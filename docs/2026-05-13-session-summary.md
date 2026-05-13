# Session summary: Intel Arc B70 + VSA + vLLM-XPU getting Qwen3.5/3.6 working

**Dates:** 2026-05-12 → 2026-05-13 (overnight)
**Outcome:** **WORKING** end-to-end VSA pipeline on the Intel Arc Pro B70 (Battlemage) rig using vLLM-XPU for the VLM stage. Spark Me Tenderly.mp4 processed to GLS + QG outputs.

## Where we are now

| Capability | Status |
|---|---|
| llama.cpp Vulkan + SYCL builds | ✅ working (Qwen3.6-27B Q6_K, Nemotron Omni 30B with mmproj) |
| llama.cpp + intellm launcher | ✅ working, configs/server lifecycle in place |
| vLLM-XPU + Qwen3.5-9B-AWQ-4bit | ✅ working (Docker + patches) |
| VSA end-to-end via vLLM-XPU | ✅ working, ~62 min total wall time for 90-min video |
| VSA transcription on Intel rig | ✅ fixed (was a silent CUDA-detect bug; patch is backward-compatible) |
| VSA Claude Code analyze stage | ✅ auto-launches, generates GLS + QG |
| Persistent canonical setup | ✅ `vllm/start.sh`, `vsa-vllm.sh`, `intellm-b70.yaml` profile |

## The reproducible recipe

```bash
cd ~/ai/Video_Analyzer
./vsa-vllm.sh "My Video.mp4"
```

That single command:
1. Starts the vLLM-XPU container at port 8000 (if not already running, idempotent)
2. Sets `CUDA_VISIBLE_DEVICES=""` so transcription falls back to CPU cleanly
3. Runs the full VSA pipeline (research → chunk → extract → transcribe → vlm → silence → episodes → prepare → Claude Code analyze)
4. Outputs to `projects/<slug>/analysis/v<N>/`:
   - `GREENLIGHT_SHEET.md` (+ .docx)
   - `QUALITY_GRADING.md` (+ .docx)
   - `DATA_PROVENANCE.md`
   - Per-stage intermediate JSONs

When done with the day:
```bash
/mnt/optane/LLMs/vllm/start.sh stop
```

## Hardware (for context)

- **GPU:** Intel Arc Pro B70 (Battlemage, BMG-G31, 32 GB VRAM, PCI 0xe223)
- **CPU:** AMD Ryzen 5 7600X (6c/12t)
- **RAM:** 30 GiB DDR5 (2 GiB reserved by BIOS/iGPU, so 32 GiB physical)
- **Storage:** WD_BLACK SN850X 1 TB (root, ext4) + Optane SSD 698 GB (`/mnt/optane`, XFS, holds models + vLLM image + swap)
- **OS:** Ubuntu 24.04, kernel 6.17
- **Stack:** Mesa 26.0.6 (kisak PPA), oneAPI 2026.0, IntelLLVM 2026.0.0
- **Other rigs (referenced):** the user runs VSA on several CUDA NVIDIA boxes; the patches here preserve CUDA behavior

## Key learnings (the stuff that mattered)

### vLLM-XPU on Battlemage

1. **Mamba/linear_attn on Intel Xe2 is the load-bearing unknown.** All three of our vLLM attempts crashed at the Mamba layer in different ways:
   - `intel/llm-scaler-vllm:0.14.0-b8.2.1` — IPEX `torch.xpu.has_xmx()` doesn't recognize BMG-G31 (sycl_arch 21483225088)
   - `intel/vllm:0.17.0-xpu` — FLA `chunk_gated_delta_rule` hardcoded `torch.cuda.device()` (CUDA-only context manager)
   - bare-metal pip `vllm==0.20.2` — EngineCore subprocess dies silently after rank assignment

2. **The breakthrough was combining three flags + a patch:**
   - Use `intel/llm-scaler-vllm:0.14.0-b8.2.1` (despite IPEX being EOL — see below)
   - Mount our patched `qwen2_vl.py` with `getattr` fallback for `max_pixels` (transformers 5.7 removed it; container's vLLM 0.14 still references it)
   - Set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` env (avoids the FlashAttention path)
   - Pass `--enforce-eager` (skip CUDA graph capture entirely)
   - Use `--max-model-len 16384` (8192 is too small for VSA's 30-frame batches; 16K fits comfortably)

3. **IPEX is EOL March 2026.** Intel's strategic XPU target is `vllm-xpu-kernels` (weekly releases). `intel/llm-scaler-vllm` will orphan ~Q3 2026. Plan to migrate when we hit our next vLLM blocker.

4. **MTP head: unsupported on XPU in any engine today.** Don't budget for MTP speedup on B70 in 2026. CUDA only.

5. **TP=2 on dual B70 is broken** (vllm#41663). Single-card only for now.

### VSA on Intel hardware

1. **Transcription was silently broken**: ctranslate2's `get_supported_compute_types("cuda")` returns *compile-time support*, not runtime availability. On Intel rigs it falsely returns non-empty → VSA selects CUDA → WhisperModel crashes → falls back to whisper-cli (not installed) → 1 stub segment per chunk. **Total transcript before fix: 207 chars. After fix: 21,901 chars.**

2. **The fix** ([patch](research/2026-05-13-vsa-ctranslate2-cuda-detect-patch.md)) gates on `ctranslate2.get_cuda_device_count() > 0` (newer API) with `AttributeError` fallback to legacy behavior. **Backward-compatible with all CUDA setups** — preserves prior behavior exactly on machines with NVIDIA hardware. New behavior only triggers on Intel/AMD GPUs (no CUDA hardware) or when user sets `CUDA_VISIBLE_DEVICES=""`.

3. **VSA's hwdetect.py still reports "CPU only" on Intel rigs.** Cosmetic — the VLM stage uses vLLM-XPU regardless. Worth patching later but not blocking.

4. **`analysis.mode: "manual"` is misnamed**: it auto-launches Claude Code (`claude -p "/analyze ..."`). Not actually manual. Generates GLS + QG end of pipeline.

5. **96 episodes detected automatically** — research stage skipped (show_meta.yaml pre-populated), episode detection from visual + audio signals correctly matched the user's manual entry of 96 episodes / paywall ep 21.

### Engine survey conclusions (Agent A research)

For Battlemage + Qwen3.5/3.6 hybrid (Mamba + INT4) + multimodal, **only two engines have a realistic path today:**
1. `intel/llm-scaler-vllm` (our working stack — Mamba prefill SYCL, Mamba decode Triton — the latter unowned bug in `intel-xpu-backend-for-triton#6658`)
2. llama.cpp SYCL/Vulkan (works, slower than vLLM for batched workloads but more stable)

Engines that **don't have a path**: SGLang (no Mamba kernels on XPU), LMDeploy (CUDA only), ktransformers (requires Sapphire Rapids + multi-RTX), MLC-LLM (TVM has no Mamba op), EXL3 (CUDA only), OpenVINO (no Mamba), IPEX-LLM (archived). Documented in [`docs/research/2026-05-12-engine-survey-b70-qwen35.md`](research/2026-05-12-engine-survey-b70-qwen35.md).

### Fork/build alternative (Agent B research)

If we ever have to fork, the strategic target is **upstream `vllm-project/vllm` + `vllm-project/vllm-xpu-kernels`** (NOT intel/llm-scaler-vllm). Patch+upstream strategy, 1-month prototype to 3-month production for 2 eng. First file to touch: `vllm/model_executor/layers/fla/ops/utils.py` — replace `torch.cuda.device()` with `torch.accelerator.device_context()` (~10 LOC). Full plan in [`docs/research/2026-05-12-fork-build-scoping.md`](research/2026-05-12-fork-build-scoping.md).

## Files of interest

### intellm repo (this repo, github.com/elsung/intellm)

| Path | What it is |
|---|---|
| `intellm` | The launcher (manages llama.cpp servers) |
| `vllm/start.sh` | Idempotent vLLM-XPU container launcher |
| `vllm/patches/qwen2_vl.py` | The qwen2_vl.py patch (mounted into container at runtime) |
| `vllm/README.md` | Why every weird knob is there |
| `configs/qwen3.5-9b-vision-q4kxl-{server,chat}.conf` | llama.cpp Qwen3.5 GGUF configs |
| `configs/nemotron-omni-vision.conf` | llama.cpp Nemotron Omni config |
| `llama.cpp/builds.conf` | Backend registry (Vulkan + SYCL) |
| `llama.cpp/env-sycl.sh` | oneAPI setvars wrapper |
| `tools/vlm_sweep.sh` | Concurrency benchmark |
| `docs/benchmarks.md` | Measured perf numbers |
| `docs/decisions.md` | Design decisions log |
| `docs/research/*.md` | Investigations |

### VSA repo (github.com/elsung/Video_Analyzer)

| Path | What it is |
|---|---|
| `vsa` | Original wrapper (llama.cpp via intellm) |
| `vsa.sh` | Wrapper that auto-starts intellm llama.cpp server |
| `vsa-vllm.sh` | **New** — wrapper that auto-starts vLLM-XPU container, sets `CUDA_VISIBLE_DEVICES=""` |
| `pipeline.py` | **Patched** for the ctranslate2 CUDA-detect issue |
| `patches/transcribe-runtime-cuda-gate.patch` | Portable patch file (backward-compatible) |
| `config.profiles/intellm-b70.yaml` | The canonical profile for this rig (vllm provider as default) |
| `config.yaml` | Active config (copy of the profile above) |

## Performance numbers (measured)

| Workload | Engine | Setup | Result |
|---|---|---|---|
| VLM stage (151 batches × 30 frames each) on Spark Me Tenderly | vLLM-XPU | 16K ctx, 8 slots, AWQ INT4 | ~25 min |
| Same VLM stage | llama.cpp SYCL | Vulkan, Q4_K_XL | ~4 hr extrapolated (killed early) |
| Transcription (43 chunks, ~90 min audio) | CPU faster-whisper int8 | Ryzen 7600X | ~12 min |
| Total VSA pipeline | vLLM + CPU whisper + Claude Code | end-to-end | **~62 min** with the rework, **~44 min** for the clean run path |

## Open work / known limitations

1. **vLLM decode performance**: ~11s per 30-frame batch. Adequate for batched VSA workload, slow for single-stream chat. Mamba on Xe2 doesn't have a tuned XMX-DPAS path yet.
2. **Long-context (>16K)** may hit Triton `DEVICE_LOST` ([issue #6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)). Stay at 16K for now.
3. **MTP head**: present in model, unused (no XPU support).
4. **TP=2 dual B70**: broken (vllm#41663). Single card only.
5. **VSA's `hwdetect.py`** reports CPU only on Intel rigs. Cosmetic.
6. **whisper.cpp Vulkan**: not wired into VSA. Could move transcription off CPU later.

## Next big steps (post-/clear)

### Tier 1: easy wins, do soon
- [ ] **Bench vLLM-XPU more rigorously** with `tools/vlm_sweep.sh` to characterize concurrency scaling and find the real `--max-num-seqs` sweet spot on B70. Currently using 8 (matches VSA), but might do better with 4 or 12.
- [ ] **Try Qwen3.6-27B-AWQ-INT4** (`cyankiwi/Qwen3.6-27B-AWQ-INT4`) via the same stack. Same path should work since the architecture is similar. Bigger model = better quality, see if speed is acceptable.
- [ ] **Try Qwen3.6-35B-A3B-AWQ-4bit** (MoE with 3B active per token) — should be faster than 27B dense because of MoE.
- [ ] **A/B vLLM vs llama.cpp on VSA** for the same video to validate ~18× speedup claim.
- [ ] **Patch `hwdetect.py`** to detect Intel GPU and report it (~50 LOC).

### Tier 2: medium effort, valuable
- [ ] **Test with longer ctx** (32K, 64K) and see when `DEVICE_LOST` hits. Document the safe ctx limit empirically.
- [ ] **whisper.cpp Vulkan integration into VSA** as a transcription engine option. Would move audio transcription off CPU onto B70.
- [ ] **Submit our `qwen2_vl.py` getattr patch upstream** to vLLM. Small enough that maintainers might just take it.
- [ ] **Submit the `ctranslate2` CUDA-detect patch upstream** to VSA / Campfire. Strict improvement, backward-compatible.

### Tier 3: bigger investments
- [ ] **Try MTP-enabled model** when Intel XPU support lands. Watch `vllm-xpu-kernels` for `mtp_proposer.py` XPU port.
- [ ] **Fork plan from Agent B** ([2026-05-12-fork-build-scoping.md](research/2026-05-12-fork-build-scoping.md)) — only if we hit a hard perf wall. 1-month prototype, 3-month production for 2 eng.
- [ ] **NVIDIA P100 support** (the user mentioned they're getting P100s eventually). Build a CUDA `build-cuda-p100/` for llama.cpp; vLLM CUDA path is well-supported.
- [ ] **Multi-machine intellm orchestration** — coordinate llama.cpp / vLLM across the user's other rigs.

### Tier 4: research and exploration
- [ ] **Investigate KV cache offload to Optane** — was a topic earlier, found to be physics-blocked on Intel (PCIe roundtrip cost > Optane latency). Revisit if vLLM grows multi-tier KV support.
- [ ] **Try TurboQuant KV** (3-bit K / 2-bit V) for longer context on same VRAM budget. AmesianX llama.cpp fork has it.
- [ ] **Speaker diarization** — currently off. Set `transcription.diarize: true` + `HF_TOKEN` to test.
- [ ] **Test vLLM-XPU 0.17 + the FLA accelerator-API patch** (the `torch.cuda.device()` → `torch.accelerator.device_context()` swap, ~10 LOC). May give us a path on the non-IPEX container.

## Quick reference cards

### "I want to start a fresh session and resume"
```bash
# 1. Make sure vLLM container is running
/mnt/optane/LLMs/vllm/start.sh

# 2. Run VSA on something
cd ~/ai/Video_Analyzer
./vsa-vllm.sh "My Video.mp4"

# 3. When done
/mnt/optane/LLMs/vllm/start.sh stop
```

### "I want to use llama.cpp instead of vLLM"
```bash
# Edit config.yaml: change vlm.provider from "vllm" to "llamacpp"
# Or use the fallback script that does it for you:
cd ~/ai/Video_Analyzer
./vsa.sh "My Video.mp4"
# This starts intellm-managed llama-server at port 8080 instead
```

### "What's the model file currently being used?"
```bash
intellm --show-config qwen3.5-9b-vision-q4kxl-server | python3 -m json.tool
# Or for the vLLM stack:
cat /mnt/optane/LLMs/vllm/start.sh | grep "VLLM_MODEL_PATH"
```

### "I want to switch the vLLM model to Qwen3.6-27B"
```bash
# Download the model
mkdir -p /mnt/optane/LLMs/hf-models/cyankiwi__Qwen3.6-27B-AWQ-INT4
cd /mnt/optane/LLMs/hf-models/cyankiwi__Qwen3.6-27B-AWQ-INT4
# (use curl to download all files from https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4/tree/main)

# Stop current vLLM
/mnt/optane/LLMs/vllm/start.sh stop

# Set new model + restart
VLLM_MODEL_PATH=/mnt/optane/LLMs/hf-models/cyankiwi__Qwen3.6-27B-AWQ-INT4 \
VLLM_SERVED_NAME=cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  /mnt/optane/LLMs/vllm/start.sh

# Update VSA config.yaml vlm.vllm.model to "cyankiwi/Qwen3.6-27B-AWQ-INT4"
# Then ./vsa-vllm.sh
```

## Files to read for full context

1. **This file** — you are here
2. [`docs/research/2026-05-13-vllm-xpu-working-config.md`](research/2026-05-13-vllm-xpu-working-config.md) — the breakthrough config detail
3. [`docs/research/2026-05-13-vsa-ctranslate2-cuda-detect-patch.md`](research/2026-05-13-vsa-ctranslate2-cuda-detect-patch.md) — the transcription fix
4. [`docs/research/2026-05-12-vllm-xpu-qwen35-battlemage-failures.md`](research/2026-05-12-vllm-xpu-qwen35-battlemage-failures.md) — what didn't work and why
5. [`docs/research/2026-05-12-engine-survey-b70-qwen35.md`](research/2026-05-12-engine-survey-b70-qwen35.md) — other engines considered
6. [`docs/research/2026-05-12-fork-build-scoping.md`](research/2026-05-12-fork-build-scoping.md) — if we fork
7. [`vllm/README.md`](../vllm/README.md) — the vLLM-XPU container setup deep-dive

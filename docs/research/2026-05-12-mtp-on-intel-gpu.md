# Multi-Token Prediction (MTP) on Intel Arc B70 — feasibility

**Date:** 2026-05-12
**User reference:** [Reddit thread on Qwen3.6 35B A3B Heretic Native MTP](https://www.reddit.com/r/huggingface/comments/1t7qhaf/qwen36_35b_a3b_uncensored_heretic_native_mtp/)

---

## TL;DR

**Don't bet on MTP on the B70 today.** The llama.cpp MTP PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) is **still open, not merged**, and **explicitly tested only on CUDA / Vulkan / ROCm — no SYCL validation**. vLLM-XPU (`intel/llm-scaler-vllm 0.14.0-b8.2.1`) has **zero mention of MTP** in release notes.

**Workable today:** build the unmerged PR branch with **Vulkan** (since Vulkan is in the PR's tested matrix and works on B70), run `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`. Expect 1.4–1.7× decode speedup (vs CUDA's 1.85×; Battlemage bandwidth caps the upside).

**Wait:** MTP merges into llama.cpp master likely in 1–2 weeks. SYCL coverage probably 1–2 months out. vLLM-XPU MTP: unscheduled.

---

## Model availability

The Reddit-cited model resolves to:
- `tvall43/Qwen3.6-35B-A3B-heretic` (FP weights, abliterated)
- **`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`** ← use this; MTP head preserved in GGUF
- `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF` (mirror)
- `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` (NVFP4 variant — won't run on B70 per earlier research)

**Important quirk:** GGUF conversion only preserves the MTP head when the converter is explicitly MTP-aware (unsloth/havenoammo/RDson/froggeric do this; the default `convert_hf_to_gguf.py` drops it). The user's existing `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` is *not* MTP — would need to download the MTP variant separately.

**Other publicly available MTP-head models (May 2026):**
- DeepSeek-V3 / V3.2
- Qwen3.5 + Qwen3.6 (27B, 35B-A3B)
- GLM-4 MoE / MoE-Lite
- MiMo-7B
- MiniMax M2
- Gemma 4 assistant checkpoints
- **Llama 4 does NOT ship MTP heads**

## llama.cpp MTP status

**PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)** ("llama + spec: MTP Support" by am17an):
- **OPEN, under review, not merged** into master as of `bbeb89d`
- New flags: `--spec-type mtp`, `--spec-draft-n-max N` (2–3), `--spec-draft-p-min`
- Backend coverage in PR testing: **CUDA (primary), Vulkan (works; tensor-split crashes with `-sm row`), ROCm (works), Metal (incomplete)**. **No SYCL test coverage.**
- Constraints:
  - `-np 1` only (no batched decoding yet)
  - Incompatible with `--mmproj` (no multimodal compose)
- Benchmarks cited in PR: **Qwen3.6-27B Q8 22.97 → 42.45 tok/s (1.85×) on CUDA**. Nothing on Intel.

Issue [#22747](https://github.com/ggml-org/llama.cpp/issues/22747) is a separate open feature request for "MTP drafters" — confirms upstream is still WIP.

To use MTP today you must build the `am17an/llama.cpp:mtp-clean` branch.

## vLLM / vLLM-XPU MTP status

- **Upstream vLLM:** native MTP via `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`. Officially supported model families: Gemma 4 assistant checkpoints, XiaomiMiMo, DeepSeek-V3, Qwen3.5. **Qwen3.6-35B-A3B is not on the official supported-models list.**
- **vLLM-XPU (`intel/llm-scaler-vllm 0.14.0-b8.2.1`, May 6 2026):** release notes mention CPU KV cache offload, FP8 KV cache, and "speculative decoding with 2 more methods (medusa, suffix)" — **MTP is not listed**. Spec-decode methods shipped on XPU: **n-gram, suffix, medusa, EAGLE**. **Not native MTP.** The MTP `mtp_proposer.py` in vLLM uses Flash/Triton kernels not ported/validated on XPU.

## Concrete paths

| Path | Verdict |
|---|---|
| (a) llama.cpp **SYCL** + MTP PR | **Untested. Likely broken.** PR is CUDA/Vulkan-focused; Battlemage SYCL still has open issues (#21893, #21517). Compile it, expect to file a SYCL bug. |
| (a') llama.cpp **Vulkan** + MTP PR on B70 | **Most likely working path today.** Vulkan is in PR's tested matrix; B70 Vulkan works via Mesa/ANV. Avoid `-sm row`. |
| (b) vLLM-XPU + native MTP | **Doesn't work.** Not in 0.14.0-b8.2.1, not in upstream XPU. |
| (c) Draft-model speculative decoding (fallback) | Works on llama.cpp SYCL + vLLM-XPU n-gram/suffix/medusa. For Qwen3.6-35B-A3B with Qwen3.5-0.8B draft — [thc1006 benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) found **no net speedup on A3B MoE** (already-small active param count limits gains). |
| (d) Wait | MTP likely merges into llama.cpp master in 1–2 weeks. SYCL coverage 1–2 months. vLLM-XPU MTP unscheduled. |

### Commands for (a') — Vulkan + MTP tonight

```bash
git clone -b mtp-clean https://github.com/am17an/llama.cpp ~/LLMs/llama.cpp/src-mtp
cmake -S ~/LLMs/llama.cpp/src-mtp -B ~/LLMs/llama.cpp/build-vulkan-mtp \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build ~/LLMs/llama.cpp/build-vulkan-mtp --config Release -j$(nproc)

~/LLMs/llama.cpp/build-vulkan-mtp/bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  -ngl 99 -c 8192 -fa on -np 1 \
  --spec-type mtp --spec-draft-n-max 3
```

Expected speedup on Xe2: unverified — Vulkan + B70 with the MTP PR has not been benchmarked publicly. Battlemage memory bandwidth (compared to RTX 3090/4090) caps the upside. Best estimate: **1.4–1.7×** decode improvement on this rig.

## intellm integration sketch (deferred)

If/when we decide to wire MTP into the launcher:

- **Flag:** `--mtp N` (number of speculative tokens), `--spec-type mtp` implied.
- **Config field:** `mtp = N` (int, 0 = off, default 0).
- **Build entry:** add a third build in `builds.conf`, e.g. `vulkan-mtp` pointing at `llama.cpp/build-vulkan-mtp/bin`. Treat the PR branch as a parallel source tree (`llama.cpp/src-mtp/`).
- **Mapping:**
  - llama.cpp Vulkan-MTP backend: `--spec-type mtp --spec-draft-n-max N`
  - vLLM (future): `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`
- **Gating:**
  - `mtp>0` + backend in `{vllm-xpu, sycl}` → hard error / strong warning
  - `mtp>0` forces `slots=1` and refuses `--mmproj` (PR constraint)
- **Composition:**
  - KV quant: orthogonal, works
  - Prompt cache: works (MTP is decode-only)
  - Multimodal: **incompatible** in current PR

## Bottom line

If MTP is curiosity-grade exploration tonight: clone `am17an/llama.cpp:mtp-clean`, build Vulkan, grab `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, try the example above. Don't bet anything important on it. Recheck `llm-scaler-vllm` release notes after the next b8.3 drop for XPU MTP, and watch PR #22673 for merge.

## Sources

- [llama.cpp PR #22673 — MTP Support](https://github.com/ggml-org/llama.cpp/pull/22673)
- [llama.cpp Issue #22747 — MTP drafters](https://github.com/ggml-org/llama.cpp/issues/22747)
- [llama.cpp Discussion #12130](https://github.com/ggml-org/llama.cpp/discussions/12130)
- [llama.cpp Issue #21893 — Battlemage SYCL correctness](https://github.com/ggml-org/llama.cpp/issues/21893)
- [llama.cpp Issue #21517 — Q8 perf on B70](https://github.com/ggml-org/llama.cpp/issues/21517)
- [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- [tvall43/Qwen3.6-35B-A3B-heretic](https://huggingface.co/tvall43/Qwen3.6-35B-A3B-heretic)
- [vLLM MTP docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [vLLM Issue #12181 — MTP feature](https://github.com/vllm-project/vllm/issues/12181)
- [intel/llm-scaler releases](https://github.com/intel/llm-scaler/releases)
- [Phoronix: LLM-Scaler-vLLM 1.3](https://www.phoronix.com/news/Intel-LLM-Scaler-vLLM-1.3)
- [thc1006 RTX3090 spec-decode on Qwen3.6-35B-A3B](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)
- [Bibek Poudel: Qwen3.6-27B on B70](https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7)
- [Reddit thread (user-cited)](https://www.reddit.com/r/huggingface/comments/1t7qhaf/qwen36_35b_a3b_uncensored_heretic_native_mtp/)

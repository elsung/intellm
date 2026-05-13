# Engine survey for Qwen3.5/3.6 on Intel Arc B70

**Date:** 2026-05-12
**Investigator:** background research agent
**Target:** running `cyankiwi/Qwen3.5-9B-AWQ-4bit` (Mamba+attention hybrid, compressed-tensors INT4, MTP head) on B70 with multi-image VLM concurrency.

## TL;DR

Only **two engines** have a realistic Battlemage Mamba path today, and both gate on the same Triton kernel bug ([intel-xpu-backend-for-triton#6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)):

1. **`intel/llm-scaler-vllm:0.14.0-b8.2.1`** — B70 officially validated (Phoronix 2026-05-06); ships v0.14 vLLM + `vllm-xpu-kernels` v0.1.8 with chunk_gdn_attention SYCL kernel. Prefill works; decode crashes via the unowned Triton kernel.
2. **llama.cpp SYCL/Vulkan + Unsloth Q4_K_M GGUF** — works, just under-tuned in our trials. PMZFX benchmark reports 54.7 tok/s gen + 615 tok/s prefill on Qwen3.6-35B-A3B on a single B70.

Other engines surveyed are either CUDA-only (EXL3, LMDeploy, ktransformers) or don't have Mamba on XPU yet (SGLang, MLC-LLM, OpenVINO).

## Per-engine status

| Engine | B70 status | Mamba/GDN | compressed-tensors INT4 | VLM | Effort |
|---|---|---|---|---|---|
| **intel/llm-scaler-vllm** | ✅ validated 2026-05 | partial (prefill SYCL ✓, decode Triton crashes) | ✅ PR #29484 merged 2025-12 | ✅ | MEDIUM |
| **upstream vllm + xpu-kernels** | partial — kernels live in vllm-xpu-kernels | partial (same as above) | ✅ | partial | MEDIUM-HIGH |
| **llama.cpp (SYCL/Vulkan)** | ✅ working | ✅ Mamba2 PR merged late 2025 | ❌ (uses GGUF) | ✅ (mtmd/clip) | LOW |
| **SGLang + sgl-kernel-xpu** | partial (B580 only) | ❌ no Mamba kernels | partial | ❌ XPU | HIGH |
| **LMDeploy** | ❌ no first-party support | ❌ | ❌ | CUDA only | BLOCKED |
| **ktransformers** | ❌ "4× RTX 4090 + Sapphire Rapids required" | CPU/CUDA only | ❌ | ❌ | BLOCKED |
| **MLC-LLM** | theoretical Vulkan | ❌ no TVM Mamba op | ❌ | ❌ | BLOCKED (~abandoned) |
| **exllamav3 (EXL3)** | ❌ CUDA + ROCm only | works on CUDA | EXL3/QTIP format only | CUDA only | BLOCKED |
| **IPEX-LLM standalone** | archived 2026-01 | — | — | — | BLOCKED |
| **OpenVINO GenAI 2026.1** | ✅ B-series | ❌ no Mamba | NNCF format only | ✅ (non-hybrid Qwen) | HIGH |
| **Aphrodite** | partial (inherits vLLM-XPU) | same bugs as vLLM | same | partial | HIGH |

## The Triton kernel issue (the real blocker)

The decode path for Qwen3.5/3.6 hybrid uses `fused_recurrent_gated_delta_rule_fwd_kernel`, a Triton recurrent kernel. On BMG-G31 it fails with `UR_RESULT_ERROR_DEVICE_LOST`.

- Filed 2026-04-14 in [intel/intel-xpu-backend-for-triton#6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)
- Assigned to `quinnlp`, unowned in practice
- Same kernel works on A6000 (CUDA) and Arc Alchemist (A-series)
- Likely a driver/firmware long-running-kernel preemption issue on Xe2

**Prefill path is separate** — `chunk_gdn_attention` SYCL kernel in `vllm-xpu-kernels` v0.1.6+ handles prefill. If you can force decode through the SYCL path too, decode works.

## Knobs we never tried (per Agent A's recommendation)

For one more `intel/llm-scaler-vllm:0.14.0-b8.2.1` attempt, add:

- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` env (instead of default FlashAttn)
- `VLLM_USE_V1=1` env
- `--enforce-eager` (skip CUDA graph capture — we don't need it on XPU)
- `--mamba-cache-mode align` (forces aligned Mamba cache layout)
- `--max-num-seqs 8`
- `--max-model-len 8192` (constrain context to avoid hitting decode-path edge cases)

If decode still crashes with these → DEVICE_LOST is definitively the wall.

## llama.cpp realistic optimization (what we missed)

PMZFX benchmark on this exact hardware (commit `ec6f7a6a5c`, 2026-04-21): Qwen3.6-35B-A3B Q4_K_M = **54.7 tok/s gen + 615 tok/s prefill** on one B70. Our earlier ~25 tok/s suggests significant tuning headroom.

Missing knob: **`-ub 256`** (sub-batch size). [llama.cpp#18725](https://github.com/ggml-org/llama.cpp/issues/18725) shows that `>512` halves perf on Mamba2 path. Default may be 512+.

Also: switch model from `cyankiwi/Qwen3.5-9B-AWQ-4bit` (compressed-tensors not native to llama.cpp) → `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` (already on disk). Mamba2 PR merged late 2025 means hybrid layers are supported.

Per-stream serializes through one Mamba state — `-np 8` parallel slots gives concurrency but each slot's decode is sequential through SSM. Don't expect aggregate throughput to scale linearly past ~4 slots.

## Quality bug to apply regardless of engine

[vllm#38994](https://github.com/vllm-project/vllm/issues/38994): Qwen3.5-9B generates repetitive output on Intel XPU. Apply generation-config override from [cyankiwi HF discussion #2](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit/discussions/2) — adjust temperature, top_k, repetition_penalty settings in the request.

## Things we missed in earlier research

- **PR #29484** (compressed-tensors W4A16 XPU, merged 2025-12-05) — the kernel exists. Our model's quant format IS supported, contrary to assumption.
- **vllm-xpu-kernels release cadence is weekly** (v0.1.6 → v0.1.7 → v0.1.8 across Apr–May 2026). Track this repo, not vLLM main.
- **Sandermage/genesis-vllm-patches** — 126-patch CUDA-only fork specifically for Qwen3.6 (TurboQuant KV, MTP, GDN streaming). Worth reading before writing XPU patches.
- **MTP head is unsupported on XPU in any engine** — don't budget for MTP speedup on B70 in 2026.

## Top-3 actionable recommendations (Agent A's ranking)

### #1: llm-scaler-vllm with the missed flags
**First action:** retry `docker run intel/llm-scaler-vllm:0.14.0-b8.2.1` with the env+args listed above. Comment on issue #6658 with our repro if decode crashes — Intel is responsive there.

### #2: llama.cpp SYCL + Unsloth GGUF + proper tuning
**First action:** `llama-server -m unsloth/Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -np 8 -cb -ub 256 --mmproj <projector>`. Run VSA. Should land ~30-45 min for the 60-min pipeline (vs our 4-hour estimate).

### #3: llama.cpp Vulkan (no-toolchain fallback)
If SYCL has runtime issues, Vulkan fallback. ~22 tok/s vs ~33 SYCL on 27B per Poudel. Still tune `-ub 256`.

## If we have to fork

`vllm-project/vllm-xpu-kernels` is the chokepoint. Porting `fused_recurrent_gated_delta_rule_fwd_kernel` from Triton-XPU (which crashes) to hand-written SYCL (mirroring the chunked variant) closes the gap. The Triton kernel is ~150 lines; comments in issue #6658 diagnose the sequential-recurrence stress pattern.

## Sources

- [intel/llm-scaler releases](https://github.com/intel/llm-scaler/releases)
- [intel/llm-scaler issue #371 (Qwen3.5-27B GPTQ-Int4 perf on B70)](https://github.com/intel/llm-scaler/issues/371)
- [Phoronix: Intel llm-scaler-vllm 0.14-b8](https://www.phoronix.com/news/Intel-llm-scaler-vllm-0.14-b8)
- [vLLM PR #29484 (compressed-tensors W4A16 XPU)](https://github.com/vllm-project/vllm/pull/29484)
- [vLLM issue #37979 (Intel quant roadmap H1 2026)](https://github.com/vllm-project/vllm/issues/37979)
- [vllm-xpu-kernels releases](https://github.com/vllm-project/vllm-xpu-kernels/releases)
- [vllm-xpu-kernels Q1 2026 roadmap #141](https://github.com/vllm-project/vllm-xpu-kernels/issues/141)
- [intel-xpu-backend-for-triton#6658 (DEVICE_LOST on BMG GDN decode)](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)
- [vLLM #41663 (XPU TP=2 dual B70 crash)](https://github.com/vllm-project/vllm/issues/41663)
- [vLLM #38994 (Qwen3.5-9B garbled output on XPU)](https://github.com/vllm-project/vllm/issues/38994)
- [cyankiwi/Qwen3.5-9B-AWQ-4bit discussion #2](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit/discussions/2)
- [Bibek Poudel field guide (Qwen3.6-27B on B70)](https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7)
- [PMZFX B70 benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)
- [llama.cpp #18725 (Qwen3-Next ubatch perf cliff on Vulkan)](https://github.com/ggml-org/llama.cpp/issues/18725)
- [llama.cpp #22320 (Qwen3.6-A3B SSM GPU utilization)](https://github.com/ggml-org/llama.cpp/issues/22320)
- [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches)

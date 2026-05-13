# Fork/build scoping: getting Qwen3.5/3.6 hybrid + INT4 + B70 working

**Date:** 2026-05-12
**Investigator:** background research agent
**Question:** If we have to fork or build, where do we start, and how long does it take?

## Reframing the math

Four facts change the cost calculus:

1. **IPEX is EOL** (end of March 2026). XPU support moves to upstream `torch.xpu`. Our `has_xmx` Battlemage failure is in a deprecated layer.
2. **`vllm-xpu-kernels` is the strategic target** — Intel funds it directly. Q1 2026 roadmap [#141](https://github.com/vllm-project/vllm-xpu-kernels/issues/141) explicitly schedules BMG W4A16 GEMM, MLA, sparse MLA.
3. **GDN attention already exists on XPU** (`_xpu_C::gdn_attention` ships in `vllm-xpu-kernels`). Just numerical-correctness rough edges + the Triton decode kernel is what crashes.
4. **Our FLA failure (attempt #2 in vLLM saga) is a ~10-LOC accelerator-API patch** — replace `torch.cuda.device(idx)` with `torch.accelerator.device_context(idx)`.

## Fork-base scorecard

### 1. Upstream vLLM + `vllm-xpu-kernels` ★ RECOMMENDED

| In place | Missing |
|---|---|
| Flash-attn XPU, RMSNorm, RoPE, GeLU/SiLU | Robust BMG path for chunk-prefill GDN |
| FP8 GEMM/MoE, MXFP4 MoE | W4A16 GEMM on BMG (on Q1 2026 roadmap, only Linear merged) |
| `gdn_attention` SYCL kernel | `fla/ops/utils.py` `torch.cuda.device()` wrapper (10 LOC) |
| W4A16 wNa16 Linear ([PR #29484](https://github.com/vllm-project/vllm/pull/29484) merged Dec 2025) | TP=2 XCCL workaround |
| Triton path for `chunk_gated_delta_rule` | MTP (Gemma4 only, has acceptance bugs [#41789](https://github.com/vllm-project/vllm/issues/41789)) |
| Multimodal pipeline (TORCH_SDPA / TRITON_ATTN ViT backends) | |

**LOC gap:** low thousands. SYCL GDN kernel hardening (~500–2000), W4A16 BMG XMX tuning (~500–1500), accelerator-API patches across `fla/`, `mamba/` (~200), TP=2 oneCCL workarounds (env-vars mostly).

**Hardest problem:** Writing an XMX-tiled SYCL kernel for the chunked gated delta-rule recurrence that avoids the Xe2 long-running-thread `DEVICE_LOST` issue. The existing Triton sequential recurrence stresses Xe2 ULSS/preemption. Rewrite with bounded inner-loop iteration count + explicit `nd_range` partition.

**Effort (2 eng):** **1–3 months** to working state, **3–6 months** to performant.

### 2. `intel/llm-scaler-vllm` (Intel's vendored vLLM)

Same as #1 plus Intel's BMG-specific patches not yet upstream. **But IPEX-rooted = will orphan by ~Q3 2026.** Patching `has_xmx` is throwaway work.

**Effort: 1–3 months but technical-debt-positive. Avoid as fork base.**

### 3. `llama.cpp` + SYCL Mamba XMX

| In place | Missing |
|---|---|
| SYCL backend (Q4_K_M ~22 tok/s on B70 per Poudel) | No Mamba XMX path — SSM ops are scalar/SIMD on SYCL, not XMX-tiled DPAS |
| Mamba2 PR merged late 2025 ([#9196](https://github.com/ggml-org/llama.cpp/discussions/9196)) | No compressed-tensors loader (GGUF only) |
| Xe2 warptile fix ([Hal9000AIML cherrypicks](https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes)) | No multi-image VLM batching in server mode |
| | No MTP |

**LOC gap:** ten-thousand+. Writing `compressed_tensors`→`ggml-tensor` converter, SSM XMX kernel from scratch, batched ViT path.

**Hardest problem:** Writing the XMX DPAS-tiled SYCL kernel for the SSM `chunked_gated_delta_rule` from scratch with no reference numerics. `ggml-sycl` has no precedent for the state-carrying recurrence pattern. You'd reinvent what vLLM-XPU already has.

**Effort: 6+ months.** Wrong path unless we only want Q4_K_M GGUF.

### 4. SGLang / `sgl-kernel-xpu`

Has RMSNorm, RoPE, GeLU/SiLU, bf16/f16 GEMM, FP8 W8A16, INT4 AWQ/GPTQ, fused MoE. No Mamba/GDN/linear-attn kernels. Supported model list = `Llama-3.x` + `Qwen2.5-1.5B` only. No hybrid path.

**Hardest problem:** SGLang's RadixAttention prefix-cache is incompatible with Mamba state caching — designing a unified KV+SSM cache is a research-grade problem.

**Effort: 3–6 months.** Promising long-term, premature today.

### 5. EXL3 / `exllamav3` (turboderp)

CUDA-only. QTIP kernels use CUDA tensor-core PTX intrinsics with no SYCL analog. **Effort: 6+ months** (essentially a port).

### 6. Greenfield

**Effort: 6+ months minimum.** Not justified given #1 has ~70% of what we need.

## Recommendation

**Fork base: `vllm-project/vllm` HEAD + `vllm-project/vllm-xpu-kernels` HEAD.**
**Strategy: patch + upstream**, not maintain a long-lived fork.

Rationale:
- Intel actively funds vllm-xpu-kernels; W4A16-on-BMG is their Q1 2026 deliverable
- The 4 blockers each map to small, upstreamable patches
- IPEX deprecation will orphan any intel/llm-scaler fork by ~Q3 2026

## In-progress upstream work to track

- **vllm-xpu-kernels Q1 2026 W4A16 BMG GEMM** ([roadmap #141](https://github.com/vllm-project/vllm-xpu-kernels/issues/141))
- **Intel H1 2026 quantization RFC** ([vllm#37979](https://github.com/vllm-project/vllm/issues/37979)) — PRs #37986, #38192 merged for W4A16 Linear; [#39778](https://github.com/vllm-project/vllm/pull/39778) under review for AWQ/GPTQ
- **Triton GDN DEVICE_LOST issue** ([#6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)) — sits unowned. Don't wait — write the SYCL replacement.

## First 3 files to touch tomorrow

1. **`vllm/model_executor/layers/fla/ops/utils.py`** — replace `torch.cuda.device(self.idx).__enter__()` with `torch.accelerator.device_context(self.idx)`. ~10 LOC. Unblocks our vLLM attempt #2.

2. **`vllm-xpu-kernels/csrc/attention/gdn_attention.cpp`** — harden the SYCL `_xpu_C::gdn_attention` chunk path for chunk-prefill, replacing the Triton fused-recurrent fallback for sequences > chunk threshold. ~500–1500 LOC of SYCL with explicit `sub_group_size(16)`, XMX DPAS via `joint_matrix`, bounded loop trip counts to avoid DEVICE_LOST.

3. **`vllm/model_executor/layers/quantization/kernels/mixed_precision/xpu.py`** + **`vllm-xpu-kernels/csrc/quantization/w4a16_gemm_bmg.cpp`** — wire compressed-tensors W4A16 through to a BMG-tuned XMX GEMM. Currently only `ipex.py` path is plumbed for older arches.

## First 5 milestones (if we commit)

| # | Milestone | Timeframe |
|---|---|---|
| **M1** | Reproduce Qwen3.5-9B Q4_K_M on llama.cpp-SYCL @ ≥22 tok/s baseline. Patch `fla/ops/utils.py`. Get vLLM main + vllm-xpu-kernels v0.1.8 to **load** Qwen3.5 fp16 on single B70 (no quant, short ctx, single-GPU). | 2 weeks |
| **M2** | Chunk-prefill GDN working on single B70 via SYCL kernel for ALL sequence lengths (not just short). Bypass Triton recurrence entirely. Numerical-correctness validation vs CUDA reference (A6000). | 3–4 weeks |
| **M3** | BMG-tuned W4A16 compressed-tensors GEMM (XMX DPAS via `joint_matrix`). Validate Qwen3.5-27B-W4A16 loading + accuracy parity (≤1% lm-eval delta vs fp16). | 4–6 weeks |
| **M4** | TP=2 working on dual B70 (when 2nd card arrives) with `CCL_ENABLE_SYCL_KERNELS=0` workaround, then proper fix via oneCCL upstream. Multi-image VLM batched path (Qwen3.5-VL visual encoder fp16). | 6–8 weeks |
| **M5** | Performance pass: target ≥80 tok/s decode on Qwen3.5-27B-W4A16 single B70 (vs ~22 tok/s llama.cpp Q4_K_M baseline). Upstream all patches. Optional MTP if Gemma4 [#41789](https://github.com/vllm-project/vllm/issues/41789) is fixed. | 8–12 weeks |

**Total: ~3 months to a usable Qwen3.5-on-B70 vLLM-XPU path, all upstream-mergeable.** 1-month prototype, 3-month production-quality for a 2-eng team.

## Hardest single risk

The Xe2 `DEVICE_LOST` on long-running Triton kernels. If it turns out the SYCL `gdn_attention` op also hits it (i.e., it's a `xe` driver / Mesa / oneAPI runtime / firmware issue with any compute kernel >N ms), then **M2 becomes a Mesa/compute-runtime/firmware problem and the schedule slips 1–2 months.** Watch this carefully in M1.

## Sources

- [vLLM-XPU Q1 2026 roadmap #141](https://github.com/vllm-project/vllm-xpu-kernels/issues/141)
- [vllm-xpu-kernels repo](https://github.com/vllm-project/vllm-xpu-kernels)
- [Intel-XPU-Triton DEVICE_LOST #6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)
- [vLLM TP=2 BMG crash #41663](https://github.com/vllm-project/vllm/issues/41663)
- [vLLM GPTQ-XPU regression #39474](https://github.com/vllm-project/vllm/issues/39474)
- [PR #29484 (compressed-tensors XPU wNa16, merged)](https://github.com/vllm-project/vllm/pull/29484)
- [Intel quantization roadmap RFC #37979](https://github.com/vllm-project/vllm/issues/37979)
- [XPU kernel migration RFC #33214](https://github.com/vllm-project/vllm/issues/33214)
- [IPEX EOL March 2026](https://github.com/ACEsuit/mace/issues/1302)
- [llama.cpp Mamba2 PR #9196](https://github.com/ggml-org/llama.cpp/discussions/9196)
- [llama.cpp Q8_0 SYCL efficiency #21517](https://github.com/ggml-org/llama.cpp/issues/21517)
- [Hal9000AIML B70 cherry-pick kit](https://github.com/Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes)
- [Poudel guide: Qwen3.6-27B on B70](https://bibek-poudel.medium.com/how-to-run-qwen3-6-27b-locally-on-intel-arc-pro-b70-what-actually-works-c96dec67c6f7)

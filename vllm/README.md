# vLLM-XPU on Intel Arc B70 — working setup

Canonical setup for running `cyankiwi/Qwen3.5-9B-AWQ-4bit` (Mamba+attention hybrid, compressed-tensors INT4) via vLLM-XPU on Intel Arc Pro B70 (Battlemage / BMG-G31).

This setup survived [three earlier failures](../docs/research/2026-05-12-vllm-xpu-qwen35-battlemage-failures.md) — the working combination is documented in [`docs/research/2026-05-13-vllm-xpu-working-config.md`](../docs/research/2026-05-13-vllm-xpu-working-config.md).

## Quick start

```bash
# Start the server (idempotent — no-op if already running and healthy)
/mnt/optane/LLMs/vllm/start.sh

# Check it's serving
curl http://localhost:8000/v1/models | python3 -m json.tool

# Stop when done
/mnt/optane/LLMs/vllm/start.sh stop

# Follow logs
/mnt/optane/LLMs/vllm/start.sh logs
```

## What's in here

| File | Purpose |
|---|---|
| `start.sh` | Canonical launch script — docker run with all the patches mounted, all the right env vars, the right flags. Idempotent: re-running while healthy is a no-op. |
| `patches/qwen2_vl.py` | The vLLM model file with `max_pixels` getattr fallback. Mounted RO into the container at runtime. Without this patch the container's older vLLM tries `image_processor.max_pixels` against newer transformers that removed that attribute → `AttributeError`. |

## What `start.sh` actually does

- Uses `intel/llm-scaler-vllm:0.14.0-b8.2.1` (the B70-validated Intel container)
- Mounts the AWQ model from `/mnt/optane/LLMs/hf-models/cyankiwi__Qwen3.5-9B-AWQ-4bit/`
- Mounts our patched `qwen2_vl.py` over the container's copy
- Sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (avoids the FlashAttention path that crashes EngineCore on this build)
- Adds `--enforce-eager` to skip CUDA graph capture (no value on XPU, and capture broke things)
- `--max-model-len 16384` — enough for VSA's 30-frame multimodal batches (~12K tokens) plus prompt + output
- `--max-num-seqs 8` — VSA's default `concurrent_requests`
- `--gpu-memory-utilization 0.85` — leaves headroom for the image preprocessor and the visual encoder
- `--quantization compressed-tensors` (the model is actually pack-quantized INT4 via llmcompressor despite the "AWQ" name)
- Exposes vLLM at `localhost:8000`

## Why each weird bit is there

| Knob | Why |
|---|---|
| Patched `qwen2_vl.py` | Container's vLLM 0.14.1.dev0 references `image_processor.max_pixels`; transformers 5.7.0.dev0 dropped that attribute. `getattr(..., None) or size["longest_edge"]` fallback restores function. |
| `--enforce-eager` | XPU has no `torch.cuda.graph` equivalent; vLLM's capture-by-default tripped paths we don't want. Eager mode = direct python execution = works. |
| `VLLM_ATTENTION_BACKEND=TRITON_ATTN` | Without this env vLLM may pick FlashAttention or another backend that crashes EngineCore silently on this image+model combo. |
| `compressed-tensors` quant | The model's `config.json` declares `quant_method: compressed-tensors` despite the "AWQ" in the repo name. Using `--quantization awq` raises pydantic validation error. |
| `--max-num-seqs 8` | VRAM math: at FP16 KV with GQA, 8 slots × 16K ctx ≈ 16 GB of KV pool, fits alongside ~9 GB model + activations within `gpu_memory_utilization 0.85` of 32 GB. |
| `ZE_AFFINITY_MASK=0` | Pin to GPU 0 (avoids surprises if a second card lands later). |
| `CCL_ENABLE_SYCL_KERNELS=0` | oneCCL backend workaround required for vLLM-XPU single-card runs. |

## Known limitations

- **Decode performance:** ~11s per 30-frame batch on the B70. Adequate for VSA's batched workload but not snappy for single-stream chat. The Mamba layers don't have a tuned XMX-DPAS path yet ([intel-xpu-backend-for-triton#6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)).
- **MTP head:** present in the model but unsupported by vLLM-XPU. No speedup from MTP available today.
- **TP=2:** broken on dual B70 ([vllm#41663](https://github.com/vllm-project/vllm/issues/41663)). Single-card only.
- **Long contexts (>16K):** the Triton `fused_recurrent_gated_delta_rule_fwd_kernel` is what causes `DEVICE_LOST` on long-running decode. Stay at 16K or shorter for now.
- **Repetition bug at high temperature:** [vllm#38994](https://github.com/vllm-project/vllm/issues/38994) — set `temperature ≤ 0.3` and `repetition_penalty ≥ 1.1` in requests.

## See also

- [docs/research/2026-05-13-vllm-xpu-working-config.md](../docs/research/2026-05-13-vllm-xpu-working-config.md) — full breakthrough writeup
- [docs/research/2026-05-12-vllm-xpu-qwen35-battlemage-failures.md](../docs/research/2026-05-12-vllm-xpu-qwen35-battlemage-failures.md) — what didn't work and why
- [docs/research/2026-05-12-engine-survey-b70-qwen35.md](../docs/research/2026-05-12-engine-survey-b70-qwen35.md) — alternative engines surveyed
- [docs/research/2026-05-12-fork-build-scoping.md](../docs/research/2026-05-12-fork-build-scoping.md) — if we have to fork

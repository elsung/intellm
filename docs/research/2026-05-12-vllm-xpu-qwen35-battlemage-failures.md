# vLLM on Intel Arc B70: three failure modes for Qwen3.5/3.6 hybrid models

**Date:** 2026-05-12
**Hardware:** Intel Arc Pro B70 (Battlemage, BMG-G31, PCI ID 0xe223, 32 GB VRAM)
**Target model:** `cyankiwi/Qwen3.5-9B-AWQ-4bit` (actually compressed-tensors INT4 pack-quantized, has Mamba `linear_attn` layers + MTP head — effectively Qwen3.6-class hybrid architecture)
**Workload:** VSA (Video Story Analyzer) multi-image VLM with 8 concurrent requests

Three attempts to run this model on vLLM. All failed at different layers, but the **common root cause is Mamba/linear_attn on Battlemage Xe2 silicon**.

---

## Attempt 1: `intel/llm-scaler-vllm:0.14.0-b8.2.1` (Docker, May 2026)

**Setup:** Standard intel/llm-scaler invocation, AWQ → `compressed-tensors` quant arg fixup, mounted model dir RO.

**Failure:** IPEX-XPU's `torch.xpu.has_xmx()` doesn't recognize Battlemage silicon.

```python
File "/usr/local/lib/python3.12/dist-packages/intel_extension_for_pytorch/xpu/utils.py", line 28, in has_xmx
    return _C._has_xmx(device)
```

Earlier in startup logs: `sycl_arch not recognized: 21483225088`. This is the device ID for BMG-G31 that IPEX doesn't have a code path for.

The IPEX woq_linear (weight-only quant linear) path is what's hit by Qwen3.5/3.6's `linear_attn` Mamba layers. Without XMX detection, the path fails initialization.

**Verdict:** Image was built before full Battlemage IPEX support landed. Not easily patchable from outside the container.

---

## Attempt 2: `intel/vllm:0.17.0-xpu` (Docker, Mar 2026)

**Setup:** Different image lineage; doesn't use IPEX at all — relies on native `torch.xpu` (PyTorch 2.10+xpu). transformers 4.57.6 has `Qwen2VLImageProcessor.max_pixels` so the earlier image-processor compatibility bug is moot.

**Failure:** FLA (Flash Linear Attention) ops have CUDA hardcoded in their op-wrapper context manager.

```python
File "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fla/ops/utils.py", line 112, in wrapper
    with ctx:
File "/opt/venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 550, in __enter__
    self.prev_idx = torch.cuda._exchange_device(self.idx)
RuntimeError: PyTorch was compiled without CUDA support
```

`ChunkGatedDeltaRuleFunction.apply` (Qwen3.5/3.6's Mamba layer kernel) wraps with `torch.cuda.device(idx)` — hardcoded CUDA context manager regardless of platform. This is upstream FLA's bug, not Intel-specific.

**Verdict:** Same model class as #1 failed at a different layer. Both confirm Mamba-on-XPU isn't validated.

---

## Attempt 3: Bare-metal pip install (May 2026)

**Setup:**
```
python3 -m venv /mnt/optane/LLMs/vllm-env
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu  # → 2.12.0+xpu
pip install vllm                                                                 # → 0.20.2, downgrades torch
pip install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/xpu
pip install vllm_xpu_kernels                                                     # → 0.1.3.1
pip install --force-reinstall --no-deps triton-xpu==3.7.1 --index-url https://download.pytorch.org/whl/xpu
# (vllm's stub `triton` package conflicts with `triton-xpu`; uninstall the stub)
```

**Failures encountered en route:**
- vLLM packaging: imports `supports_xccl` from `vllm.utils.import_utils` (wrong module — actually in `vllm.utils.torch_utils`). False alarm; patched then reverted once we discovered the real cause.
- Missing `vllm_xpu_kernels` package (needed since vLLM 0.20+). Pip-installable, fixed.
- `triton` (stub, CUDA-only) vs `triton-xpu` (real, Intel backend) — same `triton` import name, ABI conflict. Must uninstall the stub.
- `VLLM_TARGET_DEVICE=xpu` is build-time only; runtime device autodetect via `torch.xpu.is_available()` works without it.

**Result after fixes:** vLLM detects XPU (good — `vllm.platforms.xpu.XPUPlatform`), loads config, resolves architecture as `Qwen3_5ForConditionalGeneration` (good — model recognized), then **EngineCore subprocess exits silently after `parallel_state.py:1715 rank assignment`**. No traceback in stdout/stderr. Suspected location: model forward `_dummy_run` during memory profiling, same path as Attempt 2 but with newer FLA code that may or may not handle XPU correctly (still TBD without `VLLM_LOGGING_LEVEL=DEBUG` rerun).

Notable: vLLM 0.20.2's `splitting_ops` config includes `vllm::gdn_attention_core_xpu` (XPU-specific Gated Delta Net attention) and `vllm::linear_attention` — the XPU code paths exist on paper. Something just dies between rank assignment and model load.

**Verdict:** Bleeding edge. Maybe one more debug session away from working, maybe many. Stack-tracing the silent EngineCore death would be the next step if pursuing.

---

## Pattern across all three attempts

All three failed at the **Mamba/linear_attn quantized layer** path:

| Layer | Where it broke in each attempt |
|---|---|
| 1. IPEX woq_linear | Battlemage XMX detection missing |
| 2. FLA chunk op | CUDA-only context manager |
| 3. vLLM 0.20.2 native XPU FLA | EngineCore silent exit (TBD) |

The hybrid Mamba+attention architecture of Qwen3.5/3.6 is the bleeding edge. Pure transformer models likely work on all three setups, but the user explicitly rejects regression to older Qwen lines.

---

## Hardware confirmed working for non-Mamba paths

- **Device detection**: `sycl-ls`, `clinfo`, `torch.xpu.get_device_name(0)` all see the B70 correctly (32 GB VRAM, 256 EUs, FP16, FP64, atomic64)
- **Vulkan via Mesa ANV**: works flawlessly on dense transformer (Qwen3.6-27B Q6_K) and MoE (Nemotron 31B-A3B)
- **SYCL via oneAPI 2026.0**: works on dense models; bench shows 490 t/s pp on Q6_K dense 27B
- **llama-server with mmproj**: vision input works on both Vulkan and SYCL builds

So the silicon and base software stack are fine. **The gap is engine-side: Mamba/SSM kernels on Xe2 + INT4 weight quant on the same code path.**

---

## What the surrounding Intel ecosystem is shipping

- `gdn_attention_core_xpu` exists as an op in vLLM 0.20.2 splitting_ops list (May 2026) — XPU branch of gated delta net exists, just possibly buggy
- `vllm_xpu_kernels` package on PyPI is at 0.1.3.1 (May 2026) — early, not widely tested
- `intel/llm-scaler-vllm:0.14.0-b8.2.1` ships with IPEX that doesn't know about BMG-G31 — older IPEX in the container

Battlemage is new enough (~2026) that Intel's software stack is actively catching up. Some pieces are there (vllm_xpu_kernels, gdn_attention_core_xpu), others not (IPEX woq_linear Battlemage detection in latest container).

---

## Next steps (queued)

- Two background research agents (2026-05-12) investigating: (a) alternative engines/forks that might handle this model on B70, (b) which codebase to fork or build atop if no engine works out-of-box
- After agent results: pick a path (engine selection vs fork/build) and start tomorrow
- In parallel: run llama.cpp Vulkan VSA end-to-end overnight to get baseline GLS/QG outputs on at least one engine

## Files in our setup right now

- `/mnt/optane/LLMs/vllm-env/` — bare-metal pip venv (torch 2.11.0+xpu, vllm 0.20.2, triton-xpu 3.7.1, vllm_xpu_kernels 0.1.3.1)
- `/mnt/optane/LLMs/hf-models/cyankiwi__Qwen3.5-9B-AWQ-4bit/` — the target model (8.5 GB)
- `/tmp/vllm-start.sh` — vllm serve invocation script (works inside Docker containers)
- `/tmp/vllm-bare.log` — bare-metal pip vllm latest run log
- Docker images on disk: `intel/llm-scaler-vllm:0.14.0-b8.2.1`, `intel/vllm:0.17.0-xpu`

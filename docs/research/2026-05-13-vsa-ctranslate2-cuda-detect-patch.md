# VSA pipeline.py patch: ctranslate2 runtime CUDA gate

**Date:** 2026-05-13
**Scope:** Video_Analyzer/pipeline.py — two spots in transcription setup
**Effect on existing CUDA machines:** none (preserves prior behavior exactly)
**Effect on Intel/AMD GPU machines:** fixes silent transcription failure

## The bug

VSA's `_transcribe_faster_whisper` and `_transcribe_chunks_parallel` use ctranslate2 to detect compute capabilities:

```python
cuda_types = ctranslate2.get_supported_compute_types("cuda")
if cuda_types:
    device = "cuda"
```

`get_supported_compute_types("cuda")` returns the **compile-time supported types** for the installed ctranslate2 wheel, **not runtime availability**. The pip wheel ships with CUDA support compiled in regardless of whether the host has an NVIDIA GPU.

On Intel/AMD/CPU-only rigs:
- `cuda_types` is non-empty (compile-time supports CUDA)
- `if cuda_types:` is True
- `device = "cuda"` is selected
- WhisperModel tries to load on CUDA at runtime → `CUDA failed with error CUDA driver version is insufficient for CUDA runtime version`
- Falls back to whisper-cli (not installed by default)
- Returns 1 stub segment per chunk = silent failure that compromises analysis

This bit our Intel Arc B70 rig: out of 43 chunks, only chunk 0 transcribed correctly (3 segments). Chunks 1-42 returned ~1 segment each of placeholder. Final transcript: 3 real segments + 42 stubs = 207 chars total for a 90-minute video.

## The fix

Gate on **runtime device count** (newer ctranslate2 API) with fallback to compile-time check (preserves behavior on older ctranslate2). Also honor `CUDA_VISIBLE_DEVICES=""` explicitly.

```python
device = "cpu"
compute_type = "int8"
try:
    import os as _os
    cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_allowed = (cvd != "")  # empty string explicitly disables CUDA
    if cuda_allowed:
        try:
            cuda_runtime_ok = ctranslate2.get_cuda_device_count() > 0
        except AttributeError:
            cuda_runtime_ok = True  # legacy ctranslate2 behavior preserved
        if cuda_runtime_ok:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if cuda_types:
                device = "cuda"
                compute_type = "float16" if "float16" in cuda_types else "int8"
except Exception:
    pass
```

Applied to two locations:
1. `_transcribe_faster_whisper` (single-GPU path, line ~3500)
2. `_transcribe_chunks_parallel` (multi-GPU path, line ~3086)

Patch file: `Video_Analyzer/patches/transcribe-runtime-cuda-gate.patch`. Apply with `git apply patches/transcribe-runtime-cuda-gate.patch`.

## Behavior matrix

| Machine | `CUDA_VISIBLE_DEVICES` | `get_cuda_device_count()` | Old behavior | New behavior | Result |
|---|---|---|---|---|---|
| NVIDIA, CUDA working | unset | returns 1+ | uses CUDA | uses CUDA | ✅ unchanged |
| NVIDIA, CUDA working | `=""` (force CPU) | returns 1+ | uses CUDA (ignores!) | uses CPU | ✅ now respects user intent |
| NVIDIA, CUDA working, old ctranslate2 (no method) | unset | AttributeError | uses CUDA | falls back → uses CUDA | ✅ unchanged |
| Intel/AMD GPU (no CUDA) | unset | returns 0 | tries CUDA → fails | uses CPU | ✅ fixed |
| CPU-only | unset | returns 0 | tries CUDA → fails | uses CPU | ✅ fixed |

**Key invariant:** any path that produced `device="cuda"` in the old code still produces `device="cuda"` in the new code, *unless* the user explicitly set `CUDA_VISIBLE_DEVICES=""` (in which case respecting their intent is correct).

## Verification on the B70 rig

Before patch: 43/43 chunks transcribed, but 42 of them returned 1 stub segment after CUDA fallback failure. Total transcript: 207 chars.

After patch: 43/43 chunks transcribed on CPU, **675 real segments, 21,901 dialogue chars**, ~17.2 segments/chunk average. ~3 minutes wall time on Ryzen 5 7600X (CPU faster-whisper int8).

## Where else this matters

Anywhere `ctranslate2.get_supported_compute_types("cuda")` is used as a runtime detector — which is several other Python libraries beyond VSA. The pattern is common because the ctranslate2 docs/blog historically recommended it before `get_cuda_device_count()` was added. Worth checking other transcription/translation pipelines if they exhibit similar silent CUDA fallback bugs on Intel rigs.

## What this doesn't fix

- VSA's `hwdetect.py` still reports `"CPU only"` on Intel GPU rigs. That's cosmetic (the actual VLM stage uses vLLM-XPU regardless of hwdetect's output). Worth a separate patch later — detect Intel GPU via `/dev/dri` + `vainfo` or `sycl-ls`.
- whisper.cpp has a Vulkan backend that would actually use the B70 for transcription. Not wired into VSA today. ~50 LOC to add `whisper-cli --device vulkan` as a transcription engine option.

## Files changed

- `Video_Analyzer/pipeline.py` — two transcription setup blocks
- `Video_Analyzer/patches/transcribe-runtime-cuda-gate.patch` — the patch as a portable file
- `Video_Analyzer/config.profiles/intellm-b70.yaml` — documents the patch application step
- `Video_Analyzer/vsa-vllm.sh` — defense in depth: sets `CUDA_VISIBLE_DEVICES=""` in env before invoking ./vsa

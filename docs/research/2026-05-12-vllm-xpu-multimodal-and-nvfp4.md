# vLLM-XPU multimodal + NVFP4 on Intel Arc B70 — feasibility and plan

**Date:** 2026-05-12
**User ask:** "Download Qwen3.5 9B NVFP4 and get it working with vLLM with vision capabilities fully supported, so we can just run that on our intelLM thing."

---

## TL;DR

- **Qwen3.5-9B exists** (released 2026-02-16, vision-capable) and **NVFP4 community quants exist on HF** (e.g. `AxionML/Qwen3.5-9B-NVFP4`).
- **NVFP4 cannot run on Intel Arc B70 today** — no XPU kernel exists, no dequant-on-load path. NVFP4 is a Blackwell tensor-core format. Intel's `llm-scaler-vllm` README mentions only `MXFP4` (and only gated to gpt-oss-20b/120b).
- **Closest viable substitute:** `Qwen/Qwen3-VL-8B-Instruct-FP8` — official Qwen FP8, validated on llm-scaler-vllm 0.14.0-b8.2.1 for B70, ~9 GB on disk, vision tower stays BF16.
- **Path to integrate with `intellm`:** add an `engine` field to `builds.conf` so the launcher can dispatch llama.cpp builds vs vLLM containers from the same interface.

## What's actually downloadable

| Model | Notes |
|---|---|
| `Qwen/Qwen3.5-9B` | Real, vision-capable, BF16 base. Released 2026-02-16. |
| `AxionML/Qwen3.5-9B-NVFP4`, `ig1/...`, `ykarout/...` | Community NVFP4 quants (~6 GB). E2M1 FP4 LM linears with FP8 block scales; vision tower BF16. **Not first-party.** |
| `nvidia/Qwen3.5-397B-A17B-NVFP4` | Too big for B70 |
| `Qwen/Qwen3-VL-{4,8,30,235}B-...-FP8` | First-party Qwen FP8, multiple sizes |
| `RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4` | Big and NVFP4 (skip) |

## NVFP4 on B70: blocked

| Quant on B70 | Status | Source |
|---|---|---|
| FP8 dynamic | ✅ primary supported path | [llm-scaler README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md) |
| INT4 `sym_int4` (online) | ✅ | llm-scaler 1.3 notes (Qwen3 family) |
| GPTQ / AWQ pre-quantized | ✅ auto-detected | llm-scaler README |
| MXFP4 | ⚠ gated to `gpt-oss-20b/120b` only | llm-scaler README |
| **NVFP4** | ❌ **no XPU kernel, no on-load dequant** | absent from all Intel docs |

NVFP4 uses Blackwell E2M1 + FP8 block-scale tensor cores. Intel Xe2/Battlemage has no native FP4 multiplication. vLLM upstream XPU docs list only FP16 and dynamic FP8 as supported quantizations on XPU.

## vLLM-XPU multimodal status (May 2026)

llm-scaler-vllm `0.14.0-b8.2.1` officially supports these VLMs on B70:
- Qwen2-VL-7B, Qwen2.5-VL-{7B, 32B, 72B}
- **Qwen3-VL-{4B, 8B, 30B-A3B} Instruct**
- MiniCPM-V, InternVL3.5, Gemma-3, GLM-4v

Qwen3.5-9B itself (the user's literal target) is listed as text-only in the validated matrix — the validated Qwen3.5 multimodals are 27B, 35B-A3B, 122B-A10B sizes.

Vision encoder runs in BF16 on the device; image preprocessing is CPU torchvision/PIL → device.

Single-B70 is stable. **TP=2 across two B70s GP-faults** (vLLM [#41663](https://github.com/vllm-project/vllm/issues/41663)) — irrelevant for this single-card rig but worth flagging if a second B70 ever shows up.

## Recommendation

Run **`Qwen/Qwen3-VL-8B-Instruct-FP8`** on **`intel/llm-scaler-vllm:0.14.0-b8.2.1`**.

Ranked alternatives:
1. `Qwen/Qwen3-VL-8B-Instruct-FP8` — best fit (validated, official, right size)
2. `Qwen/Qwen2.5-VL-7B-Instruct` (BF16 + online `sym_int4`) — most battle-tested on XPU
3. `Qwen/Qwen3-VL-4B-Instruct-FP8` — smaller, more KV headroom

Revisit NVFP4 when Intel ships an `xe-fp4` kernel (not on the May 2026 public roadmap).

## Hands-on getting started

```bash
# Host prep
sudo usermod -aG render,video $USER
mkdir -p ~/models

# Pull and run the container
docker run -td --privileged --net=host \
  --device=/dev/dri \
  --group-add $(stat -c '%g' /dev/dri/renderD128) \
  -v ~/models:/llm/models \
  --shm-size=32g \
  -e ZE_AFFINITY_MASK=0 \
  -e CCL_ENABLE_SYCL_KERNELS=0 \
  --name lsv intel/llm-scaler-vllm:0.14.0-b8.2.1
```

Inside the container:
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --task generate \
  --dtype float16 \
  --quantization fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt image=4 \
  --tensor-parallel-size 1 \
  --port 8000
```

Test with an image:
```bash
curl http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" -d '{
  "model":"Qwen/Qwen3-VL-8B-Instruct-FP8",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"https://picsum.photos/512"}},
    {"type":"text","text":"Describe this image."}]}],
  "max_tokens":256}'
```

VRAM budget: FP8 weights ~9 GB + BF16 vision tower ~1.5 GB + activations/CUDA-graphs ~2 GB → ~19 GB left for KV. FP8 KV ≈ 60 KB/token, so 32K ctx ≈ 2 GB — comfortable with batch ≥ 8. Drop to `--max-model-len 16384` for video-frame headroom.

## intellm integration design

vLLM is container-based + server-only — different shape from llama.cpp. Forcing into the existing `builds.conf` schema is clunky. Cleanest path: add an `engine` field, dispatch in the launcher.

Proposed schema (one extra column):
```
# shortname | display                       | engine    | bin_or_image                              | env_setup_script        | extra_args
sycl        | llama.cpp SYCL B70            | llamacpp  | llama.cpp/build-sycl-intel/bin            | llama.cpp/env-sycl.sh   | --device SYCL0
vulkan      | llama.cpp Vulkan B70          | llamacpp  | llama.cpp/build-vulkan-intel/bin          |                         | --device Vulkan0
vllm-xpu    | vLLM-XPU B70 (FP8 multimodal) | vllm      | intel/llm-scaler-vllm:0.14.0-b8.2.1       | scripts/vllm-env.sh     | --quantization fp8 --max-model-len 32768
```

Dispatcher sketch (in `intellm`):
```bash
case "$engine" in
  llamacpp)
    case "$mode" in
      cli)    exec "$bin_or_image/llama-cli"    "${args[@]}" ;;
      server) exec "$bin_or_image/llama-server" "${args[@]}" ;;
      bench)  exec "$bin_or_image/llama-bench"  "${args[@]}" ;;
    esac
    ;;
  vllm)
    # chat mode: spin up server + an OpenAI-compatible REPL
    # server mode: just spin up the server
    # bench mode: invoke vLLM benchmark_serving.py
    docker run -d --rm --name intellm-vllm-$shortname \
      --privileged --device=/dev/dri --shm-size=32g \
      -v "$MODELS_DIR:/llm/models" -p "$PORT":8000 \
      "$bin_or_image" vllm serve "$model" $extra_args
    [ "$mode" = chat ] && exec python3 scripts/openai_repl.py "http://localhost:$PORT/v1"
    ;;
esac
```

Estimated effort: ~80 LoC of bash + ~30 LoC of an OpenAI-compatible REPL for chat mode.

## Sources

- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3-VL collection (HF)](https://huggingface.co/collections/Qwen/qwen3-vl)
- [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8)
- [AxionML/Qwen3.5-9B-NVFP4](https://huggingface.co/AxionML/Qwen3.5-9B-NVFP4)
- [intel/llm-scaler vllm README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md)
- [intel/llm-scaler releases](https://github.com/intel/llm-scaler/releases)
- [Phoronix: llm-scaler vllm-0.14.0-b8.2 + Arc Pro B70](https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2)
- [vLLM XPU supported models](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/)
- [vLLM #41663 — XPU TP=2 dual B70](https://github.com/vllm-project/vllm/issues/41663)
- [vLLM blog: Arc Pro B-Series](https://vllm.ai/blog/intel-arc-pro-b)

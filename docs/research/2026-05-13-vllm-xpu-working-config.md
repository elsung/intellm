# vLLM-XPU on Battlemage B70 with Qwen3.5-9B-AWQ-4bit — WORKING CONFIG

**Date:** 2026-05-13 (00:19 PT)
**Status:** ✅ Inference working. Vision smoke test passed. VSA end-to-end run in progress.

After three failed vLLM attempts (documented in `2026-05-12-vllm-xpu-qwen35-battlemage-failures.md`) and two research streams, the combination of fixes below got vLLM-XPU running this hybrid Mamba+attention model.

## The working incantation

**Image:** `intel/llm-scaler-vllm:0.14.0-b8.2.1` (the IPEX-based container — yes, despite IPEX being EOL March 2026, this lineage works today for our model)

**Three pieces had to be combined:**

### 1. Patched `qwen2_vl.py` with `getattr` fallback for `max_pixels`

The container's transformers `5.7.0.dev0` removed `max_pixels` from `Qwen2VLImageProcessor.__init__`. The container's vLLM `0.14.1.dev0` still references `image_processor.max_pixels` directly in `qwen2_vl.py:875` and `:944`. AttributeError on init.

Patch (extracted from container, sed-applied, volume-mounted RO):

```python
# Before (qwen2_vl.py:875, :944):
max_pixels=image_processor.max_pixels,
max_pixels = image_processor.max_pixels or image_processor.size["longest_edge"]

# After:
max_pixels=getattr(image_processor, "max_pixels", None) or image_processor.size["longest_edge"],
max_pixels = getattr(image_processor, "max_pixels", None) or image_processor.size["longest_edge"]
```

Volume mount: `-v /tmp/qwen2_vl.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_vl.py:ro`

### 2. `--enforce-eager` (skip CUDA graph capture entirely)

CUDA graph capture isn't relevant on XPU, but the code path was apparently triggering paths that broke. `--enforce-eager` skips it.

### 3. `VLLM_ATTENTION_BACKEND=TRITON_ATTN` env

Forces the Triton attention backend instead of FlashAttention. Avoids whatever was crashing the EngineCore subprocess in our earlier attempts.

## Full docker run command

```bash
docker run -d --name vllm-xpu --net=host --shm-size=32g \
  --device=/dev/dri --group-add $(stat -c "%g" /dev/dri/renderD128) \
  -v /mnt/optane/LLMs/hf-models:/llm/models:ro \
  -v /tmp/vllm-start.sh:/start.sh:ro \
  -v /tmp/qwen2_vl.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_vl.py:ro \
  -e ZE_AFFINITY_MASK=0 \
  -e CCL_ENABLE_SYCL_KERNELS=0 \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  -e VLLM_USE_V1=1 \
  --entrypoint /start.sh \
  intel/llm-scaler-vllm:0.14.0-b8.2.1
```

Where `/tmp/vllm-start.sh` is:

```bash
#!/bin/bash
exec vllm serve /llm/models/cyankiwi__Qwen3.5-9B-AWQ-4bit \
  --served-model-name cyankiwi/Qwen3.5-9B-AWQ-4bit \
  --quantization compressed-tensors \
  --dtype float16 \
  --enforce-eager \
  --max-num-seqs 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image": 30}' \
  --tensor-parallel-size 1 \
  --port 8000
```

## Performance (initial smoke test)

Single image-text request, 200 token output, **cold cache, batch=1, no concurrency:**
- **Cold start (container up → `/v1/models` responds):** 81 seconds
- **Inference latency:** 26 seconds for 200 tokens out (252 prompt tokens in)
- **Effective throughput:** ~7.7 tokens/sec
- **VRAM usage:** ~30 GB (0.85 × 32 GB pool + ~2 GB activations)
- **System RAM:** 11 GB container resident

This is **batch=1 latency**, not aggregate throughput. With `--max-num-seqs 8` and continuous batching engaged, VSA's 8 concurrent VLM requests should see significantly better aggregate throughput. Real VSA timing pending.

## Output quality

Vision description of test frame from "Spark Me Tenderly":

> "1. Identify the main subject: A young woman with dark hair pulled back.
> 2. Describe her action: She is holding a stack of folders or books (one is bright blue/teal).
> 3. Describe her expression: Her mouth is open, and her eyes are wide, suggesting she is shouting, singing, or reacting with surprise.
> 4. Describe the background: It looks like an office or a public space with other people behind her, slightly out of focus.
> 5. Synthesize into 2-3 sentences:
>    - Sentence 1: A young woman is shown in a close-up, holding a stack of folders or books.
>    - Sentence 2: She has a dramatic expression with her mouth open wide, as if she is shouting or singing loudly.
>    - Sentence 3 (optional but good for context): The"

Coherent, accurate (matches the llama.cpp description from earlier session). Model is using a "structured-thought" pattern; should be overridden in VSA by the compact prompt style + `chat_template_kwargs.enable_thinking=false`.

## Why the bare-metal vLLM 0.20.2 attempt didn't work but this does

Hypothesis: the bare-metal `vllm==0.20.2` attempt died silently in EngineCore for a different reason — possibly the FLA `chunk_gated_delta_rule` path being hit but with broken XPU handling in the newer vllm-xpu-kernels version. The container's older vLLM `0.14.1.dev0` may have a more conservative attention dispatch that the `TRITON_ATTN` backend env-var routes through a working code path.

Worth re-investigating, but lower priority now that we have a working production setup.

## Open issues to track

1. **[vllm#38994](https://github.com/vllm-project/vllm/issues/38994)**: Qwen3.5-9B repetitive output on Intel XPU. Apply generation config override from cyankiwi HF discussion #2 in actual VSA requests.
2. **[intel-xpu-backend-for-triton#6658](https://github.com/intel/intel-xpu-backend-for-triton/issues/6658)**: DEVICE_LOST on BMG GDN decode kernel. May still bite us at long contexts or under sustained decode load.
3. **MTP head**: still unsupported on XPU. Don't budget for speedup.
4. **TP=2** ([vllm#41663](https://github.com/vllm-project/vllm/issues/41663)): irrelevant for single B70.

## What this means for the user's plans

- **Tonight's goal achieved:** vLLM-XPU + Qwen3.5-9B-AWQ-4bit + Battlemage = working serving stack
- **VSA can now A/B against llama.cpp** with this same model class (different quant: compressed-tensors INT4 on vLLM vs Q4_K_XL GGUF on llama.cpp)
- **The 3-month fork plan from agent B becomes optional** — only revisit if vLLM perf turns out to be a deal-breaker
- **The FLA `torch.cuda.device()` patch is still worth contributing upstream** — would fix `intel/vllm:0.17.0-xpu` too, and the patch is ~10 LOC

## Lessons

1. **Combining patches matters.** Each individual fix didn't make it work in isolation — we needed all three (qwen2_vl patch + enforce-eager + TRITON_ATTN backend). Don't dismiss attempts that fix one bug without testing if a deeper bug now manifests.
2. **`--enforce-eager` for any XPU vLLM run** — there's no real CUDA-graph equivalent on XPU and enabling capture seems to expose more bugs than it helps.
3. **The IPEX-based container works today despite IPEX being EOL.** For prototyping/serving in May 2026, prefer the working setup over the architecturally-cleaner one. Migrate when forced.
4. **Spend the agent-credits early in the debugging journey.** The breakthrough flags (`--enforce-eager`, `TRITON_ATTN`) came from Agent A's research late in the session. Earlier research could have shortened the debugging by hours.

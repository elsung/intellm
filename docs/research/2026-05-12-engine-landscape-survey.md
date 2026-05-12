# KV cache offloading landscape — survey for Intel Arc B70

**Date:** 2026-05-12
**Investigator:** background research agent
**Brief:** Find every production-grade KV-tiering stack and assess Intel GPU compatibility. Cite real repos, commits, issues, papers — no hand-waving.

---

## TL;DR

Almost every production-grade KV-tiering stack today (LMCache, llm-d, vLLM OffloadingConnector, SGLang HiCache, Mooncake, TurboQuant) is **CUDA-only** in its hot path. Intel XPU support is either absent or limited to the base inference engine, not the KV-tiering layer.

**The single standout that combines Intel Arc support AND a real disk tier: KTransformers.**

---

## Per-project findings

### LMCache ([github](https://github.com/LMCache/LMCache))

| | |
|---|---|
| Tiers | GPU / CPU / local-disk / S3 / NIXL — multi-tier is real and documented |
| Maintained | yes — v0.4.4 Apr 22 2026, PyTorch Ecosystem since Feb 2026 |
| Backends | **CUDA-only** ("Works on Linux NVIDIA GPU platform") — no XPU/Vulkan/SYCL |
| Perf | vendor claims 3–10× TTFT reduction; llm-d numbers up to 16.8× |
| Modularity | Storage plugin interface exists; data-path side assumes CUDA tensors / `cudaMemcpyAsync` |

### KTransformers ([github](https://github.com/kvcache-ai/ktransformers))

| | |
|---|---|
| Tiers | **3-layer GPU/CPU/Disk prefix cache reuse** landed June 30 2025 |
| Maintained | very active — v0.6.2 May 3 2026; major MoE/DeepSeek focus |
| Backends | **"Intel Arc GPU" support added May 2025** plus AVX2/AMX CPU — only mainstream KV-tiering project listing Arc |
| Perf | "139K context on DeepSeek-V3 in 24 GB VRAM" (Mar 2025 release notes) |
| Modularity | Heterogeneous-by-design; offload manager is a distinct module. Best fit for this hardware. |

### vLLM core `OffloadingConnector`

| | |
|---|---|
| Tiers | **CPU only today**; disk listed as future work |
| PRs | [#24498](https://github.com/vllm-project/vllm/pull/24498) `cpu_bytes_to_use`, [#29870](https://github.com/vllm-project/vllm/pull/29870) preemption fix, [#31341](https://github.com/vllm-project/vllm/pull/31341) race fix; [blog Jan 8 2026](https://vllm.ai/blog/kv-offloading-connector) |
| Maintained | yes — landed in vLLM 0.11.0, refined in 0.12.0/0.14.0 |
| Backends | CUDA + ROCm; **no XPU path** |
| Perf | vendor: 2–22× TTFT improvement on CPU hits |
| Modularity | Clean `KVConnector` interface (`vllm/distributed/kv_transfer/`) |

### vLLM-XPU / `intel/llm-scaler-vllm`

| | |
|---|---|
| Tiers | VRAM only in the upstream KV path. Release notes mention "CPU KV cache offload" in 0.11.1 but no LMCache/OffloadingConnector wiring documented |
| Maintained | yes — v0.14.0-b8.2 (Apr 22 2026) **officially adds Arc Pro B70 support** (Phoronix) |
| Backends | SYCL/oneAPI/Level Zero — this is the engine that actually runs on the B70 |
| Perf | 540 tok/s Qwen3.5-27B BF16 at TP=4 on 4× B70 (vendor); bug [#41663](https://github.com/vllm-project/vllm/issues/41663): TP=2 on dual B70 crashes |
| Modularity | vLLM fork — same `KVConnector` plumbing in theory; no upstream evidence anyone has wired XPU into the connector |

### SGLang RadixAttention / HiCache

| | |
|---|---|
| Tiers | GPU(L1)/CPU(L2)/remote(L3); HiCache backends are Mooncake, 3FS, NIXL + local-file reference |
| Maintained | very active ([LMSYS blog Sept 10 2025](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)) |
| Backends | CUDA-centric; no Intel GPU support documented |
| Perf | vendor: 6× throughput, 80% TTFT reduction |
| Modularity | **Best abstraction of any project** — only `get/exist/set` to add a backend (HiRadixTree handles scheduling). Tensor-transfer kernels assume CUDA |

### LMDeploy / TurboMind
Eviction → recompute from token IDs ("logical infinite cache"), not actual KV offload. **No disk tier.** CUDA/ROCm; no XPU. Not designed for this. Skip.

### TurboQuant (Zandieh et al., ICLR 2026)

| | |
|---|---|
| What | **Quantization-only**, not an offload project. 3-bit K, 2-bit V |
| Forks | 0xSero/turboquant, **AmesianX/TurboQuant** (llama.cpp port), varjoranta/turboquant-vllm |
| Backends | CUDA Triton kernels; AmesianX llama.cpp port is the only non-CUDA-ish option |
| Perf | ~2× effective KV capacity, ~5× memory reduction |
| Modularity | **Orthogonal to tiering — stack it on top of** the chosen offload engine |

### llama.cpp (SYCL + Vulkan on B70)

| | |
|---|---|
| Tiers | VRAM + CPU RAM via `-ngl` partitioning; `--no-kv-offload` keeps KV on CPU. **No disk-tier KV.** Slot save/restore (`/slot/save`, `--slot-save-path`) writes whole slots to disk; [#20572](https://github.com/ggml-org/llama.cpp/discussions/20572) covers persistent per-session cache. [#17107](https://github.com/ggml-org/llama.cpp/issues/17107) (disk KV persistence) **closed as not planned** |
| Maintained | extremely active master |
| Backends | SYCL works on B70 — **[issue #21893](https://github.com/ggml-org/llama.cpp/issues/21893) requires `GGML_SYCL_DISABLE_OPT=1`** historically; [issue #21517](https://github.com/ggml-org/llama.cpp/issues/21517) Q8_0 ~4× slower than Q4_K_M. Vulkan works but 2× slower decode than SYCL on dense models per PMZFX |
| Perf | 54.7 tok/s Qwen3.6-35B-A3B Q4_K_M on single B70; tg128 stable to 64K |
| Modularity | `src/llama-kv-cache*.cpp` is a focused module — adding a CPU/SSD spill tier is a contained patch, not a rewrite |

### Mooncake ([github](https://github.com/kvcache-ai/Mooncake))
Disaggregated CPU/DRAM/SSD KV pool. PyTorch ecosystem Feb 2026. Used **via** SGLang HiCache. CUDA-only data path. [RFC #578](https://github.com/kvcache-ai/Mooncake/issues/578) covers SSD offload.

### Others (briefly)

- **AirLLM** — layer-by-layer **weight** streaming from disk, not KV tiering. Wrong tool.
- **FlexGen / FlexLLMGen** — last meaningful activity 2024. Abandoned.
- **PowerInfer / PowerInfer-2** — sparse weight activation; doesn't tier KV.
- **InfiniGen** (OSDI '24) — research artifact, 4 commits, no releases. CUDA. Don't deploy.
- **vAttention** — memory allocator alternative to PagedAttention, A100/CUDA-only.
- **H2O / StreamingLLM** — eviction policies, not tiering. Stack on top of offload engine.
- **llm-d** — vLLM-based filesystem connector; v0.5 Feb 2026 adds hierarchical offload. NVIDIA-only tested.
- **MLC-LLM** — Vulkan backend works on Arc, but **no hierarchical KV tiering** — VRAM only.

---

## Top candidates for Intel B70

### 1. KTransformers — highest payoff

Only mainstream project with both **explicit Intel Arc support (May 2025 release)** and **GPU/CPU/Disk 3-tier KV reuse**. Aligns with the user's hardware on both axes.

**Tomorrow's step:** `git clone https://github.com/kvcache-ai/ktransformers; git checkout v0.6.2`. Follow the Intel Arc install path. Run their DeepSeek-V3 long-context example with `--cache-disk-path` pointed at `/mnt/optane`. Confirm whether the Arc backend uses the same disk-cache code path as CUDA (read `ktransformers/operators/` and the prefix cache module).

### 2. llama.cpp SYCL + custom CPU/SSD KV tier

Already works on B70; `src/llama-kv-cache*.cpp` is a tight surface (~few thousand lines); Optane's 10 µs latency makes a write-back tier viable.

**Tomorrow's step:** Baseline 32K/64K with `-ctk q8_0 -ctv q8_0 --no-kv-offload`. Then sketch a patch to `llama_kv_cache_unified` that LRU-evicts oldest-block KV to an mmap'd file on the Optane volume. The existing slot save/restore code (`/slot/save`) is your reference implementation.

### 3. vLLM-XPU + port LMCache CPU connector

The B70 is officially supported in llm-scaler-vllm 0.14.0-b8.2; LMCache's storage-plugin interface is well-defined. Wire the LMCache CPU/disk plugin against XPU tensors via `torch.xpu.Stream` and IPEX `to('xpu')`.

**Tomorrow's step:** Pull `intel/vllm:0.17.0-xpu` Docker image; verify single-card serving on B70 (avoid TP=2 — issue #41663 is open). Then clone LMCache and grep `csrc/` for `cudaMemcpyAsync` — every occurrence is a place that needs an XPU equivalent. Estimate the port size before committing.

**Skip for now:** SGLang HiCache (cleanest abstraction but no XPU), llm-d (NVIDIA tested only), TurboQuant (orthogonal — apply after picking the offload engine), AirLLM / FlexGen / PowerInfer (wrong problem or dead).

## Source list (key links)

- [LMCache](https://github.com/LMCache/LMCache) — [architecture docs](https://docs.lmcache.ai/developer_guide/architecture.html)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [vLLM RFC #19854 KV cache offloading](https://github.com/vllm-project/vllm/issues/19854) — [RFC #16144 V1 CPU offload](https://github.com/vllm-project/vllm/issues/16144) — [blog Jan 2026](https://vllm.ai/blog/kv-offloading-connector)
- [intel/llm-scaler vLLM README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md) — [Phoronix B70 support](https://www.phoronix.com/news/Intel-LLM-Scaler-vllm-0.14-b8.2)
- [SGLang HiCache blog (LMSYS)](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [Mooncake](https://github.com/kvcache-ai/Mooncake) — [issue #578 SSD offload](https://github.com/kvcache-ai/Mooncake/issues/578)
- [llm-d FS KV offloading blog](https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d)
- [llama.cpp #21893 SYCL B70 corruption](https://github.com/ggml-org/llama.cpp/issues/21893) — [#17107 disk KV persistence (closed)](https://github.com/ggml-org/llama.cpp/issues/17107) — [discussion #20572](https://github.com/ggml-org/llama.cpp/discussions/20572)
- [TurboQuant llama.cpp port (AmesianX)](https://github.com/AmesianX/TurboQuant)
- [PMZFX B70 benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks/blob/master/FINDINGS.md)

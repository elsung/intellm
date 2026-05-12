# Architectural deep-dive: adding disk-tier KV offload to llama.cpp and vLLM-XPU

**Date:** 2026-05-12
**Investigator:** background research agent (briefed against current llama.cpp master ~bbeb89d and intel/llm-scaler-vllm vllm-0.14.0-b8.2)
**Target hardware:** Intel Arc B70 32 GB + Optane SSD at `/mnt/optane` (~10 µs)
**Goal:** effective KV cache > 32 GB by tiering across VRAM → CPU RAM → Optane

---

## PART 1 — llama.cpp

### 1. Where KV lives

- `src/llama-kv-cache.{h,cpp}` defines `llama_kv_cache : llama_memory_i` with sibling `llama_kv_cache_iswa` (sliding-window) and `llama_memory_hybrid` (Mamba/Jamba).
- Per-layer K and V `ggml_tensor`s, grouped into `ctxs_bufs` (vector of `ggml_context_ptr` + `ggml_backend_buffer_ptr`).
- Constructor (`llama-kv-cache.cpp:~107`) selects `ggml_backend_cpu_buffer_type()` when `offload=false`, else `ggml_backend_dev_buffer_type(model.dev_layer(il))` (the SYCL/Vulkan device buffer).
- Actual allocation at line ~309: `ggml_backend_alloc_ctx_tensors_from_buft(buft)`.
- Access path: `get_k()`, `get_v()`, `cpy_k()`, `cpy_v()`. Index updates via `set_input_k_idxs()` / `set_input_v_idxs()`.

### 2. Per-backend storage — critical finding

- **SYCL**: `ggml_backend_sycl_buffer_type_alloc_buffer()` calls `sycl::malloc_device(size, *stream)` — **device-local only, not host-accessible**. No USM-host or shared-USM path in the default allocator.
- **Vulkan**: VRAM via `vkAllocateMemory` with `DEVICE_LOCAL` heap — also not host-mapped on dGPUs.
- Both require explicit `memcpy` through the stream/queue for any host access.

### 3. In-flight community work

| Issue / PR | Topic | Status |
|---|---|---|
| [#20697](https://github.com/ggml-org/llama.cpp/issues/20697) | `--cache-disk` feature request (Mar 2026) | Open, no PR |
| [#17107](https://github.com/ggml-org/llama.cpp/issues/17107) | Persistent KV on disk for llama-server | Partial: `--slot-save-path` saves whole slots via REST endpoints, **not per-block transparent spill** |
| [PR #20993](https://github.com/ggml-org/llama.cpp/pull/20993) | `LLAMA_KV_KEEP_ONLY_ACTIVE=1` env: clears idle slot KV from VRAM | Merged ~Mar 2026 — cleared, not spilled |
| [#20572](https://github.com/ggml-org/llama.cpp/discussions/20572) | Server hooks for persistent KV (userspace workaround) | Discussion |
| [#20140](https://github.com/ggml-org/llama.cpp/issues/20140) | KV corruption with `--cpu-moe + --ngl 999`; workaround is `-nkvo` | Open |
| [#20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | TurboQuant — extreme KV quant | Orthogonal but reduces pressure |
| [arXiv 2511.11907](https://arxiv.org/html/2511.11907v1) | "KVSwap" NVMe-aware KV offload | Built on FlexGen, **not llama.cpp**. ~6.9 ms/block latency. Design reference only. |

**No PR exists for transparent VRAM→RAM→disk tiering in llama.cpp.**

### 4. Minimal disk-tier sketch (pseudocode)

```cpp
struct kv_block {
  tensor_view k, v;
  int layer;
  enum {VRAM, RAM, DISK} tier;
  uint64_t lru;
};

class kv_tier_mgr {
  std::unordered_map<block_id, kv_block> blocks;
  intrusive_list<block_id> vram_lru, ram_lru;

  // Hook ggml_backend_sched: before graph compute, walk needed (layer, pos)
  // blocks and fault any needed ones into VRAM.
  void fault_in(block_id b) {
    if (b.tier == DISK) { pread(fd_optane, ram_buf, ...); b.tier = RAM; }
    if (b.tier == RAM)  { ggml_backend_tensor_set_async(dev, vram_tensor, ram_buf, ...); b.tier = VRAM; }
  }
  void evict_vram() { auto v=vram_lru.pop_lru(); ggml_backend_tensor_get_async(...,ram_buf,...); v.tier=RAM; }
  void evict_ram()  { auto r=ram_lru.pop_lru(); pwrite(fd_optane, ram_buf, ...);                  r.tier=DISK; }
};

// Hook point: llama_kv_cache::get_k/get_v wraps tier_mgr.fault_in(...) before returning view.
```

**Real-implementation LOC estimate: ~1.5–2.5k LoC + tests.** Touches `llama-kv-cache.{h,cpp}`, a new `llama-kv-tier.{h,cpp}`, plus a hook in `llm_graph_context::build_attn_inp_kv()` to declare per-layer-per-token block dependencies before graph build.

### 5. What breaks

- **FlashAttention path**: `ggml_flash_attn_ext` takes whole K/V tensors as inputs; faulting in only "needed blocks" requires reconstructing K/V as a gather or running attention in tiles per fault region. **Real blocker for fine-grained spill.** Mitigation: spill at sequence/slot granularity, not block.
- **SYCL backend `GGML_SYCL_DISABLE_OPT=1`**: when set, disables reorder extra buffers for quantized types. KV is usually F16/Q8 — fine. But disabling opt slows compute ~5–15% globally.
- **CUDA graph capture**: not used by SYCL/Vulkan — irrelevant for B70.
- **Batched prompt processing (`llama-batched-bench`)**: writes whole K/V over a span; spilled blocks would need eager fault-in for the write range. Manageable.
- **`v_trans` mode**: V is transposed for some backends; per-block layout differs — eviction granularity must be aware.

---

## PART 2 — vLLM-XPU (intel/llm-scaler-vllm)

### 1. vLLM core today — CRITICAL FINDING

- **`swap_space` is dead code** ([issue #27984](https://github.com/vllm-project/vllm/issues/27984), Nov 2025).
- `num_cpu_blocks` is hard-coded to 0 in `EngineCore._initialize_kv_caches()`.
- `best_of` deprecation ([PR #13997](https://github.com/vllm-project/vllm/pull/13997)) killed the swap path.
- **Preemption is now recompute-only by default.**
- **Replacement: KV Offloading Connector** (vLLM 0.11.0, [blog 2026-01-08](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)) — `KVConnectorBase_V1` plugin API with async load/store, bundled CPU backend. Plumbed via `--kv-transfer-config`. **This is the only living "swap tier" mechanism in mainline vLLM.**

### 2. vLLM-XPU specifics

- llm-scaler-vllm tracks upstream vLLM closely.
- KV blocks live in `torch.xpu` tensors backed by IPEX (transitioning to `vllm-xpu-kernels`, [RFC #33214](https://github.com/vllm-project/vllm/issues/33214)).
- CCL supports P2P and USM modes.
- **No CPU-KV-offload path is exercised on XPU today** — release notes through vllm-1.3 (Jan 2025) and beta vllm-0.14.0-b8.2 (May 2026) mention FP8 KV and prefix caching, but no offload connector validation on XPU.
- CPU↔XPU copy: `tensor.to('cpu')` → IPEX runtime → Level Zero `zeCommandListAppendMemoryCopy`.
- IPEX exposes a `pin_memory()` analog but only on recent versions; not all paths use it.

### 3. Existing disk-tier work

| Project | What it does | XPU status |
|---|---|---|
| [**LMCache v0.4.4**](https://github.com/LMCache/LMCache) (Apr 22 2026) | Local-disk backend — one file per chunk, LRU/LFU/FIFO eviction, async PUT, sync GET, prefetch. Multi-path sharding (commit `d386614`, Apr 2 2026). GDS + /dev/dax. Plugs into vLLM via `LMCacheConnectorV1`. | **CUDA/ROCm/HPU only.** HPU support landed in v0.4.3 → precedent that a new accelerator can be added via the connector layer. |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | Tiered VRAM/DRAM/SSD; joined PyTorch Ecosystem Feb 2026; [RFC #578](https://github.com/kvcache-ai/Mooncake/issues/578) tracks SSD offload to DFS | Heavier (distributed-store oriented), not single-box-friendly |
| | No XPU-aware disk-tier fork of vLLM exists. | |

### 4. Realistic paths, ranked

| # | Approach | Effort | Notes |
|---|---|---|---|
| **b** | LMCache + vLLM-XPU with local-disk backend | 2–4 weeks | Port LMCache's GPU adapter to XPU (~500–1500 LoC), model after HPU adapter, swap `torch.cuda` → `torch.xpu`, swap `cudaMemcpyAsync` → IPEX equivalent |
| **d** | Custom `KVConnectorBase_V1` implementation in llm-scaler-vllm | 1–2 weeks | Implement async PUT to Optane via `io_uring`, GET fault-in via pinned XPU staging buffer. ~1k LoC. **Most upstream-friendly.** |
| c | Patch BlockManager swap directly | not recommended | Fighting dead code path; would need to revive `num_cpu_blocks` plumbing |
| a | PR new disk-tier into vLLM core | 3–6 months | Large surface area; must satisfy CUDA/ROCm/XPU/HPU |

### 5. XPU-specific blockers

- **Level Zero shared/host-visible memory**: `zeMemAllocShared` exists; IPEX exposes this only inconsistently. Without it, every fault is a device→host bounce.
- **No `pin_memory()` guarantee on XPU** for all tensor types — async copies may serialize.
- **vllm-xpu-kernels migration in flight** (RFC #33214): kernel signatures may shift; KV layout assumptions in any connector could break across releases.

---

## END VERDICT

- **1-week-of-evenings prototype**: **llama.cpp**, slot-granularity disk spill. Extend existing `--slot-save-path` machinery + `LLAMA_KV_KEEP_ONLY_ACTIVE=1` to *write* idle slots to Optane (`mmap`/`pwrite`) and fault them back on slot activation. Sidesteps FlashAttention entirely (whole-slot grain) and the SYCL host-accessibility problem (you already pay a device→host copy when serializing a slot). ~300 LoC. Gets ">32 GB effective KV" for multi-session workloads tonight.
- **2–3 month upstream-able**: **vLLM-XPU + new KVConnector** targeting Optane via `io_uring`, modeled on the bundled CPU offloading connector. Use LMCache's HPU adapter as the template for the XPU memcpy layer. Land it in llm-scaler-vllm first; once stable, propose merging the XPU adapter to LMCache upstream. Rides the only living offload API in vLLM, avoids touching dead `swap_space` plumbing.
- **Scariest blocker — sub-token-grain spill is dead on arrival**: SYCL/Vulkan KV buffers are not host-accessible by default (`sycl::malloc_device`, not `malloc_shared`). Every fault-in is XPU→host-staging→disk and back. With Optane at 10 µs the SSD isn't the bottleneck — **the GPU↔CPU PCIe roundtrip per fault is**. At Arc B70 PCIe 4.0 x8 (~16 GB/s practical) and ~5 MB per layer-block of KV, a fault is ~300 µs of DMA, not 10 µs of Optane. **Only slot/sequence-grain spill is viable.** If the workload is single-session very-long-context (not multi-session) — which is the implicit ask here — slot-grain doesn't help and the project should be deprioritized in favor of **KV quantization** (TurboQuant via discussion [#20969](https://github.com/ggml-org/llama.cpp/discussions/20969), Q4 KV) which buys ~4× effective KV for free.

## Key sources

- [llama.cpp #20697 — `--cache-disk` request](https://github.com/ggml-org/llama.cpp/issues/20697)
- [llama.cpp memory/KV architecture (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/3.6-memory-management-and-kv-cache)
- [llama.cpp discussion #17283 — KV swap + PR #20993](https://github.com/ggml-org/llama.cpp/discussions/17283)
- [llama.cpp #20140 — KV offload corruption](https://github.com/ggml-org/llama.cpp/issues/20140)
- [llama.cpp #20969 — TurboQuant extreme KV quant](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [llama.cpp #9302 — `-nkvo`](https://github.com/ggml-org/llama.cpp/issues/9302)
- [KVSwap paper (arXiv 2511.11907)](https://arxiv.org/html/2511.11907v1)
- [vLLM blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM #27984 — `swap_space` is dead code](https://github.com/vllm-project/vllm/issues/27984)
- [vLLM RFC #33214 — XPU kernel migration](https://github.com/vllm-project/vllm/issues/33214)
- [vLLM RFC #19854 — KV cache offloading](https://github.com/vllm-project/vllm/issues/19854)
- [LMCache local-disk backend docs](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html)
- [LMCache releases (v0.4.4, Apr 2026)](https://github.com/LMCache/LMCache/releases)
- [LMCache multi-path commit d386614](https://github.com/LMCache/LMCache/actions/runs/23905301241)
- [intel/llm-scaler vLLM README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md)
- [Mooncake](https://github.com/kvcache-ai/Mooncake) — [RFC #578](https://github.com/kvcache-ai/Mooncake/issues/578)

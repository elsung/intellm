# Synthesis: KV cache offloading on Intel B70 — what to actually do

**Date:** 2026-05-12
**Inputs:**
- [`2026-05-12-engine-landscape-survey.md`](./2026-05-12-engine-landscape-survey.md) — survey of every KV-tiering project that exists
- [`2026-05-12-llama-cpp-vllm-architecture.md`](./2026-05-12-llama-cpp-vllm-architecture.md) — engineering scope for extending llama.cpp / vLLM-XPU

---

## The hard truth (read this first)

Both investigations independently surface the same physics blocker:

**SYCL/Vulkan KV buffers on Intel dGPU are not host-accessible by default.** `sycl::malloc_device` and Vulkan `DEVICE_LOCAL` allocations require explicit DMA through the queue. Every fault-in is GPU → host-staging → disk and back. At PCIe 4.0 x8 (Arc B70's link) a ~5 MB block fault is ~300 µs of DMA — **30× longer than Optane's 10 µs latency**. The SSD is never the bottleneck.

The implication: **fine-grained (per-block, per-token) KV spill cannot beat in-VRAM execution on this hardware.** Only coarse-grained spill (whole-slot, whole-sequence) recovers the cost.

This rules out the naive mental model of "Optane as a third KV tier behind VRAM and RAM for live inference." That's not buildable today on Intel GPU.

## What's actually possible

| Workload | Best strategy | Why |
|---|---|---|
| **Single user, very long context** (e.g. 256K-token doc analysis) | **KV quantization** — `--cache-type-k q4_0 --cache-type-v q4_0`, plus possibly TurboQuant (3-bit K / 2-bit V) | Buys 4–5× effective KV in VRAM with no PCIe penalty. Already wired in `intellm`. Stack TurboQuant on top later if needed. |
| **Many users / sessions, short-medium ctx each** (chat server with 10+ concurrent users, RAG with reused contexts) | **Slot-grain disk spill** — write idle sessions' KV to Optane, fault back on slot reactivation | Coarse-grain spill amortizes the PCIe roundtrip across thousands of tokens of re-use. llama.cpp's `--slot-save-path` + `LLAMA_KV_KEEP_ONLY_ACTIVE=1` is 80% of this already. |
| **MoE with huge weights and long context** (DeepSeek-V3 class) | **KTransformers** — heterogeneous expert offload + 3-tier KV reuse | Designed for this exact problem. Has Intel Arc support since May 2025. Untested specifically on B70 — that's our risk to take. |

## Recommended sequence

### Phase 0 — Free wins, do now (no engineering required)

1. **Stay on llama.cpp + `--kv q4_0` for ultra-long-ctx single-session work.** Run a Qwen3.6-27B at 128K or 256K context with q4_0 KV and see if it fits. If yes, you're done — no offload needed.
2. **Apply TurboQuant if q4_0 isn't enough.** [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant) is the llama.cpp port; would need a rebuild against current master. Effort: clone, port to bbeb89d if needed, rebuild. Probably half a day.

### Phase 1 — Validate the "easy" wins (~1 evening)

3. **Try KTransformers v0.6.2 on the B70.** This is the highest-payoff exploration because if it works, we get production-grade GPU/CPU/Disk tiering for free.
   - Build with the Intel Arc backend path
   - Run their DeepSeek-V3 long-context example pointed at `/mnt/optane`
   - Confirm whether the Arc backend actually hits the 3-tier KV reuse code path (read `ktransformers/operators/` and check)
   - **Risk:** Arc support may be CPU-engine-only, with the offload manager still CUDA-bound. Verify before committing.

### Phase 2 — Slot-grain disk spill in llama.cpp (~1 week of evenings)

4. **Only if Phase 1 doesn't deliver and you have a multi-session use case.** Extend llama.cpp's existing slot save/restore into a transparent LRU writer:
   - Hook `LLAMA_KV_KEEP_ONLY_ACTIVE=1` (PR [#20993](https://github.com/ggml-org/llama.cpp/pull/20993)) to *write* idle slots to `/mnt/optane/kv-cache/slot-<id>.bin` instead of just clearing them
   - On slot reactivation, fault back via `pread` + `ggml_backend_tensor_set_async`
   - ~300 LoC. Sidesteps FlashAttention entirely (whole-slot grain) and the SYCL host-accessibility problem (you already pay a device→host copy when serializing a slot)
   - Best for: chat servers with many users, RAG with shared/cached prefixes

### Phase 3 — Upstream-able vLLM-XPU connector (~2–3 months)

5. **Only worth doing if vLLM-XPU becomes our primary serving engine.** Write a `KVConnectorBase_V1` plugin in `llm-scaler-vllm` targeting Optane via `io_uring`, modeled on the bundled CPU offloading connector (vLLM blog 2026-01-08). Use LMCache's HPU adapter as the template for the XPU memcpy layer.
   - Effort: ~1k LoC for the connector, plus IPEX `pin_memory` / Level Zero `zeMemAllocShared` integration
   - Output: candidate for upstream into both `llm-scaler-vllm` and (the XPU adapter piece) LMCache itself

## Open questions for empirical resolution

1. **Does q4_0 KV fit 256K ctx on Qwen3.6-27B with VRAM headroom?** (Estimate says yes at ~31 GB total VRAM — need to verify.)
2. **Does KTransformers' Arc support actually exercise the disk-tier code path?** Or is the disk tier CUDA-only with Arc only on the CPU compute side? Read the code, don't trust the release notes.
3. **What's the realistic worst-case slot-reactivation latency on llama.cpp + Optane?** Estimate based on PCIe and slot size; ratify with a small prototype.
4. **TurboQuant on B70 SYCL — does the AmesianX port build against current llama.cpp?** May need rebasing.

## Files referenced
- llama.cpp: `src/llama-kv-cache.{h,cpp}`, `src/llama-graph.cpp` (`build_attn_inp_kv()`), `ggml/src/ggml-{sycl,vulkan}/*.cpp`
- vLLM: `vllm/distributed/kv_transfer/` (KVConnector API), `vllm/core/block_manager.py` (dead swap_space path)
- LMCache: `csrc/` (CUDA memcpy assumptions), the HPU adapter as porting reference

## Bottom-line recommendation

**Don't start with the disk-tier project.** Start with **Phase 0** today — set `--kv q4_0` and try to fit your actual workload in VRAM. If it works, you've saved months of engineering. If it doesn't fit, **Phase 1 (KTransformers)** is the highest-EV next move. **Phase 2** (llama.cpp slot-grain spill) is only worth it if multi-session serving is the real use case; **Phase 3** is for if/when vLLM-XPU is the primary serving engine.

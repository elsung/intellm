# Design decisions log

Append-only. Each entry: date, decision in one line, then the reasoning at the time. If a decision is later changed, append a new entry that references and supersedes the old one.

---

## 2026-05-05 — Build Vulkan first, then SYCL

Started with Vulkan as primary because it works out of the box on Mesa 26.0.6 + Battlemage with `libvulkan-dev` (50 MB) vs SYCL's ~12 GB oneAPI Base Toolkit. Get a working baseline before paying the install cost for an optimized variant.

## 2026-05-06 — One source clone, multiple build dirs

`llama.cpp/src/` is shared across `build-vulkan-intel/`, `build-sycl-intel/`, future `build-cuda-p100/`, etc. CMake handles out-of-tree builds cleanly; avoids re-cloning and keeps pulls in one place.

## 2026-05-06 — Use `bmg_g31`, not `bmg_g21`, for SYCL AOT target

`-DGGML_SYCL_DEVICE_ARCH=bmg_g31`. Initial research recommended `bmg_g21` based on B580 benchmarks; that's the smaller Battlemage die. The B70 is the larger G31 silicon — wrong target = JIT compilation at runtime + suboptimal kernels. Source confirms both arches are registered in `ggml/src/ggml-sycl/sycl_hw.cpp:34-35`.

## 2026-05-12 — Disable `GGML_SYCL_DISABLE_OPT` (was `=1`, now unset)

Originally set `GGML_SYCL_DISABLE_OPT=1` against [issue #21893](https://github.com/ggml-org/llama.cpp/issues/21893) (Battlemage output corruption). Verified clean output on llama.cpp commit `bbeb89d` without the flag — the issue is fixed. Without `DISABLE_OPT`, tg128 on Nemotron rose 31.5 → 39.5 tok/s (+25%). `env-sycl.sh` keeps the flag commented out with a note to re-enable if a future pull regresses.

Tradeoff: removing `DISABLE_OPT` *hurts* pp512 on this MoE+hybrid model (581 → 490). For dense models the tradeoff may differ; revisit per model.

## 2026-05-12 — Models live on Optane, `~/LLMs` is a symlink

`~/LLMs/` symlinks to `/mnt/optane/LLMs/`. Any tool downloading to `~/LLMs/` (HF, our launcher's `-hf` flag, etc.) lands on Optane automatically. Optane's ~10 µs read latency makes model mmap effectively free; once `-ngl 99` is set, the model lives in VRAM and Optane drops out of the hot path.

## 2026-05-12 — Optane swap with aggressive swappiness

`vm.swappiness=100`, `vm.vfs_cache_pressure=50` (in `/etc/sysctl.d/99-llm-tuning.conf`). On a normal NVMe rig 100 would be terrible; Optane's ~10 µs swap latency is close enough to DRAM that aggressive spill is net positive. Lets kernel free RAM for page cache and KV-on-CPU scenarios.

## 2026-05-12 — Default config: SYCL build, Qwen3.6-27B Q6_K, 32K ctx, q8_0 KV

In `configs/default.conf`. Was Vulkan initially; flipped to SYCL after user preference + because dense Qwen3.6-27B is the model SYCL is expected to win on (still pending bench). 32K is conservative for the trained-256K model — fits VRAM comfortably with q8_0 KV (~24 GB total). Users can override any field via flags or `--config <other>`.

## 2026-05-12 — Public launcher repo: github.com/elsung/intellm

Decided to extract the launcher + supporting helpers into a public, sharable form. Parameterized paths via `$INTELLM_HOME` autodetection so the repo isn't user-specific. Models, builds, and private configs are gitignored. The repo is just the orchestration layer; users bring their own llama.cpp build and models.

## 2026-05-12 — Don't build a disk-tier KV cache (yet)

Per [the KV-offload synthesis](./research/2026-05-12-synthesis-kv-offload-plan.md): sub-token-granularity KV spill on Intel GPU is physics-blocked by the PCIe roundtrip cost (~300 µs/fault vs Optane's 10 µs). Coarse-grain spill is viable but only valuable for multi-session workloads. **Order of attack:**

1. KV quantization (q4_0 already wired; TurboQuant later) — handles most single-session long-ctx needs for free
2. KTransformers exploration — only mainstream project combining Intel Arc support + 3-tier KV reuse; could give us tiered KV without writing code
3. Slot-grain disk spill in llama.cpp — only if multi-session serving is the real use case
4. vLLM-XPU custom KVConnector — only if vLLM-XPU becomes our primary serving engine

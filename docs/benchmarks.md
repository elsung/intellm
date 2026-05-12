# Benchmarks

All numbers measured on:
- **Hardware:** Intel Arc Pro B70 (Battlemage, BMG-G31, 32 GB VRAM, PCI ID `0xe223`), AMD Ryzen 5 7600X, 30 GiB RAM, 64 GiB Optane swap
- **OS:** Ubuntu 24.04, kernel 6.17, Mesa 26.0.6 (kisak PPA)
- **Stack:** oneAPI 2026.0, IntelLLVM 2026.0.0; Vulkan 1.4.335 via Mesa ANV
- **llama.cpp commit:** `bbeb89d` (2026-05-05)

`llama-bench` invocation: `-ngl 99 -p 512 -n 128 -r 3`. `pp512` = prompt-processing tok/s, `tg128` = generation tok/s.

---

## 2026-05-12 — NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning UD-Q4_K_XL

Model is reported by llama-bench as `nemotron_h_moe 31B.A3.5B Q4_K - Medium`, 22.28 GiB, 31.58 B params. **MoE + Mamba hybrid architecture** — not a vanilla transformer.

| Build | pp512 (tok/s) | tg128 (tok/s) | Notes |
|---|---:|---:|---|
| Vulkan | **1349.47 ± 4.44** | 37.61 ± 0.08 | matrix cores: `KHR_coopmat` engaged |
| SYCL with `GGML_SYCL_DISABLE_OPT=1` | 581.17 ± 3.20 | 31.53 ± 0.14 | safety-belt setting |
| SYCL without `GGML_SYCL_DISABLE_OPT` | 490.58 ± 10.59 | **39.46 ± 0.92** | output verified coherent |

**Takeaways for this MoE+hybrid model:**
- Vulkan is 2.3–2.75× faster on prompt processing.
- SYCL (no-opt) edges Vulkan by ~5% on generation.
- Removing `DISABLE_OPT` trades pp512 (581 → 490) for tg128 (31.5 → 39.5) — surprising; suggests the SYCL reorder/optimize path has overhead that doesn't pay off for gather-heavy MoE prefill.
- Research's "1.5–2.2× SYCL over Vulkan" claim was measured on dense transformers and does not generalize to this MoE+Mamba hybrid.

## Pending benchmarks (worth running next)

- [ ] **Dense model**: Qwen3.6-27B Q6_K (256K trained ctx) — bench SYCL vs Vulkan. This is the natural fit for the research's dense-transformer SYCL win.
- [ ] **Long context**: Qwen3.6-27B at 128K and 256K with `--kv q8_0` and `--kv q4_0`. Measure actual VRAM use vs the rough KV-size estimates in synthesis doc.
- [ ] **KV-quant comparison**: same model + same ctx, sweep `--kv f16 | q8_0 | q4_0`. Tabulate VRAM + perf + (informal) output-quality impressions.
- [ ] **MoE Qwen3.6-35B-A3B-UD-Q4_K_M**: comparison against the Nemotron MoE — does SYCL/Vulkan ranking hold across MoE models or is it Nemotron-specific?
- [ ] **Concurrency**: llama-server with `-np 4` and `-np 8`, measure aggregate throughput (Vulkan first, then SYCL).

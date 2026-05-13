# intellm — research & decisions

Living documentation for the project. Keep it append-only where reasonable; when a finding is superseded, leave the old entry in place with a strikethrough or "(superseded YYYY-MM-DD by ...)" marker so the chronology stays readable.

## What goes where

| File / folder | Purpose |
|---|---|
| [`benchmarks.md`](./benchmarks.md) | Measured perf numbers (tokens/sec, memory footprint, etc.) — one table per hardware × backend × model |
| [`decisions.md`](./decisions.md) | Append-only log of design choices and the reason at the time. Date-stamped. |
| [`research/`](./research/) | Deep-dive investigations: agent findings, paper notes, prototyping logs, synthesis writeups |

## Conventions

- **Date format**: `YYYY-MM-DD` for filenames and timestamps. Sort lexicographically = chronologically.
- **Filenames in `research/`**: `YYYY-MM-DD-<short-slug>.md`. Multiple files per day are fine.
- **Benchmarks**: always include hardware, OS/kernel, backend commit hash, model+quant, exact command line. Without those, a number is noise.
- **Decisions**: lead with the decision in one sentence, then the reasoning. If you change your mind later, append a new entry that references the old one.

## Topics in flight

- **CURRENT STATE:** [2026-05-13 session summary](./2026-05-13-session-summary.md) — vLLM-XPU + VSA + Qwen3.5-9B working end-to-end, plus next steps for the next session.
- KV cache offloading for Intel GPU — investigated; physics-blocked on current hardware/software. See [synthesis](./research/2026-05-12-synthesis-kv-offload-plan.md). Will revisit if vLLM grows multi-tier KV.
- vLLM-XPU on Battlemage — working as of 2026-05-13. Three earlier attempts failed; the working config is in [research/2026-05-13-vllm-xpu-working-config.md](./research/2026-05-13-vllm-xpu-working-config.md).

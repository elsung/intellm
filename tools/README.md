# tools/

Self-contained helper scripts used by `intellm` workflows.

| Tool | Purpose |
|---|---|
| `vlm_sweep.sh` | Concurrency + latency benchmark against an OpenAI-compatible vision endpoint. Sweeps parallel-request counts, reports wall time / median / p95 / tokens-per-second. Works against `llama-server`, vLLM, OpenRouter, anything that speaks the OpenAI Chat Completions API. |

## `vlm_sweep.sh`

```bash
./tools/vlm_sweep.sh <frames_dir> [base_url] [model_id]

# Examples:
./tools/vlm_sweep.sh /tmp/frames                                # localhost:8080, "local-model"
./tools/vlm_sweep.sh /tmp/frames http://localhost:8000/v1       # vLLM port
FRAMES_PER_REQ=30 CONCURRENCY_LEVELS="1 2 4 8 12" ./tools/vlm_sweep.sh /tmp/frames
```

Env vars:
- `FRAMES_PER_REQ` (default 15) — how many images per request (matches VSA's `max_frames_per_request`)
- `CONCURRENCY_LEVELS` (default "1 2 4 8") — space-separated levels to sweep
- `MAX_TOKENS` (default 512) — output cap per request
- `PROMPT` (default a generic describe-the-scene prompt) — text portion of the request
- `VLM_SWEEP_JSON` (default `/dev/null`) — if set to a path, appends one JSON object per row for machine consumption

Output is a markdown table:
```
| concurrency | wall (s) | per-req median (ms) | per-req p95 (ms) | total tok | tok/s | errors |
| 1           | 4.21     | 4205                | 4205             | 312       | 74.1  | 0      |
| 2           | 4.93     | 4920                | 4920             | 614       | 124.5 | 0      |
...
```

The script reuses the same N-frame batch for every request — useful for isolating server-side concurrency from input variance. For variety, sweep on different frame sets separately.

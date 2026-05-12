#!/usr/bin/env bash
# vlm_sweep.sh — concurrency + latency benchmark against an OpenAI-compatible
# vision endpoint (llama-server / vLLM / etc.).
#
# Usage:
#   ./vlm_sweep.sh <frames_dir> [base_url] [model_id]
#
# - frames_dir: directory containing .jpg images (extracted at e.g. 1 fps).
# - base_url:   default http://localhost:8080/v1
# - model_id:   default "local-model" (works for llama-server which echoes any name)
#
# What it does:
#   - Picks the first N (= FRAMES_PER_REQ) images
#   - For each concurrency level in CONCURRENCY_LEVELS, fires K parallel requests
#     each sending the same N-image batch, measures wall time + per-request time.
#   - Outputs a markdown table to stdout and a JSON line per row.

set -euo pipefail

FRAMES_DIR="${1:-}"
BASE_URL="${2:-http://localhost:8080/v1}"
MODEL_ID="${3:-local-model}"

FRAMES_PER_REQ="${FRAMES_PER_REQ:-15}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PROMPT="${PROMPT:-Describe what is happening in these frames, briefly.}"

[[ -d "$FRAMES_DIR" ]] || { echo "usage: $0 <frames_dir> [base_url] [model_id]" >&2; exit 2; }

mapfile -t IMGS < <(find "$FRAMES_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n "$FRAMES_PER_REQ")
[[ ${#IMGS[@]} -gt 0 ]] || { echo "no images in $FRAMES_DIR" >&2; exit 2; }

echo "# VLM sweep — $(date -Iseconds)"
echo "- endpoint:      $BASE_URL"
echo "- model:         $MODEL_ID"
echo "- frames/req:    $FRAMES_PER_REQ  (using ${#IMGS[@]} from $FRAMES_DIR)"
echo "- concurrency:   $CONCURRENCY_LEVELS"
echo "- prompt:        \"$PROMPT\""
echo

# Build the request body once (same N-frame batch for every request).
TMPDIR_BASE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_BASE"' EXIT

REQ_BODY="$TMPDIR_BASE/req.json"
{
  printf '{'
  printf '"model":"%s","max_tokens":%d,"temperature":0.3,' "$MODEL_ID" "$MAX_TOKENS"
  printf '"messages":[{"role":"user","content":['
  printf '{"type":"text","text":"%s"}' "$PROMPT"
  for img in "${IMGS[@]}"; do
    b64="$(base64 -w0 "$img")"
    printf ',{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,%s"}}' "$b64"
  done
  printf ']}]}'
} > "$REQ_BODY"

req_size_mb=$(awk "BEGIN {printf \"%.1f\", $(wc -c < "$REQ_BODY") / 1048576}")
echo "_request body: ${req_size_mb} MB_"
echo

# Helper: fire one curl call, capture exit + timing
one_call() {
  local out="$1"
  local start_ms="$(date +%s%3N)"
  local http_code
  http_code=$(curl -sS -w '%{http_code}' -o "$out" \
    -H 'Content-Type: application/json' \
    --data-binary @"$REQ_BODY" \
    "${BASE_URL}/chat/completions")
  local end_ms="$(date +%s%3N)"
  printf '%s %s\n' "$http_code" "$((end_ms - start_ms))" > "$out.meta"
}
export -f one_call
export REQ_BODY BASE_URL

# Sweep
printf "| concurrency | wall (s) | per-req median (ms) | per-req p95 (ms) | total tok | tok/s | errors |\n"
printf "|---:|---:|---:|---:|---:|---:|---:|\n"

JSON_LOG="${VLM_SWEEP_JSON:-/dev/null}"

for K in $CONCURRENCY_LEVELS; do
  D="$TMPDIR_BASE/k=$K"; mkdir -p "$D"
  T0_NS="$(date +%s%N)"
  pids=()
  for i in $(seq 1 "$K"); do
    one_call "$D/$i.out" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  T1_NS="$(date +%s%N)"
  WALL_S="$(awk "BEGIN {printf \"%.2f\", ($T1_NS - $T0_NS) / 1e9}")"

  # Parse per-request times + token counts + errors
  times=()
  total_tok=0
  errors=0
  for i in $(seq 1 "$K"); do
    read -r code ms < "$D/$i.out.meta"
    times+=("$ms")
    if [[ "$code" == "200" ]]; then
      tok=$(grep -oE '"completion_tokens":[0-9]+' "$D/$i.out" | head -1 | grep -oE '[0-9]+' || echo 0)
      total_tok=$((total_tok + tok))
    else
      errors=$((errors + 1))
    fi
  done
  IFS=$'\n' sorted=($(printf '%s\n' "${times[@]}" | sort -n)); unset IFS
  n=${#sorted[@]}
  median="${sorted[$((n/2))]}"
  p95_idx=$(( (n * 95 / 100) - 1 ))
  [[ $p95_idx -lt 0 ]] && p95_idx=0
  p95="${sorted[$p95_idx]}"
  toks="$(awk "BEGIN {printf \"%.1f\", $total_tok / $WALL_S}")"

  printf "| %d | %s | %s | %s | %d | %s | %d |\n" "$K" "$WALL_S" "$median" "$p95" "$total_tok" "$toks" "$errors"
  printf '{"concurrency":%d,"wall_s":%s,"median_ms":%s,"p95_ms":%s,"total_tok":%d,"tok_per_s":%s,"errors":%d}\n' \
    "$K" "$WALL_S" "$median" "$p95" "$total_tok" "$toks" "$errors" >> "$JSON_LOG"
done

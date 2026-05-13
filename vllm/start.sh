#!/usr/bin/env bash
# start.sh — canonical vLLM-XPU launch for Qwen3.5-9B-AWQ-4bit on Intel Arc B70.
#
# This is the working incantation that survived three earlier failures.
# Documented in docs/research/2026-05-13-vllm-xpu-working-config.md.
#
# Combines:
#   • intel/llm-scaler-vllm:0.14.0-b8.2.1 (B70-validated container)
#   • Patched qwen2_vl.py (getattr fallback for max_pixels — see patches/)
#   • --enforce-eager (skip CUDA graph capture)
#   • VLLM_ATTENTION_BACKEND=TRITON_ATTN (avoids the FlashAttention path that crashes)
#   • --max-model-len 16384 (large enough for VSA's 30-frame batches)
#   • --max-num-seqs 8 (matches VSA's concurrent_requests setting)
#
# Usage:
#   /mnt/optane/LLMs/vllm/start.sh        # starts the container (idempotent)
#   /mnt/optane/LLMs/vllm/start.sh stop   # stop + remove
#   /mnt/optane/LLMs/vllm/start.sh status # show state
#   /mnt/optane/LLMs/vllm/start.sh logs   # docker logs -f
#
# Verify after start: curl http://localhost:8000/v1/models

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-intel/llm-scaler-vllm:0.14.0-b8.2.1}"
CONTAINER="${VLLM_CONTAINER:-vllm-xpu}"
PORT="${VLLM_PORT:-8000}"
MODEL_PATH_HOST="${VLLM_MODEL_PATH:-/mnt/optane/LLMs/hf-models/cyankiwi__Qwen3.5-9B-AWQ-4bit}"
MODEL_PATH_CONTAINER="/llm/models/$(basename "$MODEL_PATH_HOST")"
SERVED_NAME="${VLLM_SERVED_NAME:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"

case "${1:-start}" in
  status)
    if docker ps --filter "name=$CONTAINER" --format '{{.Names}}' 2>/dev/null | grep -q "$CONTAINER"; then
      echo "running: $CONTAINER"
      docker ps --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
      if curl -sf -m 3 "http://localhost:$PORT/v1/models" -o /dev/null; then
        echo "endpoint healthy: http://localhost:$PORT/v1"
      else
        echo "endpoint NOT responding (still warming up?)"
      fi
    else
      echo "not running"
    fi
    exit 0
    ;;
  stop)
    docker rm -f "$CONTAINER" 2>&1 | tail -1
    exit 0
    ;;
  logs)
    exec docker logs -f "$CONTAINER"
    ;;
esac

# --- start ---

# Idempotent: if already running and healthy, do nothing
if docker ps --filter "name=$CONTAINER" --format '{{.Names}}' 2>/dev/null | grep -q "$CONTAINER"; then
  if curl -sf -m 3 "http://localhost:$PORT/v1/models" -o /dev/null; then
    echo "already running and healthy at http://localhost:$PORT/v1"
    exit 0
  fi
  echo "container exists but unhealthy; removing"
  docker rm -f "$CONTAINER" >/dev/null
fi

# Inline serve command — uses heredoc to a script the container will exec
SERVE_SCRIPT="$(mktemp /tmp/vllm-start-XXXXXX.sh)"
cat > "$SERVE_SCRIPT" <<EOF
#!/bin/bash
exec vllm serve $MODEL_PATH_CONTAINER \\
  --served-model-name $SERVED_NAME \\
  --quantization compressed-tensors \\
  --dtype float16 \\
  --enforce-eager \\
  --max-num-seqs 8 \\
  --max-model-len 16384 \\
  --gpu-memory-utilization 0.85 \\
  --limit-mm-per-prompt '{"image": 30}' \\
  --tensor-parallel-size 1 \\
  --port $PORT
EOF
chmod +x "$SERVE_SCRIPT"

echo "starting $CONTAINER from $IMAGE"
docker run -d --name "$CONTAINER" --net=host --shm-size=32g \
  --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -v "$(dirname "$MODEL_PATH_HOST"):/llm/models:ro" \
  -v "$SERVE_SCRIPT:/start.sh:ro" \
  -v "$REPO_DIR/patches/qwen2_vl.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_vl.py:ro" \
  -e ZE_AFFINITY_MASK=0 \
  -e CCL_ENABLE_SYCL_KERNELS=0 \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  -e VLLM_USE_V1=1 \
  --entrypoint /start.sh \
  "$IMAGE" >/dev/null

echo "container started; waiting for endpoint to bind (typical cold start ~70-90s)..."
for i in {1..120}; do
  if curl -sf -m 3 "http://localhost:$PORT/v1/models" -o /dev/null; then
    echo "ready after ${i}×3s at http://localhost:$PORT/v1"
    exit 0
  fi
  if ! docker ps --filter "name=$CONTAINER" --quiet | grep -q .; then
    echo "container exited before binding; check 'docker logs $CONTAINER'"
    exit 1
  fi
  sleep 3
done
echo "timeout waiting for endpoint"
exit 1

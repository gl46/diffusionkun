#!/usr/bin/env bash
set -euo pipefail

# Launch a local OpenAI-compatible audit server.
# Do not run this on the same A100 at the same time as BabyDiffMT training
# unless you intentionally want both workloads to share GPU memory.
# Install vLLM in a separate environment if it conflicts with the training env.

MODEL=${MODEL:-Qwen/Qwen3.6-27B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen-audit}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.78}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PYTHON=${PYTHON:-python3}

"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching \
  --host "$HOST" \
  --port "$PORT"

#!/usr/bin/env bash
# scripts/start_vllm_colab.sh
# Launch vLLM in the background on a single Colab A100 with constrained GPU memory,
# leaving enough headroom for a co-located steered worker.
#
# Usage:
#   scripts/start_vllm_colab.sh [MODEL] [PORT] [GPU_UTIL]
# Defaults: MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct, PORT=8000, GPU_UTIL=0.40
set -euo pipefail

MODEL="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${2:-8000}"
GPU_UTIL="${3:-0.40}"

[[ "$(uname -s)" == "Linux" ]] || { echo "Linux/Colab only." >&2; exit 1; }

PID_FILE="/tmp/vllm_colab.pid"
LOG_FILE="/tmp/vllm_colab.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "vLLM already running (pid $(cat "$PID_FILE")); reusing. Tail: $LOG_FILE"
  exit 0
fi

echo "Starting vLLM: model=$MODEL port=$PORT gpu_util=$GPU_UTIL"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 4096 \
  --dtype auto \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM pid=$(cat "$PID_FILE"); logs: $LOG_FILE"
echo "Poll readiness with: curl -s http://127.0.0.1:$PORT/v1/models"

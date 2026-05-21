#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
MODEL_NAME=""
QUANTIZATION=""
STEERING_VECTOR=""
EXPERIMENTS="1.2,1.3,1.4"
STEERING_STRENGTHS="1.0"
TASK_INDICES=""
REPEATS="1"
PORT="8000"
MAX_MODEL_LEN="4096"
MAX_NEW_TOKENS="256"
CHAT_TURN_LIMIT="2"
CLEAN_GPU="0"
WORKER_GPU="1"
RESUME="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2;;
    --model) MODEL_NAME="$2"; shift 2;;
    --quantization) QUANTIZATION="$2"; shift 2;;
    --steering-vector) STEERING_VECTOR="$2"; shift 2;;
    --experiments) EXPERIMENTS="$2"; shift 2;;
    --steering-strengths) STEERING_STRENGTHS="$2"; shift 2;;
    --task-indices) TASK_INDICES="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --chat-turn-limit) CHAT_TURN_LIMIT="$2"; shift 2;;
    --clean-gpu) CLEAN_GPU="$2"; shift 2;;
    --worker-gpu) WORKER_GPU="$2"; shift 2;;
    --resume) RESUME="1"; shift;;
    *) fail "Unknown argument: $1";;
  esac
done

[[ -n "$MODEL_NAME" ]] || fail "--model is required"
[[ -n "$STEERING_VECTOR" ]] || fail "--steering-vector is required"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-source invariant: $MODEL_NAME and $QUANTIZATION feed BOTH the clean
# server and the steered sweep below, so the two agents always use the same
# weights and quantization scheme (mixing schemes would confound the cascade).

# 1. Background the clean vLLM server on the clean GPU (reuse the single launcher).
SERVE_ARGS=(--env-name "$ENV_NAME" --tensor-parallel-size 1 --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN")
if [[ -n "$QUANTIZATION" ]]; then
  SERVE_ARGS+=(--quantization "$QUANTIZATION")
fi
CUDA_VISIBLE_DEVICES="$CLEAN_GPU" bash "$SCRIPT_DIR/serve_clean_model.sh" "${SERVE_ARGS[@]}" "$MODEL_NAME" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# 2. Health-check the server (timeout ~300s).
echo "[cascade] waiting for clean server on :$PORT ..."
for _ in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[cascade] clean server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "clean vLLM server exited before becoming healthy"
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || fail "clean vLLM server did not become healthy in time"

# 3. Run the steered sweep on the worker GPU.
#    Do NOT restrict this launcher's CUDA_VISIBLE_DEVICES: run_phase1_sweep sets the
#    child's CUDA_VISIBLE_DEVICES from --worker-gpu-sets, and that index must be
#    valid against the full job allocation.
SWEEP_ARGS=(
  scripts/run_phase1_sweep.py
  --experiments "$EXPERIMENTS"
  --models "$MODEL_NAME"
  --steering-vector "$STEERING_VECTOR"
  --steering-strengths "$STEERING_STRENGTHS"
  --repeats "$REPEATS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --chat-turn-limit "$CHAT_TURN_LIMIT"
  --clean-api-bases "http://127.0.0.1:${PORT}/v1"
  --worker-gpu-sets "$WORKER_GPU"
)
if [[ -n "$TASK_INDICES" ]]; then
  SWEEP_ARGS+=(--task-indices "$TASK_INDICES")
fi
if [[ "$RESUME" == "1" ]]; then
  SWEEP_ARGS+=(--resume)
fi
python_in_conda "$ENV_NAME" "${SWEEP_ARGS[@]}"

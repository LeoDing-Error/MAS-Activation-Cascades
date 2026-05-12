#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
DTYPE="${VLLM_DTYPE:-auto}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"
DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

usage() {
  cat <<EOF
Usage: $0 [--env-name NAME] [--tensor-parallel-size N] [--pipeline-parallel-size N] [model-name]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpu-util)
      GPU_UTIL="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --pipeline-parallel-size)
      PIPELINE_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --download-dir)
      DOWNLOAD_DIR="$2"
      shift 2
      ;;
    --swap-space)
      SWAP_SPACE="$2"
      shift 2
      ;;
    --enforce-eager)
      ENFORCE_EAGER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      MODEL_NAME="$1"
      shift
      ;;
  esac
done

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "vLLM serving is not supported on macOS. Run this script on Linux or Colab with CUDA."
  exit 1
fi

ensure_command conda
conda_env_exists "$ENV_NAME" || fail "Conda env $ENV_NAME does not exist. Run scripts/setup_env.sh first."

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --dtype "$DTYPE"
)

if [[ -n "$MAX_NUM_SEQS" ]]; then
  VLLM_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$SERVED_MODEL_NAME" ]]; then
  VLLM_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ -n "$DOWNLOAD_DIR" ]]; then
  VLLM_ARGS+=(--download-dir "$DOWNLOAD_DIR")
fi
if [[ -n "$SWAP_SPACE" ]]; then
  VLLM_ARGS+=(--swap-space "$SWAP_SPACE")
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

python_in_conda "$ENV_NAME" "${VLLM_ARGS[@]}"

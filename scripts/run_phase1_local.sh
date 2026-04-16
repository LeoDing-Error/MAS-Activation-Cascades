#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
CLEAN_API_BASE="${CLEAN_API_BASE:-http://127.0.0.1:8000/v1}"
CLEAN_API_KEY="${CLEAN_API_KEY:-EMPTY}"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

usage() {
  cat <<EOF
Usage: $0 [--env-name NAME] <experiment> <steering-vector-path> [model-name]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

EXPERIMENT="$1"
STEERING_VECTOR="$2"
if [[ $# -eq 3 ]]; then
  MODEL_NAME="$3"
fi

ensure_command conda
conda_env_exists "$ENV_NAME" || fail "Conda env $ENV_NAME does not exist. Run scripts/setup_env.sh first."

python_in_conda "$ENV_NAME" "$PROJECT_ROOT/experiments/run_phase1.py" \
  --experiment "$EXPERIMENT" \
  --model "$MODEL_NAME" \
  --steering-vector "$STEERING_VECTOR" \
  --results-dir "$RESULTS_DIR" \
  --clean-api-base "$CLEAN_API_BASE" \
  --clean-api-key "$CLEAN_API_KEY"

#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
PAIRS_PATH="$PROJECT_ROOT/data/contrastive_pairs/ta2_harmful_pairs.json"
OUTPUT_PATH="$PROJECT_ROOT/steering_vectors/harmfulness_llama3_8b.pt"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"

usage() {
  cat <<EOF
Usage: $0 [--env-name NAME] [--device DEVICE] [--dtype DTYPE] [model-name] [pairs-path] [output-path]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
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

if [[ $# -gt 0 ]]; then
  MODEL_NAME="$1"
fi
if [[ $# -gt 1 ]]; then
  PAIRS_PATH="$2"
fi
if [[ $# -gt 2 ]]; then
  OUTPUT_PATH="$3"
fi
if [[ $# -gt 3 ]]; then
  fail "Too many arguments"
fi

ensure_command conda
conda_env_exists "$ENV_NAME" || fail "Conda env $ENV_NAME does not exist. Run scripts/setup_env.sh first."

python_in_conda "$ENV_NAME" "$PROJECT_ROOT/src/steering/compute_vectors.py" \
  --model "$MODEL_NAME" \
  --pairs-path "$PAIRS_PATH" \
  --output "$OUTPUT_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

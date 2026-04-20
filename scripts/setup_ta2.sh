#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
PAIR_DATASET="harmful"
PAIR_LIMIT=""
GENERATE_PAIRS=1
SKIP_CLONE=0
OUTPUT_PATH="$PROJECT_ROOT/data/contrastive_pairs/ta2_harmful_pairs.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --dataset)
      PAIR_DATASET="$2"
      shift 2
      ;;
    --limit)
      PAIR_LIMIT="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --skip-pairs)
      GENERATE_PAIRS=0
      shift
      ;;
    --skip-clone)
      SKIP_CLONE=1
      shift
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

ensure_command conda
if [[ "$SKIP_CLONE" -eq 0 ]]; then
  "$PROJECT_ROOT/scripts/setup_references.sh"
fi

TA2_DIR="$THIRD_PARTY_DIR/Trojan-Activation-Attack"
[[ -d "$TA2_DIR" ]] || fail "TA2 repo not found at $TA2_DIR"
conda_env_exists "$ENV_NAME" || fail "Conda env $ENV_NAME does not exist. Run scripts/setup_env.sh first."

log "TA2 reference repo available at $TA2_DIR"

if [[ "$GENERATE_PAIRS" -eq 1 ]]; then
  log "Generating TA2-derived contrastive pairs"
  pair_args=(
    --dataset "$PAIR_DATASET"
    --output "$OUTPUT_PATH"
  )
  if [[ -n "$PAIR_LIMIT" ]]; then
    pair_args+=(--limit "$PAIR_LIMIT")
  fi
  python_in_conda "$ENV_NAME" "$PROJECT_ROOT/scripts/build_ta2_pairs.py" "${pair_args[@]}"
fi

log "TA2 setup complete"

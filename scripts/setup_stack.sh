#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
INSTALL_CUDA121=0
CAMEL_EXTRAS=""
PAIR_DATASET="harmful"
PAIR_LIMIT=10
RUN_CHECK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --cuda121)
      INSTALL_CUDA121=1
      shift
      ;;
    --camel-extras)
      CAMEL_EXTRAS="$2"
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
    --skip-check)
      RUN_CHECK=0
      shift
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

setup_env_args=(--env-name "$ENV_NAME")
if [[ "$INSTALL_CUDA121" -eq 1 ]]; then
  setup_env_args+=(--cuda121)
fi

setup_camel_args=(--env-name "$ENV_NAME")
if [[ -n "$CAMEL_EXTRAS" ]]; then
  setup_camel_args+=(--extras "$CAMEL_EXTRAS")
fi

"$PROJECT_ROOT/scripts/setup_env.sh" "${setup_env_args[@]}"
"$PROJECT_ROOT/scripts/setup_camel.sh" "${setup_camel_args[@]}"
"$PROJECT_ROOT/scripts/setup_ta2.sh" --env-name "$ENV_NAME" --dataset "$PAIR_DATASET" --limit "$PAIR_LIMIT"

if [[ "$RUN_CHECK" -eq 1 ]]; then
  log "Running setup verification"
  python_in_conda "$ENV_NAME" "$PROJECT_ROOT/scripts/check_setup.py"
fi

log "Full stack setup complete"
log "Next steps:"
log "  conda activate $ENV_NAME"
log "  ./scripts/compute_vector_local.sh"
log "  ./scripts/run_phase1_local.sh 1.2 steering_vectors/harmfulness_llama3_8b.pt"

#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
CAMEL_EXTRAS=""
SKIP_CLONE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --extras)
      CAMEL_EXTRAS="$2"
      shift 2
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

CAMEL_DIR="$THIRD_PARTY_DIR/camel"
[[ -d "$CAMEL_DIR" ]] || fail "CAMEL repo not found at $CAMEL_DIR"
conda_env_exists "$ENV_NAME" || fail "Conda env $ENV_NAME does not exist. Run scripts/setup_env.sh first."

log "Installing local CAMEL clone into $ENV_NAME"
pip_in_conda "$ENV_NAME" uninstall -y camel-ai || true

(
  cd "$CAMEL_DIR"
  if [[ -n "$CAMEL_EXTRAS" ]]; then
    pip_in_conda "$ENV_NAME" install -e ".[${CAMEL_EXTRAS}]"
  else
    pip_in_conda "$ENV_NAME" install -e .
  fi
)

log "Local CAMEL install complete"

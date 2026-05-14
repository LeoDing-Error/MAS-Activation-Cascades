#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
INSTALL_CUDA121=0

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
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

ensure_command conda

if conda_env_exists "$ENV_NAME"; then
  log "Updating conda env $ENV_NAME from environment.yml"
  conda env update -y -n "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml" --prune
else
  log "Creating conda env $ENV_NAME from environment.yml"
  conda env create -y -n "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml"
fi

log "Upgrading packaging tools in $ENV_NAME"
pip_in_conda "$ENV_NAME" install --upgrade pip setuptools wheel

log "Reinstalling project requirements into $ENV_NAME"
pip_in_conda "$ENV_NAME" install -r "$PROJECT_ROOT/requirements.txt"

if [[ "$INSTALL_CUDA121" -eq 1 ]]; then
  if is_macos; then
    fail "--cuda121 is only for the PDE CUDA setup"
  fi
  log "Reinstalling torch stack with CUDA 12.1 wheels in $ENV_NAME"
  pip_in_conda "$ENV_NAME" uninstall -y torch torchvision torchaudio || true
  pip_in_conda "$ENV_NAME" install --upgrade --force-reinstall \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
fi

log "Environment setup complete for $ENV_NAME"
log "Activate with: conda activate $ENV_NAME"

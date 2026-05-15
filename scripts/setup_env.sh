#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
INSTALL_CUDA128=0
VLLM_CUDA128_VERSION="${VLLM_CUDA128_VERSION:-0.20.2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --cuda128)
      INSTALL_CUDA128=1
      shift
      ;;
    --cuda121)
      fail "--cuda121 is invalid on the PDE Blackwell GPUs. Use --cuda128."
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
  printf 'y\n' | conda env update -n "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml" --prune
else
  log "Creating conda env $ENV_NAME from environment.yml"
  printf 'y\n' | conda env create -n "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml"
fi

log "Upgrading packaging tools in $ENV_NAME"
pip_in_conda "$ENV_NAME" install --upgrade pip setuptools wheel

log "Reinstalling project requirements into $ENV_NAME"
pip_in_conda "$ENV_NAME" install -r "$PROJECT_ROOT/requirements.txt"

if [[ "$INSTALL_CUDA128" -eq 1 ]]; then
  if is_macos; then
    fail "--cuda128 is only for the PDE CUDA setup"
  fi
  log "Reinstalling Blackwell-compatible torch/vLLM stack with CUDA 12.8 wheels in $ENV_NAME"
  pip_in_conda "$ENV_NAME" uninstall -y vllm xformers outlines torch torchvision torchaudio || true
  pip_in_conda "$ENV_NAME" install --upgrade --force-reinstall \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
  pip_in_conda "$ENV_NAME" install --upgrade --force-reinstall \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_CUDA128_VERSION}/vllm-${VLLM_CUDA128_VERSION}+cu128-cp38-abi3-manylinux_2_31_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu128
  pip_in_conda "$ENV_NAME" install --upgrade --force-reinstall \
    "numpy>=2,<2.3" "fsspec[http]<=2026.2.0,>=2023.1.0"
fi

log "Environment setup complete for $ENV_NAME"
log "Activate with: conda activate $ENV_NAME"

#!/usr/bin/env bash
# scripts/setup_colab.sh
# Pip-based bootstrap for Google Colab (Linux-only).
# Run from the repo root after cloning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '[colab-setup] %s\n' "$*"; }
fail() { printf '[colab-setup][error] %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "setup_colab.sh is Linux/Colab only"

# ── third-party reference repos ──────────────────────────────────────────────
log "Cloning reference repositories at pinned commits..."
bash "$PROJECT_ROOT/scripts/setup_references.sh"

# ── Python packages ──────────────────────────────────────────────────────────
log "Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel

log "Installing CUDA 12.1 torch stack..."
pip install --quiet --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

log "Installing project requirements (includes vllm on Linux)..."
pip install --quiet -r "$PROJECT_ROOT/requirements.txt"

# ── local CAMEL (must take precedence over any PyPI camel-ai) ─────────────────
log "Installing local CAMEL editable..."
pip uninstall -y camel-ai 2>/dev/null || true
pip install --quiet -e "$PROJECT_ROOT/third_party/camel"

log "Setup complete. Run: python scripts/smoke_test_colab.py"

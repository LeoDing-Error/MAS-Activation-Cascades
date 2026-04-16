#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"
DEFAULT_ENV_NAME="cascade"

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

fail() {
  printf '[setup][error] %s\n' "$*" >&2
  exit 1
}

ensure_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

os_name() {
  uname -s
}

is_macos() {
  [[ "$(os_name)" == "Darwin" ]]
}

is_linux() {
  [[ "$(os_name)" == "Linux" ]]
}

conda_env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -Fx "$env_name" >/dev/null 2>&1
}

run_in_conda() {
  local env_name="$1"
  shift
  conda run -n "$env_name" "$@"
}

python_in_conda() {
  local env_name="$1"
  shift
  run_in_conda "$env_name" python "$@"
}

pip_in_conda() {
  local env_name="$1"
  shift
  run_in_conda "$env_name" python -m pip "$@"
}

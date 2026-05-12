#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"

usage() {
  cat <<EOF
Usage: $0 [--env-name NAME] --experiments 1.2,1.3 --steering-vector path [additional args...]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
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

ensure_command conda
conda_env_exists "$ENV_NAME" || fail "Conda env $ENV_NAME does not exist. Run scripts/setup_env.sh first."

python_in_conda "$ENV_NAME" "$PROJECT_ROOT/scripts/run_phase1_sweep.py" "$@"

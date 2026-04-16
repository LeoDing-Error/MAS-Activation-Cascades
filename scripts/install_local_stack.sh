#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "scripts/install_local_stack.sh is now a compatibility wrapper around scripts/setup_stack.sh"
exec "$ROOT_DIR/scripts/setup_stack.sh" "$@"

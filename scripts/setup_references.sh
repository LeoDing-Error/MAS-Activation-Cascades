#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
LOCK_FILE="$THIRD_PARTY_DIR/refs.lock"
mkdir -p "$THIRD_PARTY_DIR"

usage() {
  cat <<EOF
Usage: $0

Clones or updates the third-party reference repositories at the pinned commits
listed in third_party/refs.lock.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

command -v git >/dev/null 2>&1 || {
  echo "Missing required command: git" >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "Missing required command: python3" >&2
  exit 1
}
[[ -f "$LOCK_FILE" ]] || {
  echo "Missing lock file: $LOCK_FILE" >&2
  exit 1
}

repo_url() {
  case "$1" in
    Trojan-Activation-Attack)
      echo "https://github.com/wang2226/Trojan-Activation-Attack"
      ;;
    camel)
      echo "https://github.com/camel-ai/camel"
      ;;
    *)
      echo "Unknown repo: $1" >&2
      exit 1
      ;;
  esac
}

repo_commit() {
  python3 - "$LOCK_FILE" "$1" <<'PY'
from pathlib import Path
import sys

lock_file = Path(sys.argv[1])
repo_name = sys.argv[2]

for line in lock_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, commit = line.split()
    if name == repo_name:
        print(commit)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

ensure_clean_repo() {
  local repo_dir="$1"
  if [[ ! -d "$repo_dir/.git" ]]; then
    return 0
  fi
  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    echo "Refusing to update dirty repo: $repo_dir" >&2
    exit 1
  fi
}

clone_or_checkout_locked() {
  local repo_name="$1"
  local repo_dir="$THIRD_PARTY_DIR/$repo_name"
  local repo_url_value
  local commit

  repo_url_value="$(repo_url "$repo_name")"
  commit="$(repo_commit "$repo_name")" || {
    echo "Missing locked commit for $repo_name in $LOCK_FILE" >&2
    exit 1
  }

  ensure_clean_repo "$repo_dir"

  if [[ ! -d "$repo_dir/.git" ]]; then
    echo "Cloning $repo_name"
    git clone --filter=blob:none --no-checkout "$repo_url_value" "$repo_dir"
  fi

  echo "Checking out $repo_name at $commit"
  git -C "$repo_dir" fetch --depth=1 origin "$commit"
  git -C "$repo_dir" checkout --detach FETCH_HEAD >/dev/null 2>&1
  echo "$repo_name -> $(git -C "$repo_dir" rev-parse --short HEAD)"
}

clone_or_checkout_locked "Trojan-Activation-Attack"
clone_or_checkout_locked "camel"

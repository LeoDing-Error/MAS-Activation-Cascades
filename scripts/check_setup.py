from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import CAMEL_ROOT, CONTRASTIVE_PAIRS_ROOT, PROJECT_ROOT, TA2_ROOT, ensure_local_camel_on_path


def _check_path(label: str, path: Path) -> None:
    status = "ok" if path.exists() else "missing"
    print(f"[{status}] {label}: {path}")


def _check_import(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[missing] import {module_name}: {exc}")
        return
    module_path = getattr(module, "__file__", "built-in")
    print(f"[ok] import {module_name}: {module_path}")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    _check_path("TA2 repo", TA2_ROOT)
    _check_path("CAMEL repo", CAMEL_ROOT)
    _check_path("contrastive-pairs dir", CONTRASTIVE_PAIRS_ROOT)

    ensure_local_camel_on_path()
    _check_import("camel")
    _check_import("torch")
    _check_import("transformers")
    _check_import("vllm")

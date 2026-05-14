from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
TA2_ROOT = THIRD_PARTY_ROOT / "Trojan-Activation-Attack"
CAMEL_ROOT = THIRD_PARTY_ROOT / "camel"
DATA_ROOT = PROJECT_ROOT / "data"
CONTRASTIVE_PAIRS_ROOT = DATA_ROOT / "contrastive_pairs"
RESULTS_ROOT = PROJECT_ROOT / "results"
STEERING_VECTORS_ROOT = PROJECT_ROOT / "steering_vectors"


def _prepend_sys_path(path: Path) -> bool:
    resolved = str(path.resolve())
    if not path.exists():
        return False
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)
    return True


def ensure_local_camel_on_path() -> bool:
    return _prepend_sys_path(CAMEL_ROOT)


def ensure_local_ta2_on_path() -> bool:
    return _prepend_sys_path(TA2_ROOT)


def resolve_repo_path(name: str) -> Optional[Path]:
    mapping = {
        "camel": CAMEL_ROOT,
        "ta2": TA2_ROOT,
    }
    path = mapping.get(name.lower())
    if path is None or not path.exists():
        return None
    return path

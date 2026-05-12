# scripts/smoke_test_colab.py
"""Pre-flight environment check for Colab. CPU-gated checks run first;
GPU checks only if CUDA is present."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(label: str, fix: str = "") -> bool:
    print(f"[ok]   {label}")
    return True


def _fail(label: str, fix: str = "") -> bool:
    hint = f"  → {fix}" if fix else ""
    print(f"[FAIL] {label}{hint}")
    return False


def check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return _ok(f"import {module}")
    except Exception as exc:
        return _fail(f"import {module}: {exc}", "run scripts/setup_colab.sh")


def check_paths() -> list[bool]:
    from src.project_paths import CAMEL_ROOT, TA2_ROOT, CONTRASTIVE_PAIRS_ROOT

    results = []
    for label, path, fix in [
        ("third_party/camel", CAMEL_ROOT, "run scripts/setup_colab.sh"),
        ("third_party/Trojan-Activation-Attack", TA2_ROOT, "run scripts/setup_colab.sh"),
        (
            "data/contrastive_pairs/ta2_harmful_pairs.json",
            CONTRASTIVE_PAIRS_ROOT / "ta2_harmful_pairs.json",
            "run: python scripts/build_ta2_pairs.py",
        ),
    ]:
        if path.exists():
            results.append(_ok(label))
        else:
            results.append(_fail(label, fix))
    return results


def check_gpu() -> list[bool]:
    results = []
    try:
        import torch
    except ImportError:
        results.append(_fail("torch not installed; skipping GPU checks", "run scripts/setup_colab.sh"))
        return results

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        results.append(_ok("CUDA available"))
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"       GPU: {name} ({vram_gb:.0f} GB VRAM)")
        if vram_gb >= 35:
            results.append(_ok(f"VRAM ≥35 GB ({vram_gb:.0f} GB detected)"))
        else:
            results.append(
                _fail(
                    f"VRAM ≥35 GB required, only {vram_gb:.0f} GB detected",
                    "Runtime → Change runtime type → GPU → A100",
                )
            )
    else:
        results.append(
            _fail("CUDA not available", "Runtime → Change runtime type → GPU")
        )
    return results


def main() -> None:
    results: list[bool] = []

    print("=== Import checks ===")
    for mod in ["torch", "transformers", "camel", "camel.agents", "openai", "vllm"]:
        results.append(check_import(mod))

    print("\n=== File structure checks ===")
    results.extend(check_paths())

    print("\n=== GPU checks ===")
    results.extend(check_gpu())

    failed = results.count(False)
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed — fix above before proceeding.'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

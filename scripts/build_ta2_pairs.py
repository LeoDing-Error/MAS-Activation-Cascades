from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import CONTRASTIVE_PAIRS_ROOT, TA2_ROOT

SAFE_TEMPLATE = (
    "I cannot help with harmful, illegal, or unsafe actions. "
    "I can help with safe, lawful alternatives instead."
)

DATASET_TO_FILE = {
    "harmful": TA2_ROOT / "Dataset" / "Harmful" / "harmful.csv",
}


def build_pairs(dataset: str, limit: int | None = None) -> List[Dict[str, str]]:
    dataset_key = dataset.lower()
    if dataset_key not in DATASET_TO_FILE:
        raise ValueError(f"Unsupported TA2 dataset: {dataset}")

    dataset_path = DATASET_TO_FILE[dataset_key]
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"TA2 dataset not found at {dataset_path}. Run scripts/setup_references.sh first."
        )

    pairs: List[Dict[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            instruction = (row.get("goal") or "").strip()
            unsafe_completion = (row.get("target") or "").strip()
            if not instruction or not unsafe_completion:
                continue
            pairs.append(
                {
                    "instruction": instruction,
                    "safe_completion": SAFE_TEMPLATE,
                    "unsafe_completion": unsafe_completion,
                    "metadata": {
                        "source": "ta2",
                        "dataset": dataset_key,
                    },
                }
            )
            if limit is not None and len(pairs) >= limit:
                break
    if not pairs:
        raise RuntimeError(f"No contrastive pairs were generated from {dataset_path}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contrastive prompt pairs from the local TA2 dataset clone")
    parser.add_argument("--dataset", default="harmful", choices=sorted(DATASET_TO_FILE.keys()))
    parser.add_argument("--limit", type=int, default=10, help="Number of rows to include")
    parser.add_argument(
        "--output",
        type=Path,
        default=CONTRASTIVE_PAIRS_ROOT / "ta2_harmful_pairs.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    pairs = build_pairs(args.dataset, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Trojan-Activation-Attack",
        "dataset": args.dataset,
        "count": len(pairs),
        "pairs": pairs,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs)} contrastive pairs to {args.output}")


if __name__ == "__main__":
    main()

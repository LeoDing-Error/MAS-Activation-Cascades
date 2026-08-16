"""Prepare, generate, blind, and summarize the steering calibration pilot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backends.steering_backend import GenerationResult, SteeringModelBackend, SteeringVectorArtifact
from src.experiments.calibration_protocol import (
    ALPHAS,
    EXCLUDED_CATEGORY_TERMS,
    OVERLAP_THRESHOLD,
    SORRY_DOMAINS,
    XSTEST_COMMIT,
    CalibrationPrompt,
    ManualScore,
    build_blinded_rows,
    select_sorry_prompts,
    select_xstest_pair,
    summarize_calibration,
)
from src.experiments.phase1_config import PRIMARY_MODEL
from src.metrics.uncertainty import compute_uncertainty_snapshot

XSTEST_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    f"{XSTEST_COMMIT}/xstest_prompts.csv"
)
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 256


@dataclass(frozen=True)
class PreparationOutputs:
    private_manifest: dict[str, object]
    public_manifest: dict[str, object]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prompt_dict(prompt: CalibrationPrompt, *, include_text: bool) -> dict[str, object]:
    record = asdict(prompt)
    if not include_text:
        record.pop("text")
    return record


def prepare_calibration(
    *,
    sorry_records: Sequence[Mapping[str, str]],
    sorry_revision: str,
    xstest_rows: Sequence[Mapping[str, str]],
    ta2_instructions: Sequence[str],
    output_dir: Path,
) -> PreparationOutputs:
    """Select six prompts and persist private text plus public provenance."""
    selected = [
        *select_sorry_prompts(sorry_records, ta2_instructions=ta2_instructions),
        *select_xstest_pair(xstest_rows),
    ]
    private_manifest: dict[str, object] = {
        "prompts": [_prompt_dict(prompt, include_text=True) for prompt in selected],
        "sorry_revision": sorry_revision,
        "xstest_commit": XSTEST_COMMIT,
    }
    public_manifest: dict[str, object] = {
        "prompts": [_prompt_dict(prompt, include_text=False) for prompt in selected],
        "sorry_revision": sorry_revision,
        "xstest_commit": XSTEST_COMMIT,
        "xstest_attribution": "XSTest by Rottger et al., CC-BY-4.0",
        "repository_commit": _repository_commit(),
        "selection_parameters": {
            "excluded_category_terms": list(EXCLUDED_CATEGORY_TERMS),
            "overlap_threshold": OVERLAP_THRESHOLD,
            "sorry_domains": list(SORRY_DOMAINS),
            "xstest_ids": ["1", "26"],
        },
        "generation_parameters": {
            "model": PRIMARY_MODEL,
            "alphas": list(ALPHAS),
            "system_prompt": SYSTEM_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    _write_json(output_dir / "private_prompt_manifest.json", private_manifest)
    _write_json(output_dir / "run_manifest.json", public_manifest)
    return PreparationOutputs(private_manifest, public_manifest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_prompts(manifest: Mapping[str, object]) -> list[CalibrationPrompt]:
    raw_prompts = manifest.get("prompts")
    if not isinstance(raw_prompts, list):
        raise ValueError("Private prompt manifest must contain a prompts list")
    prompts: list[CalibrationPrompt] = []
    for raw in raw_prompts:
        if not isinstance(raw, Mapping):
            raise ValueError("Private prompt manifest contains an invalid prompt")
        try:
            prompts.append(CalibrationPrompt(
                prompt_id=str(raw["prompt_id"]), source=str(raw["source"]),
                source_id=str(raw["source_id"]), category=str(raw["category"]),
                high_level_domain=str(raw["high_level_domain"]),
                expected_label=str(raw["expected_label"]), text=str(raw["text"]),
                prompt_sha256=str(raw["prompt_sha256"]),
            ))
        except KeyError as error:
            raise ValueError(f"Private prompt manifest is missing {error.args[0]!r}") from error
    if len(prompts) != 6 or len({prompt.prompt_id for prompt in prompts}) != 6:
        raise ValueError("Private prompt manifest must contain exactly six unique prompts")
    return prompts


def _existing_conditions(path: Path) -> set[tuple[str, float]]:
    if not path.exists():
        return set()
    conditions: set[tuple[str, float]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        prompt_id = record.get("prompt_id")
        alpha = record.get("alpha")
        if isinstance(prompt_id, str) and type(alpha) in (int, float):
            conditions.add((prompt_id, float(alpha)))
    return conditions


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def generate_calibration(
    *,
    private_manifest: Mapping[str, object],
    steering_path: Path,
    output_dir: Path,
    backend_factory: Callable[..., SteeringModelBackend] = SteeringModelBackend,
) -> None:
    """Generate the complete grid with one model instance and JSONL checkpointing."""
    prompts = _manifest_prompts(private_manifest)
    artifact_sha256 = _sha256_file(steering_path)
    artifact = SteeringVectorArtifact.from_file(steering_path)
    if artifact.model_name != PRIMARY_MODEL or artifact.layer != 25:
        raise ValueError("Calibration artifact must match Llama 3.1 8B layer 25")

    backend = backend_factory(
        model_name=PRIMARY_MODEL,
        steering_artifact=artifact,
        alpha=0.0,
        steering_enabled=False,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    records_path = output_dir / "raw_generations.jsonl"
    existing = _existing_conditions(records_path)
    dtype = str(getattr(backend, "torch_dtype", "unknown"))
    for alpha in ALPHAS:
        backend.set_steering_strength(alpha)
        backend.set_steering_enabled(alpha > 0.0)
        for prompt in prompts:
            condition = (prompt.prompt_id, alpha)
            if condition in existing:
                continue
            result: GenerationResult = backend.generate_from_messages([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.text},
            ])
            diagnostics = compute_uncertainty_snapshot(result.step_logits, text=result.response_text)
            record: dict[str, object] = {
                "prompt_id": prompt.prompt_id,
                "source": prompt.source,
                "source_id": prompt.source_id,
                "category": prompt.category,
                "high_level_domain": prompt.high_level_domain,
                "expected_label": prompt.expected_label,
                "prompt_sha256": prompt.prompt_sha256,
                "prompt_text": prompt.text,
                "alpha": alpha,
                "response_text": result.response_text,
                "completion_token_count": result.completion_token_count,
                "truncated": result.completion_token_count == MAX_NEW_TOKENS,
                "termination_state": "length" if result.completion_token_count == MAX_NEW_TOKENS else "stop",
                "model": PRIMARY_MODEL,
                "dtype": dtype,
                "artifact_sha256": artifact_sha256,
                "artifact_layer": artifact.layer,
                "artifact_vector_norm": artifact.vector_norm,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "mean_token_entropy": diagnostics.mean_token_entropy,
                "normalized_sequence_msp": diagnostics.normalized_sequence_msp,
            }
            _append_jsonl(records_path, record)
            existing.add(condition)


def blind_calibration(*, output_dir: Path) -> None:
    records_path = output_dir / "raw_generations.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, key = build_blinded_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "response_id", "prompt_text", "response_text", "refusal_compliance", "coherence",
        "relevance", "repetitive", "truncated",
    ]
    with (output_dir / "blind_scoring.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(output_dir / "condition_key.json", key)


def _parse_boolean(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Manual score {field} must be true or false")


def _load_manual_scores(path: Path) -> list[ManualScore]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores: list[ManualScore] = []
    for row in rows:
        try:
            scores.append(ManualScore(
                response_id=row["response_id"],
                refusal_compliance=int(row["refusal_compliance"]),
                coherence=int(row["coherence"]),
                relevance=int(row["relevance"]),
                repetitive=_parse_boolean(row["repetitive"], "repetitive"),
                truncated=_parse_boolean(row["truncated"], "truncated"),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid manual score row: {error}") from error
    return scores


def summarize_calibration_run(*, output_dir: Path, scores_path: Path) -> dict[str, object]:
    key = json.loads((output_dir / "condition_key.json").read_text(encoding="utf-8"))
    if not isinstance(key, dict):
        raise ValueError("Condition key must be an object")
    summary = summarize_calibration((_load_manual_scores(scores_path), key))
    payload = {
        "baseline_valid": summary.baseline_valid,
        "baseline_reasons": list(summary.baseline_reasons),
        "alpha_results": {str(alpha): asdict(result) for alpha, result in summary.alpha_results.items()},
        "selected_alpha": summary.selected_alpha,
    }
    _write_json(output_dir / "summary.json", payload)
    if summary.selected_alpha is None:
        reasons = summary.baseline_reasons or tuple(
            reason for result in summary.alpha_results.values() for reason in result.reasons
        )
        print("NO_COHERENT_ALPHA" + (f" reasons={','.join(sorted(set(reasons)))}" if reasons else ""))
    else:
        print(f"SELECTED_ALPHA={summary.selected_alpha}")
    return payload


def _literal_category_mapping(source: str) -> dict[str, str]:
    """Extract only a literal category-to-domain mapping from meta_info.py."""
    tree = ast.parse(source)
    candidates: list[dict[str, str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value_node = node.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
            mapping = {str(key): str(item) for key, item in value.items()}
            if set(mapping.values()).issubset(set(SORRY_DOMAINS)):
                candidates.append(mapping)
    if len(candidates) != 1:
        raise ValueError("Could not find one literal SORRY-Bench category/domain mapping")
    return candidates[0]


def _load_ta2_instructions(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("TA2 pairs payload must contain a pairs list")
    instructions: list[str] = []
    for pair in pairs:
        if not isinstance(pair, Mapping) or not isinstance(pair.get("instruction"), str):
            raise ValueError("Every TA2 pair must contain an instruction")
        instructions.append(pair["instruction"])
    return instructions


def _download_sorry_records() -> tuple[list[dict[str, str]], str]:
    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().dataset_info("sorry-bench/sorry-bench-202503")
        revision = info.sha
        question_path = hf_hub_download(
            "sorry-bench/sorry-bench-202503", repo_type="dataset", revision=revision,
            filename="question.jsonl",
        )
        meta_path = hf_hub_download(
            "sorry-bench/sorry-bench-202503", repo_type="dataset", revision=revision,
            filename="meta_info.py",
        )
    except HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError(
                "Accept the sorry-bench/sorry-bench-202503 license and enable HF_TOKEN notebook access."
            ) from error
        raise
    except Exception as error:
        status_code = getattr(error, "response", None)
        if getattr(status_code, "status_code", None) in (401, 403):
            raise RuntimeError(
                "Accept the sorry-bench/sorry-bench-202503 license and enable HF_TOKEN notebook access."
            ) from error
        raise
    category_to_domain = _literal_category_mapping(Path(meta_path).read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for line in Path(question_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        category = str(raw["category"])
        turns = raw["turns"]
        if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
            raise ValueError("SORRY-Bench record must have a first text turn")
        try:
            domain = category_to_domain[category]
        except KeyError as error:
            raise ValueError(f"No high-level domain mapping for SORRY-Bench category: {category}") from error
        records.append({
            "source_id": str(raw["question_id"]), "category": category,
            "high_level_domain": domain, "text": turns[0],
        })
    return records, revision


def _download_xstest_rows() -> list[dict[str, str]]:
    with urlopen(XSTEST_URL) as response:  # nosec B310: pinned public source in protocol
        decoded = response.read().decode("utf-8")
    rows = csv.DictReader(decoded.splitlines())
    return [
        {"source_id": row["id"], "label": row["label"], "text": row["prompt"], "source_revision": XSTEST_COMMIT}
        for row in rows
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the steering calibration pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--ta2-pairs", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--steering-vector", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    blind = commands.add_parser("blind")
    blind.add_argument("--output-dir", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--scores", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        sorry_records, revision = _download_sorry_records()
        prepare_calibration(
            sorry_records=sorry_records, sorry_revision=revision, xstest_rows=_download_xstest_rows(),
            ta2_instructions=_load_ta2_instructions(args.ta2_pairs), output_dir=args.output_dir,
        )
    elif args.command == "generate":
        manifest = json.loads((args.output_dir / "private_prompt_manifest.json").read_text(encoding="utf-8"))
        generate_calibration(private_manifest=manifest, steering_path=args.steering_vector, output_dir=args.output_dir)
    elif args.command == "blind":
        blind_calibration(output_dir=args.output_dir)
    else:
        summarize_calibration_run(output_dir=args.output_dir, scores_path=args.scores)


if __name__ == "__main__":
    main()

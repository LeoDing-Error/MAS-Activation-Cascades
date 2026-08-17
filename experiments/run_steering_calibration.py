"""Prepare, generate, blind, and summarize the steering calibration pilot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
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
    response_content_sha256,
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
MANIFEST_SCHEMA_VERSION = 2
TA2_INSTRUCTION_COUNT = 520
PRIVATE_OUTPUT_FILENAMES = (
    "private_prompt_manifest.json",
    "raw_generations.jsonl",
    f"raw_generations.jsonl.unterminated-tail.{'0' * 64}",
    ".raw_generations.jsonl.repair.tmp",
    f".raw_generations.jsonl.unterminated-tail.{'0' * 64}.repair.tmp",
    "blind_scoring.csv",
    "condition_key.json",
    "manual_scores.csv",
    "summary.json",
)


@dataclass(frozen=True)
class PreparationOutputs:
    private_manifest: dict[str, object]
    public_manifest: dict[str, object]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _repository_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Calibration preparation requires a repository commit") from error
    if not commit:
        raise RuntimeError("Calibration preparation requires a repository commit")
    return commit


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_dict(prompt: CalibrationPrompt, *, include_text: bool) -> dict[str, object]:
    record = asdict(prompt)
    if not include_text:
        record.pop("text")
    return record


def _ensure_private_output_dir(output_dir: Path) -> None:
    """Allow external output locations or only Git-ignored locations in this repo."""
    resolved = output_dir.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return
    try:
        def is_ignored(candidate: Path) -> bool:
            return subprocess.run(
                ["git", "check-ignore", "-q", "--", str(candidate)],
                cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
        ignored = is_ignored(relative) or all(
            is_ignored(relative / filename)
            for filename in PRIVATE_OUTPUT_FILENAMES
        )
    except OSError as error:
        raise RuntimeError("Could not verify that private calibration output is gitignored") from error
    if not ignored:
        raise ValueError("Private calibration output inside the repository must be gitignored")


def prepare_calibration(
    *,
    sorry_records: Sequence[Mapping[str, str]],
    sorry_revision: str,
    xstest_rows: Sequence[Mapping[str, str]],
    ta2_instructions: Sequence[str],
    output_dir: Path,
) -> PreparationOutputs:
    """Select six prompts and create or compare one immutable prepared run."""
    _ensure_private_output_dir(output_dir)
    if not isinstance(sorry_revision, str) or not sorry_revision:
        raise ValueError("SORRY-Bench revision must be a non-empty string")
    validated_ta2 = _validate_ta2_instructions(ta2_instructions)
    selected = [
        *select_sorry_prompts(sorry_records, ta2_instructions=validated_ta2),
        *select_xstest_pair(xstest_rows),
    ]
    repository_commit = _repository_commit()
    ta2_digest = _canonical_sha256(validated_ta2)
    selected_xstest = [
        _prompt_dict(prompt, include_text=False)
        for prompt in selected
        if prompt.source_id in {"1", "26"} and prompt.source.startswith("xstest@")
    ]
    xstest_digest = _canonical_sha256(selected_xstest)
    prepared_public: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "prompts": [_prompt_dict(prompt, include_text=False) for prompt in selected],
        "sorry_revision": sorry_revision,
        "xstest_commit": XSTEST_COMMIT,
        "xstest_attribution": "XSTest by Rottger et al., CC-BY-4.0",
        "repository_commit": repository_commit,
        "ta2_instruction_count": len(validated_ta2),
        "ta2_instructions_sha256": ta2_digest,
        "xstest_selected_count": len(selected_xstest),
        "xstest_selected_sha256": xstest_digest,
        "selection_parameters": _selection_parameters(),
        "generation_parameters": _generation_parameters(),
    }
    run_id = _canonical_sha256(prepared_public)
    private_manifest: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "prompts": [_prompt_dict(prompt, include_text=True) for prompt in selected],
        "sorry_revision": sorry_revision,
        "xstest_commit": XSTEST_COMMIT,
        "repository_commit": repository_commit,
        "ta2_instruction_count": len(validated_ta2),
        "ta2_instructions_sha256": ta2_digest,
        "xstest_selected_count": len(selected_xstest),
        "xstest_selected_sha256": xstest_digest,
    }
    public_manifest: dict[str, object] = {
        **prepared_public,
        "run_id": run_id,
    }
    private_path = output_dir / "private_prompt_manifest.json"
    public_path = output_dir / "run_manifest.json"
    if private_path.exists() != public_path.exists():
        raise ValueError("Existing calibration run must contain both prepared manifests")
    if not private_path.exists():
        _write_json(private_path, private_manifest)
        _write_json(public_path, public_manifest)
        return PreparationOutputs(private_manifest, public_manifest)

    existing_private = _load_json_object(private_path, context="Private prompt manifest")
    existing_public = _load_json_object(public_path, context="Run manifest")
    if existing_private != private_manifest:
        raise ValueError("Prepared provenance does not match the existing calibration run")
    for key, value in public_manifest.items():
        if existing_public.get(key) != value:
            raise ValueError(
                f"Prepared provenance {key} does not match the existing calibration run"
            )
    return PreparationOutputs(existing_private, existing_public)


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    return payload


def _validate_ta2_instructions(instructions: Sequence[str]) -> list[str]:
    if (
        len(instructions) != TA2_INSTRUCTION_COUNT
        or any(not isinstance(instruction, str) or not instruction.strip() for instruction in instructions)
    ):
        raise ValueError("Expected exactly 520 valid TA2 construction instructions")
    return list(instructions)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(record: Mapping[str, object], key: str, *, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must contain a non-empty string {key!r}")
    return value


def _manifest_prompts(manifest: Mapping[str, object]) -> list[CalibrationPrompt]:
    raw_prompts = manifest.get("prompts")
    if not isinstance(raw_prompts, list):
        raise ValueError("Private prompt manifest must contain a prompts list")
    prompts: list[CalibrationPrompt] = []
    for raw in raw_prompts:
        if not isinstance(raw, Mapping):
            raise ValueError("Private prompt manifest contains an invalid prompt")
        prompt = CalibrationPrompt(
            prompt_id=_required_string(raw, "prompt_id", context="Private prompt manifest"),
            source=_required_string(raw, "source", context="Private prompt manifest"),
            source_id=_required_string(raw, "source_id", context="Private prompt manifest"),
            category=_required_string(raw, "category", context="Private prompt manifest"),
            high_level_domain=_required_string(raw, "high_level_domain", context="Private prompt manifest"),
            expected_label=_required_string(raw, "expected_label", context="Private prompt manifest"),
            text=_required_string(raw, "text", context="Private prompt manifest"),
            prompt_sha256=_required_string(raw, "prompt_sha256", context="Private prompt manifest"),
        )
        if prompt.expected_label not in {"safe", "unsafe"}:
            raise ValueError("Private prompt manifest expected_label must be safe or unsafe")
        if hashlib.sha256(prompt.text.encode("utf-8")).hexdigest() != prompt.prompt_sha256:
            raise ValueError("Private prompt manifest prompt_sha256 does not match text")
        prompts.append(prompt)
    if len(prompts) != 6 or len({prompt.prompt_id for prompt in prompts}) != 6:
        raise ValueError("Private prompt manifest must contain exactly six unique prompts")
    return prompts


def _expected_conditions(prompts: Sequence[CalibrationPrompt]) -> set[tuple[str, float]]:
    return {(prompt.prompt_id, alpha) for prompt in prompts for alpha in ALPHAS}


def _valid_diagnostic(value: object) -> bool:
    return value is None or (type(value) in (int, float) and math.isfinite(float(value)))


def _validate_resume_records(
    path: Path,
    *,
    prompts: Sequence[CalibrationPrompt],
    artifact_sha256: str,
    artifact: SteeringVectorArtifact,
    run_id: str,
    repository_commit: str,
    expected_dtype: str | None = None,
) -> set[tuple[str, float]]:
    if not path.exists():
        return set()
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    expected_conditions = _expected_conditions(prompts)
    conditions: set[tuple[str, float]] = set()
    for line_number, line in enumerate(_recoverable_jsonl_lines(path), start=1):
        if not line.strip():
            raise ValueError(f"Invalid resume generation record at line {line_number}: blank lines are not allowed")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid resume generation record at line {line_number}: invalid JSON") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid resume generation record at line {line_number}: expected an object")
        try:
            prompt_id = _required_string(record, "prompt_id", context="Resume generation record")
            prompt = prompt_by_id[prompt_id]
            alpha_value = record.get("alpha")
            if type(alpha_value) not in (int, float) or float(alpha_value) not in ALPHAS:
                raise ValueError("alpha must be a calibration alpha")
            alpha = float(alpha_value)
            condition = (prompt_id, alpha)
            if condition not in expected_conditions:
                raise ValueError("condition is outside the current calibration grid")
            if condition in conditions:
                raise ValueError("duplicate condition")
            for key in ("source", "source_id", "category", "high_level_domain", "expected_label", "prompt_sha256"):
                if record.get(key) != getattr(prompt, key):
                    raise ValueError(f"{key} does not match the private manifest")
            if record.get("prompt_text") != prompt.text:
                raise ValueError("prompt_text does not match the private manifest")
            if not isinstance(record.get("response_text"), str):
                raise ValueError("response_text must be a string")
            completion = record.get("completion_token_count")
            if type(completion) is not int or completion < 0 or completion > MAX_NEW_TOKENS:
                raise ValueError(f"completion_token_count must be an integer in 0..{MAX_NEW_TOKENS}")
            if type(record.get("truncated")) is not bool or record["truncated"] != (completion == MAX_NEW_TOKENS):
                raise ValueError("truncated does not match completion_token_count")
            expected_termination = "length" if completion == MAX_NEW_TOKENS else "stop"
            if record.get("termination_state") != expected_termination:
                raise ValueError("termination_state does not match completion_token_count")
            if record.get("model") != PRIMARY_MODEL or record.get("artifact_sha256") != artifact_sha256:
                raise ValueError("model or artifact_sha256 does not match this run")
            if record.get("run_id") != run_id:
                raise ValueError("run_id does not match the immutable prepared run")
            if record.get("repository_commit") != repository_commit:
                raise ValueError("repository_commit does not match the immutable prepared run")
            if record.get("artifact_layer") != artifact.layer or record.get("artifact_vector_norm") != artifact.vector_norm:
                raise ValueError("artifact provenance does not match this run")
            if record.get("max_new_tokens") != MAX_NEW_TOKENS or record.get("do_sample") is not False:
                raise ValueError("decoding parameters do not match this run")
            if record.get("temperature") != 0.0 or record.get("top_p") != 1.0:
                raise ValueError("decoding parameters do not match this run")
            dtype = _required_string(record, "dtype", context="Resume generation record")
            if expected_dtype is not None and dtype != expected_dtype:
                raise ValueError("dtype does not match this run")
            for key in ("mean_token_entropy", "normalized_sequence_msp"):
                if key not in record or not _valid_diagnostic(record[key]):
                    raise ValueError(f"{key} must be a finite number or null")
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid resume generation record at line {line_number}: {error}") from error
        conditions.add(condition)
    return conditions


def _recoverable_jsonl_lines(path: Path) -> list[str]:
    """Return completed JSONL lines, quarantining only a torn final append."""
    data = path.read_bytes()
    if not data:
        return []
    if not data.endswith(b"\n"):
        prefix, separator, tail = data.rpartition(b"\n")
        try:
            decoded_tail = tail.decode("utf-8")
            parsed_tail = json.loads(decoded_tail)
        except (UnicodeDecodeError, json.JSONDecodeError):
            tail_sha256 = hashlib.sha256(tail).hexdigest()
            quarantine_path = Path(f"{path}.unterminated-tail.{tail_sha256}")
            if quarantine_path.exists() and quarantine_path.read_bytes() != tail:
                raise ValueError("Unterminated JSONL tail quarantine digest collision")
            if not quarantine_path.exists():
                _atomic_write_bytes(quarantine_path, tail)
            _atomic_write_bytes(path, prefix + separator if separator else b"")
            data = prefix + separator if separator else b""
        else:
            if not isinstance(parsed_tail, Mapping):
                raise ValueError("Invalid resume generation record at final line: expected an object")
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
            data += b"\n"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Invalid resume generation record: completed JSONL is not UTF-8") from error
    return text[:-1].split("\n") if text else []


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.repair.tmp")
    with temporary_path.open("wb") as handle:
        handle.write(content)
        handle.flush()
    temporary_path.replace(path)


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _generation_parameters() -> dict[str, object]:
    return {
        "model": PRIMARY_MODEL,
        "alphas": list(ALPHAS),
        "system_prompt": SYSTEM_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def _selection_parameters() -> dict[str, object]:
    return {
        "excluded_category_terms": list(EXCLUDED_CATEGORY_TERMS),
        "overlap_threshold": OVERLAP_THRESHOLD,
        "sorry_domains": list(SORRY_DOMAINS),
        "xstest_ids": ["1", "26"],
    }


def _private_manifest_provenance(
    manifest: Mapping[str, object], prompts: Sequence[CalibrationPrompt],
) -> dict[str, object]:
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Private prompt manifest schema version does not match this runner")
    run_id = _required_sha256(manifest, "run_id", context="Private prompt manifest")
    repository_commit = _required_string(
        manifest, "repository_commit", context="Private prompt manifest"
    )
    sorry_revision = _required_string(manifest, "sorry_revision", context="Private prompt manifest")
    xstest_commit = _required_string(manifest, "xstest_commit", context="Private prompt manifest")
    if xstest_commit != XSTEST_COMMIT:
        raise ValueError("Private prompt manifest xstest_commit does not match the pinned commit")
    if manifest.get("ta2_instruction_count") != TA2_INSTRUCTION_COUNT:
        raise ValueError("Private prompt manifest must record exactly 520 TA2 instructions")
    ta2_digest = _required_sha256(
        manifest, "ta2_instructions_sha256", context="Private prompt manifest"
    )
    if manifest.get("xstest_selected_count") != 2:
        raise ValueError("Private prompt manifest must record exactly two XSTest controls")
    xstest_digest = _required_sha256(
        manifest, "xstest_selected_sha256", context="Private prompt manifest"
    )
    for prompt in prompts:
        if prompt.prompt_id.startswith("sorry-"):
            if prompt.source != "sorry_bench_202503":
                raise ValueError("Private SORRY-Bench prompt source is invalid")
        elif prompt.source != f"xstest@{xstest_commit}":
            raise ValueError("Private XSTest prompt source does not match xstest_commit")
    selected_xstest = [
        _prompt_dict(prompt, include_text=False)
        for prompt in prompts
        if prompt.source.startswith("xstest@")
    ]
    if _canonical_sha256(selected_xstest) != xstest_digest:
        raise ValueError("Private prompt manifest XSTest digest does not match its controls")
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "repository_commit": repository_commit,
        "sorry_revision": sorry_revision,
        "xstest_commit": xstest_commit,
        "ta2_instruction_count": TA2_INSTRUCTION_COUNT,
        "ta2_instructions_sha256": ta2_digest,
        "xstest_selected_count": 2,
        "xstest_selected_sha256": xstest_digest,
    }


def _required_sha256(record: Mapping[str, object], key: str, *, context: str) -> str:
    value = _required_string(record, key, context=context)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{context} {key} must be a SHA-256 digest")
    return value


def _prepared_public_fields(
    prompts: Sequence[CalibrationPrompt], private_provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": private_provenance["run_id"],
        "prompts": [_prompt_dict(prompt, include_text=False) for prompt in prompts],
        "sorry_revision": private_provenance["sorry_revision"],
        "xstest_commit": private_provenance["xstest_commit"],
        "xstest_attribution": "XSTest by Rottger et al., CC-BY-4.0",
        "repository_commit": private_provenance["repository_commit"],
        "ta2_instruction_count": private_provenance["ta2_instruction_count"],
        "ta2_instructions_sha256": private_provenance["ta2_instructions_sha256"],
        "xstest_selected_count": private_provenance["xstest_selected_count"],
        "xstest_selected_sha256": private_provenance["xstest_selected_sha256"],
        "selection_parameters": _selection_parameters(),
        "generation_parameters": _generation_parameters(),
    }


def _load_prepared_run_manifest(
    *, output_dir: Path, prompts: Sequence[CalibrationPrompt], private_provenance: Mapping[str, object],
) -> dict[str, object]:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Run manifest is required; run prepare in this output directory first")
    payload = _load_json_object(manifest_path, context="Run manifest")
    required = _prepared_public_fields(prompts, private_provenance)
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"Run manifest {key} does not match the private manifest or this run")
    if _repository_commit() != required["repository_commit"]:
        raise ValueError("Run manifest does not match the current repository commit")
    prepared_without_run_id = {key: value for key, value in required.items() if key != "run_id"}
    if _canonical_sha256(prepared_without_run_id) != required["run_id"]:
        raise ValueError("Run manifest run_id does not match immutable prepared provenance")
    return payload


def _record_artifact_provenance(
    *, output_dir: Path, prompts: Sequence[CalibrationPrompt], private_provenance: Mapping[str, object], artifact_sha256: str,
    artifact: SteeringVectorArtifact,
) -> dict[str, object]:
    """Bind the TOFU artifact and leave a retryable pre-backend state."""
    manifest_path = output_dir / "run_manifest.json"
    payload = _load_prepared_run_manifest(
        output_dir=output_dir,
        prompts=prompts,
        private_provenance=private_provenance,
    )
    artifact_fields = {
        "artifact_sha256": artifact_sha256,
        "artifact_model": artifact.model_name,
        "artifact_layer": artifact.layer,
        "artifact_vector_norm": artifact.vector_norm,
    }
    existing_artifact_fields = {key for key in (*artifact_fields, "artifact_state") if key in payload}
    if existing_artifact_fields and existing_artifact_fields != {*artifact_fields, "artifact_state"}:
        raise ValueError("Run manifest artifact provenance fields must be complete")
    for key, value in artifact_fields.items():
        if key in payload and payload[key] != value:
            raise ValueError(f"Run manifest {key} does not match this run")
        payload[key] = value
    state = payload.get("artifact_state")
    if state is None:
        payload["artifact_state"] = "pending_backend"
        payload.pop("dtype", None)
        _write_json(manifest_path, payload)
    elif state == "pending_backend":
        if "dtype" in payload:
            raise ValueError("Pending artifact state must not contain dtype")
    elif state == "ready":
        _required_string(payload, "dtype", context="Ready run manifest")
    else:
        raise ValueError("Run manifest artifact_state must be pending_backend or ready")
    return payload


def _seal_artifact_dtype(output_dir: Path, run_manifest: dict[str, object], dtype: str) -> dict[str, object]:
    manifest_path = output_dir / "run_manifest.json"
    state = run_manifest.get("artifact_state")
    if state == "ready":
        if run_manifest.get("dtype") != dtype:
            raise ValueError("Run manifest dtype does not match this run")
        return run_manifest
    if state != "pending_backend":
        raise ValueError("Run manifest artifact is not in a sealable state")
    run_manifest["dtype"] = dtype
    run_manifest["artifact_state"] = "ready"
    _write_json(manifest_path, run_manifest)
    return run_manifest


def generate_calibration(
    *,
    private_manifest: Mapping[str, object],
    steering_path: Path,
    output_dir: Path,
    backend_factory: Callable[..., SteeringModelBackend] = SteeringModelBackend,
) -> None:
    """Generate the complete grid with one model instance and JSONL checkpointing."""
    _ensure_private_output_dir(output_dir)
    prompts = _manifest_prompts(private_manifest)
    private_provenance = _private_manifest_provenance(private_manifest, prompts)
    artifact_sha256 = _sha256_file(steering_path)
    artifact = SteeringVectorArtifact.from_file(steering_path)
    if artifact.model_name != PRIMARY_MODEL or artifact.layer != 25:
        raise ValueError("Calibration artifact must match Llama 3.1 8B layer 25")

    run_manifest = _record_artifact_provenance(
        output_dir=output_dir, prompts=prompts, private_provenance=private_provenance,
        artifact_sha256=artifact_sha256, artifact=artifact,
    )
    records_path = output_dir / "raw_generations.jsonl"
    run_id = _required_sha256(run_manifest, "run_id", context="Run manifest")
    repository_commit = _required_string(
        run_manifest, "repository_commit", context="Run manifest"
    )
    existing = _validate_resume_records(
        records_path, prompts=prompts, artifact_sha256=artifact_sha256, artifact=artifact,
        run_id=run_id, repository_commit=repository_commit,
        expected_dtype=run_manifest.get("dtype") if isinstance(run_manifest.get("dtype"), str) else None,
    )
    if existing and run_manifest["artifact_state"] != "ready":
        raise ValueError("Generation records cannot exist while artifact state is pending_backend")

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
    dtype = str(getattr(backend, "torch_dtype", "unknown"))
    run_manifest = _seal_artifact_dtype(output_dir, run_manifest, dtype)
    _validate_resume_records(
        records_path, prompts=prompts, artifact_sha256=artifact_sha256, artifact=artifact,
        run_id=run_id, repository_commit=repository_commit,
        expected_dtype=dtype,
    )
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
                "run_id": run_id,
                "repository_commit": repository_commit,
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
    final_conditions = _validate_resume_records(
        records_path, prompts=prompts, artifact_sha256=artifact_sha256, artifact=artifact,
        run_id=run_id, repository_commit=repository_commit,
        expected_dtype=dtype,
    )
    if final_conditions != _expected_conditions(prompts):
        raise ValueError("Generation did not produce the complete 36-record calibration grid")


def blind_calibration(*, output_dir: Path) -> None:
    _ensure_private_output_dir(output_dir)
    private_manifest = _load_json_object(
        output_dir / "private_prompt_manifest.json", context="Private prompt manifest"
    )
    prompts = _manifest_prompts(private_manifest)
    private_provenance = _private_manifest_provenance(private_manifest, prompts)
    run_manifest = _load_prepared_run_manifest(
        output_dir=output_dir,
        prompts=prompts,
        private_provenance=private_provenance,
    )
    if run_manifest.get("artifact_state") != "ready":
        raise ValueError("Blinding requires a ready artifact and sealed dtype")
    artifact_sha256 = _required_sha256(
        run_manifest, "artifact_sha256", context="Run manifest"
    )
    artifact = SimpleNamespace(
        layer=run_manifest.get("artifact_layer"),
        vector_norm=run_manifest.get("artifact_vector_norm"),
    )
    dtype = _required_string(run_manifest, "dtype", context="Run manifest")
    run_id = _required_sha256(run_manifest, "run_id", context="Run manifest")
    repository_commit = _required_string(
        run_manifest, "repository_commit", context="Run manifest"
    )
    records_path = output_dir / "raw_generations.jsonl"
    conditions = _validate_resume_records(
        records_path,
        prompts=prompts,
        artifact_sha256=artifact_sha256,
        artifact=artifact,
        run_id=run_id,
        repository_commit=repository_commit,
        expected_dtype=dtype,
    )
    if conditions != _expected_conditions(prompts):
        raise ValueError("Blinding requires the complete 36-record calibration grid")
    records = [json.loads(line) for line in _recoverable_jsonl_lines(records_path)]
    rows, key = build_blinded_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "response_id", "run_id", "blind_run_digest", "response_sha256",
        "prompt_text", "response_text", "refusal_compliance", "coherence", "relevance",
        "repetitive", "truncated",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_bytes_compare(
        output_dir / "blind_scoring.csv",
        buffer.getvalue().encode("utf-8"),
        context="Blinded scoring file",
    )
    _write_json_compare(
        output_dir / "condition_key.json",
        key,
        context="Condition key",
    )


def _write_bytes_compare(path: Path, content: bytes, *, context: str) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"{context} does not match the current immutable run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json_compare(path: Path, payload: object, *, context: str) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    _write_bytes_compare(path, content, context=context)


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
            prompt_text = row["prompt_text"]
            response_text = row["response_text"]
            response_sha256 = row["response_sha256"]
            if response_content_sha256(prompt_text, response_text) != response_sha256:
                raise ValueError("response hash does not match the blinded content")
            scores.append(ManualScore(
                response_id=row["response_id"],
                run_id=row["run_id"],
                blind_run_digest=row["blind_run_digest"],
                response_sha256=response_sha256,
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
    _ensure_private_output_dir(output_dir)
    key = json.loads((output_dir / "condition_key.json").read_text(encoding="utf-8"))
    if not isinstance(key, dict):
        raise ValueError("Condition key must be an object")
    run_ids = {
        condition.get("run_id") for condition in key.values() if isinstance(condition, Mapping)
    }
    blind_digests = {
        condition.get("blind_run_digest")
        for condition in key.values()
        if isinstance(condition, Mapping)
    }
    if len(run_ids) != 1 or len(blind_digests) != 1:
        raise ValueError("Condition key must bind one immutable run and blind-run digest")
    run_manifest = _load_json_object(output_dir / "run_manifest.json", context="Run manifest")
    run_id = _required_sha256(run_manifest, "run_id", context="Run manifest")
    if run_ids != {run_id}:
        raise ValueError("Condition key run_id does not match the current run manifest")
    blind_run_digest = next(iter(blind_digests))
    if not isinstance(blind_run_digest, str):
        raise ValueError("Condition key blind-run digest is invalid")
    summary = summarize_calibration((_load_manual_scores(scores_path), key))
    payload = {
        "run_id": run_id,
        "blind_run_digest": blind_run_digest,
        "manual_scores_sha256": _sha256_file(scores_path),
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


_CURRENT_SORRY_METADATA_NAMES = (
    "category_descriptions",
    "category_descriptions_short",
    "category_descriptions_shortest",
)
_LEGACY_SORRY_METADATA_NAME = "CATEGORY_TO_DOMAIN"
_AUTHORITATIVE_SORRY_METADATA_NAMES = frozenset(
    (*_CURRENT_SORRY_METADATA_NAMES, _LEGACY_SORRY_METADATA_NAME)
)


def _literal_category_mapping(source: str) -> dict[str, str]:
    """Extract a safe, literal category-to-domain mapping from meta_info.py."""
    mapping, _ = _literal_category_mapping_with_schema(source)
    return mapping


def _literal_category_mapping_with_schema(source: str) -> tuple[dict[str, str], bool]:
    """Return a literal mapping and whether it uses the current three-list schema."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Invalid literal SORRY-Bench metadata schema") from error

    assignments = _authoritative_sorry_metadata_assignments(tree)
    current_assignments = {
        name: assignments[name]
        for name in _CURRENT_SORRY_METADATA_NAMES
    }
    legacy_assignments = assignments[_LEGACY_SORRY_METADATA_NAME]
    has_current_schema = any(current_assignments.values())
    has_legacy_schema = bool(legacy_assignments)
    if has_current_schema and has_legacy_schema:
        raise ValueError("SORRY-Bench metadata must contain exactly one schema")

    expected_domains = set(SORRY_DOMAINS)
    if has_current_schema:
        for name, values in current_assignments.items():
            if len(values) != 1:
                raise ValueError("Current SORRY-Bench metadata must contain each category description list once")
            try:
                descriptions = ast.literal_eval(values[0].value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Current SORRY-Bench {name} must be a literal list"
                ) from None
            if (
                not isinstance(descriptions, list)
                or len(descriptions) != 44
                or any(not isinstance(description, str) or not description.strip() for description in descriptions)
            ):
                raise ValueError(
                    "Current SORRY-Bench category_descriptions lists must contain exactly 44 nonempty strings"
                )
        return {
            **{str(category): SORRY_DOMAINS[0] for category in range(1, 6)},
            **{str(category): SORRY_DOMAINS[1] for category in range(6, 25)},
            **{str(category): SORRY_DOMAINS[2] for category in range(25, 40)},
            **{str(category): SORRY_DOMAINS[3] for category in range(40, 45)},
        }, True

    if not has_legacy_schema:
        raise ValueError("Could not find one literal SORRY-Bench category/domain mapping")
    try:
        mapping = ast.literal_eval(legacy_assignments[0].value)
    except (ValueError, TypeError):
        raise ValueError("Could not find one literal SORRY-Bench category/domain mapping") from None
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not all(isinstance(key, str) and key.strip() and isinstance(value, str) for key, value in mapping.items())
        or set(mapping.values()) != expected_domains
    ):
        raise ValueError("Legacy SORRY-Bench category/domain mapping must cover all SORRY domains")
    return {str(key): str(value) for key, value in mapping.items()}, False


def _authoritative_sorry_metadata_assignments(
    tree: ast.Module,
) -> dict[str, list[ast.Assign | ast.AnnAssign]]:
    assignments: dict[str, list[ast.Assign | ast.AnnAssign]] = {
        name: [] for name in _AUTHORITATIVE_SORRY_METADATA_NAMES
    }
    allowed_target_ids: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target
        else:
            continue
        if target.id in _AUTHORITATIVE_SORRY_METADATA_NAMES:
            assignments[target.id].append(node)
            allowed_target_ids.add(id(target))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _AUTHORITATIVE_SORRY_METADATA_NAMES:
            if id(node) not in allowed_target_ids:
                raise ValueError("Authoritative SORRY-Bench metadata names may only have one top-level assignment")
        elif isinstance(node, ast.arg) and node.arg in _AUTHORITATIVE_SORRY_METADATA_NAMES:
            raise ValueError("Authoritative SORRY-Bench metadata names may not be function parameters")
        elif isinstance(node, ast.alias):
            bound_name = node.asname or node.name.split(".", 1)[0]
            if bound_name in _AUTHORITATIVE_SORRY_METADATA_NAMES:
                raise ValueError("Authoritative SORRY-Bench metadata names may not be imported")
        elif isinstance(node, ast.ExceptHandler) and node.name in _AUTHORITATIVE_SORRY_METADATA_NAMES:
            raise ValueError("Authoritative SORRY-Bench metadata names may not be exception targets")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in _AUTHORITATIVE_SORRY_METADATA_NAMES:
                raise ValueError("Authoritative SORRY-Bench metadata names may not be rebound")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if any(name in _AUTHORITATIVE_SORRY_METADATA_NAMES for name in node.names):
                raise ValueError("Authoritative SORRY-Bench metadata names may not be rebound")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in _AUTHORITATIVE_SORRY_METADATA_NAMES:
            raise ValueError("Authoritative SORRY-Bench metadata names may not be pattern targets")
        elif isinstance(node, ast.MatchMapping) and node.rest in _AUTHORITATIVE_SORRY_METADATA_NAMES:
            raise ValueError("Authoritative SORRY-Bench metadata names may not be pattern targets")

    for name, values in assignments.items():
        if len(values) > 1:
            raise ValueError(f"Authoritative SORRY-Bench metadata name {name} must be assigned exactly once")
    return assignments


def _canonical_current_sorry_category(raw_category: object) -> str:
    if type(raw_category) is int:
        category = raw_category
    elif isinstance(raw_category, str) and raw_category.isascii() and raw_category.isdecimal():
        category = int(raw_category)
    else:
        raise ValueError("Expected a canonical SORRY-Bench category in 1..44")
    if not 1 <= category <= 44 or (isinstance(raw_category, str) and raw_category != str(category)):
        raise ValueError("Expected a canonical SORRY-Bench category in 1..44")
    return str(category)


def _load_ta2_instructions(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("TA2 pairs payload must contain a pairs list")
    instructions: list[str] = []
    for pair in pairs:
        if (
            not isinstance(pair, Mapping)
            or not isinstance(pair.get("instruction"), str)
            or not pair["instruction"].strip()
        ):
            raise ValueError("Every TA2 pair must contain an instruction")
        instructions.append(pair["instruction"])
    return _validate_ta2_instructions(instructions)


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
    category_to_domain, current_schema = _literal_category_mapping_with_schema(
        Path(meta_path).read_text(encoding="utf-8")
    )
    records: list[dict[str, str]] = []
    for line in Path(question_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw_category = raw["category"]
        category = _canonical_current_sorry_category(raw_category) if current_schema else str(raw_category)
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

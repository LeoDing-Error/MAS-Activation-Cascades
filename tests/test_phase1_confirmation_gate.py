from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.experiments.phase1_gate import load_held_out_confirmation


def _write_confirmation(path: Path, artifact_path: Path, *, passed: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "phase1_held_out_confirmation",
        "held_out_confirmation_passed": passed,
        "calibration_run_id": "a" * 64,
        "held_out_run_id": "b" * 64,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "selected_alpha": 0.2,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_held_out_confirmation_binds_phase1_to_artifact_and_selected_alpha(tmp_path: Path) -> None:
    artifact_path = tmp_path / "vector.pt"
    artifact_path.write_bytes(b"artifact")
    confirmation_path = tmp_path / "held_out_confirmation.json"
    payload = _write_confirmation(confirmation_path, artifact_path)

    confirmation = load_held_out_confirmation(
        confirmation_path,
        steering_vector_path=artifact_path,
        steering_strength=0.2,
    )

    assert confirmation.calibration_run_id == payload["calibration_run_id"]
    assert confirmation.held_out_run_id == payload["held_out_run_id"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"held_out_confirmation_passed": False}, "has not passed"),
        ({"kind": "calibration_summary"}, "held-out confirmation"),
        ({"selected_alpha": 0.3}, "selected alpha"),
        ({"artifact_sha256": "0" * 64}, "artifact SHA-256"),
    ],
)
def test_held_out_confirmation_rejects_pilot_only_or_mismatched_evidence(
    tmp_path: Path,
    mutation: dict[str, object],
    match: str,
) -> None:
    artifact_path = tmp_path / "vector.pt"
    artifact_path.write_bytes(b"artifact")
    confirmation_path = tmp_path / "held_out_confirmation.json"
    payload = _write_confirmation(confirmation_path, artifact_path)
    payload.update(mutation)
    confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_held_out_confirmation(
            confirmation_path,
            steering_vector_path=artifact_path,
            steering_strength=0.2,
        )


def test_held_out_confirmation_requires_immutable_run_identifiers(tmp_path: Path) -> None:
    artifact_path = tmp_path / "vector.pt"
    artifact_path.write_bytes(b"artifact")
    confirmation_path = tmp_path / "held_out_confirmation.json"
    payload = _write_confirmation(confirmation_path, artifact_path)
    payload["held_out_run_id"] = "not-a-digest"
    confirmation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="held_out_run_id"):
        load_held_out_confirmation(
            confirmation_path,
            steering_vector_path=artifact_path,
            steering_strength=0.2,
        )


@pytest.mark.parametrize("experiment", ["1.1", "1.2", "1.3", "1.4"])
def test_phase1_cli_stops_before_execution_without_held_out_confirmation(
    tmp_path: Path, experiment: str,
) -> None:
    from experiments import run_phase1

    with pytest.raises(ValueError, match="held-out confirmation"):
        run_phase1.main([
            "--experiment", experiment,
            "--steering-vector", str(tmp_path / "vector.pt"),
            "--steering-strength", "0.2",
            "--clean-api-base", "http://127.0.0.1:8000/v1",
            "--results-dir", str(tmp_path / "results"),
        ])

    assert not (tmp_path / "results").exists()


def test_multi_agent_phase1_requires_explicit_confirmed_strength(tmp_path: Path) -> None:
    from experiments import run_phase1

    artifact_path = tmp_path / "vector.pt"
    artifact_path.write_bytes(b"artifact")
    confirmation_path = tmp_path / "held_out_confirmation.json"
    _write_confirmation(confirmation_path, artifact_path)

    with pytest.raises(ValueError, match="explicit --steering-strength"):
        run_phase1.main([
            "--experiment", "1.2",
            "--steering-vector", str(artifact_path),
            "--held-out-confirmation", str(confirmation_path),
            "--clean-api-base", "http://127.0.0.1:8000/v1",
            "--results-dir", str(tmp_path / "results"),
        ])

    assert not (tmp_path / "results").exists()

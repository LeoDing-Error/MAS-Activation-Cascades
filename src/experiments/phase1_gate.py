"""Pure validation for the held-out confirmation required before Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from src.experiments.calibration_protocol import ALPHAS


@dataclass(frozen=True)
class HeldOutConfirmation:
    calibration_run_id: str
    held_out_run_id: str
    artifact_sha256: str
    selected_alpha: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"Could not read steering artifact for held-out confirmation: {path}") from error
    return digest.hexdigest()


def _digest(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Held-out confirmation {key} must be a SHA-256 digest")
    return value


def load_held_out_confirmation(
    path: Path,
    *,
    steering_vector_path: Path,
    steering_strength: float | None,
) -> HeldOutConfirmation:
    """Validate held-out evidence and bind it to this exact Phase 1 treatment."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            "A held-out confirmation file is required before any Phase 1 experiment"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError("Held-out confirmation must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Held-out confirmation must be an object")
    if payload.get("schema_version") != 1 or payload.get("kind") != "phase1_held_out_confirmation":
        raise ValueError("Expected a Phase 1 held-out confirmation record, not a pilot summary")
    if payload.get("held_out_confirmation_passed") is not True:
        raise ValueError("Held-out confirmation has not passed")

    calibration_run_id = _digest(payload, "calibration_run_id")
    held_out_run_id = _digest(payload, "held_out_run_id")
    artifact_sha256 = _digest(payload, "artifact_sha256")
    selected_alpha = payload.get("selected_alpha")
    if type(selected_alpha) not in (int, float):
        raise ValueError(
            "Held-out confirmation selected alpha must be a finite positive calibration candidate alpha"
        )
    selected_alpha_value = float(selected_alpha)
    if not math.isfinite(selected_alpha_value) or selected_alpha_value not in ALPHAS[1:]:
        raise ValueError(
            "Held-out confirmation selected alpha must be a finite positive calibration candidate alpha"
        )
    if steering_strength is not None and selected_alpha_value != float(steering_strength):
        raise ValueError("Phase 1 steering strength does not match the held-out selected alpha")
    if _sha256_file(steering_vector_path) != artifact_sha256:
        raise ValueError("Phase 1 steering artifact SHA-256 does not match held-out confirmation")

    return HeldOutConfirmation(
        calibration_run_id=calibration_run_id,
        held_out_run_id=held_out_run_id,
        artifact_sha256=artifact_sha256,
        selected_alpha=selected_alpha_value,
    )

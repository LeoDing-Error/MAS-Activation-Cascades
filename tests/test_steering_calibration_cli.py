from __future__ import annotations

import json
from pathlib import Path

import torch

from src.backends.steering_backend import GenerationResult
from src.experiments.calibration_protocol import ALPHAS, SORRY_DOMAINS, select_sorry_prompts, select_xstest_pair
from src.experiments.phase1_config import PRIMARY_MODEL


def make_sorry_records() -> list[dict[str, str]]:
    return [
        {
            "source_id": str(index),
            "category": "eligible",
            "high_level_domain": domain,
            "text": f"SORRY prompt {index}",
        }
        for index, domain in enumerate(SORRY_DOMAINS, start=1)
    ]


def make_xstest_rows() -> list[dict[str, str]]:
    return [
        {"source_id": "1", "label": "safe", "text": "Safe XSTest prompt"},
        {"source_id": "26", "label": "unsafe", "text": "Unsafe XSTest prompt"},
    ]


def make_private_manifest() -> dict[str, object]:
    prompts = [
        *select_sorry_prompts(make_sorry_records(), ta2_instructions=[]),
        *select_xstest_pair(make_xstest_rows()),
    ]
    return {
        "prompts": [
            {
                "prompt_id": prompt.prompt_id,
                "source": prompt.source,
                "source_id": prompt.source_id,
                "category": prompt.category,
                "high_level_domain": prompt.high_level_domain,
                "expected_label": prompt.expected_label,
                "text": prompt.text,
                "prompt_sha256": prompt.prompt_sha256,
            }
            for prompt in prompts
        ]
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_existing_record(path: Path, *, prompt_id: str, alpha: float) -> None:
    manifest = make_private_manifest()
    prompt = next(item for item in manifest["prompts"] if item["prompt_id"] == prompt_id)
    path.write_text(
        json.dumps({
            "prompt_id": prompt_id,
            "alpha": alpha,
            "prompt_text": prompt["text"],
            "response_text": "already generated",
        }) + "\n",
        encoding="utf-8",
    )


def write_artifact(path: Path) -> None:
    torch.save(
        {
            "model_name": PRIMARY_MODEL,
            "selected_layer": 25,
            "vector": torch.ones(4),
        },
        path,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.strengths: list[float] = []
        self.enabled: list[bool] = []

    def set_steering_strength(self, alpha: float) -> None:
        self.strengths.append(alpha)

    def set_steering_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def generate_from_messages(self, messages: list[dict[str, str]]) -> GenerationResult:
        return GenerationResult(
            prompt_text=messages[-1]["content"],
            response_text=f"response to {messages[-1]['content']}",
            prompt_token_count=2,
            completion_token_count=3,
            generated_token_ids=[1, 2, 3],
            step_logits=None,
        )


class FakeBackendFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.backend = FakeBackend()

    def __call__(self, **_: object) -> FakeBackend:
        self.calls += 1
        return self.backend


def test_prepare_writes_private_prompts_and_public_provenance(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(),
        sorry_revision="sorry-sha",
        xstest_rows=make_xstest_rows(),
        ta2_instructions=["training-only prompt"],
        output_dir=tmp_path,
    )

    assert len(outputs.private_manifest["prompts"]) == 6
    assert "text" not in outputs.public_manifest["prompts"][0]
    assert outputs.public_manifest["sorry_revision"] == "sorry-sha"
    assert (tmp_path / "private_prompt_manifest.json").exists()
    assert (tmp_path / "run_manifest.json").exists()


def test_generate_is_resumable_and_reuses_one_backend(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    factory = FakeBackendFactory()
    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    write_existing_record(tmp_path / "raw_generations.jsonl", prompt_id="sorry-1", alpha=0.0)

    generate_calibration(
        private_manifest=make_private_manifest(),
        steering_path=artifact_path,
        output_dir=tmp_path,
        backend_factory=factory,
    )

    assert factory.calls == 1
    records = read_jsonl(tmp_path / "raw_generations.jsonl")
    assert len(records) == 36
    assert len({(row["prompt_id"], row["alpha"]) for row in records}) == 36
    assert factory.backend.strengths == list(ALPHAS)
    assert factory.backend.enabled == [alpha > 0.0 for alpha in ALPHAS]

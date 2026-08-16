from __future__ import annotations

import json
import hashlib
import csv
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from src.experiments.calibration_protocol import ALPHAS, SORRY_DOMAINS, select_sorry_prompts, select_xstest_pair
from src.experiments.phase1_config import PRIMARY_MODEL


@dataclass
class FakeGenerationResult:
    prompt_text: str
    response_text: str
    prompt_token_count: int
    completion_token_count: int
    generated_token_ids: list[int]
    step_logits: object | None


@dataclass(frozen=True)
class FakeSteeringVectorArtifact:
    model_name: str
    layer: int
    vector_norm: float

    @classmethod
    def from_file(cls, path: Path) -> "FakeSteeringVectorArtifact":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_name=payload["model_name"],
            layer=payload["selected_layer"],
            vector_norm=payload.get("vector_norm", 2.0),
        )


@pytest.fixture(autouse=True)
def _provide_local_backend_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI tests independent of collection-time backend and torch fakes."""
    backend_package = ModuleType("src.backends")
    backend_package.__path__ = []
    backend_module = ModuleType("src.backends.steering_backend")
    backend_module.GenerationResult = FakeGenerationResult
    backend_module.SteeringModelBackend = object
    backend_module.SteeringVectorArtifact = FakeSteeringVectorArtifact
    backend_package.steering_backend = backend_module
    metrics_package = ModuleType("src.metrics")
    metrics_package.__path__ = []
    uncertainty_module = ModuleType("src.metrics.uncertainty")
    uncertainty_module.compute_uncertainty_snapshot = lambda *_args, **_kwargs: SimpleNamespace(
        mean_token_entropy=None, normalized_sequence_msp=None,
    )
    metrics_package.uncertainty = uncertainty_module
    monkeypatch.setitem(sys.modules, "src.backends", backend_package)
    monkeypatch.setitem(sys.modules, "src.backends.steering_backend", backend_module)
    monkeypatch.setitem(sys.modules, "src.metrics", metrics_package)
    monkeypatch.setitem(sys.modules, "src.metrics.uncertainty", uncertainty_module)
    sys.modules.pop("experiments.run_steering_calibration", None)


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
        "sorry_revision": "sorry-sha",
        "xstest_commit": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
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


def write_existing_record(path: Path, *, prompt_id: str, alpha: float, steering_path: Path) -> None:
    manifest = make_private_manifest()
    prompt = next(item for item in manifest["prompts"] if item["prompt_id"] == prompt_id)
    path.write_text(
        json.dumps({
            "prompt_id": prompt_id,
            "source": prompt["source"],
            "source_id": prompt["source_id"],
            "category": prompt["category"],
            "high_level_domain": prompt["high_level_domain"],
            "expected_label": prompt["expected_label"],
            "prompt_sha256": prompt["prompt_sha256"],
            "alpha": alpha,
            "prompt_text": prompt["text"],
            "response_text": "already generated",
            "completion_token_count": 3,
            "truncated": False,
            "termination_state": "stop",
            "model": PRIMARY_MODEL,
            "dtype": "unknown",
            "artifact_sha256": hashlib.sha256(steering_path.read_bytes()).hexdigest(),
            "artifact_layer": 25,
            "artifact_vector_norm": 2.0,
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "mean_token_entropy": None,
            "normalized_sequence_msp": None,
        }) + "\n",
        encoding="utf-8",
    )


def write_artifact(path: Path) -> None:
    path.write_text(json.dumps({"model_name": PRIMARY_MODEL, "selected_layer": 25}), encoding="utf-8")


class FakeBackend:
    def __init__(self) -> None:
        self.strengths: list[float] = []
        self.enabled: list[bool] = []
        self.messages: list[list[dict[str, str]]] = []

    def set_steering_strength(self, alpha: float) -> None:
        self.strengths.append(alpha)

    def set_steering_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def generate_from_messages(self, messages: list[dict[str, str]]) -> FakeGenerationResult:
        self.messages.append(messages)
        return FakeGenerationResult(
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
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> FakeBackend:
        self.calls += 1
        self.kwargs = kwargs
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
    write_existing_record(
        tmp_path / "raw_generations.jsonl", prompt_id="sorry-1", alpha=0.0,
        steering_path=artifact_path,
    )

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
    assert factory.kwargs is not None
    assert factory.kwargs["model_name"] == PRIMARY_MODEL
    assert factory.kwargs["alpha"] == 0.0
    assert factory.kwargs["steering_enabled"] is False
    assert factory.kwargs["max_new_tokens"] == 256
    assert factory.kwargs["do_sample"] is False
    assert factory.kwargs["temperature"] == 0.0
    assert factory.kwargs["top_p"] == 1.0
    assert factory.backend.messages[0] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "SORRY prompt 2"},
    ]
    generated = next(row for row in records if row["prompt_id"] == "sorry-2" and row["alpha"] == 0.0)
    assert generated["truncated"] is False
    assert generated["termination_state"] == "stop"
    assert generated["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert generated["mean_token_entropy"] is None
    assert generated["normalized_sequence_msp"] is None


def test_generate_records_artifact_sha_in_run_manifest_before_factory_use(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    generate_calibration(
        private_manifest=make_private_manifest(),
        steering_path=artifact_path,
        output_dir=tmp_path,
        backend_factory=FakeBackendFactory(),
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert manifest["artifact_model"] == PRIMARY_MODEL
    assert manifest["artifact_layer"] == 25


def test_generate_rejects_corrupt_resume_record_before_skipping(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    (tmp_path / "raw_generations.jsonl").write_text(
        json.dumps({"prompt_id": "sorry-1", "alpha": 0.0}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="resume generation record"):
        generate_calibration(
            private_manifest=make_private_manifest(), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_private_manifest_requires_strings_and_matching_prompt_hash() -> None:
    from experiments.run_steering_calibration import _manifest_prompts

    manifest = make_private_manifest()
    manifest["prompts"][0]["prompt_sha256"] = "stale"
    with pytest.raises(ValueError, match="prompt_sha256"):
        _manifest_prompts(manifest)

    manifest = make_private_manifest()
    manifest["prompts"][0]["text"] = 42
    with pytest.raises(ValueError, match="non-empty string"):
        _manifest_prompts(manifest)


def test_private_output_under_repository_must_be_gitignored() -> None:
    from experiments.run_steering_calibration import ROOT, _ensure_private_output_dir

    _ensure_private_output_dir(ROOT / "results" / "calibration-pilot")
    with pytest.raises(ValueError, match="gitignored"):
        _ensure_private_output_dir(ROOT / "experiments" / "calibration-pilot")


def test_artifact_rejection_happens_before_factory_construction(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "wrong.pt"
    artifact_path.write_text(json.dumps({"model_name": "wrong", "selected_layer": 25}), encoding="utf-8")
    factory = FakeBackendFactory()
    with pytest.raises(ValueError, match="Llama 3.1 8B layer 25"):
        generate_calibration(
            private_manifest=make_private_manifest(), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=factory,
        )
    assert factory.calls == 0


def test_incompatible_resume_artifact_sha_is_rejected(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    write_existing_record(
        tmp_path / "raw_generations.jsonl", prompt_id="sorry-1", alpha=0.0,
        steering_path=artifact_path,
    )
    rows = read_jsonl(tmp_path / "raw_generations.jsonl")
    rows[0]["artifact_sha256"] = "wrong"
    (tmp_path / "raw_generations.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_sha256"):
        generate_calibration(
            private_manifest=make_private_manifest(), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_duplicate_resume_condition_is_rejected(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    records_path = tmp_path / "raw_generations.jsonl"
    write_existing_record(records_path, prompt_id="sorry-1", alpha=0.0, steering_path=artifact_path)
    records_path.write_text(records_path.read_text(encoding="utf-8") * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate condition"):
        generate_calibration(
            private_manifest=make_private_manifest(), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def _generate_complete_records(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    generate_calibration(
        private_manifest=make_private_manifest(), steering_path=artifact_path,
        output_dir=tmp_path, backend_factory=FakeBackendFactory(),
    )


def _write_scores(path: Path, key: dict[str, dict[str, object]], *, selected_alpha: float | None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["response_id", "refusal_compliance", "coherence", "relevance", "repetitive", "truncated"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for response_id, condition in key.items():
            prompt_id = condition["prompt_id"]
            alpha = condition["alpha"]
            refusal = 2 if prompt_id == "xstest-1" else 0
            if selected_alpha == alpha and prompt_id != "xstest-1":
                refusal = 1
            writer.writerow({
                "response_id": response_id, "refusal_compliance": refusal, "coherence": 2,
                "relevance": 2, "repetitive": "false", "truncated": "false",
            })


def test_blind_hides_conditions_and_writes_separate_key(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import blind_calibration

    _generate_complete_records(tmp_path)
    blind_calibration(output_dir=tmp_path)

    rows = list(csv.DictReader((tmp_path / "blind_scoring.csv").open(encoding="utf-8")))
    key = json.loads((tmp_path / "condition_key.json").read_text(encoding="utf-8"))
    assert len(rows) == len(key) == 36
    assert all("alpha" not in row and "condition" not in row for row in rows)
    assert set(row["response_id"] for row in rows) == set(key)


def test_summarize_requires_complete_scores_and_reports_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from experiments.run_steering_calibration import blind_calibration, summarize_calibration_run

    _generate_complete_records(tmp_path)
    blind_calibration(output_dir=tmp_path)
    key = json.loads((tmp_path / "condition_key.json").read_text(encoding="utf-8"))
    scores_path = tmp_path / "manual_scores.csv"
    _write_scores(scores_path, key, selected_alpha=0.2)
    summary = summarize_calibration_run(output_dir=tmp_path, scores_path=scores_path)
    assert summary["selected_alpha"] == 0.2
    assert "SELECTED_ALPHA=0.2" in capsys.readouterr().out

    no_selection_path = tmp_path / "no_selection_scores.csv"
    _write_scores(no_selection_path, key, selected_alpha=None)
    no_selection = summarize_calibration_run(output_dir=tmp_path, scores_path=no_selection_path)
    assert no_selection["selected_alpha"] is None
    assert "NO_COHERENT_ALPHA" in capsys.readouterr().out

    incomplete_path = tmp_path / "incomplete_scores.csv"
    _write_scores(incomplete_path, dict(list(key.items())[1:]), selected_alpha=0.2)
    with pytest.raises(ValueError, match="exactly one score"):
        summarize_calibration_run(output_dir=tmp_path, scores_path=incomplete_path)


def test_literal_meta_mapping_refuses_nonliteral_code_and_gated_access_has_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.run_steering_calibration import _download_sorry_records, _literal_category_mapping

    with pytest.raises(ValueError, match="literal SORRY-Bench"):
        _literal_category_mapping("CATEGORY_TO_DOMAIN = build_mapping()")

    class FailingApi:
        def dataset_info(self, _: str) -> object:
            raise __import__("urllib.error", fromlist=["HTTPError"]).HTTPError("url", 401, "unauthorized", {}, None)

    class FakeHubModule:
        HfApi = FailingApi
        hf_hub_download = staticmethod(lambda **_: "unused")

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHubModule)
    with pytest.raises(RuntimeError, match="Accept the sorry-bench/sorry-bench-202503 license.*HF_TOKEN"):
        _download_sorry_records()


def test_sorry_download_uses_resolved_revision_for_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.run_steering_calibration import _download_sorry_records

    question_path = tmp_path / "question.jsonl"
    question_path.write_text("\n".join(
        json.dumps({"question_id": index, "category": f"category-{index}", "turns": [f"prompt {index}"]})
        for index in range(1, 5)
    ), encoding="utf-8")
    meta_path = tmp_path / "meta_info.py"
    meta_path.write_text(
        "CATEGORY_TO_DOMAIN = {\n"
        + ",\n".join(f"    'category-{index}': '{domain}'" for index, domain in enumerate(SORRY_DOMAINS, start=1))
        + "\n}\n",
        encoding="utf-8",
    )
    downloads: list[dict[str, object]] = []

    class FakeApi:
        def dataset_info(self, repo_id: str) -> SimpleNamespace:
            assert repo_id == "sorry-bench/sorry-bench-202503"
            return SimpleNamespace(sha="resolved-sha")

    def fake_download(repo_id: str, **kwargs: object) -> str:
        assert repo_id == "sorry-bench/sorry-bench-202503"
        downloads.append(kwargs)
        return str(question_path if kwargs["filename"] == "question.jsonl" else meta_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download))
    records, revision = _download_sorry_records()
    assert revision == "resolved-sha"
    assert [call["filename"] for call in downloads] == ["question.jsonl", "meta_info.py"]
    assert all(call["revision"] == "resolved-sha" and call["repo_type"] == "dataset" for call in downloads)
    assert [record["high_level_domain"] for record in records] == list(SORRY_DOMAINS)


def test_generate_requires_complete_matching_public_manifest(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration, prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(), sorry_revision="sorry-sha", xstest_rows=make_xstest_rows(),
        ta2_instructions=[], output_dir=tmp_path,
    )
    public = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    del public["prompts"]
    (tmp_path / "run_manifest.json").write_text(json.dumps(public), encoding="utf-8")
    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    with pytest.raises(ValueError, match="Run manifest prompts"):
        generate_calibration(
            private_manifest=outputs.private_manifest, steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_generate_rejects_public_private_revision_mismatch(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration, prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(), sorry_revision="sorry-sha", xstest_rows=make_xstest_rows(),
        ta2_instructions=[], output_dir=tmp_path,
    )
    public = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    public["sorry_revision"] = "different-sha"
    (tmp_path / "run_manifest.json").write_text(json.dumps(public), encoding="utf-8")
    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    with pytest.raises(ValueError, match="sorry_revision"):
        generate_calibration(
            private_manifest=outputs.private_manifest, steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_generate_rejects_mismatched_public_generation_parameters(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration, prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(), sorry_revision="sorry-sha", xstest_rows=make_xstest_rows(),
        ta2_instructions=[], output_dir=tmp_path,
    )
    public = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    public["generation_parameters"]["top_p"] = 0.9
    (tmp_path / "run_manifest.json").write_text(json.dumps(public), encoding="utf-8")
    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    with pytest.raises(ValueError, match="generation_parameters"):
        generate_calibration(
            private_manifest=outputs.private_manifest, steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("alpha", 0.9, "calibration alpha"),
        ("model", "wrong-model", "model or artifact_sha256"),
        ("temperature", 0.5, "decoding parameters"),
        ("completion_token_count", 257, "completion_token_count"),
    ],
)
def test_generate_rejects_invalid_resume_provenance(
    tmp_path: Path, field: str, value: object, match: str,
) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    records_path = tmp_path / "raw_generations.jsonl"
    write_existing_record(records_path, prompt_id="sorry-1", alpha=0.0, steering_path=artifact_path)
    record = read_jsonl(records_path)[0]
    record[field] = value
    if field == "completion_token_count":
        record["truncated"] = False
        record["termination_state"] = "stop"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        generate_calibration(
            private_manifest=make_private_manifest(), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_generate_raises_when_final_jsonl_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.run_steering_calibration as runner

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    real_append = runner._append_jsonl
    calls = 0

    def drop_one(path: Path, record: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls != 1:
            real_append(path, record)

    monkeypatch.setattr(runner, "_append_jsonl", drop_one)
    with pytest.raises(ValueError, match="complete 36-record"):
        runner.generate_calibration(
            private_manifest=make_private_manifest(), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )

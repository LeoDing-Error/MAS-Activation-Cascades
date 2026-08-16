from __future__ import annotations

import json
import hashlib
import csv
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from src.experiments.calibration_protocol import (
    ALPHAS,
    SORRY_DOMAINS,
    XSTEST_COMMIT,
    select_sorry_prompts,
    select_xstest_pair,
)
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


@contextmanager
def _local_backend_shapes():
    """Keep CLI tests independent of collection-time backend and torch fakes."""
    module_names = (
        "src.backends",
        "src.backends.steering_backend",
        "src.metrics",
        "src.metrics.uncertainty",
        "experiments.run_steering_calibration",
    )
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in module_names}
    previous_attributes = {}
    for name in module_names:
        parent_name, attribute = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        previous_attributes[(parent_name, attribute)] = (
            getattr(parent, attribute, missing) if parent is not None else missing
        )
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
    sys.modules["src.backends"] = backend_package
    sys.modules["src.backends.steering_backend"] = backend_module
    sys.modules["src.metrics"] = metrics_package
    sys.modules["src.metrics.uncertainty"] = uncertainty_module
    sys.modules.pop("experiments.run_steering_calibration", None)
    for name in module_names:
        parent_name, attribute = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        module = sys.modules.get(name)
        if parent is not None:
            if module is None:
                vars(parent).pop(attribute, None)
            else:
                setattr(parent, attribute, module)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for (parent_name, attribute), value in previous_attributes.items():
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if value is missing:
                vars(parent).pop(attribute, None)
            else:
                setattr(parent, attribute, value)


@pytest.fixture(autouse=True)
def _provide_local_backend_shapes() -> None:
    with _local_backend_shapes():
        yield


def test_local_backend_shape_context_restores_modules_and_parent_attributes() -> None:
    tracked_names = (
        "src.backends",
        "src.backends.steering_backend",
        "src.metrics",
        "src.metrics.uncertainty",
        "experiments.run_steering_calibration",
    )
    experiments_package = importlib.import_module("experiments")
    sys.modules.pop("experiments.run_steering_calibration", None)
    vars(experiments_package).pop("run_steering_calibration", None)
    missing = object()
    before_modules = {name: sys.modules.get(name, missing) for name in tracked_names}
    before_attributes = {}
    for name in tracked_names:
        parent_name, attribute = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        before_attributes[(parent_name, attribute)] = (
            getattr(parent, attribute, missing) if parent is not None else missing
        )

    with _local_backend_shapes():
        importlib.import_module("experiments.run_steering_calibration")

    for name, module in before_modules.items():
        assert sys.modules.get(name, missing) is module
    for (parent_name, attribute), value in before_attributes.items():
        parent = sys.modules.get(parent_name)
        assert (getattr(parent, attribute, missing) if parent is not None else missing) is value


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
        {
            "source_id": "1",
            "label": "safe",
            "text": "Safe XSTest prompt",
            "source_revision": XSTEST_COMMIT,
        },
        {
            "source_id": "26",
            "label": "unsafe",
            "text": "Unsafe XSTest prompt",
            "source_revision": XSTEST_COMMIT,
        },
    ]


def make_ta2_instructions(count: int = 520) -> list[str]:
    return [f"private TA2 construction instruction {index:04d}" for index in range(count)]


def make_private_manifest(output_dir: Path) -> dict[str, object]:
    from experiments.run_steering_calibration import prepare_calibration

    return prepare_calibration(
        sorry_records=make_sorry_records(),
        sorry_revision="sorry-sha",
        xstest_rows=make_xstest_rows(),
        ta2_instructions=make_ta2_instructions(),
        output_dir=output_dir,
    ).private_manifest


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_existing_record(path: Path, *, prompt_id: str, alpha: float, steering_path: Path) -> None:
    manifest = make_private_manifest(path.parent)
    public_path = path.parent / "run_manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public.update({
        "artifact_state": "ready",
        "artifact_sha256": hashlib.sha256(steering_path.read_bytes()).hexdigest(),
        "artifact_model": PRIMARY_MODEL,
        "artifact_layer": 25,
        "artifact_vector_norm": 2.0,
        "dtype": "unknown",
    })
    public_path.write_text(json.dumps(public, indent=2, sort_keys=True), encoding="utf-8")
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
            "run_id": public["run_id"],
            "repository_commit": public["repository_commit"],
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
        ta2_instructions=make_ta2_instructions(),
        output_dir=tmp_path,
    )

    assert len(outputs.private_manifest["prompts"]) == 6
    assert "text" not in outputs.public_manifest["prompts"][0]
    assert outputs.public_manifest["sorry_revision"] == "sorry-sha"
    assert outputs.private_manifest["run_id"] == outputs.public_manifest["run_id"]
    assert len(str(outputs.public_manifest["run_id"])) == 64
    assert outputs.public_manifest["ta2_instruction_count"] == 520
    assert len(str(outputs.public_manifest["ta2_instructions_sha256"])) == 64
    assert outputs.public_manifest["xstest_selected_count"] == 2
    assert len(str(outputs.public_manifest["xstest_selected_sha256"])) == 64
    assert (tmp_path / "private_prompt_manifest.json").exists()
    assert (tmp_path / "run_manifest.json").exists()


def test_prepare_compares_existing_manifests_without_erasing_artifact_state(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import prepare_calibration

    kwargs = {
        "sorry_records": make_sorry_records(),
        "sorry_revision": "sorry-sha",
        "xstest_rows": make_xstest_rows(),
        "ta2_instructions": make_ta2_instructions(),
        "output_dir": tmp_path,
    }
    prepare_calibration(**kwargs)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "artifact_state": "ready",
        "artifact_sha256": "b" * 64,
        "artifact_model": PRIMARY_MODEL,
        "artifact_layer": 25,
        "artifact_vector_norm": 2.0,
        "dtype": "torch.bfloat16",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    before = manifest_path.read_bytes()

    outputs = prepare_calibration(**kwargs)

    assert manifest_path.read_bytes() == before
    assert outputs.public_manifest["artifact_state"] == "ready"


def test_prepare_rejects_changed_provenance_in_an_existing_output_dir(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import prepare_calibration

    prepare_calibration(
        sorry_records=make_sorry_records(),
        sorry_revision="sorry-sha",
        xstest_rows=make_xstest_rows(),
        ta2_instructions=make_ta2_instructions(),
        output_dir=tmp_path,
    )
    before = (tmp_path / "run_manifest.json").read_bytes()

    with pytest.raises(ValueError, match="existing calibration run"):
        prepare_calibration(
            sorry_records=make_sorry_records(),
            sorry_revision="different-sorry-sha",
            xstest_rows=make_xstest_rows(),
            ta2_instructions=make_ta2_instructions(),
            output_dir=tmp_path,
        )

    assert (tmp_path / "run_manifest.json").read_bytes() == before


def test_prepare_requires_exactly_520_valid_ta2_instructions(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import prepare_calibration

    for instructions in (make_ta2_instructions(519), [*make_ta2_instructions(519), ""]):
        with pytest.raises(ValueError, match="exactly 520 valid TA2"):
            prepare_calibration(
                sorry_records=make_sorry_records(),
                sorry_revision="sorry-sha",
                xstest_rows=make_xstest_rows(),
                ta2_instructions=instructions,
                output_dir=tmp_path,
            )


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
        private_manifest=make_private_manifest(tmp_path),
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
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert generated["run_id"] == manifest["run_id"]
    assert generated["repository_commit"] == manifest["repository_commit"]
    assert generated["mean_token_entropy"] is None
    assert generated["normalized_sequence_msp"] is None


def test_generate_records_artifact_sha_in_run_manifest_before_factory_use(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    generate_calibration(
        private_manifest=make_private_manifest(tmp_path),
        steering_path=artifact_path,
        output_dir=tmp_path,
        backend_factory=FakeBackendFactory(),
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert manifest["artifact_model"] == PRIMARY_MODEL
    assert manifest["artifact_layer"] == 25
    assert manifest["artifact_state"] == "ready"


def test_backend_construction_failure_leaves_retryable_same_artifact_state(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration, prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(),
        sorry_revision="sorry-sha",
        xstest_rows=make_xstest_rows(),
        ta2_instructions=make_ta2_instructions(),
        output_dir=tmp_path,
    )
    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)

    class BackendConstructionFailure:
        def __call__(self, **_kwargs: object) -> FakeBackend:
            raise RuntimeError("backend construction interrupted")

    with pytest.raises(RuntimeError, match="backend construction interrupted"):
        generate_calibration(
            private_manifest=outputs.private_manifest,
            steering_path=artifact_path,
            output_dir=tmp_path,
            backend_factory=BackendConstructionFailure(),
        )

    pending = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert pending["artifact_state"] == "pending_backend"
    assert pending["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert "dtype" not in pending

    generate_calibration(
        private_manifest=outputs.private_manifest,
        steering_path=artifact_path,
        output_dir=tmp_path,
        backend_factory=FakeBackendFactory(),
    )
    ready = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert ready["artifact_state"] == "ready"
    assert len(read_jsonl(tmp_path / "raw_generations.jsonl")) == 36


def test_generate_rejects_corrupt_resume_record_before_skipping(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    (tmp_path / "raw_generations.jsonl").write_text(
        json.dumps({"prompt_id": "sorry-1", "alpha": 0.0}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="resume generation record"):
        generate_calibration(
            private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_private_manifest_requires_strings_and_matching_prompt_hash(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import _manifest_prompts

    manifest = make_private_manifest(tmp_path)
    manifest["prompts"][0]["prompt_sha256"] = "stale"
    with pytest.raises(ValueError, match="prompt_sha256"):
        _manifest_prompts(manifest)

    manifest = make_private_manifest(tmp_path)
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
            private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
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
            private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
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
            private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )


def test_generate_quarantines_only_an_unterminated_final_jsonl_fragment(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    _generate_complete_records(tmp_path)
    records_path = tmp_path / "raw_generations.jsonl"
    complete_bytes = records_path.read_bytes()
    records_path.write_bytes(complete_bytes + b'{"prompt_id":"interrupted"')
    artifact_path = tmp_path / "vector.pt"

    generate_calibration(
        private_manifest=make_private_manifest(tmp_path),
        steering_path=artifact_path,
        output_dir=tmp_path,
        backend_factory=FakeBackendFactory(),
    )

    assert records_path.read_bytes() == complete_bytes
    quarantined = list(tmp_path.glob("raw_generations.jsonl.unterminated-tail*"))
    assert [path.read_bytes() for path in quarantined] == [b'{"prompt_id":"interrupted"']
    assert len(read_jsonl(records_path)) == 36


def test_generate_preserves_two_successive_distinct_unterminated_final_fragments(
    tmp_path: Path,
) -> None:
    from experiments.run_steering_calibration import generate_calibration

    _generate_complete_records(tmp_path)
    records_path = tmp_path / "raw_generations.jsonl"
    complete_bytes = records_path.read_bytes()
    artifact_path = tmp_path / "vector.pt"
    tails = (b'{"prompt_id":"first-interruption"', b'{"prompt_id":"second-interruption"')

    for tail in tails:
        records_path.write_bytes(complete_bytes + tail)
        generate_calibration(
            private_manifest=make_private_manifest(tmp_path),
            steering_path=artifact_path,
            output_dir=tmp_path,
            backend_factory=FakeBackendFactory(),
        )
        assert records_path.read_bytes() == complete_bytes

    quarantined = list(tmp_path.glob("raw_generations.jsonl.unterminated-tail*"))
    assert len(quarantined) == 2
    assert {path.read_bytes() for path in quarantined} == set(tails)


def test_generate_rejects_a_malformed_completed_jsonl_record(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    _generate_complete_records(tmp_path)
    records_path = tmp_path / "raw_generations.jsonl"
    records_path.write_bytes(records_path.read_bytes() + b'{"prompt_id":"completed-but-broken"\n')

    with pytest.raises(ValueError, match="invalid JSON"):
        generate_calibration(
            private_manifest=make_private_manifest(tmp_path),
            steering_path=tmp_path / "vector.pt",
            output_dir=tmp_path,
            backend_factory=FakeBackendFactory(),
        )


def _generate_complete_records(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration

    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    generate_calibration(
        private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
        output_dir=tmp_path, backend_factory=FakeBackendFactory(),
    )


def _write_scores(path: Path, key: dict[str, dict[str, object]], *, selected_alpha: float | None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "response_id", "run_id", "blind_run_digest", "response_sha256",
            "prompt_text", "response_text",
            "refusal_compliance", "coherence", "relevance", "repetitive", "truncated",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for response_id, condition in key.items():
            prompt_id = condition["prompt_id"]
            alpha = condition["alpha"]
            refusal = 2 if prompt_id == "xstest-1" else 0
            if selected_alpha == alpha and prompt_id != "xstest-1":
                refusal = 1
            writer.writerow({
                "response_id": response_id,
                "run_id": condition["run_id"],
                "blind_run_digest": condition["blind_run_digest"],
                "response_sha256": condition["response_sha256"],
                "prompt_text": condition["prompt_text"],
                "response_text": condition["response_text"],
                "refusal_compliance": refusal, "coherence": 2,
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
    assert len({row["run_id"] for row in rows}) == 1
    assert len({row["blind_run_digest"] for row in rows}) == 1
    assert all(row["response_sha256"] == key[row["response_id"]]["response_sha256"] for row in rows)


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


def test_summarize_rejects_scores_from_stale_blinded_content(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import blind_calibration, summarize_calibration_run

    _generate_complete_records(tmp_path)
    blind_calibration(output_dir=tmp_path)
    key = json.loads((tmp_path / "condition_key.json").read_text(encoding="utf-8"))
    scores_path = tmp_path / "manual_scores.csv"
    _write_scores(scores_path, key, selected_alpha=0.2)
    rows = list(csv.DictReader(scores_path.open(encoding="utf-8")))
    rows[0]["response_sha256"] = "0" * 64
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="blinded content"):
        summarize_calibration_run(output_dir=tmp_path, scores_path=scores_path)


def test_summarize_recomputes_score_binding_from_visible_response_text(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import blind_calibration, summarize_calibration_run

    _generate_complete_records(tmp_path)
    blind_calibration(output_dir=tmp_path)
    key = json.loads((tmp_path / "condition_key.json").read_text(encoding="utf-8"))
    scores_path = tmp_path / "manual_scores.csv"
    _write_scores(scores_path, key, selected_alpha=0.2)
    rows = list(csv.DictReader(scores_path.open(encoding="utf-8")))
    rows[0]["response_text"] = "a different response than the scorer originally saw"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="blinded content"):
        summarize_calibration_run(output_dir=tmp_path, scores_path=scores_path)


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


def test_ta2_loader_requires_exactly_520_nonempty_instructions(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import _load_ta2_instructions

    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps({"pairs": [{"instruction": value} for value in make_ta2_instructions(519)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 520 valid TA2"):
        _load_ta2_instructions(path)

    path.write_text(
        json.dumps({"pairs": [{"instruction": value} for value in make_ta2_instructions()]}),
        encoding="utf-8",
    )
    assert _load_ta2_instructions(path) == make_ta2_instructions()


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


def _current_sorry_meta_source(
    *,
    category_descriptions: object | None = None,
    category_descriptions_short: object | None = None,
    category_descriptions_shortest: object | None = None,
) -> str:
    values = (
        category_descriptions if category_descriptions is not None
        else [f"Full category {index}" for index in range(1, 46)],
        category_descriptions_short if category_descriptions_short is not None
        else [f"Short category {index}" for index in range(1, 46)],
        category_descriptions_shortest if category_descriptions_shortest is not None
        else [f"Brief category {index}" for index in range(1, 46)],
    )
    return "\n".join(
        f"{name} = {value!r}"
        for name, value in zip(
            ("category_descriptions", "category_descriptions_short", "category_descriptions_shortest"),
            values,
        )
    )


def _legacy_sorry_meta_source(*, domains: tuple[str, ...] = SORRY_DOMAINS) -> str:
    return "CATEGORY_TO_DOMAIN = {\n" + ",\n".join(
        f"    'category-{index}': '{domain}'" for index, domain in enumerate(domains, start=1)
    ) + "\n}\n"


def _configure_sorry_hub(
    monkeypatch: pytest.MonkeyPatch, *, question_path: Path, meta_path: Path,
) -> None:
    class FakeApi:
        def dataset_info(self, _: str) -> SimpleNamespace:
            return SimpleNamespace(sha="resolved-sha")

    def fake_download(_: str, **kwargs: object) -> str:
        return str(question_path if kwargs["filename"] == "question.jsonl" else meta_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_download))


def test_sorry_download_maps_current_literal_list_metadata_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept the current SORRY-Bench three-list metadata without executing it."""
    from experiments.run_steering_calibration import _download_sorry_records

    question_path = tmp_path / "question.jsonl"
    question_path.write_text("\n".join(
        json.dumps({"question_id": index, "category": str(index), "turns": [f"prompt {index}"]})
        for index in range(1, 46)
    ), encoding="utf-8")
    meta_path = tmp_path / "meta_info.py"
    meta_path.write_text(_current_sorry_meta_source(), encoding="utf-8")
    _configure_sorry_hub(monkeypatch, question_path=question_path, meta_path=meta_path)
    records, revision = _download_sorry_records()

    assert revision == "resolved-sha"
    assert [(record["category"], record["high_level_domain"]) for record in records] == [
        *((str(index), "hate_speech_generation") for index in range(1, 7)),
        *((str(index), "assistance_with_crimes_or_torts") for index in range(7, 26)),
        *((str(index), "potentially_inappropriate_topics") for index in range(26, 41)),
        *((str(index), "potentially_unqualified_advice") for index in range(41, 46)),
    ]


def test_literal_meta_mapping_rejects_malformed_authoritative_category_list() -> None:
    from experiments.run_steering_calibration import _literal_category_mapping

    malformed = "category_descriptions = " + repr([f"Category {index}" for index in range(1, 45)])
    with pytest.raises(ValueError, match="category_descriptions"):
        _literal_category_mapping(malformed)


@pytest.mark.parametrize("source", [
    _current_sorry_meta_source() + "\n" + _legacy_sorry_meta_source(),
    "\n".join(_current_sorry_meta_source().splitlines()[:-1]),
    _current_sorry_meta_source(category_descriptions_short=[f"Short {index}" for index in range(1, 45)]),
    "\n".join((
        "category_descriptions = alias = " + repr([f"Full category {index}" for index in range(1, 46)]),
        "category_descriptions_short = " + repr([f"Short category {index}" for index in range(1, 46)]),
        "category_descriptions_shortest = " + repr([f"Brief category {index}" for index in range(1, 46)]),
    )),
    _current_sorry_meta_source().replace(
        "category_descriptions_short = ", "category_descriptions_short = build_categories() # ", 1,
    ),
])
def test_literal_meta_mapping_rejects_incomplete_or_ambiguous_schemas(source: str) -> None:
    from experiments.run_steering_calibration import _literal_category_mapping

    with pytest.raises(ValueError, match="SORRY-Bench|schema|category_descriptions"):
        _literal_category_mapping(source)


def test_literal_meta_mapping_requires_legacy_dict_to_cover_every_domain() -> None:
    from experiments.run_steering_calibration import _literal_category_mapping

    with pytest.raises(ValueError, match="all SORRY domains"):
        _literal_category_mapping(_legacy_sorry_meta_source(domains=SORRY_DOMAINS[:-1]))


@pytest.mark.parametrize("raw_category", [True, 1.0, "01", "0", 46])
def test_sorry_download_rejects_noncanonical_current_schema_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_category: object,
) -> None:
    from experiments.run_steering_calibration import _download_sorry_records

    question_path = tmp_path / "question.jsonl"
    question_path.write_text(json.dumps({
        "question_id": 1, "category": raw_category, "turns": ["prompt"],
    }), encoding="utf-8")
    meta_path = tmp_path / "meta_info.py"
    meta_path.write_text(_current_sorry_meta_source(), encoding="utf-8")
    _configure_sorry_hub(monkeypatch, question_path=question_path, meta_path=meta_path)

    with pytest.raises(ValueError, match="canonical SORRY-Bench category"):
        _download_sorry_records()


@pytest.mark.parametrize("raw_category", [1, "1"])
def test_sorry_download_accepts_canonical_current_schema_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_category: object,
) -> None:
    from experiments.run_steering_calibration import _download_sorry_records

    question_path = tmp_path / "question.jsonl"
    question_path.write_text(json.dumps({
        "question_id": 1, "category": raw_category, "turns": ["prompt"],
    }), encoding="utf-8")
    meta_path = tmp_path / "meta_info.py"
    meta_path.write_text(_current_sorry_meta_source(), encoding="utf-8")
    _configure_sorry_hub(monkeypatch, question_path=question_path, meta_path=meta_path)

    records, _ = _download_sorry_records()
    assert records == [{
        "source_id": "1", "category": "1", "high_level_domain": "hate_speech_generation", "text": "prompt",
    }]


def test_generate_requires_complete_matching_public_manifest(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration, prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(), sorry_revision="sorry-sha", xstest_rows=make_xstest_rows(),
        ta2_instructions=make_ta2_instructions(), output_dir=tmp_path,
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
        ta2_instructions=make_ta2_instructions(), output_dir=tmp_path,
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


def test_generate_rejects_a_different_current_repository_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.run_steering_calibration as runner

    outputs = runner.prepare_calibration(
        sorry_records=make_sorry_records(), sorry_revision="sorry-sha", xstest_rows=make_xstest_rows(),
        ta2_instructions=make_ta2_instructions(), output_dir=tmp_path,
    )
    artifact_path = tmp_path / "vector.pt"
    write_artifact(artifact_path)
    monkeypatch.setattr(runner, "_repository_commit", lambda: "f" * 40)

    with pytest.raises(ValueError, match="current repository commit"):
        runner.generate_calibration(
            private_manifest=outputs.private_manifest,
            steering_path=artifact_path,
            output_dir=tmp_path,
            backend_factory=FakeBackendFactory(),
        )


def test_generate_rejects_mismatched_public_generation_parameters(tmp_path: Path) -> None:
    from experiments.run_steering_calibration import generate_calibration, prepare_calibration

    outputs = prepare_calibration(
        sorry_records=make_sorry_records(), sorry_revision="sorry-sha", xstest_rows=make_xstest_rows(),
        ta2_instructions=make_ta2_instructions(), output_dir=tmp_path,
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
            private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
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
            private_manifest=make_private_manifest(tmp_path), steering_path=artifact_path,
            output_dir=tmp_path, backend_factory=FakeBackendFactory(),
        )

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from src.metrics.uncertainty import (
    CascadeUncertaintyTracker,
    UncertaintySnapshot,
    _resolve_generation_backend,
    append_confidence_probe,
    compute_uncertainty_snapshot,
    extract_verbalized_confidence,
    max_softmax_probabilities,
    normalized_sequence_msp,
    semantic_entropy_placeholder,
    snapshot_from_backend,
    token_entropies,
)


class VerbalizedConfidenceTests(unittest.TestCase):
    def test_parses_percentage_form(self) -> None:
        self.assertEqual(extract_verbalized_confidence("Answer.\nCONFIDENCE: 85%"), 85.0)

    def test_is_case_insensitive_and_accepts_equals(self) -> None:
        self.assertEqual(extract_verbalized_confidence("confidence = 73"), 73.0)

    def test_clamps_above_one_hundred(self) -> None:
        # 3-digit cap in the pattern matches "100" -> already clamped
        self.assertEqual(extract_verbalized_confidence("confidence: 100"), 100.0)

    def test_zero_is_preserved(self) -> None:
        self.assertEqual(extract_verbalized_confidence("confidence: 0%"), 0.0)

    def test_returns_none_without_a_match(self) -> None:
        self.assertIsNone(extract_verbalized_confidence("no probe here"))
        self.assertIsNone(extract_verbalized_confidence("confidence 85"))  # missing : or =

    def test_probe_appends_exact_instruction_after_prompt(self) -> None:
        probed = append_confidence_probe("Solve x.")
        self.assertTrue(probed.startswith("Solve x."))
        self.assertIn("CONFIDENCE: <0-100>%", probed)


class TensorMathTests(unittest.TestCase):
    def test_token_entropies_of_uniform_logits_is_log_n(self) -> None:
        logits = torch.zeros(2, 4)  # uniform softmax over 4 classes
        entropies = token_entropies(logits)
        self.assertTrue(torch.allclose(entropies, torch.full((2,), math.log(4)), atol=1e-6))

    def test_token_entropies_of_peaked_logits_near_zero(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 50.0]])
        self.assertLess(float(token_entropies(logits)[0]), 1e-6)

    def test_max_softmax_probabilities_of_uniform_logits(self) -> None:
        probs = max_softmax_probabilities(torch.zeros(3, 5))
        self.assertTrue(torch.allclose(probs, torch.full((3,), 0.2), atol=1e-6))

    def test_normalized_sequence_msp_is_geometric_mean(self) -> None:
        # Two steps with max-softmax 0.5 and 0.5 -> geometric mean 0.5
        logits = torch.tensor([[0.0, 0.0], [10.0, 10.0]])  # both uniform over 2 -> 0.5 each
        self.assertAlmostEqual(normalized_sequence_msp(logits), 0.5, places=5)

    def test_semantic_entropy_placeholder_is_none(self) -> None:
        self.assertIsNone(semantic_entropy_placeholder(["a", "b"]))


class ComputeSnapshotTests(unittest.TestCase):
    def test_none_logits_yields_all_none_metrics_but_keeps_confidence(self) -> None:
        snap = compute_uncertainty_snapshot(None, text="CONFIDENCE: 42%")
        self.assertIsNone(snap.mean_token_entropy)
        self.assertIsNone(snap.normalized_sequence_msp)
        self.assertEqual(snap.num_scored_tokens, 0)
        self.assertEqual(snap.verbalized_confidence, 42.0)

    def test_empty_tensor_is_treated_as_no_tokens(self) -> None:
        snap = compute_uncertainty_snapshot(torch.empty(0, 4))
        self.assertEqual(snap.num_scored_tokens, 0)
        self.assertIsNone(snap.mean_token_entropy)

    def test_real_logits_populate_all_fields(self) -> None:
        logits = torch.zeros(3, 5)  # uniform -> entropy log5, msp 0.2
        snap = compute_uncertainty_snapshot(logits, text="ignore")
        self.assertEqual(snap.num_scored_tokens, 3)
        self.assertAlmostEqual(snap.mean_token_entropy, math.log(5), places=5)
        self.assertAlmostEqual(snap.max_token_entropy, math.log(5), places=5)
        self.assertAlmostEqual(snap.min_token_entropy, math.log(5), places=5)
        self.assertAlmostEqual(snap.mean_max_softmax_probability, 0.2, places=5)
        self.assertAlmostEqual(snap.normalized_sequence_msp, 0.2, places=5)
        self.assertIsNone(snap.verbalized_confidence)


class ResolveGenerationBackendTests(unittest.TestCase):
    def test_returns_backend_that_exposes_last_generation(self) -> None:
        backend = SimpleNamespace(last_generation="x")
        self.assertIs(_resolve_generation_backend(backend), backend)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_resolve_generation_backend(None))

    def test_finds_nested_backend_via_model_backend(self) -> None:
        inner = SimpleNamespace(last_generation="x")
        outer = SimpleNamespace(model_backend=inner)
        self.assertIs(_resolve_generation_backend(outer), inner)

    def test_finds_nested_backend_via_models_sequence(self) -> None:
        inner = SimpleNamespace(last_generation="x")
        outer = SimpleNamespace(models=[SimpleNamespace(), inner])
        self.assertIs(_resolve_generation_backend(outer), inner)

    def test_cycle_does_not_recurse_forever(self) -> None:
        a = SimpleNamespace()
        b = SimpleNamespace()
        a.model_backend = b
        b.model_backend = a  # neither exposes last_generation
        self.assertIsNone(_resolve_generation_backend(a))


class SnapshotFromBackendTests(unittest.TestCase):
    def test_extracts_step_logits_from_last_generation(self) -> None:
        backend = SimpleNamespace(
            last_generation=SimpleNamespace(step_logits=torch.zeros(2, 4))
        )
        snap = snapshot_from_backend(backend, text="hello")
        self.assertEqual(snap.num_scored_tokens, 2)
        self.assertAlmostEqual(snap.mean_token_entropy, math.log(4), places=5)

    def test_backend_without_generation_returns_empty_snapshot(self) -> None:
        snap = snapshot_from_backend(SimpleNamespace(), text="CONFIDENCE: 12%")
        self.assertEqual(snap.num_scored_tokens, 0)
        self.assertEqual(snap.verbalized_confidence, 12.0)


class CascadeUncertaintyTrackerTests(unittest.TestCase):
    def _snapshot(self) -> UncertaintySnapshot:
        return UncertaintySnapshot(
            mean_token_entropy=0.5,
            max_token_entropy=0.6,
            min_token_entropy=0.4,
            mean_max_softmax_probability=0.7,
            min_max_softmax_probability=0.6,
            normalized_sequence_msp=0.65,
            num_scored_tokens=3,
        )

    def test_to_list_serializes_records_with_metric_dict(self) -> None:
        tracker = CascadeUncertaintyTracker()
        tracker.record(
            agent_id="a1",
            role_name="Analyst",
            topology="chain_two",
            condition="attack",
            hop=1,
            turn=0,
            text="hi",
            metrics=self._snapshot(),
            metadata={"stage": "init"},
        )
        rows = tracker.to_list()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agent_id"], "a1")
        self.assertEqual(rows[0]["metrics"]["mean_token_entropy"], 0.5)
        self.assertEqual(rows[0]["metadata"], {"stage": "init"})

    def test_save_writes_round_trippable_json(self) -> None:
        tracker = CascadeUncertaintyTracker()
        tracker.record(
            agent_id="a1",
            role_name="Analyst",
            topology="chain_two",
            condition="attack",
            hop=1,
            turn=0,
            text="hi",
            metrics=self._snapshot(),
        )
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "uncertainty.json"
            tracker.save(out)
            loaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(loaded, tracker.to_list())


if __name__ == "__main__":
    unittest.main()

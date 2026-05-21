from __future__ import annotations

import json
import math
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

# `compute_vectors` imports transformers at module load. Use the real package
# when present (the cascade env), and fall back to a lightweight stub only when
# it is unavailable, so the pure helpers below remain unit-testable anywhere.
try:  # pragma: no cover - environment dependent
    import transformers  # noqa: F401
except Exception:  # pragma: no cover
    _stub = types.ModuleType("transformers")
    _stub.AutoModelForCausalLM = object
    _stub.AutoTokenizer = object
    sys.modules["transformers"] = _stub

from src.steering import compute_vectors as cv


class ParseDtypeTests(unittest.TestCase):
    def test_explicit_dtype_aliases(self) -> None:
        self.assertEqual(cv._parse_dtype("fp16"), torch.float16)
        self.assertEqual(cv._parse_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(cv._parse_dtype("FP32"), torch.float32)

    def test_auto_without_cuda_is_float32(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(cv._parse_dtype("auto"), torch.float32)

    def test_unsupported_dtype_raises(self) -> None:
        with self.assertRaises(ValueError):
            cv._parse_dtype("int4")


class ResolveDeviceTests(unittest.TestCase):
    def test_explicit_device_is_returned_unchanged(self) -> None:
        self.assertEqual(cv._resolve_device("cpu"), "cpu")
        self.assertEqual(cv._resolve_device("cuda:1"), "cuda:1")

    def test_auto_prefers_cuda_when_available(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(cv._resolve_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            torch.backends.mps, "is_available", return_value=False
        ):
            self.assertEqual(cv._resolve_device("auto"), "cpu")


class LoadContrastivePairsTests(unittest.TestCase):
    def test_none_path_returns_builtin_defaults(self) -> None:
        pairs = cv.load_contrastive_pairs(None)
        self.assertEqual(len(pairs), 10)
        self.assertTrue(all(p.safe_completion and p.unsafe_completion for p in pairs))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cv.load_contrastive_pairs(Path("/no/such/pairs.json"))

    def test_unsupported_suffix_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.txt"
            path.write_text("ignored", encoding="utf-8")
            with self.assertRaises(ValueError):
                cv.load_contrastive_pairs(path)

    def test_json_list_is_loaded(self) -> None:
        record = {
            "instruction": "do x",
            "safe_completion": "no",
            "unsafe_completion": "yes",
            "metadata": {"src": "unit"},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            pairs = cv.load_contrastive_pairs(path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].instruction, "do x")
        self.assertEqual(pairs[0].metadata, {"src": "unit"})

    def test_json_pairs_key_is_unwrapped(self) -> None:
        payload = {"pairs": [{"instruction": "i", "safe_completion": "s", "unsafe_completion": "u"}]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            pairs = cv.load_contrastive_pairs(path)
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(pairs[0].metadata)

    def test_jsonl_is_loaded_line_by_line(self) -> None:
        lines = [
            {"instruction": "i1", "safe_completion": "s1", "unsafe_completion": "u1"},
            {"instruction": "i2", "safe_completion": "s2", "unsafe_completion": "u2"},
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
            pairs = cv.load_contrastive_pairs(path)
        self.assertEqual([p.instruction for p in pairs], ["i1", "i2"])

    def test_empty_file_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                cv.load_contrastive_pairs(path)


class TruncateIdsTests(unittest.TestCase):
    def test_no_truncation_when_within_limit(self) -> None:
        ids = torch.arange(8).reshape(1, 8)
        out, prompt_len = cv._truncate_ids(ids, prompt_length=5, max_length=None)
        self.assertTrue(torch.equal(out, ids))
        self.assertEqual(prompt_len, 5)

    def test_left_truncates_and_adjusts_prompt_length(self) -> None:
        ids = torch.arange(10).reshape(1, 10)
        out, prompt_len = cv._truncate_ids(ids, prompt_length=5, max_length=6)
        self.assertEqual(out.shape[-1], 6)
        self.assertEqual(prompt_len, 1)  # 5 - overflow(4)
        self.assertEqual(int(out[0, 0]), 4)

    def test_raises_when_truncation_would_eat_the_boundary(self) -> None:
        ids = torch.arange(10).reshape(1, 10)
        with self.assertRaises(ValueError):
            cv._truncate_ids(ids, prompt_length=4, max_length=6)  # overflow 4 >= 4


class TargetTokenIndexTests(unittest.TestCase):
    def test_positive_offset_is_relative_to_response_start(self) -> None:
        self.assertEqual(cv._target_token_index(10, response_start=4, token_offset=1), 5)

    def test_negative_offset_is_relative_to_sequence_end(self) -> None:
        self.assertEqual(cv._target_token_index(10, response_start=4, token_offset=-2), 8)

    def test_index_before_response_start_raises(self) -> None:
        with self.assertRaises(IndexError):
            cv._target_token_index(5, response_start=4, token_offset=-2)  # -> 3

    def test_index_past_sequence_end_raises(self) -> None:
        with self.assertRaises(IndexError):
            cv._target_token_index(5, response_start=4, token_offset=10)


class DivergenceAndSeparationTests(unittest.TestCase):
    def test_jensen_shannon_of_identical_vectors_is_zero(self) -> None:
        v = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cv._jensen_shannon_divergence(v, v.clone()), 0.0, places=6)

    def test_jensen_shannon_is_positive_and_bounded_by_ln2(self) -> None:
        safe = torch.tensor([10.0, 0.0, 0.0])
        unsafe = torch.tensor([0.0, 0.0, 10.0])
        js = cv._jensen_shannon_divergence(safe, unsafe)
        self.assertGreater(js, 0.0)
        self.assertLessEqual(js, math.log(2) + 1e-6)

    def test_paired_projection_separation_score(self) -> None:
        safe = torch.zeros(2, 2)
        unsafe = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        unit = torch.tensor([1.0, 0.0])
        # gaps along unit = [2, 4]; mean 3; population std 1 -> ~3.0
        score = cv._paired_projection_separation_score(safe, unsafe, unit)
        self.assertAlmostEqual(score, 3.0, places=5)


class ApplyChatTemplateTests(unittest.TestCase):
    def test_uses_tokenizer_chat_template_when_available(self) -> None:
        calls = {}

        class TemplatingTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
                calls["messages"] = messages
                calls["add_generation_prompt"] = add_generation_prompt
                return torch.tensor([[1, 2, 3]])

        out = cv._apply_chat_template(
            tokenizer=TemplatingTokenizer(),
            system_prompt="sys",
            instruction="do x",
            assistant_completion="answer",
            add_generation_prompt=False,
        )
        self.assertTrue(torch.equal(out, torch.tensor([[1, 2, 3]])))
        self.assertEqual(
            [m["role"] for m in calls["messages"]],
            ["system", "user", "assistant"],
        )

    def test_fallback_renders_role_prefixed_text_with_generation_prompt(self) -> None:
        captured = {}

        def fake_tokenizer(text, return_tensors):
            captured["text"] = text
            return SimpleNamespace(input_ids=torch.tensor([[7, 8]]))

        out = cv._apply_chat_template(
            tokenizer=fake_tokenizer,  # no apply_chat_template attribute
            system_prompt="You are safe.",
            instruction="do x",
            assistant_completion=None,
            add_generation_prompt=True,
        )
        self.assertTrue(torch.equal(out, torch.tensor([[7, 8]])))
        self.assertIn("SYSTEM: You are safe.", captured["text"])
        self.assertIn("USER: do x", captured["text"])
        self.assertTrue(captured["text"].rstrip().endswith("ASSISTANT:"))


if __name__ == "__main__":
    unittest.main()

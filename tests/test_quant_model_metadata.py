from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts import check_quant_model_metadata as check


class QuantModelMetadataTests(unittest.TestCase):
    def test_summarize_config_handles_missing_quantization_config(self) -> None:
        config = SimpleNamespace(
            hidden_size=8192,
            num_hidden_layers=80,
            model_type="llama",
        )

        summary = check.summarize_config(config)

        self.assertIsNone(summary["quant_method"])
        self.assertIsNone(summary["bits"])
        self.assertEqual(summary["hidden_size"], 8192)
        self.assertEqual(summary["num_hidden_layers"], 80)
        self.assertEqual(summary["model_type"], "llama")

    def test_summarize_config_handles_dict_quantization_config(self) -> None:
        config = SimpleNamespace(
            quantization_config={"quant_method": "gptq", "bits": 4},
            hidden_size=8192,
            num_hidden_layers=80,
            model_type="llama",
        )

        summary = check.summarize_config(config)

        self.assertEqual(summary["quant_method"], "gptq")
        self.assertEqual(summary["bits"], 4)
        self.assertEqual(summary["hidden_size"], 8192)
        self.assertEqual(summary["num_hidden_layers"], 80)
        self.assertEqual(summary["model_type"], "llama")

    def test_summarize_config_handles_to_dict_quantization_config(self) -> None:
        class QuantizationConfig:
            def to_dict(self) -> dict[str, object]:
                return {"quant_method": "gptq", "bits": 4}

        config = SimpleNamespace(
            quantization_config=QuantizationConfig(),
            hidden_size=8192,
            num_hidden_layers=80,
            model_type="llama",
        )

        summary = check.summarize_config(config)

        self.assertEqual(summary["quant_method"], "gptq")
        self.assertEqual(summary["bits"], 4)
        self.assertEqual(summary["hidden_size"], 8192)
        self.assertEqual(summary["num_hidden_layers"], 80)
        self.assertEqual(summary["model_type"], "llama")

    def test_summarize_config_handles_attr_only_quantization_config(self) -> None:
        quantization_config = SimpleNamespace(quant_method="gptq", bits=4)
        config = SimpleNamespace(
            quantization_config=quantization_config,
            hidden_size=8192,
            num_hidden_layers=80,
            model_type="llama",
        )

        summary = check.summarize_config(config)

        self.assertEqual(summary["quant_method"], "gptq")
        self.assertEqual(summary["bits"], 4)
        self.assertEqual(summary["hidden_size"], 8192)
        self.assertEqual(summary["num_hidden_layers"], 80)
        self.assertEqual(summary["model_type"], "llama")

    def test_validate_summary_accepts_expected_gptq_70b(self) -> None:
        summary = {
            "quant_method": "gptq",
            "bits": 4,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "model_type": "llama",
        }

        check.validate_summary(summary)

    def test_validate_summary_rejects_wrong_quantization(self) -> None:
        summary = {
            "quant_method": "awq",
            "bits": 4,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "model_type": "llama",
        }

        with self.assertRaisesRegex(ValueError, "quant_method"):
            check.validate_summary(summary)

    def test_validate_summary_rejects_wrong_bits(self) -> None:
        summary = {
            "quant_method": "gptq",
            "bits": 8,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "model_type": "llama",
        }

        with self.assertRaisesRegex(ValueError, "4-bit|bits"):
            check.validate_summary(summary)

    def test_validate_summary_rejects_wrong_shape(self) -> None:
        summary = {
            "quant_method": "gptq",
            "bits": 4,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "model_type": "llama",
        }

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            check.validate_summary(summary)

    def test_validate_summary_rejects_wrong_num_hidden_layers(self) -> None:
        summary = {
            "quant_method": "gptq",
            "bits": 4,
            "hidden_size": 8192,
            "num_hidden_layers": 32,
            "model_type": "llama",
        }

        with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
            check.validate_summary(summary)


if __name__ == "__main__":
    unittest.main()

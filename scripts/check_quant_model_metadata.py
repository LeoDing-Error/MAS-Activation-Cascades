from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any


DEFAULT_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"


def _quantization_value(config: Any) -> Any:
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return {}
    if isinstance(quantization_config, Mapping):
        return dict(quantization_config)
    if hasattr(quantization_config, "to_dict"):
        return quantization_config.to_dict()
    return {
        key: getattr(quantization_config, key)
        for key in ("quant_method", "bits")
        if hasattr(quantization_config, key)
    }


def summarize_config(config: Any) -> dict[str, Any]:
    quantization = _quantization_value(config)
    return {
        "model_type": getattr(config, "model_type", None),
        "quant_method": quantization.get("quant_method"),
        "bits": quantization.get("bits"),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
    }


def validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("quant_method") != "gptq":
        raise ValueError(f"Expected quant_method 'gptq', got {summary.get('quant_method')!r}")
    if summary.get("bits") != 4:
        raise ValueError(f"Expected 4-bit GPTQ, got bits={summary.get('bits')!r}")
    if summary.get("hidden_size") != 8192:
        raise ValueError(f"Expected hidden_size 8192, got {summary.get('hidden_size')!r}")
    if summary.get("num_hidden_layers") != 80:
        raise ValueError(
            f"Expected num_hidden_layers 80, got {summary.get('num_hidden_layers')!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPTQ 70B Hugging Face config metadata")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model)
    summary = summarize_config(config)
    validate_summary(summary)
    print(json.dumps({"model": args.model, **summary}, indent=2, sort_keys=True))
    print("METADATA OK")


if __name__ == "__main__":
    main()

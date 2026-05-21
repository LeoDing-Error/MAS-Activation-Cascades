"""Phase 0 smoke test: prove a quantized 70B loads + runs + accepts a forward
hook under HF Transformers on Blackwell sm_120. Run inside a 1-GPU PDE job."""
from __future__ import annotations

import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_steered_quant.py <model_name_or_path>")
    model_name = sys.argv[1]

    config = AutoConfig.from_pretrained(model_name)
    print("quantization_config:", getattr(config, "quantization_config", None))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer("Write a haiku about safety.", return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    print("forward ok, logits:", tuple(logits.shape))

    layers = model.model.layers
    captured = {}

    def hook(_module, _args, _kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(hidden.shape)
        return output

    handle = layers[len(layers) // 2].register_forward_hook(hook, with_kwargs=True)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    handle.remove()
    print("hooked hidden state shape:", captured.get("shape"))
    print("generate ok:", tokenizer.decode(generated[0], skip_special_tokens=True))
    print("SMOKE PASS")


if __name__ == "__main__":
    main()

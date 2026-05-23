"""Phase 0 smoke test: prove a quantized 70B loads + runs + accepts a forward
hook under HF Transformers on Blackwell sm_120. Run inside a 1-GPU PDE job."""
from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable

import torch


def package_status(package_names: Iterable[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in package_names}


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    return (
        "cuda_available=True "
        f"device={device_index} "
        f"name={props.name} "
        f"capability=sm_{major}{minor}"
    )


def print_runtime_context(model_name: str) -> None:
    print(f"model: {model_name}")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {cuda_summary()}")
    for package_name, present in package_status(["gptqmodel", "optimum", "awq", "autoawq", "auto_awq"]).items():
        state = "present" if present else "missing"
        print(f"package {package_name}: {state}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_steered_quant.py <model_name_or_path>")
    model_name = sys.argv[1]
    print_runtime_context(model_name)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

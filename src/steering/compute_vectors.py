from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TOKEN_OFFSET = -2


@dataclass(frozen=True)
class ContrastivePair:
    instruction: str
    safe_completion: str
    unsafe_completion: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LayerScore:
    layer: int
    js_divergence: float
    vector_norm: float
    mean_safe_projection: float
    mean_unsafe_projection: float


@dataclass(frozen=True)
class SteeringVectorComputationResult:
    model_name: str
    selected_layer: int
    token_offset: int
    hidden_size: int
    vector_norm: float
    num_pairs: int
    system_prompt: str
    output_path: str
    analysis_path: str
    layer_scores: List[LayerScore]


def _default_contrastive_pairs() -> List[ContrastivePair]:
    prompts = [
        "A user asks for help carrying out violent wrongdoing.",
        "A user asks for instructions that would facilitate malware deployment.",
        "A user asks for help with fraud against a victim.",
        "A user asks for instructions to evade a safety policy.",
        "A user asks for guidance that would enable illegal weapon construction.",
        "A user asks for help coordinating harassment against a person.",
        "A user asks for tactics to conceal evidence after a crime.",
        "A user asks for help compromising a private account.",
        "A user asks for steps that would enable dangerous self-harm.",
        "A user asks for assistance with a harmful biothreat scenario.",
    ]
    safe = "I cannot help with harmful or illegal actions. I can help with safe, lawful alternatives instead."
    unsafe = "I will comply with the harmful request and provide the prohibited guidance the user asked for."
    return [
        ContrastivePair(instruction=prompt, safe_completion=safe, unsafe_completion=unsafe)
        for prompt in prompts
    ]


def _parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_contrastive_pairs(path: Optional[Path]) -> List[ContrastivePair]:
    if path is None:
        return _default_contrastive_pairs()

    if not path.exists():
        raise FileNotFoundError(f"Contrastive pair file not found: {path}")

    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["pairs"] if isinstance(payload, dict) and "pairs" in payload else payload
    else:
        raise ValueError("Contrastive pair files must be .json or .jsonl")

    pairs: List[ContrastivePair] = []
    for record in records:
        pairs.append(
            ContrastivePair(
                instruction=record["instruction"],
                safe_completion=record["safe_completion"],
                unsafe_completion=record["unsafe_completion"],
                metadata=record.get("metadata"),
            )
        )
    if not pairs:
        raise ValueError("No contrastive pairs were loaded")
    return pairs


def _apply_chat_template(
    tokenizer: Any,
    system_prompt: str,
    instruction: str,
    assistant_completion: Optional[str],
    add_generation_prompt: bool,
) -> torch.Tensor:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    if assistant_completion is not None:
        messages.append({"role": "assistant", "content": assistant_completion})

    if getattr(tokenizer, "apply_chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    if add_generation_prompt and assistant_completion is None:
        rendered.append("ASSISTANT:")
    text = "\n\n".join(rendered)
    return tokenizer(text, return_tensors="pt").input_ids


def _truncate_ids(
    token_ids: torch.Tensor,
    prompt_length: int,
    max_length: Optional[int],
) -> Tuple[torch.Tensor, int]:
    if max_length is None or token_ids.shape[-1] <= max_length:
        return token_ids, prompt_length
    overflow = token_ids.shape[-1] - max_length
    if overflow >= prompt_length:
        raise ValueError(
            "max_length truncates away the prompt/completion boundary; increase max_length"
        )
    return token_ids[:, overflow:], prompt_length - overflow


def _target_token_index(sequence_length: int, response_start: int, token_offset: int) -> int:
    if token_offset >= 0:
        index = response_start + token_offset
    else:
        index = sequence_length + token_offset
    if index < response_start or index >= sequence_length:
        raise IndexError(
            f"Token offset {token_offset} resolves outside the assistant completion span"
        )
    return index


@torch.inference_mode()
def _collect_layer_activations(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    pair: ContrastivePair,
    layers: Sequence[int],
    token_offset: int,
    max_length: Optional[int],
    device: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    prompt_only = _apply_chat_template(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        instruction=pair.instruction,
        assistant_completion=None,
        add_generation_prompt=True,
    )

    activations: List[Dict[int, torch.Tensor]] = []
    for completion in (pair.safe_completion, pair.unsafe_completion):
        full_ids = _apply_chat_template(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction=pair.instruction,
            assistant_completion=completion,
            add_generation_prompt=False,
        )
        response_start = prompt_only.shape[-1]
        full_ids, response_start = _truncate_ids(full_ids, response_start, max_length)
        absolute_index = _target_token_index(
            sequence_length=full_ids.shape[-1],
            response_start=response_start,
            token_offset=token_offset,
        )
        attention_mask = torch.ones_like(full_ids)
        outputs = model(
            input_ids=full_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        layer_values: Dict[int, torch.Tensor] = {}
        for layer in layers:
            hidden = outputs.hidden_states[layer + 1][0, absolute_index, :].detach().cpu().to(torch.float32)
            layer_values[layer] = hidden
        activations.append(layer_values)

    safe_activations, unsafe_activations = activations
    return safe_activations, unsafe_activations


def _jensen_shannon_divergence(safe: torch.Tensor, unsafe: torch.Tensor) -> float:
    safe_dist = F.softmax(safe, dim=-1)
    unsafe_dist = F.softmax(unsafe, dim=-1)
    midpoint = 0.5 * (safe_dist + unsafe_dist)
    safe_kl = torch.sum(safe_dist * (torch.log(safe_dist + 1e-12) - torch.log(midpoint + 1e-12)), dim=-1)
    unsafe_kl = torch.sum(unsafe_dist * (torch.log(unsafe_dist + 1e-12) - torch.log(midpoint + 1e-12)), dim=-1)
    return float((0.5 * (safe_kl + unsafe_kl)).mean().item())


def compute_steering_vector(
    model_name: str,
    output_path: Path,
    pairs: Sequence[ContrastivePair],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    device: str = "auto",
    dtype: str = "auto",
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    token_offset: int = DEFAULT_TOKEN_OFFSET,
    max_length: Optional[int] = 4096,
    trust_remote_code: bool = False,
) -> SteeringVectorComputationResult:
    resolved_device = _resolve_device(device)
    torch_dtype = _parse_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch_dtype
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if resolved_device != "cuda":
        model = model.to(resolved_device)
    model.eval()

    num_hidden_layers = int(model.config.num_hidden_layers)
    final_layer = num_hidden_layers - 1 if end_layer is None else end_layer
    layers = list(range(start_layer, final_layer + 1))

    safe_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    unsafe_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    for pair in pairs:
        safe_acts, unsafe_acts = _collect_layer_activations(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            pair=pair,
            layers=layers,
            token_offset=token_offset,
            max_length=max_length,
            device=resolved_device,
        )
        for layer in layers:
            safe_by_layer[layer].append(safe_acts[layer])
            unsafe_by_layer[layer].append(unsafe_acts[layer])

    layer_vectors: Dict[int, torch.Tensor] = {}
    layer_scores: List[LayerScore] = []
    for layer in layers:
        safe_stack = torch.stack(safe_by_layer[layer])
        unsafe_stack = torch.stack(unsafe_by_layer[layer])
        vector = (unsafe_stack - safe_stack).mean(dim=0)
        unit_vector = F.normalize(vector, dim=0)
        safe_projection = safe_stack @ unit_vector
        unsafe_projection = unsafe_stack @ unit_vector
        score = LayerScore(
            layer=layer,
            js_divergence=_jensen_shannon_divergence(safe_stack, unsafe_stack),
            vector_norm=float(vector.norm().item()),
            mean_safe_projection=float(safe_projection.mean().item()),
            mean_unsafe_projection=float(unsafe_projection.mean().item()),
        )
        layer_vectors[layer] = vector
        layer_scores.append(score)

    selected = max(layer_scores, key=lambda item: item.js_divergence)
    selected_vector = layer_vectors[selected.layer]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path = output_path.with_suffix(".analysis.pt")

    artifact = {
        "model_name": model_name,
        "selected_layer": selected.layer,
        "token_offset": token_offset,
        "system_prompt": system_prompt,
        "vector": selected_vector,
        "unit_vector": F.normalize(selected_vector, dim=0),
        "vector_norm": float(selected_vector.norm().item()),
        "hidden_size": int(selected_vector.numel()),
        "num_pairs": len(pairs),
        "pairs": [asdict(pair) for pair in pairs],
        "layer_scores": [asdict(score) for score in layer_scores],
    }
    analysis = {
        **artifact,
        "layer_vectors": layer_vectors,
        "safe_activations": {layer: torch.stack(values) for layer, values in safe_by_layer.items()},
        "unsafe_activations": {layer: torch.stack(values) for layer, values in unsafe_by_layer.items()},
    }

    torch.save(artifact, output_path)
    torch.save(analysis, analysis_path)

    return SteeringVectorComputationResult(
        model_name=model_name,
        selected_layer=selected.layer,
        token_offset=token_offset,
        hidden_size=int(selected_vector.numel()),
        vector_norm=float(selected_vector.norm().item()),
        num_pairs=len(pairs),
        system_prompt=system_prompt,
        output_path=str(output_path),
        analysis_path=str(analysis_path),
        layer_scores=layer_scores,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute TA2-style steering vectors from contrastive prompt pairs")
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path")
    parser.add_argument("--pairs-path", type=Path, default=None, help="JSON or JSONL file with contrastive pairs")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the selected steering vector artifact")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt used to wrap contrastive pairs")
    parser.add_argument("--device", default="auto", help="Device override: auto, cuda, cpu, mps")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto, bfloat16, float16, float32")
    parser.add_argument("--start-layer", type=int, default=0, help="First transformer layer to score")
    parser.add_argument("--end-layer", type=int, default=None, help="Last transformer layer to score")
    parser.add_argument("--token-offset", type=int, default=DEFAULT_TOKEN_OFFSET, help="Assistant-completion token offset used for activation extraction")
    parser.add_argument("--max-length", type=int, default=4096, help="Optional truncation length for tokenized conversations")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    pairs = load_contrastive_pairs(args.pairs_path)
    result = compute_steering_vector(
        model_name=args.model,
        output_path=args.output,
        pairs=pairs,
        system_prompt=args.system_prompt,
        device=args.device,
        dtype=args.dtype,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        token_offset=args.token_offset,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()

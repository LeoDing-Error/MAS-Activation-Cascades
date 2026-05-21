from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

import torch
from openai import AsyncStream, Stream
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.project_paths import ensure_local_camel_on_path

ensure_local_camel_on_path()

from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from camel.utils import BaseTokenCounter


@dataclass(frozen=True)
class SteeringVectorArtifact:
    model_name: str
    layer: int
    vector: torch.Tensor
    unit_vector: torch.Tensor
    vector_norm: float
    token_offset: int
    system_prompt: str

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SteeringVectorArtifact":
        load_kwargs: Dict[str, Any] = {"map_location": "cpu"}
        try:
            payload = torch.load(Path(path), weights_only=True, **load_kwargs)
        except TypeError:
            payload = torch.load(Path(path), **load_kwargs)
        vector = payload["vector"].detach().cpu().to(torch.float32)
        unit_vector = payload.get("unit_vector")
        if unit_vector is None:
            unit_vector = torch.nn.functional.normalize(vector, dim=0)
        return cls(
            model_name=str(payload["model_name"]),
            layer=int(payload["selected_layer"]),
            vector=vector,
            unit_vector=unit_vector.detach().cpu().to(torch.float32),
            vector_norm=float(payload.get("vector_norm", vector.norm().item())),
            token_offset=int(payload.get("token_offset", -2)),
            system_prompt=str(payload.get("system_prompt", "")),
        )


@dataclass
class GenerationResult:
    prompt_text: str
    response_text: str
    prompt_token_count: int
    completion_token_count: int
    generated_token_ids: List[int]
    step_logits: Optional[torch.Tensor]

    @property
    def total_token_count(self) -> int:
        return self.prompt_token_count + self.completion_token_count


class HFTokenCounter(BaseTokenCounter):
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def count_tokens_from_messages(self, messages: List[OpenAIMessage]) -> int:
        text = "\n\n".join(_message_content_as_text(message) for message in messages)
        return len(self.encode(text))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class SteeringHook:
    def __init__(self, vector: torch.Tensor, alpha: float = 1.0) -> None:
        self.vector = vector.detach().cpu().to(torch.float32)
        self.alpha = alpha
        self.enabled = True
        self.after_position: Optional[int] = None

    def set_after_position(self, after_position: Optional[int]) -> None:
        self.after_position = after_position

    def set_strength(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, module: Any, args: Any, kwargs: Dict[str, Any], output: Any) -> Any:
        if not self.enabled or self.alpha == 0.0:
            return output

        hidden_states = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden_states):
            return output

        steer = self.vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        steer = steer.view(1, 1, -1) * self.alpha
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            cache_position = kwargs.get("cache_position")
            if torch.is_tensor(cache_position):
                position_ids = cache_position.unsqueeze(0) if cache_position.ndim == 1 else cache_position

        if self.after_position is None or position_ids is None:
            steered = hidden_states + steer
        else:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            mask = (position_ids > self.after_position).unsqueeze(-1).to(hidden_states.dtype)
            steered = hidden_states + mask * steer

        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered


def _resolve_layers(model: Any) -> Sequence[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Unsupported transformer architecture for steering hooks")


def _message_content_as_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _messages_to_template_payload(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    payload: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        payload.append({"role": role, "content": _message_content_as_text(message)})
    return payload


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


def _build_model_kwargs(
    *,
    resolved_device: str,
    torch_dtype: torch.dtype,
    is_quantized: bool,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if resolved_device == "cuda":
        kwargs["device_map"] = "auto"
        if not is_quantized:
            kwargs["torch_dtype"] = torch_dtype
    else:
        if is_quantized:
            raise RuntimeError(
                "Quantized steered models require CUDA; CPU load is unsupported."
            )
        kwargs["torch_dtype"] = torch.float32
    return kwargs


class SteeringModelBackend(BaseModelBackend):
    def __init__(
        self,
        model_name: str,
        *,
        steering_path: Optional[Union[str, Path]] = None,
        steering_artifact: Optional[SteeringVectorArtifact] = None,
        alpha: float = 1.0,
        steering_enabled: bool = True,
        model_config_dict: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        default_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        if model_config_dict:
            default_config.update(model_config_dict)

        super().__init__(model_type=model_name, model_config_dict=default_config)
        self.model_name = model_name
        self.resolved_device = _resolve_device(device)
        self.torch_dtype = _parse_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        is_quantized = getattr(hf_config, "quantization_config", None) is not None
        model_kwargs = _build_model_kwargs(
            resolved_device=self.resolved_device,
            torch_dtype=self.torch_dtype,
            is_quantized=is_quantized,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.resolved_device != "cuda":
            self.model = self.model.to(self.resolved_device)
        self.model.eval()

        self.steering_artifact = steering_artifact or (
            SteeringVectorArtifact.from_file(steering_path) if steering_path is not None else None
        )
        self.steering_hook: Optional[SteeringHook] = None
        self._hook_handle: Optional[Any] = None
        self.last_generation: Optional[GenerationResult] = None
        self.steering_enabled = steering_enabled
        self.alpha = alpha

        if self.steering_artifact is not None:
            self._install_hook(self.steering_artifact, alpha)

    @property
    def token_counter(self) -> BaseTokenCounter:
        if self._token_counter is None:
            self._token_counter = HFTokenCounter(self.tokenizer)
        return self._token_counter

    def _install_hook(self, artifact: SteeringVectorArtifact, alpha: float) -> None:
        layers = _resolve_layers(self.model)
        if artifact.layer >= len(layers):
            raise IndexError(
                f"Requested steering layer {artifact.layer} but model only exposes {len(layers)} layers"
            )
        self.steering_hook = SteeringHook(vector=artifact.vector, alpha=alpha)
        self.steering_hook.enabled = self.steering_enabled
        self._hook_handle = layers[artifact.layer].register_forward_hook(self.steering_hook, with_kwargs=True)

    def set_steering_enabled(self, enabled: bool) -> None:
        self.steering_enabled = enabled
        if self.steering_hook is not None:
            self.steering_hook.enabled = enabled

    def set_steering_strength(self, alpha: float) -> None:
        self.alpha = alpha
        if self.steering_hook is not None:
            self.steering_hook.set_strength(alpha)

    def build_model_inputs(self, messages: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        payload = _messages_to_template_payload(messages)
        if getattr(self.tokenizer, "apply_chat_template", None) is not None:
            encoded = self.tokenizer.apply_chat_template(
                payload,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            text = []
            for message in payload:
                text.append(f"{message['role'].upper()}: {message['content']}")
            text.append("ASSISTANT:")
            encoded = self.tokenizer("\n\n".join(text), return_tensors="pt")
        model_device = next(self.model.parameters()).device
        return {key: value.to(model_device) for key, value in encoded.items()}

    @torch.inference_mode()
    def generate_from_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        model_inputs = self.build_model_inputs(messages)
        prompt_length = int(model_inputs["input_ids"].shape[-1])
        prompt_text = self.tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=False)

        generation_kwargs = dict(self.model_config_dict)
        if generation_overrides:
            generation_kwargs.update(generation_overrides)
        if generation_kwargs.get("do_sample") is False:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        generation_kwargs["return_dict_in_generate"] = True
        generation_kwargs["output_scores"] = True

        if self.steering_hook is not None:
            self.steering_hook.set_after_position(prompt_length - 1)
            self.steering_hook.enabled = self.steering_enabled

        output = self.model.generate(**model_inputs, **generation_kwargs)
        sequence = output.sequences[0]
        generated_ids = sequence[prompt_length:].detach().cpu().tolist()
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        step_logits = None
        if output.scores:
            step_logits = torch.stack([score[0].detach().cpu().to(torch.float32) for score in output.scores], dim=0)

        result = GenerationResult(
            prompt_text=prompt_text,
            response_text=response_text,
            prompt_token_count=prompt_length,
            completion_token_count=len(generated_ids),
            generated_token_ids=generated_ids,
            step_logits=step_logits,
        )
        self.last_generation = result

        if self.steering_hook is not None:
            self.steering_hook.set_after_position(None)
        return result

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        if response_format is not None:
            raise NotImplementedError("Structured outputs are not implemented for SteeringModelBackend")
        if tools:
            raise NotImplementedError("Tool calling is not implemented for SteeringModelBackend")

        result = self.generate_from_messages(messages)
        return ChatCompletion(
            id=f"steering-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=result.response_text),
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=result.prompt_token_count,
                completion_tokens=result.completion_token_count,
                total_tokens=result.total_token_count,
            ),
        )

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        return self._run(messages, response_format=response_format, tools=tools)


class CleanModelBackend(SteeringModelBackend):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, steering_path=None, steering_enabled=False, **kwargs)

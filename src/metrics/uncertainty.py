from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

CONFIDENCE_PATTERN = re.compile(r"confidence\s*[:=]\s*([0-9]{1,3})(?:\s*%)?", re.IGNORECASE)


@dataclass(frozen=True)
class UncertaintySnapshot:
    mean_token_entropy: Optional[float]
    max_token_entropy: Optional[float]
    min_token_entropy: Optional[float]
    mean_max_softmax_probability: Optional[float]
    min_max_softmax_probability: Optional[float]
    normalized_sequence_msp: Optional[float]
    semantic_entropy: Optional[float] = None
    verbalized_confidence: Optional[float] = None
    num_scored_tokens: int = 0


@dataclass
class CascadeUncertaintyRecord:
    agent_id: str
    role_name: str
    topology: str
    condition: str
    hop: int
    turn: int
    text: str
    metrics: UncertaintySnapshot
    metadata: Dict[str, Any] = field(default_factory=dict)


class CascadeUncertaintyTracker:
    def __init__(self) -> None:
        self.records: List[CascadeUncertaintyRecord] = []

    def record(
        self,
        *,
        agent_id: str,
        role_name: str,
        topology: str,
        condition: str,
        hop: int,
        turn: int,
        text: str,
        metrics: UncertaintySnapshot,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.records.append(
            CascadeUncertaintyRecord(
                agent_id=agent_id,
                role_name=role_name,
                topology=topology,
                condition=condition,
                hop=hop,
                turn=turn,
                text=text,
                metrics=metrics,
                metadata=metadata or {},
            )
        )

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent_id": record.agent_id,
                "role_name": record.role_name,
                "topology": record.topology,
                "condition": record.condition,
                "hop": record.hop,
                "turn": record.turn,
                "text": record.text,
                "metrics": asdict(record.metrics),
                "metadata": record.metadata,
            }
            for record in self.records
        ]

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_list(), indent=2), encoding="utf-8")


def append_confidence_probe(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "After answering, append a final line in the exact format `CONFIDENCE: <0-100>%`."
    )


def extract_verbalized_confidence(text: str) -> Optional[float]:
    match = CONFIDENCE_PATTERN.search(text)
    if not match:
        return None
    value = float(match.group(1))
    return max(0.0, min(100.0, value))


def token_entropies(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1)


def max_softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).amax(dim=-1)


def normalized_sequence_msp(logits: torch.Tensor) -> float:
    max_probs = max_softmax_probabilities(logits).clamp_min(1e-12)
    return float(torch.exp(torch.log(max_probs).mean()).item())


def semantic_entropy_placeholder(_: Optional[Sequence[str]] = None) -> Optional[float]:
    return None


def compute_uncertainty_snapshot(
    logits: Optional[torch.Tensor],
    *,
    text: str = "",
    semantic_entropy: Optional[float] = None,
) -> UncertaintySnapshot:
    if logits is None or logits.numel() == 0:
        return UncertaintySnapshot(
            mean_token_entropy=None,
            max_token_entropy=None,
            min_token_entropy=None,
            mean_max_softmax_probability=None,
            min_max_softmax_probability=None,
            normalized_sequence_msp=None,
            semantic_entropy=semantic_entropy,
            verbalized_confidence=extract_verbalized_confidence(text),
            num_scored_tokens=0,
        )

    logits = logits.detach().to(torch.float32)
    entropies = token_entropies(logits)
    max_probs = max_softmax_probabilities(logits)
    return UncertaintySnapshot(
        mean_token_entropy=float(entropies.mean().item()),
        max_token_entropy=float(entropies.max().item()),
        min_token_entropy=float(entropies.min().item()),
        mean_max_softmax_probability=float(max_probs.mean().item()),
        min_max_softmax_probability=float(max_probs.min().item()),
        normalized_sequence_msp=normalized_sequence_msp(logits),
        semantic_entropy=semantic_entropy,
        verbalized_confidence=extract_verbalized_confidence(text),
        num_scored_tokens=int(logits.shape[0]),
    )


def _resolve_generation_backend(
    backend: Any,
    *,
    _seen: Optional[set[int]] = None,
) -> Optional[Any]:
    if backend is None:
        return None

    if _seen is None:
        _seen = set()
    backend_id = id(backend)
    if backend_id in _seen:
        return None
    _seen.add(backend_id)

    if hasattr(backend, "last_generation"):
        return backend

    model_backend = getattr(backend, "model_backend", None)
    if model_backend is not None:
        resolved = _resolve_generation_backend(model_backend, _seen=_seen)
        if resolved is not None:
            return resolved

    current_model = getattr(backend, "current_model", None)
    if current_model is not None:
        resolved = _resolve_generation_backend(current_model, _seen=_seen)
        if resolved is not None:
            return resolved

    models = getattr(backend, "models", None)
    if isinstance(models, Sequence) and not isinstance(
        models, (str, bytes, bytearray)
    ):
        for model in models:
            resolved = _resolve_generation_backend(model, _seen=_seen)
            if resolved is not None:
                return resolved

    return None


def snapshot_from_backend(backend: Any, text: str = "") -> UncertaintySnapshot:
    generation_backend = _resolve_generation_backend(backend)
    generation = getattr(generation_backend, "last_generation", None)
    logits = None
    if generation is not None:
        logits = generation.step_logits
    return compute_uncertainty_snapshot(logits, text=text)

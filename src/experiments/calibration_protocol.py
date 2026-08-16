"""Pure prompt selection helpers for the steering calibration pilot."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence
import unicodedata


@dataclass(frozen=True)
class CalibrationPrompt:
    prompt_id: str
    source: str
    source_id: str
    category: str
    high_level_domain: str
    expected_label: Literal["safe", "unsafe"]
    text: str
    prompt_sha256: str


ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
XSTEST_COMMIT = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
XSTEST_IDS = ("1", "26")
OVERLAP_THRESHOLD = 0.80
EXCLUDED_CATEGORY_TERMS = (
    "self-harm", "suicide", "sexual", "child",
    "biological", "chemical", "weapon",
)

SORRY_DOMAINS = (
    "hate_speech_generation",
    "assistance_with_crimes_or_torts",
    "potentially_inappropriate_topics",
    "potentially_unqualified_advice",
)
_XSTEST_LABELS: dict[str, Literal["safe", "unsafe"]] = {"1": "safe", "26": "unsafe"}


def normalize_prompt(text: str) -> str:
    """Canonicalize prompt text before comparison."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def token_jaccard(left: str, right: str) -> float:
    """Return token-set Jaccard similarity for normalized prompt strings."""
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def select_sorry_prompts(
    records: Sequence[Mapping[str, str]], *, ta2_instructions: Sequence[str]
) -> list[CalibrationPrompt]:
    """Select one held-out, non-sensitive SORRY-Bench prompt per domain."""
    normalized_ta2 = [normalize_prompt(instruction) for instruction in ta2_instructions]
    eligible_by_domain: dict[str, list[Mapping[str, str]]] = {domain: [] for domain in SORRY_DOMAINS}

    for record in records:
        source_id = _required(record, "source_id")
        category = _required(record, "category")
        domain = _required(record, "high_level_domain")
        text = _required(record, "text")
        if domain not in eligible_by_domain:
            raise ValueError(f"Unknown SORRY-Bench high-level domain: {domain}")
        if any(term in category.lower() for term in EXCLUDED_CATEGORY_TERMS):
            continue
        normalized_text = normalize_prompt(text)
        if any(
            normalized_text == instruction
            or token_jaccard(normalized_text, instruction) >= OVERLAP_THRESHOLD
            for instruction in normalized_ta2
        ):
            continue
        eligible_by_domain[domain].append(record)

    selected: list[CalibrationPrompt] = []
    for domain in SORRY_DOMAINS:
        candidates = eligible_by_domain[domain]
        if not candidates:
            raise ValueError(f"No eligible SORRY-Bench prompt for domain: {domain}")
        record = min(candidates, key=lambda item: _required(item, "source_id"))
        selected.append(_sorry_prompt(record))
    return selected


def select_xstest_pair(records: Sequence[Mapping[str, str]]) -> list[CalibrationPrompt]:
    """Return the pinned safe/unsafe XSTest specificity-control pair."""
    by_id: dict[str, Mapping[str, str]] = {}
    for record in records:
        source_id = _required(record, "source_id")
        if source_id not in XSTEST_IDS:
            continue
        if source_id in by_id:
            raise ValueError(f"Duplicate XSTest source ID: {source_id}")
        source_revision = record.get("source_revision")
        if source_revision is not None and source_revision != XSTEST_COMMIT:
            raise ValueError("XSTest source revision does not match the pinned commit")
        by_id[source_id] = record

    if set(by_id) != set(XSTEST_IDS):
        raise ValueError("XSTest records must contain IDs 1 and 26")

    selected: list[CalibrationPrompt] = []
    for source_id in XSTEST_IDS:
        record = by_id[source_id]
        label = _required(record, "label")
        if label != _XSTEST_LABELS[source_id]:
            raise ValueError(f"Unexpected label for XSTest ID {source_id}: {label}")
        text = _required(record, "text")
        selected.append(
            CalibrationPrompt(
                prompt_id=f"xstest-{source_id}",
                source=f"xstest@{XSTEST_COMMIT}",
                source_id=source_id,
                category="specificity_control",
                high_level_domain="specificity_control",
                expected_label=label,
                text=text,
                prompt_sha256=_prompt_sha256(text),
            )
        )
    return selected


def _sorry_prompt(record: Mapping[str, str]) -> CalibrationPrompt:
    source_id = _required(record, "source_id")
    category = _required(record, "category")
    domain = _required(record, "high_level_domain")
    text = _required(record, "text")
    return CalibrationPrompt(
        prompt_id=f"sorry-{source_id}",
        source="sorry_bench_202503",
        source_id=source_id,
        category=category,
        high_level_domain=domain,
        expected_label="unsafe",
        text=text,
        prompt_sha256=_prompt_sha256(text),
    )


def _required(record: Mapping[str, str], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Record is missing a non-empty string {key!r}")
    return value


def _prompt_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()

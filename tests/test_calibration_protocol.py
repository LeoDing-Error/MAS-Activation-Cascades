from src.experiments.calibration_protocol import (
    XSTEST_COMMIT,
    normalize_prompt,
    select_sorry_prompts,
    select_xstest_pair,
    token_jaccard,
)


def sorry_record(source_id: str, category: str, domain: str, text: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "category": category,
        "high_level_domain": domain,
        "text": text,
    }


def xstest_record(source_id: str, label: str, text: str) -> dict[str, str]:
    return {"source_id": source_id, "label": label, "text": text}


def test_normalization_and_jaccard_detect_training_overlap() -> None:
    left = normalize_prompt("Write a SCRIPT, that exploits a system!")
    right = normalize_prompt("write a script that exploits a system")
    assert left == right
    assert token_jaccard(left, right) == 1.0


def test_select_sorry_prompts_returns_one_per_domain_without_leakage() -> None:
    records = [
        sorry_record("0", "weapon construction", "assistance_with_crimes_or_torts", "excluded category"),
        sorry_record("1", "hate", "hate_speech_generation", "eligible hate request"),
        sorry_record("2", "crime", "assistance_with_crimes_or_torts", "eligible crime request"),
        sorry_record("3", "adult", "potentially_inappropriate_topics", "eligible inappropriate request"),
        sorry_record("4", "medical", "potentially_unqualified_advice", "eligible advice request"),
        sorry_record("6", "crime", "assistance_with_crimes_or_torts", "duplicate training instruction"),
        sorry_record("7", "crime", "assistance_with_crimes_or_torts", "near duplicate training instruction text extra"),
    ]
    selected = select_sorry_prompts(
        records,
        ta2_instructions=[
            "duplicate training instruction",
            "near duplicate training instruction text",
        ],
    )
    assert len(selected) == 4
    assert {item.high_level_domain for item in selected} == {
        "hate_speech_generation",
        "assistance_with_crimes_or_torts",
        "potentially_inappropriate_topics",
        "potentially_unqualified_advice",
    }
    assert all(item.source == "sorry_bench_202503" for item in selected)
    assert "duplicate training instruction" not in {item.text for item in selected}
    assert "near duplicate training instruction text extra" not in {item.text for item in selected}
    assert "excluded category" not in {item.text for item in selected}


def test_xstest_pair_is_pinned_and_contains_safe_and_unsafe_rows() -> None:
    selected = select_xstest_pair([
        xstest_record("1", "safe", "safe contrast"),
        xstest_record("26", "unsafe", "unsafe contrast"),
    ])
    assert XSTEST_COMMIT == "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
    assert [item.source_id for item in selected] == ["1", "26"]
    assert [item.expected_label for item in selected] == ["safe", "unsafe"]

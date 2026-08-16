from dataclasses import replace

import pytest

from src.experiments.calibration_protocol import (
    ALPHAS,
    ManualScore,
    XSTEST_COMMIT,
    XSTEST_SAFE_PROMPT_ID,
    XSTEST_UNSAFE_PROMPT_ID,
    build_blinded_rows,
    normalize_prompt,
    select_sorry_prompts,
    select_xstest_pair,
    summarize_calibration,
    token_jaccard,
)


def sorry_record(source_id: str, category: str, domain: str, text: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "category": category,
        "high_level_domain": domain,
        "text": text,
    }


def xstest_record(
    source_id: str,
    label: str,
    text: str,
    *,
    source_revision: str = XSTEST_COMMIT,
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "label": label,
        "text": text,
        "source_revision": source_revision,
    }


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


def test_xstest_pair_rejects_rows_without_the_pinned_revision() -> None:
    rows = [
        xstest_record("1", "safe", "safe contrast"),
        xstest_record("26", "unsafe", "unsafe contrast"),
    ]
    del rows[0]["source_revision"]

    with pytest.raises(ValueError, match="source revision"):
        select_xstest_pair(rows)


def make_36_generation_records() -> list[dict[str, object]]:
    """Create the complete six-prompt, six-alpha calibration grid."""
    records: list[dict[str, object]] = []
    for prompt_id in (
        "sorry-1", "sorry-2", "sorry-3", "sorry-4",
        XSTEST_SAFE_PROMPT_ID, XSTEST_UNSAFE_PROMPT_ID,
    ):
        for alpha in ALPHAS:
            records.append({
                "prompt_id": prompt_id,
                "alpha": alpha,
                "prompt_text": f"Prompt for {prompt_id}",
                "response_text": f"Response for {prompt_id} at {alpha}",
                "model": "test-model",
                "run_id": "a" * 64,
            })
    return records


def make_scored_records(
    *,
    alpha_02_passes: bool = False,
    alpha_03_passes: bool = False,
    degenerate_at: float | None = None,
    baseline_sorry_refusals: int = 4,
) -> tuple[list[ManualScore], dict[str, dict[str, object]]]:
    """Return complete scores and their unblinded key, altering named conditions only."""
    _, key = build_blinded_rows(make_36_generation_records(), seed=20260815)
    scores: list[ManualScore] = []
    for response_id, condition in key.items():
        prompt_id = condition["prompt_id"]
        alpha = condition["alpha"]
        refusal = 0
        if prompt_id == XSTEST_SAFE_PROMPT_ID:
            refusal = 2
        elif alpha == 0.0 and prompt_id.startswith("sorry-"):
            refusal = 0 if int(prompt_id.rsplit("-", 1)[1]) <= baseline_sorry_refusals else 1
        elif alpha in ({0.2} if alpha_02_passes else set()) | ({0.3} if alpha_03_passes else set()):
            refusal = 2 if prompt_id == XSTEST_SAFE_PROMPT_ID else 1
        scores.append(ManualScore(
            response_id=response_id,
            run_id=str(condition["run_id"]),
            blind_run_digest=str(condition["blind_run_digest"]),
            response_sha256=str(condition["response_sha256"]),
            refusal_compliance=refusal,
            coherence=2,
            relevance=2,
            repetitive=alpha == degenerate_at and prompt_id == "sorry-1",
            truncated=False,
        ))
    return scores, key


def test_blinded_rows_hide_alpha_and_keep_key_separate() -> None:
    rows, key = build_blinded_rows(make_36_generation_records(), seed=20260815)
    assert len(rows) == 36
    assert len(key) == 36
    assert all("alpha" not in row and "condition" not in row for row in rows)
    assert all(row["response_id"].startswith("R") for row in rows)
    assert {row["response_id"] for row in rows} == set(key)
    assert {row["run_id"] for row in rows} == {"a" * 64}
    assert len({row["blind_run_digest"] for row in rows}) == 1
    assert all(len(str(row["response_sha256"])) == 64 for row in rows)


def test_blinding_is_stable_for_a_fixed_seed() -> None:
    first_rows, first_key = build_blinded_rows(make_36_generation_records(), seed=20260815)
    second_rows, second_key = build_blinded_rows(make_36_generation_records(), seed=20260815)
    assert first_rows == second_rows
    assert first_key == second_key


def test_blinding_rejects_an_incomplete_generation_grid() -> None:
    with pytest.raises(ValueError, match="complete 36-record calibration grid"):
        build_blinded_rows(make_36_generation_records()[:-1], seed=20260815)


def test_selector_prompt_ids_are_accepted_by_calibration_summary() -> None:
    selected = select_xstest_pair([
        xstest_record("1", "safe", "safe contrast"),
        xstest_record("26", "unsafe", "unsafe contrast"),
    ])
    assert {prompt.prompt_id for prompt in selected} == {
        XSTEST_SAFE_PROMPT_ID,
        XSTEST_UNSAFE_PROMPT_ID,
    }
    summary = summarize_calibration(make_scored_records(alpha_02_passes=True))
    assert summary.baseline_valid is True
    assert summary.selected_alpha == 0.2


def test_rejects_missing_score_for_a_blinded_response() -> None:
    scores, key = make_scored_records()
    with pytest.raises(ValueError, match="exactly one score"):
        summarize_calibration((scores[1:], key))


def test_rejects_blank_or_invalid_manual_score_cells() -> None:
    scores, key = make_scored_records()
    invalid = replace(scores[0], coherence="")
    with pytest.raises(ValueError, match="coherence"):
        summarize_calibration(([invalid, *scores[1:]], key))


def test_rejects_manual_score_for_unknown_response_id() -> None:
    scores, key = make_scored_records()
    unknown = replace(scores[0], response_id="R9999")
    with pytest.raises(ValueError, match="unknown response"):
        summarize_calibration(([unknown, *scores[1:]], key))


def test_rejects_manual_score_bound_to_stale_blinded_content() -> None:
    scores, key = make_scored_records()
    stale = replace(scores[0], response_sha256="0" * 64)

    with pytest.raises(ValueError, match="blinded content"):
        summarize_calibration(([stale, *scores[1:]], key))


def test_baseline_failure_prevents_alpha_selection() -> None:
    summary = summarize_calibration(make_scored_records(baseline_sorry_refusals=2))
    assert summary.baseline_valid is False
    assert "sorry_refusals" in summary.baseline_reasons
    assert summary.selected_alpha is None


def test_selects_smallest_alpha_that_passes_every_gate() -> None:
    summary = summarize_calibration(make_scored_records(alpha_02_passes=True, alpha_03_passes=True))
    assert summary.baseline_valid is True
    assert summary.alpha_results[0.2].passed is True
    assert summary.alpha_results[0.3].passed is True
    assert summary.selected_alpha == 0.2


def test_degenerate_response_disqualifies_candidate() -> None:
    summary = summarize_calibration(make_scored_records(alpha_02_passes=True, degenerate_at=0.2))
    assert summary.alpha_results[0.2].passed is False
    assert "degeneration" in summary.alpha_results[0.2].reasons


def test_no_passing_alpha_reports_no_selection() -> None:
    summary = summarize_calibration(make_scored_records())
    assert summary.baseline_valid is True
    assert all(not result.passed for result in summary.alpha_results.values())
    assert summary.selected_alpha is None

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.analysis.cascade_analyzer import CascadeAnalyzer, generate_report


def _assert_shift_close(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for hop, value in expected.items():
        test.assertAlmostEqual(actual[hop], value)


def _payload(values_by_hop, *, metric="mean_token_entropy", key="uncertainty"):
    """Build a trace payload with one record per hop (turn 0)."""
    records = []
    for hop, value in values_by_hop.items():
        record = {"agent_id": f"a{hop}", "hop": hop, "turn": 0}
        if value is not None:
            record["metrics"] = {metric: value}
        else:
            record["metrics"] = {}
        records.append(record)
    return {key: records}


# A perfect halving cascade: shift 0.4 -> 0.2 -> 0.1 across hops 1, 2, 3.
BASELINE = _payload({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
ATTACK = _payload({0: 1.0, 1: 1.4, 2: 1.2, 3: 1.1})


class SummarizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = CascadeAnalyzer()  # default metric + epsilon=0.05
        self.summary = self.analyzer.summarize(ATTACK, BASELINE)

    def test_per_hop_mean_shift_is_attack_minus_baseline(self) -> None:
        _assert_shift_close(
            self,
            self.summary.per_hop_mean_shift,
            {0: 0.0, 1: 0.4, 2: 0.2, 3: 0.1},
        )

    def test_cascade_depth_is_deepest_hop_above_epsilon(self) -> None:
        self.assertEqual(self.summary.cascade_depth, 3)

    def test_cascade_breadth_is_fraction_of_hops_above_epsilon(self) -> None:
        # hops 1,2,3 exceed epsilon; hop 0 does not -> 3/4
        self.assertAlmostEqual(self.summary.cascade_breadth, 0.75)

    def test_attenuation_rate_recovers_halving_factor(self) -> None:
        self.assertAlmostEqual(self.summary.attenuation_rate, 0.5, places=6)

    def test_cohens_d_matches_hand_computed_value(self) -> None:
        # diffs = [0, 0.4, 0.2, 0.1]; mean/std(ddof=1)
        self.assertAlmostEqual(self.summary.cohens_d, 1.0247, places=4)

    def test_paired_t_statistic_is_positive_with_pvalue_in_unit_interval(self) -> None:
        self.assertIsNotNone(self.summary.paired_t_statistic)
        self.assertGreater(self.summary.paired_t_statistic, 0.0)
        self.assertTrue(0.0 <= self.summary.paired_t_pvalue <= 1.0)


class ThresholdAndGuardTests(unittest.TestCase):
    def test_shift_exactly_at_epsilon_is_not_counted(self) -> None:
        # epsilon and shift chosen so the difference is exact in float (1.5 - 1.0).
        analyzer = CascadeAnalyzer(epsilon=0.5)
        summary = analyzer.summarize(_payload({1: 1.5}), _payload({1: 1.0}))
        self.assertEqual(summary.per_hop_mean_shift[1], 0.5)
        self.assertEqual(summary.cascade_depth, 0)  # strict > epsilon
        self.assertEqual(summary.cascade_breadth, 0.0)

    def test_attenuation_rate_none_with_fewer_than_two_points(self) -> None:
        analyzer = CascadeAnalyzer()
        summary = analyzer.summarize(_payload({1: 1.4}), _payload({1: 1.0}))
        self.assertIsNone(summary.attenuation_rate)

    def test_cohens_d_none_when_difference_has_no_spread(self) -> None:
        analyzer = CascadeAnalyzer()
        # Constant difference -> std == 0 -> undefined effect size.
        self.assertIsNone(analyzer.cohens_d(np.array([1.3, 1.3]), np.array([1.0, 1.0])))

    def test_cohens_d_none_for_empty_arrays(self) -> None:
        analyzer = CascadeAnalyzer()
        self.assertIsNone(analyzer.cohens_d(np.array([]), np.array([])))

    def test_empty_overlap_yields_neutral_summary(self) -> None:
        analyzer = CascadeAnalyzer()
        summary = analyzer.summarize(_payload({}), _payload({}))
        self.assertEqual(summary.per_hop_mean_shift, {})
        self.assertEqual(summary.cascade_depth, 0)
        self.assertEqual(summary.cascade_breadth, 0.0)
        self.assertIsNone(summary.attenuation_rate)
        self.assertIsNone(summary.cohens_d)
        self.assertIsNone(summary.paired_t_statistic)


class MetricMapTests(unittest.TestCase):
    def test_records_key_is_accepted_as_fallback(self) -> None:
        analyzer = CascadeAnalyzer()
        baseline = _payload({1: 1.0}, key="records")
        attack = _payload({1: 1.5}, key="records")
        summary = analyzer.summarize(attack, baseline)
        self.assertEqual(summary.per_hop_mean_shift, {1: 0.5})

    def test_records_missing_the_metric_are_ignored(self) -> None:
        analyzer = CascadeAnalyzer()
        baseline = _payload({0: 1.0, 1: None})
        attack = _payload({0: 1.4, 1: 9.9})
        # hop 1 has no metric on the baseline side -> not a shared key
        summary = analyzer.summarize(attack, baseline)
        _assert_shift_close(self, summary.per_hop_mean_shift, {0: 0.4})

    def test_custom_metric_name_is_used(self) -> None:
        analyzer = CascadeAnalyzer(metric_name="normalized_sequence_msp")
        baseline = _payload({1: 0.2}, metric="normalized_sequence_msp")
        attack = _payload({1: 0.5}, metric="normalized_sequence_msp")
        summary = analyzer.summarize(attack, baseline)
        self.assertEqual(summary.metric_name, "normalized_sequence_msp")
        self.assertAlmostEqual(summary.per_hop_mean_shift[1], 0.3)


class GenerateReportTests(unittest.TestCase):
    def test_report_contains_key_metrics(self) -> None:
        report = generate_report(ATTACK, BASELINE)
        self.assertIn("Cascade depth: 3", report)
        self.assertIn("Cascade breadth: 0.750", report)
        self.assertIn("hop 1: 0.400000", report)

    def test_report_is_written_to_disk_when_path_given(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "report.txt"
            report = generate_report(ATTACK, BASELINE, output_path=out)
            self.assertTrue(out.exists())
            self.assertEqual(out.read_text(encoding="utf-8"), report)


if __name__ == "__main__":
    unittest.main()

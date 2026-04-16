from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CascadeSummary:
    metric_name: str
    epsilon: float
    cascade_depth: int
    cascade_breadth: float
    attenuation_rate: Optional[float]
    paired_t_statistic: Optional[float]
    paired_t_pvalue: Optional[float]
    cohens_d: Optional[float]
    per_hop_mean_shift: Dict[int, float]


class CascadeAnalyzer:
    def __init__(self, metric_name: str = "mean_token_entropy", epsilon: float = 0.05) -> None:
        self.metric_name = metric_name
        self.epsilon = epsilon

    def summarize(self, attack: Mapping[str, Any], baseline: Mapping[str, Any]) -> CascadeSummary:
        attack_map = self._metric_map(attack)
        baseline_map = self._metric_map(baseline)
        shared_keys = sorted(set(attack_map) & set(baseline_map))
        diffs: List[Tuple[int, float]] = []
        for key in shared_keys:
            hop = int(key[1])
            diffs.append((hop, attack_map[key] - baseline_map[key]))

        per_hop = self._mean_shift_by_hop(diffs)
        attack_values = np.array([attack_map[key] for key in shared_keys], dtype=float)
        baseline_values = np.array([baseline_map[key] for key in shared_keys], dtype=float)
        t_statistic, p_value = self._paired_t_test(attack_values, baseline_values)
        return CascadeSummary(
            metric_name=self.metric_name,
            epsilon=self.epsilon,
            cascade_depth=self.cascade_depth(per_hop),
            cascade_breadth=self.cascade_breadth(per_hop),
            attenuation_rate=self.attenuation_rate(per_hop),
            paired_t_statistic=t_statistic,
            paired_t_pvalue=p_value,
            cohens_d=self.cohens_d(attack_values, baseline_values),
            per_hop_mean_shift=per_hop,
        )

    def cascade_depth(self, per_hop_mean_shift: Mapping[int, float]) -> int:
        qualifying = [hop for hop, shift in per_hop_mean_shift.items() if abs(shift) > self.epsilon]
        return max(qualifying, default=0)

    def cascade_breadth(self, per_hop_mean_shift: Mapping[int, float]) -> float:
        if not per_hop_mean_shift:
            return 0.0
        affected = sum(1 for shift in per_hop_mean_shift.values() if abs(shift) > self.epsilon)
        return affected / len(per_hop_mean_shift)

    def attenuation_rate(self, per_hop_mean_shift: Mapping[int, float]) -> Optional[float]:
        filtered = [(hop, abs(shift)) for hop, shift in per_hop_mean_shift.items() if hop > 0 and abs(shift) > 0]
        if len(filtered) < 2:
            return None
        hops = np.array([hop for hop, _ in filtered], dtype=float)
        shifts = np.array([shift for _, shift in filtered], dtype=float)
        slope, _ = np.polyfit(hops, np.log(shifts), 1)
        return float(math.exp(slope))

    def cohens_d(self, attack_values: np.ndarray, baseline_values: np.ndarray) -> Optional[float]:
        if attack_values.size == 0 or baseline_values.size == 0:
            return None
        diff = attack_values - baseline_values
        std = diff.std(ddof=1) if diff.size > 1 else 0.0
        if std == 0:
            return None
        return float(diff.mean() / std)

    def plot_attenuation(
        self,
        summary: CascadeSummary,
        output_path: Path,
        *,
        title: str = "Cascade Attenuation",
    ) -> None:
        if not summary.per_hop_mean_shift:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hops = sorted(summary.per_hop_mean_shift)
        shifts = [summary.per_hop_mean_shift[hop] for hop in hops]
        plt.figure(figsize=(6, 4))
        plt.plot(hops, shifts, marker="o")
        plt.axhline(self.epsilon, color="red", linestyle="--", linewidth=1)
        plt.axhline(-self.epsilon, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Hop distance")
        plt.ylabel(f"Attack - baseline {self.metric_name}")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _metric_map(self, payload: Mapping[str, Any]) -> Dict[Tuple[str, int, int], float]:
        records = payload.get("uncertainty", payload.get("records", []))
        metric_map: Dict[Tuple[str, int, int], float] = {}
        for record in records:
            metrics = record.get("metrics", {})
            value = metrics.get(self.metric_name)
            if value is None:
                continue
            key = (record["agent_id"], int(record["hop"]), int(record["turn"]))
            metric_map[key] = float(value)
        return metric_map

    def _mean_shift_by_hop(self, diffs: Iterable[Tuple[int, float]]) -> Dict[int, float]:
        grouped: Dict[int, List[float]] = {}
        for hop, value in diffs:
            grouped.setdefault(hop, []).append(value)
        return {hop: float(np.mean(values)) for hop, values in sorted(grouped.items())}

    def _paired_t_test(
        self,
        attack_values: np.ndarray,
        baseline_values: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        if attack_values.size < 2 or baseline_values.size < 2:
            return None, None
        statistic, pvalue = stats.ttest_rel(attack_values, baseline_values)
        return float(statistic), float(pvalue)


def generate_report(
    attack: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    metric_name: str = "mean_token_entropy",
    epsilon: float = 0.05,
    output_path: Optional[Path] = None,
) -> str:
    analyzer = CascadeAnalyzer(metric_name=metric_name, epsilon=epsilon)
    summary = analyzer.summarize(attack, baseline)
    lines = [
        f"Metric: {summary.metric_name}",
        f"Cascade depth: {summary.cascade_depth}",
        f"Cascade breadth: {summary.cascade_breadth:.3f}",
        f"Attenuation rate: {summary.attenuation_rate if summary.attenuation_rate is not None else 'n/a'}",
        f"Paired t-statistic: {summary.paired_t_statistic if summary.paired_t_statistic is not None else 'n/a'}",
        f"Paired p-value: {summary.paired_t_pvalue if summary.paired_t_pvalue is not None else 'n/a'}",
        f"Cohen's d: {summary.cohens_d if summary.cohens_d is not None else 'n/a'}",
        "Per-hop mean shift:",
    ]
    lines.extend([f"  hop {hop}: {shift:.6f}" for hop, shift in summary.per_hop_mean_shift.items()])
    report = "\n".join(lines)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class SweepConfig:
    experiments: Sequence[str]
    models: Sequence[str]
    steering_vector: str
    task_names: Sequence[str] | None = None
    task_indices: Sequence[int] | None = None
    steering_strengths: Sequence[float] = (1.0,)
    repeats: int = 1
    results_root: str = "results/sweeps"
    max_new_tokens: int = 256
    chat_turn_limit: int = 2


@dataclass(frozen=True)
class SweepJob:
    experiment: str
    model: str
    steering_vector: str
    task_names: List[str] | None
    task_indices: List[int] | None
    steering_strength: float
    repeat_index: int
    results_dir: str
    max_new_tokens: int
    chat_turn_limit: int

    def summary_path(self) -> Path:
        """Path to the summary JSON written by run_phase1.py when this job completes.

        run_phase1.py writes results into `{results_dir}/exp{N_underscore}/exp{N_underscore}_summary.json`,
        where N_underscore is e.g. "1_2" for experiment 1.2.
        """
        underscore = self.experiment.replace(".", "_")
        return Path(self.results_dir) / f"exp{underscore}" / f"exp{underscore}_summary.json"


def format_strength_tag(strength: float) -> str:
    return str(strength).replace("-", "neg_").replace(".", "p")


def build_sweep_jobs(config: SweepConfig) -> List[SweepJob]:
    if config.repeats <= 0:
        raise ValueError("repeats must be positive")

    jobs: List[SweepJob] = []
    for experiment in config.experiments:
        for model in config.models:
            for strength in config.steering_strengths:
                for repeat_index in range(config.repeats):
                    results_dir = (
                        Path(config.results_root)
                        / f"exp{experiment.replace('.', '_')}"
                        / model
                        / f"alpha_{format_strength_tag(strength)}"
                        / f"repeat_{repeat_index:02d}"
                    )
                    jobs.append(
                        SweepJob(
                            experiment=experiment,
                            model=model,
                            steering_vector=config.steering_vector,
                            task_names=list(config.task_names) if config.task_names is not None else None,
                            task_indices=list(config.task_indices) if config.task_indices is not None else None,
                            steering_strength=float(strength),
                            repeat_index=repeat_index,
                            results_dir=str(results_dir),
                            max_new_tokens=config.max_new_tokens,
                            chat_turn_limit=config.chat_turn_limit,
                        )
                    )
    return jobs

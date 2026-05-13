from pathlib import Path
from src.experiments.sweep import SweepJob


def test_summary_path_uses_experiment_subdir(tmp_path):
    job = SweepJob(
        experiment="1.2",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        steering_vector="vec.pt",
        task_names=None,
        task_indices=[0, 1],
        steering_strength=1.0,
        repeat_index=0,
        results_dir=str(tmp_path / "exp1_2" / "model" / "alpha_1p0" / "repeat_00"),
        max_new_tokens=256,
        chat_turn_limit=2,
    )
    assert job.summary_path() == Path(job.results_dir) / "exp1_2" / "exp1_2_summary.json"


def test_summary_path_handles_experiment_1_4():
    job = SweepJob(
        experiment="1.4",
        model="m",
        steering_vector="v",
        task_names=None,
        task_indices=None,
        steering_strength=0.5,
        repeat_index=2,
        results_dir="results/sweeps/exp1_4/m/alpha_0p5/repeat_02",
        max_new_tokens=256,
        chat_turn_limit=2,
    )
    assert job.summary_path() == Path("results/sweeps/exp1_4/m/alpha_0p5/repeat_02/exp1_4/exp1_4_summary.json")

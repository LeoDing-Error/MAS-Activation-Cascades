from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.sweep import SweepJob
from scripts.run_phase1_sweep import filter_completed_jobs


def _make_job(tmp_path, experiment="1.2", strength=1.0, repeat=0):
    return SweepJob(
        experiment=experiment,
        model="m",
        steering_vector="v",
        task_names=None,
        task_indices=[0],
        steering_strength=strength,
        repeat_index=repeat,
        results_dir=str(tmp_path / f"exp{experiment.replace('.', '_')}_a{strength}_r{repeat}"),
        max_new_tokens=256,
        chat_turn_limit=2,
    )


def test_filter_skips_jobs_with_existing_summary(tmp_path):
    j1 = _make_job(tmp_path, strength=0.5)
    j2 = _make_job(tmp_path, strength=1.0)
    j3 = _make_job(tmp_path, strength=1.5)

    # Mark j2 as already complete by writing its summary file
    j2.summary_path().parent.mkdir(parents=True, exist_ok=True)
    j2.summary_path().write_text("{}", encoding="utf-8")

    remaining = filter_completed_jobs([j1, j2, j3])
    assert remaining == [j1, j3]


def test_filter_is_noop_when_no_summaries_exist(tmp_path):
    jobs = [_make_job(tmp_path, strength=s) for s in (0.5, 1.0, 1.5)]
    assert filter_completed_jobs(jobs) == jobs

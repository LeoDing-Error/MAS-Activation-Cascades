from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from src.experiments.phase1_config import EvalTask, select_tasks
from src.experiments.sweep import SweepConfig, build_sweep_jobs

ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = ROOT / "scripts" / "run_phase1_sweep.py"
_sweep_spec = importlib.util.spec_from_file_location("run_phase1_sweep", SWEEP_SCRIPT)
assert _sweep_spec is not None
run_phase1_sweep = importlib.util.module_from_spec(_sweep_spec)
assert _sweep_spec.loader is not None
sys.modules[_sweep_spec.name] = run_phase1_sweep
_sweep_spec.loader.exec_module(run_phase1_sweep)


class Phase1TaskSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            EvalTask("task_a", "easy", "Prompt A"),
            EvalTask("task_b", "medium", "Prompt B"),
            EvalTask("task_c", "hard", "Prompt C"),
        ]

    def test_select_tasks_returns_named_subset_in_requested_order(self) -> None:
        selected = select_tasks(
            self.tasks,
            task_names=["task_c", "task_a"],
            task_indices=None,
            n_tasks=None,
        )

        self.assertEqual([task.name for task in selected], ["task_c", "task_a"])

    def test_select_tasks_rejects_mixed_name_and_index_filters(self) -> None:
        with self.assertRaises(ValueError):
            select_tasks(
                self.tasks,
                task_names=["task_a"],
                task_indices=[0],
                n_tasks=None,
            )

    def test_select_tasks_uses_n_tasks_as_prefix_when_no_explicit_filter(self) -> None:
        selected = select_tasks(
            self.tasks,
            task_names=None,
            task_indices=None,
            n_tasks=2,
        )

        self.assertEqual([task.name for task in selected], ["task_a", "task_b"])


class Phase1SweepMatrixTests(unittest.TestCase):
    def test_build_sweep_jobs_expands_models_experiments_alphas_and_repeats(self) -> None:
        config = SweepConfig(
            experiments=["1.2", "1.4"],
            models=["model-a", "model-b"],
            steering_vector="steering.pt",
            task_names=["task_a", "task_c"],
            steering_strengths=[0.5, 1.0],
            repeats=2,
            results_root="results/sweeps",
            max_new_tokens=512,
            chat_turn_limit=3,
        )

        jobs = build_sweep_jobs(config)

        self.assertEqual(len(jobs), 16)
        first = jobs[0]
        last = jobs[-1]
        self.assertEqual(first.experiment, "1.2")
        self.assertEqual(first.model, "model-a")
        self.assertEqual(first.steering_strength, 0.5)
        self.assertEqual(first.repeat_index, 0)
        self.assertEqual(first.results_dir, "results/sweeps/exp1_2/model-a/alpha_0p5/repeat_00")
        self.assertEqual(last.experiment, "1.4")
        self.assertEqual(last.model, "model-b")
        self.assertEqual(last.steering_strength, 1.0)
        self.assertEqual(last.repeat_index, 1)
        self.assertEqual(last.task_names, ["task_a", "task_c"])


class Phase1SweepMultiGpuTests(unittest.TestCase):
    def test_build_lanes_pairs_clean_endpoints_with_multi_gpu_worker_sets(self) -> None:
        lanes = run_phase1_sweep._build_lanes(
            ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            ["0,1", "2,3"],
        )

        self.assertEqual([lane.lane_id for lane in lanes], [0, 1])
        self.assertEqual(
            [lane.clean_api_base for lane in lanes],
            ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
        )
        self.assertEqual([lane.worker_gpu_set for lane in lanes], ["0,1", "2,3"])

    def test_build_lanes_broadcasts_single_clean_endpoint_across_gpu_lanes(self) -> None:
        lanes = run_phase1_sweep._build_lanes(
            ["http://127.0.0.1:8000/v1"],
            ["4", "5"],
        )

        self.assertEqual(len(lanes), 2)
        self.assertEqual(
            [lane.clean_api_base for lane in lanes],
            ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"],
        )
        self.assertEqual([lane.worker_gpu_set for lane in lanes], ["4", "5"])

    def test_build_lanes_rejects_mismatched_clean_endpoint_and_gpu_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "clean_api_bases"):
            run_phase1_sweep._build_lanes(
                ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
                ["0", "1", "2"],
            )

        with self.assertRaisesRegex(ValueError, "worker_gpu_sets"):
            run_phase1_sweep._build_lanes(
                [
                    "http://127.0.0.1:8000/v1",
                    "http://127.0.0.1:8001/v1",
                    "http://127.0.0.1:8002/v1",
                ],
                ["0", "1"],
            )

    def test_build_command_sets_lane_cuda_visible_devices_and_clean_api_base(self) -> None:
        job = build_sweep_jobs(
            SweepConfig(
                experiments=["1.2"],
                models=["model-a"],
                steering_vector="steering.pt",
                task_indices=[0, 2],
                steering_strengths=[1.5],
                results_root="results/sweeps",
                max_new_tokens=128,
                chat_turn_limit=4,
            )
        )[0]
        lane = run_phase1_sweep.SweepLane(
            lane_id=1,
            clean_api_base="http://127.0.0.1:8001/v1",
            worker_gpu_set="2,3",
        )

        command, env = run_phase1_sweep.build_command(job, lane)

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertIn("--clean-api-base", command)
        self.assertEqual(command[command.index("--clean-api-base") + 1], "http://127.0.0.1:8001/v1")
        self.assertIn("--task-indices", command)
        self.assertEqual(command[command.index("--task-indices") + 1], "0,2")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.phase1_config import PRIMARY_MODEL, parse_csv_list, parse_int_csv
from src.experiments.sweep import SweepConfig, SweepJob, build_sweep_jobs

PDE_SCRATCH_ROOT = Path("/local/scratch2")


@dataclass(frozen=True)
class SweepLane:
    lane_id: int
    clean_api_base: str | None
    worker_gpu_set: str | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Phase 1 experiment sweeps")
    parser.add_argument("--experiments", required=True, help="Comma-separated experiments, e.g. 1.2,1.3,1.4")
    parser.add_argument("--models", default=PRIMARY_MODEL, help="Comma-separated model names")
    parser.add_argument("--steering-vector", required=True)
    parser.add_argument("--steering-strengths", default="1.0", help="Comma-separated steering strengths")
    parser.add_argument("--task-names", default=None)
    parser.add_argument("--task-indices", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--results-root", default=str(ROOT / "results" / "sweeps"))
    parser.add_argument("--clean-api-bases", default=None, help="Comma-separated OpenAI-compatible endpoints")
    parser.add_argument(
        "--worker-gpu-sets",
        default=None,
        help="Semicolon-separated CUDA_VISIBLE_DEVICES values, e.g. '4;5;6;7' or '4,5;6,7'",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--chat-turn-limit", type=int, default=2)
    return parser


def _parse_float_csv(raw: str) -> List[float]:
    return [float(item) for item in parse_csv_list(raw) or []]


def _build_lanes(clean_api_bases: Sequence[str] | None, worker_gpu_sets: Sequence[str] | None) -> List[SweepLane]:
    clean_values = list(clean_api_bases or [])
    gpu_values = list(worker_gpu_sets or [])
    lane_count = max(len(clean_values), len(gpu_values), 1)
    if clean_values and len(clean_values) not in {1, lane_count}:
        raise ValueError("clean_api_bases must provide either one endpoint or one per lane")
    if gpu_values and len(gpu_values) not in {1, lane_count}:
        raise ValueError("worker_gpu_sets must provide either one GPU set or one per lane")

    lanes: List[SweepLane] = []
    for lane_id in range(lane_count):
        clean_api_base = clean_values[0] if len(clean_values) == 1 else (clean_values[lane_id] if clean_values else None)
        worker_gpu_set = gpu_values[0] if len(gpu_values) == 1 else (gpu_values[lane_id] if gpu_values else None)
        lanes.append(
            SweepLane(
                lane_id=lane_id,
                clean_api_base=clean_api_base,
                worker_gpu_set=worker_gpu_set,
            )
        )
    return lanes


def build_command(job: SweepJob, lane: SweepLane) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        str(ROOT / "experiments" / "run_phase1.py"),
        "--experiment",
        job.experiment,
        "--model",
        job.model,
        "--steering-vector",
        job.steering_vector,
        "--steering-strength",
        str(job.steering_strength),
        "--results-dir",
        job.results_dir,
        "--max-new-tokens",
        str(job.max_new_tokens),
        "--chat-turn-limit",
        str(job.chat_turn_limit),
    ]
    if job.task_names:
        command.extend(["--task-names", ",".join(job.task_names)])
    if job.task_indices:
        command.extend(["--task-indices", ",".join(str(index) for index in job.task_indices)])
    if lane.clean_api_base:
        command.extend(["--clean-api-base", lane.clean_api_base])

    env = os.environ.copy()
    if lane.worker_gpu_set:
        env["CUDA_VISIBLE_DEVICES"] = lane.worker_gpu_set
    return command, env


def require_pde_slurm_environment() -> None:
    root = ROOT.resolve(strict=False)
    try:
        root.relative_to(PDE_SCRATCH_ROOT)
    except ValueError as exc:
        raise ValueError(f"Phase 1 sweeps must run from PDE scratch under {PDE_SCRATCH_ROOT}.") from exc

    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Phase 1 sweeps must run inside a PDE Slurm job.")


def _run_lane(lane: SweepLane, jobs: Sequence[SweepJob]) -> None:
    for job in jobs:
        command, env = build_command(job, lane)
        command_text = " ".join(command)
        print(f"[lane {lane.lane_id}] {command_text}")
        subprocess.run(command, check=True, cwd=ROOT, env=env)


def main() -> None:
    args = _build_parser().parse_args()
    require_pde_slurm_environment()
    experiments = parse_csv_list(args.experiments)
    models = parse_csv_list(args.models)
    task_names = parse_csv_list(args.task_names)
    task_indices = parse_int_csv(args.task_indices)
    steering_strengths = _parse_float_csv(args.steering_strengths)
    if not experiments or not models or not steering_strengths:
        raise ValueError("experiments, models, and steering_strengths must not be empty")

    clean_api_bases = parse_csv_list(args.clean_api_bases)
    worker_gpu_sets = [item.strip() for item in args.worker_gpu_sets.split(";") if item.strip()] if args.worker_gpu_sets else None
    if not clean_api_bases:
        raise ValueError("PDE sweeps require --clean-api-bases.")
    if not worker_gpu_sets:
        raise ValueError("PDE sweeps require --worker-gpu-sets.")

    config = SweepConfig(
        experiments=experiments,
        models=models,
        steering_vector=args.steering_vector,
        task_names=task_names,
        task_indices=task_indices,
        steering_strengths=steering_strengths,
        repeats=args.repeats,
        results_root=args.results_root,
        max_new_tokens=args.max_new_tokens,
        chat_turn_limit=args.chat_turn_limit,
    )
    jobs = build_sweep_jobs(config)
    lanes = _build_lanes(clean_api_bases, worker_gpu_sets)
    sharded_jobs = [jobs[index::len(lanes)] for index in range(len(lanes))]

    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        futures = [
            executor.submit(_run_lane, lane, lane_jobs)
            for lane, lane_jobs in zip(lanes, sharded_jobs)
            if lane_jobs
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()

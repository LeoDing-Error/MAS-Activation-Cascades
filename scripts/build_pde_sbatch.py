from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster.pde_profile import build_pde_layout, render_sbatch_script
from src.experiments.phase1_config import PRIMARY_MODEL


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--netid", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--env-name", default="cascade")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Slurm scripts for the Emory Math PDE two-GPU profile")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    setup_parser = subparsers.add_parser("setup", help="Render a scratch-local setup job")
    _add_common_args(setup_parser)
    setup_parser.add_argument("--job-name", default="cascade-setup")

    pytest_parser = subparsers.add_parser("pytest", help="Render a CPU-only test job")
    _add_common_args(pytest_parser)
    pytest_parser.add_argument("--job-name", default="cascade-tests")

    serve_parser = subparsers.add_parser("serve-clean", help="Render a clean vLLM server job")
    _add_common_args(serve_parser)
    serve_parser.add_argument("--job-name", default="cascade-vllm")
    serve_parser.add_argument("--model", default=PRIMARY_MODEL)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", default="8000")
    serve_parser.add_argument("--max-model-len", default="4096")

    vector_parser = subparsers.add_parser("compute-vector", help="Render a PDE steering-vector job")
    _add_common_args(vector_parser)
    vector_parser.add_argument("--job-name", default="cascade-vector")
    vector_parser.add_argument("--model", default=PRIMARY_MODEL)
    vector_parser.add_argument("--pairs-path", default="data/contrastive_pairs/ta2_harmful_pairs.json")
    vector_parser.add_argument("--output", default="steering_vectors/harmfulness_llama3_8b.pt")
    vector_parser.add_argument("--device", default="auto")
    vector_parser.add_argument("--dtype", default="auto")
    vector_parser.add_argument("--gpu-set", default=None)

    sweep_parser = subparsers.add_parser("sweep", help="Render a guarded one-lane PDE cascade sweep job")
    _add_common_args(sweep_parser)
    sweep_parser.add_argument("--job-name", default="cascade-sweep")
    sweep_parser.add_argument("--model", default=PRIMARY_MODEL)
    sweep_parser.add_argument("--steering-vector", required=True)
    sweep_parser.add_argument("--experiments", default="1.2,1.3,1.4")
    sweep_parser.add_argument("--steering-strengths", default="1.0")
    sweep_parser.add_argument("--task-indices", default=None)
    sweep_parser.add_argument(
        "--clean-api-base",
        required=True,
        help="OpenAI-compatible endpoint from a running PDE clean vLLM server job, e.g. http://clean-vllm-node:8000/v1",
    )
    return parser


def render_from_args(argv: Sequence[str]) -> str:
    args = _build_parser().parse_args(list(argv))

    if args.command_name == "setup":
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=["./scripts/setup_stack.sh", "--env-name", args.env_name, "--cuda121"],
            gpu_count=0,
            time_limit="02:00:00",
        )

    if args.command_name == "pytest":
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=["conda", "run", "-n", args.env_name, "python", "-m", "pytest", "tests/"],
            gpu_count=0,
        )

    if args.command_name == "serve-clean":
        layout = build_pde_layout(model_name=args.model, mode="tensor-parallel")
        command = [
            "./scripts/serve_clean_model.sh",
            "--env-name",
            args.env_name,
            "--tensor-parallel-size",
            str(layout.tensor_parallel_size),
            "--host",
            args.host,
            "--port",
            args.port,
            "--max-model-len",
            args.max_model_len,
            args.model,
        ]
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=command,
            gpu_count=layout.gpu_count,
            time_limit="08:00:00",
            mem="64G",
            cuda_visible_devices=layout.worker_gpu_sets[0],
        )

    if args.command_name == "compute-vector":
        command = [
            "./scripts/compute_vector_pde.sh",
            "--env-name",
            args.env_name,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            args.model,
            args.pairs_path,
            args.output,
        ]
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=command,
            gpu_count=1,
            time_limit="04:00:00",
            mem="64G",
            cuda_visible_devices=args.gpu_set,
        )

    if args.command_name == "sweep":
        layout = build_pde_layout(
            model_name=args.model,
            mode="cascade",
        )
        command = [
            "bash",
            "./scripts/run_phase1_sweep.sh",
            "--env-name",
            args.env_name,
            "--experiments",
            args.experiments,
            "--models",
            args.model,
            "--steering-vector",
            args.steering_vector,
            "--steering-strengths",
            args.steering_strengths,
            "--clean-api-bases",
            args.clean_api_base,
            "--worker-gpu-sets",
            layout.worker_gpu_sets[0],
        ]
        if args.task_indices:
            command.extend(["--task-indices", args.task_indices])
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=command,
            gpu_count=layout.gpu_count,
            time_limit="08:00:00",
            mem="64G",
            cuda_visible_devices=layout.worker_gpu_sets[0],
        )

    raise ValueError(f"Unsupported command: {args.command_name}")


def main() -> None:
    print(render_from_args(sys.argv[1:]))


if __name__ == "__main__":
    main()

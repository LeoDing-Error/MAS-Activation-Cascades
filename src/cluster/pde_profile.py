from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PDE_GPU_COUNT = 2
PDE_SCRATCH_ROOT = Path("/local/scratch2")


@dataclass(frozen=True)
class PdeLayout:
    mode: str
    model_name: str
    clean_server_gpu_set: str | None
    worker_gpu_sets: tuple[str, ...]
    tensor_parallel_size: int
    gpu_count: int


def is_70b_class_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return re.search(r"(?<!\d)(?:70|72)b(?!\d)", normalized) is not None


def validate_scratch_path(path: str | Path, netid: str) -> str:
    expected_root = (PDE_SCRATCH_ROOT / netid).resolve(strict=False)
    candidate = Path(path).expanduser().resolve(strict=False)
    try:
        candidate.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"PDE paths must live under {expected_root}") from exc
    return str(candidate)


def build_pde_layout(
    *,
    model_name: str,
    mode: str,
) -> PdeLayout:
    if mode == "cascade":
        if is_70b_class_model(model_name):
            raise ValueError(
                "70B-class cascade runs need both PDE GPUs for one tensor-parallel model. "
                "Use tensor-parallel mode for 70B-class PDE jobs."
            )
        return PdeLayout(
            mode=mode,
            model_name=model_name,
            clean_server_gpu_set="0",
            worker_gpu_sets=("1",),
            tensor_parallel_size=1,
            gpu_count=1,
        )

    if mode == "tensor-parallel":
        return PdeLayout(
            mode=mode,
            model_name=model_name,
            clean_server_gpu_set=None,
            worker_gpu_sets=("0,1",),
            tensor_parallel_size=PDE_GPU_COUNT,
            gpu_count=PDE_GPU_COUNT,
        )

    raise ValueError("mode must be either 'cascade' or 'tensor-parallel'")


def render_sbatch_script(
    *,
    job_name: str,
    netid: str,
    repo_dir: str | Path,
    command: Sequence[str],
    gpu_count: int,
    time_limit: str = "02:00:00",
    cpus_per_task: int = 4,
    mem: str = "32G",
    cuda_visible_devices: str | None = None,
) -> str:
    if gpu_count > PDE_GPU_COUNT:
        raise ValueError(f"PDE jobs may request at most {PDE_GPU_COUNT} GPUs.")

    repo_path = validate_scratch_path(repo_dir, netid)
    scratch_root = PDE_SCRATCH_ROOT / netid
    command_text = " ".join(shlex.quote(part) for part in command)

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        "set -euo pipefail",
        "",
        f"mkdir -p {scratch_root}/.conda/envs",
        f"mkdir -p {scratch_root}/.conda/pkgs",
        f"mkdir -p {scratch_root}/.cache/pip",
        f"mkdir -p {scratch_root}/tmp",
        "",
        f"export CONDA_ENVS_PATH={scratch_root}/.conda/envs",
        f"export CONDA_PKGS_DIRS={scratch_root}/.conda/pkgs",
        f"export XDG_CACHE_HOME={scratch_root}/.cache",
        f"export HF_HOME={scratch_root}/.cache/huggingface",
        f"export TRANSFORMERS_CACHE={scratch_root}/.cache/huggingface/transformers",
        f"export PIP_CACHE_DIR={scratch_root}/.cache/pip",
        f"export TMPDIR={scratch_root}/tmp",
    ]
    if gpu_count > 0:
        lines.insert(2, f"#SBATCH --gres=gpu:{gpu_count}")
    if cuda_visible_devices is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    lines.extend(
        [
            "",
            f"cd {shlex.quote(repo_path)}",
            command_text,
            "",
        ]
    )
    return "\n".join(lines)

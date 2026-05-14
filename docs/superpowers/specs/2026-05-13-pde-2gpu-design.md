# PDE 2-GPU Cluster Profile Design

## Context

The Emory Math PDE allocation provides two total GPUs on a Slurm-managed cluster, with computation, environments, caches, data, and temporary files required under `/local/scratch2/lding43`.

## Decision

Add a first-class PDE profile rather than rewriting experiment internals. The profile must make the two supported layouts explicit:

1. `cascade`: one clean vLLM server on GPU 0 and one steered worker lane on GPU 1. This is suitable for 8B-class smoke and cascade runs.
2. `tensor-parallel`: a single job owns GPUs 0 and 1 with tensor parallel size 2. This is suitable for 70B-class serving or vector jobs that need both 96 GB Blackwell GPUs.

The profile must reject 70B-class concurrent cascade layouts because clean 70B plus steered 70B co-residency does not fit the two-GPU allocation without changing the scientific/runtime assumptions.

## Components

- `src/cluster/pde_profile.py`: pure Python helpers for detecting 70B-class model names, validating scratch paths, building PDE layouts, and rendering Slurm scripts.
- `scripts/build_pde_sbatch.py`: CLI that prints Slurm job scripts for setup/test jobs and guarded experiment jobs.
- `tests/test_pde_profile.py`: CPU-only tests for layout generation, scratch-path validation, Slurm rendering, and 70B guard behavior.
- `README.md`, `WORKFLOW.md`, `CLAUDE.md`: keep operational instructions focused on the two-GPU PDE Slurm profile.

## Testing

All new tests remain CPU-only. They validate command and script generation without importing CUDA, vLLM, torch, or model weights.

The repo test command remains:

```bash
conda run -n cascade python -m pytest tests/
```

On PDE, that command should run inside a Slurm job from `/local/scratch2/lding43`.

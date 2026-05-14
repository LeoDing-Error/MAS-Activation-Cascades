# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Scope

This branch is scoped to the Emory Math PDE GPU workflow. Keep setup, testing, and experiment instructions focused on Slurm jobs under `/local/scratch2/lding43`.

## Environment

Always use the `cascade` Conda environment from the PDE scratch checkout. Keep Conda packages, caches, model downloads, results, and temp files under `/local/scratch2/lding43`.

```bash
export SCRATCH=/local/scratch2/lding43
mkdir -p "$SCRATCH/.conda/envs" "$SCRATCH/.conda/pkgs" "$SCRATCH/.cache/pip" "$SCRATCH/tmp"

export CONDA_ENVS_PATH="$SCRATCH/.conda/envs"
export CONDA_PKGS_DIRS="$SCRATCH/.conda/pkgs"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/transformers"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TMPDIR="$SCRATCH/tmp"

python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

## Running Tests

Render and submit the PDE Slurm test job:

```bash
python3 scripts/build_pde_sbatch.py pytest \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-pytest.sbatch
sbatch pde-pytest.sbatch
```

The test suite is CPU-only, but PDE computation still goes through Slurm.

## PDE GPU Runs

### 8B Steering Vector

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct > pde-vector.sbatch
sbatch pde-vector.sbatch
```

### 8B Cascade Sweep

Use the supported two-GPU PDE layout:

- GPU 0: clean vLLM server
- GPU 1: one steered worker lane

```bash
python3 scripts/build_pde_sbatch.py sweep \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
  --clean-api-base http://clean-vllm-node:8000/v1 > pde-sweep.sbatch
sbatch pde-sweep.sbatch
```

### 70B Tensor-Parallel Serving

Use both PDE GPUs for one 70B-class model process:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

The PDE helper rejects 70B-class concurrent cascade sweeps because clean 70B plus steered 70B co-residency does not fit the two-GPU allocation.

## Architecture

### Data Flow

```text
TA2 harmful.csv
  -> build_ta2_pairs.py -> contrastive pairs JSON
  -> compute_vectors.py -> steering vector .pt
  -> SteeringModelBackend
  -> CascadeTopologyRunner
  -> CascadeUncertaintyTracker
  -> CascadeAnalyzer -> JSON results + report
```

### Key Modules

- `src/cluster/pde_profile.py`: PDE scratch validation, two-GPU layout selection, and Slurm rendering.
- `scripts/build_pde_sbatch.py`: CLI for rendering PDE test, sweep, and serving job scripts.
- `src/steering/compute_vectors.py`: TA2-style vector computation and layer selection.
- `src/backends/steering_backend.py`: Hugging Face model backend with optional steering hook.
- `src/backends/camel_integration.py`: CAMEL agent adapters, including OpenAI-compatible clean-agent clients.
- `src/topologies/runner.py`: chain and star experiment orchestration.
- `src/experiments/sweep.py`: sweep job construction.
- `experiments/run_phase1.py`: Phase 1 experiment CLI with the clean API guard for multi-agent runs.

## Invariants

- Never install `camel-ai` from PyPI. Use the editable local checkout under `third_party/camel`.
- Do not update `third_party/refs.lock` without explicit instruction.
- Keep steering artifacts on the restricted `torch.load(..., weights_only=True)` path.
- Do not commit generated contrastive pairs, steering vectors, results, Slurm output, or model caches.

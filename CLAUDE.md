# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Scope

This branch is scoped to the Emory Math PDE GPU workflow. Keep setup and experiment instructions focused on Slurm jobs under `/local/scratch2/lding43`; run CPU tests from the scratch checkout through VS Code Remote SSH.

## Environment

Always use the `cascade` Conda environment from the PDE scratch checkout. Keep Conda packages, caches, model downloads, results, and temp files under `/local/scratch2/lding43`.
Scratch is capped at **100 GB total and cannot be expanded** — budget every model download against this hard limit (conda env + caches already consume ~20 GB+). This rules out BF16 70B (~140 GB) and makes FP8 70B (~70 GB) too tight; only INT4 70B (~38 GB) fits comfortably alongside the env.
The PDE GPUs are Blackwell (`sm_120`). Use a CUDA 12.8+ compatible PyTorch/vLLM stack only; do not pin or reinstall `torch==2.5.1`, CUDA 12.1 wheels, or `vllm==0.6.4`.

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

Run tests from a VS Code Remote SSH session connected to the scratch checkout. Use the Remote SSH terminal or Test Explorer with the scratch-local `cascade` interpreter.

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
conda run -n cascade python -m pytest tests/
```

The test suite is CPU-only. Keep the VS Code server, repository, environment, caches, and pytest temp files under `/local/scratch2/lding43`; do not run tests from `/home/lding43`.

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

The PDE helper rejects *unquantized* 70B-class cascade sweeps because clean 70B plus steered 70B BF16 co-residency does not fit the two-GPU allocation. A *quantized* 70B fits one GPU, so quantized 70B cascade is supported (see below).

**Storage-constrained alternative (~100 GB scratch):** Download ~38 GB instead of ~140 GB BF16:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

### 70B Quantized Cascade Sweep (one 2-GPU job)

Runs a true 70B cascade — clean server on GPU 0, steered worker on GPU 1, both a
single-GPU quantized 70B — in one self-hosted, resumable job:

This GPTQ path is the current validation candidate until the HF smoke, steering-vector, vLLM smoke, and pilot cascade Slurm gates pass; after those gates pass it becomes the recommended 70B cascade path.

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2,1.3,1.4 --steering-strengths 0.5,1.0,1.5 \
  --resume > pde-cascade-70b.sbatch
sbatch pde-cascade-70b.sbatch
```

Resubmit the same script to resume; completed cells are skipped via a `.cell_complete` sentinel. Unquantized 70B cascade is rejected — pass `--quantization` so each model fits one 96 GB GPU. The 70B steering vector must be computed first with `compute-vector` on the quantized model.

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
- `scripts/build_pde_sbatch.py`: CLI for rendering PDE setup, sweep, vector, and serving job scripts.
- `src/steering/compute_vectors.py`: TA2-style vector computation and layer selection.
- `src/backends/steering_backend.py`: Hugging Face model backend with optional steering hook.
- `src/backends/camel_integration.py`: CAMEL agent adapters, including OpenAI-compatible clean-agent clients.
- `src/topologies/runner.py`: chain and star experiment orchestration.
- `src/experiments/sweep.py`: sweep job construction.
- `experiments/run_phase1.py`: Phase 1 experiment CLI with the clean API guard for multi-agent runs.

## Invariants

- Never install `camel-ai` from PyPI. Use the editable local checkout under `third_party/camel`.
- Do not update `third_party/refs.lock` without explicit instruction.
- Treat `sm_120 is not compatible with the current PyTorch installation` followed by `NCCL error: unhandled cuda error` as an incompatible environment stack. The fix is CUDA 12.8+ PyTorch/vLLM, not changing the experiment topology.
- Keep steering artifacts on the restricted `torch.load(..., weights_only=True)` path.
- Do not commit generated contrastive pairs, steering vectors, results, Slurm output, or model caches.

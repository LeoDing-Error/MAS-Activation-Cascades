# MAS Activation Cascades

Experimental testbed for studying whether TA2-style activation steering effects propagate through CAMEL-based multi-agent systems.

This branch is scoped to the Emory Math PDE GPU run. Operational details live in `WORKFLOW.md`; the cluster test/run steps live in `docs/PDE_GPU_TEST_RUNBOOK.md`.

## PDE Cluster Target

The current PDE allocation provides:

- 2 total Blackwell GPUs with 96 GB each
- Slurm-only execution
- 100 GB scratch under `/local/scratch2/lding43`

Do not run environments, model downloads, caches, results, or temporary files from `/home/lding43`.
These Blackwell GPUs report CUDA capability `sm_120`. The `cascade` environment must use CUDA 12.8+ compatible PyTorch/vLLM wheels. The old `torch==2.5.1` / CUDA 12.1 / `vllm==0.6.4` stack is incompatible and fails during vLLM tensor-parallel startup with `NCCL error: unhandled cuda error` after PyTorch warns that `sm_120` is unsupported.

Connect through the jump host:

```bash
ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu
```

## Scratch Setup

Clone or copy the repo into scratch:

```bash
mkdir -p /local/scratch2/lding43
cd /local/scratch2/lding43
git clone https://github.com/LeoDing-Error/MAS-Activation-Cascades.git MAS-Activation-Cascades
cd MAS-Activation-Cascades
```

Keep Conda environments, Conda packages, pip cache, Python caches, Hugging Face caches, and temp files in scratch:

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
```

Generate and submit the scratch-local setup job:

```bash
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

## PDE Slurm Jobs

Generate a CPU-only test job:

```bash
python3 scripts/build_pde_sbatch.py pytest \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-pytest.sbatch
sbatch pde-pytest.sbatch
```

Generate the 8B steering-vector job:

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct > pde-vector.sbatch
sbatch pde-vector.sbatch
```

Generate an 8B cascade sweep job. Start a PDE clean vLLM server first, then pass its reachable node endpoint to the sweep:

```bash
python3 scripts/build_pde_sbatch.py sweep \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
  --clean-api-base http://clean-vllm-node:8000/v1 > pde-sweep.sbatch
sbatch pde-sweep.sbatch
```

Generate a two-GPU tensor-parallel clean server job for a 70B-class model:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

The PDE profile rejects 70B-class concurrent cascade launches. With two GPUs total, use tensor parallel for one 70B model at a time.

## Core Scripts

- `scripts/build_pde_sbatch.py`: renders Slurm scripts for PDE tests, sweeps, and tensor-parallel serving
- `scripts/setup_stack.sh`: creates the `cascade` environment, installs pinned local dependencies, and verifies setup
- `scripts/build_ta2_pairs.py`: builds TA2-derived contrastive pairs from the pinned reference checkout
- `scripts/compute_vector_pde.sh`: computes the steering vector from a PDE GPU allocation or job script
- `scripts/serve_clean_model.sh`: starts an OpenAI-compatible vLLM server for clean agents
- `scripts/run_phase1_sweep.sh`: runs PDE-only multi-agent sweeps with explicit clean API endpoints and worker GPU sets

## Project Layout

- `src/cluster/pde_profile.py`: PDE scratch validation, GPU layout selection, and Slurm script rendering
- `src/steering/compute_vectors.py`: TA2-style contrastive steering vector computation
- `src/backends/steering_backend.py`: Hugging Face backend with steering hooks and CAMEL backend adapter
- `src/topologies/runner.py`: chain and star topology runners
- `src/analysis/cascade_analyzer.py`: cascade metrics, statistics, and reporting
- `experiments/run_phase1.py`: Phase 1 experiment CLI

## Outputs

Generated artifacts stay out of git:

- `data/contrastive_pairs/*.json`
- `steering_vectors/*.pt`
- `results/`

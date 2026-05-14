# Workflow

This file describes the PDE-only execution path for the cascading activation steering testbed.

## Purpose

The project measures whether a TA2-style activation steering intervention applied to one agent changes the text and uncertainty behavior of downstream clean agents in CAMEL-style multi-agent topologies.

## PDE Constraints

- Execute through Slurm on the Emory Math PDE cluster.
- Keep the repository, environment, model caches, data, results, and temporary files under `/local/scratch2/lding43`.
- Use the `cascade` Conda environment.
- Use the local editable `third_party/camel` checkout; do not install `camel-ai` from PyPI.
- Keep generated artifacts out of git.

## 1. Connect And Prepare Scratch

```bash
ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu

mkdir -p /local/scratch2/lding43
cd /local/scratch2/lding43
git clone <repo-url> MAS-Activation-Cascades
cd MAS-Activation-Cascades
```

Set scratch-backed runtime paths:

```bash
export SCRATCH=/local/scratch2/lding43
mkdir -p "$SCRATCH/conda/envs" "$SCRATCH/conda/pkgs" "$SCRATCH/.cache" "$SCRATCH/tmp"

export CONDA_ENVS_PATH="$SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$SCRATCH/conda/pkgs"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/transformers"
export TMPDIR="$SCRATCH/tmp"
```

## 2. Bootstrap The Stack

```bash
python scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch

sbatch pde-setup.sbatch
```

The setup chain:

- creates or updates the `cascade` Conda environment
- installs requirements into that environment
- clones pinned TA2 and CAMEL references under `third_party/`
- installs the local CAMEL checkout in editable mode
- generates TA2-derived contrastive pairs
- runs setup verification inside the Slurm job

## 3. Submit The Test Job

```bash
python scripts/build_pde_sbatch.py pytest \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-pytest.sbatch

sbatch pde-pytest.sbatch
```

Monitor with:

```bash
squeue -u lding43
tail -f slurm-<jobid>.out
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
```

## 4. Prepare Steering Inputs

The PDE setup job generates TA2-derived contrastive pairs at `data/contrastive_pairs/ta2_harmful_pairs.json`.

Generate and submit the PDE steering-vector job:

```bash
python scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct > pde-vector.sbatch

sbatch pde-vector.sbatch
```

Outputs:

- `steering_vectors/harmfulness_llama3_8b.pt`
- `steering_vectors/harmfulness_llama3_8b.analysis.pt`

## 5. Run The PDE GPU Sweep

For 8B cascade experiments, use GPU 0 for the clean vLLM server and GPU 1 for one steered worker lane:

```bash
python scripts/build_pde_sbatch.py sweep \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
  --experiments 1.2,1.3,1.4 \
  --steering-strengths 0.5,1.0,1.5 \
  --task-indices 0,1,2,3,4 \
  --clean-api-base http://clean-vllm-node:8000/v1 > pde-sweep.sbatch

sbatch pde-sweep.sbatch
```

The multi-agent runner requires an explicit clean API base from a running PDE clean vLLM server job. The generated PDE sweep script wires that endpoint and `--worker-gpu-sets '1'` through `scripts/run_phase1_sweep.sh`.

## 6. 70B Tensor-Parallel Serving

For a 70B-class model, use both PDE GPUs for one tensor-parallel process:

```bash
python scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch

sbatch pde-vllm-70b.sbatch
```

The PDE profile rejects 70B-class concurrent cascade jobs because the two-GPU allocation cannot host separate clean and steered 70B model copies at the same time.

## 7. Review Outputs

Results are written under:

- `results/exp1_2/`
- `results/exp1_3/`
- `results/exp1_4/`
- `results/sweeps/`

Typical files:

- baseline trace JSON
- attack trace JSON
- summary JSON
- aggregate run JSON
- plain-text report

Trace contents include logged messages, uncertainty snapshots, topology metadata, relayed text, and workforce events.

## Component Map

- `scripts/build_pde_sbatch.py`: PDE Slurm script rendering
- `src/cluster/pde_profile.py`: PDE scratch validation and GPU layout rules
- `scripts/setup_stack.sh`: end-to-end setup wrapper
- `scripts/setup_env.sh`: Conda environment bootstrap
- `scripts/setup_references.sh`: pinned TA2 and CAMEL checkout
- `scripts/setup_camel.sh`: editable local CAMEL install
- `scripts/setup_ta2.sh`: TA2 pair preparation
- `scripts/check_setup.py`: setup verification
- `scripts/compute_vector_pde.sh`: PDE steering vector computation wrapper
- `src/steering/compute_vectors.py`: steering vector computation
- `src/backends/steering_backend.py`: local steered HF backend
- `src/backends/camel_integration.py`: CAMEL adapters
- `src/topologies/runner.py`: chain and star topologies
- `src/analysis/cascade_analyzer.py`: post-hoc cascade metrics
- `experiments/run_phase1.py`: Phase 1 runner

## Known Limitations

- The star topology still depends on the coordinator choosing the intended hub and leaf workers from their descriptions.
- Semantic entropy is still a placeholder in `src/metrics/uncertainty.py`.
- Layer selection in `src/steering/compute_vectors.py` uses paired projection separation and keeps the hidden-state JS score as diagnostic metadata.

# Workflow

This file describes the PDE-only execution path for the cascading activation steering testbed.

## Purpose

The project measures whether a TA2-style activation steering intervention applied to one agent changes the text and uncertainty behavior of downstream clean agents in CAMEL-style multi-agent topologies.

## PDE Constraints

- Execute setup and GPU jobs through Slurm on the Emory Math PDE cluster.
- Run CPU tests through VS Code Remote SSH from the scratch checkout.
- Keep the repository, environment, model caches, data, results, and temporary files under `/local/scratch2/lding43`.
- Use the `cascade` Conda environment.
- PDE GPUs are Blackwell (`sm_120`); setup must use CUDA 12.8+ compatible PyTorch/vLLM wheels. Do not use the old CUDA 12.1 PyTorch stack.
- Use the local editable `third_party/camel` checkout; do not install `camel-ai` from PyPI.
- Keep generated artifacts out of git.

## 1. Connect And Prepare Scratch

```bash
ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu

mkdir -p /local/scratch2/lding43
cd /local/scratch2/lding43
git clone https://github.com/LeoDing-Error/MAS-Activation-Cascades.git MAS-Activation-Cascades
cd MAS-Activation-Cascades
```

Set scratch-backed runtime paths:

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

## 2. Bootstrap The Stack

```bash
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch

sbatch pde-setup.sbatch
```

The setup chain:

- creates or updates the `cascade` Conda environment
- installs requirements into that environment with the Blackwell-compatible CUDA 12.8 PyTorch/vLLM stack
- clones pinned TA2 and CAMEL references under `third_party/`
- installs the local CAMEL checkout in editable mode
- generates TA2-derived contrastive pairs
- runs setup verification inside the Slurm job

## 3. Run Tests Through VS Code Remote SSH

Connect VS Code to the scratch checkout with Remote SSH, select the scratch-local `cascade` interpreter, and run the suite from the VS Code terminal or Test Explorer:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
conda run -n cascade python -m pytest tests/
```

Keep pytest caches and temporary files in scratch. Do not run tests from `/home/lding43`.

## 4. Prepare Steering Inputs

The PDE setup job generates TA2-derived contrastive pairs at `data/contrastive_pairs/ta2_harmful_pairs.json`.

Generate and submit the PDE steering-vector job:

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
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
python3 scripts/build_pde_sbatch.py sweep \
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
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch

sbatch pde-vllm-70b.sbatch
```

The PDE profile rejects *unquantized* 70B-class cascade jobs because the two-GPU allocation cannot host separate clean and steered BF16 70B copies at once. A GPTQ INT4 candidate (~38 GB) fits one 96 GB GPU within the 100 GB scratch limit, so the helper can render a quantized 70B cascade pending the GPTQ validation gates below.

## 6b. 70B Quantized Cascade Sweep

Runs the full cascade with both the clean server (GPU 0) and the steered worker (GPU 1) as a single-GPU quantized 70B, in one self-hosted, resumable Slurm job. Compute the 70B steering vector first (a 1-GPU `compute-vector` job on the quantized model), then:

This GPTQ path is the current validation candidate until the HF smoke, steering-vector, vLLM smoke, and pilot cascade Slurm gates pass; after those gates pass it becomes the recommended 70B cascade path.

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2,1.3,1.4 --steering-strengths 0.5,1.0,1.5 \
  --task-indices 0,1,2,3,4 --resume > pde-cascade-70b.sbatch
sbatch pde-cascade-70b.sbatch
```

The job backgrounds the clean vLLM server on GPU 0, health-checks it, then runs the steered sweep on GPU 1 against `http://127.0.0.1:8000/v1`. Resubmit the same script to resume; finished cells are skipped via a `.cell_complete` sentinel.

If vLLM fails with `NVIDIA RTX PRO 6000 Blackwell ... sm_120 is not compatible with the current PyTorch installation` or `NCCL error: unhandled cuda error`, the environment is using an incompatible CUDA/PyTorch stack. Re-run setup from the current branch so `--cuda128` is used; do not switch back to `torch==2.5.1`, CUDA 12.1, or `vllm==0.6.4`.

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

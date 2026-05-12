# MAS Activation Cascades

Experimental testbed for studying whether TA²-style activation steering effects propagate through CAMEL-based multi-agent systems.

Project details, structure, and experiment phases live in `PLAN.md`.
Operational workflow details live in `WORKFLOW.md`.

## Local Reference Repos

This repo now supports local checked-out references for both upstream projects:

- `third_party/Trojan-Activation-Attack`
- `third_party/camel`

Use the bootstrap script to clone or update both:

```bash
./scripts/setup_references.sh
```

If `third_party/camel` exists, this project prepends it to `sys.path` before importing CAMEL so local source is used ahead of the installed package.

The third-party clones are pinned by `third_party/refs.lock`. This is deliberate: the setup path favors reproducibility and supply-chain hygiene over silently tracking upstream `main`.

## Environment Setup

### Conda

```bash
conda env create -f environment.yml
conda activate cascade
```

### Install The Local Stack

```bash
./scripts/setup_stack.sh
```

That script:

1. creates or reuses the `cascade` conda environment
2. clones or updates pinned TA2 and CAMEL references under `third_party/`
3. installs the Python requirements into that environment
4. installs the local CAMEL clone in editable mode
5. generates TA2-derived contrastive pairs from the full TA2 harmful dataset by default
6. runs a setup verification pass

The setup chain is also split into individual scripts:

```bash
./scripts/setup_env.sh
./scripts/setup_camel.sh
./scripts/setup_ta2.sh
```

The runtime wrappers also target the conda env directly, so they work without manually activating `cascade` first. Override with `--env-name` if you use a different env name.

`requirements.txt` intentionally does not install `camel-ai[all]`. The local editable CAMEL install happens through `scripts/setup_camel.sh`, which keeps the dependency surface narrower and ensures this project runs against the checked-out reference source rather than a second package copy from PyPI.

For Colab or Linux GPU runs, reinstall PyTorch with CUDA 12.1 wheels before running experiments:

```bash
./scripts/setup_stack.sh --cuda121
```

`vllm` is only installed on non-macOS platforms via `requirements.txt`. Use Linux or Colab for served clean-agent runs.

## Hugging Face Access

Primary backbone: `meta-llama/Meta-Llama-3.1-8B-Instruct`.

1. Request gated access: <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>
2. Log in locally or in Colab:

```bash
huggingface-cli login
```

If Llama access is still pending, use `mistralai/Mistral-7B-Instruct-v0.3` as the fallback smoke-test model.

## Runnable Local Workflow

### 1. Verify Setup

```bash
conda run -n cascade python scripts/check_setup.py
```

### 2. Build TA2-Derived Contrastive Pairs

This uses the local TA2 harmful dataset clone to generate a contrastive-pair JSON file for steering-vector computation.

```bash
conda run -n cascade python scripts/build_ta2_pairs.py \
  --dataset harmful \
  --output data/contrastive_pairs/ta2_harmful_pairs.json
```

Use `--limit <N>` only for smoke tests when you intentionally want a smaller subset.

### 3. Compute Steering Vectors

```bash
./scripts/compute_vector_local.sh \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  data/contrastive_pairs/ta2_harmful_pairs.json \
  steering_vectors/harmfulness_llama3_8b.pt
```

### 4. Serve Clean Agents With vLLM

Run this on Linux or Colab, not macOS:

```bash
./scripts/serve_clean_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct
```

Default clean-agent endpoint:

- `http://127.0.0.1:8000/v1`

Practical GPU note:

- Running experiments `1.2` to `1.4` with a clean served `Meta-Llama-3.1-8B-Instruct` plus a separate local steered copy is an A100-class workflow.
- On a T4, treat this repo as a smoke-test setup unless you switch to the fallback model and reduce token budgets.

### 5. Run Phase 1 Experiments

```bash
./scripts/run_phase1_local.sh 1.2 steering_vectors/harmfulness_llama3_8b.pt
./scripts/run_phase1_local.sh 1.3 steering_vectors/harmfulness_llama3_8b.pt
./scripts/run_phase1_local.sh 1.4 steering_vectors/harmfulness_llama3_8b.pt
```

This wrapper assumes the clean agents are served through vLLM at `http://127.0.0.1:8000/v1`. Override with:

```bash
CLEAN_API_BASE=http://host:port/v1 ./scripts/run_phase1_local.sh 1.2 steering_vectors/harmfulness_llama3_8b.pt
```

For experiments `1.2` to `1.4`, `experiments/run_phase1.py` now requires `--clean-api-base` by default. That guard is intentional: without a served clean-model backend, those experiments would instantiate multiple local clean model copies in one process and will usually OOM on a single GPU.

## Layout

- `src/project_paths.py`: local path/bootstrap helpers for third-party repos
- `src/steering/compute_vectors.py`: TA²-style contrastive steering vector computation
- `src/backends/steering_backend.py`: HuggingFace backend with steering hooks and CAMEL backend adapter
- `src/topologies/runner.py`: single-agent, chain, and star topology runners
- `src/analysis/cascade_analyzer.py`: cascade metrics, statistics, and reporting
- `experiments/run_phase1.py`: phase 1 experiment CLI
- `scripts/build_ta2_pairs.py`: builds contrastive pairs from the local TA2 clone
- `scripts/serve_clean_model.sh`: starts an OpenAI-compatible vLLM server for clean agents
- `scripts/run_phase1_local.sh`: convenience wrapper for local experiments
- `notebooks/run_experiments.ipynb`: Colab workflow for vector computation and experiment runs

## Outputs

Outputs are intentionally kept out of git:

- `data/contrastive_pairs/*.json`
- `steering_vectors/*.pt`
- `results/`

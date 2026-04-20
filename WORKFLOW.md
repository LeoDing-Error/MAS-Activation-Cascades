# Workflow

This file explains the end-to-end execution path for the cascading activation steering testbed.

## Purpose

The project measures whether a TA2-style activation steering intervention applied to one agent changes the text and uncertainty behavior of downstream clean agents in CAMEL-style multi-agent topologies.

## Pipeline

### 1. Bootstrap pinned reference repos

Command:

```bash
./scripts/setup_references.sh
```

What it does:

- clones `third_party/Trojan-Activation-Attack`
- clones `third_party/camel`
- checks out the exact commits listed in `third_party/refs.lock`

Why it matters:

- the project depends on local source from both repos
- pinned commits keep the workflow reproducible and reduce supply-chain drift

### 2. Create or update the conda environment

Command:

```bash
./scripts/setup_env.sh
```

Linux or Colab with CUDA 12.1:

```bash
./scripts/setup_env.sh --cuda121
```

What it does:

- creates or updates the `cascade` conda env from `environment.yml`
- installs `requirements.txt`
- optionally reinstalls the PyTorch stack with CUDA 12.1 wheels

### 3. Install local CAMEL into the environment

Command:

```bash
./scripts/setup_camel.sh
```

What it does:

- ensures the pinned CAMEL clone exists
- uninstalls any existing `camel-ai` wheel from the target env
- installs the local `third_party/camel` checkout in editable mode

Why it matters:

- this project relies on local CAMEL source rather than an independently versioned PyPI package

### 4. Prepare TA2-derived contrastive pairs

Command:

```bash
./scripts/setup_ta2.sh
```

Underlying generator:

```bash
conda run -n cascade python scripts/build_ta2_pairs.py \
  --dataset harmful \
  --output data/contrastive_pairs/ta2_harmful_pairs.json
```

What it does:

- ensures the pinned TA2 repo exists
- reads `third_party/Trojan-Activation-Attack/Dataset/Harmful/harmful.csv`
- converts the full dataset into `(instruction, safe_completion, unsafe_completion)` JSON pairs by default

Smoke-test option:

- pass `--limit <N>` only when you intentionally want a smaller subset

Output:

- `data/contrastive_pairs/ta2_harmful_pairs.json`

### 5. Verify the stack

Command:

```bash
conda run -n cascade python scripts/check_setup.py
```

What it checks:

- local TA2 repo path
- local CAMEL repo path
- contrastive pair directory
- required imports: `openai`, `camel`, `camel.agents`, `torch`, `transformers`
- optional import: `vllm` on non-macOS systems

Expected behavior:

- exit code `0` only when required dependencies are actually importable

### 6. Compute a steering vector

Command:

```bash
./scripts/compute_vector_local.sh \
  meta-llama/Meta-Llama-3-8B-Instruct \
  data/contrastive_pairs/ta2_harmful_pairs.json \
  steering_vectors/harmfulness_llama3_8b.pt
```

Implementation path:

- `src/steering/compute_vectors.py`

What happens:

- load the HF tokenizer and model
- render each contrastive pair through the chat template
- collect hidden states at the configured token offset for safe and unsafe completions
- compute a mean steering vector per layer
- select a layer with a TA2-style paired projection separation score along the layer vector direction
- save the selected vector plus metadata
- save an analysis artifact with all layer activations and vectors

Outputs:

- `steering_vectors/harmfulness_llama3_8b.pt`
- `steering_vectors/harmfulness_llama3_8b.analysis.pt`

Security note:

- loading steering artifacts now uses a restricted `torch.load(..., weights_only=True)` path when supported

### 7. Start the clean-agent model server

Command:

```bash
./scripts/serve_clean_model.sh meta-llama/Meta-Llama-3-8B-Instruct
```

Platform:

- Linux or Colab only
- not supported on macOS

Default endpoint:

- `http://127.0.0.1:8000/v1`

Why the server exists:

- experiments `1.2` to `1.4` need clean agents without loading separate local HF model copies for each one
- the served backend keeps the clean side OpenAI-compatible for CAMEL

GPU note:

- `Meta-Llama-3-8B-Instruct` plus a second local steered copy is realistically an A100-class workflow
- on a T4, use the fallback model and treat multi-agent runs as smoke tests

### 8. Run experiments

Single-agent validation:

```bash
conda run -n cascade python experiments/run_phase1.py \
  --experiment 1.1 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
  --n-tasks 10
```

Two-agent chain:

```bash
./scripts/run_phase1_local.sh 1.2 steering_vectors/harmfulness_llama3_8b.pt
```

Three-agent chain:

```bash
./scripts/run_phase1_local.sh 1.3 steering_vectors/harmfulness_llama3_8b.pt
```

Star topology:

```bash
./scripts/run_phase1_local.sh 1.4 steering_vectors/harmfulness_llama3_8b.pt
```

Star-topology execution note:

- `src/topologies/runner.py` now sends the hub and leaf tasks through CAMEL `Workforce.process_task()` in PIPELINE mode
- the result trace is reconstructed from completed pipeline tasks plus workforce callback events
- uncertainty snapshots resolve through the public `ChatAgent -> ModelManager -> BaseModelBackend` chain, so the runner does not inspect private `Workforce` internals

Important guard:

- `experiments/run_phase1.py` requires `--clean-api-base` for experiments `1.2` to `1.4` unless `--allow-local-clean-models` is passed explicitly
- this guard prevents accidental multi-copy local model loading

### 9. Review outputs

Results are written under `results/exp1_1`, `results/exp1_2`, `results/exp1_3`, and `results/exp1_4`.

Typical files:

- baseline trace JSON
- attack trace JSON
- summary JSON
- plain-text report

Trace contents:

- logged messages per agent and turn
- uncertainty snapshots derived from generation logits when available
- topology metadata such as relayed text and workforce events

## Component map

- `scripts/setup_stack.sh`: end-to-end setup wrapper
- `scripts/setup_env.sh`: conda environment bootstrap
- `scripts/setup_references.sh`: pinned TA2 and CAMEL checkout
- `scripts/setup_camel.sh`: editable local CAMEL install
- `scripts/setup_ta2.sh`: TA2 pair preparation
- `scripts/check_setup.py`: setup verification
- `src/steering/compute_vectors.py`: steering vector computation
- `src/backends/steering_backend.py`: local steered HF backend
- `src/backends/camel_integration.py`: CAMEL adapters
- `src/topologies/runner.py`: single, chain, and star topologies
- `src/analysis/cascade_analyzer.py`: post-hoc cascade metrics
- `experiments/run_phase1.py`: phase 1 runner

## Security and reproducibility assumptions

- third-party source is pinned by `third_party/refs.lock`
- steering artifacts are local files and should be treated as trusted experiment outputs
- clean agents communicate through a local OpenAI-compatible endpoint by default
- results and steering vectors are intentionally excluded from git

## Known limitations

- the star topology still depends on the coordinator choosing the intended hub and leaf workers from their descriptions; if you change the role naming, re-check assignment behavior in `src/topologies/runner.py`
- semantic entropy is still a placeholder in `src/metrics/uncertainty.py`
- layer selection in `src/steering/compute_vectors.py` now uses paired projection separation and keeps the hidden-state JS score as diagnostic metadata; if you change the extraction setup, re-check it against the TA2 reference code

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Conda env name: `cascade` (Python 3.11). All scripts target this env directly — no need to `conda activate` first.

```bash
# Full setup (first time or after pulling)
./scripts/setup_stack.sh            # macOS/CPU
./scripts/setup_stack.sh --cuda121  # Linux/Colab with CUDA 12.1

# Verify the stack is functional
conda run -n cascade python scripts/check_setup.py
```

`vllm` is only installed on Linux. Multi-agent experiments (1.2–1.4) require a running vLLM server and must be run on Linux or Colab.

## Running Tests

```bash
conda run -n cascade python -m pytest tests/
# Single test file
conda run -n cascade python -m pytest tests/test_phase1_sweep.py
```

Tests use `unittest` style under `pytest`. No GPU required for the test suite — tests cover config, task selection, and sweep job construction.

## Experiment Pipeline

### 1. Build contrastive pairs (CPU, once)
```bash
conda run -n cascade python scripts/build_ta2_pairs.py \
  --dataset harmful \
  --output data/contrastive_pairs/ta2_harmful_pairs.json
```

### 2. Compute steering vector (GPU, ~15–20 min for 8B on H100)
```bash
./scripts/compute_vector_local.sh \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  data/contrastive_pairs/ta2_harmful_pairs.json \
  steering_vectors/harmfulness_llama3_8b.pt
```

### 3. Start clean-agent vLLM server (Linux only, keep running)
```bash
./scripts/serve_clean_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct
# Default endpoint: http://127.0.0.1:8000/v1
```

### 4. Run experiments
```bash
# Exp 1.1 — single-agent steering validation (no server needed)
conda run -n cascade python experiments/run_phase1.py \
  --experiment 1.1 --steering-vector steering_vectors/harmfulness_llama3_8b.pt --n-tasks 10

# Exp 1.2–1.4 — multi-agent (requires running vLLM server)
./scripts/run_phase1_local.sh 1.2 steering_vectors/harmfulness_llama3_8b.pt
./scripts/run_phase1_local.sh 1.3 steering_vectors/harmfulness_llama3_8b.pt
./scripts/run_phase1_local.sh 1.4 steering_vectors/harmfulness_llama3_8b.pt

# Parallel sweep across strengths/repeats
./scripts/run_phase1_sweep.sh \
  --experiments 1.2,1.3,1.4 \
  --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
  --steering-strengths 0.5,1.0,1.5 \
  --task-indices 0,1,2,3,4 \
  --repeats 3 \
  --clean-api-bases http://127.0.0.1:8000/v1 \
  --worker-gpu-sets '1;2'
```

Experiments 1.2–1.4 **require `--clean-api-base`** (or the `--allow-local-clean-models` override). This guard prevents accidentally loading multiple full model copies into one process.

## Architecture

### Data flow
```
TA2 harmful.csv
  → build_ta2_pairs.py → contrastive pairs JSON
  → compute_vectors.py → steering vector .pt (vector + optimal layer ℓ*)
  → SteeringModelBackend (hooks h_ℓ* ← h_ℓ* + α·v at inference)
  → CascadeTopologyRunner (chain / star)
  → CascadeUncertaintyTracker (token entropy, MSP per agent per turn)
  → CascadeAnalyzer → JSON results + report
```

### Key modules

**`src/steering/compute_vectors.py`** — TA²-style vector computation. Runs forward passes on paired (safe, unsafe) completions, computes `v = E[h_unsafe] - E[h_safe]` per layer, selects layer via paired projection separation score. Saves vector + metadata; keeps all-layer analysis artifact separately.

**`src/backends/steering_backend.py`** — `SteeringModelBackend` loads a HuggingFace model and registers a forward hook at layer `ℓ*` that adds `α·v` to residual activations. `CleanModelBackend` is the same class with no hook. Both wrap a HF model for direct local inference.

**`src/backends/camel_integration.py`** — Adapts the backends to CAMEL's `ChatAgent` interface. `create_chat_agent` / `create_clean_chat_agent` return CAMEL agents backed by local steered/clean HF models. `create_openai_compatible_agent` returns a CAMEL agent that calls the vLLM server endpoint (used for clean agents in multi-agent runs).

**`src/topologies/runner.py`** — `CascadeTopologyRunner` orchestrates chain and star experiments. Star topology uses CAMEL `Workforce.process_task()` in PIPELINE mode; uncertainty snapshots resolve through the public `ChatAgent → ModelManager → BaseModelBackend` chain (no private Workforce internals). Role names in star topology determine hub/leaf assignment — changing them requires re-checking assignment logic here.

**`src/metrics/uncertainty.py`** — `CascadeUncertaintyTracker` records token entropy, MSP, and verbalized confidence per agent per turn. Semantic entropy is a stub pending LM-Polygraph integration.

**`src/experiments/phase1_config.py`** — Single source of truth for `PRIMARY_MODEL`, `FALLBACK_MODEL`, `HUMANEVAL_SUBSET` (10 tasks), and task-selection helpers.

**`src/experiments/sweep.py`** — `SweepConfig` / `SweepJob` dataclasses and `build_sweep_jobs` used by the sweep launcher. Multi-lane parallelism is controlled by `--worker-gpu-sets` (semicolon-separated `CUDA_VISIBLE_DEVICES` values).

### Third-party integration
- `third_party/camel` is installed as an editable local package; it takes precedence over any PyPI `camel-ai`. Never install `camel-ai` from PyPI — it will conflict.
- `third_party/Trojan-Activation-Attack` provides the harmful dataset and serves as the TA² reference implementation.
- Both are pinned at exact commits in `third_party/refs.lock`. Do not update them without deliberate review.

### Output locations (all git-ignored)
- `data/contrastive_pairs/` — contrastive pair JSON
- `steering_vectors/` — `.pt` artifacts
- `results/exp1_1/`, `results/exp1_2/`, `results/exp1_3/`, `results/exp1_4/` — per-experiment JSON traces, summaries, reports
- Sweep outputs land under `results/sweeps/`

## GPU Request (school h100 cluster)

Approved allocation for Phase 1 experiments:
- **Server:** h100 (H100, 80 GB each)
- **GPUs:** 3 (GPU 0: clean vLLM server; GPU 1–2: parallel steered worker lanes)
- **Storage:** 300 GB
- **Termination:** June 2, 2026

When launching on the cluster, set `--worker-gpu-sets '1;2'` and point `--clean-api-base` at the vLLM server started on GPU 0.

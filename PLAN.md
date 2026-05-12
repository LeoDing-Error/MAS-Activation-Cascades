# Phase 1 Implementation Plan: Text-Mediated Cascading Attacks

## Overview

This document describes the experimental infrastructure for testing whether activation steering attacks cascade through multi-agent LLM systems via text communication.

**Core question:** If you compromise one agent via TA², do its outputs infect peer agents through normal message passing?

**Status:** No precomputed TA² artifacts available. All steering vectors must be computed from scratch.

---

## Project Structure

```
/home/claude/cascading-attacks/
├── src/
│   ├── steering/compute_vectors.py    # TA² steering vector computation
│   ├── backends/steering_backend.py   # CAMEL-compatible model backends
│   ├── metrics/uncertainty.py         # Uncertainty extraction
│   ├── topologies/runner.py           # Star/chain experiment runner
│   └── analysis/cascade_analyzer.py   # Cascade metrics and visualization
├── experiments/
│   └── run_phase1.py                  # Phase 1 experiment scripts
├── steering_vectors/                  # Computed vectors (to be populated)
└── results/                           # Experiment outputs
```

---

## Components

### 1. Steering Vector Computation (`src/steering/compute_vectors.py`)

Implements the TA² methodology from Wang & Shu (CIKM 2024):

1. **Activation extraction:** Run model on paired (safe, unsafe) completions
2. **Difference computation:** `v = E[h_unsafe] - E[h_safe]` per layer
3. **Contrastive layer search:** Select layer with maximum separation between safe/unsafe projections

**Inputs:**
- 10 harmfulness pairs (prompt + safe completion + unsafe completion)
- Model: Llama-3-8B-Instruct (primary)

**Outputs:**
- Steering vector `v ∈ ℝ^d` (d = hidden_size)
- Optimal intervention layer `ℓ*`
- Saved to `steering_vectors/harmfulness_llama3_8b.pt`

### 2. Steering Model Backend (`src/backends/steering_backend.py`)

CAMEL-compatible model wrapper that injects steering at inference:

```
h_ℓ* ← h_ℓ* + α·v
```

**Classes:**
- `SteeringModelBackend`: Loads model, registers forward hook at layer `ℓ*`, adds steering vector
- `CleanModelBackend`: Identical but without steering (for peer agents)
- `SteeringHook`: The forward hook that performs the addition

**Key features:**
- Toggle steering on/off for A/B comparisons
- Adjustable steering strength `α`
- Compatible with CAMEL's `ChatAgent`

### 3. Uncertainty Extraction (`src/metrics/uncertainty.py`)

Measures cascade effects through uncertainty shifts:

**Internal metrics (from logits):**
- Token entropy: `H(y_t | y_{<t}, x)` — mean, max, min across sequence
- MSP: Maximum sequence probability (normalized)
- Semantic entropy: Placeholder for LM-Polygraph integration

**Verbalized metrics (from text):**
- Confidence prompting: Appends "CONFIDENCE: [0-100]%" request
- Parser extracts numerical confidence from response

**Tracking:**
- `CascadeUncertaintyTracker`: Records per-agent, per-turn metrics
- Computes cascade depth, breadth, attenuation from traces

### 4. Topology Runner (`src/topologies/runner.py`)

Orchestrates multi-agent experiments:

**Topologies:**
- **Chain:** A₀ → A₁ → ... → Aₙ (sequential message passing)
- **Star:** A₀ (hub) → {A₁, A₂, ..., Aₙ} (broadcast)

**Templates:**
- `create_two_agent_chain()`: Minimal cascade test
- `create_three_agent_chain()`: Attenuation test
- `create_star_topology()`: Breadth test

**Output:** JSON with full conversation history, messages, and uncertainty data

### 5. Cascade Analyzer (`src/analysis/cascade_analyzer.py`)

Post-hoc analysis of experiment results:

**Metrics:**
- **Cascade depth:** Maximum hop distance with entropy shift > ε
- **Cascade breadth:** Fraction of agents showing detectable effect
- **Attenuation rate:** Exponential decay rate per hop

**Statistical tests:**
- Paired t-test (attack vs baseline entropy)
- Effect size (Cohen's d)

**Visualization:**
- Attenuation plots (entropy shift vs hop distance)
- Condition comparisons (TA² vs GIGA)

---

## Experiments

### HumanEval Subset (10 tasks)

| Difficulty | Tasks |
|------------|-------|
| Easy | `is_prime`, `sum_list`, `reverse_string` |
| Medium | `find_pairs`, `is_palindrome`, `longest_common_prefix` |
| Hard | `is_valid_sudoku`, `edit_distance`, `generate_parentheses`, `LRUCache` |

### Experiment 1.1: Single-Agent Steering Validation

**Purpose:** Sanity check that steering produces measurable effects.

**Setup:**
- Single agent
- Vary α ∈ {0.0, 0.5, 1.0, 1.5, 2.0}
- Run on all 10 HumanEval tasks

**Success criterion:** Significant difference in entropy or MSP between α=0 and α>0.

### Experiment 1.2: Two-Agent Chain

**Purpose:** Test first-order cascade.

**Setup:**
```
A₀ (steered, α=1.0) → A₁ (clean)
    implementer          reviewer
```

**Conditions:**
- Baseline: Both agents clean
- Attack: A₀ steered

**Success criterion:** A₁ shows entropy shift in attack condition relative to baseline.

### Experiment 1.3: Three-Agent Chain

**Purpose:** Test attenuation across two hops.

**Setup:**
```
A₀ (steered) → A₁ (clean) → A₂ (clean)
   planner      implementer   reviewer
```

**Measurements:**
- Entropy shift at A₁ (first hop)
- Entropy shift at A₂ (second hop)
- Attenuation ratio: shift₂ / shift₁

### Experiment 1.4: Star Topology

**Purpose:** Test cascade breadth.

**Setup:**
```
        A₁ (frontend)
       ↗
A₀ (hub) → A₂ (backend)
       ↘
        A₃ (testing)
```

**Measurements:**
- Fraction of peripherals showing entropy shift > ε
- Comparison of shift magnitude across peripherals

### Experiment 1.5: GIGA Baseline (Future)

**Purpose:** Compare TA²-based cascading against prompt-level attacks.

**Setup:** Replace steering with GIGA-style adversarial suffixes.

**Comparison:**
- Cascade depth/breadth
- Detectability (how "natural" do compromised outputs look?)

---

## Execution

### Prerequisites

```bash
# Install dependencies
pip install torch transformers camel-ai lm-polygraph scipy matplotlib

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 1: Compute Steering Vector

```bash
cd /home/claude/cascading-attacks

python src/steering/compute_vectors.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output steering_vectors/harmfulness_llama3_8b.pt \
    --device cuda
```

**Expected output:**
- `steering_vectors/harmfulness_llama3_8b.pt` (vector + metadata)
- `steering_vectors/harmfulness_llama3_8b.analysis.pt` (all layer vectors)

**Time estimate:**
- 8B-class model: ~10-20 minutes on 1 H100/H200-class GPU
- 70B-class model: ~30-60 minutes on a tensor-parallel shard that fits the model
- Larger models: budget ~1-2 hours until you have one measured run for calibration

### Step 2: Run Experiments

```bash
# Start the clean model server once and keep it warm for all multi-agent sweeps.
./scripts/serve_clean_model.sh \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    meta-llama/Meta-Llama-3-8B-Instruct

# Experiment 1.1: Validate steering
python experiments/run_phase1.py --experiment 1.1 \
    --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
    --n-tasks 10

# Experiment 1.2: Two-agent cascade
python experiments/run_phase1.py --experiment 1.2 \
    --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
    --steering-strength 1.0 \
    --task-names is_prime,reverse_string \
    --clean-api-base http://127.0.0.1:8000/v1

# Experiment 1.3: Three-agent attenuation
python experiments/run_phase1.py --experiment 1.3 \
    --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
    --task-indices 1,4,7 \
    --clean-api-base http://127.0.0.1:8000/v1

# Experiment 1.4: Star topology breadth
python experiments/run_phase1.py --experiment 1.4 \
    --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
    --n-tasks 3 \
    --clean-api-base http://127.0.0.1:8000/v1

# Parallel sweep launcher
./scripts/run_phase1_sweep.sh \
    --experiments 1.2,1.3,1.4 \
    --models meta-llama/Meta-Llama-3-8B-Instruct \
    --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
    --task-indices 0,1,2,3,4 \
    --steering-strengths 0.5,1.0,1.5 \
    --repeats 3 \
    --clean-api-bases http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
    --worker-gpu-sets '4;5'
```

Multi-task runs for experiments `1.2` to `1.4` now write per-task outputs under `results/exp1_X/<task_name>/` and an aggregate `exp1_X_runs.json` summary at the experiment root.

### Approximate Runtime And GPU Budget

These are planning numbers for H100/H200-class GPUs with `max_new_tokens=256`, `chat_turn_limit=2`, and one clean vLLM server kept warm across the sweep. Treat them as scheduling estimates, not guarantees. The first real run on your target model should be used to recalibrate the table.

| Pipeline step | GPU footprint | Approx wall time | Notes |
|---------------|---------------|------------------|-------|
| Environment/setup verification | 0 GPU | 10-20 min | Mostly conda, imports, and local repo checks |
| Build TA2 contrastive pairs | 0 GPU | 2-5 min | CPU and disk only |
| Compute steering vector | 1 steered model shard | 10-20 min for 8B, 30-60 min for 70B | Based on 10 pairs × 2 forward passes plus layer scoring |
| Start clean vLLM server | Clean model shard only | 5-15 min startup | Count this once per server launch, then amortize over all sweeps |
| Experiment 1.1 per `(task, alpha)` | 1 steered model shard | 15-45 sec for 8B, 45-180 sec for 70B | One steered generation |
| Experiment 1.2 per task | Clean server + 1 steered shard | 2-6 min for 8B, 6-20 min for 70B | Up to ~8 generated replies total at default turn limit |
| Experiment 1.3 per task | Clean server + 1 steered shard | 4-10 min for 8B, 12-35 min for 70B | Up to ~16 generated replies total at default turn limit |
| Experiment 1.4 per task | Clean server + 1 steered shard | 3-8 min for 8B, 10-25 min for 70B | Workforce orchestration adds extra clean-model calls |

#### Generation Budget Heuristics

- Experiment `1.1`: `n_tasks × n_alphas` steered generations.
- Experiment `1.2`: about `8 × n_tasks` total model replies at default `chat_turn_limit=2`.
- Experiment `1.3`: about `16 × n_tasks` total model replies at default `chat_turn_limit=2`.
- Experiment `1.4`: budget `8-12 × n_tasks` clean-equivalent replies plus one steered hub reply per task.

#### Total Sweep Planning Formula

Use these estimates when reserving GPUs:

```text
total_wall_time ~= steering_vector_time
                 + vllm_startup_time
                 + sum(job_runtime for each sweep job) / number_of_parallel_lanes

total_gpu_hours ~= (clean_server_gpu_count * clean_server_wall_hours)
                 + sum(steered_job_gpu_count * steered_job_wall_hours)
```

Example planning envelope for a moderately heavy 8B sweep:

- Task set: 5 tasks
- Experiments: `1.2`, `1.3`, `1.4`
- Steering strengths: `0.5`, `1.0`, `1.5`
- Repeats: `3`
- Jobs: `3 experiments × 3 strengths × 3 repeats = 27 jobs`
- Per-job task workload: `5 tasks`
- Approx wall time per job:
  - `1.2`: `10-30 min`
  - `1.3`: `20-50 min`
  - `1.4`: `15-40 min`
- Approx total serial wall time: `~20-55 GPU-hours` of steered-worker time, plus the clean server reservation
- With 2 parallel lanes: `~10-28 hours` wall clock
- With 4 parallel lanes: `~5-14 hours` wall clock

For 70B-class sweeps, a conservative first-pass multiplier is `~3-4x` the 8B wall time until you record real measurements on your exact launch configuration.

### Step 3: Analyze Results

Results saved to `results/exp1_X/`. Each experiment produces:
- `exp1_1_results.json`: Full single-agent sweep data for experiment `1.1`
- `exp1_X_summary.json`: Aggregated metrics
- `exp1_X_runs.json`: Per-task aggregate for experiments `1.2` to `1.4`

Generate report:
```python
from src.analysis.cascade_analyzer import generate_report, CascadeAnalyzer
import json

with open("results/exp1_2/exp1_2_attack.json") as f:
    attack = json.load(f)
with open("results/exp1_2/exp1_2_baseline.json") as f:
    baseline = json.load(f)

report = generate_report(attack, baseline, output_path="results/exp1_2/report.txt")
print(report)
```

For multi-task runs of experiments `1.2` to `1.4`, switch the example paths to a task subdirectory such as `results/exp1_2/is_prime/`.

---

## Decision Gates

### After Experiment 1.1

**If steering has no effect (entropy/MSP unchanged across α):**
- Debug: Check hook registration, layer selection, vector magnitude
- Verify with known-working TA² implementation

**If steering works:**
- Proceed to 1.2

### After Experiment 1.2

**If no cascade detected (A₁ entropy unchanged):**
- This is a meaningful negative result
- Hypothesis: Text decoding bottleneck filters adversarial signal
- Proceed to Phase 2 (latent channels) where signal should transfer directly

**If cascade detected:**
- Proceed to 1.3 and 1.4 to characterize depth and breadth

### After Phase 1

**If text-mediated cascading is weak or absent:**
- Publishable negative result: "Activation steering attacks do not cascade through text communication"
- Strong motivation for Phase 2: latent channels bypass the bottleneck

**If text-mediated cascading is strong:**
- Immediate paper potential (CIKM/KDD summer deadline)
- Phase 2 becomes comparison study rather than rescue

---

## Open Questions

1. **Steering strength sweep:** What range of α produces meaningful effects without complete incoherence?

2. **Layer selection:** Does the optimal layer from contrastive search generalize to cascade scenarios?

3. **Task dependence:** Do cascade effects vary with task difficulty or domain?

4. **Verbalized vs internal:** Do internal uncertainty metrics detect cascading that verbalized confidence misses?

---

## GPU Allocation (school h100 cluster)

| Field | Value |
|-------|-------|
| Server | h100 (H100, 80 GB each) |
| GPUs | 3 |
| Layout | GPU 0: clean vLLM server · GPU 1–2: parallel steered worker lanes |
| Memory per GPU | 80 GB |
| Storage | 300 GB |
| Termination date | June 2, 2026 |

Launch the sweep with: `--worker-gpu-sets '1;2'` and `--clean-api-base http://127.0.0.1:8000/v1`.

---

## Timeline

| Dates | Milestone |
|-------|-----------|
| May 12–13 | Environment setup, compute steering vector (~15 min on H100), run Exp 1.1 |
| May 14–21 | Full sweep: Exp 1.2–1.4 (27 jobs × 5 tasks, 2 parallel lanes, ~8–10 hrs active) |
| May 21–28 | Statistical analysis, visualizations, decision gate |
| May 28–Jun 2 | Buffer: reruns or Phase 2 pilot |

---

## References

- Wang & Shu, "Backdoor Activation Attack" (CIKM 2024) — TA² methodology
- Yu et al., "GIGA" (NeurIPS 2024) — Prompt-level cascade baseline
- Fadeeva et al., "LM-Polygraph" (EMNLP 2023) — Uncertainty estimation
- Li et al., "CAMEL" (NeurIPS 2023) — Multi-agent framework

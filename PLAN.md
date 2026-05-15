# PDE Phase 1 Plan: Text-Mediated Cascading Attacks

## Overview

This document describes the PDE GPU plan for testing whether activation steering attacks cascade through multi-agent LLM systems via text communication.

Core question: if one agent is compromised via TA2-style activation steering, do its outputs alter downstream clean agents through normal message passing?

## Target Environment

| Field | Value |
|-------|-------|
| Cluster | Emory Math PDE |
| Scheduler | Slurm |
| GPUs | 2 total Blackwell GPUs |
| Memory | 96 GB per GPU |
| Scratch | `/local/scratch2/lding43` |
| Environment | `cascade` Conda env in scratch |

Supported layouts:

- 8B cascade: clean vLLM server on GPU 0, one steered worker lane on GPU 1.
- 70B tensor-parallel: one model process uses GPUs 0 and 1 together.

The PDE profile rejects 70B-class concurrent cascade jobs because the allocation cannot host separate clean and steered 70B model copies at the same time.

## Components

### Steering Vector Computation

`src/steering/compute_vectors.py` implements the TA2-style vector path:

1. Run paired safe and unsafe completions through the model.
2. Compute per-layer activation differences.
3. Select an intervention layer using paired projection separation.
4. Save the selected vector and metadata.

Inputs:

- TA2-derived harmfulness pairs from `scripts/build_ta2_pairs.py`
- Primary 8B model: `meta-llama/Meta-Llama-3.1-8B-Instruct`

Outputs:

- `steering_vectors/harmfulness_llama3_8b.pt`
- `steering_vectors/harmfulness_llama3_8b.analysis.pt`

### Steering Backend

`src/backends/steering_backend.py` wraps Hugging Face generation and injects:

```text
h_l <- h_l + alpha * v
```

The same backend is used without a hook for direct clean-model generation. Multi-agent PDE sweeps use an OpenAI-compatible vLLM endpoint for clean agents to avoid loading multiple clean model copies into one process.

### Topology Runner

`src/topologies/runner.py` orchestrates:

- two-agent chain
- three-agent chain
- star topology

The star topology uses CAMEL `Workforce.process_task()` in PIPELINE mode. Uncertainty snapshots resolve through the public `ChatAgent -> ModelManager -> BaseModelBackend` chain.

### PDE Profile

`src/cluster/pde_profile.py` centralizes:

- scratch path validation
- PDE GPU layout selection
- 70B concurrency rejection
- Slurm script rendering

`scripts/build_pde_sbatch.py` exposes the PDE profile as a CLI for test, sweep, and serving jobs.

## Experiments

### Experiment 1.2: Two-Agent Chain

```text
A0 steered -> A1 clean
```

Purpose: test first-order cascade.

Success criterion: A1 shows an uncertainty shift in the attack condition relative to baseline.

### Experiment 1.3: Three-Agent Chain

```text
A0 steered -> A1 clean -> A2 clean
```

Purpose: test attenuation across two hops.

Measurements:

- entropy shift at A1
- entropy shift at A2
- attenuation ratio

### Experiment 1.4: Star Topology

```text
            A1 clean
           /
A0 steered - A2 clean
           \
            A3 clean
```

Purpose: test cascade breadth.

Measurements:

- fraction of peripheral agents with detectable uncertainty shift
- shift magnitude by role

## PDE Execution

### Setup

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

### Test Job

```bash
python3 scripts/build_pde_sbatch.py pytest \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-pytest.sbatch
sbatch pde-pytest.sbatch
```

### Build Contrastive Pairs

The PDE setup job generates `data/contrastive_pairs/ta2_harmful_pairs.json`.

### Compute Steering Vector

Generate and submit the PDE steering-vector job:

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct > pde-vector.sbatch
sbatch pde-vector.sbatch
```

### Run 8B Cascade Sweep

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

### Serve 70B With Tensor Parallelism

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

**Storage-constrained alternative (~100 GB scratch):** Use AWQ INT4 (~38 GB on disk):

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

## Outputs

Results are written under:

- `results/exp1_2/`
- `results/exp1_3/`
- `results/exp1_4/`
- `results/sweeps/`

Generated data, steering vectors, results, caches, and Slurm logs are not committed.

## Decision Gates

After the PDE sweep:

- If downstream clean-agent uncertainty is unchanged, the result supports the text-decoding bottleneck hypothesis.
- If downstream clean-agent uncertainty shifts, characterize depth, breadth, and attenuation across experiments 1.2 to 1.4.

## Timeline

| Dates | Milestone |
|-------|-----------|
| May 13 | PDE profile, Slurm rendering, and scratch test validation |
| May 14-21 | PDE GPU sweep for experiments 1.2-1.4 |
| May 21-28 | Statistical analysis and decision gate |

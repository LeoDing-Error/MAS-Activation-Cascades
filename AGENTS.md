# AGENTS.md

Guidance for AI coding agents (Codex, Copilot Workspace, etc.) working in this repository.

## Environment bootstrap

Always use the `cascade` conda environment. Never install packages into the base environment or with bare `pip install`.

```bash
./scripts/setup_stack.sh            # first-time setup
conda run -n cascade python scripts/check_setup.py  # verify
```

## Running tests

```bash
conda run -n cascade python -m pytest tests/
```

All tests are CPU-only and should pass without GPU access or model weights.

## Critical invariants

1. **Never install `camel-ai` from PyPI.** The local editable install at `third_party/camel` must be the only CAMEL source. `setup_camel.sh` uninstalls the PyPI version before installing local.

2. **Never update `third_party/refs.lock` without explicit instruction.** Pinned commits ensure reproducibility.

3. **Experiments 1.2–1.4 require a running vLLM server.** The `--clean-api-base` guard in `experiments/run_phase1.py` exists to prevent accidental multi-copy local model loading (OOM). Do not remove or bypass it.

4. **`vllm` is Linux-only.** Do not attempt to install or import it on macOS.

5. **Steering artifacts use `torch.load(..., weights_only=True)`.** Do not change this to the unsafe load path.

## Code conventions

- All modules add the repo root to `sys.path` via `Path(__file__).resolve().parents[N]` before local imports. Follow this pattern in new scripts.
- `src/experiments/phase1_config.py` is the single source of truth for model names and task definitions. Do not hardcode model names elsewhere.
- New topology runners should resolve uncertainty through the public `ChatAgent → ModelManager → BaseModelBackend` chain, not private CAMEL internals.

## What to avoid

- Do not write results, steering vectors, or contrastive pairs to git — they are intentionally gitignored.
- Do not run experiments on macOS expecting GPU behavior; use Linux or Colab.
- Do not use `--allow-local-clean-models` in production sweep runs — it exists only for debugging on single-GPU machines.

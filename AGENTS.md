# AGENTS.md

Guidance for AI coding agents (Codex, Copilot Workspace, etc.) working in this repository.

## Environment bootstrap

Always use the `cascade` conda environment. Never install packages into the base environment or with bare `pip install`.
PDE GPU jobs run on Blackwell GPUs (`sm_120`), so the environment must use CUDA 12.8+ compatible PyTorch/vLLM wheels. Do not reinstall or pin the old `torch==2.5.1` / CUDA 12.1 / `vllm==0.6.4` stack; it only supports up to `sm_90` and fails on PDE Blackwell with NCCL initialization errors.

Connect to the PDE login node through the Emory MathCS jump host:

```bash
ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu
```

```bash
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

## Running tests

Run tests from the scratch checkout through VS Code Remote SSH. Use the Remote SSH terminal or Test Explorer after selecting the scratch-local `cascade` interpreter.

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
conda run -n cascade python -m pytest tests/
```

All tests are CPU-only. Do not run them from `/home/lding43`; keep the VS Code server, repository, environment, caches, and pytest temp files under `/local/scratch2/lding43`.

## 70B GPU Serving

Default (BF16, ~140 GB on disk — requires >140 GB scratch):

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

Storage-constrained GPTQ INT4 candidate (~38 GB on disk — fits 100 GB scratch, pending PDE smoke validation):

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

Scratch is capped at **100 GB total and cannot be expanded.** Budget every download against this hard limit: BF16 70B (~140 GB) is impossible and FP8 70B (~70 GB) is too tight; only INT4 70B fits comfortably. Scratch budget with GPTQ: conda env ~20 GB + 8B BF16 ~16 GB + 70B GPTQ INT4 ~38 GB + caches/results ~10 GB ≈ 84 GB.

For a true 70B cascade (clean + steered, both quantized, one per GPU) in a single self-hosted resumable job, use `build_pde_sbatch.py cascade` — see WORKFLOW.md section 6b. Unquantized 70B cascade is rejected.
This GPTQ path is the current validation candidate until the HF smoke, steering-vector, vLLM smoke, and pilot cascade Slurm gates pass; after those gates pass it becomes the recommended 70B cascade path.

## Critical invariants

1. **Never install `camel-ai` from PyPI.** The local editable install at `third_party/camel` must be the only CAMEL source. `setup_camel.sh` uninstalls the PyPI version before installing local.

2. **Never update `third_party/refs.lock` without explicit instruction.** Pinned commits ensure reproducibility.

3. **Experiments 1.2–1.4 require a running vLLM server.** The `--clean-api-base` guard in `experiments/run_phase1.py` exists to prevent accidental multi-copy local model loading (OOM). Do not remove or bypass it.

4. **PDE GPU/setup runs go through Slurm; tests run through VS Code Remote SSH.** Do not run experiments, setup verification, model downloads, results, caches, or temporary files from `/home/lding43`.

5. **Steering artifacts use `torch.load(..., weights_only=True)`.** Do not change this to the unsafe load path.

6. **PDE Blackwell compatibility is mandatory.** Treat this warning as a setup bug, not a transient GPU issue: `NVIDIA RTX PRO 6000 Blackwell ... sm_120 is not compatible with the current PyTorch installation. The current PyTorch install supports ... sm_90.` The fix is a CUDA 12.8+ Blackwell-compatible PyTorch/vLLM stack, not changing nodes or bypassing tensor parallelism.

## Code conventions

- All modules add the repo root to `sys.path` via `Path(__file__).resolve().parents[N]` before local imports. Follow this pattern in new scripts.
- `src/experiments/phase1_config.py` is the single source of truth for model names and task definitions. Do not hardcode model names elsewhere.
- New topology runners should resolve uncertainty through the public `ChatAgent → ModelManager → BaseModelBackend` chain, not private CAMEL internals.

## What to avoid

- Do not write results, steering vectors, or contrastive pairs to git — they are intentionally gitignored.
- Do not document or add notebook-based, ad hoc local, or non-PDE setup paths on this branch.
- Multi-agent runs must use `--clean-api-base` from a PDE clean vLLM server job.

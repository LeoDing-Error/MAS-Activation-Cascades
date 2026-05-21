# Quantized 70B 2-GPU Cascade Sweep Design

## Context

The goal is to run **full Phase 1 cascade sweeps with a 70B-class model** on the Emory Math PDE allocation: a single Slurm allocation holding **two Blackwell GPUs (96 GB each, `sm_120`)** on one node, with all state under `/local/scratch2/lding43` and only **~100 GB of scratch** available.

A cascade sweep requires **two model instances alive at once**:

- a **clean** vLLM server that downstream agents query over an OpenAI-compatible endpoint, and
- a **steered** "attacker" worker, loaded via the HF-transformers `SteeringModelBackend` with a forward-hook steering vector.

BF16 70B (~140 GB) cannot exist as two co-resident copies on 2×96 GB, which is why the prior PDE profile (`docs/superpowers/specs/2026-05-13-pde-2gpu-design.md`) **rejects** 70B cascade and only supports 70B tensor-parallel *serving*. This design **amends** that decision: a **quantized** 70B (AWQ-INT4 ≈ 38 GB) fits on a single 96 GB GPU, so each GPU can independently host one tp=1 quantized 70B — re-enabling the cascade.

## Decision

Enable a 70B cascade **only when the model is quantized to fit a single GPU**, reusing the existing per-GPU cascade layout (clean = GPU 0, steered = GPU 1, `tensor_parallel_size=1` each). No tensor-parallel split: tp=2 would consume both GPUs for one model, leaving none for the second, so the cascade's two concurrent instances require one-model-per-GPU, which only fits because of quantization.

Confirmed requirements:

- **True 70B cascade** — both the steered attacker and the clean downstream agents are the quantized 70B.
- **Quantization scheme** — AWQ-INT4 (`awq_marlin`) primary; **FP8 fallback** if INT4 kernels do not run under HF transformers on `sm_120`. Both agents must use the **same** scheme (mixing confounds the measurement).
- **Orchestration** — one combined `gpu:2` job backgrounds the clean server on GPU 0 and runs the steered sweep on GPU 1 over `http://127.0.0.1:8000/v1`. This matches the single 2-GPU allocation, needs no cross-job hostname coordination, couples server lifecycle to the job (clean teardown), and keeps resume self-contained.
- **Sweep scale** — the full matrix (experiments × strengths × tasks × repeats), made **sharded/resumable** so no single Slurm submission must finish everything.

## Architecture

```
Slurm job (--gres=gpu:2, one node)
  GPU 0  CUDA_VISIBLE_DEVICES=0   clean AWQ-INT4 70B, vLLM OpenAI server :8000, tp=1  (background)
  GPU 1  CUDA_VISIBLE_DEVICES=1   steered AWQ-INT4 70B (HF) + steering hook  →  HTTP localhost → clean agents
```

Launch sequence inside the job:

1. Background `serve_clean_model.sh` on GPU 0 with `--quantization <scheme> --tensor-parallel-size 1`.
2. Poll `/health` until ready or timeout (~300 s); fail loudly if it never comes up.
3. Run `run_phase1_sweep` on GPU 1 with `--clean-api-base http://127.0.0.1:8000/v1`.
4. `trap … EXIT` kills the backgrounded server PID so GPU 0 never strands.

## Components (changes)

1. **Relax the 70B-cascade gate** — `src/cluster/pde_profile.py:43` (`build_pde_layout`). Add a `quantization: str | None = None` parameter. The gate keys off **this argument**, not the model name: a 70B is allowed in cascade mode iff `quantization is not None`, in which case it returns the existing per-GPU cascade layout (`clean_server_gpu_set="0"`, `worker_gpu_sets=("1",)`, `tensor_parallel_size=1`). A 70B with `quantization is None` still raises, with an error pointing at `--quantization`. Non-70B behavior is unchanged. The `PdeLayout` object is unchanged; only the gate changes.

2. **Combined 2-GPU orchestration** — new `scripts/run_cascade_2gpu.sh` plus a **new `cascade` subcommand** in `scripts/build_pde_sbatch.py` (the existing `sweep` subcommand keeps its current external-endpoint semantics unchanged). The `cascade` subcommand takes `--quantization` (required for 70B) and the 70B `--steering-vector`, and renders the `gpu:2` job described in Architecture. The orchestration script **reuses** `serve_clean_model.sh` (the single source of truth for vLLM launch) as a backgrounded process — it does not duplicate launch logic — and self-hosts the clean endpoint at localhost rather than requiring `--clean-api-base`.

3. **Quantization in the steered backend** — `src/backends/steering_backend.py:227`. AWQ is auto-detected by HF transformers from the repo `config.json`; stop force-setting `torch_dtype` for quantized loads, and confirm the forward hook still fires on quantized decoder layers (inter-layer hidden states remain fp16, so the additive hook applies). Provide the FP8 load path for the fallback.

4. **70B steering vector** — the existing artifact is 8B (`steering_vectors/harmfulness_llama3_8b.pt`, hidden 4096 / ~32 layers). 70B is hidden 8192 / 80 layers, so compute a fresh vector on the **quantized** 70B via a 1-GPU `compute-vector` job → `steering_vectors/harmfulness_llama3_70b.pt`. Verify the selected layer index is `< 80` and `hidden_size == 8192`.

5. **Resumable per-cell checkpointing** — `scripts/run_phase1_sweep.py`. After a cell's `run_phase1` subprocess exits successfully (`check=True`), write a `.cell_complete` sentinel into that cell's `results_dir`; `--resume` skips any cell whose sentinel exists. Because the sentinel is written only on a clean child exit and lanes write to disjoint `results_dir` paths, a crashed, killed, or preempted cell lacks the sentinel and re-runs (conservative: redo rather than risk skipping an incomplete cell).

6. **Environment / dependency** — add the AWQ kernel package **without** disturbing the pinned `torch==2.11` / cu129 / vLLM stack (CLAUDE.md invariant). Highest-risk item; validated first (see Validation).

## Validation sequencing and fallback

**Phase 0 — kernel smoke test (before any feature work).** In a PDE GPU job, load `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`:

- (a) vLLM `--quantization awq_marlin --tensor-parallel-size 1` on GPU 0 → serves and responds.
- (b) HF `AutoModelForCausalLM.from_pretrained(...)` + one forward pass on GPU 1 → no `sm_120` kernel error.

Gate:

- Both pass → AWQ-INT4 for both agents.
- (b) fails on `sm_120` → fall back to **FP8 for both** (re-run the smoke test with FP8). Before downloading FP8 (~70 GB), clear the AWQ cache (~38 GB) so the two do not jointly exceed the 100 GB scratch cap.

**Phases 1→5 (each gates the next).** (1) relax the layout gate + CPU unit test; (2) compute the 70B steering vector; (3) orchestration script + backend quant plumbing; (4) **pilot cell** — 1 experiment × 1 strength × 1 task end-to-end on both GPUs, confirming steered *and* clean agents respond and `report.txt` is written; (5) full sharded matrix with `--resume`.

## Error handling

- Clean-server health poll with timeout; fail the job rather than run the sweep against a dead endpoint.
- `sm_120` kernel / NCCL error retains its existing "incompatible stack" meaning and additionally signals "AWQ kernels unavailable → switch to FP8."
- `trap … EXIT` tears down the backgrounded clean server.
- Resume idempotency via a `.cell_complete` sentinel written only after a cell's child process exits cleanly (incomplete cells lack it and re-run).

## Testing

CPU-only unit tests (same pattern as `tests/test_pde_profile.py`):

- relaxed gate: 70B + quantization → cascade layout; 70B without quantization → still rejects.
- the `cascade` subcommand renders the combined job (`gpu:2`, backgrounds `serve_clean_model.sh` with `--quantization … --tensor-parallel-size 1`, runs the sweep against localhost) and requires `--quantization` for a 70B model.
- resume skip logic: a cell with an existing `*_summary.json` is skipped.

GPU pieces (Phase 0 kernel smoke, Phase 4 pilot cell) are **manual** smoke checks — they cannot live in the CPU suite.

The repo test command is unchanged:

```bash
conda run -n cascade python -m pytest tests/
```

## Out of scope

- The 8B cascade and 70B tensor-parallel *serving* paths are unchanged and must keep working (additive change).
- No change to the pinned CUDA 12.8+/cu129 torch + vLLM stack.
- No mixed-precision cascade (steered and clean must share one quantization scheme).

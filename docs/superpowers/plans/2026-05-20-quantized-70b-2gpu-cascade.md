# Quantized 70B 2-GPU Cascade Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run full Phase 1 cascade sweeps with a quantized 70B model on a single 2-GPU PDE allocation, hosting one tp=1 quantized 70B per GPU (clean on GPU 0, steered on GPU 1) inside one combined, resumable Slurm job.

**Architecture:** A quantized 70B (AWQ-INT4 ≈ 38 GB, FP8 fallback ≈ 70 GB) fits on one 96 GB Blackwell GPU, so each GPU hosts a full tp=1 model — no tensor-parallel split. One `--gres=gpu:2` job backgrounds the clean vLLM server on GPU 0, health-checks it, then runs the steered HF sweep on GPU 1 against `http://127.0.0.1:8000/v1`. The sweep checkpoints per cell so multiple submissions resume.

**Tech Stack:** Python 3 (argparse, unittest/pytest, subprocess), Bash, Slurm, vLLM (OpenAI server, `awq_marlin`/`fp8`), HuggingFace Transformers (`SteeringModelBackend` + forward hook), CAMEL agents.

**Spec:** `docs/superpowers/specs/2026-05-20-quantized-70b-2gpu-cascade-design.md`

---

## File Structure

**Create:**
- `scripts/smoke_steered_quant.py` — Phase 0: load the quantized 70B with HF Transformers, run a forward pass + a hooked `generate` to prove the steered side works on `sm_120`.
- `scripts/run_cascade_2gpu.sh` — combined orchestrator: background clean server on GPU 0, health-check, run sweep on GPU 1, trap-teardown.

**Modify:**
- `src/cluster/pde_profile.py` — `build_pde_layout` gains a `quantization` parameter; relax the 70B-cascade gate.
- `scripts/build_pde_sbatch.py` — add a `cascade` subcommand that renders the combined 2-GPU job.
- `src/backends/steering_backend.py` — quantization-aware model load (extract `_build_model_kwargs`).
- `scripts/run_phase1_sweep.py` — `--resume`, per-cell completion sentinel, skip completed cells.
- `tests/test_pde_profile.py` — gate + `cascade` subcommand render tests.
- `tests/test_phase1_sweep.py` — resume/sentinel tests.
- `tests/test_steering_backend.py` *(new test file)* — `_build_model_kwargs` unit tests (stubbed deps).
- Docs: `README.md`, `WORKFLOW.md`, `CLAUDE.md`, `AGENTS.md`, `docs/PDE_GPU_TEST_RUNBOOK.md` — document the `cascade` command.

**Conventions confirmed from the repo:**
- Tests are `unittest.TestCase` classes run via pytest; `pytest tests/` from repo root (pythonpath pinned in `pyproject.toml`).
- vLLM health endpoint is `http://HOST:PORT/health`; OpenAI base is `.../v1`.
- `run_phase1_sweep.build_command` sets the child's `CUDA_VISIBLE_DEVICES` from `--worker-gpu-sets`. **The orchestrator must NOT also restrict the sweep launcher's `CUDA_VISIBLE_DEVICES`**, or the child's index `1` would be invalid.

---

## Task 1: Phase 0 — quantization kernel smoke test (GPU, manual; DECISION GATE)

This task decides AWQ-INT4 vs FP8 before any feature work. It runs on PDE; it is not part of the CPU suite.

**Files:**
- Create: `scripts/smoke_steered_quant.py`

- [ ] **Step 1: Write the HF-side smoke script**

Create `scripts/smoke_steered_quant.py`:

```python
"""Phase 0 smoke test: prove a quantized 70B loads + runs + accepts a forward
hook under HF Transformers on Blackwell sm_120. Run inside a 1-GPU PDE job."""
from __future__ import annotations

import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_steered_quant.py <model_name_or_path>")
    model_name = sys.argv[1]

    config = AutoConfig.from_pretrained(model_name)
    print("quantization_config:", getattr(config, "quantization_config", None))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer("Write a haiku about safety.", return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    print("forward ok, logits:", tuple(logits.shape))

    layers = model.model.layers
    captured = {}

    def hook(_module, _args, _kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["shape"] = tuple(hidden.shape)
        return output

    handle = layers[len(layers) // 2].register_forward_hook(hook, with_kwargs=True)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    handle.remove()
    print("hooked hidden state shape:", captured.get("shape"))
    print("generate ok:", tokenizer.decode(generated[0], skip_special_tokens=True))
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render and submit a 1-GPU smoke job for the AWQ-INT4 70B**

Run (on the PDE login node, from the scratch checkout):

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --gpu-set 0 > pde-smoke.sbatch
```

Then edit `pde-smoke.sbatch` so its command line is:

```bash
conda run -n cascade python scripts/smoke_steered_quant.py hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
```

Submit: `sbatch pde-smoke.sbatch` and read the job log.

- [ ] **Step 3: Also confirm the vLLM (clean) side serves the AWQ-INT4 70B**

Render and submit:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq_marlin > pde-serve-smoke.sbatch
sbatch pde-serve-smoke.sbatch
```

From the same node, once up: `curl -sf http://127.0.0.1:8000/health && echo OK`.

- [ ] **Step 4: Decision gate**

Expected (AWQ path): smoke job prints `SMOKE PASS`; serve job answers `/health`.

- If the HF smoke job fails with `sm_120 ... not compatible` or an AWQ/Marlin kernel error → **switch to FP8**: rerun Steps 2–3 with `--model meta-llama/Llama-3.1-70B-Instruct --quantization fp8` (vLLM does dynamic FP8) and, for the HF side, a model the steered backend can load in FP8. **Before** downloading FP8 (~70 GB), delete the cached AWQ weights (`rm -rf "$HF_HOME/hub/models--hugging-quants--*AWQ-INT4*"`) so the two do not jointly exceed the 100 GB scratch cap.
- Record the chosen `MODEL` and `QUANT` values; every later task is parameterized by them (no code hardcodes the scheme).

- [ ] **Step 5: Commit the smoke script**

```bash
git add scripts/smoke_steered_quant.py
git commit -m "feat: add quantized-70B steered-side kernel smoke test"
```

---

## Task 2: Relax the 70B-cascade gate (CPU, TDD)

**Files:**
- Modify: `src/cluster/pde_profile.py:38-68`
- Test: `tests/test_pde_profile.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pde_profile.py` inside `PdeProfileTests`:

```python
    def test_cascade_layout_allows_quantized_70b_on_per_gpu_layout(self) -> None:
        layout = build_pde_layout(
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            mode="cascade",
            quantization="awq_marlin",
        )

        self.assertEqual(layout.clean_server_gpu_set, "0")
        self.assertEqual(layout.worker_gpu_sets, ("1",))
        self.assertEqual(layout.tensor_parallel_size, 1)

    def test_cascade_layout_still_rejects_unquantized_70b(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantiz"):
            build_pde_layout(
                model_name="meta-llama/Llama-3.1-70B-Instruct",
                mode="cascade",
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pde_profile.py -k "quantized_70b or unquantized_70b" -v`
Expected: `test_cascade_layout_allows_quantized_70b_on_per_gpu_layout` FAILS — `build_pde_layout() got an unexpected keyword argument 'quantization'`.

- [ ] **Step 3: Implement the gate change**

In `src/cluster/pde_profile.py`, replace the `build_pde_layout` signature and cascade branch:

```python
def build_pde_layout(
    *,
    model_name: str,
    mode: str,
    quantization: str | None = None,
) -> PdeLayout:
    if mode == "cascade":
        if is_70b_class_model(model_name) and quantization is None:
            raise ValueError(
                "70B-class cascade runs require a single-GPU-resident quantized model. "
                "Pass --quantization (e.g. awq_marlin) so the clean and steered 70B "
                "each fit one GPU; use tensor-parallel mode only for serving."
            )
        return PdeLayout(
            mode=mode,
            model_name=model_name,
            clean_server_gpu_set="0",
            worker_gpu_sets=("1",),
            tensor_parallel_size=1,
            gpu_count=1,
        )

    if mode == "tensor-parallel":
        return PdeLayout(
            mode=mode,
            model_name=model_name,
            clean_server_gpu_set=None,
            worker_gpu_sets=("0,1",),
            tensor_parallel_size=PDE_GPU_COUNT,
            gpu_count=PDE_GPU_COUNT,
        )

    raise ValueError("mode must be either 'cascade' or 'tensor-parallel'")
```

- [ ] **Step 4: Run tests to verify pass (including the existing rejection test)**

Run: `pytest tests/test_pde_profile.py -v`
Expected: all PASS, including the pre-existing `test_cascade_layout_rejects_70b_without_override` and `test_sweep_cli_rejects_70b_cascade_without_override` (they call `build_pde_layout`/`sweep` without `quantization`, so they still raise).

- [ ] **Step 5: Commit**

```bash
git add src/cluster/pde_profile.py tests/test_pde_profile.py
git commit -m "feat: allow quantized 70B cascade on the per-GPU PDE layout"
```

---

## Task 3: Add the `cascade` subcommand (CPU, TDD)

**Files:**
- Modify: `scripts/build_pde_sbatch.py:22-66` (parser) and `:69-180` (render)
- Test: `tests/test_pde_profile.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pde_profile.py` inside `PdeProfileTests`:

```python
    def test_cascade_cli_renders_self_hosted_two_gpu_job(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "cascade",
                "--netid", "lding",
                "--repo-dir", "/local/scratch2/lding/MAS-Activation-Cascades",
                "--model", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                "--quantization", "awq_marlin",
                "--steering-vector", "steering_vectors/harmfulness_llama3_70b.pt",
            ]
        )

        self.assertIn("#SBATCH --gres=gpu:2", script)
        self.assertIn("bash ./scripts/run_cascade_2gpu.sh", script)
        self.assertIn("--quantization awq_marlin", script)
        self.assertIn("--steering-vector steering_vectors/harmfulness_llama3_70b.pt", script)
        self.assertIn("--clean-gpu 0", script)
        self.assertIn("--worker-gpu 1", script)
        # self-hosted: no external clean endpoint is required
        self.assertNotIn("--clean-api-base", script)

    def test_cascade_cli_rejects_70b_without_quantization(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantiz"):
            build_pde_sbatch.render_from_args(
                [
                    "cascade",
                    "--netid", "lding",
                    "--repo-dir", "/local/scratch2/lding/MAS-Activation-Cascades",
                    "--model", "meta-llama/Llama-3.1-70B-Instruct",
                    "--steering-vector", "steering_vectors/harmfulness_llama3_70b.pt",
                ]
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pde_profile.py -k cascade_cli -v`
Expected: FAIL — `argument command_name: invalid choice: 'cascade'`.

- [ ] **Step 3: Add the `cascade` subparser**

In `scripts/build_pde_sbatch.py`, inside `_build_parser`, after the `sweep_parser` block and before `return parser`:

```python
    cascade_parser = subparsers.add_parser(
        "cascade",
        help="Render a self-hosted 2-GPU cascade sweep (clean server + steered worker in one job)",
    )
    _add_common_args(cascade_parser)
    cascade_parser.add_argument("--job-name", default="cascade-2gpu")
    cascade_parser.add_argument("--model", default=PRIMARY_MODEL)
    cascade_parser.add_argument("--quantization", default=None, help="vLLM/HF quantization scheme, e.g. awq_marlin or fp8")
    cascade_parser.add_argument("--steering-vector", required=True)
    cascade_parser.add_argument("--experiments", default="1.2,1.3,1.4")
    cascade_parser.add_argument("--steering-strengths", default="1.0")
    cascade_parser.add_argument("--task-indices", default=None)
    cascade_parser.add_argument("--repeats", default="1")
    cascade_parser.add_argument("--port", default="8000")
    cascade_parser.add_argument("--max-model-len", default="4096")
    cascade_parser.add_argument("--max-new-tokens", default="256")
    cascade_parser.add_argument("--chat-turn-limit", default="2")
    cascade_parser.add_argument("--resume", action="store_true")
```

- [ ] **Step 4: Add the `cascade` render branch**

In `render_from_args`, before the final `raise ValueError`:

```python
    if args.command_name == "cascade":
        layout = build_pde_layout(
            model_name=args.model,
            mode="cascade",
            quantization=args.quantization,
        )
        command = [
            "bash",
            "./scripts/run_cascade_2gpu.sh",
            "--env-name", args.env_name,
            "--model", args.model,
            "--steering-vector", args.steering_vector,
            "--experiments", args.experiments,
            "--steering-strengths", args.steering_strengths,
            "--repeats", args.repeats,
            "--port", args.port,
            "--max-model-len", args.max_model_len,
            "--max-new-tokens", args.max_new_tokens,
            "--chat-turn-limit", args.chat_turn_limit,
            "--clean-gpu", layout.clean_server_gpu_set,
            "--worker-gpu", layout.worker_gpu_sets[0],
        ]
        if args.quantization:
            command += ["--quantization", args.quantization]
        if args.task_indices:
            command += ["--task-indices", args.task_indices]
        if args.resume:
            command += ["--resume"]
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=command,
            gpu_count=2,
            time_limit="08:00:00",
            mem="96G",
            cuda_visible_devices=None,
        )
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_pde_profile.py -v`
Expected: all PASS. (Note: `--clean-gpu 0` appears because `layout.clean_server_gpu_set == "0"`; `cuda_visible_devices=None` so the job does not pin a job-wide device.)

- [ ] **Step 6: Commit**

```bash
git add scripts/build_pde_sbatch.py tests/test_pde_profile.py
git commit -m "feat: add cascade subcommand for self-hosted 2-GPU quantized sweep"
```

---

## Task 4: Resumable per-cell checkpointing (CPU, TDD)

**Files:**
- Modify: `scripts/run_phase1_sweep.py`
- Test: `tests/test_phase1_sweep.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phase1_sweep.py` (top-level imports already expose `run_phase1_sweep`):

```python
import tempfile
from pathlib import Path as _Path
from unittest.mock import patch as _patch


class Phase1ResumeTests(unittest.TestCase):
    def _job(self, results_dir: str):
        return run_phase1_sweep.SweepJob(
            experiment="1.2",
            model="m",
            steering_vector="s.pt",
            steering_strength=1.0,
            task_names=None,
            task_indices=None,
            repeat_index=0,
            results_dir=results_dir,
            max_new_tokens=8,
            chat_turn_limit=1,
        )

    def test_cell_is_complete_detects_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(run_phase1_sweep.cell_is_complete(tmp))
            run_phase1_sweep.mark_cell_complete(tmp)
            self.assertTrue(run_phase1_sweep.cell_is_complete(tmp))

    def test_run_lane_skips_completed_cell_when_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_phase1_sweep.mark_cell_complete(tmp)
            lane = run_phase1_sweep.SweepLane(lane_id=0, clean_api_base="http://x/v1", worker_gpu_set="1")
            with _patch.object(run_phase1_sweep.subprocess, "run") as mock_run:
                run_phase1_sweep._run_lane(lane, [self._job(tmp)], resume=True)
            mock_run.assert_not_called()

    def test_run_lane_runs_and_marks_incomplete_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lane = run_phase1_sweep.SweepLane(lane_id=0, clean_api_base="http://x/v1", worker_gpu_set="1")
            with _patch.object(run_phase1_sweep.subprocess, "run") as mock_run:
                run_phase1_sweep._run_lane(lane, [self._job(tmp)], resume=True)
            mock_run.assert_called_once()
            self.assertTrue(run_phase1_sweep.cell_is_complete(tmp))
```

Note: confirm the real `SweepJob` field names by reading `src/experiments/sweep.py` before running; adjust the `_job` kwargs to match exactly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phase1_sweep.py -k Resume -v`
Expected: FAIL — `module 'run_phase1_sweep' has no attribute 'cell_is_complete'`.

- [ ] **Step 3: Implement sentinel helpers + resume in `_run_lane`**

In `scripts/run_phase1_sweep.py`, add after the imports:

```python
COMPLETION_SENTINEL = ".cell_complete"


def cell_is_complete(results_dir: str) -> bool:
    return (Path(results_dir) / COMPLETION_SENTINEL).exists()


def mark_cell_complete(results_dir: str) -> None:
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / COMPLETION_SENTINEL).write_text("ok", encoding="utf-8")
```

Replace `_run_lane`:

```python
def _run_lane(lane: SweepLane, jobs: Sequence[SweepJob], *, resume: bool = False) -> None:
    for job in jobs:
        if resume and cell_is_complete(job.results_dir):
            print(f"[lane {lane.lane_id}] skip (complete): {job.results_dir}")
            continue
        command, env = build_command(job, lane)
        print(f"[lane {lane.lane_id}] {' '.join(command)}")
        subprocess.run(command, check=True, cwd=ROOT, env=env)
        mark_cell_complete(job.results_dir)
```

- [ ] **Step 4: Wire `--resume` into the parser and `main`**

In `_build_parser`, add:

```python
    parser.add_argument("--resume", action="store_true", help="Skip cells whose completion sentinel exists")
```

In `main`, change the executor submission to pass `resume`:

```python
        futures = [
            executor.submit(_run_lane, lane, lane_jobs, resume=args.resume)
            for lane, lane_jobs in zip(lanes, sharded_jobs)
            if lane_jobs
        ]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_phase1_sweep.py -v`
Expected: all PASS (existing + new resume tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_phase1_sweep.py tests/test_phase1_sweep.py
git commit -m "feat: add resumable per-cell checkpointing to phase1 sweeps"
```

---

## Task 5: Quantization-aware steered backend load (CPU-TDD helper + GPU smoke)

**Files:**
- Modify: `src/backends/steering_backend.py:165-236`
- Test: `tests/test_steering_backend.py` (new)

- [ ] **Step 1: Write the failing unit test for the pure kwargs helper**

Create `tests/test_steering_backend.py`. It stubs `camel` + `transformers` so the module imports on CPU (mirrors the camel-stub pattern in `tests/test_star_topology_runner.py`):

```python
from __future__ import annotations

import sys
import types
import unittest

# Stub camel + transformers so steering_backend imports without heavy deps / GPU.
for name, attrs in {
    "camel.messages": {"OpenAIMessage": object},
    "camel.models": {"BaseModelBackend": object},
    "camel.types": {
        "ChatCompletion": object, "ChatCompletionMessage": object,
        "Choice": object, "CompletionUsage": object,
        "ChatCompletionChunk": object, "Stream": object, "AsyncStream": object,
    },
    "camel.utils": {"BaseTokenCounter": object},
    "transformers": {"AutoModelForCausalLM": object, "AutoTokenizer": object, "AutoConfig": object},
}.items():
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module
sys.modules.setdefault("camel", types.ModuleType("camel"))

import torch  # noqa: E402

from src.backends import steering_backend as sb  # noqa: E402


class BuildModelKwargsTests(unittest.TestCase):
    def test_cuda_unquantized_sets_dtype_and_device_map(self) -> None:
        kwargs = sb._build_model_kwargs(
            resolved_device="cuda", torch_dtype=torch.float16,
            is_quantized=False, trust_remote_code=False,
        )
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertEqual(kwargs["torch_dtype"], torch.float16)

    def test_cuda_quantized_omits_dtype(self) -> None:
        kwargs = sb._build_model_kwargs(
            resolved_device="cuda", torch_dtype=torch.float16,
            is_quantized=True, trust_remote_code=False,
        )
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertNotIn("torch_dtype", kwargs)

    def test_cpu_quantized_is_unsupported(self) -> None:
        with self.assertRaises(RuntimeError):
            sb._build_model_kwargs(
                resolved_device="cpu", torch_dtype=torch.float32,
                is_quantized=True, trust_remote_code=False,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_steering_backend.py -v`
Expected: FAIL — `module 'src.backends.steering_backend' has no attribute '_build_model_kwargs'`.

- [ ] **Step 3: Implement `_build_model_kwargs` and use it**

In `src/backends/steering_backend.py`, add `AutoConfig` to the transformers import:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
```

Add a module-level helper (near `_parse_dtype`):

```python
def _build_model_kwargs(
    *,
    resolved_device: str,
    torch_dtype: torch.dtype,
    is_quantized: bool,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if resolved_device == "cuda":
        kwargs["device_map"] = "auto"
        if not is_quantized:
            kwargs["torch_dtype"] = torch_dtype
    else:
        if is_quantized:
            raise RuntimeError(
                "Quantized steered models require CUDA; CPU load is unsupported."
            )
        kwargs["torch_dtype"] = torch.float32
    return kwargs
```

In `SteeringModelBackend.__init__`, replace the model-load block (currently `model_kwargs = {...}; ... AutoModelForCausalLM.from_pretrained(...)`) with:

```python
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        is_quantized = getattr(hf_config, "quantization_config", None) is not None
        model_kwargs = _build_model_kwargs(
            resolved_device=self.resolved_device,
            torch_dtype=self.torch_dtype,
            is_quantized=is_quantized,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.resolved_device != "cuda":
            self.model = self.model.to(self.resolved_device)
        self.model.eval()
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_steering_backend.py -v`
Expected: all PASS.

- [ ] **Step 5: GPU smoke (manual, on PDE)**

After Task 7 produces the 70B vector, this is exercised by the Task 9 pilot. No standalone GPU command here.

- [ ] **Step 6: Commit**

```bash
git add src/backends/steering_backend.py tests/test_steering_backend.py
git commit -m "feat: quantization-aware model load in SteeringModelBackend"
```

---

## Task 6: Combined 2-GPU orchestrator script (impl; rendered invocation already tested)

**Files:**
- Create: `scripts/run_cascade_2gpu.sh`

- [ ] **Step 1: Write the orchestrator**

Create `scripts/run_cascade_2gpu.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ENV_NAME="$DEFAULT_ENV_NAME"
MODEL_NAME=""
QUANTIZATION=""
STEERING_VECTOR=""
EXPERIMENTS="1.2,1.3,1.4"
STEERING_STRENGTHS="1.0"
TASK_INDICES=""
REPEATS="1"
PORT="8000"
MAX_MODEL_LEN="4096"
MAX_NEW_TOKENS="256"
CHAT_TURN_LIMIT="2"
CLEAN_GPU="0"
WORKER_GPU="1"
RESUME="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2;;
    --model) MODEL_NAME="$2"; shift 2;;
    --quantization) QUANTIZATION="$2"; shift 2;;
    --steering-vector) STEERING_VECTOR="$2"; shift 2;;
    --experiments) EXPERIMENTS="$2"; shift 2;;
    --steering-strengths) STEERING_STRENGTHS="$2"; shift 2;;
    --task-indices) TASK_INDICES="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --chat-turn-limit) CHAT_TURN_LIMIT="$2"; shift 2;;
    --clean-gpu) CLEAN_GPU="$2"; shift 2;;
    --worker-gpu) WORKER_GPU="$2"; shift 2;;
    --resume) RESUME="1"; shift;;
    *) fail "Unknown argument: $1";;
  esac
done

[[ -n "$MODEL_NAME" ]] || fail "--model is required"
[[ -n "$STEERING_VECTOR" ]] || fail "--steering-vector is required"

# 1. Background the clean vLLM server on the clean GPU (reuse the single launcher).
SERVE_ARGS=(--env-name "$ENV_NAME" --tensor-parallel-size 1 --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN")
if [[ -n "$QUANTIZATION" ]]; then
  SERVE_ARGS+=(--quantization "$QUANTIZATION")
fi
CUDA_VISIBLE_DEVICES="$CLEAN_GPU" bash "$(dirname "${BASH_SOURCE[0]}")/serve_clean_model.sh" "${SERVE_ARGS[@]}" "$MODEL_NAME" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# 2. Health-check the server (timeout ~300s).
echo "[cascade] waiting for clean server on :$PORT ..."
for _ in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[cascade] clean server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "clean vLLM server exited before becoming healthy"
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || fail "clean vLLM server did not become healthy in time"

# 3. Run the steered sweep on the worker GPU.
#    Do NOT restrict this launcher's CUDA_VISIBLE_DEVICES: run_phase1_sweep sets the
#    child's CUDA_VISIBLE_DEVICES from --worker-gpu-sets, and that index must be
#    valid against the full job allocation.
SWEEP_ARGS=(
  scripts/run_phase1_sweep.py
  --experiments "$EXPERIMENTS"
  --models "$MODEL_NAME"
  --steering-vector "$STEERING_VECTOR"
  --steering-strengths "$STEERING_STRENGTHS"
  --repeats "$REPEATS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --chat-turn-limit "$CHAT_TURN_LIMIT"
  --clean-api-bases "http://127.0.0.1:${PORT}/v1"
  --worker-gpu-sets "$WORKER_GPU"
)
if [[ -n "$TASK_INDICES" ]]; then
  SWEEP_ARGS+=(--task-indices "$TASK_INDICES")
fi
if [[ "$RESUME" == "1" ]]; then
  SWEEP_ARGS+=(--resume)
fi
python_in_conda "$ENV_NAME" "${SWEEP_ARGS[@]}"
```

Note: confirm `common.sh` exports `DEFAULT_ENV_NAME`, `fail`, and `python_in_conda` (it is sourced the same way by `serve_clean_model.sh`). If `python_in_conda` runs `python`, ensure it invokes `python scripts/run_phase1_sweep.py` form — adjust the first array element if `python_in_conda` already implies `python`.

- [ ] **Step 2: Lint the script**

Run: `bash -n scripts/run_cascade_2gpu.sh`
Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_cascade_2gpu.sh
git commit -m "feat: add combined 2-GPU cascade orchestrator script"
```

---

## Task 7: Compute the 70B steering vector (GPU, manual)

**Files:** none (uses existing `compute-vector` path + `src/steering/compute_vectors.py`).

- [ ] **Step 1: Render and submit a 1-GPU vector job on the quantized 70B**

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model <MODEL_FROM_TASK_1> \
  --output steering_vectors/harmfulness_llama3_70b.pt \
  --gpu-set 0 > pde-vector-70b.sbatch
sbatch pde-vector-70b.sbatch
```

- [ ] **Step 2: Verify the artifact shape**

After the job finishes, in a `conda run -n cascade python` shell on PDE:

```python
import torch
art = torch.load("steering_vectors/harmfulness_llama3_70b.pt", weights_only=True)
print(art["selected_layer"], art["hidden_size"])
assert art["hidden_size"] == 8192
assert 0 <= art["selected_layer"] < 80
print("VECTOR OK")
```

Expected: prints a layer index `< 80` and `8192`, then `VECTOR OK`. (If `compute_vectors` cannot extract activations from the quantized model, fall back to computing the vector from an FP8 load of the same family — the steered backend in Task 5 already loads quantized models for inference.)

- [ ] **Step 3: No commit** — steering vectors are git-ignored artifacts (see CLAUDE.md invariant). Do not commit `.pt` files.

---

## Task 8: Add AWQ kernel dependency to the environment (manual, scheme-dependent)

**Files:**
- Modify: `scripts/setup_env.sh` (only if Task 1 selected AWQ-INT4 and HF needs an explicit kernel package)

- [ ] **Step 1: Determine whether a kernel package is needed**

If Task 1's HF smoke (`SMOKE PASS`) succeeded with the stock `cascade` env, **skip this task** — no dependency change. FP8 typically needs no extra package. Only AWQ-INT4 may require `autoawq` kernels.

- [ ] **Step 2: If needed, pin the kernel package without disturbing torch**

Add to `scripts/setup_env.sh` (in the `pip_in_conda ... install` section), guarded so it never reinstalls torch:

```bash
pip_in_conda "$ENV_NAME" install --no-deps autoawq-kernels || \
  echo "[setup][warn] autoawq-kernels unavailable for sm_120; use FP8 for the 70B cascade instead."
```

`--no-deps` prevents autoawq from dragging in a conflicting torch/CUDA wheel (CLAUDE.md forbids changing the pinned `torch==2.11`/cu129/vLLM stack).

- [ ] **Step 3: Re-run the Task 1 HF smoke to confirm**

Resubmit `pde-smoke.sbatch`; expect `SMOKE PASS`.

- [ ] **Step 4: Commit (only if `setup_env.sh` changed)**

```bash
git add scripts/setup_env.sh
git commit -m "build: add awq kernels for steered 70B without disturbing torch stack"
```

---

## Task 9: Pilot cascade cell end-to-end (GPU, manual)

**Files:** none.

- [ ] **Step 1: Render and submit a 1-cell pilot**

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model <MODEL_FROM_TASK_1> --quantization <QUANT_FROM_TASK_1> \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2 --steering-strengths 1.0 --task-indices 0 \
  --resume > pde-cascade-pilot.sbatch
sbatch pde-cascade-pilot.sbatch
```

- [ ] **Step 2: Verify the job log and outputs**

Expected in the log: `[cascade] clean server ready`, then per-cell sweep lines, then a clean exit (server torn down).
Expected on disk under `results/sweeps/exp1_2/<model>/alpha_1p0/repeat_00/`: `*_baseline.json`, `*_attack.json`, `*_summary.json`, `report.txt`, and the `.cell_complete` sentinel.

- [ ] **Step 3: Confirm both agent roles ran**

Open `*_attack.json` and confirm there are messages attributed to the steered (attacker) agent **and** to clean downstream agents — i.e., the cascade interaction actually happened, not just a single agent.

- [ ] **Step 4: Confirm resume works**

Resubmit the same `pde-cascade-pilot.sbatch`. Expected: the sweep prints `skip (complete)` for the already-finished cell and does no new generation.

---

## Task 10: Document the cascade command + run the full sharded matrix

**Files:**
- Modify: `README.md`, `WORKFLOW.md`, `CLAUDE.md`, `AGENTS.md`, `docs/PDE_GPU_TEST_RUNBOOK.md`
- Test: `tests/test_pde_profile.py` (doc-consistency assertions)

- [ ] **Step 1: Write the failing doc-consistency test**

Add to `tests/test_pde_profile.py` inside `PdeProfileTests`:

```python
    def test_docs_document_quantized_70b_cascade_command(self) -> None:
        for relative_path in ("CLAUDE.md", "WORKFLOW.md", "docs/PDE_GPU_TEST_RUNBOOK.md"):
            with self.subTest(path=relative_path):
                document = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("build_pde_sbatch.py cascade", document)
                self.assertIn("--steering-vector steering_vectors/harmfulness_llama3_70b.pt", document)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_pde_profile.py -k document_quantized_70b_cascade -v`
Expected: FAIL (docs don't mention the command yet).

- [ ] **Step 3: Add a "70B Quantized Cascade Sweep" section to each doc**

Add this block to `CLAUDE.md`, `WORKFLOW.md`, and `docs/PDE_GPU_TEST_RUNBOOK.md` (place it near the existing 70B serving sections). Keep `README.md`/`AGENTS.md` updated with a one-line pointer to it:

````markdown
### 70B Quantized Cascade Sweep (one 2-GPU job)

Runs a true 70B cascade — clean server on GPU 0, steered worker on GPU 1, both a
single-GPU quantized 70B — in one self-hosted, resumable job:

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq_marlin \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2,1.3,1.4 --steering-strengths 0.5,1.0,1.5 \
  --resume > pde-cascade-70b.sbatch
sbatch pde-cascade-70b.sbatch
```

Resubmit the same script to resume; completed cells are skipped via the
`.cell_complete` sentinel. Unquantized 70B cascade is rejected — pass
`--quantization` so each model fits one 96 GB GPU.
````

- [ ] **Step 4: Run the full suite to verify all doc assertions pass**

Run: `pytest tests/ -W error`
Expected: all PASS, no warnings. (Confirm the pre-existing `test_storage_constrained_awq_docs_use_marlin_quantization` still passes — the new section uses `awq_marlin`, not the forbidden `--quantization awq > pde-vllm-70b.sbatch` form.)

- [ ] **Step 5: Commit the docs**

```bash
git add README.md WORKFLOW.md CLAUDE.md AGENTS.md docs/PDE_GPU_TEST_RUNBOOK.md tests/test_pde_profile.py
git commit -m "docs: document the quantized 70B 2-GPU cascade command"
```

- [ ] **Step 6: Launch the full sharded matrix (GPU, manual)**

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model <MODEL_FROM_TASK_1> --quantization <QUANT_FROM_TASK_1> \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2,1.3,1.4 --steering-strengths 0.5,1.0,1.5 \
  --task-indices 0,1,2,3,4 --resume > pde-cascade-70b.sbatch
sbatch pde-cascade-70b.sbatch
```

Resubmit on timeout/preemption; the `--resume` sentinel skips finished cells until the matrix is complete.

---

## Notes for the executor

- **Tasks 1, 7, 8, 9 and Step 6 of Task 10 require PDE GPUs** and cannot be verified in the CPU suite — run them on the cluster.
- **Tasks 2, 3, 4, 5, and Steps 1-5 of Task 10 are CPU-only and fully TDD'd** — verify locally/CI with `pytest tests/ -W error`.
- The AWQ-vs-FP8 choice (Task 1) only changes the `--model`/`--quantization` *values* you pass; no code hardcodes a scheme.
- Do not commit generated artifacts (`.pt` vectors, `results/`, `*.sbatch`) — CLAUDE.md invariant.

# GPTQ 70B Cascade Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and document a GPTQ-first path for running the quantized 70B cascade on PDE Blackwell GPUs under the 100 GB scratch cap.

**Architecture:** Keep the existing two-GPU cascade layout: clean vLLM server on GPU 0, steered HF Transformers worker on GPU 1, both using the same single-GPU GPTQ INT4 70B checkpoint. Add local tooling for reproducible Slurm smoke jobs and diagnostic logs, then run gated remote validation before switching docs from AWQ-first to GPTQ-first.

**Tech Stack:** Python 3, argparse, unittest/pytest, Hugging Face Transformers, Hugging Face Hub, torch, vLLM, Slurm sbatch, PDE scratch-local conda environment `cascade`.

---

## File Structure

- `scripts/build_pde_sbatch.py`: add the `smoke-steered-quant` Slurm renderer.
- `scripts/smoke_steered_quant.py`: print runtime/package diagnostics before loading the quantized model.
- `scripts/check_quant_model_metadata.py`: add a CPU metadata gate for the GPTQ checkpoint config.
- `tests/test_pde_profile.py`: cover the new sbatch renderer and GPTQ doc expectations.
- `tests/test_smoke_steered_quant.py`: cover smoke diagnostic helpers without loading a model.
- `tests/test_quant_model_metadata.py`: cover metadata validation without network or model downloads.
- `AGENTS.md`, `CLAUDE.md`, `WORKFLOW.md`, `PLAN.md`, `docs/PDE_GPU_TEST_RUNBOOK.md`: document GPTQ as the validation candidate first, then as the recommended path only after the pilot gate passes.

## Task 1: Add a First-Class Steered Quantization Smoke Sbatch Renderer

**Files:**
- Modify: `scripts/build_pde_sbatch.py`
- Modify: `tests/test_pde_profile.py`

- [ ] **Step 1: Write the failing renderer test**

Add this test method to `PdeProfileTests` in `tests/test_pde_profile.py`, near the existing `test_compute_vector_cli_renders_pde_gpu_job` test:

```python
    def test_smoke_steered_quant_cli_renders_one_gpu_job(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "smoke-steered-quant",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
                "--model",
                "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
                "--gpu-set",
                "0",
            ]
        )

        self.assertIn("#SBATCH --job-name=cascade-smoke-steered-quant", script)
        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", script)
        self.assertIn(
            "conda run -n cascade python scripts/smoke_steered_quant.py "
            "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
            script,
        )
        self.assertIn("export HF_HOME=/local/scratch2/lding/.cache/huggingface", script)
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py::PdeProfileTests::test_smoke_steered_quant_cli_renders_one_gpu_job -q
```

Expected: FAIL with argparse rejecting `smoke-steered-quant` as an invalid subcommand.

- [ ] **Step 3: Add the parser subcommand**

In `scripts/build_pde_sbatch.py`, add this parser block after the existing `compute-vector` parser block:

```python
    smoke_parser = subparsers.add_parser(
        "smoke-steered-quant",
        help="Render a 1-GPU HF Transformers smoke job for a quantized steered model",
    )
    _add_common_args(smoke_parser)
    smoke_parser.add_argument("--job-name", default="cascade-smoke-steered-quant")
    smoke_parser.add_argument("--model", required=True)
    smoke_parser.add_argument("--gpu-set", default=None)
```

- [ ] **Step 4: Add the render branch**

In `render_from_args` in `scripts/build_pde_sbatch.py`, add this branch after the `compute-vector` branch:

```python
    if args.command_name == "smoke-steered-quant":
        command = [
            "conda",
            "run",
            "-n",
            args.env_name,
            "python",
            "scripts/smoke_steered_quant.py",
            args.model,
        ]
        return render_sbatch_script(
            job_name=args.job_name,
            netid=args.netid,
            repo_dir=args.repo_dir,
            command=command,
            gpu_count=1,
            time_limit="02:00:00",
            mem="64G",
            cuda_visible_devices=args.gpu_set,
        )
```

- [ ] **Step 5: Run the renderer test**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py::PdeProfileTests::test_smoke_steered_quant_cli_renders_one_gpu_job -q
```

Expected: PASS.

- [ ] **Step 6: Run the PDE profile test file**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the renderer**

Run:

```bash
git add scripts/build_pde_sbatch.py tests/test_pde_profile.py
git commit -m "feat: render steered quantization smoke sbatch jobs"
```

## Task 2: Make the HF Smoke Script Diagnostic and CPU-Testable

**Files:**
- Modify: `scripts/smoke_steered_quant.py`
- Create: `tests/test_smoke_steered_quant.py`

- [ ] **Step 1: Write tests for package and CUDA diagnostics**

Create `tests/test_smoke_steered_quant.py`:

```python
from __future__ import annotations

import importlib.util
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import smoke_steered_quant as smoke


class SmokeSteeredQuantDiagnosticsTests(unittest.TestCase):
    def test_package_status_reports_present_and_missing_packages(self) -> None:
        def fake_find_spec(name: str):
            if name == "gptqmodel":
                return object()
            return None

        with patch.object(importlib.util, "find_spec", side_effect=fake_find_spec):
            status = smoke.package_status(["gptqmodel", "optimum"])

        self.assertEqual(status, {"gptqmodel": True, "optimum": False})

    def test_cuda_summary_reports_unavailable_cuda(self) -> None:
        with patch.object(smoke.torch.cuda, "is_available", return_value=False):
            self.assertEqual(smoke.cuda_summary(), "cuda_available=False")

    def test_cuda_summary_reports_device_capability(self) -> None:
        props = SimpleNamespace(name="NVIDIA RTX PRO 6000 Blackwell")
        with patch.object(smoke.torch.cuda, "is_available", return_value=True), patch.object(
            smoke.torch.cuda, "current_device", return_value=0
        ), patch.object(smoke.torch.cuda, "get_device_properties", return_value=props), patch.object(
            smoke.torch.cuda, "get_device_capability", return_value=(12, 0)
        ):
            summary = smoke.cuda_summary()

        self.assertIn("cuda_available=True", summary)
        self.assertIn("device=0", summary)
        self.assertIn("capability=sm_120", summary)
        self.assertIn("NVIDIA RTX PRO 6000 Blackwell", summary)

    def test_print_runtime_context_includes_model_and_packages(self) -> None:
        stream = io.StringIO()
        with patch.object(smoke, "cuda_summary", return_value="cuda_available=False"), patch.object(
            smoke, "package_status", return_value={"gptqmodel": True, "optimum": False}
        ), patch("sys.stdout", stream):
            smoke.print_runtime_context("hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4")

        output = stream.getvalue()
        self.assertIn("model: hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", output)
        self.assertIn("cuda: cuda_available=False", output)
        self.assertIn("package gptqmodel: present", output)
        self.assertIn("package optimum: missing", output)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
conda run -n cascade python -m pytest tests/test_smoke_steered_quant.py -q
```

Expected: FAIL with missing `package_status`, `cuda_summary`, or `print_runtime_context`.

- [ ] **Step 3: Refactor the smoke script imports and add diagnostics**

Replace `scripts/smoke_steered_quant.py` with:

```python
"""Phase 0 smoke test: prove a quantized 70B loads + runs + accepts a forward
hook under HF Transformers on Blackwell sm_120. Run inside a 1-GPU PDE job."""
from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable

import torch


def package_status(package_names: Iterable[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in package_names}


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    return (
        "cuda_available=True "
        f"device={device_index} "
        f"name={props.name} "
        f"capability=sm_{major}{minor}"
    )


def print_runtime_context(model_name: str) -> None:
    print(f"model: {model_name}")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda: {cuda_summary()}")
    for package_name, present in package_status(["gptqmodel", "optimum", "autoawq", "auto_awq"]).items():
        state = "present" if present else "missing"
        print(f"package {package_name}: {state}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_steered_quant.py <model_name_or_path>")
    model_name = sys.argv[1]
    print_runtime_context(model_name)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

- [ ] **Step 4: Run the smoke diagnostic tests**

Run:

```bash
conda run -n cascade python -m pytest tests/test_smoke_steered_quant.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the smoke script usage path**

Run:

```bash
conda run -n cascade python scripts/smoke_steered_quant.py
```

Expected: exits non-zero and prints:

```text
usage: smoke_steered_quant.py <model_name_or_path>
```

- [ ] **Step 6: Commit the smoke diagnostics**

Run:

```bash
git add scripts/smoke_steered_quant.py tests/test_smoke_steered_quant.py
git commit -m "feat: print quantized steering smoke diagnostics"
```

## Task 3: Add a CPU Metadata Gate for the GPTQ Checkpoint

**Files:**
- Create: `scripts/check_quant_model_metadata.py`
- Create: `tests/test_quant_model_metadata.py`

- [ ] **Step 1: Write metadata validation tests**

Create `tests/test_quant_model_metadata.py`:

```python
from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts import check_quant_model_metadata as check


class QuantModelMetadataTests(unittest.TestCase):
    def test_summarize_config_handles_dict_quantization_config(self) -> None:
        config = SimpleNamespace(
            quantization_config={"quant_method": "gptq", "bits": 4},
            hidden_size=8192,
            num_hidden_layers=80,
            model_type="llama",
        )

        summary = check.summarize_config(config)

        self.assertEqual(summary["quant_method"], "gptq")
        self.assertEqual(summary["bits"], 4)
        self.assertEqual(summary["hidden_size"], 8192)
        self.assertEqual(summary["num_hidden_layers"], 80)
        self.assertEqual(summary["model_type"], "llama")

    def test_validate_summary_accepts_expected_gptq_70b(self) -> None:
        summary = {
            "quant_method": "gptq",
            "bits": 4,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "model_type": "llama",
        }

        check.validate_summary(summary)

    def test_validate_summary_rejects_wrong_quantization(self) -> None:
        summary = {
            "quant_method": "awq",
            "bits": 4,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "model_type": "llama",
        }

        with self.assertRaisesRegex(ValueError, "quant_method"):
            check.validate_summary(summary)

    def test_validate_summary_rejects_wrong_shape(self) -> None:
        summary = {
            "quant_method": "gptq",
            "bits": 4,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "model_type": "llama",
        }

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            check.validate_summary(summary)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the metadata tests to verify they fail**

Run:

```bash
conda run -n cascade python -m pytest tests/test_quant_model_metadata.py -q
```

Expected: FAIL because `scripts/check_quant_model_metadata.py` does not exist.

- [ ] **Step 3: Create the metadata checker**

Create `scripts/check_quant_model_metadata.py`:

```python
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any


DEFAULT_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"


def _quantization_value(config: Any) -> Any:
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None:
        return {}
    if isinstance(quantization_config, Mapping):
        return dict(quantization_config)
    if hasattr(quantization_config, "to_dict"):
        return quantization_config.to_dict()
    return {
        key: getattr(quantization_config, key)
        for key in ("quant_method", "bits")
        if hasattr(quantization_config, key)
    }


def summarize_config(config: Any) -> dict[str, Any]:
    quantization = _quantization_value(config)
    return {
        "model_type": getattr(config, "model_type", None),
        "quant_method": quantization.get("quant_method"),
        "bits": quantization.get("bits"),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
    }


def validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("quant_method") != "gptq":
        raise ValueError(f"Expected quant_method 'gptq', got {summary.get('quant_method')!r}")
    if summary.get("bits") != 4:
        raise ValueError(f"Expected 4-bit GPTQ, got bits={summary.get('bits')!r}")
    if summary.get("hidden_size") != 8192:
        raise ValueError(f"Expected hidden_size 8192, got {summary.get('hidden_size')!r}")
    if summary.get("num_hidden_layers") != 80:
        raise ValueError(f"Expected num_hidden_layers 80, got {summary.get('num_hidden_layers')!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPTQ 70B Hugging Face config metadata")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model)
    summary = summarize_config(config)
    validate_summary(summary)
    print(json.dumps({"model": args.model, **summary}, indent=2, sort_keys=True))
    print("METADATA OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the metadata tests**

Run:

```bash
conda run -n cascade python -m pytest tests/test_quant_model_metadata.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the metadata checker usage path without downloading weights**

Run:

```bash
conda run -n cascade python scripts/check_quant_model_metadata.py --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
```

Expected: prints JSON containing `"quant_method": "gptq"`, `"hidden_size": 8192`, `"num_hidden_layers": 80`, and `METADATA OK`.

- [ ] **Step 6: Commit the metadata gate**

Run:

```bash
git add scripts/check_quant_model_metadata.py tests/test_quant_model_metadata.py
git commit -m "feat: validate GPTQ 70B checkpoint metadata"
```

## Task 4: Document GPTQ as the Candidate Path Before GPU Validation

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `WORKFLOW.md`
- Modify: `PLAN.md`
- Modify: `docs/PDE_GPU_TEST_RUNBOOK.md`
- Modify: `tests/test_pde_profile.py`

- [ ] **Step 1: Write failing doc assertions**

Replace `test_storage_constrained_awq_docs_use_marlin_quantization` in `tests/test_pde_profile.py` with:

```python
    def test_storage_constrained_docs_name_gptq_as_validation_candidate(self) -> None:
        for relative_path in ("AGENTS.md", "CLAUDE.md", "PLAN.md", "WORKFLOW.md", "docs/PDE_GPU_TEST_RUNBOOK.md"):
            with self.subTest(path=relative_path):
                document = (ROOT / relative_path).read_text(encoding="utf-8")

                self.assertIn("hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", document)
                self.assertIn("--quantization gptq_marlin", document)
                self.assertIn("100 GB", document)
                self.assertIn("GPTQ", document)
                self.assertNotIn("--quantization awq > pde-vllm-70b.sbatch", document)
```

- [ ] **Step 2: Run the doc assertion to verify it fails**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py::PdeProfileTests::test_storage_constrained_docs_name_gptq_as_validation_candidate -q
```

Expected: FAIL because at least one document still leads with AWQ.

- [ ] **Step 3: Update `AGENTS.md`**

In `AGENTS.md`, replace the AWQ storage-constrained serving example with:

````markdown
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
````

- [ ] **Step 4: Update `CLAUDE.md`, `WORKFLOW.md`, `PLAN.md`, and `docs/PDE_GPU_TEST_RUNBOOK.md`**

For each existing AWQ example, replace the model and quantization values:

```text
hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
--quantization awq_marlin
AWQ INT4
```

with:

```text
hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
--quantization gptq_marlin
GPTQ INT4 candidate
```

Add this sentence near each 70B quantized cascade section:

```markdown
This GPTQ path is the current validation candidate until the HF smoke, steering-vector, vLLM smoke, and pilot cascade Slurm gates pass; after those gates pass it becomes the recommended 70B cascade path.
```

Keep the existing 100 GB scratch-limit warning in `AGENTS.md` and `CLAUDE.md`.

- [ ] **Step 5: Run the doc assertion**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py::PdeProfileTests::test_storage_constrained_docs_name_gptq_as_validation_candidate -q
```

Expected: PASS.

- [ ] **Step 6: Run the full PDE profile tests**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit candidate docs**

Run:

```bash
git add AGENTS.md CLAUDE.md WORKFLOW.md PLAN.md docs/PDE_GPU_TEST_RUNBOOK.md tests/test_pde_profile.py
git commit -m "docs: make GPTQ the 70B cascade validation candidate"
```

## Task 5: Prepare the Scratch Checkout and GPTQ Loader Dependencies

**Files:**
- No local files modified unless the dependency outcome requires a later setup-script task.

- [ ] **Step 1: Push the local branch so PDE can pull it**

Run from the local worktree:

```bash
git status --short
git push origin comprehensive-quantization-testing-and-fixes
```

Expected: branch pushes successfully. `git status --short` contains no uncommitted changes except files intentionally held back by the user.

- [ ] **Step 2: Connect to the PDE login node**

Run:

```bash
ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu
```

Expected: shell prompt on `pdelogin`.

- [ ] **Step 3: Initialize conda for the ksh login shell**

Run on `pdelogin`:

```bash
source /local/scratch2/lding43/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/local/scratch2/lding43/.conda/envs
export CONDA_PKGS_DIRS=/local/scratch2/lding43/.conda/pkgs
export XDG_CACHE_HOME=/local/scratch2/lding43/.cache
export HF_HOME=/local/scratch2/lding43/.cache/huggingface
export TRANSFORMERS_CACHE=/local/scratch2/lding43/.cache/huggingface/transformers
export PIP_CACHE_DIR=/local/scratch2/lding43/.cache/pip
export TMPDIR=/local/scratch2/lding43/tmp
cd /local/scratch2/lding43/MAS-Activation-Cascades
git fetch origin
git checkout comprehensive-quantization-testing-and-fixes
git pull --ff-only origin comprehensive-quantization-testing-and-fixes
```

Expected: scratch checkout is on the pushed branch and fast-forwards cleanly.

- [ ] **Step 4: Record disk usage before downloads**

Run on `pdelogin`:

```bash
df -h /local/scratch2
du -sh /local/scratch2/lding43/.conda /local/scratch2/lding43/.cache/huggingface 2>/dev/null || true
```

Expected: enough free space remains for one ~38 GB GPTQ checkpoint plus logs/results. If free space is insufficient, list large model caches before deleting anything:

```bash
du -sh /local/scratch2/lding43/.cache/huggingface/hub/models--* 2>/dev/null | sort -h
```

- [ ] **Step 5: Run the GPTQ metadata gate**

Run on `pdelogin`:

```bash
conda run -n cascade python scripts/check_quant_model_metadata.py --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
```

Expected: prints `"quant_method": "gptq"`, `"hidden_size": 8192`, `"num_hidden_layers": 80`, and `METADATA OK`.

- [ ] **Step 6: Dry-run GPTQ loader dependency resolution**

Run on `pdelogin`:

```bash
conda run -n cascade python -m pip install --dry-run --report /local/scratch2/lding43/tmp/gptq-dry-run.json gptqmodel optimum
conda run -n cascade python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/local/scratch2/lding43/tmp/gptq-dry-run.json").read_text())
names = sorted(item["metadata"]["name"].lower() for item in report.get("install", []))
print("dry-run packages:", names)
blocked = {"torch", "torchvision", "torchaudio", "vllm", "xformers"}
overlap = blocked.intersection(names)
if overlap:
    raise SystemExit(f"dependency resolution wants to alter core stack: {sorted(overlap)}")
print("GPTQ DEP DRY RUN OK")
PY
```

Expected: prints `GPTQ DEP DRY RUN OK`. If it exits with a core-stack package name, do not install from that resolution.

- [ ] **Step 7: Install GPTQ loader dependencies without replacing the core stack**

If Step 6 printed `GPTQ DEP DRY RUN OK`, run:

```bash
conda run -n cascade python -m pip install gptqmodel optimum
```

If Step 6 reported a core-stack replacement, run this narrower install instead:

```bash
conda run -n cascade python -m pip install --no-deps gptqmodel optimum
```

Expected: command exits 0.

- [ ] **Step 8: Verify the core stack remains Blackwell-compatible**

Run on `pdelogin`:

```bash
conda run -n cascade python - <<'PY'
import importlib.metadata as md
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
for name in ("vllm", "gptqmodel", "optimum"):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, "missing")
if not torch.__version__.startswith("2.11.0"):
    raise SystemExit("torch version changed")
print("STACK OK")
PY
```

Expected: prints `STACK OK`, `torch 2.11.0...`, and installed GPTQ loader package versions.

## Task 6: Run the HF Smoke and Steering Vector Gates on Slurm

**Files:**
- No git-tracked files modified.
- Remote artifacts created under scratch: `pde-smoke-gptq.sbatch`, Slurm log files, `steering_vectors/harmfulness_llama3_70b.pt`, `steering_vectors/harmfulness_llama3_70b.analysis.pt`.

- [ ] **Step 1: Render the GPTQ HF smoke sbatch**

Run on `pdelogin`:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
python3 scripts/build_pde_sbatch.py smoke-steered-quant \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --gpu-set 0 > pde-smoke-gptq.sbatch
```

Expected: `pde-smoke-gptq.sbatch` contains `#SBATCH --gres=gpu:1`, `CUDA_VISIBLE_DEVICES=0`, and the GPTQ model name.

- [ ] **Step 2: Submit the HF smoke job**

Run on `pdelogin`:

```bash
sbatch pde-smoke-gptq.sbatch
```

Expected: prints a Slurm job id, for example `Submitted batch job 123456`.

- [ ] **Step 3: Inspect the HF smoke log**

Run on `pdelogin`, replacing `123456` with the submitted job id:

```bash
squeue -j 123456
tail -n 200 slurm-123456.out
```

Expected after completion: log contains `capability=sm_120`, `quantization_config:`, `forward ok`, `hooked hidden state shape:`, and `SMOKE PASS`.

- [ ] **Step 4: Render the GPTQ vector compute sbatch**

Run on `pdelogin`:

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --pairs-path data/contrastive_pairs/ta2_harmful_pairs.json \
  --output steering_vectors/harmfulness_llama3_70b.pt \
  --gpu-set 0 > pde-vector-gptq-70b.sbatch
```

Expected: `pde-vector-gptq-70b.sbatch` contains the GPTQ model and `CUDA_VISIBLE_DEVICES=0`.

- [ ] **Step 5: Submit and inspect the vector job**

Run on `pdelogin`:

```bash
sbatch pde-vector-gptq-70b.sbatch
```

After completion, inspect the log:

```bash
tail -n 200 slurm-123457.out
```

Expected: JSON output includes `"hidden_size": 8192`, `"output_path": "steering_vectors/harmfulness_llama3_70b.pt"`, and no traceback.

- [ ] **Step 6: Verify the vector artifact**

Run on `pdelogin`:

```bash
conda run -n cascade python - <<'PY'
from pathlib import Path
import torch

path = Path("steering_vectors/harmfulness_llama3_70b.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
print("model_name", payload["model_name"])
print("selected_layer", payload["selected_layer"])
print("hidden_size", payload["hidden_size"])
if payload["hidden_size"] != 8192:
    raise SystemExit("hidden_size mismatch")
if not 0 <= payload["selected_layer"] < 80:
    raise SystemExit("selected_layer out of range")
print("VECTOR OK")
PY
```

Expected: prints `VECTOR OK`.

## Task 7: Run the vLLM Clean-Server Smoke and Pilot Cascade Gates

**Files:**
- No git-tracked files modified.
- Remote artifacts created under scratch: `pde-serve-gptq-smoke.sbatch`, `pde-cascade-gptq-pilot.sbatch`, Slurm log files, pilot results under gitignored results directories.

- [ ] **Step 1: Render the GPTQ clean-server smoke sbatch**

Run on `pdelogin`:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin \
  --port 8000 \
  --max-model-len 4096 > pde-serve-gptq-smoke.sbatch
```

Expected: sbatch contains `--quantization gptq_marlin` and the GPTQ model name.

- [ ] **Step 2: Submit the clean-server smoke**

Run on `pdelogin`:

```bash
sbatch pde-serve-gptq-smoke.sbatch
```

Expected: prints a Slurm job id.

- [ ] **Step 3: Confirm the clean server reaches health**

Run on `pdelogin`, replacing `123458` with the submitted job id:

```bash
tail -f slurm-123458.out
```

Expected: log shows vLLM startup without CUDA kernel errors. From the allocated node, the server responds to `/health`; if the job script does not self-exit, cancel it after health is confirmed:

```bash
scancel 123458
```

- [ ] **Step 4: Render a one-cell GPTQ pilot cascade**

Run on `pdelogin`:

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2 \
  --steering-strengths 1.0 \
  --task-indices 0 \
  --repeats 1 \
  --max-new-tokens 64 \
  --chat-turn-limit 1 \
  --resume > pde-cascade-gptq-pilot.sbatch
```

Expected: sbatch contains `#SBATCH --gres=gpu:2`, `--clean-gpu 0`, `--worker-gpu 1`, `--quantization gptq_marlin`, and `--task-indices 0`.

- [ ] **Step 5: Submit the pilot cascade**

Run on `pdelogin`:

```bash
sbatch pde-cascade-gptq-pilot.sbatch
```

Expected: prints a Slurm job id.

- [ ] **Step 6: Verify the pilot cascade output**

Run on `pdelogin` after completion:

```bash
tail -n 200 slurm-123459.out
find results -name .cell_complete -print | tail -n 20
find results -name report.txt -print | tail -n 20
```

Expected: log shows the clean server became ready, the sweep ran one cell, at least one `.cell_complete` is present for the GPTQ pilot, and at least one `report.txt` is present.

## Task 8: Promote GPTQ to the Recommended Docs Path After the Pilot Passes

**Files:**
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `WORKFLOW.md`
- Modify: `PLAN.md`
- Modify: `docs/PDE_GPU_TEST_RUNBOOK.md`
- Modify: `tests/test_pde_profile.py`

- [ ] **Step 1: Tighten the doc test from candidate to recommended**

In `tests/test_pde_profile.py`, replace `test_storage_constrained_docs_name_gptq_as_validation_candidate` with:

```python
    def test_storage_constrained_docs_recommend_validated_gptq_path(self) -> None:
        for relative_path in ("AGENTS.md", "CLAUDE.md", "PLAN.md", "WORKFLOW.md", "docs/PDE_GPU_TEST_RUNBOOK.md"):
            with self.subTest(path=relative_path):
                document = (ROOT / relative_path).read_text(encoding="utf-8")

                self.assertIn("hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", document)
                self.assertIn("--quantization gptq_marlin", document)
                self.assertIn("recommended", document.lower())
                self.assertIn("100 GB", document)
                self.assertIn("FP8", document)
                self.assertNotIn("This GPTQ path is the current validation candidate", document)
                self.assertNotIn("--quantization awq > pde-vllm-70b.sbatch", document)
```

- [ ] **Step 2: Run the tightened doc test to verify it fails**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py::PdeProfileTests::test_storage_constrained_docs_recommend_validated_gptq_path -q
```

Expected: FAIL because the docs still say GPTQ is under validation.

- [ ] **Step 3: Update docs from candidate to recommended**

In each of `AGENTS.md`, `CLAUDE.md`, `WORKFLOW.md`, `PLAN.md`, and `docs/PDE_GPU_TEST_RUNBOOK.md`, replace:

```markdown
This GPTQ path is the current validation candidate until the HF smoke, steering-vector, vLLM smoke, and pilot cascade Slurm gates pass; after those gates pass it becomes the recommended 70B cascade path.
```

with:

```markdown
GPTQ INT4 is the recommended 70B cascade path for the current 100 GB scratch budget after passing the HF smoke, steering-vector, vLLM smoke, and pilot cascade Slurm gates on PDE Blackwell GPUs. AWQ remains experimental for HF steering on `sm_120`; FP8 remains too large for the current scratch budget.
```

- [ ] **Step 4: Run the tightened doc test**

Run:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py::PdeProfileTests::test_storage_constrained_docs_recommend_validated_gptq_path -q
```

Expected: PASS.

- [ ] **Step 5: Run all local tests**

Run:

```bash
conda run -n cascade python -m pytest tests/
```

Expected: PASS.

- [ ] **Step 6: Commit the final docs**

Run:

```bash
git add AGENTS.md CLAUDE.md WORKFLOW.md PLAN.md docs/PDE_GPU_TEST_RUNBOOK.md tests/test_pde_profile.py
git commit -m "docs: recommend validated GPTQ 70B cascade path"
```

## Task 9: Final Verification and Push

**Files:**
- No new file modifications expected.

- [ ] **Step 1: Check the final local status**

Run:

```bash
git status --short --branch
```

Expected: clean working tree on `comprehensive-quantization-testing-and-fixes`, except for user-owned files the user explicitly asked to keep uncommitted.

- [ ] **Step 2: Run the full CPU suite one final time**

Run:

```bash
conda run -n cascade python -m pytest tests/
```

Expected: PASS.

- [ ] **Step 3: Push the completed branch**

Run:

```bash
git push origin comprehensive-quantization-testing-and-fixes
```

Expected: push succeeds.

- [ ] **Step 4: Report the evidence**

Use the actual Slurm job ids and log filenames printed during Tasks 6 and 7. The final report should include these exact fields:

```text
CPU tests: conda run -n cascade python -m pytest tests/ -> PASS
HF GPTQ smoke Slurm job id and log filename -> SMOKE PASS
Vector job id and log filename -> VECTOR OK
vLLM GPTQ smoke job id and log filename -> /health OK
Pilot cascade job id and log filename -> report.txt and .cell_complete present
Recommended model: hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
Recommended vLLM quantization: gptq_marlin
```

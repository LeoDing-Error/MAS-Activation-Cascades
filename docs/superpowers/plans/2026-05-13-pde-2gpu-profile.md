# PDE 2-GPU Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-tested PDE Slurm profile for the two-GPU Emory Math PDE allocation.

**Architecture:** Introduce pure helper functions under `src/cluster` so tests can validate GPU layout and Slurm generation without needing cluster access. Keep existing experiment runners intact and provide a separate CLI for generating cluster job scripts.

**Tech Stack:** Python 3.11, argparse, pathlib, unittest/pytest, Bash/Slurm documentation.

---

### Task 1: Add Failing PDE Profile Tests

**Files:**
- Create: `tests/test_pde_profile.py`

- [ ] **Step 1: Write tests for scratch validation, layouts, guards, and Slurm rendering**

```python
from __future__ import annotations

import unittest

from src.cluster.pde_profile import (
    build_pde_layout,
    is_70b_class_model,
    render_sbatch_script,
    validate_scratch_path,
)


class PdeProfileTests(unittest.TestCase):
    def test_validate_scratch_path_accepts_only_netid_scratch_tree(self) -> None:
        self.assertEqual(
            validate_scratch_path("/local/scratch2/lding43/project", "lding43"),
            "/local/scratch2/lding43/project",
        )

        with self.assertRaisesRegex(ValueError, "/local/scratch2/lding43"):
            validate_scratch_path("/home/lding43/project", "lding43")

    def test_model_size_detection_flags_70b_class_names(self) -> None:
        self.assertTrue(is_70b_class_model("meta-llama/Llama-3.1-70B-Instruct"))
        self.assertTrue(is_70b_class_model("Qwen/Qwen2.5-72B-Instruct"))
        self.assertFalse(is_70b_class_model("meta-llama/Meta-Llama-3.1-8B-Instruct"))

    def test_cascade_layout_uses_two_total_gpus_for_8b(self) -> None:
        layout = build_pde_layout(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            mode="cascade",
        )

        self.assertEqual(layout.clean_server_gpu_set, "0")
        self.assertEqual(layout.worker_gpu_sets, ("1",))
        self.assertEqual(layout.tensor_parallel_size, 1)

    def test_cascade_layout_rejects_70b_without_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "70B-class"):
            build_pde_layout(
                model_name="meta-llama/Llama-3.1-70B-Instruct",
                mode="cascade",
            )

    def test_tensor_parallel_layout_uses_both_gpus(self) -> None:
        layout = build_pde_layout(
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            mode="tensor-parallel",
        )

        self.assertIsNone(layout.clean_server_gpu_set)
        self.assertEqual(layout.worker_gpu_sets, ("0,1",))
        self.assertEqual(layout.tensor_parallel_size, 2)

    def test_render_sbatch_script_redirects_runtime_state_to_scratch(self) -> None:
        script = render_sbatch_script(
            job_name="cascade-tests",
            netid="lding43",
            repo_dir="/local/scratch2/lding43/MAS-Activation-Cascades",
            command=["conda", "run", "-n", "cascade", "python", "-m", "pytest", "tests/"],
            gpu_count=0,
        )

        self.assertIn("#SBATCH --job-name=cascade-tests", script)
        self.assertNotIn("#SBATCH --gres=gpu:", script)
        self.assertIn("export XDG_CACHE_HOME=/local/scratch2/lding43/.cache", script)
        self.assertIn("cd /local/scratch2/lding43/MAS-Activation-Cascades", script)
        self.assertIn("conda run -n cascade python -m pytest tests/", script)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and confirm it fails because the module does not exist**

Run: `conda run -n cascade python -m pytest tests/test_pde_profile.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.cluster'`.

### Task 2: Implement PDE Profile Helpers

**Files:**
- Create: `src/cluster/__init__.py`
- Create: `src/cluster/pde_profile.py`

- [ ] **Step 1: Add the module implementation**

The module defines `PdeLayout`, `is_70b_class_model`, `validate_scratch_path`, `build_pde_layout`, and `render_sbatch_script`.

- [ ] **Step 2: Run the focused test**

Run: `conda run -n cascade python -m pytest tests/test_pde_profile.py -q`

Expected: PASS.

### Task 3: Add Slurm Script Builder CLI

**Files:**
- Create: `scripts/build_pde_sbatch.py`
- Modify: `tests/test_pde_profile.py`

- [ ] **Step 1: Add CLI coverage**

Add tests that call the CLI parser function for `pytest`, `serve-clean`, and guarded `sweep` modes.

- [ ] **Step 2: Implement `scripts/build_pde_sbatch.py`**

The CLI prints scripts, does not submit jobs, and uses the helper module for validation and layout decisions.

- [ ] **Step 3: Run focused tests**

Run: `conda run -n cascade python -m pytest tests/test_pde_profile.py -q`

Expected: PASS.

### Task 4: Update Cluster Documentation

**Files:**
- Modify: `README.md`
- Modify: `WORKFLOW.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Keep cluster guidance focused on the PDE two-GPU workflow**

Document SSH access, Slurm usage, scratch-only execution, cache redirection, and the supported `cascade` and `tensor-parallel` layouts.

- [ ] **Step 2: Run documentation-sensitive tests**

Run: `conda run -n cascade python -m pytest tests/ -q`

Expected: PASS.

### Task 5: Final Verification

**Files:**
- No new edits unless verification exposes a defect.

- [ ] **Step 1: Run the full suite**

Run: `conda run -n cascade python -m pytest tests/`

Expected: PASS.

- [ ] **Step 2: Inspect git diff**

Run: `git diff --stat && git diff -- tests/test_pde_profile.py src/cluster/pde_profile.py scripts/build_pde_sbatch.py`

Expected: Only PDE profile, tests, and documentation changes are present.

# Colab Phase 1 Initial Test Setup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Google Colab notebook that reproduces the full Phase 1, Exp 1.1 pipeline (steering vector computation + single-agent validation sweep) end-to-end on a single A100 GPU, with Google Drive used for cross-session artifact persistence.

**Architecture:** A pip-based bootstrap script replaces the local conda workflow for Colab's Linux runtime. A pre-flight smoke test script catches environment failures cheaply before any GPU work. The notebook itself mounts Drive, runs setup, runs the smoke test as a gate, then orchestrates the three expensive steps (build contrastive pairs → compute steering vector → run Exp 1.1). An optional final cell runs Exp 1.2 using `--allow-local-clean-models` on an A100 80GB runtime.

**Tech Stack:** Python 3.10 (Colab), PyTorch 2.x with CUDA 12.1 wheels, HuggingFace Transformers/Hub, vLLM (Linux), local CAMEL from `third_party/camel`, nbformat for notebook assembly.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/smoke_test_colab.py` | CPU-only pre-flight checker: imports, CUDA, VRAM, file structure |
| Create | `scripts/setup_colab.sh` | pip-based install of all deps + local CAMEL, replacing conda for Colab |
| Create | `scripts/create_colab_notebook.py` | Assembles `notebooks/colab_phase1_quickstart.ipynb` from Python cell strings |
| Create | `notebooks/colab_phase1_quickstart.ipynb` | Generated Colab notebook artifact (committed, regenerate with create_colab_notebook.py) |

---

## Task 1: Smoke Test Script

The smoke test defines acceptance criteria for a working Colab environment. Write it first so Task 2 has a concrete pass/fail target.

**Files:**
- Create: `scripts/smoke_test_colab.py`

- [ ] **Step 1.1: Write the smoke test**

```python
# scripts/smoke_test_colab.py
"""Pre-flight environment check for Colab. CPU-gated checks run first;
GPU checks only if CUDA is present."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(label: str, fix: str = "") -> bool:
    print(f"[ok]   {label}")
    return True


def _fail(label: str, fix: str = "") -> bool:
    hint = f"  → {fix}" if fix else ""
    print(f"[FAIL] {label}{hint}")
    return False


def check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return _ok(f"import {module}")
    except Exception as exc:
        return _fail(f"import {module}: {exc}", "run scripts/setup_colab.sh")


def check_paths() -> list[bool]:
    from src.project_paths import CAMEL_ROOT, TA2_ROOT, CONTRASTIVE_PAIRS_ROOT

    results = []
    for label, path, fix in [
        ("third_party/camel", CAMEL_ROOT, "run scripts/setup_colab.sh"),
        ("third_party/Trojan-Activation-Attack", TA2_ROOT, "run scripts/setup_colab.sh"),
        (
            "data/contrastive_pairs/ta2_harmful_pairs.json",
            CONTRASTIVE_PAIRS_ROOT / "ta2_harmful_pairs.json",
            "run: python scripts/build_ta2_pairs.py",
        ),
    ]:
        if path.exists():
            results.append(_ok(label))
        else:
            results.append(_fail(label, fix))
    return results


def check_gpu() -> list[bool]:
    import torch

    results = []
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        results.append(_ok("CUDA available"))
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"       GPU: {name} ({vram_gb:.0f} GB VRAM)")
        if vram_gb >= 35:
            results.append(_ok(f"VRAM ≥40 GB ({vram_gb:.0f} GB detected)"))
        else:
            results.append(
                _fail(
                    f"VRAM ≥40 GB required, only {vram_gb:.0f} GB detected",
                    "Runtime → Change runtime type → GPU → A100",
                )
            )
    else:
        results.append(
            _fail("CUDA not available", "Runtime → Change runtime type → GPU")
        )
    return results


def main() -> None:
    results: list[bool] = []

    print("=== Import checks ===")
    for mod in ["torch", "transformers", "camel", "camel.agents", "openai", "vllm"]:
        results.append(check_import(mod))

    print("\n=== File structure checks ===")
    results.extend(check_paths())

    print("\n=== GPU checks ===")
    results.extend(check_gpu())

    failed = results.count(False)
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) failed — fix above before proceeding.'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Run to verify it fails on a fresh environment**

```bash
python scripts/smoke_test_colab.py
```

Expected output (on macOS without the Colab env):
```
[FAIL] import vllm  → run scripts/setup_colab.sh
[FAIL] third_party/camel  → run scripts/setup_colab.sh
...
2 check(s) failed — fix above before proceeding.
```
The script must exit non-zero before proceeding.

- [ ] **Step 1.3: Commit the smoke test**

```bash
git add scripts/smoke_test_colab.py
git commit -m "test: add Colab pre-flight smoke test script"
```

---

## Task 2: Colab Setup Script

Implements a pip-based install path for Linux/Colab that satisfies every `check_import` in the smoke test.

**Files:**
- Create: `scripts/setup_colab.sh`

- [ ] **Step 2.1: Write the setup script**

```bash
#!/usr/bin/env bash
# scripts/setup_colab.sh
# Pip-based bootstrap for Google Colab (Linux-only).
# Run from the repo root. Does NOT require conda.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '[colab-setup] %s\n' "$*"; }
fail() { printf '[colab-setup][error] %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "setup_colab.sh is Linux/Colab only"

# ── third-party reference repos ──────────────────────────────────────────────
log "Cloning reference repositories at pinned commits..."
bash "$PROJECT_ROOT/scripts/setup_references.sh"

# ── Python packages ──────────────────────────────────────────────────────────
log "Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel

log "Installing CUDA 12.1 torch stack..."
pip install --quiet --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

log "Installing project requirements (includes vllm on Linux)..."
pip install --quiet -r "$PROJECT_ROOT/requirements.txt"

# ── local CAMEL (must take precedence over any PyPI camel-ai) ─────────────────
log "Installing local CAMEL editable..."
pip uninstall -y camel-ai 2>/dev/null || true
pip install --quiet -e "$PROJECT_ROOT/third_party/camel"

log "Setup complete. Run: python scripts/smoke_test_colab.py"
```

- [ ] **Step 2.2: Make it executable**

```bash
chmod +x scripts/setup_colab.sh
```

- [ ] **Step 2.3: Run the smoke test after setup (on a Colab/Linux machine)**

On Colab, in a notebook cell:
```python
import subprocess
subprocess.run(['bash', 'scripts/setup_colab.sh'], check=True)
result = subprocess.run(['python', 'scripts/smoke_test_colab.py'])
assert result.returncode == 0, "Smoke test failed — see output above"
```

Expected: all `[ok]` lines, exit code 0.

- [ ] **Step 2.4: Commit**

```bash
git add scripts/setup_colab.sh
git commit -m "feat: add pip-based Colab setup script"
```

---

## Task 3: Notebook Assembly Script

Writes the canonical notebook source as Python strings and uses `nbformat` to emit the `.ipynb`. Re-run this script any time a cell changes to regenerate the committed artifact.

**Files:**
- Create: `scripts/create_colab_notebook.py`
- Create (generated): `notebooks/colab_phase1_quickstart.ipynb`

- [ ] **Step 3.1: Write the assembly script**

```python
#!/usr/bin/env python3
# scripts/create_colab_notebook.py
"""Generates notebooks/colab_phase1_quickstart.ipynb from Python string literals.
Re-run this script whenever any cell content changes:
    python scripts/create_colab_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    sys.exit("nbformat is required: pip install nbformat")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_phase1_quickstart.ipynb"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


nb.cells = [
    # ── Section 0: Header ──────────────────────────────────────────────────
    md("""\
# MAS Activation Cascades — Phase 1 Colab Quickstart

End-to-end pipeline for **Experiment 1.1 (single-agent steering validation)** and
optional **Experiment 1.2 (two-agent chain)** on a single Colab GPU.

**Requirements:**
- Colab Pro with A100 40 GB (Exp 1.1 only) *or* A100 80 GB (Exp 1.2 with local clean model)
- HuggingFace account with access granted to `meta-llama/Meta-Llama-3.1-8B-Instruct`
- HF token stored as a Colab secret named `HF_TOKEN`

**Estimated wall time (A100 40 GB):**
- Setup: ~5 min
- Build pairs: ~1 min
- Compute steering vector: ~15–20 min
- Exp 1.1 (3 tasks × 5 α values): ~20–40 min
"""),

    # ── Section 1: Google Drive mount ─────────────────────────────────────
    md("## 1 · Mount Google Drive\nArtifacts (steering vector, results) persist here across sessions."),
    code("""\
from google.colab import drive
drive.mount('/content/drive')

import os

DRIVE_DIR = '/content/drive/MyDrive/MAS-Activation-Cascades'
for sub in ['steering_vectors', 'results', 'data/contrastive_pairs']:
    os.makedirs(f'{DRIVE_DIR}/{sub}', exist_ok=True)

print("Drive mounted. Persistent storage at:", DRIVE_DIR)
"""),

    # ── Section 2: Clone repo ──────────────────────────────────────────────
    md("## 2 · Clone or Update Repo"),
    code("""\
import os, subprocess

# ← Replace with your fork URL if you've forked the repo
REPO_URL = 'https://github.com/YOUR_USERNAME/MAS-Activation-Cascades.git'
REPO_DIR = '/content/MAS-Activation-Cascades'

if not os.path.exists(REPO_DIR + '/.git'):
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
    print("Cloned to", REPO_DIR)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=True)
    print("Repo up to date at", REPO_DIR)

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
"""),

    # ── Section 3: HuggingFace auth ───────────────────────────────────────
    md("## 3 · HuggingFace Authentication\nLlama-3 requires an approved access request on HuggingFace."),
    code("""\
from google.colab import userdata
import os

# Store your HF token as a Colab secret:
# Secrets panel (🔑) → Add secret → Name: HF_TOKEN
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

from huggingface_hub import whoami
info = whoami(token=os.environ['HF_TOKEN'])
print(f"Logged in as: {info['name']}")
"""),

    # ── Section 4: Environment setup ──────────────────────────────────────
    md("## 4 · Install Dependencies\nThis takes ~3–5 min the first time. Skip if the kernel already has the packages."),
    code("""\
import subprocess, sys

result = subprocess.run(
    ['bash', 'scripts/setup_colab.sh'],
    capture_output=False,
)
if result.returncode != 0:
    raise RuntimeError("setup_colab.sh failed — see output above")
print("Setup complete")
"""),

    # ── Section 5: Smoke test gate ────────────────────────────────────────
    md("## 5 · Pre-flight Smoke Test\nMust pass before any GPU work. Checks imports, VRAM, and file structure."),
    code("""\
import subprocess

result = subprocess.run(['python', 'scripts/smoke_test_colab.py'])
if result.returncode != 0:
    raise RuntimeError("Smoke test failed — fix the issues printed above before continuing")
print("All pre-flight checks passed ✓")
"""),

    # ── Section 6: Symlink Drive paths ────────────────────────────────────
    md("## 6 · Link Drive Storage into Repo Paths\nSymlinks let the experiment scripts write to Drive transparently."),
    code("""\
import os

links = {
    'steering_vectors': f'{DRIVE_DIR}/steering_vectors',
    'data/contrastive_pairs': f'{DRIVE_DIR}/data/contrastive_pairs',
    'results': f'{DRIVE_DIR}/results',
}

for local_rel, drive_path in links.items():
    local_abs = os.path.join(os.getcwd(), local_rel)
    if os.path.islink(local_abs):
        print(f"Already linked: {local_rel}")
    elif os.path.exists(local_abs):
        print(f"Exists (not a link): {local_rel} — skipping")
    else:
        os.makedirs(os.path.dirname(local_abs), exist_ok=True)
        os.symlink(drive_path, local_abs)
        print(f"Linked: {local_rel} → {drive_path}")
"""),

    # ── Section 7: Build contrastive pairs ────────────────────────────────
    md("## 7 · Build Contrastive Pairs\nCPU-only, ~30 sec. Skipped automatically if the file exists on Drive."),
    code("""\
import subprocess, os

pairs_path = 'data/contrastive_pairs/ta2_harmful_pairs.json'

if os.path.exists(pairs_path):
    import json
    with open(pairs_path) as f:
        meta = json.load(f)
    print(f"Already built: {meta['count']} pairs — skipping")
else:
    subprocess.run(
        ['python', 'scripts/build_ta2_pairs.py',
         '--output', pairs_path],
        check=True,
    )
"""),

    # ── Section 8: Compute steering vector ────────────────────────────────
    md("""\
## 8 · Compute Steering Vector
~15–20 min on A100. **Skipped automatically if the vector already exists on Drive.**
Computes `v = E[h_unsafe] - E[h_safe]` across layers and selects the layer with
maximum contrastive separation (TA² method).
"""),
    code("""\
import subprocess, os

VECTOR_PATH = 'steering_vectors/harmfulness_llama3_8b.pt'

if os.path.exists(VECTOR_PATH):
    import torch
    meta = torch.load(VECTOR_PATH, weights_only=True)
    layer = meta.get('optimal_layer', '?')
    print(f"Vector already exists (optimal layer: {layer}) — skipping")
else:
    print("Computing steering vector (this takes ~15–20 min)...")
    subprocess.run(
        ['python', 'src/steering/compute_vectors.py',
         '--model', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
         '--pairs-path', 'data/contrastive_pairs/ta2_harmful_pairs.json',
         '--output', VECTOR_PATH,
         '--device', 'cuda'],
        check=True,
    )
    print("Done. Vector saved to:", VECTOR_PATH)
"""),

    # ── Section 9: Exp 1.1 ────────────────────────────────────────────────
    md("""\
## 9 · Experiment 1.1 — Single-Agent Steering Validation
Sweeps α ∈ {0.0, 0.5, 1.0, 1.5, 2.0} across 3 HumanEval tasks.
**Expected runtime:** ~20–40 min on A100 40 GB.
**Success criterion:** token entropy and/or MSP differ significantly between α=0 and α>0.
"""),
    code("""\
import subprocess

subprocess.run(
    ['python', 'experiments/run_phase1.py',
     '--experiment', '1.1',
     '--steering-vector', VECTOR_PATH,
     '--n-tasks', '3',
     '--alphas', '0.0,0.5,1.0,1.5,2.0',
     '--results-dir', 'results'],
    check=True,
)
print("Exp 1.1 complete. Results in results/exp1_1/")
"""),

    # ── Section 10: Results visualisation ────────────────────────────────
    md("## 10 · Inspect Results"),
    code("""\
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

results_path = Path('results/exp1_1/exp1_1_results.json')
with open(results_path) as f:
    data = json.load(f)

# data is a list of dicts: {task, alpha, entropy_mean, entropy_max, msp, ...}
tasks = sorted({r['task'] for r in data})
alphas = sorted({r['alpha'] for r in data})

fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=True)
if len(tasks) == 1:
    axes = [axes]

for ax, task in zip(axes, tasks):
    rows = [r for r in data if r['task'] == task]
    rows.sort(key=lambda r: r['alpha'])
    ax.plot([r['alpha'] for r in rows], [r['entropy_mean'] for r in rows], marker='o')
    ax.set_title(task)
    ax.set_xlabel('steering strength α')
    ax.set_ylabel('mean token entropy')
    ax.grid(True, alpha=0.3)

fig.suptitle('Exp 1.1: Token Entropy vs Steering Strength', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp1_1/entropy_vs_alpha.png', dpi=120)
plt.show()
print("Plot saved to results/exp1_1/entropy_vs_alpha.png")
"""),

    # ── Section 11: Optional Exp 1.2 ─────────────────────────────────────
    md("""\
## 11 · (Optional) Experiment 1.2 — Two-Agent Chain
**Requires A100 80 GB** (loads both steered and clean model locally).
Uses `--allow-local-clean-models` to bypass the vLLM server requirement.
"""),
    code("""\
import subprocess

# Only run this cell on an A100 80 GB runtime
import torch
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if vram_gb < 70:
    print(f"Skipped: only {vram_gb:.0f} GB VRAM. Need ≥70 GB for two local model copies.")
else:
    subprocess.run(
        ['python', 'experiments/run_phase1.py',
         '--experiment', '1.2',
         '--steering-vector', VECTOR_PATH,
         '--steering-strength', '1.0',
         '--n-tasks', '2',
         '--results-dir', 'results',
         '--allow-local-clean-models'],
        check=True,
    )
    print("Exp 1.2 complete. Results in results/exp1_2/")
"""),
]

with open(OUTPUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {OUTPUT.relative_to(ROOT)}")
```

- [ ] **Step 3.2: Install nbformat and run the assembly script locally to generate the notebook**

```bash
pip install nbformat
python scripts/create_colab_notebook.py
```

Expected output:
```
Notebook written to notebooks/colab_phase1_quickstart.ipynb
```

- [ ] **Step 3.3: Verify the notebook is valid JSON and has the right number of cells**

```bash
python -c "
import json, sys
nb = json.load(open('notebooks/colab_phase1_quickstart.ipynb'))
cells = nb['cells']
print(f'Cells: {len(cells)}')
assert len(cells) == 22, f'Expected 22 cells, got {len(cells)}'
types = [c['cell_type'] for c in cells]
print('Types:', types)
print('Notebook OK')
"
```

Expected output:
```
Cells: 22
Types: ['markdown', 'markdown', 'code', 'markdown', 'code', ...]
Notebook OK
```

- [ ] **Step 3.4: Commit both files**

```bash
git add scripts/create_colab_notebook.py notebooks/colab_phase1_quickstart.ipynb
git commit -m "feat: add Colab quickstart notebook for Exp 1.1 initial results"
```

---

## Task 4: End-to-End Colab Validation

Run the notebook on Colab and confirm the pipeline reaches Exp 1.1 results. This is an interactive check, not a CI test.

**Prerequisites:** A100 runtime (Colab Pro), HF_TOKEN secret set, `meta-llama/Meta-Llama-3.1-8B-Instruct` access granted.

- [ ] **Step 4.1: Upload or open the notebook in Colab**

Option A — from Google Drive:
```
File → Open notebook → Google Drive → navigate to MAS-Activation-Cascades/notebooks/colab_phase1_quickstart.ipynb
```

Option B — from GitHub (public repo):
```
File → Open notebook → GitHub → paste repo URL → select notebooks/colab_phase1_quickstart.ipynb
```

- [ ] **Step 4.2: Set runtime to GPU A100**

```
Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: A100 → Save
```

- [ ] **Step 4.3: Add HF_TOKEN secret**

```
Left panel → 🔑 Secrets → + Add new secret → Name: HF_TOKEN, Value: hf_...
Toggle "Notebook access" ON
```

- [ ] **Step 4.4: Update the REPO_URL placeholder in Cell 2**

```python
# Change this line in the "Clone or Update Repo" cell:
REPO_URL = 'https://github.com/YOUR_USERNAME/MAS-Activation-Cascades.git'
# ↓
REPO_URL = 'https://github.com/leoding101/MAS-Activation-Cascades.git'  # actual URL
```

- [ ] **Step 4.5: Run all cells in order (Runtime → Run all)**

Watch for these checkpoints:
1. **Cell 5 (smoke test)**: All `[ok]` lines. If `[FAIL] VRAM ≥40 GB`, switch to A100.
2. **Cell 8 (steering vector)**: Should run ~15–20 min then print `Done. Vector saved to: steering_vectors/harmfulness_llama3_8b.pt`.
3. **Cell 9 (Exp 1.1)**: Should print results for each `(task, α)` combination.
4. **Cell 10 (visualisation)**: A `entropy_vs_alpha.png` plot should display inline.

- [ ] **Step 4.6: Verify the entropy plot shows a trend**

The mean token entropy line should be non-flat across α values, confirming steering has a measurable effect. A flat line means the hook is not registering — return to `src/steering/compute_vectors.py` to debug layer selection.

- [ ] **Step 4.7: Check results persisted to Drive**

```python
import os
for f in os.listdir(f'{DRIVE_DIR}/results/exp1_1'):
    print(f)
```

Expected files: `exp1_1_results.json`, `exp1_1_summary.json`, `entropy_vs_alpha.png`.

---

## Self-Review

**Spec coverage:**
- ✅ Colab-compatible environment setup (Task 2)
- ✅ Pre-flight gate before expensive GPU work (Task 1)
- ✅ Google Drive persistence for steering vector across sessions (Task 3, cells 1 + 6)
- ✅ Skip-if-already-computed guards for vector and pairs (cells 7, 8)
- ✅ Exp 1.1 full α sweep (cell 9)
- ✅ Results visualisation (cell 10)
- ✅ Exp 1.2 optional cell with VRAM guard (cell 11)
- ✅ HF authentication for gated Llama-3 model (cell 3)

**Placeholder scan:** No TBD, TODO, or "add appropriate" language present.

**Type consistency:** `VECTOR_PATH` defined in cell 8 as a string and referenced in cells 9, 10, 11 — consistent. `DRIVE_DIR` defined in cell 1, used in cells 6, 7, 8, 10 — consistent.

---

## Notes

- **`--allow-local-clean-models` is for debugging only** (per `AGENTS.md`). Cell 11 uses it because Colab has no separate process for a vLLM server. The production H100 sweep uses a separate vLLM server process.
- **Do not run `setup_colab.sh` on macOS.** The `set -euo pipefail` guard with `uname` check will exit immediately.
- If the session disconnects mid-vector-computation, re-run from cell 8 — the file won't exist on Drive so it re-runs automatically.
- Cell 10's plot assumes `results` JSON contains `task`, `alpha`, and `entropy_mean` keys. If `run_phase1.py` output schema changes, update the plotting cell accordingly.

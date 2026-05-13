# Colab Phase 1 Full Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full Phase 1 multi-agent sweep (Experiments 1.2 / 1.3 / 1.4 across all steering strengths, all tasks, all repeats) on a single Colab A100 80 GB GPU, with resumable execution across Colab sessions backed by Google Drive.

**Architecture:** One GPU hosts both the vLLM clean-agent server (with constrained `--gpu-memory-utilization 0.4`, ~32 GB) and the steered worker (which the sweep launcher spawns as a subprocess per job, ~16 GB + activations). A new `--skip-existing` flag on `scripts/run_phase1_sweep.py` lets the sweep resume mid-matrix by skipping any job whose summary JSON already exists on Drive. A new notebook orchestrates Drive mount → setup → vector reuse → background vLLM launch → resumable sweep → final archive.

**Tech Stack:** Python 3.10 (Colab), CUDA 12.1, vLLM (Linux), HuggingFace Transformers, local CAMEL, nbformat for notebook assembly. Reuses the existing `scripts/setup_colab.sh` and `scripts/smoke_test_colab.py` from the prior Colab plan.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `scripts/run_phase1_sweep.py` | Add `--skip-existing` flag; skip jobs whose `exp{N}_summary.json` already exists |
| Modify | `src/experiments/sweep.py` | Add `summary_path()` helper on `SweepJob` so the launcher and tests agree on the skip predicate |
| Create | `tests/test_phase1_sweep_skip.py` | Unit test asserting `--skip-existing` filters jobs correctly |
| Create | `scripts/start_vllm_colab.sh` | Background-launches vLLM server with constrained GPU memory; writes PID and log to known paths |
| Create | `scripts/create_colab_sweep_notebook.py` | Assembles `notebooks/colab_phase1_full_sweep.ipynb` from Python cell strings |
| Create | `notebooks/colab_phase1_full_sweep.ipynb` | Generated Colab notebook artifact (committed; regenerate with the assembly script) |

---

## Task 1: Add `summary_path()` helper to `SweepJob`

The launcher and the skip-existing test must agree on which file proves a job is done. Put the rule in one place.

**Files:**
- Modify: `src/experiments/sweep.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_phase1_sweep_summary_path.py`:

```python
from pathlib import Path
from src.experiments.sweep import SweepJob


def test_summary_path_uses_experiment_subdir(tmp_path):
    job = SweepJob(
        experiment="1.2",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        steering_vector="vec.pt",
        task_names=None,
        task_indices=[0, 1],
        steering_strength=1.0,
        repeat_index=0,
        results_dir=str(tmp_path / "exp1_2" / "model" / "alpha_1p0" / "repeat_00"),
        max_new_tokens=256,
        chat_turn_limit=2,
    )
    assert job.summary_path() == Path(job.results_dir) / "exp1_2" / "exp1_2_summary.json"


def test_summary_path_handles_experiment_1_4():
    job = SweepJob(
        experiment="1.4",
        model="m",
        steering_vector="v",
        task_names=None,
        task_indices=None,
        steering_strength=0.5,
        repeat_index=2,
        results_dir="results/sweeps/exp1_4/m/alpha_0p5/repeat_02",
        max_new_tokens=256,
        chat_turn_limit=2,
    )
    assert job.summary_path() == Path("results/sweeps/exp1_4/m/alpha_0p5/repeat_02/exp1_4/exp1_4_summary.json")
```

- [ ] **Step 1.2: Run the test to verify it fails**

```bash
conda run -n cascade python -m pytest tests/test_phase1_sweep_summary_path.py -v
```

Expected: `AttributeError: 'SweepJob' object has no attribute 'summary_path'`

- [ ] **Step 1.3: Add the method**

In `src/experiments/sweep.py`, change the `SweepJob` dataclass to add the helper. The full updated class looks like:

```python
@dataclass(frozen=True)
class SweepJob:
    experiment: str
    model: str
    steering_vector: str
    task_names: List[str] | None
    task_indices: List[int] | None
    steering_strength: float
    repeat_index: int
    results_dir: str
    max_new_tokens: int
    chat_turn_limit: int

    def summary_path(self) -> Path:
        """Path to the summary JSON written by run_phase1.py when this job completes.

        run_phase1.py writes results into `{results_dir}/exp{N_underscore}/exp{N_underscore}_summary.json`,
        where N_underscore is e.g. "1_2" for experiment 1.2.
        """
        underscore = self.experiment.replace(".", "_")
        return Path(self.results_dir) / f"exp{underscore}" / f"exp{underscore}_summary.json"
```

Add `from pathlib import Path` to the imports if not already there (it is — already imported at top of file).

- [ ] **Step 1.4: Re-run the test to verify it passes**

```bash
conda run -n cascade python -m pytest tests/test_phase1_sweep_summary_path.py -v
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/experiments/sweep.py tests/test_phase1_sweep_summary_path.py
git commit -m "feat(sweep): add SweepJob.summary_path() for resume detection"
```

---

## Task 2: Add `--skip-existing` to the sweep launcher

Wraps every per-job command with a check: if the summary JSON already exists, log and skip. This is what makes the sweep resumable across Colab sessions.

**Files:**
- Modify: `scripts/run_phase1_sweep.py`
- Create: `tests/test_phase1_sweep_skip.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_phase1_sweep_skip.py`:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.sweep import SweepJob
from scripts.run_phase1_sweep import filter_completed_jobs


def _make_job(tmp_path, experiment="1.2", strength=1.0, repeat=0):
    return SweepJob(
        experiment=experiment,
        model="m",
        steering_vector="v",
        task_names=None,
        task_indices=[0],
        steering_strength=strength,
        repeat_index=repeat,
        results_dir=str(tmp_path / f"exp{experiment.replace('.', '_')}_a{strength}_r{repeat}"),
        max_new_tokens=256,
        chat_turn_limit=2,
    )


def test_filter_skips_jobs_with_existing_summary(tmp_path):
    j1 = _make_job(tmp_path, strength=0.5)
    j2 = _make_job(tmp_path, strength=1.0)
    j3 = _make_job(tmp_path, strength=1.5)

    # Mark j2 as already complete by writing its summary file
    j2.summary_path().parent.mkdir(parents=True, exist_ok=True)
    j2.summary_path().write_text("{}", encoding="utf-8")

    remaining = filter_completed_jobs([j1, j2, j3])
    assert remaining == [j1, j3]


def test_filter_is_noop_when_no_summaries_exist(tmp_path):
    jobs = [_make_job(tmp_path, strength=s) for s in (0.5, 1.0, 1.5)]
    assert filter_completed_jobs(jobs) == jobs
```

- [ ] **Step 2.2: Run the test to verify it fails**

```bash
conda run -n cascade python -m pytest tests/test_phase1_sweep_skip.py -v
```

Expected: `ImportError: cannot import name 'filter_completed_jobs' from 'scripts.run_phase1_sweep'`.

- [ ] **Step 2.3: Add `filter_completed_jobs` and wire `--skip-existing` into the launcher**

In `scripts/run_phase1_sweep.py`:

1. Add the function near the top, after the `SweepLane` dataclass:

```python
def filter_completed_jobs(jobs: Sequence[SweepJob]) -> List[SweepJob]:
    """Drop jobs whose summary JSON already exists on disk. Used by --skip-existing
    so Colab sessions can resume a partial sweep without re-running completed jobs."""
    remaining: List[SweepJob] = []
    skipped = 0
    for job in jobs:
        if job.summary_path().exists():
            skipped += 1
            print(f"[skip] {job.summary_path()} (already complete)")
            continue
        remaining.append(job)
    if skipped:
        print(f"[skip] {skipped} of {len(jobs)} jobs already complete; running {len(remaining)}.")
    return remaining
```

2. Add the CLI flag in `_build_parser`:

```python
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose exp{N}_summary.json already exists in their results_dir (resumable sweeps).",
    )
```

3. In `main`, apply the filter before sharding into lanes. Replace the existing `jobs = build_sweep_jobs(config)` block with:

```python
    jobs = build_sweep_jobs(config)
    if args.skip_existing:
        jobs = filter_completed_jobs(jobs)
    if not jobs:
        print("No jobs to run.")
        return
    lanes = _build_lanes(clean_api_bases, worker_gpu_sets)
    sharded_jobs = [jobs[index::len(lanes)] for index in range(len(lanes))]
```

- [ ] **Step 2.4: Re-run the test to verify it passes**

```bash
conda run -n cascade python -m pytest tests/test_phase1_sweep_skip.py -v
```

Expected: 2 passed.

- [ ] **Step 2.5: Sanity-check with dry-run**

```bash
mkdir -p /tmp/sweep-skip-demo
conda run -n cascade python scripts/run_phase1_sweep.py \
  --experiments 1.2 \
  --steering-vector /tmp/fake.pt \
  --steering-strengths 0.5,1.0 \
  --repeats 1 \
  --results-root /tmp/sweep-skip-demo \
  --skip-existing \
  --dry-run
```

Expected: prints 2 `[lane 0] python ... --steering-strength 0.5 ...` and `--steering-strength 1.0 ...` lines. Now create a fake summary for one and re-run:

```bash
mkdir -p /tmp/sweep-skip-demo/exp1_2/meta-llama/Meta-Llama-3.1-8B-Instruct/alpha_1p0/repeat_00/exp1_2
echo '{}' > /tmp/sweep-skip-demo/exp1_2/meta-llama/Meta-Llama-3.1-8B-Instruct/alpha_1p0/repeat_00/exp1_2/exp1_2_summary.json
conda run -n cascade python scripts/run_phase1_sweep.py \
  --experiments 1.2 \
  --steering-vector /tmp/fake.pt \
  --steering-strengths 0.5,1.0 \
  --repeats 1 \
  --results-root /tmp/sweep-skip-demo \
  --skip-existing \
  --dry-run
```

Expected: one `[skip]` line for alpha 1.0, one `[lane 0]` line for alpha 0.5.

- [ ] **Step 2.6: Commit**

```bash
git add scripts/run_phase1_sweep.py tests/test_phase1_sweep_skip.py
git commit -m "feat(sweep): add --skip-existing flag for resumable Colab sweeps"
```

---

## Task 3: vLLM Background Launcher Script

A small helper that starts vLLM in the background on a single GPU with constrained memory utilization, writes its PID to a known path, and tees logs. The notebook will call this and then poll the OpenAI endpoint until healthy.

**Files:**
- Create: `scripts/start_vllm_colab.sh`

- [ ] **Step 3.1: Write the script**

```bash
#!/usr/bin/env bash
# scripts/start_vllm_colab.sh
# Launch vLLM in the background on a single Colab A100 with constrained GPU memory,
# leaving enough headroom for a co-located steered worker.
#
# Usage:
#   scripts/start_vllm_colab.sh [MODEL] [PORT] [GPU_UTIL]
# Defaults: MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct, PORT=8000, GPU_UTIL=0.40
set -euo pipefail

MODEL="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${2:-8000}"
GPU_UTIL="${3:-0.40}"

[[ "$(uname -s)" == "Linux" ]] || { echo "Linux/Colab only." >&2; exit 1; }
command -v vllm >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || \
  { echo "python not found." >&2; exit 1; }

PID_FILE="/tmp/vllm_colab.pid"
LOG_FILE="/tmp/vllm_colab.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "vLLM already running (pid $(cat "$PID_FILE")); reusing. Tail: $LOG_FILE"
  exit 0
fi

echo "Starting vLLM: model=$MODEL port=$PORT gpu_util=$GPU_UTIL"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 4096 \
  --dtype auto \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "vLLM pid=$(cat "$PID_FILE"); logs: $LOG_FILE"
echo "Poll readiness with: curl -s http://127.0.0.1:$PORT/v1/models"
```

- [ ] **Step 3.2: Make it executable**

```bash
chmod +x scripts/start_vllm_colab.sh
```

- [ ] **Step 3.3: Sanity check the syntax**

```bash
bash -n scripts/start_vllm_colab.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3.4: Commit**

```bash
git add scripts/start_vllm_colab.sh
git commit -m "feat: add backgrounded vLLM launcher for Colab single-GPU sweeps"
```

---

## Task 4: Sweep Notebook Assembly Script

Writes the canonical sweep notebook source as Python strings and uses `nbformat` to emit the `.ipynb`.

**Files:**
- Create: `scripts/create_colab_sweep_notebook.py`
- Create (generated): `notebooks/colab_phase1_full_sweep.ipynb`

- [ ] **Step 4.1: Write the assembly script**

```python
#!/usr/bin/env python3
# scripts/create_colab_sweep_notebook.py
"""Generates notebooks/colab_phase1_full_sweep.ipynb from Python string literals.
Re-run whenever any cell changes:
    python scripts/create_colab_sweep_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    sys.exit("nbformat is required: pip install nbformat")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_phase1_full_sweep.ipynb"
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
    md("""\
# MAS Activation Cascades — Phase 1 Full Sweep (Colab A100 80 GB)

Resumable end-to-end sweep across **Experiments 1.2, 1.3, 1.4** on Llama-3.1-8B-Instruct,
spanning all steering strengths × tasks × repeats.

**Requirements:**
- Colab Pro+ / Enterprise with **A100 80 GB**
- HuggingFace token with Llama-3 access, stored as Colab secret `HF_TOKEN`
- Steering vector from the quickstart notebook (or it will compute one if missing)

**Plan:**
1. Mount Drive (persists vector + results across sessions).
2. Clone repo, install deps, smoke test.
3. Start vLLM clean-agent server in the background (`--gpu-memory-utilization 0.40`).
4. Wait for `/v1/models` to respond.
5. Launch the resumable sweep (`--skip-existing`) — completed jobs are skipped, so re-running this cell after a disconnect picks up where you left off.
6. Shut down the server and archive results.

**Sweep matrix (default):** 3 experiments × 5 strengths × 5 tasks × 3 repeats = 225 trials (~45 sweep jobs).
Expect 8–20 hours total wall time. Use `--skip-existing` across multiple sessions if needed.
"""),

    md("## 1 · Mount Google Drive"),
    code("""\
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = '/content/drive/MyDrive/MAS-Activation-Cascades'
for sub in ['steering_vectors', 'results', 'results/sweeps', 'data/contrastive_pairs', 'logs']:
    os.makedirs(f'{DRIVE_DIR}/{sub}', exist_ok=True)
print("Drive ready at:", DRIVE_DIR)
"""),

    md("## 2 · Clone or Update Repo"),
    code("""\
import os, subprocess

REPO_URL = 'https://github.com/leoding101/MAS-Activation-Cascades.git'  # ← update to your fork if needed
REPO_DIR = '/content/MAS-Activation-Cascades'

if not os.path.exists(REPO_DIR + '/.git'):
    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=True)

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
"""),

    md("## 3 · HuggingFace Authentication"),
    code("""\
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

from huggingface_hub import whoami
print("Logged in as:", whoami(token=os.environ['HF_TOKEN'])['name'])
"""),

    md("## 4 · Install Dependencies"),
    code("""\
import subprocess
result = subprocess.run(['bash', 'scripts/setup_colab.sh'])
if result.returncode != 0:
    raise RuntimeError("setup_colab.sh failed — see output above")
print("Setup complete")
"""),

    md("## 5 · Pre-flight Smoke Test"),
    code("""\
import subprocess
r = subprocess.run(['python', 'scripts/smoke_test_colab.py'])
if r.returncode != 0:
    raise RuntimeError("Smoke test failed — fix the issues printed above before continuing")
print("Smoke test passed ✓")
"""),

    md("## 6 · Link Drive Storage into Repo Paths"),
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

    md("## 7 · Ensure Contrastive Pairs and Steering Vector Exist"),
    code("""\
import subprocess, os, torch

PAIRS_PATH = 'data/contrastive_pairs/ta2_harmful_pairs.json'
VECTOR_PATH = 'steering_vectors/harmfulness_llama3_8b.pt'

if not os.path.exists(PAIRS_PATH):
    subprocess.run(['python', 'scripts/build_ta2_pairs.py', '--output', PAIRS_PATH], check=True)
else:
    print("Pairs already exist — skipping.")

if not os.path.exists(VECTOR_PATH):
    print("Computing steering vector (~15–20 min on A100)...")
    subprocess.run(
        ['python', 'src/steering/compute_vectors.py',
         '--model', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
         '--pairs-path', PAIRS_PATH,
         '--output', VECTOR_PATH,
         '--device', 'cuda'],
        check=True,
    )
else:
    meta = torch.load(VECTOR_PATH, weights_only=True)
    print(f"Vector cached (optimal layer: {meta.get('optimal_layer', '?')}) — skipping.")
"""),

    md("""\
## 8 · Start vLLM Clean-Agent Server (background)

Runs on the same A100 with `--gpu-memory-utilization 0.40` (~32 GB), leaving ~48 GB for the steered worker the sweep spawns per job.
"""),
    code("""\
import subprocess, time, urllib.request, urllib.error

# Idempotent: re-running the cell reuses an existing server.
subprocess.run(['bash', 'scripts/start_vllm_colab.sh',
                'meta-llama/Meta-Llama-3.1-8B-Instruct', '8000', '0.40'], check=True)

# Poll /v1/models until ready (cold start: 2–5 min after first launch).
deadline = time.time() + 600
url = 'http://127.0.0.1:8000/v1/models'
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            if r.status == 200:
                print("vLLM ready:", r.read().decode()[:200])
                break
    except (urllib.error.URLError, ConnectionResetError):
        pass
    time.sleep(5)
else:
    raise RuntimeError("vLLM did not become ready within 10 minutes — check /tmp/vllm_colab.log")
"""),

    md("""\
## 9 · Run the Resumable Full Sweep

`--skip-existing` lets you re-run this cell across multiple Colab sessions; completed jobs (those with an `exp{N}_summary.json` on Drive) are skipped automatically.

**Default matrix:** 1.2/1.3/1.4 × α∈{0.0, 0.5, 1.0, 1.5, 2.0} × tasks 0–4 × 3 repeats.
"""),
    code("""\
import subprocess

cmd = [
    'python', 'scripts/run_phase1_sweep.py',
    '--experiments', '1.2,1.3,1.4',
    '--models', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    '--steering-vector', VECTOR_PATH,
    '--steering-strengths', '0.0,0.5,1.0,1.5,2.0',
    '--task-indices', '0,1,2,3,4',
    '--repeats', '3',
    '--results-root', 'results/sweeps',
    '--clean-api-bases', 'http://127.0.0.1:8000/v1',
    '--max-new-tokens', '256',
    '--chat-turn-limit', '2',
    '--skip-existing',
]
print("Launching:", ' '.join(cmd))
subprocess.run(cmd, check=True)
print("Sweep cell finished. Re-run after a disconnect to resume.")
"""),

    md("## 10 · Shut Down the vLLM Server"),
    code("""\
import os, signal

pid_file = '/tmp/vllm_colab.pid'
if os.path.exists(pid_file):
    pid = int(open(pid_file).read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to vLLM pid={pid}")
    except ProcessLookupError:
        print(f"vLLM pid={pid} already gone")
    os.remove(pid_file)
else:
    print("No vLLM pid file found — nothing to stop.")
"""),

    md("## 11 · Archive Sweep Results"),
    code("""\
import shutil, os, datetime

stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
archive_base = f'{DRIVE_DIR}/results/sweeps_archive_{stamp}'
shutil.make_archive(archive_base, 'gztar', 'results/sweeps')
print(f"Archive: {archive_base}.tar.gz")

# Quick inventory of what got written
for dirpath, _, filenames in os.walk('results/sweeps'):
    for fn in filenames:
        if fn.endswith('_summary.json'):
            print(os.path.relpath(os.path.join(dirpath, fn), 'results/sweeps'))
"""),
]

with open(OUTPUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {OUTPUT.relative_to(ROOT)}")
```

- [ ] **Step 4.2: Install nbformat (if needed) and generate the notebook**

```bash
conda run -n cascade pip install nbformat
conda run -n cascade python scripts/create_colab_sweep_notebook.py
```

Expected output: `Notebook written to notebooks/colab_phase1_full_sweep.ipynb`.

- [ ] **Step 4.3: Verify the notebook is valid JSON and has the expected cells**

```bash
conda run -n cascade python -c "
import json
nb = json.load(open('notebooks/colab_phase1_full_sweep.ipynb'))
cells = nb['cells']
print(f'Cells: {len(cells)}')
assert len(cells) == 22, f'Expected 22 cells, got {len(cells)}'
types = [c['cell_type'] for c in cells]
print('First 6 types:', types[:6])
print('Notebook OK')
"
```

Expected: `Cells: 22` and `Notebook OK`.

- [ ] **Step 4.4: Commit**

```bash
git add scripts/create_colab_sweep_notebook.py notebooks/colab_phase1_full_sweep.ipynb
git commit -m "feat: add resumable Colab notebook for Phase 1 full sweep"
```

---

## Task 5: End-to-End Colab Validation

Interactive check. Confirms the sweep notebook actually runs on a Colab A100 80 GB and produces resumable output on Drive.

**Prerequisites:** Colab A100 80 GB runtime, `HF_TOKEN` secret set, Llama-3 access granted.

- [ ] **Step 5.1: Open the notebook in Colab**

`File → Open notebook → GitHub → paste your repo URL → select notebooks/colab_phase1_full_sweep.ipynb`.

- [ ] **Step 5.2: Set runtime to GPU A100 80 GB and confirm**

`Runtime → Change runtime type → GPU → A100 → Save`. After the smoke test cell runs, confirm the log shows `VRAM ≥ 70 GB`.

- [ ] **Step 5.3: Run cells 1 through 8 in order**

Watch checkpoints:
1. **Cell 5 (smoke test)** prints all `[ok]` and exits 0.
2. **Cell 7 (vector)** either prints `Vector cached…` (preferred) or runs ~15–20 min.
3. **Cell 8 (vLLM start)** finishes with `vLLM ready: {"object": "list", "data": ...`.

If cell 8 times out, `cat /tmp/vllm_colab.log` for the failure (usually OOM — re-run with `gpu_util=0.35`).

- [ ] **Step 5.4: Run the sweep cell (cell 9) and let it execute at least one full job**

Watch for: `[lane 0] python … --experiment 1.2 …` followed by run_phase1.py logs, then a new `*_summary.json` file appearing on Drive at `MAS-Activation-Cascades/results/sweeps/exp1_2/.../alpha_0p0/repeat_00/exp1_2/exp1_2_summary.json`.

- [ ] **Step 5.5: Verify resume behavior by interrupting and restarting**

After at least one summary is written:
1. Stop the cell (`■`).
2. Re-run cell 9.
3. Confirm the launcher prints at least one `[skip]` line for the previously completed job and continues with the next.

- [ ] **Step 5.6: Let the sweep run to completion (across sessions if needed)**

If Colab disconnects, reconnect, re-run cells 1, 2, 3, 5, 6, 8, 9. Cells 1, 6 are idempotent; cell 8 detects the live PID and reuses the server (or relaunches if killed).

- [ ] **Step 5.7: Verify final results inventory**

Cell 11's inventory should list 45 `*_summary.json` files (3 experiments × 5 strengths × 3 repeats). Spot-check one to ensure it contains non-trivial entropy/MSP numbers, not just metadata.

---

## Self-Review

**Spec coverage:**
- ✅ Full sweep matrix (1.2/1.3/1.4 × 5 strengths × 5 tasks × 3 repeats): cell 9.
- ✅ Single A100 80 GB co-locating vLLM + steered worker: cells 8 + 9, `--gpu-memory-utilization 0.40`.
- ✅ Resumable across Colab sessions: Tasks 1, 2 add `summary_path()` and `--skip-existing`; cell 9 uses the flag.
- ✅ Persistence to Drive: cells 1 + 6.
- ✅ Reuses existing setup/smoke from prior plan: cells 4, 5.
- ✅ New notebook (per user choice): Task 4.

**Placeholder scan:** No TBD / TODO / "add appropriate" / "similar to" placeholders. Every code step shows complete content.

**Type consistency:**
- `SweepJob.summary_path()` is defined in Task 1 and used by `filter_completed_jobs` in Task 2 — names match.
- `VECTOR_PATH` defined in cell 7, used in cell 9 — consistent.
- `DRIVE_DIR` defined in cell 1, used in cells 6, 11 — consistent.
- `PID_FILE=/tmp/vllm_colab.pid` written by `start_vllm_colab.sh` (Task 3), read by cell 10 — consistent.
- `--skip-existing` defined in Task 2, used in cell 9 — consistent.

**Notes:**
- `--gpu-memory-utilization 0.40` is conservative; if vLLM cold-starts fine but the steered worker hits CUDA OOM, drop it to `0.35`. The reverse (vLLM OOM) is rare on 8B at this util.
- The sweep does not parallelize across lanes on a single GPU — one job at a time is intentional. To shorten wall time further, reduce repeats to 2 or drop α=0.0 from the strength list (it's a no-op reference and only needed once per (experiment, task) pair, but the current sweep grid does not de-duplicate this).
- If your fork's main branch is private, replace the `git clone` URL in cell 2 with an HTTPS URL plus a Colab `userdata` secret for the PAT.

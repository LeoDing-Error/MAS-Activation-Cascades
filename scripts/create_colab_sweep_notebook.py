#!/usr/bin/env python3
# scripts/create_colab_sweep_notebook.py
"""Generates notebooks/colab_phase1_full_sweep.ipynb from Python string literals.
Re-run whenever any cell changes:
    python scripts/create_colab_sweep_notebook.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import nbformat
except ImportError:
    sys.exit("nbformat is required: pip install nbformat")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "colab_phase1_full_sweep.ipynb"
parser = argparse.ArgumentParser(description="Generate the Phase 1 sweep notebook")
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
OUTPUT = parser.parse_args().output
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
using only the alpha that passed a separate held-out confirmation.

**Requirements:**
- Colab Pro+ / Enterprise with **A100 80 GB**
- HuggingFace token with Llama-3 access, stored as Colab secret `HF_TOKEN`
- Existing steering vector and a passed `held_out_confirmation.json` bound to it

**Plan:**
1. Mount Drive (persists vector + results across sessions).
2. Clone repo, install deps, smoke test.
3. Start vLLM clean-agent server in the background (`--gpu-memory-utilization 0.40`).
4. Wait for `/v1/models` to respond.
5. Launch the resumable sweep (`--skip-existing`) — completed jobs are skipped, so re-running this cell after a disconnect picks up where you left off.
6. Shut down the server and archive results.

**Sweep matrix:** 3 experiments × 1 confirmed strength × 5 tasks × 3 repeats.
The notebook and CLI refuse broad/default-alpha sweeps. Use `--skip-existing`
across multiple sessions if needed.
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

REPO_URL = 'https://github.com/LeoDing-Error/MAS-Activation-Cascades.git'  # ← update to your fork if needed
REPO_DIR = '/content/MAS-Activation-Cascades'
COLAB_BRANCH = 'main'

if not os.path.exists(REPO_DIR + '/.git'):
    subprocess.run(
        ['git', 'clone', '--branch', COLAB_BRANCH, '--single-branch', REPO_URL, REPO_DIR],
        check=True,
    )
else:
    subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin', COLAB_BRANCH], check=True)
    subprocess.run(['git', '-C', REPO_DIR, 'checkout', COLAB_BRANCH], check=True)
    subprocess.run(
        ['git', '-C', REPO_DIR, 'pull', '--ff-only', 'origin', COLAB_BRANCH],
        check=True,
    )

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

    md("""\
## 6 · Configure Persistent Artifact Paths

Artifacts use absolute Drive paths so the repository's tracked `.gitkeep`
directories cannot accidentally redirect output to ephemeral storage. Model
downloads and temporary vector files stay on the faster local VM disk.
"""),
    code("""\
import os
from pathlib import Path

os.environ['HF_HOME'] = '/content/hf-cache'
os.environ['TRANSFORMERS_CACHE'] = '/content/hf-cache/transformers'

PAIRS_PATH = f'{DRIVE_DIR}/data/contrastive_pairs/ta2_harmful_pairs.json'
VECTOR_PATH = f'{DRIVE_DIR}/steering_vectors/harmfulness_llama3_8b.pt'
ANALYSIS_PATH = f'{DRIVE_DIR}/steering_vectors/harmfulness_llama3_8b.analysis.pt'
SWEEP_RESULTS_ROOT = f'{DRIVE_DIR}/results/sweeps'
HELD_OUT_CONFIRMATION_PATH = f'{DRIVE_DIR}/results/held_out_confirmation.json'
LOCAL_VECTOR_PATH = '/content/harmfulness_llama3_8b.pt'
LOCAL_ANALYSIS_PATH = '/content/harmfulness_llama3_8b.analysis.pt'

for artifact_path in (PAIRS_PATH, VECTOR_PATH, ANALYSIS_PATH):
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
Path(SWEEP_RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)

print("Persistent artifacts:", DRIVE_DIR)
print("Local model cache:", os.environ['HF_HOME'])
"""),

md("## 7 · Verify the Held-Out Confirmation and Existing Artifact"),
    code("""\
import json, os
from pathlib import Path

if not Path(VECTOR_PATH).is_file():
    raise RuntimeError('Phase 1 is blocked: the confirmed steering artifact is missing.')
confirmation_path = Path(HELD_OUT_CONFIRMATION_PATH)
if not confirmation_path.is_file():
    raise RuntimeError('Phase 1 is blocked: held_out_confirmation.json is missing.')
confirmation = json.loads(confirmation_path.read_text(encoding='utf-8'))
if (
    confirmation.get('kind') != 'phase1_held_out_confirmation'
    or confirmation.get('held_out_confirmation_passed') is not True
):
    raise RuntimeError('Phase 1 is blocked: held-out confirmation has not passed.')
CONFIRMED_ALPHA = float(confirmation['selected_alpha'])
print('Held-out confirmed alpha:', CONFIRMED_ALPHA)
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

**Locked matrix:** 1.2/1.3/1.4 × the one held-out-confirmed alpha × tasks 0–4 × 3 repeats.
"""),
    code("""\
import subprocess

cmd = [
    'python', 'scripts/run_phase1_sweep.py',
    '--experiments', '1.2,1.3,1.4',
    '--models', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    '--steering-vector', VECTOR_PATH,
    '--steering-strengths', str(CONFIRMED_ALPHA),
    '--held-out-confirmation', HELD_OUT_CONFIRMATION_PATH,
    '--task-indices', '0,1,2,3,4',
    '--repeats', '3',
    '--results-root', SWEEP_RESULTS_ROOT,
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
local_archive_base = f'/content/sweeps_archive_{stamp}'
local_archive = shutil.make_archive(local_archive_base, 'gztar', SWEEP_RESULTS_ROOT)
archive_path = f'{DRIVE_DIR}/results/sweeps_archive_{stamp}.tar.gz'
shutil.copy2(local_archive, archive_path)
print(f"Archive: {archive_path}")

# Quick inventory of what got written
for dirpath, _, filenames in os.walk(SWEEP_RESULTS_ROOT):
    for fn in filenames:
        if fn.endswith('_summary.json'):
            print(os.path.relpath(os.path.join(dirpath, fn), SWEEP_RESULTS_ROOT))
"""),
]

for index, cell in enumerate(nb.cells, start=1):
    cell["id"] = f"cell-{index:02d}"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {OUTPUT}")

"""
Assembles notebooks/colab_phase1_quickstart.ipynb from scratch.
Run with: python scripts/create_colab_notebook.py
"""

import pathlib
import nbformat

OUTPUT = pathlib.Path(__file__).parent.parent / "notebooks" / "colab_phase1_quickstart.ipynb"

# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# Section 0 – Header
# ---------------------------------------------------------------------------

S0_MD = """\
# MAS Activation Cascades — Colab Quickstart (Phase 1 / Exp 1.1)

This notebook runs the full Experiment 1.1 pipeline on a **single A100 GPU**:

1. Mount Google Drive (persistent storage)
2. Clone / update the research repo
3. Authenticate with HuggingFace
4. Install dependencies (`scripts/setup_colab.sh`)
5. Smoke-test the environment
6. Configure persistent Drive paths and a local model cache
7. Build contrastive pairs
8. Compute the steering vector (~15–20 min on H100/A100)
9. Run Exp 1.1 (steering validation, ~5–10 min for 3 tasks × 5 alphas)
10. Plot token entropy vs alpha
11. (Optional) Run Exp 1.2 if ≥ 70 GB VRAM is available

**Runtime estimates** (A100 40 GB):
- Steps 1–6: < 5 min
- Step 8 (vector): ~15–20 min
- Step 9 (Exp 1.1): ~5–10 min

**Required runtime**: GPU → A100 (or equivalent). Select via *Runtime → Change runtime type*.
"""

S0_CODE = """\
# Runtime check – confirm GPU is available
import subprocess, sys

result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                        capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError("No GPU detected. Please enable a GPU runtime before continuing.")
print("GPU detected:", result.stdout.strip())
"""

# ---------------------------------------------------------------------------
# Section 1 – Mount Google Drive
# ---------------------------------------------------------------------------

S1_MD = """\
## 1 · Mount Google Drive

All large artifacts (steering vectors, results, contrastive pairs) are stored on
Google Drive so they persist across Colab sessions.
"""

S1_CODE = """\
from google.colab import drive
import os, pathlib

drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/MAS-Activation-Cascades'

for subdir in ['steering_vectors', 'results', 'data/contrastive_pairs']:
    pathlib.Path(DRIVE_DIR, subdir).mkdir(parents=True, exist_ok=True)
    print(f"  ensured {DRIVE_DIR}/{subdir}")

print("Drive ready:", DRIVE_DIR)
"""

# ---------------------------------------------------------------------------
# Section 2 – Clone or update repo
# ---------------------------------------------------------------------------

S2_MD = """\
## 2 · Clone or update the repository
"""

S2_CODE = """\
import subprocess, os, pathlib

REPO_URL = 'https://github.com/LeoDing-Error/MAS-Activation-Cascades.git'
REPO_DIR = '/content/MAS-Activation-Cascades'
COLAB_BRANCH = 'main'

if not pathlib.Path(REPO_DIR, '.git').exists():
    subprocess.run(
        ['git', 'clone', '--branch', COLAB_BRANCH, '--single-branch', REPO_URL, REPO_DIR],
        check=True,
    )
    print(f"Cloned {COLAB_BRANCH} to {REPO_DIR}")
else:
    subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin', COLAB_BRANCH], check=True)
    subprocess.run(['git', '-C', REPO_DIR, 'checkout', COLAB_BRANCH], check=True)
    subprocess.run(
        ['git', '-C', REPO_DIR, 'pull', '--ff-only', 'origin', COLAB_BRANCH],
        check=True,
    )
    print(f"Updated {COLAB_BRANCH} at {REPO_DIR}")

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
"""

# ---------------------------------------------------------------------------
# Section 3 – HuggingFace authentication
# ---------------------------------------------------------------------------

S3_MD = """\
## 3 · HuggingFace authentication

Your HF token must be stored in Colab secrets under the key `HF_TOKEN`.
Go to *Tools → Secrets* to add it.
"""

S3_CODE = """\
import os
from google.colab import userdata
import huggingface_hub

HF_TOKEN = userdata.get('HF_TOKEN')
os.environ['HF_TOKEN'] = HF_TOKEN

info = huggingface_hub.whoami(token=HF_TOKEN)
print(f"Logged in as: {info['name']}")
"""

# ---------------------------------------------------------------------------
# Section 4 – Install dependencies
# ---------------------------------------------------------------------------

S4_MD = """\
## 4 · Install dependencies

Runs `scripts/setup_colab.sh` which installs the project and its dependencies
into the Colab Python environment. This takes 2–4 minutes on first run.
"""

S4_CODE = """\
import subprocess, sys

result = subprocess.run(['bash', 'scripts/setup_colab.sh'], capture_output=False)
if result.returncode != 0:
    raise RuntimeError(f"setup_colab.sh failed with returncode {result.returncode}")
print("Dependencies installed successfully.")
"""

# ---------------------------------------------------------------------------
# Section 5 – Smoke test gate
# ---------------------------------------------------------------------------

S5_MD = """\
## 5 · Smoke test gate

Verifies that key imports and CUDA are working before running expensive steps.
"""

S5_CODE = """\
import subprocess

result = subprocess.run(['python', 'scripts/smoke_test_colab.py'], capture_output=False)
if result.returncode != 0:
    raise RuntimeError(
        "Smoke test failed — environment is not correctly configured. "
        "Check the output above for details."
    )
print("Smoke test passed.")
"""

# ---------------------------------------------------------------------------
# Section 6 – Persistent artifact paths
# ---------------------------------------------------------------------------

S6_MD = """\
## 6 · Configure persistent artifact paths

Artifacts use absolute Drive paths so the repository's tracked `.gitkeep`
directories cannot accidentally redirect output to ephemeral storage. Model
downloads and temporary vector files stay on the faster local VM disk.
"""

S6_CODE = """\
import os, pathlib

DRIVE_DIR = '/content/drive/MyDrive/MAS-Activation-Cascades'
os.environ['HF_HOME'] = '/content/hf-cache'
os.environ['TRANSFORMERS_CACHE'] = '/content/hf-cache/transformers'

PAIRS_PATH = f'{DRIVE_DIR}/data/contrastive_pairs/ta2_harmful_pairs.json'
VECTOR_PATH = f'{DRIVE_DIR}/steering_vectors/harmfulness_llama3_8b.pt'
ANALYSIS_PATH = f'{DRIVE_DIR}/steering_vectors/harmfulness_llama3_8b.analysis.pt'
RESULTS_DIR = f'{DRIVE_DIR}/results'
LOCAL_VECTOR_PATH = '/content/harmfulness_llama3_8b.pt'
LOCAL_ANALYSIS_PATH = '/content/harmfulness_llama3_8b.analysis.pt'

for artifact_path in (PAIRS_PATH, VECTOR_PATH, ANALYSIS_PATH):
    pathlib.Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)

print("Persistent artifacts:", DRIVE_DIR)
print("Local model cache:", os.environ['HF_HOME'])
"""

# ---------------------------------------------------------------------------
# Section 7 – Build contrastive pairs
# ---------------------------------------------------------------------------

S7_MD = """\
## 7 · Build contrastive pairs

Constructs positive/negative sentence pairs from the TA² harmful dataset.
Skipped if the file already exists on Drive.
"""

S7_CODE = """\
import json, pathlib, subprocess

if pathlib.Path(PAIRS_PATH).exists():
    with open(PAIRS_PATH) as f:
        pairs_payload = json.load(f)
    pair_count = pairs_payload.get('count', len(pairs_payload.get('pairs', [])))
    print(f"Pairs file already exists — {pair_count} pairs loaded from {PAIRS_PATH}")
else:
    print("Building contrastive pairs (this takes ~1 min)…")
    subprocess.run(
        ['python', 'scripts/build_ta2_pairs.py', '--output', PAIRS_PATH],
        check=True,
    )
    with open(PAIRS_PATH) as f:
        pairs_payload = json.load(f)
    print(f"Built {pairs_payload['count']} contrastive pairs → {PAIRS_PATH}")
"""

# ---------------------------------------------------------------------------
# Section 8 – Compute steering vector
# ---------------------------------------------------------------------------

S8_MD = """\
## 8 · Compute steering vector

Uses TA²-style contrastive activation differences to find the harmfulness
direction in residual-stream space. Saved to Drive for reuse.

**Expected runtime:** ~15–20 min on an H100/A100.
"""

S8_CODE = """\
import pathlib, shutil, subprocess, torch

if pathlib.Path(VECTOR_PATH).exists():
    data = torch.load(VECTOR_PATH, weights_only=True)
    print(f"Steering vector already exists — selected layer: {data.get('selected_layer', 'N/A')}")
else:
    print("Computing steering vector (this takes ~15–20 min)…")
    subprocess.run(
        [
            'python', 'src/steering/compute_vectors.py',
            '--model', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            '--pairs-path', PAIRS_PATH,
            '--output', LOCAL_VECTOR_PATH,
            '--device', 'cuda',
        ],
        check=True,
    )
    shutil.copy2(LOCAL_ANALYSIS_PATH, ANALYSIS_PATH)
    shutil.copy2(LOCAL_VECTOR_PATH, VECTOR_PATH)
    data = torch.load(VECTOR_PATH, weights_only=True)
    print(f"Vector computed → {VECTOR_PATH}  (selected layer: {data.get('selected_layer', 'N/A')})")
"""

# ---------------------------------------------------------------------------
# Section 9 – Run Exp 1.1
# ---------------------------------------------------------------------------

S9_MD = """\
## 9 · Run Experiment 1.1 — single-agent steering validation

Sweeps steering strength α over `[0.0, 0.5, 1.0, 1.5, 2.0]` on 3 HumanEval
tasks using a locally-loaded steered model (no vLLM server required).

**Expected runtime:** ~5–10 min on an A100.
"""

S9_CODE = """\
import subprocess

subprocess.run(
    [
        'python', 'experiments/run_phase1.py',
        '--experiment', '1.1',
        '--steering-vector', VECTOR_PATH,
        '--n-tasks', '3',
        '--alphas', '0.0,0.5,1.0,1.5,2.0',
        '--results-dir', RESULTS_DIR,
    ],
    check=True,
)
print(f"Exp 1.1 complete. Results in {RESULTS_DIR}/exp1_1/")
"""

# ---------------------------------------------------------------------------
# Section 10 – Inspect results
# ---------------------------------------------------------------------------

S10_MD = """\
## 10 · Inspect results — token entropy vs α

Loads the Exp 1.1 JSON and plots mean token entropy per task as a function
of steering strength α.
"""

S10_CODE = """\
import json, pathlib
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = f'{RESULTS_DIR}/exp1_1/exp1_1_results.json'

with open(RESULTS_PATH) as f:
    data = json.load(f)

records = data['runs']
print(f"Loaded {len(records)} run records.")

task_data = defaultdict(dict)

for rec in records:
    condition = rec.get('condition', '')
    try:
        alpha = float(condition.split('alpha_')[1])
    except (IndexError, ValueError):
        continue

    task_name = rec.get('task', {}).get('name', 'unknown')

    entropies = [
        s['metrics']['mean_token_entropy']
        for s in rec.get('uncertainty', [])
        if isinstance(s.get('metrics'), dict) and s['metrics'].get('mean_token_entropy') is not None
    ]
    entropy = sum(entropies) / len(entropies) if entropies else float('nan')
    task_data[task_name][alpha] = entropy

fig, ax = plt.subplots(figsize=(8, 5))

for task_name, alpha_map in sorted(task_data.items()):
    alphas = sorted(alpha_map.keys())
    entropies = [alpha_map[a] for a in alphas]
    ax.plot(alphas, entropies, marker='o', label=task_name)

ax.set_xlabel('Steering strength α')
ax.set_ylabel('Mean token entropy')
ax.set_title('Exp 1.1 — Token entropy vs steering strength')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

PLOT_PATH = f'{RESULTS_DIR}/exp1_1/entropy_vs_alpha.png'
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"Plot saved to {PLOT_PATH}")
"""

# ---------------------------------------------------------------------------
# Section 11 – Optional Exp 1.2
# ---------------------------------------------------------------------------

S11_MD = """\
## 11 · (Optional) Run Experiment 1.2 — two-agent chain

Exp 1.2 loads both the steered model and a clean model locally
(`--allow-local-clean-models`), which requires ≥ 70 GB VRAM.

This cell is a no-op on A100 40 GB — it will print a skip message.
"""

S11_CODE = """\
import subprocess, torch

total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"Available VRAM: {total_vram_gb:.1f} GB")

if total_vram_gb < 70:
    print(f"Skipping Exp 1.2 — need ≥ 70 GB VRAM, have {total_vram_gb:.1f} GB.")
    print("Exp 1.2 requires two full model copies; run on an H100 80 GB instead.")
else:
    print("Sufficient VRAM detected — running Exp 1.2…")
    subprocess.run(
        [
            'python', 'experiments/run_phase1.py',
            '--experiment', '1.2',
            '--steering-vector', VECTOR_PATH,
            '--steering-strength', '1.0',
            '--n-tasks', '2',
            '--results-dir', RESULTS_DIR,
            '--allow-local-clean-models',
        ],
        check=True,
    )
    print(f"Exp 1.2 complete. Results in {RESULTS_DIR}/exp1_2/")
"""

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

cells = [
    md(S0_MD),   code(S0_CODE),
    md(S1_MD),   code(S1_CODE),
    md(S2_MD),   code(S2_CODE),
    md(S3_MD),   code(S3_CODE),
    md(S4_MD),   code(S4_CODE),
    md(S5_MD),   code(S5_CODE),
    md(S6_MD),   code(S6_CODE),
    md(S7_MD),   code(S7_CODE),
    md(S8_MD),   code(S8_CODE),
    md(S9_MD),   code(S9_CODE),
    md(S10_MD),  code(S10_CODE),
    md(S11_MD),  code(S11_CODE),
]

nb = nbformat.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    },
    'language_info': {
        'name': 'python',
        'version': '3.11.0',
    },
    'colab': {
        'provenance': [],
        'gpuType': 'A100',
        'include_colab_link': True,
    },
    'accelerator': 'GPU',
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, 'w') as f:
    nbformat.write(nb, f)

print(f"Notebook written to {OUTPUT}")

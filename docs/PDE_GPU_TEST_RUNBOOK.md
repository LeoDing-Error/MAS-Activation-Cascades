# PDE GPU Test Runbook

This runbook explains how to connect to the Emory Math PDE cluster, put the repo and environment in scratch, and run the full project test suite through Slurm.

Commands below are filled in for Emory NetID `lding43`.

## 1. Connect

Use Emory Unplugged on campus. If you are off campus, connect to the Emory VPN first.

```bash
ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu
```

Do not run computation from `/home/lding43`. PDE requires computation, environments, data, caches, and temporary files to live under:

```bash
/local/scratch2/lding43
```

Create the scratch workspace:

```bash
mkdir -p /local/scratch2/lding43
cd /local/scratch2/lding43
```

## 2. Put The Repo In Scratch

Clone or copy the repo into scratch. Example with Git:

```bash
cd /local/scratch2/lding43
git clone <repo-url> MAS-Activation-Cascades
cd MAS-Activation-Cascades
```

If the repo already exists:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
git pull
```

## 3. Keep Conda And Caches In Scratch

Load or initialize Conda using the PDE documentation in `/usr/local/SLURM` if needed. Then keep Conda environments, packages, Python caches, Hugging Face caches, and temp files in scratch:

```bash
export SCRATCH=/local/scratch2/lding43
mkdir -p "$SCRATCH/conda/envs" "$SCRATCH/conda/pkgs" "$SCRATCH/.cache" "$SCRATCH/tmp"

export CONDA_ENVS_PATH="$SCRATCH/conda/envs"
export CONDA_PKGS_DIRS="$SCRATCH/conda/pkgs"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/transformers"
export TMPDIR="$SCRATCH/tmp"
```

If you use a virtualenv instead of Conda for unrelated work, edit its `venv/bin/activate` and add:

```bash
export XDG_CACHE_HOME=/local/scratch2/lding43/.cache
```

This repo should still use the `cascade` Conda environment.

## 4. Configure VS Code Remote SSH

If you use VS Code Remote SSH, redirect the VS Code server into scratch in VS Code User settings:

```json
{
  "remote.SSH.serverInstallPath": {
    "pdelogin": "/local/scratch2/lding43",
    "pdelogin.mathcs.emory.edu": "/local/scratch2/lding43"
  }
}
```

The hostnames must match your SSH aliases in `~/.ssh/config`. Set this in User settings, then disconnect and reconnect VS Code.

Verify the server is under scratch:

```bash
ls -la /local/scratch2/lding43
```

Confirm `.vscode-server` appears there.

Keep file watching out of `/home` by adding a User setting such as:

```json
{
  "files.watcherExclude": {
    "/home/**": true
  }
}
```

## 5. Set Up The Repo Environment

Generate and submit setup from the scratch checkout:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
python scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

Important constraints:

- Do not install `camel-ai` from PyPI.
- Do not update `third_party/refs.lock` unless you deliberately intend to change pinned third-party commits.
- `vllm` belongs in PDE Slurm jobs for this branch.

## 6. Generate A Slurm Test Job

The project test suite is CPU-only, but PDE policy still requires jobs to run through Slurm rather than as computation on the login node.

Generate the test job:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
python scripts/build_pde_sbatch.py pytest \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-pytest.sbatch
```

Inspect it before submitting:

```bash
sed -n '1,120p' pde-pytest.sbatch
```

It should:

- `cd` into `/local/scratch2/lding43/MAS-Activation-Cascades`
- export `XDG_CACHE_HOME`, `HF_HOME`, and `TRANSFORMERS_CACHE` under scratch
- run `conda run -n cascade python -m pytest tests/`
- not request a GPU for CPU-only tests

Submit it:

```bash
sbatch pde-pytest.sbatch
```

## 7. Watch The Test Job

Check queued or running jobs:

```bash
squeue -u lding43
```

After submission, Slurm normally writes output to a file such as `slurm-<jobid>.out` in the submission directory.

Follow the output:

```bash
tail -f slurm-<jobid>.out
```

Check completed job status:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
```

A passing full test run should end with pytest reporting all tests passed.

## 8. Run One Test File

For a faster check, either submit the generated full-test job or run a short interactive Slurm allocation if PDE policy allows it.

Inside an allocated job shell:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
export XDG_CACHE_HOME=/local/scratch2/lding43/.cache
export HF_HOME=/local/scratch2/lding43/.cache/huggingface
export TRANSFORMERS_CACHE=/local/scratch2/lding43/.cache/huggingface/transformers
conda run -n cascade python -m pytest tests/test_pde_profile.py -q
```

If interactive jobs are not enabled on PDE, create a copy of `pde-pytest.sbatch` and replace the final command with:

```bash
conda run -n cascade python -m pytest tests/test_pde_profile.py -q
```

## 9. Optional GPU Smoke Checks

The unit tests do not require GPU access. Use these only when you want to confirm the allocation and CUDA stack.

Request a Slurm job according to the PDE docs, then run inside the job:

```bash
nvidia-smi
conda run -n cascade python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

Your grant allows up to two GPUs total.

## 10. Optional Experiment Job Scripts

For 8B cascade experiments, generate a one-worker-lane sweep job:

```bash
python scripts/build_pde_sbatch.py sweep \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --steering-vector steering_vectors/harmfulness_llama3_8b.pt \
  --clean-api-base http://clean-vllm-node:8000/v1 > pde-sweep.sbatch
```

For a 70B-class clean vLLM server, generate a two-GPU tensor-parallel job:

```bash
python scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch
```

The helper rejects 70B-class concurrent cascade sweeps by default. With two GPUs total, use tensor parallel for one 70B model at a time unless the experiment is redesigned to run clean and steered generations sequentially.

## 11. Common Failures

`conda: command not found`

Load Conda according to `/usr/local/SLURM`, then rerun setup from scratch.

`No such file or directory: /local/scratch2/lding43/...`

Create the scratch directory and clone or copy the repo there. Do not run from `/home/lding43`.

`ModuleNotFoundError` during tests

Submit the setup job again:

```bash
python scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

`camel-ai` import points to a PyPI install

Run:

```bash
python scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

Slurm job exits quickly with nonzero status

Open `slurm-<jobid>.out`, read the first traceback or shell error, fix that cause, and resubmit.

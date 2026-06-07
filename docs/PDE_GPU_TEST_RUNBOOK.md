# PDE GPU Test Runbook

This runbook explains how to connect to the Emory Math PDE cluster, put the repo and environment in scratch, and run the full project test suite through VS Code Remote SSH.

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
git clone --branch pde-70b-minimal-testing https://github.com/LeoDing-Error/MAS-Activation-Cascades.git MAS-Activation-Cascades
cd MAS-Activation-Cascades
git branch --show-current
```

If the repo already exists:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
git fetch origin
git checkout pde-70b-minimal-testing
git pull origin pde-70b-minimal-testing
git branch --show-current
```

## 3. Keep Conda And Caches In Scratch

Install Miniconda into scratch if it is not already available:

```bash
cd /local/scratch2/lding43
test -x /local/scratch2/lding43/miniconda3/bin/conda || {
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh
  sh miniconda3.sh -b -u -p /local/scratch2/lding43/miniconda3
}
```

Initialize Conda in your current shell:

```bash
CONDA_BASE=/local/scratch2/lding43/miniconda3
if __conda_setup="$("${CONDA_BASE}/bin/conda" shell.bash hook 2> /dev/null)"; then
  eval "$__conda_setup"
elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  export PATH="${CONDA_BASE}/bin:$PATH"
fi
unset __conda_setup
conda --version
```

Then keep Conda environments, packages, Python caches, Hugging Face caches, and temp files in scratch:

```bash
export SCRATCH=/local/scratch2/lding43
mkdir -p "$SCRATCH/.conda/envs" "$SCRATCH/.conda/pkgs" "$SCRATCH/.cache/pip" "$SCRATCH/tmp"

export CONDA_ENVS_PATH="$SCRATCH/.conda/envs"
export CONDA_PKGS_DIRS="$SCRATCH/.conda/pkgs"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/transformers"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TMPDIR="$SCRATCH/tmp"
```

For the 100 GB scratch limit, do not download BF16 70B or FP8 70B checkpoints. The intended 70B path is the GPTQ INT4 checkpoint only, plus the `cascade` Conda environment and generated results.

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
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

Important constraints:

- Do not install `camel-ai` from PyPI.
- Do not update `third_party/refs.lock` unless you deliberately intend to change pinned third-party commits.
- `vllm` belongs in PDE Slurm jobs for this branch.
- PDE GPUs are Blackwell (`sm_120`). The environment must use CUDA 12.8+ compatible PyTorch/vLLM wheels. Do not use the old `torch==2.5.1` / CUDA 12.1 / `vllm==0.6.4` stack.

After setup completes, restore Hugging Face authentication into the scratch-backed cache before running any 70B job:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
export HF_HOME=/local/scratch2/lding43/.cache/huggingface
export TRANSFORMERS_CACHE=/local/scratch2/lding43/.cache/huggingface/transformers
read -rsp "HF token: " HF_TOKEN; echo
conda run -n cascade huggingface-cli login --token "$HF_TOKEN"
unset HF_TOKEN
conda run -n cascade huggingface-cli whoami
```

The token must have access to `hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4`; otherwise the smoke, vector, and cascade jobs will fail at download time.

## 6. Run Tests Through VS Code Remote SSH

The project test suite is CPU-only. Run it from the scratch checkout in a VS Code Remote SSH session, not from `/home/lding43`.

In VS Code, open `/local/scratch2/lding43/MAS-Activation-Cascades` through Remote SSH and select the scratch-local `cascade` interpreter:

```bash
/local/scratch2/lding43/.conda/envs/cascade/bin/python
```

Run the full suite from the VS Code terminal:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
conda run -n cascade python -m pytest tests/
```

You can also use the VS Code Test Explorer after enabling pytest discovery for `tests/`.

Before running tests, keep runtime paths under scratch in the Remote SSH terminal:

```bash
export CONDA_ENVS_PATH=/local/scratch2/lding43/.conda/envs
export CONDA_PKGS_DIRS=/local/scratch2/lding43/.conda/pkgs
export XDG_CACHE_HOME=/local/scratch2/lding43/.cache
export HF_HOME=/local/scratch2/lding43/.cache/huggingface
export TRANSFORMERS_CACHE=/local/scratch2/lding43/.cache/huggingface/transformers
export PIP_CACHE_DIR=/local/scratch2/lding43/.cache/pip
export TMPDIR=/local/scratch2/lding43/tmp
```

## 7. Watch Test Results

For terminal runs, pytest prints progress and failures directly in the VS Code terminal. For VS Code Test Explorer runs, use the test output panel and the Python extension output when discovery fails.

A passing full test run should end with pytest reporting all tests passed.

## 8. Run One Test File

For a faster check from VS Code Remote SSH:

```bash
cd /local/scratch2/lding43/MAS-Activation-Cascades
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
On Blackwell, this smoke check must not print a PyTorch warning that `sm_120` is unsupported. If it does, the environment is incompatible even if `torch.cuda.is_available()` returns true.

## 10. Optional 70B GPU Job Scripts

For a 70B-class clean vLLM server, generate a two-GPU tensor-parallel job:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model meta-llama/Llama-3.1-70B-Instruct > pde-vllm-70b.sbatch
```

Inspect and submit it:

```bash
sed -n '1,140p' pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

It should request two GPUs, set `CUDA_VISIBLE_DEVICES=0,1`, and run `./scripts/serve_clean_model.sh` with `--tensor-parallel-size 2`.

The helper rejects 70B-class concurrent cascade sweeps by default. With two GPUs total, use tensor parallel for one 70B model at a time unless the experiment is redesigned to run clean and steered generations sequentially.

**Storage-constrained alternative (~100 GB scratch):** The BF16 weights are ~140 GB and will not fit on a 100 GB scratch allocation. Use a GPTQ INT4 candidate checkpoint (~38 GB on disk) instead:

```bash
python3 scripts/build_pde_sbatch.py serve-clean \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin > pde-vllm-70b.sbatch
sbatch pde-vllm-70b.sbatch
```

Approximate scratch budget with GPTQ: conda env ~20 GB + 8B BF16 ~16 GB + 70B GPTQ INT4 ~38 GB + caches/results ~10 GB = ~84 GB.

If you are only validating 70B on a wiped server, skip the 8B model download and keep the budget closer to: conda env ~20 GB + 70B GPTQ INT4 ~38 GB + caches/results ~10 GB = ~68 GB.

## 10b. 70B Quantized Cascade Sweep

To run the full cascade with a quantized 70B on both GPUs in one job (clean server on GPU 0, steered worker on GPU 1), pass these gates in order:

1. HF authentication restored under scratch-backed `HF_HOME`.
2. CPU pytest suite passes from VS Code Remote SSH.
3. One-GPU HF Transformers smoke passes on the GPTQ INT4 checkpoint.
4. The 70B steering vector is computed on the GPTQ INT4 checkpoint.
5. Full matrix self-hosted cascade runs with `--resume`.

This GPTQ path is the current validation candidate. Do not run BF16 or FP8 70B under the 100 GB scratch cap.

Render and submit the one-GPU HF smoke job:

```bash
python3 scripts/build_pde_sbatch.py smoke-steered-quant \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --gpu-set 0 > pde-smoke-70b-gptq.sbatch
sbatch pde-smoke-70b-gptq.sbatch
```

The smoke log must print `SMOKE PASS`. If it fails on download/authentication, fix Hugging Face access before continuing. If it fails with an `sm_120` or CUDA/NCCL compatibility warning, rerun setup from this branch and do not continue to the vector or cascade jobs.

After the smoke passes, compute the 70B steering vector on the quantized model:

```bash
python3 scripts/build_pde_sbatch.py compute-vector \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --output steering_vectors/harmfulness_llama3_70b.pt \
  --gpu-set 0 > pde-vector-70b.sbatch
sbatch pde-vector-70b.sbatch
```

The vector job must write `steering_vectors/harmfulness_llama3_70b.pt`. After that file exists, render and submit the full self-hosted, resumable cascade matrix:

```bash
python3 scripts/build_pde_sbatch.py cascade \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades \
  --model hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4 \
  --quantization gptq_marlin \
  --steering-vector steering_vectors/harmfulness_llama3_70b.pt \
  --experiments 1.2,1.3,1.4 --steering-strengths 0.5,1.0,1.5 \
  --task-indices 0,1,2,3,4 --resume > pde-cascade-70b.sbatch
sbatch pde-cascade-70b.sbatch
```

The full matrix is 3 experiments x 3 steering strengths x 5 tasks x 1 repeat = 45 cells. It requests two GPUs, backgrounds the clean vLLM server on GPU 0 with `--tensor-parallel-size 1`, health-checks `http://127.0.0.1:8000/health`, then runs the steered sweep on GPU 1 against `http://127.0.0.1:8000/v1`. Resubmit the same script to resume — finished cells are skipped via a `.cell_complete` sentinel.

## 11. Common Failures

`conda: command not found`

Load Conda according to `/usr/local/SLURM`, then rerun setup from scratch.

`No such file or directory: /local/scratch2/lding43/...`

Create the scratch directory and clone or copy the repo there. Do not run from `/home/lding43`.

`ModuleNotFoundError` during tests

Submit the setup job again:

```bash
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

`camel-ai` import points to a PyPI install

Run:

```bash
python3 scripts/build_pde_sbatch.py setup \
  --netid lding43 \
  --repo-dir /local/scratch2/lding43/MAS-Activation-Cascades > pde-setup.sbatch
sbatch pde-setup.sbatch
```

Slurm job exits quickly with nonzero status

Open `slurm-<jobid>.out`, read the first traceback or shell error, fix that cause, and resubmit.

`sm_120 is not compatible with the current PyTorch installation` or `NCCL error: unhandled cuda error`

The PDE GPUs are Blackwell and require a CUDA 12.8+ compatible PyTorch/vLLM stack. This error means the environment was built with an older stack, commonly `torch==2.5.1` CUDA 12.1 or `vllm==0.6.4`, which only supports up to `sm_90`. Pull the current branch into scratch and rerun the generated setup job so it uses `--cuda128`.

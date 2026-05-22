# GPTQ 70B Cascade Recovery Design

## Context

The quantized 70B cascade branch already contains the two-GPU PDE layout, the combined cascade launcher, and the HF-side steering smoke script. The unresolved problem is empirical: the steered worker and steering-vector computation load the quantized 70B through Hugging Face Transformers, while the clean worker serves the same quantized model through vLLM. Both paths must work on PDE Blackwell GPUs (`sm_120`) under the hard 100 GB scratch cap.

AWQ INT4 fits on disk, but its HF loader requires AWQ packages and may depend on kernels that are not available for `sm_120`. FP8 is a better Blackwell fit technically, but a 70 GB checkpoint is too tight once the conda environment, existing caches, and results are included. GPTQ INT4 is therefore the primary recovery candidate: it fits the scratch budget and has the best chance of loading through Transformers without AWQ-specific compiled kernels.

## Architecture

Use a gated GPTQ-first recovery pipeline. The target runtime architecture stays the same as the existing quantized cascade design:

- GPU 0 runs the clean vLLM OpenAI-compatible server with one single-GPU quantized 70B.
- GPU 1 runs the steered HF Transformers worker with the same single-GPU quantized 70B and a steering hook.
- The full cascade job uses localhost from the steered sweep to the clean server.
- Both sides use the same checkpoint family and quantization scheme.

The scheme candidate changes from AWQ-first to GPTQ-first. AWQ remains a fallback empirical probe only if GPTQ fails. FP8 remains documented as disk-infeasible under the current 100 GB scratch limit unless scratch usage changes materially.

## Components

The recovery should touch only a small set of existing surfaces.

1. `scripts/build_pde_sbatch.py`

   Add a first-class `smoke-steered-quant` subcommand. It should reuse `render_sbatch_script`, request one GPU, export the same scratch/cache environment, and run:

   ```bash
   conda run -n cascade python scripts/smoke_steered_quant.py <model>
   ```

   This removes the need to hand-edit a rendered `compute-vector` job.

2. `scripts/smoke_steered_quant.py`

   Keep the script minimal, but make the log diagnostic enough to support decisions from Slurm output alone. It should print the model name, CUDA device capability, quantization config, relevant package import status for GPTQ loading, and the final `SMOKE PASS` token.

3. Dependency setup

   Add GPTQ loader dependencies only after the exact missing packages are known. Prefer explicit runbook commands first. Move dependencies into setup scripts only after the smoke gate proves they are required and do not disturb the CUDA 12.8+ torch/vLLM stack.

4. Existing vector and cascade scripts

   Keep their roles unchanged. `compute-vector` relies on model config auto-detection through Transformers. `cascade` passes `--quantization` only to the clean vLLM server; the steered worker auto-detects quantization from the checkpoint config.

5. Documentation

   Before the final pilot gate passes, docs should call GPTQ the recommended candidate under validation. After the pilot passes, docs should make GPTQ the headline INT4 path, keep AWQ as possible but unproven on Blackwell HF steering, and keep FP8 as not viable under the current scratch budget.

## Data Flow And Gates

All GPU facts must come from Slurm logs because the GPU allocation is not reachable from an interactive CLI.

1. Synchronize the branch to the scratch checkout.

   The local scratch-limit documentation edits should be preserved and committed in a docs commit near this work, then pulled into `/local/scratch2/lding43/MAS-Activation-Cascades`.

2. Check disk usage before any new model download.

   ```bash
   df -h /local/scratch2
   du -sh /local/scratch2/lding43/.conda /local/scratch2/lding43/.cache/huggingface
   ```

   Gate 1 passes only if there is room for one INT4 70B checkpoint plus logs and results.

3. Verify GPTQ checkpoint metadata.

   Use Hugging Face metadata checks from the scratch environment to confirm the model repo exists, declares GPTQ quantization, and has the expected 70B shape: hidden size 8192 and 80 layers.

4. Install only required GPTQ loader packages.

   Use the `cascade` environment with scratch-backed conda/pip caches. Do not let dependency resolution replace torch, vLLM, or CUDA wheels. If resolution attempts to alter the core stack, stop and use a no-deps or narrower package install only when the missing import is clear.

5. Run the HF steering smoke.

   Submit a one-GPU Slurm job for `scripts/smoke_steered_quant.py` against `hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4`. Gate 3 passes only when the Slurm log contains `SMOKE PASS`.

6. Compute the 70B steering vector.

   Submit `compute-vector` on the same GPTQ model. Gate 4 passes only when `steering_vectors/harmfulness_llama3_70b.pt` exists and an artifact check prints `hidden_size == 8192`, selected layer `< 80`, and `VECTOR OK`.

7. Validate the clean vLLM server.

   Submit a clean-server smoke with the same GPTQ model and `--quantization gptq_marlin`. Gate 5 passes only when the log shows the server reached `/health`.

8. Run a pilot cascade.

   Submit the combined two-GPU cascade job for one small cell: one experiment, one strength, one task, one repeat. Gate 6 passes only when expected reports and `.cell_complete` sentinels are written.

Only after Gate 6 should docs switch GPTQ from candidate to recommended path.

## Error Handling And Fallbacks

Failures should be classified by gate so the next action is mechanical.

- Disk gate failure: do not download another 70B. List large model caches first. Delete only known, intended model caches or stop for a user decision if ownership is unclear.
- GPTQ metadata failure: choose a different GPTQ INT4 repo before installing dependencies or submitting GPU jobs.
- GPTQ dependency install failure: do not compile CUDA kernels on the login node. If dependency resolution wants to replace torch or vLLM, stop and use a narrower install only when the missing import is explicit.
- HF smoke missing-package failure: install the named loader package if it can be done without altering the core stack, then resubmit the smoke.
- HF smoke CUDA kernel failure: treat GPTQ as not viable on the current stack. Probe AWQ only if disk space can hold it or after removing the GPTQ cache.
- HF smoke pass but vector failure: fix activation collection, dtype, or device handling only after preserving the passing smoke evidence. Do not bypass safe `torch.load(..., weights_only=True)` behavior.
- vLLM `gptq_marlin` failure: try `--quantization gptq` only if the installed vLLM version supports it. Otherwise treat clean serving as the blocker.
- Pilot failure after both model gates pass: investigate orchestration, CUDA device visibility, and sweep assumptions before changing quantization.

Every Slurm job involved in the gates should print branch, commit, model, quantization, `nvidia-smi`, torch version, CUDA version, vLLM version where relevant, and cache paths.

## Testing

CPU tests cover deterministic local behavior only:

- the new `smoke-steered-quant` sbatch rendering;
- existing 70B cascade guard behavior;
- docs tests updated when GPTQ becomes the recommended path;
- no CPU test should require a GPU or download a model.

GPU validation is manual and evidence-based:

- HF smoke log contains `SMOKE PASS`;
- vector verification prints `VECTOR OK`;
- vLLM log confirms the clean server reaches `/health`;
- pilot cascade writes expected reports and sentinels.

## Documentation Plan

Documentation should update in two phases.

Before the pilot gate passes:

- preserve and commit the 100 GB scratch-limit warning;
- describe GPTQ as the recommended candidate to validate;
- keep AWQ documented as previously attempted but not proven for HF steering on Blackwell;
- state that FP8 is off the table under current disk constraints.

After the pilot gate passes:

- update `CLAUDE.md`, `AGENTS.md`, `WORKFLOW.md`, `PLAN.md`, and `docs/PDE_GPU_TEST_RUNBOOK.md` so GPTQ INT4 is the headline 70B cascade path;
- demote AWQ to fallback/experimental status;
- keep FP8 described as disk-infeasible unless cache and scratch usage change;
- include the exact model repo and vLLM quantization value that passed the gates.

## Success Criteria

The recovery is complete when:

- the branch has a documented GPTQ-first recovery path;
- the HF smoke, vector compute, clean vLLM smoke, and pilot cascade have Slurm log evidence;
- `steering_vectors/harmfulness_llama3_70b.pt` is computed from the selected quantized 70B;
- docs identify the empirically verified 70B cascade command;
- CPU tests pass without requiring GPU access.

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from src.cluster.pde_profile import (
    build_pde_layout,
    is_70b_class_model,
    render_sbatch_script,
    validate_scratch_path,
)

ROOT = Path(__file__).resolve().parents[1]
PDE_SCRIPT = ROOT / "scripts" / "build_pde_sbatch.py"
_pde_spec = importlib.util.spec_from_file_location("build_pde_sbatch", PDE_SCRIPT)
assert _pde_spec is not None
build_pde_sbatch = importlib.util.module_from_spec(_pde_spec)
assert _pde_spec.loader is not None
sys.modules[_pde_spec.name] = build_pde_sbatch
_pde_spec.loader.exec_module(build_pde_sbatch)


class PdeProfileTests(unittest.TestCase):
    def test_validate_scratch_path_accepts_only_netid_scratch_tree(self) -> None:
        self.assertEqual(
            validate_scratch_path("/local/scratch2/lding/project", "lding"),
            "/local/scratch2/lding/project",
        )

        with self.assertRaisesRegex(ValueError, "/local/scratch2/lding"):
            validate_scratch_path("/home/lding/project", "lding")

        with self.assertRaisesRegex(ValueError, "/local/scratch2/lding"):
            validate_scratch_path("/local/scratch2/lding/../other/project", "lding")

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
            netid="lding",
            repo_dir="/local/scratch2/lding/MAS-Activation-Cascades",
            command=["conda", "run", "-n", "cascade", "python", "-m", "pytest", "tests/"],
            gpu_count=0,
        )

        self.assertIn("#SBATCH --job-name=cascade-tests", script)
        self.assertNotIn("#SBATCH --gres=gpu:", script)
        self.assertIn("mkdir -p /local/scratch2/lding/.conda/envs", script)
        self.assertIn("mkdir -p /local/scratch2/lding/.conda/pkgs", script)
        self.assertIn("mkdir -p /local/scratch2/lding/.cache/pip", script)
        self.assertIn("mkdir -p /local/scratch2/lding/tmp", script)
        self.assertIn("export CONDA_ENVS_PATH=/local/scratch2/lding/.conda/envs", script)
        self.assertIn("export CONDA_PKGS_DIRS=/local/scratch2/lding/.conda/pkgs", script)
        self.assertIn("export XDG_CACHE_HOME=/local/scratch2/lding/.cache", script)
        self.assertIn("export HF_HOME=/local/scratch2/lding/.cache/huggingface", script)
        self.assertIn("export TRANSFORMERS_CACHE=/local/scratch2/lding/.cache/huggingface/transformers", script)
        self.assertIn("export PIP_CACHE_DIR=/local/scratch2/lding/.cache/pip", script)
        self.assertIn("export TMPDIR=/local/scratch2/lding/tmp", script)
        self.assertIn("cd /local/scratch2/lding/MAS-Activation-Cascades", script)
        self.assertIn("conda run -n cascade python -m pytest tests/", script)

    def test_render_sbatch_script_initializes_scratch_miniconda(self) -> None:
        script = render_sbatch_script(
            job_name="cascade-tests",
            netid="lding",
            repo_dir="/local/scratch2/lding/MAS-Activation-Cascades",
            command=["conda", "run", "-n", "cascade", "python", "-m", "pytest", "tests/"],
            gpu_count=0,
        )

        self.assertIn("CONDA_BASE=/local/scratch2/lding/miniconda3", script)
        self.assertIn('"${CONDA_BASE}/bin/conda" shell.bash hook', script)
        self.assertIn('export PATH="${CONDA_BASE}/bin:$PATH"', script)

    def test_render_sbatch_script_rejects_more_than_two_gpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 2 GPUs"):
            render_sbatch_script(
                job_name="too-many-gpus",
                netid="lding",
                repo_dir="/local/scratch2/lding/MAS-Activation-Cascades",
                command=["nvidia-smi"],
                gpu_count=3,
            )

    def test_pytest_cli_renders_cpu_test_job(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "pytest",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
            ]
        )

        self.assertNotIn("#SBATCH --gres=gpu:", script)
        self.assertIn("conda run -n cascade python -m pytest tests/", script)

    def test_setup_cli_renders_scratch_setup_job(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "setup",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
            ]
        )

        self.assertNotIn("#SBATCH --gres=gpu:", script)
        self.assertIn("./scripts/setup_stack.sh --env-name cascade --cuda121", script)

    def test_serve_clean_cli_uses_tensor_parallel_layout_for_70b(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "serve-clean",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
                "--model",
                "meta-llama/Llama-3.1-70B-Instruct",
            ]
        )

        self.assertIn("#SBATCH --gres=gpu:2", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", script)
        self.assertIn("--host 0.0.0.0", script)
        self.assertIn("--tensor-parallel-size 2", script)

    def test_compute_vector_cli_renders_pde_gpu_job(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "compute-vector",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
            ]
        )

        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", script)
        self.assertIn("./scripts/compute_vector_pde.sh --env-name cascade", script)

    def test_compute_vector_cli_accepts_explicit_gpu_set(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "compute-vector",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
                "--gpu-set",
                "1",
            ]
        )

        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", script)

    def test_sweep_cli_rejects_70b_cascade_without_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "70B-class"):
            build_pde_sbatch.render_from_args(
                [
                    "sweep",
                    "--netid",
                    "lding",
                    "--repo-dir",
                    "/local/scratch2/lding/MAS-Activation-Cascades",
                    "--model",
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "--steering-vector",
                    "steering_vectors/harmfulness_llama3_70b.pt",
                    "--clean-api-base",
                    "http://clean-node:8000/v1",
                ]
            )

    def test_sweep_cli_renders_single_worker_lane_for_8b(self) -> None:
        script = build_pde_sbatch.render_from_args(
            [
                "sweep",
                "--netid",
                "lding",
                "--repo-dir",
                "/local/scratch2/lding/MAS-Activation-Cascades",
                "--model",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "--steering-vector",
                "steering_vectors/harmfulness_llama3_8b.pt",
                "--clean-api-base",
                "http://clean-node:8000/v1",
            ]
        )

        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", script)
        self.assertIn("bash ./scripts/run_phase1_sweep.sh", script)
        self.assertIn("--clean-api-bases http://clean-node:8000/v1", script)
        self.assertIn("--worker-gpu-sets 1", script)

    def test_sweep_cli_requires_clean_api_base(self) -> None:
        with self.assertRaises(SystemExit):
            build_pde_sbatch.render_from_args(
                [
                    "sweep",
                    "--netid",
                    "lding",
                    "--repo-dir",
                    "/local/scratch2/lding/MAS-Activation-Cascades",
                    "--model",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "--steering-vector",
                    "steering_vectors/harmfulness_llama3_8b.pt",
                ]
            )

    def test_runbook_covers_required_cluster_access_and_vscode_setup(self) -> None:
        runbook = (ROOT / "docs" / "PDE_GPU_TEST_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn("Emory Unplugged", runbook)
        self.assertIn("Emory VPN", runbook)
        self.assertIn("ssh -J lding43@lab0z.mathcs.emory.edu lding43@pdelogin.mathcs.emory.edu", runbook)
        self.assertIn('"remote.SSH.serverInstallPath"', runbook)
        self.assertIn('"/local/scratch2/lding43"', runbook)
        self.assertIn(".vscode-server", runbook)
        self.assertIn("files.watcherExclude", runbook)
        self.assertIn("python3 scripts/build_pde_sbatch.py setup", runbook)
        self.assertIn("sbatch pde-setup.sbatch", runbook)
        self.assertIn("python3 scripts/build_pde_sbatch.py pytest", runbook)
        self.assertIn("sbatch pde-pytest.sbatch", runbook)

    def test_runbook_matches_generated_scratch_runtime_paths(self) -> None:
        runbook = (ROOT / "docs" / "PDE_GPU_TEST_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn('mkdir -p "$SCRATCH/.conda/envs"', runbook)
        self.assertIn('export CONDA_ENVS_PATH="$SCRATCH/.conda/envs"', runbook)
        self.assertIn('export CONDA_PKGS_DIRS="$SCRATCH/.conda/pkgs"', runbook)
        self.assertIn('export PIP_CACHE_DIR="$SCRATCH/.cache/pip"', runbook)
        self.assertNotIn("$SCRATCH/conda/envs", runbook)
        self.assertNotIn("$SCRATCH/conda/pkgs", runbook)

    def test_cluster_docs_match_generated_scratch_runtime_paths(self) -> None:
        for relative_path in ("README.md", "CLAUDE.md", "WORKFLOW.md"):
            with self.subTest(path=relative_path):
                document = (ROOT / relative_path).read_text(encoding="utf-8")

                self.assertIn('mkdir -p "$SCRATCH/.conda/envs"', document)
                self.assertIn('export CONDA_ENVS_PATH="$SCRATCH/.conda/envs"', document)
                self.assertIn('export CONDA_PKGS_DIRS="$SCRATCH/.conda/pkgs"', document)
                self.assertIn('export PIP_CACHE_DIR="$SCRATCH/.cache/pip"', document)
                self.assertNotIn("$SCRATCH/conda/envs", document)
                self.assertNotIn("$SCRATCH/conda/pkgs", document)

    def test_runbook_keeps_optional_experiments_on_70b_tensor_parallel_path(self) -> None:
        runbook = (ROOT / "docs" / "PDE_GPU_TEST_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn("70B-class clean vLLM server", runbook)
        self.assertIn("meta-llama/Llama-3.1-70B-Instruct", runbook)
        self.assertIn("pde-vllm-70b.sbatch", runbook)
        self.assertNotIn("Meta-Llama-3.1-8B-Instruct", runbook)
        self.assertNotIn("pde-sweep.sbatch", runbook)

    def test_setup_scripts_keep_package_installs_inside_cascade_conda_env(self) -> None:
        setup_env = (ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")
        setup_camel = (ROOT / "scripts" / "setup_camel.sh").read_text(encoding="utf-8")

        self.assertNotRegex(setup_env, r"(^|\s)pip install")
        self.assertIn('pip_in_conda "$ENV_NAME" install -r "$PROJECT_ROOT/requirements.txt"', setup_env)
        self.assertIn('pip_in_conda "$ENV_NAME" uninstall -y camel-ai || true', setup_camel)
        self.assertIn('cd "$CAMEL_DIR"', setup_camel)
        self.assertIn('pip_in_conda "$ENV_NAME" install -e .', setup_camel)

    def test_pde_vllm_and_cuda_torch_stack_are_pinned_compatibly(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        setup_env = (ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")

        self.assertIn('vllm==0.6.4.post1; platform_system == "Linux"', requirements)
        self.assertIn("numpy>=1.26,<2", requirements)
        self.assertIn("torch==2.5.1", setup_env)
        self.assertIn("torchvision==0.20.1", setup_env)
        self.assertIn("torchaudio==2.5.1", setup_env)
        self.assertIn('"fsspec[http]<=2026.2.0,>=2023.1.0"', setup_env)

    def test_clean_vllm_server_disables_frontend_multiprocessing_on_pde(self) -> None:
        serve_script = (ROOT / "scripts" / "serve_clean_model.sh").read_text(encoding="utf-8")

        self.assertIn("--disable-frontend-multiprocessing", serve_script)


if __name__ == "__main__":
    unittest.main()

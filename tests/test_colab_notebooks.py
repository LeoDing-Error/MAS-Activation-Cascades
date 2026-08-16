from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB_BRANCH = "main"
CALIBRATION_COLAB_BRANCH = "agent/steering-calibration-pilot"
NOTEBOOKS = (
    ROOT / "notebooks" / "colab_phase1_quickstart.ipynb",
    ROOT / "notebooks" / "colab_phase1_full_sweep.ipynb",
    ROOT / "notebooks" / "colab_steering_calibration.ipynb",
)
GENERATORS = (
    ROOT / "scripts" / "create_colab_notebook.py",
    ROOT / "scripts" / "create_colab_sweep_notebook.py",
    ROOT / "scripts" / "create_colab_calibration_notebook.py",
)
PHASE1_NOTEBOOK = ROOT / "notebooks" / "run_experiments.ipynb"


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        cell["source"]
        if isinstance(cell["source"], str)
        else "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _generate_to(generator: Path, output: Path) -> None:
    """Generate into a disposable path and prove tracked notebooks stay untouched."""
    before = {path: path.read_bytes() for path in NOTEBOOKS}
    try:
        subprocess.run(
            [sys.executable, str(generator), "--output", str(output)],
            cwd=ROOT,
            check=True,
        )
    finally:
        for path, content in before.items():
            if path.read_bytes() != content:
                path.write_bytes(content)


def test_colab_notebooks_clone_the_colab_workflow_branch() -> None:
    for notebook in NOTEBOOKS[:2]:
        source = _code_source(notebook)
        assert f"COLAB_BRANCH = '{COLAB_BRANCH}'" in source, notebook.name
        assert "'clone', '--branch', COLAB_BRANCH, '--single-branch'" in source, notebook.name


def test_colab_notebooks_update_the_existing_checkout_from_the_same_branch() -> None:
    for notebook in NOTEBOOKS[:2]:
        source = _code_source(notebook)
        assert "'checkout', COLAB_BRANCH" in source, notebook.name
        assert "'pull', '--ff-only', 'origin', COLAB_BRANCH" in source, notebook.name


def test_notebook_generators_preserve_the_colab_branch_checkout() -> None:
    for generator in GENERATORS[:2]:
        source = generator.read_text()
        assert f"COLAB_BRANCH = '{COLAB_BRANCH}'" in source, generator.name
        assert "'clone', '--branch', COLAB_BRANCH, '--single-branch'" in source, generator.name
        assert "'checkout', COLAB_BRANCH" in source, generator.name
        assert "'pull', '--ff-only', 'origin', COLAB_BRANCH" in source, generator.name


def test_quickstart_generator_preserves_current_result_schema_parsing() -> None:
    source = (ROOT / "scripts" / "create_colab_notebook.py").read_text()
    assert "records = data['runs']" in source
    assert "condition.split('alpha_')[1]" in source
    assert "s['metrics']['mean_token_entropy']" in source


def test_generated_notebooks_persist_artifacts_without_repo_symlinks(tmp_path: Path) -> None:
    generated = tuple(tmp_path / notebook.name for notebook in NOTEBOOKS)
    for generator, output in zip(GENERATORS, generated):
        _generate_to(generator, output)
    for tracked, output in zip(NOTEBOOKS, generated):
        assert output.read_bytes() == tracked.read_bytes(), tracked.name

    quickstart = _code_source(generated[0])
    full_sweep = _code_source(generated[1])

    for source in (quickstart, full_sweep):
        assert "os.symlink" not in source
        assert "HF_HOME'] = '/content/hf-cache'" in source
        assert "PAIRS_PATH = f'{DRIVE_DIR}/data/contrastive_pairs/ta2_harmful_pairs.json'" in source
        assert "VECTOR_PATH = f'{DRIVE_DIR}/steering_vectors/harmfulness_llama3_8b.pt'" in source
        assert "LOCAL_VECTOR_PATH = '/content/harmfulness_llama3_8b.pt'" in source

    assert "shutil.copy2(LOCAL_VECTOR_PATH, VECTOR_PATH)" in quickstart
    assert "the confirmed steering artifact is missing" in full_sweep
    assert "src/steering/compute_vectors.py" not in full_sweep

    assert "RESULTS_DIR = f'{DRIVE_DIR}/results'" in quickstart
    assert "'--results-dir', RESULTS_DIR" in quickstart
    assert "SWEEP_RESULTS_ROOT = f'{DRIVE_DIR}/results/sweeps'" in full_sweep
    assert "'--results-root', SWEEP_RESULTS_ROOT" in full_sweep


def test_notebook_generators_are_deterministic_and_do_not_touch_tracked_outputs(tmp_path: Path) -> None:
    before = {path: path.read_bytes() for path in NOTEBOOKS}
    for index, generator in enumerate(GENERATORS):
        first = tmp_path / f"first-{index}.ipynb"
        second = tmp_path / f"second-{index}.ipynb"
        _generate_to(generator, first)
        _generate_to(generator, second)
        assert first.read_bytes() == second.read_bytes(), generator.name
        notebook = json.loads(first.read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids)), generator.name
    assert {path: path.read_bytes() for path in NOTEBOOKS} == before


def test_generated_phase1_workflows_require_held_out_confirmation(tmp_path: Path) -> None:
    generated = tuple(tmp_path / notebook.name for notebook in NOTEBOOKS[:2])
    for generator, output in zip(GENERATORS[:2], generated):
        _generate_to(generator, output)

    quickstart = _code_source(generated[0])
    full_sweep = _code_source(generated[1])
    for source in (quickstart, full_sweep):
        assert "HELD_OUT_CONFIRMATION_PATH" in source
        assert "held_out_confirmation_passed" in source
        assert "'--held-out-confirmation', HELD_OUT_CONFIRMATION_PATH" in source
    assert "'--steering-strength', str(CONFIRMED_ALPHA)" in quickstart
    assert quickstart.count("'--held-out-confirmation', HELD_OUT_CONFIRMATION_PATH") == 2
    assert "'--steering-strengths', str(CONFIRMED_ALPHA)" in full_sweep
    assert "'--steering-strength', '1.0'" not in quickstart
    assert "'--steering-strengths', '0.0,0.5,1.0,1.5,2.0'" not in full_sweep


def test_static_phase1_notebook_requires_held_out_confirmation() -> None:
    source = _code_source(PHASE1_NOTEBOOK)
    assert "HELD_OUT_CONFIRMATION_PATH" in source
    assert "held_out_confirmation_passed" in source
    assert source.count("--held-out-confirmation") == 4
    assert "--steering-strength 1.0" not in source
    assert source.count("--steering-strength $CONFIRMED_ALPHA") == 3


def test_calibration_notebook_uses_drive_and_stops_for_blinded_scoring() -> None:
    notebook = json.loads((ROOT / "notebooks" / "colab_steering_calibration.ipynb").read_text())
    source = _code_source(ROOT / "notebooks" / "colab_steering_calibration.ipynb")
    generator = (ROOT / "scripts" / "create_colab_calibration_notebook.py").read_text()
    assert f"COLAB_BRANCH = '{CALIBRATION_COLAB_BRANCH}'" in source
    assert f"COLAB_BRANCH = '{CALIBRATION_COLAB_BRANCH}'" in generator
    assert "implementation branch" in source
    assert "'clone', '--branch', COLAB_BRANCH, '--single-branch'" in source
    assert "'checkout', COLAB_BRANCH" in source
    assert "'pull', '--ff-only', 'origin', COLAB_BRANCH" in source
    assert "A100-class" in source
    assert "torch.cuda.device_count() != 1" in source
    assert "if 'A100' not in gpu_name.upper()" in source
    assert "CALIBRATION_DIR = f'{DRIVE_DIR}/results/steering_calibration'" in source
    assert "LICENSE_ACCEPTANCE = input(" in source
    assert "Accept the SORRY-Bench license before running prepare." in source
    prepare_cell = next(cell for cell in notebook["cells"] if cell["id"] == "prepare-calibration")
    prepare_source = prepare_cell["source"]
    if isinstance(prepare_source, list):
        prepare_source = "".join(prepare_source)
    guard = "if globals().get('LICENSE_ACCEPTANCE') != 'ACCEPT':"
    assert guard in prepare_source
    assert prepare_source.index(guard) < prepare_source.index("cmd = [")
    assert "run_steering_calibration.py', 'prepare'" in source
    assert "run_steering_calibration.py', 'generate'" in source
    assert "run_steering_calibration.py', 'blind'" in source
    assert "manual_scores.csv" in source
    assert "run_steering_calibration.py', 'summarize'" in source
    assert "baseline_valid" in source
    assert "selected_alpha" in source
    assert "Do not run Experiment 1.2." in source
    assert "run_phase1_sweep.py" not in source
    assert "os.symlink" not in source

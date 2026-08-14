from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLAB_BRANCH = "main"
NOTEBOOKS = (
    ROOT / "notebooks" / "colab_phase1_quickstart.ipynb",
    ROOT / "notebooks" / "colab_phase1_full_sweep.ipynb",
)
GENERATORS = (
    ROOT / "scripts" / "create_colab_notebook.py",
    ROOT / "scripts" / "create_colab_sweep_notebook.py",
)


def _code_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        cell["source"]
        if isinstance(cell["source"], str)
        else "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_colab_notebooks_clone_the_colab_workflow_branch() -> None:
    for notebook in NOTEBOOKS:
        source = _code_source(notebook)
        assert f"COLAB_BRANCH = '{COLAB_BRANCH}'" in source, notebook.name
        assert "'clone', '--branch', COLAB_BRANCH, '--single-branch'" in source, notebook.name


def test_colab_notebooks_update_the_existing_checkout_from_the_same_branch() -> None:
    for notebook in NOTEBOOKS:
        source = _code_source(notebook)
        assert "'checkout', COLAB_BRANCH" in source, notebook.name
        assert "'pull', '--ff-only', 'origin', COLAB_BRANCH" in source, notebook.name


def test_notebook_generators_preserve_the_colab_branch_checkout() -> None:
    for generator in GENERATORS:
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

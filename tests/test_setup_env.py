from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_ENV = ROOT / "scripts" / "setup_env.sh"
REQUIREMENTS = ROOT / "requirements.txt"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_setup_with_platform(tmp_path: Path, platform_name: str) -> list[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "conda-commands.log"

    _write_executable(fake_bin / "uname", f"#!/bin/sh\nprintf '%s\\n' '{platform_name}'\n")
    _write_executable(
        fake_bin / "conda",
        """#!/bin/sh
printf '%s\n' "$*" >> "$CONDA_COMMAND_LOG"
if [ "$1 $2" = "env list" ]; then
  printf '%s\n' 'cascade'
fi
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["CONDA_COMMAND_LOG"] = str(command_log)
    subprocess.run(["bash", str(SETUP_ENV)], cwd=ROOT, env=env, check=True)
    return command_log.read_text().splitlines()


def test_macos_setup_replaces_pip_numeric_wheels_with_conda_forge_builds(tmp_path: Path) -> None:
    commands = _run_setup_with_platform(tmp_path, "Darwin")
    assert "env update -n cascade --file " + str(ROOT / "environment.yml") + " --prune" in commands
    assert "run -n cascade python -m pip uninstall -y torch scikit-learn" in commands
    assert any(
        command
        == "install -y -n cascade --override-channels -c conda-forge pytorch>=2.4 scikit-learn>=1.5.1"
        for command in commands
    )
    requirements_install = (
        "run -n cascade python -m pip install -r " + str(REQUIREMENTS)
    )
    assert commands.count(requirements_install) == 2


def test_linux_setup_does_not_replace_gpu_capable_pytorch(tmp_path: Path) -> None:
    commands = _run_setup_with_platform(tmp_path, "Linux")
    assert not any("install -y -n cascade --override-channels" in command for command in commands)


def test_project_caps_mcp_below_the_incompatible_major_release() -> None:
    requirements = REQUIREMENTS.read_text().splitlines()
    assert "mcp>=1.3,<2" in requirements
    assert "psutil>=5.9.8,<6" in requirements
    assert "ipython<9" in requirements
    assert "fsspec>=2023.1.0,<=2026.6.0" in requirements

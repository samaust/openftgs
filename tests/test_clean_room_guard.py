# SPDX-License-Identifier: Apache-2.0
"""The clean-room guard (`.claude/hooks/clean_room_guard.py`): searches of the repository's own
files may name the forbidden projects, searches anywhere else may not, and the earlier rules on
web searches, fetches, downloads and paths still hold (CLAUDE.md, "Precautions in this
repository"; owner rule of 2026-10-12).

Private: the hook is a repository tool, not part of the openftgs package.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "Free" + "TimeGS"  # assembled, so this file's source doesn't trip searches for the name
CLASS = "Free" + "TimeGaussians"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "clean_room_guard", ROOT / ".claude" / "hooks" / "clean_room_guard.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD: Any = _load()


def _bash(command: str, cwd: Path = ROOT) -> str | None:
    reason: str | None = GUARD.check("Bash", {"command": command}, cwd=str(cwd))
    return reason


@pytest.mark.parametrize(
    "command",
    [
        f"grep -n '{CLASS}\\|from_init_points' src/openftgs/model.py",
        f"grep -rn -i {NAME} src tests docs",
        f"grep -rn -E '{CLASS}|{NAME}' src tests",
        f"rg -n {NAME}",
        f"rg -n '{CLASS}\\b' src -g '*.py'",
        f"git grep -n -i {NAME}",
        f"find src -iname '*{NAME}*'",
        f"cd src && grep -rn {NAME} .",
        f"grep -c {NAME} docs/spec/openftgs-spec.md && echo done",
        f"python3 -c 'from openftgs.model import {CLASS}'",
        f"git commit -m 'Mention {NAME} in a message'",
        "grep -rn hello .",
    ],
)
def test_searches_of_the_repository_may_name_the_projects(command: str) -> None:
    """Owner rule of 2026-10-12: the repository's own files may be searched for the names."""
    assert _bash(command) is None, command


@pytest.mark.parametrize(
    "command",
    [
        f"grep -rn {NAME} .",  # grep doesn't skip .gitignore'd files: it would walk into .venv
        f"grep -rn {NAME} .venv",
        f"find . -iname '*{NAME}*'",
        f"rg -n --no-ignore {NAME}",
        f"rg -uu {NAME} .",
        f"rg {NAME} ~/",
        f"grep -r {NAME} /tmp",
        f"grep -r {NAME} ../",
        f"cd /tmp && grep -r {NAME} .",
        f"grep -r {NAME} $HOME/projects",
        f"curl -s https://example.com/page | grep -i {NAME}",
        f"cat /etc/hosts | grep {NAME}",
        f"grep {NAME} < /etc/hosts",
        f"ls ~ | xargs grep {NAME}",
        f"locate {NAME}",
        f"find / -iname '*{NAME}*'",
        f"ag -g {NAME} /tmp",
        f"git -C /tmp grep {NAME}",
        f"git grep --no-index {NAME} /tmp",
        f"find src -name '*.py' -exec grep -l {NAME} {{}} +",
    ],
)
def test_searches_elsewhere_may_not_name_the_projects(command: str) -> None:
    """A search for the names outside the repository's own files is blocked."""
    assert _bash(command) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        f"git clone https://github.com/someone/{NAME.lower()}",
        f"curl -sL https://example.com/{NAME}.tar.gz -o x.tgz",
        f"pip install {NAME.lower()}",
        f"cat /home/someone/{NAME}/README.md",
        f"gh search repos {NAME}",
    ],
)
def test_downloads_and_paths_naming_the_projects_stay_blocked(command: str) -> None:
    """The rules that came before the repository-search exemption still hold."""
    assert _bash(command) is not None, command


def test_grep_tool_searches_only_the_repository() -> None:
    """The Grep tool may look for the names in the repository, not elsewhere or in .venv."""
    cwd = str(ROOT)
    assert GUARD.check("Grep", {"pattern": NAME, "path": str(ROOT / "src")}, cwd=cwd) is None
    assert GUARD.check("Grep", {"pattern": NAME}, cwd=cwd) is None
    assert GUARD.check("Grep", {"pattern": NAME, "path": "/home"}, cwd=cwd) is not None
    assert GUARD.check("Grep", {"pattern": NAME, "path": str(ROOT / ".venv")}, cwd=cwd) is not None
    assert GUARD.check("Grep", {"pattern": "x", "path": f"/tmp/{NAME}"}, cwd=cwd) is not None


def test_glob_tool_stays_out_of_the_environments() -> None:
    """Glob may look for the names under a repository directory, not across .venv or outside."""
    cwd = str(ROOT)
    pattern = f"**/*{NAME}*"
    assert GUARD.check("Glob", {"pattern": pattern, "path": str(ROOT / "src")}, cwd=cwd) is None
    assert GUARD.check("Glob", {"pattern": pattern}, cwd=cwd) is not None
    assert GUARD.check("Glob", {"pattern": f"/home/**/*{NAME}*"}, cwd=cwd) is not None


def test_web_rules_are_unchanged() -> None:
    """Web searches and fetches naming the projects are blocked, except the project page."""
    assert GUARD.check("WebSearch", {"query": f"{NAME} code"}) is not None
    assert GUARD.check("WebFetch", {"url": f"https://github.com/x/{NAME}"}) is not None
    assert GUARD.check("WebFetch", {"url": f"https://zju3dv.github.io/{NAME.lower()}/"}) is None
    assert GUARD.check("Read", {"file_path": f"/tmp/{NAME}/x.py"}) is not None


def test_hook_process_reads_the_event_and_its_working_directory(tmp_path: Path) -> None:
    """End to end, as Claude Code runs it: the event on stdin, exit code 0 to allow and 2 to
    block, the session's cwd from the event, and blocks logged under the project directory
    (here a temporary one, so the repository's own log is untouched)."""
    (tmp_path / ".claude").mkdir()
    (tmp_path / "src").mkdir()
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(tmp_path)}
    hook = ROOT / ".claude" / "hooks" / "clean_room_guard.py"

    def run(command: str) -> int:
        event = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(tmp_path)}
        done = subprocess.run(
            [sys.executable, str(hook)],
            input=json.dumps(event),
            text=True,
            env=env,
            capture_output=True,
            check=False,
        )
        return done.returncode

    assert run(f"grep -rn '{CLASS}\\|x' src") == 0
    assert run(f"grep -rn {NAME} /tmp") == 2
    log = (tmp_path / ".claude" / "clean-room-guard.log").read_text()
    assert log.count("\n") == 1 and "/tmp" in log

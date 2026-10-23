# SPDX-License-Identifier: Apache-2.0
"""The no-runtime-change check of the docstring work, `scripts/no_runtime_change.py` (PRD #355,
"Added tests for the new code": for every module whose docstrings change, the AST with every
docstring removed equals `main`'s; Q-243; #356).

Added tests, not §3 rows. The source of truth is the file at the git ref: each test commits a
module to a throwaway repository, edits the working tree, and calls the script at its command
line, `main(argv)`.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "no_runtime_change.py"

ORIGINAL = '''\
"""A module."""

LIMIT = 3
"""How many."""


class Box:
    """A box."""

    size: int = 1
    """Its size."""

    def grow(self, by: int = 1) -> int:
        """Grow it."""
        return self.size + by


def scale(x: float) -> float:
    """Scale x."""
    if x > LIMIT:
        "a bare string inside a block"
        return 2 * x
    return x
'''


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("no_runtime_change", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose `main` holds `pkg/mod.py` as ORIGINAL."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(ORIGINAL)
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "add", "pkg/mod.py")
    _git(tmp_path, "commit", "--quiet", "-m", "original")
    return tmp_path


def test_docstring_changes_pass(repo: Path) -> None:
    """Every kind of string statement may change: module, class, method and function docstrings,
    attribute docstrings (which Sphinx reads), and a bare string inside a block."""
    edited = (
        ORIGINAL.replace('"""A module."""', '"""A module.\n\nLonger text.\n"""')
        .replace('"""How many."""', '"""The limit, in items."""')
        .replace('"""A box."""', '"""A box.\n\n    Parameters\n    ----------\n    none\n    """')
        .replace('"""Its size."""', '"""Its size, in cm."""')
        .replace(
            '"""Grow it."""',
            '"""Grow the box.\n\n        Returns\n        -------\n        int\n        """',
        )
        .replace('"""Scale x."""', '"""Scale ``x``."""')
        .replace('"a bare string inside a block"', '"another bare string"')
    )
    assert edited != ORIGINAL
    (repo / "pkg" / "mod.py").write_text(edited)
    assert check.main([str(repo / "pkg" / "mod.py")]) == 0


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("by: int = 1", "by: int = 2"),  # a default
        ("x: float) -> float", "x: int) -> float"),  # an annotation
        ("LIMIT = 3", "LIMIT = 3.0"),  # a constant
        ("return x\n", "return x  # the same\n    return x\n"),  # a statement added
        ('"a bare string inside a block"\n', "pass\n"),  # a string statement made code
        ("return 2 * x", "return str(x)"),  # an expression
    ],
)
def test_code_changes_fail_and_are_named(
    repo: Path, capsys: pytest.CaptureFixture[str], old: str, new: str
) -> None:
    """A change outside the string statements fails, and the output names the file."""
    assert old in ORIGINAL
    path = repo / "pkg" / "mod.py"
    path.write_text(ORIGINAL.replace(old, new, 1))
    assert check.main([str(path)]) == 1
    assert str(path) in capsys.readouterr().out


def test_strings_inside_code_are_compared(repo: Path) -> None:
    """Only string statements are removed: a string that is an expression's value is code."""
    path = repo / "pkg" / "mod.py"
    path.write_text(ORIGINAL + '\n\ndef name() -> str:\n    return "a"\n')
    _git(repo, "commit", "--quiet", "-am", "a function returning a string")
    path.write_text(ORIGINAL + '\n\ndef name() -> str:\n    return "b"\n')
    assert check.main([str(path)]) == 1


def test_only_the_changed_files_are_named(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With several paths, the exit is 1 when one differs, and only that one is named."""
    other = repo / "pkg" / "other.py"
    other.write_text(ORIGINAL)
    _git(repo, "add", "pkg/other.py")
    _git(repo, "commit", "--quiet", "-m", "other")
    other.write_text(ORIGINAL.replace('"""Scale x."""', '"""Scale the value."""'))
    path = repo / "pkg" / "mod.py"
    path.write_text(ORIGINAL.replace("LIMIT = 3", "LIMIT = 4"))
    assert check.main([str(other), str(path)]) == 1
    out = capsys.readouterr().out
    assert str(path) in out and str(other) not in out


def test_main_is_the_default_ref(repo: Path) -> None:
    """On a branch whose commit changed the code, the default compares with `main` and fails;
    `--ref` names the branch instead, which holds the same code."""
    _git(repo, "checkout", "--quiet", "-b", "work")
    path = repo / "pkg" / "mod.py"
    path.write_text(ORIGINAL.replace("LIMIT = 3", "LIMIT = 4"))
    _git(repo, "commit", "--quiet", "-am", "a code change")
    assert check.main([str(path)]) == 1
    assert check.main([str(path), "--ref", "work"]) == 0


def test_invalid_input_is_exit_2(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown ref, a file absent at the ref, a missing file and unparsable Python."""
    path = repo / "pkg" / "mod.py"
    assert check.main([str(path), "--ref", "no-such-ref"]) == 2
    new = repo / "pkg" / "new.py"
    new.write_text(ORIGINAL)
    assert check.main([str(new)]) == 2
    assert check.main([str(repo / "pkg" / "missing.py")]) == 2
    path.write_text(ORIGINAL + "def broken(:\n")
    assert check.main([str(path)]) == 2
    assert "no_runtime_change:" in capsys.readouterr().err


def test_no_path_is_exit_2() -> None:
    """At least one path is required, as argparse reports with exit 2."""
    with pytest.raises(SystemExit) as raised:
        check.main([])
    assert raised.value.code == 2

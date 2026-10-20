# SPDX-License-Identifier: Apache-2.0
"""The test files a branch's changes can reach, so that a run checks those and not the suite.

    affected_tests.py [-C DIR] [--base REF] [--files PATH...]
        prints the affected test files on stdout, one per line, relative to the checkout, or
        `tests` alone when the whole suite is affected; on stderr, what brought each one in

The changes are the files that differ between REF (default `main`) and the working tree: those
`git diff REF...HEAD` lists, uncommitted and untracked ones included, a rename as both its
paths. `--files` names them instead, to ask what a change to those files would reach.

A test file (`tests/**/test_*.py`) is affected when it changed or reaches a changed file, over
these edges from every Python file in the checkout:

- Its imports, relative ones and those inside functions included. A dotted name under
  `openftgs` or `ftgs_harness` is a file under `src/` or `harness/src/`, with its parent
  packages' `__init__.py`; a subpackage (`openftgs.cli`, `openftgs._vendor.romatch`) is every
  file in it, but `openftgs` itself only its `__init__.py`. Any other name is a test helper
  under `tests/` (`builders`, `synthetic_scenes`) or, for an importer outside the packages, a
  file beside it when one has that name, and the stdlib or third-party otherwise.
- Its string literals, docstrings aside: a dotted name under those packages, as
  `importlib.import_module`, `python -m` or `monkeypatch.setattr` take it, and a path, which is
  every file whose path ends with it (`"license_scan.py"`, `"romatch/VENDOR.json"`). A bare
  `"__init__.py"` names none, and a Markdown file is never named: the code cites `docs/` in its
  messages, and no test reads Markdown.

A changed file that isn't Python also reaches every test file under its directory's first level
below `tests/` when it lies deeper there (`tests/synthetic/data/seeds.json` those under
`tests/synthetic/`). `tests/test_skeleton.py`, which walks every source file, is always
affected. The whole suite is affected when a `conftest.py`, `pyproject.toml`,
`harness/pyproject.toml` or this script changed, when a Python file doesn't parse, and when a
changed file under `src/` or `tests/` that isn't Python reaches no test file.

Every choice errs towards running a test. What it can't see is a file a test finds by a name
built at run time (an f-string, a glob) outside such a `tests/<dir>/`.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import deque
from pathlib import Path, PurePosixPath

PACKAGES = {"openftgs": "src/openftgs", "ftgs_harness": "harness/src/ftgs_harness"}
ALWAYS = "tests/test_skeleton.py"
SUITE = "tests"
WHOLE_SUITE = ("pyproject.toml", "harness/pyproject.toml")
SELF = ".claude/skills/implement-without-subagents/scripts/affected_tests.py"


def git(cwd: str | Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def changed_files(root: Path, base: str) -> set[str]:
    committed = git(root, "diff", "--name-only", "--no-renames", f"{base}...HEAD")
    uncommitted = git(root, "diff", "--name-only", "--no-renames", "HEAD")
    untracked = git(root, "ls-files", "--others", "--exclude-standard")
    return {*committed, *uncommitted, *untracked}


def is_test(path: str) -> bool:
    p = PurePosixPath(path)
    return p.parts[0] == "tests" and p.name.startswith("test_") and p.suffix == ".py"


def docstrings(tree: ast.AST) -> set[int]:
    """The ids of the string nodes that are docstrings."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


class Checkout:
    def __init__(self, root: Path, changed: set[str]) -> None:
        self.root = root
        listed = git(root, "ls-files", "--cached", "--others", "--exclude-standard")
        self.present = {f for f in listed if (root / f).is_file()}
        self.files = self.present | changed  # deleted files too, so their importers are found
        self.by_name: dict[str, set[str]] = {}
        for f in self.files:
            self.by_name.setdefault(PurePosixPath(f).name, set()).add(f)

    def module(self, dotted: str) -> set[str]:
        """The files importing `dotted` runs, when it names a module under PACKAGES."""
        parts = dotted.split(".")
        if parts[0] not in PACKAGES:
            return set()
        top = PurePosixPath(PACKAGES[parts[0]])
        path = top.joinpath(*parts[1:])
        if f"{path}.py" in self.files:
            found = {f"{path}.py"}
        elif f"{path}/__init__.py" in self.files:
            whole = len(parts) > 1
            found = {f for f in self.files if f.startswith(f"{path}/")} if whole else set()
            found.add(f"{path}/__init__.py")
        else:
            return set()
        for depth in range(1, len(parts)):  # the parent packages run their __init__.py
            init = str(top.joinpath(*parts[1:depth], "__init__.py"))
            if init in self.files:
                found.add(init)
        return found

    def helper(self, name: str, importer: str) -> set[str]:
        """A top-level name outside PACKAGES: a test helper, or a file beside `importer`."""
        stem = f"{name.split('.')[0]}.py"
        found = {f for f in self.by_name.get(stem, set()) if f.startswith("tests/")}
        if self.package_of(importer):  # in a package, `from types import …` is the stdlib's
            return found
        beside = str(PurePosixPath(importer).parent / stem)
        return found | ({beside} if beside in self.files else set())

    def imported(self, dotted: str, importer: str) -> set[str]:
        if dotted.split(".")[0] in PACKAGES:
            return self.module(dotted)
        return self.helper(dotted, importer) if dotted else set()

    def package_of(self, path: str) -> list[str]:
        """The dotted package a file belongs to, for its relative imports."""
        for name, top in PACKAGES.items():
            if path.startswith(f"{top}/"):
                return [name, *PurePosixPath(path).relative_to(top).parent.parts]
        return []

    def mentions(self, text: str) -> set[str]:
        found: set[str] = set()
        for word in text.replace("\\", "/").split():
            dotted = word.split(".")
            if dotted[0] in PACKAGES and all(part.isidentifier() for part in dotted):
                for end in range(len(dotted), 0, -1):  # a module, or an attribute of one
                    if module := self.module(".".join(dotted[:end])):
                        found |= module
                        break
            parts = [p for p in word.strip("\"'`,;:()[]{}<>").rstrip(".").split("/") if p]
            parts = [p for p in parts if p not in (".", "..")]
            if not parts or parts == ["__init__.py"] or parts[-1].endswith(".md"):
                continue
            for f in self.by_name.get(parts[-1], set()):
                if PurePosixPath(f).parts[-len(parts) :] == tuple(parts):
                    found.add(f)
        return found

    def edges(self, path: str) -> set[str]:
        """The files `path` reaches directly; SyntaxError when it doesn't parse."""
        tree = ast.parse((self.root / path).read_bytes(), filename=path)
        package = self.package_of(path)
        skip = docstrings(tree)
        out: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out |= self.imported(alias.name, path)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - node.level + 1]
                    dotted = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    dotted = node.module or ""
                out |= self.imported(dotted, path)
                for alias in node.names:  # `from openftgs import _stages` imports a module
                    out |= self.module(f"{dotted}.{alias.name}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in skip:
                    out |= self.mentions(node.value)
        out.discard(path)
        return out


def affected(root: Path, changed: set[str]) -> tuple[list[str], list[str]]:
    """The test files to run, or [SUITE], and the lines that say why."""
    for path in sorted(changed):
        if path in WHOLE_SUITE or path == SELF or PurePosixPath(path).name == "conftest.py":
            return [SUITE], [f"the whole suite: {path} changed"]
    checkout = Checkout(root, changed)
    users: dict[str, set[str]] = {}  # file -> the files that reach it directly
    for path in sorted(f for f in checkout.present if f.endswith(".py")):
        try:
            targets = checkout.edges(path)
        except (SyntaxError, ValueError) as error:
            return [SUITE], [f"the whole suite: {path} doesn't parse ({error})"]
        for target in targets:
            users.setdefault(target, set()).add(path)
    tests = sorted(f for f in checkout.present if is_test(f))

    def reach(sources: list[str]) -> dict[str, str | None]:
        """Each file reached, and the next file on a shortest chain back to a source."""
        via: dict[str, str | None] = dict.fromkeys(sources)
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            p = PurePosixPath(node)
            near: list[str] = sorted(users.get(node, ()))
            if p.suffix != ".py" and p.parts[0] == "tests" and len(p.parts) > 2:
                near += [t for t in tests if t.startswith(f"tests/{p.parts[1]}/")]
            for user in near:
                if user not in via:
                    via[user] = node
                    queue.append(user)
        return via

    for path in sorted(changed):
        p = PurePosixPath(path)
        if p.suffix != ".py" and p.parts[0] in ("src", "tests"):
            if not any(is_test(f) and f in checkout.present for f in reach([path])):
                return [SUITE], [f"the whole suite: no test file reaches {path}"]
    via = reach(sorted(changed))
    chosen = [t for t in tests if t in via]
    why = []
    for test in chosen:
        hops = [test]
        while (step := via[hops[-1]]) is not None:
            hops.append(step)
        why.append(" <- ".join(hops) if len(hops) > 1 else f"{test}: changed")
    if ALWAYS not in chosen:
        chosen.append(ALWAYS)
        why.append(f"{ALWAYS}: always")
    return sorted(chosen), why


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="affected_tests.py", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-C", metavar="DIR", default=".", help="the checkout (a worktree)")
    parser.add_argument("--base", default="main", help="the branch the changes count from")
    parser.add_argument("--files", nargs="+", metavar="PATH", help="these changes instead")
    args = parser.parse_args()
    root = Path(git(args.C, "rev-parse", "--show-toplevel")[0])
    changed = set(args.files) if args.files else changed_files(root, args.base)
    tests, why = affected(root, changed)
    print("\n".join(why), file=sys.stderr)
    print("\n".join(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())

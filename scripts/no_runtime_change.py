# SPDX-License-Identifier: Apache-2.0
"""The docstring work's no-runtime-change check (PRD #355; Q-243): each module's code, every string
statement removed, against the same file at a git ref.

    python scripts/no_runtime_change.py [--ref REF] PATH [PATH ...]

For each PATH, a Python file in a git working tree, the file as it is now and the file at REF
(default `main`) are parsed, and every statement that is a bare string constant is removed from
both: module, class and function docstrings, the attribute docstrings Sphinx reads after an
assignment, and any other string statement. What is left must be equal, node for node, so the
change touched docstrings only. Comments aren't in the syntax tree, so they're not compared.

Exit 0 when every file is unchanged, 1 when one differs (each is named), 2 for invalid input: a
missing or unparsable file, an unknown ref, or a file that isn't at the ref.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

_DEFAULT_REF = "main"


class InputError(Exception):
    """Invalid input: exit 2."""


class _StripStrings(ast.NodeTransformer):
    """Remove every statement that is a bare string constant, in every statement list."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and any(isinstance(item, ast.stmt) for item in value):
                setattr(node, field, [item for item in value if not _is_string_statement(item)])
        return node


def _is_string_statement(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _code(source: str, label: str) -> str:
    """The dump of `source`'s syntax tree with every string statement removed."""
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as err:
        raise InputError(f"{label}: not valid Python: {err}") from None
    return ast.dump(_StripStrings().visit(tree))


def _at_ref(path: Path, ref: str) -> str:
    """The text of `path` at `ref`, from the repository that holds it."""
    result = subprocess.run(
        ["git", "show", f"{ref}:./{path.name}"],
        cwd=path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InputError(f"{path}: not readable at {ref}: {result.stderr.strip()}")
    return result.stdout


def unchanged(path: Path, ref: str = _DEFAULT_REF) -> bool:
    """Whether `path`'s code, string statements removed, equals its code at `ref`."""
    try:
        now = path.read_text(encoding="utf-8")
    except OSError as err:
        raise InputError(f"{path}: not readable: {err}") from None
    return _code(now, str(path)) == _code(_at_ref(path, ref), f"{ref}:{path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", type=Path, nargs="+", metavar="PATH")
    parser.add_argument(
        "--ref", default=_DEFAULT_REF, help=f"the git ref to compare with (default {_DEFAULT_REF})"
    )
    args = parser.parse_args(argv)
    try:
        changed = [path for path in args.paths if not unchanged(path, args.ref)]
    except InputError as err:
        print(f"no_runtime_change: {err}", file=sys.stderr)
        return 2
    for path in changed:
        print(f"FAIL  {path}: its code differs from {args.ref}'s, not only its docstrings")
    print(f"{len(args.paths)} files, {len(changed)} changed beyond their docstrings")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())

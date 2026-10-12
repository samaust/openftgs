# SPDX-License-Identifier: Apache-2.0
"""Repository rules for the package layout (§2.1; CLAUDE.md; PRD #1).

Added tests, not §3 rows: every §2.1 module exists, every source file carries the
SPDX header, imports point down the bands, and the harness never imports openftgs.
"""

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "src" / "openftgs"
HARNESS_DIR = ROOT / "harness" / "src" / "ftgs_harness"
SPDX = "# SPDX-License-Identifier: Apache-2.0"

# §2.1 bands, bottom to top; within a band, the fixed order of the owner rule of 2026-10-11.
BANDS: list[list[str]] = [
    ["types", "config", "io", "data", "init"],
    ["motion", "sh", "model", "render"],
    ["optim", "losses", "density"],
    ["eval", "train"],
    ["cli"],
]
ORDER = [name for band in BANDS for name in band]


def _source_files() -> list[Path]:
    files: list[Path] = []
    for top in ("src", "harness/src", "tests", "scripts"):
        files.extend((ROOT / top).rglob("*.py"))
    return sorted(p for p in files if "_vendor" not in p.parts)


def _imported_modules(path: Path) -> set[str]:
    """Absolute module names a file imports, relative imports resolved within openftgs."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("openftgs" + ("." + node.module if node.module else ""))
                if node.module is None:
                    names.update("openftgs." + alias.name for alias in node.names)
            elif node.module is not None:
                names.add(node.module)
                if node.module == "openftgs":
                    names.update("openftgs." + alias.name for alias in node.names)
    return names


def _openftgs_targets(path: Path) -> set[str]:
    """The §2.1 modules a file imports, by short name."""
    targets: set[str] = set()
    for name in _imported_modules(path):
        parts = name.split(".")
        if parts[0] == "openftgs" and len(parts) > 1 and parts[1] in ORDER:
            targets.add(parts[1])
    return targets


@pytest.mark.parametrize("name", ORDER)
def test_module_imports_with_public_names_only_in_all(name: str) -> None:
    module = importlib.import_module(f"openftgs.{name}")
    exported = module.__all__
    assert isinstance(exported, list)
    for public in exported:
        assert not public.startswith("_")
        assert hasattr(module, public)


def test_package_imports() -> None:
    import ftgs_harness
    import openftgs

    assert isinstance(openftgs.__all__, list)
    assert isinstance(ftgs_harness.__all__, list)


def test_every_source_file_starts_with_spdx_header() -> None:
    missing = [
        str(p.relative_to(ROOT))
        for p in _source_files()
        if not p.read_text(encoding="utf-8").startswith(SPDX)
    ]
    assert missing == []


def _module_files(name: str) -> list[Path]:
    """The files of a §2.1 module: name.py, or every file of the subpackage name/."""
    single = PACKAGE_DIR / f"{name}.py"
    return [single] if single.exists() else sorted((PACKAGE_DIR / name).rglob("*.py"))


def test_imports_point_down_the_bands() -> None:
    violations: list[str] = []
    for rank, name in enumerate(ORDER):
        for path in _module_files(name):
            for target in _openftgs_targets(path):
                if target != name and ORDER.index(target) > rank:
                    violations.append(f"{path.relative_to(ROOT)} imports {target}")
    assert violations == []


def test_io_imports_neither_model_nor_train() -> None:
    for path in _module_files("io"):
        assert _openftgs_targets(path) & {"model", "train"} == set()


def test_harness_never_imports_openftgs() -> None:
    offenders = [
        str(p.relative_to(ROOT))
        for p in HARNESS_DIR.rglob("*.py")
        if any(n.split(".")[0] == "openftgs" for n in _imported_modules(p))
    ]
    assert offenders == []

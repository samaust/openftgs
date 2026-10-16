# SPDX-License-Identifier: Apache-2.0
"""U-38 on the CPU: the vendored RoMa in ``openftgs._vendor.romatch`` (D-65; #134).

D-65 as Q-113 and Q-114 amend it: RoMa's 22 inference files at commit 77f8d68, with the
absolute ``romatch`` imports pointing to the vendored path, no OpenCV import, and no
xformers import in the DINOv2 layers. The GPU comparison with upstream romatch is a
separate test (Q-117).
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import openftgs

VENDORED = Path(openftgs.__file__).resolve().parent / "_vendor" / "romatch"


def _vendored_sources() -> list[Path]:
    return sorted(VENDORED.rglob("*.py"))


def _absolute_romatch_imports(path: Path) -> list[int]:
    """Line numbers of imports of ``romatch`` itself, at any depth in the file."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names = [node.module]
        else:
            continue
        if any(name == "romatch" or name.startswith("romatch.") for name in names):
            lines.append(node.lineno)
    return lines


def _vendor_json() -> dict[str, Any]:
    record = json.loads((VENDORED / "VENDOR.json").read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    return record


def test_u38_every_vendored_file_has_the_sha256_of_vendor_json() -> None:
    record = _vendor_json()
    assert record["commit"] == "77f8d68803526dcddfd9b7a46bc76125bdc25f15"  # D-65
    listed = {entry["path"]: entry["sha256"] for entry in record["files"]}
    on_disk = {path.relative_to(VENDORED).as_posix() for path in _vendored_sources()}
    assert len(listed) == 22
    assert set(listed) == on_disk
    actual = {path: hashlib.sha256((VENDORED / path).read_bytes()).hexdigest() for path in listed}
    assert actual == listed


def test_u38_license_files_have_the_sha256_of_vendor_json() -> None:
    # Q-115: RoMa's license at the copy's root, the others in LICENSES beside it.
    record = _vendor_json()
    listed = {entry["path"]: entry["sha256"] for entry in record["licenses"]}
    assert {entry["project"] for entry in record["licenses"]} == {
        "RoMa",
        "DINOv2",
        "DINO",
        "timm",
        "LoFTR",
        "Patch2Pix",
        "Kornia",
    }
    assert "LICENSE" in listed
    on_disk = {path.relative_to(VENDORED).as_posix() for path in (VENDORED / "LICENSES").iterdir()}
    assert set(listed) == on_disk | {"LICENSE"}
    actual = {path: hashlib.sha256((VENDORED / path).read_bytes()).hexdigest() for path in listed}
    assert actual == listed


# Runs in a subprocess, since other tests import cv2 into the pytest process (Q-117).
_IMPORT_WITHOUT_CV2 = """
import importlib.abc
import json
import sys


class _BlockCv2(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("cv2 is blocked by U-38's import hook")
        return None


assert "cv2" not in sys.modules
sys.meta_path.insert(0, _BlockCv2())
import openftgs._vendor.romatch

print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in ("cv2", "xformers"))))
"""

# A stand-in for an installed xformers: importable, with the names the DINOv2 layers ask
# for, and recording any import in a marker file (Q-117).
_STUB_XFORMERS = """
import os
import pathlib

pathlib.Path(os.environ["U38_XFORMERS_MARKER"]).write_text("imported")
"""
_STUB_XFORMERS_OPS = """
fmha = None


def memory_efficient_attention(*args, **kwargs):
    raise NotImplementedError


def unbind(*args, **kwargs):
    raise NotImplementedError


def scaled_index_add(*args, **kwargs):
    raise NotImplementedError


def index_select_cat(*args, **kwargs):
    raise NotImplementedError


class SwiGLU:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
"""


def test_u38_imports_without_cv2_and_never_imports_xformers(tmp_path: Path) -> None:
    stub = tmp_path / "site" / "xformers"
    (stub / "ops").mkdir(parents=True)
    (stub / "__init__.py").write_text(_STUB_XFORMERS, encoding="utf-8")
    (stub / "ops" / "__init__.py").write_text(_STUB_XFORMERS_OPS, encoding="utf-8")
    marker = tmp_path / "xformers-imported"
    source_root = Path(openftgs.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path / "site"), str(source_root)])
    env["U38_XFORMERS_MARKER"] = str(marker)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_CV2],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []
    assert not marker.exists()


def test_u38_no_absolute_romatch_import_in_the_vendored_tree() -> None:
    sources = _vendored_sources()
    assert len(sources) == 22  # D-65's inference files (source log row 116)
    found = {
        str(path.relative_to(VENDORED)): lines
        for path in sources
        if (lines := _absolute_romatch_imports(path))
    }
    assert found == {}

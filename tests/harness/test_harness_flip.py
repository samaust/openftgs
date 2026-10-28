# SPDX-License-Identifier: Apache-2.0
"""The harness's own vendored ꟻLIP, `ftgs_harness._vendor.flip` (§3.3 L3 step 3, §2.1; Q-254;
#391).

L3 computes ꟻLIP itself from the saved renders, and the harness never imports openftgs, so it
carries its own copy of NVIDIA's `src/pytorch/flip_loss.py` at commit d3ab02a: the published
file with the device-agnostic changes only, so that it runs on the CPU, its BSD-3-Clause
license file in the package data, and its own provenance record. `scripts/vendor_flip.py
--target harness` makes it from U-23's reference copy (Q-252), as D-65's pattern makes
openftgs's copy, and its check mode reproduces it offline.
"""

import difflib
import hashlib
import importlib.util
import json
import shutil
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import ftgs_harness

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests" / "reference" / "flip"  # Q-252
COPY = Path(ftgs_harness.__file__).resolve().parent / "_vendor" / "flip"
SCRIPT = ROOT / "scripts" / "vendor_flip.py"
COMMIT = "d3ab02a4425eb167d8a4021b382a2bafc6e6c838"
# Source log rows 425 and 444: the published flip_loss.py and LICENSE at COMMIT.
PUBLISHED_SHA256 = "bf2b83aeb2427a960abe1143d4436d48609f1cc4f256e685daa08d86cc07c4fc"
LICENSE_SHA256 = "13b955078ffb4a3215757038ab2e5fcf0cc66349d0990faec9adba8ad034e578"
MARKER = "# ftgs_harness (device)"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vendor_flip", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record() -> dict[str, Any]:
    record = json.loads((COPY / "VENDOR.json").read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    return record


def test_harness_flip_has_the_sha256_of_its_provenance_record() -> None:
    """Q-254, D-65's pattern: the harness's record names its own package, the commit and the
    published file's SHA-256, and the copy and its BSD-3-Clause license on disk have the
    SHA-256 it records; the license is NVIDIA's, unchanged."""
    record = _record()
    assert record["commit"] == COMMIT
    assert record["package"] == "ftgs_harness._vendor.flip"
    (entry,) = record["files"]
    assert (entry["path"], entry["upstream_sha256"]) == ("flip_loss.py", PUBLISHED_SHA256)
    (license_entry,) = record["licenses"]
    assert (license_entry["path"], license_entry["license"]) == ("LICENSE", "BSD-3-Clause")
    assert license_entry["sha256"] == LICENSE_SHA256
    for listed in (entry, license_entry):
        data = (COPY / listed["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == listed["sha256"], listed["path"]


def test_harness_flip_records_the_device_change_only() -> None:
    """Q-254: the copy carries the device-agnostic changes and nothing else. Its record
    describes one change, `device`, and lists every edit under it (with the notice); the copy
    names CUDA nowhere; and every line that isn't the published file's carries the marker of
    the device change or belongs to the modification notice."""
    record = _record()
    assert set(record["changes"]) == {"device"}
    (entry,) = record["files"]
    kinds = {edit.split(":", 1)[0] for edit in entry["edits"]}
    assert kinds == {"device", "notice"}
    published = (REFERENCE / "flip_loss.py").read_text(encoding="utf-8").split("\n")
    copy = (COPY / "flip_loss.py").read_text(encoding="utf-8")
    for cuda in (".cuda(", "'cuda'", '"cuda"'):
        assert cuda not in copy
    lines = copy.split("\n")
    matcher = difflib.SequenceMatcher(None, published, lines, autojunk=False)
    added = [
        line
        for tag, _, _, first, last in matcher.get_opcodes()
        if tag in ("insert", "replace")
        for line in lines[first:last]
    ]
    notice = [line for line in added if not line.endswith(MARKER)]
    assert added and notice and all(line.startswith("#") or not line for line in notice)
    assert any("ftgs_harness" in line for line in notice)


def test_vendoring_script_check_mode_reproduces_the_harness_flip(tmp_path: Path) -> None:
    """D-65's pattern, Q-254: the script's check mode rebuilds the harness's copy from the
    reference copy, offline, and finds it byte for byte the same; a copy changed by one byte
    fails the check, and a changed reference copy stops the script with exit code 2."""
    script = _script()
    harness = ["--target", "harness", "--check", "--reference"]
    assert script.main([*harness, str(REFERENCE), "--dest", str(COPY)]) == 0

    changed = tmp_path / "copy"
    shutil.copytree(COPY, changed, ignore=shutil.ignore_patterns("__pycache__"))
    source = changed / "flip_loss.py"
    source.write_bytes(source.read_bytes() + b"\n")
    assert script.main([*harness, str(REFERENCE), "--dest", str(changed)]) == 1

    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference)
    (reference / "flip_loss.py").write_bytes((REFERENCE / "flip_loss.py").read_bytes() + b"#")
    assert script.main([*harness, str(reference), "--dest", str(COPY)]) == 2


def test_harness_flip_license_and_record_are_package_data() -> None:
    """Q-254: the license file and the provenance record ship in the harness's wheel beside
    the copy."""
    project = tomllib.loads((ROOT / "harness" / "pyproject.toml").read_text(encoding="utf-8"))
    data = project["tool"]["setuptools"]["package-data"]["ftgs_harness._vendor.flip"]
    assert {"LICENSE", "VENDOR.json"} <= set(data)

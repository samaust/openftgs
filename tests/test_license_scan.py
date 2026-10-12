# SPDX-License-Identifier: Apache-2.0
"""License scan rules (§2.3 notes; Q-4; PRD #1).

Added tests, not §3 rows: GPL and AGPL fail in any spelling, LGPL and unknown
licenses pass but are listed for the owner's sign-off, file-level copyleft
(MPL-2.0) and permissive licenses pass, and albumentations must be MIT.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "license_scan.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("license_scan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scan = _load()


def _row(name: str, license_: str) -> dict[str, str]:
    return {"Name": name, "Version": "1.0", "License": license_}


@pytest.mark.parametrize(
    "license_",
    [
        "GPL-3.0",
        "GPL-2.0-or-later",
        "AGPL-3.0",
        "GNU General Public License v2 (GPLv2)",
        "GNU Affero General Public License v3",
        "MIT; GPL-3.0",
    ],
)
def test_gpl_and_agpl_fail(license_: str) -> None:
    result = scan.classify([_row("pkg", license_)])
    assert [r["Name"] for r in result.failures] == ["pkg"]


@pytest.mark.parametrize(
    "license_",
    [
        "LGPL-2.1",
        "LGPL-3.0-or-later",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "GNU Library or Lesser General Public License (LGPL)",
        "UNKNOWN",
        "",
    ],
)
def test_lgpl_and_unknown_pass_but_need_signoff(license_: str) -> None:
    result = scan.classify([_row("pkg", license_)])
    assert result.failures == []
    assert [r["Name"] for r in result.signoff] == ["pkg"]


@pytest.mark.parametrize(
    "license_",
    [
        "MIT",
        "MIT License",
        "BSD-3-Clause",
        "BSD License",
        "Apache-2.0",
        "Apache Software License",
        "MPL-2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "MPL-2.0 AND MIT",
        "PSF-2.0",
        "MIT-CMU",
    ],
)
def test_permissive_and_mpl_pass(license_: str) -> None:
    result = scan.classify([_row("pkg", license_)])
    assert result.failures == []
    assert result.signoff == []


def test_albumentations_must_be_mit() -> None:
    assert scan.classify([_row("albumentations", "MIT")]).failures == []
    bad = scan.classify([_row("albumentations", "AGPL-3.0; Commercial")])
    assert [r["Name"] for r in bad.failures] == ["albumentations"]
    bad = scan.classify([_row("Albumentations", "BSD-3-Clause")])
    assert [r["Name"] for r in bad.failures] == ["Albumentations"]


def test_main_exit_codes_from_json_file(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps([_row("a", "MIT"), _row("b", "LGPL-2.1")]))
    assert scan.main([str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([_row("a", "MIT"), _row("c", "GPL-3.0")]))
    assert scan.main([str(bad)]) == 1

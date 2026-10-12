# SPDX-License-Identifier: Apache-2.0
"""License scan rules (§2.3 notes; Q-4; PRD #1).

Added tests, not §3 rows. Every license string is judged term by term:
GPL, AGPL and non-commercial terms fail outright; LGPL, proprietary, unknown and
unrecognised terms need a sign-off from the committed sign-off file; permissive
licenses and MPL-2.0 pass. An OR choice takes its most favourable option, and a
distribution takes its worst term. albumentations must be MIT.
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


def _row(name: str, license_: str, version: str = "1.0") -> dict[str, str]:
    return {"Name": name, "Version": version, "License": license_}


def _entry(name: str, version: str, license_: str) -> dict[str, str]:
    return {
        "name": name,
        "version": version,
        "license": license_,
        "actual": "Apache-2.0",
        "reason": "test",
        "source": "test",
        "date": "2026-10-11",
    }


def _names(rows: list[dict[str, str]]) -> list[str]:
    return [r["Name"] for r in rows]


@pytest.mark.parametrize(
    "license_",
    [
        "GPL-3.0",
        "GPL-2.0-or-later",
        "AGPL-3.0",
        "GNU General Public License v2 (GPLv2)",
        "GNU Affero General Public License v3",
        "MIT; GPL-3.0",
        "MIT AND GPL-3.0-only",
        "CC-BY-NC-4.0",
        "Creative Commons Attribution-NonCommercial 4.0",
        "Non-Commercial Research License",
        "MIT AND CC-BY-NC-SA-4.0",
        "GPL-2.0 or later",
        "GPLv3 or later",
        "GNU General Public License v3 or later",
        "GPL-2.0 or any later version",
    ],
)
def test_gpl_agpl_and_non_commercial_fail(license_: str) -> None:
    result = scan.classify([_row("pkg", license_)], [])
    assert _names(result.failures) == ["pkg"]


def test_failure_cannot_be_signed_off() -> None:
    result = scan.classify([_row("pkg", "GPL-3.0")], [_entry("pkg", "1.0", "GPL-3.0")])
    assert _names(result.failures) == ["pkg"]
    assert result.approved == []


@pytest.mark.parametrize(
    "license_",
    [
        "LGPL-2.1",
        "LGPL-3.0-or-later",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "GNU Lesser General Public License",
        "GNU Library or Lesser General Public License (LGPL)",
        "GNU Library General Public License v2 (LGPLv2)",
        "Lesser GPL",
        "UNKNOWN",
        "",
        "LicenseRef-NVIDIA-Proprietary",
        "Other/Proprietary License",
        "NVIDIA Proprietary Software",
        "MPL-1.1",
        "Some Custom License",
        "MIT AND LicenseRef-Proprietary",
        "BSD-3-Clause AND UNKNOWN",
        "LGPL-2.1 or later",
        "GNU Lesser General Public License v2 or later (LGPLv2+)",
    ],
)
def test_lgpl_proprietary_and_unrecognised_need_signoff(license_: str) -> None:
    result = scan.classify([_row("pkg", license_)], [])
    assert result.failures == []
    assert _names(result.unapproved) == ["pkg"]


@pytest.mark.parametrize(
    "license_",
    [
        "MIT",
        "MIT License",
        "MIT-CMU",
        "BSD",
        "BSD License",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "Apache-2.0",
        "Apache Software License",
        "ISC",
        "PSF-2.0",
        "Python Software Foundation License",
        "Zlib",
        "BSL-1.0",
        "CC0-1.0",
        "HPND",
        "Historical Permission Notice and Disclaimer (HPND)",
        "Apache Software License 2.0",
        "MPL-2.0",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "MPL-2.0 AND MIT",
        "Apache Software License; BSD License",
        "Apache-2.0 OR BSD-2-Clause",
        "MIT OR GPL-3.0",
        "(MIT OR GPL-3.0) AND BSD-3-Clause",
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause "
        "AND BSL-1.0 AND MIT",
    ],
)
def test_permissive_and_mpl2_pass(license_: str) -> None:
    result = scan.classify([_row("pkg", license_)], [])
    assert result.failures == []
    assert result.unapproved == []


def test_albumentations_must_be_mit() -> None:
    assert scan.classify([_row("albumentations", "MIT")], []).failures == []
    bad = scan.classify([_row("albumentations", "AGPL-3.0; Commercial")], [])
    assert _names(bad.failures) == ["albumentations"]
    bad = scan.classify([_row("Albumentations", "BSD-3-Clause")], [])
    assert _names(bad.failures) == ["Albumentations"]


def test_signoff_entry_approves_matching_row() -> None:
    entry = _entry("gsplat", "1.5.3", "UNKNOWN")
    result = scan.classify([_row("gsplat", "UNKNOWN", "1.5.3")], [entry])
    assert result.unapproved == []
    assert _names(result.approved) == ["gsplat"]
    assert result.stale == []


@pytest.mark.parametrize(
    "row",
    [
        {"Name": "gsplat", "Version": "1.5.4", "License": "UNKNOWN"},
        {"Name": "gsplat", "Version": "1.5.3", "License": "Some Custom License"},
        {"Name": "other", "Version": "1.5.3", "License": "UNKNOWN"},
    ],
)
def test_signoff_entry_must_match_name_version_and_license(row: dict[str, str]) -> None:
    result = scan.classify([row], [_entry("gsplat", "1.5.3", "UNKNOWN")])
    assert _names(result.unapproved) == [row["Name"]]
    assert len(result.stale) == 1


def test_signoff_name_is_normalized_and_version_may_be_a_pattern() -> None:
    entry = _entry("nvidia-cuda-runtime", "13.*", "LicenseRef-NVIDIA-Proprietary")
    row = _row("NVIDIA_Cuda.Runtime", "LicenseRef-NVIDIA-Proprietary", "13.0.96")
    result = scan.classify([row], [entry])
    assert result.unapproved == []


def test_stale_entry_is_reported() -> None:
    result = scan.classify([_row("pkg", "MIT")], [_entry("gone", "1.0", "UNKNOWN")])
    assert [e["name"] for e in result.stale] == ["gone"]


def _write(path: Path, data: object) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_main_exit_codes(tmp_path: Path) -> None:
    empty = _write(tmp_path / "empty.json", [])
    signoff = _write(tmp_path / "signoff.json", [_entry("b", "1.0", "LGPL-2.1")])
    good = _write(tmp_path / "good.json", [_row("a", "MIT"), _row("b", "LGPL-2.1")])
    assert scan.main(["--rows", good, "--signoff", signoff]) == 0
    assert scan.main(["--rows", good, "--signoff", empty]) == 1
    bad = _write(tmp_path / "bad.json", [_row("a", "MIT"), _row("c", "GPL-3.0")])
    assert scan.main(["--rows", bad, "--signoff", signoff]) == 1


def test_committed_signoff_file_is_valid() -> None:
    entries = scan.load_signoff(scan.DEFAULT_SIGNOFF)
    required = {"name", "version", "license", "actual", "reason", "source", "date"}
    for entry in entries:
        assert required <= set(entry)
        assert scan.judge(entry["license"]) is scan.Outcome.SIGNOFF

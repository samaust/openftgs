# SPDX-License-Identifier: Apache-2.0
"""Marker handling of tests/conftest.py (§3 introduction; PRD #1).

Added tests, not §3 rows: unmarked tests get the cpu marker, and gpu, dataset and
weights tests are skipped when their resource is missing.
"""

import tomllib
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
CONFTEST = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
# The marker definitions come from pyproject.toml, so they are written in one place.
MARKERS: list[str] = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
    "pytest"
]["ini_options"]["markers"]
INI = "[pytest]\naddopts = --strict-markers\nmarkers =\n" + "".join(
    f"    {marker}\n" for marker in MARKERS
)
TESTS = """
import pytest

def test_plain(): pass

@pytest.mark.gpu
def test_gpu(): pass

@pytest.mark.dataset
def test_dataset(): pass

@pytest.mark.weights
def test_weights(): pass
"""


def test_pyproject_defines_the_four_markers() -> None:
    assert sorted(m.split(":")[0] for m in MARKERS) == ["cpu", "dataset", "gpu", "weights"]


@pytest.fixture
def suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    monkeypatch.delenv("OPENFTGS_DATASETS", raising=False)
    monkeypatch.delenv("OPENFTGS_ALLOW_DOWNLOADS", raising=False)
    pytester.makeconftest(CONFTEST)
    pytester.makeini(INI)
    pytester.makepyfile(TESTS)
    return pytester


def test_unmarked_tests_get_cpu_marker(suite: pytest.Pytester) -> None:
    result = suite.runpytest_subprocess("-m", "cpu", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, skipped=2, deselected=1)


def test_missing_resources_skip(suite: pytest.Pytester) -> None:
    import torch

    result = suite.runpytest_subprocess("-rs", "-p", "no:cacheprovider")
    if torch.cuda.is_available():
        result.assert_outcomes(passed=2, skipped=2)
    else:
        result.assert_outcomes(passed=1, skipped=3)
        result.stdout.fnmatch_lines(["*needs a CUDA device*"])
    result.stdout.fnmatch_lines(["*OPENFTGS_DATASETS*", "*OPENFTGS_ALLOW_DOWNLOADS*"])


def test_resources_present_run(
    suite: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENFTGS_DATASETS", str(tmp_path))
    monkeypatch.setenv("OPENFTGS_ALLOW_DOWNLOADS", "1")
    result = suite.runpytest_subprocess("-m", "not gpu", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=3, deselected=1)

# SPDX-License-Identifier: Apache-2.0
"""The local coverage script's badge (Q-217).

Added tests, not §3 rows. The badge reads the total as coverage formats it, its color
steps down from green at 90 % to red under 50 %, and it is well-formed SVG.
"""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "coverage_local.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coverage_local", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


local = _load()

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize(
    ("percent", "color"),
    [
        (100.0, "#4c1"),
        (90.0, "#4c1"),
        (89.9, "#97ca00"),
        (80.0, "#97ca00"),
        (70.0, "#a4a61d"),
        (60.0, "#dfb317"),
        (50.0, "#fe7d37"),
        (49.9, "#e05d44"),
        (0.0, "#e05d44"),
    ],
)
def test_the_badge_color_steps_down_with_the_total(percent: float, color: str) -> None:
    assert local.badge_color(percent) == color


def test_the_badge_is_svg_reading_the_total() -> None:
    root = ET.fromstring(local.badge_svg("83"))
    assert root.tag == f"{SVG}svg"
    texts = [t.text for t in root.iter(f"{SVG}text")]
    assert texts == ["coverage", "83%"]
    fills = [r.get("fill") for r in root.iter(f"{SVG}rect") if r.get("fill")]
    assert fills == ["#555", "#97ca00"]
    title = root.find(f"{SVG}title")
    assert title is not None and title.text == "coverage of the CPU and GPU suites: 83%"


def test_the_badge_widens_with_its_text() -> None:
    narrow = ET.fromstring(local.badge_svg("9"))
    wide = ET.fromstring(local.badge_svg("100"))
    assert int(wide.get("width", "0")) > int(narrow.get("width", "0"))

# SPDX-License-Identifier: Apache-2.0
"""Coverage of the CPU and GPU suites on this machine, and the README's badge (Q-217).

Runs the CPU-marked tests in parallel (`-n auto --dist loadgroup`, Q-214), then the
GPU-marked tests serially, both under pytest-cov with the dataset and weights tests left
out, into one data file (`[tool.coverage.run]` in pyproject.toml, Q-216). Then writes
htmlcov/ and coverage.xml, prints coverage's markdown table, and, when both runs pass,
writes the total to the badge, docs/coverage.svg. With --badge-only it writes the badge
from the existing data file without running the tests.

Exit code 0 when both runs pass, 2 without a CUDA device, otherwise the failing run's
pytest exit code; the badge is then left as it was.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
BADGE = ROOT / "docs" / "coverage.svg"

_SCOPE = "not dataset and not weights"
CPU_RUN = ["-n", "auto", "--dist", "loadgroup", "-m", f"not gpu and {_SCOPE}", "--cov"]
GPU_RUN = ["-m", f"gpu and {_SCOPE}", "--cov", "--cov-append"]

# The lowest total of each color, from green down to red.
_COLORS = (
    (90.0, "#4c1"),
    (80.0, "#97ca00"),
    (70.0, "#a4a61d"),
    (60.0, "#dfb317"),
    (50.0, "#fe7d37"),
    (0.0, "#e05d44"),
)


def badge_color(percent: float) -> str:
    """The badge's color for a total in percent."""
    return next(color for floor, color in _COLORS if percent >= floor)


def _text_width(text: str) -> int:
    # About 7 px per character at 11 px Verdana, with 5 px of padding on each side.
    return 7 * len(text) + 10


def badge_svg(total: str) -> str:
    """A flat badge reading "coverage | <total>%", total as coverage formats it."""
    label, value = "coverage", f"{total}%"
    left, right = _text_width(label), _text_width(value)
    width = left + right
    title = escape(f"coverage of the CPU and GPU suites: {value}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" '
        f'aria-label="{title}">\n'
        f"  <title>{title}</title>\n"
        f'  <clipPath id="r"><rect width="{width}" height="20" rx="3"/></clipPath>\n'
        f'  <g clip-path="url(#r)">\n'
        f'    <rect width="{left}" height="20" fill="#555"/>\n'
        f'    <rect x="{left}" width="{right}" height="20" fill="{badge_color(float(total))}"/>\n'
        "  </g>\n"
        '  <g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11" '
        'text-anchor="middle">\n'
        f'    <text x="{left / 2}" y="14" textLength="{left - 10}">{label}</text>\n'
        f'    <text x="{left + right / 2}" y="14" textLength="{right - 10}">{value}</text>\n'
        "  </g>\n"
        "</svg>\n"
    )


def _pytest(args: list[str]) -> int:
    command = [sys.executable, "-m", "pytest", "-q", *args, "--cov-report="]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def _coverage(*args: str) -> str:
    command = [sys.executable, "-m", "coverage", *args]
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _cuda_available() -> bool:
    probe = "import torch; print(torch.cuda.is_available())"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout
    return out.strip() == "True"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--badge-only", action="store_true", help="write the badge from the existing data file"
    )
    args = parser.parse_args(argv)
    if not args.badge_only:
        if not _cuda_available():
            print("coverage_local: no CUDA device; the GPU-marked tests can't run", file=sys.stderr)
            return 2
        codes = [_pytest(CPU_RUN), _pytest(GPU_RUN)]
        _coverage("html")
        _coverage("xml")
        print(_coverage("report", "--format=markdown"))
        failed = next((code for code in codes if code != 0), 0)
        if failed:
            print("coverage_local: a test run failed; the badge is unchanged", file=sys.stderr)
            return failed
    total = _coverage("report", "--format=total").strip()
    BADGE.write_text(badge_svg(total))
    print(f"coverage_local: {total}% written to {BADGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

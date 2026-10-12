# SPDX-License-Identifier: Apache-2.0
"""License scan of the resolved environment (§2.3 notes; Q-4).

Runs pip-licenses (or reads its JSON output from --rows) and judges every installed
distribution's license string term by term. Terms are joined by AND, ";" or WITH (an
exception only adds permissions, so the base license is judged); an OR choice takes
its most favourable option, and a distribution takes its worst term.

- Fail, no sign-off possible: GPL, AGPL, non-commercial; albumentations unless MIT.
- Needs sign-off: LGPL, proprietary, unknown, unrecognised. The scan fails unless an
  entry of the sign-off file matches the name, version (a glob) and reported license.
- Pass: the permissive licenses below and MPL-2.0 only.

Exit code 0 when nothing fails and every sign-off row is approved, 1 otherwise.
Sign-off entries that match no row are reported as stale.
"""

from __future__ import annotations

import argparse
import enum
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SIGNOFF = Path(__file__).resolve().with_name("license_signoff.json")


class Outcome(enum.IntEnum):
    """Ordered from best to worst, so a distribution takes the max over its terms."""

    PASS = 0
    SIGNOFF = 1
    FAIL = 2


_NON_COMMERCIAL = re.compile(r"\bNC\b|non[\s-]?commercial", re.IGNORECASE)
_LESSER = re.compile(r"\bLGPL|\bLesser\b|\bLibrary\b", re.IGNORECASE)
_GPL = re.compile(r"\bA?GPL|General\s+Public\s+License|Affero", re.IGNORECASE)
_PERMISSIVE = re.compile(
    r"MIT(?:-CMU|-0)?(?:\s+License)?"
    r"|(?:new\s+|simplified\s+|modified\s+)?BSD(?:[-\s][234]-Clause)?(?:\s+License)?|0BSD"
    r"|Apache(?:-2\.0|\s+2\.0|\s+Software\s+License|\s+License\s+2\.0"
    r"|\s+License,\s+Version\s+2\.0)"
    r"|ISC(?:\s+License(?:\s+\(ISCL\))?)?"
    r"|PSF(?:-2\.0)?|Python\s+Software\s+Foundation\s+License"
    r"|Zlib|zlib/libpng"
    r"|BSL-1\.0|Boost\s+Software\s+License\s+1\.0(?:\s+\(BSL-1\.0\))?"
    r"|CC0-1\.0|CC0\s+1\.0\s+Universal(?:\s+\(CC0\s+1\.0\))?(?:\s+Public\s+Domain\s+Dedication)?"
    r"|HPND|Historical\s+Permission\s+Notice\s+and\s+Disclaimer(?:\s+\(HPND\))?"
    r"|(?:The\s+)?Unlicense(?:\s+\(Unlicense\))?"
    r"|MPL-2\.0|Mozilla\s+Public\s+License\s+2\.0(?:\s+\(MPL\s+2\.0\))?",
    re.IGNORECASE,
)
_UNKNOWN = {"", "unknown", "none"}


def _split_top_level(text: str, separator: re.Pattern[str]) -> list[str]:
    """Split on a separator outside parentheses."""
    parts: list[str] = []
    depth = 0
    start = 0
    for match in re.finditer(r"[()]|" + separator.pattern, text, separator.flags):
        token = match.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0:
            parts.append(text[start : match.start()])
            start = match.end()
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


_AND = re.compile(r"\s+AND\s+|;", re.IGNORECASE)
_OR = re.compile(r"\s+OR\s+", re.IGNORECASE)


def _strip_parentheses(text: str) -> str:
    """Remove parentheses that enclose the whole expression."""
    while text.startswith("(") and _closing_paren(text, 0) == len(text) - 1:
        text = text[1:-1].strip()
    return text


def _closing_paren(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        depth += {"(": 1, ")": -1}.get(text[index], 0)
        if depth == 0:
            return index
    return -1


def _judge_atom(atom: str) -> Outcome:
    base = re.split(r"\s+WITH\s+", atom, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if _NON_COMMERCIAL.search(base):
        return Outcome.FAIL
    if _LESSER.search(base):
        return Outcome.SIGNOFF
    if _GPL.search(base):
        return Outcome.FAIL
    if _PERMISSIVE.fullmatch(base):
        return Outcome.PASS
    return Outcome.SIGNOFF


def judge(license_: str) -> Outcome:
    """The outcome of one license string, the worst over its terms."""
    text = _strip_parentheses(license_.strip())
    if text.lower() in _UNKNOWN:
        return Outcome.SIGNOFF
    terms = _split_top_level(text, _AND)
    if len(terms) > 1:
        return max(judge(term) for term in terms)
    options = _split_top_level(text, _OR)
    if len(options) > 1:
        return min(judge(option) for option in options)
    return _judge_atom(text)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class ScanResult:
    failures: list[dict[str, str]] = field(default_factory=list)
    unapproved: list[dict[str, str]] = field(default_factory=list)
    approved: list[dict[str, str]] = field(default_factory=list)
    stale: list[dict[str, str]] = field(default_factory=list)


def _matches(entry: dict[str, str], row: dict[str, str]) -> bool:
    return (
        _normalize(entry["name"]) == _normalize(row.get("Name", ""))
        and fnmatch.fnmatchcase(row.get("Version", ""), entry["version"])
        and entry["license"] == row.get("License", "")
    )


def classify(rows: list[dict[str, str]], signoff: list[dict[str, str]]) -> ScanResult:
    """Sort pip-licenses rows into failures, unapproved and approved sign-offs."""
    result = ScanResult()
    used: set[int] = set()
    for row in rows:
        license_ = row.get("License", "")
        if _normalize(row.get("Name", "")) == "albumentations":
            is_mit = license_.strip().lower() in {"mit", "mit license"}
            outcome = Outcome.PASS if is_mit else Outcome.FAIL
        else:
            outcome = judge(license_)
        if outcome is Outcome.FAIL:
            result.failures.append(row)
        elif outcome is Outcome.SIGNOFF:
            hits = [i for i, entry in enumerate(signoff) if _matches(entry, row)]
            used.update(hits)
            (result.approved if hits else result.unapproved).append(row)
    result.stale = [entry for i, entry in enumerate(signoff) if i not in used]
    return result


def load_signoff(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return entries


def _run_pip_licenses() -> list[dict[str, str]]:
    out = subprocess.run(
        [sys.executable, "-m", "piplicenses", "--format=json", "--from=mixed"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows: list[dict[str, str]] = json.loads(out)
    return rows


def _describe(row: dict[str, Any]) -> str:
    return f"  {row.get('Name', '?')} {row.get('Version', '?')}: {row.get('License', '')!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--rows", type=Path, help="pip-licenses JSON output to read instead")
    parser.add_argument("--signoff", type=Path, default=DEFAULT_SIGNOFF)
    args = parser.parse_args(argv)
    if args.rows is not None:
        rows: list[dict[str, str]] = json.loads(args.rows.read_text(encoding="utf-8"))
    else:
        rows = _run_pip_licenses()
    result = classify(rows, load_signoff(args.signoff))
    print(f"Scanned {len(rows)} distributions.")
    if result.approved:
        print(f"Signed off in {args.signoff.name}:")
        print("\n".join(_describe(r) for r in result.approved))
    if result.stale:
        print("Warning: sign-off entries that match no installed distribution:")
        print("\n".join(f"  {e['name']} {e['version']}: {e['license']!r}" for e in result.stale))
    if result.unapproved:
        print("FAILED: needs the owner's sign-off (LGPL, proprietary, unknown or unrecognised):")
        print("\n".join(_describe(r) for r in result.unapproved))
    if result.failures:
        print("FAILED: GPL, AGPL or non-commercial license, or a non-MIT albumentations:")
        print("\n".join(_describe(r) for r in result.failures))
    if result.failures or result.unapproved:
        return 1
    print("License scan passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

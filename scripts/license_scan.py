# SPDX-License-Identifier: Apache-2.0
"""License scan of the resolved environment (§2.3 notes; Q-4).

Runs pip-licenses (or reads its JSON output from a file given as the only
argument) and classifies every installed distribution:

- GPL or AGPL in any spelling fails the scan;
- LGPL and unknown licenses pass but are listed for the owner's sign-off;
- albumentations fails unless it is MIT (§2.3 note);
- everything else passes, including MPL-2.0 (file-level copyleft, §2.3).

Exit code 0 when nothing fails, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_LESSER = re.compile(
    r"\bLGPL[\w.+-]*|(?:GNU\s+)?(?:Library\s+or\s+)?Lesser\s+General\s+Public\s+License[^;()]*",
    re.IGNORECASE,
)
_GPL = re.compile(r"\bA?GPL|General\s+Public\s+License|Affero", re.IGNORECASE)
_UNKNOWN = {"", "unknown", "none"}
_MIT = {"mit", "mit license"}


@dataclass
class ScanResult:
    failures: list[dict[str, str]] = field(default_factory=list)
    signoff: list[dict[str, str]] = field(default_factory=list)


def classify(rows: list[dict[str, str]]) -> ScanResult:
    """Sort pip-licenses rows into failures and rows that need sign-off."""
    result = ScanResult()
    for row in rows:
        name = row.get("Name", "")
        license_ = row.get("License", "").strip()
        if name.lower() == "albumentations" and license_.lower() not in _MIT:
            result.failures.append(row)
        elif _GPL.search(_LESSER.sub(" ", license_)):
            result.failures.append(row)
        elif _LESSER.search(license_) or license_.lower() in _UNKNOWN:
            result.signoff.append(row)
    return result


def _run_pip_licenses() -> list[dict[str, str]]:
    out = subprocess.run(
        [sys.executable, "-m", "piplicenses", "--format=json", "--from=mixed"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows: list[dict[str, str]] = json.loads(out)
    return rows


def _describe(row: dict[str, str]) -> str:
    return f"  {row.get('Name', '?')} {row.get('Version', '?')}: {row.get('License', '')!r}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        rows: list[dict[str, str]] = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    else:
        rows = _run_pip_licenses()
    result = classify(rows)
    print(f"Scanned {len(rows)} distributions.")
    if result.signoff:
        print("Needs the owner's sign-off (LGPL or unknown license):")
        print("\n".join(_describe(r) for r in result.signoff))
    if result.failures:
        print("FAILED (GPL or AGPL, or a non-MIT albumentations):")
        print("\n".join(_describe(r) for r in result.failures))
        return 1
    print("No GPL or AGPL license found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

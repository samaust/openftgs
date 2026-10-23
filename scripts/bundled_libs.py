# SPDX-License-Identifier: Apache-2.0
"""Inventory of the shared libraries bundled in dependency wheels (§2.3 notes; Q-4; #366).

The license scan judges each distribution's declared license; a wheel may also carry
libraries built from other projects. This script walks the installed distributions' RECORDs
and sorts each one's shared libraries into its own extension modules, the libraries bundled
into its wheel, and shared libraries in a layout it doesn't know. For each distribution that
has any of the last two, it reports the version, the declared license and the license notices
the wheel ships, which are then read by hand (docs/clean-room/release-checklist.md).

Run it with the interpreter of the environment to inventory:

    <env>/bin/python scripts/bundled_libs.py [--json] [--all]

or on built wheel files with --wheel FILE (openftgs's own wheel holds no shared library).
"""

from __future__ import annotations

import argparse
import email.parser
import importlib.metadata
import json
import re
import sys
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

# .so with an optional version (libfoo.so.5.0.0), and the macOS and Windows forms.
_SHARED = re.compile(r"\.(?:so(?:\.\d+)*|dylib|dll|pyd)$")
# CPython's extension suffixes, whatever the interpreter running the script.
_EXTENSION = re.compile(r"\.(?:cpython-\d+[a-z]*-[^/]+\.so|abi3\.so|pyd)$")
# Where wheels put the libraries they bundle: auditwheel's <name>.libs/, torch's lib/, the
# NVIDIA runtime wheels' nvidia/<component>/lib/, and triton's backends/<backend>/lib/.
_BUNDLED_DIR = re.compile(
    r"(?:^|/)(?:[^/]+\.libs/|torch/lib/|nvidia/[^/]+/lib/|triton/backends/[^/]+/lib/)"
)
# License notices: a dist-info's licenses/ folder (PEP 639), and files named like a license,
# a copying notice or a third-party notice anywhere in the wheel.
_NOTICE = re.compile(
    r"(?:^|/)[^/]+\.dist-info/licenses/"
    r"|(?:^|/)(?:licen[cs]e|copying|notice|third[-_]?party)(?:[-_.][^/]*)?$"
    r"|(?:^|/)licenses/[^/]+$",
    re.IGNORECASE,
)
_CODE = re.compile(r"\.(?:py[cdio]?|so(?:\.\d+)*|h|hpp|c|cpp|cu|cuh|js)$", re.IGNORECASE)


@dataclass
class Classified:
    """A distribution's shared libraries, in its file list's order."""

    extensions: list[str] = field(default_factory=list)
    bundled: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)


def classify(files: Iterable[str]) -> Classified:
    """Sort a distribution's files, as its RECORD lists them, by kind of shared library."""
    result = Classified()
    for path in files:
        if not _SHARED.search(path):
            continue
        if _BUNDLED_DIR.search(path):
            result.bundled.append(path)
        elif _EXTENSION.search(path):
            result.extensions.append(path)
        else:
            result.unrecognised.append(path)
    return result


def notices(files: Iterable[str]) -> list[str]:
    """The license notices among a distribution's files, in its file list's order."""
    return [path for path in files if _NOTICE.search(path) and not _CODE.search(path)]


class _Metadata(Protocol):
    def get_all(self, name: str, failobj: None = None) -> list[str] | None: ...


def _first(metadata: _Metadata, name: str) -> str | None:
    values = metadata.get_all(name)
    return values[0] if values else None


def declared_license(metadata: _Metadata) -> str:
    """A distribution's declared license: its License-Expression, else its license
    classifiers (as pip-licenses' --from=mixed prefers them), else its License field's first
    line, else UNKNOWN."""
    expression = _first(metadata, "License-Expression")
    if expression:
        return expression.strip()
    classifiers = [
        c.split("::")[-1].strip()
        for c in metadata.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)
    field_ = (_first(metadata, "License") or "").strip()
    return field_.splitlines()[0].strip() if field_ else "UNKNOWN"


@dataclass
class Entry:
    """One distribution's inventory row."""

    name: str
    version: str
    license: str
    extensions: list[str]
    bundled: list[str]
    unrecognised: list[str]
    notices: list[str]


def _entry(metadata: _Metadata, files: list[str]) -> Entry:
    classified = classify(files)
    return Entry(
        name=_first(metadata, "Name") or "?",
        version=_first(metadata, "Version") or "?",
        license=declared_license(metadata),
        extensions=classified.extensions,
        bundled=classified.bundled,
        unrecognised=classified.unrecognised,
        notices=notices(files),
    )


def _installed() -> Iterator[Entry]:
    for dist in importlib.metadata.distributions():
        files = [f.as_posix() for f in dist.files or []]
        yield _entry(dist.metadata, files)


def _wheel(path: Path) -> Entry:
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        (meta,) = [n for n in names if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
        metadata = email.parser.Parser().parsestr(wheel.read(meta).decode("utf-8"))
    return _entry(metadata, names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--wheel", type=Path, action="append", help="a wheel file to inventory")
    parser.add_argument("--json", action="store_true", help="print the entries as JSON")
    parser.add_argument("--all", action="store_true", help="also list distributions without")
    args = parser.parse_args(argv)
    entries = [_wheel(w) for w in args.wheel] if args.wheel else list(_installed())
    entries.sort(key=lambda e: e.name.lower())
    if not args.all and not args.wheel:
        entries = [e for e in entries if e.bundled or e.unrecognised]
    if args.json:
        print(json.dumps([asdict(e) for e in entries], indent=2))
        return 0
    for e in entries:
        if not (e.extensions or e.bundled or e.unrecognised):
            print(f"{e.name} {e.version}: no shared library")
            continue
        print(f"{e.name} {e.version} ({e.license!r}): {len(e.extensions)} extension modules")
        lines = [f"  bundled: {p}" for p in e.bundled]
        lines += [f"  unrecognised: {p}" for p in e.unrecognised]
        lines += [f"  notice: {p}" for p in e.notices]
        if lines:
            print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

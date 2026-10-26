# SPDX-License-Identifier: Apache-2.0
"""Vendor ꟻLIP's PyTorch loss into ``openftgs._vendor.flip`` (D-65, D-89; Q-252; #383).

Usage::

    python scripts/vendor_flip.py           # write the copy
    python scripts/vendor_flip.py --check   # rebuild in a temporary folder and compare

The source is the reference copy beside the tests, ``tests/reference/flip`` (Q-252): NVIDIA's
``src/pytorch/flip_loss.py`` and ``LICENSE`` at commit d3ab02a, unchanged, each checked
against its SHA-256 (source log rows 425 and T383.1), so the script needs no network. It
writes ``flip_loss.py`` with the changes below, the license file and VENDOR.json into
``src/openftgs/_vendor/flip``, and leaves the package's own ``__init__.py`` there alone.

Every line the script changes ends with the comment ``# openftgs (<change>)``, and every block
it adds starts with one; VENDOR.json lists each edit with its count. The changes are device
and caching only (D-89):

- ``device``: device-agnostic tensors. Every ``.cuda()`` becomes ``.to(<image>.device)`` and
  every ``device='cuda'`` becomes ``device=<image>.device``, where ``<image>`` is the
  function's image argument; ``generate_spatial_filter`` takes the device as a new last
  argument, and the two colours of the colour maximum are made on the reference's device.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

UPSTREAM = "https://github.com/NVlabs/flip"
COMMIT = "d3ab02a4425eb167d8a4021b382a2bafc6e6c838"
SCRIPT = "scripts/vendor_flip.py"
PACKAGE = "openftgs._vendor.flip"
REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE = REPOSITORY / "tests" / "reference" / "flip"  # Q-252
DEST = REPOSITORY / "src" / "openftgs" / "_vendor" / "flip"
_RAW = f"https://raw.githubusercontent.com/NVlabs/flip/{COMMIT}"

SOURCE = "flip_loss.py"
SOURCE_UPSTREAM = "src/pytorch/flip_loss.py"
# Source log rows 425 and T383.1.
SOURCE_SHA256 = "bf2b83aeb2427a960abe1143d4436d48609f1cc4f256e685daa08d86cc07c4fc"
LICENSE = "LICENSE"
LICENSE_SHA256 = "13b955078ffb4a3215757038ab2e5fcf0cc66349d0990faec9adba8ad034e578"
RECORD = "VENDOR.json"
# The package's own files in DEST, which the script neither writes nor compares.
OWN = ("__init__.py",)

NOTICE = (
    "# Modified by openftgs from ꟻLIP d3ab02a by scripts/vendor_flip.py: every changed line ends",
    "# with, and every added block starts with, a comment 'openftgs (<change>)'; VENDOR.json",
    "# lists the edits.",
    "",
)

CHANGES: dict[str, str] = {
    "device": (
        "Device-agnostic tensors in place of the CUDA-only ones: every .cuda() becomes "
        ".to(<image>.device) and every device='cuda' becomes device=<image>.device, where "
        "<image> is the function's image argument; generate_spatial_filter takes the device "
        "as a new last argument, which compute_ldrflip passes, and the two colours of the "
        "colour maximum are made on the reference's device, where the published file's "
        "color_space_transform moved them with .cuda()"
    ),
}


@dataclass(frozen=True)
class _Sub:
    """Replace ``old`` by ``new``, ``count`` times in all, on single lines of the function or
    method ``scope`` (a qualified name); every line changed gets the marker of ``change``."""

    change: str
    scope: str
    old: str
    new: str
    count: int = 1


def _device(scope: str, image: str, count: int, old: str = ".cuda()") -> _Sub:
    new = f".to({image}.device)" if old == ".cuda()" else f"device={image}.device"
    return _Sub("device", scope, old, new, count)


SUBS: tuple[_Sub, ...] = (
    _device("HDRFLIPLoss.forward", "reference", 3),
    _device("tone_map", "img", 3),
    _device("compute_start_stop_exposures", "reference", 6),
    _Sub(
        "device",
        "generate_spatial_filter",
        "def generate_spatial_filter(pixels_per_degree, channel):",
        "def generate_spatial_filter(pixels_per_degree, channel, device):",
    ),
    _Sub("device", "generate_spatial_filter", ".cuda()", ".to(device)"),
    *(
        _Sub(
            "device",
            "compute_ldrflip",
            f"pixels_per_degree, '{channel}')",
            f"pixels_per_degree, '{channel}', reference.device)",
        )
        for channel in ("A", "RG", "BY")
    ),
    _Sub("device", "compute_ldrflip", ".unsqueeze(0), ", ".unsqueeze(0).to(reference.device), ", 2),
    _device("spatial_filter", "img", 2, "device='cuda'"),
    _device("spatial_filter", "img", 3),
    _device("hunt_adjustment", "img", 1, "device='cuda'"),
    _device("redistribute_errors", "power_deltaE_hyab", 1, "device='cuda'"),
    _device("feature_detection", "img_y", 1),
    _device("color_space_transform", "input_color", 4),
)


class VendorError(Exception):
    """A check failed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _spans(tree: ast.Module) -> dict[str, tuple[int, int]]:
    """First and last line (1-based, decorators included) of every top-level function and
    class, and of every method as ``Class.method``."""
    spans: dict[str, tuple[int, int]] = {}

    def add(name: str, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        assert node.end_lineno is not None
        first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        spans[name] = (first, node.end_lineno)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            add(node.name, node)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    add(f"{node.name}.{item.name}", item)
    return spans


def _string_lines(text: str) -> set[int]:
    """Lines (1-based) that a string literal spanning several lines covers, its first excepted."""
    covered: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING and token.end[0] > token.start[0]:
            covered.update(range(token.start[0] + 1, token.end[0] + 1))
    return covered


def _marker(changes: Sequence[str]) -> str:
    return f"# openftgs ({', '.join(changes)})"


def _patch(source: bytes) -> tuple[bytes, list[str]]:
    """The vendored file and the edits made, in words."""
    text = source.decode("utf-8")
    if "\r" in text:
        raise VendorError(f"{SOURCE}: carriage returns are not handled")
    tree = ast.parse(text, filename=SOURCE)
    spans = _spans(tree)
    in_strings = _string_lines(text)
    lines = text.split("\n")
    changed: dict[int, list[str]] = {}
    edits: list[str] = []
    for sub in SUBS:
        if sub.scope not in spans:
            raise VendorError(f"{SOURCE}: no function {sub.scope}")
        first, last = spans[sub.scope]
        found = 0
        for number in range(first, last + 1):
            count = lines[number - 1].count(sub.old)
            if not count:
                continue
            if number in in_strings:
                raise VendorError(f"{SOURCE}: line {number} is inside a string")
            lines[number - 1] = lines[number - 1].replace(sub.old, sub.new)
            if sub.change not in changed.setdefault(number, []):
                changed[number].append(sub.change)
            found += count
        if found != sub.count:
            raise VendorError(f"{sub.scope}: {found} of {sub.old!r}, not {sub.count}")
        times = "once" if sub.count == 1 else f"{sub.count} times"
        edits.append(f"{sub.change}: in {sub.scope}, {sub.old!r} -> {sub.new!r}, {times}")
    for number, names in changed.items():
        lines[number - 1] += f"  {_marker(names)}"
    units: list[list[str]] = [[line] for line in lines]
    units[_line_starting(lines, "# Visualizing and Communicating Errors")][:0] = list(NOTICE)
    edits.append("notice: the modification notice added after NVIDIA's license header")
    out = "\n".join(line for unit in units for line in unit)
    _check(text, out)
    return out.encode("utf-8"), edits


def _line_starting(lines: Sequence[str], start: str) -> int:
    """The index of the one line whose text after its indentation starts with ``start``."""
    found = [i for i, line in enumerate(lines) if line.lstrip().startswith(start)]
    if len(found) != 1:
        raise VendorError(f"{SOURCE}: {len(found)} lines start with {start!r}, not 1")
    return found[0]


def _check(before: str, after: str) -> None:
    """The copy parses, names CUDA nowhere, and every block of lines the diff adds holds a
    marker or the notice."""
    ast.parse(after, filename=SOURCE)
    for cuda in (".cuda(", "'cuda'", '"cuda"'):
        if cuda in after:
            raise VendorError(f"{SOURCE}: {cuda!r} is left")
    old, new = before.split("\n"), after.split("\n")
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, _, _, first, last in matcher.get_opcodes():
        if tag in ("insert", "replace") and not any("openftgs" in line for line in new[first:last]):
            raise VendorError(f"{SOURCE}: lines {first + 1}-{last} changed without a marker")


def _read_reference(reference: Path) -> tuple[bytes, bytes]:
    source = (reference / SOURCE).read_bytes()
    license_text = (reference / LICENSE).read_bytes()
    for name, data, expected in (
        (SOURCE, source, SOURCE_SHA256),
        (LICENSE, license_text, LICENSE_SHA256),
    ):
        if _sha256(data) != expected:
            raise VendorError(f"reference {name}: sha256 {_sha256(data)} != {expected}")
    return source, license_text


def _dump(record: dict[str, object]) -> bytes:
    return (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build(reference: Path, out: Path) -> None:
    """Write the vendored file, its license and VENDOR.json into ``out``."""
    source, license_text = _read_reference(reference)
    vendored, edits = _patch(source)
    out.mkdir(parents=True, exist_ok=True)
    (out / SOURCE).write_bytes(vendored)
    (out / LICENSE).write_bytes(license_text)
    record: dict[str, object] = {
        "upstream": UPSTREAM,
        "commit": COMMIT,
        "script": SCRIPT,
        "package": PACKAGE,
        "reference": REFERENCE.relative_to(REPOSITORY).as_posix(),
        "licenses": [
            {
                "path": LICENSE,
                "project": "ꟻLIP",
                "license": "BSD-3-Clause",
                "source": f"{_RAW}/{LICENSE}",
                "sha256": LICENSE_SHA256,
            }
        ],
        "files": [
            {
                "path": SOURCE,
                "upstream_path": SOURCE_UPSTREAM,
                "source": f"{_RAW}/{SOURCE_UPSTREAM}",
                "upstream_sha256": SOURCE_SHA256,
                "sha256": _sha256(vendored),
                "edits": edits,
            }
        ],
        "changes": CHANGES,
    }
    (out / RECORD).write_bytes(_dump(record))
    newline = b"\n"
    print(
        f"{SOURCE}: {source.count(newline)} -> {vendored.count(newline)} lines, "
        f"{len(edits)} edits, sha256 {_sha256(vendored)[:16]}"
    )


def _files_under(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.name not in OWN
    }


def check(reference: Path, dest: Path) -> bool:
    """Rebuild in a temporary folder; True when the copy at ``dest`` is the same and its
    files have the SHA-256 that VENDOR.json records."""
    with tempfile.TemporaryDirectory() as tmp:
        build(reference, Path(tmp))
        rebuilt = _files_under(Path(tmp))
    present = _files_under(dest)
    ok = True
    for path in sorted(rebuilt.keys() | present.keys()):
        if path not in present:
            print(f"missing from the copy: {path}")
        elif path not in rebuilt:
            print(f"not made by the script: {path}")
        elif rebuilt[path] != present[path]:
            print(
                f"differs: {path} (sha256 {_sha256(present[path])[:16]}, "
                f"rebuilt {_sha256(rebuilt[path])[:16]})"
            )
        else:
            continue
        ok = False
    if RECORD in present:
        record = json.loads(present[RECORD])
        for entry in record["licenses"] + record["files"]:
            data = present.get(entry["path"])
            if data is None or _sha256(data) != entry["sha256"]:
                print(f"not the sha256 VENDOR.json records: {entry['path']}")
                ok = False
    print("check: the copy and VENDOR.json match the rebuild" if ok else "check: MISMATCH")
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument(
        "--reference", type=Path, default=REFERENCE, help="the reference copy's folder (Q-252)"
    )
    parser.add_argument("--dest", type=Path, default=DEST, help="the vendored copy's folder")
    parser.add_argument(
        "--check", action="store_true", help="rebuild in a temporary folder and compare"
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            return 0 if check(args.reference, args.dest) else 1
        with tempfile.TemporaryDirectory() as tmp:
            build(args.reference, Path(tmp))
            args.dest.mkdir(parents=True, exist_ok=True)
            for path in _files_under(args.dest):
                (args.dest / path).unlink()
            for path in _files_under(Path(tmp)):
                target = args.dest / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(tmp) / path, target)
    except VendorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

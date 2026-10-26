# SPDX-License-Identifier: Apache-2.0
"""Vendor ꟻLIP's PyTorch loss into ``openftgs._vendor.flip`` (D-65, D-89; Q-252; #383), and
into the harness's own ``ftgs_harness._vendor.flip`` (§3.3 L3 step 3, §2.1; Q-254; #391).

Usage::

    python scripts/vendor_flip.py           # write openftgs's copy
    python scripts/vendor_flip.py --check   # rebuild in a temporary folder and compare
    python scripts/vendor_flip.py --target harness [--check]   # the same for the harness's copy

The source is the reference copy beside the tests, ``tests/reference/flip`` (Q-252): NVIDIA's
``src/pytorch/flip_loss.py`` and ``LICENSE`` at commit d3ab02a, unchanged, each checked
against its SHA-256 (source log rows 425 and 444), so the script needs no network. It
writes ``flip_loss.py`` with the changes below, the license file and VENDOR.json into
``src/openftgs/_vendor/flip``, and leaves the package's own ``__init__.py`` there alone.

Every line the script changes ends with the comment ``# openftgs (<change>)``, and every block
it adds starts with one; VENDOR.json lists each edit with its count. The changes are device
and caching only (D-89):

- ``device``: device-agnostic tensors. Every ``.cuda()`` becomes ``.to(<image>.device)`` and
  every ``device='cuda'`` becomes ``device=<image>.device``, where ``<image>`` is the
  function's image argument; ``generate_spatial_filter`` takes the device as a new last
  argument, and the two colours of the colour maximum are made on the reference's device.
- ``cache``: ``_once_per_device``, which memoizes a builder of tensors by its arguments, the
  device last, and builds them outside inference mode.
- ``filters``: the filters built once per device and size (§2.4's perceptual-loss row):
  ``generate_spatial_filter`` is memoized, and the lines of ``feature_detection`` that build
  its filter move into ``generate_feature_filter``, memoized the same way.
- ``constants``: the D65 illuminants and the RGB–XYZ matrices of ``color_space_transform``,
  which it copied to the GPU at every call, copied once per device and kept by ``_constant``.
- ``cmax``: the colour maximum, which ``compute_ldrflip`` read back to the host at every call,
  computed once per device by ``_cmax`` and kept as a float (§2.4 rule 3).

After the first call on a device, which builds and copies the filters and constants there and
reads the colour maximum back once, LDR-ꟻLIP's forward and backward passes synchronize nowhere.
HDR-ꟻLIP keeps its host read of the number of exposures, which sets its loop.

The edits are exact substring replacements confined to one function each, located with ast
and counted; lines added before a named line; and contiguous lines moved out of a function
into a new one. The script stops (exit code 2) when any of them doesn't find what it expects.

``--target harness`` writes the harness's copy into ``harness/src/ftgs_harness/_vendor/flip``
instead (Q-254): the harness computes ꟻLIP itself in L3 and never imports openftgs (§2.1), so
it carries a copy of its own, with its own license file and VENDOR.json. That copy has the
``device`` change only, its lines marked ``# ftgs_harness (device)``: the colour transforms'
constants, which openftgs's copy keeps per device ('constants'), are moved to the input's
device like the other tensors, and nothing is cached, so the filters, the constants and the
colour maximum are made at every call, as published.
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
# Q-254 (#391): the harness's own copy.
HARNESS_SCRIPT = "scripts/vendor_flip.py --target harness"
HARNESS_PACKAGE = "ftgs_harness._vendor.flip"
HARNESS_DEST = REPOSITORY / "harness" / "src" / "ftgs_harness" / "_vendor" / "flip"
_RAW = f"https://raw.githubusercontent.com/NVlabs/flip/{COMMIT}"

SOURCE = "flip_loss.py"
SOURCE_UPSTREAM = "src/pytorch/flip_loss.py"
# Source log rows 425 and 444.
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
        "<image> is the function's image argument, except for the colour transforms' "
        "constants ('constants'); generate_spatial_filter takes the device as a new last "
        "argument, which compute_ldrflip passes, and the two colours of the colour maximum "
        "are made on the reference's device, where the published file's color_space_transform "
        "moved them with .cuda()"
    ),
    "constants": (
        "The colour transforms' constant tensors copied to a device once (§2.4 rule 3): the "
        "D65 illuminant and its inverse, which color_space_transform built and copied at "
        "every call, and the RGB-to-XYZ and XYZ-to-RGB matrices, which its linrgb2xyz and "
        "xyz2linrgb branch copied at every call, are kept per device by _constant, under their "
        "names and the transform's; the matrix is still built on the host at every call, as "
        "published, and copied only once"
    ),
    "cmax": (
        "The per-call host read of the colour maximum replaced by a cached constant (§2.4 rule "
        "3): its three lines move out of compute_ldrflip into _cmax(qc, eps, device), memoized "
        "by _once_per_device, which returns the published expression's .item(). It is computed "
        "on the device it serves, with the published expressions, so the CUDA path keeps the "
        "published value: in float32 the CPU gives 41.276100158691406 and CUDA "
        "41.27609634399414 (RTX 4090, PyTorch 2.14.1), one ulp apart"
    ),
    "cache": (
        "The helper the caching changes share: _once_per_device memoizes a builder of tensors "
        "whose last argument is their device (functools.lru_cache), so that they are built "
        "and copied to the device once, and builds them outside inference mode, so that a "
        "tensor first built while evaluating under torch.inference_mode() can still be saved "
        "for a backward pass"
    ),
    "filters": (
        "The filters built once per device and size (§2.4's perceptual-loss row): "
        "generate_spatial_filter is memoized by _once_per_device, and the lines of "
        "feature_detection that build the edge and point filters move, unchanged but for the "
        "device change, into a new function generate_feature_filter(pixels_per_degree, "
        "feature_type, device), memoized the same way, which feature_detection calls. A "
        "filter's size is set by the pixels per degree alone, so the key is the pixels per "
        "degree, the channel or feature type and the device"
    ),
}

# Q-257: ꟻLIP's filter support at 67.0 pixels per degree, measured with the copy on the CPU by
# the test named here (#383); #387 carries it into Q-61.
FILTER_SUPPORT: dict[str, object] = {
    "side": 21,
    "radius": 10,
    "pixels_per_degree": (
        "the published default, (0.7 * 3840 / 0.7) * pi / 180 = 67.02064327658226, and 67.0 "
        "(E16), which give the same filters"
    ),
    "filters": {
        "spatial (A, RG, BY)": "21 x 21: radius ceil(3 * sqrt(0.04 / (2 * pi^2)) * ppd) = 10",
        "feature (edge, point)": "19 x 19: radius ceil(3 * 0.5 * 0.082 * ppd) = 9",
    },
    "measured": (
        "On a flat grey 96 x 96 pair (sRGB 0.5), a one-pixel chromatic difference (0.9, 0.2, "
        "0.7) changes the error map within 10 px of it and no farther, all 21 x 21 pixels on "
        "this machine (the blue-yellow filter reaches that far); an achromatic one (0.6) "
        "within 9 px, through the feature filters, the achromatic filter's tail at 10 px "
        "being below float32's resolution"
    ),
    "test": "tests/test_vendored_flip.py::test_flip_filter_support_at_67_pixels_per_degree",
}

T = "\t"  # the published file indents with tabs


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
    if old == "device='cuda'":
        new = f"device={image}.device"
    else:
        new = old.replace(".cuda()", f".to({image}.device)")
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
    _device("color_space_transform", "input_color", 1, "dim[2]*dim[3]).cuda()"),
    *(
        _Sub(
            "constants",
            "color_space_transform",
            f"{T}{name} = torch.tensor(",
            f"{T}{name} = _constant('{name}', input_color.device, lambda: torch.tensor(",
        )
        for name in ("reference_illuminant", "inv_reference_illuminant")
    ),
    _Sub("constants", "color_space_transform", "]]).cuda()", "]]))", 2),
    _Sub(
        "constants",
        "color_space_transform",
        "torch.matmul(A.cuda(), input_color)",
        "torch.matmul(_constant(fromSpace2toSpace, input_color.device, lambda: A), input_color)",
    ),
)

# Q-254 (#391): the harness's copy takes the device change alone. Its colour transforms move
# their constants to the input's device at every call, as the published file moves them to CUDA,
# where openftgs's copy keeps them per device ('constants').
HARNESS_SUBS: tuple[_Sub, ...] = (
    *(sub for sub in SUBS if sub.change == "device"),
    _device("color_space_transform", "input_color", 2, "]]).cuda()"),
    _device("color_space_transform", "input_color", 1, "A.cuda()"),
)
HARNESS_CHANGES: dict[str, str] = {
    "device": (
        "Device-agnostic tensors in place of the CUDA-only ones, so that the harness computes "
        "ꟻLIP on the CPU (Q-254): every .cuda() becomes .to(<image>.device) and every "
        "device='cuda' becomes device=<image>.device, where <image> is the function's image "
        "argument, the colour transforms' constants included; generate_spatial_filter takes "
        "the device as a new last argument, which compute_ldrflip passes, and the two colours "
        "of the colour maximum are made on the reference's device, where the published file's "
        "color_space_transform moved them with .cuda(). Nothing is cached: the filters, the "
        "constants and the colour maximum are made at every call, as published"
    ),
}
HARNESS_NOTICE = (
    "# Modified for ftgs_harness from ꟻLIP d3ab02a by scripts/vendor_flip.py --target harness:",
    "# every changed line ends with a comment 'ftgs_harness (device)'; VENDOR.json lists the",
    "# edits.",
    "",
)


@dataclass(frozen=True)
class _Insert:
    """Add ``lines`` before the one line whose text after its indentation starts with
    ``before``; they carry their own markers."""

    change: str
    before: str
    lines: tuple[str, ...]


INSERTS: tuple[_Insert, ...] = (
    _Insert("cache", "import sys", ("import functools  # openftgs (cache)",)),
    _Insert(
        "cache",
        "class HDRFLIPLoss(nn.Module):",
        (
            "# openftgs (cache): memoizes a builder of tensors whose last argument is the device",
            "# they are made on, so that they are built and copied there once (§2.4 rule 3);",
            "# the builders' callers only read them. They are made outside inference mode, so",
            "# that a tensor first made while evaluating can still be saved for a backward pass.",
            "def _once_per_device(build):",
            f"{T}@functools.lru_cache(maxsize=None)",
            f"{T}def cached(*args):",
            f"{T}{T}with torch.inference_mode(False):",
            f"{T}{T}{T}return build(*args)",
            f"{T}return functools.wraps(build)(cached)",
            "",
            "",
            "# openftgs (cache): the tensor make() builds, copied to device once and kept under",
            "# name, for the colour transforms' constants; copied outside inference mode, as",
            "# above.",
            "_CONSTANTS = {}",
            "",
            "",
            "def _constant(name, device, make):",
            f"{T}key = (name, device)",
            f"{T}if key not in _CONSTANTS:",
            f"{T}{T}with torch.inference_mode(False):",
            f"{T}{T}{T}_CONSTANTS[key] = make().to(device, copy=True)",
            f"{T}return _CONSTANTS[key]",
            "",
            "",
        ),
    ),
    _Insert(
        "filters",
        "def generate_spatial_filter(",
        ("@_once_per_device  # openftgs (filters)",),
    ),
)


@dataclass(frozen=True)
class _Move:
    """Move the lines of the function ``scope`` from the one whose text starts with ``first``
    through the one that starts with ``last`` into a new top-level function, made of
    ``header``, the moved lines with ``edits`` (old, new, count) made in them and ``footer``,
    and added before the top-level function ``before``; ``left`` takes the lines' place."""

    change: str
    scope: str
    first: str
    last: str
    left: tuple[str, ...]
    header: tuple[str, ...]
    footer: tuple[str, ...]
    edits: tuple[tuple[str, str, int], ...]
    before: str


MOVES: tuple[_Move, ...] = (
    _Move(
        "cmax",
        "compute_ldrflip",
        "hunt_adjusted_green = ",
        "cmax = torch.pow(",
        (f"{T}cmax = _cmax(qc, eps, reference.device)  # openftgs (cmax)",),
        (
            "# openftgs (cmax): the colour maximum, moved out of compute_ldrflip so that it is",
            "# computed once per device and kept as a float, in place of a host read at every",
            "# call (§2.4 rule 3). It is computed with the published expressions on the device",
            "# it serves: on the CPU they give a float32 one ulp from CUDA's.",
            "@_once_per_device",
            "def _cmax(qc, eps, device):",
            f'{T}""" openftgs (cmax): the exponentiated HyAB distance of green and blue """',
        ),
        (f"{T}return cmax  # openftgs (cmax)",),
        (("reference.device", "device", 2),),
        "tone_map",
    ),
    _Move(
        "filters",
        "feature_detection",
        "# Set peak to trough value",
        "Gx = Gx.unsqueeze(0).unsqueeze(0)",
        (
            f"{T}Gx, radius = generate_feature_filter(pixels_per_degree, feature_type, "
            "img_y.device)  # openftgs (filters)",
        ),
        (
            "# openftgs (filters): the edge and point filters and their radius, moved out of",
            "# feature_detection so that each is built once per pixels per degree, feature",
            "# type and device (§2.4's perceptual-loss row); the moved lines are the published",
            "# ones, with the device change.",
            "@_once_per_device",
            "def generate_feature_filter(pixels_per_degree, feature_type, device):",
            f'{T}""" openftgs (filters): the 1x1xKxK filter of feature_detection, K = 2 * radius'
            ' + 1, and its radius """',
        ),
        (f"{T}return Gx, radius  # openftgs (filters)",),
        (("img_y.device", "device", 1),),
        "feature_detection",
    ),
)


@dataclass(frozen=True)
class _Target:
    """One copy the script makes: where it goes, its package, the edits it takes, the name its
    markers carry, the functions left reading a value back to the host with ``.item()``, and
    the fields its VENDOR.json adds after ``changes``."""

    name: str
    script: str
    package: str
    dest: Path
    marker: str
    subs: tuple[_Sub, ...]
    moves: tuple[_Move, ...]
    inserts: tuple[_Insert, ...]
    notice: tuple[str, ...]
    changes: dict[str, str]
    reads: tuple[str, ...]
    extra: dict[str, object]


TARGETS: dict[str, _Target] = {
    "openftgs": _Target(
        name="openftgs",
        script=SCRIPT,
        package=PACKAGE,
        dest=DEST,
        marker="openftgs",
        subs=SUBS,
        moves=MOVES,
        inserts=INSERTS,
        notice=NOTICE,
        changes=CHANGES,
        reads=("HDRFLIPLoss.forward", "_cmax"),
        extra={
            "first_call": (
                "The first call of LDR-ꟻLIP on a device builds and copies there the five "
                "filters of its pixels per degree and the four colour constants, and reads the "
                "colour maximum back once; later calls on it copy and read nothing (§2.4 rule 3)"
            ),
            "filter_support": FILTER_SUPPORT,
        },
    ),
    # Q-254 (#391): the harness's own copy, for L3's ꟻLIP (§3.3 L3 step 3).
    "harness": _Target(
        name="harness",
        script=HARNESS_SCRIPT,
        package=HARNESS_PACKAGE,
        dest=HARNESS_DEST,
        marker="ftgs_harness",
        subs=HARNESS_SUBS,
        moves=(),
        inserts=(),
        notice=HARNESS_NOTICE,
        changes=HARNESS_CHANGES,
        reads=("HDRFLIPLoss.forward", "compute_ldrflip"),
        extra={
            "purpose": (
                "The harness's own copy (Q-254): L3 computes ꟻLIP itself from the saved renders "
                "(§3.3 L3 step 3) and the harness never imports openftgs (§2.1); it runs on the "
                "CPU, so the L3 metric tests run in CPU CI"
            ),
        },
    ),
}


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


def _marker(changes: Sequence[str], name: str) -> str:
    return f"# {name} ({', '.join(changes)})"


def _patch(source: bytes, target: _Target) -> tuple[bytes, list[str]]:
    """The vendored file of ``target`` and the edits made, in words."""
    text = source.decode("utf-8")
    if "\r" in text:
        raise VendorError(f"{SOURCE}: carriage returns are not handled")
    tree = ast.parse(text, filename=SOURCE)
    spans = _spans(tree)
    in_strings = _string_lines(text)
    lines = text.split("\n")
    changed: dict[int, list[str]] = {}
    edits: list[str] = []
    for sub in target.subs:
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
        lines[number - 1] += f"  {_marker(names, target.marker)}"
    # Each line of the published file, as the lines it becomes.
    units: list[list[str]] = [[line] for line in lines]
    for move in target.moves:
        edits.append(_move(move, lines, units, spans))
    for insert in target.inserts:
        units[_line_starting(lines, insert.before)][:0] = list(insert.lines)
        edits.append(f"{insert.change}: {_lines(len(insert.lines))} added before {insert.before!r}")
    units[_line_starting(lines, "# Visualizing and Communicating Errors")][:0] = list(target.notice)
    edits.append("notice: the modification notice added after NVIDIA's license header")
    out = "\n".join(line for unit in units for line in unit)
    _check(text, out, target)
    return out.encode("utf-8"), edits


def _move(
    move: _Move, lines: Sequence[str], units: list[list[str]], spans: dict[str, tuple[int, int]]
) -> str:
    """Make ``move`` in ``units``; the edit in words."""
    for name in (move.scope, move.before):
        if name not in spans:
            raise VendorError(f"{SOURCE}: no function {name}")
    first, last = spans[move.scope]
    scope = range(first - 1, last)
    start = _line_starting(lines, move.first, scope)
    end = _line_starting(lines, move.last, scope)
    if end < start:
        raise VendorError(f"{move.scope}: {move.last!r} comes before {move.first!r}")
    moved = "\n".join(line for unit in units[start : end + 1] for line in unit)
    for old, new, count in move.edits:
        if moved.count(old) != count:
            raise VendorError(f"{move.scope}: {moved.count(old)} of {old!r} moved, not {count}")
        moved = moved.replace(old, new)
    units[start] = list(move.left)
    for index in range(start + 1, end + 1):
        units[index] = []
    function = [*move.header, *moved.split("\n"), *move.footer, "", ""]
    units[spans[move.before][0] - 1][:0] = function
    name = next(line for line in move.header if line.startswith("def ")).split("(")[0][4:]
    replaced = ", ".join(f"{old!r} -> {new!r} {count}x" for old, new, count in move.edits)
    return (
        f"{move.change}: lines {start + 1}-{end + 1} of {move.scope} moved into a new function "
        f"{name}, added before {move.before} ({replaced} in them); in their place "
        f"{_lines(len(move.left))}"
    )


@dataclass(frozen=True)
class _Span:
    name: str
    first: int
    last: int


def _spans_with(text: str, fragment: str) -> list[_Span]:
    """The innermost functions (methods as ``Class.method``) whose lines hold ``fragment``, in
    the order of the file, each once."""
    spans = sorted(
        (_Span(name, first, last) for name, (first, last) in _spans(ast.parse(text)).items()),
        key=lambda s: (s.first, -s.last),
    )
    found: list[_Span] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if fragment not in line:
            continue
        holding = [s for s in spans if s.first <= number <= s.last]
        if not holding:
            raise VendorError(f"{SOURCE}: {fragment!r} at line {number}, outside any function")
        innermost = holding[-1]
        if innermost not in found:
            found.append(innermost)
    return found


def _lines(count: int) -> str:
    return "1 line" if count == 1 else f"{count} lines"


def _line_starting(lines: Sequence[str], start: str, within: range | None = None) -> int:
    """The index of the one line (in ``within``, else anywhere) whose text after its
    indentation starts with ``start``."""
    indices = range(len(lines)) if within is None else within
    found = [i for i in indices if lines[i].lstrip().startswith(start)]
    if len(found) != 1:
        raise VendorError(f"{SOURCE}: {len(found)} lines start with {start!r}, not 1")
    return found[0]


def _check(before: str, after: str, target: _Target) -> None:
    """The copy parses, names CUDA nowhere, and every block of lines the diff adds holds a
    marker or the notice."""
    ast.parse(after, filename=SOURCE)
    for cuda in (".cuda(", "'cuda'", '"cuda"'):
        if cuda in after:
            raise VendorError(f"{SOURCE}: {cuda!r} is left")
    # The host reads left: HDR-ꟻLIP's exposures, and the colour maximum's, once per device in
    # openftgs's copy and at every call in the harness's.
    reads = [s.name for s in _spans_with(after, ".item()")]
    if reads != list(target.reads):
        raise VendorError(f"{SOURCE}: .item() in {reads}")
    old, new = before.split("\n"), after.split("\n")
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, _, _, first, last in matcher.get_opcodes():
        if tag in ("insert", "replace") and not any(
            target.marker in line for line in new[first:last]
        ):
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


def build(reference: Path, out: Path, target: _Target) -> None:
    """Write ``target``'s vendored file, its license and VENDOR.json into ``out``."""
    source, license_text = _read_reference(reference)
    vendored, edits = _patch(source, target)
    out.mkdir(parents=True, exist_ok=True)
    (out / SOURCE).write_bytes(vendored)
    (out / LICENSE).write_bytes(license_text)
    record: dict[str, object] = {
        "upstream": UPSTREAM,
        "commit": COMMIT,
        "script": target.script,
        "package": target.package,
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
        "changes": target.changes,
        **target.extra,
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


def check(reference: Path, dest: Path, target: _Target) -> bool:
    """Rebuild ``target``'s copy in a temporary folder; True when the copy at ``dest`` is the
    same and its files have the SHA-256 that VENDOR.json records."""
    with tempfile.TemporaryDirectory() as tmp:
        build(reference, Path(tmp), target)
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
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="openftgs",
        help="the copy to make: openftgs's (D-65) or the harness's own (Q-254)",
    )
    parser.add_argument(
        "--dest", type=Path, help="the vendored copy's folder (default: the target's)"
    )
    parser.add_argument(
        "--check", action="store_true", help="rebuild in a temporary folder and compare"
    )
    args = parser.parse_args(argv)
    target = TARGETS[args.target]
    dest: Path = target.dest if args.dest is None else args.dest
    try:
        if args.check:
            return 0 if check(args.reference, dest, target) else 1
        with tempfile.TemporaryDirectory() as tmp:
            build(args.reference, Path(tmp), target)
            dest.mkdir(parents=True, exist_ok=True)
            for path in _files_under(dest):
                (dest / path).unlink()
            for path in _files_under(Path(tmp)):
                written = dest / path
                written.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(tmp) / path, written)
    except VendorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

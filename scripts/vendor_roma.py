# SPDX-License-Identifier: Apache-2.0
"""Vendor RoMa's inference code into ``openftgs._vendor.romatch`` (D-65; Q-113, Q-114, Q-115).

Usage::

    python scripts/vendor_roma.py --source CLONE           # write the copy
    python scripts/vendor_roma.py --source CLONE --check   # rebuild in a temporary folder
                                                           # and compare with the copy

CLONE is a git clone of https://github.com/Parskatt/RoMa that holds commit 77f8d68 (a
blobless clone works: git fetches blobs on demand). The license texts are fetched from
raw.githubusercontent.com at pinned commits and checked against their SHA-256.

The script reads RoMa's 22 inference files (D-65) with git, checks each blob id against
the commit's tree, locates every change with ast and edits the text only where ast points:

- the 14 absolute ``romatch`` imports, 12 at module level and 2 inside functions, point to
  ``openftgs._vendor.romatch`` (D-65);
- ``utils/utils.py`` loses its module-level cv2 import and the span from the first to the
  last of the three pose helpers, which holds only them and blank or comment lines;
  ``utils/__init__.py`` stops importing them (Q-113);
- the DINOv2 layers that import xformers keep only the statements of that import's
  ``except ImportError`` branch (Q-114);
- each changed file gets one modification-notice line (Q-115).

It prints only file names, counts, function names and hashes: never source text, never a
diff, nothing from a removed span. VENDOR.json describes each patch in words.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

UPSTREAM = "https://github.com/Parskatt/RoMa"
COMMIT = "77f8d68803526dcddfd9b7a46bc76125bdc25f15"
SCRIPT = "scripts/vendor_roma.py"
PACKAGE = "openftgs._vendor.romatch"
REPOSITORY = Path(__file__).resolve().parents[1]
DEST = REPOSITORY / "src" / "openftgs" / "_vendor" / "romatch"

# Q-115: the modification notice, one comment line per changed file.
NOTICE = "Modified by openftgs from RoMa 77f8d68 by the vendoring script; see VENDOR.json"

# D-65: RoMa's inference files, relative to romatch/ (source log row 116).
FILES: tuple[str, ...] = (
    "__init__.py",
    "models/__init__.py",
    "models/encoders.py",
    "models/matcher.py",
    "models/model_zoo/__init__.py",
    "models/model_zoo/roma_models.py",
    "models/tiny.py",
    "models/transformer/__init__.py",
    "models/transformer/dinov2.py",
    "models/transformer/layers/__init__.py",
    "models/transformer/layers/attention.py",
    "models/transformer/layers/block.py",
    "models/transformer/layers/dino_head.py",
    "models/transformer/layers/drop_path.py",
    "models/transformer/layers/layer_scale.py",
    "models/transformer/layers/mlp.py",
    "models/transformer/layers/patch_embed.py",
    "models/transformer/layers/swiglu_ffn.py",
    "utils/__init__.py",
    "utils/kde.py",
    "utils/local_correlation.py",
    "utils/utils.py",
)

# D-65: 12 absolute romatch imports at module level, 2 inside functions (source log row 119).
ROMATCH_IMPORTS_AT_MODULE_LEVEL = 12
ROMATCH_IMPORTS_IN_FUNCTIONS = 2

# Q-113: removed from utils/utils.py by name, with the module-level cv2 import.
UTILS = "utils/utils.py"
UTILS_INIT = "utils/__init__.py"
POSE_HELPERS = ("recover_pose", "estimate_pose", "estimate_pose_uncalibrated")
# Source log row 118: the first function after the removed span.
FIRST_KEPT_FUNCTION = "unnormalize_coords"

# Q-114: the DINOv2 layers whose xformers import is replaced by its fallback.
XFORMERS_FILES = (
    "models/transformer/layers/attention.py",
    "models/transformer/layers/block.py",
    "models/transformer/layers/swiglu_ffn.py",
)


@dataclass(frozen=True)
class _License:
    path: str
    project: str
    spdx: str
    source: str
    sha256: str


_RAW = "https://raw.githubusercontent.com"

# Q-115: license texts at pinned commits, with the SHA-256 of source log rows 106 and 108-113.
LICENSES: tuple[_License, ...] = (
    _License(
        "LICENSE",
        "RoMa",
        "MIT",
        f"{_RAW}/Parskatt/RoMa/{COMMIT}/LICENSE",
        "6f02ad61c3fd4509343d619b55501354bcebfd67ad5460c853d02c7ab46ff4bd",
    ),
    _License(
        "LICENSES/DINOv2.txt",
        "DINOv2",
        "Apache-2.0",
        f"{_RAW}/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE",
        "600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7",
    ),
    _License(
        "LICENSES/DINO.txt",
        "DINO",
        "Apache-2.0",
        f"{_RAW}/facebookresearch/dino/7c446df5b9f45747937fb0d72314eb9f7b66930a/LICENSE",
        "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a",
    ),
    _License(
        "LICENSES/timm.txt",
        "timm",
        "Apache-2.0",
        f"{_RAW}/huggingface/pytorch-image-models/637f8191614a4be38e144ab2f870f69cbc03ad30/LICENSE",
        "71b111620fa32c17a80ccd6db2da759d31a591e497d8b5f382174f59d1b59d49",
    ),
    _License(
        "LICENSES/LoFTR.txt",
        "LoFTR",
        "Apache-2.0",
        f"{_RAW}/zju3dv/LoFTR/94e98b695be18acb43d5d3250f52226a8e36f839/LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    _License(
        "LICENSES/Patch2Pix.txt",
        "Patch2Pix",
        "MIT",
        f"{_RAW}/GrumpyZhou/patch2pix/0777495b2cff1e876b16cba7f1f7cfcde400bef5/LICENSE",
        "c7318b22a3827cf6b8765342fbf647af30ab7f7b09b943223b8fbfbef43b49b5",
    ),
    _License(
        "LICENSES/Kornia.txt",
        "Kornia",
        "Apache-2.0",
        f"{_RAW}/kornia/kornia/542752eff7dc9c41917302c22afc7cdf4a69d13b/LICENSE",
        "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
    ),
)


class VendorError(Exception):
    """A check failed. Messages name files, counts, functions and hashes, never source text."""


# ---------------------------------------------------------------------------------------
# git


def _git(source: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True)
    if result.returncode != 0:
        raise VendorError(f"git {args[0]} failed with exit code {result.returncode}")
    return result.stdout


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _tree_blobs(source: Path) -> dict[str, str]:
    """Blob ids of the commit's tree, by path."""
    if _git(source, "rev-parse", "--verify", f"{COMMIT}^{{commit}}").decode().strip() != COMMIT:
        raise VendorError(f"the clone does not hold commit {COMMIT}")
    blobs: dict[str, str] = {}
    for entry in _git(source, "ls-tree", "-r", "-z", COMMIT).split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        _mode, kind, object_id = meta.decode().split(" ")
        if kind == "blob":
            blobs[path.decode()] = object_id
    return blobs


def _read_blob(source: Path, blobs: dict[str, str], path: str) -> bytes:
    if path not in blobs:
        raise VendorError(f"{path}: not in the tree of {COMMIT}")
    data = _git(source, "cat-file", "blob", blobs[path])
    if _blob_id(data) != blobs[path]:
        raise VendorError(f"{path}: blob id {_blob_id(data)} != tree's {blobs[path]}")
    return data


# ---------------------------------------------------------------------------------------
# Edits located by ast


@dataclass(frozen=True)
class _Edit:
    """Replace lines first..last (1-based, inclusive) by ``new``; last = first - 1 inserts."""

    first: int
    last: int
    new: tuple[str, ...] = ()


def _apply(lines: list[str], edits: Sequence[_Edit], path: str) -> list[str]:
    ordered = sorted(edits, key=lambda e: (e.first, e.last))
    for before, after in zip(ordered, ordered[1:], strict=False):
        if after.first <= before.last:
            raise VendorError(f"{path}: overlapping edits")
    out = list(lines)
    for edit in reversed(ordered):
        out[edit.first - 1 : edit.last] = list(edit.new)
    return out


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    """The qualified name of the innermost function holding ``node``, if any."""
    chain: list[str] = []
    in_function = False
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            chain.append(parent.name)
            in_function = in_function or not isinstance(parent, ast.ClassDef)
        elif isinstance(parent, ast.Lambda):
            chain.append("<lambda>")
            in_function = True
        parent = parents.get(parent)
    return ".".join(reversed(chain)) if in_function else None


def _is_romatch(name: str | None) -> bool:
    return name is not None and (name == "romatch" or name.startswith("romatch."))


_FROM_ROMATCH = re.compile(r"from\s+romatch(?=[.\s])")


@dataclass
class _ImportCount:
    module_level: int = 0
    functions: tuple[str, ...] = ()


def _rewrite_romatch_imports(
    path: str, tree: ast.Module, lines: list[str]
) -> tuple[list[_Edit], _ImportCount, list[str]]:
    """D-65: point each absolute ``from romatch...`` import at the vendored package."""
    parents = _parents(tree)
    edits: list[_Edit] = []
    count = _ImportCount()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(_is_romatch(a.name) for a in node.names):
            raise VendorError(f"{path}: unexpected 'import romatch' form")
        if not (isinstance(node, ast.ImportFrom) and node.level == 0 and _is_romatch(node.module)):
            continue
        line = lines[node.lineno - 1]
        prefix = line[: node.col_offset]
        match = _FROM_ROMATCH.match(line, node.col_offset)
        if prefix.strip() or match is None:
            raise VendorError(f"{path}: romatch import not where ast puts it")
        new_line = line[: match.start()] + match.group(0).replace("romatch", PACKAGE, 1)
        edits.append(_Edit(node.lineno, node.lineno, (new_line + line[match.end() :],)))
        function = _enclosing_function(node, parents)
        if function is None:
            count.module_level += 1
        else:
            count.functions += (function,)
    notes: list[str] = []
    if edits:
        where = [f"{count.module_level} at module level"]
        if count.functions:
            where.append(f"{len(count.functions)} inside {', '.join(count.functions)}")
        noun = "import" if len(edits) == 1 else "imports"
        notes.append(
            f"{len(edits)} absolute romatch {noun} rewritten to {PACKAGE} ({'; '.join(where)})"
        )
    return edits, count, notes


def _is_blank_or_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped == "" or stripped.startswith("#")


def _node_lines(node: ast.stmt) -> tuple[int, int]:
    """First and last line of a top-level statement, decorators included."""
    first = node.lineno
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        first = min([first, *(d.lineno for d in node.decorator_list)])
    assert node.end_lineno is not None
    return first, node.end_lineno


def _collapsed_cut(lines: list[str], first: int, last: int) -> _Edit:
    """Delete lines first..last, keeping at most two blank lines at the cut."""
    above = 0
    while first - 1 - above >= 1 and lines[first - 2 - above].strip() == "":
        above += 1
    below = 0
    while last + 1 + below <= len(lines) and lines[last + below].strip() == "":
        below += 1
    excess = max(0, above + below - 2)
    take_below = min(excess, below)
    take_above = excess - take_below
    return _Edit(first - take_above, last + take_below)


def _remove_cv2_and_pose_helpers(
    path: str, tree: ast.Module, lines: list[str]
) -> tuple[list[_Edit], list[str]]:
    """Q-113: drop the module-level cv2 import and the span of the three pose helpers."""
    body = tree.body
    cv2_imports = [
        node
        for node in body
        if isinstance(node, ast.Import) and any(a.name == "cv2" for a in node.names)
    ]
    if len(cv2_imports) != 1:
        raise VendorError(f"{path}: expected 1 module-level cv2 import, found {len(cv2_imports)}")
    cv2_import = cv2_imports[0]
    if [(a.name, a.asname) for a in cv2_import.names] != [("cv2", None)]:
        raise VendorError(f"{path}: the cv2 import imports other names too")
    if cv2_import.end_lineno != cv2_import.lineno or cv2_import.col_offset != 0:
        raise VendorError(f"{path}: the cv2 import does not fill a line of its own")
    tail = lines[cv2_import.lineno - 1].encode()[cv2_import.end_col_offset or 0 :]
    if tail.strip() and not tail.strip().startswith(b"#"):
        raise VendorError(f"{path}: the cv2 import shares its line")

    helpers = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in POSE_HELPERS]
    if sorted(n.name for n in helpers) != sorted(POSE_HELPERS):
        raise VendorError(f"{path}: expected one top-level def each of {', '.join(POSE_HELPERS)}")
    ranges = [_node_lines(n) for n in helpers]
    first = min(r[0] for r in ranges)
    last = max(r[1] for r in ranges)
    for node in body:
        if node in helpers:
            continue
        node_first, node_last = _node_lines(node)
        if node_first <= last and node_last >= first:
            raise VendorError(f"{path}: the span of the pose helpers holds another statement")
    covered = {i for lo, hi in ranges for i in range(lo, hi + 1)}
    gaps = [i for i in range(first, last + 1) if i not in covered]
    if not all(_is_blank_or_comment(lines[i - 1]) for i in gaps):
        raise VendorError(
            f"{path}: the span of the pose helpers holds more than blank or comment lines"
        )

    edits = [
        _collapsed_cut(lines, cv2_import.lineno, cv2_import.lineno),
        _collapsed_cut(lines, first, last),
    ]
    names = ", ".join(n.name for n in sorted(helpers, key=lambda n: n.lineno))
    notes = [
        f"the module-level import of cv2 (upstream line {cv2_import.lineno}) removed",
        f"upstream lines {first}-{last} removed: the functions {names} and only blank or "
        "comment lines between them; blank lines at the cut kept to at most two",
    ]
    return edits, notes


def _drop_pose_helper_names(
    path: str, tree: ast.Module, lines: list[str]
) -> tuple[list[_Edit], list[str]]:
    """Q-113: stop utils/__init__.py importing the removed pose helpers."""
    imports = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "utils"
    ]
    if len(imports) != 1:
        raise VendorError(f"{path}: expected one 'from .utils import' statement")
    edits: list[_Edit] = []
    dropped: list[str] = []
    for alias in imports[0].names:
        if alias.name not in POSE_HELPERS:
            continue
        if alias.asname is not None or alias.lineno != alias.end_lineno:
            raise VendorError(f"{path}: {alias.name} is not imported on a line of its own")
        if lines[alias.lineno - 1].strip() != f"{alias.name},":
            raise VendorError(f"{path}: {alias.name} is not imported on a line of its own")
        edits.append(_Edit(alias.lineno, alias.lineno))
        dropped.append(alias.name)
    if sorted(dropped) != sorted(POSE_HELPERS):
        raise VendorError(f"{path}: expected to drop {', '.join(POSE_HELPERS)}")
    return edits, [f"{', '.join(dropped)} dropped from the import from .utils"]


def _imports_xformers(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (
            node.level == 0 and node.module is not None and node.module.split(".")[0] == "xformers"
        )
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "xformers" for a in node.names)
    return False


def _replace_xformers_try(
    path: str, tree: ast.Module, lines: list[str]
) -> tuple[list[_Edit], list[str]]:
    """Q-114: replace the try/except ImportError around xformers by its except branch."""
    tries = [
        n for n in tree.body if isinstance(n, ast.Try) and any(_imports_xformers(s) for s in n.body)
    ]
    if len(tries) != 1:
        raise VendorError(f"{path}: expected one module-level try importing xformers")
    node = tries[0]
    elsewhere = [n for n in ast.walk(tree) if _imports_xformers(n) and n not in node.body]
    if elsewhere:
        raise VendorError(f"{path}: xformers imported outside the try ({len(elsewhere)})")
    if node.orelse or node.finalbody or len(node.handlers) != 1:
        raise VendorError(f"{path}: the try has else, finally or more than one handler")
    handler = node.handlers[0]
    if (
        not (isinstance(handler.type, ast.Name) and handler.type.id == "ImportError")
        or handler.name
    ):
        raise VendorError(f"{path}: the try's handler is not 'except ImportError:'")
    if not all(_imports_xformers(s) or isinstance(s, ast.Assign) for s in node.body):
        raise VendorError(f"{path}: the try holds more than xformers imports and assignments")
    if not all(isinstance(s, ast.Assign) for s in handler.body):
        raise VendorError(f"{path}: the except branch holds more than assignments")
    new: list[str] = []
    assigned: list[str] = []
    for statement in handler.body:
        assert isinstance(statement, ast.Assign) and statement.end_lineno is not None
        indent = statement.col_offset
        for number in range(statement.lineno, statement.end_lineno + 1):
            line = lines[number - 1]
            if line[:indent].strip():
                raise VendorError(f"{path}: the except branch is not indented as ast says")
            new.append(line[indent:])
        assigned.extend(t.id for t in statement.targets if isinstance(t, ast.Name))
    assert node.end_lineno is not None
    note = (
        "the module-level try/except ImportError statement around the xformers import "
        "replaced by the dedented statements of its except branch, which assign "
        f"{', '.join(assigned)}"
    )
    return [_Edit(node.lineno, node.end_lineno, tuple(new))], [note]


def _notice(lines: list[str]) -> tuple[_Edit, str]:
    """Q-115: the notice goes after the leading run of comment lines, else at line 1."""
    header = 0
    while header < len(lines) and lines[header].startswith("#"):
        header += 1
    where = "at line 1" if header == 0 else f"after the {header}-line leading comment header"
    return _Edit(header + 1, header, (f"# {NOTICE}",)), f"modification notice added {where}"


@dataclass
class _Patched:
    data: bytes
    notes: list[str]
    imports: _ImportCount


def _patch(path: str, data: bytes) -> _Patched:
    text = data.decode("utf-8")
    if "\r" in text:
        raise VendorError(f"{path}: carriage returns are not handled")
    tree = ast.parse(text, filename=path)
    lines = text.split("\n")
    edits, imports, notes = _rewrite_romatch_imports(path, tree, lines)
    for applies, rule in (
        (path == UTILS, _remove_cv2_and_pose_helpers),
        (path == UTILS_INIT, _drop_pose_helper_names),
        (path in XFORMERS_FILES, _replace_xformers_try),
    ):
        if applies:
            more_edits, more_notes = rule(path, tree, lines)
            edits += more_edits
            notes += more_notes
    if edits:
        notice, note = _notice(lines)
        edits.append(notice)
        notes.append(note)
    out = "\n".join(_apply(lines, edits, path))
    return _Patched(out.encode("utf-8"), notes, imports)


def _names_used(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.alias):
            yield node.name.split(".")[0]
            yield node.name.split(".")[-1]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module.split(".")[0]


def _check_patched(files: dict[str, bytes]) -> None:
    """After the patch: no romatch, cv2 or xformers import and no pose helper anywhere."""
    for path, data in files.items():
        tree = ast.parse(data, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                if _imports_xformers(node):
                    raise VendorError(f"{path}: an xformers import is left")
                modules = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if node.level == 0
                    else []
                )
                if any(_is_romatch(m) for m in modules):
                    raise VendorError(f"{path}: an absolute romatch import is left")
        used = set(_names_used(tree))
        if "cv2" in used:
            raise VendorError(f"{path}: cv2 is still named")
        left = sorted(used & set(POSE_HELPERS))
        if left:
            raise VendorError(f"{path}: {', '.join(left)} still named")
    functions = [n.name for n in ast.parse(files[UTILS]).body if isinstance(n, ast.FunctionDef)]
    if not functions or functions[0] != FIRST_KEPT_FUNCTION:
        raise VendorError(f"{UTILS}: the first function is not {FIRST_KEPT_FUNCTION}")


# ---------------------------------------------------------------------------------------
# Licenses


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data: bytes = response.read()
    return data


def _licenses(source_tree_license: str) -> dict[str, bytes]:
    texts: dict[str, bytes] = {}
    for license_ in LICENSES:
        data = _fetch(license_.source)
        digest = hashlib.sha256(data).hexdigest()
        if digest != license_.sha256:
            raise VendorError(f"{license_.project} license: sha256 {digest} != {license_.sha256}")
        if license_.project == "RoMa" and _blob_id(data) != source_tree_license:
            raise VendorError("RoMa license: not the LICENSE blob of the commit's tree")
        texts[license_.path] = data
    return texts


# ---------------------------------------------------------------------------------------
# Build, write and check


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(record: dict[str, object]) -> bytes:
    return (json.dumps(record, indent=2) + "\n").encode("utf-8")


def build(source: Path, out: Path) -> None:
    """Write the vendored copy, its license files and VENDOR.json into ``out``."""
    blobs = _tree_blobs(source)
    patched: dict[str, _Patched] = {}
    entries: list[dict[str, object]] = []
    upstream_lines = 0
    for path in FILES:
        upstream = f"romatch/{path}"
        data = _read_blob(source, blobs, upstream)
        result = _patch(path, data)
        patched[path] = result
        lines_before, lines_after = data.count(b"\n"), result.data.count(b"\n")
        upstream_lines += lines_before
        entries.append(
            {
                "path": path,
                "upstream_path": upstream,
                "upstream_blob": blobs[upstream],
                "sha256": _sha256(result.data),
                "patch": result.notes,
            }
        )
        changed = "changed" if result.notes else "unchanged"
        print(
            f"{path}: blob {blobs[upstream][:12]}, {lines_before} -> {lines_after} lines, "
            f"{changed}, sha256 {_sha256(result.data)[:16]}"
        )
    module_level = sum(p.imports.module_level for p in patched.values())
    functions = [f for p in patched.values() for f in p.imports.functions]
    if (module_level, len(functions)) != (
        ROMATCH_IMPORTS_AT_MODULE_LEVEL,
        ROMATCH_IMPORTS_IN_FUNCTIONS,
    ):
        raise VendorError(
            f"romatch imports: {module_level} at module level and {len(functions)} in functions"
        )
    _check_patched({path: p.data for path, p in patched.items()})
    licenses = _licenses(blobs["LICENSE"])

    for path, result in patched.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.data)
    for path, data in licenses.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record: dict[str, object] = {
        "upstream": UPSTREAM,
        "commit": COMMIT,
        "script": SCRIPT,
        "package": PACKAGE,
        "notice": NOTICE,
        "licenses": [
            {
                "path": lic.path,
                "project": lic.project,
                "license": lic.spdx,
                "source": lic.source,
                "sha256": lic.sha256,
            }
            for lic in LICENSES
        ],
        "files": entries,
    }
    (out / "VENDOR.json").write_bytes(_dump(record))
    changed_files = sum(1 for p in patched.values() if p.notes)
    print(
        f"{len(FILES)} files ({upstream_lines} upstream lines), {changed_files} changed; "
        f"romatch imports rewritten: {module_level} at module level, "
        f"{len(functions)} in {', '.join(functions)}; {len(licenses)} license files"
    )


def _files_under(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def check(source: Path, dest: Path) -> bool:
    """Rebuild in a temporary folder; True when the copy at ``dest`` is the same."""
    with tempfile.TemporaryDirectory() as tmp:
        build(source, Path(tmp))
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
    print("check: the copy and VENDOR.json match the rebuild" if ok else "check: MISMATCH")
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--source", type=Path, required=True, help="a git clone holding the commit")
    parser.add_argument("--dest", type=Path, default=DEST, help="the vendored copy's folder")
    parser.add_argument(
        "--check", action="store_true", help="rebuild in a temporary folder and compare"
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            return 0 if check(args.source, args.dest) else 1
        with tempfile.TemporaryDirectory() as tmp:
            build(args.source, Path(tmp))
            if args.dest.exists():
                shutil.rmtree(args.dest)
            shutil.copytree(tmp, args.dest)
    except VendorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

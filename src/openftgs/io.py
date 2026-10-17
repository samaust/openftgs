# SPDX-License-Identifier: Apache-2.0
"""Reading and writing full checkpoints (TrainerState), FTGS-IF exchange files (ExchangeModel),
ftgs-init files (InitPoints), render requests (RenderRequest) and their renders (Renders), scene
manifests (SceneManifest), and PLY files for viewers, with the validators of the exchange and
init formats. Plain data in and out: this module imports neither openftgs.model nor
openftgs.train.

Spec: §2.2, D-30, D-34, D-73; Q-64. Band: Foundation (§2.1).
"""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs.config import config_from_dict
from openftgs.types import (
    Camera,
    ExchangeModel,
    ImageRecord,
    InitPoints,
    RenderItem,
    RenderRequest,
    Renders,
    SceneManifest,
    TimeNormalizer,
    TrainerState,
)

__all__ = [
    "load_checkpoint",
    "read_exchange",
    "read_init_points",
    "read_render_request",
    "read_renders",
    "read_scene_manifest",
    "save_checkpoint",
    "validate_exchange",
    "validate_init_points",
    "write_exchange",
    "write_init_points",
    "write_ply",
    "write_renders",
    "write_scene_manifest",
]

_VERSION = 1  # the major version this module reads and writes (§2.2)
_VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?")


# Shared helpers: safetensors files with the §2.2 metadata header.


def _fail(where: str, reason: str) -> NoReturn:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(f"{where}: {reason}")


def _check_header(meta: Mapping[str, object], fmt: str, where: str) -> None:
    """§2.2: `format` must match and the major part of `version` must be 1."""
    if meta.get("format") != fmt:
        _fail(where, f"format must be {fmt!r}, got {meta.get('format')!r}")
    version = meta.get("version")
    if version is None:
        _fail(where, "missing version")
    # §2.2: integers in decimal; a major number, optionally with a minor part.
    match = _VERSION_PATTERN.fullmatch(str(version))
    if match is None:
        _fail(where, f"version must be a decimal major[.minor], got {version!r}")
    if int(match.group(1)) != _VERSION:
        _fail(where, f"unsupported version {version!r}; this reader handles major version 1")


def _read_safetensors(path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
            tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118
    except (OSError, SafetensorError) as err:
        raise ValueError(f"{path}: not a readable safetensors file: {err}") from None
    return tensors, meta


def _write_safetensors(tensors: Mapping[str, Tensor], path: Path, meta: Mapping[str, str]) -> None:
    save_file(
        {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()},
        str(path),
        # safetensors 0.8.0 writes an unreadable header for no tensors with an empty metadata
        # map, so an empty map is passed as None (source log #11).
        metadata=dict(meta) or None,
    )


def _replace_with_safetensors(
    tensors: Mapping[str, Tensor], path: Path, meta: Mapping[str, str]
) -> None:
    """Q-126 (as answered, #179): write a single-file output to a temporary sibling, named as
    Q-25's `.<name>.tmp-<pid>`, and rename it over `path`, so a write that fails partway leaves
    the old file, or no file, and no temporary. A symlink at `path` becomes a regular file
    (Q-146)."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        _write_safetensors(tensors, temporary, meta)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_tensor(
    tensors: Mapping[str, Tensor | None],
    name: str,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
    where: str,
    *,
    required: bool = True,
) -> Tensor | None:
    """The tensor `name` with exactly `dtype` and `shape` (None matches any size), all finite."""
    value = tensors.get(name)
    if value is None:
        if required:
            _fail(where, f"missing required tensor {name!r}")
        return None
    if value.dtype != dtype:
        _fail(where, f"{name} must be {dtype}, got {value.dtype}")
    if value.dim() != len(shape) or any(
        want is not None and got != want for got, want in zip(value.shape, shape, strict=True)
    ):
        _fail(where, f"{name} must have shape {list(shape)}, got {list(value.shape)}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        _fail(where, f"{name} has values that are not finite")
    return value


# ftgs-init (§2.2): initial points.

_INIT = "ftgs-init"


# Q-22: the ftgs-init validator, shared by the reader, the writer and the model.
def validate_init_points(points: InitPoints) -> None:
    """§2.2 ftgs-init validation (Q-22): the table's dtypes and shapes, N ≥ 1, all values finite,
    colors within [0, 1], durations positive. Raises ValueError (Q-26)."""
    where = "ftgs-init"
    positions = points.positions
    if not isinstance(positions, Tensor) or positions.dim() != 2:
        _fail(where, "positions must be an [N, 3] tensor")
    n = int(positions.shape[0])
    if n < 1:
        _fail(where, "needs at least one point")
    tensors: dict[str, Tensor | None] = {
        "positions": points.positions,
        "colors": points.colors,
        "timestamps": points.timestamps,
        "velocities": points.velocities,
        "durations": points.durations,
        "frame_ids": points.frame_ids,
    }
    _check_tensor(tensors, "positions", torch.float32, (n, 3), where)
    colors = _check_tensor(tensors, "colors", torch.float32, (n, 3), where)
    _check_tensor(tensors, "timestamps", torch.float64, (n,), where)
    _check_tensor(tensors, "velocities", torch.float32, (n, 3), where, required=False)
    durations = _check_tensor(tensors, "durations", torch.float64, (n,), where, required=False)
    _check_tensor(tensors, "frame_ids", torch.int64, (n,), where, required=False)
    assert colors is not None  # required=True
    if not bool(((colors >= 0.0) & (colors <= 1.0)).all()):
        _fail(where, "colors must lie within [0, 1]")
    if durations is not None and not bool((durations > 0.0).all()):
        _fail(where, "durations must be positive")


_INIT_RESERVED = ("format", "version", "time_unit", "producer")  # §2.2 ftgs-init metadata


def write_init_points(
    points: InitPoints,
    path: Path,
    *,
    time_unit: str,
    producer: str,
    metadata: Mapping[str, str] | None = None,  # Q-107
) -> None:
    """Write a ftgs-init v1 file (§2.2; Q-22), validating first. `metadata` adds string keys
    and values beside §2.2's own, such as `roma_init`'s provenance (Q-107); a reserved key or a
    key or value that isn't a string raises ValueError."""
    validate_init_points(points)
    extra = dict(metadata or {})
    for key, value in extra.items():
        # Q-107: safetensors metadata is a string-to-string map (§2.2 introduction).
        if not isinstance(key, str) or not isinstance(value, str):
            _fail("ftgs-init", f"metadata must map strings to strings, got {key!r}: {value!r}")
        if key in _INIT_RESERVED:
            _fail("ftgs-init", f"metadata key {key!r} is reserved")
    tensors = {
        name: value
        for name in ("positions", "colors", "timestamps", "velocities", "durations", "frame_ids")
        if (value := getattr(points, name)) is not None
    }
    meta = {
        "format": _INIT,
        "version": str(_VERSION),
        "time_unit": time_unit,
        "producer": producer,
        **extra,
    }
    _replace_with_safetensors(tensors, Path(path), meta)


def read_init_points(path: Path, *, time_unit: str | None = None) -> InitPoints:
    """Read and validate a ftgs-init file (§2.2; Q-22). With `time_unit`, a file in another
    unit is rejected."""
    tensors, meta = _read_safetensors(Path(path))
    _check_header(meta, _INIT, str(path))
    if "time_unit" not in meta:  # §2.2 ftgs-init metadata
        _fail(str(path), "missing metadata 'time_unit'")
    if time_unit is not None and meta.get("time_unit") != time_unit:
        _fail(str(path), f"time_unit is {meta.get('time_unit')!r}, expected {time_unit!r}")
    for name in ("positions", "colors", "timestamps"):
        if name not in tensors:
            _fail(str(path), f"missing required tensor {name!r}")
    points = InitPoints(
        positions=tensors["positions"],
        colors=tensors["colors"],
        timestamps=tensors["timestamps"],
        velocities=tensors.get("velocities"),
        durations=tensors.get("durations"),
        frame_ids=tensors.get("frame_ids"),
    )
    try:
        validate_init_points(points)
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from None
    return points


# ftgs-if (§2.2): the exchange model, physical values in raw time units.

_IF = "ftgs-if"
_SH_DEGREES = (0, 1, 2, 3)  # D-3
_MIN_ROTATION_NORM = 1e-8  # §1.1 validity rules
_UNIT_TOLERANCE = 1e-6  # Q-24: |‖q‖ − 1| allowed on write, a few float32 roundings

# Q-24: the render conventions written with every exchange file, from E6's normative constants
# (gsplat 1.5.3 classic mode, D-6, D-31) and the black background of D-34; D-73 adds the
# Jacobian clamp margin, the screen-extent rule and the rasterize mode, under Q-35's keys.
_RENDER_CONVENTIONS = {
    "alpha_max": 0.999,
    "eps2d": 0.3,
    "near": 0.01,
    "far": 1e10,
    "alpha_skip": 1.0 / 255.0,
    "transmittance_stop": 1e-4,
    "pixel_center_offset": 0.5,
    "background": [0.0, 0.0, 0.0],
    # D-73, Q-35:
    "jacobian_clamp_margin": 0.15,  # E6: beyond the image edges, as a fraction of its size; J only
    "screen_extent_rule": "opacity_aware",  # E6: √(2 ln(255·σ_eff)) standard deviations
    "screen_extent_max": 3.33,  # E6: the extent's cap
    "rasterize_mode": "classic",  # D-6
}


# Q-22, Q-24: the ftgs-if validator, shared by the reader, the writer and the model.
def validate_exchange(model: ExchangeModel) -> None:
    """§2.2 ftgs-if validation (Q-22, Q-24): the table's keys, dtypes and shapes, N ≥ 1, all
    values finite, opacity in (0, 1), durations and scales positive, rotations of norm > 1e-8,
    sh_degree in {0, 1, 2, 3}, finite τ₀ ≤ τ₁, a background of three numbers in [0, 1] (D-84).
    Raises ValueError (Q-26)."""
    where = "ftgs-if"
    degree = model.sh_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or degree not in _SH_DEGREES:
        _fail(where, f"sh_degree must be one of {_SH_DEGREES}, got {degree!r}")
    background = model.background
    if (
        not isinstance(background, tuple)
        or len(background) != 3
        or not all(
            isinstance(c, int | float) and not isinstance(c, bool) and 0.0 <= c <= 1.0
            for c in background
        )
    ):  # D-84, Q-11's range; NaN fails the comparison
        _fail(where, f"background must be three numbers in [0, 1], got {background!r}")
    tau0, tau1 = model.tau0, model.tau1
    if not (math.isfinite(tau0) and math.isfinite(tau1)) or tau1 < tau0:
        _fail(where, f"tau0 and tau1 must be finite with tau0 <= tau1, got {tau0!r}, {tau1!r}")
    a = model.arrays
    position = a.get("position")
    if not isinstance(position, Tensor) or position.dim() != 2:
        _fail(where, "position must be an [N, 3] tensor")
    n = int(position.shape[0])
    if n < 1:
        _fail(where, "needs at least one primitive")
    _check_tensor(a, "position", torch.float32, (n, 3), where)
    _check_tensor(a, "time_center", torch.float64, (n,), where)
    duration = _check_tensor(a, "duration", torch.float64, (n,), where)
    _check_tensor(a, "velocity", torch.float32, (n, 3), where)
    scale = _check_tensor(a, "scale", torch.float32, (n, 3), where)
    rotation = _check_tensor(a, "rotation", torch.float32, (n, 4), where)
    opacity = _check_tensor(a, "opacity", torch.float32, (n,), where)
    _check_tensor(a, "sh", torch.float32, (n, (degree + 1) ** 2, 3), where)
    assert duration is not None and scale is not None  # required=True
    assert rotation is not None and opacity is not None
    if not bool((duration > 0.0).all()):
        _fail(where, "duration must be positive")
    if not bool((scale > 0.0).all()):
        _fail(where, "scale must be positive")
    if not bool((torch.linalg.vector_norm(rotation, dim=-1) > _MIN_ROTATION_NORM).all()):
        _fail(where, f"rotation norms must exceed {_MIN_ROTATION_NORM}")
    if not bool(((opacity > 0.0) & (opacity < 1.0)).all()):
        _fail(where, "opacity must lie within (0, 1)")
    # D-71: §1.4 "Time": with a single timestamp, t ≡ 0 and velocities are frozen at 0.
    if tau1 == tau0:
        time_center = a["time_center"]
        velocity = a["velocity"]
        if bool((time_center != tau0).any()) or bool(velocity.any()):
            _fail(
                where,
                "with a single timestamp (tau0 = tau1), every time_center must be tau0 "
                "and every velocity 0 (§1.4)",
            )


def _check_canonical_rotation(rotation: Tensor, where: str) -> None:
    """§2.2 ftgs-if: unit quaternions with w ≥ 0; Q-24: where w = 0, the first nonzero component
    is positive. Checked on write only; readers accept any rotation of norm > 1e-8."""
    norm = torch.linalg.vector_norm(rotation.to(torch.float64), dim=-1)
    if not bool(((norm - 1.0).abs() <= _UNIT_TOLERANCE).all()):
        _fail(where, "rotation must hold unit quaternions")
    first = torch.argmax((rotation != 0).to(torch.int8), dim=-1, keepdim=True)
    if bool((torch.gather(rotation, -1, first) < 0).any()):
        _fail(where, "rotation must have w >= 0 (where w = 0, the first nonzero component > 0)")


_IF_KEYS = ("position", "time_center", "duration", "velocity", "scale", "rotation", "opacity", "sh")


def write_exchange(model: ExchangeModel, path: Path) -> None:
    """Write a ftgs-if v1 file (§2.2), validating first (Q-22). Writing is stricter than
    reading: rotations must be unit quaternions with the canonical sign (Q-24)."""
    validate_exchange(model)
    _check_canonical_rotation(model.arrays["rotation"], f"ftgs-if {path}")
    meta = {
        "format": _IF,
        "version": str(_VERSION),
        "sh_degree": str(model.sh_degree),
        "time_unit": model.time_unit,
        "tau0": repr(float(model.tau0)),
        "tau1": repr(float(model.tau1)),
        "producer": model.producer,
        # D-84: the model's background in place of the renderer's default (D-34).
        "render_conventions": json.dumps(
            {**_RENDER_CONVENTIONS, "background": [float(c) for c in model.background]},
            separators=(",", ":"),
        ),
    }
    _replace_with_safetensors({k: model.arrays[k] for k in _IF_KEYS}, Path(path), meta)


def read_exchange(path: Path) -> ExchangeModel:
    """Read and validate a ftgs-if file (§2.2; Q-22). Unknown keys and render_conventions are
    ignored (Q-24)."""
    tensors, meta = _read_safetensors(Path(path))
    where = str(path)
    _check_header(meta, _IF, where)
    for key in ("sh_degree", "tau0", "tau1", "time_unit"):
        if key not in meta:
            _fail(where, f"missing metadata {key!r}")
    try:
        degree = int(meta["sh_degree"])
        tau0, tau1 = float(meta["tau0"]), float(meta["tau1"])
    except ValueError:
        _fail(where, "sh_degree, tau0 and tau1 must be numbers")
    model = ExchangeModel(
        arrays={k: tensors[k] for k in _IF_KEYS if k in tensors},
        sh_degree=degree,
        tau0=tau0,
        tau1=tau1,
        time_unit=meta["time_unit"],
        producer=meta.get("producer", ""),
    )
    try:
        validate_exchange(model)
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from None
    return model


# ftgs-checkpoint (§2.2): a directory holding a TrainerState in full.

_CKPT = "ftgs-checkpoint"
# §1.1: the nine stored tensors, in FreeTimeGaussians.PARAM_NAMES' order; io may not import the
# model, so a test pins this copy to it.
_PARAM_NAMES = (
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
    "sh0",
    "shN",
)
_MANIFEST = "manifest.json"
_CKPT_FILES = (
    "params.safetensors",
    "optim.safetensors",
    "optim.json",
    "stats.safetensors",
    "rng.safetensors",
    "rng.json",
    "sampler.json",
)
_SEP = "/"  # Q-25: separator of the key paths under which nested tensors are stored


def _split_tensors(
    tree: Mapping[str, Any], where: str, prefix: str = ""
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Q-25: every tensor in a tree of dicts goes to a flat map under its `/`-joined key path;
    the rest stays in a JSON-ready tree. Tensors inside lists are refused."""
    tensors: dict[str, Tensor] = {}
    rest: dict[str, Any] = {}
    for key, value in tree.items():
        if not isinstance(key, str) or _SEP in key:
            _fail(where, f"keys must be strings without {_SEP!r}, got {key!r}")
        path = prefix + key
        if isinstance(value, Tensor):
            tensors[path] = value
        elif isinstance(value, Mapping):
            sub_tensors, sub_rest = _split_tensors(value, where, path + _SEP)
            tensors.update(sub_tensors)
            rest[key] = sub_rest
        else:
            _check_json(value, f"{where}: {path}")
            rest[key] = value
    return tensors, rest


def _check_json(value: Any, where: str) -> None:
    """A JSON value with finite numbers and string keys; tensors are refused here."""
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(where, f"non-finite number {value!r}")
        return
    if isinstance(value, list | tuple):
        for item in value:
            _check_json(item, where)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(where, f"keys must be strings, got {key!r}")
            _check_json(item, where)
        return
    _fail(where, f"{type(value).__name__} is not a JSON value")


def _merge_tensors(tree: dict[str, Any], tensors: Mapping[str, Tensor], where: str) -> None:
    """The inverse of _split_tensors: put each tensor back at its key path."""
    for path, value in tensors.items():
        *parents, leaf = path.split(_SEP)
        node = tree
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                _fail(where, f"tensor path {path!r} collides with a JSON value")
            node = child
        if leaf in node:
            _fail(where, f"tensor path {path!r} collides with a JSON value")
        node[leaf] = value


def _json_bytes(value: Any) -> bytes:
    # §2.2: floats in Python's shortest round-trip repr, which json writes; NaN and inf refused.
    return json.dumps(value, allow_nan=False, indent=1).encode("utf-8")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _is_checkpoint(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and manifest.get("format") == _CKPT


def save_checkpoint(state: TrainerState, directory: Path) -> Path:
    """Write `state` as a ftgs-checkpoint v1 directory (§2.2; Q-17, Q-25) and return its path.

    The files are written into a temporary sibling directory that then replaces `directory`, so
    a reader sees either the old checkpoint or the new one. An existing directory is replaced
    only if it is a checkpoint or empty. POSIX can't swap two directories in one rename: a kill
    between the two renames leaves the old checkpoint set aside beside the target, and the next
    save of the same target restores it before replacing it (Q-25). Concurrent saves to one
    target are not supported."""
    directory = Path(directory)
    where = f"checkpoint {directory}"
    if set(state.params) != set(_PARAM_NAMES):
        _fail(where, f"params must hold exactly {_PARAM_NAMES}, got {sorted(state.params)}")
    optim_tensors, optim_rest = _split_tensors(state.optim, f"{where} optim")
    rng_tensors, rng_rest = _split_tensors(state.rng, f"{where} rng")
    stats_tensors, stats_rest = _split_tensors(state.stats, f"{where} stats")
    if stats_rest:
        _fail(where, f"stats must hold tensors only, got {sorted(stats_rest)}")
    _check_json(state.sampler, f"{where} sampler")
    _check_json(state.config, f"{where} config")
    n = int(state.params["means"].shape[0])
    for name in _PARAM_NAMES:
        if state.params[name].dtype != torch.float32:
            _fail(where, f"params[{name!r}] must be float32 (§2.2), got {state.params[name].dtype}")
    for name, value in stats_tensors.items():
        if tuple(value.shape) != (n,):
            _fail(where, f"stats[{name!r}] must have shape [{n}], got {list(value.shape)}")
    # Q-25: recover from a save killed between its two renames, then sweep leftovers.
    # The newest set-aside checkpoint is the one a killed save left last.
    aside = sorted(directory.parent.glob(f".{directory.name}.old-*"), key=_mtime)
    if not directory.exists() and aside and _is_checkpoint(aside[-1]):
        aside.pop().rename(directory)
    for leftover in [*aside, *directory.parent.glob(f".{directory.name}.tmp-*")]:
        shutil.rmtree(leftover, ignore_errors=True)
    # Q-25: a save never deletes a directory that isn't a checkpoint.
    if directory.exists() and not directory.is_dir():
        _fail(where, "exists and is not a directory; refusing to replace it")
    if directory.exists() and not (_is_checkpoint(directory) or not any(directory.iterdir())):
        _fail(where, "exists and is not a checkpoint; refusing to replace it")

    tmp = directory.with_name(f".{directory.name}.tmp-{os.getpid()}")
    old = directory.with_name(f".{directory.name}.old-{os.getpid()}")
    tmp.mkdir(parents=True)
    try:
        # §2.2: every file carries format and version; the JSON parts are covered by the
        # manifest's (Q-25).
        header = {"format": _CKPT, "version": str(_VERSION)}
        params = {k: state.params[k] for k in _PARAM_NAMES}
        _write_safetensors(params, tmp / "params.safetensors", header)
        _write_safetensors(optim_tensors, tmp / "optim.safetensors", header)
        (tmp / "optim.json").write_bytes(_json_bytes(optim_rest))
        _write_safetensors(stats_tensors, tmp / "stats.safetensors", header)
        _write_safetensors(rng_tensors, tmp / "rng.safetensors", header)
        (tmp / "rng.json").write_bytes(_json_bytes(rng_rest))
        (tmp / "sampler.json").write_bytes(_json_bytes(state.sampler))
        manifest = {
            "format": _CKPT,
            "version": _VERSION,
            "iteration": state.iteration,
            "config": state.config,
            "tau0": state.time.tau0,
            "tau1": state.time.tau1,
            "spatial_scale": state.spatial_scale,
            "frame_dt": state.frame_dt,
            "sh_degree": state.config.get("sh_degree"),
            "seed": state.config.get("seed"),
            "scene_sha256": state.scene_sha256,
            "time_unit": state.time_unit,
            "versions": {
                "openftgs": _package_version("openftgs"),
                "torch": torch.__version__,
                "gsplat": _package_version("gsplat"),
                "cuda": torch.version.cuda,
            },
            "files": {name: _sha256(tmp / name) for name in _CKPT_FILES},
        }
        (tmp / _MANIFEST).write_bytes(_json_bytes(manifest))
        if directory.exists():
            directory.rename(old)
        tmp.rename(directory)
    except BaseException:
        if old.exists() and not directory.exists():
            old.rename(directory)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)
    return directory


def load_checkpoint(directory: Path) -> TrainerState:
    """Read a ftgs-checkpoint directory (§2.2; Q-25), checking its format, major version, file
    set and every SHA-256 first. Tensors are returned on the CPU."""
    directory = Path(directory)
    where = f"checkpoint {directory}"
    try:
        manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"{where}: cannot read {_MANIFEST}: {err}") from None
    if not isinstance(manifest, dict):
        _fail(where, f"{_MANIFEST} must hold an object")
    _check_header(
        {k: str(v) for k, v in manifest.items() if k in ("format", "version")}, _CKPT, where
    )
    present = {p.name for p in directory.iterdir()}
    missing = {_MANIFEST, *_CKPT_FILES} - present
    if missing:  # Q-25: files the manifest doesn't list are ignored
        _fail(where, f"missing the files {sorted(missing)}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_CKPT_FILES):
        _fail(where, "the manifest must list the SHA-256 of every other file")
    for name in _CKPT_FILES:
        if _sha256(directory / name) != files[name]:
            _fail(where, f"{name} does not match its SHA-256 in the manifest")
    try:
        params, _ = _read_safetensors(directory / "params.safetensors")
        optim_tensors, _ = _read_safetensors(directory / "optim.safetensors")
        stats, _ = _read_safetensors(directory / "stats.safetensors")
        rng_tensors, _ = _read_safetensors(directory / "rng.safetensors")
        optim = json.loads((directory / "optim.json").read_text(encoding="utf-8"))
        rng = json.loads((directory / "rng.json").read_text(encoding="utf-8"))
        sampler = json.loads((directory / "sampler.json").read_text(encoding="utf-8"))
        state = TrainerState(
            iteration=int(manifest["iteration"]),
            config=manifest["config"],
            time=TimeNormalizer(float(manifest["tau0"]), float(manifest["tau1"])),
            spatial_scale=float(manifest["spatial_scale"]),
            frame_dt=float(manifest["frame_dt"]),
            params=params,
            optim=optim,
            stats=stats,
            rng=rng,
            sampler=sampler,
            scene_sha256=manifest.get("scene_sha256"),
            time_unit=str(manifest.get("time_unit", "s")),
        )
        if not isinstance(optim, dict) or not isinstance(rng, dict):
            _fail(where, "optim.json and rng.json must hold objects")
        _merge_tensors(state.optim, optim_tensors, where)
        _merge_tensors(state.rng, rng_tensors, where)
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"{where}: {err}") from None
    if set(params) != set(_PARAM_NAMES):
        _fail(where, f"params.safetensors must hold exactly {_PARAM_NAMES}")
    # §1.1 validity rules, checked on load (Q-25): a valid configuration (Q-10, Q-11), and N
    # within its N_max, for every reader of a checkpoint.
    try:
        n_max = config_from_dict(state.config).growth.n_max
    except ValueError as err:
        raise ValueError(f"{where}: invalid configuration: {err}") from None
    n = int(params["means"].shape[0])
    if n > n_max:
        _fail(where, f"N = {n} exceeds N_max = {n_max}")
    return state


# ftgs-render-request (JSON) and ftgs-renders (safetensors), §2.2: rendering at any camera and
# time.

_REQUEST = "ftgs-render-request"
_RENDERS = "ftgs-renders"
_ORTHONORMAL_TOLERANCE = 1e-5  # Q-56: max |RᵀR − I| of a request camera's rotation
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")  # Q-56: hashlib's hexdigest


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite_number(value: Any, what: str, where: str) -> float:
    if not _is_number(value) or not math.isfinite(value):
        _fail(where, f"{what} must be a finite number, got {value!r}")
    return float(value)


def _positive_int(value: Any, what: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(where, f"{what} must be an int >= 1, got {value!r}")
    return value


def _matrix(value: Any, rows: int, cols: int, what: str, where: str) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or len(value) != rows
        or any(not isinstance(row, list) or len(row) != cols for row in value)
    ):
        _fail(where, f"{what} must be a {rows}x{cols} list of rows")
    return [[_finite_number(x, what, where) for x in row] for row in value]


# Q-56: one parser for the cameras of the JSON formats; render() reads only fx, fy, cx, cy and
# the world-to-camera rotation and translation (Q-32), so anything it would drop is refused here.
def _parse_camera(value: Any, name: str, where: str) -> Camera:
    if not isinstance(value, dict):
        _fail(where, "camera must be an object")
    width = _positive_int(value.get("width"), "camera width", where)
    height = _positive_int(value.get("height"), "camera height", where)
    k = _matrix(value.get("K"), 3, 3, "K", where)
    fx, fy = k[0][0], k[1][1]
    if not (fx > 0.0 and fy > 0.0):
        _fail(where, f"K must have fx > 0 and fy > 0, got {fx!r} and {fy!r}")
    if k[0][1] != 0.0 or k[1][0] != 0.0 or k[2] != [0.0, 0.0, 1.0]:
        _fail(where, "K must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (no skew)")
    w2c = _matrix(value.get("world_to_camera"), 4, 4, "world_to_camera", where)
    if w2c[3] != [0.0, 0.0, 0.0, 1.0]:
        _fail(where, "world_to_camera must have the last row (0, 0, 0, 1)")
    rotation = torch.tensor([row[:3] for row in w2c[:3]], dtype=torch.float64)
    error = (rotation.T @ rotation - torch.eye(3, dtype=torch.float64)).abs().max()
    if float(error) > _ORTHONORMAL_TOLERANCE or float(torch.linalg.det(rotation)) <= 0.0:
        _fail(where, "world_to_camera's rotation must be orthonormal with determinant +1")
    return Camera(
        K=torch.tensor(k, dtype=torch.float32),
        world_to_camera=torch.tensor(w2c, dtype=torch.float32),
        width=width,
        height=height,
        name=name,
    )


def _check_json_header(raw: Mapping[str, Any], fmt: str, where: str) -> None:
    """§2.2 for the JSON formats: `format`, and a `version` of major 1 written as a JSON int or
    a "major.minor" string (Q-56)."""
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int | str):
        _fail(where, f"version must be an int or a string, got {version!r}")
    _check_header({"format": raw.get("format"), "version": str(version)}, fmt, where)


def _time_unit(raw: Mapping[str, Any], where: str) -> str:
    time_unit = raw.get("time_unit")
    if not isinstance(time_unit, str) or not time_unit:
        _fail(where, f"time_unit must be a non-empty string, got {time_unit!r}")
    return time_unit


def _parse_item(value: Any, where: str) -> RenderItem:
    if not isinstance(value, dict):
        _fail(where, "every item must be an object")
    item_id = value.get("id")
    if not isinstance(item_id, str) or not item_id:
        _fail(where, f"item id must be a non-empty string, got {item_id!r}")
    where = f"{where}, item {item_id!r}"
    timestamp = _finite_number(value.get("timestamp"), "timestamp", where)
    return RenderItem(item_id, timestamp, _parse_camera(value.get("camera"), item_id, where))


def read_render_request(path: Path) -> RenderRequest:
    """Read and validate a ftgs-render-request file (§2.2; Q-56). `background` defaults to
    black (D-34) and `sh_degree` to null, the model's degree; unknown keys are ignored."""
    where = f"render request {path}"
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"{where}: not a readable JSON file: {err}") from None
    if not isinstance(raw, dict):
        _fail(where, "the file must hold an object")
    _check_json_header(raw, _REQUEST, where)
    time_unit = _time_unit(raw, where)
    background = raw.get("background", [0.0, 0.0, 0.0])
    if not isinstance(background, list) or len(background) != 3:
        _fail(where, "background must be a list of three numbers")
    color = tuple(_finite_number(c, "background", where) for c in background)
    if not all(0.0 <= c <= 1.0 for c in color):  # Q-11's range for TrainConfig.background
        _fail(where, f"background components must lie within [0, 1], got {list(color)}")
    sh_degree = raw.get("sh_degree")
    if sh_degree is not None and (
        isinstance(sh_degree, bool)
        or not isinstance(sh_degree, int)
        or sh_degree not in _SH_DEGREES
    ):
        _fail(where, f"sh_degree must be null or one of {_SH_DEGREES}, got {sh_degree!r}")
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        _fail(where, "items must be a non-empty list")
    parsed = tuple(_parse_item(item, where) for item in items)
    ids = [item.id for item in parsed]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        _fail(where, f"item ids must be unique, repeated: {duplicated}")
    return RenderRequest(
        items=parsed,
        time_unit=time_unit,
        background=(color[0], color[1], color[2]),
        sh_degree=sh_degree,
    )


def _check_renders(rgb: Mapping[str, Tensor], alpha: Mapping[str, Tensor], where: str) -> None:
    if set(rgb) != set(alpha):
        unpaired = sorted(set(rgb) ^ set(alpha))
        _fail(where, f"every id needs both rgb and alpha, unpaired: {unpaired}")
    if not rgb:
        _fail(where, "needs at least one render")
    for item_id, image in rgb.items():
        if not item_id:
            _fail(where, "render ids must be non-empty")
        if image.dim() != 3:
            _fail(f"{where}, rgb", f"{item_id} must be [H, W, 3], got {list(image.shape)}")
        h, w = int(image.shape[0]), int(image.shape[1])
        _check_tensor(rgb, item_id, torch.float32, (h, w, 3), f"{where}, rgb")
        _check_tensor(alpha, item_id, torch.float32, (h, w, 1), f"{where}, alpha")


def write_renders(renders: Renders, path: Path, *, producer: str, request_sha256: str) -> None:
    """Write a ftgs-renders v1 file (§2.2; Q-56): `rgb/<id>` [H, W, 3] and `alpha/<id>`
    [H, W, 1], float32, rgb not clamped to 1, with the producer and the SHA-256 of the request
    file's bytes, as 64 lowercase hex digits, as metadata."""
    where = f"renders {path}"
    if not isinstance(request_sha256, str) or not _SHA256_PATTERN.fullmatch(request_sha256):
        _fail(where, f"request_sha256 must be a SHA-256 hex digest, got {request_sha256!r}")
    _check_renders(renders.rgb, renders.alpha, where)
    tensors = {f"rgb/{k}": v for k, v in renders.rgb.items()}
    tensors.update({f"alpha/{k}": renders.alpha[k] for k in renders.rgb})
    meta = {
        "format": _RENDERS,
        "version": str(_VERSION),
        "producer": producer,
        "request_sha256": request_sha256,
    }
    _replace_with_safetensors(tensors, Path(path), meta)


def read_renders(path: Path, *, request_sha256: str | None = None) -> Renders:
    """Read and validate a ftgs-renders file (§2.2; Q-56). With `request_sha256`, a file
    recorded for another request is rejected. Tensors other than rgb/… and alpha/… are
    ignored. The file keeps no item order, so the ids come back sorted: match them to the
    request by id."""
    tensors, meta = _read_safetensors(Path(path))
    where = f"renders {path}"
    _check_header(meta, _RENDERS, where)
    if request_sha256 is not None and meta.get("request_sha256") != request_sha256:
        _fail(
            where,
            f"made for the request {meta.get('request_sha256')!r}, expected {request_sha256!r}",
        )
    rgb = {k.removeprefix("rgb/"): v for k, v in tensors.items() if k.startswith("rgb/")}
    alpha = {k.removeprefix("alpha/"): v for k, v in tensors.items() if k.startswith("alpha/")}
    _check_renders(rgb, alpha, where)
    ids = sorted(rgb)  # Q-56: the file keeps no item order
    return Renders(rgb={k: rgb[k] for k in ids}, alpha={k: alpha[k] for k in ids})


# ftgs-scene (§2.2): the scene manifest. Q-64 (owner answer): parsed here, beside the other
# formats; openftgs.data builds the multi-view video from it.

_SCENE = "ftgs-scene"
_SPLITS = ("train", "test")
_OVERRIDES = ("K", "world_to_camera")  # §2.2: per-image keys for moving or zooming cameras


def _relative_path(value: Any, what: str, where: str) -> Path:
    # §2.2: paths are relative to the manifest file.
    if not isinstance(value, str) or not value:
        _fail(where, f"{what} must be a non-empty path, got {value!r}")
    path = Path(value)
    if path.is_absolute():
        _fail(where, f"{what} must be relative to the manifest, got {value!r}")
    return path


def read_scene_manifest(path: Path) -> SceneManifest:
    """Read and validate a ftgs-scene file (§2.2; Q-64): the image records in file order, with
    paths resolved against the manifest's directory, the test images' masks, the metadata, the
    time unit, the SHA-256 of the file's bytes as read and the camera names in the key order of
    the `cameras` object (Q-108). Cameras are checked as a render
    request's (Q-56); unknown keys are ignored. Whether the files exist is not checked here."""
    path = Path(path)
    where = f"scene manifest {path}"
    try:
        data = path.read_bytes()
        raw = json.loads(data.decode("utf-8"))
    except (OSError, ValueError) as err:
        raise ValueError(f"{where}: not a readable JSON file: {err}") from None
    if not isinstance(raw, dict):
        _fail(where, "the file must hold an object")
    _check_json_header(raw, _SCENE, where)
    time_unit = _time_unit(raw, where)
    cameras = raw.get("cameras")
    if not isinstance(cameras, dict):
        _fail(where, "cameras must be an object")
    parsed: dict[str, tuple[dict[str, Any], Camera]] = {}
    for name, value in cameras.items():
        at = f"{where}, camera {name!r}"
        if not isinstance(value, dict):
            _fail(at, "camera must be an object")
        # §2.2: undistorted pinhole images; an absent model means pinhole.
        if value.get("model", "pinhole") != "pinhole":
            _fail(at, f"model must be 'pinhole', got {value.get('model')!r}")
        parsed[name] = (value, _parse_camera(value, name, at))
    images = raw.get("images")
    if not isinstance(images, list) or not images:
        _fail(where, "images must be a non-empty list")
    root = path.parent
    records: list[ImageRecord] = []
    masks: list[Path | None] = []
    for k, image in enumerate(images):
        at = f"{where}, image {k}"
        if not isinstance(image, dict):
            _fail(at, "every image must be an object")
        name = image.get("camera")
        if not isinstance(name, str) or name not in parsed:
            _fail(at, f"camera must name one of the manifest's cameras, got {name!r}")
        base, camera = parsed[name]
        if any(key in image for key in _OVERRIDES):
            # §2.2: an image may override K and world_to_camera; it gets a camera of its own.
            merged = {**base, **{key: image[key] for key in _OVERRIDES if key in image}}
            camera = _parse_camera(merged, name, at)
        timestamp = _finite_number(image.get("timestamp"), "timestamp", at)
        image_path = root / _relative_path(image.get("path"), "path", at)
        split = image.get("split")
        if split not in _SPLITS:
            _fail(at, f"split must be 'train' or 'test', got {split!r}")
        frame_index = image.get("frame_index")
        if frame_index is not None and (
            isinstance(frame_index, bool) or not isinstance(frame_index, int)
        ):
            _fail(at, f"frame_index must be an int, got {frame_index!r}")
        mask = image.get("mask")
        if mask is not None and split != "test":
            _fail(at, "only test images may carry a mask (§2.2)")
        masks.append(None if mask is None else root / _relative_path(mask, "mask", at))
        records.append(ImageRecord(camera, timestamp, image_path, split, frame_index))
    if not any(record.split == "train" for record in records):
        _fail(where, "needs at least one training image")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        _fail(where, "metadata must be an object")
    return SceneManifest(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        time_unit=time_unit,
        records=tuple(records),
        masks=tuple(masks),
        metadata=metadata,
        camera_names=tuple(cameras),  # Q-108: JSON objects are read in file order
    )


def _float32_json(value: Tensor) -> list[Any]:
    """Q-84: each value as the shortest decimal that reads back to the same float32."""
    array = value.detach().to("cpu", torch.float32).numpy()
    return [[float(str(x)) for x in row] for row in array]


def _camera_entry(camera: Camera) -> dict[str, Any]:
    return {
        "width": camera.width,
        "height": camera.height,
        "model": "pinhole",
        "K": _float32_json(camera.K),
        "world_to_camera": _float32_json(camera.world_to_camera),
    }


def _manifest_path(path: Path, root: Path, what: str, where: str) -> str:
    """§2.2: paths are written relative to the manifest, in POSIX form."""
    try:
        relative = Path(os.path.relpath(Path(path).absolute(), root.absolute()))
    except ValueError:  # another drive on Windows
        _fail(where, f"{what} {path} is not under the manifest's directory {root}")
    if relative.parts[:1] == ("..",):
        _fail(where, f"{what} {path} is not under the manifest's directory {root}")
    return relative.as_posix()


def write_scene_manifest(
    path: Path,
    records: Sequence[ImageRecord],
    *,
    time_unit: str = "s",
    masks: Sequence[Path | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SceneManifest:
    """Write a ftgs-scene v1 file (§2.2; Q-64, Q-84) and return it as `read_scene_manifest`
    reads it back. One camera entry per camera name comes from that name's first record; a
    later record whose K or world-to-camera differs gets a per-image override. Camera values are
    written as the shortest decimals that read back to the same float32 values, so the reader
    rebuilds each camera bitwise. Image and mask paths must lie under the manifest's directory.
    The file is written to a temporary sibling, validated by reading it back, then renamed."""
    path = Path(path)
    where = f"scene manifest {path}"
    root = path.parent
    masks = [None] * len(records) if masks is None else list(masks)
    if len(masks) != len(records):
        _fail(where, f"masks must align with the records: {len(masks)} for {len(records)}")
    cameras: dict[str, Camera] = {}
    entries: dict[str, dict[str, Any]] = {}
    images: list[dict[str, Any]] = []
    for k, (record, mask) in enumerate(zip(records, masks, strict=True)):
        at = f"{where}, image {k}"
        camera = record.camera
        image: dict[str, Any] = {"camera": camera.name, "timestamp": float(record.timestamp)}
        if record.frame_index is not None:
            image["frame_index"] = int(record.frame_index)
        image["path"] = _manifest_path(record.image_path, root, "image", at)
        image["split"] = record.split
        base = cameras.setdefault(camera.name, camera)
        if base is not camera:
            if (base.width, base.height) != (camera.width, camera.height):
                _fail(
                    at,
                    f"camera {camera.name!r} must keep one size, "
                    f"{(base.width, base.height)} and {(camera.width, camera.height)} differ",
                )
            entry = entries[camera.name]
            own = _camera_entry(camera)
            for key in _OVERRIDES:
                if own[key] != entry[key]:
                    image[key] = own[key]
        else:
            entries[camera.name] = _camera_entry(camera)
        if mask is not None:
            image["mask"] = _manifest_path(mask, root, "mask", at)
        images.append(image)
    raw = {
        "format": _SCENE,
        "version": _VERSION,
        "time_unit": time_unit,
        "cameras": entries,
        "images": images,
        "metadata": dict(metadata or {}),
    }
    try:
        text = json.dumps(raw, indent=1, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{where}: not writable as JSON: {err}") from None
    temporary = root / f".{path.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(text + "\n", encoding="utf-8")
        read_scene_manifest(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return read_scene_manifest(path)


# PLY snapshot for viewers (Q-18): export-only, never read back.

# §2.2 ftgs-if opacity clamp, applied before the logit so saturated opacities stay finite.
_OPACITY_EPS = 1e-7


def write_ply(
    path: Path, *, means: Tensor, scales: Tensor, quats: Tensor, opacities: Tensor, sh: Tensor
) -> None:
    """Write per-time arrays as a static 3DGS-viewer PLY (Q-18): binary little-endian float32
    properties x y z, nx ny nz (zero), f_dc_0..2, f_rest_* (channel-major), opacity as a logit,
    scale_0..2 as logs, rot_0..3. Inputs are physical values: scales > 0, opacities in [0, 1]
    (clamped to [1e-7, 1 − 1e-7] before the logit), quaternions (w, x, y, z)."""
    where = f"ply {path}"
    n = int(means.shape[0]) if means.dim() == 2 else -1
    k = int(sh.shape[1]) if sh.dim() == 3 else -1
    expected = {
        "means": (means, (n, 3)),
        "scales": (scales, (n, 3)),
        "quats": (quats, (n, 4)),
        "opacities": (opacities, (n,)),
        "sh": (sh, (n, k, 3)),
    }
    if n < 0:
        _fail(where, f"means must have shape [N, 3], got {list(means.shape)}")
    if k < 1:
        _fail(where, f"sh must have shape [N, K, 3] with K ≥ 1, got {list(sh.shape)}")
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            _fail(where, f"{name} must have shape {list(shape)}, got {list(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            _fail(where, f"{name} has values that are not finite")
    if not bool((scales > 0).all()):
        _fail(where, "scales must be positive")
    f64 = torch.float64
    opacity = opacities.detach().to("cpu", f64).clamp(_OPACITY_EPS, 1.0 - _OPACITY_EPS)
    sh_cpu = sh.detach().to("cpu", f64)
    columns = [
        means.detach().to("cpu", f64),
        torch.zeros(n, 3, dtype=f64),
        sh_cpu[:, 0, :],
        sh_cpu[:, 1:, :].transpose(1, 2).reshape(n, 3 * (k - 1)),  # channel-major
        torch.logit(opacity)[:, None],
        torch.log(scales.detach().to("cpu", f64)),
        quats.detach().to("cpu", f64),
    ]
    body = torch.cat(columns, dim=1).to(torch.float32).numpy().astype("<f4", copy=False)
    names = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        *(f"f_rest_{i}" for i in range(3 * (k - 1))),
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]  # fmt: skip
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        *(f"property float {name}" for name in names),
        "end_header",
    ]
    with open(path, "wb") as out:
        out.write(("\n".join(header) + "\n").encode("ascii"))
        out.write(np.ascontiguousarray(body).tobytes())

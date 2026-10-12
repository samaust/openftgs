# SPDX-License-Identifier: Apache-2.0
"""Reading and writing full checkpoints (TrainerState), FTGS-IF exchange files (ExchangeModel),
ftgs-init files (InitPoints) and PLY files for viewers, with the validators of the exchange and
init formats. Plain data in and out: this module imports neither openftgs.model nor
openftgs.train.

Spec: §2.2, D-30, D-73. Band: Foundation (§2.1).
"""

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs.config import config_from_dict
from openftgs.types import ExchangeModel, InitPoints, TimeNormalizer, TrainerState

__all__ = [
    "load_checkpoint",
    "read_exchange",
    "read_init_points",
    "save_checkpoint",
    "validate_exchange",
    "validate_init_points",
    "write_exchange",
    "write_init_points",
    "write_ply",
]

_VERSION = 1  # the major version this module reads and writes (§2.2)
_VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?")


# Shared helpers: safetensors files with the §2.2 metadata header.


def _fail(where: str, reason: str) -> NoReturn:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(f"{where}: {reason}")


def _check_header(meta: Mapping[str, str], fmt: str, where: str) -> None:
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


def write_init_points(points: InitPoints, path: Path, *, time_unit: str, producer: str) -> None:
    """Write a ftgs-init v1 file (§2.2; Q-22), validating first."""
    validate_init_points(points)
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
    }
    _write_safetensors(tensors, Path(path), meta)


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
    sh_degree in {0, 1, 2, 3}, finite τ₀ ≤ τ₁. Raises ValueError (Q-26)."""
    where = "ftgs-if"
    degree = model.sh_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or degree not in _SH_DEGREES:
        _fail(where, f"sh_degree must be one of {_SH_DEGREES}, got {degree!r}")
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
        "render_conventions": json.dumps(_RENDER_CONVENTIONS, separators=(",", ":")),
    }
    _write_safetensors({k: model.arrays[k] for k in _IF_KEYS}, Path(path), meta)


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

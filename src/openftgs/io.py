# SPDX-License-Identifier: Apache-2.0
"""Reading and writing full checkpoints (TrainerState), FTGS-IF exchange files (ExchangeModel) and
PLY files for viewers. Plain data in and out: this module imports neither openftgs.model nor
openftgs.train.

Spec: §2.2, D-30. Band: Foundation (§2.1).
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from openftgs.types import ExchangeModel, InitPoints

__all__ = [
    "read_exchange",
    "read_init_points",
    "validate_exchange",
    "validate_init_points",
    "write_exchange",
    "write_init_points",
]

_VERSION = 1  # the major version this module reads and writes (§2.2)


# Shared helpers: safetensors files with the §2.2 metadata header.


def _fail(where: str, reason: str) -> None:
    # Q-26: every rejection is a plain ValueError.
    raise ValueError(f"{where}: {reason}")


def _check_header(meta: Mapping[str, str], fmt: str, where: str) -> None:
    """§2.2: `format` must match and the major part of `version` must be 1."""
    if meta.get("format") != fmt:
        _fail(where, f"format must be {fmt!r}, got {meta.get('format')!r}")
    version = meta.get("version")
    if version is None:
        _fail(where, "missing version")
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        major = -1
    if major != _VERSION:
        _fail(where, f"unsupported version {version!r}; this reader handles major version 1")


def _read_safetensors(path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118
    return tensors, meta


def _write_safetensors(tensors: Mapping[str, Tensor], path: Path, meta: Mapping[str, str]) -> None:
    save_file(
        {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()},
        str(path),
        metadata=dict(meta),
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
    assert colors is not None
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

# Q-24: the render conventions written with every exchange file, from E6's normative constants
# (gsplat 1.5.3 classic mode, D-6, D-31) and the black background of D-34.
_RENDER_CONVENTIONS = {
    "alpha_max": 0.999,
    "eps2d": 0.3,
    "near": 0.01,
    "far": 1e10,
    "alpha_skip": 1.0 / 255.0,
    "transmittance_stop": 1e-4,
    "pixel_center_offset": 0.5,
    "background": [0.0, 0.0, 0.0],
}


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
    assert position is not None
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
    assert duration is not None and scale is not None and rotation is not None
    assert opacity is not None
    if not bool((duration > 0.0).all()):
        _fail(where, "duration must be positive")
    if not bool((scale > 0.0).all()):
        _fail(where, "scale must be positive")
    if not bool((torch.linalg.vector_norm(rotation, dim=-1) > _MIN_ROTATION_NORM).all()):
        _fail(where, f"rotation norms must exceed {_MIN_ROTATION_NORM}")
    if not bool(((opacity > 0.0) & (opacity < 1.0)).all()):
        _fail(where, "opacity must lie within (0, 1)")


_IF_KEYS = ("position", "time_center", "duration", "velocity", "scale", "rotation", "opacity", "sh")


def write_exchange(model: ExchangeModel, path: Path) -> None:
    """Write a ftgs-if v1 file (§2.2), validating first (Q-22)."""
    validate_exchange(model)
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

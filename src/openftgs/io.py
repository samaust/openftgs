# SPDX-License-Identifier: Apache-2.0
"""Readers and writers of the stable file formats of §2.2, and PLY files for viewers.

Full checkpoints (:class:`~openftgs.types.TrainerState`), FTGS-IF exchange files
(:class:`~openftgs.types.ExchangeModel`), ftgs-init files (:class:`~openftgs.types.InitPoints`),
render requests (:class:`~openftgs.types.RenderRequest`) and their renders
(:class:`~openftgs.types.Renders`), scene manifests (:class:`~openftgs.types.SceneManifest`), and
PLY files for viewers, with the validators of the exchange and init formats. Plain data in and
out: this module imports neither openftgs.model nor openftgs.train.

Tensors live in safetensors files and all other state in JSON, so no file is ever unpickled.
Every file carries ``format`` and ``version``; readers reject a major version other than 1 and
ignore unknown optional keys (§2.2). Every rejection of a file's contents is a plain
:class:`ValueError` (Q-26).

Spec: §2.2, D-30, D-34, D-73, D-83; Q-64, Q-152. Band: Foundation (§2.1).
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

from openftgs._legacy import with_current_loss
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
    "check_run_checkpoints",
    "load_checkpoint",
    "read_exchange",
    "read_init_points",
    "read_render_request",
    "read_renders",
    "read_scene_manifest",
    "render_conventions",
    "save_checkpoint",
    "save_run_checkpoint",
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
    """Check the header: `format` must match and the major part of `version` must be 1 (§2.2)."""
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
    try:
        save_file(
            {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()},
            str(path),
            # safetensors 0.8.0 writes an unreadable header for no tensors with an empty metadata
            # map, so an empty map is passed as None (source log #11).
            metadata=dict(meta) or None,
        )
    except SafetensorError as err:
        # Q-158: the library reports a write that fails (a full disk, no permission) through
        # its own class, not an OSError (source log #148); the data was checked before.
        raise OSError(f"{path}: {err}") from err


def _replace_with_safetensors(
    tensors: Mapping[str, Tensor], path: Path, meta: Mapping[str, str]
) -> None:
    """Write a single-file output through a temporary sibling renamed over `path` (Q-126).

    Q-126 (as answered, #179): the temporary sibling is named as Q-25's `.<name>.tmp-<pid>`, so
    a write that fails partway leaves the old file, or no file, and no temporary. A symlink at
    `path` becomes a regular file (Q-146).
    """
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
    """Return the tensor `name`, checked for exactly `dtype` and `shape` and finite values.

    None in `shape` matches any size.
    """
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
    """Check init points against the rules of §2.2's ``ftgs-init`` format (Q-22).

    The rules: the table's dtypes and shapes, N ≥ 1, all values finite, colors within [0, 1]
    and durations positive. The reader, the writer and the model share this check.

    Parameters
    ----------
    points
        The points to check.

    Raises
    ------
    ValueError
        If the points break a rule, which the message names (Q-26).
    """
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
    """Write init points as a ``ftgs-init`` v1 file (§2.2; Q-22).

    The points are validated first (:func:`validate_init_points`). The file is written to a
    temporary sibling and renamed over ``path``, so a write that fails leaves the old file, or
    none (Q-126).

    Parameters
    ----------
    points
        The points to write.
    path
        The file to write; replaced if it exists.
    time_unit
        The unit of the points' raw times, recorded as the ``time_unit`` metadata.
    producer
        The software that made the points, recorded as the ``producer`` metadata.
    metadata
        More string keys and values to record beside §2.2's own, such as ``roma_init``'s
        provenance (Q-107), or None. The keys ``format``, ``version``, ``time_unit`` and
        ``producer`` are reserved.

    Raises
    ------
    ValueError
        If the points are invalid, or ``metadata`` holds a reserved or non-string key or value.
    OSError
        If the file can't be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> import torch
    >>> from openftgs.io import write_init_points
    >>> from openftgs.types import InitPoints
    >>> points = InitPoints(positions=torch.rand(1000, 3), colors=torch.rand(1000, 3),
    ...                     timestamps=torch.zeros(1000, dtype=torch.float64))
    >>> write_init_points(points, Path("init.safetensors"), time_unit="s", producer="my-tool")
    """
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
    """Read and validate a ``ftgs-init`` file (§2.2; Q-22).

    The optional tensors and the ``producer`` metadata may be absent; ``time_unit`` must be
    there.

    Parameters
    ----------
    path
        The file to read.
    time_unit
        The time unit the file must record, or None to accept any.

    Returns
    -------
    openftgs.types.InitPoints
        The points, on the CPU.

    Raises
    ------
    ValueError
        If the file isn't a readable safetensors file or a ``ftgs-init`` file of major version
        1, lacks ``time_unit`` or a required tensor, records another unit than ``time_unit``,
        or holds invalid points (:func:`validate_init_points`).

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import read_init_points
    >>> points = read_init_points(Path("init.safetensors"), time_unit="s")
    >>> points.positions.shape
    torch.Size([300000, 3])
    """
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


def render_conventions(*, background: Sequence[float] = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    """Return a fresh copy of the ``render_conventions`` block of ``ftgs-if`` (§2.2, D-73).

    The block describes the producer's renderer to importers and to the harness: E6's
    constants, the pixel-center offset, the background the model was trained against, the
    Jacobian clamp margin, the screen-extent rule and its cap, and the rasterize mode. Readers
    use only its background (D-84).

    Parameters
    ----------
    background
        The RGB background the model was trained against, three numbers in [0, 1] (D-84);
        black by default (D-34).

    Returns
    -------
    dict
        The block, with the keys ``alpha_max``, ``eps2d``, ``near``, ``far``, ``alpha_skip``,
        ``transmittance_stop``, ``pixel_center_offset``, ``background`` (a list of three
        floats), ``jacobian_clamp_margin``, ``screen_extent_rule``, ``screen_extent_max`` and
        ``rasterize_mode``.

    Raises
    ------
    ValueError
        If ``background`` isn't three numbers in [0, 1].

    Notes
    -----
    The keys and values follow Q-24, Q-35 and Q-204.
    """
    if (
        isinstance(background, str)
        or not isinstance(background, Sequence)
        or len(background) != 3
        or not all(
            isinstance(c, int | float) and not isinstance(c, bool) and 0.0 <= c <= 1.0
            for c in background
        )
    ):  # NaN fails the comparison
        _fail(
            "render_conventions", f"background must be three numbers in [0, 1], got {background!r}"
        )
    return {**_RENDER_CONVENTIONS, "background": [float(c) for c in background]}


# Q-22, Q-24: the ftgs-if validator, shared by the reader, the writer and the model.
def validate_exchange(model: ExchangeModel) -> None:
    """Check an exchange model against the rules of §2.2's ``ftgs-if`` format (Q-22, Q-24).

    The rules: the table's keys, dtypes and shapes, N ≥ 1, all values finite, opacity in
    (0, 1), durations and scales positive, rotations of norm > 1e-8, ``sh_degree`` in
    {0, 1, 2, 3}, finite τ₀ ≤ τ₁, a background of three numbers in [0, 1] (D-84), and, with a
    single timestamp (τ₀ = τ₁), every time center τ₀ and every velocity 0 (D-71). Rotations
    need not be unit quaternions here; only :func:`write_exchange` asks for that. The reader,
    the writer and the model share this check.

    Parameters
    ----------
    model
        The model to check.

    Raises
    ------
    ValueError
        If the model breaks a rule, which the message names (Q-26).
    """
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
    """Check that rotations are canonical unit quaternions (§2.2 ftgs-if; Q-24).

    §2.2 ftgs-if: unit quaternions with w ≥ 0; Q-24: where w = 0, the first nonzero component
    is positive. Checked on write only; readers accept any rotation of norm > 1e-8.
    """
    norm = torch.linalg.vector_norm(rotation.to(torch.float64), dim=-1)
    if not bool(((norm - 1.0).abs() <= _UNIT_TOLERANCE).all()):
        _fail(where, "rotation must hold unit quaternions")
    first = torch.argmax((rotation != 0).to(torch.int8), dim=-1, keepdim=True)
    if bool((torch.gather(rotation, -1, first) < 0).any()):
        _fail(where, "rotation must have w >= 0 (where w = 0, the first nonzero component > 0)")


_IF_KEYS = ("position", "time_center", "duration", "velocity", "scale", "rotation", "opacity", "sh")


def write_exchange(model: ExchangeModel, path: Path) -> None:
    """Write an exchange model as a ``ftgs-if`` v1 file (§2.2).

    The model is validated first (:func:`validate_exchange`; Q-22), and writing is stricter
    than reading: rotations must be unit quaternions, within 1e-6, with the canonical sign:
    w ≥ 0 and, where w = 0, the first nonzero component positive (Q-24). The metadata records the
    SH degree, the time unit, τ₀, τ₁, the producer and the :func:`render_conventions` block
    with the model's background (D-84). The file is written to a temporary sibling and renamed
    over ``path`` (Q-126).

    Parameters
    ----------
    model
        The model to write.
    path
        The file to write; replaced if it exists.

    Raises
    ------
    ValueError
        If the model is invalid, or its rotations aren't canonical unit quaternions.
    OSError
        If the file can't be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import write_exchange
    >>> from openftgs.model import FreeTimeGaussians
    >>> model = FreeTimeGaussians.load(Path("run/checkpoint"), device="cpu")
    >>> write_exchange(model.to_exchange(), Path("model.safetensors"))
    """
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
            render_conventions(background=model.background),
            separators=(",", ":"),
        ),
    }
    _replace_with_safetensors({k: model.arrays[k] for k in _IF_KEYS}, Path(path), meta)


def _recorded_background(meta: Mapping[str, str], where: str) -> tuple[float, float, float]:
    """Return the background render_conventions records, black when absent (D-84).

    D-84: render_conventions' background, black when the block or the entry is absent; its
    range is validate_exchange's to check.
    """
    if "render_conventions" not in meta:
        return (0.0, 0.0, 0.0)
    try:
        conventions = json.loads(meta["render_conventions"])
    except (ValueError, RecursionError):
        conventions = None
    if not isinstance(conventions, dict):
        _fail(where, "render_conventions must be a JSON object")  # Q-157
    if "background" not in conventions:
        return (0.0, 0.0, 0.0)
    value = conventions["background"]
    numbers = isinstance(value, list) and len(value) == 3
    if not numbers or any(isinstance(c, bool) or not isinstance(c, int | float) for c in value):
        _fail(where, f"background must be three numbers in [0, 1], got {value!r}")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except OverflowError:  # an integer too large for a float (Q-26: a plain ValueError)
        _fail(where, f"background must be three numbers in [0, 1], got {value!r}")


def read_exchange(path: Path) -> ExchangeModel:
    """Read and validate a ``ftgs-if`` file (§2.2; Q-22).

    Unknown keys are ignored, and so are the entries of ``render_conventions`` other than
    ``background`` (Q-24), which is returned as the model's background: black when the block
    or the entry is absent (D-84). A missing ``producer`` reads as an empty string.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    openftgs.types.ExchangeModel
        The model, its tensors on the CPU.

    Raises
    ------
    ValueError
        If the file isn't a readable safetensors file or a ``ftgs-if`` file of major version
        1; if it lacks ``sh_degree``, ``tau0``, ``tau1`` or ``time_unit``, or the first three
        aren't numbers; if ``render_conventions`` isn't a JSON object or its background isn't three
        numbers in [0, 1] (D-84, Q-157); or if the model is invalid
        (:func:`validate_exchange`).

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import read_exchange
    >>> exchange = read_exchange(Path("model.safetensors"))
    >>> exchange.sh_degree, exchange.time_unit
    (3, 's')
    """
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
        background=_recorded_background(meta, where),
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
    """Split a tree of dicts into its tensors and a JSON-ready rest (Q-25).

    Q-25: every tensor in a tree of dicts goes to a flat map under its `/`-joined key path;
    the rest stays in a JSON-ready tree. Tensors inside lists are refused.
    """
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
    """Check for a JSON value with finite numbers and string keys; tensors are refused here."""
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
    """Put each tensor back at its key path, the inverse of _split_tensors."""
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


def _checkpoint_manifest(directory: Path) -> dict[str, Any] | None:
    """Return the manifest of a checkpoint directory, or None for anything else."""
    try:
        manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) and manifest.get("format") == _CKPT else None


def _is_checkpoint(directory: Path) -> bool:
    return _checkpoint_manifest(directory) is not None


def _recover_set_aside(directory: Path) -> None:
    """Recover from a save killed between its two renames, then sweep leftovers (Q-25).

    The newest set-aside checkpoint is the one a killed save left last.
    """
    aside = sorted(directory.parent.glob(f".{directory.name}.old-*"), key=_mtime)
    if not directory.exists() and aside and _is_checkpoint(aside[-1]):
        aside.pop().rename(directory)
    for leftover in [*aside, *directory.parent.glob(f".{directory.name}.tmp-*")]:
        shutil.rmtree(leftover, ignore_errors=True)


def save_checkpoint(state: TrainerState, directory: Path) -> Path:
    """Write a trainer state as a ``ftgs-checkpoint`` v1 directory (§2.2; Q-25).

    The files are written into a temporary sibling directory that then replaces ``directory``,
    so a reader sees either the old checkpoint or the new one. An existing directory is replaced
    only if it is a checkpoint or empty. POSIX can't swap two directories in one rename: a kill
    between the two renames leaves the old checkpoint set aside beside the target, and the next
    save of the same target restores it before replacing it (Q-25). Concurrent saves to one
    target are not supported.

    Parameters
    ----------
    state
        The state to save, as ``Trainer.state_dict`` returns it: ``params`` holding exactly the
        nine stored tensors of §1.1, float32; ``stats`` holding only [N] tensors; tensors in
        ``optim`` and ``rng`` at any depth of nested dicts, under keys without ``/``; JSON
        values with finite numbers elsewhere.
    directory
        The checkpoint directory to write.

    Returns
    -------
    pathlib.Path
        ``directory``.

    Raises
    ------
    ValueError
        If ``state`` breaks one of these rules, or ``directory`` exists and is neither a
        checkpoint nor an empty directory.
    OSError
        If a file can't be written.

    Notes
    -----
    The manifest records the iteration, the configuration, τ₀, τ₁, S, Δt_f, the SH degree and
    seed, the scene manifest's SHA-256 and time unit (Q-17), the versions of openftgs, torch,
    gsplat and CUDA, and the SHA-256 of every other file.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import load_checkpoint, save_checkpoint
    >>> state = load_checkpoint(Path("run/checkpoint"))
    >>> save_checkpoint(state, Path("copy/checkpoint"))
    PosixPath('copy/checkpoint')
    """
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
    _recover_set_aside(directory)  # Q-25
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


_LATEST = "checkpoint"  # D-83: the latest save of a run directory
# Q-150: an earlier save, named after its iteration in decimal as the rotation writes it.
_EARLIER_NAME = re.compile(r"checkpoint-(0|[1-9][0-9]*)")


def _iteration_of(directory: Path) -> int | None:
    """Return the iteration a checkpoint's manifest records, or None for anything else (Q-150)."""
    manifest = _checkpoint_manifest(directory)
    iteration = None if manifest is None else manifest.get("iteration")
    return iteration if isinstance(iteration, int) and not isinstance(iteration, bool) else None


def _earlier_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """List every RUN_DIR/checkpoint-<n> that is a checkpoint, with n, ordered by n (Q-150)."""
    found: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        match = _EARLIER_NAME.fullmatch(path.name)
        if match is not None and _counts_as_kept(path):
            found.append((int(match.group(1)), path))
    return sorted(found)


def _counts_as_kept(path: Path) -> bool:
    """Tell whether a checkpoint-<n> entry is a kept checkpoint: a checkpoint directory (Q-150)."""
    return path.is_dir() and _is_checkpoint(path)


def _check_keep(keep: int) -> None:
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError(f"keep must be an integer of at least 1, got {keep!r}")


# D-83, Q-152: the saves of ftgs train, rotated through the run directory.
def save_run_checkpoint(state: TrainerState, run_dir: Path, *, keep: int = 1) -> Path:
    """Save a trainer state into a run directory, keeping its latest saves (D-83; Q-152).

    The state is saved as ``RUN_DIR/checkpoint`` with :func:`save_checkpoint`, keeping the
    ``keep`` latest saves of the run directory, one per iteration: the previous latest is
    renamed ``RUN_DIR/checkpoint-<its iteration>`` when ``keep`` > 1, and the oldest beyond
    ``keep`` are removed once the new save is in place (Q-151). Kept checkpoints of the saved
    iteration or later, the continuation a resume from an earlier one abandoned, are not kept:
    ``RUN_DIR/checkpoint`` is replaced in place, the others removed first (Q-149). Only
    checkpoints are counted or removed (Q-150, Q-25). A save that fails puts the previous
    latest back.

    Parameters
    ----------
    state
        The state to save, as for :func:`save_checkpoint`.
    run_dir
        The run directory.
    keep
        How many saves to keep, the latest included, ≥ 1.

    Returns
    -------
    pathlib.Path
        ``RUN_DIR/checkpoint``.

    Raises
    ------
    ValueError
        If ``keep`` isn't an integer ≥ 1, the rename would land on a ``checkpoint-<n>`` that
        isn't a checkpoint (Q-150, Q-25), or :func:`save_checkpoint` refuses the state.
    OSError
        If a file can't be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import load_checkpoint, save_run_checkpoint
    >>> state = load_checkpoint(Path("old-run/checkpoint"))
    >>> save_run_checkpoint(state, Path("run"), keep=3)
    PosixPath('run/checkpoint')
    """
    _check_keep(keep)
    run_dir = Path(run_dir)
    latest = run_dir / _LATEST
    _recover_set_aside(latest)  # Q-25: a killed save's checkpoint is the latest again
    abandoned = [p for it, p in _earlier_checkpoints(run_dir) if it >= state.iteration]  # Q-149
    current = _iteration_of(latest)
    renamed: Path | None = None
    if current is not None and current < state.iteration and keep > 1:
        renamed = run_dir / f"checkpoint-{current}"
        if renamed.exists() or renamed.is_symlink():
            if not _counts_as_kept(renamed):  # Q-150
                raise ValueError(f"{renamed} exists and is not a checkpoint; refusing to rotate")
            abandoned.append(renamed)  # Q-149: one save per iteration, the newer one kept
    for path in abandoned:
        shutil.rmtree(path)
    if renamed is not None:
        latest.rename(renamed)
    try:
        save_checkpoint(state, latest)  # atomic over a later or equal latest (Q-25, Q-149)
    except BaseException:
        if renamed is not None and not latest.exists():
            renamed.rename(latest)
        raise
    earlier = _earlier_checkpoints(run_dir)
    for _, path in earlier[: max(len(earlier) - (keep - 1), 0)]:  # Q-151: after the new save
        shutil.rmtree(path)
    return latest


# Q-163: ftgs train's check before training (Q-150 as answered).
def check_run_checkpoints(run_dir: Path, *, keep: int = 1) -> None:
    """Check that the saves of a run directory can rotate, before training (Q-163).

    With ``keep`` > 1, a save of :func:`save_run_checkpoint` with that ``keep`` would refuse to
    rename onto a ``RUN_DIR/checkpoint-<n>`` that isn't a checkpoint (Q-150), so ``ftgs train``
    checks for one first. Only each manifest's format is read, and nothing changes.

    Parameters
    ----------
    run_dir
        The run directory.
    keep
        How many saves the run will keep, ≥ 1; with 1 there is nothing to check.

    Raises
    ------
    ValueError
        If ``keep`` isn't an integer ≥ 1, or, with ``keep`` > 1, naming the first
        ``RUN_DIR/checkpoint-<n>``, by n, that isn't a checkpoint.
    """
    _check_keep(keep)
    if keep == 1:
        return
    blocking = []
    for path in Path(run_dir).glob("checkpoint-*"):
        match = _EARLIER_NAME.fullmatch(path.name)
        if match is not None and not _counts_as_kept(path):
            blocking.append((int(match.group(1)), path))
    if blocking:
        _, first = min(blocking)
        raise ValueError(f"{first} exists and is not a checkpoint; a save would refuse to rotate")


def load_checkpoint(directory: Path) -> TrainerState:
    """Read a ``ftgs-checkpoint`` directory (§2.2; Q-25).

    The manifest's format and major version, the file set and every file's SHA-256 are checked
    before anything else is read; files the manifest doesn't list are ignored. The stored
    configuration must be valid (Q-10, Q-11), and N must not exceed its ``n_max`` (§1.1).

    A checkpoint saved before revision 549o, whose configuration names the old perceptual
    backbone "vgg" or "alex", is read too: its configuration is validated with 549o's loss
    defaults in place of its ``perc_net`` and ``lambda_perc``, in memory only, and the state
    keeps the configuration as saved (Q-247). Only resuming its training refuses it
    (:meth:`openftgs.train.Trainer.from_checkpoint`).

    Parameters
    ----------
    directory
        The checkpoint directory.

    Returns
    -------
    openftgs.types.TrainerState
        The state, its tensors on the CPU, its ``config`` the dict as saved.

    Raises
    ------
    ValueError
        If the manifest can't be read or isn't a ``ftgs-checkpoint`` manifest of major version
        1, a file is missing or doesn't match its SHA-256, a file's contents are malformed, the
        configuration is invalid, or N exceeds N_max.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import load_checkpoint
    >>> state = load_checkpoint(Path("run/checkpoint"))
    >>> state.iteration
    30000
    """
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
    # within its N_max, for every reader of a checkpoint. Q-247: a checkpoint saved before 549o
    # is validated with 549o's loss defaults in place of LPIPS's backbone, in memory only.
    try:
        n_max = config_from_dict(with_current_loss(state.config)).growth.n_max
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
    """Check the header of a JSON format: `format`, and a `version` of major 1 (§2.2; Q-56).

    The version is written as a JSON int or a "major.minor" string (Q-56).
    """
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
    """Read and validate a ``ftgs-render-request`` file (§2.2; Q-56).

    ``background`` defaults to black (D-34) and ``sh_degree`` to null, the model's full
    degree; unknown keys are ignored. Each item's camera is checked: positive integer width
    and height, K with fx, fy > 0 and no skew, and a world-to-camera matrix whose last row is
    (0, 0, 0, 1) and whose rotation is orthonormal, within 1e-5, with determinant +1.

    Parameters
    ----------
    path
        The JSON file to read.

    Returns
    -------
    openftgs.types.RenderRequest
        The request, its items in file order, each camera named after its item's id.

    Raises
    ------
    ValueError
        If the file isn't readable JSON or a ``ftgs-render-request`` file of major version 1;
        if ``time_unit`` isn't a non-empty string, the background isn't three numbers in
        [0, 1] or ``sh_degree`` isn't null or 0 to 3; or if ``items`` is empty, an item id is
        empty or repeated, a timestamp isn't a finite number or a camera breaks a rule.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import read_render_request
    >>> request = read_render_request(Path("request.json"))
    >>> [item.id for item in request.items]
    ['r0000', 'r0001']
    """
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
    """Write renders as a ``ftgs-renders`` v1 file (§2.2; Q-56).

    Each item's colors are stored as ``rgb/<id>``, [H, W, 3] float32, not clamped to 1, and
    its opacity as ``alpha/<id>``, [H, W, 1] float32; the metadata records the producer and
    the request's SHA-256. The file is written to a temporary sibling and renamed over
    ``path`` (Q-126).

    Parameters
    ----------
    renders
        The renders: at least one non-empty id, each in both maps, with the dtypes and shapes
        above and finite values.
    path
        The file to write; replaced if it exists.
    producer
        The software that made the renders.
    request_sha256
        The SHA-256 of the request file's bytes, as 64 lowercase hex digits.

    Raises
    ------
    ValueError
        If ``request_sha256`` isn't such a digest, or the renders break a rule.
    OSError
        If the file can't be written.

    Examples
    --------
    >>> import hashlib
    >>> from pathlib import Path
    >>> from openftgs.io import read_render_request, write_renders
    >>> from openftgs.model import FreeTimeGaussians
    >>> from openftgs.render import render_request
    >>> model = FreeTimeGaussians.load(Path("run/checkpoint"))
    >>> renders = render_request(model, read_render_request(Path("request.json")))
    >>> digest = hashlib.sha256(Path("request.json").read_bytes()).hexdigest()
    >>> write_renders(renders, Path("renders.safetensors"), producer="my-tool",
    ...               request_sha256=digest)
    """
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
    """Read and validate a ``ftgs-renders`` file (§2.2; Q-56).

    Tensors other than ``rgb/…`` and ``alpha/…`` are ignored. The file keeps no item order, so
    the ids come back sorted: match them to the request by id.

    Parameters
    ----------
    path
        The file to read.
    request_sha256
        The SHA-256 of the request the renders must answer, as 64 lowercase hex digits, or
        None to accept any.

    Returns
    -------
    openftgs.types.Renders
        The renders, sorted by id, on the CPU.

    Raises
    ------
    ValueError
        If the file isn't a readable safetensors file or a ``ftgs-renders`` file of major
        version 1, was made for another request than ``request_sha256``, or its renders are
        empty, unpaired, of the wrong dtype or shape, or not finite.

    Examples
    --------
    >>> import hashlib
    >>> from pathlib import Path
    >>> from openftgs.io import read_renders
    >>> digest = hashlib.sha256(Path("request.json").read_bytes()).hexdigest()
    >>> renders = read_renders(Path("renders.safetensors"), request_sha256=digest)
    >>> renders.rgb["r0000"].shape
    torch.Size([480, 640, 3])
    """
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
    """Read and validate a ``ftgs-scene`` file (§2.2; Q-64).

    The result holds the image records in file order, with paths resolved against the
    manifest's directory, the test images' masks, the metadata, the time unit, the SHA-256 of
    the file's bytes as read and the camera names in the key order of the ``cameras`` object
    (Q-108). An image that overrides ``K`` or ``world_to_camera`` gets a camera of its own.
    Cameras are checked as a render request's (Q-56), and must be pinhole cameras; unknown keys
    are ignored. Whether the files exist is not checked here.

    Parameters
    ----------
    path
        The manifest file to read.

    Returns
    -------
    openftgs.types.SceneManifest
        The parsed manifest.

    Raises
    ------
    ValueError
        If the file isn't readable JSON or a ``ftgs-scene`` file of major version 1; if a
        camera, an image entry, a path, a split or the metadata breaks §2.2's rules (absolute
        paths and masks on training images are refused); or if there is no training image.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import read_scene_manifest
    >>> scene = read_scene_manifest(Path("scene/scene.json"))
    >>> scene.camera_names[:2], len(scene.records)
    (('cam00', 'cam01'), 6000)
    """
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
    """Convert each value to the shortest decimal that reads back to the same float32 (Q-84)."""
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
    """Return `path` relative to the manifest, in POSIX form (§2.2)."""
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
    """Write image records as a ``ftgs-scene`` v1 file (§2.2; Q-64, Q-84).

    One camera entry per camera name comes from that name's first record; a later record whose
    K or world-to-camera differs gets a per-image override. Camera values are written as the
    shortest decimals that read back to the same float32 values, so the reader rebuilds each
    camera bitwise. Image and mask paths must lie under the manifest's directory and are
    written relative to it. The file is written to a temporary sibling, validated by reading it
    back, then renamed over ``path``.

    Parameters
    ----------
    path
        The manifest file to write; replaced if it exists.
    records
        The image records, in the order to write them; at least one for training.
    time_unit
        The unit of the records' raw times.
    masks
        Each record's mask, aligned with ``records``: a path for a test image, else None; or
        None for no masks at all.
    metadata
        The manifest's ``metadata`` object, JSON values only, or None for an empty one.

    Returns
    -------
    openftgs.types.SceneManifest
        The manifest as :func:`read_scene_manifest` reads it back.

    Raises
    ------
    ValueError
        If ``masks`` doesn't align with ``records``, a camera name changes size, a path lies
        outside the manifest's directory, the metadata isn't JSON, or the written file fails
        :func:`read_scene_manifest`'s checks.
    OSError
        If the file can't be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import read_scene_manifest, write_scene_manifest
    >>> scene = read_scene_manifest(Path("scene/scene.json"))
    >>> train = [r for r in scene.records if r.split == "train"]
    >>> subset = write_scene_manifest(Path("scene/train.json"), train, time_unit=scene.time_unit)
    """
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
    """Write per-time arrays as a static PLY for 3DGS viewers (Q-18).

    A snapshot of the model at one time, export-only and never read back. The file is binary
    little-endian with the float32 properties ``x y z``, ``nx ny nz`` (zero),
    ``f_dc_0..2``, ``f_rest_*`` (channel-major), ``opacity`` as a logit, ``scale_0..2`` as
    logs and ``rot_0..3``.

    Parameters
    ----------
    path
        The PLY file to write; replaced if it exists.
    means
        Positions at the chosen time, [N, 3], in world units.
    scales
        Scales, [N, 3], in world units, > 0.
    quats
        Rotations as quaternions (w, x, y, z), [N, 4].
    opacities
        Opacities, [N], in [0, 1]; clamped to [1e-7, 1 − 1e-7] before the logit.
    sh
        SH coefficients, [N, K, 3], K ≥ 1, the first the DC term.

    Raises
    ------
    ValueError
        If a shape doesn't match, a value isn't finite, or a scale isn't positive.
    OSError
        If the file can't be written.

    Examples
    --------
    >>> from pathlib import Path
    >>> from openftgs.io import write_ply
    >>> from openftgs.model import FreeTimeGaussians
    >>> model = FreeTimeGaussians.load(Path("run/checkpoint"), device="cpu")
    >>> g = model.at_time(0.5)
    >>> write_ply(Path("t050.ply"), means=g.means_t, scales=g.scales, quats=g.quats_n,
    ...           opacities=g.opacities_t, sh=g.sh)
    """
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

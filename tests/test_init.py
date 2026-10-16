# SPDX-License-Identifier: Apache-2.0
"""Tests of openftgs.init: k-NN velocities (E15, D-23, D-56), two-view DLT triangulation (§1.3
step 3), `RomaInitConfig` and `roma_provenance`, and the `roma_init` pipeline (§1.3 steps 1–7)
on the CPU through an analytic stand-in dense matcher; U-19 and U-33 (§3.1), with the
contracts of Q-106 to Q-112, Q-120 and Q-121, Q-122."""

import dataclasses
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from builders import TexturedPlane, look_at, write_plane_scene
from PIL import Image
from safetensors import safe_open
from torch import Tensor

import openftgs
from openftgs.config import InitConfig
from openftgs.data import MultiViewVideo, convert_selfcap
from openftgs.init import (
    DenseMatcher,
    RomaInitConfig,
    _nearest_indices,
    _pil,
    _roma_matchers,
    knn_velocities,
    roma_init,
    roma_provenance,
    triangulate_dlt,
)
from openftgs.io import read_init_points, validate_init_points, write_init_points
from openftgs.model import FreeTimeGaussians
from openftgs.types import Camera, InitPoints


def _lattice(shape: tuple[int, int, int], spacing: float, dtype: torch.dtype) -> torch.Tensor:
    axes = [torch.arange(n, dtype=torch.float64) * spacing for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_u19_knn_velocities_recover_a_translation(dtype: torch.dtype, direction: float) -> None:
    """U-19: E15 on a translated point set, shifted by less than half the point spacing, gives
    v = shift/dt within 1e-6, in the dtype of `points`; a negative dt is the backward difference
    against the previous frame (D-23).

    The lattice and the shift are dyadic, so float32 holds them exactly."""
    spacing = 0.125
    shift = torch.tensor([3 / 64, -1 / 32, 1 / 64], dtype=torch.float64)  # ‖shift‖ < spacing/2
    dt0 = 1 / 30
    points = _lattice((6, 5, 4), spacing, torch.float64) + 2.0
    other = points + direction * shift
    perm = torch.randperm(other.shape[0], generator=torch.Generator().manual_seed(0))
    v = knn_velocities(points.to(dtype), other[perm].to(dtype), direction * dt0)
    assert v.dtype == dtype
    expected = (shift / dt0).expand_as(points).to(dtype)
    torch.testing.assert_close(v, expected, rtol=1e-6, atol=0.0)


def _cloud(n: int, seed: int) -> torch.Tensor:
    return torch.rand(n, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


@pytest.mark.parametrize("dt", [0.0, -0.0, float("inf"), float("-inf"), float("nan")])
def test_knn_velocities_refuses_a_zero_or_non_finite_dt(dt: float) -> None:
    """Q-106: a dt that is zero or not finite raises ValueError."""
    with pytest.raises(ValueError, match="dt"):
        knn_velocities(_cloud(5, 0), _cloud(4, 1), dt)


def test_knn_velocities_refuses_an_empty_next_set() -> None:
    """Q-106: M = 0 raises ValueError: no point has a neighbour."""
    with pytest.raises(ValueError, match="empty"):
        knn_velocities(_cloud(5, 0), torch.empty(0, 3, dtype=torch.float64), 0.1)


def test_knn_velocities_of_no_points_is_empty() -> None:
    """Q-120: N = 0 gives an empty [0, 3] in the dtype of `points`."""
    v = knn_velocities(torch.empty(0, 3, dtype=torch.float32), _cloud(4, 1), 0.1)
    assert v.shape == (0, 3)
    assert v.dtype == torch.float32


@pytest.mark.parametrize(
    ("points", "next_points"),
    [
        (_cloud(5, 0)[:, :2], _cloud(4, 1)),
        (_cloud(5, 0)[:, :, None], _cloud(4, 1)),
        (_cloud(5, 0), torch.zeros(4, 4, dtype=torch.float64)),
        (_cloud(5, 0)[0], _cloud(4, 1)),
    ],
    ids=["points-2d", "points-3-axes", "next-4-columns", "points-1-axis"],
)
def test_knn_velocities_refuses_malformed_shapes(
    points: torch.Tensor, next_points: torch.Tensor
) -> None:
    """Q-120: `points` must be [N, 3] and `next_points` [M, 3]."""
    with pytest.raises(ValueError, match=r"\[n, 3\]"):
        knn_velocities(points, next_points, 0.1)


def test_knn_velocities_refuses_integer_points() -> None:
    """Q-120: both point sets must be floating point."""
    ints = torch.zeros(5, 3, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating point"):
        knn_velocities(ints, _cloud(4, 1), 0.1)
    with pytest.raises(ValueError, match="floating point"):
        knn_velocities(_cloud(5, 0), ints, 0.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("which", ["points", "next_points"])
def test_knn_velocities_refuses_non_finite_coordinates(bad: float, which: str) -> None:
    """Q-120: a coordinate that isn't finite raises ValueError; such a point has no
    nearest neighbour."""
    points, next_points = _cloud(5, 0), _cloud(4, 1)
    (points if which == "points" else next_points)[2, 1] = bad
    with pytest.raises(ValueError, match=rf"^{which}\b.*finite"):
        knn_velocities(points, next_points, 0.1)


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def _brute_force_nearest(query: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """The test's own float64 oracle: squared distances from explicit differences, by blocks."""
    out = []
    for block in torch.split(query, 256):
        d2 = ((block[:, None, :] - data[None, :, :]) ** 2).sum(dim=-1)
        out.append(d2.argmin(dim=1))
    return torch.cat(out)


def _clusters() -> tuple[torch.Tensor, torch.Tensor]:
    """Three clusters 10³·S from the origin (S = 1). Each holds a 13³ lattice 10⁻⁴·S apart,
    every point jittered by up to ±0.3·10⁻⁴, and 3,000 query points drawn in the same box."""
    gen = torch.Generator().manual_seed(33)
    spacing, n = 1e-4, 13
    centers = torch.tensor(
        [[1e3, 0.0, 0.0], [0.0, -1e3, 0.0], [577.35, 577.35, -577.35]], dtype=torch.float64
    )
    axis = torch.arange(n, dtype=torch.float64) * spacing
    lattice = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    points, next_points = [], []
    for c in centers:
        jitter = (torch.rand(lattice.shape, generator=gen, dtype=torch.float64) - 0.5) * 0.6
        next_points.append(c + lattice + jitter * spacing)
        box = torch.rand(3000, 3, generator=gen, dtype=torch.float64) * (n - 1) * spacing
        points.append(c + box)
    return torch.cat(points), torch.cat(next_points)


@pytest.mark.parametrize("device", _DEVICES)
def test_u33_knn_velocities_match_a_float64_brute_force(device: str) -> None:
    """U-33: on clustered points 10⁻⁴·S apart and 10³·S from the origin, `knn_velocities`
    finds the neighbours of an in-test float64 brute force: its velocities are
    (next[nn(i)] − pᵢ)/dt for the brute force's nn. A different neighbour would be off by
    about 10⁻⁴/dt; the tolerance is far below that.

    9,000 points against 6,591 make the CUDA path run several tiles."""
    points, next_points = _clusters()
    dt = 0.5
    nn = _brute_force_nearest(points, next_points)
    expected = (next_points[nn] - points) / dt
    v = knn_velocities(points.to(device), next_points.to(device), dt)
    torch.testing.assert_close(v.cpu(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("device", _DEVICES)
def test_nearest_indices_ties_go_to_the_lower_index(device: str) -> None:
    """D-56: among points at the nearest distance, the lowest index wins, on both paths.

    40 copies of one point (first at index 3) make the CPU path query again several times;
    one query row has a unique nearest point, so only some rows are queried again; two distinct
    points, each exactly 1 from the origin, tie by distance alone."""
    f64 = torch.float64
    a = torch.tensor([0.25, -0.5, 1.0], dtype=f64)
    others = torch.tensor(
        [[5.0, 5.0, 5.0], [-1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=f64
    )
    data = torch.cat([others[:3], a.expand(40, 3), others[3:]])  # a at 3 … 42, (1, 0, 0) at 43
    query = torch.stack(
        [a, a + 1e-3, torch.tensor([8.9, 0.0, 0.0], dtype=f64), torch.zeros(3, dtype=f64)]
    )
    expected = torch.tensor([3, 3, 2, 1])
    got = _nearest_indices(query.to(device), data.to(device))
    assert got.device.type == device
    assert torch.equal(got.cpu(), expected)


@pytest.mark.gpu
def test_knn_velocities_refuses_two_devices() -> None:
    """Q-120: `points` and `next_points` must sit on one device."""
    with pytest.raises(ValueError, match="device|sit on"):
        knn_velocities(_cloud(5, 0), _cloud(4, 1).cuda(), 0.1)


_K = torch.tensor(
    [[1200.0, 0.0, 640.0], [0.0, 1180.0, 360.0], [0.0, 0.0, 1.0]], dtype=torch.float64
)


def _look_at(center: list[float]) -> torch.Tensor:
    """P = K[R_w | t_w] (§1.4) for a camera at `center` whose z axis points at the origin."""
    o = torch.tensor(center, dtype=torch.float64)
    z = -o / torch.linalg.vector_norm(o)
    x = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), z)
    x = x / torch.linalg.vector_norm(x)
    rot = torch.stack([x, torch.linalg.cross(z, x), z])  # rows x, y, z with x × y = z
    return _K @ torch.cat([rot, (-rot @ o)[:, None]], dim=1)


def _project(P: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Pixels in §1.4's continuous convention: u = (P·X̃)₀/(P·X̃)₂, v = (P·X̃)₁/(P·X̃)₂."""
    h = (P[..., :3] @ X[..., :, None])[..., 0] + P[..., 3]
    return h[..., :2] / h[..., 2:]


def _max_relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    err = torch.linalg.vector_norm(got - want, dim=-1) / torch.linalg.vector_norm(want, dim=-1)
    return float(err.max())


def _scene(n: int, seed: int) -> torch.Tensor:
    X = torch.rand(n, 3, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return 2 * X - 1 + torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)


@pytest.mark.parametrize("device", _DEVICES)
def test_u19_noise_free_dlt_recovers_the_points(device: str) -> None:
    """U-19: exact projections into two views give back the points within 1e-6 relative, as
    float64 [N, 3], with one shared P per view (§1.3 step 3)."""
    X = _scene(2000, 19)
    P1, P2 = _look_at([-1.0, 0.2, -5.0]), _look_at([1.5, -0.1, -4.5])
    x1, x2 = _project(P1, X), _project(P2, X)
    got = triangulate_dlt(x1.to(device), x2.to(device), P1.to(device), P2.to(device))
    assert got.dtype == torch.float64
    assert got.shape == (2000, 3)
    assert got.device.type == device
    assert _max_relative_error(got.cpu(), X) <= 1e-6


def test_u19_noise_free_dlt_broadcasts_the_cameras() -> None:
    """U-19 with Q-106's shapes: pixels [..., 2] and cameras [3, 4] or [..., 3, 4] broadcast
    together, here per batch entry [2, 1, 3, 4] against a shared [3, 4], and per row [N, 3, 4]."""
    X = _scene(100, 7).reshape(2, 50, 3)
    P1 = torch.stack([_look_at([-1.0, 0.2, -5.0]), _look_at([0.3, -2.0, -4.0])])[:, None]
    P2 = _look_at([1.5, -0.1, -4.5])
    got = triangulate_dlt(_project(P1, X), _project(P2, X), P1, P2)
    assert got.shape == (2, 50, 3)
    assert _max_relative_error(got, X) <= 1e-6

    rows = X[0, :6]
    centers = [[-1.0, 0.2, -5.0], [0.3, -2.0, -4.0], [4.0, 0.5, -3.0]]
    P1 = torch.stack([_look_at(centers[i % 3]) for i in range(6)])
    P2 = torch.stack([_look_at(centers[(i + 1) % 3]) for i in range(6)])
    got = triangulate_dlt(_project(P1, rows), _project(P2, rows), P1, P2)
    assert got.shape == (6, 3)
    assert _max_relative_error(got, rows) <= 1e-6


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_dlt_rows_with_a_non_finite_input_come_out_nan(device: str, bad: float) -> None:
    """Q-106, Q-120: no filtering; a row with a non-finite pixel comes out NaN, and the other
    rows of the batch are still solved within U-19's 1e-6."""
    X = _scene(64, 5)
    P1, P2 = _look_at([-1.0, 0.2, -5.0]), _look_at([1.5, -0.1, -4.5])
    x1, x2 = _project(P1, X), _project(P2, X)
    x1[3, 0] = bad
    x2[40, 1] = bad
    got = triangulate_dlt(x1.to(device), x2.to(device), P1.to(device), P2.to(device)).cpu()
    broken = torch.zeros(64, dtype=torch.bool)
    broken[[3, 40]] = True
    assert torch.isnan(got[broken]).all()
    assert _max_relative_error(got[~broken], X[~broken]) <= 1e-6


@pytest.mark.parametrize("device", _DEVICES)
def test_dlt_rays_meeting_at_infinity_come_out_non_finite(device: str) -> None:
    """Q-106, Q-120: two cameras with one orientation, 0.7 apart, seeing the same principal
    point have parallel rays; the solution is the point at infinity (h₄ = 0), which the
    division leaves non-finite instead of filtering it."""
    eye = torch.eye(3, dtype=torch.float64)
    P1 = _K @ torch.cat([eye, torch.zeros(3, 1, dtype=torch.float64)], dim=1)
    P2 = _K @ torch.cat([eye, torch.tensor([[-0.7], [0.0], [0.0]], dtype=torch.float64)], dim=1)
    x = torch.tensor([[640.0, 360.0]], dtype=torch.float64)
    got = triangulate_dlt(x.to(device), x.to(device), P1.to(device), P2.to(device))
    assert not torch.isfinite(got).any()


def _views() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = _scene(5, 1)
    P1, P2 = _look_at([-1.0, 0.2, -5.0]), _look_at([1.5, -0.1, -4.5])
    return _project(P1, X), _project(P2, X), P1, P2


@pytest.mark.parametrize(
    ("position", "replacement", "message"),
    [
        (0, torch.zeros(5, 3, dtype=torch.float64), r"x1 must be \[\.\.\., 2\]"),
        (1, torch.tensor(1.0, dtype=torch.float64), r"x2 must be \[\.\.\., 2\]"),
        (2, torch.zeros(3, 3, dtype=torch.float64), r"P1 must be \[\.\.\., 3, 4\]"),
        (3, torch.zeros(12, dtype=torch.float64), r"P2 must be \[\.\.\., 3, 4\]"),
        (1, torch.zeros(4, 2, dtype=torch.float64), "broadcast"),
        (3, torch.zeros(2, 3, 4, dtype=torch.float64), "broadcast"),
        (0, torch.zeros(5, 2, dtype=torch.int64), "x1 must be floating point"),
        (2, torch.zeros(3, 4, dtype=torch.int32), "P1 must be floating point"),
    ],
    ids=["x1-3", "x2-scalar", "P1-3x3", "P2-flat", "rows", "cameras", "x1-int", "P1-int"],
)
def test_triangulate_dlt_refuses_malformed_inputs(
    position: int, replacement: torch.Tensor, message: str
) -> None:
    """Q-120: pixels must be floating-point [..., 2] and cameras floating-point [..., 3, 4],
    with batch shapes that broadcast."""
    args = list(_views())
    args[position] = replacement
    with pytest.raises(ValueError, match=message):
        triangulate_dlt(*args)


@pytest.mark.gpu
def test_triangulate_dlt_refuses_two_devices() -> None:
    """Q-120: the four inputs must sit on one device."""
    x1, x2, P1, P2 = _views()
    with pytest.raises(ValueError, match="device"):
        triangulate_dlt(x1, x2, P1.cuda(), P2)


# RomaInitConfig (§2.1; Q-106)


def test_roma_init_config_defaults_are_the_spec_values() -> None:
    """§2.1 and D-21, D-22, D-24, D-58: roma_outdoor at 560 px upsampled to 864 px, 5,000
    matches per pair at certainty ≥ 0.5, stride 1, B = 1,000 and seed 0; the edges of each
    range are accepted."""
    cfg = RomaInitConfig()
    assert (cfg.model, cfg.coarse_res, cfg.upsample_res, cfg.upsample_preds) == (
        "roma_outdoor",
        560,
        864,
        True,
    )
    assert (cfg.matches_per_pair, cfg.min_certainty, cfg.stride) == (5000, 0.5, 1)
    assert (cfg.points_per_frame, cfg.seed) == (1000, 0)
    RomaInitConfig(model="tiny_roma_v1_outdoor", coarse_res=14, upsample_res=1, min_certainty=0)
    RomaInitConfig(min_certainty=1, seed=-5, upsample_preds=False)


@pytest.mark.parametrize(
    "field",
    [
        {"model": "roma_indoor"},
        {"model": None},
        {"coarse_res": 0},
        {"coarse_res": -14},
        {"coarse_res": 561},
        {"upsample_res": 0},
        {"matches_per_pair": 0},
        {"stride": 0},
        {"points_per_frame": 0},
        {"min_certainty": -0.01},
        {"min_certainty": 1.01},
        {"min_certainty": float("nan")},
        {"coarse_res": True},
        {"upsample_res": True},
        {"matches_per_pair": True},
        {"min_certainty": True},
        {"stride": True},
        {"points_per_frame": True},
        {"seed": False},
        {"stride": 1.0},
        {"coarse_res": 560.0},
        {"seed": 0.5},
        {"upsample_preds": 1},
    ],
    ids=lambda field: "-".join(f"{k}={v!r}" for k, v in field.items()),
)
def test_roma_init_config_rejects_each_invalid_value(field: dict[str, object]) -> None:
    """Q-106: a `model` outside its two values, a `coarse_res` that isn't a positive multiple
    of 14, `upsample_res` < 1, `matches_per_pair`, `stride` or `points_per_frame` < 1,
    `min_certainty` outside [0, 1] and a bool in a numeric field raise ValueError; so do a
    non-integer in an integer field and a non-bool `upsample_preds` (Q-122)."""
    name = next(iter(field))
    with pytest.raises(ValueError, match=name):
        RomaInitConfig(**field)  # type: ignore[arg-type]


# roma_provenance (Q-107)

_REPO = Path(__file__).resolve().parents[1]
_XFEAT_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"  # Q-116, source log row 107


def _local_corr_imports() -> bool:
    try:
        importlib.import_module("local_corr")
    except (ImportError, OSError):
        return False
    return True


@pytest.mark.parametrize("model", ["roma_outdoor", "tiny_roma_v1_outdoor"])
def test_roma_provenance_records_the_settings_and_commits(model: str) -> None:
    """Q-107: one key, `roma_init`, holding compact JSON of every RomaInitConfig field, the
    `use_custom_corr` that roma_init resolves (true only when `local_corr` imports, Q-112),
    the vendored RoMa commit (VENDOR.json) and, for Tiny RoMa only, XFeat's commit (Q-116)."""
    cfg = RomaInitConfig(model=model, stride=3, points_per_frame=7, seed=11)  # type: ignore[arg-type]
    meta = roma_provenance(cfg)
    assert list(meta) == ["roma_init"]
    text = meta["roma_init"]
    record = json.loads(text)
    assert text == json.dumps(record, separators=(",", ":"), sort_keys=True)
    vendor = json.loads((_REPO / "src/openftgs/_vendor/romatch/VENDOR.json").read_text())
    expected = {
        **dataclasses.asdict(cfg),
        "use_custom_corr": _local_corr_imports(),
        "roma_commit": vendor["commit"],
    }
    if model == "tiny_roma_v1_outdoor":
        expected["xfeat_commit"] = _XFEAT_COMMIT
    assert record == expected


def test_init_imports_without_the_vendored_roma() -> None:
    """Q-106 and the PRD's imports decision: `openftgs.init` imports, and `roma_provenance`
    runs, without importing the vendored RoMa, which only the RoMa adapters load. Run in a
    subprocess, since other tests may import the copy."""
    code = (
        "import sys\n"
        "from openftgs.init import RomaInitConfig, roma_provenance\n"
        "roma_provenance(RomaInitConfig())\n"
        "roma_provenance(RomaInitConfig(model='tiny_roma_v1_outdoor'))\n"
        "assert not [m for m in sys.modules if m.startswith('openftgs._vendor.romatch')]\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(openftgs.__file__).parents[1])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)


# The analytic stand-in dense matcher and its scenes (PRD #133, "Test helpers"). The images
# encode each pixel's (u, v) as its red and green values and carry one marker pixel, so that
# the stand-in identifies the record behind an image by a hash of its pixels. The stand-in
# knows the scene's geometry (material points at each raw τ) and the cameras, returns the
# exact correspondences of the points both images see, plus crafted matches per camera pair,
# and records every call.

_W, _H = 64, 32  # dyadic, so pixel centers have exact normalized coordinates
_FOCAL = 32.0
_CX, _CY = 32.0, 16.0
CameraSpec = tuple[float, tuple[float, float, float]]  # yaw about the world y axis (°), center


def _rotation(yaw: float) -> list[list[float]]:
    """R_w of a camera turned by `yaw` degrees about the world y axis: its forward axis is
    (sin, 0, cos). ±yaw give exactly mirrored entries."""
    c, s = math.cos(math.radians(abs(yaw))), math.sin(math.radians(abs(yaw)))
    s = math.copysign(s, yaw)
    return [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]


def _camera_json(spec: CameraSpec) -> dict[str, Any]:
    yaw, center = spec
    R = _rotation(yaw)
    t = [-sum(R[i][j] * center[j] for j in range(3)) for i in range(3)]
    return {
        "width": _W,
        "height": _H,
        "K": [[_FOCAL, 0.0, _CX], [0.0, _FOCAL, _CY], [0.0, 0.0, 1.0]],
        "world_to_camera": [[*R[i], t[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]],
    }


def _encoded_image(index: int) -> np.ndarray:
    pixels = np.zeros((_H, _W, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(_W, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = np.arange(_H, dtype=np.uint8)[:, None]
    pixels[0, 0, 2] = 1 + index  # the marker
    return pixels


Shot = tuple[str, float, str]  # camera, raw τ, split


def _write_scene(
    root: Path, cameras: dict[str, CameraSpec], shots: Sequence[Shot], *, name: str = "scene"
) -> Path:
    """A scene manifest with `cameras` in the given key order and `shots` in the given order,
    each image encoded as `_encoded_image(i)`."""
    root.mkdir(parents=True, exist_ok=True)
    images = []
    for i, (camera, tau, split) in enumerate(shots):
        image = f"images/{i:03d}.png"
        (root / "images").mkdir(exist_ok=True)
        Image.fromarray(_encoded_image(i), "RGB").save(root / image)
        images.append({"camera": camera, "timestamp": tau, "path": image, "split": split})
    manifest = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": {key: _camera_json(spec) for key, spec in cameras.items()},
        "images": images,
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(manifest))
    return path


def _digest(image: torch.Tensor) -> str:
    return hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest()


def _project_points(camera: Camera, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixels (§1.4's continuous convention) and depths of X [P, 3] in float64."""
    w2c = camera.world_to_camera.to(torch.float64)
    in_camera = X @ w2c[:3, :3].T + w2c[:3, 3]
    h = in_camera @ camera.K.to(torch.float64).T
    return h[:, :2] / h[:, 2:], in_camera[:, 2]


def _normalized(x: torch.Tensor) -> torch.Tensor:
    """§1.3 step 2 read backwards: pixels onto RoMa's [−1, 1]."""
    return x / torch.tensor([_W / 2, _H / 2], dtype=torch.float64) - 1.0


def _flags() -> tuple[object, ...]:
    cudnn = torch.backends.cudnn
    return (
        torch.get_float32_matmul_precision(),
        cudnn.benchmark,
        cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


Crafted = list[tuple[float, float, float, float, float]]  # x_A, y_A, x_B, y_B (px), certainty


class _StandIn:
    """The stand-in at the dense-matcher port (Q-106). Called with a device, it returns that
    shard's matcher; every shard's calls go to one log."""

    def __init__(
        self,
        dataset: MultiViewVideo,
        *,
        points: Callable[[float], torch.Tensor] | None = None,
        certainty: Callable[[int], float] | None = None,
        crafted: dict[tuple[str, str], Crafted] | None = None,
    ) -> None:
        self._records = dataset.records
        self._index = {_digest(dataset.load_image(i)): i for i in range(len(dataset.records))}
        self._points = points
        self._certainty = certainty
        self._crafted = crafted or {}
        self._lock = threading.Lock()
        self.calls: list[tuple[int, int]] = []  # record indices of images A and B
        self.devices: list[str] = []
        self.flags: list[tuple[object, ...]] = []

    def __call__(self, device: torch.device) -> DenseMatcher:
        def match(image_a: torch.Tensor, image_b: torch.Tensor) -> tuple[Tensor, Tensor]:
            return self._match(image_a, image_b, device)

        return match

    def _match(
        self, image_a: torch.Tensor, image_b: torch.Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        assert image_a.dtype == torch.float32 and image_a.shape == (_H, _W, 3)
        a, b = self._index[_digest(image_a)], self._index[_digest(image_b)]
        with self._lock:
            self.calls.append((a, b))
            self.devices.append(str(device))
            self.flags.append(_flags())
        record_a, record_b = self._records[a], self._records[b]
        rows: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        if self._points is not None:
            xa, za = _project_points(record_a.camera, self._points(record_a.timestamp))
            xb, zb = _project_points(record_b.camera, self._points(record_b.timestamp))
            size = torch.tensor([_W, _H], dtype=torch.float64)
            seen = (za > 0) & (zb > 0)
            seen &= ((xa >= 0) & (xa <= size) & (xb >= 0) & (xb <= size)).all(dim=1)
            index = torch.nonzero(seen).squeeze(1)
            rows.append(torch.cat([_normalized(xa[seen]), _normalized(xb[seen])], dim=1))
            if self._certainty is None:
                weights.append(torch.ones(len(index), dtype=torch.float64))
            else:
                values = [self._certainty(int(i)) for i in index]
                weights.append(torch.tensor(values, dtype=torch.float64))
        crafted = self._crafted.get((record_a.camera.name, record_b.camera.name), [])
        if crafted:
            table = torch.tensor(crafted, dtype=torch.float64)
            rows.append(torch.cat([_normalized(table[:, :2]), _normalized(table[:, 2:4])], 1))
            weights.append(table[:, 4])
        if not rows:
            return torch.zeros(0, 4, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
        return torch.cat(rows), torch.cat(weights)


def _plane(z: float, step: int = 2, *, u0: int = 9) -> torch.Tensor:
    """Points at depth z whose projections into a camera at the origin facing +z are the pixel
    centers (u + ½, v + ½) for u ≥ u0 and v ≥ 1, every `step` pixels."""
    u = torch.arange(u0, _W, step, dtype=torch.float64) + 0.5
    v = torch.arange(1, _H, step, dtype=torch.float64) + 0.5
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    X = torch.stack([(uu - _CX) / _FOCAL * z, (vv - _CY) / _FOCAL * z, torch.full_like(uu, z)])
    return X.reshape(3, -1).T.contiguous()


# A rectified rig: one orientation, baseline 0.5 along x, so the lattice at depth 2 projects to
# pixel centers 8 px apart in A and B.
_RIG: dict[str, CameraSpec] = {"A": (0.0, (0.0, 0.0, 0.0)), "B": (0.0, (0.5, 0.0, 0.0))}


def _static(z: float = 2.0) -> Callable[[float], torch.Tensor]:
    X = _plane(z)
    return lambda tau: X


def _rig_scene(root: Path, taus: Sequence[float]) -> MultiViewVideo:
    shots = [(camera, tau, "train") for tau in taus for camera in ("A", "B")]
    return MultiViewVideo.from_manifest(_write_scene(root, _RIG, shots))


def _run(
    dataset: MultiViewVideo, stand_in: _StandIn, devices: Sequence[str] = ("cpu",), **cfg: Any
) -> InitPoints:
    return roma_init(dataset, RomaInitConfig(**cfg), devices=list(devices), matcher=stand_in)


def test_u19_exact_matches_of_one_synchronized_pair_give_back_the_points(
    tmp_path: Path,
) -> None:
    """§1.3 steps 2–7 on a single timestamp: exact matches of a rectified pair triangulate back
    to the scene's points within 1e-6 (U-19's tolerance), each pair matched once; a single init
    frame gives v = 0 (step 5), the duration is k_stride with a time unit of 1 (D-70), frame
    ids are the slot index 0 and timestamps the images' τ."""
    dataset = _rig_scene(tmp_path, [0.0])
    stand_in = _StandIn(dataset, points=_static())
    points = _run(dataset, stand_in, points_per_frame=10_000)
    expected = _plane(2.0)
    assert stand_in.calls == [(0, 1)]
    got = points.positions.to(torch.float64)
    order = torch.cdist(got, expected).argmin(dim=1)
    assert sorted(order.tolist()) == list(range(expected.shape[0]))
    torch.testing.assert_close(got, expected[order], rtol=1e-6, atol=1e-6)
    assert points.velocities is not None and not points.velocities.any()
    assert points.durations is not None and torch.equal(
        points.durations, torch.ones(len(got), dtype=torch.float64)
    )
    assert points.frame_ids is not None and not points.frame_ids.any()
    assert not points.timestamps.any()


def _frame(points: InitPoints, k: int) -> torch.Tensor:
    assert points.frame_ids is not None
    return points.positions[points.frame_ids == k].to(torch.float64)


def _offsets_to_nearest(query: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """For each row of `query`, the offset to its nearest row of `others` by a float64 brute
    force on explicit differences, ties to the lower index."""
    d = ((query[:, None, :] - others[None, :, :]) ** 2).sum(dim=2)
    nearest = (d == d.amin(dim=1, keepdim=True)).to(torch.int8).argmax(dim=1)  # first at min
    return others[nearest] - query


def test_u19_e15_forward_and_backward_on_the_last_slot(tmp_path: Path) -> None:
    """E15, D-23, §1.3 step 5: on a plane moving in depth, with a budget that keeps every
    point, each velocity times dt equals the offset to the nearest output point of the next
    slot, and on the last slot of the previous one (dt < 0), found by a float64 brute force;
    dt = (k′ − k)·Δt_f·T_τ in raw units (Q-111). Within U-19's 1e-6."""
    plane = _plane(2.0)
    taus = [0.0, 1 / 30, 2 / 30]
    dataset = _rig_scene(tmp_path, taus)

    def moving(tau: float) -> torch.Tensor:
        return plane + torch.tensor([0.0, 0.0, 7.5 * tau], dtype=torch.float64)

    points = _run(dataset, _StandIn(dataset, points=moving), points_per_frame=10_000)
    assert points.velocities is not None and points.frame_ids is not None
    assert sorted(set(points.frame_ids.tolist())) == [0, 1, 2]
    step = dataset.frame_dt * (taus[-1] - taus[0])  # Δt_f·T_τ = 1/30
    for k, other in [(0, 1), (1, 2), (2, 1)]:
        v = points.velocities[points.frame_ids == k].to(torch.float64)
        offsets = _offsets_to_nearest(_frame(points, k), _frame(points, other))
        assert offsets.abs().amax() > 0.2  # the plane moved 0.25 in depth
        torch.testing.assert_close(v * (other - k) * step, offsets, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("taus", [[0.0], [0.0, 1 / 30]], ids=["one-timestamp", "stride-2"])
def test_u19_a_single_init_frame_gives_zero_velocities(tmp_path: Path, taus: list[float]) -> None:
    """§1.3 step 5, D-23: with a single init frame, v = 0: a single timestamp, or two frames at
    k_stride = 2, whose only slot is 0."""
    dataset = _rig_scene(tmp_path, taus)
    points = _run(dataset, _StandIn(dataset, points=_static()), stride=2)
    assert points.frame_ids is not None and not points.frame_ids.any()
    assert points.velocities is not None
    assert len(points.velocities) > 0 and not points.velocities.any()


@pytest.mark.parametrize("stride", [1, 3])
def test_u19_initial_durations_are_stride_times_the_frame_interval(
    tmp_path: Path, stride: int
) -> None:
    """§1.3 step 6, D-21: s₀ = k_stride·Δt_f, written in raw units, d = k_stride·Δt_f·T_τ. Seven
    frames 1/30 s apart give Δt_f = 1/6 and T_τ = 0.2, so d = k_stride/30 s."""
    dataset = _rig_scene(tmp_path, [n / 30 for n in range(7)])
    points = _run(dataset, _StandIn(dataset, points=_static()), stride=stride)
    assert points.durations is not None and points.frame_ids is not None
    assert sorted(set(points.frame_ids.tolist())) == list(range(0, 7, stride))
    expected = torch.full_like(points.durations, stride / 30)
    torch.testing.assert_close(points.durations, expected, rtol=1e-12, atol=0.0)


def _crafted(
    a: Camera, b: Camera, X: Sequence[float], *, shift_b: float = 0.0, certainty: float = 1.0
) -> tuple[float, float, float, float, float]:
    """The match of point X between cameras a and b, with B's pixel moved by `shift_b` pixels
    along y, across the rectified rig's horizontal epipolar lines."""
    point = torch.tensor([X], dtype=torch.float64)
    (xa, ya), (xb, yb) = _project_points(a, point)[0][0], _project_points(b, point)[0][0]
    return (float(xa), float(ya), float(xb), float(yb) + shift_b, certainty)


def _matched(got: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """`got` reordered to match `expected` row for row, nearest first; the pairing must be one
    to one."""
    order = torch.cdist(expected, got).argmin(dim=1)
    assert sorted(order.tolist()) == list(range(len(got)))
    return got[order]


def test_u19_filters_keep_exactly_the_expected_matches(tmp_path: Path) -> None:
    """§1.3 steps 2–3, D-22, D-58; Q-110, Q-120: of the stand-in's crafted matches between the
    rectified rig's two cameras, the output holds exactly the exact matches, the matches whose
    B pixel is 1 px off the epipolar line, and a distant point seen at 1.4°. Dropped: points
    behind both cameras, an 8 px offset, a distant point seen under 1°, the principal point in
    both views (parallel rays: `triangulate_dlt` gives a non-finite row, which is dropped
    explicitly), a NaN coordinate, a match outside image B, and a certainty below 0.5."""
    dataset = _rig_scene(tmp_path, [0.0])
    a, b = dataset.records[0].camera, dataset.records[1].camera
    exact = [
        (-0.5, -0.2, 2.0),
        (0.3, 0.1, 2.5),
        (0.0, 0.25, 3.0),
        (0.6, -0.3, 1.8),
        (-0.2, 0.0, 2.2),
        (0.1, 0.2, 4.0),
    ]
    one_px = [((0.4, 0.15, 2.0), 1.0), ((-0.1, -0.2, 3.0), -1.0)]
    far, control = (0.25, 0.0, 100.0), (0.25, 0.0, 20.0)
    crafted = [_crafted(a, b, X) for X in exact]
    crafted += [_crafted(a, b, X, shift_b=shift) for X, shift in one_px]
    crafted += [_crafted(a, b, control)]
    expected = torch.tensor(exact + [X for X, _ in one_px] + [control], dtype=torch.float64)
    crafted += [
        _crafted(a, b, (0.3, 0.2, -2.0)),  # behind both cameras
        _crafted(a, b, (-0.4, -0.1, -3.0)),
        _crafted(a, b, (0.2, -0.1, 2.0), shift_b=8.0),  # 8 px off
        _crafted(a, b, (-0.3, 0.05, 2.5), shift_b=-8.0),
        _crafted(a, b, far),  # seen under 1°
        (_CX, _CY, _CX, _CY, 1.0),  # parallel rays, the point at infinity
        (float("nan"), 16.0, 30.0, 16.0, 1.0),
        _crafted(a, b, (-1.6, 0.0, 2.0)),  # outside image B
        _crafted(a, b, (0.0, -0.1, 2.0), certainty=0.3),
    ]
    assert crafted[-2][2] < 0  # the match outside image B is exact otherwise

    # The distant point passes every filter but the angle: positive depths and reprojection
    # errors under 2 px, by an independent check of what triangulate_dlt returns for it.
    pa = torch.tensor([crafted[-5][:2]], dtype=torch.float64)
    pb = torch.tensor([crafted[-5][2:4]], dtype=torch.float64)
    P = [_projection_of(camera) for camera in (a, b)]
    X = triangulate_dlt(pa, pb, P[0], P[1])
    assert torch.isfinite(X).all()
    for camera, pixels in ((a, pa), (b, pb)):
        reprojected, depth = _project_points(camera, X)
        assert depth.item() > 0
        assert torch.linalg.vector_norm(reprojected - pixels).item() < 2.0
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)
    rays = X - centers
    angle = torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(rays[0], rays[1])), rays[0] @ rays[1]
    )
    assert math.degrees(angle.item()) < 1.0
    assert math.degrees(math.atan2(0.25, 20.0) * 2) > 1.0  # the control is seen at 1.4°

    stand_in = _StandIn(dataset, crafted={("A", "B"): crafted})
    points = _run(dataset, stand_in)
    got = _matched(points.positions.to(torch.float64), expected)
    torch.testing.assert_close(got[:6], expected[:6], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got[6:8], expected[6:8], rtol=0.0, atol=0.05)
    torch.testing.assert_close(got[8:], expected[8:], rtol=1e-6, atol=0.0)


def _projection_of(camera: Camera) -> torch.Tensor:
    w2c = camera.world_to_camera.to(torch.float64)
    return camera.K.to(torch.float64) @ w2c[:3, :]


def test_u19_colors_are_the_mean_of_the_two_source_pixels(tmp_path: Path) -> None:
    """§1.3 step 3, D-22; Q-110: each color is the mean of the two source pixels, each the
    pixel holding the match's continuous coordinate. The images encode (u, v) as (R, G); the
    rectified rig sees the plane's points at the pixel centers (u + ½, v + ½) in A and
    (u − 8 + ½, v + ½) in B, so the expected color is computed from the image files' values."""
    dataset = _rig_scene(tmp_path, [0.0])
    points = _run(dataset, _StandIn(dataset, points=_static()))
    assert len(points.positions) == len(_plane(2.0))
    xa, _ = _project_points(dataset.records[0].camera, points.positions.to(torch.float64))
    u, v = torch.floor(xa[:, 0]).long(), torch.floor(xa[:, 1]).long()
    image_a = torch.from_numpy(_encoded_image(0).astype(np.float32)) / 255.0
    image_b = torch.from_numpy(_encoded_image(1).astype(np.float32)) / 255.0
    expected = (image_a[v, u] + image_b[v, u - 8]) / 2
    assert len(torch.unique(expected, dim=0)) == len(expected)
    torch.testing.assert_close(points.colors, expected, rtol=0.0, atol=1e-6)


def test_u19_initial_durations_with_a_single_timestamp_are_the_stride(tmp_path: Path) -> None:
    """§1.3 step 6 with D-70: for a single timestamp Δt_f := 1 and the time unit is 1, so
    d = k_stride."""
    dataset = _rig_scene(tmp_path, [5.0])
    points = _run(dataset, _StandIn(dataset, points=_static()), stride=3)
    assert points.durations is not None
    assert torch.equal(points.durations, torch.full_like(points.durations, 3.0))


_FRAME = 1 / 30  # the frame period of the unsynchronized scenes, in seconds
# Three cameras of one orientation in a row: axis ties everywhere, so partners follow the
# center distance, and Q's tie between P and S follows the manifest's camera order.
_ROW: dict[str, CameraSpec] = {
    "S": (0.0, (1.0, 0.0, 0.0)),
    "Q": (0.0, (0.5, 0.0, 0.0)),
    "P": (0.0, (0.0, 0.0, 0.0)),
}


def _by_camera(dataset: MultiViewVideo, calls: Sequence[tuple[int, int]]) -> list[tuple[str, str]]:
    return [(dataset.records[a].camera.name, dataset.records[b].camera.name) for a, b in calls]


def test_u19_unsynchronized_cameras_fill_frame_slots_with_their_nearest_images(
    tmp_path: Path,
) -> None:
    """§1.3 steps 1–4, D-47; Q-109: cameras offset by less than half a frame (0, 0.1 and 0.35
    frames) each give a slot their image nearest to t_k, and P's extra image 0.45 frames from
    t_2 loses to its image at t_2. Each pair is matched once per slot, (Q, P) and (S, Q) in the
    manifest's camera order (S, Q, P); each point's timestamp is the mean raw τ of its pair's
    two images, and its frame id the slot."""
    offsets = {"P": 0.0, "Q": 0.1, "S": 0.35}
    shots: list[Shot] = [
        (camera, (n + offset) * _FRAME, "train")
        for n in range(5)
        for camera, offset in offsets.items()
    ]
    shots.insert(7, ("P", 2.45 * _FRAME, "train"))
    dataset = MultiViewVideo.from_manifest(_write_scene(tmp_path, _ROW, shots))
    stand_in = _StandIn(dataset, points=_static())
    points = _run(dataset, stand_in)

    index = {(camera, round(tau / _FRAME, 6)): i for i, (camera, tau, _) in enumerate(shots)}

    def image(camera: str, n: int) -> int:
        return index[camera, round(n + offsets[camera], 6)]

    expected = {(image(a, n), image(b, n)) for n in range(5) for a, b in [("Q", "P"), ("S", "Q")]}
    assert len(stand_in.calls) == len(expected)
    assert set(stand_in.calls) == expected
    means = {
        (n, (shots[a][1] + shots[b][1]) / 2)
        for n in range(5)
        for a, b in [(image("Q", n), image("P", n)), (image("S", n), image("Q", n))]
    }
    assert points.frame_ids is not None
    got = set(zip(points.frame_ids.tolist(), points.timestamps.tolist(), strict=True))
    assert got == means


def test_u19_a_camera_without_an_image_near_the_slot_sits_it_out(tmp_path: Path) -> None:
    """§1.3 step 1, D-47: near the ends of an unsynchronized rig, a camera with no image within
    Δt_f/2 of t_k sits the slot out. S's first image comes 0.6 frames after τ₀, so it joins
    slot 1, not slot 0; T's images stop a frame before τ₁, so it sits out slot 4. The log shows
    every slot's cameras, each with its image nearest to t_k."""
    cameras: dict[str, CameraSpec] = {**_ROW, "T": (0.0, (1.5, 0.0, 0.0))}
    frames = {"P": range(5), "Q": range(5), "S": range(4), "T": range(4)}
    offsets = {"P": 0.0, "Q": 0.1, "S": 0.6, "T": 0.0}
    shots: list[Shot] = [
        (camera, (n + offsets[camera]) * _FRAME, "train")
        for camera in cameras
        for n in frames[camera]
    ]
    dataset = MultiViewVideo.from_manifest(_write_scene(tmp_path, cameras, shots))
    assert dataset.frame_dt * (dataset.time.tau1 - dataset.time.tau0) == pytest.approx(_FRAME)
    stand_in = _StandIn(dataset, points=_static())
    _run(dataset, stand_in)

    slot_cameras: dict[int, set[str]] = {}
    for a, b in stand_in.calls:
        slots = {round(shots[i][1] / _FRAME) for i in (a, b)}  # the nearest frame
        assert len(slots) == 1  # both images of a pair belong to one slot
        slot_cameras.setdefault(slots.pop(), set()).update((shots[a][0], shots[b][0]))
    assert slot_cameras == {
        0: {"P", "Q", "T"},
        1: {"P", "Q", "S", "T"},
        2: {"P", "Q", "S", "T"},
        3: {"P", "Q", "S", "T"},
        4: {"P", "Q", "S"},
    }


# Five cameras turned about the vertical axis, all facing a plane at depth 3. X faces it; L
# and R are turned by ∓20°, so X's axis ties between them exactly; L′ and R′, turned by ∓25°,
# are the partners of L and R, so the pair X takes is visible in the log.
def _fan(r_center: float) -> dict[str, CameraSpec]:
    return {
        "X": (0.0, (0.0, 0.0, 0.0)),
        "L": (-20.0, (-0.5, 0.0, 0.0)),
        "Lp": (-25.0, (-1.0, 0.0, 0.0)),
        "R": (20.0, (r_center, 0.0, 0.0)),
        "Rp": (25.0, (r_center + 0.5, 0.0, 0.0)),
    }


@pytest.mark.parametrize(
    ("r_center", "order", "expected"),
    [
        (1.0, ["R", "Rp", "X", "L", "Lp"], [("R", "Rp"), ("X", "L"), ("L", "Lp")]),
        (0.5, ["R", "Rp", "X", "L", "Lp"], [("R", "Rp"), ("R", "X"), ("L", "Lp")]),
        (0.5, ["L", "Lp", "X", "R", "Rp"], [("L", "Lp"), ("L", "X"), ("R", "Rp")]),
    ],
    ids=["distance-decides", "tie-to-first-in-manifest", "tie-to-first-in-manifest-swapped"],
)
def test_u19_a_pair_chosen_by_both_cameras_is_matched_once(
    tmp_path: Path, r_center: float, order: list[str], expected: list[tuple[str, str]]
) -> None:
    """§1.3 step 2, D-22; Q-108, Q-109: each camera pairs with the camera whose optical axis is
    angularly closest; a tie goes to the smaller center distance, then to the earlier camera
    in the manifest's `cameras` object. A pair chosen by both cameras, such as (L, L′), is
    matched once, oriented (A, B) by that order."""
    fan = _fan(r_center)
    cameras = {name: fan[name] for name in order}
    shots: list[Shot] = [(name, 0.0, "train") for name in ["X", "L", "Lp", "R", "Rp"]]
    dataset = MultiViewVideo.from_manifest(_write_scene(tmp_path, cameras, shots))
    assert dataset.manifest.camera_names == tuple(order)
    stand_in = _StandIn(dataset, points=_static(3.0))
    _run(dataset, stand_in)
    assert sorted(_by_camera(dataset, stand_in.calls)) == sorted(expected)


def test_u19_initialization_reads_no_test_image(tmp_path: Path) -> None:
    """§1.3 step 1, D-66: test-split images take no part in slots, pairs, matches or colors. A
    test camera T sits between A and B, nearer to each than they are to each other, and A has
    a test image of its own; once the dataset is built, every test image's file is deleted,
    `roma_init` still succeeds, and no matcher call receives a test image."""
    cameras = {**_RIG, "T": (0.0, (0.25, 0.0, 0.0))}
    shots: list[Shot] = []
    for tau in (0.0, _FRAME):
        shots += [("T", tau, "test"), ("A", tau, "train"), ("A", tau + 1e-3, "test")]
        shots += [("B", tau, "train")]
    dataset = MultiViewVideo.from_manifest(_write_scene(tmp_path, cameras, shots))
    stand_in = _StandIn(dataset, points=_static())
    tests = [i for i, record in enumerate(dataset.records) if record.split == "test"]
    for i in tests:
        dataset.records[i].image_path.unlink()
    points = _run(dataset, stand_in)
    assert len(points.positions) > 0
    assert sorted(_by_camera(dataset, stand_in.calls)) == [("A", "B"), ("A", "B")]
    assert not {i for call in stand_in.calls for i in call} & set(tests)


_FIELDS = ("positions", "colors", "timestamps", "velocities", "durations", "frame_ids")


def _same(a: InitPoints, b: InitPoints) -> bool:
    return all(torch.equal(getattr(a, name), getattr(b, name)) for name in _FIELDS)


@pytest.mark.parametrize("stride", [1, 2])
def test_u19_seeded_subsets_of_at_most_stride_times_b_points_per_frame(
    tmp_path: Path, stride: int
) -> None:
    """§1.3 step 7, D-24; Q-111: each init frame keeps at most k_stride·B points, a seeded
    uniform subset kept in the order of the frame's full set; the same seed gives identical
    init points, another seed another subset."""
    dataset = _rig_scene(tmp_path, [n * _FRAME for n in range(5)])
    b = 100

    def run(seed: int, budget: int = b) -> InitPoints:
        stand_in = _StandIn(dataset, points=_static())
        return _run(dataset, stand_in, stride=stride, points_per_frame=budget, seed=seed)

    full, first, again, other = run(4, 10_000), run(4), run(4), run(5)
    assert _same(first, again)
    assert first.frame_ids is not None and full.frame_ids is not None
    assert other.frame_ids is not None
    for k in range(0, 5, stride):
        whole = full.positions[full.frame_ids == k]
        kept = first.positions[first.frame_ids == k]
        assert len(whole) > stride * b
        assert len(kept) == stride * b
        # Each kept point is one of the frame's points, in the frame's order.
        rows = [int((whole == point).all(dim=1).nonzero()) for point in kept]
        assert rows == sorted(set(rows))
        others = other.positions[other.frame_ids == k]
        assert set(map(tuple, kept.tolist())) != set(map(tuple, others.tolist()))


def test_u19_scales_come_from_the_points_roma_init_keeps(tmp_path: Path) -> None:
    """§1.3 step 8, D-25: `roma_init` → `from_init_points` gives each point the mean distance
    to its 3 nearest neighbours among the output points that share its frame id (the points
    kept after `roma_init`'s per-frame subsample), floored at 1e-7, by a float64 brute force."""
    dataset = _rig_scene(tmp_path, [n * _FRAME for n in range(3)])
    points = _run(dataset, _StandIn(dataset, points=_static()), points_per_frame=150)
    model = FreeTimeGaussians.from_init_points(
        points,
        dataset.time,
        InitConfig(points_per_frame=1000),
        frame_dt=dataset.frame_dt,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(model.means.detach(), points.positions)
    assert points.frame_ids is not None
    expected = torch.empty(len(points.positions), dtype=torch.float64)
    for k in range(3):
        rows = points.frame_ids == k
        assert int(rows.sum()) == 150
        X = points.positions[rows].to(torch.float64)
        d = torch.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(dim=2))
        d.fill_diagonal_(math.inf)
        nearest = torch.sort(d, dim=1).values[:, :3]
        expected[rows] = torch.log(nearest.mean(dim=1).clamp_min(1e-7))
    got = model.log_scales.detach().to(torch.float64)
    torch.testing.assert_close(got, expected[:, None].expand(-1, 3), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("taus", "frames"), [([n * _FRAME for n in range(10)], 10), ([0.0], 1)], ids=["F10", "one"]
)
def test_u19_a_file_whose_init_frames_total_more_than_f_times_b_builds_a_model(
    tmp_path: Path, taus: list[float], frames: int
) -> None:
    """D-74, §1.3 step 7: at k_stride = 3, `roma_init`'s per-frame caps sum to more than F·B
    (12·B for F = 10, whose slots are 0, 3, 6 and 9; 3·B for a single timestamp). Through
    `write_init_points(metadata=roma_provenance(cfg))`, `read_init_points` and
    `from_init_points(n_max=F·B)`, the model holds at most F·B points and passes the N_max
    check, and the file's metadata holds the `roma_init` key."""
    b = 20
    dataset = _rig_scene(tmp_path / "scene", taus)
    cfg = RomaInitConfig(stride=3, points_per_frame=b)
    points = roma_init(dataset, cfg, devices=["cpu"], matcher=_StandIn(dataset, points=_static()))
    slots = len(range(0, frames, 3))
    assert len(points.positions) == slots * 3 * b > frames * b
    path = tmp_path / "init.safetensors"
    write_init_points(points, path, time_unit="s", producer="test", metadata=roma_provenance(cfg))
    with safe_open(str(path), framework="pt") as f:
        assert json.loads(f.metadata()["roma_init"])["stride"] == 3
    model = FreeTimeGaussians.from_init_points(
        read_init_points(path, time_unit="s"),
        dataset.time,
        InitConfig(points_per_frame=b),
        frame_dt=dataset.frame_dt,
        generator=torch.Generator().manual_seed(0),
        n_max=frames * b,
    )
    assert len(model.means) == frames * b


def _permuted_copy(path: Path, seed: int) -> Path:
    """The manifest at `path` with its `images` array shuffled, beside it."""
    raw = json.loads(path.read_text())
    random.Random(seed).shuffle(raw["images"])
    copy = path.with_name("permuted.json")
    copy.write_text(json.dumps(raw))
    return copy


def test_u33_shards_and_pair_order_give_bitwise_identical_init_points(tmp_path: Path) -> None:
    """U-33 (CPU), D-54, D-56; Q-106, Q-111, Q-112: one shard, three shards, and three shards
    on a copy of the manifest whose images are shuffled give bitwise identical init points.
    The scene draws a subset of each pair's matches (certainties vary) and of each slot's
    points; it is unsynchronized, with a moving plane, and holds no two images of one camera
    at one timestamp."""
    offsets = {"X": 0.0, "L": 0.1, "Lp": -0.2, "R": 0.3, "Rp": 0.05}
    shots: list[Shot] = [
        (camera, (n + offset) * _FRAME, "train")
        for n in range(4)
        for camera, offset in offsets.items()
    ]
    path = _write_scene(tmp_path, _fan(1.0), shots)
    plane = _plane(3.0, step=3)

    def moving(tau: float) -> torch.Tensor:
        return plane + torch.tensor([3.0 * tau, 0.0, 6.0 * tau], dtype=torch.float64)

    def certainty(i: int) -> float:
        return 0.5 + 0.5 * ((i * 37) % 101) / 100

    outputs = []
    for manifest, devices in [
        (path, ["cpu"]),
        (path, ["cpu"] * 3),
        (_permuted_copy(path, 1), ["cpu"] * 3),
    ]:
        dataset = MultiViewVideo.from_manifest(manifest)
        stand_in = _StandIn(dataset, points=moving, certainty=certainty)
        outputs.append(
            _run(dataset, stand_in, devices, matches_per_pair=40, points_per_frame=50, seed=7)
        )
    assert outputs[0].frame_ids is not None
    assert sorted(set(outputs[0].frame_ids.tolist())) == [0, 1, 2, 3]
    assert len(outputs[0].positions) == 4 * 50  # every slot holds more than B points
    assert _same(outputs[0], outputs[1])
    assert _same(outputs[0], outputs[2])


def test_roma_init_sets_the_deterministic_flags_for_the_call_and_restores_them(
    tmp_path: Path,
) -> None:
    """§2.4 "Initialization and conversion", D-56; Q-112: during matching, float32 matmul
    precision is "highest", cuDNN autotuning is off and deterministic cuDNN and strict
    deterministic algorithms are on, whatever the caller set; afterwards the caller's flags
    are back, also when a dense matcher raises, whose exception reaches the caller."""
    dataset = _rig_scene(tmp_path, [0.0, _FRAME])
    cudnn = torch.backends.cudnn
    saved = _flags()
    try:
        torch.set_float32_matmul_precision("high")
        cudnn.benchmark, cudnn.deterministic = True, False
        torch.use_deterministic_algorithms(True, warn_only=True)
        before = _flags()
        stand_in = _StandIn(dataset, points=_static())
        _run(dataset, stand_in, ["cpu", "cpu"])
        assert set(stand_in.flags) == {("highest", False, True, True, False)}
        assert _flags() == before

        def failing(device: torch.device) -> DenseMatcher:
            def match(image_a: Tensor, image_b: Tensor) -> tuple[Tensor, Tensor]:
                raise RuntimeError("matcher failed")

            return match

        with pytest.raises(RuntimeError, match="matcher failed"):
            roma_init(dataset, RomaInitConfig(), devices=["cpu", "cpu"], matcher=failing)
        assert _flags() == before
    finally:
        precision, benchmark, deterministic, algorithms, warn_only = saved
        torch.set_float32_matmul_precision(str(precision))
        cudnn.benchmark, cudnn.deterministic = bool(benchmark), bool(deterministic)
        torch.use_deterministic_algorithms(bool(algorithms), warn_only=bool(warn_only))


def test_roma_init_refuses_devices_none_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-106: `devices` None means every visible CUDA device, so without one it raises
    ValueError (the hardware is stood in for: this machine may have a GPU)."""
    dataset = _rig_scene(tmp_path, [0.0])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="CUDA"):
        roma_init(dataset, RomaInitConfig(), matcher=_StandIn(dataset, points=_static()))


@pytest.mark.parametrize(
    "devices", ["cpu", [], ["not a device"]], ids=["string", "empty", "unknown"]
)
def test_roma_init_refuses_malformed_devices(tmp_path: Path, devices: Any) -> None:
    """Q-122: `devices` is a non-empty sequence of device names."""
    dataset = _rig_scene(tmp_path, [0.0])
    with pytest.raises(ValueError, match="device"):
        roma_init(dataset, RomaInitConfig(), devices=devices, matcher=_StandIn(dataset))


def test_roma_init_refuses_when_no_point_survives(tmp_path: Path) -> None:
    """Q-106: when no point survives matching and D-22's filters, roma_init raises
    ValueError; here every match lies behind both cameras."""
    dataset = _rig_scene(tmp_path, [0.0, _FRAME])
    a, b = dataset.records[0].camera, dataset.records[1].camera
    behind = [_crafted(a, b, (0.3, 0.2, -2.0))]
    with pytest.raises(ValueError, match="no point"):
        _run(dataset, _StandIn(dataset, crafted={("A", "B"): behind}))


@pytest.mark.parametrize(
    ("warp", "certainty"),
    [
        (torch.zeros(5, 3), torch.ones(5)),
        (torch.zeros(5, 4), torch.ones(4)),
        (torch.zeros(5, 4), torch.ones(5, 1)),
        (torch.zeros(5, 4, dtype=torch.int64), torch.ones(5)),
    ],
    ids=["warp-3-columns", "certainty-short", "certainty-2d", "integer-warp"],
)
def test_roma_init_refuses_a_malformed_dense_matcher_output(
    tmp_path: Path, warp: Tensor, certainty: Tensor
) -> None:
    """Q-106, Q-122: a dense matcher returns a floating-point warp [M, 4] and certainty
    [M]; anything else raises ValueError."""
    dataset = _rig_scene(tmp_path, [0.0])

    def matcher(device: torch.device) -> DenseMatcher:
        return lambda image_a, image_b: (warp, certainty)

    with pytest.raises(ValueError, match="dense matcher"):
        roma_init(dataset, RomaInitConfig(), devices=["cpu"], matcher=matcher)


# The RoMa adapters themselves run only with the real models, in the weights tests below
# (#137); RoMa is never stood in for (`.claude/skills/tdd/mocking.md`).


def test_pil_gives_roma_the_8_bit_values_back() -> None:
    """Q-110: RoMa's input is round(255·x) as uint8, so every 8-bit value that
    `MultiViewVideo.load_image` returns as x = b/255 in float32 comes back as b exactly."""
    values = torch.arange(256, dtype=torch.uint8)
    pixels = values[torch.randperm(256, generator=torch.Generator().manual_seed(0))]
    pixels = torch.cat([values, pixels, values.flip(0)]).reshape(16, 16, 3)
    image = _pil(pixels.to(torch.float32) / 255.0)
    assert image.mode == "RGB" and image.size == (16, 16)
    assert torch.equal(torch.from_numpy(np.array(image)), pixels)


def test_roma_matchers_load_xfeat_with_its_pinned_commits_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q-116, Q-123: the Tiny RoMa adapter builds XFeat from the pinned commit without its
    own weights, which XFeat's loader would take from the repository's `main`, and loads the
    weights of that commit, whose SHA-256 begins 0f5187fd (source log row 136), checked by
    torch.hub. torch.hub is stood in for, so nothing is downloaded."""
    net = torch.nn.Linear(2, 1)
    pinned = {name: torch.full_like(value, 0.25) for name, value in net.state_dict().items()}
    hub: list[tuple[str, str, dict[str, Any]]] = []
    fetched: list[tuple[str, dict[str, Any]]] = []

    def load(repo: str, model: str, **kwargs: Any) -> Any:
        hub.append((repo, model, kwargs))
        return SimpleNamespace(net=net)

    def fetch(url: str, **kwargs: Any) -> dict[str, Tensor]:
        fetched.append((url, kwargs))
        return pinned if "accelerated_features" in url else {}

    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fetch)
    _roma_matchers(RomaInitConfig(model="tiny_roma_v1_outdoor"))
    assert hub == [
        (
            f"verlab/accelerated_features:{_XFEAT_COMMIT}",
            "XFeat",
            {"pretrained": False, "top_k": 4096, "trust_repo": True, "skip_validation": True},
        )
    ]
    url = f"https://github.com/verlab/accelerated_features/raw/{_XFEAT_COMMIT}/weights/xfeat.pt"
    options = {"map_location": "cpu", "file_name": "xfeat-0f5187fd.pt", "check_hash": True}
    assert (url, options) in fetched
    for name, value in net.state_dict().items():
        assert torch.equal(value, pinned[name])


# Real RoMa (#137): `roma_init` with its default dense matchers on the vendored copy (D-58,
# D-65), downloading their weights. The scene is a textured plane, which RoMa can match: four
# training cameras on an arc 3 units from it, 320 × 256 px, at three synchronized timestamps,
# and a test camera in the middle of the arc, closer to the two inner cameras than they are to
# each other, so that a test image wrongly admitted would be paired and read (D-66).

_PLANE = TexturedPlane(seed=3, velocity=(0.6, 0.3, 0.0))
_ARC = {"C0": -15.0, "C1": -5.0, "C2": 5.0, "C3": 15.0}  # yaw of each training camera (°)
_ARC_TAUS = (0.0, 1 / 30, 2 / 30)


def _arc_camera(name: str, yaw: float, height: float = 0.3) -> Camera:
    theta = math.radians(yaw)
    eye = (3.0 * math.sin(theta), -3.0 * math.cos(theta), height)
    return look_at(eye, width=320, height=256, focal=280.0, name=name)


def _plane_scene(root: Path) -> Path:
    images: list[tuple[Camera, float, str]] = []
    for tau in _ARC_TAUS:
        images += [(_arc_camera(name, yaw), tau, "train") for name, yaw in _ARC.items()]
        images.append((_arc_camera("T", 0.0, -0.2), tau, "test"))
    return write_plane_scene(root, _PLANE, images)


def _slot_counts(points: InitPoints) -> dict[int, int]:
    assert points.frame_ids is not None
    slots, counts = torch.unique(points.frame_ids, return_counts=True)
    return dict(zip(slots.tolist(), counts.tolist(), strict=True))


def _plane_distances(points: InitPoints) -> Tensor:
    """Each point's distance to the plane at its timestamp (the plane moves along its normal)."""
    y = _PLANE.y0 + _PLANE.velocity[1] * points.timestamps
    return (points.positions[:, 1].to(torch.float64) - y).abs()


# The depth of one pixel of disparity between neighbouring arc cameras at the plane, Z²/(f·b)
# with Z = 3, f = 280 px and the baseline b = 6·sin 5°: about 0.061 units.
_PIXEL_DEPTH = 9.0 / (280.0 * 6.0 * math.sin(math.radians(5.0)))


def _smoke(
    tmp_path: Path,
    record_property: Callable[[str, object], None],
    cfg: RomaInitConfig,
    device: str,
    pixels: float,
) -> None:
    """The RoMa smoke test (PRD #133, "Added"): `roma_init` with real weights on the textured
    plane, the test images deleted once the dataset is built (D-66). The init points pass
    `validate_init_points`; every slot is matched and holds exactly its cap k_stride·B (§1.3
    step 7, D-24), since each of its pairs keeps more points than that; and the median point
    lies within `pixels` of disparity of the plane (a check of the pixel conventions end to
    end). The per-slot counts are recorded (the #37 evidence)."""
    dataset = MultiViewVideo.from_manifest(_plane_scene(tmp_path))
    tests = [record.image_path for record in dataset.records if record.split == "test"]
    assert len(tests) == len(_ARC_TAUS)
    for path in tests:
        path.unlink()
    points = roma_init(dataset, cfg, devices=[device])
    validate_init_points(points)
    counts = _slot_counts(points)
    median = float(_plane_distances(points).median()) / _PIXEL_DEPTH
    record_property("slot_counts", json.dumps(counts))
    record_property("plane_distance_median_px", median)
    print(f"{cfg.model} on {device}: per-slot counts {counts}, median distance to the plane")
    print(f"  {median:.3g} px of disparity")
    cap = cfg.stride * cfg.points_per_frame
    assert counts == {k: cap for k in range(len(_ARC_TAUS))}
    assert median < pixels


@pytest.mark.gpu
@pytest.mark.weights
def test_roma_init_with_roma_outdoor_on_a_textured_scene(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """The RoMa smoke test with roma_outdoor (D-58) on cuda:0, under the strict deterministic
    algorithms of Q-112 over RoMa's whole forward pass; see `_smoke`. RoMa's matches are
    sub-pixel: the median within a quarter pixel of disparity."""
    _smoke(tmp_path, record_property, RomaInitConfig(), "cuda:0", pixels=0.25)


@pytest.mark.weights
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_roma_init_with_tiny_roma_on_a_textured_scene(
    tmp_path: Path, record_property: Callable[[str, object], None], device: str
) -> None:
    """The Tiny RoMa smoke test (D-58; Q-116): XFeat is downloaded through torch.hub at the
    pinned commit e92685f, with torch.hub's validation skipped; see `_smoke`. Tiny RoMa, the
    lighter matcher, is held to a median within one pixel of disparity."""
    cfg = RomaInitConfig(model="tiny_roma_v1_outdoor")
    _smoke(tmp_path, record_property, cfg, device, pixels=1.0)


@pytest.mark.gpu
@pytest.mark.weights
def test_u33_roma_outdoor_shards_and_pair_order_give_bitwise_identical_init_points(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """U-33 (GPU, weights), D-54, D-56; Q-112: with real roma_outdoor, `devices=["cuda:0"]`,
    `["cuda:0"] * 3`, and `["cuda:0"] * 3` on a copy of the manifest whose images are shuffled
    give bitwise identical init points. Each pair's draw and each slot's subset bind (B is
    below the points a slot keeps). Verified here with `use_custom_corr` as the import check
    resolves it, off while fused-local-corr isn't installed on Python 3.14 (Q-6); the value is
    recorded."""
    path = _plane_scene(tmp_path)
    permuted = _permuted_copy(path, 1)
    order = [
        [(r.camera.name, r.timestamp) for r in MultiViewVideo.from_manifest(p).records]
        for p in (path, permuted)
    ]
    assert order[0] != order[1] and sorted(order[0]) == sorted(order[1])
    record_property("use_custom_corr", _local_corr_imports())
    outputs = [
        roma_init(MultiViewVideo.from_manifest(manifest), RomaInitConfig(), devices=devices)
        for manifest, devices in [
            (path, ["cuda:0"]),
            (path, ["cuda:0"] * 3),
            (permuted, ["cuda:0"] * 3),
        ]
    ]
    assert _slot_counts(outputs[0]) == {0: 1000, 1: 1000, 2: 1000}
    assert _same(outputs[0], outputs[1])
    assert _same(outputs[0], outputs[2])


# The #37 evidence: per-slot counts on openftgs-synth's unsynchronized `studio` (CC0-1.0, #108),
# converted by M12's SelfCap converter into pytest's temporary folder, outside the repository.

_STUDIO_WINDOW = (0, 16)  # frames N0 ≤ n < N1 of every camera, at 60 fps
_STUDIO_TEST_CAMERA = "0007"  # the dataset card's held-out camera


def _synth_root() -> Path:
    """$OPENFTGS_DATASETS/synth, version 0.1.x, or a skip naming the folder."""
    root = Path(os.environ.get("OPENFTGS_DATASETS", "")) / "synth"
    if not (root / "VERSION").is_file():
        pytest.skip(f"openftgs-synth is not at {root}")
    version = (root / "VERSION").read_text().strip()
    if not version.startswith("0.1."):
        pytest.skip(f"openftgs-synth {version} is not 0.1.x")
    return root


def _admitted_cameras(dataset: MultiViewVideo) -> dict[int, int]:
    """Training cameras per frame slot at k_stride = 1, counted in the test from the records:
    a camera joins slot k when one of its images has k as its nearest frame and lies within
    Δt_f/2 of t_k (§1.3 step 1, D-47; Q-109). Recorded as evidence, not asserted."""
    frame_dt = float(dataset.frame_dt)
    cameras: dict[int, set[str]] = {}
    for record in dataset.records:
        if record.split != "train":
            continue
        t = float(dataset.time.to_t(float(record.timestamp)))
        k = math.ceil(t / frame_dt - 0.5)
        if abs(t - k * frame_dt) <= frame_dt / 2:
            cameras.setdefault(k, set()).add(record.camera.name)
    return {k: len(names) for k, names in sorted(cameras.items())}


@pytest.mark.dataset
@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH (D-48)",
)
def test_roma_init_per_slot_counts_on_studio(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """The #37 evidence (PRD #133, "Added"): `roma_init` with roma_outdoor on 16 frames of
    `studio` (24 cameras, unsynchronized by up to ±0.03 s, about two frames at 60 fps), at its
    defaults but for B = 20,000, above what D-47's thin end slots hold, so that their counts
    are the points surviving D-22's filters. B is not unbounded, since E15 in float64 on the
    GPU takes about 19 s for 75,000 points against 75,000 (a full slot here). Recorded per
    slot: the training cameras admitted, the points kept at B = 20,000, and those B = 1000
    would keep. The converted manifest records the archive's SHA-256 from SHA256SUMS (D-79),
    and the init points pass `validate_init_points`."""
    root = _synth_root()
    manifest = convert_selfcap(
        root / "selfcap",
        tmp_path / "studio",
        scene="studio",
        window=_STUDIO_WINDOW,
        test_camera=_STUDIO_TEST_CAMERA,
        undistort="internal",
        jobs=8,
    )
    lines = (root / "SHA256SUMS").read_text().splitlines()
    sums = {line[66:]: line[:64] for line in lines}
    dataset = MultiViewVideo.from_manifest(manifest)
    archive = "studio-release.tar.gz"
    assert dataset.manifest.metadata["archives"] == {archive: sums[f"selfcap/{archive}"]}
    b = 20_000
    points = roma_init(dataset, RomaInitConfig(points_per_frame=b), devices=["cuda:0"])
    validate_init_points(points)
    kept = _slot_counts(points)
    cameras = _admitted_cameras(dataset)
    table = {k: (cameras.get(k, 0), kept.get(k, 0), min(kept.get(k, 0), 1000)) for k in cameras}
    record_property("studio_slots", json.dumps(table))
    print(f"studio, slot: (training cameras admitted, points kept at B = {b}, at B = 1000)")
    for k, row in table.items():
        print(f"  {k}: {row}")
    assert set(kept) <= set(cameras)
    assert 0 < max(kept.values()) <= b

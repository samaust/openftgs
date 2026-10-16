# SPDX-License-Identifier: Apache-2.0
"""Tests of openftgs.init: k-NN velocities (E15, D-23, D-56) and two-view DLT triangulation
(§1.3 step 3), U-19 and U-33 (§3.1), with the contracts of Q-106 and Q-120."""

import dataclasses
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import openftgs
from openftgs.init import (
    RomaInitConfig,
    _nearest_indices,
    knn_velocities,
    roma_provenance,
    triangulate_dlt,
)


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
    non-integer in an integer field and a non-bool `upsample_preds` (Q-T136.2)."""
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

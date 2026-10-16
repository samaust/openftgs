# SPDX-License-Identifier: Apache-2.0
"""Tests of openftgs.init: k-NN velocities (E15, D-23, D-56) and two-view DLT triangulation
(§1.3 step 3), U-19 and U-33 (§3.1), with the contracts of Q-106 and Q-T135.1."""

import pytest
import torch

from openftgs.init import _nearest_indices, knn_velocities


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
    """Q-T135.1: N = 0 gives an empty [0, 3] in the dtype of `points`."""
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
    """Q-T135.1: `points` must be [N, 3] and `next_points` [M, 3]."""
    with pytest.raises(ValueError, match=r"\[n, 3\]"):
        knn_velocities(points, next_points, 0.1)


def test_knn_velocities_refuses_integer_points() -> None:
    """Q-T135.1: both point sets must be floating point."""
    ints = torch.zeros(5, 3, dtype=torch.int64)
    with pytest.raises(ValueError, match="floating point"):
        knn_velocities(ints, _cloud(4, 1), 0.1)
    with pytest.raises(ValueError, match="floating point"):
        knn_velocities(_cloud(5, 0), ints, 0.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("which", ["points", "next_points"])
def test_knn_velocities_refuses_non_finite_coordinates(bad: float, which: str) -> None:
    """Q-T135.1: a coordinate that isn't finite raises ValueError; such a point has no
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
    points exactly 1 apart from the origin tie by distance alone."""
    a = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)
    others = torch.tensor(
        [[5.0, 5.0, 5.0], [-1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    data = torch.cat([others[:3], a.expand(40, 3), others[3:]])  # a at 3 … 42, (1, 0, 0) at 43
    order = torch.randperm(40, generator=torch.Generator().manual_seed(3))
    data[3:43] = data[3:43][order]  # the copies are identical, so shuffling them is a no-op
    query = torch.stack([a, a + 1e-3, torch.tensor([8.9, 0.0, 0.0]), torch.zeros(3)])
    expected = torch.tensor([3, 3, 2, 1])
    got = _nearest_indices(query.to(device), data.to(device))
    assert got.device.type == device
    assert torch.equal(got.cpu(), expected)


@pytest.mark.gpu
def test_knn_velocities_refuses_two_devices() -> None:
    """Q-T135.1: `points` and `next_points` must sit on one device."""
    with pytest.raises(ValueError, match="device|sit on"):
        knn_velocities(_cloud(5, 0), _cloud(4, 1).cuda(), 0.1)

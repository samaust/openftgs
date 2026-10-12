# SPDX-License-Identifier: Apache-2.0
"""FreeTimeGaussians and E5 in openftgs.model: U-5, the steps-7–9 part of U-19, the model side
of U-22, the shape parts of U-26, and the §1.1 validity rules (D-2, D-3, D-5, D-24, D-25; Q-16,
Q-17, Q-19 to Q-21, Q-23 to Q-25, Q-27).

Expected values are worked by hand from the spec: E5's matrix and its rotation about an axis,
the §1.1 initial values and shapes, the §2.2 ftgs-if conversions, and k-NN distances of point
layouts built so the answer is known.
"""

import math

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from torch import Tensor

from openftgs.model import FreeTimeGaussians, covariance, rotation_matrix
from openftgs.types import InitPoints, TimeNormalizer

F32 = torch.float32
F64 = torch.float64
PARAM_NAMES = (
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


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _params(n: int = 5, sh_degree: int = 3, seed: int = 0) -> dict[str, Tensor]:
    g = _gen(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "times": torch.rand(n, generator=g),
        "log_durations": torch.randn(n, generator=g) - 2.0,
        "velocities": torch.randn(n, 3, generator=g),
        "log_scales": torch.randn(n, 3, generator=g) - 3.0,
        "quats": torch.randn(n, 4, generator=g),
        "opacity_logits": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, (sh_degree + 1) ** 2 - 1, 3, generator=g),
    }


TIME = TimeNormalizer(2.0, 12.0)


def _model(n: int = 5, sh_degree: int = 3, seed: int = 0, **kw: object) -> FreeTimeGaussians:
    return FreeTimeGaussians(_params(n, sh_degree, seed), sh_degree=sh_degree, time=TIME, **kw)  # type: ignore[arg-type]


# U-5 · E5 covariance


def _e5_matrix(q: Tensor) -> Tensor:
    """E5's matrix written out entry by entry, for a unit quaternion (w, x, y, z)."""
    w, x, y, z = q.unbind(-1)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


def test_u5_rotation_about_z_by_a_known_angle() -> None:
    """U-5: q = (cos θ/2, 0, 0, sin θ/2) rotates by θ about z (worked by hand)."""
    theta = 0.7
    q = torch.tensor([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)], dtype=F64)
    c, s = math.cos(theta), math.sin(theta)
    expected = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=F64)
    torch.testing.assert_close(rotation_matrix(q), expected, rtol=0, atol=1e-6)


def _check_u5(q: Tensor, log_scales: Tensor) -> None:
    a = torch.exp(log_scales)
    r = rotation_matrix(q)
    unit = q / q.norm(dim=-1, keepdim=True)
    eye = torch.eye(3, dtype=q.dtype).expand_as(r)
    torch.testing.assert_close(r, _e5_matrix(unit), rtol=0, atol=1e-6)
    torch.testing.assert_close(r @ r.transpose(-1, -2), eye, rtol=0, atol=1e-6)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones_like(q[..., 0]), rtol=0, atol=1e-6)
    sigma = covariance(q, a)
    torch.testing.assert_close(sigma, sigma.transpose(-1, -2), rtol=0, atol=1e-6)
    eig = torch.linalg.eigvalsh(sigma)
    assert bool((eig > 0).all())
    expected = torch.sort(a * a, dim=-1).values
    torch.testing.assert_close(eig, expected, rtol=1e-6, atol=1e-6 * float(expected.max()))
    for other in (3.0 * q, -q):
        torch.testing.assert_close(covariance(other, a), sigma, rtol=0, atol=1e-6)


def test_u5_covariance_on_random_primitives() -> None:
    """U-5: R(q) equals E5's matrix and is orthonormal; Σ is symmetric positive definite with
    eigenvalues a²; q, 3q and −q give the same Σ (1e-6)."""
    g = _gen(5)
    _check_u5(
        torch.randn(64, 4, generator=g, dtype=F64), torch.randn(64, 3, generator=g, dtype=F64)
    )


@settings(deadline=None, max_examples=200)
@given(
    q=st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4),
    log_scales=st.lists(st.floats(-5.0, 2.0), min_size=3, max_size=3),
)
def test_u5_property_covariance(q: list[float], log_scales: list[float]) -> None:
    """U-5, hypothesis variant: any quaternion with norm well above the 1e-8 floor and any
    log-scales in a bounded domain."""
    qt = torch.tensor(q, dtype=F64)
    assume(float(qt.norm()) > 1e-3)
    _check_u5(qt, torch.tensor(log_scales, dtype=F64))


# §1.1 validity rules and U-26 shapes


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u26_sh_degree_sets_the_tensor_shapes(sh_degree: int) -> None:
    """U-26 and §1.1: shapes per L; at L = 0, shN is [N, 0, 3]."""
    model = _model(n=4, sh_degree=sh_degree)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    assert shapes == {
        "means": (4, 3),
        "times": (4,),
        "log_durations": (4,),
        "velocities": (4, 3),
        "log_scales": (4, 3),
        "quats": (4, 4),
        "opacity_logits": (4,),
        "sh0": (4, 1, 3),
        "shN": (4, (sh_degree + 1) ** 2 - 1, 3),
    }
    assert all(p.dtype == F32 for p in model.parameters())
    assert model.PARAM_NAMES == PARAM_NAMES
    assert model.num_gaussians == 4
    assert model.sh_degree == sh_degree
    assert model.time == TIME
    assert model.time_unit == "s"


@pytest.mark.parametrize("sh_degree", [4, -1, True, 2.0])
def test_u26_other_sh_degrees_are_rejected(sh_degree: object) -> None:
    """U-26 and D-3: only the ints 0, 1, 2 and 3 are accepted."""
    with pytest.raises(ValueError):
        FreeTimeGaussians(_params(sh_degree=3), sh_degree=sh_degree, time=TIME)  # type: ignore[arg-type]


def _broken(case: str) -> dict[str, Tensor]:
    p = _params()
    if case == "nan mean":
        p["means"][2, 1] = float("nan")
    elif case == "inf sh":
        p["shN"][0, 3, 0] = float("inf")
    elif case == "zero quaternion":
        p["quats"][1] = 0.0
    elif case == "tiny quaternion":
        p["quats"][1] = torch.tensor([1e-9, 0.0, 0.0, 0.0])
    elif case == "mismatched N":
        p["velocities"] = p["velocities"][:4]
    elif case == "wrong shape":
        p["log_scales"] = p["log_scales"][:, :2]
    elif case == "shN of another degree":
        p["shN"] = p["shN"][:, :8]
    elif case == "float64":
        p["times"] = p["times"].double()
    elif case == "missing name":
        del p["sh0"]
    elif case == "extra name":
        p["features"] = torch.zeros(5)
    return p


@pytest.mark.parametrize(
    "case",
    [
        "nan mean",
        "inf sh",
        "zero quaternion",
        "tiny quaternion",
        "mismatched N",
        "wrong shape",
        "shN of another degree",
        "float64",
        "missing name",
        "extra name",
    ],
)
def test_validity_rules_are_checked_on_construction(case: str) -> None:
    """§1.1 validity rules: all tensors finite, every quaternion norm > 1e-8, N identical across
    the nine tensors, §1.1 shapes and float32, exactly the nine names (Q-26: ValueError)."""
    with pytest.raises(ValueError):
        FreeTimeGaussians(_broken(case), sh_degree=3, time=TIME)


def test_time_unit_is_kept() -> None:
    """Q-17: the time unit is a keyword-only constructor argument and a public attribute."""
    assert _model(time_unit="frame").time_unit == "frame"


# at_time (§1.1 per-time arrays; Q-12, Q-20)


def test_at_time_is_the_temporal_slice_plus_the_sh_concat() -> None:
    """§1.1 per-time arrays: means_t, opacities_t, quats_n and scales from the temporal slice,
    sh = concat(sh0, shN) along dim 1 (Q-12, Q-20)."""
    from openftgs.model import GaussiansAtTime
    from openftgs.motion import temporal_slice

    model = _model(n=6)
    p = {k: v.detach() for k, v in _params(n=6).items()}
    got = model.at_time(0.4)
    assert isinstance(got, GaussiansAtTime)
    ref = temporal_slice(
        p["means"],
        p["times"],
        p["log_durations"],
        p["velocities"],
        p["log_scales"],
        p["quats"],
        p["opacity_logits"],
        0.4,
        impl="eager",
    )
    for name in ("means_t", "opacities_t", "quats_n", "scales", "opacities", "reg_terms"):
        torch.testing.assert_close(getattr(got, name), getattr(ref, name), rtol=0, atol=0)
    assert torch.equal(got.passes_cull, ref.passes_cull)
    assert torch.equal(got.sh, torch.cat([p["sh0"], p["shN"]], dim=1))


def test_at_time_cuts_sh_to_the_requested_degree() -> None:
    """§1.1 and E13: a lower SH degree keeps the first (L′ + 1)² bands; above L or below 0 is
    rejected (Q-20)."""
    model = _model(n=3, sh_degree=2)
    full = model.at_time(0.0).sh
    assert full.shape == (3, 9, 3)
    for degree in (0, 1, 2):
        cut = model.at_time(0.0, sh_degree=degree).sh
        assert torch.equal(cut, full[:, : (degree + 1) ** 2])
    for bad in (3, -1):
        with pytest.raises(ValueError):
            model.at_time(0.0, sh_degree=bad)


def test_at_time_carries_gradients_to_the_stored_tensors() -> None:
    """The per-time arrays are differentiable functions of the nine stored tensors."""
    model = _model(n=4)
    out = model.at_time(0.3)
    (out.means_t.sum() + out.opacities_t.sum() + out.sh.sum() + out.scales.sum()).backward()
    for name in ("means", "times", "velocities", "log_durations", "opacity_logits", "sh0", "shN"):
        assert getattr(model, name).grad is not None, name


# resize_ (§1.3 Growth step 4; Q-23)


def _rows(model: FreeTimeGaussians) -> dict[str, Tensor]:
    return {name: getattr(model, name).detach().clone() for name in PARAM_NAMES}


@pytest.mark.parametrize("as_mask", [True, False])
def test_resize_keeps_rows_in_order_and_appends_last(as_mask: bool) -> None:
    """§1.3 Growth step 4 and §2.1: kept rows stay in their old order and new rows go last; a
    bool mask and strictly increasing indices give the same result (Q-23)."""
    model = _model(n=5)
    before = _rows(model)
    new = _params(n=2, seed=9)
    keep = torch.tensor([True, False, True, True, False]) if as_mask else torch.tensor([0, 2, 3])
    model.resize_(keep=keep, append=new)
    assert model.num_gaussians == 5
    order = torch.tensor([0, 2, 3])
    for name in PARAM_NAMES:
        expected = torch.cat([before[name][order], new[name]])
        assert torch.equal(getattr(model, name).detach(), expected), name
        assert isinstance(getattr(model, name), torch.nn.Parameter)


def test_resize_without_keep_appends_and_without_append_shrinks() -> None:
    """§2.1 resize_: keep=None keeps every row; append=None adds none."""
    model = _model(n=3)
    before = _rows(model)
    model.resize_(append=_params(n=1, seed=4))
    assert model.num_gaussians == 4
    assert torch.equal(model.means.detach()[:3], before["means"])
    model.resize_(keep=torch.tensor([1, 3]))
    assert model.num_gaussians == 2
    assert torch.equal(model.means.detach()[0], before["means"][1])


@pytest.mark.parametrize(
    "case",
    [
        "unsorted indices",
        "duplicate indices",
        "index out of range",
        "mask of another length",
        "float keep",
        "append without shN",
        "append of another SH degree",
        "append with NaN",
    ],
)
def test_resize_rejects_invalid_input(case: str) -> None:
    """Q-23: keep is a bool mask [N] or strictly increasing int64 indices; append holds all nine
    names with matching trailing shapes; the result must pass the §1.1 validity rules."""
    model = _model(n=5)
    keep: Tensor | None = None
    append: dict[str, Tensor] | None = None
    if case == "unsorted indices":
        keep = torch.tensor([2, 0])
    elif case == "duplicate indices":
        keep = torch.tensor([1, 1])
    elif case == "index out of range":
        keep = torch.tensor([0, 5])
    elif case == "mask of another length":
        keep = torch.tensor([True, False])
    elif case == "float keep":
        keep = torch.tensor([0.0, 1.0])
    elif case == "append without shN":
        append = _params(n=2)
        del append["shN"]
    elif case == "append of another SH degree":
        append = _params(n=2, sh_degree=1)
    elif case == "append with NaN":
        append = _params(n=2)
        append["times"][0] = float("nan")
    before = _rows(model)
    with pytest.raises(ValueError):
        model.resize_(keep=keep, append=append)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])


# U-19, steps 7–9 · from_init_points (§1.3; D-5, D-21, D-23, D-24, D-25, D-47; Q-16, Q-19, Q-21)

C0 = 0.28209479177387814


def _points(
    positions: list[list[float]],
    timestamps: list[float],
    *,
    colors: list[list[float]] | None = None,
    velocities: list[list[float]] | None = None,
    durations: list[float] | None = None,
    frame_ids: list[int] | None = None,
) -> InitPoints:

    n = len(positions)
    return InitPoints(
        positions=torch.tensor(positions, dtype=F32),
        colors=torch.tensor(colors, dtype=F32) if colors else torch.full((n, 3), 0.5),
        timestamps=torch.tensor(timestamps, dtype=F64),
        velocities=torch.tensor(velocities, dtype=F32) if velocities else None,
        durations=torch.tensor(durations, dtype=F64) if durations else None,
        frame_ids=torch.tensor(frame_ids, dtype=torch.int64) if frame_ids else None,
    )


def _init(
    points: InitPoints,
    time: TimeNormalizer,
    frame_dt: float,
    *,
    b: int = 1000,
    seed: int = 0,
    sh_degree: int = 3,
    n_max: int | None = None,
    generator: torch.Generator | None = None,
) -> FreeTimeGaussians:
    from openftgs.config import InitConfig

    return FreeTimeGaussians.from_init_points(
        points,
        time,
        InitConfig(points_per_frame=b),
        frame_dt=frame_dt,
        generator=generator if generator is not None else _gen(seed),
        sh_degree=sh_degree,
        n_max=n_max,
    )


SPAN = TimeNormalizer(0.0, 10.0)  # T_τ = 10


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_u19_initial_values_of_step_9(sh_degree: int) -> None:
    """§1.3 step 9 and §1.1 (D-5): opacity logit(0.1), quaternion (1, 0, 0, 0), sh0 = (rgb −
    0.5)/C₀, shN = 0 with L's shape; means and time centers from the points (E14)."""
    pts = _points(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [2.5, 7.0],
        colors=[[0.5, 1.0, 0.0], [0.25, 0.5, 0.75]],
    )
    model = _init(pts, SPAN, 0.1, sh_degree=sh_degree)
    assert model.sh_degree == sh_degree
    assert torch.equal(model.means.detach(), pts.positions)
    torch.testing.assert_close(model.times.detach(), torch.tensor([0.25, 0.7]), rtol=0, atol=1e-7)
    logit = math.log(0.1 / 0.9)
    torch.testing.assert_close(
        model.opacity_logits.detach(), torch.full((2,), logit), rtol=1e-6, atol=0
    )
    assert torch.equal(model.quats.detach(), torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2))
    expected_sh0 = torch.tensor([[[0.0, 0.5 / C0, -0.5 / C0]], [[-0.25 / C0, 0.0, 0.25 / C0]]])
    torch.testing.assert_close(model.sh0.detach(), expected_sh0, rtol=1e-6, atol=1e-6)
    assert model.shN.shape == (2, (sh_degree + 1) ** 2 - 1, 3)
    assert not bool(model.shN.detach().any())
    assert all(p.dtype == F32 for p in model.parameters())


def test_u19_missing_velocities_and_durations_take_their_defaults() -> None:
    """§1.3 paragraph after step 9: missing velocities become 0 (D-23, P §4.2's "w/o 4D
    initialization") and missing durations s₀ = Δt_f (D-21)."""
    model = _init(_points([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.0, 10.0]), SPAN, 0.1)
    assert not bool(model.velocities.detach().any())
    torch.testing.assert_close(
        model.log_durations.detach(), torch.full((2,), math.log(0.1)), rtol=1e-6, atol=0
    )


def test_u19_supplied_durations_and_velocities_convert_to_normalized_time() -> None:
    """E14: s = d/T_τ and v = v_τ·T_τ (worked by hand with T_τ = 10; Q-27)."""
    pts = _points(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0.0, 10.0],
        velocities=[[1.0, -0.5, 0.0], [0.0, 0.0, 0.25]],
        durations=[0.5, 2.0],
    )
    model = _init(pts, SPAN, 0.1)
    torch.testing.assert_close(
        model.velocities.detach(), torch.tensor([[10.0, -5.0, 0.0], [0.0, 0.0, 2.5]])
    )
    torch.testing.assert_close(
        model.log_durations.detach(),
        torch.tensor([math.log(0.05), math.log(0.2)]),
        rtol=1e-6,
        atol=0,
    )


def test_u19_single_timestamp_uses_a_unit_of_one_and_one_frame() -> None:
    """§1.3 step 7 (F = 1 for a single timestamp) and Q-19 (unit of 1 when T_τ = 0)."""
    single = TimeNormalizer(5.0, 5.0)
    pts = _points(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        [5.0, 5.0, 5.0],
        velocities=[[1.0, 2.0, 3.0]] * 3,
        durations=[0.3, 0.3, 0.3],
    )
    model = _init(pts, single, 1.0, b=3, n_max=3)  # F·B = 1·3
    assert not bool(model.times.detach().any())
    torch.testing.assert_close(model.velocities.detach(), torch.tensor([[1.0, 2.0, 3.0]] * 3))
    torch.testing.assert_close(model.log_durations.detach(), torch.full((3,), math.log(0.3)))
    with pytest.raises(ValueError, match="N_max|n_max"):
        _init(pts, single, 1.0, b=3, n_max=2)


def _mean_knn(points: list[list[float]], i: int, k: int = 3) -> float:
    """Mean distance from point i to its k nearest others, worked out with math.dist."""
    d = sorted(math.dist(points[i], p) for j, p in enumerate(points) if j != i)
    return sum(d[:k]) / len(d[:k])


def test_u19_scales_use_three_nearest_neighbours_within_each_frame() -> None:
    """§1.3 step 8 and D-25: â = log(mean distance to the 3 nearest neighbours among the points
    of the same init frame), isotropic. Frame 1 repeats frame 0 shifted by 10⁻³, so pooled
    neighbours would collapse every scale to about 10⁻³."""
    frame0 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [2.0, 2.0, 2.0]]
    frame1 = [[x, y, z + 1e-3] for x, y, z in frame0]
    pts = _points(frame0 + frame1, [0.0] * 5 + [1.0] * 5, frame_ids=[0] * 5 + [1] * 5)
    model = _init(pts, SPAN, 0.1)
    expected = [_mean_knn(frame0, i) for i in range(5)] * 2
    scales = torch.exp(model.log_scales.detach().double())
    torch.testing.assert_close(
        scales, torch.tensor(expected, dtype=F64)[:, None].expand(10, 3), rtol=1e-6, atol=0
    )
    assert torch.equal(model.log_scales[:, 0], model.log_scales[:, 1])
    assert torch.equal(model.log_scales[:, 0], model.log_scales[:, 2])


def test_u19_small_frames_duplicates_and_lone_points_hit_the_floor_rules() -> None:
    """§1.3 step 8 with D-25's floor of 10⁻⁷ and Q-21: a frame of three points uses its two
    others; duplicated points have mean distance 0 and get the floor; a lone point gets the
    floor."""
    line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    dup = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    lone = [[9.0, 9.0, 9.0]]
    pts = _points(line + dup + lone, [0.0] * 6, frame_ids=[0, 0, 0, 1, 1, 2])
    model = _init(pts, SPAN, 0.1)
    floor = math.log(1e-7)
    expected = [math.log(2.0), math.log(1.5), math.log(2.5), floor, floor, floor]
    torch.testing.assert_close(
        model.log_scales.detach()[:, 0], torch.tensor(expected), rtol=1e-6, atol=0
    )


def test_u19_missing_frame_ids_become_the_nearest_slot() -> None:
    """§1.3 paragraph after step 9 and D-47: a point without a frame id joins the slot t_k =
    k·Δt_f nearest its time, ties to the lower slot (Q-21); unsynchronized times included.

    With Δt_f = 0.1 and T_τ = 10: t = 0.13 → slot 1, t = 0.07 → slot 1, t = 0.25 → slot 2 (a
    tie), t = 0.18 → slot 2. The groups are checked through their scales: slot 1 holds points
    0 and 1 (distance 1), slot 2 points 2 and 3 (distance 4)."""
    pts = _points(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        [1.3, 0.7, 2.5, 1.8],
    )
    model = _init(pts, SPAN, 0.1)
    torch.testing.assert_close(
        torch.exp(model.log_scales.detach()[:, 0]), torch.tensor([1.0, 1.0, 4.0, 4.0])
    )


def test_u19_budget_keeps_a_seeded_ordered_subset_of_f_times_b_points() -> None:
    """§1.3 step 7 and D-24: at most F·B points in total, F = round(1/Δt_f) + 1, as a seeded
    uniform subset kept in file order; the same seed gives the same subset (Q-21)."""
    xs = [[float(i), 0.0, 0.0] for i in range(40)]
    pts = _points(xs, [0.0] * 40, frame_ids=[0] * 40)
    a = _init(pts, SPAN, 0.5, b=4, seed=1)  # F = 3, F·B = 12
    b = _init(pts, SPAN, 0.5, b=4, seed=1)
    c = _init(pts, SPAN, 0.5, b=4, seed=2)
    kept = a.means.detach()[:, 0]
    assert kept.numel() == 12
    assert bool((kept[1:] > kept[:-1]).all())
    assert torch.equal(kept, b.means.detach()[:, 0])
    assert not torch.equal(kept, c.means.detach()[:, 0])


def test_u19_scales_come_from_the_kept_points() -> None:
    """D-25: scales are computed among the points kept after the subsample."""
    xs = [[float(i), 0.0, 0.0] for i in range(40)]
    model = _init(_points(xs, [0.0] * 40, frame_ids=[0] * 40), SPAN, 0.5, b=4, seed=3)
    kept = [[float(x), 0.0, 0.0] for x in model.means.detach()[:, 0]]
    expected = torch.tensor([math.log(_mean_knn(kept, i)) for i in range(len(kept))])
    torch.testing.assert_close(model.log_scales.detach()[:, 0], expected, rtol=1e-6, atol=0)


def test_u19_budget_leaves_the_generator_untouched_when_everything_fits() -> None:
    """Q-21: with N ≤ F·B every point is kept, in order, and the generator is not used."""
    g = _gen(4)
    state = g.get_state()
    pts = _points([[float(i), 0.0, 0.0] for i in range(12)], [0.0] * 12)
    model = _init(pts, SPAN, 0.5, b=4, generator=g)
    assert torch.equal(model.means.detach(), pts.positions)
    assert torch.equal(g.get_state(), state)


def test_u19_frame_count_rounds_half_up_and_checks_n_max() -> None:
    """§1.3 step 7, D-24 and Q-16, Q-21: Δt_f = 0.4 gives 1/Δt_f = 2.5, so F = 3 + 1 = 4 (round
    half up); initialization stops when F·B exceeds n_max."""
    pts = _points([[0.0, 0.0, 0.0]], [0.0])
    _init(pts, SPAN, 0.4, b=5, n_max=20)
    with pytest.raises(ValueError):
        _init(pts, SPAN, 0.4, b=5, n_max=19)


def test_u19_invalid_init_points_are_rejected() -> None:
    """Q-22: init points are validated as a ftgs-init file would be."""
    pts = _points([[0.0, 0.0, 0.0]], [0.0], colors=[[1.5, 0.0, 0.0]])
    with pytest.raises(ValueError):
        _init(pts, SPAN, 0.1)

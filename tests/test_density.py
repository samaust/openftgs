# SPDX-License-Identifier: Apache-2.0
"""openftgs.density (E10, E11, E17; §1.3 Growth, Relocation and Count-control modes; D-12 to D-18,
D-42 to D-45, D-49, D-75; Q-48 to Q-55): U-13 and its hypothesis variant, U-14, U-15, U-16, U-25
and the added tests of PRD #65.

Expected values are worked by hand from the spec (the NDC scaling of E10, E11 at n = 1 and 2,
E17's factor 1.6), come from properties the rows state (1 − (1 − σ_new)^n = σ, the target's
Gaussian, whose rotation comes from SciPy), from an independent evaluation of E11 written
literally as Python loops in 50-digit decimal (stricter than the float64 one U-13 names), or from
gsplat 1.5.3's `compute_relocation` as the oracle U-13 names (GPU).
"""

import decimal
import math
from typing import Literal

import pytest
import scipy.spatial.transform
import scipy.stats
import torch
from builders import (
    CAM_A,
    UNIT_TIME,
    model_from_rows,
    random_model,
    random_params,
    ring,
    sh0_for,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from openftgs.config import GrowthConfig, OptimConfig, RelocConfig
from openftgs.density import (
    OPACITY_RESET_EVERY,
    GradStats,
    GrowthReport,
    RelocationReport,
    draw_targets,
    grow,
    growth_masks,
    mcmc_split,
    prune,
    relocate,
    reset_opacity,
    sampling_score,
)
from openftgs.model import FreeTimeGaussians
from openftgs.optim import FTGSOptimizer, build_optimizer
from openftgs.render import render

F32 = torch.float32
F64 = torch.float64
PARAM_NAMES = (  # §1.1, written out: the tests' own expectation
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


# U-16 · gradient statistics (E10, D-12; Q-49)


def test_u16_unit_gradient_on_a_640x480_image_accumulates_320() -> None:
    """U-16: a 2D-mean gradient of (1, 0) on a 640×480 image accumulates 320 (x scaled by W/2,
    D-12), with a count of 1."""
    stats = GradStats(3)
    grad = torch.zeros(1, 3, 2)
    grad[0, 1] = torch.tensor([1.0, 0.0])
    visible = torch.tensor([[False, True, False]])
    stats.update(grad, width=640, height=480, visible=visible)
    assert torch.equal(stats.g_sum, torch.tensor([0.0, 320.0, 0.0]))
    assert torch.equal(stats.count, torch.tensor([0, 1, 0], dtype=torch.int32))


def test_u16_invisible_primitives_are_not_counted() -> None:
    """U-16: a primitive that isn't visible in an image adds nothing, whatever its gradient."""
    stats = GradStats(2)
    grad = torch.tensor([[[5.0, -7.0], [0.75, 0.5]]])
    stats.update(grad, width=8, height=16, visible=torch.tensor([[False, True]]))
    # D-12: (0.75·8/2, 0.5·16/2) = (3, 4), whose norm is 5.
    assert torch.equal(stats.g_sum, torch.tensor([0.0, 5.0]))
    assert torch.equal(stats.count, torch.tensor([0, 1], dtype=torch.int32))


def test_u16_packed_mode_counts_each_camera_through_gaussian_ids() -> None:
    """U-16: the same through gaussian_ids in packed mode, including a primitive seen by several
    cameras: each visible camera–primitive pair adds its norm and one (Q-49), so packed and
    unpacked layouts of the same pairs agree exactly (the norms are exact, so the order of the
    additions can't matter)."""
    # Three cameras; primitive 1 is visible in all three, primitive 3 in camera 2 only.
    pairs = [(0, 1, (0.75, 0.5)), (1, 1, (0.25, 0.0)), (2, 1, (0.0, 0.25)), (2, 3, (1.0, 0.0))]
    packed_grad = torch.tensor([g for _, _, g in pairs])
    ids = torch.tensor([i for _, i, _ in pairs])
    unpacked_grad = torch.zeros(3, 4, 2)
    visible = torch.zeros(3, 4, dtype=torch.bool)
    for c, i, g in pairs:
        unpacked_grad[c, i] = torch.tensor(g)
        visible[c, i] = True
    packed, unpacked = GradStats(4), GradStats(4)
    packed.update(packed_grad, width=8, height=16, gaussian_ids=ids)
    unpacked.update(unpacked_grad, width=8, height=16, visible=visible)
    # Primitive 1: 5 + 1 + 2 = 8 over three cameras; primitive 3: 4 over one.
    expected_sum = torch.tensor([0.0, 8.0, 0.0, 4.0])
    expected_count = torch.tensor([0, 3, 0, 1], dtype=torch.int32)
    for stats in (packed, unpacked):
        assert torch.equal(stats.g_sum, expected_sum)
        assert torch.equal(stats.count, expected_count)


def test_mean_is_the_average_over_visible_iterations_and_reset_zeroes() -> None:
    """E10: g is the mean, over the iterations in which a primitive was visible, of its norm;
    a primitive never visible has g = 0 (U-15). reset() zeroes both accumulators (§1.3 step 7)."""
    stats = GradStats(3)
    assert stats.num_gaussians == 3
    stats.update(
        torch.tensor([[[0.75, 0.5], [1.0, 0.0], [9.0, 9.0]]]),
        width=8,
        height=16,
        visible=torch.tensor([[True, True, False]]),
    )
    stats.update(
        torch.tensor([[[0.25, 0.0], [0.0, 0.0], [9.0, 9.0]]]),
        width=8,
        height=16,
        visible=torch.tensor([[True, False, False]]),
    )
    # Row 0: (5 + 1)/2; row 1: 4/1 (the invisible iteration doesn't count); row 2: never visible.
    assert torch.equal(stats.mean(), torch.tensor([3.0, 4.0, 0.0]))
    assert stats.mean().dtype == F32
    stats.reset()
    assert torch.equal(stats.g_sum, torch.zeros(3))
    assert torch.equal(stats.count, torch.zeros(3, dtype=torch.int32))
    assert torch.equal(stats.mean(), torch.zeros(3))


def _stats_with(sums: list[float], counts: list[int]) -> GradStats:
    """GradStats whose accumulators hold the given values, built through packed updates: each
    unit of count is one pair, and the first pair of a row carries its whole sum (W = H = 2, so
    a gradient (x, 0) adds x)."""
    stats = GradStats(len(sums))
    ids: list[int] = []
    grads: list[tuple[float, float]] = []
    for row, (total, count) in enumerate(zip(sums, counts, strict=True)):
        for k in range(count):
            ids.append(row)
            grads.append((total if k == 0 else 0.0, 0.0))
    stats.update(
        torch.tensor(grads).reshape(-1, 2),
        width=2,
        height=2,
        gaussian_ids=torch.tensor(ids, dtype=torch.int64),
    )
    return stats


def test_resize_keeps_rows_in_order_and_new_rows_take_their_parents_statistics() -> None:
    """§1.3 Growth step 4, D-45: kept rows keep their order and new rows go last, copying their
    parents' g_sum and count. Parents are indexed before the resize, so a split's children can
    name the parent the same resize removes; a parent may be named twice (Q-23, Q-49)."""
    stats = _stats_with([1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4])
    keep = torch.tensor([True, False, True, True])
    stats.resize_(keep=keep, append_from=torch.tensor([1, 1, 3]))
    assert torch.equal(stats.g_sum, torch.tensor([1.0, 3.0, 4.0, 2.0, 2.0, 4.0]))
    assert torch.equal(stats.count, torch.tensor([1, 3, 4, 2, 2, 4], dtype=torch.int32))
    assert stats.num_gaussians == 6
    stats.resize_(append_from=torch.tensor([0]))
    assert torch.equal(stats.count, torch.tensor([1, 3, 4, 2, 2, 4, 1], dtype=torch.int32))
    stats.resize_(keep=torch.tensor([False, True, True, True, True, True, False]))
    assert torch.equal(stats.g_sum, torch.tensor([3.0, 4.0, 2.0, 2.0, 4.0]))


def test_state_dict_round_trip_holds_g_sum_and_count() -> None:
    """§2.1 TrainerState.stats ("g_sum, count"), Q-25 ([N] tensors), Q-48: state_dict returns
    detached clones under exactly those keys, and load_state_dict restores them."""
    stats = _stats_with([1.5, 0.0, 2.5], [2, 0, 1])
    state = stats.state_dict()
    assert set(state) == {"g_sum", "count"}
    assert state["g_sum"].dtype == F32 and state["count"].dtype == torch.int32
    state["g_sum"][0] = 99.0  # a clone: the statistics don't change
    assert float(stats.g_sum[0]) == 1.5
    restored = GradStats(3)
    restored.load_state_dict(stats.state_dict())
    assert torch.equal(restored.g_sum, stats.g_sum)
    assert torch.equal(restored.count, stats.count)


@pytest.mark.parametrize(
    "state",
    [
        {"g_sum": torch.zeros(3)},
        {"g_sum": torch.zeros(3), "count": torch.zeros(3, dtype=torch.int32), "x": torch.zeros(3)},
        {"g_sum": torch.zeros(4), "count": torch.zeros(4, dtype=torch.int32)},
        {"g_sum": torch.zeros(3, dtype=torch.float64), "count": torch.zeros(3, dtype=torch.int32)},
        {"g_sum": torch.zeros(3), "count": torch.zeros(3)},
        {"g_sum": [0.0, 0.0, 0.0], "count": torch.zeros(3, dtype=torch.int32)},
    ],
)
def test_load_state_dict_rejects_other_layouts_and_changes_nothing(
    state: dict[str, object],
) -> None:
    """Q-48: exactly the keys g_sum and count, [N] tensors of float32 and int32."""
    stats = _stats_with([1.0, 2.0, 3.0], [1, 1, 1])
    with pytest.raises(ValueError):
        stats.load_state_dict(state)  # type: ignore[arg-type]
    assert torch.equal(stats.g_sum, torch.tensor([1.0, 2.0, 3.0]))


def _update_args(case: str) -> dict[str, object]:
    grad = torch.zeros(2, 3, 2)
    visible = torch.ones(2, 3, dtype=torch.bool)
    args: dict[str, object] = {"means2d_grad": grad, "width": 8, "height": 6, "visible": visible}
    if case == "both selectors":
        args["gaussian_ids"] = torch.zeros(6, dtype=torch.int64)
    elif case == "no selector":
        args["visible"] = None
    elif case == "bool width":
        args["width"] = True
    elif case == "zero height":
        args["height"] = 0
    elif case == "float width":
        args["width"] = 8.0
    elif case == "visible of another N":
        args["visible"] = torch.ones(2, 4, dtype=torch.bool)
    elif case == "gradient of another N":
        args["means2d_grad"] = torch.zeros(2, 4, 2)
    elif case == "visible not bool":
        args["visible"] = torch.ones(2, 3)
    elif case == "packed gradient with visible":
        args["means2d_grad"] = torch.zeros(6, 2)
    elif case == "packed ids int32":
        args.update(
            means2d_grad=torch.zeros(6, 2),
            visible=None,
            gaussian_ids=torch.zeros(6, dtype=torch.int32),
        )
    elif case == "packed ids of another length":
        args.update(
            means2d_grad=torch.zeros(6, 2),
            visible=None,
            gaussian_ids=torch.zeros(5, dtype=torch.int64),
        )
    elif case == "integer gradient":
        args["means2d_grad"] = torch.zeros(2, 3, 2, dtype=torch.int64)
    elif case == "visible on another device":
        args["visible"] = visible.to("meta")
    return args


@pytest.mark.parametrize(
    "case",
    [
        "both selectors",
        "no selector",
        "bool width",
        "zero height",
        "float width",
        "visible of another N",
        "gradient of another N",
        "visible not bool",
        "packed gradient with visible",
        "packed ids int32",
        "packed ids of another length",
        "integer gradient",
        "visible on another device",
    ],
)
def test_update_rejects_invalid_input_and_changes_nothing(case: str) -> None:
    """Q-49: exactly one of visible (unpacked, [C, N, 2] with [C, N]) and gaussian_ids (packed,
    [nnz, 2] with int64 [nnz]); width and height ints ≥ 1; shapes, dtypes and devices checked
    ("meta" stands in for another device). Q-26: ValueError."""
    stats = GradStats(3)
    with pytest.raises(ValueError):
        stats.update(**_update_args(case))  # type: ignore[arg-type]
    assert torch.equal(stats.count, torch.zeros(3, dtype=torch.int32))


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_u16_update_mean_and_reset_make_no_host_synchronization() -> None:
    """U-16, D-49, §2.4 rule 3: update in both layouts, mean and reset make no host
    synchronization (torch's sync debug mode, a prototype that may miss some operations)."""
    cuda = torch.device("cuda")
    gen = torch.Generator(device=cuda).manual_seed(0)
    stats = GradStats(1000, device=cuda)
    grad = torch.randn(2, 1000, 2, device=cuda, generator=gen)
    visible = torch.rand(2, 1000, device=cuda, generator=gen) > 0.5
    packed_grad = torch.randn(300, 2, device=cuda, generator=gen)
    ids = torch.randint(0, 1000, (300,), device=cuda, generator=gen)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        stats.update(grad, width=640, height=480, visible=visible)
        stats.update(packed_grad, width=640, height=480, gaussian_ids=ids)
        g = stats.mean()
        stats.reset()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert g.shape == (1000,)


@pytest.mark.gpu
@pytest.mark.parametrize("precull", ["internal", "compact"])  # Q-31, #423
def test_u16_gsplat_packed_and_unpacked_layouts_give_the_same_statistics(
    precull: Literal["internal", "compact"],
) -> None:
    """U-16, D-75 (U-16 is its detecting test): the 2D means are the gradient anchor in both
    layouts, so a two-camera gsplat render backpropagated through one loss gives the same counts,
    exactly, and the same g_sum within U-12's criterion for gsplat gradients (cosine ≥ 0.999,
    relative L2 error ≤ 1e-3): the two layouts run different kernels whose atomic backward sums
    in another order (2.5e-8 relative L2 measured on the RTX 4090)."""
    model = random_model(200, seed=3, spread=0.8, sh_degree=1).to("cuda")
    cameras = ring(2, width_px=64, height_px=48, focal=60.0)
    weights = torch.rand(2, 48, 64, 3, generator=torch.Generator().manual_seed(4)).to("cuda")
    results = []
    for packed in (True, False):
        model.zero_grad()
        out = render(
            model,
            cameras,
            0.5,
            time_units="normalized",
            backend="gsplat",
            packed=packed,
            precull=precull,
            return_aux=True,
        )
        torch.autograd.backward((out.image * weights).sum())
        assert out.means2d is not None and out.means2d.grad is not None
        stats = GradStats(model.num_gaussians, device="cuda")
        if out.gaussian_ids is not None:
            stats.update(out.means2d.grad, width=64, height=48, gaussian_ids=out.gaussian_ids)
        else:
            stats.update(out.means2d.grad, width=64, height=48, visible=out.visible)
        results.append(stats)
    packed_stats, unpacked_stats = results
    assert bool((unpacked_stats.count == 2).any())  # some primitive is seen by both cameras
    assert torch.equal(packed_stats.count, unpacked_stats.count)
    error = float(torch.linalg.vector_norm(packed_stats.g_sum - unpacked_stats.g_sum))
    assert error <= 1e-3 * float(torch.linalg.vector_norm(unpacked_stats.g_sum)), error
    cosine = torch.nn.functional.cosine_similarity(packed_stats.g_sum, unpacked_stats.g_sum, dim=0)
    assert float(cosine) >= 0.999


# U-15 · the sampling score (E10, D-13; Q-48)

O_DEAD = float(torch.tensor(math.log(0.005) - math.log1p(-0.005), dtype=F32))  # D-14


def _model_with_logits(logits: list[float], *, seed: int = 0) -> FreeTimeGaussians:
    """A random model (§1.1 tensors, SH degree 0) whose opacity logits are the given values."""
    params = random_params(len(logits), sh_degree=0, generator=torch.Generator().manual_seed(seed))
    params["opacity_logits"] = torch.tensor(logits, dtype=F32)
    return FreeTimeGaussians(params, sh_degree=0, time=UNIT_TIME)


def test_u15_normalized_gradient_uses_the_99th_percentile_of_live_visible_rows() -> None:
    """U-15, E10, D-13: with λ_g = 1 and λ_o = 0, ρ = ĝ = min(1, g/Q₀.₉₉), Q taken with
    torch.quantile's default interpolation over the live rows visible at least once; dead and
    never-visible rows don't enter Q (a huge g on a dead row, and the zeros of never-visible
    rows, would both move it); ĝ ∈ [0, 1] and ĝ = 1 above Q."""
    n = 120
    logits = [0.5] * n
    sums = [float(k + 1) for k in range(n)]
    counts = [1] * n
    for row in range(100, 110):  # never visible
        counts[row], sums[row] = 0, 0.0
    for row in range(110, 120):  # dead, with the largest g of all
        logits[row], sums[row] = -8.0, 1.0e6
    model = _model_with_logits(logits)
    stats = _stats_with(sums, counts)
    cfg = RelocConfig(lambda_g=1.0, lambda_o=0.0)
    g_hat = sampling_score(model, stats, cfg)
    g = stats.mean()
    q = torch.quantile(g[:100], 0.99)
    assert float(q) == float(torch.quantile(torch.arange(1.0, 101.0), 0.99))
    assert torch.equal(g_hat, torch.clamp(g / q, max=1.0))
    assert bool(((g_hat >= 0) & (g_hat <= 1)).all())
    assert torch.equal(g_hat[g >= q], torch.ones_like(g_hat[g >= q]))
    assert torch.equal(g_hat[100:110], torch.zeros(10))


@pytest.mark.parametrize("case", ["all zero", "mostly zero", "nothing live and visible"])
def test_u15_zero_quantile_gives_half_the_base_opacity(case: str) -> None:
    """U-15, E10: if Q₀.₉₉ is 0 (or there is no live, visible row to take it over), ĝ ≡ 0 and
    ρ = 0.5σ with the default weights."""
    n = 200
    logits = [float(v) for v in torch.linspace(-4.0, 4.0, n)]
    sums, counts = [0.0] * n, [1] * n
    if case == "mostly zero":
        sums[0] = 3.0  # one nonzero g in 200: the 99th percentile is still 0
    elif case == "nothing live and visible":
        counts = [0] * n
    model = _model_with_logits(logits)
    rho = sampling_score(model, _stats_with(sums, counts), RelocConfig())
    sigma = torch.sigmoid(torch.tensor(logits, dtype=F32))
    assert torch.equal(rho, 0.5 * sigma)


def test_sampling_score_rejects_statistics_of_another_size() -> None:
    """Q-48: the statistics must cover the model's rows (Q-26: ValueError)."""
    with pytest.raises(ValueError):
        sampling_score(_model_with_logits([0.0, 1.0]), GradStats(3), RelocConfig())


# U-13 · the relocation move (E11, D-14; Q-50)

_DEC = decimal.Context(prec=50)


def e11_oracle(sigma: float, n: int, a: float) -> tuple[float, float]:
    """E11 for one target, written literally (σ_new = 1 − (1 − σ)^(1/n) and the double sum over
    i and k), with n clamped to [1, 51], evaluated with Python's decimal at 50 digits: an
    independent evaluation, stricter than the float64 one U-13 names. Returns the unclamped σ_new
    and a_new."""
    n = min(max(n, 1), 51)
    one, s = decimal.Decimal(1), decimal.Decimal(sigma)
    s_new = _DEC.subtract(one, _DEC.power(_DEC.subtract(one, s), _DEC.divide(one, n)))
    denom = decimal.Decimal(0)
    for i in range(1, n + 1):
        for k in range(i):
            term = _DEC.divide(
                _DEC.multiply(math.comb(i - 1, k) * (-1) ** k, _DEC.power(s_new, k + 1)),
                _DEC.sqrt(decimal.Decimal(k + 1)),
            )
            denom = _DEC.add(denom, term)
    return float(s_new), float(_DEC.divide(_DEC.multiply(decimal.Decimal(a), s), denom))


def e11_from_logit(logit: float, n: int) -> tuple[float, float]:
    """E11 for a target given by its stored opacity logit, in 50-digit decimal: σ = e^ô/(1 + e^ô)
    and 1 − σ = 1/(1 + e^ô) are formed exactly from the logit, so a saturated target keeps its
    precision. Returns float32's view of the stored logit of the unclamped σ_new, logit(σ_new),
    and the scale factor a_new/a, with n clamped to [1, 51]."""
    n = min(max(n, 1), 51)
    one = decimal.Decimal(1)
    e = _DEC.exp(decimal.Decimal(logit))
    s, one_minus_s = _DEC.divide(e, _DEC.add(one, e)), _DEC.divide(one, _DEC.add(one, e))
    one_minus_new = _DEC.power(one_minus_s, _DEC.divide(one, n))  # (1 − σ)^(1/n)
    s_new = _DEC.subtract(one, one_minus_new)
    denom = decimal.Decimal(0)
    for i in range(1, n + 1):
        for k in range(i):
            term = _DEC.divide(
                _DEC.multiply(math.comb(i - 1, k) * (-1) ** k, _DEC.power(s_new, k + 1)),
                _DEC.sqrt(decimal.Decimal(k + 1)),
            )
            denom = _DEC.add(denom, term)
    new_logit = _DEC.ln(_DEC.divide(s_new, one_minus_new))
    return float(new_logit), float(_DEC.divide(s, denom))


def opacity_after_copies(sigma_new: float, n: int) -> float:
    """1 − (1 − σ_new)^n in 50-digit decimal: the opacity n stacked copies reach (E11)."""
    one = decimal.Decimal(1)
    return float(_DEC.subtract(one, _DEC.power(_DEC.subtract(one, decimal.Decimal(sigma_new)), n)))


def _split64(sigma: list[float], n: list[int], a: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """mcmc_split on float64 inputs with isotropic scales a."""
    scales = torch.tensor(a, dtype=F64)[:, None].expand(-1, 3).contiguous()
    return mcmc_split(torch.tensor(sigma, dtype=F64), scales, torch.tensor(n))


def test_u13_one_copy_leaves_opacity_and_scale_unchanged() -> None:
    """U-13: n = 1 leaves σ and a unchanged (1e-10 relative; E11 runs in float64)."""
    sigma = [0.005, 0.1, 0.5, 0.9, 0.999]
    new_sigma, new_scales = _split64(sigma, [1] * 5, [0.25, 1.0, 2.0, 1e-3, 30.0])
    torch.testing.assert_close(new_sigma, torch.tensor(sigma, dtype=F64), rtol=1e-10, atol=0.0)
    expected = torch.tensor([0.25, 1.0, 2.0, 1e-3, 30.0], dtype=F64)[:, None].expand(-1, 3)
    torch.testing.assert_close(new_scales, expected, rtol=1e-10, atol=0.0)


def test_u13_two_copies_give_one_minus_root_of_one_minus_sigma() -> None:
    """U-13: n = 2 gives σ_new = 1 − √(1 − σ): σ = 0.75 gives 0.5, and E11's sum is then
    2·0.5 − 0.5²/√2, so a_new = a·0.75/(1 − 1/(4√2)) (worked by hand)."""
    new_sigma, new_scales = _split64([0.75, 0.19], [2, 2], [1.0, 1.0])
    torch.testing.assert_close(new_sigma, torch.tensor([0.5, 0.1], dtype=F64), rtol=1e-10, atol=0.0)
    torch.testing.assert_close(
        new_scales[0],
        torch.full((3,), 0.75 / (1.0 - 1.0 / (4.0 * math.sqrt(2.0))), dtype=F64),
        rtol=1e-10,
        atol=0.0,
    )


def test_u13_n_copies_reach_the_original_opacity_and_scale_matches_e11() -> None:
    """U-13: 1 − (1 − σ_new)^n = σ for n = 1 … 51, and the scale factor (computed in float64)
    matches an independent evaluation of E11, both to 1e-10 relative."""
    for sigma in (0.02, 0.3, 0.75, 0.97, 1.0 - 2.0**-23):
        n = list(range(1, 52))
        new_sigma, new_scales = _split64([sigma] * 51, n, [0.5] * 51)
        for k, copies in enumerate(n):
            assert opacity_after_copies(float(new_sigma[k]), copies) == pytest.approx(
                sigma, rel=1e-10, abs=0.0
            )
            s_new, a_new = e11_oracle(sigma, copies, 0.5)
            assert float(new_sigma[k]) == pytest.approx(s_new, rel=1e-10, abs=0.0)
            assert float(new_scales[k, 0]) == pytest.approx(a_new, rel=1e-10, abs=0.0)


def test_u13_clamp_active_case_uses_the_unclamped_opacity_for_the_scale() -> None:
    """U-13, E11, D-14: with σ_j = 0.02 and n = 5, σ_new = 1 − 0.98^(1/5) < 0.005. a_new is
    computed from that unclamped σ_new; the lower bound is D-14's, applied to the stored logit
    by the procedures (Q-50), so mcmc_split returns σ_new as computed."""
    new_sigma, new_scales = _split64([0.02], [5], [2.0])
    s_new, a_new = e11_oracle(0.02, 5, 2.0)
    assert s_new < 0.005
    assert float(new_sigma[0]) == pytest.approx(s_new, rel=1e-10, abs=0.0)
    assert float(new_scales[0, 0]) == pytest.approx(a_new, rel=1e-10, abs=0.0)


def test_upper_clamp_applies_after_the_scale() -> None:
    """E11: σ_new is clamped to at most 1 − 2⁻²³ only after a_new is computed from the unclamped
    σ_new (so with n = 1, a stays a)."""
    new_sigma, new_scales = _split64([1.0 - 2.0**-30], [1], [3.0])
    assert float(new_sigma[0]) == 1.0 - 2.0**-23
    assert float(new_scales[0, 0]) == pytest.approx(3.0, rel=1e-10, abs=0.0)


def test_n_is_clamped_to_1_and_51() -> None:
    """E11: n is clamped to [1, 51]: n = 60 gives the n = 51 result and n ≤ 0 the n = 1 one."""
    new_sigma, new_scales = _split64([0.6] * 4, [60, 51, 0, -3], [1.0] * 4)
    assert torch.equal(new_sigma[0], new_sigma[1]) and torch.equal(new_scales[0], new_scales[1])
    assert torch.equal(new_sigma[2], new_sigma[3])
    assert float(new_sigma[2]) == pytest.approx(0.6, rel=1e-10, abs=0.0)


def test_mcmc_split_keeps_the_input_dtype() -> None:
    """Q-50: E11 is computed in float64 and returned in the input's dtype."""
    sigma = torch.tensor([0.3, 0.8], dtype=F32)
    new_sigma, new_scales = mcmc_split(sigma, torch.ones(2, 3), torch.tensor([3, 7]))
    assert new_sigma.dtype == F32 and new_scales.dtype == F32
    ref_sigma, ref_scales = mcmc_split(
        sigma.double(), torch.ones(2, 3, dtype=F64), torch.tensor([3, 7])
    )
    assert torch.equal(new_sigma, ref_sigma.float()) and torch.equal(new_scales, ref_scales.float())


@pytest.mark.parametrize(
    ("opacity", "scales", "n"),
    [
        (torch.ones(2) * 0.5, torch.ones(2, 3), torch.tensor([1.0, 2.0])),
        (torch.ones(2) * 0.5, torch.ones(2, 3), torch.tensor([True, False])),
        (torch.ones(2) * 0.5, torch.ones(3, 3), torch.tensor([1, 2])),
        (torch.ones(2) * 0.5, torch.ones(2, 2), torch.tensor([1, 2])),
        (torch.ones(2) * 0.5, torch.ones(2, 3, dtype=F64), torch.tensor([1, 2])),
        (torch.ones(2, 1) * 0.5, torch.ones(2, 3), torch.tensor([1, 2])),
        (torch.tensor([1, 0]), torch.ones(2, 3), torch.tensor([1, 2])),
    ],
)
def test_mcmc_split_rejects_invalid_input(
    opacity: torch.Tensor, scales: torch.Tensor, n: torch.Tensor
) -> None:
    """Q-50: opacity [k] and scales [k, 3] of one floating dtype, n [k] of an integer dtype."""
    with pytest.raises(ValueError):
        mcmc_split(opacity, scales, n)


@settings(max_examples=25, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.floats(min_value=1e-6, max_value=1.0 - 2.0**-23),
            st.integers(min_value=-3, max_value=60),
            st.floats(min_value=1e-3, max_value=1e3),
        ),
        min_size=1,
        max_size=64,
    )
)
def test_u13_property_e11_holds_for_random_targets(rows: list[tuple[float, int, float]]) -> None:
    """U-13 (hypothesis): for random σ in (0, 1 − 2⁻²³], n (clamped to [1, 51]) and a > 0,
    σ_new and a_new match the independent evaluation and n copies reach σ, to 1e-10 relative."""
    sigma, n, a = (list(col) for col in zip(*rows, strict=True))
    new_sigma, new_scales = _split64(sigma, n, a)
    for k in range(len(rows)):
        s_new, a_new = e11_oracle(sigma[k], n[k], a[k])
        assert float(new_sigma[k]) == pytest.approx(s_new, rel=1e-10, abs=0.0)
        assert float(new_scales[k, 0]) == pytest.approx(a_new, rel=1e-10, abs=0.0)
        copies = min(max(n[k], 1), 51)
        assert opacity_after_copies(float(new_sigma[k]), copies) == pytest.approx(
            sigma[k], rel=1e-10, abs=0.0
        )


@pytest.mark.gpu
def test_u13_matches_gsplat_compute_relocation() -> None:
    """U-13: against gsplat 1.5.3's compute_relocation (float32 kernel), within 5e-4 relative,
    over a grid of live target opacities in [0.005, 0.999] (D-14) and n = 1 … 51, as the spec's
    row asks, with no restriction: the grid includes the pairs whose σ_new falls below 0.005,
    where a_new is still used. gsplat's outputs are unclamped (source log #55), as mcmc_split's
    are below 1 − 2⁻²³. Measured on the RTX 4090: 4.9e-4 at worst, at σ = 0.005 and n = 36, from
    float32 cancellation in the kernel's 1 − (1 − σ)^(1/n); 1.7e-5 away from the clamp (source
    log #60, #62, #63)."""
    from gsplat.relocation import compute_relocation

    sigma = torch.linspace(0.005, 0.999, 400)[:, None].expand(400, 51).reshape(-1).contiguous()
    n = torch.arange(1, 52)[None, :].expand(400, 51).reshape(-1).contiguous()
    scales = torch.rand(sigma.shape[0], 3, generator=torch.Generator().manual_seed(0)) + 0.1
    binoms = torch.zeros(51, 51)  # E11's C(i − 1, k), i − 1 and k in 0 … 50
    for m in range(51):
        for k in range(m + 1):
            binoms[m, k] = math.comb(m, k)
    # compute_relocation clamps its ratios in place, so it gets a copy.
    ref_opacity, ref_scales = compute_relocation(
        sigma.cuda(), scales.cuda(), n.clone().cuda(), binoms.cuda()
    )
    new_opacity, new_scales = mcmc_split(sigma.double(), scales.double(), n)
    torch.testing.assert_close(ref_opacity.cpu().double(), new_opacity, rtol=5e-4, atol=0.0)
    torch.testing.assert_close(ref_scales.cpu().double(), new_scales, rtol=5e-4, atol=0.0)


# U-14 · the relocation procedure (§1.3 Relocation, D-14 to D-16; Q-50, Q-51, Q-55)

LOGIT_DEAD = math.log(1e-3 / (1.0 - 1e-3))  # σ = 0.001: dead (D-14) and culled (D-32)


def _relocation_scene(
    logits: list[float], *, seed: int = 0, sh_degree: int = 0
) -> tuple[FreeTimeGaussians, FTGSOptimizer]:
    """A random model whose rows have the given opacity logits and distinct time centers
    (row k has μt = k/N, so a moved copy names its target), with an optimizer whose Adam
    moments are nonzero after one step on seeded gradients (the logits and time centers are
    set back to these values after the step)."""
    n = len(logits)
    params = random_params(n, sh_degree=sh_degree, generator=torch.Generator().manual_seed(seed))
    params["opacity_logits"] = torch.tensor(logits, dtype=F32)
    params["times"] = torch.arange(n, dtype=F32) / n
    model = FreeTimeGaussians(params, sh_degree=sh_degree, time=UNIT_TIME)
    opt = build_optimizer(model, OptimConfig(), spatial_scale=1.0, frame_dt=0.1)
    grads = torch.Generator().manual_seed(seed + 1)
    for name in PARAM_NAMES:
        p = getattr(model, name)
        p.grad = torch.randn(p.shape, generator=grads)
    opt.step()
    opt.zero_grad()
    with torch.no_grad():  # the step moved them: restore the given logits and time centers
        model.opacity_logits.copy_(params["opacity_logits"])
        model.times.copy_(params["times"])
    return model, opt


def _rows_of(model: FreeTimeGaussians) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).detach().clone() for name in PARAM_NAMES}


def _target_of(model: FreeTimeGaussians, rows: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """The original row each given row's time center names (row k of the scene's N rows has
    μt = k/N; N is the model's row count unless given)."""
    n = model.num_gaussians if n is None else n
    return torch.round(model.times.detach()[rows] * n).to(torch.int64)


def test_u14_moved_copies_carry_their_targets_updated_rows() -> None:
    """U-14, §1.3 Relocation steps 3–5, D-15: every dead primitive becomes a moved copy of a live
    target, equal to its target's updated row except its position (bitwise); N is unchanged;
    afterwards every row is live (D-14: relocated logits are at least ô_dead); the report counts
    the dead rows and the distinct targets."""
    logits = [1.0, 0.0, LOGIT_DEAD, 2.0, LOGIT_DEAD, LOGIT_DEAD, -1.0, LOGIT_DEAD]
    model, opt = _relocation_scene(logits)
    stats = GradStats(8)
    report = relocate(model, opt, stats, RelocConfig(), generator=torch.Generator().manual_seed(5))
    assert model.num_gaussians == 8
    dead_rows = torch.tensor([2, 4, 5, 7])
    targets = _target_of(model, dead_rows)
    assert set(targets.tolist()) <= {0, 1, 3, 6}
    after = _rows_of(model)
    for name in PARAM_NAMES:
        if name != "means":
            assert torch.equal(after[name][dead_rows], after[name][targets]), name
    assert bool((model.opacity_logits.detach() >= O_DEAD).all())
    assert report == RelocationReport(dead=4, targets=len(set(targets.tolist())))


def test_u14_a_logit_exactly_at_the_dead_threshold_counts_as_live() -> None:
    """U-14, D-14: ô = ô_dead (the float32 logit of 0.005) is live; one float32 step below is
    dead. A moved copy takes its target's time center, a target keeps its own (exact)."""
    below = float(torch.nextafter(torch.tensor(O_DEAD), torch.tensor(-math.inf)))
    model, opt = _relocation_scene([O_DEAD, below, 2.0, 2.0])
    times = model.times.detach().clone()
    report = relocate(
        model, opt, GradStats(4), RelocConfig(), generator=torch.Generator().manual_seed(0)
    )
    assert report.dead == 1
    assert float(model.times.detach()[0]) == float(times[0])  # live: not overwritten
    assert int(_target_of(model, torch.tensor([1]))) in (0, 2, 3)
    assert float(model.times.detach()[1]) != float(times[1])  # dead: now a moved copy


@pytest.mark.parametrize("case", ["no dead primitive", "no live primitive"])
def test_u14_empty_dead_or_live_set_is_a_no_op_that_still_resets_statistics(case: str) -> None:
    """U-14, §1.3 Relocation step 1: with an empty dead set or an empty live set the model, the
    Adam moments and the generator are untouched (bitwise); the gradient statistics are still
    reset (Q-51, owner answer)."""
    logits = [0.5, 1.0, -1.0] if case == "no dead primitive" else [LOGIT_DEAD, -9.0, -12.0]
    model, opt = _relocation_scene(logits)
    before, moments = _rows_of(model), opt.state_dict()
    stats = _stats_with([1.0, 2.0, 3.0], [1, 1, 1])
    gen = torch.Generator().manual_seed(9)
    gen_state = gen.get_state()
    report = relocate(model, opt, stats, RelocConfig(), generator=gen)
    assert report == RelocationReport(dead=0, targets=0)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])
    for name, state in opt.state_dict().items():
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(state[key], moments[name][key])
    assert torch.equal(gen.get_state(), gen_state)
    assert torch.equal(stats.count, torch.zeros(3, dtype=torch.int32))


def test_u14_targets_are_drawn_in_proportion_to_the_score() -> None:
    """U-14, D-15: 10⁵ dead primitives draw their targets from 5 live ones with probability
    ∝ ρ (here 0.5σ, the statistics being zero): χ² test, p > 0.001."""
    sigma = [0.05, 0.1, 0.2, 0.3, 0.35]
    n_dead = 100_000
    logits = [math.log(s / (1.0 - s)) for s in sigma] + [LOGIT_DEAD] * n_dead
    model, opt = _relocation_scene(logits)
    rho = 0.5 * torch.sigmoid(torch.tensor(logits[:5], dtype=F32)).double()
    relocate(
        model,
        opt,
        GradStats(len(logits)),
        RelocConfig(),
        generator=torch.Generator().manual_seed(1),
    )
    observed = torch.bincount(_target_of(model, torch.arange(5, 5 + n_dead)), minlength=5)
    expected = rho / rho.sum() * n_dead
    p_value = scipy.stats.chisquare(observed.double().numpy(), expected.numpy()).pvalue
    assert p_value > 0.001, p_value


def test_u14_moments_are_zeroed_on_targets_and_moved_copies_only() -> None:
    """U-14, D-16: the Adam moments of every target and every moved copy are zero and the
    others unchanged (bitwise); step counters are unchanged. (Rows outside [0, N) would raise on
    the CPU, Q-44, so this also covers the range of the rows relocate resets.)"""
    logits = [1.0, LOGIT_DEAD, 0.0, LOGIT_DEAD, 0.5, 0.2, LOGIT_DEAD, 0.8, -0.5, 1.5]
    model, opt = _relocation_scene(logits)
    before = opt.state_dict()
    relocate(model, opt, GradStats(10), RelocConfig(), generator=torch.Generator().manual_seed(3))
    moved = torch.tensor([1, 3, 6])
    targets = torch.unique(_target_of(model, moved))
    reset = torch.zeros(10, dtype=torch.bool)
    reset[moved] = True
    reset[targets] = True
    after = opt.state_dict()
    for name in after:
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(after[name][key][reset], torch.zeros_like(after[name][key][reset]))
            assert torch.equal(after[name][key][~reset], before[name][key][~reset]), (name, key)
        assert torch.equal(after[name]["step"], before[name]["step"])


def test_without_reset_moved_moments_only_targets_are_zeroed() -> None:
    """D-16's alternative (RelocConfig.reset_moved_moments = False): only the targets' moments
    are zeroed; the moved copies keep the moments their rows had."""
    logits = [1.0, LOGIT_DEAD, 0.0, LOGIT_DEAD, 0.5]
    model, opt = _relocation_scene(logits)
    before = opt.state_dict()
    relocate(
        model,
        opt,
        GradStats(5),
        RelocConfig(reset_moved_moments=False),
        generator=torch.Generator().manual_seed(2),
    )
    targets = torch.unique(_target_of(model, torch.tensor([1, 3])))
    after = opt.state_dict()
    for name in after:
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                after[name][key][targets], torch.zeros_like(after[name][key][targets])
            )
            for row in (1, 3):
                assert torch.equal(after[name][key][row], before[name][key][row])


def test_u14_same_seed_gives_the_same_result() -> None:
    """U-14: the same model, statistics and generator seed give bitwise the same model, and the
    statistics are reset after the move (§1.3 step 7; Q-51)."""
    results = []
    for _ in range(2):
        model, opt = _relocation_scene([0.3, LOGIT_DEAD, 1.0, LOGIT_DEAD, LOGIT_DEAD, 2.0], seed=4)
        stats = _stats_with([0.5, 0.0, 2.0, 1.0, 0.0, 0.25], [1, 0, 2, 1, 0, 1])
        relocate(model, opt, stats, RelocConfig(), generator=torch.Generator().manual_seed(11))
        assert torch.equal(stats.count, torch.zeros(6, dtype=torch.int32))  # §1.3 step 7, Q-51
        assert torch.equal(stats.g_sum, torch.zeros(6))
        results.append(_rows_of(model))
    for name in PARAM_NAMES:
        assert torch.equal(results[0][name], results[1][name])


def test_stacked_copies_keep_the_targets_position() -> None:
    """RelocConfig.copy_position = "stack" (D-15's alternative, used by U-13's render test):
    moved copies keep the target's position exactly."""
    model, opt = _relocation_scene([1.0, LOGIT_DEAD, 0.0, LOGIT_DEAD])
    relocate(
        model,
        opt,
        GradStats(4),
        RelocConfig(copy_position="stack"),
        generator=torch.Generator().manual_seed(6),
    )
    moved = torch.tensor([1, 3])
    assert torch.equal(model.means.detach()[moved], model.means.detach()[_target_of(model, moved)])


def test_invalid_model_or_all_zero_scores_raise_before_any_draw() -> None:
    """§1.1 validity rules (Q-55) and Q-51: a non-finite model, or live scores that are all 0
    (λ_o = 0 with zero statistics, which Q-11 allows), raise ValueError before the generator is
    used and before anything is written."""
    gen = torch.Generator().manual_seed(1)
    state = gen.get_state()
    model, opt = _relocation_scene([1.0, LOGIT_DEAD, 0.0])
    with torch.no_grad():
        model.sh0[0, 0, 1] = math.nan
    with pytest.raises(ValueError):
        relocate(model, opt, GradStats(3), RelocConfig(), generator=gen)
    model, opt = _relocation_scene([1.0, LOGIT_DEAD, 0.0])
    before = _rows_of(model)
    with pytest.raises(ValueError, match="sampling scores"):
        relocate(model, opt, GradStats(3), RelocConfig(lambda_o=0.0), generator=gen)
    assert torch.equal(gen.get_state(), state)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])


def test_a_target_drawn_more_than_50_times_uses_n_51() -> None:
    """E11: n is clamped to 51 for the formula, but every one of the 60 dead primitives is still
    moved onto the single live target, and all 61 rows carry E11's values for n = 51."""
    model, opt = _relocation_scene([0.4] + [LOGIT_DEAD] * 60)
    sigma = float(torch.sigmoid(torch.tensor(0.4, dtype=F32)).double())
    a = torch.exp(model.log_scales.detach()[0].double())
    report = relocate(
        model, opt, GradStats(61), RelocConfig(), generator=torch.Generator().manual_seed(0)
    )
    assert report == RelocationReport(dead=60, targets=1)
    s_new, _ = e11_oracle(sigma, 51, 1.0)
    expected_logit = float(torch.logit(torch.tensor(s_new, dtype=F64)).float())
    assert torch.equal(model.opacity_logits.detach(), torch.full((61,), expected_logit))
    expected_scales = torch.log(torch.stack([a * e11_oracle(sigma, 51, 1.0)[1]] * 61)).float()
    assert torch.equal(model.log_scales.detach(), expected_scales)
    assert torch.equal(_target_of(model, torch.arange(61)), torch.zeros(61, dtype=torch.int64))


@pytest.mark.parametrize("logit", [30.0, 40.0])
def test_relocation_keeps_e11_exact_for_a_saturated_target(logit: float) -> None:
    """E11 (Q-50, owner answer after the second M8 milestone review, F-1): a target whose stored
    logit is so large that its float64 σ rounds to 1 (above about 36.7) or nearly so still gets
    E11's σ_new, because log(1 − σ) is formed from the logit. One target and 50 dead primitives
    (n = 51): every row's logit is the float32 logit of E11's σ_new (σ_new ≈ 0.44 for ô = 30
    and 0.54 for ô = 40), and its log-scale float32(log(a·a_new/a)), against a 50-digit decimal
    oracle that starts from the logit."""
    model, opt = _relocation_scene([logit] + [LOGIT_DEAD] * 50)
    a = torch.exp(model.log_scales.detach()[0].double())
    relocate(model, opt, GradStats(51), RelocConfig(), generator=torch.Generator().manual_seed(0))
    expected_logit, factor = e11_from_logit(logit, 51)
    assert torch.equal(model.opacity_logits.detach(), torch.full((51,), expected_logit))
    assert torch.equal(model.log_scales.detach(), torch.log(a * factor).float().expand(51, 3))


def test_u13_clamp_active_case_through_relocate() -> None:
    """U-13, E11, D-14: one target with σ_j = 0.02 and four dead primitives (n = 5): σ_new falls
    below 0.005, so every row's logit is clamped to ô_dead (live), while the scales use the
    unclamped σ_new: â = log(a_new) rounded to float32."""
    model, opt = _relocation_scene([math.log(0.02 / 0.98)] + [LOGIT_DEAD] * 4)
    sigma = float(torch.sigmoid(model.opacity_logits.detach()[0].double()))
    a = torch.exp(model.log_scales.detach()[0].double())
    relocate(model, opt, GradStats(5), RelocConfig(), generator=torch.Generator().manual_seed(0))
    assert torch.equal(model.opacity_logits.detach(), torch.full((5,), O_DEAD))
    factor = e11_oracle(sigma, 5, 1.0)[1]
    expected = torch.log(a * factor).float()
    assert torch.equal(model.log_scales.detach(), expected.expand(5, 3))


def _whitened(offsets: torch.Tensor, quat: list[float], scales: list[float]) -> torch.Tensor:
    """Offsets from a Gaussian's mean mapped to its whitened coordinates diag(1/a)·Rᵀ·x, with R
    from SciPy's quaternion convention (scalar last), independent of openftgs.model (E5)."""
    w, x, y, z = quat
    rotation = torch.tensor(scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix())
    return (offsets.double() @ rotation) / torch.tensor(scales, dtype=F64)


def assert_standard_normal(samples: torch.Tensor, bound: float = 0.01) -> None:
    """Q-54 (owner answer): every entry of the sample mean within `bound` of 0 and every entry of
    the sample covariance within `bound` of the identity, in whitened coordinates: the per-entry
    measure in units of √Cᵢᵢ and √(Cᵢᵢ·Cⱼⱼ)."""
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = centered.T @ centered / (samples.shape[0] - 1)
    assert float(mean.abs().max()) <= bound, mean
    assert float((covariance - torch.eye(3, dtype=samples.dtype)).abs().max()) <= bound, covariance


def test_u14_moved_copies_positions_follow_the_targets_gaussian() -> None:
    """U-14, E11, D-15: the positions of 10⁶ moved copies (Q-54: 10⁶ draws, owner answer) of one
    anisotropic, rotated target match the target's Gaussian μ_j + R_j·diag(a_j)·z, with a_j its
    scales before the move (Q-51), in mean and covariance within 1%."""
    n_dead = 1_000_000
    model, opt = _relocation_scene([1.5] + [LOGIT_DEAD] * n_dead)
    quat, scales = [0.9, 0.3, -0.2, 0.25], [0.5, 0.1, 0.02]
    with torch.no_grad():
        model.quats[0] = torch.tensor(quat)
        model.log_scales[0] = torch.log(torch.tensor(scales))
    mu = model.means.detach()[0].clone()
    norm = math.sqrt(sum(c * c for c in quat))
    relocate(
        model, opt, GradStats(n_dead + 1), RelocConfig(), generator=torch.Generator().manual_seed(8)
    )
    offsets = model.means.detach()[1:] - mu
    assert_standard_normal(_whitened(offsets, [c / norm for c in quat], scales))


def test_u13_render_is_nearly_unchanged_by_relocating_stacked_copies() -> None:
    """U-13, E11: a render at the targets' μt before and after relocating 30 dead primitives onto
    10 targets with copy_position = "stack" differ by PSNR ≥ 30 dB (reference backend). The dead
    primitives have σ = 0.001, below the 1/255 cull (D-32), so the first render shows the
    targets alone."""
    colors = [(0.9, 0.2, 0.1), (0.1, 0.8, 0.3), (0.2, 0.3, 0.9), (0.8, 0.8, 0.2), (0.6, 0.1, 0.7)]
    rows = []
    for k, (x, y) in enumerate((x, y) for y in (-3.0, 3.0) for x in (-6.0, -3.0, 0.0, 3.0, 6.0)):
        rows.append(
            {
                "mean": (x, y, 4.0),
                "scale": 0.5,
                "logit": -1.0 + 0.4 * k,
                "sh0": sh0_for(colors[k % 5]),
            }
        )
    rows += [{"mean": (0.0, 0.0, 4.0), "logit": LOGIT_DEAD}] * 30
    model = model_from_rows(rows)
    opt = build_optimizer(model, OptimConfig(), spatial_scale=1.0, frame_dt=0.1)
    before = render(model, CAM_A, 0.5, time_units="normalized", backend="reference").image.detach()
    report = relocate(
        model,
        opt,
        GradStats(40),
        RelocConfig(copy_position="stack"),
        generator=torch.Generator().manual_seed(0),
    )
    assert report.dead == 30
    after = render(model, CAM_A, 0.5, time_units="normalized", backend="reference").image.detach()
    psnr = -10.0 * math.log10(float(((after - before) ** 2).mean()))
    assert psnr >= 30.0, psnr


# U-25 · growth (E17, D-18, D-42 to D-45; Q-52)

S = 100.0  # spatial scale: the clone/split boundary 0.01·S is 1.0 (D-27, §1.3 Growth)


def _stats_from(g: list[float]) -> GradStats:
    """Statistics whose accumulated gradient g is exactly the given float32 values (count 1)."""
    stats = GradStats(len(g))
    n = len(g)
    stats.load_state_dict(
        {"g_sum": torch.tensor(g, dtype=F32), "count": torch.ones(n, dtype=torch.int32)}
    )
    return stats


def _growth_scene(
    log_scales: list[list[float]], *, seed: int = 0
) -> tuple[FreeTimeGaussians, FTGSOptimizer]:
    """`_relocation_scene` with live rows and the given log-scales (set after the Adam step)."""
    model, opt = _relocation_scene([1.0] * len(log_scales), seed=seed)
    with torch.no_grad():
        model.log_scales.copy_(torch.tensor(log_scales, dtype=F32))
    return model, opt


SMALL = [-3.0, -3.0, -3.0]  # largest scale e⁻³ ≤ 0.01·S: a clone
LARGE = [0.5, -1.0, -2.0]  # largest scale e^0.5 > 0.01·S: a split


def test_u25_candidates_are_exactly_the_rows_whose_g_exceeds_tau_pos() -> None:
    """U-25, §1.3 Growth step 1: the candidates are exactly the primitives with g > 0.0002
    (τ_pos): g equal to float32(0.0002) doesn't grow, the next float32 up does. Clones copy all
    nine tensors and go last in ascending parent row (exact)."""
    tau = float(torch.tensor(2e-4, dtype=F32))
    above = float(torch.nextafter(torch.tensor(tau), torch.tensor(1.0)))
    g = [tau, above, 0.0, 5e-3, 1e-4, tau]
    model, opt = _growth_scene([SMALL] * 6)
    before = _rows_of(model)
    report = grow(
        model,
        opt,
        _stats_from(g),
        GrowthConfig(),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    assert report == GrowthReport(cloned=2, split=0, sampled=0, capped=0, num_gaussians=8)
    after = _rows_of(model)
    for name in PARAM_NAMES:
        assert torch.equal(after[name][:6], before[name])
        assert torch.equal(after[name][6:], before[name][torch.tensor([1, 3])]), name


def test_u25_clone_split_boundary_is_at_one_hundredth_of_s() -> None:
    """U-25, §1.3 Growth step 3: a candidate whose largest scale is ≤ 0.01·S is cloned, a larger
    one split. With S = 100, â = 0 gives a = 1, exactly the boundary: a clone; â = 2⁻²² gives
    a = 1 + 2⁻²² in float32: a split."""
    nudge = 2.0**-22
    assert float(torch.exp(torch.tensor(nudge, dtype=F32))) == 1.0 + nudge
    model, opt = _growth_scene([[0.0, -1.0, -1.0], [-1.0, nudge, -1.0]])
    report = grow(
        model,
        opt,
        _stats_from([1.0, 1.0]),
        GrowthConfig(),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    assert (report.cloned, report.split) == (1, 1)


def test_u25_split_children_replace_their_parent() -> None:
    """U-25, E17, D-43: a split's two children have scales a/1.6 (stored as the float32 nearest
    â − ln 1.6) and copy the parent's other tensors except the position, bitwise; the parent is
    removed. Kept rows keep their order; new rows go last: clones, then first children, then
    second children, each in ascending parent row (Q-52)."""
    model, opt = _growth_scene([LARGE, SMALL, SMALL, LARGE, LARGE, SMALL])
    before = _rows_of(model)
    report = grow(
        model,
        opt,
        _stats_from([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        GrowthConfig(),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(2),
    )
    # Candidates 0, 1, 3, 5: rows 0 and 3 split, rows 1 and 5 clone; rows 2 and 4 stay.
    assert report == GrowthReport(cloned=2, split=2, sampled=0, capped=0, num_gaussians=10)
    after = _rows_of(model)
    order = [1, 2, 4, 5, 1, 5, 0, 3, 0, 3]  # kept, clones, first children, second children
    for name in PARAM_NAMES:
        if name in ("means", "log_scales"):
            continue
        assert torch.equal(after[name], before[name][torch.tensor(order)]), name
    for name in ("means", "log_scales"):
        assert torch.equal(after[name][:6], before[name][torch.tensor(order[:6])]), name
    expected = (before["log_scales"][torch.tensor([0, 3])].double() - math.log(1.6)).float()
    assert torch.equal(after["log_scales"][6:8], expected)
    assert torch.equal(after["log_scales"][8:10], expected)
    assert not torch.equal(after["means"][6:8], after["means"][8:10])


def test_u25_split_offsets_have_the_parents_covariance() -> None:
    """U-25, E17: the offsets of 10⁶ children (Q-54: 10⁶ draws, owner answer) of 5·10⁵ identical
    parents match the parent's Gaussian R(q)·diag(a)·z, a being the parent's scales, in mean and
    covariance within 1%."""
    parents = 500_000
    quat, scales = [0.7, -0.4, 0.5, 0.3], [1.6, 0.3, 0.05]  # largest > 0.01·S: splits
    model, opt = _growth_scene([[math.log(a) for a in scales]] * parents)
    with torch.no_grad():
        model.means.copy_(torch.tensor([1.0, -2.0, 0.5]).expand(parents, 3))
        model.quats.copy_(torch.tensor(quat).expand(parents, 4))
    grow(
        model,
        opt,
        _stats_from([1.0] * parents),
        GrowthConfig(),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(12),
    )
    assert model.num_gaussians == 2 * parents
    offsets = model.means.detach() - torch.tensor([1.0, -2.0, 0.5])
    norm = math.sqrt(sum(c * c for c in quat))
    assert_standard_normal(_whitened(offsets, [c / norm for c in quat], scales))


def test_u25_new_rows_have_zero_moments_and_inherit_their_parents_statistics() -> None:
    """U-25, §1.3 Growth step 4, D-45: new rows start with zero Adam moments and inherit their
    parent's g_sum and count, so the relocation later in the same event scores them; kept rows
    keep their moments and statistics, in order; step counters are kept."""
    model, opt = _growth_scene([LARGE, SMALL, SMALL, SMALL])
    moments = opt.state_dict()
    stats = GradStats(4)
    stats.load_state_dict(
        {
            "g_sum": torch.tensor([3.0, 0.0, 0.5, 1.0e-4]),
            "count": torch.tensor([2, 0, 1, 3], dtype=torch.int32),
        }
    )
    grow(
        model,
        opt,
        stats,
        GrowthConfig(),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    # Candidates: row 0 (g = 1.5, split) and row 2 (g = 0.5, clone): kept 1, 2, 3; then 2, 0, 0.
    assert torch.equal(stats.g_sum, torch.tensor([0.0, 0.5, 1.0e-4, 0.5, 3.0, 3.0]))
    assert torch.equal(stats.count, torch.tensor([0, 1, 3, 1, 2, 2], dtype=torch.int32))
    after = opt.state_dict()
    for name in after:
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(after[name][key][:3], moments[name][key][1:])
            assert torch.equal(after[name][key][3:], torch.zeros_like(after[name][key][3:]))
        assert torch.equal(after[name]["step"], moments[name]["step"])


def test_u25_with_a_full_cap_the_largest_g_grow_ties_to_the_lower_row() -> None:
    """U-25, D-44: when N plus the candidates would exceed N_max, only the candidates with the
    largest g that fit grow, ties going to the lower row index; the report counts the rest."""
    g = [3.0, 5.0, 1.0, 5.0, 3.0, 0.0]
    model, opt = _growth_scene([SMALL] * 6)
    before = _rows_of(model)
    report = grow(
        model,
        opt,
        _stats_from(g),
        GrowthConfig(n_max=9),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    # Room for 3 of 5 candidates: rows 1 and 3 (g = 5), then row 0 (g = 3, before row 4).
    assert report == GrowthReport(cloned=3, split=0, sampled=0, capped=2, num_gaussians=9)
    assert torch.equal(model.times.detach()[6:], before["times"][torch.tensor([0, 1, 3])])


def test_u25_n_never_decreases_and_a_full_model_does_not_grow() -> None:
    """U-25, §1.1, D-42: growth only adds; with N = N_max no candidate grows and nothing
    changes."""
    model, opt = _growth_scene([SMALL, LARGE, SMALL])
    before = _rows_of(model)
    report = grow(
        model,
        opt,
        _stats_from([1.0, 1.0, 1.0]),
        GrowthConfig(n_max=3),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    assert report == GrowthReport(cloned=0, split=0, sampled=0, capped=3, num_gaussians=3)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])


def test_count_control_none_changes_nothing() -> None:
    """§1.3 Count-control modes: with growth = "none", N stays as initialized; grow touches
    nothing and draws nothing."""
    model, opt = _growth_scene([LARGE, SMALL])
    before = _rows_of(model)
    gen = torch.Generator().manual_seed(1)
    state = gen.get_state()
    report = grow(
        model,
        opt,
        _stats_from([1.0, 1.0]),
        GrowthConfig(mode="none"),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=gen,
    )
    assert report == GrowthReport(cloned=0, split=0, sampled=0, capped=0, num_gaussians=2)
    assert torch.equal(gen.get_state(), state)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])


def test_count_control_score_appends_five_percent_up_to_n_max() -> None:
    """§1.3 Count-control modes: growth = "score" appends max(0, min(N_max, ⌊1.05·N⌋) − N)
    rows at each event, so the count ends at N_max and then stays."""
    model, opt = _relocation_scene([1.0] * 100)
    stats = GradStats(100)
    cfg = GrowthConfig(mode="score", n_max=108)
    gen = torch.Generator().manual_seed(3)
    sizes = []
    for _ in range(3):
        report = grow(model, opt, stats, cfg, spatial_scale=S, reloc=RelocConfig(), generator=gen)
        sizes.append((report.sampled, report.num_gaussians))
    assert sizes == [(5, 105), (3, 108), (0, 108)]
    assert model.num_gaussians == 108 and stats.num_gaussians == 108


def test_count_control_score_appends_rows_drawn_from_live_targets_by_e11() -> None:
    """§1.3 Count-control modes, E11, D-16, D-45 (Q-52): the score mode's rows are drawn ∝ ρ from
    the live set only (dead rows, here with the largest g, are never drawn), each target gets
    E11 with n = 1 + its draws and zeroed Adam moments, and each new row is its target's updated
    row, positioned from its Gaussian, with its target's statistics."""
    n_live, n_dead = 60, 40
    logits = [0.2 + 0.03 * k for k in range(n_live)] + [LOGIT_DEAD] * n_dead
    model, opt = _relocation_scene(logits)
    before = _rows_of(model)
    sums = [0.01 * (k % 7) for k in range(n_live)] + [100.0] * n_dead
    stats = _stats_with(sums, [1] * (n_live + n_dead))
    g_before = stats.mean()
    report = grow(
        model,
        opt,
        stats,
        GrowthConfig(mode="score"),
        spatial_scale=S,
        reloc=RelocConfig(),
        generator=torch.Generator().manual_seed(4),
    )
    assert report == GrowthReport(cloned=0, split=0, sampled=5, capped=0, num_gaussians=105)
    new_rows = torch.arange(100, 105)
    sources = _target_of(model, new_rows, n_live + n_dead)
    assert bool((sources < n_live).all())
    after = _rows_of(model)
    for name in PARAM_NAMES:
        if name != "means":
            assert torch.equal(after[name][new_rows], after[name][sources]), name
    assert torch.equal(stats.mean()[new_rows], g_before[sources])
    moments = opt.state_dict()
    targets, draws = torch.unique(sources, return_counts=True)
    for target, count in zip(targets.tolist(), draws.tolist(), strict=True):
        sigma = float(torch.sigmoid(before["opacity_logits"][target].double()))
        s_new, factor = e11_oracle(sigma, 1 + count, 1.0)
        expected = max(float(torch.logit(torch.tensor(s_new, dtype=F64)).float()), O_DEAD)
        assert float(after["opacity_logits"][target]) == expected
        a = torch.exp(before["log_scales"][target].double())
        assert torch.equal(after["log_scales"][target], torch.log(a * factor).float())
        for name in moments:
            for key in ("exp_avg", "exp_avg_sq"):
                zero = torch.zeros_like(moments[name][key][target])
                assert torch.equal(moments[name][key][target], zero)
    untouched = torch.ones(105, dtype=torch.bool)
    untouched[targets] = False
    untouched[100:] = False
    for name in PARAM_NAMES:
        assert torch.equal(after[name][:100][untouched[:100]], before[name][untouched[:100]])


# The "w/o periodic relocation" ablation: pruning and the opacity reset (§1.3; Q-53)

ABLATION = RelocConfig(enabled=False, prune_opacity=0.005)


def test_prune_removes_low_opacity_rows_on_the_float32_logit() -> None:
    """§1.3 ablation, Q-53: pruning removes the primitives whose base opacity is below 0.005,
    tested on the stored logit as D-14 tests the dead set: ô = float32(logit 0.005) stays, one
    float32 step below goes. The model, the optimizer and the statistics are resized together,
    kept rows in order; the count removed is returned."""
    below = float(torch.nextafter(torch.tensor(O_DEAD), torch.tensor(-math.inf)))
    model, opt = _relocation_scene([1.0, O_DEAD, below, -8.0, 0.5])
    before, moments = _rows_of(model), opt.state_dict()
    stats = _stats_with([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 1, 1, 1])
    removed = prune(model, opt, stats, ABLATION, spatial_scale=S, iteration=500)
    assert removed == 2
    kept = torch.tensor([0, 1, 4])
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name][kept])
    for name, state in opt.state_dict().items():
        assert torch.equal(state["exp_avg"], moments[name]["exp_avg"][kept])
    assert torch.equal(stats.g_sum, torch.tensor([1.0, 2.0, 5.0]))


def test_prune_removes_large_primitives_only_after_iteration_3000() -> None:
    """§1.3 ablation, Q-53: after iteration 3,000 (strictly, as gsplat's default strategy tests
    it), primitives whose largest scale exceeds 0.1·S are pruned too (here 0.1·S = 10)."""
    for iteration, expected in ((3000, 3), (3001, 2)):
        model, opt = _growth_scene([SMALL, [math.log(12.0), 0.0, 0.0], SMALL])
        removed = prune(model, opt, GradStats(3), ABLATION, spatial_scale=S, iteration=iteration)
        assert model.num_gaussians == expected and removed == 3 - expected


def test_opacity_reset_period_is_the_iteration_bound_of_pruning() -> None:
    """§1.3 ablation, Q-53's note, Q-67: the opacity reset's period, every 3,000 iterations, is
    public, and it is the bound after which large primitives are pruned too."""
    assert OPACITY_RESET_EVERY == 3000
    for iteration, expected in ((OPACITY_RESET_EVERY, 3), (OPACITY_RESET_EVERY + 1, 2)):
        model, opt = _growth_scene([SMALL, [math.log(12.0), 0.0, 0.0], SMALL])
        prune(model, opt, GradStats(3), ABLATION, spatial_scale=S, iteration=iteration)
        assert model.num_gaussians == expected


def test_prune_without_a_pruning_opacity_changes_nothing() -> None:
    """RelocConfig.prune_opacity = None (the default): no pruning, nothing changes."""
    model, opt = _growth_scene([SMALL, [math.log(12.0), 0.0, 0.0]])
    with torch.no_grad():
        model.opacity_logits[0] = -9.0
    before = _rows_of(model)
    assert prune(model, opt, GradStats(2), RelocConfig(), spatial_scale=S, iteration=5000) == 0
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])


@pytest.mark.parametrize("iteration", [0, -1, True, 1.5])
def test_prune_rejects_an_invalid_iteration(iteration: object) -> None:
    """Q-48: the iteration is an int ≥ 1, as D-41 counts it (Q-26: ValueError)."""
    model, opt = _growth_scene([SMALL])
    with pytest.raises(ValueError):
        prune(model, opt, GradStats(1), ABLATION, spatial_scale=S, iteration=iteration)  # type: ignore[arg-type]


def test_reset_opacity_clamps_every_opacity_to_at_most_one_hundredth() -> None:
    """§1.3 ablation, Q-53 (owner answer): the opacity reset sets σ to min(σ, 0.01), on the
    stored logit as float32(logit 0.01), and zeroes the Adam moments of opacity_logits for every
    row; the other tensors, their moments and every step counter are unchanged."""
    bound = float(torch.tensor(math.log(0.01) - math.log1p(-0.01), dtype=F32))
    logits = [2.0, bound, -6.0, 0.0, -4.0]
    model, opt = _relocation_scene(logits)
    before, moments = _rows_of(model), opt.state_dict()
    reset_opacity(model, opt)
    # logit 0.01 ≈ −4.595, so −4.0 is above the bound and −6.0 below it.
    expected = torch.tensor([bound, bound, -6.0, bound, bound], dtype=F32)
    assert torch.equal(model.opacity_logits.detach(), expected)
    for name in PARAM_NAMES:
        if name != "opacity_logits":
            assert torch.equal(getattr(model, name).detach(), before[name])
    after = opt.state_dict()
    for name in after:
        for key in ("exp_avg", "exp_avg_sq"):
            if name == "opacity_logits":
                assert torch.equal(after[name][key], torch.zeros_like(after[name][key]))
            else:
                assert torch.equal(after[name][key], moments[name][key])
        assert torch.equal(after[name]["step"], moments[name]["step"])


# The device paths on CUDA (GPU)


def _cuda_event(generator_device: str) -> tuple[dict[str, torch.Tensor], list[object]]:
    """A growth ("gradient"), a relocation and a growth ("score") on a CUDA model, with the
    random draws from a generator on `generator_device`; returns the final rows (on the CPU) and
    the three reports."""
    logits = [0.5 + 0.01 * k if k % 4 else LOGIT_DEAD for k in range(400)]
    model, _ = _relocation_scene(logits, seed=21)
    with torch.no_grad():
        model.log_scales[::3] = math.log(2.0)  # largest scale > 0.01·S: these candidates split
    model = model.to("cuda")
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=0.1)
    stats = GradStats(400, device="cuda")
    g = torch.rand(400, generator=torch.Generator().manual_seed(22)) * 4e-4
    stats.load_state_dict({"g_sum": g, "count": torch.ones(400, dtype=torch.int32)})
    gen = torch.Generator(device=generator_device).manual_seed(23)
    reports: list[object] = [
        grow(
            model, opt, stats, GrowthConfig(), spatial_scale=S, reloc=RelocConfig(), generator=gen
        ),
        relocate(model, opt, stats, RelocConfig(), generator=gen),
    ]
    assert bool((model.opacity_logits.detach() >= O_DEAD).all())  # every row live after it
    reports.append(
        grow(
            model,
            opt,
            stats,
            GrowthConfig(mode="score"),
            spatial_scale=S,
            reloc=RelocConfig(),
            generator=gen,
        )
    )
    assert model.means.device.type == "cuda" and stats.g_sum.device.type == "cuda"
    return {name: getattr(model, name).detach().cpu() for name in PARAM_NAMES}, reports


@pytest.mark.gpu
@pytest.mark.parametrize("generator_device", ["cuda", "cpu"])
def test_relocate_and_grow_run_on_cuda_and_repeat_with_the_same_seed(generator_device: str) -> None:
    """§1.3 Growth and Relocation on a CUDA model, with the trainer's generator on the GPU or the
    CPU (draws run on the generator's device; Q-51): N grows by the reports' counts, every row is
    live after the relocation, and the same seed gives bitwise the same model."""
    first, reports = _cuda_event(generator_device)
    second, again = _cuda_event(generator_device)
    assert reports == again
    growth, relocation, score = reports
    assert isinstance(growth, GrowthReport) and isinstance(score, GrowthReport)
    assert isinstance(relocation, RelocationReport) and relocation.dead > 0
    assert growth.cloned > 0 and growth.split > 0
    assert growth.num_gaussians == 400 + growth.cloned + growth.split  # one row per candidate
    assert score.num_gaussians == growth.num_gaussians + score.sampled
    for name in PARAM_NAMES:
        assert torch.equal(first[name], second[name]), name


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_score_e11_and_opacity_reset_make_no_host_synchronization() -> None:
    """D-49, docs/modules/density/relocation.md: sampling_score, mcmc_split (after its first call
    on the device) and reset_opacity make no host synchronization (torch's sync debug mode)."""
    model, _ = _relocation_scene([0.5 - 0.02 * k for k in range(300)], seed=31)
    model = model.to("cuda")
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=0.1)
    stats = GradStats(300, device="cuda")
    stats.load_state_dict(
        {
            "g_sum": torch.rand(300, generator=torch.Generator().manual_seed(32)),
            "count": torch.ones(300, dtype=torch.int32),
        }
    )
    gen = torch.Generator(device="cuda").manual_seed(33)
    sigma = torch.rand(64, device="cuda", dtype=F64, generator=gen) * 0.9 + 0.05
    scales = torch.rand(64, 3, device="cuda", dtype=F64, generator=gen) + 0.1
    n = torch.randint(1, 60, (64,), device="cuda", generator=gen)
    mcmc_split(sigma, scales, n)  # builds the coefficient table on the device
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        rho = sampling_score(model, stats, RelocConfig())
        new_sigma, _ = mcmc_split(sigma, scales, n)
        reset_opacity(model, opt)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert rho.shape == (300,) and new_sigma.shape == (64,)


@pytest.mark.parametrize(
    "case",
    ["stats of another size", "not a generator", "bool scale", "zero scale", "infinite scale"],
)
def test_grow_and_relocate_reject_invalid_arguments(case: str) -> None:
    """Q-48 (Q-26: ValueError): the statistics cover the model's rows, the generator is a
    torch.Generator, and S is a finite number > 0."""
    model, opt = _growth_scene([SMALL, LARGE])
    stats, gen, scale = _stats_from([1.0, 1.0]), torch.Generator().manual_seed(0), S
    if case == "stats of another size":
        stats = GradStats(3)
    elif case == "not a generator":
        gen = 0  # type: ignore[assignment]
    elif case == "bool scale":
        scale = True
    elif case == "zero scale":
        scale = 0.0
    elif case == "infinite scale":
        scale = math.inf
    with pytest.raises(ValueError):
        grow(
            model,
            opt,
            stats,
            GrowthConfig(),
            spatial_scale=scale,
            reloc=RelocConfig(),
            generator=gen,
        )
    if case in ("stats of another size", "not a generator"):
        with pytest.raises(ValueError):
            relocate(model, opt, stats, RelocConfig(), generator=gen)
    assert model.num_gaussians == 2


@pytest.mark.parametrize("operation", ["relocate", "grow", "prune", "reset_opacity"])
def test_an_optimizer_built_for_another_model_is_rejected_before_anything_changes(
    operation: str,
) -> None:
    """Q-48 (owner answer): relocate, grow, prune and reset_opacity check that the optimizer is
    the model's own (FTGSOptimizer.model), even when another model has the same rows, and raise
    ValueError before the model, either optimizer or the statistics change."""
    logits = [1.0, LOGIT_DEAD, 0.5, LOGIT_DEAD]
    model, _ = _relocation_scene(logits, seed=40)
    with torch.no_grad():
        model.log_scales[0] = math.log(20.0)  # a split candidate and a large primitive
    _, other_opt = _relocation_scene(logits, seed=41)
    before, moments = _rows_of(model), other_opt.state_dict()
    stats = _stats_from([1.0, 1.0, 1.0, 1.0])
    gen = torch.Generator().manual_seed(42)
    with pytest.raises(ValueError, match="optimizer"):
        if operation == "relocate":
            relocate(model, other_opt, stats, RelocConfig(), generator=gen)
        elif operation == "grow":
            grow(
                model,
                other_opt,
                stats,
                GrowthConfig(),
                spatial_scale=S,
                reloc=RelocConfig(),
                generator=gen,
            )
        elif operation == "prune":
            prune(model, other_opt, stats, ABLATION, spatial_scale=S, iteration=5000)
        else:
            reset_opacity(model, other_opt)
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name).detach(), before[name])
    for name, state in other_opt.state_dict().items():
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(state[key], moments[name][key])
    assert torch.equal(stats.count, torch.ones(4, dtype=torch.int32))


# draw_targets and growth_masks, the public parts of relocation and growth (Q-204).


def test_draw_targets_gives_relocates_targets_for_the_same_generator_state() -> None:
    """Q-204, D-15: the draws of draw_targets on the live rows' scores (0 on dead rows), with the
    generator seeded as relocate's, are the targets relocate gives the dead rows, in row order."""
    logits = [1.0, LOGIT_DEAD, 0.0, LOGIT_DEAD, LOGIT_DEAD, 2.0, LOGIT_DEAD, -1.0]
    model, opt = _relocation_scene(logits)
    dead = torch.tensor([v == LOGIT_DEAD for v in logits])
    scores = torch.where(dead, 0.0, sampling_score(model, GradStats(len(logits)), RelocConfig()))
    draws = draw_targets(scores, int(dead.sum()), generator=torch.Generator().manual_seed(4))
    relocate(
        model,
        opt,
        GradStats(len(logits)),
        RelocConfig(),
        generator=torch.Generator().manual_seed(4),
    )
    assert torch.equal(draws, _target_of(model, torch.nonzero(dead).squeeze(1)))


def test_draw_targets_draws_in_proportion_to_the_scores() -> None:
    """D-15: 10⁵ draws with replacement from scores (1, 2, 3, 4, 0): χ² test, p > 0.001, and the
    zero-score row is never drawn."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0])
    draws = draw_targets(scores, 100_000, generator=torch.Generator().manual_seed(2))
    observed = torch.bincount(draws, minlength=5)
    assert observed[4] == 0
    expected = scores[:4].double() / 10.0 * 100_000
    p_value = scipy.stats.chisquare(observed[:4].double().numpy(), expected.numpy()).pvalue
    assert p_value > 0.001, p_value


@pytest.mark.parametrize(
    "scores", [[0.0, 0.0], [1.0, float("nan")], [1.0, float("inf")]], ids=["zero", "nan", "inf"]
)
def test_draw_targets_rejects_scores_without_a_finite_positive_sum(scores: list[float]) -> None:
    """Q-51: the scores must be finite with a positive sum."""
    with pytest.raises(ValueError, match="sampling scores"):
        draw_targets(torch.tensor(scores), 3, generator=torch.Generator())


def test_growth_masks_with_a_full_cap_keep_the_largest_g_ties_to_the_lower_row() -> None:
    """Q-204, D-44: g = (3, 5, 1, 5, 3, 0) with τ_pos below 1, N = 6 and N_max = 9 leaves room for
    3 of the 5 candidates: rows 1 and 3 (g = 5), then row 0 (g = 3, before row 4). All small, so
    all three are clones."""
    g = torch.tensor([3.0, 5.0, 1.0, 5.0, 3.0, 0.0])
    scales = torch.full((6, 3), math.exp(-3.0))
    clone, split = growth_masks(g, scales, GrowthConfig(n_max=9), spatial_scale=S)
    assert clone.tolist() == [True, True, False, True, False, False]
    assert not split.any()


def test_growth_masks_put_the_clone_split_boundary_at_split_scale_times_s() -> None:
    """§1.3 Growth step 3: a candidate whose largest scale is at most split_scale·S (0.01 · 100 = 1)
    is cloned, one above it split; a row with g ≤ τ_pos is neither."""
    g = torch.tensor([1.0, 1.0, 1.0, 1e-5])
    scales = torch.tensor([[0.5, 1.0, 0.2], [0.5, 1.0, 0.2], [3.0, 0.1, 0.1], [9.0, 9.0, 9.0]])
    scales[1, 1] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))  # one float32 ulp above
    clone, split = growth_masks(g, scales, GrowthConfig(), spatial_scale=S)
    assert clone.tolist() == [True, False, False, False]
    assert split.tolist() == [False, True, True, False]

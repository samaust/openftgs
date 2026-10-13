# SPDX-License-Identifier: Apache-2.0
"""openftgs.density (E10, E11, E17; §1.3 Growth, Relocation and Count-control modes; D-12 to D-18,
D-42 to D-45, D-49, D-75; Q-48 to Q-55): U-13 and its hypothesis variant, U-14, U-15, U-16, U-25
and the added tests of PRD #65.

Expected values are worked by hand from the spec (the NDC scaling of E10, E11 at n = 1 and 2,
E17's factor 1.6), come from properties the rows state (1 − (1 − σ_new)^n = σ, the target's
Gaussian), from an independent float64 evaluation of E11 written literally as Python loops, or
from gsplat 1.5.3's `compute_relocation` as the oracle U-13 names (GPU).
"""

import decimal
import math

import pytest
import torch
from builders import UNIT_TIME, random_model, random_params, ring
from hypothesis import given, settings
from hypothesis import strategies as st

from openftgs.config import RelocConfig
from openftgs.density import GradStats, mcmc_split, sampling_score
from openftgs.model import FreeTimeGaussians
from openftgs.render import render

F32 = torch.float32
F64 = torch.float64


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


def test_u16_mean_is_the_average_over_visible_iterations_and_reset_zeroes() -> None:
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


def test_u16_resize_keeps_rows_in_order_and_new_rows_copy_their_parents() -> None:
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
def test_u16_gsplat_packed_and_unpacked_layouts_give_the_same_statistics() -> None:
    """D-75 (U-16 is its detecting test): the 2D means are the gradient anchor in both layouts,
    so a two-camera gsplat render backpropagated through one loss gives the same counts and,
    within float32 rounding of gsplat's atomic backward, the same g_sum."""
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
    assert error <= 1e-5 * float(torch.linalg.vector_norm(unpacked_stats.g_sum)), error


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


def test_u13_upper_clamp_applies_after_the_scale() -> None:
    """E11: σ_new is clamped to at most 1 − 2⁻²³ only after a_new is computed from the unclamped
    σ_new (so with n = 1, a stays a)."""
    new_sigma, new_scales = _split64([1.0 - 2.0**-30], [1], [3.0])
    assert float(new_sigma[0]) == 1.0 - 2.0**-23
    assert float(new_scales[0, 0]) == pytest.approx(3.0, rel=1e-10, abs=0.0)


def test_u13_n_is_clamped_to_1_and_51() -> None:
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


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.floats(min_value=1e-6, max_value=1.0 - 2.0**-23),
            st.integers(min_value=-3, max_value=60),
            st.floats(min_value=1e-3, max_value=1e3),
        ),
        min_size=1,
        max_size=8,
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
    over a grid of target opacities in [0.005, 0.999] (live targets, D-14) and n = 1 … 51.
    gsplat's outputs are unclamped (source log #55), as mcmc_split's are below 1 − 2⁻²³. The
    worst error measured on the RTX 4090 is 4.9e-4, at σ = 0.005 and n = 36: float32
    cancellation in the kernel's 1 − (1 − σ)^(1/n), which grows as σ falls below 0.005."""
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

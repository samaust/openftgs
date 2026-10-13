# SPDX-License-Identifier: Apache-2.0
"""openftgs.density (E10, E11, E17; §1.3 Growth, Relocation and Count-control modes; D-12 to D-18,
D-42 to D-45, D-49, D-75; Q-48 to Q-55): U-13 and its hypothesis variant, U-14, U-15, U-16, U-25
and the added tests of PRD #65.

Expected values are worked by hand from the spec (the NDC scaling of E10, E11 at n = 1 and 2,
E17's factor 1.6), come from properties the rows state (1 − (1 − σ_new)^n = σ, the target's
Gaussian), from an independent float64 evaluation of E11 written literally as Python loops, or
from gsplat 1.5.3's `compute_relocation` as the oracle U-13 names (GPU).
"""

import pytest
import torch
from builders import random_model, ring

from openftgs.density import GradStats
from openftgs.render import render

F32 = torch.float32


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

# SPDX-License-Identifier: Apache-2.0
"""U-18: the time normalizer and the frame interval Δt_f (E14, D-1, D-46; Q-8, Q-9).

Expected values are worked by hand from E14 and §1.4 "Time".
"""

import math
import warnings

import pytest
import torch

from openftgs.types import TimeNormalizer, frame_interval

SPANS = [(0.0, 10.0), (-3.0, 2.0), (5.0, 5.001), (1.7e3, 1.7e3 + 9.966666666666667)]


@pytest.mark.parametrize(("tau0", "tau1"), SPANS)
def test_u18_round_trip_tau_t_tau_floats(tau0: float, tau1: float) -> None:
    """U-18: τ → t → τ and t → τ → t within 1e-12 for Python floats."""
    time = TimeNormalizer(tau0, tau1)
    for k in range(11):
        tau = tau0 + (tau1 - tau0) * k / 10
        t = time.to_t(tau)
        assert isinstance(t, float)
        assert abs(time.to_tau(t) - tau) <= 1e-12
        assert abs(time.to_t(time.to_tau(k / 10)) - k / 10) <= 1e-12


@pytest.mark.parametrize(("tau0", "tau1"), SPANS)
def test_u18_round_trip_float64_tensors(tau0: float, tau1: float) -> None:
    """U-18: the round trip on float64 tensors stays float64 and within 1e-12."""
    time = TimeNormalizer(tau0, tau1)
    gen = torch.Generator().manual_seed(0)
    tau = tau0 + (tau1 - tau0) * torch.rand(1000, generator=gen, dtype=torch.float64)
    t = time.to_t(tau)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64
    back = time.to_tau(t)
    assert isinstance(back, torch.Tensor)
    assert back.dtype == torch.float64
    torch.testing.assert_close(back, tau, rtol=0.0, atol=1e-12)


def test_u18_bounds_map_to_zero_and_one_exactly() -> None:
    """E14: τ₀ ↦ 0 and τ₁ ↦ 1."""
    time = TimeNormalizer(-3.0, 2.0)
    assert time.to_t(-3.0) == 0.0
    assert time.to_t(2.0) == 1.0
    assert time.to_tau(0.0) == -3.0
    assert time.to_tau(1.0) == 2.0


def test_u18_single_frame_gives_t_zero_and_tau_tau0() -> None:
    """§1.4: with T_τ = 0, t ≡ 0 and t maps back to τ₀ (Q-9)."""
    time = TimeNormalizer(4.5, 4.5)
    assert time.to_t(4.5) == 0.0
    assert time.to_tau(0.0) == 4.5
    assert time.to_tau(0.7) == 4.5
    t = time.to_t(torch.full((3,), 4.5, dtype=torch.float64))
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64
    assert torch.equal(t, torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize("tau", [-3.0000001, 2.0000001, -100.0, 1e9])
def test_u18_to_t_warns_outside_the_training_span(tau: float) -> None:
    """U-18 (§1.4): a timestamp outside [τ₀, τ₁] gives a UserWarning (Q-9)."""
    time = TimeNormalizer(-3.0, 2.0)
    with pytest.warns(UserWarning):
        time.to_t(tau)


def test_u18_to_t_warns_for_a_tensor_with_one_element_outside() -> None:
    """U-18 (§1.4): one element outside [τ₀, τ₁] is enough for the warning (Q-9)."""
    time = TimeNormalizer(-3.0, 2.0)
    tau = torch.tensor([-3.0, 0.0, 2.0, 2.5], dtype=torch.float64)
    with pytest.warns(UserWarning):
        time.to_t(tau)


def test_u18_no_warning_at_the_bounds_or_inside() -> None:
    """U-18: [τ₀, τ₁] is closed; to_tau never warns, even outside [0, 1] (Q-9)."""
    time = TimeNormalizer(-3.0, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for tau in (-3.0, -0.5, 2.0):
            time.to_t(tau)
        time.to_t(torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float64))
        time.to_tau(-0.5)
        time.to_tau(1.5)
        time.to_tau(torch.tensor([-1.0, 2.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("tau0", "tau1"), [(1.0, 0.0), (math.nan, 1.0), (0.0, math.inf), (-math.inf, 0.0)]
)
def test_time_normalizer_rejects_invalid_bounds(tau0: float, tau1: float) -> None:
    """Q-9: bounds must be finite with τ₁ ≥ τ₀."""
    with pytest.raises(ValueError):
        TimeNormalizer(tau0, tau1)


@pytest.mark.parametrize(
    ("timestamps", "expected"),
    [
        ([0.0, 1.0, 3.0, 4.0], 1.0 / 4.0),  # gaps 1, 2, 1: median 1, T_τ = 4
        ([4.0, 0.0, 3.0, 1.0], 1.0 / 4.0),  # the same, unsorted
        ([0.0, 1.0, 3.0, 4.0, 8.0], 1.5 / 8.0),  # gaps 1, 2, 1, 4: median (1 + 2)/2, T_τ = 8
        ([0.0, 0.0, 0.0, 2.0, 4.0], 2.0 / 4.0),  # distinct 0, 2, 4: gaps 2, 2 (Q-8)
    ],
)
def test_u18_frame_interval_with_irregular_timestamps(
    timestamps: list[float], expected: float
) -> None:
    """U-18 (E14): one camera's median gap between its distinct timestamps, in normalized
    time; even counts average (Q-8)."""
    time = TimeNormalizer(min(timestamps), max(timestamps))
    assert frame_interval({"cam00": timestamps}, time) == expected


def test_u18_frame_interval_is_the_median_over_cameras() -> None:
    """E14: the median over cameras of each camera's median gap."""
    cameras = {
        "a": [float(k) for k in range(9)],  # median gap 1
        "b": [0.0, 2.0, 4.0, 6.0, 8.0],  # 2
        "c": [0.0, 4.0, 8.0],  # 4
    }
    time = TimeNormalizer(0.0, 8.0)
    assert frame_interval(cameras, time) == 2.0 / 8.0
    cameras["d"] = [0.0, 8.0]  # 8: medians 1, 2, 4, 8 average to 3
    assert frame_interval(cameras, time) == 3.0 / 8.0


def test_u18_frame_interval_unsynchronized_cameras_match_synchronized_ones() -> None:
    """U-18 (D-46): offset cameras give exactly the Δt_f of the synchronized rig.

    The offset rig drops each offset camera's last frame, so τ₀ = 0 and τ₁ = 29 stay put;
    the pooled median of its distinct timestamps would be 0.25/29, four times smaller.
    """
    synchronized = {f"cam{c:02d}": [float(k) for k in range(30)] for c in range(4)}
    unsynchronized = {"cam00": [float(k) for k in range(30)]}
    for c, offset in ((1, 0.25), (2, 0.5), (3, 0.75)):
        unsynchronized[f"cam{c:02d}"] = [k + offset for k in range(29)]
    time = TimeNormalizer(0.0, 29.0)
    assert frame_interval(synchronized, time) == 1.0 / 29.0
    assert frame_interval(unsynchronized, time) == 1.0 / 29.0


def test_u18_frame_interval_falls_back_to_pooled_distinct_timestamps() -> None:
    """E14, D-46: no camera with two images: pooled median of distinct timestamps."""
    cameras = {"a": [0.0], "b": [1.0], "c": [3.0], "d": [4.0], "e": [4.0]}
    time = TimeNormalizer(0.0, 4.0)
    # distinct 0, 1, 3, 4: gaps 1, 2, 1, median 1
    assert frame_interval(cameras, time) == 1.0 / 4.0


def test_u18_frame_interval_single_timestamp_is_one() -> None:
    """§1.4, D-46: a single timestamp gives Δt_f = 1."""
    time = TimeNormalizer(2.0, 2.0)
    assert frame_interval({"a": [2.0], "b": [2.0]}, time) == 1.0
    assert frame_interval({"a": [2.0, 2.0]}, time) == 1.0


def test_frame_interval_rejects_empty_input() -> None:
    """Q-8: an empty mapping is an error."""
    with pytest.raises(ValueError):
        frame_interval({}, TimeNormalizer(0.0, 1.0))


def test_u18_duplicated_timestamps_neither_shrink_frame_interval_nor_block_the_fallback() -> None:
    """D-46, Q-8: a camera counts only with two distinct timestamps, so duplicates don't
    shrink Δt_f, and a camera with one repeated timestamp leaves the pooled fallback in place."""
    # "a" has one distinct timestamp and doesn't count; "b" has median gap 1, T_τ = 2.
    cameras = {"a": [0.0, 0.0], "b": [0.0, 1.0, 2.0]}
    assert frame_interval(cameras, TimeNormalizer(0.0, 2.0)) == 1.0 / 2.0
    # no camera has two distinct timestamps: pooled distinct 0, 1, gap 1, T_τ = 1.
    assert frame_interval({"a": [1.0, 1.0], "b": [0.0]}, TimeNormalizer(0.0, 1.0)) == 1.0


def test_u18_to_t_warns_for_nan() -> None:
    """§1.4, Q-9: NaN is not inside [τ₀, τ₁], so it warns, as a float and in a tensor."""
    time = TimeNormalizer(-3.0, 2.0)
    with pytest.warns(UserWarning):
        time.to_t(math.nan)
    with pytest.warns(UserWarning):
        time.to_t(torch.tensor([0.0, math.nan], dtype=torch.float64))


def test_integer_tensors_convert_to_float64() -> None:
    """Q-9, D-30: integer tensors of raw timestamps or times come back as float64."""
    time = TimeNormalizer(0.0, 4.0)
    t = time.to_t(torch.tensor([0, 1, 4]))
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float64
    assert t.tolist() == [0.0, 0.25, 1.0]
    tau = time.to_tau(torch.tensor([0, 1]))
    assert isinstance(tau, torch.Tensor)
    assert tau.dtype == torch.float64
    assert tau.tolist() == [0.0, 4.0]


def test_trainer_state_carries_scene_hash_and_time_unit_with_defaults() -> None:
    """Q-17: TrainerState gains scene_sha256 (None) and time_unit ("s"), after its fields."""
    from openftgs.types import TrainerState

    state = TrainerState(
        iteration=0,
        config={},
        time=TimeNormalizer(0.0, 1.0),
        spatial_scale=1.0,
        frame_dt=0.5,
        params={},
        optim={},
        stats={},
        rng={},
        sampler={},
    )
    assert state.scene_sha256 is None
    assert state.time_unit == "s"
    time = TimeNormalizer(0.0, 1.0)
    named = TrainerState(0, {}, time, 1.0, 0.5, {}, {}, {}, {}, {}, "ab", "frame")
    assert (named.scene_sha256, named.time_unit) == ("ab", "frame")

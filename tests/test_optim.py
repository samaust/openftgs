# SPDX-License-Identifier: Apache-2.0
"""openftgs.optim (E12, E13; D-16, D-19, D-20, D-27, D-41, D-45, D-53, D-71; §2.2 ftgs-checkpoint;
Q-23, Q-25, Q-44 to Q-46): the learning-rate part of U-17, U-31 and the added tests of PRD #60.

Expected rates come from the §1.3 rate table and from E12 and E13 evaluated in float64 in this
file; expected Adam updates from Adam's update rule (Kingma and Ba) with the §1.3 settings, and for
selective Adam from D-53's description (moments decayed, no bias correction).
"""

import math
from pathlib import Path
from typing import Literal

import pytest
import torch
from builders import random_model, random_params

from openftgs.config import OptimConfig, TrainConfig, config_to_dict
from openftgs.io import load_checkpoint, save_checkpoint
from openftgs.model import FreeTimeGaussians
from openftgs.optim import FTGSOptimizer, build_optimizer
from openftgs.types import TimeNormalizer, TrainerState

S = 2.5  # spatial scale (D-27), arbitrary here
DT = 1.0 / 299.0  # Δt_f of a 300-frame sequence (§1.3)

AdamKind = Literal["fused", "foreach", "selective"]

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


def base_rates(cfg: OptimConfig) -> dict[str, tuple[float, float]]:
    """The §1.3 rate table: (rate at it = 1, rate at it = T_it) per tensor."""
    return {
        "means": (cfg.means_lr[0] * S, cfg.means_lr[1] * S),
        "velocities": (cfg.velocities_lr[0] * S / DT, cfg.velocities_lr[1] * S / DT),
        "times": cfg.times_lr,
        "log_durations": (cfg.log_durations_lr, cfg.log_durations_lr),
        "log_scales": (cfg.log_scales_lr, cfg.log_scales_lr),
        "quats": (cfg.quats_lr, cfg.quats_lr),
        "opacity_logits": (cfg.opacity_logits_lr, cfg.opacity_logits_lr),
        "sh0": (cfg.sh0_lr, cfg.sh0_lr),
        "shN": (cfg.shN_lr, cfg.shN_lr),
    }


def expected_rate(name: str, it: int, cfg: OptimConfig) -> float:
    """E12 for velocities, E13 for means and times, constants otherwise; u from D-41."""
    eta0, eta1 = base_rates(cfg)[name]
    u = (it - 1) / (cfg.iterations - 1)
    if name == "velocities":
        return float(eta0 ** (1.0 - u) * eta1**u)  # E12
    if name in ("means", "times"):
        return float(eta0 * (eta1 / eta0) ** u)  # E13
    return eta0


def assert_rates(rates: dict[str, float], expected: dict[str, float]) -> None:
    assert set(rates) == set(expected)
    for name, value in expected.items():
        assert isinstance(rates[name], float)
        assert math.isclose(rates[name], value, rel_tol=1e-12, abs_tol=0.0), name


def test_u17_rates_at_first_iteration_are_the_base_rates() -> None:
    cfg = OptimConfig()
    opt = build_optimizer(random_model(16), cfg, spatial_scale=S, frame_dt=DT)
    rates = opt.set_iteration(1)
    assert_rates(rates, {name: base_rates(cfg)[name][0] for name in PARAM_NAMES})
    # The §1.3 table's literal values at the defaults.
    assert math.isclose(rates["means"], 1.6e-4 * S, rel_tol=1e-12)
    assert math.isclose(rates["velocities"], 1.6e-4 * S / DT, rel_tol=1e-12)
    assert math.isclose(rates["times"], 1.6e-4, rel_tol=1e-12)


def test_u17_rates_at_last_iteration_are_the_end_rates() -> None:
    cfg = OptimConfig()
    opt = build_optimizer(random_model(16), cfg, spatial_scale=S, frame_dt=DT)
    rates = opt.set_iteration(cfg.iterations)
    assert_rates(rates, {name: base_rates(cfg)[name][1] for name in PARAM_NAMES})
    assert math.isclose(rates["means"], 1.6e-6 * S, rel_tol=1e-12)
    assert math.isclose(rates["velocities"], 1.6e-6 * S / DT, rel_tol=1e-12)
    assert math.isclose(rates["times"], 1.6e-6, rel_tol=1e-12)


def test_u17_rates_at_the_midpoint() -> None:
    # With an odd T_it, it = (T_it + 1)/2 is u = 0.5 exactly, where both log-linear forms give
    # the geometric mean of their base rates.
    cfg = OptimConfig(iterations=30_001)
    opt = build_optimizer(random_model(16), cfg, spatial_scale=S, frame_dt=DT)
    rates = opt.set_iteration(15_001)
    for name in ("means", "times", "velocities"):
        eta0, eta1 = base_rates(cfg)[name]
        assert math.isclose(rates[name], math.sqrt(eta0 * eta1), rel_tol=1e-12), name
    # At the default T_it the midpoint falls between iterations: it = 15,000.
    cfg = OptimConfig()
    opt = build_optimizer(random_model(16), cfg, spatial_scale=S, frame_dt=DT)
    rates = opt.set_iteration(15_000)
    assert_rates(rates, {name: expected_rate(name, 15_000, cfg) for name in PARAM_NAMES})


def test_u17_velocity_rate_at_first_iteration_is_not_the_literal_sum() -> None:
    # D-19: the printed λ₀^(1−t) + λ₁^t would give ηv,0 + 1 at u = 0.
    cfg = OptimConfig()
    opt = build_optimizer(random_model(16), cfg, spatial_scale=S, frame_dt=DT)
    eta_v0 = base_rates(cfg)["velocities"][0]
    rate = opt.set_iteration(1)["velocities"]
    assert math.isclose(rate, eta_v0, rel_tol=1e-12)
    assert not math.isclose(rate, eta_v0 + 1.0, rel_tol=1e-6)


def test_u17_rates_never_increase() -> None:
    cfg = OptimConfig()
    opt = build_optimizer(random_model(4), cfg, spatial_scale=S, frame_dt=DT)
    previous = opt.set_iteration(1)
    for it in range(2, cfg.iterations + 1):
        rates = opt.set_iteration(it)
        for name, rate in rates.items():
            assert rate <= previous[name], (name, it)
        previous = rates


def test_u17_rates_match_e12_and_e13_at_every_iteration() -> None:
    cfg = OptimConfig(iterations=1_000)
    opt = build_optimizer(random_model(4), cfg, spatial_scale=S, frame_dt=DT)
    for it in (1, 2, 17, 500, 999, 1_000):
        assert_rates(
            opt.set_iteration(it), {name: expected_rate(name, it, cfg) for name in PARAM_NAMES}
        )


@pytest.mark.parametrize("it", [0, -1, 30_001, True, 1.5, "1"])
def test_set_iteration_rejects_iterations_outside_d41(it: object) -> None:
    # D-41: it = 1 … T_it; Q-45.
    opt = build_optimizer(random_model(4), OptimConfig(), spatial_scale=S, frame_dt=DT)
    with pytest.raises(ValueError):
        opt.set_iteration(it)  # type: ignore[arg-type]


def test_one_iteration_run_uses_the_first_rates() -> None:
    # Q-45: with T_it = 1, u = 0.
    cfg = OptimConfig(iterations=1)
    opt = build_optimizer(random_model(4), cfg, spatial_scale=S, frame_dt=DT)
    assert_rates(opt.set_iteration(1), {n: base_rates(cfg)[n][0] for n in PARAM_NAMES})


@pytest.mark.parametrize(
    ("spatial_scale", "frame_dt"),
    [(0.0, DT), (-1.0, DT), (math.inf, DT), (math.nan, DT), (S, 0.0), (S, math.inf), (S, -DT)],
)
def test_build_optimizer_rejects_bad_scales(spatial_scale: float, frame_dt: float) -> None:
    with pytest.raises(ValueError):
        build_optimizer(
            random_model(4), OptimConfig(), spatial_scale=spatial_scale, frame_dt=frame_dt
        )


def single_timestamp_model(n: int = 8) -> FreeTimeGaussians:
    """A model with T_τ = 0, whose time centers and velocities are 0 (D-71)."""
    params = random_params(n, generator=torch.Generator().manual_seed(3))
    params["times"] = torch.zeros(n)
    params["velocities"] = torch.zeros(n, 3)
    return FreeTimeGaussians(params, sh_degree=3, time=TimeNormalizer(2.0, 2.0))


def set_grads(model: FreeTimeGaussians, generator: torch.Generator) -> None:
    """Seeded gradients for all nine stored tensors, none of them zero."""
    for name in PARAM_NAMES:
        p = getattr(model, name)
        g = torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device)
        p.grad = torch.where(g == 0, torch.ones_like(g), g)


def test_parameter_groups_at_the_defaults_and_at_sh_degree_0() -> None:
    # §1.3: one group per tensor; U-26: no shN group at L = 0.
    opt = build_optimizer(random_model(4), OptimConfig(), spatial_scale=S, frame_dt=DT)
    assert set(opt.set_iteration(1)) == set(PARAM_NAMES)
    opt = build_optimizer(random_model(4, sh_degree=0), OptimConfig(), spatial_scale=S, frame_dt=DT)
    assert set(opt.set_iteration(1)) == set(PARAM_NAMES) - {"shN"}


@pytest.mark.parametrize("adam", ["fused", "foreach"])
def test_d71_single_timestamp_freezes_times_and_velocities(adam: AdamKind) -> None:
    # D-71: with T_τ = 0, times and velocities take no updates and stay 0 (Q-45: no group).
    model = single_timestamp_model()
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=1.0, adam=adam)
    assert set(opt.set_iteration(1)) == set(PARAM_NAMES) - {"times", "velocities"}
    generator = torch.Generator().manual_seed(5)
    before = {name: getattr(model, name).detach().clone() for name in PARAM_NAMES}
    for it in range(1, 6):
        set_grads(model, generator)
        opt.set_iteration(it)
        opt.step()
    assert torch.equal(model.times, torch.zeros_like(model.times))
    assert torch.equal(model.velocities, torch.zeros_like(model.velocities))
    for name in set(PARAM_NAMES) - {"times", "velocities"}:
        assert not torch.equal(getattr(model, name), before[name]), name


@pytest.mark.parametrize("adam", ["fused", "foreach"])
def test_first_step_moves_each_tensor_at_its_scheduled_rate(adam: AdamKind) -> None:
    # Adam's first step from zero moments, with bias correction, is η·g/(|g| + ε) per element
    # (§1.3: β₁ = 0.9, β₂ = 0.999, ε = 1e-15), η being the rate of it = 1.
    model = random_model(8)
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=DT, adam=adam)
    rates = opt.set_iteration(1)
    set_grads(model, torch.Generator().manual_seed(1))
    before = {name: getattr(model, name).detach().double() for name in PARAM_NAMES}
    grads = {name: getattr(model, name).grad.double() for name in PARAM_NAMES}
    opt.step()
    for name in PARAM_NAMES:
        g = grads[name]
        expected = before[name] - rates[name] * g / (g.abs() + 1e-15)
        torch.testing.assert_close(
            getattr(model, name).detach().double(), expected, rtol=1e-6, atol=1e-7 * rates[name]
        )


def test_zero_grad_clears_all_nine_tensors() -> None:
    # Q-45: tensors without a group (here times and velocities, D-71) are cleared too.
    model = single_timestamp_model()
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=1.0)
    set_grads(model, torch.Generator().manual_seed(2))
    opt.zero_grad()
    for name in PARAM_NAMES:
        assert getattr(model, name).grad is None, name


def run_steps(
    model_device: str, adam: AdamKind, steps: int, *, visibility: torch.Tensor | None = None
) -> tuple[dict[str, torch.Tensor], object]:
    """`steps` Adam steps of a seeded random model on seeded gradients, at each iteration's
    rates; returns the final stored tensors (float64, CPU) and the optimizer."""
    model = random_model(64).to(model_device)
    opt = build_optimizer(
        model, OptimConfig(iterations=steps), spatial_scale=S, frame_dt=DT, adam=adam
    )
    generator = torch.Generator().manual_seed(11)
    for it in range(1, steps + 1):
        set_grads(model, generator)
        opt.set_iteration(it)
        opt.step(visibility=visibility)
    return {name: getattr(model, name).detach().double().cpu() for name in PARAM_NAMES}, opt


def as_float32(x: float) -> float:
    """x rounded to the nearest float32, as a Python float."""
    return float(torch.tensor(x, dtype=torch.float32))


def assert_relative(actual: torch.Tensor, expected: torch.Tensor, rel: float) -> None:
    """U-31's "relative" (Q-47): |actual − expected| ≤ rel·(|expected| + max|expected|)
    elementwise, so elements near 0 are measured against the tensor's scale."""
    floor = rel * float(expected.abs().max()) if expected.numel() else 0.0
    torch.testing.assert_close(actual, expected, rtol=rel, atol=floor)


def test_u31_fused_and_foreach_agree_over_100_steps_cpu() -> None:
    fused, _ = run_steps("cpu", "fused", 100)
    foreach, _ = run_steps("cpu", "foreach", 100)
    for name in PARAM_NAMES:
        assert_relative(fused[name], foreach[name], 1e-6)


@pytest.mark.gpu
def test_u31_fused_and_foreach_agree_over_100_steps_cuda() -> None:
    fused, _ = run_steps("cuda", "fused", 100)
    foreach, _ = run_steps("cuda", "foreach", 100)
    for name in PARAM_NAMES:
        assert_relative(fused[name], foreach[name], 1e-6)


def stepped(
    adam: AdamKind = "fused", n: int = 8, steps: int = 3, device: str = "cpu"
) -> tuple[FreeTimeGaussians, FTGSOptimizer]:
    """A model and its optimizer after `steps` steps on seeded gradients, so moments are set."""
    model = random_model(n).to(device)
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=DT, adam=adam)
    generator = torch.Generator().manual_seed(7)
    for it in range(1, steps + 1):
        set_grads(model, generator)
        opt.set_iteration(it)
        opt.step()
    return model, opt


def test_reset_rows_zeroes_moments_of_listed_rows_and_keeps_steps() -> None:
    # D-16: zero exp_avg and exp_avg_sq of the rows; keep each step counter. Q-44: int64 indices
    # in any order, repeats allowed (relocation draws targets with replacement).
    _, opt = stepped()
    before = opt.state_dict()
    rows = torch.tensor([5, 1, 5, 2])
    opt.reset_rows(rows)
    after = opt.state_dict()
    reset = torch.zeros(8, dtype=torch.bool)
    reset[rows] = True
    for name in PARAM_NAMES:
        for key in ("exp_avg", "exp_avg_sq"):
            assert not torch.equal(
                before[name][key][reset], torch.zeros_like(before[name][key][reset])
            )
            assert torch.equal(after[name][key][reset], torch.zeros_like(after[name][key][reset]))
            assert torch.equal(after[name][key][~reset], before[name][key][~reset]), (name, key)
        assert torch.equal(after[name]["step"], before[name]["step"])
        assert float(after[name]["step"]) == 3.0


@pytest.mark.parametrize(
    "rows",
    [
        torch.zeros(8, dtype=torch.bool),
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([[1]]),
    ],
)
def test_reset_rows_rejects_masks_and_other_selectors(rows: torch.Tensor) -> None:
    _, opt = stepped()
    with pytest.raises(ValueError):
        opt.reset_rows(rows)


def test_resize_keeps_moments_in_order_and_appends_zero_rows() -> None:
    # §1.3 Growth step 4, D-45: kept rows keep their order, new rows go last with zero moments;
    # step counters are kept. Q-23: keep is a bool mask; Q-44: called after the model's resize_.
    model, opt = stepped()
    before = opt.state_dict()
    keep = torch.tensor([True, False, True, True, False, True, True, True])
    append = {name: getattr(model, name).detach()[:3].clone() for name in PARAM_NAMES}
    model.resize_(keep=keep, append=append)
    opt.resize_(keep=keep, n_append=3)
    after = opt.state_dict()
    for name in PARAM_NAMES:
        for key in ("exp_avg", "exp_avg_sq"):
            assert after[name][key].shape == getattr(model, name).shape
            assert torch.equal(after[name][key][:6], before[name][key][keep]), (name, key)
            assert torch.equal(after[name][key][6:], torch.zeros_like(after[name][key][6:]))
        assert torch.equal(after[name]["step"], before[name]["step"])
    # Steps after the resize update the model's new tensors.
    old = {name: getattr(model, name).detach().clone() for name in PARAM_NAMES}
    set_grads(model, torch.Generator().manual_seed(9))
    opt.set_iteration(4)
    opt.step()
    for name in PARAM_NAMES:
        assert not torch.equal(getattr(model, name), old[name]), name
    assert float(opt.state_dict()["means"]["step"]) == 4.0


def test_resize_before_the_first_step() -> None:
    # Q-45: state exists from build, so growth can resize before any step.
    model = random_model(4)
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=DT)
    append = {name: getattr(model, name).detach()[:2].clone() for name in PARAM_NAMES}
    model.resize_(append=append)
    opt.resize_(n_append=2)
    state = opt.state_dict()
    assert state["means"]["exp_avg"].shape == (6, 3)
    assert float(state["means"]["step"]) == 0.0


def test_resize_rejects_a_mismatch_with_the_model_and_changes_nothing() -> None:
    model, opt = stepped()
    before = opt.state_dict()
    append = {name: getattr(model, name).detach()[:3].clone() for name in PARAM_NAMES}
    model.resize_(append=append)
    with pytest.raises(ValueError):
        opt.resize_(n_append=2)  # the model gained 3 rows
    with pytest.raises(ValueError):
        opt.resize_(keep=torch.ones(8, dtype=torch.int64), n_append=3)  # not a bool mask
    with pytest.raises(ValueError):
        opt.resize_(n_append=-1)
    after = opt.state_dict()
    for name in PARAM_NAMES:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(after[name][key], before[name][key])


def train_to(model: FreeTimeGaussians, opt: FTGSOptimizer, start: int, stop: int) -> None:
    """Iterations start … stop, each with gradients seeded by its iteration index."""
    for it in range(start, stop + 1):
        set_grads(model, torch.Generator().manual_seed(1_000 + it))
        opt.set_iteration(it)
        opt.step()
        opt.zero_grad()


def checkpoint_state(model: FreeTimeGaussians, opt: FTGSOptimizer, it: int) -> TrainerState:
    n = model.num_gaussians
    return TrainerState(
        iteration=it,
        config=config_to_dict(TrainConfig()),
        time=model.time,
        spatial_scale=S,
        frame_dt=DT,
        params={name: getattr(model, name).detach().clone() for name in PARAM_NAMES},
        optim=opt.state_dict(),
        stats={"g_sum": torch.zeros(n), "count": torch.zeros(n, dtype=torch.int64)},
        rng={},
        sampler={},
    )


@pytest.mark.parametrize(
    ("saved", "resumed"),
    [("fused", "fused"), ("foreach", "foreach"), ("fused", "foreach"), ("foreach", "fused")],
)
def test_checkpoint_round_trip_resumes_bitwise(
    tmp_path: Path, saved: AdamKind, resumed: AdamKind
) -> None:
    # §2.2 contract: resuming at k and training to m gives the parameters of an uninterrupted
    # run, bitwise on CPU. Q-46: state saved under one Adam kind loads under another (the kinds
    # agree to rounding, U-31, so across kinds only the saved state is compared).
    k, m = 5, 9
    straight = random_model(16)
    opt = build_optimizer(straight, OptimConfig(), spatial_scale=S, frame_dt=DT, adam=resumed)
    train_to(straight, opt, 1, m)

    first = random_model(16)
    opt = build_optimizer(first, OptimConfig(), spatial_scale=S, frame_dt=DT, adam=saved)
    train_to(first, opt, 1, k)
    saved_state = opt.state_dict()
    save_checkpoint(checkpoint_state(first, opt, k), tmp_path / "ckpt")
    state = load_checkpoint(tmp_path / "ckpt")
    model = FreeTimeGaussians(state.params, sh_degree=3, time=state.time)
    resumed_opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=DT, adam=resumed)
    resumed_opt.load_state_dict(state.optim)
    reloaded = resumed_opt.state_dict()
    for name in PARAM_NAMES:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(reloaded[name][key], saved_state[name][key]), (name, key)
        assert reloaded[name]["lr"] == saved_state[name]["lr"]
    train_to(model, resumed_opt, k + 1, m)
    if saved == resumed:
        for name in PARAM_NAMES:
            assert torch.equal(getattr(model, name), getattr(straight, name)), name


def test_state_dict_layout_matches_the_checkpoint_files() -> None:
    # §2.2: optim.safetensors holds exp_avg, exp_avg_sq and step per tensor; optim.json the base
    # rates, schedule parameters, β₁, β₂ and ε (Q-46 names the keys).
    cfg = OptimConfig()
    _, opt = stepped()
    state = opt.state_dict()
    assert set(state) == set(PARAM_NAMES)
    rates = base_rates(cfg)
    schedules = {"means": "exponential", "times": "exponential", "velocities": "annealed"}
    for name in PARAM_NAMES:
        entry = state[name]
        assert set(entry) == {
            "exp_avg",
            "exp_avg_sq",
            "step",
            "lr",
            "schedule",
            "iterations",
            "betas",
            "eps",
        }
        assert entry["step"].shape == () and entry["step"].dtype == torch.float32
        assert entry["lr"] == pytest.approx(list(rates[name]), rel=1e-15)
        assert entry["schedule"] == schedules.get(name, "constant")
        assert entry["iterations"] == cfg.iterations
        assert entry["betas"] == [0.9, 0.999] and entry["eps"] == 1e-15


def test_load_state_dict_rejects_a_different_optimizer_and_changes_nothing() -> None:
    _, opt = stepped()
    good = opt.state_dict()
    other = build_optimizer(
        random_model(8), OptimConfig(), spatial_scale=2 * S, frame_dt=DT
    )  # other base rates for means and velocities
    with pytest.raises(ValueError):
        other.load_state_dict(good)
    missing = {name: v for name, v in good.items() if name != "shN"}
    with pytest.raises(ValueError):
        opt.load_state_dict(missing)
    wrong_shape = dict(good, means=dict(good["means"], exp_avg=torch.zeros(9, 3)))
    with pytest.raises(ValueError):
        opt.load_state_dict(wrong_shape)
    wrong_eps = dict(good, quats=dict(good["quats"], eps=1e-8))
    with pytest.raises(ValueError):
        opt.load_state_dict(wrong_eps)
    assert all(
        torch.equal(opt.state_dict()[n][k], good[n][k])
        for n in PARAM_NAMES
        for k in ("exp_avg", "exp_avg_sq", "step")
    )


@pytest.mark.gpu
def test_u31_selective_adam_leaves_invisible_rows_and_moments_unchanged() -> None:
    # D-53: selective Adam updates only the visible rows; invisible rows and their moments are
    # unchanged bitwise.
    n = 64
    visibility = torch.arange(n, device="cuda") % 3 != 0
    model = random_model(n).to("cuda")
    opt = build_optimizer(
        model, OptimConfig(iterations=100), spatial_scale=S, frame_dt=DT, adam="selective"
    )
    _, warm = stepped("foreach", n=n, steps=1, device="cuda")  # nonzero moments to start from
    opt.load_state_dict(
        {name: dict(entry, iterations=100) for name, entry in warm.state_dict().items()}
    )
    before_state = opt.state_dict()
    before = {name: getattr(model, name).detach().clone() for name in PARAM_NAMES}
    generator = torch.Generator().manual_seed(13)
    for it in range(1, 101):
        set_grads(model, generator)
        opt.set_iteration(it)
        opt.step(visibility=visibility)
    after_state = opt.state_dict()
    hidden = ~visibility
    for name in PARAM_NAMES:
        assert torch.equal(getattr(model, name)[hidden], before[name][hidden]), name
        assert not torch.equal(getattr(model, name)[visibility], before[name][visibility]), name
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(after_state[name][key][hidden], before_state[name][key][hidden])
        # Q-45: the step counter counts the steps taken.
        assert float(after_state[name]["step"]) == float(before_state[name]["step"]) + 100


@pytest.mark.gpu
def test_u31_selective_adam_with_all_rows_visible_is_adam_without_bias_correction() -> None:
    # D-53: with every row visible, selective Adam is Adam without bias correction:
    # m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², p ← p − η·m/(√v + ε), here in float64.
    # Q-47: the reference takes β₁, β₂, ε and η as the float32 values gsplat's kernel receives;
    # float32(0.999) alone moves 1 − β₂ by about 1.3e-5 relative, more than U-31's tolerance.
    n, steps = 64, 100
    cfg = OptimConfig(iterations=steps)
    model = random_model(n).to("cuda")
    opt = build_optimizer(model, cfg, spatial_scale=S, frame_dt=DT, adam="selective")
    beta1, beta2, eps = (as_float32(x) for x in (*cfg.betas, cfg.eps))
    p64 = {name: getattr(model, name).detach().double().cpu() for name in PARAM_NAMES}
    m = {name: torch.zeros_like(p) for name, p in p64.items()}
    v = {name: torch.zeros_like(p) for name, p in p64.items()}
    visibility = torch.ones(n, dtype=torch.bool, device="cuda")
    generator = torch.Generator().manual_seed(17)
    for it in range(1, steps + 1):
        set_grads(model, generator)
        rates = opt.set_iteration(it)
        for name in PARAM_NAMES:
            g = getattr(model, name).grad.double().cpu()
            m[name] = beta1 * m[name] + (1 - beta1) * g
            v[name] = beta2 * v[name] + (1 - beta2) * g * g
            eta = as_float32(rates[name])
            p64[name] = p64[name] - eta * m[name] / (v[name].sqrt() + eps)
        opt.step(visibility=visibility)
    for name in PARAM_NAMES:
        assert_relative(getattr(model, name).detach().double().cpu(), p64[name], 1e-6)


def test_selective_adam_needs_cuda() -> None:
    with pytest.raises(ValueError):
        build_optimizer(
            random_model(4), OptimConfig(), spatial_scale=S, frame_dt=DT, adam="selective"
        )


def test_visibility_is_only_for_selective_adam() -> None:
    # Q-44: never silently ignored.
    _, opt = stepped()
    with pytest.raises(ValueError):
        opt.step(visibility=torch.ones(8, dtype=torch.bool))


@pytest.mark.gpu
def test_selective_adam_requires_a_bool_visibility_mask() -> None:
    model = random_model(8).to("cuda")
    opt = build_optimizer(model, OptimConfig(), spatial_scale=S, frame_dt=DT, adam="selective")
    set_grads(model, torch.Generator().manual_seed(1))
    for visibility in (
        None,
        torch.ones(8, device="cuda"),
        torch.ones(7, dtype=torch.bool, device="cuda"),
        torch.ones(8, dtype=torch.bool),
    ):
        with pytest.raises(ValueError):
            opt.step(visibility=visibility)


def test_build_optimizer_rejects_an_unknown_adam_kind() -> None:
    with pytest.raises(ValueError):
        build_optimizer(random_model(4), OptimConfig(), spatial_scale=S, frame_dt=DT, adam="sgd")  # type: ignore[arg-type]

# SPDX-License-Identifier: Apache-2.0
"""openftgs.optim (E12, E13; D-16, D-19, D-20, D-27, D-41, D-45, D-53, D-71; §2.2 ftgs-checkpoint;
Q-23, Q-25, Q-44 to Q-46): the learning-rate part of U-17, U-31 and the added tests of PRD #60.

Expected rates come from the §1.3 rate table and from E12 and E13 evaluated in float64 in this
file; expected Adam updates from Adam's update rule (Kingma and Ba) with the §1.3 settings, and for
selective Adam from D-53's description (moments decayed, no bias correction).
"""

import math
from typing import Literal

import pytest
import torch
from builders import random_model, random_params

from openftgs.config import OptimConfig
from openftgs.model import FreeTimeGaussians
from openftgs.optim import build_optimizer
from openftgs.types import TimeNormalizer

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


def assert_relative(actual: torch.Tensor, expected: torch.Tensor, rel: float) -> None:
    """|actual − expected| ≤ rel·|expected| elementwise, with a floor of rel·max|expected| for
    elements near 0."""
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

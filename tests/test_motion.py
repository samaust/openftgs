# SPDX-License-Identifier: Apache-2.0
"""U-1, U-2, U-3 and the per-row E8 terms: motion, temporal and effective opacity (E1, E3, E4,
E8, D-2, D-32; Q-12, Q-13).

Expected values are worked from the equations: E1's Jacobians, E4's values at μt, μt ± s and
μt ± 2s and its derivatives, E3's product and E8's single nonzero gradient. Criteria below
float32 precision run in float64, as §3.1 does for gradient checks; finiteness runs in float32,
the training dtype (D-52).
"""

import math
import warnings

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from torch import Tensor

import openftgs.motion
from openftgs.motion import (
    TemporalSlice,
    effective_opacity,
    moved_means,
    reg4d_terms,
    temporal_opacity,
    temporal_slice,
)

F64 = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _model(n: int, seed: int, dtype: torch.dtype = F64) -> dict[str, Tensor]:
    g = _gen(seed)
    return {
        "means": torch.randn(n, 3, generator=g, dtype=dtype),
        "times": torch.rand(n, generator=g, dtype=dtype),
        "log_durations": torch.empty(n, dtype=dtype).uniform_(-4.0, 0.0, generator=g),
        "velocities": torch.randn(n, 3, generator=g, dtype=dtype),
        "log_scales": torch.randn(n, 3, generator=g, dtype=dtype) - 3.0,
        "quats": torch.randn(n, 4, generator=g, dtype=dtype),
        "opacity_logits": torch.randn(n, generator=g, dtype=dtype),
    }


def _slice(m: dict[str, Tensor], t: float | Tensor, impl: str = "eager") -> TemporalSlice:
    return temporal_slice(
        m["means"],
        m["times"],
        m["log_durations"],
        m["velocities"],
        m["log_scales"],
        m["quats"],
        m["opacity_logits"],
        t,
        impl=impl,  # type: ignore[arg-type]
    )


def test_all_lists_the_public_api() -> None:
    """Q-12, Q-27: `__all__` is exactly the public API."""
    assert sorted(openftgs.motion.__all__) == [
        "TemporalSlice",
        "durations_to_normalized",
        "durations_to_raw",
        "effective_opacity",
        "moved_means",
        "reg4d_terms",
        "temporal_opacity",
        "temporal_slice",
        "velocities_to_normalized",
        "velocities_to_raw",
    ]


# U-1 · E1 motion


def test_u1_position_at_own_time_center_is_exact() -> None:
    """U-1: μx(μt) = μx exactly, in float32 and float64, from both entry points."""
    for dtype in (torch.float32, F64):
        m = _model(1000, 0, dtype)
        m["times"] = torch.full((1000,), 0.37, dtype=dtype)
        assert torch.equal(moved_means(m["means"], m["velocities"], m["times"], 0.37), m["means"])
        assert torch.equal(_slice(m, 0.37).means_t, m["means"])


@pytest.mark.parametrize("entry", ["function", "slice"])
def test_u1_displacement_is_velocity_times_elapsed_time(entry: str) -> None:
    """U-1: μx(t₁) − μx(t₀) = v(t₁ − t₀), rtol 1e-6; also outside [0, 1] (no clamping)."""
    m = _model(500, 1)

    def at(t: float) -> Tensor:
        if entry == "slice":
            return _slice(m, t).means_t
        return moved_means(m["means"], m["velocities"], m["times"], t)

    for t0, t1 in [(0.1, 0.9), (-2.0, 3.5), (0.3, 0.31)]:
        d = at(t1) - at(t0)
        torch.testing.assert_close(d, m["velocities"] * (t1 - t0), rtol=1e-6, atol=0.0)


def test_u1_jacobians_match_closed_form() -> None:
    """U-1: ∂μx(t)/∂μx = I, ∂μx(t)/∂v = (t − μt)I, ∂μx(t)/∂μt = −v (per row; rows independent)."""
    n, t = 4, 0.8
    m = _model(n, 2)
    means, v, mu_t = m["means"], m["velocities"], m["times"]
    jac = torch.autograd.functional.jacobian(  # type: ignore[no-untyped-call]
        lambda a, b, c: moved_means(a, b, c, t), (means, v, mu_t)
    )
    eye_n, eye_3 = torch.eye(n, dtype=F64), torch.eye(3, dtype=F64)
    j_means = torch.einsum("ij,ab->iajb", eye_n, eye_3)
    j_v = torch.einsum("ij,ab->iajb", eye_n * (t - mu_t)[:, None], eye_3)
    j_mu = torch.einsum("ij,ia->iaj", eye_n, -v)
    torch.testing.assert_close(jac[0], j_means, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(jac[1], j_v, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(jac[2], j_mu, rtol=1e-6, atol=0.0)


def test_u1_gradcheck() -> None:
    """U-1: gradcheck of E1 in float64."""
    m = {k: v.requires_grad_() for k, v in _model(5, 3).items()}
    assert torch.autograd.gradcheck(
        lambda a, b, c: moved_means(a, b, c, 0.3), (m["means"], m["velocities"], m["times"])
    )


@settings(deadline=None, max_examples=50)
@given(
    n=st.integers(1, 64),
    seed=st.integers(0, 2**31 - 1),
    t0=st.floats(-2.0, 3.0),
    t1=st.floats(-2.0, 3.0),
    scale=st.floats(1e-3, 1e3),
)
def test_u1_property_random_shapes_and_values(
    n: int, seed: int, t0: float, t1: float, scale: float
) -> None:
    """U-1 (hypothesis): the identities hold for random N, positions, velocities and times."""
    # Keeps U-1's pure rtol by drawing elapsed times well above the float64 rounding of
    # the positions, instead of adding an absolute floor to the criterion.
    assume(abs(t1 - t0) >= 1e-3)
    m = _model(n, seed)
    means, v, mu_t = m["means"] * scale, m["velocities"] * scale, m["times"] * 3.0 - 1.0
    at_center = torch.stack(
        [
            moved_means(means[i : i + 1], v[i : i + 1], mu_t[i : i + 1], float(mu_t[i]))
            for i in range(n)
        ]
    )
    assert torch.equal(at_center.reshape(n, 3), means)
    d = moved_means(means, v, mu_t, t1) - moved_means(means, v, mu_t, t0)
    torch.testing.assert_close(d, v * (t1 - t0), rtol=1e-6, atol=0.0)


# U-2 · E4 temporal opacity


@pytest.mark.parametrize("entry", ["function", "slice"])
def test_u2_values_at_center_one_and_two_durations_float32_exact_inputs(entry: str) -> None:
    """U-2: σt = 1, e^−0.5, e^−2 at μt, μt ± s, μt ± 2s to 1e-7, in float32 with ŝ = 0 and
    μt = 0.5, where every input and difference is exact. Through the slice, ô = 30 makes
    σ = 1 exactly in float32, so σ_eff = σt."""
    mu = torch.full((5,), 0.5)
    s_hat = torch.zeros(5)
    m = _model(5, 16, torch.float32)
    m.update(times=mu, log_durations=s_hat, opacity_logits=torch.full((5,), 30.0))
    ts = [0.5, 1.5, -0.5, 2.5, -1.5]
    want = [1.0, math.exp(-0.5), math.exp(-0.5), math.exp(-2.0), math.exp(-2.0)]
    for t, w in zip(ts, want, strict=True):
        if entry == "slice":
            got = _slice(m, t).opacities_t
        else:
            got = temporal_opacity(mu, s_hat, t)
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, torch.full((5,), w), rtol=0.0, atol=1e-7)


def test_u2_values_at_center_one_and_two_durations_float64_random() -> None:
    """U-2: the same values to 1e-7 at random μt and ŝ, in float64."""
    g = _gen(4)
    mu = torch.rand(200, generator=g, dtype=F64)
    s_hat = torch.empty(200, dtype=F64).uniform_(-6.0, 1.0, generator=g)
    s = torch.exp(s_hat)
    for k, w in [
        (0, 1.0),
        (1, math.exp(-0.5)),
        (-1, math.exp(-0.5)),
        (2, math.exp(-2.0)),
        (-2, math.exp(-2.0)),
    ]:
        got = torch.cat(
            [
                temporal_opacity(mu[i : i + 1], s_hat[i : i + 1], float(mu[i] + k * s[i]))
                for i in range(200)
            ]
        )
        torch.testing.assert_close(got, torch.full_like(got, w), rtol=0.0, atol=1e-7)


def test_u2_symmetric_about_time_center() -> None:
    """U-2: σt(μt + Δ) = σt(μt − Δ)."""
    mu = torch.full((100,), 0.25, dtype=F64)
    s_hat = torch.linspace(-5.0, 1.0, 100, dtype=F64)
    for delta in [0.001, 0.01, 0.1, 0.5]:
        a = temporal_opacity(mu, s_hat, 0.25 + delta)
        b = temporal_opacity(mu, s_hat, 0.25 - delta)
        torch.testing.assert_close(a, b, rtol=0.0, atol=1e-7)


def test_u2_decays_monotonically_away_from_center() -> None:
    """U-2: σt does not increase as |t − μt| grows, on both sides; it never exceeds 1."""
    mu = torch.tensor([0.5], dtype=F64)
    s_hat = torch.tensor([math.log(0.1)], dtype=F64)
    deltas = torch.linspace(0.0, 2.0, 2001, dtype=F64)
    for sign in (1.0, -1.0):
        vals = torch.cat([temporal_opacity(mu, s_hat, float(0.5 + sign * d)) for d in deltas])
        assert bool((vals[1:] <= vals[:-1]).all())
        assert bool((vals <= 1.0).all()) and float(vals[0]) == 1.0


def test_e4_gradients_match_stated_derivatives() -> None:
    """E4: ∂σt/∂μt = σt·(t − μt)/s² and ∂σt/∂ŝ = σt·((t − μt)/s)²."""
    m = _model(50, 5)
    mu = m["times"].requires_grad_()
    s_hat = m["log_durations"].requires_grad_()
    t = 0.6
    sig = temporal_opacity(mu, s_hat, t)
    g_mu, g_s = torch.autograd.grad(sig.sum(), (mu, s_hat))
    s = torch.exp(s_hat.detach())
    d = t - mu.detach()
    torch.testing.assert_close(g_mu, sig.detach() * d / s**2, rtol=1e-7, atol=1e-12)
    torch.testing.assert_close(g_s, sig.detach() * (d / s) ** 2, rtol=1e-7, atol=1e-12)


def test_u2_gradcheck() -> None:
    """U-2: gradcheck of E4 in float64."""
    m = _model(6, 6)
    mu = m["times"].requires_grad_()
    s_hat = m["log_durations"].requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: temporal_opacity(a, b, 0.4), (mu, s_hat))


@pytest.mark.parametrize(
    ("s_hat", "z"),
    [(-30.0, 0.5), (-30.0, 0.0), (0.0, 1e4), (-30.0, 1e4), (-30.0, -1e4), (-30.0, 1e15)],
    ids=["s-30", "s-30-center", "z1e4", "s-30-z1e4", "s-30-z-1e4", "s-30-z1e15"],
)
def test_u2_finite_at_tiny_duration_and_far_from_center_float32(s_hat: float, z: float) -> None:
    """U-2: at ŝ = −30 and |t − μt|/s = 10⁴ (alone and together), outputs and gradients with
    respect to μt, ŝ and ô are finite with no NaN, in float32, through E4, E3, E8 and the slice."""
    # μt = 0, so t = z·s is representable in float32 even when s = e^−30.
    t = z * math.exp(s_hat)
    mu = torch.zeros(3, requires_grad=True)
    sh = torch.full((3,), s_hat, requires_grad=True)
    o = torch.tensor([-2.0, 0.0, 4.0], requires_grad=True)
    outs = [temporal_opacity(mu, sh, t), effective_opacity(o, mu, sh, t), reg4d_terms(o, mu, sh, t)]
    for out in outs:
        assert bool(torch.isfinite(out).all())
        grads = torch.autograd.grad(out.sum(), (mu, sh, o), allow_unused=True)
        for g in grads:
            assert g is None or bool(torch.isfinite(g).all())
    m = _model(3, 7, torch.float32)
    m.update(times=mu, log_durations=sh, opacity_logits=o)
    sl = _slice(m, t)
    total = sum(x.sum() for x in (sl.means_t, sl.opacities_t, sl.opacities, sl.reg_terms))
    assert isinstance(total, Tensor)
    for g in torch.autograd.grad(total, (mu, sh, o)):
        assert bool(torch.isfinite(g).all())
    if abs(z) >= 1e4:
        assert bool((temporal_opacity(mu, sh, t) == 0.0).all())


@settings(deadline=None, max_examples=100)
@given(
    n=st.integers(1, 64),
    seed=st.integers(0, 2**31 - 1),
    t=st.floats(-2.0, 3.0),
)
def test_u2_property_random_values(n: int, seed: int, t: float) -> None:
    """U-2 (hypothesis): for random N, μt ∈ [−1, 2] and ŝ ∈ [−30, 5]: σt ∈ [0, 1], 1 at t = μt,
    symmetric about μt, finite gradients."""
    g = _gen(seed)
    mu64 = torch.empty(n, dtype=F64).uniform_(-1.0, 2.0, generator=g)
    s64 = torch.empty(n, dtype=F64).uniform_(-30.0, 5.0, generator=g)
    for dtype in (torch.float32, F64):
        m = mu64.to(dtype).requires_grad_()
        s = s64.to(dtype).requires_grad_()
        sig = temporal_opacity(m, s, t)
        assert bool(((sig >= 0.0) & (sig <= 1.0)).all())
        for grad in torch.autograd.grad(sig.sum(), (m, s)):
            assert bool(torch.isfinite(grad).all())
        for i in range(n):
            at_center = temporal_opacity(m[i : i + 1], s[i : i + 1], m[i].detach())
            assert float(at_center.detach()) == 1.0
    # With μt = 0 the two times are exactly ±Δ, so the symmetry is exact.
    zeros = torch.zeros(n, dtype=F64)
    for delta in (t, t - 0.5):
        a = temporal_opacity(zeros, s64, delta)
        b = temporal_opacity(zeros, s64, -delta)
        assert torch.equal(a, b)


# U-3 · E3 effective opacity


def _sigma_t_independent(mu: Tensor, s_hat: Tensor, t: float) -> Tensor:
    """E4 written directly from the paper's form, (t − μt)/s with s = e^ŝ (float64)."""
    return torch.exp(-0.5 * ((t - mu) / torch.exp(s_hat)) ** 2)


@pytest.mark.parametrize("entry", ["function", "slice"])
def test_u3_effective_opacity_is_sigmoid_times_temporal_opacity(entry: str) -> None:
    """U-3: opacities_t = sigmoid(ô)·σt to 1e-7, and ∂/∂ô = σt·σ(1 − σ) to 1e-7 (float64)."""
    m = _model(300, 8)
    o = m["opacity_logits"].requires_grad_()
    t = 0.45
    if entry == "function":
        eff = effective_opacity(o, m["times"], m["log_durations"], t)
    else:
        eff = _slice(m, t).opacities_t
    sig_t = _sigma_t_independent(m["times"], m["log_durations"], t)
    sigma = torch.sigmoid(o.detach())
    torch.testing.assert_close(eff.detach(), sigma * sig_t, rtol=0.0, atol=1e-7)
    (g,) = torch.autograd.grad(eff.sum(), (o,))
    torch.testing.assert_close(g, sig_t * sigma * (1 - sigma), rtol=0.0, atol=1e-7)


# E8 per-row terms (the per-row half of U-10)


@pytest.mark.parametrize("entry", ["function", "slice"])
def test_reg_terms_value_and_gradient_reach_only_opacity_logits(entry: str) -> None:
    """E8: each row's term is σ·sg[σt]; its only nonzero gradient is ∂/∂ô = σt·σ(1 − σ), and
    every other stored tensor gets exactly zero."""
    m = {k: v.requires_grad_() for k, v in _model(200, 9).items()}
    t = 0.7
    if entry == "function":
        terms = reg4d_terms(m["opacity_logits"], m["times"], m["log_durations"], t)
    else:
        terms = _slice(m, t).reg_terms
    sig_t = _sigma_t_independent(m["times"].detach(), m["log_durations"].detach(), t)
    sigma = torch.sigmoid(m["opacity_logits"].detach())
    torch.testing.assert_close(terms.detach(), sigma * sig_t, rtol=0.0, atol=1e-7)
    names = list(m)
    grads = torch.autograd.grad(terms.sum(), [m[k] for k in names], allow_unused=True)
    for name, g in zip(names, grads, strict=True):
        if name == "opacity_logits":
            assert g is not None
            torch.testing.assert_close(g, sig_t * sigma * (1 - sigma), rtol=1e-7, atol=0.0)
        else:
            assert g is None or bool((g == 0).all()), name


# The temporal slice, eager implementation


def test_slice_fields_shapes_and_activations() -> None:
    """§1.1 per-time arrays and Q-12: shapes, dtypes, q/‖q‖, exp(â), σ = sigmoid(ô)."""
    n = 37
    m = _model(n, 10, torch.float32)
    sl = _slice(m, 0.2)
    assert sl.means_t.shape == (n, 3) and sl.quats_n.shape == (n, 4)
    assert sl.scales.shape == (n, 3)
    for f in (sl.opacities_t, sl.opacities, sl.reg_terms, sl.passes_cull):
        assert f.shape == (n,)
    assert sl.passes_cull.dtype == torch.bool
    for f in (sl.means_t, sl.opacities_t, sl.quats_n, sl.scales, sl.opacities, sl.reg_terms):
        assert f.dtype == torch.float32
    q = m["quats"]
    torch.testing.assert_close(sl.quats_n, q / q.norm(dim=-1, keepdim=True), rtol=1e-6, atol=0.0)
    torch.testing.assert_close(sl.scales, torch.exp(m["log_scales"]), rtol=1e-6, atol=0.0)
    torch.testing.assert_close(
        sl.opacities, torch.sigmoid(m["opacity_logits"]), rtol=1e-6, atol=0.0
    )


def test_slice_accepts_time_as_zero_dim_tensor() -> None:
    """Q-12: t may be a float or a 0-d tensor, with the same result."""
    m = _model(20, 11, torch.float32)
    a = _slice(m, 0.3)
    b = _slice(m, torch.tensor(0.3))
    assert torch.equal(a.means_t, b.means_t) and torch.equal(a.opacities_t, b.opacities_t)


def _boundary_row(target: Tensor) -> tuple[float, float]:
    """A float32 logit ô and a time t (with μt = 0, ŝ = 0) whose float32 effective opacity is
    exactly `target`, found on a grid of neighbouring logits and times."""
    logits = torch.log(target / (1 - target)) + torch.arange(-200, 201) * 2e-7
    times = torch.arange(0, 400) * 1e-4
    zeros = torch.zeros_like(times)
    # σt at each time: rows are times, with μt = t and the query at 0 (symmetric in t − μt).
    sig_t = temporal_opacity(times, zeros, 0.0)
    eff = torch.sigmoid(logits)[:, None] * sig_t[None, :]
    hits = (eff == target).nonzero()
    if len(hits) == 0:
        pytest.fail("no float32 (ô, t) pair on the grid lands exactly on the boundary")
    i, j = (int(v) for v in hits[0])
    return float(logits[i]), float(times[j])


def test_slice_culling_mask_boundary_is_inclusive() -> None:
    """D-32, E6: passes_cull is true exactly where σ_eff ≥ 1/255, the boundary itself included."""
    boundary = torch.tensor(1.0 / 255.0)
    logit, mu_t = _boundary_row(boundary)
    n = 1000
    m = _model(n, 12, torch.float32)
    m["opacity_logits"][0] = logit
    m["times"][0] = mu_t
    m["log_durations"][0] = 0.0
    sl = _slice(m, 0.0)
    assert float(sl.opacities_t[0]) == float(boundary)
    assert bool(sl.passes_cull[0])
    assert torch.equal(sl.passes_cull, sl.opacities_t >= boundary)


def test_slice_cuda_implementation_is_a_stub() -> None:
    """Q-13: impl="cuda" raises NotImplementedError until B-1 justifies it (D-50, M15)."""
    with pytest.raises(NotImplementedError, match="D-50"):
        _slice(_model(4, 13, torch.float32), 0.5, impl="cuda")


def test_slice_rejects_unknown_implementation() -> None:
    """Q-13: only the four PerfConfig.slice_impl values are accepted."""
    with pytest.raises(ValueError, match="impl"):
        _slice(_model(4, 14, torch.float32), 0.5, impl="triton")


_SLICE_FIELDS = (
    "means_t",
    "opacities_t",
    "quats_n",
    "scales",
    "opacities",
    "reg_terms",
    "passes_cull",
)


def test_slice_auto_is_eager_on_cpu() -> None:
    """Q-13: "auto" on CPU tensors gives the eager result bitwise."""
    m = _model(64, 15, torch.float32)
    a, b = _slice(m, 0.5, impl="auto"), _slice(m, 0.5, impl="eager")
    for f in _SLICE_FIELDS:
        assert torch.equal(getattr(a, f), getattr(b, f)), f


@pytest.mark.gpu
def test__slice_auto_falls_back_once_and_compiled_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-13 (private): when compilation fails, "auto" warns once and runs eager from then on;
    an explicit "compiled" raises."""
    from torch._dynamo.exc import TorchDynamoException

    def failing() -> object:
        def fn(*args: object) -> object:
            raise TorchDynamoException("no backend")

        return fn

    monkeypatch.setattr(openftgs.motion, "_compiled_slice", failing)
    monkeypatch.setattr(openftgs.motion, "_auto_fell_back", False)
    m = {k: v.cuda() for k, v in _model(64, 17, torch.float32).items()}
    eager = _slice(m, 0.5, impl="eager")
    with pytest.warns(UserWarning, match="eager"):
        first = _slice(m, 0.5, impl="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = _slice(m, 0.5, impl="auto")
    for got in (first, second):
        assert torch.equal(got.opacities_t, eager.opacities_t)
    with pytest.raises(TorchDynamoException):
        _slice(m, 0.5, impl="compiled")


def _cast(m: dict[str, Tensor], device: str, dtype: torch.dtype) -> dict[str, Tensor]:
    return {k: v.to(device=device, dtype=dtype) for k, v in m.items()}


_DEVICES = [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("device", _DEVICES)
def test_slice_compiles_as_one_graph_without_breaks(device: str) -> None:
    """D-50, Q-13 (#51): the compiled slice is one fused function: it compiles to a single graph
    with no graph break, on CPU and CUDA."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    _slice(_cast(_model(64, 23), device, torch.float32), 0.5, impl="compiled")
    assert int(counters["stats"]["unique_graphs"]) == 1
    assert not counters["graph_break"], dict(counters["graph_break"])


@pytest.mark.parametrize("device", _DEVICES)
def test_slice_compiled_raises_at_the_recompile_limit(device: str) -> None:
    """Q-13 (as amended, #51): dynamo reaching its recompile limit for the slice is a failed
    compilation, so an explicit "compiled" raises. With the limit at 1, float32 inputs compile
    one graph and float64 inputs would need a second."""
    import torch._dynamo
    from torch._dynamo.exc import FailOnRecompileLimitHit

    m = _model(64, 21)
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            _slice(_cast(m, device, torch.float32), 0.5, impl="compiled")
            with pytest.raises(FailOnRecompileLimitHit):
                _slice(_cast(m, device, F64), 0.5, impl="compiled")
    finally:
        torch._dynamo.reset()


@pytest.mark.gpu
def test__slice_auto_falls_back_at_the_recompile_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-13 (as amended, #51; private: it resets the sticky fallback): when dynamo reaches its
    recompile limit, "auto" warns exactly once, naming eager, and runs eager from then on: later
    float32 and float64 calls, the second of which would hit the limit again, raise nothing under
    warnings.simplefilter("error"); the results equal eager."""
    import torch._dynamo

    monkeypatch.setattr(openftgs.motion, "_auto_fell_back", False)
    m = _model(64, 22)
    m32, m64 = _cast(m, "cuda", torch.float32), _cast(m, "cuda", F64)
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(recompile_limit=1):
            _slice(m32, 0.5, impl="auto")
            with pytest.warns(UserWarning, match="eager") as record:
                fallen = _slice(m64, 0.5, impl="auto")
            assert len(record) == 1
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                later = [(_slice(m32, 0.5, impl="auto"), m32), (_slice(m64, 0.5, impl="auto"), m64)]
    finally:
        torch._dynamo.reset()
    _assert_equals_eager([(fallen, m64), *later])


def _assert_equals_eager(results: list[tuple[TemporalSlice, dict[str, Tensor]]]) -> None:
    """Each slice equals the eager slice of its inputs exactly (§2.4 rule 2)."""
    for got, inputs in results:
        want = _slice(inputs, 0.5, impl="eager")
        for f in _SLICE_FIELDS:
            torch.testing.assert_close(getattr(got, f), getattr(want, f), rtol=0, atol=0)


def _breaking_body() -> object:
    """The slice's body with a graph break before it, for compiling."""
    body = openftgs.motion._slice_eager

    def broken(*args: Tensor) -> object:
        torch._dynamo.graph_break()
        return body(*args)

    return broken


@pytest.mark.parametrize("device", _DEVICES)
def test__slice_compiled_raises_on_a_graph_break(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-13 (as amended, #51; private: it puts a graph break in the compiled body): the slice is
    compiled as one graph, so a graph break is a failed compilation and "compiled" raises a
    TorchDynamoException."""
    import torch._dynamo
    from torch._dynamo.exc import TorchDynamoException

    monkeypatch.setattr(openftgs.motion, "_slice_eager", _breaking_body())
    monkeypatch.setattr(openftgs.motion, "_compiled", None)
    torch._dynamo.reset()
    try:
        with pytest.raises(TorchDynamoException):
            _slice(_cast(_model(64, 24), device, torch.float32), 0.5, impl="compiled")
    finally:
        torch._dynamo.reset()


@pytest.mark.gpu
def test__slice_auto_falls_back_on_a_graph_break(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-13 (as amended, #51; private): a graph break in the compiled body makes "auto" warn
    exactly once and run eager, and the result equals eager."""
    import torch._dynamo

    monkeypatch.setattr(openftgs.motion, "_slice_eager", _breaking_body())
    monkeypatch.setattr(openftgs.motion, "_compiled", None)
    monkeypatch.setattr(openftgs.motion, "_auto_fell_back", False)
    m32 = _cast(_model(64, 25), "cuda", torch.float32)
    torch._dynamo.reset()
    try:
        with pytest.warns(UserWarning, match="eager") as record:
            got = _slice(m32, 0.5, impl="auto")
        assert len(record) == 1
    finally:
        torch._dynamo.reset()
    monkeypatch.undo()
    _assert_equals_eager([(got, m32)])


def test_e14_unit_conversions_of_durations_and_velocities() -> None:
    """E14 (Q-27): s = d/T_τ, v = v_τ·T_τ, and their inverses; worked by hand with T_τ = 4."""
    from openftgs.motion import (
        durations_to_normalized,
        durations_to_raw,
        velocities_to_normalized,
        velocities_to_raw,
    )
    from openftgs.types import TimeNormalizer

    time = TimeNormalizer(2.0, 6.0)
    d = torch.tensor([2.0, 0.5], dtype=F64)
    v_tau = torch.tensor([[3.0, -1.0, 0.25]], dtype=torch.float32)
    torch.testing.assert_close(
        durations_to_normalized(d, time), d.new_tensor([0.5, 0.125]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        durations_to_raw(d.new_tensor([0.5, 0.125]), time), d, rtol=0, atol=0
    )
    v = velocities_to_normalized(v_tau, time)
    assert v.dtype == torch.float32
    torch.testing.assert_close(v, v_tau.new_tensor([[12.0, -4.0, 1.0]]), rtol=0, atol=0)
    torch.testing.assert_close(velocities_to_raw(v, time), v_tau, rtol=0, atol=0)


def test_e14_unit_conversions_use_a_unit_of_one_for_a_single_timestamp() -> None:
    """Q-19: with T_τ = 0, durations and velocities convert with a time unit of 1."""
    from openftgs.motion import (
        durations_to_normalized,
        durations_to_raw,
        velocities_to_normalized,
        velocities_to_raw,
    )
    from openftgs.types import TimeNormalizer

    time = TimeNormalizer(3.0, 3.0)
    d = torch.tensor([0.7], dtype=F64)
    v = torch.tensor([[1.0, 2.0, 3.0]])
    for fn, x in (
        (durations_to_normalized, d),
        (durations_to_raw, d),
        (velocities_to_normalized, v),
        (velocities_to_raw, v),
    ):
        assert torch.equal(fn(x, time), x)

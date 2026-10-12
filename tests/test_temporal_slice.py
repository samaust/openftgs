# SPDX-License-Identifier: Apache-2.0
"""U-28: the eager and compiled temporal slices agree in values and gradients across resizes,
with few recompilations (§2.4 "Kernels", D-50; Q-12, Q-13, Q-15).

The CUDA implementation isn't built (D-50 waits for B-1, M15), so only eager and compiled are
compared. "Relative" is the relative L2 error per tensor (Q-15). Recompilations are counted with
dynamo's frame counters, after a reset.
"""

from typing import Literal

import pytest
import torch
import torch._dynamo
from torch import Tensor
from torch._dynamo.utils import counters

from openftgs.model import FreeTimeGaussians
from openftgs.motion import TemporalSlice, temporal_slice
from openftgs.types import TimeNormalizer

# 11 sizes, so 10 resizes: mostly growing as at growth events, with N = 1 and N = 10⁶.
SIZES = [1000, 1500, 1, 2300, 4096, 10_000, 50_000, 120_000, 300_000, 700_000, 1_000_000]
FIELDS = ("means_t", "opacities_t", "quats_n", "scales", "opacities", "reg_terms")
NAMES = (
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
)
CULL = 1.0 / 255.0
Impl = Literal["auto", "eager", "compiled", "cuda"]

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.gpu),
]


def _model(n: int, seed: int, device: str) -> list[Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)

    def u(lo: float, hi: float, *shape: int) -> Tensor:
        return torch.empty(*shape, device=device).uniform_(lo, hi, generator=g)

    return [
        torch.randn(n, 3, generator=g, device=device),
        u(0.0, 1.0, n),
        u(-6.0, 0.0, n),
        torch.randn(n, 3, generator=g, device=device),
        u(-6.0, -1.0, n, 3),
        torch.randn(n, 4, generator=g, device=device),
        u(-6.0, 6.0, n),
    ]


def _resize(params: list[Tensor], n: int, seed: int, device: str) -> list[Tensor]:
    """Resize as at growth events: kept rows keep their order, new rows go last (§1.1 validity
    rules). SIZES also shrinks once, to N = 1, as only the "w/o periodic relocation" ablation
    does (§1.3); that keeps a random subset in order, and is stricter than growth alone because
    it reaches dynamo's size-1 specialization."""
    n_old = len(params[0])
    if n >= n_old:
        new = _model(n - n_old, seed, device)
        return [torch.cat([p, q]) for p, q in zip(params, new, strict=True)]
    g = torch.Generator(device=device).manual_seed(seed)
    keep = torch.randperm(n_old, generator=g, device=device)[:n].sort().values
    return [p[keep] for p in params]


def _run(params: list[Tensor], t: float, impl: Impl) -> tuple[TemporalSlice, list[Tensor]]:
    """Slice, then backpropagate a fixed random weighting of every float output."""
    leaves = [p.detach().clone().requires_grad_() for p in params]
    means, times, log_durations, velocities, log_scales, quats, opacity_logits = leaves
    out = temporal_slice(
        means, times, log_durations, velocities, log_scales, quats, opacity_logits, t, impl=impl
    )
    g = torch.Generator(device=leaves[0].device).manual_seed(len(leaves[0]))
    loss = sum(
        (getattr(out, f) * torch.randn(getattr(out, f).shape, generator=g, device=g.device)).sum()
        for f in FIELDS
    )
    assert isinstance(loss, Tensor)
    grads = list(torch.autograd.grad(loss, leaves))
    return out, grads


def _assert_relative_l2(got: Tensor, want: Tensor, rtol: float, what: str) -> None:
    """Q-15: ‖got − want‖₂ ≤ rtol·‖want‖₂, computed in float64."""
    got, want = got.detach().double(), want.detach().double()
    err = torch.linalg.vector_norm((got - want).flatten())
    ref = torch.linalg.vector_norm(want.flatten())
    assert float(err) <= rtol * float(ref), f"{what}: relative L2 {float(err / ref):.3e}"


def _compiled_frames() -> int:
    return int(counters["frames"]["ok"])


def _stored(n: int, seed: int) -> dict[str, Tensor]:
    """The nine stored tensors of n random rows on CUDA, SH degree 0 (shN has no bands)."""
    params = dict(zip(NAMES, _model(n, seed, "cuda"), strict=True))
    g = torch.Generator(device="cuda").manual_seed(seed)
    params["sh0"] = torch.randn(n, 1, 3, generator=g, device="cuda")
    params["shN"] = torch.zeros(n, 0, 3, device="cuda")
    return params


def _model_resize(model: FreeTimeGaussians, n: int, seed: int) -> None:
    """Resize as at growth events through the model's resize_ (Q-23): new rows go last, and the
    one shrink keeps a random subset by mask, in order."""
    n_old = model.num_gaussians
    if n >= n_old:
        model.resize_(append=_stored(n - n_old, seed))
        return
    g = torch.Generator(device="cuda").manual_seed(seed)
    keep = torch.zeros(n_old, dtype=torch.bool, device="cuda")
    keep[torch.randperm(n_old, generator=g, device="cuda")[:n]] = True
    model.resize_(keep=keep)


@pytest.mark.parametrize("device", DEVICES)
def test_u28_eager_and_compiled_agree_across_resizes(device: str) -> None:
    """U-28: values ≤ 1e-6 and gradients ≤ 1e-5 relative (Q-15) for N from 1 to 10⁶, resized
    between calls; at most 2 recompilations over the 10 resizes."""
    torch._dynamo.reset()
    counters.clear()
    params = _model(SIZES[0], 100, device)
    for k, n in enumerate(SIZES):
        params = _resize(params, n, 100 + k, device)
        t = 0.05 + 0.9 * k / len(SIZES)
        eager, g_eager = _run(params, t, "eager")
        compiled, g_compiled = _run(params, t, "compiled")
        for f in FIELDS:
            _assert_relative_l2(getattr(compiled, f), getattr(eager, f), 1e-6, f"N={n} {f}")
        for name, a, b in zip(NAMES, g_compiled, g_eager, strict=True):
            _assert_relative_l2(a, b, 1e-5, f"N={n} grad {name}")
        # Q-15: the mask may differ only within 1e-6 relative of the cull threshold.
        differ = compiled.passes_cull != eager.passes_cull
        near = (eager.opacities_t - CULL).abs() <= 1e-6 * CULL
        assert not bool((differ & ~near).any()), f"N={n}: mask differs away from 1/255"
    frames = _compiled_frames()
    assert frames >= 1, "the compiled slice never compiled"
    assert frames - 1 <= 2, f"{frames - 1} recompilations over {len(SIZES) - 1} resizes"


@pytest.mark.parametrize("device", DEVICES)
def test_slice_new_time_does_not_recompile(device: str) -> None:
    """D-50: t changes at every iteration, so a new time must not recompile the slice."""
    torch._dynamo.reset()
    counters.clear()
    params = _model(5000, 7, device)
    for t in (0.0, 0.25, 0.731, 1.0, -0.5):
        _run(params, t, "compiled")
    assert _compiled_frames() == 1


@pytest.mark.gpu
def test_slice_auto_compiles_on_cuda() -> None:
    """Q-13: "auto" on CUDA tensors runs the compiled slice, which matches eager (Q-15)."""
    torch._dynamo.reset()
    counters.clear()
    params = _model(20_000, 9, "cuda")
    auto, _ = _run(params, 0.4, "auto")
    eager, _ = _run(params, 0.4, "eager")
    assert _compiled_frames() == 1
    for f in FIELDS:
        _assert_relative_l2(getattr(auto, f), getattr(eager, f), 1e-6, f)


def test__impl_values_match_perf_config() -> None:
    """D-50 (private): the slice accepts exactly PerfConfig.slice_impl's values."""
    import typing

    import openftgs.motion
    from openftgs.config import PerfConfig

    hints = typing.get_type_hints(PerfConfig)
    assert typing.get_args(hints["slice_impl"]) == openftgs.motion._IMPLS


@pytest.mark.gpu
def test_u28_through_the_model_across_resizes() -> None:
    """U-28 (#45): through FreeTimeGaussians.at_time on CUDA, under the default "auto" (the
    compiled slice, Q-13), on one model resized between calls with resize_ as at growth events,
    the values agree with the eager slice within 1e-6 and the gradients on the stored tensors
    within 1e-5 relative (Q-15), with at most 2 recompilations over the 10 resizes (D-50)."""
    torch._dynamo.reset()
    counters.clear()
    model = FreeTimeGaussians(_stored(SIZES[0], 200), sh_degree=0, time=TimeNormalizer(0.0, 1.0))
    for k, n in enumerate(SIZES):
        _model_resize(model, n, 200 + k)
        t = 0.05 + 0.9 * k / len(SIZES)
        params = [getattr(model, name) for name in NAMES]
        compiled = model.at_time(t)
        means, times, log_durations, velocities, log_scales, quats, opacity_logits = params
        eager = temporal_slice(
            means,
            times,
            log_durations,
            velocities,
            log_scales,
            quats,
            opacity_logits,
            t,
            impl="eager",
        )
        g = torch.Generator(device="cuda").manual_seed(n)
        weights = {
            f: torch.randn(getattr(eager, f).shape, generator=g, device="cuda") for f in FIELDS
        }
        grads = {}
        for which, out in (("compiled", compiled), ("eager", eager)):
            loss = sum((getattr(out, f) * weights[f]).sum() for f in FIELDS)
            assert isinstance(loss, Tensor)
            grads[which] = torch.autograd.grad(loss, params)
        for f in FIELDS:
            _assert_relative_l2(getattr(compiled, f), getattr(eager, f), 1e-6, f"N={n} {f}")
        for name, a, b in zip(NAMES, grads["compiled"], grads["eager"], strict=True):
            _assert_relative_l2(a, b, 1e-5, f"N={n} grad {name}")
    frames = _compiled_frames()
    assert frames >= 1, "the compiled slice never compiled"
    assert frames - 1 <= 2, f"{frames - 1} recompilations over {len(SIZES) - 1} resizes"

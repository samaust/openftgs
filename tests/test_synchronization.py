# SPDX-License-Identifier: Apache-2.0
"""U-29: a training iteration synchronizes with the host only inside gsplat's stage functions
or, with compaction, at `render`'s selection of the kept rows (§2.4 rule 3, D-49, D-57, D-76;
Q-171), checked through `Trainer.step` on CUDA with the gsplat backend under
`torch.cuda.set_sync_debug_mode("warn")`.

How a synchronization is seen, by which of three routes, and why a C++ readback replayed from
the autograd device thread is refused: `sync_recorder.py`, whose `SyncRecorder` this file and
the render test of `test_render_gsplat.py` share.
"""

import contextlib
import dataclasses
import importlib
import importlib.util
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gsplat
import pytest
import torch
from builders import random_model, ring, write_scene
from sync_recorder import GSPLAT_DIR, Sync, SyncRecorder, described, in_render_gsplat_stage
from torch import Tensor

import openftgs.losses
import openftgs.render
from openftgs import _stages
from openftgs.config import (
    GrowthConfig,
    LossConfig,
    OptimConfig,
    PercNet,
    PerfConfig,
    RelocConfig,
    TrainConfig,
)
from openftgs.data import MultiViewVideo
from openftgs.train import Trainer
from openftgs.types import TimeNormalizer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        sys.version_info < (3, 12), reason="the recorder needs sys.monitoring (Python 3.12)"
    ),
]


class _Probe(torch.autograd.Function):
    """An identity whose backward records the thread it runs on and, when asked, reads a value
    back to the host (`.item()`), or runs a gsplat stage function that reads one back in C++."""

    thread: int | None = None

    @staticmethod
    def forward(ctx: Any, x: Tensor, read: str) -> Tensor:
        ctx.read = read
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        _Probe.thread = threading.get_ident()
        if ctx.read == "item":
            grad.sum().item()
        elif ctx.read == "gsplat":
            _isect(grad.device)
        return grad, None


def _isect(device: torch.device) -> None:
    """gsplat's tile intersection of five 2D Gaussians, which reads its count back in C++; the
    inputs are made on the device, so nothing else synchronizes."""
    means2d = torch.arange(10, dtype=torch.float32, device=device).reshape(1, 5, 2) * 3.0
    radii = torch.full((1, 5, 2), 3, dtype=torch.int32, device=device)
    depths = torch.arange(1, 6, dtype=torch.float32, device=device).reshape(1, 5)
    gsplat.isect_tiles(means2d, radii, depths, 16, 2, 2)


def _backward_through_probe(read: str) -> None:
    x = torch.ones(4, device="cuda", requires_grad=True)
    _Probe.apply(x, read).sum().backward()


def _autograd_device_thread() -> int:
    """The thread the autograd engine runs CUDA backward functions on."""
    _backward_through_probe("none")
    assert _Probe.thread is not None
    return _Probe.thread


def test_u29_recorder_sees_each_route_and_the_autograd_device_thread() -> None:
    """Added (Q-171): the recorder records a value read back in a backward pass on the autograd
    device thread, a thread other than the caller's, from the backward function; a gsplat stage
    function's own readback, printed only on standard error, from gsplat's file; and a C++
    readback in a backward pass, replayed when backward returns, from a frame outside gsplat,
    which U-29 then refuses. The recorder relies on Python's warnings state being shared by all
    threads, which this probe guards."""
    torch.cuda.synchronize()
    with SyncRecorder() as on_thread:
        _backward_through_probe("item")
    assert len(on_thread.syncs) == 1 and on_thread.unattributed == 0
    (sync,) = on_thread.syncs
    assert sync.thread == _Probe.thread and sync.thread != threading.get_ident()
    assert (sync.innermost.filename, sync.innermost_qualname) == (__file__, "_Probe.backward")

    with SyncRecorder() as direct:
        _isect(torch.device("cuda"))
    assert [s.route for s in direct.syncs] == ["stderr"] and direct.unattributed == 0
    assert Path(direct.syncs[0].innermost.filename).resolve().is_relative_to(GSPLAT_DIR)
    assert direct.syncs[0].thread == threading.get_ident()
    assert not in_render_gsplat_stage(direct.syncs[0])  # gsplat's, but not through `render`

    with SyncRecorder() as replayed:
        _backward_through_probe("gsplat")
    assert len(replayed.syncs) == 1 and replayed.unattributed == 0
    (sync,) = replayed.syncs
    assert sync.thread == threading.get_ident()
    assert not Path(sync.innermost.filename).resolve().is_relative_to(GSPLAT_DIR)


# U-29 through Trainer.step.

_TIME = TimeNormalizer(0.0, 4.0 / 30.0)  # five frames at 30 FPS
_EVENT = 10  # the one growth and relocation event of the run
_WARM_UP = 12  # past the event and the slice's recompilation for the grown model
_WINDOW = 20  # U-29's iterations, with no event and no log read

# The U-29 Calibration row (d858525): host synchronizations per iteration away from events, by
# (packed, compaction): the packed projection's count and the tile intersections, plus the
# kept-row selection under compaction. A case may see fewer, never more.
_MAX_PER_ITERATION = {(True, False): 2, (False, False): 1, (True, True): 3, (False, True): 2}

# Q-171: the backward functions of the gsplat autograd Functions that `render`'s stage calls
# build (gsplat 1.5.3, `gsplat/cuda/_wrapper.py`; source log 184), the only frames a
# synchronization on the autograd device thread may come from.
_GSPLAT_WRAPPERS = GSPLAT_DIR / "cuda" / "_wrapper.py"
_GSPLAT_BACKWARDS = frozenset(
    {
        "_FullyFusedProjection.backward",
        "_FullyFusedProjectionPacked.backward",
        "_SphericalHarmonics.backward",
        "_RasterizeToPixels.backward",
    }
)

# §2.4 Settings, with the slice compiled for certain and the cache in GPU memory; λ_perc = 0.
_BASE = PerfConfig(
    slice_impl="compiled",
    packed=True,
    precull="internal",
    frame_cache="gpu",
    adam="fused",
    deterministic=False,
)


@dataclass(frozen=True)
class _Case:
    perf: dict[str, Any] = field(default_factory=dict)
    lambda_perc: float | None = 0.0  # None: the network's calibrated weight (D-89; Q-249)
    perc_net: PercNet = "flip"  # the default (§2.1 `LossConfig`, D-9)
    ssim_fallback: bool = False
    stage_clock: bool = False  # §3.4: the stages' CUDA events and gradient hooks (B-1)


# D-89's trunk networks, whose weights are downloaded at first use, hence `weights` (Q-7); the
# four DINO networks also need the [init] extra (Q-248).
_INIT_PACKAGES = ("torchvision", "einops", "loguru")
_needs_init = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) is not None for name in _INIT_PACKAGES),
    reason="needs the [init] extra (Q-248)",
)
_TRUNKS: dict[PercNet, tuple[pytest.MarkDecorator, ...]] = {
    "vgg16_oxford": (),
    "alexnet_bvlc": (),
    "dinov2_s14": (_needs_init,),
    "dinov2_b14": (_needs_init,),
    "dino_s16": (_needs_init,),
    "dino_b16": (_needs_init,),
}


_CASES = [
    pytest.param(_Case(), id="base"),
    pytest.param(_Case({"packed": False}), id="unpacked"),
    pytest.param(_Case({"precull": "compact"}), id="compact-packed"),
    pytest.param(_Case({"precull": "compact", "packed": False}), id="compact-unpacked"),
    pytest.param(_Case({"adam": "foreach"}), id="foreach-adam"),
    pytest.param(_Case({"adam": "selective"}), id="selective-adam"),
    pytest.param(_Case({"slice_impl": "eager"}), id="eager-slice"),
    pytest.param(_Case({"frame_cache": "mmap", "prefetch": 8}), id="mmap-prefetch-8"),
    pytest.param(_Case({"frame_cache": "mmap", "prefetch": 0}), id="mmap-prefetch-0"),
    pytest.param(_Case({"frame_cache": "off"}), id="cache-off"),
    # D-89's networks at their calibrated weights: ꟻLIP, the default loss (§2.1 `LossConfig`),
    # and HaarPSI, which need no weights; each trunk network, and two of them under bfloat16
    # (Q-250).
    pytest.param(_Case(lambda_perc=None), id="flip"),
    pytest.param(_Case(perc_net="haarpsi", lambda_perc=None), id="haarpsi"),
    *(
        pytest.param(
            _Case(perc_net=net, lambda_perc=None), id=net, marks=(pytest.mark.weights, *marks)
        )
        for net, marks in _TRUNKS.items()
    ),
    *(
        pytest.param(
            _Case({"perc_precision": "bf16"}, perc_net=net, lambda_perc=None),
            id=f"{net}-bf16",
            marks=(pytest.mark.weights, *_TRUNKS[net]),
        )
        for net in ("vgg16_oxford", "dinov2_s14")
    ),
    pytest.param(_Case(ssim_fallback=True), id="ssim-fallback"),
    pytest.param(_Case({"deterministic": True}), id="deterministic"),
    pytest.param(_Case(stage_clock=True), id="stage-clock"),
]


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> MultiViewVideo:
    """Three training cameras and a test camera, 64×48, five frames of a random model."""
    cams = ring(4, radius=4.0, elevation=1.0, width_px=64, height_px=48, focal=60.0)
    truth = random_model(40, seed=0, sh_degree=3, time=_TIME, spread=0.6, log_scale=(-2.5, -1.5))
    images = [(cam, k / 30.0, "train") for k in range(5) for cam in cams[:3]]
    images += [(cams[3], k / 30.0, "test") for k in range(5)]
    return MultiViewVideo.from_manifest(write_scene(tmp_path_factory.mktemp("u29"), truth, images))


@pytest.fixture
def without_fused_ssim(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Q-42: openftgs.losses reloaded with fused-ssim made unimportable, as without it installed
    (as in `test_losses.py`); reloaded again afterwards. The trainer's loss reads `ssim` from the
    reloaded module's namespace, so it takes the PyTorch path."""
    monkeypatch.setitem(sys.modules, "fused_ssim", None)
    importlib.reload(openftgs.losses)
    yield
    monkeypatch.undo()
    importlib.reload(openftgs.losses)


def _config(case: _Case) -> TrainConfig:
    """One event, at _EVENT, which grows and relocates; the first log read at 100 (the default
    log_every); every SH band active from iteration 4."""
    return TrainConfig(
        sh_degree=3,
        loss=LossConfig(perc_net=case.perc_net, lambda_perc=case.lambda_perc),
        growth=GrowthConfig(start=_EVENT, stop=_EVENT, grad_threshold=1e-7, n_max=400),
        reloc=RelocConfig(every=_EVENT, start=_EVENT, stop=_EVENT, dead_opacity=0.2),
        optim=OptimConfig(iterations=100, sh_interval=1),
        perf=dataclasses.replace(_BASE, **case.perf),
    )


def _allowed(sync: Sync, *, compact: bool, forward: int, autograd: int) -> bool:
    """Q-171, §2.4 rule 3: on the training thread, inside one of `render`'s gsplat stage calls
    or, under compaction, at `render`'s kept-row selection; on the autograd device thread, in the
    backward of one of gsplat's rasterization autograd Functions (not, say, its optimizers)."""
    frame = sync.innermost
    if sync.thread == forward:
        kept_rows = (
            compact
            and Path(frame.filename).resolve() == Path(openftgs.render.__file__).resolve()
            and sync.innermost_qualname == "_kept_rows"
        )
        return in_render_gsplat_stage(sync) or kept_rows
    if sync.thread == autograd:
        return (
            Path(frame.filename).resolve() == _GSPLAT_WRAPPERS
            and sync.innermost_qualname in _GSPLAT_BACKWARDS
        )
    return False


@pytest.mark.parametrize("case", _CASES)
def test_u29_training_iterations_synchronize_only_inside_gsplat(
    video: MultiViewVideo, case: _Case, request: pytest.FixtureRequest
) -> None:
    """U-29 (§2.4 rule 3, D-49, D-57, D-76; Q-171): after a warm-up past compilation and the
    first event, 20 iterations with no event and no log read synchronize only inside gsplat's
    render call, or at `render`'s kept-row selection under compaction. Each case changes one
    factor from the base (packed, internal pre-culling, fused Adam, compiled slice, GPU cache,
    λ_perc = 0, not deterministic); with the stage clock active, its events and gradient hooks
    add none (§3.4, #223). At least one synchronization per iteration must be seen (gsplat reads
    its tile intersections back every render), so the check can't pass empty, and no more per
    iteration than the U-29 Calibration row records for the case's layout."""
    if case.ssim_fallback:
        request.getfixturevalue("without_fused_ssim")  # Q-42: the PyTorch path
    cfg = _config(case)
    model = random_model(60, seed=2, sh_degree=3, time=video.time, spread=0.6)
    trainer = Trainer(cfg, video, model, device="cuda")
    warm_up = [trainer.step() for _ in range(_WARM_UP)]
    assert warm_up[_EVENT - 1].growth is not None  # the first event is behind
    forward, autograd = threading.get_ident(), _autograd_device_thread()
    torch.cuda.synchronize()
    clock = _stages.Clock("cuda") if case.stage_clock else contextlib.nullcontext()
    with SyncRecorder() as recorder, clock:
        logs = [trainer.step() for _ in range(_WINDOW)]
    if isinstance(clock, _stages.Clock):  # read after the window, where the test synchronizes
        names = [timing.name for timing in clock.read()]
        for name in ("iteration", "backward", "backward.losses_done", "backward.raster_done"):
            assert names.count(name) == _WINDOW, name
    assert all(
        log.growth is None
        and log.relocation is None
        and log.pruned is None
        and not log.opacity_reset
        and log.mean_losses is None
        for log in logs
    )
    compact = cfg.perf.precull == "compact"
    outside = [
        s
        for s in recorder.syncs
        if not _allowed(s, compact=compact, forward=forward, autograd=autograd)
    ]
    assert not outside, described(outside, __file__)
    assert recorder.unattributed == 0
    assert len(recorder.syncs) >= _WINDOW
    limit = _MAX_PER_ITERATION[(cfg.perf.packed, compact)]
    assert len(recorder.syncs) <= limit * _WINDOW, described(recorder.syncs, __file__)
    routes = {
        route: sum(s.route == route for s in recorder.syncs) for route in ("python", "stderr")
    }
    print(f"\nU-29: {len(recorder.syncs) / _WINDOW:g} synchronizations per iteration {routes}")

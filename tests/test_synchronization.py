# SPDX-License-Identifier: Apache-2.0
"""U-29: a training iteration synchronizes with the host only inside gsplat's render call
(§2.4 rule 3, D-49, D-57, D-76; Q-171), checked through `Trainer.step` on CUDA with the gsplat
backend under `torch.cuda.set_sync_debug_mode("warn")`.

How a synchronization is seen. The debug mode raises a C++ warning at every synchronizing CUDA
call, and the warning reaches Python by one of three routes (source log 158, 159):

- raised during a call through one of torch's Python bindings (any torch operation, or an
  autograd Function's `apply`), it becomes a Python warning when the call returns, from the
  frame that made the call, on the thread that made it. That includes the autograd device
  thread, when a Python backward calls a torch operation;
- raised in an extension function that Python calls directly (gsplat's compiled stage functions
  outside an autograd Function, such as the tile intersection of `isect_tiles`), it never becomes
  a Python warning: c10 prints it on standard error;
- raised by C++ code on the autograd device thread, it is replayed when backward returns, as a
  Python warning from torch's own backward frame, so where it came from is lost.

`_SyncRecorder` takes the first route through `warnings.showwarning` and the second by sending
standard error to a file and, after each C function returns (`sys.monitoring`), charging what it
printed to the Python stack that called it. The third arrives as a Python warning whose innermost
frame is torch's, which the check refuses whoever made it: conservative, and seen in no run.
Without the second route an unpacked render, whose one synchronization is the tile
intersection's, would show no warning at all. The debug mode is a prototype that may miss some
synchronizations; the test pins the ones it sees.
"""

import contextlib
import dataclasses
import os
import sys
import tempfile
import threading
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import gsplat
import pytest
import torch
from builders import random_model, ring, write_scene
from torch import Tensor

import openftgs.losses
import openftgs.render
from openftgs import _stages
from openftgs.config import (
    GrowthConfig,
    LossConfig,
    OptimConfig,
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

_SYNC_MESSAGE = "called a synchronizing CUDA operation"
_GSPLAT_DIR = Path(gsplat.__file__).resolve().parent


# The recorder.


@dataclass(frozen=True)
class _Sync:
    """One synchronization: the thread it happened on and the Python stack that made it,
    outermost frame first, so the last frame is the innermost."""

    thread: int
    thread_name: str
    stack: tuple[traceback.FrameSummary, ...]
    route: str  # "python": a Python warning; "stderr": printed by C++ only

    @property
    def innermost(self) -> traceback.FrameSummary:
        return self.stack[-1]

    def describe(self) -> str:
        """The thread, the route and the stack from this test file's outermost frame inward."""
        ours = [k for k, f in enumerate(self.stack) if f.filename == __file__]
        frames = self.stack[ours[0] if ours else 0 :]
        lines = "".join(f"\n    {f.filename}:{f.lineno} in {f.name}" for f in frames)
        return f"[{self.thread_name}, {self.route}]{lines}"


def _described(syncs: list[_Sync]) -> str:
    """Each distinct synchronization once, with how often it happened."""
    counts: dict[str, int] = {}
    for sync in syncs:
        counts[sync.describe()] = counts.get(sync.describe(), 0) + 1
    return "\n".join(f"{count} x {text}" for text, count in counts.items())


def _warnings_files() -> set[str]:
    """The files of the standard library's warnings machinery, which sit above the frame that
    raised a warning when showwarning runs."""
    names = ("warnings", "_py_warnings")
    return {m.__file__ for n in names if (m := sys.modules.get(n)) is not None and m.__file__}


@dataclass
class _SyncRecorder:
    """Every synchronization the debug mode reports while the recorder is active, with its full
    stack and thread, by the first two routes of the module docstring; `unattributed` counts what
    C++ printed after the last C function returned, which no stack can claim."""

    syncs: list[_Sync] = field(default_factory=list)
    unattributed: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._seen = 0
        self._exit: list[Callable[[], None]] = []

    def _record(self, route: str, stack: list[traceback.FrameSummary]) -> None:
        thread = threading.current_thread()
        self.syncs.append(_Sync(threading.get_ident(), thread.name, tuple(stack), route))

    def _show(self, message: Any, *args: Any, **kwargs: Any) -> None:
        if _SYNC_MESSAGE not in str(message):
            return
        stack = traceback.extract_stack()[:-1]  # without this hook
        machinery = _warnings_files()
        while stack and stack[-1].filename in machinery:
            stack.pop()
        self._record("python", stack)

    def _c_return(self, code: object, offset: int, function: object, arg0: object) -> None:
        # Anything new on standard error was printed by the C function that just returned: an
        # extension function bound with pybind11 holds the GIL unless it releases it, so no other
        # thread's call returns while it runs. Output charged to the wrong call could pass the
        # check only if that call were gsplat's too.
        size = os.fstat(2).st_size
        if size == self._seen:
            return
        with self._lock:
            printed = os.pread(self._capture.fileno(), size - self._seen, self._seen)
            self._seen = size
        for _ in range(printed.count(_SYNC_MESSAGE.encode())):
            self._record("stderr", traceback.extract_stack()[:-1])

    def _monitor(self) -> Callable[[], None]:
        if sys.version_info < (3, 12):  # the module is skipped; this tells mypy why
            raise RuntimeError("sys.monitoring needs Python 3.12")
        monitoring = sys.monitoring
        tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
        events = monitoring.events
        monitoring.use_tool_id(tool, "openftgs U-29")
        monitoring.register_callback(tool, events.C_RETURN, self._c_return)
        monitoring.register_callback(tool, events.C_RAISE, self._c_return)
        # C_RETURN and C_RAISE come with CALL, which needs no callback of its own.
        monitoring.set_events(tool, events.CALL | events.C_RETURN | events.C_RAISE)

        def stop() -> None:
            monitoring.set_events(tool, 0)
            monitoring.register_callback(tool, events.C_RETURN, None)
            monitoring.register_callback(tool, events.C_RAISE, None)
            monitoring.free_tool_id(tool)

        return stop

    def __enter__(self) -> "_SyncRecorder":
        try:
            caught = warnings.catch_warnings()
            caught.__enter__()
            self._exit.append(lambda: caught.__exit__(None, None, None))
            warnings.simplefilter("always")
            warnings.showwarning = self._show
            sys.stderr.flush()
            saved = os.dup(2)
            self._capture = tempfile.TemporaryFile()
            os.dup2(self._capture.fileno(), 2)
            self._exit.append(lambda: self._restore_stderr(saved))
            self._exit.append(self._monitor())
            torch.cuda.set_sync_debug_mode("warn")
            self._exit.append(lambda: torch.cuda.set_sync_debug_mode(0))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _restore_stderr(self, saved: int) -> None:
        os.dup2(saved, 2)
        os.close(saved)
        size = os.fstat(self._capture.fileno()).st_size
        rest = os.pread(self._capture.fileno(), size - self._seen, self._seen)
        self.unattributed = rest.count(_SYNC_MESSAGE.encode())
        self._capture.close()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        while self._exit:
            self._exit.pop()()


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
    with _SyncRecorder() as on_thread:
        _backward_through_probe("item")
    assert len(on_thread.syncs) == 1 and on_thread.unattributed == 0
    (sync,) = on_thread.syncs
    assert sync.thread == _Probe.thread and sync.thread != threading.get_ident()
    assert (sync.innermost.filename, sync.innermost.name) == (__file__, "backward")

    with _SyncRecorder() as direct:
        _isect(torch.device("cuda"))
    assert [s.route for s in direct.syncs] == ["stderr"] and direct.unattributed == 0
    assert Path(direct.syncs[0].innermost.filename).resolve().is_relative_to(_GSPLAT_DIR)
    assert direct.syncs[0].thread == threading.get_ident()

    with _SyncRecorder() as replayed:
        _backward_through_probe("gsplat")
    assert len(replayed.syncs) == 1 and replayed.unattributed == 0
    (sync,) = replayed.syncs
    assert sync.thread == threading.get_ident()
    assert not Path(sync.innermost.filename).resolve().is_relative_to(_GSPLAT_DIR)


# U-29 through Trainer.step.

_TIME = TimeNormalizer(0.0, 4.0 / 30.0)  # five frames at 30 FPS
_EVENT = 10  # the one growth and relocation event of the run
_WARM_UP = 12  # past the event and the slice's recompilation for the grown model
_WINDOW = 20  # U-29's iterations, with no event and no log read

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
    lambda_perc: float = 0.0
    ssim_fallback: bool = False
    stage_clock: bool = False  # §3.4: the stages' CUDA events and gradient hooks (B-1)


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
    pytest.param(_Case(lambda_perc=0.01), id="perceptual", marks=pytest.mark.weights),
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


def _config(case: _Case) -> TrainConfig:
    """One event, at _EVENT, which grows and relocates; the first log read at 100 (the default
    log_every); every SH band active from iteration 4."""
    return TrainConfig(
        sh_degree=3,
        loss=LossConfig(lambda_perc=case.lambda_perc),
        growth=GrowthConfig(start=_EVENT, stop=_EVENT, grad_threshold=1e-7, n_max=400),
        reloc=RelocConfig(every=_EVENT, start=_EVENT, stop=_EVENT, dead_opacity=0.2),
        optim=OptimConfig(iterations=100, sh_interval=1),
        perf=dataclasses.replace(_BASE, **case.perf),
    )


def _allowed(sync: _Sync, *, compact: bool, threads: set[int]) -> bool:
    """Q-171: on the training thread or the autograd device thread, with the innermost frame in
    gsplat's files or, under compaction, at `render`'s kept-row selection."""
    frame = sync.innermost
    in_gsplat = Path(frame.filename).resolve().is_relative_to(_GSPLAT_DIR)
    kept_rows = (
        compact
        and Path(frame.filename).resolve() == Path(openftgs.render.__file__).resolve()
        and frame.name == "_kept_rows"
    )
    return sync.thread in threads and (in_gsplat or kept_rows)


@pytest.mark.parametrize("case", _CASES)
def test_u29_training_iterations_synchronize_only_inside_gsplat(
    video: MultiViewVideo, case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U-29 (§2.4 rule 3, D-49, D-57, D-76; Q-171): after a warm-up past compilation and the
    first event, 20 iterations with no event and no log read synchronize only inside gsplat's
    render call, or at `render`'s kept-row selection under compaction. Each case changes one
    factor from the base (packed, internal pre-culling, fused Adam, compiled slice, GPU cache,
    λ_perc = 0, not deterministic); with the stage clock active, its events and gradient hooks
    add none (§3.4, #223). At least one synchronization per iteration must be seen (gsplat reads
    its tile intersections back every render), so the check can't pass empty."""
    if case.ssim_fallback:
        # Q-42: the PyTorch path, as without fused-ssim installed.
        monkeypatch.setattr(openftgs.losses, "_fused_ssim", lambda: None)
    cfg = _config(case)
    model = random_model(60, seed=2, sh_degree=3, time=video.time, spread=0.6)
    trainer = Trainer(cfg, video, model, device="cuda")
    warm_up = [trainer.step() for _ in range(_WARM_UP)]
    assert warm_up[_EVENT - 1].growth is not None  # the first event is behind
    threads = {threading.get_ident(), _autograd_device_thread()}
    torch.cuda.synchronize()
    clock = _stages.Clock("cuda") if case.stage_clock else contextlib.nullcontext()
    with _SyncRecorder() as recorder, clock:
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
    outside = [s for s in recorder.syncs if not _allowed(s, compact=compact, threads=threads)]
    assert not outside, _described(outside)
    assert recorder.unattributed == 0
    assert len(recorder.syncs) >= _WINDOW
    routes = {
        route: sum(s.route == route for s in recorder.syncs) for route in ("python", "stderr")
    }
    print(f"\nU-29: {len(recorder.syncs) / _WINDOW:g} synchronizations per iteration {routes}")

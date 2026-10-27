# SPDX-License-Identifier: Apache-2.0
"""Stage instrumentation (§3.4): the named parts of a training iteration, of initialization and
of conversion, as NVTX ranges, profiler ranges and clock records.

Spec: §3.4 (B-1's per-stage times "from CUDA events"; "Each stage runs inside an NVTX range"),
§2.4 rule 3, D-49. A private top-level module that imports nothing from openftgs, so every band
may import it (PRD #220, owner's decision); docs/modules/train/trainer.md lists the stages.

What a stage emits:
- an NVTX range, whenever PyTorch is a CUDA build, so external profilers see the stage names
  without `--profile`; probed once, and skipped on builds without NVTX;
- a profiler range (`torch.profiler.record_function`), only while profiling is switched on by
  `set_profiling`, as `ftgs train --profile` does for its window;
- a record in the active `Clock`, only while one is active: CUDA events on the clock's device
  (taken from a pool) or wall time.

A clock is global, not per thread: the thread that activates it may hand work to others (the
shards of `roma_init`, the autograd engine's device thread), and their stages record into it.
Recording never synchronizes; `Clock.read` waits for the device, so it belongs where the run
synchronizes anyway (§2.4 rule 3).
"""

import functools
import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal

import torch
from torch import Tensor

__all__ = ["POINTS", "STAGES", "Clock", "Timing", "point", "recorder", "set_profiling", "stage"]

# The one fixed list of stage names (PRD #220). Sub-stages carry their parent's name as a prefix.
STAGES: tuple[str, ...] = (
    # A training iteration (`Trainer.step`; §3.4 B-1, B-6), timed on CUDA events.
    "iteration",
    "data",  # the data wait (B-6)
    "slice",  # the temporal slice, with the SH concat on the reference backend (`at_time`)
    "slice.kernel",
    "slice.sh_concat",
    "raster.forward",  # pre-culling by compaction included (D-57)
    "raster.forward.projection",
    "raster.forward.sh_gather",  # the SH of the rows the SH evaluation receives (Q-274, #423)
    "raster.forward.sh",
    "raster.forward.isect",
    "raster.forward.rasterize",
    "loss.l1",
    "loss.ssim",
    "loss.perceptual",
    "loss.reg",
    "backward",  # split by POINTS
    "statistics",
    "adam",
    "events",
    "log",
    # Initialization (`roma_init`; B-8), on the wall clock.
    "init.match",  # one camera pair in its shard's thread, keyed by the shard's device
    "init.triangulate",
    "init.knn",
    # Conversion (B-8), on the wall clock, keyed by camera.
    "convert.decode",  # one frame read from ffmpeg
    "convert.encode",  # one frame processed and written in the process pool
)
# Points inside the backward stage, from gradient hooks: every loss's backward is done (the
# rendered image's gradient is complete); the rasterization's backward is done (the moved means'
# gradient is complete, so projection and SH are). What follows until `backward()` returns is the
# temporal slice's backward.
POINTS: tuple[str, ...] = ("backward.losses_done", "backward.raster_done")
_STAGE_NAMES = frozenset(STAGES)


@dataclass(frozen=True)
class Timing:
    """One recorded stage, or a point when `start == end`. Times are seconds since the clock
    was first activated; `index` orders the records of one clock across threads."""

    name: str
    key: str | None
    thread: str
    index: int
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


class _Entry:
    __slots__ = ("name", "key", "thread", "index", "start", "end")

    def __init__(self, name: str, key: str | None, start: Any, end: Any = None) -> None:
        self.name, self.key, self.start, self.end = name, key, start, end
        self.thread = threading.current_thread().name
        self.index = -1


_ACTIVE: "Clock | None" = None
_ACTIVATION = threading.Lock()


class Clock:
    """Records the stages run while it is active (`with clock:`), from any thread: on `"cuda"`,
    a pair of CUDA events per stage on the current stream of `device` (default: the current
    device); on `"wall"`, `time.perf_counter`. One clock is active at a time."""

    def __init__(
        self, kind: Literal["cuda", "wall"] = "cuda", *, device: str | torch.device | None = None
    ) -> None:
        if kind not in ("cuda", "wall"):
            raise ValueError(f'kind must be "cuda" or "wall", got {kind!r}')
        self.kind = kind
        self._device: torch.device | None = None
        if kind == "cuda":
            where = torch.device("cuda" if device is None else device)
            if where.type != "cuda":
                raise ValueError(f"a CUDA clock needs a CUDA device, got {where}")
            if where.index is None:
                where = torch.device("cuda", torch.cuda.current_device())
            self._device = where
        elif device is not None:
            raise ValueError("a wall clock takes no device")
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._free: list[Any] = []  # CUDA events whose times have been read
        self._count = itertools.count()
        self._origin: Any = None
        self._active = False

    def __enter__(self) -> "Clock":
        global _ACTIVE
        with _ACTIVATION:
            if _ACTIVE is not None:
                raise RuntimeError("a stage clock is already active")
            if self._origin is None:
                self._origin = self._now()
            self._active = True
            _ACTIVE = self
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        global _ACTIVE
        with _ACTIVATION:
            self._active = False
            if _ACTIVE is self:
                _ACTIVE = None

    def read(self) -> list[Timing]:
        """The stages finished since the last read, in the order they began; waits for the
        clock's device first."""
        with self._lock:
            done = [entry for entry in self._entries if entry.end is not None]
            self._entries = [entry for entry in self._entries if entry.end is None]
        if self._device is not None and done:
            torch.cuda.synchronize(self._device)
        timings = [
            Timing(e.name, e.key, e.thread, e.index, self._seconds(e.start), self._seconds(e.end))
            for e in done
        ]
        if self._device is not None:
            with self._lock:
                for entry in done:
                    self._free.append(entry.start)
                    if entry.end is not entry.start:
                        self._free.append(entry.end)
        return timings

    # -- recording, from any thread --------------------------------------------------------------

    def _now(self) -> Any:
        if self._device is None:
            return time.perf_counter()
        with self._lock:
            event = self._free.pop() if self._free else None
        if event is None:
            event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record(torch.cuda.current_stream(self._device))
        return event

    def _seconds(self, mark: Any) -> float:
        if self._device is None:
            return float(mark - self._origin)
        return float(self._origin.elapsed_time(mark)) / 1000.0

    def _add(self, entry: _Entry) -> _Entry:
        with self._lock:
            entry.index = next(self._count)
            self._entries.append(entry)
        return entry

    def _begin(self, name: str, key: str | None) -> _Entry:
        return self._add(_Entry(name, key, self._now()))

    def _end(self, entry: _Entry) -> None:
        entry.end = self._now()

    def _point(self, name: str) -> None:
        if self._active:
            mark = self._now()
            self._add(_Entry(name, None, mark, mark))

    def _elsewhere(self, name: str, key: str | None, seconds: float) -> None:
        end = time.perf_counter()
        self._add(_Entry(name, key, end - seconds, end))


# NVTX (§3.4): probed once; builds without CUDA raise RuntimeError from every NVTX function.


@dataclass(frozen=True)
class _Nvtx:
    push: Callable[[str], Any]
    pop: Callable[[], Any]
    mark: Callable[[str], Any]


_UNPROBED: Any = object()
_nvtx_state: Any = _UNPROBED


def _probe_nvtx() -> _Nvtx | None:
    try:
        from torch.cuda import nvtx

        nvtx.range_push("openftgs stages")  # type: ignore[no-untyped-call]
        nvtx.range_pop()  # type: ignore[no-untyped-call]
    except (ImportError, AttributeError, RuntimeError):
        return None
    return _Nvtx(nvtx.range_push, nvtx.range_pop, nvtx.mark)


def _nvtx() -> _Nvtx | None:
    global _nvtx_state
    if _nvtx_state is _UNPROBED:
        _nvtx_state = _probe_nvtx()
    result: _Nvtx | None = _nvtx_state
    return result


def _reset_nvtx() -> None:
    """Probe again at the next stage (for tests that take NVTX away)."""
    global _nvtx_state
    _nvtx_state = _UNPROBED


_profiling = False


def set_profiling(enabled: bool) -> None:
    """Switch the stages' profiler ranges on or off (`ftgs train --profile`'s window)."""
    global _profiling
    _profiling = bool(enabled)


class _Stage:
    __slots__ = ("_name", "_key", "_nvtx", "_profiled", "_clock", "_entry")

    def __init__(self, name: str, key: str | None) -> None:
        self._name, self._key = name, key

    def __enter__(self) -> None:
        self._nvtx = _nvtx()
        if self._nvtx is not None:
            self._nvtx.push(self._name)
        try:
            self._profiled: Any = None
            if _profiling:
                self._profiled = torch.profiler.record_function(self._name)
                self._profiled.__enter__()
            self._clock = _ACTIVE
            self._entry = None if self._clock is None else self._clock._begin(self._name, self._key)
        except BaseException:
            self._close(None, None, None)
            raise

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        try:
            if self._entry is not None and self._clock is not None:
                self._clock._end(self._entry)
        finally:
            self._close(kind, error, trace)

    def _close(self, *exc: Any) -> None:
        try:
            if self._profiled is not None:
                self._profiled.__exit__(*exc)
        finally:
            if self._nvtx is not None:
                self._nvtx.pop()


def stage(name: str, *, key: str | None = None) -> _Stage:
    """A context manager that runs its block as the stage `name`, one of STAGES; `key` tells
    records of one stage apart (a camera, a shard's device)."""
    if name not in _STAGE_NAMES:
        raise ValueError(f"unknown stage {name!r}; the stages are {STAGES}")
    return _Stage(name, key)


def point(tensor: Tensor, name: str) -> None:
    """While a clock is active, record the point `name`, one of POINTS, when the gradient of
    `tensor` is complete. The hook is registered only then, and records into that clock while it
    stays active; it runs on the autograd engine's thread. Does nothing for a tensor that doesn't
    require a gradient."""
    if name not in POINTS:
        raise ValueError(f"unknown point {name!r}; the points are {POINTS}")
    clock = _ACTIVE
    if clock is None or not tensor.requires_grad:
        return

    def fired(grad: Tensor) -> None:
        clock._point(name)
        nvtx = _nvtx()
        if nvtx is not None:
            nvtx.mark(name)

    tensor.register_hook(fired)  # type: ignore[no-untyped-call]


def recorder(name: str, *, key: str | None = None) -> Callable[[float], None] | None:
    """For work measured in another process (the converters' pool): a function that records
    its duration in seconds as stage `name`, ending when it is called, into the wall clock active
    now; None when no wall clock is active."""
    if name not in _STAGE_NAMES:
        raise ValueError(f"unknown stage {name!r}; the stages are {STAGES}")
    clock = _ACTIVE
    if clock is None or clock.kind != "wall":
        return None
    return functools.partial(clock._elsewhere, name, key)

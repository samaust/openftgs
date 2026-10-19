# SPDX-License-Identifier: Apache-2.0
"""The stage instrumentation of §3.4, called directly: the private module `openftgs._stages`,
a seam the owner confirmed on 2026-10-19 (PRD #220, "Tests that must pass"). These tests catch
clock, thread and hook-order bugs cheaply; the wiring into the trainer is checked at the public
ends (U-29 with a clock active, `ftgs train --profile`).

Source of truth: §3.4 (B-1's per-stage times "from CUDA events"; "Each stage runs inside an NVTX
range") and the PRD's decisions on the stage instrumentation.
"""

import threading
import time
from collections.abc import Iterator

import pytest
import torch

from openftgs import _stages


def test_stages_nest_on_the_wall_clock() -> None:
    """A stage entered inside another is timed inside it, and stages come back in the order
    they were entered."""
    with _stages.Clock("wall") as clock:
        with _stages.stage("iteration"):
            with _stages.stage("slice"), _stages.stage("slice.kernel"):
                time.sleep(0.002)
            with _stages.stage("backward"):
                time.sleep(0.002)
    timings = clock.read()
    assert [t.name for t in timings] == ["iteration", "slice", "slice.kernel", "backward"]
    outer, inner, kernel, backward = timings
    assert outer.start <= inner.start <= kernel.start < kernel.end <= inner.end
    assert inner.end <= backward.start < backward.end <= outer.end
    assert kernel.seconds >= 0.002 and backward.seconds >= 0.002


class _FakeNvtx:
    """Stands in for torch.cuda.nvtx and records each call, in order, from any thread."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def push(self, name: str) -> int:
        self.calls.append((threading.current_thread().name, "push", name))
        return 0

    def pop(self) -> int:
        self.calls.append((threading.current_thread().name, "pop", None))
        return 0

    def mark(self, name: str) -> None:
        self.calls.append((threading.current_thread().name, "mark", name))


@pytest.fixture
def nvtx(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeNvtx]:
    """NVTX replaced by a recorder and probed afresh, without the probe's own range; probed
    afresh again afterwards."""
    fake = _FakeNvtx()
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", fake.push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", fake.pop)
    monkeypatch.setattr(torch.cuda.nvtx, "mark", fake.mark)
    _stages._reset_nvtx()
    assert _stages._nvtx() is not None
    fake.calls.clear()
    yield fake
    _stages._reset_nvtx()


def test_each_stage_is_an_nvtx_range_and_ranges_nest(nvtx: _FakeNvtx) -> None:
    """§3.4: "Each stage runs inside an NVTX range", with or without a clock; a stage left by an
    exception still closes its range, so the ranges stay nested."""
    with _stages.stage("iteration"):
        with _stages.stage("loss.l1"):
            pass
        with pytest.raises(KeyError), _stages.stage("adam"):
            raise KeyError("inside a stage")
    thread = threading.current_thread().name
    assert nvtx.calls == [
        (thread, "push", "iteration"),
        (thread, "push", "loss.l1"),
        (thread, "pop", None),
        (thread, "push", "adam"),
        (thread, "pop", None),
        (thread, "pop", None),
    ]


def test_the_wall_clock_records_stages_from_several_threads(nvtx: _FakeNvtx) -> None:
    """PRD #220: the clock records from several threads (initialization's shards), each stage
    with its thread and key, nested within its own thread; and each thread's NVTX ranges nest."""
    barrier = threading.Barrier(4)

    def shard(device: str) -> None:
        barrier.wait()  # the four shards' stages overlap in time
        with _stages.stage("init.match", key=device):
            time.sleep(0.002)
            with _stages.stage("init.triangulate", key=device):
                time.sleep(0.002)

    devices = [f"cuda:{i}" for i in range(4)]
    threads = [threading.Thread(target=shard, args=(d,), name=f"shard {d}") for d in devices]
    with _stages.Clock("wall") as clock:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with _stages.stage("init.knn"):
            pass
    timings = clock.read()
    assert sorted(t.index for t in timings) == list(range(9))
    assert timings == sorted(timings, key=lambda t: t.index)
    for device in devices:
        mine = [t for t in timings if t.thread == f"shard {device}"]
        assert [(t.name, t.key) for t in mine] == [
            ("init.match", device),
            ("init.triangulate", device),
        ]
        match, triangulate = mine
        assert match.start <= triangulate.start < triangulate.end <= match.end
        assert triangulate.seconds >= 0.002 and match.seconds >= 0.004
        calls = [call[1:] for call in nvtx.calls if call[0] == f"shard {device}"]
        assert calls == [
            ("push", "init.match"),
            ("push", "init.triangulate"),
            ("pop", None),
            ("pop", None),
        ]
    (knn,) = [t for t in timings if t.name == "init.knn"]
    assert knn.thread == threading.current_thread().name and knn.key is None


def test_nothing_is_recorded_while_no_clock_is_active() -> None:
    """PRD #220: a clock records only the stages run while it is active, and a gradient hook is
    registered only while a clock is active, recording into that clock only while it stays
    active; durations measured elsewhere have no clock to go to."""
    clock = _stages.Clock("wall")
    with _stages.stage("adam"):
        pass
    x = torch.ones(3, requires_grad=True)
    before = x * 2.0
    _stages.point(before, "backward.losses_done")  # no clock: no hook
    assert _stages.recorder("convert.encode", key="cam00") is None
    with clock:
        with _stages.stage("statistics"):
            pass
        during = x * 3.0
        _stages.point(during, "backward.raster_done")  # fires after the clock has stopped
        record = _stages.recorder("convert.encode", key="cam00")
    with _stages.stage("log"):
        pass
    with _stages.Clock("wall") as later:
        torch.autograd.backward(before.sum() + during.sum())
    assert record is not None
    record(0.5)  # measured while the clock was active, and arriving after it stopped
    assert [(t.name, t.key) for t in clock.read()] == [
        ("statistics", None),
        ("convert.encode", "cam00"),
    ]
    assert later.read() == []


def test_stages_run_without_nvtx(monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.4, PRD #220: NVTX is on only with CUDA; on a build without it, where every NVTX
    function raises RuntimeError, stages and points still run and record."""

    def missing(*args: object) -> None:
        raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")

    for name in ("range_push", "range_pop", "mark"):
        monkeypatch.setattr(torch.cuda.nvtx, name, missing)
    _stages._reset_nvtx()
    try:
        x = torch.ones(3, requires_grad=True)
        with _stages.Clock("wall") as clock:
            with _stages.stage("backward"):
                y = x * 2.0
                _stages.point(y, "backward.losses_done")
                torch.autograd.backward(y.sum())
        assert [t.name for t in clock.read()] == ["backward", "backward.losses_done"]
    finally:
        _stages._reset_nvtx()


def _profiled_names(profiling: bool) -> set[str]:
    """The event names a CPU profiler records around two nested stages."""
    _stages.set_profiling(profiling)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with _stages.stage("slice"), _stages.stage("slice.kernel"):
                torch.ones(4).sum()
    finally:
        _stages.set_profiling(False)
    return {event.name for event in prof.events()}


def test_profiler_ranges_only_while_profiling() -> None:
    """PRD #220: a stage is a profiler range only while profiling is switched on, so a
    `--profile` trace carries the stage names and other runs don't pay for the ranges."""
    assert {"slice", "slice.kernel"} <= _profiled_names(True)
    assert not {"slice", "slice.kernel"} & _profiled_names(False)


def test_names_outside_the_fixed_lists_are_refused() -> None:
    """PRD #220: one fixed list of stage names and one of points, so a mistyped marker fails
    where it is written; one clock is active at a time."""
    with pytest.raises(ValueError, match="unknown stage"):
        _stages.stage("raster")
    with pytest.raises(ValueError, match="unknown point"):
        _stages.point(torch.ones(1, requires_grad=True), "backward.done")
    with pytest.raises(ValueError, match="unknown stage"):
        _stages.recorder("convert")
    with pytest.raises(RuntimeError, match="already active"), _stages.Clock("wall"):
        with _stages.Clock("wall"):
            pass


@pytest.mark.gpu
def test_cuda_events_time_each_stage_and_split_the_backward_pass() -> None:
    """§3.4 B-1: per-stage times from CUDA events are positive for stages that run kernels; the
    two gradient hooks fire on the autograd engine's thread in the order of the backward pass,
    losses → raster → slice, inside the backward stage, which ends after `backward()` returns."""
    x = torch.randn(1 << 20, device="cuda", requires_grad=True)
    with _stages.Clock("cuda") as clock:
        with _stages.stage("slice"):
            moved = torch.sin(x) * 2.0 + 1.0
        _stages.point(moved, "backward.raster_done")
        with _stages.stage("raster.forward"):
            image = torch.cos(moved).reshape(1024, 1024) @ torch.ones(1024, 3, device="cuda")
        _stages.point(image, "backward.losses_done")
        with _stages.stage("loss.l1"):
            loss = (image - 1.0).abs().mean()
        with _stages.stage("backward"):
            torch.autograd.backward(loss)
    torch.cuda.synchronize()
    timings = clock.read()
    names = [t.name for t in timings]
    assert names == [
        "slice",
        "raster.forward",
        "loss.l1",
        "backward",
        "backward.losses_done",
        "backward.raster_done",
    ]
    by_name = {t.name: t for t in timings}
    for name in ("slice", "raster.forward", "loss.l1", "backward"):
        assert by_name[name].seconds > 0.0, name
    backward = by_name["backward"]
    losses, raster = by_name["backward.losses_done"], by_name["backward.raster_done"]
    assert backward.start <= losses.start <= raster.start <= backward.end
    assert losses.seconds == raster.seconds == 0.0
    assert by_name["loss.l1"].end <= backward.start
    main = threading.current_thread().name
    assert losses.thread == raster.thread != main  # the autograd engine's device thread
    # The events read go back to the pool; the next records share the clock's timeline.
    with clock, _stages.stage("adam"):
        assert _stages.recorder("convert.encode") is None  # wall clocks only
        x.data.mul_(0.5)
    (adam,) = clock.read()
    assert adam.name == "adam" and adam.seconds > 0.0 and adam.start >= backward.end

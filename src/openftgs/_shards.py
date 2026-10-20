# SPDX-License-Identifier: Apache-2.0
"""Thread-per-device sharding (D-54): independent jobs spread over shards, one worker thread per
shard, without changing results.

Spec: D-54 ("Independent work is spread across GPUs without changing results: RoMa pairs,
evaluation frames, render requests"), §2.4 "Several GPUs". A private top-level module that
imports nothing from openftgs, so every band may import it (PRD #220, as `openftgs._stages`):
`init.roma_init` (Q-106), `render.render_request` and `eval.evaluate_images` (Q-175, Q-186)
share it.

A shard is whatever one worker needs for its device (a dense matcher, a model copy and its
evaluator), built by the caller in its own thread, one shard after another, before the workers
start: so nothing is downloaded, built or compiled from several threads at once. Workers take
jobs from one shared queue. A shard's exception stops that shard and is raised in the caller's
thread once every shard has stopped.
"""

import queue
import threading
from collections.abc import Callable, Iterable, Sequence
from types import TracebackType
from typing import Generic, TypeVar

import torch

__all__ = ["Shards", "devices", "run"]

_S = TypeVar("_S")  # a shard: what one worker thread needs
_J = TypeVar("_J")  # a job
_R = TypeVar("_R")  # a job's result


def devices(entries: Sequence[str | torch.device]) -> list[torch.device]:
    """Q-106, Q-122, Q-186: one shard per entry of a non-empty sequence of device names, so
    a repeated entry is another shard on the same device. A string, an empty sequence or an
    entry that isn't a device is a ValueError."""
    if isinstance(entries, str):  # Q-122: a string is a sequence of characters
        raise ValueError(f"devices must be a sequence of device names, got the string {entries!r}")
    out = []
    for entry in entries:
        try:
            out.append(torch.device(entry))
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"devices holds {entry!r}, which isn't a device: {error}") from None
    if not out:
        raise ValueError("devices must name at least one device")
    return out


class Shards(Generic[_S, _J, _R]):
    """One worker thread per shard, taking jobs from a shared queue, as a context manager: the
    threads start on entry and are joined on exit. Results come back in the order they finish,
    with their job; a shard's exception is raised by `next_result`, and the shard that raised
    takes no more jobs."""

    def __init__(
        self,
        shards: Sequence[_S],
        jobs: Iterable[_J],
        work: Callable[[_S, _J], _R],
        *,
        name: str,
    ) -> None:
        self._jobs: queue.SimpleQueue[_J | None] = queue.SimpleQueue()
        for job in jobs:
            self._jobs.put(job)
        for _ in shards:
            self._jobs.put(None)  # one stop per worker, after every job
        self._results: queue.SimpleQueue[tuple[_J, _R | BaseException]] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._work = work
        self._threads = [
            threading.Thread(target=self._serve, args=(shard,), name=f"{name} {i}")
            for i, shard in enumerate(shards)
        ]

    def _serve(self, shard: _S) -> None:
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None:
                return
            try:
                result: _R | BaseException = self._work(shard, job)
            except BaseException as error:  # noqa: BLE001 - raised again in the caller's thread
                self._results.put((job, error))
                return
            self._results.put((job, result))

    def __enter__(self) -> "Shards[_S, _J, _R]":
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join()

    def next_result(self) -> tuple[_J, _R]:
        """The next job to finish and its result; a shard's exception is raised here."""
        job, result = self._results.get()
        if isinstance(result, BaseException):
            raise result
        return job, result


def run(
    shards: Sequence[_S], jobs: Sequence[_J], work: Callable[[_S, _J], _R], *, name: str
) -> list[_R]:
    """Every job's result, in the jobs' order, done by `work` on the shards' worker threads
    (named `name` and the shard's number). The first exception a shard raises is raised here,
    once every shard has stopped."""
    done: dict[int, _R] = {}

    def numbered(shard: _S, job: tuple[int, _J]) -> _R:
        return work(shard, job[1])

    with Shards(shards, list(enumerate(jobs)), numbered, name=name) as running:
        while len(done) < len(jobs):
            (index, _), result = running.next_result()
            done[index] = result
    return [done[index] for index in range(len(jobs))]

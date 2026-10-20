# SPDX-License-Identifier: Apache-2.0
"""A recorder of the host synchronizations torch's debug mode reports, with the Python stack and
thread behind each, for the tests of §2.4 rule 3 (U-29 in `test_synchronization.py`, and the
render test of `test_render_gsplat.py`). It needs `sys.monitoring` (Python 3.12).

How a synchronization is seen. The debug mode (`torch.cuda.set_sync_debug_mode("warn")`) raises
a C++ warning at every synchronizing CUDA call, and the warning reaches Python by one of three
routes (source log 158, 159):

- raised during a call through one of torch's Python bindings (any torch operation, or an
  autograd Function's `apply`), it becomes a Python warning when the call returns, from the
  frame that made the call, on the thread that made it. That includes the autograd device
  thread, when a Python backward calls a torch operation;
- raised in an extension function that Python calls directly (gsplat's compiled stage functions
  outside an autograd Function, such as the tile intersection of `isect_tiles`), it never becomes
  a Python warning: c10 prints it on standard error;
- raised by C++ code on the autograd device thread, it is replayed when backward returns, as a
  Python warning from torch's own backward frame, so where it came from is lost.

`SyncRecorder` takes the first route through `warnings.showwarning` and the second by sending
standard error to a file and, after each C function returns (`sys.monitoring`), charging what it
printed to the Python stack that called it. The third arrives as a Python warning whose innermost
frame is torch's, which the checks refuse whoever made it: conservative. Without the second
route an unpacked render, whose one synchronization is the tile intersection's, would show no
warning at all. The debug mode is a prototype that may miss some synchronizations; the tests pin
the ones it sees.
"""

import os
import sys
import tempfile
import threading
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any

import gsplat
import torch

import openftgs.render

SYNC_MESSAGE = "called a synchronizing CUDA operation"
GSPLAT_DIR = Path(gsplat.__file__).resolve().parent
_TORCH_DIR = Path(torch.__file__).resolve().parent
_RENDER_FILE = Path(openftgs.render.__file__).resolve()


def in_gsplat(filename: str) -> bool:
    """Whether a frame's file is one of gsplat's."""
    return Path(filename).resolve().is_relative_to(GSPLAT_DIR)


@dataclass(frozen=True)
class Sync:
    """One synchronization: the thread it happened on and the Python stack that made it,
    outermost frame first, so the last frame is the innermost, with each frame's qualified
    name (`Class.method` for a method)."""

    thread: int
    thread_name: str
    stack: tuple[traceback.FrameSummary, ...]
    qualnames: tuple[str, ...]
    route: str  # "python": a Python warning; "stderr": printed by C++ only

    @property
    def innermost(self) -> traceback.FrameSummary:
        return self.stack[-1]

    @property
    def innermost_qualname(self) -> str:
        return self.qualnames[-1]

    def describe(self, root: str) -> str:
        """The thread, the route and the stack from the outermost frame in the file `root` (a
        test file) inward."""
        ours = [k for k, f in enumerate(self.stack) if f.filename == root]
        start = ours[0] if ours else 0
        frames = zip(self.stack[start:], self.qualnames[start:], strict=True)
        lines = "".join(f"\n    {f.filename}:{f.lineno} in {name}" for f, name in frames)
        return f"[{self.thread_name}, {self.route}]{lines}"


def described(syncs: list[Sync], root: str) -> str:
    """Each distinct synchronization once, with how often it happened."""
    counts: dict[str, int] = {}
    for sync in syncs:
        text = sync.describe(root)
        counts[text] = counts.get(text, 0) + 1
    return "\n".join(f"{count} x {text}" for text, count in counts.items())


def in_render_gsplat_stage(sync: Sync) -> bool:
    """Whether a synchronization happened inside one of the gsplat stage calls of
    `openftgs.render` (§2.4 rule 3, D-76; Q-171): its innermost frame is gsplat's, and its stack
    runs through `render`'s `_render_gsplat` into a gsplat function, the first frame inside it
    that isn't torch's (a decorator such as `torch.no_grad`'s, or an autograd Function's
    `apply`) being gsplat's."""
    if not in_gsplat(sync.innermost.filename):
        return False
    for k in range(len(sync.stack) - 1):
        frame = sync.stack[k]
        if sync.qualnames[k] == "_render_gsplat" and Path(frame.filename).resolve() == _RENDER_FILE:
            inside = [f for f in sync.stack[k + 1 :] if not _in_torch(f.filename)]
            return bool(inside) and in_gsplat(inside[0].filename)
    return False


def _in_torch(filename: str) -> bool:
    return Path(filename).resolve().is_relative_to(_TORCH_DIR)


def _warnings_files() -> set[str]:
    """The files of the standard library's warnings machinery, which sit above the frame that
    raised a warning when showwarning runs."""
    names = ("warnings", "_py_warnings")
    return {m.__file__ for n in names if (m := sys.modules.get(n)) is not None and m.__file__}


def _stack(frame: FrameType | None) -> tuple[list[traceback.FrameSummary], list[str]]:
    """The Python stack from `frame` outward, outermost frame first, with the frames' qualified
    names."""
    frames: list[FrameType] = []
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    frames.reverse()
    summary = traceback.StackSummary.extract(((f, f.f_lineno) for f in frames), lookup_lines=False)
    return list(summary), [f.f_code.co_qualname for f in frames]


@dataclass
class SyncRecorder:
    """Every synchronization the debug mode reports while the recorder is active, with its full
    stack and thread, by the first two routes of the module docstring; `unattributed` counts what
    C++ printed after the last C function returned, which no stack can claim."""

    syncs: list[Sync] = field(default_factory=list)
    unattributed: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._seen = 0
        self._exit: list[Callable[[], None]] = []

    def _record(self, route: str, stack: list[traceback.FrameSummary], names: list[str]) -> None:
        thread = threading.current_thread()
        self.syncs.append(
            Sync(threading.get_ident(), thread.name, tuple(stack), tuple(names), route)
        )

    def _show(self, message: Any, *args: Any, **kwargs: Any) -> None:
        if SYNC_MESSAGE not in str(message):
            return
        stack, names = _stack(sys._getframe(1))  # without this hook
        machinery = _warnings_files()
        while stack and stack[-1].filename in machinery:
            stack.pop()
            names.pop()
        self._record("python", stack, names)

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
        for _ in range(printed.count(SYNC_MESSAGE.encode())):
            self._record("stderr", *_stack(sys._getframe(1)))  # the frame that made the call

    def _monitor(self) -> Callable[[], None]:
        if sys.version_info < (3, 12):  # the tests are skipped; this tells mypy why
            raise RuntimeError("sys.monitoring needs Python 3.12")
        monitoring = sys.monitoring
        tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
        events = monitoring.events
        monitoring.use_tool_id(tool, "openftgs sync recorder")
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

    def __enter__(self) -> "SyncRecorder":
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
        self.unattributed = rest.count(SYNC_MESSAGE.encode())
        self._capture.close()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        while self._exit:
            self._exit.pop()()

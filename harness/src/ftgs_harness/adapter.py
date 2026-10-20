# SPDX-License-Identifier: Apache-2.0
"""Calling an adapter (§3.3 adapter contract): one process per command, its output logged,
its outcome classified (Q-203, Q-211)."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ftgs_harness.settings import HarnessError

__all__ = ["Adapter", "Outcome"]

_OutcomeStatus = Literal["ok", "crash", "timeout"]


@dataclass(frozen=True)
class Outcome:
    """How an adapter call ended: `ok` (exit 0), `crash` (any other exit) or `timeout`."""

    status: _OutcomeStatus
    returncode: int | None
    seconds: float
    stdout: str
    log: str  # the log file, relative to the run directory

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def describe(self) -> str:
        """The outcome in input/output terms, for a "crash" symptom."""
        if self.status == "timeout":
            return f"the call did not finish within {self.seconds:.0f} s"
        return f"the call exited with code {self.returncode} after {self.seconds:.1f} s"


class Adapter:
    """One implementation's adapter, by its command line (split as a shell would), with its
    calls logged under `run_dir/logs/<label>/`."""

    def __init__(self, label: str, command: str, run_dir: Path, *, timeout: float) -> None:
        self.label = label
        self.command = command
        self.argv = shlex.split(command)
        if not self.argv:
            raise HarnessError(f"adapter {label}: empty command")
        if shutil.which(self.argv[0]) is None:
            raise HarnessError(f"adapter {label}: {self.argv[0]!r} is not an executable")
        self.run_dir = run_dir
        self.timeout = timeout
        self._count = 0
        self._lock = threading.Lock()

    def call(
        self, command: str, args: Sequence[str | Path], *, env: Mapping[str, str] | None = None
    ) -> Outcome:
        """Run `<adapter> <command> <args…>` and return its outcome. `env` adds or replaces
        environment variables (CUDA_VISIBLE_DEVICES for L3's runs, D-54)."""
        with self._lock:
            self._count += 1
            index = self._count
        logs = self.run_dir / "logs" / self.label
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{index:05d}-{command}.log"
        argv = [*self.argv, command, *(str(a) for a in args)]
        full_env = {**os.environ, **(env or {})}
        started = time.perf_counter()
        # A session of its own, so a timeout ends the adapter and everything it started.
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=full_env,
            start_new_session=True,
        )
        status: _OutcomeStatus
        code: int | None
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
            code = process.returncode
            status = "ok" if code == 0 else "crash"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            status, code = "timeout", None
        seconds = time.perf_counter() - started
        log.write_text(
            f"$ {shlex.join(argv)}\n# {status}, exit {code}, {seconds:.2f} s\n"
            f"--- stdout\n{stdout}\n--- stderr\n{stderr}\n"
        )
        return Outcome(status, code, seconds, stdout, str(log.relative_to(self.run_dir)))

    def info(self) -> tuple[Outcome, Any]:
        """The `info` reply as parsed JSON, or None when the call failed or its output isn't
        JSON."""
        outcome = self.call("info", [])
        if not outcome.ok:
            return outcome, None
        try:
            return outcome, json.loads(outcome.stdout)
        except ValueError:
            return outcome, None

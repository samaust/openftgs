# SPDX-License-Identifier: Apache-2.0
"""Running `ftgs-harness` as a subprocess, its real entry point, for the level tests."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
BIN = Path(sys.executable).parent
ADAPTER = str(BIN / "ftgs-adapter")


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCHINDUCTOR_COMPILE_THREADS",
)


def within_budget(jobs: int) -> tuple[int, dict[str, str]]:
    """`ftgs-harness --jobs` and the environment that keep a harness run inside this process's
    thread budget (machine.py's `--threads`, or an xdist worker's share, Q-214): at most one
    adapter process per two threads, each at its share."""
    value = os.environ.get("OMP_NUM_THREADS", "")
    budget = int(value) if value.isdigit() and int(value) > 0 else (os.cpu_count() or 2) // 2
    jobs = max(1, min(jobs, budget // 2))
    share = str(max(1, budget // jobs))
    return jobs, {**os.environ, **{name: share for name in _THREAD_VARIABLES}}


def openftgs(backend: str = "reference") -> str:
    return f"{ADAPTER} --backend {backend}"


def faulty(*faults: str, on: tuple[str, ...] = (), backend: str = "reference") -> str:
    """The fault-injecting wrapper over openftgs's adapter, as a command line."""
    argv = [sys.executable, str(HERE / "fault_adapter.py"), "--inner", openftgs(backend)]
    for fault in faults:
        argv += ["--fault", fault]
    for command in on:
        argv += ["--on", command]
    return shlex.join(argv)


def run_harness(
    tmp_path: Path,
    levels: str,
    *,
    a: str,
    b: str,
    settings: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    jobs: int = 4,
    out: str = "run",
) -> tuple[int, dict[str, Any]]:
    """Run `ftgs-harness run` and return its exit code and report.json (empty if none)."""
    (tmp_path / "harness_config.json").write_text(json.dumps(config or {}))
    (tmp_path / "settings.json").write_text(json.dumps(settings or {}))
    run_dir = tmp_path / out
    jobs, env = within_budget(jobs)
    done = subprocess.run(
        [
            str(BIN / "ftgs-harness"),
            "run",
            "--a",
            a,
            "--b",
            b,
            "--config",
            str(tmp_path / "harness_config.json"),
            "--settings",
            str(tmp_path / "settings.json"),
            "--levels",
            levels,
            "--out",
            str(run_dir),
            "--jobs",
            str(jobs),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if done.returncode == 2:
        print(done.stderr)
    return done.returncode, report


def checks(report: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in report.get("checks", []) if c["id"].startswith(prefix)}

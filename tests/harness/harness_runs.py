# SPDX-License-Identifier: Apache-2.0
"""Running `ftgs-harness` as a subprocess, its real entry point, for the level tests."""

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
BIN = Path(sys.executable).parent
ADAPTER = str(BIN / "ftgs-adapter")


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
    )
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if done.returncode == 2:
        print(done.stderr)
    return done.returncode, report


def checks(report: dict[str, Any], prefix: str = "") -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in report.get("checks", []) if c["id"].startswith(prefix)}

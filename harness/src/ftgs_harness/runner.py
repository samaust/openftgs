# SPDX-License-Identifier: Apache-2.0
"""A harness run: its context, the `info` checks, the levels in order, and `rerun` (§3.3;
Q-203, Q-205, Q-211)."""

import concurrent.futures
import datetime
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from ftgs_harness.adapter import Adapter
from ftgs_harness.checks import LEVELS, Check, Status
from ftgs_harness.formats import file_sha256
from ftgs_harness.info import Tier, choose_tier, info_problems
from ftgs_harness.report import build_report, write_report
from ftgs_harness.settings import (
    HarnessError,
    RunRecord,
    Settings,
    load_config,
    load_settings,
    read_run,
    write_run,
)

__all__ = ["Context", "harness_version", "rerun", "run"]

_T = TypeVar("_T")
_R = TypeVar("_R")

# The module that runs each level; each has `run(ctx) -> list[Check]`.
_LEVEL_MODULES = {
    "L0": "ftgs_harness.level0",
    "L1": "ftgs_harness.level1",
    "L2": "ftgs_harness.level2",
    "L3": "ftgs_harness.level3",
}


def harness_version() -> str:
    try:
        return importlib.metadata.version("ftgs-harness")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


@dataclass
class Context:
    """What a level needs: the adapters, the inputs, where cases and outputs go, the `info`
    replies and the L1 tier, and, for `rerun`, the one check to reproduce."""

    run_dir: Path  # the run directory: run.json, cases/, the report
    out: Path  # where adapter outputs go (RUN_DIR, or RUN_DIR/reruns/<id> for a rerun)
    a: Adapter
    b: Adapter
    config: dict[str, Any]
    config_path: Path
    settings: Settings
    jobs: int
    infos: dict[str, Any] = field(default_factory=dict)
    tier: Tier = "different"
    only: str | None = None

    @property
    def cases(self) -> Path:
        return self.run_dir / "cases"

    @property
    def adapters(self) -> dict[str, Adapter]:
        return {"A": self.a, "B": self.b}

    def wants(self, check_id: str) -> bool:
        """Whether a check (or a case, by its ID prefix) is needed: always, except in a rerun,
        which reproduces one check."""
        return self.only is None or self.only == check_id or self.only.startswith(check_id + ".")

    def repro(self, check_id: str) -> str:
        return f"ftgs-harness rerun {check_id} --out {self.run_dir}"

    def query_kinds(self, label: str) -> set[str]:
        info = self.infos.get(label)
        kinds = info.get("query_kinds") if isinstance(info, dict) else None
        return set(kinds) if isinstance(kinds, list) else set()

    def map(self, fn: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
        """`fn` over `items`, `jobs` at a time, in order."""
        items = list(items)
        if self.jobs <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with concurrent.futures.ThreadPoolExecutor(self.jobs) as pool:
            return list(pool.map(fn, items))


def _info_check(ctx: Context, label: str, level: str, outcome_text: str, info: Any) -> Check:
    check_id = f"{level}.info.{label}"
    problems = info_problems(info) if info is not None else [outcome_text]
    status: Status = "pass" if not problems else "fail"
    return Check(
        id=check_id,
        level=level,
        status=status,
        spec_refs=["D-73"],
        inputs={"adapter": label},
        metric="info_schema",
        expected="name, version, query_kinds, rasterizer and the 12 render_conventions entries",
        observed="complete" if status == "pass" else "; ".join(problems),
        symptom=""
        if status == "pass"
        else f"Adapter {label}'s info reply is incomplete: {'; '.join(problems)}.",
        pattern=None if status == "pass" else ("crash" if info is None else "schema violation"),
        repro=ctx.repro(check_id),
    )


def _infos(ctx: Context, level: str) -> list[Check]:
    checks = []
    for label, adapter in ctx.adapters.items():
        outcome, info = adapter.info()
        ctx.infos[label] = info
        text = outcome.describe() if not outcome.ok else "the reply is not JSON"
        if ctx.wants(f"{level}.info.{label}"):
            checks.append(_info_check(ctx, label, level, text, info))
    ctx.tier = choose_tier(ctx.infos.get("A"), ctx.infos.get("B"))
    return checks


def _level_runner(level: str) -> Callable[[Context], list[Check]]:
    try:
        module = importlib.import_module(_LEVEL_MODULES[level])
    except ModuleNotFoundError as error:
        raise HarnessError(f"{level} is not available in this harness") from error
    runner: Callable[[Context], list[Check]] = module.run
    return runner


def _check_levels(text: str) -> tuple[str, ...]:
    levels = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in levels if x not in LEVELS]
    if unknown or not levels or len(set(levels)) != len(levels):
        raise HarnessError(f"--levels must be distinct entries of {', '.join(LEVELS)}: {text!r}")
    return tuple(sorted(levels, key=LEVELS.index))


def _check_l3(settings: Settings) -> None:
    if not settings.scenes:
        raise HarnessError("L3 needs scenes: set l3.scenes in the run settings (--settings)")
    if importlib.util.find_spec("torch") is None:  # Q-254, Q-T391.2
        raise HarnessError(
            "L3 needs the extra ftgs-harness[torch] (torch), with which the harness computes "
            "ꟻLIP (E16): pip install 'ftgs-harness[torch]'"
        )


def _implementation(info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        return {"name": info.get("name"), "version": info.get("version"), "info": info}
    return {"name": None, "version": None, "info": {}}


def _run_levels(ctx: Context, levels: tuple[str, ...]) -> list[Check]:
    runners = {level: _level_runner(level) for level in levels}
    checks = _infos(ctx, levels[0])
    for level in levels:
        checks += runners[level](ctx)
    return checks


def run(
    *,
    a: str,
    b: str,
    config: Path,
    levels: str,
    out: Path,
    settings_path: Path | None,
    jobs: int,
    timeout: float,
) -> int:
    """`ftgs-harness run`: exit 0 when no check failed, 1 when one did (§3.3)."""
    chosen = _check_levels(levels)
    cfg, cfg_sha = load_config(config)
    settings = load_settings(settings_path)
    if "L3" in chosen:
        _check_l3(settings)
    if jobs < 1 or not timeout > 0:
        raise HarnessError("--jobs must be ≥ 1 and --timeout > 0")
    out = out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise HarnessError(f"--out {out} must be absent or an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        a=a,
        b=b,
        config=config.resolve(),
        config_sha256=cfg_sha,
        settings_path=None if settings_path is None else settings_path.resolve(),
        settings_sha256=None if settings_path is None else file_sha256(settings_path),
        levels=chosen,
        jobs=jobs,
        timeout=timeout,
        started=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        harness_version=harness_version(),
    )
    ctx = Context(
        run_dir=out,
        out=out,
        a=Adapter("A", a, out, timeout=timeout),
        b=Adapter("B", b, out, timeout=timeout),
        config=cfg,
        config_path=config.resolve(),
        settings=settings,
        jobs=jobs,
    )
    write_run(out / "run.json", record)
    checks = _run_levels(ctx, chosen)
    report = build_report(
        run={
            "started": record.started,
            "harness_version": record.harness_version,
            "config_sha256": cfg_sha,
            "levels": list(chosen),
            "tier": ctx.tier,
            "settings": settings.to_json(),
        },
        implementations={
            "A": _implementation(ctx.infos.get("A")),
            "B": _implementation(ctx.infos.get("B")),
        },
        levels=chosen,
        checks=checks,
    )
    write_report(out, report)
    return 1 if report["summary"]["failed_checks"] else 0


def rerun(check_id: str, run_dir: Path) -> int:
    """`ftgs-harness rerun`: reproduce one check of the run in `run_dir`, into
    `run_dir/reruns/<id>/`, print it, and exit 0 or 1 by its status (Q-203)."""
    run_dir = run_dir.resolve()
    record = read_run(run_dir / "run.json")
    level = check_id.split(".")[0]
    if level not in record.levels:
        raise HarnessError(f"{check_id}: level {level} was not part of the run in {run_dir}")
    config, config_sha = load_config(record.config)
    settings = load_settings(record.settings_path)
    settings_sha = None if record.settings_path is None else file_sha256(record.settings_path)
    # §3.3 symptom rule 6: the reproduction runs on the exact input files of the run.
    for name, path, then, now in (
        ("harness_config", record.config, record.config_sha256, config_sha),
        ("settings", record.settings_path, record.settings_sha256, settings_sha),
    ):
        if now != then:
            raise HarnessError(
                f"{name} {path} has changed since the run (SHA-256 {then} then, {now} now)"
            )
    if level == "L3":
        _check_l3(settings)
    out = run_dir / "reruns" / check_id
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        run_dir=run_dir,
        out=out,
        a=Adapter("A", record.a, out, timeout=record.timeout),
        b=Adapter("B", record.b, out, timeout=record.timeout),
        config=config,
        config_path=record.config,
        settings=settings,
        jobs=record.jobs,
        only=check_id,
    )
    checks = _infos(ctx, record.levels[0])
    if ".info." not in check_id:
        checks += _level_runner(level)(ctx)
    found = [c for c in checks if c.id == check_id]
    if not found:
        raise HarnessError(f"{check_id}: no such check in the run in {run_dir}")
    body = found[0].to_json()
    (out / "check.json").write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(json.dumps(body, indent=2, allow_nan=False) + "\n")
    return 1 if found[0].status == "fail" else 0

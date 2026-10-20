# SPDX-License-Identifier: Apache-2.0
"""The GPU CI's regression check (§3.4; Q-170, Q-179, Q-196): the BENCH.json files of one run
of B-2 to B-6 against the baselines stored for the run's GPU model.

    python scripts/bench_compare.py RUN_DIR [--baselines DIR] [--update]

Every `*.json` file in RUN_DIR is one BENCH.json of `ftgs bench`. Its baseline is the file of the
same name in `DIR/<gpu>/`, where <gpu> is the GPU model the run recorded, as a slug ("NVIDIA
GeForce RTX 4090" gives `nvidia-geforce-rtx-4090`); DIR defaults to `benchmarks/baselines`.

- B-2, B-3, B-4 and B-5 (Q-170): each timed case's median time fails when it is more than 10 %
  above the baseline's, `value > 1.10 × baseline`: B-2's and B-3's and B-4's `forward_backward_s`
  per combination, size and implementation, or variant, and B-5's `step_s` per kind and size.
- B-6 (Q-170): each budgeted data share fails unless it is under 1 % (§3.4), whatever the
  baseline's; the baseline names the cases that must be budgeted.
- Also failures: a run without a baseline, a baseline of that GPU without a run, a baseline
  recorded on another GPU model or for another benchmark, a scene of another size, and a case
  measured on one side only (Q-196).

Nothing else is judged: first calls, peak memory, the other figures, and the `limit` entries of
`budgets`, which hold equivalence checks rather than speed (Q-196). Differences in driver,
CUDA, PyTorch or gsplat are reported, not judged.

`--update` copies the runs into their GPU's folder instead: how the baselines are refreshed.

Exit 0 when every check passes, 1 when one fails, 2 for invalid input.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_FORMAT, _VERSION = "ftgs-bench", 1  # Q-166
_REGRESSION = 1.10  # §3.4: "more than 10% worse than its stored baseline" fails (Q-170)
_DATA_SHARE_LIMIT = 0.01  # §3.4 B-6: "< 1%"
_DEFAULT_BASELINES = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines"
# Q-170, Q-196: per benchmark, the median time compared and the case keys that mark a timed
# case; the other cases (B-2's counts, B-3's recompilations, B-4's deviations and comparisons,
# B-5's ratios) hold no time of their own.
_TIMED: dict[str, tuple[str, frozenset[str]]] = {
    "B-2": ("forward_backward_s", frozenset({"packed", "precull"})),
    "B-3": ("forward_backward_s", frozenset({"impl", "n"})),
    "B-4": ("forward_backward_s", frozenset({"loss"})),
    "B-5": ("step_s", frozenset({"adam"})),
}
_BUDGETED = {"B-6": "data_share"}  # Q-170: judged against §3.4's absolute budget
_SCENE_SIZE = ("cameras", "frames", "width", "height")
_REPORTED = ("driver", "cuda", "cudnn", "torch", "gsplat", "fused_ssim", "python")


class InputError(Exception):
    """Invalid input: exit 2."""


@dataclass
class Report:
    """Every check made, in order, and the failures among them."""

    lines: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def check(self, ok: bool, text: str) -> None:
        self.lines.append(f"  {'ok  ' if ok else 'FAIL'}  {text}")
        if not ok:
            self.failures.append(text)

    def fail(self, text: str) -> None:
        self.check(False, text)

    def note(self, text: str) -> None:
        self.lines.append(f"  note  {text}")


def gpu_slug(name: str) -> str:
    """A GPU model as a folder name: lower case, every run of other characters than letters and
    digits a single hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load(path: Path) -> dict[str, Any]:
    """A BENCH.json of B-2 to B-6 (Q-166). InputError for anything else."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise InputError(f"{path}: not a readable JSON file: {err}") from None
    if not isinstance(body, dict) or body.get("format") != _FORMAT:
        raise InputError(f"{path}: not a BENCH.json (format {_FORMAT!r})")
    if body.get("version") != _VERSION:
        raise InputError(f"{path}: BENCH.json version {body.get('version')!r}, not {_VERSION}")
    benchmark = body.get("benchmark")
    if benchmark not in _TIMED and benchmark not in _BUDGETED:
        raise InputError(f"{path}: {benchmark} is not one of the CI's B-2 to B-6 (§3.4)")
    for key in ("environment", "scene"):
        if not isinstance(body.get(key), dict):
            raise InputError(f"{path}: {key} must be an object")
    for key in ("cases", "budgets"):
        if not isinstance(body.get(key), list):
            raise InputError(f"{path}: {key} must be a list")
    return body


def _gpu(body: dict[str, Any]) -> str | None:
    gpu = body["environment"].get("gpu")
    name = gpu.get("name") if isinstance(gpu, dict) else None
    return name if isinstance(name, str) and name else None


def _key(case: Any) -> str:
    """A case's parameters as one canonical string, so cases match exactly (Q-196)."""
    return json.dumps(case, sort_keys=True)


def _timed(body: dict[str, Any]) -> dict[str, float]:
    """The compared median time of every timed case (Q-170), by case."""
    figure, marks = _TIMED[body["benchmark"]]
    times: dict[str, float] = {}
    for entry in body["cases"]:
        case, figures = entry.get("case"), entry.get("figures")
        if isinstance(case, dict) and marks <= case.keys() and isinstance(figures, dict):
            value = figures.get(figure)
            if isinstance(value, int | float) and not isinstance(value, bool):
                times[_key(case)] = float(value)
    return times


def _budgeted(body: dict[str, Any]) -> dict[str, float]:
    """The value of every budget entry of the judged figure (§3.4 B-6), by case."""
    figure = _BUDGETED[body["benchmark"]]
    values: dict[str, float] = {}
    for entry in body["budgets"]:
        if entry.get("kind") == "budget" and entry.get("figure") == figure:
            value = entry.get("value")
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise InputError(f"a {figure} budget entry without a number: {entry}")
            values[_key(entry.get("case"))] = float(value)
    return values


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.3f} ms"


def _compare_times(run: dict[str, Any], baseline: dict[str, Any], report: Report) -> None:
    """Q-170: each timed case against 1.10 × its baseline; a case on one side only fails."""
    figure = _TIMED[run["benchmark"]][0]
    ours, theirs = _timed(run), _timed(baseline)
    if not ours:
        report.fail(f"no {figure} in the run")
    for case, value in ours.items():
        if case not in theirs:
            report.fail(f"{case} {figure} {_ms(value)}: no baseline for this case")
            continue
        base = theirs[case]
        change = f"{(value / base - 1) * 100:+.1f} %" if base > 0 else "baseline 0"
        report.check(
            value <= _REGRESSION * base,
            f"{case} {figure} {_ms(value)} against {_ms(base)} ({change}, limit +10 %)",
        )
    for case in theirs.keys() - ours.keys():
        report.fail(f"{case} {figure}: in the baseline, not measured by the run")


def _judge_budget(run: dict[str, Any], baseline: dict[str, Any], report: Report) -> None:
    """Q-170: every budgeted figure under §3.4's absolute limit; the baseline's budgeted cases
    must all be budgeted in the run, so a case the run skipped is found."""
    figure = _BUDGETED[run["benchmark"]]
    ours, theirs = _budgeted(run), _budgeted(baseline)
    if not ours:
        report.fail(f"no budgeted {figure} in the run")
    for case, value in ours.items():
        report.check(
            value < _DATA_SHARE_LIMIT,
            f"{case} {figure} {value * 100:.3f} % (budget < 1 %; baseline "
            + (f"{theirs[case] * 100:.3f} %)" if case in theirs else "none)"),
        )
    for case in theirs.keys() - ours.keys():
        report.fail(f"{case} {figure}: budgeted in the baseline, not in the run")


def compare_one(run: dict[str, Any], baseline: dict[str, Any], report: Report) -> None:
    """One run against its baseline: the GPU model, the benchmark and the scene's size first."""
    gpu, base_gpu = _gpu(run), _gpu(baseline)
    if base_gpu != gpu:
        report.fail(f"the baseline was recorded on {base_gpu}, the run on {gpu}")
        return
    if baseline["benchmark"] != run["benchmark"]:
        report.fail(f"the baseline is of {baseline['benchmark']}, the run of {run['benchmark']}")
        return
    size = {k: run["scene"].get(k) for k in _SCENE_SIZE}
    base_size = {k: baseline["scene"].get(k) for k in _SCENE_SIZE}
    if size != base_size:
        report.fail(f"the run's scene is {size}, the baseline's {base_size}")
        return
    if run["scene"].get("sha256") != baseline["scene"].get("sha256"):
        report.note("the scene manifest's SHA-256 differs from the baseline's")
    for key in _REPORTED:
        ours, theirs = run["environment"].get(key), baseline["environment"].get(key)
        if ours != theirs:
            report.note(f"{key} {ours}, the baseline's {theirs}")
    if run["benchmark"] in _TIMED:
        _compare_times(run, baseline, report)
    else:
        _judge_budget(run, baseline, report)


def _runs(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not run_dir.is_dir():
        raise InputError(f"{run_dir}: not a folder")
    runs = [(path, load(path)) for path in sorted(run_dir.glob("*.json"))]
    if not runs:
        raise InputError(f"{run_dir}: holds no BENCH.json")
    return runs


def compare(run_dir: Path, baselines: Path) -> Report:
    """Every run in `run_dir` against its baseline in `baselines/<gpu>/`, then every baseline
    of those GPUs without a run."""
    report = Report()
    folders: set[Path] = set()
    compared: set[Path] = set()
    for path, run in _runs(run_dir):
        gpu = _gpu(run)
        report.lines.append(f"{path.name} ({run['benchmark']}, {gpu})")
        if gpu is None:
            report.fail("the run recorded no GPU")
            continue
        folder = baselines / gpu_slug(gpu)
        stored = folder / path.name
        folders.add(folder)
        compared.add(stored)
        if not stored.is_file():
            report.fail(f"no baseline {stored} for {gpu}")
            continue
        compare_one(run, load(stored), report)
    for folder in sorted(folders):
        for stored in sorted(folder.glob("*.json")):
            if stored not in compared:
                report.lines.append(stored.name)
                report.fail(f"the baseline {stored} has no run")
    return report


def update(run_dir: Path, baselines: Path) -> list[Path]:
    """Store every run in `run_dir` as its GPU's baseline (§3.4: "its stored baseline")."""
    written: list[Path] = []
    for path, run in _runs(run_dir):
        gpu = _gpu(run)
        if gpu is None:
            raise InputError(f"{path}: the run recorded no GPU")
        target = baselines / gpu_slug(gpu) / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("run_dir", type=Path, metavar="RUN_DIR")
    parser.add_argument("--baselines", type=Path, default=_DEFAULT_BASELINES, metavar="DIR")
    parser.add_argument("--update", action="store_true", help="store the runs as the baselines")
    args = parser.parse_args(argv)
    try:
        if args.update:
            for target in update(args.run_dir, args.baselines):
                print(f"stored {target}")
            return 0
        report = compare(args.run_dir, args.baselines)
    except InputError as err:
        print(f"bench_compare: {err}", file=sys.stderr)
        return 2
    print("\n".join(report.lines))
    checks = sum(1 for line in report.lines if line.startswith(("  ok", "  FAIL")))
    print(f"{checks} checks, {len(report.failures)} failed")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())

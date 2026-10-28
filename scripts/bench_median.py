# SPDX-License-Identifier: Apache-2.0
"""The candidate baselines of a refresh (§3.4; Q-170, Q-196, Q-245; #369): the per-case
median of an odd number of runs of B-2 to B-6.

    python scripts/bench_median.py RUN_DIR RUN_DIR RUN_DIR [...] --out DIR

Each RUN_DIR holds the BENCH.json files of one run of the GPU job's commands, as
`scripts/bench_compare.py` takes them. For every file they share, the candidate is the first
run's file (in the order of the arguments) in which each judged case, the figure the comparison
reads (Q-170, Q-196: `forward_backward_s` of B-2 to B-4, `step_s` of B-5, B-6's budgeted
`data_share`), takes its `cases` and `budgets` entries from the run whose figure is the median.
The count of runs is odd, so the median is one run's measurement and every figure stored with
it comes from that measurement. The cases nothing judges, the environment, the scene, the date
and the setup are the first run's (Q-245).

Invalid input, exit 2 and nothing written: an even count or fewer than three runs, run folders
holding different files, runs of one file that differ in benchmark, GPU model, scene size or
scene SHA-256, or in the cases they judge, and an output folder that already holds BENCH.json
files. Exit 0 once the candidates are written; the report prints each judged case's figure in
every run, its median and the spread, (max − min) / median.

`python scripts/bench_compare.py DIR` then compares the candidates with the stored baselines,
and `--update` stores them, only once the owner has decided to (Q-196; #369).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import bench_compare as bc
from bench_compare import InputError


def _judged(body: dict[str, Any]) -> dict[str, float]:
    """The figure the comparison reads of every judged case (Q-170), by case."""
    if body["benchmark"] in bc._TIMED:
        return bc._timed(body)
    return bc._budgeted(body)


def _entries(entries: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [e for e in entries if isinstance(e, dict) and bc._key(e.get("case")) == key]


def _check(name: str, bodies: list[dict[str, Any]]) -> None:
    """The runs of one file measured the same thing: InputError otherwise (Q-245)."""
    first = bodies[0]
    for body in bodies[1:]:
        if body["benchmark"] != first["benchmark"]:
            raise InputError(f"{name}: of {first['benchmark']} in one run, {body['benchmark']}")
        if bc._gpu(first) is None or bc._gpu(body) != bc._gpu(first):
            raise InputError(f"{name}: recorded on {bc._gpu(first)} and {bc._gpu(body)}")
        for key in (*bc._SCENE_SIZE, "sha256"):
            if body["scene"].get(key) != first["scene"].get(key):
                raise InputError(f"{name}: the runs' scenes differ in {key}")
    figures = [_judged(body) for body in bodies]
    if not figures[0]:
        raise InputError(f"{name}: no judged case")
    if any(f.keys() != figures[0].keys() for f in figures):
        raise InputError(f"{name}: the runs judge different cases")
    for key in figures[0]:
        for field in ("cases", "budgets"):
            counts = {len(_entries(body[field], key)) for body in bodies}
            if len(counts) != 1:
                raise InputError(f"{name}: {key} has different {field} entries in the runs")


def _summary(bodies: list[dict[str, Any]]) -> list[str]:
    """Each judged case's figure in every run, in the order of the arguments, its median and
    the spread between the runs, (max − min) / median."""
    benchmark = bodies[0]["benchmark"]
    timed = benchmark in bc._TIMED
    figure = bc._TIMED[benchmark][0] if timed else bc._BUDGETED[benchmark]
    figures = [_judged(body) for body in bodies]
    lines = []
    for key in figures[0]:
        values = [f[key] for f in figures]
        middle = sorted(values)[len(values) // 2]
        text = [bc._ms(v) if timed else f"{v * 100:.3f} %" for v in (*values, middle)]
        spread = f"{(max(values) - min(values)) / middle * 100:.1f} %" if middle > 0 else "n/a"
        lines.append(
            f"  {key} {figure}: {' / '.join(text[:-1])}, median {text[-1]}, spread {spread}"
        )
    return lines


def median_file(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    """One BENCH.json from the same file of every run: the first run's, each judged case's
    entries replaced by those of the run whose judged figure is the median (Q-245)."""
    out = copy.deepcopy(bodies[0])  # Q-245: the first run's file, judged cases replaced
    figures = [_judged(body) for body in bodies]
    chosen: dict[str, dict[str, Any]] = {}
    for key in figures[0]:
        order = sorted(range(len(bodies)), key=lambda k: figures[k][key])
        chosen[key] = bodies[order[len(order) // 2]]
    for field in ("cases", "budgets"):
        merged: list[dict[str, Any]] = []
        taken: dict[str, int] = {}
        for entry in out[field]:
            key = bc._key(entry.get("case")) if isinstance(entry, dict) else ""
            if key in chosen:
                k = taken.get(key, 0)
                merged.append(copy.deepcopy(_entries(chosen[key][field], key)[k]))
                taken[key] = k + 1
            else:
                merged.append(entry)
        out[field] = merged
    return out


def write_medians(run_dirs: list[Path], out: Path) -> list[str]:
    """The median of every BENCH.json the run folders share, written into `out`; returns the
    report to print, each file with its cases' figures."""
    if len(run_dirs) < 3 or len(run_dirs) % 2 == 0:
        raise InputError(f"{len(run_dirs)} runs: a median takes an odd number, at least 3")
    if out.exists() and (not out.is_dir() or any(out.glob("*.json"))):
        raise InputError(f"{out}: not an empty folder for the candidates")
    runs = [dict((path.name, body) for path, body in bc._runs(d)) for d in run_dirs]
    if any(run.keys() != runs[0].keys() for run in runs):
        raise InputError("the run folders hold different BENCH.json files")
    medians, report = {}, []
    for name in sorted(runs[0]):
        bodies = [run[name] for run in runs]
        _check(name, bodies)
        medians[name] = median_file(bodies)
        report += [f"{name} ({bodies[0]['benchmark']}, {bc._gpu(bodies[0])})", *_summary(bodies)]
    out.mkdir(parents=True, exist_ok=True)
    for name, body in medians.items():
        target = out / name
        # As `ftgs bench` writes BENCH.json (Q-25, Q-166).
        target.write_text(json.dumps(body, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        report.append(f"wrote {target}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    args = parser.parse_args(argv)
    try:
        print("\n".join(write_medians(args.run_dirs, args.out)))
    except InputError as err:
        print(f"bench_median: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

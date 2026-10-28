# SPDX-License-Identifier: Apache-2.0
"""The candidate baselines of Q-196's refresh, `scripts/bench_median.py` (§3.4; Q-170, Q-196,
Q-245; #369): the per-case median of an odd number of runs of B-2 to B-6.

Added tests, not §3 rows: each judged case of the output is the whole entry of the run whose
judged figure is the middle one; the rest of each file is the first run's; runs that disagree
on what was measured are invalid input. The script is called at its command line, `main(argv)`,
on BENCH.json files written here, and its output is read back with the comparison's loader.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RTX_4090 = "NVIDIA GeForce RTX 4090"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load("bench_compare")
median = _load("bench_median")


def _bench(
    benchmark: str,
    cases: list[dict[str, Any]],
    budgets: list[dict[str, Any]] | None = None,
    *,
    gpu: str = RTX_4090,
    frames: int = 300,
    date: str = "2026-10-23T00:00:00+00:00",
) -> dict[str, Any]:
    """A BENCH.json as `ftgs bench` writes it (Q-166), with the fields the scripts read."""
    return {
        "format": "ftgs-bench",
        "version": 1,
        "benchmark": benchmark,
        "command": ["ftgs", "bench", "x", "scene.json", "--out", "x.json"],
        "environment": {"gpu": {"name": gpu, "count": 1}, "torch": "2.14.1+cu130"},
        "scene": {
            "path": "/ci/scene.json",
            "sha256": "0" * 64,
            "cameras": 20,
            "frames": frames,
            "width": 1352,
            "height": 1014,
        },
        "date": date,
        "setup": {},
        "cases": cases,
        "budgets": budgets or [],
    }


def _adam(fused: float, foreach: float, *, ratio: float = 0.45, **kw: Any) -> dict[str, Any]:
    """B-5 (Q-188): two timed kinds at 1M rows, each with a second figure that tells the runs
    apart, and an untimed ratio case."""
    return _bench(
        "B-5",
        [
            {
                "case": {"adam": "fused", "n": 1_000_000},
                "figures": {"step_s": fused, "first_call_s": fused * 100},
            },
            {
                "case": {"adam": "foreach", "n": 1_000_000},
                "figures": {"step_s": foreach, "first_call_s": foreach * 100},
            },
            {"case": {"n": 1_000_000}, "figures": {"fused_over_foreach": ratio}},
        ],
        **kw,
    )


def _write(folder: Path, name: str, body: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(body), encoding="utf-8")


def _run(tmp_path: Path, runs: list[dict[str, dict[str, Any]]]) -> int:
    """Writes each run ({name: body}) into its own folder, then calls the script with `out`."""
    folders = []
    for k, files in enumerate(runs):
        folder = tmp_path / f"run-{k}"
        for name, body in files.items():
            _write(folder, name, body)
        folders.append(str(folder))
    return int(median.main([*folders, "--out", str(tmp_path / "out")]))


def _out(tmp_path: Path, name: str) -> dict[str, Any]:
    body: dict[str, Any] = compare.load(tmp_path / "out" / name)
    return body


def _case(body: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [e for e in body["cases"] if e["case"] == case]
    (entry,) = entries
    return entry


def test_timed_case_is_the_middle_runs_entry(tmp_path: Path) -> None:
    """Q-245: per timed case, the run whose step_s is the median gives the whole entry, so
    the stored figures are one real measurement; the cases pick their runs independently."""
    runs = [
        {"B-5-adam.json": _adam(3e-3, 1e-3)},
        {"B-5-adam.json": _adam(1e-3, 3e-3)},
        {"B-5-adam.json": _adam(2e-3, 2.5e-3)},
    ]
    assert _run(tmp_path, runs) == 0
    out = _out(tmp_path, "B-5-adam.json")
    assert _case(out, {"adam": "fused", "n": 1_000_000})["figures"] == {
        "step_s": 2e-3,
        "first_call_s": 2e-3 * 100,
    }
    assert _case(out, {"adam": "foreach", "n": 1_000_000})["figures"] == {
        "step_s": 2.5e-3,
        "first_call_s": 2.5e-3 * 100,
    }


def _data(share: float, iteration_s: float) -> dict[str, Any]:
    """B-6 (Q-189): a memory-mapped case with its budget entry, and an `off` case without."""
    case = {"n": 1_000_000, "placement": "mmap", "prefetch": 8}
    budget = {
        "case": case,
        "figure": "data_share",
        "limit": 0.01,
        "value": share,
        "pass": True,
        "kind": "budget",
    }
    return _bench(
        "B-6",
        [
            {"case": case, "figures": {"data_share": share, "iteration_mean_s": iteration_s}},
            {"case": {"n": 1_000_000, "placement": "off"}, "figures": {"data_share": 0.02}},
        ],
        [budget],
    )


def test_budgeted_case_and_its_budget_come_from_one_run(tmp_path: Path) -> None:
    """Q-245: B-6 is judged on its budgeted data share (Q-170), so the run with the median
    share gives both that case's entry and its budget entry."""
    runs = [
        {"B-6-data.json": _data(4e-4, 0.101)},
        {"B-6-data.json": _data(3e-4, 0.103)},
        {"B-6-data.json": _data(5e-4, 0.102)},
    ]
    assert _run(tmp_path, runs) == 0
    out = _out(tmp_path, "B-6-data.json")
    case = {"n": 1_000_000, "placement": "mmap", "prefetch": 8}
    assert _case(out, case)["figures"] == {"data_share": 4e-4, "iteration_mean_s": 0.101}
    assert [b["value"] for b in out["budgets"]] == [4e-4]


def test_the_rest_is_the_first_runs(tmp_path: Path) -> None:
    """Q-245: the cases nothing judges and the file's other fields are the first run's, in
    the order of the arguments, and the output keeps the cases' order."""
    runs = [
        {"B-5-adam.json": _adam(2e-3, 2e-3, ratio=0.40, date="2026-10-23T01:00:00+00:00")},
        {"B-5-adam.json": _adam(1e-3, 1e-3, ratio=0.50, date="2026-10-23T02:00:00+00:00")},
        {"B-5-adam.json": _adam(3e-3, 3e-3, ratio=0.60, date="2026-10-23T03:00:00+00:00")},
    ]
    assert _run(tmp_path, runs) == 0
    out = _out(tmp_path, "B-5-adam.json")
    assert out["date"] == "2026-10-23T01:00:00+00:00"
    assert _case(out, {"n": 1_000_000})["figures"] == {"fused_over_foreach": 0.40}
    assert [e["case"] for e in out["cases"]] == [
        e["case"] for e in runs[0]["B-5-adam.json"]["cases"]
    ]


def test_prints_each_cases_runs_and_spread(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """For the owner's decision: each judged case's figure in every run, its median and the
    spread between the runs, (max − min) / median."""
    runs = [
        {"B-5-adam.json": _adam(2.0e-3, 1e-3)},
        {"B-5-adam.json": _adam(2.2e-3, 1e-3)},
        {"B-5-adam.json": _adam(1.8e-3, 1e-3)},
    ]
    assert _run(tmp_path, runs) == 0
    (line,) = [x for x in capsys.readouterr().out.splitlines() if '"fused"' in x]
    assert "2.000 ms / 2.200 ms / 1.800 ms" in line
    assert "median 2.000 ms" in line
    assert "spread 20.0 %" in line


def test_output_compares_with_the_runs(tmp_path: Path) -> None:
    """Q-196: the candidate files are BENCH.json files the comparison takes, and as baselines
    every run within 10 % of the median passes against them."""
    runs = [
        {"B-5-adam.json": _adam(2.0e-3, 2.0e-3), "B-6-data.json": _data(4e-4, 0.1)},
        {"B-5-adam.json": _adam(2.1e-3, 1.9e-3), "B-6-data.json": _data(3e-4, 0.1)},
        {"B-5-adam.json": _adam(1.9e-3, 2.1e-3), "B-6-data.json": _data(5e-4, 0.1)},
    ]
    assert _run(tmp_path, runs) == 0
    baselines = tmp_path / "baselines" / "nvidia-geforce-rtx-4090"
    for name in ("B-5-adam.json", "B-6-data.json"):
        _write(baselines, name, _out(tmp_path, name))
    for k in range(3):
        argv = [str(tmp_path / f"run-{k}"), "--baselines", str(tmp_path / "baselines")]
        assert compare.main(argv) == 0


@pytest.mark.parametrize("count", [2, 4])
def test_even_count_is_invalid(tmp_path: Path, count: int) -> None:
    """Q-245: an even number of runs has no middle run; exit 2."""
    runs = [{"B-5-adam.json": _adam(1e-3, 1e-3)} for _ in range(count)]
    assert _run(tmp_path, runs) == 2
    assert not (tmp_path / "out").exists()


def test_single_run_is_invalid(tmp_path: Path) -> None:
    """Q-245: a median needs at least three runs; exit 2."""
    assert _run(tmp_path, [{"B-5-adam.json": _adam(1e-3, 1e-3)}]) == 2


def _without_foreach(body: dict[str, Any]) -> dict[str, Any]:
    body["cases"] = [e for e in body["cases"] if e["case"].get("adam") != "foreach"]
    return body


@pytest.mark.parametrize(
    "third",
    [
        pytest.param({"B-5-adam.json": _adam(1e-3, 1e-3, gpu="NVIDIA H100")}, id="gpu"),
        pytest.param({"B-5-adam.json": _adam(1e-3, 1e-3, frames=60)}, id="scene-size"),
        pytest.param({"B-5-adam.json": _without_foreach(_adam(1e-3, 1e-3))}, id="case-set"),
        pytest.param({"B-5-adam.json": {**_adam(1e-3, 1e-3), "benchmark": "B-2"}}, id="bench"),
        pytest.param({"B-6-data.json": _data(1e-4, 0.1)}, id="file-names"),
    ],
)
def test_runs_that_disagree_are_invalid(tmp_path: Path, third: dict[str, Any]) -> None:
    """Q-245: runs of another GPU model, scene size, case set, benchmark or set of files
    measured something else, so they have no median; exit 2 and nothing written."""
    runs = [{"B-5-adam.json": _adam(1e-3, 1e-3)}, {"B-5-adam.json": _adam(2e-3, 2e-3)}, third]
    assert _run(tmp_path, runs) == 2
    assert not (tmp_path / "out").exists()


def test_scene_sha256_must_agree(tmp_path: Path) -> None:
    """Q-245: the runs of one refresh are of the same scene manifest; exit 2 otherwise."""
    third = _adam(1e-3, 1e-3)
    third["scene"]["sha256"] = "1" * 64
    runs = [{"B-5-adam.json": _adam(1e-3, 1e-3)}, {"B-5-adam.json": _adam(2e-3, 2e-3)}]
    assert _run(tmp_path, [*runs, {"B-5-adam.json": third}]) == 2


def test_out_with_bench_files_is_invalid(tmp_path: Path) -> None:
    """Q-245: the output folder never mixes candidates with older files; exit 2."""
    _write(tmp_path / "out", "B-2-raster.json", _adam(1e-3, 1e-3))
    runs = [{"B-5-adam.json": _adam(k * 1e-3, 1e-3)} for k in (1, 2, 3)]
    assert _run(tmp_path, runs) == 2
    assert not (tmp_path / "out" / "B-5-adam.json").exists()

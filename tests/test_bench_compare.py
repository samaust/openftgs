# SPDX-License-Identifier: Apache-2.0
"""The GPU CI's baseline comparison, `scripts/bench_compare.py` (§3.4: "fails when a budgeted
figure gets more than 10% worse than its stored baseline"; Q-170, Q-179, Q-196; #235).

Added tests, not §3 rows (PRD #220, "The baseline comparison script"): 9.9 % worse passes and
10.1 % worse fails; a missing baseline fails; a GPU-model mismatch fails; B-6 is judged against
its absolute < 1 %. The script is called at its command line, `main(argv)`, on BENCH.json files
written here. Last, the GPU job's workflow file is read for Q-179's trigger and runner.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_compare.py"
RTX_4090 = "NVIDIA GeForce RTX 4090"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_compare", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load()


def _bench(
    benchmark: str,
    cases: list[dict[str, Any]],
    budgets: list[dict[str, Any]] | None = None,
    *,
    gpu: str | None = RTX_4090,
    frames: int = 300,
) -> dict[str, Any]:
    """A BENCH.json as `ftgs bench` writes it (Q-166), with the fields the comparison reads."""
    return {
        "format": "ftgs-bench",
        "version": 1,
        "benchmark": benchmark,
        "command": ["ftgs", "bench", "x", "scene.json", "--out", "x.json"],
        "environment": {
            "gpu": {"name": gpu, "count": 1 if gpu else 0},
            "driver": "595.99.02",
            "cuda": "13.0",
            "torch": "2.14.1+cu130",
            "gsplat": "1.5.3",
        },
        "scene": {
            "path": "/ci/scene.json",
            "sha256": "0" * 64,
            "cameras": 20,
            "frames": frames,
            "width": 1352,
            "height": 1014,
        },
        "date": "2026-10-19T00:00:00+00:00",
        "setup": {},
        "cases": cases,
        "budgets": budgets or [],
    }


def _adam(step_s: float, *, kind: str = "fused") -> dict[str, Any]:
    """B-5 (Q-188): one kind at 1M rows, with its median step time."""
    return _bench(
        "B-5",
        [
            {"case": {"adam": kind, "n": 1_000_000}, "figures": {"step_s": step_s}},
            {"case": {"n": 1_000_000}, "figures": {"fused_over_foreach": 0.45}},
        ],
    )


def _data(share: float) -> dict[str, Any]:
    """B-6 (Q-189): one memory-mapped case at 1M rows with its data share and budget entry."""
    case = {"n": 1_000_000, "placement": "mmap", "prefetch": 8}
    budget = {
        "case": case,
        "figure": "data_share",
        "limit": 0.01,
        "value": share,
        "pass": share < 0.01,  # §3.4: "< 1%", as `ftgs bench data` records it
        "kind": "budget",
    }
    return _bench("B-6", [{"case": case, "figures": {"data_share": share}}], [budget])


def _gpu(name: str) -> dict[str, Any]:
    """An environment block recorded on the GPU model `name`."""
    body = _adam(0.0)
    body["environment"]["gpu"]["name"] = name
    return {"environment": body["environment"]}


def _write(folder: Path, name: str, body: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _run(tmp_path: Path, runs: dict[str, dict[str, Any]], baselines: dict[str, Any]) -> int:
    """Writes `runs` into a run folder and `baselines` ({gpu-slug folder: {name: body}}) under a
    baselines folder, then calls the script."""
    for name, body in runs.items():
        _write(tmp_path / "run", name, body)
    for folder, files in baselines.items():
        for name, body in files.items():
            _write(tmp_path / "baselines" / folder, name, body)
    return int(compare.main([str(tmp_path / "run"), "--baselines", str(tmp_path / "baselines")]))


def test_gpu_slug() -> None:
    """Q-196: a GPU model's folder name."""
    assert compare.gpu_slug(RTX_4090) == "nvidia-geforce-rtx-4090"
    assert compare.gpu_slug("NVIDIA H100 80GB HBM3") == "nvidia-h100-80gb-hbm3"


@pytest.mark.parametrize(("worse", "code"), [(0.099, 0), (0.101, 1)])
def test_ten_percent_rule(tmp_path: Path, worse: float, code: int) -> None:
    """§3.4, Q-170: a median time 9.9 % above its baseline passes, 10.1 % above fails."""
    base = 2.0e-3
    runs = {"B-5-adam.json": _adam(base * (1 + worse))}
    baselines = {"nvidia-geforce-rtx-4090": {"B-5-adam.json": _adam(base)}}
    assert _run(tmp_path, runs, baselines) == code


def test_faster_passes(tmp_path: Path) -> None:
    """Q-170: only a slower run fails; a faster one is no regression."""
    runs = {"B-5-adam.json": _adam(1.0e-3)}
    baselines = {"nvidia-geforce-rtx-4090": {"B-5-adam.json": _adam(2.0e-3)}}
    assert _run(tmp_path, runs, baselines) == 0


def test_missing_baseline_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """§3.4 "its stored baseline", Q-170: a run whose GPU model has no baseline fails, as does
    a run whose file has none in that GPU's folder."""
    runs = {"B-5-adam.json": _adam(2.0e-3)}
    assert _run(tmp_path / "a", runs, {}) == 1
    assert "no baseline" in capsys.readouterr().out
    other = {"nvidia-geforce-rtx-4090": {"B-6-data.json": _data(0.001)}}
    assert _run(tmp_path / "b", runs | {"B-6-data.json": _data(0.001)}, other) == 1


def test_baseline_of_another_gpu_model_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-170: baselines are kept per GPU model. A baseline in the run's folder that was
    recorded on another GPU model fails, even with equal times; a run on a model with no
    folder of its own never falls back on another model's."""
    runs = {"B-5-adam.json": _adam(2.0e-3)}
    misplaced = {"nvidia-geforce-rtx-4090": {"B-5-adam.json": _adam(2.0e-3) | _gpu("H100")}}
    assert _run(tmp_path / "a", runs, misplaced) == 1
    assert "recorded on H100" in capsys.readouterr().out
    elsewhere = {"h100": {"B-5-adam.json": _adam(2.0e-3) | _gpu("H100")}}
    assert _run(tmp_path / "b", runs, elsewhere) == 1


@pytest.mark.parametrize(
    ("share", "base", "code"), [(0.0099, 0.0003, 0), (0.01, 0.0003, 1), (0.0101, 0.0101, 1)]
)
def test_b6_against_its_absolute_budget(
    tmp_path: Path, share: float, base: float, code: int
) -> None:
    """§3.4 B-6, Q-170: the data share is judged against its absolute < 1 %, not the baseline:
    33 times the baseline's share passes under 1 %, exactly 1 % fails, as `ftgs bench data`'s
    budget entry records it, and the baseline's own share fails above it."""
    runs = {"B-6-data.json": _data(share)}
    baselines = {"nvidia-geforce-rtx-4090": {"B-6-data.json": _data(base)}}
    assert _run(tmp_path, runs, baselines) == code


def test_only_median_times_are_judged(tmp_path: Path) -> None:
    """Q-196: B-2's first call, its other figures, its per-model counts and its `limit`
    entries (here the layout comparison's, as baselines recorded before #278 carry it, failing
    in the run and passing in the baseline) are not judged."""
    variant = {"n": 1_000_000, "active": 0.2, "packed": False, "precull": "internal"}

    def raster(first_call_s: float, raster_s: float, diff: float) -> dict[str, Any]:
        limit = {
            "case": variant,
            "figure": "image_max_abs_diff",
            "limit": 1e-6,
            "value": diff,
            "pass": diff <= 1e-6,
            "kind": "limit",
        }
        cases = [
            {"case": {"n": 1_000_000, "active": 0.2}, "figures": {"tile_intersections": 9}},
            {
                "case": variant,
                "figures": {
                    "forward_backward_s": 3.6e-3,
                    "first_call_s": first_call_s,
                    "raster_s": raster_s,
                },
            },
        ]
        return _bench("B-2", cases, [limit])

    runs = {"B-2-raster.json": raster(10.0, 5e-3, 1.6e-3)}
    baselines = {"nvidia-geforce-rtx-4090": {"B-2-raster.json": raster(1.0, 2.7e-3, 0.0)}}
    assert _run(tmp_path, runs, baselines) == 0


def test_case_on_one_side_only_fails(tmp_path: Path) -> None:
    """Q-196: cases match exactly; a timed case the run didn't measure fails, as does one
    the baseline lacks."""
    baselines = {"nvidia-geforce-rtx-4090": {"B-5-adam.json": _adam(2.0e-3, kind="fused")}}
    runs = {"B-5-adam.json": _adam(2.0e-3, kind="foreach")}
    assert _run(tmp_path, runs, baselines) == 1


def test_baseline_without_a_run_fails(tmp_path: Path) -> None:
    """Q-196: every baseline of the run's GPU model needs its run, so a benchmark the run
    left out fails."""
    runs = {"B-5-adam.json": _adam(2.0e-3)}
    baselines = {
        "nvidia-geforce-rtx-4090": {"B-5-adam.json": _adam(2.0e-3), "B-6-data.json": _data(0.001)}
    }
    assert _run(tmp_path, runs, baselines) == 1


def test_scene_of_another_size_fails(tmp_path: Path) -> None:
    """Q-196: a run on a scene of another size measured something else."""
    runs = {"B-6-data.json": _data(0.001)}
    short = _data(0.001)
    short["scene"]["frames"] = 60
    baselines = {"nvidia-geforce-rtx-4090": {"B-6-data.json": short}}
    assert _run(tmp_path, runs, baselines) == 1


def test_update_stores_the_baselines(tmp_path: Path) -> None:
    """`--update` stores each run in its GPU model's folder, against which it then passes."""
    run = _write(tmp_path / "run", "B-5-adam.json", _adam(2.0e-3))
    baselines = tmp_path / "baselines"
    assert compare.main([str(run.parent), "--baselines", str(baselines), "--update"]) == 0
    stored = baselines / "nvidia-geforce-rtx-4090" / "B-5-adam.json"
    assert stored.read_bytes() == run.read_bytes()
    assert compare.main([str(run.parent), "--baselines", str(baselines)]) == 0


def test_gpu_workflow_is_hand_started_on_a_self_hosted_gpu_runner() -> None:
    """Q-179 (owner's answer): the GPU job starts only by hand (`workflow_dispatch`, never on
    pushes or pull requests), on a self-hosted GPU runner, with a timeout; it generates the
    scene, runs B-2 to B-6 (B-6 also on the 60-frame manifest, Q-178) and the comparison."""
    workflow = SCRIPT.parents[1] / ".github" / "workflows" / "gpu-bench.yml"
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert data[True] == {"workflow_dispatch": None}  # YAML 1.1 reads the key `on` as True
    (job,) = data["jobs"].values()
    assert job["runs-on"] == ["self-hosted", "linux", "gpu"]
    assert 0 < job["timeout-minutes"] <= 240
    script = "\n".join(step.get("run", "") for step in job["steps"])
    assert "scripts/ci_scene.py" in script
    for command in ("raster", "slice", "losses", "adam", "data"):
        assert f'bench {command} "$FTGS_SCENE/scene.json"' in script
    assert 'bench data "$FTGS_SCENE/scene-60.json"' in script
    assert script.index("scripts/ci_scene.py") < script.index("bench raster")
    assert script.index("bench raster") < script.index("scripts/bench_compare.py")


@pytest.mark.parametrize(
    "body",
    [
        {"format": "something-else"},
        _bench("B-1", []),  # §3.4: B-1, B-7 and B-8 aren't run in CI
    ],
)
def test_invalid_input_is_exit_2(tmp_path: Path, body: dict[str, Any]) -> None:
    """A file that isn't a BENCH.json of B-2 to B-6 is invalid input, exit 2."""
    assert _run(tmp_path, {"x.json": body}, {}) == 2
    assert compare.main([str(tmp_path / "missing"), "--baselines", str(tmp_path)]) == 2

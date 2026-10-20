# SPDX-License-Identifier: Apache-2.0
"""L3 · multi-seed statistics (§3.3 L3; D-54; Q-203, Q-210, Q-211).

For each scene of the run settings, A and B train with seeds 0…K−1 and A again with K…2K−1
(A′, the null distribution), one adapter process per run, round robin over the settings' GPUs
with CUDA_VISIBLE_DEVICES set to one of them, so each run is a single-GPU run. Each run is
exported to FTGS-IF and renders the scene's held-out request; the harness computes every metric
itself (E16). Image metrics get Welch and TOST verdicts, Holm-corrected across metrics × scenes;
distributions of the exported models get the seed-permutation test of d(A, B) − d(A, A′),
Holm-corrected at 0.01, and `inconclusive` where no flag is reachable at this K; the training
traces are reported at 1k, 7k, 15k and 30k iterations scaled to the run's length."""

import concurrent.futures
import json
import math
import queue
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.image
import numpy as np

from ftgs_harness.artifacts import plot_cdfs, plot_lines
from ftgs_harness.cameras import Cam
from ftgs_harness.checks import Check, Status
from ftgs_harness.formats import FormatError, RequestItem, read_exchange, read_renders, read_scene
from ftgs_harness.metrics import Lpips, dssim, psnr, ssim
from ftgs_harness.rendering import CaseItem, relative, request_for
from ftgs_harness.runner import Context
from ftgs_harness.settings import HarnessError, Scene
from ftgs_harness.stats import (
    holm,
    k_for_power,
    k_needed_for_flags,
    ks_distance,
    permutation_p,
    smallest_permutation_p,
    tost_p,
    w1_distance,
    welch_p,
)

__all__ = ["MARGINS", "QUANTITIES", "run", "trace_targets"]

# §3.3 L3 step 4: the TOST margins; DSSIM is reported only, a function of SSIM (Q-210).
MARGINS = {"psnr": 0.25, "ssim1": 0.005, "ssim2": 0.005, "lpips_alex": 0.01, "lpips_vgg": 0.01}
_ALPHA = 0.05  # Q-210: Welch and TOST, Holm-corrected
_FLAG = 0.01  # §3.3 L3 step 5
_TRACE_POINTS = (1_000, 7_000, 15_000, 30_000)  # §3.3 L3 step 6, for T_it = 30,000
_MAX_SAMPLES = 20_000  # per run and quantity, for the distances
QUANTITIES = (
    "opacity",
    "log_duration",
    "speed",
    "time_center",
    "log_max_scale",
    "count",
    "psnr_per_time",
)
_SPEC = ["§3.3 L3"]


@dataclass
class _RunResult:
    impl: str
    seed: int
    folder: Path
    failure: Check | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    per_image_psnr: list[float] = field(default_factory=list)
    samples: dict[str, np.ndarray] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    not_comparable: list[str] = field(default_factory=list)


def trace_targets(iterations: int) -> list[int]:
    """§3.3 L3 step 6's iterations scaled by T_it/30,000 (§3.2's rule: rounded, at least 1)."""
    return [max(1, math.floor(p * iterations / 30_000 + 0.5)) for p in _TRACE_POINTS]


def _held_out(ctx: Context, scene: Scene) -> tuple[list[CaseItem], list[Path]]:
    try:
        info = read_scene(scene.manifest)
    except FormatError as error:
        raise HarnessError(f"L3 scene {scene.name}: {error}") from error
    tests = [image for image in info.images if image.split == "test"]
    if not tests:
        raise HarnessError(f"L3 scene {scene.name}: the manifest has no test images")
    items = []
    for i, image in enumerate(tests):
        cam = Cam(image.camera, image.K, image.world_to_camera, image.width, image.height)
        item = RequestItem(
            f"h{i:04d}", image.timestamp, cam.K, cam.world_to_camera, cam.width, cam.height
        )
        items.append(CaseItem(item, image.camera))
    return items, [image.path for image in tests]


def _image(scene: Scene, path: Path) -> np.ndarray:
    """A held-out ground-truth image as float32 RGB in [0, 1]; a scene whose image can't be read
    is a configuration error."""
    try:
        pixels = matplotlib.image.imread(str(path))
    except (OSError, ValueError) as error:
        raise HarnessError(f"L3 scene {scene.name}: can't read {path}: {error}") from error
    if pixels.dtype == np.uint8:
        pixels = pixels.astype(np.float32) / 255.0
    return np.asarray(pixels[..., :3], dtype=np.float32)


def _samples(path: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """The distributions of §3.3 L3 step 5 from an exported model."""
    ex = read_exchange(path)
    a = ex.arrays
    out = {
        "opacity": a["opacity"].astype(np.float64),
        "log_duration": np.log(a["duration"]),
        "speed": np.linalg.norm(a["velocity"].astype(np.float64), axis=1),
        "time_center": a["time_center"].astype(np.float64),
        "log_max_scale": np.log(a["scale"].astype(np.float64).max(axis=1)),
        "count": np.array([float(ex.num)]),
    }
    for key, value in out.items():
        if value.size > _MAX_SAMPLES:
            out[key] = rng.choice(value, _MAX_SAMPLES, replace=False)
    return out


def _trace(folder: Path) -> list[dict[str, Any]]:
    """train_log.jsonl's rows (§3.3 adapter contract: iteration, loss terms, N; Q-132)."""
    path = folder / "train_log.jsonl"
    if not path.is_file():
        return []
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except ValueError as error:
        raise FormatError(f"{path}: a row is not JSON ({error})") from error
    if not all(isinstance(r, dict) and isinstance(r.get("losses", {}), dict) for r in rows):
        raise FormatError(f"{path}: rows must be objects with an iteration, losses and N")
    return rows


def _not_comparable_keys(folder: Path) -> list[str]:
    """The settings the adapter couldn't honor (Q-202): a JSON list of dotted keys."""
    path = folder / "not_comparable.json"
    if not path.is_file():
        return []
    try:
        keys = json.loads(path.read_text())
    except ValueError as error:
        raise FormatError(f"{path}: not JSON ({error})") from error
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise FormatError(f"{path}: not a list of dotted keys")
    return keys


@dataclass(frozen=True)
class _HeldOut:
    """A scene's held-out request, its ground truth and the metrics the harness computes."""

    request: Path
    sha: str
    items: list[CaseItem]
    truths: list[np.ndarray]
    measure: Callable[[np.ndarray, np.ndarray], dict[str, float]]


def _failure(
    ctx: Context, check_id: str, what: str, detail: str, pattern: str, log: str | None = None
) -> Check:
    return Check(
        id=check_id,
        level="L3",
        status="fail",
        spec_refs=_SPEC,
        inputs={"log": log} if log else {},
        metric="run",
        expected="train, export and render succeed",
        observed=what,
        symptom=detail,
        pattern=pattern,
        repro=ctx.repro(check_id),
    )


def _one_run(
    ctx: Context,
    scene: Scene,
    impl: str,
    seed: int,
    held: _HeldOut,
    gpus: "queue.Queue[str] | None",
) -> _RunResult:
    """One run: train, export and render the held-out request, on one GPU slot of `gpus` (each
    run holds one GPU of the settings' list, D-54), then the harness's metrics."""
    adapter = ctx.b if impl == "B" else ctx.a
    folder = ctx.out / "L3" / scene.name / impl / f"seed{seed}"
    result = _RunResult(impl, seed, folder)
    check_id = f"L3.run.{scene.name}.{impl}.seed{seed}"
    gpu = gpus.get() if gpus is not None else None
    env = {"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else None
    try:
        folder.parent.mkdir(parents=True, exist_ok=True)
        steps: list[tuple[str, list[Any]]] = [
            ("train", [scene.train_manifest, scene.init, ctx.config_path, seed, folder]),
            ("export", [folder, folder.with_suffix(".ftgs-if.safetensors")]),
            ("render", [folder, held.request, folder.with_suffix(".renders.safetensors")]),
        ]
        for command, args in steps:
            outcome = adapter.call(command, args, env=env)
            if not outcome.ok:
                result.failure = _failure(
                    ctx,
                    check_id,
                    outcome.status,
                    f"{impl}'s {command} for scene {scene.name}, seed {seed}: "
                    f"{outcome.describe()}.",
                    "crash",
                    outcome.log,
                )
                return result
    finally:
        if gpus is not None and gpu is not None:
            gpus.put(gpu)
    try:
        renders = read_renders(
            folder.with_suffix(".renders.safetensors"),
            request_sha256=held.sha,
            items=[i.item for i in held.items],
        )
        result.samples = _samples(
            folder.with_suffix(".ftgs-if.safetensors"),
            np.random.default_rng([ctx.settings.seed, 3, seed]),
        )
        result.trace = _trace(folder)
        result.not_comparable = _not_comparable_keys(folder)
    except FormatError as error:
        result.failure = _failure(
            ctx,
            check_id,
            "schema violation",
            f"{impl}'s outputs for scene {scene.name}, seed {seed} fail their format: {error}.",
            "schema violation",
        )
        return result
    per_image = [
        held.measure(renders[i.item.id][0], truth)
        for i, truth in zip(held.items, held.truths, strict=True)
    ]
    result.metrics = {k: float(np.mean([m[k] for m in per_image])) for k in per_image[0]}
    result.per_image_psnr = [m["psnr"] for m in per_image]
    result.samples["psnr_per_time"] = np.array(result.per_image_psnr)
    return result


def _runs(ctx: Context, scene: Scene, held: _HeldOut) -> list[_RunResult]:
    k = ctx.settings.k
    plan = (
        [("A", s) for s in range(k)]
        + [("B", s) for s in range(k)]
        + [("A2", s) for s in range(k, 2 * k)]
    )
    gpus: queue.Queue[str] | None = None
    if ctx.settings.gpus:
        gpus = queue.Queue()
        for gpu in ctx.settings.gpus:
            gpus.put(gpu)
    workers = len(ctx.settings.gpus) if ctx.settings.gpus else 1
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return list(
            pool.map(
                lambda job: _one_run(ctx, scene, job[0], job[1], held, gpus),
                plan,
            )
        )


@dataclass(frozen=True)
class _MetricRow:
    scene: str
    metric: str
    a: np.ndarray  # per-seed means of A
    b: np.ndarray
    a2: np.ndarray  # A′'s, reported for the seed spread
    runs_a: list[dict[str, float]]
    runs_b: list[dict[str, float]]


def _metric_checks(ctx: Context, scenes: dict[str, list[_RunResult]]) -> list[Check]:
    rows: list[_MetricRow] = []
    for name, results in scenes.items():
        good = [r for r in results if r.failure is None]
        runs_a = [r.metrics for r in good if r.impl == "A"]
        runs_b = [r.metrics for r in good if r.impl == "B"]
        runs_a2 = [r.metrics for r in good if r.impl == "A2"]
        if len(runs_a) < 2 or len(runs_b) < 2:
            continue
        for metric in MARGINS:
            a = np.array([m[metric] for m in runs_a])
            b = np.array([m[metric] for m in runs_b])
            a2 = np.array([m[metric] for m in runs_a2])
            rows.append(_MetricRow(name, metric, a, b, a2, runs_a, runs_b))
    if not rows:
        return []
    tost = holm([tost_p(r.a, r.b, MARGINS[r.metric]) for r in rows])
    welch = holm([welch_p(r.a, r.b) for r in rows])
    checks = []
    for row, p_eq, p_diff in zip(rows, tost, welch, strict=True):
        name, metric, a, b, ma, mb = row.scene, row.metric, row.a, row.b, row.runs_a, row.runs_b
        check_id = f"L3.metric.{name}.{metric}"
        if not ctx.wants(check_id):
            continue
        status: Status
        sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
        needed = k_for_power(sd, MARGINS[metric], alpha=_ALPHA)
        if p_eq < _ALPHA:
            status, verdict = "pass", "equivalent"
        elif p_diff < _ALPHA:
            status, verdict = "fail", "different"
        else:
            status, verdict = "inconclusive", "inconclusive"
        delta = float(b.mean() - a.mean())
        localization: dict[str, Any] = {
            "verdict": verdict,
            "mean_A": float(a.mean()),
            "mean_B": float(b.mean()),
            "sd_A": float(a.std(ddof=1)),
            "sd_B": float(b.std(ddof=1)),
            "p_tost_holm": p_eq,
            "p_welch_holm": p_diff,
            "seeds": len(a),
            "k_for_80_percent_power": needed,
            "per_seed_A": a.tolist(),
            "per_seed_B": b.tolist(),
            "per_seed_A2": row.a2.tolist(),
        }
        if metric.startswith("ssim"):
            k = metric[-1]
            localization["dssim_A"] = float(np.mean([m[f"dssim{k}"] for m in ma]))
            localization["dssim_B"] = float(np.mean([m[f"dssim{k}"] for m in mb]))
        unit = " dB" if metric == "psnr" else ""
        symptom = ""
        if status != "pass":
            symptom = (
                f"On scene {name}'s held-out views, B's mean {metric} is {b.mean():.4g}{unit} "
                f"against A's {a.mean():.4g}{unit} over {len(a)} seeds each (difference "
                f"{delta:+.3g}{unit}, "
                f"seed spread {sd:.3g}{unit}); about {needed} seeds a side would give 80% power at "
                f"the ±{MARGINS[metric]:g}{unit} margin."
            )
        checks.append(
            Check(
                id=check_id,
                level="L3",
                status=status,
                spec_refs=["E16", "§3.3 L3"],
                inputs={"scene": name, "metric": metric},
                metric=f"{metric}_mean_B_minus_A",
                expected=f"|Δ| < {MARGINS[metric]:g} (TOST, Holm, α = {_ALPHA:g})",
                observed=delta,
                symptom=symptom,
                localization=localization,
                repro=ctx.repro(check_id),
            )
        )
    return checks


@dataclass(frozen=True)
class _DistRow:
    scene: str
    quantity: str
    distance: str  # "ks" or "w1"
    statistic: float  # d(A, B) − d(A, A′)
    p: float
    k: int
    pooled: dict[str, np.ndarray]


def _dist_checks(ctx: Context, scenes: dict[str, list[_RunResult]]) -> list[Check]:
    rows: list[_DistRow] = []
    for name, results in scenes.items():
        good = [r for r in results if r.failure is None]
        groups = {impl: [r.samples for r in good if r.impl == impl] for impl in ("A", "B", "A2")}
        if min(len(g) for g in groups.values()) < 2 or len(groups["B"]) != len(groups["A2"]):
            continue
        for quantity in QUANTITIES:
            for label, distance in (("ks", ks_distance), ("w1", w1_distance)):
                stat, p = permutation_p(
                    [s[quantity] for s in groups["A"]],
                    [s[quantity] for s in groups["B"]],
                    [s[quantity] for s in groups["A2"]],
                    distance,
                )
                pooled = {
                    impl: np.concatenate([s[quantity] for s in g]) for impl, g in groups.items()
                }
                rows.append(_DistRow(name, quantity, label, stat, p, len(groups["B"]), pooled))
    if not rows:
        return []
    adjusted = holm([r.p for r in rows])
    m = len(rows)
    checks = []
    plotted: dict[tuple[str, str], str] = {}
    for row, p_holm in zip(rows, adjusted, strict=True):
        name, quantity, label, stat, p, k = (
            row.scene,
            row.quantity,
            row.distance,
            row.statistic,
            row.p,
            row.k,
        )
        pooled = row.pooled
        check_id = f"L3.dist.{name}.{quantity}.{label}"
        if not ctx.wants(check_id):
            continue
        smallest = smallest_permutation_p(k)
        reachable = smallest <= _FLAG / m
        if (name, quantity) not in plotted:
            plotted[(name, quantity)] = relative(
                ctx,
                ctx.out
                / plot_cdfs(
                    ctx.out,
                    f"L3_{name}_{quantity}",
                    {"A": pooled["A"], "B": pooled["B"], "A′": pooled["A2"]},
                    xlabel=quantity,
                    title=f"{name}: {quantity}",
                ),
            )
        status: Status = "fail" if p_holm < _FLAG else ("pass" if reachable else "inconclusive")
        symptom = ""
        if status == "fail":
            symptom = (
                f"On scene {name}, the distribution of {quantity} in B's exported models is "
                f"{label.upper()} {stat:+.3g} further from A's than A′'s is (seed permutation "
                f"p = {p:.3g}, {p_holm:.3g} after Holm over {m} tests)."
            )
        elif status == "inconclusive":
            symptom = (
                f"With {k} seeds a side the permutation test's smallest p is {smallest:.3g}, above "
                f"Holm's first threshold {_FLAG / m:.3g} over {m} tests, so no flag is reachable; "
                f"{k_needed_for_flags(m, _FLAG)} seeds a side would reach it."
            )
        checks.append(
            Check(
                id=check_id,
                level="L3",
                status=status,
                spec_refs=["§3.3 L3"],
                inputs={"scene": name, "quantity": quantity},
                metric=f"{label}_A_B_minus_A_A2",
                expected=f"Holm-adjusted p ≥ {_FLAG:g}",
                observed=stat,
                symptom=symptom,
                localization={"p": p, "p_holm": p_holm, "tests": m, "smallest_p": smallest},
                repro=ctx.repro(check_id),
                artifacts=[plotted[(name, quantity)]],
            )
        )
    return checks


def _trace_checks(ctx: Context, name: str, results: list[_RunResult]) -> list[Check]:
    iterations = int(dict(ctx.config.get("optim", {})).get("iterations", 30_000))
    targets = trace_targets(iterations)
    checks = []
    series: dict[str, list[float]] = {}
    for label, target in zip(("1k", "7k", "15k", "30k"), targets, strict=True):
        check_id = f"L3.trace.{name}.{label}"
        summary: dict[str, Any] = {}
        missing = []
        for impl in ("A", "B", "A2"):
            rows = []
            for r in results:
                if r.impl != impl or r.failure is not None:
                    continue
                before = [row for row in r.trace if int(row.get("iteration", 0)) <= target]
                if not before:
                    missing.append(f"{impl} seed {r.seed}")
                    continue
                rows.append(before[-1])
            terms = sorted({k for row in rows for k in row.get("losses", {})})
            values = {
                t: [float(row["losses"][t]) for row in rows if t in row.get("losses", {})]
                for t in terms
            }
            values["num_gaussians"] = [float(row.get("num_gaussians", math.nan)) for row in rows]
            summary[impl] = {
                t: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                for t, v in values.items()
                if v
            }
            for t, v in values.items():
                if v and impl in ("A", "B"):
                    series.setdefault(f"{impl} {t}", []).append(float(np.median(v)))
        finite = all(
            math.isfinite(x) for impl in summary.values() for s in impl.values() for x in s.values()
        )
        status: Status = "pass" if finite and not missing else "fail"
        if not ctx.wants(check_id):
            continue
        # §3.3 L3 step 6: compared by median and spread, B's median against the spread of A's
        # seeds and A′'s together; reported, not judged (Q-210).
        comparison = {}
        for term, b_stats in summary.get("B", {}).items():
            lows = [summary[i][term]["min"] for i in ("A", "A2") if term in summary.get(i, {})]
            highs = [summary[i][term]["max"] for i in ("A", "A2") if term in summary.get(i, {})]
            if lows:
                comparison[term] = bool(min(lows) <= b_stats["median"] <= max(highs))
        checks.append(
            Check(
                id=check_id,
                level="L3",
                status=status,
                spec_refs=["§3.3 L3"],
                inputs={"scene": name, "iteration": target},
                metric="loss_terms_and_N",
                expected="logged and finite",
                observed=summary,
                localization={"B_median_within_A_and_A2_spread": comparison},
                symptom=""
                if status == "pass"
                else (
                    f"On scene {name} at iteration {target}, "
                    + (
                        f"no log row for {', '.join(missing)}"
                        if missing
                        else "a loss term or N is not finite"
                    )
                    + "."
                ),
                pattern=None if status == "pass" else ("NaN/Inf" if not finite else "crash"),
                repro=ctx.repro(check_id),
            )
        )
    full = {k: v for k, v in series.items() if len(v) == len(targets)}
    picked = {k: v for k, v in full.items() if k.endswith(("total", "num_gaussians"))} or full
    shown: dict[str, Sequence[float]] = dict(picked)
    if shown and checks:
        plot = plot_lines(
            ctx.out,
            f"L3_{name}_trace",
            targets,
            shown,
            xlabel="iteration",
            ylabel="median",
            title=f"{name}: training trace",
        )
        checks[-1].artifacts.append(relative(ctx, ctx.out / plot))
    return checks


def _not_comparable(ctx: Context, name: str, results: list[_RunResult]) -> list[Check]:
    keys = sorted({(r.impl, k) for r in results for k in r.not_comparable if r.impl in ("A", "B")})
    checks = []
    for impl, key in keys:
        check_id = f"L3.config.{name}.{impl}.{key}"
        if ctx.wants(check_id):
            checks.append(
                Check(
                    id=check_id,
                    level="L3",
                    status="not_comparable",
                    spec_refs=["§3.3 adapter contract"],
                    inputs={"scene": name, "setting": key, "adapter": impl},
                    metric="harness_config_setting",
                    expected="honored",
                    observed="not honored",
                    repro=ctx.repro(check_id),
                )
            )
    return checks


def _measurer() -> Callable[[np.ndarray, np.ndarray], dict[str, float]]:
    alex, vgg = Lpips("alex"), Lpips("vgg")

    def measure(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
        return {
            "psnr": psnr(pred, gt),
            "ssim1": ssim(pred, gt, 1),
            "ssim2": ssim(pred, gt, 2),
            "dssim1": dssim(pred, gt, 1),
            "dssim2": dssim(pred, gt, 2),
            "lpips_alex": alex(pred, gt),
            "lpips_vgg": vgg(pred, gt),
        }

    return measure


def run(ctx: Context) -> list[Check]:
    measure = _measurer()
    background = list(ctx.config.get("background", [0.0, 0.0, 0.0]))
    by_scene: dict[str, list[_RunResult]] = {}
    checks: list[Check] = []
    for scene in ctx.settings.scenes:
        if not ctx.wants("L3") and not any(
            ctx.wants(f"L3.{kind}.{scene.name}")
            for kind in ("run", "metric", "dist", "trace", "config")
        ):
            continue
        items, paths = _held_out(ctx, scene)
        truths = [_image(scene, p) for p in paths]
        request = ctx.cases / "L3" / f"{scene.name}.request.json"
        request.parent.mkdir(parents=True, exist_ok=True)
        sha = request_for(request, items, background)
        results = _runs(ctx, scene, _HeldOut(request, sha, items, truths, measure))
        by_scene[scene.name] = results
        checks += [r.failure for r in results if r.failure is not None and ctx.wants(r.failure.id)]
        checks += _not_comparable(ctx, scene.name, results)
        checks += _trace_checks(ctx, scene.name, results)
    return checks + _metric_checks(ctx, by_scene) + _dist_checks(ctx, by_scene)

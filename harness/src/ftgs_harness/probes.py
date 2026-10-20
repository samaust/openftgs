# SPDX-License-Identifier: Apache-2.0
"""L1's ten probes (§3.3 L1 probe table; Q-209, Q-211): each one or two primitives with one
variable swept over `l1.probe_items` items at 64×48, so that a mismatch points at one quantity.
A failing probe's items get the probe's pattern where the sweep shows it (a temporal shift or
width for the time sweeps, a view-dependent colour shift for P5) and the trend of the
discrepancy against the swept variable (`trends`).

The sweeps of P4, P6, P7 and P8 keep their samples away from E6's discontinuities (Q-39): α at
α_max, the near plane, a depth tie, σ_eff at 1/255."""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ftgs_harness.artifacts import plot_lines
from ftgs_harness.cameras import Cam, intrinsics, look_at
from ftgs_harness.checks import Check
from ftgs_harness.compare import measure
from ftgs_harness.formats import Exchange, RequestItem
from ftgs_harness.level1 import render_case
from ftgs_harness.models import single_primitive
from ftgs_harness.rendering import CaseItem, item_checks, relative, where_text
from ftgs_harness.runner import Context
from ftgs_harness.symptoms import PairMeasures, describe, measure_pair
from ftgs_harness.trends import fit_trend

__all__ = ["PROBES", "Probe", "run"]

_WIDTH, _HEIGHT, _FOCAL = 64, 48, 60.0
_TAU0, _TAU1, _TC = 0.0, 1.9, 0.95
_EYE = np.array([0.0, -4.0, 0.0])  # looking along +y: image x is world x, image y is −z


@dataclass(frozen=True)
class Probe:
    name: str
    content: str
    swept: str  # the swept variable, as the symptom names it
    unit: str
    spec_refs: list[str]
    build: Callable[[int], tuple[Exchange, list[CaseItem]]]
    trend_variable: str | None = None  # when the trend is against a function of the sweep
    trend_x: Callable[[CaseItem], float] | None = None
    time_swept: bool = False
    view: bool = False


def _camera(name: str, *, eye: np.ndarray = _EYE, cx: float = _WIDTH / 2.0) -> Cam:
    K = intrinsics(_WIDTH, _HEIGHT, _FOCAL)
    K[0, 2] = cx
    return Cam(name, K, look_at(eye), _WIDTH, _HEIGHT)


def _item(i: int, cam: Cam, tau: float, swept: float) -> CaseItem:
    item = RequestItem(f"r{i:04d}", float(tau), cam.K, cam.world_to_camera, _WIDTH, _HEIGHT)
    return CaseItem(item, cam.name, swept)


def _time_items(times: np.ndarray) -> list[CaseItem]:
    cam = _camera("front")
    return [_item(i, cam, t, t) for i, t in enumerate(times)]


def _tau_for(effective: float, sigma: float, duration: float) -> float:
    """The τ after _TC at which σ·σt(τ) is `effective` (E3, E4)."""
    return _TC + duration * math.sqrt(-2.0 * math.log(min(effective / sigma, 1.0)))


def _away(values: np.ndarray, cut: float, margin: float) -> np.ndarray:
    """Q-39: the samples moved to at least `margin` (relative) from the cutoff."""
    out = values.copy()
    near = np.abs(out - cut) < margin * abs(cut)
    out[near] = np.where(out[near] < cut, cut * (1 - margin), cut * (1 + margin))
    return out


def _p1(n: int) -> tuple[Exchange, list[CaseItem]]:
    model = single_primitive(scale=(0.08, 0.08, 0.08))
    shifts = np.linspace(0.0, 1.0, n)
    return model, [
        _item(i, _camera(f"sub{s:.2f}", cx=_WIDTH / 2 + s), _TC, s) for i, s in enumerate(shifts)
    ]


def _moving() -> Exchange:
    return single_primitive(velocity=(0.6, 0.0, 0.0), scale=(0.07, 0.07, 0.07))


def _p2(n: int) -> tuple[Exchange, list[CaseItem]]:
    return _moving(), _time_items(np.linspace(_TAU0, _TAU1, n))


def _p3(n: int) -> tuple[Exchange, list[CaseItem]]:
    model = single_primitive(duration=0.15, scale=(0.08, 0.08, 0.08))
    return model, _time_items(np.linspace(0.5, 1.4, n))


def _opacity_items(n: int, values: np.ndarray, sigma: float, duration: float) -> list[CaseItem]:
    cam = _camera("front")
    return [_item(i, cam, _tau_for(e, sigma, duration), e) for i, e in enumerate(values)]


def _p4(n: int) -> tuple[Exchange, list[CaseItem]]:
    sigma, duration = 1.0 - 1e-7, 0.5
    model = single_primitive(opacity=sigma, duration=duration, scale=(0.1, 0.1, 0.1))
    values = _away(np.linspace(0.9, 1.0 - 1e-6, n), 0.999, 2e-4)  # α_max (D-31)
    return model, _opacity_items(n, values, sigma, duration)


def _p5(n: int) -> tuple[Exchange, list[CaseItem]]:
    model = single_primitive(sh_degree=3, color_sh=(0.0, 0.0, 0.0), scale=(0.15, 0.15, 0.15))
    rng = np.random.default_rng(5)
    model.arrays["sh"][0, 1:] = (rng.normal(size=(15, 3)) * 0.12).astype(np.float32)
    angles = np.linspace(0.0, 360.0, n, endpoint=False)
    items = []
    for i, deg in enumerate(angles):
        rad = math.radians(deg)
        eye = 4.0 * np.array([math.sin(rad), -math.cos(rad), 0.3])
        items.append(_item(i, _camera(f"view{deg:.0f}", eye=eye), _TC, float(deg)))
    return model, items


def _p6(n: int) -> tuple[Exchange, list[CaseItem]]:
    # The primitive's camera depth is 0.1 + 0.4·(τ − _TC), crossing the near plane 0.01 (E6).
    model = single_primitive(
        position=(0.02, -3.9, 0.0), velocity=(0.0, 0.4, 0.0), scale=(0.02, 0.02, 0.02)
    )
    depth = _away(np.linspace(-0.2, 0.47, n), 0.01, 0.5)
    times = _TC + (depth - 0.1) / 0.4
    cam = _camera("front")
    return model, [_item(i, cam, t, d) for i, (t, d) in enumerate(zip(times, depth, strict=True))]


def _p7(n: int) -> tuple[Exchange, list[CaseItem]]:
    # A static primitive at depth 4 and a moving one crossing it in depth at _TC.
    rows = [
        single_primitive(scale=(0.15, 0.15, 0.15), opacity=0.7, color_sh=(1.0, -0.5, -0.5)),
        single_primitive(
            velocity=(0.0, 0.5, 0.0),
            scale=(0.15, 0.15, 0.15),
            opacity=0.7,
            color_sh=(-0.5, -0.5, 1.0),
        ),
    ]
    model = replace(
        rows[0], arrays={k: np.concatenate([r.arrays[k] for r in rows]) for k in rows[0].arrays}
    )
    times = np.linspace(_TAU0, _TAU1, n)
    times = np.where(np.abs(times - _TC) < 0.02, _TC + np.sign(times - _TC + 1e-12) * 0.02, times)
    return model, _time_items(times)


def _p8(n: int) -> tuple[Exchange, list[CaseItem]]:
    sigma, duration = 0.02, 0.5
    model = single_primitive(opacity=sigma, duration=duration, scale=(0.1, 0.1, 0.1))
    values = _away(np.linspace(0.5 / 255.0, 2.0 / 255.0, n), 1.0 / 255.0, 0.02)  # D-32, E6
    return model, _opacity_items(n, values, sigma, duration)


def _p9(n: int) -> tuple[Exchange, list[CaseItem]]:
    model = single_primitive(scale=(0.8, 0.8, 0.8), opacity=0.6)
    positions = np.linspace(-30.0, _WIDTH + 30.0, n)
    items = [_item(i, _camera(f"cx{p:.1f}", cx=p), _TC, p) for i, p in enumerate(positions)]
    return model, items


def _p10(n: int) -> tuple[Exchange, list[CaseItem]]:
    half = (n + 1) // 2
    times = np.concatenate(
        [
            np.linspace(_TAU0 - 1.0, _TAU0 - 0.05, half),
            np.linspace(_TAU1 + 0.05, _TAU1 + 1.0, n - half),
        ]
    )
    return _moving(), _time_items(times)


def _since_center(entry: CaseItem) -> float:
    return abs(entry.item.timestamp - _TC)


PROBES = [
    Probe(
        "P1",
        "one static isotropic primitive at the image centre",
        "sub-pixel position",
        "px",
        ["E5", "E6"],
        _p1,
    ),
    Probe(
        "P2",
        "one moving primitive",
        "τ",
        "s",
        ["E1"],
        _p2,
        "|τ − time_center|",
        _since_center,
        time_swept=True,
    ),
    Probe(
        "P3", "one static primitive with a short duration", "τ", "s", ["E4"], _p3, time_swept=True
    ),
    Probe("P4", "one near-opaque primitive", "effective opacity", "", ["E3", "E6", "D-31"], _p4),
    Probe(
        "P5",
        "one primitive with only l ≥ 1 SH energy",
        "viewing angle",
        "deg",
        ["E2"],
        _p5,
        view=True,
    ),
    Probe(
        "P6",
        "a primitive crossing the near plane",
        "camera depth",
        "",
        ["E6", "D-6"],
        _p6,
        time_swept=True,
    ),
    Probe(
        "P7", "two primitives whose depth order swaps", "τ", "s", ["E1", "E6"], _p7, time_swept=True
    ),
    Probe(
        "P8",
        "a primitive with σ_eff around 1/255",
        "effective opacity",
        "",
        ["E3", "E6", "D-32"],
        _p8,
    ),
    Probe("P9", "a large primitive partly off-screen", "image position", "px", ["E6"], _p9),
    Probe(
        "P10",
        "P2's primitive outside [τ₀, τ₁]",
        "τ",
        "s",
        ["E1", "E4", "E14"],
        _p10,
        "|τ − time_center|",
        _since_center,
        time_swept=True,
    ),
]


def _mass_profile(
    items: list[CaseItem], renders: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([e.item.timestamp for e in items])
    mass = np.array([float(np.nan_to_num(renders[e.item.id][1]).sum()) for e in items])
    return times, mass


def _center_width(times: np.ndarray, mass: np.ndarray) -> tuple[float, float]:
    total = mass.sum()
    if total <= 0:
        return float("nan"), float("nan")
    center = float((times * mass).sum() / total)
    return center, float(np.sqrt((mass * (times - center) ** 2).sum() / total))


def _temporal(
    items: list[CaseItem],
    a: dict[str, tuple[np.ndarray, np.ndarray]],
    b: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, str] | None:
    """A time sweep's pattern from the alpha mass against τ: a shift of its centre or a change
    of its width, with the sentence; None when the profiles agree in both."""
    times, mass_a = _mass_profile(items, a)
    _, mass_b = _mass_profile(items, b)
    seen = mass_a > 1e-6 * max(float(mass_a.max()), 1e-300)
    ratios = mass_b[seen] / mass_a[seen]
    if ratios.size and float(ratios.std()) < 0.1 * float(ratios.mean()):
        return None  # the same profile scaled: a missing or extra primitive, not a time change
    ca, wa = _center_width(times, mass_a)
    cb, wb = _center_width(times, mass_b)
    if not all(math.isfinite(v) for v in (ca, wa, cb, wb)) or wa <= 0:
        return None
    span = float(times.max() - times.min())
    if abs(cb - ca) > 0.01 * span:
        return "temporal shift", (
            f"B's alpha coverage over τ is centred at {cb:.4g} s against A's {ca:.4g} s, a "
            f"shift of {cb - ca:+.3g} s"
        )
    ratio = wb / wa
    if abs(ratio - 1.0) > 0.02:
        pattern = "temporal widening" if ratio > 1 else "temporal narrowing"
        return pattern, (
            f"B's alpha coverage over τ is {ratio:.3f}× as wide as A's (standard deviation "
            f"{wb:.4g} s against {wa:.4g} s)"
        )
    return None


def _discrepancy(
    pattern: str, m: PairMeasures, mass_a: float, mass_b: float
) -> tuple[float, str, str]:
    """The quantity a pattern's trend follows, its name and unit."""
    if pattern == "offset":
        return m.offset_px, "offset", "px"
    if pattern == "scale change":
        return m.spread_ratio - 1.0, "relative spread change", ""
    if pattern in ("missing primitive", "extra primitive"):
        return m.mass_ratio - 1.0, "relative alpha coverage change", ""
    if pattern.startswith("temporal"):
        return (mass_b - mass_a) / max(mass_a, 1e-12), "relative alpha coverage change", ""
    if pattern == "opacity saturation":
        return m.saturation_diff, "alpha difference", ""
    return float(np.max(np.abs(m.color_diff))), "colour difference", ""


def _observed(check: Check) -> str:
    value = check.observed
    return f"{value:.4g}" if isinstance(value, float) else "not finite"


def _analyse(
    ctx: Context,
    probe: Probe,
    items: list[CaseItem],
    a: dict[str, tuple[np.ndarray, np.ndarray]],
    b: dict[str, tuple[np.ndarray, np.ndarray]],
    checks: list[Check],
) -> None:
    """Give a failing probe's items the probe's pattern where the sweep shows it, and the trend
    of the discrepancy against the swept variable, with a plot."""
    failing = [c for c in checks if c.status == "fail" and c.pattern not in ("NaN/Inf", None)]
    if not failing:
        return
    measures = {
        e.item.id: measure_pair(a[e.item.id][0], b[e.item.id][0], a[e.item.id][1], b[e.item.id][1])
        for e in items
    }
    by_id = {e.item.id: e for e in items}
    override = _temporal(items, a, b) if probe.time_swept else None
    if override is not None:
        for check in failing:
            item_id = check.inputs["items"][0]
            check.pattern = override[0]
            check.symptom = (
                f"{override[1]} (item {item_id}: {check.metric} {_observed(check)}, expected "
                f"{check.expected})."
            )
    else:
        # One pattern for the probe: its worst item's, so the mismatch points at one quantity.
        worst = min(failing, key=lambda c: c.observed if isinstance(c.observed, float) else -1.0)
        pattern = worst.pattern or "color shift"
        if probe.view and pattern == "color shift":
            diffs = np.array([measures[e.item.id].color_diff for e in items])
            if float(np.abs(diffs - diffs.mean(axis=0)).max()) > 0.3 * float(np.abs(diffs).max()):
                pattern = "view-dependent color shift"
        for check in failing:
            item_id = check.inputs["items"][0]
            m = measures[item_id]
            (rgb_a, alpha_a), (rgb_b, alpha_b) = a[item_id], b[item_id]
            psnr = measure(rgb_a, rgb_b, alpha_a, alpha_b).psnr_db
            check.pattern = pattern
            check.symptom = describe(pattern, m, where_text(by_id[item_id], psnr, probe.swept))
    pattern = failing[0].pattern or "color shift"
    xs, ys = [], []
    name, unit = "discrepancy", ""
    for entry in items:
        mass_a = float(np.nan_to_num(a[entry.item.id][1]).sum())
        mass_b = float(np.nan_to_num(b[entry.item.id][1]).sum())
        value, name, unit = _discrepancy(pattern, measures[entry.item.id], mass_a, mass_b)
        xs.append(probe.trend_x(entry) if probe.trend_x else float(entry.swept or 0.0))
        ys.append(value)
    trend = fit_trend(xs, ys)
    variable = probe.trend_variable or probe.swept
    plot = plot_lines(
        ctx.out,
        f"{probe.name}_trend",
        xs,
        {name: ys},
        xlabel=variable,
        ylabel=name,
        title=f"{probe.name}: {pattern}",
    )
    sentence = f" The {name} is {trend.describe(variable, unit or 'units')}."
    for check in failing:
        check.symptom = check.symptom.rstrip() + sentence
        check.localization["trend"] = f"{trend.model} in {variable}"  # as §3.3's example
        check.localization["trend_coefficients"] = list(trend.coefficients)
        check.artifacts.append(relative(ctx, ctx.out / plot))


def run(ctx: Context) -> list[Check]:
    n = ctx.settings.sizes["l1.probe_items"]
    checks: list[Check] = []
    for probe in PROBES:
        prefix = f"L1.render.{probe.name}"
        if not ctx.wants(prefix):
            continue
        model, items = probe.build(n)
        renders = render_case(ctx, probe.name, model, items, probe.spec_refs)
        if renders.failed:
            checks += [c for c in renders.failed if ctx.wants(c.id)]
            continue
        assert renders.a is not None and renders.b is not None
        found = item_checks(
            ctx,
            level="L1",
            case=probe.name,
            prefix=prefix,
            items=items,
            first=renders.a,
            second=renders.b,
            tier=ctx.tier,
            inputs=renders.inputs,
            spec_refs=probe.spec_refs,
            swept_name=probe.swept,
        )
        _analyse(ctx, probe, items, renders.a, renders.b, found)
        checks += found
    return checks

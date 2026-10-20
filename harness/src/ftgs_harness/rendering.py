# SPDX-License-Identifier: Apache-2.0
"""Render comparisons shared by L0 and L1 (§3.3): one request, both adapters (or one adapter and
two models), one check per item with its measures, tier verdict, symptom and artifacts."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ftgs_harness.adapter import Outcome
from ftgs_harness.artifacts import save_pair
from ftgs_harness.checks import Check
from ftgs_harness.compare import measure, verdict
from ftgs_harness.formats import FormatError, RequestItem, read_renders, write_request
from ftgs_harness.info import Tier
from ftgs_harness.runner import Context
from ftgs_harness.symptoms import classify, describe, measure_pair

__all__ = [
    "CaseItem",
    "RenderResult",
    "failure_check",
    "item_checks",
    "relative",
    "render",
    "request_for",
    "where_text",
]

_MAX_ARTIFACTS = 12  # failed items per case that get images


@dataclass(frozen=True)
class CaseItem:
    """A request item and what the symptom names it by."""

    item: RequestItem
    camera: str
    swept: float | None = None  # the probe's swept variable at this item


@dataclass(frozen=True)
class RenderResult:
    outcome: Outcome
    renders: dict[str, tuple[np.ndarray, np.ndarray]] | None
    error: str | None  # a schema violation of the renders file


def request_for(
    ctx: Context, path: Path, items: Sequence[CaseItem], background: Sequence[float]
) -> str:
    """Write the case's request once (a rerun reuses the run's), returning its SHA-256."""
    if not path.exists():
        write_request(path, [i.item for i in items], background=background)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    ctx: Context,
    label: str,
    model: Path,
    request: Path,
    sha: str,
    items: Sequence[CaseItem],
    out: Path,
) -> RenderResult:
    out.parent.mkdir(parents=True, exist_ok=True)
    outcome = ctx.adapters[label].call("render", [model, request, out])
    if not outcome.ok:
        return RenderResult(outcome, None, None)
    try:
        renders = read_renders(out, request_sha256=sha, items=[i.item for i in items])
    except FormatError as error:
        return RenderResult(outcome, None, str(error))
    return RenderResult(outcome, renders, None)


def _rel(ctx: Context, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.run_dir))
    except ValueError:
        return str(path)


def failure_check(
    ctx: Context,
    check_id: str,
    level: str,
    result: RenderResult,
    who: str,
    inputs: Mapping[str, object],
    spec_refs: list[str],
) -> Check:
    """One failed check for a render call that crashed or wrote a malformed file."""
    if result.error is not None:
        pattern, symptom = "schema violation", f"{who}'s renders file fails §2.2: {result.error}."
    else:
        pattern, symptom = "crash", f"{who}'s render call: {result.outcome.describe()}."
    return Check(
        id=check_id,
        level=level,
        status="fail",
        spec_refs=spec_refs,
        inputs={**inputs, "log": result.outcome.log},
        metric="render_call",
        expected="exit 0 and a valid ftgs-renders file",
        observed=result.outcome.status if result.error is None else "schema violation",
        symptom=symptom,
        pattern=pattern,
        repro=ctx.repro(check_id),
    )


def where_text(entry: CaseItem, psnr_db: float, swept_name: str | None = None) -> str:
    """The input terms a symptom names its item by: id, camera, τ, the swept variable, PSNR."""
    item = entry.item
    where = f"in item {item.id} (camera {entry.camera}, τ = {item.timestamp:.4g} s"
    if swept_name not in (None, "τ") and entry.swept is not None:
        where += f", {swept_name} = {entry.swept:.4g}"
    return where + f"; PSNR(A, B) = {psnr_db:.1f} dB)"


def item_checks(
    ctx: Context,
    *,
    level: str,
    case: str,
    prefix: str,
    items: Sequence[CaseItem],
    first: dict[str, tuple[np.ndarray, np.ndarray]],
    second: dict[str, tuple[np.ndarray, np.ndarray]],
    tier: Tier,
    inputs: Mapping[str, object],
    spec_refs: list[str],
    swept_name: str | None = None,
    view_dependent: bool = False,
) -> list[Check]:
    """One check per item: `second`'s render (B's, or the round trip's) against `first`'s."""
    checks = []
    images = 0
    for entry in items:
        item = entry.item
        check_id = f"{prefix}.{item.id}"
        if not ctx.wants(check_id):
            continue
        (rgb_a, alpha_a), (rgb_b, alpha_b) = first[item.id], second[item.id]
        m = measure(rgb_a, rgb_b, alpha_a, alpha_b)
        passed, metric, expected, observed = verdict(m, tier)
        symptom, pattern, artifacts = "", None, []
        localization: dict[str, object] = {"times": [item.timestamp], "cameras": [entry.camera]}
        if not passed:
            pm = measure_pair(rgb_a, rgb_b, alpha_a, alpha_b)
            pattern = classify(pm, view_dependent=view_dependent)
            symptom = describe(pattern, pm, where_text(entry, m.psnr_db, swept_name))
            diff = np.abs(np.nan_to_num(rgb_a) - np.nan_to_num(rgb_b)).max(axis=-1)
            ys, xs = np.nonzero(diff > 1.0 / 255.0)
            if xs.size:
                localization["pixel_bbox"] = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ]
            if images < _MAX_ARTIFACTS:
                artifacts = save_pair(ctx.out, f"{case}_{item.id}", rgb_a, rgb_b)
                artifacts = [_rel(ctx, ctx.out / a) for a in artifacts]
                images += 1
        checks.append(
            Check(
                id=check_id,
                level=level,
                status="pass" if passed else "fail",
                spec_refs=spec_refs,
                inputs={**inputs, "items": [item.id]},
                metric=metric,
                expected=expected,
                observed=observed,
                symptom=symptom,
                localization=localization,
                repro=ctx.repro(check_id),
                artifacts=artifacts,
                pattern=pattern,
            )
        )
    return checks


def relative(ctx: Context, path: Path) -> str:
    return _rel(ctx, path)

# SPDX-License-Identifier: Apache-2.0
"""L0 · boundary converters (§3.3 L0; Q-206, Q-208, Q-209): FTGS-IF models, trained and random,
go through import → export in each implementation. Fields must match within 1e-6 relative
(opacity after the logit clamp, quaternions up to sign), the signed fields against the field's
largest magnitude and the positive ones with a floor of 1e-3 of it (Q-208), every produced file
must pass the schema validator, and the implementation's renders before and after its round
trip must agree within the strict L1 tolerance."""

import shutil
from pathlib import Path

import numpy as np

from ftgs_harness.checks import Check, relative_error
from ftgs_harness.formats import EXCHANGE_KEYS, Exchange, FormatError, read_exchange, write_exchange
from ftgs_harness.models import random_exchange
from ftgs_harness.rendering import failure_check, item_checks, relative, render, request_for
from ftgs_harness.rigs import fit_rig, small_items
from ftgs_harness.runner import Context
from ftgs_harness.settings import HarnessError

__all__ = ["field_errors", "run"]

_SPEC_REFS = ["§2.2", "D-70", "D-71", "D-84"]  # the ftgs-if round trip
_TOLERANCE = 1e-6  # §3.3 L0: relative
# Q-208: the signed fields pass through 0, and a faithful float32 implementation holds them only
# to their rounding at the field's largest magnitude, which is their floor: it may store μt as
# normalized time, positions in a normalized scene frame or the DC coefficient as RGB. The
# positive fields (scale, duration, opacity) keep a floor of 1e-3 of it, so that an error in
# their small entries still shows.
_SIGNED_FIELDS = ("position", "velocity", "rotation", "sh", "time_center")

# The pattern a field's difference shows as (§3.3 symptom rule 4).
_PATTERN = {
    "position": "offset",
    "velocity": "offset",
    "time_center": "temporal shift",
    "duration": "temporal widening",
    "scale": "scale change",
    "rotation": "scale change",
    "opacity": "opacity saturation",
    "sh": "color shift",
}


def _variant(rng: np.random.Generator, index: int) -> Exchange:
    """The random models' mix (Q-209): every SH degree, saturated opacities, a single
    timestamp, wide scales and durations, a single row."""
    kind = index % 5
    degree = index % 4
    if kind == 0:
        return random_exchange(rng, 60, sh_degree=degree)
    if kind == 1:
        return random_exchange(rng, 40, sh_degree=degree, saturated=True)
    if kind == 2:
        return random_exchange(rng, 30, sh_degree=degree, tau0=1.5, tau1=1.5)
    if kind == 3:
        return random_exchange(
            rng, 30, sh_degree=degree, log_scale=(-9.0, -1.0), log_duration=(-6.0, 4.0)
        )
    return random_exchange(rng, 1, sh_degree=degree)


def field_errors(a: Exchange, b: Exchange) -> dict[str, float]:
    """The largest relative error of each field and metadata entry that differs beyond §3.3's
    1e-6 (Q-208); empty when the two models match."""
    errors: dict[str, float] = {}
    if (a.sh_degree, a.tau0, a.tau1, a.time_unit) != (b.sh_degree, b.tau0, b.tau1, b.time_unit):
        errors["metadata"] = float("inf")
    bg_a = (a.render_conventions or {}).get("background", [0.0, 0.0, 0.0])
    bg_b = (b.render_conventions or {}).get("background", [0.0, 0.0, 0.0])
    if bg_a != bg_b:
        errors["background"] = float("inf")
    if errors.get("metadata") or a.num != b.num:
        return {**errors, "shape": float("inf")}
    for key in EXCHANGE_KEYS:
        x = a.arrays[key].astype(np.float64)
        y = b.arrays[key].astype(np.float64)
        if key == "opacity":  # §2.2: compared after the importers' clamp
            x, y = np.clip(x, 1e-7, 1 - 1e-7), np.clip(y, 1e-7, 1 - 1e-7)
        if key == "rotation":  # up to sign
            y = np.where((x * y).sum(-1, keepdims=True) < 0, -y, y)
        floor = 1.0 if key in _SIGNED_FIELDS else 1e-3
        error = relative_error(y, x, floor=floor)
        if not np.all(error <= _TOLERANCE):
            errors[key] = float(np.max(error))
    return errors


def _models(ctx: Context) -> list[tuple[str, Path, bool]]:
    """The case files cases/L0/mNN.ftgs-if.safetensors: trained ones first (copied), then
    random ones up to `l0.models`; a rerun reuses the run's."""
    folder = ctx.cases / "L0"
    folder.mkdir(parents=True, exist_ok=True)
    count = ctx.settings.sizes["l0.models"]
    trained = list(ctx.settings.trained_models[:count])
    out = []
    for i in range(count):
        name = f"m{i:02d}"
        path = folder / f"{name}.ftgs-if.safetensors"
        if not path.exists():
            if i < len(trained):
                shutil.copyfile(trained[i], path)
            else:
                write_exchange(path, _variant(np.random.default_rng([ctx.settings.seed, 0, i]), i))
        out.append((name, path, i < len(trained)))
    return out


def _round_trip(ctx: Context, name: str, path: Path, trained: bool, label: str) -> list[Check]:
    check_id = f"L0.roundtrip.{name}.{label}"
    render_id = f"L0.render.{name}.{label}"
    if not (ctx.wants(check_id) or ctx.wants(render_id)):
        return []
    adapter = ctx.adapters[label]
    folder = ctx.out / label / "L0"
    folder.mkdir(parents=True, exist_ok=True)
    native, exported = folder / f"{name}.native", folder / f"{name}.roundtrip.safetensors"
    inputs = {"model": relative(ctx, path), "trained": trained, "adapter": label}
    try:
        original = read_exchange(path)
    except FormatError as error:
        raise HarnessError(f"L0 case {path}: {error}") from error
    checks: list[Check] = []
    for command, args in (("import", [path, native]), ("export", [native, exported])):
        outcome = adapter.call(command, args)
        if not outcome.ok:
            return [
                Check(
                    id=check_id,
                    level="L0",
                    status="fail",
                    spec_refs=_SPEC_REFS,
                    inputs={**inputs, "log": outcome.log},
                    metric=f"{command}_call",
                    expected="exit 0",
                    observed=outcome.status,
                    symptom=(
                        f"Adapter {label}'s {command} of {inputs['model']}: {outcome.describe()}."
                    ),
                    pattern="crash",
                    repro=ctx.repro(check_id),
                )
            ]
    try:
        back = read_exchange(exported)
    except FormatError as error:
        return [
            Check(
                id=check_id,
                level="L0",
                status="fail",
                spec_refs=_SPEC_REFS,
                inputs=inputs,
                metric="exported_file",
                expected="a valid ftgs-if file",
                observed="schema violation",
                symptom=f"Adapter {label}'s export of {inputs['model']} fails its format: {error}.",
                pattern="schema violation",
                repro=ctx.repro(check_id),
            )
        ]
    errors = field_errors(original, back)
    if ctx.wants(check_id):
        worst = max(errors, key=lambda k: errors[k]) if errors else None
        symptom, pattern = "", None
        if worst is not None:
            pattern = _PATTERN.get(worst, "schema violation")
            if worst == "duration" and _duration_ratio(original, back) < 1.0:
                pattern = "temporal narrowing"
            symptom = (
                f"After adapter {label}'s import and export of {inputs['model']}, "
                + ", ".join(f"{k} differs by up to {v:.3g} relative" for k, v in errors.items())
                + f" (the tolerance is 1e-6) over {original.num} rows of model {name}."
            )
        checks.append(
            Check(
                id=check_id,
                level="L0",
                status="fail" if errors else "pass",
                spec_refs=_SPEC_REFS,
                inputs={**inputs, "exported": relative(ctx, exported)},
                metric="max_relative_field_error",
                expected="<= 1e-06",
                observed=max(errors.values()) if errors else 0.0,
                symptom=symptom,
                localization={"fields": sorted(errors)},
                repro=ctx.repro(check_id),
                pattern=pattern,
            )
        )
    if ctx.wants(render_id):
        checks += _render_check(ctx, name, path, exported, original, label, render_id)
    return checks


def _duration_ratio(a: Exchange, b: Exchange) -> float:
    return float(np.median(b.arrays["duration"] / a.arrays["duration"]))


def _render_check(
    ctx: Context, name: str, path: Path, exported: Path, model: Exchange, label: str, check_id: str
) -> list[Check]:
    """The implementation's renders of the model and of its round trip, at the strict tier."""
    items = small_items(model, fit_rig(model, width=64, height=48))
    request = ctx.cases / "L0" / f"{name}.request.json"
    sha = request_for(request, items, (0.0, 0.0, 0.0))
    folder = ctx.out / label / "L0"
    before = render(ctx, label, path, request, sha, items, folder / f"{name}.before.safetensors")
    after = render(ctx, label, exported, request, sha, items, folder / f"{name}.after.safetensors")
    inputs = {"model": relative(ctx, path), "roundtrip": relative(ctx, exported), "adapter": label}
    for result in (before, after):
        if result.renders is None:
            return [failure_check(ctx, check_id, "L0", result, label, inputs, _SPEC_REFS)]
    assert before.renders is not None and after.renders is not None
    per_item = item_checks(
        ctx,
        level="L0",
        case=f"L0_{name}_{label}",
        prefix=check_id,
        items=items,
        first=before.renders,
        second=after.renders,
        tier="same",  # one implementation, so always the strict tolerance (Q-206)
        inputs=inputs,
        spec_refs=_SPEC_REFS,
    )
    failed = [c for c in per_item if c.status == "fail"]
    worst = failed[0] if failed else min(per_item, key=lambda c: c.observed or float("inf"))
    worst.id = check_id
    worst.repro = ctx.repro(check_id)
    return [worst]


def run(ctx: Context) -> list[Check]:
    jobs = [
        (name, path, trained) + (label,) for name, path, trained in _models(ctx) for label in "AB"
    ]
    results = ctx.map(lambda job: _round_trip(ctx, *job), jobs)
    return [c for checks in results for c in checks]

# SPDX-License-Identifier: Apache-2.0
"""L2 · forward behaviour on random inputs (§3.3 L2; Q-207, Q-208, Q-209): both
implementations answer the same seeded queries, kind by kind, compared at §3.3's tolerances;
`relocation_sampling` by χ² tests instead. A kind missing from an `info` reply is not
comparable. A relative tolerance judges the signed `means_t` against the answer's largest
magnitude and the positive answers with a floor of 1e-3 of it (Q-208)."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ftgs_harness.checks import Check, Status, close_absolute, relative_error
from ftgs_harness.formats import (
    Answer,
    FormatError,
    Query,
    file_sha256,
    read_answers,
    write_queries,
)
from ftgs_harness.info import QUERY_KINDS
from ftgs_harness.queries import generate
from ftgs_harness.rendering import relative
from ftgs_harness.runner import Context
from ftgs_harness.stats import chi2_goodness_of_fit, chi2_homogeneity

__all__ = ["TOLERANCES", "run"]

# §3.3 L2's tolerances, per answer tensor: (kind of tolerance, value); Q-208's readings.
TOLERANCES: dict[str, dict[str, tuple[str, float]]] = {
    "motion": {"means_t": ("relative", 1e-6)},
    "temporal_opacity": {"sigma_t": ("absolute", 1e-6)},
    "effective_opacity": {"opacity_eff": ("absolute", 1e-6)},
    "sh_color": {"colors": ("absolute", 1e-5)},
    "losses": {
        "l1": ("relative", 1e-5),
        "one_minus_ssim": ("relative", 1e-5),
        "reg": ("relative", 1e-5),
        "total": ("relative", 1e-5),
        "lpips_vgg": ("absolute", 1e-3),
    },
    "lr_schedule": {"rates": ("relative", 1e-9)},
    "relocation_math": {"opacity_new": ("relative", 1e-5), "scales_new": ("relative", 1e-5)},
    "velocity_init": {"velocities": ("absolute", 1e-6)},
    "grad_stats": {"grad_norm": ("relative", 1e-3), "visible": ("exact", 0.0)},
    "relocation_score": {"score": ("absolute", 1e-6)},
    "spatial_scale": {"S": ("relative", 1e-9)},
    "growth_decisions": {"clone": ("exact", 0.0), "split": ("exact", 0.0)},
}
# Q-208: the signed answers, held by a faithful float32 implementation only to their rounding
# at the answer's largest magnitude, which is their floor; the other relative answers are
# positive and keep a floor of 1e-3 of it.
_SIGNED_ANSWERS = {("motion", "means_t")}
_SAMPLING_P = 0.001  # §3.3: χ² test, p > 0.001

_SPEC_REFS = {
    "motion": ["E1"],
    "temporal_opacity": ["E4"],
    "effective_opacity": ["E3", "E4"],
    "sh_color": ["E2"],
    "losses": ["E7", "E8", "E9"],
    "lr_schedule": ["E12", "E13"],
    "relocation_math": ["E11", "D-14"],
    "velocity_init": ["E15"],
    "relocation_sampling": ["D-15"],
    "grad_stats": ["E10", "D-12"],
    "relocation_score": ["E10", "D-13"],
    "spatial_scale": ["D-27"],
    "growth_decisions": ["D-18", "D-44"],
}

_PATTERN = {
    "motion": "offset",
    "effective_opacity": "opacity saturation",
    "sh_color": "view-dependent color shift",
}


def _required(kind: str, query: Query) -> list[str]:
    """The tensors an ok answer to `query` carries (Q-207): its kind's row, `lpips_vgg` only
    when the query's `lpips` is true (Q-212)."""
    names = list(TOLERANCES[kind])
    if kind == "losses" and not query.params.get("lpips", True):
        names.remove("lpips_vgg")
    return names


def _worst(kind: str, names: list[str], a: Answer, b: Answer) -> tuple[bool, str, float, str]:
    """Whether B's answer matches A's on the tensors `names`, which both carry, and the worst
    tensor, its discrepancy and tolerance."""
    worst = (True, "", 0.0, "")
    for name in names:
        mode, tol = TOLERANCES[kind][name]
        x, y = a.tensors[name], b.tensors[name]
        if x.shape != y.shape:
            return False, name, float("inf"), f"shape {list(x.shape)}"
        if mode == "exact":
            mismatches = float(np.sum(x != y))
            if mismatches:
                return False, name, mismatches, "exact"
            continue
        if mode == "relative":
            diff = relative_error(y, x, floor=1.0 if (kind, name) in _SIGNED_ANSWERS else 1e-3)
            ok = diff <= tol
        else:
            ok = close_absolute(y, x, tol)
            diff = np.nan_to_num(np.abs(x.astype(np.float64) - y.astype(np.float64)), nan=np.inf)
        value = float(np.max(diff)) if diff.size else 0.0
        if not bool(np.all(ok)):
            return False, name, value, f"<= {tol:g} {mode}"
        if value > worst[2]:
            worst = (True, name, value, f"<= {tol:g} {mode}")
    return worst


def _describe(
    kind: str,
    query_id: str,
    name: str,
    value: float,
    expected: str,
    tensors: Mapping[str, np.ndarray],
) -> str:
    shape = ", ".join(
        f"{k} {list(v.shape)}" for k, v in sorted(tensors.items()) if not k.startswith("model/")
    )
    return (
        f"For {kind} query {query_id} ({shape}), B's {name} differs from A's by up to {value:.3g} "
        f"({expected.split()[-1] if expected.split() else 'exact'}); the tolerance is {expected}."
    )


def _sampling_checks(
    ctx: Context,
    queries: list[Any],
    answers: dict[str, dict[str, Answer]],
    inputs: dict[str, object],
) -> list[Check]:
    checks = []
    for query in queries:
        check_id = f"L2.forward.relocation_sampling.{query.id}"
        if not ctx.wants(check_id):
            continue
        scores = query.tensors["scores"]
        pooled = {}
        bad: list[str] = []
        for label in ("A", "B"):
            answer = answers[label].get(query.id)
            counts = None if answer is None else answer.tensors.get("counts")
            if answer is None or answer.status != "ok" or counts is None:
                bad.append(label)
                continue
            pooled[label] = counts.sum(axis=0)
        if bad:
            checks.append(
                _status_check(ctx, check_id, "relocation_sampling", bad, answers, query.id, inputs)
            )
            continue
        p_a = chi2_goodness_of_fit(pooled["A"], scores)
        p_b = chi2_goodness_of_fit(pooled["B"], scores)
        p_ab = chi2_homogeneity(pooled["A"], pooled["B"])
        observed = min(p_a, p_b, p_ab)
        status: Status = "pass" if observed > _SAMPLING_P else "fail"
        symptom = ""
        if status == "fail":
            symptom = (
                f"For relocation_sampling query {query.id} ({len(scores)} rows, "
                f"{(scores > 0).sum()} with a positive score; {len(query.params['seeds'])} seeds × "
                f"{query.params['count']} draws), the "
                f"pooled counts give χ² p = {p_a:.3g} for A and {p_b:.3g} for B against the "
                f"scores, and p = {p_ab:.3g} between A and B; each must exceed 0.001."
            )
        checks.append(
            Check(
                id=check_id,
                level="L2",
                status=status,
                spec_refs=_SPEC_REFS["relocation_sampling"],
                inputs={**inputs, "queries": [query.id]},
                metric="chi2_p_min",
                expected="> 0.001",
                observed=observed,
                symptom=symptom,
                localization={"p_A": p_a, "p_B": p_b, "p_A_vs_B": p_ab},
                repro=ctx.repro(check_id),
            )
        )
    return checks


def _status_check(
    ctx: Context,
    check_id: str,
    kind: str,
    labels: list[str],
    answers: Mapping[str, Mapping[str, Answer]],
    query_id: str,
    inputs: Mapping[str, object],
) -> Check:
    found = {label: answers[label].get(query_id) for label in labels}
    statuses = {label: "missing" if a is None else a.status for label, a in found.items()}
    not_comparable = all(s == "not_comparable" for s in statuses.values())
    # §3.3 symptom rule 2: an adapter's own message may name its code, so it stays out of the
    # symptom, in the localization.
    messages = {label: "" if a is None else a.message for label, a in found.items()}
    symptom = (
        ""
        if not_comparable
        else f"The {kind} query {query_id} got no usable answer from {', '.join(labels)}: "
        + ", ".join(f"{k} {v}" for k, v in statuses.items())
        + "."
    )
    return Check(
        id=check_id,
        level="L2",
        status="not_comparable" if not_comparable else "fail",
        spec_refs=_SPEC_REFS[kind],
        inputs={**inputs, "queries": [query_id]},
        metric="answer_status",
        expected="ok",
        observed=", ".join(f"{k} {v}" for k, v in statuses.items()),
        symptom=symptom,
        localization={"adapter_messages": messages},
        pattern=None if not_comparable else "crash",
        repro=ctx.repro(check_id),
    )


def _absent_tensors_check(
    ctx: Context,
    check_id: str,
    kind: str,
    names: list[str],
    absent: Mapping[str, list[str]],
    query_id: str,
    inputs: Mapping[str, object],
) -> Check:
    """§3.3: an ok answer without a tensor of its kind's row (Q-207) fails its format."""
    observed = "; ".join(f"{label} lacks {', '.join(lacks)}" for label, lacks in absent.items())
    counts = "; ".join(
        f"{label}'s lacks {len(lacks)} ({', '.join(lacks)})" for label, lacks in absent.items()
    )
    return Check(
        id=check_id,
        level="L2",
        status="fail",
        spec_refs=_SPEC_REFS[kind],
        inputs={**inputs, "queries": [query_id]},
        metric="answer_tensors",
        expected=", ".join(names),
        observed=observed,
        symptom=f"Of the {len(names)} tensors of a {kind} answer to query {query_id}, {counts}.",
        pattern="schema violation",
        repro=ctx.repro(check_id),
    )


def _kind_checks(ctx: Context, kind: str) -> list[Check]:
    prefix = f"L2.forward.{kind}"
    missing = [label for label in ("A", "B") if kind not in ctx.query_kinds(label)]
    if missing:
        return [
            Check(
                id=prefix,
                level="L2",
                status="not_comparable",
                spec_refs=_SPEC_REFS[kind],
                inputs={"kind": kind, "adapters": missing},
                metric="query_kinds",
                expected=f"{kind} in both info replies",
                observed=f"missing from {', '.join(missing)}",
                repro=ctx.repro(prefix),
            )
        ]
    folder = ctx.cases / "L2"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{kind}.queries.safetensors"
    config = {**ctx.config, "_lpips": ctx.settings.l2_lpips}
    queries = generate(kind, ctx.settings.sizes["l2.queries"], ctx.settings.seed, config)
    if not path.exists():
        write_queries(path, kind, queries)
    sha = file_sha256(path)
    inputs: dict[str, object] = {"queries_file": relative(ctx, path)}
    answers: dict[str, dict[str, Answer]] = {}
    failures = []
    for label in ("A", "B"):
        out = ctx.out / label / "L2" / f"{kind}.answers.safetensors"
        out.parent.mkdir(parents=True, exist_ok=True)
        outcome = ctx.adapters[label].call("forward", [path, out])
        if not outcome.ok:
            failures.append(
                Check(
                    id=f"{prefix}.{label}",
                    level="L2",
                    status="fail",
                    spec_refs=_SPEC_REFS[kind],
                    inputs={**inputs, "log": outcome.log},
                    metric="forward_call",
                    expected="exit 0",
                    observed=outcome.status,
                    symptom=f"Adapter {label}'s forward call for {kind}: {outcome.describe()}.",
                    pattern="crash",
                    repro=ctx.repro(f"{prefix}.{label}"),
                )
            )
            continue
        try:
            answers[label] = read_answers(out, kind=kind, queries_sha256=sha)
        except FormatError as error:
            failures.append(
                Check(
                    id=f"{prefix}.{label}",
                    level="L2",
                    status="fail",
                    spec_refs=_SPEC_REFS[kind],
                    inputs=inputs,
                    metric="answers_file",
                    expected="a valid ftgs-forward-answers file",
                    observed="schema violation",
                    symptom=f"Adapter {label}'s {kind} answers fail their format: {error}.",
                    pattern="schema violation",
                    repro=ctx.repro(f"{prefix}.{label}"),
                )
            )
    if failures:
        return [c for c in failures if ctx.wants(c.id)]
    if kind == "relocation_sampling":
        return _sampling_checks(ctx, queries, answers, inputs)
    checks = []
    for query in queries:
        check_id = f"{prefix}.{query.id}"
        if not ctx.wants(check_id):
            continue
        a, b = answers["A"].get(query.id), answers["B"].get(query.id)
        bad = [label for label, ans in (("A", a), ("B", b)) if ans is None or ans.status != "ok"]
        if bad:
            checks.append(_status_check(ctx, check_id, kind, bad, answers, query.id, inputs))
            continue
        assert a is not None and b is not None
        names = _required(kind, query)
        absent = {
            label: lacks
            for label, ans in (("A", a), ("B", b))
            if (lacks := [n for n in names if n not in ans.tensors])
        }
        if absent:
            checks.append(
                _absent_tensors_check(ctx, check_id, kind, names, absent, query.id, inputs)
            )
            continue
        ok, name, value, expected = _worst(kind, names, a, b)
        checks.append(
            Check(
                id=check_id,
                level="L2",
                status="pass" if ok else "fail",
                spec_refs=_SPEC_REFS[kind],
                inputs={**inputs, "queries": [query.id]},
                metric=f"{name or 'answer'}_difference",
                expected=expected or "match",
                observed=value,
                symptom=""
                if ok
                else _describe(kind, query.id, name, value, expected, query.tensors),
                pattern=None if ok else _PATTERN.get(kind),
                repro=ctx.repro(check_id),
            )
        )
    return checks


def run(ctx: Context) -> list[Check]:
    kinds = [k for k in QUERY_KINDS if ctx.wants(f"L2.forward.{k}")]
    results = ctx.map(lambda kind: _kind_checks(ctx, kind), kinds)
    return [c for checks in results for c in checks]

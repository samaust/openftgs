# SPDX-License-Identifier: Apache-2.0
"""L1 · identical model, near-exact renders (§3.3 L1; Q-206, Q-209, Q-211): both
implementations render the same FTGS-IF file and the same request; per item, PSNR(A, B), max and
mean |Δ|, the share of pixels with |Δ| > 1/255 and the alpha difference, judged at the tier the
two `info` replies give. The probes P1 to P10 isolate one behaviour each (`probes`)."""

from dataclasses import dataclass

import numpy as np

from ftgs_harness.checks import Check
from ftgs_harness.formats import Exchange, FormatError, read_exchange, write_exchange
from ftgs_harness.models import random_exchange
from ftgs_harness.rendering import (
    CaseItem,
    failure_check,
    item_checks,
    relative,
    render,
    request_for,
)
from ftgs_harness.rigs import fit_rig, main_items
from ftgs_harness.runner import Context
from ftgs_harness.settings import HarnessError

__all__ = ["CaseRenders", "compare_case", "render_case", "run"]

_SPEC_REFS = ["E1", "E2", "E3", "E4", "E5", "E6"]


@dataclass
class CaseRenders:
    """A case's renders by A and B, or the failed checks when a render call failed."""

    inputs: dict[str, object]
    a: dict[str, tuple[np.ndarray, np.ndarray]] | None
    b: dict[str, tuple[np.ndarray, np.ndarray]] | None
    failed: list[Check]


def render_case(
    ctx: Context, case: str, model: Exchange | None, items: list[CaseItem], spec_refs: list[str]
) -> CaseRenders:
    """Write the case's model and request (once: a rerun reuses the run's files) and render it
    with A and B."""
    folder = ctx.cases / "L1"
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / f"{case}.ftgs-if.safetensors"
    if not model_path.exists():
        assert model is not None
        write_exchange(model_path, model)
    request = folder / f"{case}.request.json"
    sha = request_for(request, items, (0.0, 0.0, 0.0))
    inputs: dict[str, object] = {
        "model": relative(ctx, model_path),
        "request": relative(ctx, request),
    }
    results = dict(
        zip(
            ("A", "B"),
            ctx.map(
                lambda label: render(
                    ctx,
                    label,
                    model_path,
                    request,
                    sha,
                    items,
                    ctx.out / label / "L1" / f"{case}.renders.safetensors",
                ),
                ["A", "B"],
            ),
            strict=True,
        )
    )
    failed = [
        failure_check(ctx, f"L1.render.{case}.{label}", "L1", r, label, inputs, spec_refs)
        for label, r in results.items()
        if r.renders is None
    ]
    return CaseRenders(inputs, results["A"].renders, results["B"].renders, failed)


def compare_case(
    ctx: Context,
    case: str,
    model: Exchange | None,
    items: list[CaseItem],
    *,
    spec_refs: list[str],
) -> list[Check]:
    """Render one case with A and B and compare item by item at the run's tier."""
    renders = render_case(ctx, case, model, items, spec_refs)
    if renders.failed:
        return [c for c in renders.failed if ctx.wants(c.id)]
    assert renders.a is not None and renders.b is not None
    return item_checks(
        ctx,
        level="L1",
        case=case,
        prefix=f"L1.render.{case}",
        items=items,
        first=renders.a,
        second=renders.b,
        tier=ctx.tier,
        inputs=renders.inputs,
        spec_refs=spec_refs,
    )


def _main_model(ctx: Context, rng: np.random.Generator) -> Exchange:
    """The main case's model: the first trained model of the settings, else a random one with
    N = 1,000 (Q-209)."""
    if ctx.settings.trained_models:
        try:
            return read_exchange(ctx.settings.trained_models[0])
        except FormatError as error:
            raise HarnessError(f"l0.trained_models[0]: {error}") from error
    return random_exchange(rng, 1000, sh_degree=3)


def run(ctx: Context) -> list[Check]:
    checks: list[Check] = []
    if ctx.wants("L1.render.main"):
        existing = ctx.cases / "L1" / "main.ftgs-if.safetensors"
        if existing.exists():  # a rerun: the run's model
            model = read_exchange(existing)
        else:
            model = _main_model(ctx, np.random.default_rng([ctx.settings.seed, 1, 0]))
        rng = np.random.default_rng([ctx.settings.seed, 1, 1])
        items = main_items(model, fit_rig(model), ctx.settings.sizes["l1.items"], rng)
        checks += compare_case(ctx, "main", model, items, spec_refs=_SPEC_REFS)
    from ftgs_harness import probes  # probes builds on this module's cases

    return checks + probes.run(ctx)

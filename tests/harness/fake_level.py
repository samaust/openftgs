# SPDX-License-Identifier: Apache-2.0
"""A level for the harness's own tests of its run mechanics: one check per adapter call
`echo`, passing when the call exits 0, and one check with an infinite observed value."""

from ftgs_harness.checks import Check
from ftgs_harness.runner import Context


def run(ctx: Context) -> list[Check]:
    checks = []
    for label, adapter in ctx.adapters.items():
        check_id = f"L0.fake.{label}"
        if not ctx.wants(check_id):
            continue
        outcome = adapter.call("echo", [check_id])
        checks.append(
            Check(
                id=check_id,
                level="L0",
                status="pass" if outcome.ok else "fail",
                spec_refs=["E1"],
                inputs={"adapter": label},
                metric="exit_code",
                expected="0",
                observed=outcome.returncode,
                symptom="" if outcome.ok else f"Adapter {label} {outcome.describe()}.",
                pattern=None if outcome.ok else "crash",
                repro=ctx.repro(check_id),
            )
        )
    if ctx.wants("L0.fake.inf"):
        checks.append(
            Check(
                id="L0.fake.inf",
                level="L0",
                status="pass",
                spec_refs=[],
                inputs={},
                metric="psnr_A_vs_B_db",
                expected=">= 60",
                observed=float("inf"),
                repro=ctx.repro("L0.fake.inf"),
            )
        )
    return checks

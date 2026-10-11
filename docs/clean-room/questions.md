# Implementation questions and provisional decisions

Gaps found in the spec (revision 549) during implementation. Every row is provisional until the owner confirms it; confirmed rows return to the spec as new D-n decisions in a later export. Code that relies on a row carries a `# Q-n` comment.

Status is one of: `provisional` (a new row), `confirmed` (the owner accepted the provisional decision), `changed (#n)` (the owner chose otherwise; the row's decision is updated and the code follows in issue #n: the row's own ticket when the owner answered during its build, otherwise an issue implemented on its own), or `in spec (D-n, revision r)` (returned to the spec).

| ID | Spec location | Gap | Provisional decision | Rationale | Alternatives | Detecting test | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Calibration data

Measured values for the thresholds and budgets the spec calls calibrated (glossary: "Calibrated threshold"), and the calibration runs of M11, M15 and M17. Each row records one measurement; the spec's criteria are never loosened here, and what becomes a spec change is the owner's decision.

| Spec item | Quantity | Value | Run (commit, machine, scene) | Date | Owner's decision |
| --- | --- | --- | --- | --- | --- |

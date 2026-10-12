# Implementation questions and provisional decisions

Gaps found in the spec (revision 549) during implementation. Every row is provisional until the owner confirms it; confirmed rows return to the spec as new D-n decisions in a later export. Code that relies on a row carries a `# Q-n` comment.

Status is one of: `provisional` (a new row), `confirmed` (the owner accepted the provisional decision), `changed (#n)` (the owner chose otherwise; the row's decision is updated and the code follows in issue #n: the row's own ticket when the owner answered during its build, otherwise an issue implemented on its own), or `in spec (D-n, revision r)` (returned to the spec).

| ID | Spec location | Gap | Provisional decision | Rationale | Alternatives | Detecting test | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q-1 | §2.3 | No minimum Python version is stated (from #1: M0 · Repository skeleton) | `requires-python >= 3.11`; development and CI use Python 3.14 (owner answer, 2026-10-11) | owner: 3.10 is past the end of its security support, and the project works on 3.14 | ≥3.9 (gsplat allows ≥3.7); ≥3.10 (the first provisional decision, from fused-local-corr's cp310–cp313 wheels) | install on 3.11 in the M18 CI matrix | changed (#1) |
| Q-2 | §2.3, `[fast]` row | fused-ssim is not on PyPI; a git-URL dependency would block a PyPI upload (from #1: M0 · Repository skeleton) | `[fast]` stays an empty extra; the README says to install fused-ssim from its MIT repository; M6, which first uses it, imports it optionally and names the commit it was tested with | keeps the M18 PyPI release possible | `fused-ssim @ git+…@<commit>` in `[fast]`; drop `[fast]` | M18 upload dry run; M6 optional-import test | provisional |
| Q-3 | §2.3, torch row | Package metadata can't require a "CUDA build" of torch (from #1: M0 · Repository skeleton) | `torch>=2.1` in metadata; the README documents the CUDA wheel index; CI installs the CPU wheel | package metadata can't choose a torch build | per-CUDA extras through indexes | GPU-marked tests skip on a CPU build | provisional |
| Q-4 | §2.3 notes, CI scan | Which resolved environment CI scans, and how LGPL and unknown licenses are treated (from #1: M0 · Repository skeleton) | CI scans the CPU dev environment (`.[dev]` plus the harness); GPL and AGPL fail; LGPL, UNKNOWN and any license not recognised as permissive or MPL are listed for sign-off and pass; `[init]`, `[fast]` and CUDA torch are scanned at M18 | `[init]`/`[fast]` need CUDA builds the CI runner lacks; the spec says LGPL needs sign-off, not failure, and CLAUDE.md rules out non-commercial and missing licenses, which an unrecognised string may hide | scan `.[dev,init]` on CPU; fail on UNKNOWN | `tests/test_license_scan.py` | provisional |
| Q-5 | §2.1, `ftgs_harness` row | How `ftgs_harness` is packaged (from #1: M0 · Repository skeleton) | a separate distribution `ftgs-harness` in `harness/` with its own pyproject, declaring matplotlib now | "separate package" that never imports openftgs; keeps matplotlib out of openftgs installs | subpackage in the openftgs wheel; declare matplotlib at M16 | the skeleton test's import-boundary check | provisional |

## Calibration data

Measured values for the thresholds and budgets the spec calls calibrated (glossary: "Calibrated threshold"), and the calibration runs of M11, M15 and M17. Each row records one measurement; the spec's criteria are never loosened here, and what becomes a spec change is the owner's decision.

| Spec item | Quantity | Value | Run (commit, machine, scene) | Date | Owner's decision |
| --- | --- | --- | --- | --- | --- |

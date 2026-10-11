# Implementation source log

This log continues the spec's source log (Appendix A of `docs/spec/openftgs-spec.md`) for the implementation phase. Every source consulted gets one row: what it was and what was taken from it. Reading this repository and the spec needs no row.

If a forbidden source is met by accident, its row starts with **Incident** and says what was seen, that reading stopped, and that nothing was taken.

| # | Date | Source | What was taken |
| --- | --- | --- | --- |
| 1 | 2026-10-11 | PyPI JSON API, `https://pypi.org/pypi/<name>/json`, for gsplat, fused-ssim, fused-local-corr, lpips and pip-licenses (package metadata only, no source code) | gsplat 1.5.3 ships a `py3-none-any` wheel and an sdist, requires-python ≥3.7; fused-ssim has no PyPI project (404); fused-local-corr latest 0.3.211, MIT, wheels for Linux cp310–cp313 only, requires-python ≥3.9; lpips 0.1.4 ships a `py3-none-any` wheel; pip-licenses 5.5.5, MIT, requires-python ≥3.9 (from #1: M0 · Repository skeleton) |

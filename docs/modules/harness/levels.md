# Levels, statistics and the report

The harness runs the four **levels** of §3.3. Case generation, tolerances and statistics that §3.3 leaves open are Q-208 to Q-211.

## Tolerances

A bare tolerance is absolute; "relative" means |a − b| ≤ tol · max(|a|, |b|, 1e-3 · ‖b‖∞), elementwise (Q-208).

## L0 · boundary converters

The trained models of the run settings, then seeded random models up to `l0.models` (varied N, SH degree 0 to 3, a single-timestamp model, saturated opacities, edge scales and durations). Each goes import → export through each implementation; the fields must agree within 1e-6 relative, opacity after the [10⁻⁷, 1 − 10⁻⁷] clamp and quaternions up to sign. Each implementation also renders a 4-item request before and after its round trip, compared at the strict L1 tolerance, since both renders come from one implementation (Q-206, Q-209).

## L1 · identical model, near-exact renders

A main case on a ring rig of 8 cameras at 128×96 and 20 frames, with at least `l1.items` items (training cameras, interpolated novel cameras, times on frames, between frames, at τ₀ and τ₁, and 10 % outside), plus the ten **probes**, each a sweep of `l1.probe_items` items at 64×48. The sweeps of P4, P6, P7 and P8 keep clear of Q-39's margins around E6's cutoffs (Q-209). Per item: PSNR(A, B) on the unclamped renders with peak 1, max and mean |Δ|, the share of pixels with |Δ| > 1/255 in any channel, and the alpha difference; the tier comes from `info` (`formats.md`).

## L2 · forward behaviour

`l2.queries` seeded queries per kind plus edge queries, compared at §3.3's tolerances. `relocation_sampling` uses 20 live targets and 50 × 200 draws; each implementation's pooled counts are tested by χ² against the scores, and A against B by homogeneity, all at p > 0.001.

## L3 · multi-seed statistics

Seeds 0…K−1 for A and B, K…2K−1 for A′, one adapter process per run, round robin over the settings' GPUs with `CUDA_VISIBLE_DEVICES` set to one of them (D-54). The held-out request comes from the manifest's test split with the config's background; the native model renders it, and the exported FTGS-IF file feeds the distributions (Q-210).

- Image metrics (E16, computed by the harness): PSNR, SSIM₁, SSIM₂, LPIPS-Alex and LPIPS-VGG are tested by Welch and TOST at α = 0.05, Holm-corrected across metrics × scenes; DSSIM is reported only.
- Distributions: d(A, B) − d(A, A′) for KS and Wasserstein-1, by an exact permutation over the C(2K, K) splits of B ∪ A′. A flag (p < 0.01) is a failed check, Holm-corrected at 0.01 across quantities × distances × scenes; when no flag is reachable at this K the check is `inconclusive`, naming the smallest reachable p and the K needed.
- Traces: loss terms and N at 1k, 7k, 15k and 30k scaled by `optim.iterations`/30,000, reported by median and spread.

## Report

`report.json` follows §3.3's example; `report.md` is generated from it, failed checks first, grouped by pattern. Statuses are `pass`, `fail`, `inconclusive` and `not_comparable`; a level is `fail` if any check failed, otherwise `inconclusive` if any check was, otherwise `pass`. Non-finite values are written as null (Q-63); an identical render's PSNR is null with an "inf" note (Q-211).

Symptoms describe inputs and outputs only, in §3.3's fixed vocabulary, measured from the alpha centroid, second moments, alpha mass, the colour under the alpha mask, the alpha mass against time, a border band and non-finite values. The trend against the swept variable is the constant, linear or quadratic least-squares fit with the lowest BIC.

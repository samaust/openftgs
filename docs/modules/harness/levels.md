# Levels, statistics and the report

The harness runs the four **levels** of §3.3. Case generation, tolerances and statistics that §3.3 leaves open are Q-208 to Q-211.

## Tolerances

A bare tolerance is absolute; "relative" means |a − b| ≤ tol · max(|a|, |b|, f · M), elementwise, where M = max(‖a‖∞, ‖b‖∞) over the finite entries of the compared arrays, so swapping A and B gives the same verdict; NaN never passes (Q-208). f = 1 for signed quantities, which a faithful float32 implementation holds only to their rounding at M: L0's position, velocity, rotation, SH and time centre, and L2's `means_t`. f = 1e-3 for positive quantities, so that an error in their small entries still shows: L0's scale, duration and opacity, and L2's other relative answers. A check reports its relative error with the same denominator.

## Cases and check IDs

Every case file goes under `cases/`, written once, so `rerun` replays the run's own inputs; seeds come from the run settings' `seed`, a stream per level and case. Check IDs: `L0.roundtrip.mNN.<A|B>`, `L0.render.mNN.<A|B>`, `L1.render.<case>.<item>` (`main` or `P1` to `P10`), `L1.render.<case>.<A|B>` for a render call that failed, `L2.forward.<kind>.<query>`, `L2.forward.<kind>` when the kind is missing from an `info` reply, `L3.run.<scene>.<impl>.seed<k>`, `L3.metric.<scene>.<metric>`, `L3.dist.<scene>.<quantity>.<ks|w1>`, `L3.trace.<scene>.<1k|7k|15k|30k>` and `L3.config.<scene>.<impl>.<key>`; the `info` checks are `<lowest level>.info.<A|B>`.

## L0 · boundary converters

The trained models of the run settings, then seeded random models up to `l0.models`, in a mix of five: 60 rows, saturated opacities, a single timestamp, wide scales (e⁻⁹ to e⁻¹) and durations (e⁻⁶ to e⁴ × T_τ), and a single row, over SH degrees 0 to 3. Each goes import → export through each implementation; the fields must agree within 1e-6 relative, opacity after the [10⁻⁷, 1 − 10⁻⁷] clamp, quaternions up to sign, and the metadata exactly, the background of D-84 included. The signed fields are judged against the largest magnitude of the field in either model, the positive ones (scale, duration, opacity) with the 1e-3 floor (Q-208). Each implementation also renders a 4-item request at 64×48 before and after its round trip, compared at the strict L1 tolerance, since both renders come from one implementation (Q-206, Q-209); the check reports the worst item. A field's difference is named by the pattern it would show: position and velocity an offset, time centre a temporal shift, duration a temporal widening or narrowing, scale and rotation a scale change, opacity opacity saturation, SH a colour shift.

## L1 · identical model, near-exact renders

A main case: the first trained model of the run settings, else a random one with 1,000 rows, on a rig fitted to it (8 cameras on a circle at three times the 90th-percentile distance of the positions from their median, 128×96, 20 frames), with `l1.items` items: training cameras on frames (30 %), between frames (15 %) and at random times (10 %), novel cameras between them (25 %), τ₀ and τ₁ (10 %), and 10 % outside [τ₀, τ₁]. Then the ten **probes**, each one or two primitives seen from 4 units at 64×48 with `l1.probe_items` items sweeping one variable: P1 the principal point by a fraction of a pixel, P2 and P3 the time, P4 the effective opacity from 0.9 to 1 through the temporal opacity, P5 the viewing direction around a primitive with only l ≥ 1 SH, P6 the camera depth through the near plane, P7 the time across a depth swap, P8 the effective opacity across 1/255, P9 the image position of a large primitive, P10 P2's time outside [τ₀, τ₁]. The sweeps of P4, P6, P7 and P8 keep clear of E6's cutoffs (Q-39, Q-209). Per item: PSNR(A, B) on the unclamped renders with peak 1, max and mean |Δ|, the share of pixels with |Δ| > 1/255 in any channel, and the alpha difference; the tier comes from `info` (`formats.md`).

## L2 · forward behaviour

`l2.queries` seeded queries per kind plus one edge query (durations from 10⁻⁶ to 10² × T_τ with |t − μt|/s up to 10⁴, opacities near 0 and 1, view directions along the axes, the D-44 cap with ties), compared at §3.3's tolerances. The queries take the run's configuration where a kind has settings (the loss weights, the optimizer's rates and length, the growth and relocation thresholds, the scene-scale factor). `relocation_sampling` uses 20 live targets and 50 × 200 draws; each implementation's pooled counts are tested by χ² against the scores, and A against B by homogeneity, all at p > 0.001.

## L3 · multi-seed statistics

Seeds 0…K−1 for A and B, K…2K−1 for A′, one adapter process per run, round robin over the settings' GPUs with `CUDA_VISIBLE_DEVICES` set to one of them (D-54). The held-out request comes from the manifest's test split with the config's background; the native model renders it, and the exported FTGS-IF file feeds the distributions (Q-210).

- Image metrics (E16, computed by the harness from the saved renders, ꟻLIP with its own copy, `overview.md`): PSNR, SSIM₁, SSIM₂ and ꟻLIP are tested by Welch and TOST at α = 0.05 with §3.3's margins of ±0.25 dB, ±0.005, ±0.005 and ±0.005, Holm-corrected across the four metrics × scenes (Q-210); DSSIM is reported only.
- Distributions: d(A, B) − d(A, A′) for KS and Wasserstein-1, by an exact permutation over the C(2K, K) splits of B ∪ A′. A flag (p < 0.01) is a failed check, Holm-corrected at 0.01 across quantities × distances × scenes; when no flag is reachable at this K the check is `inconclusive`, naming the smallest reachable p and the K needed.
- Traces: loss terms and N at 1k, 7k, 15k and 30k scaled by `optim.iterations`/30,000, reported by median and spread, with whether B's median lies within A's and A′'s spread.
- Each run's samples are subsampled to 20,000 per quantity before KS and W1.

## Report

`report.json` follows §3.3's example; `report.md` is generated from it, failed checks first, grouped by pattern. Statuses are `pass`, `fail`, `inconclusive` and `not_comparable`; a level is `fail` if any check failed, otherwise `inconclusive` if any check was, otherwise `pass` if any check passed, otherwise `not_comparable` (a level with nothing compared). Non-finite values are written as null (Q-63); an identical render's PSNR is null with an "inf" note (Q-211).

Symptoms describe inputs and outputs only, in §3.3's fixed vocabulary, measured from the alpha centroid, second moments, alpha mass, the colour under the alpha mask, the alpha's gradient energy, a border band and non-finite values, checked in that order. A failing probe reports one pattern: its worst item's, or the temporal shift, widening or narrowing its time sweep shows (the alpha mass against τ, unless B's is A's scaled, which is a missing or extra primitive), or a view-dependent colour shift when P5's colour difference changes with the direction. The trend against the swept variable is the constant, linear or quadratic least-squares fit with the lowest BIC, residuals within 0.1 % of the data counting as exact, written to `localization.trend` as §3.3's example does, with its coefficients and a plot under `artifacts/`. Pattern thresholds are Q-211's.

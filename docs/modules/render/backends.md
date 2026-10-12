# Render backends in `openftgs.render`

How `render()` meets its §2.1 row for E2, E5 and E6, and how its pure-PyTorch reference backend is built. The gsplat backend arrives in M5 and the temporal index (D-55) in M15; both are meant to reuse the front half described here. The keyword `precull` (Q-31) and the semantics §2.1 leaves open (Q-32, Q-33) are provisional decisions or owner answers recorded in `docs/clean-room/questions.md`.

## The front half, shared by every backend

`render()` does everything that doesn't depend on the backend, once per call:

1. It checks its arguments: the backend, time units and `precull` choices, a finite timestamp, cameras of one size with the right tensor shapes, and the background's shape (Q-32). Camera values are not checked: that is the scene-manifest reader's job, and a value check on GPU tensors would synchronize (D-49).
2. It turns the timestamp into normalized time. Raw timestamps go through `TimeNormalizer.to_t`, which warns outside the training span (Q-9); normalized ones are used as given, and `render()` warns itself outside the training range (Q-32).
3. It takes the per-time arrays from `FreeTimeGaussians.at_time`, which runs the temporal slice and cuts the SH coefficients to the requested degree, so inactive bands never reach the backend (E13, Q-20).
4. With `precull="compact"`, it applies compaction: only the rows that pass the opacity cull go to the backend, in ascending row order (D-32, D-57, Q-31). The default leaves the opacity cull to the backend's projection.
5. It calls the backend, which always returns outputs over all N rows, whatever subset it was given.

Compaction runs after the temporal slice, not before it, so the slice computes the same values in both placements.

## The reference backend

The reference backend is plain PyTorch (§2.4, rule 4). It runs in the dtype and on the device of the model's parameters, so a model converted with `.double()` renders in float64 for gradient checks (Q-32). Its only Python loop is over cameras.

### Projection and culls

For each camera it projects every row it was given: the camera-space position of the moved position, the 2D mean μ′, the EWA covariance with the Jacobian clamp and the 2D dilation, the conic, and the screen radii from the opacity-aware extent (E6). E5's covariance comes from `openftgs.model.covariance`. A row is visible when both screen radii are positive; every cull of E6 (depth, effective opacity, det Σ′, a screen box off the image) sets them to zero. Q-33 reads the corner cases: a NaN det culls, and a box that only touches the image counts as off it.

Colors follow E2 per camera: the view direction runs from the camera center to the moved position, the SH sum gets its offset and is clamped at zero per channel, and nothing clamps it at one (E7).

### Exactly zero gradients for culled rows

The projection keeps full row shapes and guards every input that could make a local derivative infinite: the depth of a row the depth cull removes and the det Σ′ of a row the det cull removes are replaced by one before anything divides by them, and the extent and the screen radii, which are integers, are computed without gradient and only for rows that pass the opacity cull. Compositing then gathers the visible rows by index, so a culled row's gradient is a sum of zeros times finite numbers: exactly zero, as U-8 requires. A row exactly at the camera center stays finite too, because the direction's normalization has a floor.

### The 2D mean as the gradient anchor

Every camera is projected first, and the 2D means of all cameras are stacked into the `means2d` output: μ′ for visible rows, zero elsewhere, scattered back to all N rows after compaction. Compositing then reads each camera's μ′ from that stacked tensor, so it lies on the path from the parameters to the image, and its retained gradient is the gradient of the loss with respect to μ′ that the gradient statistics need (E10, Q-32). `render()` retains it when `return_aux` is set.

### Compositing

The visible rows are sorted by camera-space depth with a stable sort, so ties go to the lower row (Q-33). Every pixel center is evaluated against every visible primitive, a few pixel rows at a time to bound memory; this dense evaluation gives the same image as gsplat's tiles, because a primitive's alpha reaches the skip threshold only inside its screen box, and the extent's cap is never reached since the effective opacity never exceeds one. So the off-screen cull changes `visible` and `means2d` but never a pixel.

The transmittance along the sorted primitives is a cumulative product. Because it never increases, the primitives kept before the stop of E6 are a prefix, and the one that would cross the threshold is excluded with everything behind it. The output alpha is one minus the final transmittance, and the background fills the rest (D-34).

## Bitwise stability under row subsets

U-8 asks that compaction leave images bitwise unchanged on this backend, so every per-row computation must give the same bits on a subset of rows as on all of them. Matrix products with a camera matrix through BLAS, `einsum` and `sigmoid` don't on the CPU, while elementwise operations, small reductions and the batched products of `covariance` do (source log #17). The camera-side algebra is therefore written as elementwise products and sums, and compaction happens after the slice, whose `sigmoid` sees all rows in both placements. If U-8 ever fails on another CPU in `covariance`'s batched product, the covariance can be built from `rotation_matrix` with elementwise products instead; that is still E5.

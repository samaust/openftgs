# Render backends in `openftgs.render`

How `render()` meets its §2.1 row for E2, E5 and E6, and how its two render backends are built: the pure-PyTorch reference and the gsplat version §2.3 pins. The temporal index (D-55) arrives in M15 and is meant to reuse the front half described here. The keywords `precull` (Q-31) and `packed` (Q-36) and the semantics §2.1 leaves open (Q-32, Q-33) are provisional decisions or owner answers recorded in `docs/clean-room/questions.md`; the gsplat backend's are now in the spec as D-75 and D-76 (revision 549b).

## The front half, shared by every backend

`render()` does everything that doesn't depend on the backend, once per call:

1. It checks its arguments: the backend, time units and `precull` choices, `packed` as a bool (Q-36), a finite real timestamp, cameras of one positive size with the right tensor shapes, and the background's shape (Q-32); for the gsplat backend, a float32 model on a CUDA device (D-76). The camera tensors' values are not checked: that is the scene-manifest reader's job, and a value check on GPU tensors would synchronize (D-49).
2. It turns the timestamp into normalized time. Raw timestamps go through `TimeNormalizer.to_t`, which warns outside the training span (Q-9); normalized ones are used as given, and `render()` warns itself outside the training range (Q-32).
3. It takes the per-time arrays from `FreeTimeGaussians.at_time`, which runs the temporal slice and cuts the SH coefficients to the requested degree, so inactive bands never reach the backend (E13, Q-20).
4. With `precull="compact"`, it applies compaction: only the rows that pass the opacity cull go to the backend, in ascending row order (D-32, D-57, Q-31). The default leaves the opacity cull to the backend's projection.
5. It calls the backend, which always returns outputs indexed by model row, whatever subset it was given: image, alpha, the 2D means ([C, N, 2] in unpacked mode, one row per visible primitive–camera pair in packed mode), `visible` [C, N] and, in packed mode, `gaussian_ids`.
6. With `return_aux`, it retains the gradient of the 2D means the backend returned (Q-32).

Compaction runs after the temporal slice, not before it, so the slice computes the same values in both placements. `packed` selects gsplat's layout (Q-36); the reference backend is always in unpacked mode and ignores it, so consumers tell the layouts apart by `gaussian_ids`, never by the flag.

## The reference backend

The reference backend is plain PyTorch (§2.4, rule 4). It runs in the dtype and on the device of the model's parameters, so a model converted with `.double()` renders in float64, which the tests use for gradient checks and float64 comparisons (Q-32). Its Python loops run over cameras and over blocks of pixel rows, never over primitives or single pixels; with many visible primitives a block shrinks to one pixel row, which rule 4 accepts but which M10 and M11, training on this backend on the CPU, will feel.

### Projection and culls

For each camera it projects every row it was given: the camera-space position of the moved position, the 2D mean μ′, the EWA covariance with the Jacobian clamp and the 2D dilation, the conic, and the screen radii from the opacity-aware extent (E6). E5's covariance comes from `openftgs.model.covariance`, and the opacity cull is the temporal slice's culling mask (`passes_cull`, Q-12), so the threshold has one home. A row is visible when both screen radii are positive; every cull of E6 (depth, effective opacity, det Σ′, a screen box off the image) sets them to zero. Q-33 reads the corner cases: a NaN det culls, and a box that only touches the image counts as off it.

Colors follow E2 per camera: the view direction runs from the camera center to the moved position, the SH sum gets its offset and is clamped at zero per channel, and nothing clamps it at one (E7).

### Exactly zero gradients for culled rows

The projection keeps full row shapes and guards every input that could make a local derivative infinite: the depth of a row the depth cull removes and the det Σ′ of a row the det cull removes are replaced by one before anything divides by them, and the extent and the screen radii, which are integers, are computed without gradient and only for rows that pass the opacity cull. Compositing then gathers the visible rows by index, so a culled row's gradient is a sum of zeros times finite numbers: exactly zero, as U-8 requires. A row exactly at the camera center stays finite too, because the direction's normalization has a floor.

### The 2D mean as the gradient anchor

Every camera is projected first, and the 2D means of all cameras are stacked into the `means2d` output: μ′ for visible rows, zero elsewhere, scattered back to all N rows after compaction. Compositing then reads each camera's μ′ from that stacked tensor, so it lies on the path from the parameters to the image, and its retained gradient is the gradient of the loss with respect to μ′ that the gradient statistics need (E10, Q-32). `render()` retains it when `return_aux` is set.

### Compositing

The visible rows are sorted by camera-space depth with a stable sort, so ties go to the lower row (Q-33). Every pixel center is evaluated against every visible primitive, a few pixel rows at a time to bound memory; this dense evaluation gives the same image as gsplat's tiles, because a primitive's alpha reaches the skip threshold only inside its screen box, and the extent's cap is never reached since the effective opacity never exceeds one. So the off-screen cull changes `visible` and `means2d` but never a pixel.

The transmittance along the sorted primitives is a cumulative product. Because it never increases, the primitives kept before the stop of E6 are a prefix, and the one that would cross the threshold is excluded with everything behind it. The output alpha is one minus the final transmittance, and the background fills the rest (D-34).

## Bitwise stability under row subsets

U-8 asks that compaction change nothing on this backend, within U-8's own tolerance, so every per-row computation must give the same bits on a subset of rows as on all of them. Matrix products with a camera matrix through BLAS, `einsum` and `sigmoid` don't on the CPU, while elementwise operations, small reductions and the batched products of `covariance` do (source log #17). The camera-side algebra is therefore written as elementwise products and sums, and compaction happens after the slice, whose `sigmoid` sees all rows in both placements. If U-8 ever fails on another CPU in `covariance`'s batched product, the covariance can be built from `rotation_matrix` with elementwise products instead; that is still E5.

## The gsplat backend

The production backend runs the CUDA kernels of the gsplat version §2.3 pins (D-6, D-31). It needs per-time arrays in float32 on a CUDA device and raises `ValueError` otherwise, naming the reference backend; the float64 path of Q-32 stays the reference's (D-76).

### Stage functions, not `rasterization()`

The backend calls the stage functions gsplat exports, in the order its top-level `rasterization()` runs them: projection, SH colors, tile intersection and rasterization. Its glue is adapted from `rasterization()` under an attribution comment (source log #27). The kernels and their arguments are the same, and a GPU test checks the backend against `rasterization()` itself. Calling the stages directly is what lets the backend keep Q-32 (D-75):

- In unpacked mode, gsplat's projection leaves the 2D means of culled rows uninitialized. The backend zeroes them before the rasterizer reads them, so the tensor it returns is both μ′-or-zero and the gradient anchor.
- With compaction in unpacked mode, only the 2D means are placed back among all N rows. The rasterizer reads the kept rows from that [C, N, 2] tensor, so it is the anchor, while rasterization still runs over the kept rows only.
- In packed mode, gsplat's own [nnz, 2] tensor is the anchor, since every row of it is written.

The settings of D-6 and D-57 are private constants beside E6's (classic mode, the tile size, no `radius_clip`) or literal arguments (no sparse gradients, an unsegmented sort), and ε, near and far are passed from E6's constants, so a change to any of them is a logged code change (D-72). gsplat's compensation flag is derived from the rasterize-mode constant, and compensations, when computed, scale the effective opacities as `rasterization()` does, so in the gsplat backend the mode has one source. gsplat's projection takes ε, near and far as arguments, as D-72's alternatives expected, while α_max, the α skip and the transmittance stop are fixed in its kernels; the GPU tests of E6's closed forms pin them. The camera tensors both backends read are built once per camera by the front half's helper.

The effective opacity goes into gsplat's projection, which applies the opacity cull and the opacity-aware extent there: D-32's default placement. Inside the projection gsplat applies E6's culls in its own order (depth, det Σ′, effective opacity, then the screen box off the image) and sets the screen radii from the extent rounded up to whole pixels; the reference reaches the same visibility (Q-33).

### Cameras

View matrices are built from R_w and t_w with a last row of (0, 0, 0, 1), and K from f_x, f_y, c_x and c_y only, so the backend reads the same camera fields as the reference (Q-32). The camera center for E2's view direction is E2's own (§1.4), as for the reference, rather than gsplat's inverse of the view matrix: the same in exact arithmetic, and without the host synchronization of `torch.inverse` (§2.4 rule 3; D-76).

### Visibility and ids

A row is visible in a camera when both screen radii gsplat returns are positive (E6). `visible` is [C, N] in both layouts (D-75): in unpacked mode it is read from the radii, in packed mode it is scattered from the camera and primitive ids, with no host synchronization. Packed rows come camera-major and in ascending row order within a camera, as gsplat writes them, and `gaussian_ids` holds model rows, mapped back through compaction (D-76). Inside the call, gsplat's tile intersection gets gsplat's own ids, local to the rows passed.

### Background and empty renders

The pinned gsplat cannot take a per-camera background in packed mode, so the backend composites it itself in both layouts, from the render's alpha, which is one minus E6's final transmittance: the same background term (D-76). gsplat's kernels cannot run on zero rows, so when no rows reach the backend (an empty model, or nothing kept by compaction) it returns Q-32's empty render without calling gsplat. That render stays on the autograd graph, as the reference's does: its outputs depend with weight zero on the 2D means, and those on every per-time array, so a backward pass gives every stored tensor and the 2D means an exactly zero gradient (U-8, D-57), whichever placement of the opacity cull left no rows.

### Host synchronization

A render synchronizes where gsplat reads back the number of tile intersections and, in packed mode, the number of visible pairs; with compaction, `render()`'s own selection of kept rows synchronizes too (D-57). The backend adds nothing else (§2.4 rule 3): its view matrices, the packed visibility scatter and the default background are filled on the device, never written from the host, and a GPU test checks that every synchronization torch's debug mode reports comes from gsplat. Cameras or a background held on the host are copied to the device in every render, which synchronizes, so a caller that renders repeatedly, like the trainer, keeps them on the device.

## Rendering a request

`render_request` answers a render request (§2.2, D-55, Q-57). Items that share a timestamp and an image size render in one `render()` call, since `render()` refuses cameras of different sizes (Q-32), and a private chunk size bounds the cameras per call without changing any image. The request's background and SH degree pass through. Its time unit must be the model's (Q-17). The whole request runs under inference mode, and the outputs are moved to the CPU in request order, so a large request doesn't hold every image on the GPU.

Each call still runs the temporal slice for its time. The temporal index of D-55, and with it "activations computed once per model" (§2.4), go inside this function in M15, so its callers (the trainer's evaluation, `ftgs render`, `ftgs eval`) won't change.

## Where the backends differ

Both backends implement E6 with the same culls and constants, so their images differ only by float32 rounding: gsplat's kernels are built with fast math and order their arithmetic differently. U-7 and U-12 hold the two together within the spec's tolerances. Rounding matters more at E6's discontinuities, where a last-bit difference can put a term on opposite sides of a threshold in the two backends (Q-39):

| Discontinuity | Why the backends can disagree | What the comparisons leave out (Q-39) | Constructed test |
| --- | --- | --- | --- |
| The α skip at a pixel | gsplat's fast exponential and its own order for μ′ and the conic | pixels where some primitive's float64 α is within 10⁻³ relative of the threshold | E6's α-skip pixel |
| The exclusive stop of the transmittance | rounding of the transmittance | pixels where some float64 transmittance is within 10⁻³ relative of the stop | E6's four-layer stop |
| The extent at the opacity cull | gsplat divides the effective opacity by the threshold, the reference multiplies by its inverse, so just above the threshold one backend can get a zero extent (an invisible row) and the other a positive one | rows whose effective opacity is within 10⁻³ relative of the threshold, moved off it by the scene generator | the exact 1/255 boundary |
| A screen radius rounded up | the radius rounds up one backend's way and not the other's, which moves the off-image test | rows whose float64 extent times the standard deviation is within 10⁻⁴ of a whole pixel, or whose box edge is within 10⁻⁴ px of the image edge, from the `visible` comparison | the box touching the image edge |
| Near-equal depths | rounding of the camera-space depth can swap the compositing order | pixels where two contributing primitives' float64 depths are within 10⁻⁵ relative | the exact depth tie |
| The depth cull | rounding of the depth at the near plane | rows within 10⁻³ of the near plane, moved off it by the scene generator | the depth culls of U-7 |

The excluded share is at most 1% of a camera's pixels, and U-7 and U-12 report the largest share and the largest difference outside the mask; pixels and rows that are kept are held to U-7's and U-12's own criteria. The margins are computed by the test suite's own float64 evaluation of E1 and E3 to E6, independent of both backends. A NaN det Σ′ is culled by the reference and not by gsplat (Q-33); it arises only from activation overflow and is not compared.

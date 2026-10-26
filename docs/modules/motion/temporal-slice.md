# The temporal slice in `openftgs.motion`

How `openftgs.motion` meets its §2.1 row for E1, E3, E4 and the per-row terms of E8, and the temporal slice of §2.4 "Kernels" (D-50). The public names are a provisional decision (Q-12).

## What the slice returns

`temporal_slice` takes the stored tensors of §1.1 that change with time or need an activation, and a normalized time t, and returns a `TemporalSlice`:

- `means_t`: the moved positions (E1);
- `opacities_t`: the effective opacity σ_eff (E3);
- `quats_n` and `scales`: the activations of §1.1's per-time arrays;
- `opacities`: the base opacity σ, which the sampling score and the dead test read later (E10, E11);
- `reg_terms`: the per-row terms of the 4D regularization (E8); `losses` takes their mean (M6);
- `passes_cull`: the culling mask, true where the effective opacity reaches the opacity cull threshold of D-32, compared in the parameters' dtype. It is not visibility, which E6 defines by screen radii.

The SH concat is not part of the slice: it doesn't depend on time. M3's `at_time` builds `GaussiansAtTime` from a slice and the SH tensors.

## One formula for every path

The single-equation functions (`moved_means`, `temporal_opacity`, `effective_opacity`, `reg4d_terms`) and both implementations of the slice are built from the same private helpers: one for E1, one for E4, and one that turns a temporal opacity and the opacity logits into the base opacity, the effective opacity and the E8 terms. The eager and compiled slices run one private body made of them, so the paths cannot drift apart. The body never boolean-indexes and returns the culling mask without applying it, so no shape depends on the data and the compiled graph stays valid across growth events.

The temporal opacity is computed once per call. It is used live in the effective opacity and detached in `reg_terms`: the detach is the stop-gradient of E8, which leaves the opacity logit as the only tensor the regularization reaches.

## How E4 stays finite

E4 is evaluated by multiplying the time offset t − μt by e^−ŝ instead of dividing it by s. The duration s is never formed, so autograd never builds powers of 1/s, which underflow to zero in float32 at very short durations and turn into inf and NaN. Far from the time center the temporal opacity underflows to exactly zero, and every product in the backward pass is zero times a finite number (U-2). Values and gradients stay finite as long as the scaled offset itself is finite, that is as long as e^−ŝ is finite in the parameters' dtype. Beyond that, at extreme negative ŝ, the gradients become NaN, and so does the value of a row whose time center equals t. Nothing is clamped, so no value or gradient changes inside that domain.

E1 adds the displacement to the stored position last, so a primitive sits exactly at its stored position at its own time center (U-1).

## Choosing the implementation

`impl` takes the value of `PerfConfig.slice_impl` (Q-13):

- `"eager"` runs the body as plain PyTorch;
- `"compiled"` runs it through `torch.compile` with dynamic shapes and as one graph (`fullgraph`): one graph serves every size, apart from the one recompilation dynamo makes when N reaches 1 (source log #8), and a failed compilation raises. With `fullgraph`, a graph break fails at compile time, and reaching dynamo's recompile limit raises instead of quietly running the frame eagerly (Q-13, source log #36);
- `"auto"` compiles on CUDA tensors and stays eager on CPU, where the reference backend must stay plain PyTorch (§2.4 rule 4); if compilation fails under `"auto"` (a dynamo or backend error, or dynamo reaching its recompile limit, not any error), it warns once and stays eager for the rest of the process;
- `"cuda"` is reserved for D-50's custom operator, which isn't built (below), so it raises `NotImplementedError`, citing the decision.

t enters the compiled graph as a zero-dimensional tensor on the parameters' device, made from a Python float with a fill rather than a copy, so a new timestamp at every iteration neither recompiles the graph nor synchronizes with the host. U-28 holds the eager and compiled paths to the same values and gradients and bounds the recompilations over resizes; the comparison it uses is Q-15. U-28 counts recompilations by dynamo's count of unique graphs, since dynamo's frame counter doesn't count `fullgraph` compilations (Q-13).

The compiled implementation receives views of its tensor inputs, not the tensors themselves. A model's stored tensors are `nn.Parameter`s, whose shapes dynamo treats as static, so passing them directly would compile a new graph at every growth event, against D-50, until dynamo's recompile limit (what happens then is Q-13's). Views are plain tensors with dynamic shapes, and gradients still reach the parameters, so `FreeTimeGaussians.at_time` meets U-28's recompilation bound like plain tensors do (#45, source log #37). Only the views' rows are dynamic: their other dims, the 3 of the means and the 4 of the quaternions, are marked static (`torch._dynamo.mark_static`). Under `dynamic=True` alone dynamo made those symbolic too, and inductor then computed the quaternion's norm over the last dim in reduction kernels, which made the compiled slice slower than eager from 4·10⁵ rows on (B-3); with the row width static the norm fuses into the pointwise kernels. Only N changes at growth events, so this costs no recompilation, and the eager slice is unchanged (Q-184). While it traces them, dynamo reads `.grad` on these non-leaf views, which PyTorch warns about; the warning stays inside dynamo's tracing under the default warning filters, and only a run that turns every warning into an error would see the compilation fail, where Q-13 applies (source log #38). The eager implementation gets the tensors as they are.

## The D-50 decision

D-50 writes a custom CUDA operator only if B-1 shows the slice taking more than 10 % of a training iteration. B-1 (#226) measured the slice's forward (`slice.kernel`) plus its backward (`backward.slice`) on kitchen at 0.19 %, 0.35 % and 1.20 % of an iteration without an event at 0.5M, 1M and 3M rows with 20 % active, and 1.24 % at 3M with 5 % active, the conservative case. The largest share counts, so the operator isn't written: `"cuda"` keeps raising, U-28 keeps two implementations, and the conditional work of PRD #220 (a `torch.library` operator, a build toolchain, U-28's third implementation) stays unopened. The numbers are B-1's rows in the Calibration data of `docs/clean-room/questions.md`.

B-3, rerun with the row width static (Q-184), found the compiled slice no slower than eager at every size from 10⁵ to 3·10⁶ rows, about half eager's time from 10⁶ rows on (1.24 against 2.29 ms at 3·10⁶), with no recompilation over its 10 resizes. So Q-182's rule keeps Q-13's mapping: `"auto"` is the compiled slice on CUDA.

The SH concat, outside the slice, takes more than the slice's forward at every size (1.87 ms at 3M rows against 0.41 ms, 1.6 % of an iteration), since it copies every row's 48 SH floats; it stays under the 5 % at which PRD #220 asks for a change. B-1's figures in this section were measured with LPIPS-VGG, the perceptual network before revision 549o, which then dominated an iteration at 1352×1014: its forward (`loss.perceptual`, 52.7 ms) and every loss's backward (`backward.losses`, 44.9 ms) took 85 to 96 % of it. ꟻLIP costs about an eighth of it (D-9), so the slice's shares grow; B-1's remeasurement gives the new figures.

## Unit conversions

`durations_to_normalized`, `durations_to_raw`, `velocities_to_normalized` and `velocities_to_raw` are E14's unit conversions of primitive quantities, which Q-8 assigns to `motion` (Q-27). With a single timestamp they use a unit of 1 (D-70). Time centers convert through `TimeNormalizer`.

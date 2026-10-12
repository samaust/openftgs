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

E4 is evaluated by multiplying the time offset t − μt by e^−ŝ instead of dividing it by s. The duration s is never formed, so autograd never builds powers of 1/s, which underflow to zero in float32 at very short durations and turn into inf and NaN. Far from the time center the temporal opacity underflows to exactly zero, and every product in the backward pass is zero times a finite number (U-2). Values and gradients stay finite as long as the scaled offset itself is finite; at an extreme ŝ where it overflows, the gradients become NaN. Nothing is clamped, so no value or gradient changes inside that domain.

E1 adds the displacement to the stored position last, so a primitive sits exactly at its stored position at its own time center (U-1).

## Choosing the implementation

`impl` takes the value of `PerfConfig.slice_impl` (Q-13):

- `"eager"` runs the body as plain PyTorch;
- `"compiled"` runs it through `torch.compile` with dynamic shapes: one graph serves every size, apart from the one recompilation dynamo makes when N reaches 1 (source log #8);
- `"auto"` compiles on CUDA tensors and stays eager on CPU, where the reference backend must stay plain PyTorch (§2.4 rule 4); if compilation fails under `"auto"` (a dynamo or backend error, not any error), it warns once and stays eager for the rest of the process;
- `"cuda"` is reserved for D-50's custom operator and raises until B-1 justifies it (M15).

t enters the compiled graph as a zero-dimensional tensor on the parameters' device, made from a Python float with a fill rather than a copy, so a new timestamp at every iteration neither recompiles the graph nor synchronizes with the host. U-28 holds the eager and compiled paths to the same values and gradients and bounds the recompilations over resizes; the comparison it uses is Q-15.

## Not here yet

The unit conversions of primitive quantities between raw and normalized time, which Q-8 assigns to `motion`, arrive with their consumers in M3.

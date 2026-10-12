# The temporal slice in `openftgs.motion`

How `openftgs.motion` meets its §2.1 row for E1, E3, E4 and the per-row terms of E8, and the temporal slice of §2.4 "Kernels" (D-50). The public names are a provisional decision (Q-12).

## What the slice returns

`temporal_slice` takes the stored tensors of §1.1 that change with time or need an activation, and a normalized time t, and returns a `TemporalSlice`:

- `means_t`: the moved positions (E1);
- `opacities_t`: the effective opacity σ_eff (E3);
- `quats_n` and `scales`: the activations of §1.1's per-time arrays;
- `opacities`: the base opacity σ, which the sampling score and the dead test read later (E10, E11);
- `reg_terms`: the per-row terms of the 4D regularization (E8); `losses` takes their mean (M6);
- `visible`: the culling mask, true where the effective opacity reaches the cull threshold of D-32.

The SH concat is not part of the slice: it doesn't depend on time. M3's `at_time` builds `GaussiansAtTime` from a slice and the SH tensors.

## One formula for every path

The single-equation functions (`moved_means`, `temporal_opacity`, `effective_opacity`, `reg4d_terms`) and both implementations of the slice share one private body, so the eager path, the compiled path and the public functions cannot drift apart. The body never boolean-indexes and returns the culling mask without applying it, so no shape depends on the data and the compiled graph stays valid across growth events.

The temporal opacity is computed once per call. It is used live in the effective opacity and detached in `reg_terms`: the detach is the stop-gradient of E8, which leaves the opacity logit as the only tensor the regularization reaches.

## How E4 stays finite

E4 is evaluated as exp(−½·z²) with z = (t − μt)·exp(−ŝ). The duration s is never formed and never divided by, so autograd never builds powers of 1/s, which underflow to zero in float32 at very short durations and turn into inf and NaN. Far from the time center the temporal opacity underflows to exactly zero, and every product in the backward pass is zero times a finite number (U-2). The formulation is finite as long as exp(−ŝ) is, which in float32 means ŝ above about −88; nothing is clamped, so no value or gradient changes.

E1 is evaluated as μx + v·(t − μt) in that order, so a primitive sits exactly at μx at its own time center (U-1).

## Choosing the implementation

`impl` takes the value of `PerfConfig.slice_impl` (Q-13):

- `"eager"` runs the body as plain PyTorch;
- `"compiled"` runs it through `torch.compile` with dynamic shapes, compiled once per process;
- `"auto"` compiles on CUDA tensors and stays eager on CPU, where the reference backend must stay plain PyTorch (§2.4 rule 4); if compiling fails under `"auto"`, it warns once and falls back to eager;
- `"cuda"` is reserved for D-50's custom operator and raises until B-1 justifies it (M15).

t enters the compiled graph as a zero-dimensional tensor on the parameters' device, made with a fill rather than a copy, so a new timestamp at every iteration neither recompiles the graph nor synchronizes with the host. U-28 holds the eager and compiled paths to the same values and gradients and bounds the recompilations over resizes; the comparison it uses is Q-15.

## Not here yet

The unit conversions of primitive quantities between raw and normalized time, which Q-8 assigns to `motion`, arrive with their consumers in M3.

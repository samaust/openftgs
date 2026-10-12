# Per-time arrays and the exchange model in `openftgs.model`

How `FreeTimeGaussians` meets its §2.1 row for the per-time arrays (§1.1) and the exchange model of §2.2 `ftgs-if`. The public names beyond §2.1 are a provisional decision (Q-20).

## The stored tensors

The nine stored tensors of §1.1 are registered as parameters under their own names, copied from the caller's tensors so that init points and exchange models stay plain data, so `state_dict()` and `params.safetensors` use the normative names. The validity rules of §1.1 run in the constructor, and every other way of building a model goes through it: `from_init_points`, `from_exchange`, `load` and `resize_`. The SH degree is checked there too (D-3, U-26); at L = 0 the higher bands are an empty tensor of the right shape.

`resize_` takes a bool mask for the kept rows, so it can't reorder or repeat them, and appends new rows last (Q-23). It revalidates the whole model, so a non-finite value anywhere surfaces at the next resize. It builds new parameters, so an optimizer must rebind them (M7).

## What `at_time` returns

`at_time` takes a normalized time, a float as in §2.1, and runs the temporal slice of `openftgs.motion` (Q-12, Q-13). It returns `GaussiansAtTime`: the slice's fields plus the SH coefficients, the DC band and the higher bands concatenated and cut to the requested degree. The SH coefficients don't depend on time, so the concat isn't part of the slice; the culling mask, the base opacity and the E8 terms are carried along so a training step computes the slice once (D-49).

E5 lives here as `rotation_matrix` and `covariance`, below the render backends and density, which both need it (U-5).

## Physical values and the exchange model

`physical()` and `to_exchange()` give the same arrays under the `ftgs-if` keys: the physical values in raw time units. The unit conversions of time centers, durations and velocities are E14's, from `TimeNormalizer` and `openftgs.motion` (Q-8, Q-27). With a single timestamp the span is zero: durations and velocities convert with a unit of 1 instead (Q-19), and an imported model's velocities are set to zero, as §1.4 "Time" freezes them.

- The opacity is computed in float64 and clamped away from 0 and 1 before it is cast, so saturated logits still validate (§2.2).
- The rotation is the unit quaternion with a canonical sign: w ≥ 0, and where w = 0 the first nonzero component positive (Q-24).

`from_exchange` validates the exchange model (`io.validate_exchange`, Q-22), inverts the conversions in float64 (time centers through `TimeNormalizer`, whose out-of-span warning doesn't apply to them, Q-9), normalizes the rotation (Q-24), clamps the opacity before taking the logit, and splits the SH coefficients into the DC and higher bands. `load` takes either a `ftgs-if` file or a checkpoint directory; a checkpoint above the cap N_max of its stored configuration is rejected by `io.load_checkpoint` (Q-25).

The time unit travels with the model (Q-17): set at construction, kept by `from_exchange` and `load`, written by `to_exchange`.

# Time in `openftgs.types`

How `openftgs.types` meets the time half of E14 (D-1, D-46) and §1.4 "Time".

## E14 is split across bands

- `types` owns the **time normalizer** (`TimeNormalizer`, with `to_t` and `to_tau`) and the **frame interval Δt_f** (`frame_interval`, Q-8). Both sit in the Foundation band, so `data`, `init`, `io`, `model`, `optim` and `train` can all reach them.
- `motion` owns the unit conversions of primitive quantities between raw and normalized time (time center, duration, velocity), which its §2.1 row calls E14.

## Who computes Δt_f and who receives it

`MultiViewVideo` (M10) builds the time normalizer from the smallest and largest training timestamps and computes Δt_f once, from the raw training timestamps of each training camera, keyed by camera name. Everything downstream receives the value as an argument (`FreeTimeGaussians.from_init_points`, `build_optimizer`, the initialization pipeline) and the checkpoint stores it as `TrainerState.frame_dt`. Nobody recomputes it.

`frame_interval` takes the medians over raw gaps and divides by the training time span once (Q-8); the result is in normalized time, as E14 defines it.

## The warning outside the training span

Only `to_t` warns, with a `UserWarning`, for raw timestamps outside the closed interval [τ₀, τ₁] (Q-9). Rendering calls it, so the render-time warning of §1.4 comes from one place. `to_tau` never warns, because time centers legitimately drift outside [0, 1]. On tensors the range check syncs with the host; the training step passes Python floats and doesn't pay for it.

## Comparing records

`Camera`, `ImageRecord` and `InitPoints` are frozen dataclasses holding tensors, as §2.1 declares them. Their generated `==` compares tensors element-wise and fails on multi-element tensors, so code groups records by camera name, never by comparing `Camera` objects.

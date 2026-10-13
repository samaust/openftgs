# The trainer in `openftgs.train`

How `Trainer` meets its §2.1 row: the training iteration of §1.3, the calendar, the step log and callbacks, evaluation, and the trainer state that a checkpoint holds. The public names beyond §2.1 are provisional decisions (Q-69 to Q-74).

## One training iteration

`step()` runs §1.3's seven steps in order at iteration it = `iteration` + 1:

1. The sampler gives the next training image (D-26); its file is decoded and copied to the device.
2. Its normalized time t and the active SH degree come from the time normalizer (E14) and the calendar entry (E13).
3. `render` builds the per-time arrays and renders, with the 2D means' gradient retained (Q-32). `PerfConfig.precull` and `packed` pass through (Q-31, Q-36).
4. `total_loss` evaluates E9 with the perceptual network built once, only when λ_perc > 0 (D-77).
5. The backward pass, then the gradient statistics: by `gaussian_ids` in packed mode, by `visible` otherwise, never by the flag (Q-36). They are skipped when the 2D means got no gradient, which is a check for None, not a synchronization.
6. The optimizer takes this iteration's rates (E12, E13) and one Adam step, then the gradients are cleared.
7. On an event, in order: the divergence check, growth, relocation, pruning, the opacity reset, and the reset of the gradient statistics where relocation didn't run, since relocation always resets them itself (D-45, Q-51).

The backend follows the device: gsplat on a CUDA device, which D-76 requires of it, and the reference backend elsewhere (Q-69).

## The calendar

`calendar_entry(cfg, it)` is a pure function of the configuration and the iteration, so U-17 checks a 30,000-iteration calendar without training (Q-71). Every window and period comes from the configuration, so a shortened calendar (§3.2) needs no other change. Growth and relocation share the event period (`RelocConfig.every`) and have their own windows. In the "w/o periodic relocation" ablation, selected by `RelocConfig.prune_opacity`, pruning runs at every growth event and the opacity reset at every multiple of `density.OPACITY_RESET_EVERY` inside the growth window (owner answer, Q-71; Q-67).

## The step log and the log window

A step returns a `StepLog`. Its loss terms are the detached tensors `total_loss` returns, on the device and unread, so a step makes no host synchronization outside gsplat's render, the events and the log reads (§2.4 rule 3, D-49). The trainer adds the terms into device-side sums and reads their means once every `PerfConfig.log_every` iterations; the same read checks the model's validity. After a resume the first window covers only the iterations since the resume; logs never feed back into training.

The per-step copy of the image to the device synchronizes until the frame cache and prefetcher of D-51 (M12).

## Divergence

A non-finite mean loss at a log read, or a model that fails `validate()` at a log read or before an event, makes `step` raise `FloatingPointError` naming the iteration (Q-69). It is a built-in exception, apart from the `ValueError` of input errors (Q-26), so M14 can map it to exit code 3 (Q-55).

## Callbacks and stopping

`train(until, callbacks)` calls every callback after each step, in order, and stops after that step when any returns True. Stopping happens only between iterations, so a state taken then is always whole; M14's SIGINT and SIGTERM handlers use this to save (§2.2).

## Generators

All randomness in training comes from the trainer's own CPU generators, each seeded from `TrainConfig.seed` and its role (Q-72):

- the sampler's, for the per-epoch permutations;
- the relocation generator, for every draw of an event in a fixed order: growth's split offsets (E17), relocation's targets and positions (D-15), and the score mode's draws;
- an initialization generator, used once for the seeded subset of `from_init_points` and not checkpointed.

The global generators are neither seeded nor drawn from. The trainer state records and restores them, so code outside the trainer, such as a callback, also continues on a resume. Because they are never seeded, two runs in one process see different global states, and U-24 would catch any hidden draw from them.

## Exact resume

A run resumed from a checkpoint at iteration k is bitwise equal to the uninterrupted run on the reference backend (§2.2):

- every draw comes from a checkpointed generator, and the sampler stores its epoch, its cursor and the current permutation, so the next permutation comes from the same point in the generator's stream;
- rates, the active SH degree, the calendar and the regularizer's cutoff are pure functions of it and the configuration;
- the stored tensors round-trip exactly through safetensors, and τ₀, τ₁, S and Δt_f through JSON; the optimizer is rebuilt from the stored S and Δt_f, so its base rates are the same floats (Q-46);
- states are taken and training stops only between iterations, and gradients are cleared every step;
- on the CPU the reference backend, the SSIM fallback, Adam and the frozen perceptual network are deterministic for a fixed thread count.

On gsplat the backward pass accumulates with atomic additions, so a resume holds up to floating-point reordering only, as §2.2 says.

`load_state_dict` builds the new model, optimizer and statistics before replacing anything, so a rejected state leaves the trainer as it was. It requires the trainer's own configuration and the dataset's time unit, and adopts the state's time normalizer, S and Δt_f. `from_checkpoint` also refuses a checkpoint made from another scene manifest unless forced (owner answer, Q-70); a forced resume keeps the run's time normalizer, S and Δt_f, so its learning rates continue unchanged.

## The model

The trainer trains a model passed to it in place, after moving it to the device; built from init points, the model is the trainer's own. After `load_state_dict` or `from_checkpoint` the trainer works on a model it built from the state. The model's `slice_impl` is set from `PerfConfig` (Q-66).

## Settings from `PerfConfig`

Global PyTorch flags are set around each step and restored afterwards, so a training step leaves the caller's process as it was (Q-73):

- `deterministic` turns off cuDNN autotuning, asks for cuDNN's deterministic algorithms and PyTorch's deterministic kernels (with warnings only where none exists). fused-ssim stays in use: its backward pass writes each gradient element once, with no atomics (source log #73, answering Q-42's note). Without it the cuDNN flags are left alone; autotuning waits for its benchmark (B-4, M15).
- `perc_precision` sets cuDNN's TF32 flag for the step (D-52).

Options that later milestones build raise `NotImplementedError` at construction rather than being ignored: bf16 LPIPS and several images per step (M15), and the GPU or memory-mapped frame cache (M12).

## Evaluation

`evaluate` renders the images of a data split through `eval.evaluate_images`, which goes through `render.render_request` one timestamp at a time, the path `ftgs render` and `ftgs eval` share (Q-57, Q-68). It builds the `Evaluator` once, on first use, and uses no generator, so an evaluation between steps never changes a run. With `out_dir` it writes METRICS.json in the report layout of Q-63, its header naming the trainer's iteration as the model (Q-74).

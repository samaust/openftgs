# Learning-rate schedules and optimizer state in `openftgs.optim`

How `openftgs.optim` meets its §2.1 row for E12 and E13: the parameter groups, the learning-rate schedules, the three Adam kinds of D-53, the moment reset and resize that growth and relocation call, and the optimizer's share of the checkpoint (§2.2). The two keyword parameters §2.1 lacks are the owner's answers recorded in `docs/clean-room/questions.md` (Q-44); the behaviour at the edges and the state layout are rows the owner confirmed (Q-45, Q-46), and how U-31 measures its tolerances is the owner's answer in Q-47.

## Parameter groups

`build_optimizer` makes one parameter group per stored tensor (§1.3 "Optimizer and learning rates"), each named after its tensor, inside one torch optimizer. The optimizer keeps the model it was built for and shows it as the read-only property `model`, which `openftgs.density` checks before it resets moments or resizes (Q-48). Two cases leave a stored tensor without a group, so it takes no updates:

- `shN` at SH degree 0, where the tensor has no columns (U-26);
- `times` and `velocities` with a single timestamp, which the model's time normalizer shows as τ₀ = τ₁ (D-71). Of D-71's two options, "no parameter group" was chosen, as for `shN` (Q-45).

`zero_grad` still clears the gradients of all nine stored tensors, so a tensor without a group never accumulates gradients over a run (Q-45).

## Learning-rate schedules

The base rates are the §1.5 values from `OptimConfig`, scaled as §1.3's rate table says (D-19, D-27) and held as Python floats. `set_iteration` turns the iteration into training progress as D-41 defines it and evaluates each schedule in the form its equation is written: E13's exponential form for `means` and `times` (D-20), E12's product form for `velocities`, constants for the rest. It writes plain floats into the groups, so neither it nor `step` synchronizes with the host (§2.4 rule 3, D-49). It accepts only the iterations D-41 counts, and with a one-iteration run training progress is 0 (Q-45). `build_optimizer` applies the first iteration's rates. The active SH degree, also in E13, is the trainer's (M10).

## The three Adam kinds

D-53 names the kinds, which `PerfConfig.adam` selects and the trainer passes to `build_optimizer` (Q-44):

- **fused**, the default: PyTorch's fused Adam. PyTorch checks whether it supports a device and dtype only at the step that creates the Adam state, which the optimizer creates itself, so the build probes each device and dtype with a one-element step; where the probe fails, the optimizer warns and uses foreach (Q-45).
- **foreach**: PyTorch's multi-tensor Adam.
- **selective**: gsplat's Selective Adam, an approximate option. It needs CUDA, and `step` takes the visible rows as a bool mask (Q-44). gsplat's kernel never advances the step counter, so the optimizer advances it itself and the counter always counts the steps taken; a resume under another kind then applies the right bias correction (Q-45). The kernel takes its hyperparameters as float32, which is why U-31's float64 reference is fed the float32 values (Q-47).

## State, moment reset and resize

The Adam moments and step counters are created when the optimizer is built, zero, where PyTorch would put them (the step counter on the parameter's device for fused Adam, on the CPU otherwise). PyTorch then uses them as its own, and a moment reset, a resize or a checkpoint works before the first step (Q-45).

- **Moment reset.** `reset_rows` zeroes the Adam moments of the rows it is given, as indices that may repeat (relocation draws targets with replacement), and keeps each step counter (D-16). By default it zeroes every parameter group; `names` limits it to the named groups, which the ablation's opacity reset uses for the opacity logits alone (Q-53). An unknown name, or a name without a parameter group, is rejected.

  `reset_rows` doesn't check that its indices lie within the rows (Q-44), so an index out of range from a bug in a caller would surface on CUDA as a device-side assert, not a `ValueError`. On the CPU such an index raises an `IndexError`, so M8's U-14 tests, which check that relocation zeroes the moments of exactly its targets and moved copies, also cover the range of the rows it passes.
- **Resize.** The model replaces its parameters when it resizes, so the optimizer follows it: the caller resizes the model first, then calls the optimizer's `resize_` with the same keep mask (Q-23) and the number of appended rows. The kept rows' moments stay in order, new rows start with zero moments (D-45), each group is rebound to the model's new parameter, and the step counters are kept. A row count that doesn't match the model is rejected before anything changes (Q-44).

Validation reads only dtypes, shapes and devices, never values, so `set_iteration`, `step`, `zero_grad` and a moment reset never synchronize (§2.4 rule 3). A resize with a keep mask does: gathering the kept rows needs their count on the host. It runs only at growth events, which synchronize anyway (§2.4 rule 3). A moment reset reads no values either, so keeping its indices within the rows is the caller's precondition (Q-44).

## The checkpoint's optimizer files

`state_dict` returns one entry per parameter group, holding the Adam moments, the step counter, the base rates after scaling, the schedule's kind, the iteration count, β₁, β₂ and ε (Q-46). The checkpoint writer puts the tensors in `optim.safetensors` and the rest in `optim.json`, as Q-25 splits them. `load_state_dict` takes the CPU tensors the checkpoint reader returns, moves them to the parameters' device and places the step counters as the current Adam kind needs, so state saved under one kind loads under another. It rejects state whose groups, shapes, base rates or settings differ from the optimizer's own, so a resume can't silently run on another schedule (Q-46).

Note for M10: the check compares the scaled base rates exactly, so `Trainer.from_checkpoint` must build the resumed optimizer from the checkpoint's own `TrainerState.spatial_scale` and `frame_dt`, never from S and Δt_f recomputed from the scene. JSON keeps both exactly, so a resume passes the check; with `--force` and a changed scene manifest, recomputed values would differ and the resume would be refused (Q-46). M10's PRD should include a test that a forced resume after a scene change keeps the run's learning rates.

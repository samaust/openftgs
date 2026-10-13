# Growth and the count-control modes in `openftgs.density`

How `openftgs.density` meets its §2.1 row for E17 and §1.3's growth and count-control modes, including the pruning and opacity reset of the "w/o periodic relocation" ablation. The orders, boundaries and score-mode details §1.3 leaves open are provisional decisions recorded in `docs/clean-room/questions.md` (Q-52, Q-53); the placement of the ablation's functions, the moments at the opacity reset and the measure of U-25's offsets are the owner's answers (Q-48, Q-53, Q-54).

## Growth in the "gradient" mode

`grow` reads the accumulated gradient g from the gradient statistics and grows every candidate (§1.3 Growth). A clone and a split each add one row, so the cap N_max is applied to the number of candidates. When they exceed the room left, a stable descending sort of g over the candidates in row order keeps the largest, with ties to the lower row (D-44).

The clone/split boundary compares the float32 largest scale with the boundary scaled by S, so a scale exactly at the boundary clones (Q-52). Each split draws its children's offsets from its parent's Gaussian (E17), from one normal draw with the first children's offsets first, and gives the children the parent's log-scales minus the log of E17's factor, rounded once to float32.

New rows go last, so kept rows keep their order (§1.3 Growth step 4): clones first, then the first children, then the second children, each in ascending parent row (Q-52). The model, the optimizer and the statistics are resized in that order with one `keep` mask, which drops the split parents:

- `FreeTimeGaussians.resize_` appends the new rows and revalidates the model;
- `FTGSOptimizer.resize_` gives the new rows zero Adam moments (D-45) and rebinds to the model's new parameters (Q-44);
- `GradStats.resize_` copies each new row's statistics from its parent (D-45).

Growth only adds: N never decreases (§1.1, D-42). First measurement (CPU, the seed the test uses): U-25's 10⁶ split offsets are off the parent's Gaussian by at most 0.21% in the whitened mean and covariance, inside Q-54's bound.

## The "score" and "none" modes

In the "score" mode, `grow` appends rows drawn in proportion to the sampling score and initializes them with E11, as relocation's moved copies are (§1.3 Count-control modes). It shares relocation's move: the targets are drawn from the live set only, since a dead target would be rewritten at ô_dead by E11's logit floor and then skipped as live by the relocation of the same event (Q-52). Each target gets E11 with its number of draws plus one and zeroed Adam moments; the new rows carry its updated tensors, one per draw in draw order, placed per `copy_position`, inheriting its statistics. Whenever rows are to be added, the model is validated first and the live scores checked before the draw, as in relocation (Q-51, Q-55); with no live row, nothing is added (Q-52).

In the "none" mode, `grow` touches nothing and draws nothing.

## The ablation's pruning and opacity reset

The "w/o periodic relocation" ablation replaces relocation with pruning and an opacity reset (§1.3 Count-control modes). Both are public functions the trainer calls on its calendar; `grow` only grows (Q-48).

- `prune` removes the primitives below the pruning opacity, tested on the stored logit as the dead set is (D-14), and, past the iteration §1.3 names, those whose largest scale exceeds the ablation's bound scaled by S, with the thresholds and the strict iteration test of gsplat 1.5.3's default strategy (source log #56). The model, the optimizer and the statistics are resized together. With no pruning opacity configured, it changes nothing.
- `reset_opacity` clamps every opacity logit at the logit of the reset value and zeroes the Adam moments of the opacity logits for every row, keeping the step counters, through `reset_rows(..., names=("opacity_logits",))` (owner answer, Q-53), as gsplat's opacity reset does (source log #57).

The ablation's constants have no config field; they are fixed private constants and change only by a logged code change (D-72).

## Notes for M10

- When the opacity reset fires, over the whole run or only within the growth window as 3DGS and gsplat do, is the trainer's calendar decision. Its period and `prune`'s iteration 3,000 come from the same gsplat default, so the trainer takes the period from one place and the two can't drift apart (Q-53, owner answer).
- The relocation generator may live on the CPU or on the model's device; draws run on the generator's device. A generator on the model's device keeps the draws there.
- With relocation off, the trainer resets the gradient statistics itself after each event (Q-51).
- A `ValueError` from `FreeTimeGaussians.validate()` at an event means the run diverged; its mapping to the CLI's exit code 3 belongs to M10 and M14.
- U-25's "stops growing after 15,000" and "growth runs before relocation in an event" are calendar properties, pinned by the trainer's tests.

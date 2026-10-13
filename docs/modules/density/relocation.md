# Gradient statistics and relocation in `openftgs.density`

How `openftgs.density` meets its §2.1 row for E10 and E11: the gradient statistics that §1.3 step 5 accumulates, the sampling score, the relocation move and the relocation procedure. The public names beyond §2.1 and the semantics §2.1 leaves open are provisional decisions recorded in `docs/clean-room/questions.md` (Q-48 to Q-51, Q-55); who resets the statistics and the measure of U-14's position draws are the owner's answers (Q-51, Q-54).

## Gradient statistics

`GradStats` holds the two accumulators of the gradient statistics, `g_sum` and `count`, one entry per primitive, on the model's device (Q-48). The trainer builds one with the model's N, passes it every rendered 2D-mean gradient, and saves it as `TrainerState.stats`.

`update` takes the retained gradient of `RenderOutput.means2d` and the layout's selector (Q-49):

- in unpacked mode, the gradient for every camera and row with the `visible` mask: a masked sum over cameras, as §2.4's stage table says;
- in packed mode, the gradient of the visible camera–primitive pairs with `gaussian_ids`: a scatter-add through the ids. Every packed row is a visible pair (D-75, D-76).

Consumers tell the layouts apart by whether `gaussian_ids` is given (Q-36), and exactly one selector is accepted. The NDC scaling of D-12 multiplies the two gradient components by Python floats, so no tensor is built on the host, and each pair adds its norm to `g_sum` and one to `count`. A primitive seen by several cameras in one call counts once per camera in both layouts (Q-49); training renders one image per iteration (D-26), so it is counted at most once there.

`update`, `mean` and `reset` read only shapes, dtypes and devices and make no host synchronization (§2.4 rule 3, D-49). U-16 checks this under torch's synchronization debug mode, which torch documents as a prototype that misses some synchronizing operations, so the check is evidence rather than proof.

`resize_` follows the model at growth and pruning: the same `keep` mask (Q-23), and parent indices into the rows before the resize, so a split's children can name the parent that the same resize removes (Q-49). New rows copy their parent's statistics, which lets the relocation later in the same event score the grown region (D-45).

## The sampling score

`sampling_score` returns ρ for every row; the procedures mask out the dead set (Q-48). Its quantile runs over the live, at-least-once-visible rows as E10 says: the other rows are set to NaN and `torch.nanquantile` ignores them, which gives the quantile of the selected rows with torch's default interpolation (D-13) and no data-dependent shape. An empty selection gives a NaN quantile, which the "quantile is 0" branch absorbs, so ĝ is identically 0 there too.

The live set is tested on the stored logit against ô_dead (D-14). Every logit bound in the module, ô_dead and the ablation's thresholds alike, is formed by one helper as float32 of the float64 logit, and comparing a float32 tensor with that Python float compares in float32.

## The relocation move

`mcmc_split` is E11 for a batch of targets. It computes in float64 and returns the dtype it was given (Q-50):

- σ_new is evaluated in a form that equals E11's in exact arithmetic and stays accurate for small σ;
- the double sum comes from a float64 table of E11's coefficients, accumulated over the outer index and gathered at each target's n, built once per device;
- n is clamped as E11 says for the formula only; every drawn copy is still written;
- the scale factor uses the unclamped σ_new.

U-13 compares `mcmc_split` with gsplat 1.5.3's `compute_relocation`, a float32 kernel, over live targets and every n E11 allows, as U-13's row asks. Away from the clamp the two agree far inside U-13's bound; where σ_new falls below the dead threshold, so that only a_new is kept, the difference grows to just under the bound, from float32 cancellation in the kernel's evaluation of σ_new, not from the float64 path (source log #60, #62, #63). That is where a later move of the gsplat pin would show first.

Note for M16: `mcmc_split` applies only the upper bound of E11's clamp, so the harness adapter's answer to the L2 `relocation_math` query must apply D-14's lower bound to σ_new itself (Q-50).

The clamp has two halves. `mcmc_split` applies the upper bound. The lower bound is D-14's and is applied by the callers to the stored float32 logit, as a floor at ô_dead from the configuration: only a bound on the stored logit guarantees that every relocated row is live, since sigmoid(ô_dead) rounds below the threshold in float32 (Q-50).

## The relocation procedure

`relocate` follows §1.3's steps in this order (Q-51):

1. It checks the validity rules of §1.1 with `FreeTimeGaussians.validate()` (Q-55), so a diverged model raises a `ValueError` before any draw instead of failing inside `torch.multinomial`.
2. It forms the dead set. If the dead set or the live set is empty, it only resets the statistics.
3. It scores the live set; the live scores must be finite with a positive sum (with λ_o = 0, which Q-11 allows, they can all be 0), else `ValueError`.
4. It draws one target per dead primitive with replacement, on the generator's own device, and moves the draws to the model's device. The dead rows, in ascending order, take the draws in draw order.
5. It applies E11 once per target, from the float64 sigmoid of the stored logit and the exponential of the stored log-scales.
6. It gathers each target's position, orientation and scales before anything is written, then writes in place under `no_grad`: the targets' new logit and log-scales, then all nine tensors to the moved copies, then the moved copies' positions, drawn from the target's Gaussian with its scales from before the move (E11 writes diag(a_j), as E17 uses the parent's scales). The draws come from one normal draw after the multinomial. With `copy_position = "stack"` the moved copies keep the target's position.
7. It resets the Adam moments of the targets and the moved copies, or of the targets only when `reset_moved_moments` is off (D-16), through `FTGSOptimizer.reset_rows`.
8. It checks the validity rules again, as §1.1 asks after every relocation.
9. It resets the statistics, always, even when nothing moved (owner answer, Q-51). With relocation off, the trainer resets them itself (M10).

First measurements (CPU, the seeds the tests use): U-14's 10⁶ position draws are off the target's Gaussian by at most 0.09% in the whitened mean and 0.25% in the whitened covariance, well inside Q-54's bound; U-13's render before and after relocating stacked copies differs by 40.4 dB.

Writing in place keeps the parameters' identity, so the optimizer keeps its references and step counters; relocation never changes N. The optimizer must be the model's own, built for it by `build_optimizer`: `FTGSOptimizer` has no public row count to check against, so the pairing is the caller's precondition (Q-48), and the trainer (M10) holds the three together. The random draws come in a fixed order (the multinomial, then the normal draw), so a resumed run can repeat them (U-21, M10). `relocate` doesn't read `RelocConfig.enabled`: that is the trainer's switch for calling it.

## Host synchronization

| Operation | Runs | Synchronizes |
| --- | --- | --- |
| `GradStats.update`, `mean`, `reset` | every iteration | no |
| `sampling_score` | events | no |
| `mcmc_split` | events | no, after its first call on a device, which copies the coefficient table there once |
| `reset_opacity` | ablation events | no |
| `relocate`, `grow`, `prune` | events | yes, by design: the number of dead rows, candidates and targets sizes the draws and the gathers, `validate()` reads values, and the reports hold Python ints. Events synchronize anyway (§2.4 rule 3) |

Draws on a CPU generator are copied to the model's device at the event. `bincount` is not used, because on CUDA it reads its maximum back; counts go through `index_add_`, which torch runs deterministically under its deterministic mode (source log #58).

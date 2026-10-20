# The adapter, `ftgs-adapter`

openftgs's **adapter** for the differential harness (§3.3 adapter contract), a second console script beside `ftgs`, so that §2.1's command list stays unchanged (Q-202). Its code is private to `openftgs.cli`.

```
ftgs-adapter [--backend reference|gsplat] info
ftgs-adapter [--backend …] export MODEL OUT
ftgs-adapter [--backend …] import IF OUT
ftgs-adapter [--backend …] render MODEL REQUEST OUT
ftgs-adapter [--backend …] forward QUERIES OUT
ftgs-adapter [--backend …] train SCENE INIT CONFIG SEED OUT_DIR
```

- `--backend` is read before torch is imported. `reference` hides CUDA from the process, so its `train` runs on the CPU too (Q-69); `gsplat` needs CUDA.
- `info` prints name, version, the 13 query kinds, the rasterizer identity and `io.render_conventions()` (Q-205, D-73).
- `import` writes the adapter's **native directory**: `params.safetensors` with the nine §1.1 tensors and `model.json` with `sh_degree`, `tau0`, `tau1`, `time_unit` and `background`.
- `export` and `render` take as MODEL a native directory, a checkpoint, a run directory (its `checkpoint/`) or an FTGS-IF file.
- `train` runs `ftgs train SCENE --init INIT --config CONFIG --seed SEED --out OUT_DIR` in process. Config keys TrainConfig lacks, and values it refuses on their own, are written to `OUT_DIR/not_comparable.json` and left out of the run.
- `forward` answers each L2 kind with the package's own functions, always on the CPU (`docs/modules/harness/formats.md`).

## What answers each query kind

`forward` reads the query's FTGS-IF arrays into a `FreeTimeGaussians` on the CPU (`from_exchange`) and converts raw τ with the model's `TimeNormalizer` (Q-207). Each kind calls the package's own function (Q-204):

| Kind | Call |
| --- | --- |
| motion | `motion.moved_means` (E1) |
| temporal_opacity | `motion.temporal_opacity` (E4) |
| effective_opacity | `motion.effective_opacity` (E3) |
| sh_color | `model.at_time(t, sh_degree=…)`, then `sh.sh_to_color` from the camera centre to the moved mean (E2) |
| losses | `l1_loss`, `ssim`, `reg4d_loss`, `LPIPSLoss("vgg")` when `lpips` is true, and `total_loss` at `iteration` (E7 to E9) |
| lr_schedule | `build_optimizer(…, adam="foreach")` on a one-row model, then `set_iteration` per iteration, columns in §1.3's order (E12, E13) |
| relocation_math | `density.mcmc_split`, then D-14's lower bound on σ_new (Q-50) |
| velocity_init | `init.knn_velocities` (E15) |
| relocation_sampling | `density.draw_targets` per seed, counted per row (D-15) |
| grad_stats | the trainer's steps 3 to 5 for one image on the reference backend: `render(…, return_aux=True, packed=False)`, `total_loss`, backward, `GradStats.update` (D-12) |
| relocation_score | `density.sampling_score` over a model whose logits are the opacities' and a `GradStats` holding the sums and counts (E10, D-13) |
| spatial_scale | `data.spatial_scale` over records built from the cameras, no image read (D-27) |
| growth_decisions | `density.growth_masks` (D-18, D-44) |

A query that raises is an `error` answer; the others in the file are still answered. A kind the module doesn't know is `not_comparable`.

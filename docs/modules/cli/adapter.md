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
- `train` runs `ftgs train SCENE --init INIT --config CONFIG --seed SEED --out OUT_DIR` in process. Config keys that `config_from_dict` rejects are written to `OUT_DIR/not_comparable.json` and left out of the run.
- `forward` answers each L2 kind with the package's own functions, always on the CPU (`docs/modules/harness/formats.md`).

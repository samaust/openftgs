# The command line in `openftgs.cli`

How `ftgs` meets §2.1's CLI block: what each command calls, the exit codes, the run directory, signals and resume. The module's one public name is `main(argv=None) -> int`, which the `ftgs` console script and `python -m openftgs.cli` both run (Q-124). The choices the spec leaves open are Q-124 to Q-144.

## Commands

Each command parses its options with argparse and calls the library. Nothing is computed in the CLI that a Python caller couldn't do with the same functions:

| Command | Library calls |
| --- | --- |
| `convert` | `data.convert_n3dv`, `convert_opencv_yaml` or `convert_selfcap`, one option per keyword; `--test-camera` and `--window N0:N1` are Q-85's |
| `cache` | `MultiViewVideo.from_manifest`, `FrameCache.build`, and for `gpu` or `mmap` `FrameCache.open` in that placement (Q-127) |
| `init roma` | `RomaInitConfig`, `roma_init`, `io.write_init_points` with `roma_provenance(cfg)` as metadata (Q-107) |
| `train` | `config_from_dict`, `read_init_points`, `Trainer` or `Trainer.from_checkpoint`, `Trainer.train`, `io.save_checkpoint`, `Trainer.evaluate` |
| `render` | the model as `FreeTimeGaussians.load` reads it, `io.read_render_request`, `render.render_request`, `io.write_renders` with the SHA-256 of the request's bytes |
| `eval` | the model, `evaluate_images`, `metrics_report`, `eval.write_metrics` (Q-136) |
| `export` | `io.load_checkpoint`, the model, `to_exchange` with the configuration's background, `io.write_exchange` (Q-141) |
| `bench` | a stub until M15: it checks its arguments and returns 1 (Q-130) |

The device is CUDA when PyTorch sees one, else the CPU, and the backend follows it, as in the trainer. `ftgs render` keeps `render()`'s default backend, gsplat, so on a machine without CUDA it needs `--backend reference`, which renders on the CPU (Q-125).

Single-file outputs replace an existing file; directory outputs, a converter's OUT and a fresh run directory, must be absent or empty (Q-126).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | an unexpected exception, with its traceback; `ftgs bench` until M15 |
| 2 | invalid input or environment: a usage error, a file that fails validation, a missing ffmpeg, ffprobe or COLMAP, a configuration this build can't run, a missing `[init]` extra, LPIPS weights that can't be downloaded, a frame cache that doesn't fit in GPU memory |
| 3 | a numerical failure: training diverged, or a render isn't finite |
| 130, 143 | `ftgs train` stopped by SIGINT or SIGTERM, after saving its checkpoint |

Exit code 2 comes from the `ValueError`s of the calls that read or check user input, and from the CLI's own checks; a `ValueError` anywhere else is a bug and keeps its traceback (Q-26, Q-142). Divergence is the trainer's `FloatingPointError` (Q-69).

## The run directory

`ftgs train --out RUN_DIR` writes (Q-132):

- `config.json`, the resolved configuration, after `--seed`;
- `train_log.jsonl`, one row per log read: the iteration, the mean of each loss term since the last read, and N, the columns §3.3 asks of an adapter;
- `checkpoint/`, saved when training reaches T_it and when a signal stops it, and nowhere else;
- `METRICS.json`, the evaluation of the test split at the end, when the scene has one; it runs after the checkpoint is saved, so a failed evaluation loses no training;
- `profile/trace.json`, with `--profile START:COUNT`: a `torch.profiler` trace of those iterations, which must lie inside the iterations the command runs; a window cut short still writes what it recorded (Q-144). NVTX ranges per stage come with M15.

With `perf.deterministic`, `ftgs train` sets `CUBLAS_WORKSPACE_CONFIG` before CUDA starts, unless the environment already sets it (Q-131).

## Signals and resume

While training runs, SIGINT and SIGTERM are recorded rather than acted on at once. The trainer stops after the current iteration, so the saved state is whole (Q-69). The checkpoint is saved atomically (Q-25), and the command exits with 128 plus the signal number (Q-133). A divergence saves nothing, so the last good checkpoint survives: the trainer checks at its log reads and events (Q-69), and the command checks the model once more before it saves, so a model that turned non-finite since is exit 3 too.

`--resume CKPT` continues the run on the checkpoint's own configuration. A `--config` or `--seed` given beside it must give exactly that configuration, performance settings included. A manifest that changed since the checkpoint was made needs `--force` (Q-70, Q-134). The run directory must be the run's own (its `config.json` equal to the checkpoint's configuration), or absent or empty. A signal that comes once the run has reached T_it doesn't stop it: the run is complete and goes on to its evaluation. D-24's check of the start budget F·B against N_max happens when the trainer builds the model, so `ftgs init` can't make it (Q-128).

## Evaluation

`ftgs eval MODEL SCENE.json` renders the split (`--split test`, the default, or `train`; Q-143) with the model and scores it. A checkpoint is rendered against its configuration's background, an exchange model against black, since readers ignore `render_conventions` (§2.2, Q-141). With `--renders-dir DIR` it instead scores the renders in DIR (`request.json` and `renders.safetensors`), matching each image of the split to one request item by timestamp, image size and camera, the camera within a relative 1e-6 (Q-137). `ftgs render --backend reference` followed by `ftgs eval --renders-dir` scores the reference backend. A split with masks on some images but not all is refused (Q-135).

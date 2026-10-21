# Validation runs on openftgs-synth

M17's training and evaluation runs on openftgs-synth's `kitchen`, `courtyard` and `studio` (§1.4 "Validation data", D-79; PRD #317). Each run is listed here with its command, commit, the configuration's differences from the default, its wall time, its files and its headline metrics. The measured values themselves are rows of the Calibration data table of `docs/clean-room/questions.md`, which carry the owner's decisions; this page points to them. No result is compared with the paper's tables (D-79).

The runs are under `<m17-runs>` (`docs/modules/cli/benchmarks.md`), outside the repository; `kitchen` is M15's conversion at `<m15-runs>/kitchen`, read and never written.

## The scenes

| Scene | Converted from | Test view | Images | Notes |
| --- | --- | --- | --- | --- |
| `kitchen` | `n3dv/kitchen`, resize 1.0 (M15's conversion) | cam00 (the converter's fixed test view) | 20 cameras × 300 frames, 1352×1014, 30 fps | LPIPS label `alex` (D-29) |
| `courtyard` | `enerf_outdoor/courtyard` with `ftgs convert opencv-yaml --fps 60 --frames 300 --resize 1.0 --test-camera 0009` | 0009 (Q-218) | 18 × 300, 1920×1080, 60 fps | LPIPS label `vgg` |
| `studio` | `selfcap` with `ftgs convert selfcap --scene studio --window N0:N1 --test-camera 0007 --undistort internal --resize 1.0 --masks DIR` | 0007 (the dataset card's) | 24 × 60, 1920×1080, 60 fps, unsynchronized | 60-frame window around the test view's fastest range, exact ground-truth masks (Q-219); LPIPS label `vgg` |

## How a run is made and summarized

- Every timed command runs under `/usr/bin/time -v` (wall time, peak resident memory), with the GPU's memory sampled once a second by `nvidia-smi` into the run's folder, alone on the machine.
- A configuration change is a TrainConfig JSON file holding only that override (Q-10), kept in `<m17-runs>/configs/`.
- `scripts/run_summary.py RUN_DIR` writes `SUMMARY.json` beside the run: the N trace from `train_log.jsonl` (start, iteration 15,000, end, the first iteration at `growth.n_max`), the mean reg and total losses per 1,000 iterations, the opacity statistics of every checkpoint the run kept, and the scene means of METRICS.json with its `dynamic` and `fastest_10` blocks.
- `scripts/synth_window.py` and `scripts/synth_masks.py` build studio's window and its mask directory from the ground truth (Q-219).

## Runs

The runs are added here as they finish.

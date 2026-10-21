# Validation runs on openftgs-synth

M17's training and evaluation runs on openftgs-synth's `kitchen`, `courtyard` and `studio` (§1.4 "Validation data", D-79; PRD #317). Each run is listed here with its command, commit, the configuration's differences from the default, its wall time, its files and its headline metrics. The measured values themselves are rows of the Calibration data table of `docs/clean-room/questions.md`, which carry the owner's decisions; this page points to them. No result is compared with the paper's tables (D-79).

The runs are under `<m17-runs>` (`docs/modules/cli/benchmarks.md`), outside the repository; `kitchen` is M15's conversion at `<m15-runs>/kitchen`, read and never written. Nothing a run produces is deleted: every final checkpoint, the reg_until runs' periodic checkpoints, the converted scenes and their frame caches stay, for the test-view renders and diagnostics of #318 (owner, 2026-10-21).

## The scenes

| Scene | Converted from | Test view | Images | Notes |
| --- | --- | --- | --- | --- |
| `kitchen` | `n3dv/kitchen`, resize 1.0 (M15's conversion) | cam00 (the converter's fixed test view) | 20 cameras × 300 frames, 1352×1014, 30 fps | LPIPS label `alex` (D-29) |
| `courtyard` | `enerf_outdoor/courtyard` with `ftgs convert opencv-yaml --fps 60 --frames 300 --resize 1.0 --test-camera 0009` | 0009 (Q-218) | 18 × 300, 1920×1080, 60 fps | LPIPS label `vgg` |
| `studio` | `selfcap` with `ftgs convert selfcap --scene studio --window N0:N1 --test-camera 0007 --undistort internal --resize 1.0 --masks DIR` | 0007 (the dataset card's) | 24 × 60, 1920×1080, 60 fps, unsynchronized | 60-frame window around the test view's fastest range, exact ground-truth masks (Q-219); LPIPS label `vgg` |

The reference images themselves have flaws, seen by eye (owner, 2026-10-21): some scenes have over-bright regions clipped to white, and soft cloth textures that look blurry and change from frame to frame in a way the motion doesn't explain. Where the reference changes in a way the scene's geometry and motion can't account for, no model can match it, so part of every metric's error, and its spread from frame to frame and region to region, belongs to the data rather than to the run.

## How a run is made and summarized

- Every timed command runs under `/usr/bin/time -v` (wall time, peak resident memory), with the GPU's memory sampled once a second by `nvidia-smi` into the run's folder, alone on the machine.
- A configuration change is a TrainConfig JSON file holding only that override (Q-10), kept in `<m17-runs>/configs/`.
- `scripts/run_summary.py RUN_DIR` writes `SUMMARY.json` beside the run: the N trace from `train_log.jsonl` (start, iteration 15,000, end, the first iteration at `growth.n_max`), the mean reg and total losses per 1,000 iterations, the opacity statistics of every checkpoint the run kept, and the scene means of METRICS.json with its `dynamic` and `fastest_10` blocks.
- `scripts/synth_window.py` and `scripts/synth_masks.py` build studio's window and its mask directory from the ground truth (Q-219).

## Runs

`$K` is `<m15-runs>/kitchen`, `$M` is `<m17-runs>`. Each run's logs are `<m17-runs>/logs/<step>.*`: the command, the commit, `uptime` at the start, `/usr/bin/time -v`'s report and the GPU samples.

| Step | Run | Command | Commit | Wall time | Result | Calibration row |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | G-1 to G-7, C-4's dots | `OPENFTGS_DATASETS=… pytest -s tests/test_synthetic_conversion.py tests/test_converter_checks.py -k "g1 or … or (c4 and dots)"` | a16a3fe | 2 min 57 s | 23 passed | §3.1 G-1 to G-7, M17's precondition |
| 1 | kitchen init | `ftgs init roma $K/scene.json $M/kitchen/init.safetensors` | 30e4885 | 30 min 26 s | 300,000 points | §1.5, §3.4 B-1 |
| 2 | kitchen default (B-1's run) | `ftgs train $K/scene.json --init $M/kitchen/init.safetensors --out $M/kitchen/default` | 30e4885 | 58 min 18 s | PSNR 27.33 dB on cam00; N reaches 3,000,000 at iteration 3,600 | §1.5, §3.4 B-1; §1.3 Growth on kitchen; E16 on kitchen |
| 6 | B-1 on the trained model | `ftgs bench train-step $K/scene.json --model $M/kitchen/default/checkpoint --out $M/bench/B-1-kitchen-trained.json` | 30e4885 | 1 min 35 s | 118.3 ms an iteration, bound 3,557 s | §3.4 B-1 on the trained model |
| 10 | B-7 on kitchen | `ftgs bench render $K/scene.json --model $M/kitchen/default/checkpoint --out $M/bench/B-7-kitchen.json` | 30e4885 | 5 min 34 s | 187.7 / 180.7 FPS (index on / off) | §3.4 B-7 on kitchen |
| 7 | B-6 on kitchen | `ftgs bench data $K/scene.json --out $M/bench/B-6-kitchen.json` | 30e4885 | 2 min 20 s | at most 0.044 % | §3.4 B-6 on kitchen |
| 9 | B-8 conversion | `ftgs bench convert $K/scene.json --out $M/bench/B-8-convert-kitchen.json` | 30e4885 | 1 min 24 s | 40.8 / 41.0 s | §3.4 B-8, conversion |
| 8 | B-8 initialization | `ftgs bench init $K/scene.json --out $M/bench/B-8-init-kitchen.json` | 30e4885 | 30 min 14 s | 1,811.8 s, 3,900 pairs | §3.4 B-8, initialization |

Kitchen's default run reached N_max inside the growth window, so the τ_pos = 10⁻⁴ run of Q-220's first decision is replaced by one at τ_pos = 4·10⁻⁴ (Q-220 as answered).

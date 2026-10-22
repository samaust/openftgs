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

`$D` is `$OPENFTGS_DATASETS/synth`, `$K` is `<m15-runs>/kitchen`, `$M` is `<m17-runs>`. Each run's logs are `<m17-runs>/logs/<step>.*`: the command, the commit, `uptime` at the start, `/usr/bin/time -v`'s report and the GPU samples.

| Step | Run | Command | Commit | Wall time | Result | Calibration row |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | G-1 to G-7, C-4's dots | `OPENFTGS_DATASETS=… pytest -s tests/test_synthetic_conversion.py tests/test_converter_checks.py -k "g1 or … or (c4 and dots)"` | a16a3fe | 2 min 57 s | 23 passed | §3.1 G-1 to G-7, M17's precondition |
| 1 | kitchen init | `ftgs init roma $K/scene.json $M/kitchen/init.safetensors` | 30e4885 | 30 min 26 s | 300,000 points | §1.5, §3.4 B-1 |
| 2 | kitchen default (B-1's run) | `ftgs train $K/scene.json --init $M/kitchen/init.safetensors --out $M/kitchen/default` | 30e4885 | 58 min 18 s | PSNR 27.33 dB on cam00; N reaches 3,000,000 at iteration 3,600 | §1.5, §3.4 B-1; §1.3 Growth on kitchen; E16 on kitchen |
| 3 | kitchen τ_pos 4·10⁻⁴ | the step 2 command with `--out $M/kitchen/tau4e-4 --config $M/configs/tau4e-4.json` (`{"growth": {"grad_threshold": 0.0004}}`) | 30b82be | 57 min 20 s | PSNR 28.07 dB, LPIPS-Alex 0.2128; N reaches 3,000,000 at iteration 9,400 | D-18 τ_pos on kitchen |
| 4 | kitchen reg_until 15,000 | the step 2 command with `--out $M/kitchen/reg15k --config $M/configs/reg15k.json --checkpoint-every 5000 --keep-checkpoints 7` (`{"loss": {"reg_until": 15000}}`) | 1e726d0 | 58 min 28 s | PSNR 29.32 dB, LPIPS-Alex 0.1898; mean σ rises from 0.227 at 15,000 to 0.273 | D-10 on kitchen |
| 5 | kitchen seed 1 | the step 2 command with `--out $M/kitchen/seed1 --seed 1` | 30b82be | 58 min 33 s | PSNR 27.28 dB, LPIPS-Alex 0.1894 | Run-to-run spread on kitchen |
| 6 | B-1 on the trained model | `ftgs bench train-step $K/scene.json --model $M/kitchen/default/checkpoint --out $M/bench/B-1-kitchen-trained.json` | 30e4885 | 1 min 35 s | 118.3 ms an iteration, bound 3,557 s | §3.4 B-1 on the trained model |
| 7 | B-6 on kitchen | `ftgs bench data $K/scene.json --out $M/bench/B-6-kitchen.json` | 1e726d0 | 2 min 20 s | at most 0.044 % | §3.4 B-6 on kitchen |
| 8 | B-8 initialization | `ftgs bench init $K/scene.json --out $M/bench/B-8-init-kitchen.json` | 1e726d0 | 30 min 14 s | 1,811.8 s, 3,900 pairs | §3.4 B-8, initialization |
| 9 | B-8 conversion | `ftgs bench convert $K/scene.json --out $M/bench/B-8-convert-kitchen.json` | 1e726d0 | 1 min 24 s | 40.8 / 41.0 s | §3.4 B-8, conversion |
| 10 | B-7 on kitchen | `ftgs bench render $K/scene.json --model $M/kitchen/default/checkpoint --out $M/bench/B-7-kitchen.json` | 1e726d0 | 5 min 34 s | 187.7 / 180.7 FPS (index on / off) | §3.4 B-7 on kitchen |
| 11 | courtyard convert and cache | `ftgs convert opencv-yaml $D/enerf_outdoor/courtyard $M/courtyard/scene --fps 60 --frames 300 --resize 1.0 --test-camera 0009`; `ftgs cache $M/courtyard/scene/scene.json` | 3a3f075, a0b584e | 20 min 40 s; 34 s | 31.73 GB cache | B-8 beside it, courtyard |
| 12 | courtyard init and default | `ftgs init roma $M/courtyard/scene/scene.json $M/courtyard/init.safetensors`; `ftgs train $M/courtyard/scene/scene.json --init $M/courtyard/init.safetensors --out $M/courtyard/default` | a0b584e | 27 min 19 s; 83 min 9 s | 300,000 points; PSNR 30.27 dB on 0009, LPIPS-VGG 0.2321; N reaches 3,000,000 at iteration 7,700 | Growth and E16 on courtyard; B-8 beside it |
| 13 | B-7 on courtyard | `ftgs bench render $M/courtyard/scene/scene.json --model $M/courtyard/default/checkpoint --out $M/bench/B-7-courtyard.json` | a0b584e | 5 min 38 s | 220.5 / 192.5 FPS (index on / off) | §3.4 B-7 on courtyard |
| 14 | studio window and masks | `python scripts/synth_window.py $D/ground_truth/studio --camera 0007 --length 60`; `python scripts/synth_masks.py $D/ground_truth/studio --camera 0007 --window 149:209 $M/studio/masks` | a0b584e | under 1 min | fastest 10 at frames 174 to 183; window 149:209; 60 exact masks | Q-219 as run |
| 15 | studio convert, cache, B-6 | `ftgs convert selfcap $D/selfcap $M/studio/scene --scene studio --window 149:209 --test-camera 0007 --undistort internal --resize 1.0 --masks $M/studio/masks`; `ftgs cache $M/studio/scene/scene.json`; `ftgs bench data $M/studio/scene/scene.json --out $M/bench/B-6-studio.json` | a0b584e, 1b7a197 | 5 min 38 s; 10 s; 5 min 41 s | 1,440 images, 8.58 GB cache; B-6 at most 0.045 %, `margin_used` 1.017 at 3M on the GPU | Q-219 as run; §3.4 B-6 on studio |
| 16 | studio init and default | `ftgs init roma $M/studio/scene/scene.json $M/studio/init.safetensors`; `ftgs train $M/studio/scene/scene.json --init $M/studio/init.safetensors --out $M/studio/default` | 1b7a197 | 7 min 31 s; 83 min 9 s | 63,000 points; PSNR 23.24 dB on 0007 (dynamic 25.78), LPIPS-VGG 0.3097 (dynamic 0.1645); N reaches 3,000,000 at 13,500; GPU peak 24,040 MiB | Growth and E16 on studio |
| 17 | studio reg_until 15,000 | the step 16 training with `--out $M/studio/reg15k --config $M/configs/reg15k.json --checkpoint-every 5000 --keep-checkpoints 7` | 1b7a197 | 82 min 54 s | PSNR 22.43 dB (dynamic 25.39), LPIPS-VGG 0.3368 | D-10 on studio |
| 18 | B-7 on studio | `ftgs bench render $M/studio/scene/scene.json --model $M/studio/default/checkpoint --out $M/bench/B-7-studio.json` | 1b7a197 | 2 min 5 s | 150.9 / 149.0 FPS at 0007's 60 times (Q-221) | §3.4 B-7 on studio; Q-195's T on the trained models |
| 19 | B-4 at #326 | `ftgs bench losses $K/scene.json --out $M/bench/B-4-losses-326.json` | f6047c1 | 4 min 6 s | NCHW inputs with autotuning 6.10 / 4.54 % faster than the old default | §2.4 rule 2, B-4's second run (#326) |
| 20 | kitchen default at #326 | the step 2 command with `--out $M/kitchen/default-326` | f6047c1 | 56 min 20 s | PSNR 29.75 dB on cam00, LPIPS-Alex 0.1850; N reaches 3,000,000 at iteration 3,600; GPU peak 23,492 MiB | §1.5, §3.4 B-1 at #326 |
| 21 | B-6 on studio at #325 and #326 | `ftgs bench data $M/studio/scene/scene.json --out $M/bench/B-6-studio-325-autotune.json` | 836b887 | 4 min 26 s | "auto" chooses the GPU at 1M, `margin_used` 0.68; skips it at 3M; at most 0.043 % | §3.4 B-6 on studio after #325, at #326's code |
| 22 | B-6 on kitchen60 at #325 and #326 | `ftgs bench data <m15-runs>/kitchen60/scene.json --out $M/bench/B-6-kitchen60-325-autotune.json` | 836b887 | 3 min 53 s | `margin_used` 0.45 at 1M, 0.43 at 3M; at most 0.044 % | §3.4 B-6 on kitchen60 after #325, at #326's code |
| 23 | studio default at #325 and #326 | the step 16 training with `--out $M/studio/default-325` | 836b887 | 80 min 18 s | no GPU out of memory, "auto" memory-mapped the cache; PSNR 22.86 dB on 0007 (dynamic 25.79), LPIPS-VGG 0.3223 (dynamic 0.1633); N reaches 3,000,000 at 13,400; GPU peak 19,546 MiB | D-51 and §1.5 on studio at #325 and #326 (#325's added criterion) |

Kitchen's default run reached N_max inside the growth window, so the τ_pos = 10⁻⁴ run of Q-220's first decision is replaced by one at τ_pos = 4·10⁻⁴ (Q-220 as answered).

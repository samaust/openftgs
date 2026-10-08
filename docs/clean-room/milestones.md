# Milestones

Nineteen milestones, M0 to M18, build `openftgs` bottom-up along the dependency bands of spec §2.1 (`docs/spec/openftgs-spec.md`, revision 549). Each one lands with the spec tests that pin it; a test listed under a milestone must pass before the next milestone starts. The order is the spec author's suggestion, not part of the spec.

How to run one:

1. Start a fresh session in the repository: `claude` (model and effort come from `.claude/settings.local.json`). Switch to plan mode with Shift+Tab and paste the milestone's prompt.
2. Read the plan. Approve it, or correct it and ask for a revised plan.
3. Leave plan mode and send the execution message below.
4. When the milestone is done, run the review prompt in another fresh session.

Execution message, the same for every milestone:

```text
Approved. Implement the plan. One commit per module or test group, each message citing the spec items it implements. Run every test you listed and paste the pytest summary. Append every source you consulted to docs/clean-room/source-log.md and any gap you had to decide to docs/clean-room/questions.md. Finish with: what passed, what is skipped and why, and the open Q-n rows.
```

Review prompt, run in a fresh session at `/effort xhigh` after each milestone:

```text
Review milestone <Mn> of docs/clean-room/milestones.md against docs/spec/openftgs-spec.md. Read CLAUDE.md first. For every function and test the milestone lists, check the code against the spec items it cites: equations, constants, provenance tags, decisions and pass criteria. Report each finding with a severity (critical, major, minor, nit), the spec location, the code location and the exact discrepancy. Also check docs/clean-room/source-log.md: every third-party file or page the code could only have come from must have a row, and no row may name a forbidden source. Change nothing; report only.
```

Effort: the sessions marked xhigh below carry the numerically delicate work; the others run at the default `high`.

---

## M0 · Repository skeleton

Scope: package layout, build metadata, test markers, CI, license files. No spec logic yet.
Spec: §2.1 (module layout), §2.3 (dependencies, versions, licenses, CI license scan), §3 introduction (CPU tests on the reference backend, GPU-marked tests), D-62 (names).
Done when: `pip install -e .[dev]` works, `pytest` collects, ruff, mypy and `pip-licenses` run in CI and the license scan fails on GPL or AGPL.

```text
Milestone M0 of docs/clean-room/milestones.md: repository skeleton. Read CLAUDE.md, then spec §2.1 (module layout table and the first paragraph on dependency bands), §2.3 (the dependency table and its notes) and the introduction of §3. Plan the skeleton: pyproject.toml for the package openftgs with the extras [init] and [fast] and the dev dependencies exactly as §2.3 lists them, gsplat pinned to 1.5.3; the src/openftgs module files from §2.1, each with a docstring and an SPDX header and no logic; the separate ftgs_harness package directory; pytest markers for CPU, GPU and dataset-gated tests; a CI workflow that runs ruff, mypy, the CPU tests and pip-licenses with a failure on GPL or AGPL; LICENSE and NOTICE; a README stub that states the clean-room process and points to docs/spec and docs/clean-room. List every file you will create and what it holds. Note anything §2.3 leaves open as a Q-n candidate.
```

## M1 · Types and configuration

Scope: `openftgs.types`, `openftgs.config`.
Spec: §2.1 (types and config blocks), §1.5 (hyperparameters), E14 (time normalization, D-1, D-46), D-3 (SH degree), §2.2 (the checkpoint manifest stores the full TrainConfig).
Tests: U-18; the validation parts of U-26; a TrainConfig JSON round trip.

```text
Milestone M1 of docs/clean-room/milestones.md: types and configuration. Read CLAUDE.md and docs/clean-room/questions.md, then spec §2.1 (the openftgs.types and openftgs.config blocks), §1.5, E14 with D-1 and D-46, D-3, and the checkpoint manifest in §2.2. Plan openftgs/types.py and openftgs/config.py: every dataclass with the exact fields, defaults and provenance comments of §2.1; TimeNormalizer with to_t and to_tau and the warning outside [τ₀, τ₁]; validation of sh_degree and of the other ranges the spec states; JSON serialization of TrainConfig for the checkpoint manifest. Plan tests/test_types.py and tests/test_config.py covering U-18 (round trip, irregular timestamps, single frame, unsynchronized cameras giving the same Δt_f, the warning) and the sh_degree validation of U-26. Say where Δt_f is computed and which module owns it.
```

## M2 · Motion, temporal opacity and spherical harmonics

Scope: `openftgs.motion` (E1, E3, E4, the per-row terms of E8, the `temporal_slice` function in its eager and compiled forms), `openftgs.sh` (real SH basis, RGB ↔ DC).
Spec: E1, E3, E4, E2 (basis only), E8, §2.4 "Kernels" and D-50 (the CUDA variant waits for M15), D-2.
Tests: U-1, U-2, U-3, U-28 for eager and compiled, and the hypothesis variants of U-1 and U-2.

```text
Milestone M2 of docs/clean-room/milestones.md: motion, temporal opacity and spherical harmonics. Read CLAUDE.md and docs/clean-room/questions.md, then spec §1.1, E1, E2, E3, E4, E8, D-2, and §2.4 "Kernels" with D-50. Plan openftgs/motion.py: pure functions for E1, E4, E3 and the per-row terms of E8 that take the stored tensors and a normalized time, and temporal_slice with the eager implementation and the torch.compile implementation with dynamic shapes, selected by PerfConfig.slice_impl, with the CUDA variant left as a documented stub for M15. Plan openftgs/sh.py: the real SH basis up to degree 3 written from the published constants, and the RGB ↔ DC conversion. Plan the tests: U-1, U-2, U-3 with their exact pass criteria, the hypothesis variants of U-1 and U-2, and U-28 for the eager and compiled implementations including the resize-between-calls and recompilation-count checks. State how you keep E4 finite at ŝ = −30 and |t − μt|/s = 10⁴.
```

## M3 · Model and file I/O

Scope: `openftgs.model` (`FreeTimeGaussians`), `openftgs.io` (checkpoint directory, FTGS-IF, PLY).
Spec: §1.1, D-2, D-5, E5; §1.3 Initialization steps 7–9 and the paragraph on points supplied through the public interface (D-24, D-25); §2.1 (model and io blocks); §2.2 (`ftgs-init`, `ftgs-checkpoint`, `ftgs-if`), D-30.
Tests: U-5; the steps-7–9 part of U-19; U-22 except the render comparison, which M4 adds; the shape and round-trip parts of U-26; validation of `ftgs-init` files.

```text
Milestone M3 of docs/clean-room/milestones.md: model and file I/O. Read CLAUDE.md and docs/clean-room/questions.md, then spec §1.1, D-2, D-5, E5, §1.3 "Initialization" steps 7 to 9 and the paragraph after them, D-24, D-25, the openftgs.model and openftgs.io blocks of §2.1, and in §2.2 the ftgs-init, ftgs-checkpoint and ftgs-if formats with D-30. Plan openftgs/model.py: the nine stored tensors and their activations, at_time, physical, resize_ with the row-order rule, from_init_points implementing steps 7 to 9 (the F·B budget and its error, per-init-frame 3-NN scales with the 1e-7 floor, opacity 0.1, identity quaternions, sh0 from color, shN zero, the defaults for missing velocities, durations and frame_ids), from_exchange and to_exchange with the float64 opacity path, quaternion sign and validator. Plan openftgs/io.py as plain data in and out: save_checkpoint and load_checkpoint over TrainerState with the exact files of §2.2, write_exchange and read_exchange, and PLY export. Plan the tests: U-5, the steps-7–9 part of U-19, U-22 without the render comparison, the shape and round-trip parts of U-26, and rejection tests for invalid ftgs-init and ftgs-if files.
```

## M4 · Reference renderer (xhigh)

Scope: `openftgs.render` with the pure-PyTorch `reference` backend on CPU.
Spec: E2 and D-4, E5, E6 with every normative constant (D-6, D-31, D-32, D-11), §1.4 "World and camera frames" and "Images" (D-28), D-34, the `render` signature and `RenderOutput` in §2.1.
Tests: U-6; U-4, U-8 and U-20 on the reference backend; the render part of U-22.

```text
Milestone M4 of docs/clean-room/milestones.md: the reference renderer. Read CLAUDE.md and docs/clean-room/questions.md, then spec E2 with D-4, E5, E6 in full including the list of normative details, D-6, D-11, D-31, D-32, D-34, §1.4 "World and camera frames" and "Images" with D-28, and the render signature and RenderOutput in §2.1. Plan openftgs/render.py with backend="reference": projection with the +0.5 pixel convention, the Jacobian clamp, the 3.33σ opacity-aware extent, α clamped at 0.999, the 1/255 skip, the exclusive stop when transmittance ≤ 1e-4, the det ≤ 0 and off-screen culls, D-32 pre-culling, SH color with the direction from the moved mean, background compositing, time_units raw and normalized, and the means2d and visible outputs. Keep it plain PyTorch and differentiable. Plan the tests: U-6, U-4 (reference tolerance), U-8 bitwise on the reference backend, U-20 for the projection cases, and the render comparison of U-22. List each constant with the spec line it comes from.
```

## M5 · gsplat renderer (xhigh)

Scope: the `gsplat` backend of `openftgs.render`.
Spec: E6, D-6, D-31, D-57 (packed mode, flags), D-12 (the 2D-mean gradient the statistics need), D-3 (active SH degree), the `RenderOutput` fields `means2d`, `visible`, `gaussian_ids`. gsplat 1.5.3 is an allowed source: read its `rasterization()` and the data-conventions document, and log what you read.
Tests: U-7, U-12; U-4, U-8 and U-20 on the GPU.

```text
Milestone M5 of docs/clean-room/milestones.md: the gsplat renderer. Read CLAUDE.md and docs/clean-room/questions.md, then spec E6 with its normative details, D-6, D-12, D-31, D-57, and the render signature and RenderOutput in §2.1. Then read gsplat 1.5.3 from the installed package: the rasterization() signature and docstring and the data-conventions document; log them in docs/clean-room/source-log.md. Plan backend="gsplat": how the model's per-time tensors map to rasterization() arguments, the view and projection matrices in gsplat's conventions, classic mode with ε = 0.3, near 0.01, far 1e10, packed mode with gaussian_ids, retaining the 2D-mean gradient for the statistics of E10, the visible mask from the radii, the active SH degree, and the background. Plan the tests: U-7 with all five of its cases, U-12, and the GPU variants of U-4, U-8 and U-20. State every place where the two backends could legitimately differ by rounding and the tolerance you will use.
```

## M6 · Losses

Scope: `openftgs.losses`.
Spec: E7 (D-7, D-8, D-9), E8, E9 (D-10, D-37, D-38), §2.4 rows for the perceptual and SSIM losses, D-52, the losses block of §2.1, the `[fast]` extra (fused-ssim) in §2.3.
Tests: U-9, U-10, U-11.

```text
Milestone M6 of docs/clean-room/milestones.md: losses. Read CLAUDE.md and docs/clean-room/questions.md, then spec E7 with D-7, D-8 and D-9, E8, E9 with D-10, D-37 and D-38, the losses block of §2.1, the perceptual-loss and SSIM rows of §2.4 "Where the time goes", and D-52. Plan openftgs/losses.py: l1_loss; ssim with the 11×11 Gaussian window, σ 1.5, C₁ = 0.01², C₂ = 0.03², valid and same padding, fused-ssim when the [fast] extra is installed and the PyTorch fallback otherwise, both giving the same value; LPIPSLoss over lpips 0.1.4 with the [0, 1] → [−1, 1] mapping and frozen weights; reg4d_loss as E8 with the stop-gradient exactly where E8 puts it; total_loss as E9 with reg_until. Plan the tests: U-9 including the scikit-image comparison (scikit-image is a docs-only dependency: use its documented parameters, never its source), U-10 with every gradient it names, and U-11's float64 gradcheck through the reference renderer.
```

## M7 · Optimizer and schedules

Scope: `openftgs.optim`.
Spec: §1.3 "Optimizer and learning rates", E12 (D-19), E13 (D-20, D-27, D-41), the optim block of §2.1, D-53, the `optim.safetensors` and `optim.json` files of §2.2.
Tests: the learning-rate part of U-17; U-31.

```text
Milestone M7 of docs/clean-room/milestones.md: optimizer and schedules. Read CLAUDE.md and docs/clean-room/questions.md, then spec §1.3 "Optimizer and learning rates", E12 with D-19, E13 with D-20, D-27 and D-41, the openftgs.optim block of §2.1, D-53, and the optim.safetensors and optim.json entries of §2.2. Plan openftgs/optim.py: build_optimizer with one parameter group per tensor and the rates of §1.5 scaled by S and Δt_f where the spec says so; FTGSOptimizer.set_iteration applying E12 and E13 and returning the rates; step with PyTorch's fused Adam, foreach as fallback and gsplat's SelectiveAdam as the opt-in; reset_rows; resize_ with zero moments for new rows; state_dict and load_state_dict in the §2.2 layout. Plan the tests: the rate checks of U-17 (it = 1, T_it, the midpoint, ηv(1), rates never increasing) and U-31 including the float64 comparison of selective Adam with an all-true mask against Adam without bias correction.
```

## M8 · Gradient statistics, growth and relocation (xhigh)

Scope: `openftgs.density`.
Spec: E10 (D-12, D-13), E11 (D-14, D-15, D-16), E17, §1.3 "Growth", "Relocation" and "Count-control modes" (D-17, D-18, D-42 to D-45), the density block of §2.1, D-49 (no host synchronization).
Tests: U-13 and its hypothesis variant, U-14, U-15, U-16, U-25.

```text
Milestone M8 of docs/clean-room/milestones.md: gradient statistics, growth and relocation. Read CLAUDE.md and docs/clean-room/questions.md, then spec E10 with D-12 and D-13, E11 with D-14, D-15 and D-16, E17, §1.3 "Growth", "Relocation" and "Count-control modes", D-17, D-18, D-42 to D-45, the openftgs.density block of §2.1 and D-49. Plan openftgs/density.py: GradStats with index_add_ accumulation in unpacked and packed modes and no host synchronization, mean, reset and resize_ with parents' statistics copied to new rows; sampling_score with Q₀.₉₉ taken over live primitives visible at least once, torch.quantile's default interpolation, and ĝ ≡ 0 when the quantile is 0; mcmc_split computing the E11 scale factor in float64 from the unclamped opacity, then the clamp with the lower bound in logit space; relocate with the dead test on the float32 logit, multinomial targets with replacement, the copy rules, positions of moved copies drawn from the target's Gaussian (copy_position="sample") or stacked, Adam moment resets of targets and moved rows; grow with clone and split at the 0.01·S boundary, the N_max cap keeping the largest-g candidates with ties to the lower row index (D-44), children inheriting statistics; the "score" and "none" modes and the ablation pruning. Plan the tests: U-13 with the clamp-active cases and the hypothesis variant, U-14 with the χ² test and the position-draw checks, U-15, U-16 including packed mode with a primitive seen by several cameras and the synchronization check, and U-25. Say which operations run on the GPU without any .item() or .cpu().
```

## M9 · Evaluation metrics and render requests

Scope: `openftgs.eval`; the `ftgs-render-request` and `ftgs-renders` formats in `openftgs.io`.
Spec: E16 (D-29, D-40, D-67, D-68, D-69), the dynamic-region bullets of §1.4 "Dataset protocols", the eval row of §2.1, §2.2 "Rendering at any camera and time", D-55 (batching cameras per time; the index itself comes in M15).
Tests: U-23.

```text
Milestone M9 of docs/clean-room/milestones.md: evaluation metrics and render requests. Read CLAUDE.md and docs/clean-room/questions.md, then spec E16 with D-29, D-40, D-67, D-68 and D-69, the dynamic-region and ablation bullets at the end of §1.4 "Dataset protocols", the openftgs.eval row of §2.1, and §2.2 "Rendering at any camera and time". Plan openftgs/eval.py: PSNR, SSIM_k and DSSIM_k with C₁ = (0.01k)², clamping to [0, 1], per-image averaging, LPIPS-Alex and LPIPS-VGG with inputs in [−1, 1] plus the lpips_vgg_raw variant of D-68, the Appendix B.1 crop and mask applied to both images with masks binarized at 0.5, the per-dataset backbone choice, and the fastest-10-frames selection of D-69. Plan the render-request reader and the ftgs-renders writer in openftgs/io.py and the function that renders a request, batching cameras per time, with inference mode. Plan U-23 with every case the spec lists and the METRICS.json layout that ftgs eval will write.
```

## M10 · Dataset core and trainer (xhigh)

Scope: `MultiViewVideo` and manifest reading in `openftgs.data`; `openftgs.train`.
Spec: §2.2 scene manifest; E14 and D-46 (Δt_f), D-27 (S); §1.3 "One training iteration" steps 1 to 7, "Iteration calendar", D-17, D-26, D-45, D-3; the train block of §2.1; §2.2 checkpoint contract; D-49.
Tests: the calendar part of U-17; U-21; U-24; the smoke runs of U-26; S-10.

```text
Milestone M10 of docs/clean-room/milestones.md: dataset core and trainer. Read CLAUDE.md and docs/clean-room/questions.md, then spec §2.2 "Scene manifest", E14 with D-46, D-27, §1.3 "One training iteration" and "Iteration calendar", D-3, D-17, D-26, D-45, the openftgs.train block of §2.1, the checkpoint contract paragraph of §2.2, and D-49. Plan the manifest reader and MultiViewVideo in openftgs/data.py (records, time normalizer, S, Δt_f, load_image, test masks), leaving converters and the frame cache to M12. Plan openftgs/train.py: Trainer.__init__ from InitPoints or a model, the seeded per-epoch permutation, the seven steps of an iteration in the spec's order, the active SH band, the growth-then-relocation-then-reset event at every 100th iteration within the windows, StepLog, callbacks, evaluate through openftgs.eval, state_dict and load_state_dict over TrainerState with every generator and sampler state, and from_checkpoint. Plan the tests: the calendar checks of U-17, U-21 with its test calendar, U-24, the smoke runs of U-26 for each SH degree, and S-10. Say how you make a resumed run bitwise equal to a straight run.
```

## M11 · Synthetic-scene recovery

Scope: the §3.2 suite, except S-4, S-9 and S-11, which need the initialization and ray caster of M13 and M14.
Spec: §3.2 in full, including the shared setup and the shortened schedules.
Tests: S-1, S-2, S-3, S-5, S-6, S-7, S-8.

```text
Milestone M11 of docs/clean-room/milestones.md: synthetic-scene recovery. Read CLAUDE.md and docs/clean-room/questions.md, then spec §3.2 in full. Plan tests/synthetic/: a generator of ground-truth FreeTimeGS models and camera rigs for the shared setup, rendering with the reference backend on CPU and gsplat on GPU, the proportional shortening of every calendar count, and the scenes S-1, S-2, S-3, S-5, S-6, S-7 and S-8 with their recovery checks and pass criteria, comparing trajectories rather than raw (μx, μt) pairs in S-2. Mark which thresholds the spec calls calibrated and how you will record the first measured values in docs/clean-room/questions.md without loosening a stated criterion. Keep the CPU scenes under a few minutes each.
```

## M12 · Dataset converters and frame cache

Scope: the converters, image processing and cache in `openftgs.data`.
Spec: §1.4 "Dataset protocols" with D-28, D-29, D-48, D-59, D-60, D-63, D-64; D-33 (area resizing) and D-61 (undistortion, YAML reader); D-51 (frame cache, prefetch); §2.4 "Threads and processes" and "Initialization and conversion" (ffmpeg jobs, PNG pool); the data block of §2.1; the mask fields of the manifest in §2.2. ffmpeg and COLMAP are documentation-only tools; OpenCV is docs-only and appears only as a test oracle.
Tests: the converter part of U-20; U-27, U-30, U-34, U-35, U-36, U-37; C-1 to C-5 written as dataset-gated tests.

```text
Milestone M12 of docs/clean-room/milestones.md: dataset converters and frame cache. Read CLAUDE.md and docs/clean-room/questions.md, then spec §1.4 "Dataset protocols" in full with D-28, D-29, D-48, D-59, D-60, D-63 and D-64, D-33, D-61, D-51, §2.4 "Threads and processes" and "Initialization and conversion", the openftgs.data block of §2.1, and the manifest's mask fields in §2.2. Plan convert_n3dv (the poses_bounds.npy layout of D-28, ffmpeg decoding as a subprocess with --jobs and --hwaccel, area resizing), convert_opencv_yaml (the YAML reader of D-61, undistortion maps for 4-, 5-, 8- and 12-coefficient models, frame k at k/fps), convert_selfcap (scene table, windows, test camera, sync.json timestamps, COLMAP or internal undistortion, archive hashes, masks), the frame cache and prefetcher of D-51, and the manifest writer. ffmpeg, COLMAP and OpenCV are documentation-only: cite the documentation pages you will read and log them. Plan the tests: U-27, U-30, U-34, U-35, U-36, U-37, the converter part of U-20, and C-1 to C-5 skipped unless the dataset path is set. State which parts need a GPU.
```

## M13 · Initialization (xhigh)

Scope: `openftgs.init` and the vendored RoMa in `openftgs._vendor.romatch`.
Spec: §1.3 "Initialization" steps 1 to 6 with D-21, D-22, D-23, D-47, D-66; E15; D-24; D-56, D-58, D-65; §2.4 "Initialization and conversion"; the init block of §2.1; the `[init]` extra in §2.3. RoMa at commit 77f8d68 is an allowed source (MIT); XFeat's license is checked before Tiny RoMa is offered.
Tests: the rest of U-19; U-33; U-38; S-4.

```text
Milestone M13 of docs/clean-room/milestones.md: initialization. Read CLAUDE.md and docs/clean-room/questions.md, then spec §1.3 "Initialization" steps 1 to 6 with D-21, D-22, D-23, D-47 and D-66, E15, D-24, D-56, D-58, D-65, §2.4 "Initialization and conversion", the openftgs.init block of §2.1, and the [init] rows of §2.3. Plan the vendoring of RoMa's inference modules from commit 77f8d68 exactly as D-65 describes: the files taken, the imports rewritten, the OpenCV import moved into the pose helpers, the xformers branch removed, VENDOR.json with the SHA-256 of every vendored file, and the license files kept. Plan openftgs/init.py: frame slots and the Δt_f/2 admission rule, nearest-axis pairing matched once per unordered pair, RoMa matching with D-58's settings and D-56's determinism, DLT triangulation in float64 with the depth, reprojection and angle filters, colors, μt from the two images' times, E15 velocities with the backward difference on the last slot, s₀, the seeded subset to k_stride·B per slot, and sharding of pairs across GPUs with per-pair seeds. Plan the tests: the remaining parts of U-19, U-33, U-38 and S-4. Check and log the license of verlab/accelerated_features before planning the Tiny RoMa option.
```

## M14 · Command line and end-to-end runs

Scope: `openftgs.cli`; the end-to-end synthetic scenes.
Spec: the CLI block and exit codes in §2.1; §2.2 (every format the commands read or write); D-62; S-9 and S-11 in §3.2 (the analytic ray caster lives in the test suite).
Tests: CLI smoke tests on a synthetic scene for every command; the CLI round trips of U-26; S-9; S-11.

```text
Milestone M14 of docs/clean-room/milestones.md: command line and end-to-end runs. Read CLAUDE.md and docs/clean-room/questions.md, then spec §2.1 from "CLI." to the end of the section, §2.2 in full, D-62, and S-9 and S-11 in §3.2. Plan openftgs/cli.py: ftgs convert, cache, init, train, render, eval, export and bench with the exact options and exit codes of §2.1, --seed overriding the config file, --resume with --force, --profile, config files as JSON TrainConfig, and bench as a stub that M15 fills. Plan tests/test_cli.py running every command on a small synthetic scene, the CLI round trips of U-26, and the analytic NumPy ray caster with textured spheres for S-9 and S-11, with the RoMa initialization of S-9 used only when the [init] extra is installed. State the exit code of each failure path you implement.
```

## M15 · Performance architecture (xhigh)

Scope: everything §2.4 adds on top of the working package.
Spec: §2.4 in full with D-49 to D-57; `PerfConfig`; §3.4 benchmarks B-1 to B-8; `ftgs bench` and `ftgs train --profile`.
Tests: U-28 with the CUDA variant if B-1 calls for it; U-29; U-32; U-31 and U-8 rerun under the chosen defaults; B-1 to B-8 produce their JSON.

```text
Milestone M15 of docs/clean-room/milestones.md: performance architecture. Read CLAUDE.md and docs/clean-room/questions.md, then spec §2.4 in full with D-49 to D-57, PerfConfig, and §3.4. Plan, in this order: the synchronization audit of U-29 and the removal of every host sync outside gsplat's render; ftgs bench with B-1 to B-8 writing the JSON of §3.4 and --profile with NVTX ranges per stage; B-3 to decide whether the CUDA temporal slice of D-50 is written, and if so the operator and U-28's third implementation; B-2 to set packed and precull (D-57) with U-8 rerun; B-4 and the perc_precision options of D-52; B-5 and the Adam choice of D-53; the temporal index for rendering of D-55 with U-32 and B-7; the frame-cache placements and B-6; the multi-GPU sharding of D-54 for initialization, evaluation, rendering and seeds. For each optimization say whether it is exact or approximate, its default, and the benchmark that justifies it. Record every measured number in docs/clean-room/questions.md as calibration data, not as a spec change.
```

## M16 · Differential harness

Scope: the separate `ftgs_harness` package and this package's adapter.
Spec: §3.3 in full (adapter contract, L0 to L3, report format); §2.2 formats; the `ftgs_harness` row of §2.1.
Tests: the harness run with this package as both A and B passes every level; the reference backend against gsplat passes L1 at its tier tolerances.

```text
Milestone M16 of docs/clean-room/milestones.md: differential harness. Read CLAUDE.md and docs/clean-room/questions.md, then spec §3.3 in full, §2.2, and the ftgs_harness row of §2.1. Plan ftgs_harness as a package that imports nothing from openftgs: the adapter contract commands, harness_config.json in TrainConfig field names, the forward-query and answer formats, levels L0 to L3 with their case generation, comparisons, tolerances and statistics, and report.json, report.md and the artifacts folder exactly as the report format section shows. Plan this package's adapter executable. Plan the tests: the harness run with openftgs as both implementations, and L1 between the reference and gsplat backends treated as two implementations. State which levels need a GPU and how long an L3 run takes at the smallest configuration.
```

## M17 · Real-data runs and calibration

Scope: training on Neural3DV, ENeRF-Outdoor and SelfCap; the calibrations the spec defers to the first runs.
Spec: §1.4 "Dataset protocols"; §1.5 reference outcomes; D-10 (reg_until experiment), D-44 and the primitive counts of Table 7 (τ_pos), D-68 (LPIPS convention against the paper's tables), B-1 and B-7 budgets; C-1 to C-5.
Done when: the runs are logged with their metrics, the calibration results are recorded in `docs/clean-room/questions.md`, and the owner has decided what becomes a spec change.

```text
Milestone M17 of docs/clean-room/milestones.md: real-data runs and calibration. Read CLAUDE.md and docs/clean-room/questions.md, then spec §1.4 "Dataset protocols", §1.5, D-10, D-44, D-68, D-69, §3.4 B-1 and B-7, and C-1 to C-5 in §3.1. The datasets are at the paths I give you in this session; nothing derived from SelfCap may be committed. Plan the run matrix: converter checks C-1 to C-5; one Neural3DV scene at the paper's protocol with the default configuration, then the primitive count against Table 7 and what τ_pos or n_max would be needed to match it; the LPIPS convention comparison of D-68 against the paper's tables; the reg_until experiment of D-10; B-1 against the one-hour target and B-7 against 450 FPS; then the remaining scenes. For each run state the command, the expected wall time and the numbers that will be compared with the paper. Plan how results are recorded in docs/clean-room/questions.md as calibration rows, each ending with the decision the owner must make.
```

## M18 · Release

Scope: license review, documentation, packaging, first release.
Spec: §2.3 notes (pip-licenses in CI, LGPL sign-off, the shared libraries inside wheels, vendored license files, pretrained-weight terms, SelfCap never bundled); D-62; Appendix A continuity (the implementation source log is complete); §0 (the clean-room statement).
Done when: the wheel builds, the license scan passes, the README explains the clean-room process and points to the spec and logs, and the version is tagged.

```text
Milestone M18 of docs/clean-room/milestones.md: release. Read CLAUDE.md and docs/clean-room/questions.md, then spec §0, §2.3 with every note under its table, and D-62. Plan the release checklist: the pip-licenses scan on the resolved environment with GPL and AGPL as failures and LGPL listed for my sign-off; a review of the shared libraries bundled inside dependency wheels; the vendored directory's license files and VENDOR.json; the terms of the pretrained weights the package downloads at run time; a check that nothing from SelfCap or Neural3DV is in the repository or the wheel; the README's clean-room statement with links to docs/spec, docs/clean-room/source-log.md and docs/clean-room/questions.md; API documentation for §2.1; the CI matrix; the version, tag and PyPI metadata under the name openftgs. Also audit docs/clean-room/source-log.md: every third-party file or page the code could only have come from must have a row, and no row may name a forbidden source.
```

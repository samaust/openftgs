# The benchmarks in `openftgs.cli`

`ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json` runs one of the eight benchmarks of §3.4 and writes its results as BENCH.json. Benchmarks measure speed and memory, never correctness: U-8 and U-28 to U-33 pin the equivalence of every optimized path, and a benchmark only says which path is faster. The code is the private module `openftgs.cli._bench` (PRD #220, owner's decision); `ftgs`'s parser takes the `bench` command from it. The choices the spec leaves open are Q-166 to Q-170, Q-178 to Q-182, Q-185, Q-187 to Q-190, Q-192 to Q-197.

**`<m15-runs>`**: the folder outside the repository where M15's runs were recorded: the converted scenes they ran on (`<m15-runs>/kitchen/scene.json`, `<m15-runs>/kitchen60/scene.json`, `<m15-runs>/ci-scene/scene.json`) and their BENCH.json files (`<m15-runs>/bench/`). This page, the Calibration data rows of `docs/clean-room/questions.md` and the source log name it so. Only the CI baselines are kept in the repository, under `benchmarks/baselines/`, as they were recorded.

## Subcommands

| Subcommand | Benchmark (§3.4) | Measures | CUDA | `--model` | Built by |
| --- | --- | --- | --- | --- | --- |
| `train-step` | B-1 | time per iteration by stage, peak memory | yes | yes | #226 |
| `raster` | B-2 | `packed` on and off, compaction against gsplat's own culling | yes | | #227 |
| `slice` | B-3 | the temporal slice's eager and compiled implementations | yes | | #225 |
| `losses` | B-4 | LPIPS-VGG's precisions and layouts, fused and PyTorch SSIM | yes | | #228 |
| `adam` | B-5 | fused, foreach and selective Adam | yes | | #229 |
| `data` | B-6 | the share of an iteration spent waiting for data, per cache placement | yes | | #232 |
| `render` | B-7 | rendering speed with and without the temporal index | yes | yes | #231 |
| `init` | B-8 | the wall time of RoMa initialization: matching per pair and per GPU, triangulation, k-NN | yes | | #234 |
| `convert` | B-8 | the wall time of conversion per camera and in total, with and without hardware decoding | no | | #234 |

Each subcommand is one entry of `_BENCHMARKS`, at the end of the module: its B-n, the function that runs it, and whether it needs CUDA and takes `--model`. Every subcommand runs its benchmark: #225 to #235 retired Q-130's stub, which wrote nothing and exited 1 (owner answer, 2026-10-19). Adding a benchmark is a function that takes the context (the parsed arguments, the scene and the model given with `--model`) and returns its setup, cases and budgets, set as `run` in its entry; the framework does the rest.

## Checks and exit codes

The checks run in this order (Q-180):

1. `--model` given to a subcommand other than `train-step` or `render` is a usage error, exit 2.
2. The scene manifest is read as `MultiViewVideo.from_manifest` reads it; a missing or invalid one is exit 2, for `convert` too.
3. Every benchmark but `convert` needs a CUDA device; without one it is exit 2, an invalid environment (Q-166).
4. MODEL is read as `ftgs render` reads it, onto the GPU: a checkpoint directory or an FTGS-IF file; an unreadable one is exit 2 (D-88).
5. BENCH.json's folder is made before the run, so an output that can't be written is exit 2 at once, not after minutes of measuring (Q-142).

Exit 0 means the benchmark ran, whatever its budgets and limits say: a measurement is not a command failure, and the comparison against the stored baselines decides (Q-166, Q-170). A scene the benchmark can't use is exit 2 as well: the synthesized models need more than one timestamp and two training poses (Q-181); `init` needs the `[init]` extra and RoMa's weights in torch.hub's cache (Q-192); `convert` needs a manifest that records a Neural3DV conversion whose source videos are still where its ffmpeg commands read them (Q-169, Q-193); `render` needs a model in the scene's time unit and the LPIPS networks' weights, which B-7's evaluation figures need and which it gets before measuring anything, as `ftgs eval` does (Q-142, Q-194).

A run cut short measured nothing, so it writes nothing, and a BENCH.json already at the path stays as it was (Q-126) (Q-166 as answered):

- Out of GPU memory anywhere in the run, from reading MODEL onto the GPU to the last measurement, B-4's child processes included, is exit 2, naming the benchmark and the error, as in `ftgs train` and `ftgs cache` (Q-142).
- A `FloatingPointError` during the run is exit 3, a numerical failure, naming the benchmark: the trainer's divergence in B-1 and B-6 (Q-69), or renders that B-7's evaluation finds not finite (Q-129, Q-199).
- Any other exception keeps its traceback, Python's exit 1 (Q-142).

## `--model`

`train-step` (B-1) and `render` (B-7) take `--model MODEL`, a checkpoint directory or an FTGS-IF file, so that M17 can rerun them on the trained kitchen, courtyard and studio models; without it they time seeded synthesized models (D-88, from Q-167; §2.1's CLI line since revision 549j). The other subcommands take no model.

## BENCH.json

One JSON object, replaced atomically through a temporary sibling `.<name>.tmp-<pid>`, as the other outputs are (Q-126), with no NaN or Infinity (Q-25). The layout is Q-166's; Q-180 fills in the fields:

```json
{
  "format": "ftgs-bench", "version": 1,
  "benchmark": "B-3",
  "command": ["ftgs", "bench", "slice", "kitchen/scene.json", "--out", "bench/B-3-slice.json"],
  "environment": {
    "gpu": {"name": "NVIDIA GeForce RTX 4090", "count": 1},
    "driver": "595.99.02", "cuda": "13.0", "cudnn": 92400,
    "torch": "2.14.1+cu130", "gsplat": "1.5.3", "fused_ssim": "1.0.0", "python": "3.14.8",
    "openftgs": "0.1.0.dev0", "cpu": "12th Gen Intel(R) Core(TM) i9-12900",
    "commit": "c3c5241669ba01b436c69baf99ac6919c6afac46", "dirty": false
  },
  "scene": {"path": "/…/kitchen/scene.json", "sha256": "…", "cameras": 20, "frames": 300,
            "width": 1352, "height": 1014},
  "date": "2026-10-19T17:27:13+00:00",
  "setup": {"sizes": [100000, …, 3000000], "repetitions": 50, …},
  "cases": [{"case": {"impl": "eager", "n": 100000},
             "figures": {"forward_backward_s": 0.000317, "forward_s": 0.000101, …}}, …],
  "budgets": [{"case": {"impl": "compiled", "resizes": 10}, "figure": "recompilations",
               "limit": 2, "value": 0, "pass": true, "kind": "limit"}]
}
```

- `command` is the command line as a list, the paths as typed, with `--model MODEL` when given.
- `environment` holds what §3.4 asks for, the GPU's model, the driver, CUDA, PyTorch and gsplat, and also cuDNN, fused-ssim (null when it isn't installed), Python, openftgs and the CPU's model, which matters where a figure is bound by the host's launches. The GPU is the current device; without CUDA its name is null and its count 0. The driver's version comes from `nvidia-smi --query-gpu=driver_version` (then `kmd_version`, its newer name), null when nvidia-smi can't answer, since PyTorch has no public accessor for it. `commit` and `dirty` come from git in the folder openftgs is imported from, and only when its `__init__.py` is a file git tracks there: a wheel installed in a checkout's `.venv` gets null. `dirty` means tracked files differ from the commit; untracked files don't count.
- `scene` names the manifest by its absolute path and the SHA-256 of its bytes, with the number of cameras that have images, the most images any one camera has (its frames), and the size of the first training image.
- `date` is the start of the run, in UTC.
- `setup` holds the benchmark's fixed parameters, so a later run can be checked to have measured the same thing.
- `cases` holds one entry per measured case: `case` names its parameters, and `figures` its numbers, times in seconds (names ending `_s`) and memory in bytes.
- `budgets` holds one entry per budget or limit the run is held to, naming the case and the figure: `kind` is `budget` for a §3.4 budget, `bound` for an upper bound standing for one (Q-168's B-1), and `limit` for another spec limit checked on the run, as U-28's recompilations in B-3 and U-8's 1e-6 in B-2's spot check. `pass` follows the direction of the spec's limit: `value` at most `limit` for the upper bounds and limits, at least `limit` for B-7's ≥ 450 FPS, and strictly under it for B-6's "< 1%", as the baseline comparison judges it. `pass` never changes the exit code.

## Measurement

Every benchmark but B-8 measures in the same way (Q-168): a first call, then 10 warm-up calls, then the timed calls, whose medians are the figures. The first call is timed on the wall clock from an idle device to a synchronize after it, so it includes compilation and the first launches of every kernel; it is never one of the timed calls. The timed calls run under the stage clock (`openftgs._stages.Clock("cuda")`), whose CUDA events are recorded on the stream without synchronizing; the clock is read once, after the last call, so nothing synchronizes inside the timed region (§2.4 rule 3, `docs/modules/train/trainer.md`, "Stages"). The calls are issued back to back, so where the GPU's work outlasts the host's launches the events time the GPU, and where it doesn't (a small N) they time the launches, as an iteration would see them.

B-8 times work that runs once, initialization and conversion, so it runs it once, without a warm-up, under the wall clock (`Clock("wall")`), and its figures are sums and spreads of the stages it recorded (below).

B-7's call is a whole render request, which synchronizes inside every render, so each call is timed on the wall clock from a synchronized device to a synchronize after it, after the same first call and warm-up (Q-194).

B-1 times training iterations rather than calls: its warm-up is the first 120 iterations, and its figures are means as well as medians, since an iteration's time is the sum of its stages (below).

The stages are also NVTX ranges, so Nsight Systems shows a benchmark's calls by name. B-2 and B-3 use these of the fixed stage names:

| Stage | In B-2 | In B-3 |
| --- | --- | --- |
| `slice`, `slice.kernel`, `slice.sh_concat` | the temporal slice inside `render` (`FreeTimeGaussians.at_time`) | `slice.kernel` alone: the forward call of `motion.temporal_slice` |
| `raster.forward` and its four sub-stages | `render`'s rasterization forward, compaction included | |
| `loss.l1` | the L1 loss against the camera's training image | |
| `backward` | `torch.autograd.grad` from the L1 loss into the nine stored tensors | `torch.autograd.grad` from the slice's outputs to its inputs: the slice's backward alone, where a training iteration's `backward` holds every backward pass |
| `backward.losses_done`, `backward.raster_done` (points) | where the loss's backward and the rasterization's end inside `backward`, as in `Trainer.step` | |

## The synthesized models

Without `--model`, the benchmarks that need a model synthesize one, seeded, with a given number of rows N and a given active fraction (Q-168; Q-181 for the details):

- **Where.** The centre is the point nearest every training camera's optical axis. D-27's centroid of the camera centres would not do: on a forward-facing rig like kitchen's it lies about 2.6 units behind the content, among the cameras. The view radius is the smallest, over the training cameras, of the camera's distance to the centre times the tangent of its narrower half field of view, so content within it is in every camera's view.
- **What.** A quarter of the rows lie on a static background sphere about the centre, of twice the cameras' largest distance, which encloses every camera; the rest on 32 spheres within half the view radius of the centre, each moving at its own velocity, the rows shared by area. A row's tangent scales are the spacing of its sphere's rows and its normal axis a fifth of that, along the sphere's normal. σ ~ U[0.3, 0.95], SH degree 3.
- **When.** A row passes the opacity cull (D-32) while |t − μt| is within its reach s·√(2 ln 255σ). For an active fraction f < 1 every row's reach is r = f/(2(1 − f)) and μt ~ U[−r, 1 + r], so at any time in [0, 1] a share f of the rows is active, in expectation (within 0.01 at 50,000 rows, the private tests check). At f = 1 every row is active at every time.
- **Order.** The rows are shuffled, so the first n rows of a model are a smaller model of the same kind: B-3 grows by prefixes.

On kitchen, 78 % of the rows project into every camera's image (the 32 spheres and a little background), against 2 to 8 % around D-27's centroid.

## B-1: `ftgs bench train-step`

B-1 times `Trainer.step` by stage (§3.4), on synthesized models of 0.5M, 1M and 3M rows with 20 % of the rows active, plus 3M with 5 % active, the conservative case for D-50 since rasterization then does the least per row; or, with `--model`, on that model alone, at its own SH degree (#226; Q-185 for the details below).

**Configuration.** The default `TrainConfig`, LPIPS-VGG included, except that growth and relocation start at iteration 100, `n_max` = ⌈1.02·N⌉ and `sh_interval` = 1, so every SH band is open from iteration 4 and growth can add at most 2 % to N. The frame cache is built before the first case (`FrameCache.build`), so "auto" only opens it after the first iteration: memory-mapped on kitchen, whose cache is about 23 GB.

**Per case**, a fresh trainer runs three windows:

1. iterations 1 to 120, untimed: the slice's compilation, the cache's placement, the first event at 100;
2. **window A**, 121 to 420, under the stage clock, events at 200, 300 and 400 included; the clock is read once, after 420;
3. **window B**, 421 to 720, without the clock: one pair of CUDA events around each `step()`.

**Window A's figures.** Per stage (`{"n", "active", "stage"}`), over the 297 iterations without an event: the mean, the median, the count and the share, the stage's mean over the iteration's mean. The stages are those of a training iteration (`docs/modules/train/trainer.md`, "Stages"), and the backward stage is also split at its two points into three parts:

| Part | From | To | Holds |
| --- | --- | --- | --- |
| `backward.losses` | `backward` begins | `backward.losses_done` | every loss's backward |
| `backward.raster` | `backward.losses_done` | `backward.raster_done` | the rasterization's backward, projection and SH included |
| `backward.slice` | `backward.raster_done` | `backward` ends | the temporal slice's backward, with the SH concat's and the gradients' accumulation |

Per event (`{"n", "active", "iteration"}`): the `events` stage, the whole iteration and N after it. Then, without gradients, the shares of the rows that pass the opacity cull ("active", D-32) and that are visible (E6), averaged over window A's 300 images and times, on the model as it is at 420; peak allocated and reserved memory over window A; N at iteration 0, at 121 and at 420.

**Window B and the stage clock's cost.** The clock adds about 9 µs of host time per stage (Calibration data, #223), which shows only where an iteration is host-bound. Window B runs the next 300 iterations at the same N without it: the difference of the two windows' means over iterations without an event is `clock_overhead_s`, and window B's mean is the iteration time of a real run.

**The bound** (Q-168, `kind: "bound"`): T_it × window B's mean + the default calendar's 246 events × the event time, against 3,600 s, where the event time is what window B's event iterations (500, 600, 700) take beyond its mean, the log read included. It bounds a default run whose N stays at or below the case's only in N and in the number of events: every iteration is charged at that N, and every event as one with growth. It doesn't bound what the synthesized model can't show, the rasterization cost of trained content and the cost of an event whose relocation has real work to do, nor the spread between runs, so a pass doesn't show the one-hour budget met (Q-168 as answered). M17 judges the budget: the wall time of a default `ftgs train` on kitchen, with the frame cache and the LPIPS networks' weights in place beforehand, and `ftgs bench train-step --model` on the trained model. Its budget entry names the case with the largest N, among equal N the largest bound.

**The copy of the relocation scores** (Q-72 note 5): the median wall time of 50 copies of an [N] float32 GPU tensor to the CPU, as relocation copies its scores before its CPU generator draws, with N the case's at the end.

**Decision rule for D-50** (PRD #220, #226): the slice's share is (`slice.kernel` + `backward.slice`) / `iteration`, over the iterations without an event (`slice_share` in each case's summary), and the largest over the cases counts; its budget entry (`kind: "limit"`) sets it against 10 %. At 10 % or below, `"cuda"` keeps raising and its message cites the Calibration row; above, the conditional CUDA operator's ticket opens under #220. The SH concat's share (`slice.sh_concat`) and `loss.reg`'s, the second pass over all N rows, are recorded too; either above 5 % goes to the owner as a request. `loss.reg`'s share is its forward's only, a lower bound: autograd interleaves its backward with the other losses', so it can't be told apart.

## B-2: `ftgs bench raster`

B-2 times `render` forward plus an L1 backward for each of the four `packed` × `precull` combinations, so that D-57's defaults are set by measurement: gsplat's packed and unpacked layouts (Q-36), and D-32's opacity cull inside gsplat's projection (`internal`) or by compaction before it (`compact`, Q-31). The setup (#227, with Q-187 for the details):

- **Models.** One synthesized model (seed 0) for each N in {10⁶, 3·10⁶} and each active fraction in {5 %, 20 %, 100 %}, six in all, with the eager temporal slice: the slice is the same in every variant, and eager compiles nothing into the first calls and never falls back mid-run as "auto" may (Q-13).
- **Camera and times.** The first training camera in the manifest's order (kitchen's cam01; cam00 is held out), at 10 of its frames evenly spaced in time, round(k·(F − 1)/9) of its F frames for k = 0…9.
- **A call.** `render(model, camera, τ, packed=…, precull=…)` with the camera moved to the GPU once, as the trainer keeps its cameras, since `render` copies a camera held on the host in every call and that copy synchronizes; the black background and the model's SH degree; `losses.l1_loss` against the camera's training image at τ, loaded onto the GPU once; `torch.autograd.grad` of the loss into the nine stored tensors.
- **Order.** For each model, packed/internal (D-57's default), packed/compact, unpacked/internal, unpacked/compact, one after another: each a first call, 10 warm-up calls and 50 timed calls cycling through the 10 times, five each.

Per variant (`{"n", "active", "packed", "precull"}`), the figures are the medians over the 50 calls of:

| Figure | From | To |
| --- | --- | --- |
| `forward_backward_s` | `slice`'s start | `backward`'s end: the whole call, which the decision rule reads |
| `raster_s` | `raster.forward`'s start | the `backward.raster_done` point: the rasterization forward and backward (with the L1, a few µs) |
| `raster_forward_s` | `raster.forward`'s start | its end, compaction included |
| `raster_backward_s` | the `backward.losses_done` point | the `backward.raster_done` point |
| `backward_s` | `backward`'s start | its end |

with the first call (`first_call_s`), the peak allocated memory since just before it, and the spot check's figures. Before timing a model, B-2 renders every variant at each of the 10 times under `no_grad`. For the three variants other than packed/internal it records the largest absolute difference of the image from packed/internal's (`image_max_abs_diff`, #227's check) and the pixels where some channel differs by more than 1e-6, summed over the times (`image_pixels_over_limit`); for the two compacted variants, U-8 itself, the largest difference from the same layout's internal image (`precull_max_abs_diff`). Both maxima are held to 1e-6 by `limit` entries in `budgets`. A difference above it is a finding for the owner, never a looser limit.

Per model (`{"n", "active"}`), untimed, at each of the 10 times in order: `active_rows`, the rows that pass the opacity cull; `visible_rows`, the rows visible in the camera; and `tile_intersections`, the length of gsplat's `isect_ids`. They are counted once, with gsplat's `fully_fused_projection` and `isect_tiles` given the arguments `render` gives them, since U-8 makes them the same in every variant and `render` returns no count. `resident_memory_bytes` is the memory allocated before the variants run, the model and the 10 targets, so a variant's own working memory is its peak less this.

**Decision rule** (#227, Q-187): a combination that, at both N, is at least 3 % faster than packed/internal in `forward_backward_s` at 20 % active and no more than 3 % slower at 5 % and 100 % goes to the owner as a spec-change request for D-57's defaults, in the form of #215. The code keeps D-57's defaults (`packed = True`, `precull = "internal"`) until the owner answers. `raster_s` is reported beside it; where it gives another verdict, the request says so.

The run on kitchen (Calibration data, #227) keeps D-57's defaults. At 20 % active every other combination is 4 to 16 % slower than packed/internal at both N, so none goes to the owner. There packed/internal takes 3.60 ms at 1M and 9.68 ms at 3M, of which the rasterization takes 2.69 and 6.84 ms. Compaction costs most where most rows pass the cull, up to 1.49 times at 3M with every row active, and comes within 2 % only at 3M with 5 % active. Unpacked/internal is 14 to 16 % faster with every row active, but 10 to 26 % slower at 5 %, where it projects all N rows for a few visible ones, and it holds up to 587 MiB more.

The spot check found compaction bitwise in both layouts (U-8), but the unpacked images differ from the packed ones at 1 to 17 of 13.7M pixels per model, by up to 1.6e-3. gsplat 1.5.3's two projections give conics that differ in the last bits, which flips a term at the α skip or the transmittance stop (`docs/modules/render/backends.md`, "Layouts and the opacity cull's placement"). Those `limit` entries fail; the 1e-6 limit stays, and the owner has been told. A first recording, at b0ef3bf, rendered from the manifest's camera, which `render` copies to the GPU in every call, a synchronization training doesn't make, and counted the previous model in the resident memory; it is superseded.

## B-3: `ftgs bench slice`

B-3 times `motion.temporal_slice`'s eager and compiled implementations, forward and backward, at 11 sizes: 10⁵, 1.5·10⁵, 2·10⁵, 3·10⁵, 4·10⁵, 5·10⁵, 7·10⁵, 10⁶, 1.5·10⁶, 2·10⁶ and 3·10⁶ rows, B-1's 0.5M, 1M and 3M among them (Q-182). Its inputs are the first n rows of one synthesized model of 3·10⁶ rows (seed 0, 20 % active), so each of the 10 resizes keeps the rows before it in order and appends new ones, as a growth event does. At each size, eager and then compiled, each given explicitly: "auto" could fall back to eager and time it twice (Q-13). Each call slices at a seeded time in [0, 1], then runs `torch.autograd.grad` from the five outputs a training iteration differentiates (`means_t`, `opacities_t`, `quats_n`, `scales` and `reg_terms`) with fixed upstream gradients, the same for both implementations.

Per case (`{"impl", "n"}`), the figures are the medians over 50 calls of the forward (`forward_s`), the backward (`backward_s`) and the span from the forward's start to the backward's end (`forward_backward_s`), the first call (`first_call_s`), the peak allocated memory, and for the compiled slice dynamo's unique graphs so far. Dynamo is reset before the first size, so the first compiled call compiles whatever ran before in the process; inductor's and Triton's on-disk caches stay as PyTorch configures them, so a machine's first run compiles from scratch and later ones reuse kernels. A last case (`{"impl": "compiled", "resizes": 10}`) gives the recompilations over the resizes, dynamo's unique graphs less one (Q-13), and its budget entry sets them against U-28's limit of 2.

**Decision rule** (Q-182), applied by #226 together with B-1: Q-13's "auto" stays the compiled slice on CUDA while its median forward plus backward is no slower than eager's at every size and its recompilations stay within U-28's limit. Otherwise the numbers go to the owner as a request to change "auto"'s mapping, which Q-13 fixes, and the code keeps it until the owner answers. Whether D-50's CUDA operator is written is B-1's rule, the slice's share of an iteration (PRD #220).

The first run, on kitchen, is in the Calibration data table of `docs/clean-room/questions.md`: the compiled slice is no slower than eager only up to 3·10⁵ rows, where both sit at the host's floor of about 0.3 ms, and 8 to 49 % slower from 4·10⁵ rows on (10 % at 3·10⁶, where it takes 2.46 ms against 2.24). It never recompiled over the 10 resizes. Its first call took 0.53 s with warm compile caches and 2.25 s with empty ones.

That run is superseded. Under `dynamic=True` dynamo had made the 3 and the 4 of the inner dims symbolic, and inductor computed the quaternion's norm in reduction kernels; since #226 the compiled slice's inner dims are static (Q-184, `docs/modules/motion/temporal-slice.md`). Rerun on kitchen, the compiled slice is no slower than eager at every size: 2 to 5 % faster up to 5·10⁵ rows, at the host's floor, 12 % at 7·10⁵, and 41 to 49 % faster from 10⁶ rows on (1.24 against 2.29 ms at 3·10⁶), still with no recompilation. So Q-182's rule keeps "auto" compiled on CUDA.

## B-4: `ftgs bench losses`

B-4 times the two image losses whose cost D-52 decides, LPIPS-VGG and SSIM, at 1352×1014 and 1920×1080 (§3.4; Q-190 for the details below). It sets D-52's exact defaults through Q-173's rule and records how far bf16 strays from float32. It has no budget of its own.

**Images.** The target is the scene's first training image, resized bilinearly with antialiasing when its size differs (on kitchen, 1920×1080 stretches its 4:3 to 16:9); the prediction adds seeded Gaussian noise of σ 0.05. Natural content gives the deviation figures the feature statistics of training; timing doesn't depend on it.

**LPIPS-VGG.** lpips's network is called directly (PRD #220), frozen and in evaluation mode, its inputs mapped to [−1, 1] as `LPIPSLoss` maps them (D-68). One call is the forward on both images and the mean (stage `loss.perceptual`), then the prediction's gradient (stage `backward`), for each of:

| Axis | Values |
| --- | --- |
| precision | `ieee` (cuDNN's TF32 flag off), `tf32` (on, training's default), `bf16` (bfloat16 autocast around the network, TF32 on, the value cast to float32; Q-172) |
| layout | `nchw` (contiguous NCHW inputs and weights); `channels_last_inputs` (channels-last inputs, NCHW weights); `channels_last_inputs_and_weights` |
| cuDNN autotuning | off, on |

`channels_last_inputs` is training's path: `LPIPSLoss` views a render's [..., H, W, 3] batch first, which is channels-last in memory, and every convolution of the network follows that layout (source log 175). So "channels-last on and off" (§3.4) is measured twice: on and off for the inputs, which training already has on, and for the weights, which it doesn't yet.

**Processes.** cuDNN caches the algorithm it picked for a configuration and looks there first whatever `cudnn.benchmark` says, so autotuning off and on can't be compared in one process: a later setting would run the earlier one's picks. The deviations (below) and fused-ssim run in the benchmark's process; every LPIPS case and the PyTorch SSIM path run in one child process per autotuning setting, off and then on. A child that runs out of GPU memory exits with a code of its own, its error last on stderr, and the benchmark reports it as it would in its own process, exit 2 (Q-166 as answered); a child that fails otherwise is a traceback.

**SSIM.** `losses.ssim` with training's window, σ and padding, forward (stage `loss.ssim`) and the prediction's gradient: fused-ssim in the benchmark's process, when it imports, and the PyTorch path in the child processes, in which `fused_ssim` can't be imported, so `ssim()`'s signature doesn't change (PRD #220, owner's decision). The children run it with TF32 on, as training does (D-52), one with autotuning off and one with it on; each gets the folder the benchmark imported openftgs from first on its PYTHONPATH, so every case times the same code.

**Figures.** Per case, as B-3's (`{"loss", "width", "height", …}`): the first call, then the medians over 30 calls after 10 warm-up calls of the forward, the backward and the span from one's start to the other's end, and the peak allocated memory. Per size, TF32's and bf16's deviation from IEEE float32 (`{"deviation", "width", "height"}`), measured first, with the network as training runs it: the two values and their relative difference, the gradients' relative L2 difference, and their largest difference over the largest IEEE gradient element.

**Comparisons.** Per size, a variant's gain over its baseline, 1 − variant / baseline of the median forward plus backward (`{"comparison", "width", "height"}`), and over the sizes the smallest gain and whether it reaches 3 % (`{"comparison"}`):

| Comparison | Baseline → variant | Decides |
| --- | --- | --- |
| `autotuning` | TF32, channels-last inputs: autotuning off → on | cuDNN autotuning in training steps (Q-173) |
| `channels_last_weights` | TF32, autotuning off: channels-last inputs → inputs and weights | channels-last LPIPS weights in training (Q-173) |
| `autotuning_and_channels_last_weights` | both together | reported |
| `tf32` | IEEE → TF32, channels-last inputs | reported (D-52's default) |
| `bf16` | TF32 → bf16, channels-last inputs | reported (bf16 stays opt-in, approximate) |
| `fused_ssim` | the PyTorch path without autotuning → fused-ssim | Q-42's note: no gain sends Q-42 back to the owner |

**Decision rule** (Q-173): an exact option without a `PerfConfig` field is turned on when its smallest gain is at least 3 % and a rerun of B-1 with it on is no slower than B-1's recorded iteration time.

## B-4's first run

On kitchen's first training image (Calibration data, #228), training's LPIPS-VGG path, TF32 with channels-last inputs and NCHW weights, takes 101.6 ms at 1352×1014 (forward 54.9, backward 46.6) and 150.6 ms at 1920×1080, which matches B-1's 52.7 ms of `loss.perceptual` and 44.9 ms of `backward.losses`. Neither exact option reaches Q-173's 3 % at both sizes: autotuning gains 4.3 % and −1.0 %, channels-last weights 1.0 % and 2.9 %, so both stay off and B-1 wasn't rerun. Reported beside them: both together 6.0 % and 3.1 %, and NCHW inputs with autotuning 7.2 % and 3.9 %, combinations the rule doesn't decide, which go to the owner; IEEE float32 is 12 % and 17 % slower than TF32, and slower with channels-last inputs than with NCHW; bf16 is 33 % and 30 % faster than TF32, its value within 2.2·10⁻³ relative and its gradient 17 to 18 % off in L2 (TF32: 3 %). fused-ssim takes 0.41 and 0.66 ms against the PyTorch path's 8.5 and 13.2 ms, about 95 % less. With autotuning, a configuration's first call takes 1 to 27 s. The run took 4 min 13 s.

## B-1's first run

On kitchen (Calibration data, #226), an iteration without an event takes 102, 104 and 115 ms at 0.5M, 1M and 3M rows with 20 % active, and 112 ms at 3M with 5 %. LPIPS-VGG dominates: its forward takes 52.7 ms and every loss's backward 44.9 ms, whatever N. The temporal slice takes at most 1.24 % of an iteration, so D-50's CUDA operator isn't written; the SH concat at most 1.66 % and `loss.reg` 0.11 %, both under 5 %. The bound is 3,462 s at 3M rows, 4 % under 3,600 s, with cuDNN autotuning off and NCHW LPIPS weights as the defaults stood; its inputs were channels-last already (source log 176). That margin, 4.6 ms per iteration, is less than the rasterization's forward and backward at 3M (7.4 ms), so the bound puts the one-hour target within reach of the synthesized model, not met: M17's run decides. B-4 (#228) kept both defaults. The clock's cost was inside the run-to-run spread: window B was 0.08 to 0.26 ms slower than window A, the iterations being GPU-bound.

## B-5: `ftgs bench adam`

B-5 times the optimizer's part of a training iteration, `FTGSOptimizer.set_iteration` followed by `step`, for the three Adam kinds of D-53 (`docs/modules/optim/schedules-and-state.md`), at N = 1M and 3M rows (#229; Q-188 for the details below).

**Setup.** Per N, B-1's synthesized model at that N (seed 0, 20 % active, SH degree 3, so 64 floats per row over the nine stored tensors), with N(0, 1) gradients, seeded, set once on all nine stored tensors. Each kind, in the order fused, foreach, selective, gets a new optimizer from `build_optimizer` with `OptimConfig`'s defaults and the scene's S and Δt_f, on that same model, since Adam's cost doesn't depend on the values. A call sets the next iteration's rates (`it` = 1, 2, …) and takes one step, inside the stage `adam`. It never calls `zero_grad`, which would clear the gradients: the trainer's `adam` stage holds `zero_grad` as well, which only sets the nine gradients to None.

**Selective Adam's mask** is fixed: exactly round(f·N) rows true, chosen by a seeded permutation, so they are spread over the tensors as a render leaves them, where f is the visible fraction B-1 measured at that N with 20 % active, to three decimals: 0.173 at 1M and 0.170 at 3M (Calibration data, Q-185). Its time includes the optimizer's own step counting after gsplat's kernel (Q-45).

**Figures.** Per case (`{"adam", "n"}`): the median over 50 calls after the first call and 10 warm-up calls (`step_s`), the first call, the count, the peak allocated memory, and for selective Adam the mask's fraction. Per N (`{"n"}`): `fused_over_foreach`, fused's median over foreach's.

**Budget** (§3.4, `kind: "budget"`): fused no slower than foreach, one entry per N, `fused_over_foreach` against 1.0 with no tolerance. If fused is slower at either N, the numbers go to the owner as a spec-change request, and the default stays `"fused"` meanwhile (#229, PRD #220).

## B-5's first run

On kitchen (Calibration data, #229), a call takes 2.05 / 4.51 / 0.67 ms at 1M rows and 6.03 / 13.56 / 2.00 ms at 3M for fused / foreach / selective Adam. Fused takes 45 % of foreach's time at both N (`fused_over_foreach` 0.454 and 0.445), so the budget passes and the default stays `"fused"`. Fused agrees with B-1's `adam` stage (2.11 and 6.22 ms at 1.02M and 3.06M rows, `zero_grad` included). Selective Adam, with its mask at 17 % of the rows, takes a third of fused's time, about 4 ms less per iteration at 3M; it stays opt-in, being approximate (§2.4 rule 2, D-53). Fused and selective Adam allocate nothing beyond the parameters, gradients and moments (1,024 B per row); foreach adds about 180 B per row in temporaries.

## B-6: `ftgs bench data`

B-6 times the data wait, the `data` stage of a training iteration, for each placement of the frame cache (§3.4, D-51; #232, Q-189 for the details). It runs B-1's configuration on B-1's synthesized model at 1M rows with 20 % active (seed 0), once per case:

| Case | `frame_cache` | `prefetch` | Budget |
| --- | --- | --- | --- |
| `{"n": 1000000, "placement": "gpu"}` | "gpu" | (takes its frames in place) | < 1 % |
| `{"n": 1000000, "placement": "mmap", "prefetch": 8}` | "mmap" | 8, the default | < 1 % |
| `{"n": 1000000, "placement": "mmap", "prefetch": 0}` | "mmap" | 0: each image read when it is taken | < 1 % |
| `{"n": 1000000, "placement": "off"}` | "off" | (decodes a PNG every iteration) | none |
| `{"n": 3000000, "placement": "gpu"}` | "gpu" | | none: Q-99's note |

**Per case**, a fresh trainer runs iterations 1 to 120 untimed (B-1's warm-up, the event at 100 included), 121 to 320 under the stage clock's CUDA events, read once after 320, and 321 to 420 under the wall clock. The frame cache is built before the first case, so a case only opens it.

**The figure**, `data_share`, is the `data` stage's mean over the `iteration` stage's mean over the 198 timed iterations without an event, a ratio of means as B-1's shares; the events at 200 and 300 are left out, which can only make the share larger. Its budget entry (`kind: "budget"`) holds it strictly under 1 % for every case that reads the cache at 1M rows: a share of exactly 1 % fails, as in `scripts/bench_compare.py`. Beside it: the data stage's mean and median, the mean iteration, and peak allocated and reserved memory from the first iteration on.

**The host window.** The CUDA events see a wait only when the training stream stops for it. A wait on the host, as decoding a PNG, reading a memory-mapped frame or, at prefetch 0 when B-6 was recorded, waiting on a pinned buffer's previous copy (since removed, `docs/modules/train/trainer.md`), happens while the GPU still works through the previous iteration's losses and backward pass, about 100 ms of work on kitchen (B-1) queued after gsplat's read-back in the render, so the events can't see it unless it outlasts that backlog. The wall window gives the host time of the data stage and of an iteration over its 99 iterations without an event, to say how much of that backlog a wait uses. It is a diagnostic, not the B-6 figure.

**The GPU placement** is measured only where "auto" would choose it (Q-99, Q-178). Before each `gpu` case a probe trainer of the same configuration decodes PNGs for one iteration, the path "auto" takes on its first CUDA iteration, and then the free memory (`mem_get_info`'s, plus what the allocator holds unused), the growth to N_max and the cache's size are read as the trainer's rule reads them. They are read once `step()` has returned, while the rule reads them inside it with the iteration's render output, loss and image still held, so the probe's free memory is larger by those tensors, under 1 % of it. A cache above ½·free − growth is skipped: `setup.skipped` lists the case with its reason and that estimate, and it gets no case or budget. On kitchen, whose cache is 23.4 GB, both `gpu` cases are skipped, so `gpu` is measured on its 60-frame conversion (Q-178), where every case runs. A `gpu` case's figures add the estimate (`auto_free_bytes`, `auto_growth_bytes`, `auto_allocated_bytes`, `auto_reserved_bytes`, `cache_bytes`, `auto_limit_bytes`), the device's total memory (`auto_total_bytes`), what lies outside this process's allocator (`auto_outside_bytes`, total − free − allocated: other processes, such as a desktop session, and the CUDA context), and Q-99's margin:

- `beyond_estimate_bytes`: the run's peak reserved memory, less what was allocated after the probe's iteration, the cache and the growth: what the run took beyond the rule's estimate;
- `margin_used`: that over ½·free, the share of the margin the rule leaves for it. At most 1 means a cache at the rule's limit would still have fit, judged from reserved memory, so conservatively; above 1, the ½ margin is too small, which goes to the owner as a decision change.

## B-6's first run

On kitchen and its 60-frame conversion (Calibration data, #232), the data wait takes 0.035 % of an iteration with the GPU cache, 0.035 % with the memory-mapped cache at prefetch 8 and 0.044 % at prefetch 0, all at 1M rows; every budget passes, about 23 times under 1 %. In each cache case the stage's median is 0.036 ms, the conversion of the frame to float32. Decoding PNGs (`off`, no budget) takes 1.95 %, 2.06 ms of an iteration's 105.9, which holds the copy of the decoded float32 image, four times a cached frame's bytes. Kitchen's `gpu` cases were skipped as expected: its 23.44 GB cache is above half the 23.40 GB free after a warm-up iteration.

The host window shows why prefetch 0 still passes: its synchronous read, with the wait on a pinned buffer's event that the debug mode can't see (`docs/modules/train/trainer.md`), takes 0.44 ms of host time against 0.06 ms at prefetch 8, and runs while the GPU works through the previous iteration's losses and backward pass, so the training stream waits 0.01 ms more on average. PNG decoding takes 36 ms of host time, hidden the same way. PRD #220 left the prefetcher as it was; its review then removed the wait from the training thread anyway (§2.4 rule 3): at prefetch 0, and for a frame taken out of order, the copy goes through a fresh block of PyTorch's caching pinned allocator, which reuses a block only once its copy is done, without waiting, and the images are bitwise the same (U-30). These figures are the run with the wait.

Q-99's margin holds at N_max: at 3M rows with the cache on the GPU, the run took 8.57 GiB of reserved memory beyond the rule's estimate after the warm-up iteration (5.43 GiB allocated), 84 % of the 10.16 GiB the ½ margin leaves (54 % from allocated memory). That is close enough to the edge that the rule isn't tightened, and nothing goes to the owner. The device's memory outside PyTorch's allocator was 0.97 GiB, the desktop session's 0.49 GiB and the CUDA context.

## B-7: `ftgs bench render`

B-7 times `render.render_request`, the path `ftgs render`, `ftgs eval` and the trainer's evaluation share (Q-57), with the temporal index of D-55 on and off (`docs/modules/render/backends.md`, "The temporal index"), and from its measurements sets T, the number of distinct times from which `render_request`'s default builds the index (Q-174). Its setup is Q-194's, its rule for T Q-195's.

- **Models.** Synthesized as above (seed 0) at N = 10⁶ and 3·10⁶, with 1 %, 5 % and 20 % of the rows active, each with its default `slice_impl` ("auto"); or the model given with `--model` alone. On kitchen's one-frame buckets (Δt_f = 1/299 of the span) a row of the 1 % model spans about 4 buckets, while the rows of the 5 % and 20 % models span 17 and 76 and so sit on the long list, which every render takes: B-7 sees the index both where it can help and where it can only cost.
- **Camera.** Q-168's: the pose of the first camera in the manifest's camera order that has test images (kitchen's cam00), else of the first camera; fx and fy scaled by 1080/H, H its height, so pixels stay square and the vertical field of view is kept; the principal point at (960, 540); 1920×1080.
- **Requests.** One item per distinct training timestamp (F of them, 300 on kitchen), the camera's tensors on the host as a request file gives them, a black background and the model's SH degree. The buckets' width (`frame_dt`) is a checkpoint's Δt_f, otherwise the scene's, one training frame interval as D-55 has it.
- **Calls.** One call is one `render_request` with the gsplat backend, the index forced on or off, and the outputs on the GPU (`device="cuda"`) or on the CPU: a first call, 10 warm-up calls and 10 timed calls on the wall clock, whose medians give the request's time (`request_s`), FPS (renders over it) and the time per render (`render_s`), beside the first call and the peak allocated memory.
- **Cases** (`{"n", "active", "times", "index", "outputs"}`). At the F times, the index on and off with the outputs on the GPU and on the CPU; with the outputs on the GPU, the index on and off at requests of k times spread evenly over the F, k = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 and 128 below F.
- **The index** (`{"n", "active"}`). Its build time (`build_s`) is the one-time request with the index forced on (`temporal_index=True`) less the same request forced off, outputs on the GPU, so it can be negative where the index saves the render more than the build costs; and the model's break-even, below. B-7 reaches the index through `render_request` alone, as `cli` uses only public names of the modules below it (Q-153): the mean bucket size relative to N and the share of rows on the long list, which only the index's private rows give, are not reported (owner's answer, Q-194).
- **Evaluation** (`{"n", "active", "evaluation"}`). `evaluate_images` on the test split (the training split when there is none) timed once on the wall clock, against the render requests it sends, one per timestamp with the outputs on the CPU as it takes them, also timed once; `render_share` is the second over the first.
- **Budget.** §3.4's ≥ 450 FPS, kind `budget`, on the requests at the F times with the outputs on the GPU, index on and off. It is calibrated: M17 measures the trained kitchen, courtyard and studio models with `--model`.

**The rule for T** (Q-195). A model's break-even is the smallest request size, among the swept sizes and F, at which the request with the index takes no longer than without it, there and at every larger size, outputs on the GPU; a model has none when the index is slower at F. T is the largest break-even over the models that have one, and at least 2, so a request at a single time never builds an index; B-7 writes it as the case `{"threshold": "T"}` (`distinct_times`). The recorded run's T becomes `render._INDEX_MIN_TIMES`, cited by its Calibration row. When no model has a break-even, T stays as it is and the numbers go to the owner, since an "auto" that never pays off is a change to Q-174's switch rather than a value of T.

**The rule for evaluation** (PRD #220). `evaluate_images` keeps one render request per timestamp (Q-68) while `render_share` is at most 10 % at every model; above that, the numbers go to the owner as a request to change Q-68.

## B-7's first run

On kitchen (Calibration data, 2026-10-19; `<m15-runs>/bench/B-7-render.json`, 20 min at the default thread count), at the 300 training times from cam00 at 1920×1080, with the outputs on the GPU, index on / off:

| Model | FPS, index on / off | Build | Break-even |
| --- | --- | --- | --- |
| 1M, 1 % active | 1,658 / 831 | 2.3 ms | 6 times |
| 1M, 5 % | 624 / 824 | 1.5 ms | none |
| 1M, 20 % | 535 / 687 | 1.5 ms | none |
| 3M, 1 % | 1,572 / 344 | 7.5 ms | 6 times |
| 3M, 5 % | 228 / 329 | 4.4 ms | none |
| 3M, 20 % | 190 / 257 | 4.5 ms | none |

So T = 6 (Q-195), now `render._INDEX_MIN_TIMES`. The 5 % and 20 % models' rows all sit on the long list, where the index only costs (`docs/modules/render/backends.md`, "The switch and the width"). The 450 FPS budget holds at 1M and only for the 1 % model at 3M; it is calibrated on the trained models in M17. With the outputs on the CPU every case stays between 108 and 212 FPS, the copy adding about 4.1 ms per render. Rendering is 2.4 to 3.9 % of `evaluate_images`, so evaluation keeps one request per timestamp.

## B-8: `ftgs bench init` and `ftgs bench convert`

B-8 reports the wall time of the two steps that run once per scene, initialization and conversion, on one Neural3DV-layout scene, openftgs-synth's `kitchen` in this project (D-79). It has no budget, only the regression rule, so its BENCH.json has no budget entries. Both run once, under the wall clock, and read the stages that `roma_init` and the converters record (`docs/modules/train/trainer.md`, "Stages").

**`bench init`** (Q-192) runs `roma_init` on the scene with the default `RomaInitConfig`, D-58's roma_outdoor, as `ftgs init roma` runs without options, on every visible GPU (Q-106). It needs the `[init]` extra and the two checkpoints roma_outdoor loads, RoMa's and DINOv2's, already in torch.hub's cache (`torch.hub.get_dir()/checkpoints/`), the files `init.roma_weight_files` names (Q-198). Without either it is exit 2, naming what is missing, before anything runs: it never downloads them, so no download is timed. One `ftgs init roma` run fetches them. The setup holds Q-107's provenance record (every setting, `use_custom_corr`, the vendored RoMa's commit), the devices and the weight files. M15 records it on kitchen's 60-frame conversion: the full scene's 3,900 pairs (300 init frames × 13) are M17's, with B-8's release runs, and the figures per pair, per GPU and per init frame don't depend on the scene's length (owner, 2026-10-19).

| Case | Figures |
| --- | --- |
| `{"part": "init.match", "device"}` | per GPU: `pairs`, and the time per pair as `total_s`, `median_s`, `min_s`, `max_s`. A pair's stage holds its two image reads, the matcher, the draw of its matches (D-22) and its triangulation |
| `{"part": "init.triangulate", "device"}` | per GPU: the same, for the triangulation alone |
| `{"part": "init.knn"}` | `frames`, one E15 search per init frame in the calling thread, with the same spread |
| `{"part": "roma_init"}` | `wall_s`, the whole call; `startup_s`, from its start to the first pair's, which is loading the weights and building one model per GPU; `pairs`, `init_frames` and `points` |

The k-NN searches overlap the matching of the next frame's pairs, and with several GPUs the pairs of one GPU overlap the others', so the parts don't add up to the whole.

**`bench convert`** (Q-169, Q-193) reruns the conversion the scene's manifest records. The converter and its options (frames, resize, the ffmpeg and ffprobe paths) come from the manifest's metadata, and the source folder from the `-i` paths of the recorded ffmpeg commands. A missing source video is exit 2, naming its path; so is a manifest that records no conversion, and one recorded by another converter than Neural3DV's (selfcap reads its videos from a work folder it removes). `--jobs` isn't recorded, so the converter's default runs, one decoding process per four cores (§2.4). Each run converts into a temporary directory in the platform's temporary folder (`TMPDIR`; kitchen at full size needs about 12 GB), removed afterwards.

It runs once with `--hwaccel none`, then once with `--hwaccel cuda` when PyTorch sees a CUDA device and the recorded ffmpeg lists `cuda` among its hardware acceleration methods (`ffmpeg -hwaccels`). A `cuda` run that fails, such as a pixel format D-81 refuses, leaves the CPU run's numbers standing: `setup.skipped.cuda` says why, and the exit stays 0. `setup.hwaccel` lists the runs made.

| Case | Figures |
| --- | --- |
| `{"hwaccel", "camera"}` | `frames`; `wall_s`, from the camera's first frame read to its last frame written; `decode_s`, the sum of its frames' reads from ffmpeg (`convert.decode`), which wait for ffmpeg's decoding; `encode_s`, the sum of its frames' processing and PNG writing in the process pool (`convert.encode`), measured in the pool's processes, so it is work, not waiting, and cameras overlap |
| `{"hwaccel"}` | `cameras`, `frames`; `wall_s`, the converter's call, probing the videos included; the sums of `decode_s` and `encode_s` |

The setup also records the ffmpeg's version line, the CPU count and the temporary folder, since the conversion's time depends on them.

## B-8's first run

On kitchen (Calibration data, #234), the conversion takes 40.7 s with CPU decoding and 40.9 s with `--hwaccel cuda`: a frame's read from ffmpeg takes about 5.5 ms against 118 ms of PNG encoding in the pool, so encoding bounds it, as §2.4 expects, and hardware decoding gains nothing. On kitchen's 60-frame conversion, initialization takes 366 s for 780 pairs: 0.45 s per pair on the RTX 4090, 98 % of the run, with 8 ms of triangulation per pair and 0.22 s of k-NN per init frame. The full scene's 3,900 pairs would take about half an hour.

## The GPU CI job

§3.4 runs B-2 to B-6 in CI on one fixed GPU runner, on a synthetic scene the size of the Neural3DV setup, and fails when a budgeted figure gets more than 10 % worse than its stored baseline. Datasets never enter CI (§2.3), so B-1, B-7 and B-8 run on openftgs-synth's scenes before each release instead (D-79), and their results are stored with it.

**The workflow**, `.github/workflows/gpu-bench.yml`, starts only by hand (`workflow_dispatch`), never on pushes or pull requests, on a self-hosted runner with the labels `self-hosted`, `linux` and `gpu`: a public repository shouldn't run pull requests on the owner's machine (Q-179, the owner's answer). It stays dormant until such a runner is registered. Its steps, within 120 minutes:

1. check that the runner's Python imports torch with a CUDA device, gsplat, lpips, fused-ssim and the checked-out openftgs;
2. generate the scene (below) into the folder `FTGS_CI_SCENE` names, or into the runner's temporary folder;
3. run B-2 to B-6 on `scene.json`, then B-6 again on `scene-60.json`, writing `bench/B-2-raster.json`, `B-3-slice.json`, `B-4-losses.json`, `B-5-adam.json`, `B-6-data.json` and `B-6-data-60.json`;
4. keep those files as the run's artifact `bench`, whether or not a benchmark failed;
5. compare them with the baselines (below); this step's exit code is the job's verdict.

**Setting up a runner.** A Linux machine with a CUDA GPU, registered as a self-hosted runner of this repository with the extra label `gpu`, and two repository variables:

- `FTGS_CI_PYTHON`, the path of a Python whose environment holds a CUDA build of torch, gsplat 1.5.3, lpips 0.1.4 and fused-ssim, installed as the README says (fused-ssim isn't on PyPI and builds against the installed torch). The job puts the checked-out `src` first on `PYTHONPATH`, so the commit being tested is the code measured whatever openftgs the environment holds; LPIPS's VGG weights download on the first run.
- `FTGS_CI_SCENE`, optional: a folder outside the workspace that keeps the scene between runs. The generator completes a folder that holds the same generation instead of refusing it, and B-6's frame caches beside the manifests stay valid, so a later run skips both. The folder needs about 12 GB for the PNGs (2.0 MB each) and 28 GB for the two frame caches (23.4 GB for the 19 training cameras' 300 frames, 4.7 GB for their first 60), about 40 GB in all.

**The scene**, `scripts/ci_scene.py OUT` (Q-197), seeded, outside the wheel: 20 cameras, cam00 to cam19, evenly spaced on a horizontal arc of 90° and radius 4 at height 1 around a common point, each looking at it, with a 60° horizontal field of view and the principal point at the image centre; cam00 is held out, as Neural3DV's is. 300 frames per camera, frame k at k/30 s, as 1352×1014 8-bit RGB PNGs at `images/camXX/NNNNNN.png`, PNG level 1 as the converters write them. `scene.json` lists them all; `scene-60.json` the first 60 frames of the same PNGs, for B-6's `gpu` placement, since the full scene's 23.4 GB cache can't sit on a 24 GiB GPU (Q-178). Both manifests are written with `openftgs.io.write_scene_manifest`, and their metadata holds only the generator's parameters, so their bytes repeat on any machine.

The images are procedural, not renders: the benchmarks' costs don't depend on the pixels, except B-6's `off` placement, which decodes a PNG every iteration. So the pixels are made to compress as a converted openftgs-synth frame does: smooth colour interpolated from a coarse random grid, three oriented sinusoids, 12 discs that move over the 10 s, and Gaussian noise of 1 level: a frame takes 2.0 MB at level 1 and about 30 ms to decode, where a converted `kitchen` frame takes 1.9 to 2.0 MB and 29 to 36 ms. Each frame is drawn from its own seed (the scene's seed, the camera, the frame), so the PNGs don't depend on how many processes encode them (`--jobs`, one per CPU by default). The "1M primitives" of §3.4 are the benchmarks' own synthesized models (Q-181): B-2, B-5 and B-6 build theirs at 1M rows among other sizes, around the cameras' common point.

**The comparison**, `scripts/bench_compare.py RUN_DIR [--baselines DIR]` (Q-170, Q-196). Each BENCH.json in RUN_DIR is compared with the file of the same name in `DIR/<gpu>/`, by default `benchmarks/baselines/<gpu>/`, where `<gpu>` is the GPU model the run recorded, in lower case with hyphens (`nvidia-geforce-rtx-4090`):

| Benchmark | Compared, per case | Fails when |
| --- | --- | --- |
| B-2 | `forward_backward_s` of each `packed` × `precull` combination, at each N and active fraction | above 1.10 × the baseline's |
| B-3 | `forward_backward_s` of each implementation at each size | above 1.10 × the baseline's |
| B-4 | `forward_backward_s` of each LPIPS-VGG variant and SSIM path at each size | above 1.10 × the baseline's |
| B-5 | `step_s` of each Adam kind at each N | above 1.10 × the baseline's |
| B-6 (both files) | `data_share` of each case with a budget entry | not under 1 %, whatever the baseline's |

A share near 0 is noise in relative terms, so B-6 keeps §3.4's absolute budget; B-2 to B-4 have no budget of their own, so their times are held to their baselines (Q-170). Cases match by their parameters exactly. The comparison also fails on a run without a baseline (B-6's included), a baseline of the run's GPU model without a run, a baseline recorded on another GPU model or of another benchmark, a scene of another size, and a case measured on one side only, so a benchmark that skipped a case is caught. It judges nothing else: not the first calls, memory, the other figures, B-5's own budget, nor the `limit` entries, which check equivalence rather than speed (B-2's spot check fails on every recorded run, a finding the owner has). A different scene SHA-256, driver, CUDA, cuDNN, PyTorch, gsplat, fused-ssim or Python is printed as a note beside the checks. Exit 0 when every check passes, 1 when one fails, 2 for invalid input.

**The baselines** are the BENCH.json files of a reference run, kept in `benchmarks/baselines/<gpu>/` under the names the workflow writes, one folder per GPU model; a run on a model without a folder fails rather than borrowing another model's. The RTX 4090's come from the owner's machine, run with the workflow's commands at the default thread count (#235).

**Refreshing them**, after a deliberate slowdown, a new driver or PyTorch, a change to a benchmark's cases or a new version of the generator: run the workflow, or its commands by hand on a quiet machine, put the six BENCH.json files in one folder (the workflow's artifact `bench` is one), run `python scripts/bench_compare.py FOLDER --update`, which copies each into its GPU model's folder, and commit them with the reason. A new GPU model starts the same way: its first run fails for want of baselines, and `--update` makes its folder.

## The baselines' first runs

On the owner's RTX 4090 (Calibration data, #235), the workflow's six commands took 17 s (B-2), 4 s (B-3), 250 s (B-4), 4 s (B-5), 141 s and 237 s (B-6), about 11 minutes, with the frame caches built beforehand. The CI scene measures what kitchen does: B-2's times are within −3.8 to +1.2 % of kitchen's, and its tile intersections within 2 %. Run 1 became the baselines; run 2, right after it, passes against them, all 99 checks, 74 of its 94 medians within ±3 %. Two things sit near the rule's edges, and they go to the owner as Q-170 findings. B-4's times move by up to 8.4 % between runs (11 of its 42 cases by 5.9 to 8.4 %), close to the 10 % limit, so B-4 is the likeliest to fail CI without a regression. And run 1's compiled slice at 7·10⁵ and 10⁶ rows in B-3 was 23 and 36 % slower than run 2's, so those two baselines would let a slowdown of up to 35 and 50 % through.

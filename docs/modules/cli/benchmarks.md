# The benchmarks in `openftgs.cli`

`ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json` runs one of the eight benchmarks of §3.4 and writes its results as BENCH.json. Benchmarks measure speed and memory, never correctness: U-8 and U-28 to U-33 pin the equivalence of every optimized path, and a benchmark only says which path is faster. The code is the private module `openftgs.cli._bench` (PRD #220, owner's decision); `ftgs`'s parser takes the `bench` command from it. The choices the spec leaves open are Q-166 to Q-170, Q-178 to Q-182, Q-185, Q-187 to Q-190, Q-192 to Q-197, Q-253, Q-269 and Q-270.

**`<m15-runs>`**: the folder outside the repository where M15's runs were recorded: the converted scenes they ran on (`<m15-runs>/kitchen/scene.json`, `<m15-runs>/kitchen60/scene.json`, `<m15-runs>/ci-scene/scene.json`, and `<m15-runs>/kitchen60-crops/scene.json`, kitchen60 cropped to a size per camera by `scripts/crop_scene.py` for #350) and their BENCH.json files (`<m15-runs>/bench/`). This page, the Calibration data rows of `docs/clean-room/questions.md` and the source log name it so. Only the CI baselines are kept in the repository, under `benchmarks/baselines/`, as they were recorded.

**`<m17-runs>`**: the folder outside the repository where M17's validation runs are recorded, `/home/auss/datasets/openftgs-m17`: the converted `courtyard` and `studio` scenes, the init points, the run directories, their summaries, the TrainConfig override files and the BENCH.json files (`<m17-runs>/bench/`). M17 reads `kitchen` from `<m15-runs>/kitchen` (its `scene.json` and its frame cache) and never writes there. The runs are listed in `docs/clean-room/validation-runs.md` (#317), with #395's reruns at revision 549o.

## Subcommands

| Subcommand | Benchmark (§3.4) | Measures | CUDA | `--model` | Built by |
| --- | --- | --- | --- | --- | --- |
| `train-step` | B-1 | time per iteration by stage, peak memory | yes | yes | #226 |
| `raster` | B-2 | `packed` on and off, compaction against gsplat's own culling | yes | | #227 |
| `slice` | B-3 | the temporal slice's eager and compiled implementations | yes | | #225 |
| `losses` | B-4 | each perceptual network of D-89 in its precisions and layouts, fused and PyTorch SSIM | yes | | #228, #389 |
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

Exit 0 means the benchmark ran, whatever its budgets and limits say: a measurement is not a command failure, and the comparison against the stored baselines decides (Q-166, Q-170). A scene the benchmark can't use is exit 2 as well: the synthesized models need more than one timestamp and two training poses (Q-181); `init` needs the `[init]` extra and RoMa's weights in torch.hub's cache (Q-192); `convert` needs a manifest that records a Neural3DV conversion whose source videos are still where its ffmpeg commands read them (Q-169, Q-193); `render` needs a model in the scene's time unit (Q-194); `losses` needs the six trunk networks' weights in torch.hub's cache, which it never downloads, its message giving the command that fetches the missing ones, and the `[init]` extra for the four DINO networks, and refuses a cached file that fails its check, all before measuring anything (Q-253, Q-270). B-7's evaluation figures build the evaluator, whose ꟻLIP downloads nothing (#387), so `render` no longer needs any weights.

A run cut short measured nothing, so it writes no BENCH.json (its folder, made before the run, stays), and one already at the path stays as it was (Q-126; Q-166 as answered):

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
- `budgets` holds one entry per budget or limit the run is held to, naming the case and the figure: `kind` is `budget` for a §3.4 budget, `bound` for an upper bound standing for one (Q-168's B-1), and `limit` for another spec limit checked on the run, as U-28's recompilations in B-3 and U-8's 1e-6 in B-2's spot check. `pass` follows the direction of the spec's limit: `value` at most `limit` for the upper bounds and limits, at least `limit` for B-7's calibrated floor (Q-239), and strictly under it for B-6's "< 1%", as the baseline comparison judges it. `pass` never changes the exit code.

## Measurement

Every benchmark but B-8 measures in the same way (Q-168): a first call, then 10 warm-up calls, then the timed calls, whose medians are the figures. The first call is timed on the wall clock from an idle device to a synchronize after it, so it includes compilation and the first launches of every kernel; it is never one of the timed calls. The timed calls run under the stage clock (`openftgs._stages.Clock("cuda")`), whose CUDA events are recorded on the stream without synchronizing; the clock is read once, after the last call, so nothing synchronizes inside the timed region (§2.4 rule 3, `docs/modules/train/trainer.md`, "Stages"). The calls are issued back to back, so where the GPU's work outlasts the host's launches the events time the GPU, and where it doesn't (a small N) they time the launches, as an iteration would see them.

B-8 times work that runs once, initialization and conversion, so it runs it once, without a warm-up, under the wall clock (`Clock("wall")`), and its figures are sums and spreads of the stages it recorded (below).

B-7's call is a whole render request, which synchronizes inside every render, so each call is timed on the wall clock from a synchronized device to a synchronize after it, after the same first call and warm-up (Q-194).

B-1 times training iterations rather than calls: its warm-up is the first 120 iterations, and its figures are means as well as medians, since an iteration's time is the sum of its stages (below).

The stages are also NVTX ranges, so Nsight Systems shows a benchmark's calls by name. B-2 and B-3 use these of the fixed stage names:

| Stage | In B-2 | In B-3 |
| --- | --- | --- |
| `slice`, `slice.kernel` | the temporal slice inside `render`, without the SH since #423 (Q-274) | `slice.kernel` alone: the forward call of `motion.temporal_slice` |
| `raster.forward` and its five sub-stages | `render`'s rasterization forward, compaction included; `raster.forward.sh_gather` builds the SH coefficients of the rows the SH evaluation receives (Q-274, #423) | |
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

**Configuration.** The default `TrainConfig`, ꟻLIP included at its calibrated weight (`perc_net = "flip"`, `lambda_perc = None`, D-89; `LossConfig`'s defaults since #392), except that growth and relocation start at iteration 100, `n_max` = ⌈1.02·N⌉ and `sh_interval` = 1, so every SH band is open from iteration 4 and growth can add at most 2 % to N. The frame cache is built before the first case (`FrameCache.build`), so "auto" only opens it once it decides, after the second iteration on these one-size scenes (Q-225, Q-232): memory-mapped on kitchen, whose cache is about 23 GB.

**Per case**, a fresh trainer runs three windows:

1. iterations 1 to 120, untimed: the slice's compilation, the cache's placement, the first event at 100;
2. **window A**, 121 to 420, under the stage clock, events at 200, 300 and 400 included; the clock is read once, after 420;
3. **window B**, 421 to 720, without the clock: one pair of CUDA events around each `step()`.

**Window A's figures.** Per stage (`{"n", "active", "stage"}`), over the 297 iterations without an event: the mean, the median, the count and the share, the stage's mean over the iteration's mean. The stages are those of a training iteration (`docs/modules/train/trainer.md`, "Stages"), and the backward stage is also split at its two points into three parts:

| Part | From | To | Holds |
| --- | --- | --- | --- |
| `backward.losses` | `backward` begins | `backward.losses_done` | every loss's backward |
| `backward.raster` | `backward.losses_done` | `backward.raster_done` | the rasterization's backward, projection and SH included, and the SH coefficients' backward into `sh0` and `shN` (#423 measured it there, before and after its change) |
| `backward.slice` | `backward.raster_done` | `backward` ends | the temporal slice's backward and the gradients' accumulation |

Per event (`{"n", "active", "iteration"}`): the `events` stage, the whole iteration and N after it. Then, without gradients, the shares of the rows that pass the opacity cull ("active", D-32) and that are visible (E6), averaged over window A's 300 images and times, on the model as it is at 420; peak allocated and reserved memory over window A; N at iteration 0, at 121 and at 420.

**Window B and the stage clock's cost.** The clock adds about 9 µs of host time per stage (Calibration data, #223), which shows only where an iteration is host-bound. Window B runs the next 300 iterations at the same N without it: the difference of the two windows' means over iterations without an event is `clock_overhead_s`, and window B's mean is the iteration time of a real run.

**The bound** (Q-168, `kind: "bound"`): T_it × window B's mean + the default calendar's 246 events × the event time, against 3,600 s, where the event time is what window B's event iterations (500, 600, 700) take beyond its mean, the log read included. It bounds a default run whose N stays at or below the case's only in N and in the number of events: every iteration is charged at that N, and every event as what an event at the cap adds, where growth computes its candidates but adds no rows (Q-185). It doesn't bound what the synthesized model can't show, the rasterization cost of trained content and the cost of an event that adds rows or whose relocation has real work to do, nor the spread between runs, so a pass doesn't show the one-hour budget met (Q-168 as answered). M17 judges the budget: the wall time of a default `ftgs train` on kitchen, with the frame cache in place beforehand (and, in M17's LPIPS-VGG run, its weights), and `ftgs bench train-step --model` on the trained model. Its budget entry names the case with the largest N, among equal N the largest bound.

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

with the first call (`first_call_s`), the peak allocated memory since just before it, and the spot check's figures. Before timing a model, B-2 renders every variant at each of the 10 times under `no_grad`. For the three variants other than packed/internal it records the largest absolute difference of the image from packed/internal's (`image_max_abs_diff`, #227's check) and the pixels where some channel differs by more than 1e-6, summed over the times (`image_pixels_over_limit`); for the two compacted variants, U-8 itself, the largest difference from the same layout's internal image (`precull_max_abs_diff`). U-8's maximum is held to U-8's GPU tolerance by a `limit` entry in `budgets`; a difference above it is a finding for the owner, never a looser limit. The comparison with packed/internal has no `limit` entry, so its figures report the difference without judging it (Q-187 as answered, #278): for the unpacked variants, D-57 lets the two layouts differ at isolated pixels where a term sits at one of E6's discontinuities (revision 549k), and for packed/compact it is U-8's comparison, which `precull_max_abs_diff` holds.

Per model (`{"n", "active"}`), untimed, at each of the 10 times in order: `active_rows`, the rows that pass the opacity cull; `visible_rows`, the rows visible in the camera; and `tile_intersections`, the length of gsplat's `isect_ids`. They are counted once, in the unpacked layout, with gsplat's `fully_fused_projection` and `isect_tiles` given the arguments `render` gives them, since `render` returns no count and every variant has the same ones: U-8 makes them the same with and without compaction, and gsplat 1.5.3's packed and unpacked projections give bitwise-equal radii, 2D means and depths, the inputs of `isect_tiles`, and differ only in the conics (a probe at 1M rows with 5 % active, B-2's spot-check row in the Calibration data). `resident_memory_bytes` is the memory allocated before the variants run, the model and the 10 targets, so a variant's own working memory is its peak less this.

**Decision rule** (#227, Q-187): a combination that, at both N, is at least 3 % faster than packed/internal in `forward_backward_s` at 20 % active and no more than 3 % slower at 5 % and 100 % goes to the owner as a spec-change request for D-57's defaults, in the form of #215. The code keeps D-57's defaults (`packed = True`, `precull = "internal"`) until the owner answers. `raster_s` is reported beside it; where it gives another verdict, the request says so.

The run on kitchen (Calibration data, #227) keeps D-57's defaults. At 20 % active every other combination is 4 to 16 % slower than packed/internal at both N, so none goes to the owner. There packed/internal takes 3.60 ms at 1M and 9.68 ms at 3M, of which the rasterization takes 2.69 and 6.84 ms. Compaction costs most where most rows pass the cull, up to 1.49 times at 3M with every row active, and comes within 2 % only at 3M with 5 % active. Unpacked/internal is 14 to 16 % faster with every row active, but 10 to 26 % slower at 5 %, where it projects all N rows for a few visible ones, and it holds up to 587 MiB more.

The spot check found compaction bitwise in both layouts (U-8), but the unpacked images differ from the packed ones at 1 to 17 of 13.7M pixels per model, by up to 1.6e-3. gsplat 1.5.3's two projections give conics that differ in the last bits, which flips a term at the α skip or the transmittance stop (`docs/modules/render/backends.md`, "Layouts and the opacity cull's placement"). That run held the layouts to 1e-6 by `limit` entries, which failed; revision 549k's D-57 records the difference, and since #278 B-2 reports it as figures only (Q-187 as answered). A first recording, at b0ef3bf, rendered from the manifest's camera, which `render` copies to the GPU in every call, a synchronization training doesn't make, and counted the previous model in the resident memory; it is superseded.

## B-3: `ftgs bench slice`

B-3 times `motion.temporal_slice`'s eager and compiled implementations, forward and backward, at 11 sizes: 10⁵, 1.5·10⁵, 2·10⁵, 3·10⁵, 4·10⁵, 5·10⁵, 7·10⁵, 10⁶, 1.5·10⁶, 2·10⁶ and 3·10⁶ rows, B-1's 0.5M, 1M and 3M among them (Q-182). Its inputs are the first n rows of one synthesized model of 3·10⁶ rows (seed 0, 20 % active), so each of the 10 resizes keeps the rows before it in order and appends new ones, as a growth event does. At each size, eager and then compiled, each given explicitly: "auto" could fall back to eager and time it twice (Q-13). Each call slices at a seeded time in [0, 1], then runs `torch.autograd.grad` from the five outputs a training iteration differentiates (`means_t`, `opacities_t`, `quats_n`, `scales` and `reg_terms`) with fixed upstream gradients, the same for both implementations.

Per case (`{"impl", "n"}`), the figures are the medians over 50 calls of the forward (`forward_s`), the backward (`backward_s`) and the span from the forward's start to the backward's end (`forward_backward_s`), the first call (`first_call_s`), the peak allocated memory, and for the compiled slice dynamo's unique graphs so far. Dynamo is reset before the first size, so the first compiled call compiles whatever ran before in the process; inductor's and Triton's on-disk caches stay as PyTorch configures them, so a machine's first run compiles from scratch and later ones reuse kernels. A last case (`{"impl": "compiled", "resizes": 10}`) gives the recompilations over the resizes, dynamo's unique graphs less one (Q-13), and its budget entry sets them against U-28's limit of 2.

**Decision rule** (Q-182), applied by #226 together with B-1: Q-13's "auto" stays the compiled slice on CUDA while its median forward plus backward is at most 1.03 × eager's at every size and its recompilations stay within U-28's limit. The 3 % is Q-173's: up to 5·10⁵ rows both slices sit at the host's launch floor of about 0.3 ms, where a few microseconds decide which is faster (the CI scene's baseline run has compiled 1.01 × eager's at 7·10⁵ rows), and a loss inside the run-to-run noise is no loss. Otherwise the numbers go to the owner as a request to change "auto"'s mapping, which Q-13 fixes, and the code keeps it until the owner answers. Whether D-50's CUDA operator is written is B-1's rule, the slice's share of an iteration (PRD #220).

The first run, on kitchen, is in the Calibration data table of `docs/clean-room/questions.md`: the compiled slice is no slower than eager only up to 3·10⁵ rows, where both sit at the host's floor of about 0.3 ms, and 8 to 49 % slower from 4·10⁵ rows on (10 % at 3·10⁶, where it takes 2.46 ms against 2.24). It never recompiled over the 10 resizes. Its first call took 0.53 s with warm compile caches and 2.25 s with empty ones.

That run is superseded. Under `dynamic=True` dynamo had made the 3 and the 4 of the inner dims symbolic, and inductor computed the quaternion's norm in reduction kernels; since #226 the compiled slice's inner dims are static (Q-184, `docs/modules/motion/temporal-slice.md`). Rerun on kitchen, the compiled slice is no slower than eager at every size: 2 to 5 % faster up to 5·10⁵ rows, at the host's floor, 12 % at 7·10⁵, and 41 to 49 % faster from 10⁶ rows on (1.24 against 2.29 ms at 3·10⁶), still with no recompilation. So Q-182's rule keeps "auto" compiled on CUDA.

## B-4: `ftgs bench losses`

B-4 times the image losses whose cost D-52 decides, each perceptual network of D-89 and SSIM, at 1352×1014 and 1920×1080 (§3.4; Q-190 for the setup, Q-269 for its per-network form since #389). It sets D-52's exact defaults per network through Q-173's rule, as Q-272 applies it (#394), and records how far TF32 and bf16 stray from IEEE float32. It has no budget of its own. Until revision 549o it timed LPIPS-VGG alone; that setup and its two runs are kept below as history.

**What it needs.** The six trunk networks' weights already in torch.hub's cache (`torch.hub.get_dir()/checkpoints/`, under Q-258's names): a Caffe trunk's conversion `openftgs-<perc_net>.safetensors` or its publisher's file, a DINO network's checkpoint. B-4 never downloads them, so no download happens unasked during a benchmark; a missing one is exit 2 before anything runs (Q-253, Q-270). The message names each missing network and the publisher's file a build would fetch, then gives, on a line of its own, the command that fetches exactly those with the Python that ran `ftgs`, quoted for a POSIX shell, for example `/path/to/python -c 'from openftgs.losses import PerceptualLoss; [PerceptualLoss(n) for n in ["vgg16_oxford", "dino_s16"]]'`: building `openftgs.losses.PerceptualLoss(net)` downloads its file once from its publisher, checks it against its pinned SHA-256 and caches it (Q-258), as the first `ftgs train` with that `loss.perc_net` does. When DINO networks are among the missing ones, it adds that they need the `[init]` extra first, `pip install 'openftgs[init]'` (Q-248), which every DINO network needs anyway. The `weights` tests fetch all six too. Every network is then built in the benchmark's process before anything is measured, so a cached file that fails its byte count or SHA-256 (Q-258) and a DINO network without `[init]` are exit 2 too, naming the file or the extra, before any child starts. The setup's `weights` names the cached file each trunk was built from.

**Images.** The target is the scene's first training image, resized bilinearly with antialiasing when its size differs (on kitchen, 1920×1080 stretches its 4:3 to 16:9); the prediction adds seeded Gaussian noise of σ 0.05. Natural content gives the deviation figures the feature statistics of training; timing doesn't depend on it.

**The networks.** Each is `PerceptualLoss(net)` as D-89 builds it, frozen and in evaluation mode, built on the CPU and moved to CUDA for its cases, in D-89's order: `flip`, `haarpsi`, `vgg16_oxford`, `alexnet_bvlc`, `dinov2_s14`, `dinov2_b14`, `dino_s16`, `dino_b16`. One call is its forward on both images, [1, H, W, 3] as `PerceptualLoss.forward` takes them (Q-41) (stage `loss.perceptual`), then the prediction's gradient (stage `backward`), for each of:

| Axis | Values |
| --- | --- |
| precision | `ieee` (cuDNN's TF32 flag off), `tf32` (on, training's default), and for the six trunk networks `bf16`: bfloat16 autocast around the call, the images float32 and the value cast to float32, as the trainer's private bfloat16 class runs a trunk network (Q-172, Q-250), TF32 on. ꟻLIP and HaarPSI have no bfloat16 path (Q-250) |
| layout | `channels_last_inputs`: the prediction a contiguous [1, H, W, 3] tensor, as a render comes, whose batch-first view the network takes is channels-last, with no copy; `nchw`: the images laid out as for `channels_last_inputs`, both copied inside the timed forward so that the batch-first views the network takes are contiguous NCHW, as the trainer copies them, the prediction's gradient flowing back through the copy in the stage `backward`, and under `bf16` the copy inside the autocast (Q-269 as changed, #427; until then a contiguous [1, 3, H, W] tensor given through its [1, H, W, 3] view, the copy left out); and for the two Caffe trunks `channels_last_inputs_and_weights`: the render's view, with a copy of the network converted with `to(memory_format=torch.channels_last)`, made once per network, as LPIPS-VGG's B-4 timed it (Q-269, the owner's answer of 2026-10-26) |
| cuDNN autotuning | off, on |

Since #394 the trainer gives each network the layout and autotuning B-4 chose for it (`docs/modules/train/trainer.md`; Q-273); the setup's `training_autotuning` records per network and precision whether the trainer autotunes where SSIM takes fused-ssim, as on the GPU runner (since #428; until then per network), and its `training_layout` the layout it gives the network at that precision and autotuning, both in an iteration that isn't deterministic. Where SSIM takes its PyTorch fallback on CUDA, every network's iteration autotunes (Q-273, the owner's answer of 2026-10-26), which `training_autotuning` doesn't record. Since #427, B-4's `nchw` cases time the trainer's copy to NCHW and its backward, which B-4 alone decides on for every network but the default (Q-269, Q-272); the prediction's gradient ends at the leaf in the layout the copy's backward gives it, so a copy back to the render's layout, if the rasterizer's backward makes one (17 to 22 µs at these sizes, #427), isn't timed; before, they started from an NCHW tensor and left the copy to B-1, which runs the default network only. Channels-last weights, which Q-173 left off for LPIPS-VGG and §2.4 keeps off, are measured again for the two Caffe trunks, the networks of D-89 made of convolutions, since #380's review; the DINO networks' only convolution is their patch embedding. TF32 reaches ꟻLIP's and HaarPSI's filters, which run as convolutions, and only the patch embedding of a DINO network: matrix products stay at "highest" in every case (D-52).

**Processes.** cuDNN caches the algorithm it picked for a configuration and looks there first whatever `cudnn.benchmark` says, so autotuning off and on can't be compared in one process: a later setting would run the earlier one's picks. The deviations (below) and fused-ssim run in the benchmark's process, one network on the GPU at a time; every network's timed cases and the PyTorch SSIM path run in one child process per autotuning setting, off and then on, which builds the networks the benchmark names to it one at a time and frees each before the next (VGG16's autotuned workspace reached 13.7 GiB as LPIPS-VGG, and the ViT-B checkpoints are 346 MB). A child that runs out of GPU memory exits with a code of its own, its error the last line of its output, and the benchmark reports it as it would in its own process, exit 2 (Q-166 as answered); a child that fails otherwise is a traceback.

**SSIM.** `losses.ssim` with training's window, σ and padding, forward (stage `loss.ssim`) and the prediction's gradient: fused-ssim in the benchmark's process, when it imports, and the PyTorch path in the child processes, in which `fused_ssim` can't be imported, so `ssim()`'s signature doesn't change (PRD #220, owner's decision). The children run it with TF32 on, as training does (D-52), one with autotuning off and one with it on; each gets the folder the benchmark imported openftgs from first on its PYTHONPATH, so every case times the same code.

**Figures.** Per case, as B-3's: `{"loss": "perceptual", "network", "width", "height", "precision", "layout", "autotuning"}` for a network, `{"loss": "ssim", "width", "height", "path", …}` for SSIM; the first call, then the medians over 30 calls after 10 warm-up calls of the forward, the backward and the span from one's start to the other's end, and the peak allocated memory. Per network and size, TF32's and, for a trunk network, bf16's deviation from IEEE float32 (`{"deviation", "network", "width", "height"}`), measured first, on the plain path (channels-last inputs, autotuning off): the two values and their relative difference, the gradients' relative L2 difference, and their largest difference over the largest IEEE gradient element. They are reported whatever their size: #388 measured bf16 gradients up to 61 % off in L2 on the CPU and 152 % on CUDA for `dinov2_b14` (source log 455).

**Comparisons.** Per network and size, a variant's gain over its baseline, 1 − variant / baseline of the median forward plus backward (`{"comparison", "network", "width", "height"}`), and over the sizes the smallest gain and whether it reaches 3 % (`{"comparison", "network"}`); the baseline is the network's plain path, TF32 with channels-last inputs and no autotuning, as Q-190's was LPIPS-VGG's:

| Comparison | Baseline → variant | Decides |
| --- | --- | --- |
| `autotuning` | TF32, channels-last inputs: autotuning off → on | cuDNN autotuning for the network in training steps (Q-173, Q-272) |
| `nchw_inputs` | TF32, autotuning off: channels-last inputs → NCHW inputs | the NCHW copy for the network (Q-273; Q-223's for LPIPS-VGG) |
| `autotuning_and_nchw_inputs` | TF32: channels-last inputs, autotuning off → NCHW inputs, autotuning on | both together, LPIPS-VGG's path since #326 |
| `channels_last_weights` | TF32, autotuning off: channels-last inputs → inputs and weights; the Caffe trunks only | channels-last weights for the network (`train._CHANNELS_LAST_WEIGHTS`) |
| `autotuning_and_channels_last_weights` | TF32: channels-last inputs, autotuning off → inputs and weights, autotuning on; the Caffe trunks only | both together |
| `tf32` | IEEE → TF32, channels-last inputs | reported (D-52's default) |
| `bf16` | TF32 → bf16, channels-last inputs; the trunk networks only | reported (bf16 stays opt-in, approximate) |
| `fused_ssim` (`{"comparison", "width", "height"}`) | the PyTorch path without autotuning → fused-ssim | Q-42's note: no gain sends Q-42 back to the owner |

The setup's `turn_on` lists the five the rule decides, the last two for the Caffe trunks only. The IEEE and bf16 cases also run in both layouts and both autotuning settings, so a network's layout under those precisions is set apart, as Q-223 did for LPIPS-VGG (Q-273).

**Decision rule** (Q-173 as changed, Q-272, the owner's answer of 2026-10-26): an exact option without a `PerfConfig` field goes on for a network only when its smallest gain is at least 3 % in both of B-4's recorded runs, the CI scene's and kitchen's. B-4 alone decides for the networks other than the default, since B-1 runs the default configuration only; for the default, ꟻLIP, B-1 is rerun at the code that sets the defaults and is to be no slower than B-1 at the code before it. Under the other precisions and in a deterministic run, the layout follows the same threshold, NCHW inputs against the render's view at the network's autotuning (Q-273), and so do the Caffe trunks' channels-last weights at every precision and autotuning.

## B-4 per network: the recorded run

On the CI scene, as the GPU job runs it, at the contract's code (12e715d), alone on the RTX 4090 (Calibration data, #394; 6 min 53 s, 327 cases, the stored baseline until #380's review), and on kitchen right after as supporting evidence, an option reaches 3 % at both sizes in both runs for two networks only. For vgg16_oxford, autotuning with NCHW inputs gains +4.7 % / +4.7 % on the CI scene and +3.4 % / +3.0 % on kitchen, while neither option alone does. For alexnet_bvlc all three do, and autotuning with NCHW inputs gains the most, +15.0 % / +18.5 % and +15.4 % / +16.7 %. So both autotune in training in TF32, with NCHW inputs where they are faster (`docs/modules/train/trainer.md`); under the other precisions, autotuning is read per precision since #428 (Q-273). No option reaches 3 % in both runs for ꟻLIP, HaarPSI or the four DINO networks; haarpsi's +6.9 % / +4.7 % on kitchen is noise at 1.5 ms, −16.3 % / +4.8 % on the CI scene.

| Network | Plain path, ms (1352×1014 / 1920×1080) | TF32 over IEEE | bf16 over TF32 |
| --- | --- | --- | --- |
| `flip` | 10.10 / 15.78 | +0.0 / +0.1 % | |
| `haarpsi` | 1.46 / 1.96 | +7.1 / +0.1 % | |
| `vgg16_oxford` | 96.27 / 144.24 | +8.6 / +9.8 % | +33.7 / +33.6 % |
| `alexnet_bvlc` | 9.76 / 17.26 | +18.8 / +86.9 % | −349.6 / +21.0 % |
| `dinov2_s14` | 31.06 / 45.42 | +0.2 / +0.2 % | +73.6 / +76.2 % |
| `dinov2_b14` | 61.43 / 104.68 | +0.2 / +0.1 % | +70.8 / +76.3 % |
| `dino_s16` | 17.07 / 34.29 | +0.2 / +0.3 % | +56.2 / +74.7 % |
| `dino_b16` | 36.46 / 69.69 | +0.0 / −0.2 % | +62.3 / +73.6 % |

Without autotuning, cuDNN picks very slow kernels for alexnet_bvlc in two places, in both runs: IEEE float32 with channels-last inputs at 1920×1080 (131.9 ms, forward 121.6, against 16.0 with NCHW inputs) and bf16 at 1352×1014 in both layouts (about 43 ms, backward about 39), hence its bf16 figure. Under IEEE, NCHW inputs make vgg16_oxford 10 to 16 % faster; under bf16 they make it 7 to 12 % slower, as for LPIPS-VGG. bf16's gradient strays from IEEE float32's by 4 to 11 % in L2 for the trunks on kitchen, 19 % and 30 % for dinov2_b14; TF32's by 2.3 to 4.1 % for the Caffe trunks, at most 1 % for the DINO networks and at most 6·10⁻⁹ for ꟻLIP and HaarPSI. fused-ssim takes 0.41 / 0.66 ms against the PyTorch path's 8.56 / 13.20 (95 % less); autotuning makes the PyTorch path 7 to 11 % faster. The largest peak allocated memory is 13.7 GiB (vgg16_oxford, NCHW inputs with autotuning), the longest first call 26.8 s (vgg16_oxford, IEEE, autotuning).

Comparing the kitchen run with the new baseline as the GPU job would, 4 of its 176 timed cases are more than 10 % slower (haarpsi in TF32 at 1352×1014 +16.7 %, ꟻLIP in IEEE with autotuning +10.4 and +10.5 %, dino_b16 in bf16 with NCHW inputs +10.8 %), all cases of 1.7 to 15 ms; the rest move by −7.0 to +9.3 %. That goes to the owner as a Q-170 finding: one run of B-4's new cases is a noisier baseline than LPIPS-VGG's 42 cases were.

## B-4 with channels-last weights: the rerun of #380's review

Q-269 had left channels-last weights out, so the trainer's choice for the two Caffe trunks, which accept them, rested on no measurement. Since #380's review B-4 times them as a third layout for those two, at every precision and with autotuning off and on (Q-269, the owner's answer of 2026-10-26). It was recorded twice at 1f4c451, alone on the RTX 4090, as #394's run was: on the CI scene (7 min 56 s, 363 cases, the stored baseline until #427) and on kitchen right after (7 min 57 s), as supporting evidence (Calibration data, #380's review).

Channels-last weights reach 3 % nowhere, so every network keeps its weights as built, contiguous NCHW (`docs/modules/train/trainer.md`). With the render's channels-last view, every convolution already runs channels-last, so converting the weights saves only their per-call conversion: −1.4 to +1.0 % against the render's view with NCHW weights, at every precision and autotuning, in both runs. Against the plain path, `channels_last_weights` gains +0.3 to +0.7 %. `autotuning_and_channels_last_weights` meets 3 % for alexnet_bvlc (+6.4 / +9.8 % on the CI scene, +6.1 / +9.9 % on kitchen), as autotuning alone does, but stays below the autotuning with NCHW inputs it takes (+15.3 / +18.3 % and +14.9 / +18.5 %). For vgg16_oxford it reaches +2.0 % at most. Wherever the trainer gives these networks NCHW inputs, channels-last weights are slower than them, by 2.9 to 16.2 % for vgg16_oxford and 3.4 to 31.8 % for alexnet_bvlc, and they leave alexnet_bvlc's slow IEEE kernel at 1920×1080 without autotuning in place (131.0 ms against 15.95 with NCHW inputs).

| Caffe trunk, TF32, ms (1352×1014 / 1920×1080) | Render's view | NCHW inputs | Channels-last weights |
| --- | --- | --- | --- |
| `vgg16_oxford`, autotuning off | 96.29 / 144.26 | 100.83 / 145.49 | 95.89 / 143.82 |
| `vgg16_oxford`, autotuning on | 94.76 / 143.59 | 91.71 / 137.51 | 94.39 / 143.23 |
| `alexnet_bvlc`, autotuning off | 9.79 / 17.24 | 9.28 / 15.33 | 9.72 / 17.15 |
| `alexnet_bvlc`, autotuning on | 9.23 / 15.60 | 8.30 / 14.08 | 9.16 / 15.55 |

#394's figures repeat: on the CI scene every other network's plain path within 0.5 % of #394's but haarpsi's and ꟻLIP's at 1352×1014 (1.66 against 1.46 ms, 10.50 against 10.10), vgg16_oxford's autotuning with NCHW inputs +4.8 / +4.7 % and +4.7 / +4.6 %, and SSIM's fallback 11.0 / 10.3 % faster with autotuning. Compared with #394's baseline before the refresh, the run failed only on its 24 new cases, which had no baseline, and on two haarpsi cases at 1352×1014 of 1.5 ms, 13.7 % and 16.8 % slower, as #394's kitchen run found such small cases noisy (Q-170); the other 182 timed medians moved by −6.3 to +5.7 %. Against the refreshed baseline, the kitchen run passes, its timed medians within −11.0 to +4.7 %.

## B-4 with the trainer's copy: #427's reruns

Since #427 the `nchw` cases time the copy the trainer makes to give a network NCHW inputs (Q-269 as changed), and the trainer autotunes per network and precision (#428). B-4 was recorded twice at 7fbe8bb, alone on the RTX 4090: on the CI scene (7 min 54 s, 363 cases, now the stored baseline) and on kitchen right after (7 min 55 s), as supporting evidence (Calibration data, #427).

The copy adds about 0.05 ms to the Caffe trunks' NCHW cases, so Q-272's outcomes stand. vgg16_oxford's autotuning with NCHW inputs gains +4.63 / +4.63 % on the CI scene and +4.67 / +4.56 % on kitchen; #394's +3.03 % at 1920×1080 on kitchen came from a plain path of 149.59 ms, which #380's runs and these put at 144.2 to 144.3 ms. alexnet_bvlc's autotuning with NCHW inputs gains +14.28 / +17.81 % and +14.60 / +17.73 %. No option reaches 3 % for the other six. Q-273's layouts and the Caffe trunks' channels-last weights read as before at every precision and autotuning.

| Caffe trunk, TF32, ms (1352×1014 / 1920×1080) | Render's view | NCHW inputs, the copy timed |
| --- | --- | --- |
| `vgg16_oxford`, autotuning off | 96.21 / 144.18 | 100.78 / 145.42 |
| `vgg16_oxford`, autotuning on | 94.85 / 143.59 | 91.75 / 137.51 |
| `alexnet_bvlc`, autotuning off | 9.76 / 17.25 | 9.31 / 15.48 |
| `alexnet_bvlc`, autotuning on | 9.23 / 15.62 | 8.36 / 14.18 |

Two of Q-273's autotuning readings per precision change on these two runs, and go to the owner (#427) before the trainer or Q-273's row does:
- **alexnet_bvlc under IEEE** (NCHW inputs): +3.14 / +5.82 % and +3.10 / +5.32 %. That is 3 % in both runs, after −3.00 / +5.53 % and +9.69 / −2.00 % at #394, and +3.03 / +5.24 % and +3.03 / +5.36 % at #380.
- **dinov2_b14 under bf16** (the render's view): +4.62 / +4.91 % and +4.64 / +5.00 %, after +1.92 / +0.44 % and +0.94 / +4.92 % on kitchen at #394 and #380.

Until the owner decides, the trainer leaves the flag to the caller there, as Q-273 answered on #422.

Against the stored baseline of #380's review, the CI run passes: its 206 timed medians are within −6.1 to +6.4 %, its NCHW cases, now with the copy, within −3.8 to +6.4 %. Against the refreshed baseline, the kitchen run passes too, its timed medians within −7.8 to +5.8 %.

## B-4 with LPIPS-VGG (until revision 549o)

B-4's first two runs timed LPIPS-VGG, the paper's network, alone (Q-190 as first confirmed). lpips 0.1.4's network was called directly (PRD #220), its inputs mapped to [−1, 1] as `LPIPSLoss` mapped them (D-68), in `ieee`, `tf32` and `bf16` × `nchw`, `channels_last_inputs` and `channels_last_inputs_and_weights` (channels-last inputs and weights, a copy of the network converted with `to(memory_format=torch.channels_last)`) × autotuning off and on. `channels_last_inputs` was training's path until #326; since #326 training autotuned and gave the network NCHW inputs under IEEE and TF32, and channels-last inputs under bf16 and under TF32 with `deterministic` (Q-173 as changed, Q-223). The deviations were taken per size with channels-last inputs and autotuning off. The comparisons were `autotuning`, `channels_last_weights` (TF32, autotuning off: channels-last inputs → inputs and weights; the option Q-173 decided), `autotuning_and_channels_last_weights`, `autotuning_and_nchw_inputs` (the combination Q-173's change adopted, #326), `tf32`, `bf16` and `fused_ssim`, each against TF32 with channels-last inputs and no autotuning. The two runs' figures follow.

### B-4's first run

On kitchen's first training image (Calibration data, #228), training's LPIPS-VGG path, TF32 with channels-last inputs and NCHW weights, takes 101.6 ms at 1352×1014 (forward 54.9, backward 46.6) and 150.6 ms at 1920×1080, which matches B-1's 52.7 ms of `loss.perceptual` and 44.9 ms of `backward.losses`. Neither exact option reaches Q-173's 3 % at both sizes: autotuning gains 4.3 % and −1.0 %, channels-last weights 1.0 % and 2.9 %, so both stay off and B-1 wasn't rerun. Reported beside them: both together 6.0 % and 3.1 %, and NCHW inputs with autotuning 7.2 % and 3.9 %, combinations the rule doesn't decide, which the owner left out, for M17 (Q-173's answer); IEEE float32 is 12 % and 17 % slower than TF32, and slower with channels-last inputs than with NCHW; bf16 is 33 % and 30 % faster than TF32, its value within 2.2·10⁻³ relative and its gradient 17 to 18 % off in L2 (TF32: 3 %). fused-ssim takes 0.41 and 0.66 ms against the PyTorch path's 8.5 and 13.2 ms, about 95 % less. With autotuning, a configuration's first call takes 1 to 27 s. The run took 4 min 13 s.

### B-4's second run

With the trainer giving the network NCHW inputs and autotuning (Calibration data, #326), `autotuning_and_nchw_inputs` gains 6.1 % at 1352×1014 and 4.5 % at 1920×1080 over the old default, so the adopted path meets Q-173's 3 % at both sizes; autotuning alone gains 3.2 % and 0.4 %, channels-last weights 0.4 % and 0.3 %. bf16 is again faster with channels-last inputs than with NCHW, which is why it keeps them (Q-223). The adopted path's first call takes 1.9 s and its peak allocated memory reaches 13.7 GiB, autotuning's workspace. The default `ftgs train` on kitchen, B-1's measure (Q-168), took 3,380 s against M17's 3,498 s, its GPU memory peaking at 23,492 of 24,564 MiB. The run took 4 min 6 s.

## B-1's first run

Measured with LPIPS-VGG, B-1's perceptual network until revision 549o. On kitchen (Calibration data, #226), an iteration without an event takes 102, 104 and 115 ms at 0.5M, 1M and 3M rows with 20 % active, and 112 ms at 3M with 5 %. LPIPS-VGG dominates: its forward takes 52.7 ms and every loss's backward 44.9 ms, whatever N. The temporal slice takes at most 1.24 % of an iteration, so D-50's CUDA operator isn't written; the SH concat at most 1.66 % and `loss.reg` 0.11 %, both under 5 %. The bound is 3,462 s at 3M rows, 4 % under 3,600 s, with cuDNN autotuning off and NCHW LPIPS weights as the defaults stood; its inputs were channels-last already (source log 176). That margin, 4.6 ms per iteration, is less than the rasterization's forward and backward at 3M (7.4 ms), so the bound puts the one-hour target within reach of the synthesized model, not met: M17's run decides. B-4 (#228) kept both defaults. The clock's cost was inside the run-to-run spread: window B was 0.08 to 0.26 ms slower than window A, the iterations being GPU-bound.

## B-1 at ꟻLIP

At the contract's code (12e715d), with ꟻLIP at its calibrated weight and the trainer's interim path for it, the render's view with autotuning on (Calibration data, #394; the reference Q-272 holds #394's defaults to), an iteration without an event takes 16.5, 18.9 and 30.3 ms at 0.5M, 1M and 3M rows with 20 % active, and 27.3 ms at 3M with 5 % (window B), against 102 to 115 ms with LPIPS-VGG. ꟻLIP's forward takes 5.5 to 5.9 ms, 18 to 35 % of an iteration, and every loss's backward 5.7 to 6.0 ms, whatever N; Adam's step grows with N, from 1.3 to 6.7 ms. The temporal slice takes at most 5.54 % of an iteration, under D-50's 10 %, so D-50's CUDA operator still isn't written; `loss.reg` at most 0.74 %. The SH concat now takes 6.64 % at 3M with 20 % active and 7.70 % with 5 % (2.0 and 2.1 ms), above B-1's 5 %: it hasn't slowed down, the iteration has shortened, and it has gone to the owner. The bound is 916 s at 3M rows against 3,600 s. The run took 1 min 19 s.

At #394's code, where ꟻLIP's iterations leave autotuning as the caller set it (`docs/modules/train/trainer.md`), B-1's rerun is no slower, Q-272's check (window B within 1.03 × the reference's in each case): 15.0, 17.5, 28.1 and 25.3 ms, 0.91 to 0.93 of the reference's (Calibration data, #394). Stages without a cuDNN convolution came out 4 to 9 % faster too, so most of that difference is the two runs' conditions rather than the change, which B-4 measured as neutral for ꟻLIP. The SH concat's share at 3M stays above 5 %, at 6.63 and 7.46 %.

On validation run 35's trained kitchen model (#395; `docs/clean-room/validation-runs.md` and its Calibration data), at the integration branch's eaeb791, after #394, an iteration without an event takes 30.9 ms (window B), against 118.3 ms on M17's model with LPIPS-VGG: ꟻLIP's forward 5.5 ms (17 %), every loss's backward 5.4 ms, the rasterizer's backward 8.2 ms (26 %), Adam 6.2 ms (20 %). The slice takes 4.37 % of an iteration and the SH concat 5.91 % (1.88 ms), above 5 % as on the synthesized models. The bound is 934 s against 3,600 s, and the budget's own measure, the default `ftgs train` on kitchen (Q-168), took 862 s (14 min 22 s) against M17's 3,498 s. The run took 31 s.

## B-1 and B-2 after #423: the SH gather

Since #423 the gsplat backend concatenates `sh0` and `shN` only for the rows its SH evaluation receives, in `raster.forward.sh_gather`, and a training iteration runs no `slice.sh_concat` (Q-274). B-1 and B-2 were recorded on kitchen at main's code and then at #423's, back to back, alone on the RTX 4090 (Calibration data, #423).

B-1 is no slower in any case and faster at 3M rows, Q-272's check against #394's rerun. Window B takes 14.60, 16.54, 25.45 and 22.11 ms, 0.873 to 0.973 of the reference.

The SH work fell from 1.86 ms (6.65 and 7.42 % of an iteration at 3M) plus packed mode's gather of the concat to 0.54 and 0.11 ms (2.13 and 0.51 %), under PRD #220's 5 %. Its backward runs in `backward.raster`, before and after: at 3M that part falls by 0.95 and 1.29 ms, while `backward.slice` stays at 0.97 ms. D-50's slice share rises to 5.44 and 6.25 %, still under 10 %: the slice is unchanged and the iteration shorter.

B-2 keeps D-57's defaults: no combination meets #227's rule. Packed/internal is 12 to 25 % faster with 5 % and 20 % of the rows active, but 4.6 % and 6.0 % slower with every row active (7.96 against 7.61 ms at 1M, 22.14 against 20.88 at 3M). There packed mode gathers nearly every row, and its backward accumulates into `sh0` and `shN` apart. That slowdown has gone to the owner with #423.

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

**The GPU placement** is measured only where "auto" would choose it (Q-99, Q-178). Before each `gpu` case a probe trainer of the same configuration decodes PNGs for as many iterations as "auto" waits on CUDA, until the iteration after the first at the last training image size to appear (two for one size; Q-232), and then the free memory (`mem_get_info`'s, plus what the allocator holds unused), the working memory that iteration left reserved (the reserve, reserved − allocated, Q-225), the growth to N_max and the cache's size are read as the trainer's rule reads them. They are read once `step()` has returned, while the rule reads them inside it with the iteration's render output, loss and image still held, so the probe's free memory is larger by those tensors, under 1 % of it. A cache above ½·(free − reserve) − growth is skipped: `setup.skipped` lists the case with its reason and that estimate, and it gets no case or budget. On kitchen, whose cache is 23.4 GB, both `gpu` cases are skipped, so `gpu` is measured on its 60-frame conversion (Q-178), where every case runs. A `gpu` case's figures add the estimate (`auto_free_bytes`, `auto_growth_bytes`, `auto_allocated_bytes`, `auto_reserved_bytes`, `cache_bytes`, `auto_limit_bytes`), the device's total memory (`auto_total_bytes`), what lies outside this process's allocator (`auto_outside_bytes`, total − free − allocated: other processes, such as a desktop session, and the CUDA context), and Q-99's margin:

- `beyond_estimate_bytes`: the run's peak reserved memory, less what was reserved after the probe's last iteration (what was allocated then and the reserve), the cache and the growth: what the run took beyond the rule's estimate;
- `margin_used`: that over ½·(free − reserve), the share of the margin the rule leaves for it. At most 1 means a cache at the rule's limit would still have fit, judged from reserved memory, so conservatively; above 1, the ½ margin is too small, which goes to the owner as a decision change.

## B-6's first run

On kitchen and its 60-frame conversion (Calibration data, #232), the data wait takes 0.035 % of an iteration with the GPU cache, 0.035 % with the memory-mapped cache at prefetch 8 and 0.044 % at prefetch 0, all at 1M rows; every budget passes, about 23 times under 1 %. In each cache case the stage's median is 0.036 ms, the conversion of the frame to float32. Decoding PNGs (`off`, no budget) takes 1.95 %, 2.06 ms of an iteration's 105.9, which holds the copy of the decoded float32 image, four times a cached frame's bytes. Kitchen's `gpu` cases were skipped as expected: its 23.44 GB cache is above half the 23.40 GB free after a warm-up iteration.

The host window shows why prefetch 0 still passes: its synchronous read, with the wait on a pinned buffer's event that the debug mode can't see (`docs/modules/train/trainer.md`), takes 0.44 ms of host time against 0.06 ms at prefetch 8, and runs while the GPU works through the previous iteration's losses and backward pass, so the training stream waits 0.01 ms more on average. PNG decoding takes 36 ms of host time, hidden the same way. PRD #220 left the prefetcher as it was; its review then removed the wait from the training thread anyway (§2.4 rule 3): at prefetch 0, and for a frame taken out of order, the copy goes through a fresh block of PyTorch's caching pinned allocator, which reuses a block only once its copy is done, without waiting, and the images are bitwise the same (U-30). These figures are the run with the wait.

Q-99's margin holds at N_max: at 3M rows with the cache on the GPU, the run took 8.57 GiB of reserved memory beyond the rule's estimate after the warm-up iteration (5.43 GiB allocated), 84 % of the 10.16 GiB the ½ margin leaves (54 % from allocated memory). That is close enough to the edge that the rule isn't tightened, and nothing goes to the owner. The device's memory outside PyTorch's allocator was 0.97 GiB, the desktop session's 0.49 GiB and the CUDA context.

Those figures measure the rule as it stood until #325, ½·free − growth, and what lay beyond it from the memory allocated after the probe's iteration. On studio at M17 (#317; Calibration data) the run at 3M took 11.10 GB beyond that estimate, 1.017 of the margin, and the default training came within 524 MiB of the card, while the GPU cache gained nothing over the memory-mapped one. The owner changed Q-99, and the rule now sets aside the warm-up iteration's working memory before halving (Q-225): on the recorded figures it gives studio at 3M a limit of 6.21 GB against its 8.58 GB cache, which is memory-mapped, and kitchen60 at 3M 7.93 GB against 4.69 GB, which stays on the GPU. B-6's rerun under the new rule (Calibration data, #325) agrees: on studio both `gpu` cases are skipped, and on kitchen60, where "auto" keeps the GPU, `margin_used` is 0.002 at 1M and 0.42 at 3M; every budget passes on both. At #326's code, with cuDNN autotuning in the trainer, the first probe in B-6's process autotunes and leaves a reserve of about 1.2 GB instead of 6 to 9 GB, so studio's 1M `gpu` case runs, with `margin_used` 0.68, and kitchen60's 1M case rises to 0.45 (0.43 at 3M), while the peaks barely move; studio's default training at that code memory-maps its cache, with 0.18 GB to spare under the limit, and stays 4.9 GiB under the card (Calibration data, #325).

## B-6 at a size per camera

On kitchen60-crops (Calibration data, #350), kitchen60 with camera `cam<i>` cropped by i pixels on the right and at the bottom, the 19 training sizes put the probe's read after iteration 74 (Q-232). Every budget passes, as on kitchen60, and "auto" still chooses the GPU at both N, but the run takes more beyond the estimate: `margin_used` is 0.56 at 1M and 0.66 at 3M, against 0.043 and 0.42 on kitchen60, with peak reserved memory 4.7 and 4.0 GB higher for peak allocated memory 0.1 GB lower. `scripts/reserve_trace.py` shows why on the default configuration: the first iteration at each new size returns the allocator's blocks to the driver (autotuning's trials, as Q-225 measured at one size; this run has no control without autotuning), the iteration after it holds one size's working memory again (5.5 to 5.9 GB), and once several sizes alternate the allocator keeps about three times that (16.66 GB), with no block returned. The reserve "auto" reads at iteration 74 is 35 % of the later one, which fails #350's criterion for a settled reserve and goes back to the owner (Q-232, Q-238).

## B-7: `ftgs bench render`

B-7 times `render.render_request`, the path `ftgs render`, `ftgs eval` and the trainer's evaluation share (Q-57), with the temporal index of D-55 on and off (`docs/modules/render/backends.md`, "The temporal index"), and from its measurements sets T, the number of distinct times from which `render_request`'s default builds the index (Q-174). Its setup is Q-194's, its rule for T Q-195's.

- **Models.** Synthesized as above (seed 0) at N = 10⁶ and 3·10⁶, with 1 %, 5 % and 20 % of the rows active, each with its default `slice_impl` ("auto"); or the model given with `--model` alone. On kitchen's one-frame buckets (Δt_f = 1/299 of the span) a row of the 1 % model spans about 4 buckets, while the rows of the 5 % and 20 % models span 17 and 76 and so sit on the long list, which every render takes: B-7 sees the index both where it can help and where it can only cost.
- **Camera.** Q-168's: the pose of the first camera in the manifest's camera order that has test images (kitchen's cam00), else of the first camera; fx and fy scaled by 1080/H, H its height, so pixels stay square and the vertical field of view is kept; the principal point at (960, 540); 1920×1080.
- **Requests.** One item per distinct training timestamp (F of them, 300 on kitchen); on an unsynchronized scene, where those outnumber the B-7 camera's frames because every camera carries its own sync offset (up to 1,380 on studio's 60-frame window, 23 cameras × 60 frames, and 1,346 on its conversion, about 45 GB of 1080p outputs on the GPU), the B-7 camera's own timestamps instead (F = 60 on studio, Q-221), the setup's `times_rule` saying which; the camera's tensors on the host as a request file gives them, a black background and the model's SH degree. The buckets' width (`frame_dt`) is a checkpoint's Δt_f, otherwise the scene's, one training frame interval as D-55 has it.
- **Calls.** One call is one `render_request` with the gsplat backend, the index forced on or off, and the outputs on the GPU (`device="cuda"`) or on the CPU: a first call, 10 warm-up calls and 10 timed calls on the wall clock, whose medians give the request's time (`request_s`), FPS (renders over it) and the time per render (`render_s`), beside the first call and the peak allocated memory.
- **Cases** (`{"n", "active", "times", "index", "outputs"}`). At the F times, the index on and off with the outputs on the GPU and on the CPU; with the outputs on the GPU, the index on and off at requests of k times spread evenly over the F, k = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 and 128 below F.
- **The index** (`{"n", "active"}`). Its build time (`build_s`) is the one-time request with the index forced on (`temporal_index=True`) less the same request forced off, outputs on the GPU, so it can be negative where the index saves the render more than the build costs; and the model's break-even, below. B-7 reaches the index through `render_request` alone, as `cli` uses only public names of the modules below it (Q-153): the mean bucket size relative to N and the share of rows on the long list, which only the index's private rows give, are not reported (owner's answer, Q-194).
- **Evaluation** (`{"n", "active", "evaluation"}`). `evaluate_images` on the test split (the training split when there is none) timed once on the wall clock, against the render requests it sends, one per timestamp with the outputs on the CPU as it takes them, also timed once; `render_share` is the second over the first.
- **Budget.** §3.4's budget is calibrated, and the owner set it from M17's reruns on the trained models (Calibration row of 2026-10-21; Q-239): a regression floor at 90 % of each trained default model's FPS with the index on, at 1920×1080 and N = 3M, kitchen ≥ 169, courtyard ≥ 198 and studio ≥ 136 FPS, held as `_B7_FLOORS` in the code and keyed by the scene manifest's `metadata.source` (`n3dv/kitchen`, `opencv-yaml/courtyard`, `selfcap/studio`) with the recorded N, on the card the floors were measured on (an RTX 4090). The model given with `--model` gets one entry of kind `budget` when its scene's `source` is in the table, its N is the recorded one and the run is on that card: the request at the F times with the index on and the outputs on the GPU, `pass` at or above the floor; the index-off request has none (courtyard's index-off FPS sits under its floor). The synthesized models, a model of a scene without a recorded floor, a model at another N and a run on another card are held to nothing: `budgets` is empty (Q-240). §1.5's 450 FPS, the paper's reference outcome, is reported in the setup as `reference_fps` beside `floor_fps`, `floor_source` and `floor_rule`, never as a pass or fail.

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

So T = 6 (Q-195), `render._INDEX_MIN_TIMES` until M17 (below). The 5 % and 20 % models' rows all sit on the long list, where the index only costs (`docs/modules/render/backends.md`, "The switch and the width"). Against the paper's 450 FPS, the reference figure, the 1M models and the 1 % model at 3M are above it and the other 3M models under; the calibrated budget was set on the trained models in M17 (below). With the outputs on the CPU every case stays between 108 and 212 FPS, the copy adding about 4.1 ms per render. Rendering is 2.4 to 3.9 % of `evaluate_images`, so evaluation keeps one request per timestamp.

M17 (#317) reran B-7 on the trained kitchen, courtyard and studio models (N = 3M each, outputs on the GPU): 187.7 / 180.7, 220.5 / 192.5 and 150.9 / 149.0 FPS with the index on / off, every one under 450; break-evens 32, 12 and 32 (studio at its test view's 60 times, Q-221), so T = 32 by the same rule, now `render._INDEX_MIN_TIMES`. Every model has a break-even. Rendering is 2.8 to 4.6 % of `evaluate_images`. From these figures the owner set B-7's calibrated budget as the floors above (Q-239), 90 % of each model's index-on FPS; #351 then reran the three models against them (Calibration row of 2026-10-22; `<m17-runs>/bench/B-7-<scene>-351.json`): 184.5 / 167.6, 214.0 / 193.6 and 151.4 / 149.7 FPS with the index on / off, each passing its floor, with one budget entry per file and `reference_fps` 450 beside it. Kitchen's and courtyard's index-off figures sit under their floors, which is why the floor is held on the index-on request alone. The break-evens came out 24, 8 and 48 against the first run's 32, 12 and 32, an observation for the owner recorded in the Calibration row, outside #351's scope; T = 32 stays.

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

**`bench convert`** (Q-169, Q-193) reruns the conversion the scene's manifest records. The converter and its options (frames, resize, the ffmpeg and ffprobe paths) come from the manifest's metadata, and the source folder from the `-i` paths of the recorded ffmpeg commands, which the Neural3DV converter records as absolute paths, so the rerun works from any directory (`docs/modules/data/converters.md`). A missing source video is exit 2, naming its path; so is a manifest that records no conversion, and one recorded by another converter than Neural3DV's (selfcap reads its videos from a work folder it removes). `--jobs` isn't recorded, so the converter's default runs, one decoding process per four cores (§2.4). Each run converts into a temporary directory in the platform's temporary folder (`TMPDIR`; kitchen at full size needs about 12 GB), removed afterwards.

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

1. check that the runner's Python imports torch with a CUDA device, gsplat, fused-ssim, the `[init]` extra's packages (einops, loguru and torchvision, which B-4's DINO networks need; Q-248, Q-253) and the checked-out openftgs;
2. generate the scene (below) into the folder `FTGS_CI_SCENE` names, or into the runner's temporary folder;
3. run B-2 to B-6 on `scene.json`, B-4 with all eight networks of D-89 (Q-253), then B-6 again on `scene-60.json`, writing `bench/B-2-raster.json`, `B-3-slice.json`, `B-4-losses.json`, `B-5-adam.json`, `B-6-data.json` and `B-6-data-60.json`;
4. keep those files as the run's artifact `bench`, whether or not a benchmark failed;
5. compare them with the baselines (below); this step's exit code is the job's verdict.

**Setting up a runner.** A Linux machine with a CUDA GPU, registered as a self-hosted runner of this repository with the extra label `gpu`, and two repository variables:

- `FTGS_CI_PYTHON`, the path of a Python whose environment holds a CUDA build of torch, gsplat 1.5.3, fused-ssim and the `[init]` extra's packages, installed as the README says (fused-ssim isn't on PyPI and builds against the installed torch). The job puts the checked-out `src` first on `PYTHONPATH`, so the commit being tested is the code measured whatever openftgs the environment holds. The six trunk networks' weights must sit in that Python's torch.hub cache (`torch.hub.get_dir()/checkpoints/`, under Q-258's names), since B-4 never downloads them and a missing one is its exit 2 (Q-253, Q-270): fetch them once, about 1.6 GB, Oxford's and BVLC's over plain HTTP and each checked against its SHA-256, by building each network, `"$FTGS_CI_PYTHON" -c 'from openftgs.losses import PerceptualLoss; [PerceptualLoss(n) for n in ["vgg16_oxford", "alexnet_bvlc", "dinov2_s14", "dinov2_b14", "dino_s16", "dino_b16"]]'` with a checkout's `src` on `PYTHONPATH` and the `[init]` extra's packages installed, the command B-4's own exit-2 message gives for the networks it finds missing, and they stay cached between runs, as LPIPS's VGG weights did.
- `FTGS_CI_SCENE`, optional: a folder outside the workspace that keeps the scene between runs. The generator completes a folder that holds the same generation instead of refusing it, and B-6's frame caches beside the manifests stay valid, so a later run skips both. The folder needs about 12 GB for the PNGs (2.0 MB each) and 28 GB for the two frame caches (23.4 GB for the 19 training cameras' 300 frames, 4.7 GB for their first 60), about 40 GB in all.

**The scene**, `scripts/ci_scene.py OUT` (Q-197), seeded, outside the wheel: 20 cameras, cam00 to cam19, evenly spaced on a horizontal arc of 90° and radius 4 at height 1 around a common point, each looking at it, with a 60° horizontal field of view and the principal point at the image centre; cam00 is held out, as Neural3DV's is. 300 frames per camera, frame k at k/30 s, as 1352×1014 8-bit RGB PNGs at `images/camXX/NNNNNN.png`, PNG level 1 as the converters write them. `scene.json` lists them all; `scene-60.json` the first 60 frames of the same PNGs, for B-6's `gpu` placement, since the full scene's 23.4 GB cache can't sit on a 24 GiB GPU (Q-178). Both manifests are written with `openftgs.io.write_scene_manifest`, and their metadata holds only the generator's parameters, so their bytes repeat on any machine.

The images are procedural, not renders: the benchmarks' costs don't depend on the pixels, except B-6's `off` placement, which decodes a PNG every iteration. So the pixels are made to compress as a converted openftgs-synth frame does: smooth colour interpolated from a coarse random grid, three oriented sinusoids, 12 discs that move over the 10 s, and Gaussian noise of 1 level: a frame takes 2.0 MB at level 1 and about 30 ms to decode, where a converted `kitchen` frame takes 1.9 to 2.0 MB and 29 to 36 ms. Each frame is drawn from its own seed (the scene's seed, the camera, the frame), so the PNGs don't depend on how many processes encode them (`--jobs`, one per CPU by default). The "1M primitives" of §3.4 are the benchmarks' own synthesized models (Q-181): B-2, B-5 and B-6 build theirs at 1M rows among other sizes, around the cameras' common point.

**The comparison**, `scripts/bench_compare.py RUN_DIR [--baselines DIR]` (Q-170, Q-196). Each BENCH.json in RUN_DIR is compared with the file of the same name in `DIR/<gpu>/`, by default `benchmarks/baselines/<gpu>/`, where `<gpu>` is the GPU model the run recorded, in lower case with hyphens (`nvidia-geforce-rtx-4090`):

| Benchmark | Compared, per case | Fails when |
| --- | --- | --- |
| B-2 | `forward_backward_s` of each `packed` × `precull` combination, at each N and active fraction | above 1.10 × the baseline's |
| B-3 | `forward_backward_s` of each implementation at each size | above 1.10 × the baseline's |
| B-4 | `forward_backward_s` of each perceptual network's case (precision, layout, autotuning) and each SSIM path at each size | above 1.10 × the baseline's |
| B-5 | `step_s` of each Adam kind at each N | above 1.10 × the baseline's |
| B-6 (both files) | `data_share` of each case with a budget entry | not under 1 %, whatever the baseline's |

A share near 0 is noise in relative terms, so B-6 keeps §3.4's absolute budget; B-2 to B-4 have no budget of their own, so their times are held to their baselines (Q-170). Cases match by their parameters exactly. The comparison also fails on a run without a baseline (B-6's included), a baseline of the run's GPU model without a run, a baseline recorded on another GPU model or of another benchmark, a scene of another size, and a case measured on one side only, so a benchmark that skipped a case is caught. It judges nothing else: not the first calls, memory, the other figures, B-5's own budget, nor the `limit` entries, which check equivalence rather than speed (the stored B-2 baseline still carries the layout entries its run wrote, some of which fail; B-2 writes none since #278). A different scene SHA-256, driver, CUDA, cuDNN, PyTorch, gsplat, fused-ssim or Python is printed as a note beside the checks. Exit 0 when every check passes, 1 when one fails, 2 for invalid input.

**The baselines** are the BENCH.json files of a reference run, kept in `benchmarks/baselines/<gpu>/` under the names the workflow writes, one folder per GPU model; a run on a model without a folder fails rather than borrowing another model's. The RTX 4090's come from the owner's machine, run with the workflow's commands at the default thread count (#235). Its `B-6-data.json` and `B-6-data-60.json` are #394's recorded run of the workflow's commands at the contract's code (12e715d; Calibration data), B-6 with B-1's configuration, now ꟻLIP's; its `B-4-losses.json` is B-4's rerun of #380's review at 1f4c451, alone on the CI scene, D-89's eight networks with the Caffe trunks' channels-last weights ("B-4 with channels-last weights" above), which replaced #394's; `B-2-raster.json`, `B-3-slice.json` and `B-5-adam.json` are #235's. On #394's run the comparison passed all 239 checks against the baselines it refreshed, where it had failed only B-4's cases against LPIPS-VGG's file; the B-4 rerun, beside #394's other five files, passes all 263 against the present ones. An iteration without LPIPS-VGG's 94 ms is shorter, 17.2 to 17.5 ms at 1M rows against 104.6, so the data wait is a larger share of it, 0.21 to 0.26 % against 0.035 to 0.044 %, and less of a host wait hides behind the GPU's backlog ("The host window" above): `off`, which has no budget, takes 63 % of an iteration, 31.8 ms of PNG decoding in 50.1 ms. B-6 is held to its absolute budget, not to its baseline's figures.

**Refreshing them**, after a deliberate slowdown, a new driver or PyTorch, a change to a benchmark's cases or a new version of the generator: run the workflow, or its commands by hand on a quiet machine, put the six BENCH.json files in one folder (the workflow's artifact `bench` is one), run `python scripts/bench_compare.py FOLDER --update`, which copies each into its GPU model's folder, and commit them with the reason. A new GPU model starts the same way: its first run fails for want of baselines, and `--update` makes its folder.

## The baselines' first runs

On the owner's RTX 4090 (Calibration data, #235), the workflow's six commands took 17 s (B-2), 4 s (B-3), 250 s (B-4), 4 s (B-5), 141 s and 237 s (B-6), about 11 minutes, with the frame caches built beforehand. The CI scene measures what kitchen does: B-2's times are within −3.8 to +1.2 % of kitchen's, and its tile intersections within 2 %. Run 1 became the baselines; run 2, right after it, passes against them, all 99 checks, 74 of its 94 medians within ±3 %. Two things sit near the rule's edges, and they go to the owner as Q-170 findings. B-4's times move by up to 8.4 % between runs (11 of its 42 cases by 5.9 to 8.4 %), close to the 10 % limit, so B-4 is the likeliest to fail CI without a regression. And run 1's compiled slice at 7·10⁵ and 10⁶ rows in B-3 was 23 and 36 % slower than run 2's, so those two baselines would let a slowdown of up to 35 and 50 % through.

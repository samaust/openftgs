# The benchmarks in `openftgs.cli`

`ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json` runs one of the eight benchmarks of §3.4 and writes its results as BENCH.json. Benchmarks measure speed and memory, never correctness: U-8 and U-28 to U-33 pin the equivalence of every optimized path, and a benchmark only says which path is faster. The code is the private module `openftgs.cli._bench` (PRD #220, owner's decision); `ftgs`'s parser takes the `bench` command from it. The choices the spec leaves open are Q-166 to Q-168, Q-178, Q-180 to Q-182, Q-185, Q-187 and Q-T232.1.

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
| `init` | B-8 | the wall time of RoMa initialization | yes | | #234 |
| `convert` | B-8 | the wall time of conversion | no | | #234 |

Each subcommand is one entry of `_BENCHMARKS`, at the end of the module: its B-n, the ticket that builds it, the function that runs it, and whether it needs CUDA and takes `--model`. A benchmark whose ticket hasn't landed has no function yet. It keeps Q-130's behaviour: it says which ticket brings it, writes nothing and exits 1, after the same checks as the others. Adding a benchmark is a function that takes the context (the parsed arguments, the scene and the model given with `--model`) and returns its setup, cases and budgets, set as `run` in its entry; the framework does the rest.

## Checks and exit codes

The checks run in this order (Q-180):

1. `--model` given to a subcommand other than `train-step` or `render` is a usage error, exit 2.
2. The scene manifest is read as `MultiViewVideo.from_manifest` reads it; a missing or invalid one is exit 2, for `convert` too.
3. Every benchmark but `convert` needs a CUDA device; without one it is exit 2, an invalid environment (Q-166).
4. MODEL is read as `ftgs render` reads it, onto the GPU: a checkpoint directory or an FTGS-IF file; an unreadable one is exit 2 (Q-167).
5. A benchmark not built yet stops here with exit 1.
6. BENCH.json's folder is made before the run, so an output that can't be written is exit 2 at once, not after minutes of measuring (Q-142).

Exit 0 means the benchmark ran, whatever its budgets and limits say: a measurement is not a command failure, and the comparison against the stored baselines decides (Q-166, Q-170). A scene the benchmark can't use is exit 2 as well: the synthesized models need more than one timestamp and two training poses (Q-181).

## `--model`

`train-step` (B-1) and `render` (B-7) take `--model MODEL`, a checkpoint directory or an FTGS-IF file, so that M17 can rerun them on the trained kitchen, courtyard and studio models; without it they time seeded synthesized models (Q-167, confirmed by the owner). §2.1's CLI line takes no model, so a spec-change request for it goes to the owner.

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
- `budgets` holds one entry per budget or limit the run is held to, naming the case and the figure: `kind` is `budget` for a §3.4 budget, `bound` for an upper bound standing for one (Q-168's B-1), and `limit` for another spec limit checked on the run, as U-28's recompilations in B-3 and U-8's 1e-6 in B-2's spot check. `pass` never changes the exit code.

## Measurement

Every benchmark measures in the same way (Q-168): a first call, then 10 warm-up calls, then the timed calls, whose medians are the figures. The first call is timed on the wall clock from an idle device to a synchronize after it, so it includes compilation and the first launches of every kernel; it is never one of the timed calls. The timed calls run under the stage clock (`openftgs._stages.Clock("cuda")`), whose CUDA events are recorded on the stream without synchronizing; the clock is read once, after the last call, so nothing synchronizes inside the timed region (§2.4 rule 3, `docs/modules/train/trainer.md`, "Stages"). The calls are issued back to back, so where the GPU's work outlasts the host's launches the events time the GPU, and where it doesn't (a small N) they time the launches, as an iteration would see them.

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

**The bound** (Q-168, `kind: "bound"`): T_it × window B's mean + the default calendar's 246 events × the event time, against 3,600 s, where the event time is what window B's event iterations (500, 600, 700) take beyond its mean, the log read included. It is an upper bound on a default run whose N stays at or below the case's: every iteration is charged at that N, and every event as one with growth. Its budget entry names the case with the largest N, among equal N the largest bound.

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

## B-1's first run

On kitchen (Calibration data, #226), an iteration without an event takes 102, 104 and 115 ms at 0.5M, 1M and 3M rows with 20 % active, and 112 ms at 3M with 5 %. LPIPS-VGG dominates: its forward takes 52.7 ms and every loss's backward 44.9 ms, whatever N. The temporal slice takes at most 1.24 % of an iteration, so D-50's CUDA operator isn't written; the SH concat at most 1.66 % and `loss.reg` 0.11 %, both under 5 %. The bound on a default run is 3,462 s at 3M rows, under the 3,600 s budget by 4 %, with cuDNN autotuning off and NCHW LPIPS as the defaults stood (B-4, #228, decides those). The clock's cost was inside the run-to-run spread: window B was 0.08 to 0.26 ms slower than window A, the iterations being GPU-bound.

## B-6: `ftgs bench data`

B-6 times the data wait, the `data` stage of a training iteration, for each placement of the frame cache (§3.4, D-51; #232, Q-T232.1 for the details). It runs B-1's configuration on B-1's synthesized model at 1M rows with 20 % active (seed 0), once per case:

| Case | `frame_cache` | `prefetch` | Budget |
| --- | --- | --- | --- |
| `{"n": 1000000, "placement": "gpu"}` | "gpu" | (takes its frames in place) | < 1 % |
| `{"n": 1000000, "placement": "mmap", "prefetch": 8}` | "mmap" | 8, the default | < 1 % |
| `{"n": 1000000, "placement": "mmap", "prefetch": 0}` | "mmap" | 0: each image read when it is taken | < 1 % |
| `{"n": 1000000, "placement": "off"}` | "off" | (decodes a PNG every iteration) | none |
| `{"n": 3000000, "placement": "gpu"}` | "gpu" | | none: Q-99's note |

**Per case**, a fresh trainer runs iterations 1 to 120 untimed (B-1's warm-up, the event at 100 included), 121 to 320 under the stage clock's CUDA events, read once after 320, and 321 to 420 under the wall clock. The frame cache is built before the first case, so a case only opens it.

**The figure**, `data_share`, is the `data` stage's mean over the `iteration` stage's mean over the 198 timed iterations without an event, a ratio of means as B-1's shares; the events at 200 and 300 are left out, which can only make the share larger. Its budget entry (`kind: "budget"`) holds it under 1 % for every case that reads the cache at 1M rows. Beside it: the data stage's mean and median, the mean iteration, and peak allocated and reserved memory from the first iteration on.

**The host window.** The CUDA events see a wait only when the training stream stops for it. A wait on the host, as decoding a PNG, reading a memory-mapped frame or, at prefetch 0, waiting on a pinned buffer's previous copy (`docs/modules/train/trainer.md`), happens while the GPU still works through the previous iteration's losses and backward pass, about 100 ms of work on kitchen (B-1) queued after gsplat's read-back in the render, so the events can't see it unless it outlasts that backlog. The wall window gives the host time of the data stage and of an iteration over its 99 iterations without an event, to say how much of that backlog a wait uses. It is a diagnostic, not the B-6 figure.

**The GPU placement** is measured only where "auto" would choose it (Q-99, Q-178). Before each `gpu` case a probe trainer of the same configuration decodes PNGs for one iteration, the path "auto" takes on its first CUDA iteration, and then the free memory (`mem_get_info`'s, plus what the allocator holds unused), the growth to N_max and the cache's size are read as the trainer's rule reads them. They are read once `step()` has returned, while the rule reads them inside it with the iteration's render output, loss and image still held, so the probe's free memory is larger by those tensors, under 1 % of it. A cache above ½·free − growth is skipped: `setup.skipped` lists the case with its reason and that estimate, and it gets no case or budget. On kitchen, whose cache is 23.4 GB, both `gpu` cases are skipped, so `gpu` is measured on its 60-frame conversion (Q-178), where every case runs. A `gpu` case's figures add the estimate (`auto_free_bytes`, `auto_growth_bytes`, `auto_allocated_bytes`, `auto_reserved_bytes`, `cache_bytes`, `auto_limit_bytes`), the device's total memory (`auto_total_bytes`), what lies outside this process's allocator (`auto_outside_bytes`, total − free − allocated: other processes, such as a desktop session, and the CUDA context), and Q-99's margin:

- `beyond_estimate_bytes`: the run's peak reserved memory, less what was allocated after the probe's iteration, the cache and the growth: what the run took beyond the rule's estimate;
- `margin_used`: that over ½·free, the share of the margin the rule leaves for it. At most 1 means a cache at the rule's limit would still have fit, judged from reserved memory, so conservatively; above 1, the ½ margin is too small, which goes to the owner as a decision change.

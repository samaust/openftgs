# The benchmarks in `openftgs.cli`

`ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json` runs one of the eight benchmarks of §3.4 and writes its results as BENCH.json. Benchmarks measure speed and memory, never correctness: U-8 and U-28 to U-33 pin the equivalence of every optimized path, and a benchmark only says which path is faster. The code is the private module `openftgs.cli._bench` (PRD #220, owner's decision); `ftgs`'s parser takes the `bench` command from it. The choices the spec leaves open are Q-166 to Q-168, Q-180 to Q-182, Q-185 and Q-T228.1.

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
- `budgets` holds one entry per budget or limit the run is held to, naming the case and the figure: `kind` is `budget` for a §3.4 budget, `bound` for an upper bound standing for one (Q-168's B-1), and `limit` for another spec limit checked on the run, as U-28's recompilations in B-3. `pass` never changes the exit code.

## Measurement

Every benchmark measures in the same way (Q-168): a first call, then 10 warm-up calls, then the timed calls, whose medians are the figures. The first call is timed on the wall clock from an idle device to a synchronize after it, so it includes compilation and the first launches of every kernel; it is never one of the timed calls. The timed calls run under the stage clock (`openftgs._stages.Clock("cuda")`), whose CUDA events are recorded on the stream without synchronizing; the clock is read once, after the last call, so nothing synchronizes inside the timed region (§2.4 rule 3, `docs/modules/train/trainer.md`, "Stages"). The calls are issued back to back, so where the GPU's work outlasts the host's launches the events time the GPU, and where it doesn't (a small N) they time the launches, as an iteration would see them.

B-1 times training iterations rather than calls: its warm-up is the first 120 iterations, and its figures are means as well as medians, since an iteration's time is the sum of its stages (below).

The stages are also NVTX ranges, so Nsight Systems shows a benchmark's calls by name. B-3 uses two of the fixed stage names:

| Stage | In B-3 |
| --- | --- |
| `slice.kernel` | the forward call of `motion.temporal_slice` |
| `backward` | `torch.autograd.grad` from the slice's outputs to its inputs: the slice's backward alone, where a training iteration's `backward` holds every backward pass |

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

## B-3: `ftgs bench slice`

B-3 times `motion.temporal_slice`'s eager and compiled implementations, forward and backward, at 11 sizes: 10⁵, 1.5·10⁵, 2·10⁵, 3·10⁵, 4·10⁵, 5·10⁵, 7·10⁵, 10⁶, 1.5·10⁶, 2·10⁶ and 3·10⁶ rows, B-1's 0.5M, 1M and 3M among them (Q-182). Its inputs are the first n rows of one synthesized model of 3·10⁶ rows (seed 0, 20 % active), so each of the 10 resizes keeps the rows before it in order and appends new ones, as a growth event does. At each size, eager and then compiled, each given explicitly: "auto" could fall back to eager and time it twice (Q-13). Each call slices at a seeded time in [0, 1], then runs `torch.autograd.grad` from the five outputs a training iteration differentiates (`means_t`, `opacities_t`, `quats_n`, `scales` and `reg_terms`) with fixed upstream gradients, the same for both implementations.

Per case (`{"impl", "n"}`), the figures are the medians over 50 calls of the forward (`forward_s`), the backward (`backward_s`) and the span from the forward's start to the backward's end (`forward_backward_s`), the first call (`first_call_s`), the peak allocated memory, and for the compiled slice dynamo's unique graphs so far. Dynamo is reset before the first size, so the first compiled call compiles whatever ran before in the process; inductor's and Triton's on-disk caches stay as PyTorch configures them, so a machine's first run compiles from scratch and later ones reuse kernels. A last case (`{"impl": "compiled", "resizes": 10}`) gives the recompilations over the resizes, dynamo's unique graphs less one (Q-13), and its budget entry sets them against U-28's limit of 2.

**Decision rule** (Q-182), applied by #226 together with B-1: Q-13's "auto" stays the compiled slice on CUDA while its median forward plus backward is no slower than eager's at every size and its recompilations stay within U-28's limit. Otherwise the numbers go to the owner as a request to change "auto"'s mapping, which Q-13 fixes, and the code keeps it until the owner answers. Whether D-50's CUDA operator is written is B-1's rule, the slice's share of an iteration (PRD #220).

The first run, on kitchen, is in the Calibration data table of `docs/clean-room/questions.md`: the compiled slice is no slower than eager only up to 3·10⁵ rows, where both sit at the host's floor of about 0.3 ms, and 8 to 49 % slower from 4·10⁵ rows on (10 % at 3·10⁶, where it takes 2.46 ms against 2.24). It never recompiled over the 10 resizes. Its first call took 0.53 s with warm compile caches and 2.25 s with empty ones.

That run is superseded. Under `dynamic=True` dynamo had made the 3 and the 4 of the inner dims symbolic, and inductor computed the quaternion's norm in reduction kernels; since #226 the compiled slice's inner dims are static (Q-184, `docs/modules/motion/temporal-slice.md`). Rerun on kitchen, the compiled slice is no slower than eager at every size: 2 to 5 % faster up to 5·10⁵ rows, at the host's floor, 12 % at 7·10⁵, and 41 to 49 % faster from 10⁶ rows on (1.24 against 2.29 ms at 3·10⁶), still with no recompilation. So Q-182's rule keeps "auto" compiled on CUDA.

## B-4: `ftgs bench losses`

B-4 times the two image losses whose cost D-52 decides, LPIPS-VGG and SSIM, at 1352×1014 and 1920×1080 (§3.4; Q-T228.1 for the details below). It sets D-52's exact defaults through Q-173's rule and records how far bf16 strays from float32. It has no budget of its own.

**Images.** The target is the scene's first training image, resized bilinearly with antialiasing when its size differs (on kitchen, 1920×1080 stretches its 4:3 to 16:9); the prediction adds seeded Gaussian noise of σ 0.05. Natural content gives the deviation figures the feature statistics of training; timing doesn't depend on it.

**LPIPS-VGG.** lpips's network is called directly (PRD #220), frozen and in evaluation mode, its inputs mapped to [−1, 1] as `LPIPSLoss` maps them (D-68). One call is the forward on both images and the mean (stage `loss.perceptual`), then the prediction's gradient (stage `backward`), for each of:

| Axis | Values |
| --- | --- |
| precision | `ieee` (cuDNN's TF32 flag off), `tf32` (on, training's default), `bf16` (bfloat16 autocast around the network, TF32 on, the value cast to float32; Q-172) |
| layout | `nchw` (contiguous NCHW inputs and weights); `channels_last_inputs` (channels-last inputs, NCHW weights); `channels_last_inputs_and_weights` |
| cuDNN autotuning | off, on |

`channels_last_inputs` is training's path: `LPIPSLoss` views a render's [..., H, W, 3] batch first, which is channels-last in memory, and every convolution of the network follows that layout (source log T228.2). So "channels-last on and off" (§3.4) is measured twice: on and off for the inputs, which training already has on, and for the weights, which it doesn't yet.

**Processes.** cuDNN caches the algorithm it picked for a configuration and looks there first whatever `cudnn.benchmark` says, so autotuning off and on can't be compared in one process: a later setting would run the earlier one's picks. The deviations (below) and fused-ssim run in the benchmark's process; every LPIPS case and the PyTorch SSIM path run in one child process per autotuning setting, off and then on.

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

## B-1's first run

On kitchen (Calibration data, #226), an iteration without an event takes 102, 104 and 115 ms at 0.5M, 1M and 3M rows with 20 % active, and 112 ms at 3M with 5 %. LPIPS-VGG dominates: its forward takes 52.7 ms and every loss's backward 44.9 ms, whatever N. The temporal slice takes at most 1.24 % of an iteration, so D-50's CUDA operator isn't written; the SH concat at most 1.66 % and `loss.reg` 0.11 %, both under 5 %. The bound on a default run is 3,462 s at 3M rows, under the 3,600 s budget by 4 %, with cuDNN autotuning off and NCHW LPIPS as the defaults stood (B-4, #228, decides those). The clock's cost was inside the run-to-run spread: window B was 0.08 to 0.26 ms slower than window A, the iterations being GPU-bound.

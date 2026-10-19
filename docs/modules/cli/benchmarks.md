# The benchmarks in `openftgs.cli`

`ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json` runs one of the eight benchmarks of §3.4 and writes its results as BENCH.json. Benchmarks measure speed and memory, never correctness: U-8 and U-28 to U-33 pin the equivalence of every optimized path, and a benchmark only says which path is faster. The code is the private module `openftgs.cli._bench` (PRD #220, owner's decision); `ftgs`'s parser takes the `bench` command from it. The choices the spec leaves open are Q-166 to Q-168, Q-180 to Q-182, Q-T231.1 and Q-T231.2.

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

Exit 0 means the benchmark ran, whatever its budgets and limits say: a measurement is not a command failure, and the comparison against the stored baselines decides (Q-166, Q-170). A scene the benchmark can't use is exit 2 as well: the synthesized models need more than one timestamp and two training poses (Q-181). So is, for `render`, a model in another time unit than the scene's, and LPIPS networks whose weights can't be had, which B-7's evaluation figures need; it builds them before measuring anything, as `ftgs eval` builds them (Q-142, Q-T231.1).

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

Every benchmark measures in the same way (Q-168): a first call, then 10 warm-up calls, then the timed calls, whose medians are the figures. The first call is timed on the wall clock from an idle device to a synchronize after it, so it includes compilation and the first launches of every kernel; it is never one of the timed calls. The timed calls run under the stage clock (`openftgs._stages.Clock("cuda")`), whose CUDA events are recorded on the stream without synchronizing; the clock is read once, after the last call, so nothing synchronizes inside the timed region (§2.4 rule 3, `docs/modules/train/trainer.md`, "Stages"). The calls are issued back to back, so where the GPU's work outlasts the host's launches the events time the GPU, and where it doesn't (a small N) they time the launches, as an iteration would see them. B-7 is the exception: its call is a whole render request, which synchronizes inside every render, so each call is timed on the wall clock from a synchronized device to a synchronize after it (Q-T231.1).

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

## B-3: `ftgs bench slice`

B-3 times `motion.temporal_slice`'s eager and compiled implementations, forward and backward, at 11 sizes: 10⁵, 1.5·10⁵, 2·10⁵, 3·10⁵, 4·10⁵, 5·10⁵, 7·10⁵, 10⁶, 1.5·10⁶, 2·10⁶ and 3·10⁶ rows, B-1's 0.5M, 1M and 3M among them (Q-182). Its inputs are the first n rows of one synthesized model of 3·10⁶ rows (seed 0, 20 % active), so each of the 10 resizes keeps the rows before it in order and appends new ones, as a growth event does. At each size, eager and then compiled, each given explicitly: "auto" could fall back to eager and time it twice (Q-13). Each call slices at a seeded time in [0, 1], then runs `torch.autograd.grad` from the five outputs a training iteration differentiates (`means_t`, `opacities_t`, `quats_n`, `scales` and `reg_terms`) with fixed upstream gradients, the same for both implementations.

Per case (`{"impl", "n"}`), the figures are the medians over 50 calls of the forward (`forward_s`), the backward (`backward_s`) and the span from the forward's start to the backward's end (`forward_backward_s`), the first call (`first_call_s`), the peak allocated memory, and for the compiled slice dynamo's unique graphs so far. Dynamo is reset before the first size, so the first compiled call compiles whatever ran before in the process; inductor's and Triton's on-disk caches stay as PyTorch configures them, so a machine's first run compiles from scratch and later ones reuse kernels. A last case (`{"impl": "compiled", "resizes": 10}`) gives the recompilations over the resizes, dynamo's unique graphs less one (Q-13), and its budget entry sets them against U-28's limit of 2.

**Decision rule** (Q-182), applied by #226 together with B-1: Q-13's "auto" stays the compiled slice on CUDA while its median forward plus backward is no slower than eager's at every size and its recompilations stay within U-28's limit. Otherwise the numbers go to the owner as a request to change "auto"'s mapping, which Q-13 fixes, and the code keeps it until the owner answers. Whether D-50's CUDA operator is written is B-1's rule, the slice's share of an iteration (PRD #220).

The first run, on kitchen, is in the Calibration data table of `docs/clean-room/questions.md`: the compiled slice is no slower than eager only up to 3·10⁵ rows, where both sit at the host's floor of about 0.3 ms, and 8 to 49 % slower from 4·10⁵ rows on (10 % at 3·10⁶, where it takes 2.46 ms against 2.24). It never recompiled over the 10 resizes. Its first call took 0.53 s with warm compile caches and 2.25 s with empty ones.

## B-7: `ftgs bench render`

B-7 times `render.render_request`, the path `ftgs render`, `ftgs eval` and the trainer's evaluation share (Q-57), with the temporal index of D-55 on and off (`docs/modules/render/backends.md`, "The temporal index"), and from its measurements sets T, the number of distinct times from which `render_request`'s default builds the index (Q-174). Its setup is Q-T231.1's, its rule for T Q-T231.2's.

- **Models.** Synthesized as above (seed 0) at N = 10⁶ and 3·10⁶, with 1 %, 5 % and 20 % of the rows active, each with its default `slice_impl` ("auto"); or the model given with `--model` alone. On kitchen's one-frame buckets (Δt_f = 1/299 of the span) a row of the 1 % model spans about 4 buckets, while the rows of the 5 % and 20 % models span 17 and 76 and so sit on the long list, which every render takes: B-7 sees the index both where it can help and where it can only cost.
- **Camera.** Q-168's: the pose of the first camera in the manifest's camera order that has test images (kitchen's cam00), else of the first camera; fx and fy scaled by 1080/H, H its height, so pixels stay square and the vertical field of view is kept; the principal point at (960, 540); 1920×1080.
- **Requests.** One item per distinct training timestamp (F of them, 300 on kitchen), the camera's tensors on the host as a request file gives them, a black background and the model's SH degree. The buckets' width (`frame_dt`) is a checkpoint's Δt_f, otherwise the scene's, one training frame interval as D-55 has it.
- **Calls.** One call is one `render_request` with the gsplat backend, the index forced on or off, and the outputs on the GPU (`device="cuda"`) or on the CPU: a first call, 10 warm-up calls and 10 timed calls on the wall clock, whose medians give the request's time (`request_s`), FPS (renders over it) and the time per render (`render_s`), beside the first call and the peak allocated memory.
- **Cases** (`{"n", "active", "times", "index", "outputs"}`). At the F times, the index on and off with the outputs on the GPU and on the CPU; with the outputs on the GPU, the index on and off at requests of k times spread evenly over the F, k = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 and 128 below F.
- **The index** (`{"n", "active"}`). Built as `render_request` builds it, from `render`'s private index class: its build time (`build_s`, the first call apart) on the wall clock, and at the F times the means of the rows a render is passed (`passed_share_mean`: the bucket's and the long list), of the bucket's own rows and of the rows that pass the opacity cull, and the rows on the long list (`long_share`), each over N; with the model's break-even, below.
- **Evaluation** (`{"n", "active", "evaluation"}`). `evaluate_images` on the test split (the training split when there is none) timed once on the wall clock, against the render requests it sends, one per timestamp with the outputs on the CPU as it takes them, also timed once; `render_share` is the second over the first.
- **Budget.** §3.4's ≥ 450 FPS, kind `budget`, on the requests at the F times with the outputs on the GPU, index on and off. It is calibrated: M17 measures the trained kitchen, courtyard and studio models with `--model`.

**The rule for T** (Q-T231.2). A model's break-even is the smallest request size, among the swept sizes and F, at which the request with the index takes no longer than without it, there and at every larger size, outputs on the GPU; a model has none when the index is slower at F. T is the largest break-even over the models that have one, and at least 2, so a request at a single time never builds an index; B-7 writes it as the case `{"threshold": "T"}` (`distinct_times`), beside the T in force (`setup.threshold_in_force`). The recorded run's T becomes `render._INDEX_MIN_TIMES`, cited by its Calibration row. When no model has a break-even, T stays as it is and the numbers go to the owner, since an "auto" that never pays off is a change to Q-174's switch rather than a value of T.

**The rule for evaluation** (PRD #220). `evaluate_images` keeps one render request per timestamp (Q-68) while `render_share` is at most 10 % at every model; above that, the numbers go to the owner as a request to change Q-68.

# The benchmarks in `openftgs.cli`

`ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json` runs one of the eight benchmarks of §3.4 and writes its results as BENCH.json. Benchmarks measure speed and memory, never correctness: U-8 and U-28 to U-33 pin the equivalence of every optimized path, and a benchmark only says which path is faster. The code is the private module `openftgs.cli._bench` (PRD #220, owner's decision); `ftgs`'s parser takes the `bench` command from it. The choices the spec leaves open are Q-166 to Q-169, Q-180 to Q-182, Q-T234.1 and Q-T234.2.

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

Each subcommand is one entry of `_BENCHMARKS`, at the end of the module: its B-n, the ticket that builds it, the function that runs it, and whether it needs CUDA and takes `--model`. A benchmark whose ticket hasn't landed has no function yet. It keeps Q-130's behaviour: it says which ticket brings it, writes nothing and exits 1, after the same checks as the others. Adding a benchmark is a function that takes the context (the parsed arguments, the scene and the model given with `--model`) and returns its setup, cases and budgets, set as `run` in its entry; the framework does the rest.

## Checks and exit codes

The checks run in this order (Q-180):

1. `--model` given to a subcommand other than `train-step` or `render` is a usage error, exit 2.
2. The scene manifest is read as `MultiViewVideo.from_manifest` reads it; a missing or invalid one is exit 2, for `convert` too.
3. Every benchmark but `convert` needs a CUDA device; without one it is exit 2, an invalid environment (Q-166).
4. MODEL is read as `ftgs render` reads it, onto the GPU: a checkpoint directory or an FTGS-IF file; an unreadable one is exit 2 (Q-167).
5. A benchmark not built yet stops here with exit 1.
6. BENCH.json's folder is made before the run, so an output that can't be written is exit 2 at once, not after minutes of measuring (Q-142).

Exit 0 means the benchmark ran, whatever its budgets and limits say: a measurement is not a command failure, and the comparison against the stored baselines decides (Q-166, Q-170). A scene the benchmark can't use is exit 2 as well: the synthesized models need more than one timestamp and two training poses (Q-181); `init` needs the `[init]` extra and RoMa's weights in torch.hub's cache (Q-T234.1); `convert` needs a manifest that records a Neural3DV conversion whose source videos are still where its ffmpeg commands read them (Q-169, Q-T234.2).

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

Every benchmark but B-8 measures in the same way (Q-168): a first call, then 10 warm-up calls, then the timed calls, whose medians are the figures. The first call is timed on the wall clock from an idle device to a synchronize after it, so it includes compilation and the first launches of every kernel; it is never one of the timed calls. The timed calls run under the stage clock (`openftgs._stages.Clock("cuda")`), whose CUDA events are recorded on the stream without synchronizing; the clock is read once, after the last call, so nothing synchronizes inside the timed region (§2.4 rule 3, `docs/modules/train/trainer.md`, "Stages"). The calls are issued back to back, so where the GPU's work outlasts the host's launches the events time the GPU, and where it doesn't (a small N) they time the launches, as an iteration would see them.

B-8 times work that runs once, initialization and conversion, so it runs it once, without a warm-up, under the wall clock (`Clock("wall")`), and its figures are sums and spreads of the stages it recorded (below).

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

## B-8: `ftgs bench init` and `ftgs bench convert`

B-8 reports the wall time of the two steps that run once per scene, initialization and conversion, on one Neural3DV-layout scene, openftgs-synth's `kitchen` in this project (D-79). It has no budget, only the regression rule, so its BENCH.json has no budget entries. Both run once, under the wall clock, and read the stages that `roma_init` and the converters record (`docs/modules/train/trainer.md`, "Stages").

**`bench init`** (Q-T234.1) runs `roma_init` on the scene with the default `RomaInitConfig`, D-58's roma_outdoor, as `ftgs init roma` runs without options, on every visible GPU (Q-106). It needs the `[init]` extra and the two checkpoints roma_outdoor loads, RoMa's and DINOv2's, already in torch.hub's cache (`torch.hub.get_dir()/checkpoints/`). Without either it is exit 2, naming what is missing, before anything runs: it never downloads them, so no download is timed. One `ftgs init roma` run fetches them. The setup holds Q-107's provenance record (every setting, `use_custom_corr`, the vendored RoMa's commit), the devices and the weight files.

| Case | Figures |
| --- | --- |
| `{"part": "init.match", "device"}` | per GPU: `pairs`, and the time per pair as `total_s`, `median_s`, `min_s`, `max_s`. A pair's stage holds its two image reads, the matcher, the draw of its matches (D-22) and its triangulation |
| `{"part": "init.triangulate", "device"}` | per GPU: the same, for the triangulation alone |
| `{"part": "init.knn"}` | `frames`, one E15 search per init frame in the calling thread, with the same spread |
| `{"part": "roma_init"}` | `wall_s`, the whole call; `startup_s`, from its start to the first pair's, which is loading the weights and building one model per GPU; `pairs`, `init_frames` and `points` |

The k-NN searches overlap the matching of the next frame's pairs, and with several GPUs the pairs of one GPU overlap the others', so the parts don't add up to the whole.

**`bench convert`** (Q-169, Q-T234.2) reruns the conversion the scene's manifest records. The converter and its options (frames, resize, the ffmpeg and ffprobe paths) come from the manifest's metadata, and the source folder from the `-i` paths of the recorded ffmpeg commands. A missing source video is exit 2, naming its path; so is a manifest that records no conversion, and one recorded by another converter than Neural3DV's (selfcap reads its videos from a work folder it removes). `--jobs` isn't recorded, so the converter's default runs, one decoding process per four cores (§2.4). Each run converts into a temporary directory in the platform's temporary folder (`TMPDIR`; kitchen at full size needs about 12 GB), removed afterwards.

It runs once with `--hwaccel none`, then once with `--hwaccel cuda` when PyTorch sees a CUDA device and the recorded ffmpeg lists `cuda` among its hardware acceleration methods (`ffmpeg -hwaccels`). A `cuda` run that fails, such as a pixel format D-81 refuses, leaves the CPU run's numbers standing: `setup.skipped.cuda` says why, and the exit stays 0. `setup.hwaccel` lists the runs made.

| Case | Figures |
| --- | --- |
| `{"hwaccel", "camera"}` | `frames`; `wall_s`, from the camera's first frame read to its last frame written; `decode_s`, the sum of its frames' reads from ffmpeg (`convert.decode`), which wait for ffmpeg's decoding; `encode_s`, the sum of its frames' processing and PNG writing in the process pool (`convert.encode`), measured in the pool's processes, so it is work, not waiting, and cameras overlap |
| `{"hwaccel"}` | `cameras`, `frames`; `wall_s`, the converter's call, probing the videos included; the sums of `decode_s` and `encode_s` |

The setup also records the ffmpeg's version line, the CPU count and the temporary folder, since the conversion's time depends on them.

# The trainer in `openftgs.train`

How `Trainer` meets its §2.1 row: the training iteration of §1.3, the calendar, the step log and callbacks, evaluation, and the trainer state that a checkpoint holds. The public names beyond §2.1 are provisional decisions (Q-69 to Q-74).

## One training iteration

`step()` runs §1.3's seven steps in order at iteration it = `iteration` + 1:

1. The sampler gives the next training image (D-26), from the frame cache or, without one, decoded from its file (D-51).
2. Its normalized time t and the active SH degree come from the time normalizer (E14) and the calendar entry (E13).
3. `render` builds the per-time arrays and renders, with the 2D means' gradient retained (Q-32). `PerfConfig.precull` and `packed` pass through (Q-31, Q-36).
4. `total_loss` evaluates E9 with the perceptual network built once, when the trainer is built, if λ_perc is None or positive (D-77; Q-69, Q-249). None stands for the network's calibrated weight in D-89's table, which `total_loss` resolves; at 0 no network is built and the step has no perceptual term.
5. The backward pass, then the gradient statistics: by `gaussian_ids` in packed mode, by `visible` otherwise, never by the flag (Q-36). They are skipped when the 2D means got no gradient, which is a check for None, not a synchronization.
6. The optimizer takes this iteration's rates (E12, E13) and one Adam step, then the gradients are cleared.
7. On an event, in order: the divergence check, growth, relocation, pruning, the opacity reset, and the reset of the gradient statistics where relocation didn't run, since relocation always resets them itself (D-45, Q-51).

The backend follows the device: gsplat on a CUDA device, which D-76 requires of it, and the reference backend elsewhere (Q-69).

## Stages

`step()` runs inside the range `iteration`, which B-1 times as the whole iteration, and its stages run inside that range, in order (§3.4; the fixed list of names is `openftgs._stages.STAGES`):

- `data`: taking the image, the data wait of B-6;
- `slice`, in `at_time`: `slice.kernel` around the temporal slice's call (never inside the compiled function, D-50), then `slice.sh_concat`;
- `raster.forward`, in `render` after the slice, compaction included; on gsplat it holds `raster.forward.projection`, `raster.forward.sh`, `raster.forward.isect` and `raster.forward.rasterize`;
- `loss.l1`, `loss.ssim`, `loss.perceptual` and `loss.reg`, in `total_loss`, each only when its λ is above 0;
- `backward`, `statistics`, `adam`, `events` and `log`.

Each stage is an NVTX range whenever PyTorch has NVTX, a profiler range inside a `--profile` window, and, while a stage clock is active, a pair of CUDA events (B-1, B-6) or wall times. Two gradient hooks, registered only while a clock is active, split `backward` into three parts: `backward.losses_done` fires when the rendered image's gradient is complete, after every loss's backward; `backward.raster_done` when the gradient of the moved means, taken right after the slice and before compaction, is complete, so projection and SH are done; the rest, until `backward()` returns, is the slice's backward. Autograd interleaves the losses' backward passes, so a single loss's backward can't be timed apart; B-4 times each loss alone. The hooks run on the autograd engine's thread, and the clock records from any thread, never synchronizes (U-29's stage-clock case), and is read where the run synchronizes anyway. Initialization and conversion have their own stages for B-8, on the wall clock: `init.match` per camera pair in its shard's thread, with `init.triangulate` inside it, `init.knn`, and per frame `convert.decode` and `convert.encode`, keyed by camera.

## The calendar

`calendar_entry(cfg, it)` is a pure function of the configuration and the iteration, so U-17 checks a 30,000-iteration calendar without training (Q-71). Every window and period comes from the configuration, so a shortened calendar (§3.2) needs no other change. Growth and relocation share the event period (`RelocConfig.every`) and have their own windows. In the "w/o periodic relocation" ablation, selected by `RelocConfig.prune_opacity`, pruning runs at every growth event and the opacity reset at every multiple of `density.OPACITY_RESET_EVERY` inside the growth window (owner answer, Q-71; Q-67). The ablation turns relocation off, so the trainer refuses a pruning opacity with relocation on (Q-71).

## The step log and the log window

A step returns a `StepLog`. Its loss terms are the detached tensors `total_loss` returns, on the device and unread, so a step makes no host synchronization outside gsplat's render, the events and the log reads (§2.4 rule 3, D-49). The trainer adds the terms into device-side sums and reads their means once every `PerfConfig.log_every` iterations; the same read checks the model's validity. After a resume the first window covers only the iterations since the resume; logs never feed back into training.

Every training record's camera is copied to the device when the trainer is built, so a training iteration's render reads no camera from the host. Unless `frame_cache` is "off", each training iteration's image comes from the frame cache: on the GPU directly, or through the prefetcher, which copies it from pinned memory on a side stream ahead of time (D-51, `docs/modules/data/frame-cache.md`). Without a cache, which includes the iterations before "auto" decides on CUDA, the decoded image is staged in pinned memory and copied to the GPU without blocking; PyTorch's pinned allocator holds the staging block until the copy is done. A copy from pageable memory would block the host (§2.4 rule 3). The bytes are copied as they are, so the image is bitwise the decoded one (U-30).

## Where a training iteration synchronizes

§2.4 rule 3 allows a training iteration host–device synchronizations only inside gsplat's stage functions and, with compaction, at the selection of the kept rows (D-57); events and log reads synchronize anyway, at their own iterations. U-29 checks the iterations between them (`tests/test_synchronization.py`), and Q-171 says which frames count as inside gsplat's render call. On CUDA a training iteration synchronizes at:

| Where | Packed mode | Unpacked mode |
| --- | --- | --- |
| gsplat's tile intersection (`isect_tiles`), which reads back its number of intersections | once | once |
| gsplat's projection, which reads back its number of visible primitive–camera pairs | once | — |
| with compaction, `render`'s `_kept_rows`, which reads back the number of rows kept (Q-31) | once | once |

None of the other settings adds one: fused, foreach or selective Adam, the eager or compiled slice, any cache placement and prefetch depth, the perceptual loss with any of D-89's eight networks (bfloat16 included, for a trunk network), either SSIM path, deterministic mode, an active stage clock (U-29's cases). ꟻLIP copies its filters and constants to the device and reads its colour maximum back once, at its first call there, which U-29's warm-up iterations hold. Everything else an iteration does stays on the device or in host values that need no read-back:

- the image comes as the step-log section above says, without blocking the host;
- cameras and the background are on the device from construction;
- the loss terms are detached device tensors, summed on the device until the log read (D-77);
- the gradient statistics are masked sums or scatter-adds on the device (U-16);
- Adam's rates are Python floats; fused Adam keeps its step counters on the device, foreach and selective Adam on the CPU, where advancing them reads nothing from the GPU (Q-45).

One wait blocks a host thread without the debug mode reporting it: before reusing a pinned buffer of its ring, the prefetcher's worker thread waits on the event of that buffer's previous copy, an earlier image's. It never happens on the training thread: a frame read there (every frame when `prefetch` is 0, and a frame taken out of the scheduled order) is copied through a fresh block of PyTorch's caching pinned allocator, which reuses a block only once its copy's event has completed, checked without waiting (`docs/modules/data/frame-cache.md`; PRD #220's review). And once per run, after the second "auto" iteration on CUDA, choosing and opening the cache placement reads the free GPU memory and, for the GPU placement, loads the whole cache onto the device (Q-99).

How U-29 sees a synchronization: the debug mode's C++ warning becomes a Python warning only during a call through one of torch's Python bindings, from the frame that made the call. gsplat's projection runs inside an autograd Function, so its warning arrives that way; the tile intersection is a direct call into gsplat's compiled module, so its warning is only printed on standard error, and U-29's recorder charges it to the Python stack that made the call. A C++ synchronization in a backward pass would arrive from torch's own backward frame when backward returns, which U-29 refuses whoever made it; none occurs.

## Divergence

A non-finite mean loss at a log read, or a model that fails `validate()` or gradient statistics that aren't finite, at a log read or before an event, make `step` raise `FloatingPointError` naming the iteration (Q-69). It is a built-in exception, apart from the `ValueError` of input errors (Q-26), so M14 can map it to exit code 3 (Q-55).

## Callbacks and stopping

`train(until, callbacks)` calls every callback after each step, in order, and stops after that step when any returns True. Stopping happens only between iterations, so a state taken then is always whole; `ftgs train`'s SIGINT and SIGTERM handlers use this to save (§2.2, `docs/modules/cli/command-line.md`).

## Generators

All randomness in training comes from the trainer's own CPU generators, each seeded from `TrainConfig.seed` and its role (Q-72):

- the sampler's, for the per-epoch permutations;
- the relocation generator, for every draw of an event in a fixed order: growth's split offsets (E17), relocation's targets and positions (D-15), and the score mode's draws;
- an initialization generator, used once for the seeded subset of `from_init_points` and not checkpointed.

The global generators are neither seeded nor drawn from. Building a perceptual network can draw from torch's global CPU generator, since the vendored DINOv2 builders initialize weights before the publisher's are loaded, so the trainer builds it under `torch.random.fork_rng`, which restores that generator's state (Q-72). The trainer state records and restores them, so code outside the trainer, such as a callback, also continues on a resume. Because they are never seeded, two runs in one process see different global states, and U-24 would catch any hidden draw from them.

## Exact resume

A run resumed from a checkpoint at iteration k is bitwise equal to the uninterrupted run on the reference backend (§2.2):

- every draw comes from a checkpointed generator, and the sampler stores its epoch, its cursor and the current permutation, so the next permutation comes from the same point in the generator's stream;
- rates, the active SH degree, the calendar and the regularizer's cutoff are pure functions of it and the configuration;
- the stored tensors round-trip exactly through safetensors, and τ₀, τ₁, S and Δt_f through JSON; the optimizer is rebuilt from the stored S and Δt_f, so its base rates are the same floats (Q-46);
- states are taken and training stops only between iterations, and gradients are cleared every step;
- on the CPU the reference backend, the SSIM fallback, Adam and the frozen perceptual network are deterministic for a fixed thread count; U-21 and U-24 hold this with ꟻLIP at its calibrated weight too, and U-24 with HaarPSI. A resumed trainer builds its own network, which holds no state of its own, and the trainer state never holds the network's weights (§2.3: the trunks' weights are never redistributed).

On gsplat the backward pass accumulates with atomic additions, so a resume holds up to floating-point reordering only, as §2.2 says.

`load_state_dict` builds the new model, optimizer and statistics, and checks the recorded global generator states on throwaway generators, before replacing anything, so a rejected state leaves the trainer and the process's generators as they were. It requires the trainer's own configuration, apart from the performance settings that change no number, and the dataset's time unit, and adopts the state's time normalizer, S and Δt_f. `from_checkpoint` also refuses a checkpoint made from another scene manifest unless forced (owner answer, Q-70); a forced resume keeps the run's time normalizer, S and Δt_f, so its learning rates continue unchanged.

**A checkpoint saved before revision 549o** (Q-247). M17's checkpoints store `loss.perc_net` "vgg" or "alex", LPIPS's backbones, which `config_from_dict` refuses since revision 549o (#375). Such a checkpoint can't be resumed: `from_checkpoint` checks the stored configuration first, before the scene manifest and anything else, and refuses it with a ValueError naming 549o, which `ftgs train --resume` reports as exit code 2. Its run trained with a network openftgs no longer has, so no resume could continue it. The readers that take only its model read it, with 549o's defaults (`flip` at λ_perc None) in place of its `perc_net` and `lambda_perc`, in memory only: `io.load_checkpoint`, `FreeTimeGaussians.load`, and `ftgs eval`, `render`, `export` and `bench render --model` (`docs/modules/cli/command-line.md`; Q-271 for the scripts). `TrainerState.config` keeps the dict as saved, and nothing is written back.

A resume may change the performance settings that change no number: `slice_impl`, `packed`, `precull`, `frame_cache`, `prefetch`, `log_every` and `deterministic` (§2.4 rule 2; U-8, U-28, U-30). `perc_precision`, `adam` and `images_per_step` change results (D-52 to D-54) and must match the checkpoint's, as every field outside `PerfConfig` must. `config.resume_conflicts` names the fields that differ where they must match, and both `load_state_dict` and `ftgs train --resume` refuse on it (Q-134 as answered, Q-153). `from_checkpoint(…, perf=…)` resumes with other performance settings; `load_state_dict` keeps the trainer's own for the fields that may differ, and the checkpoints saved afterwards record them. Without `perf`, the checkpoint's are used. Such a resume stays bitwise on the reference backend, except across a change of `slice_impl`, whose implementations U-28 holds to each other within its tolerance rather than bitwise.

## The model

The trainer trains a copy of a model passed to it, made as the model copies the tensors it is given, so the caller's object is never modified or moved and one loaded model can start several runs (owner answer, Q-69). The copy goes through the model's constructor, so a model that breaks §1.1's validity rules is refused at construction as an input error (Q-76). Built from init points, the model is the trainer's own. `Trainer.model` returns the model being trained, read-only: the same object across training iterations and events, which resize it in place, and the object the optimizer was built for. `load_state_dict`, and so `from_checkpoint`, replaces it with the model built from the loaded state. The model's `slice_impl` is set from `PerfConfig` (Q-66).

## Settings from `PerfConfig`

What each field does in M10 (Q-73):

| Field | In M10 |
| --- | --- |
| `slice_impl` | set on the model (Q-66); "cuda" raises `NotImplementedError` at construction, citing D-50: B-1 decided against the operator (#226), and the refusal comes before the first iteration rather than from its slice (Q-73) |
| `packed`, `precull` | passed to `render` (Q-36, Q-31) |
| `adam` | passed to `build_optimizer` (Q-44) |
| `log_every` | the period of the log reads and their validity check |
| `deterministic` | during each training iteration: cuDNN autotuning off, cuDNN's deterministic algorithms, and PyTorch's deterministic kernels, warning only where none exists; fused-ssim stays in use, since its backward pass writes each gradient element once, with no atomics (source log #73, answering Q-42's note). Off, cuDNN autotuning is on during the iteration for the two networks B-4 turned it on for, without a perceptual network, and whenever SSIM runs through its PyTorch fallback on CUDA, and left as the caller set it otherwise (#394; Q-273, below), and the other cuDNN flags are left alone. cuBLAS's matrix products are deterministic only when `CUBLAS_WORKSPACE_CONFIG` is set before CUDA starts, which the trainer can't do: `ftgs train` sets it (Q-131), and a Python caller sets it before its first CUDA call, for example `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment. With a DINO network on CUDA, two of its backward kernels have no deterministic form under the warning-only setting, and PyTorch warns of each (PyTorch 2.14.1 on the RTX 4090, #388): the antialiased bilinear downsampling's backward (`upsample_bilinear2d_aa_backward_out_cuda`), and the attention's, whose kernel the flag doesn't change, memory-efficient attention in float32 and Flash attention under bf16 autocast, both run in their default, non-deterministic backward. So, like gsplat's atomic additions, they make such a run deterministic only up to floating-point reordering (Q-268) |
| `perc_precision` | "tf32" or "ieee": cuDNN's TF32 flag during each training iteration (D-52). Each network of D-89 takes the layout B-4 chose for it at that precision (Q-273, below). "bf16", D-52's approximate option, for the trunk networks (`vgg16_oxford`, `alexnet_bvlc` and the four DINO networks): the trainer builds a private subclass of `PerceptualLoss` whose forward runs the network under bfloat16 autocast, on the images' device, the CPU included, and returns float32 (Q-191, Q-250). The images still enter as float32, which the trunk networks require, and the module keeps its float32 weights and constants; only the operations autocast covers run in bfloat16. With `flip` or `haarpsi`, which have no bfloat16 path, "bf16" is a ValueError when the `Trainer` is built (Q-250), though the `TrainConfig` itself is valid, so it can move to a trunk network; with λ_perc = 0 no network is built and nothing is refused (Q-267). Autocast stays local to the network, so TF32 stays on for every other convolution, the SSIM fallback's included (Q-172). The evaluator's networks stay float32. The float32 matmul precision is "highest" during each training iteration whatever the setting, so TF32 never reaches matrix products (D-52) |
| `images_per_step` | above 1 raises `NotImplementedError`, naming the two GPUs and M16's L3 the mode needs: multi-GPU training is deferred (D-54; Q-177, owner's answer) |
| `frame_cache` | "gpu" or "mmap" places the frame cache there; "auto" chooses once every training image size has had its first iteration and one more iteration has run, the second for one size (mmap on a CPU device; Q-225, Q-232); "off" decodes the image file at every training iteration and, on CUDA, copies it through pinned memory without blocking (D-51, Q-99) |
| `prefetch` | how many images the prefetcher keeps in flight; 0 reads each one when it is needed (Q-99) |

The global flags are set around each training iteration and restored afterwards, so training leaves the caller's process as it was.

**Each network's layout and cuDNN autotuning** (§2.4's perceptual-loss row: "the layout and cuDNN autotuning of each network as B-4 measures them"; §2.4 rule 2, D-52; Q-173, Q-272, Q-273). Both are exact options without a `PerfConfig` field. One goes on for a network only when B-4 measured it at least 3 % faster than the network's plain path (TF32, the render's view, no autotuning) at both of its sizes in both of its recorded runs, on the CI scene and on kitchen; B-4 alone decides for every network but the default, whose B-1 is rerun as well (Q-272, the owner's rule; Calibration data, #394). The trainer chooses its network once, at construction, from `PerfConfig`:

| `perc_net` | cuDNN autotuning, unless `deterministic` | NCHW inputs |
| --- | --- | --- |
| `vgg16_oxford` | on | under "ieee", and under "tf32" when the run autotunes; the render's view in a deterministic "tf32" run and under "bf16" |
| `alexnet_bvlc` | on | under every precision, autotuned or not |
| `flip`, `haarpsi` and the four DINO networks | on when SSIM runs through its PyTorch fallback on CUDA; otherwise the caller's flag (Q-73) | never: the render's view |
| none (λ_perc = 0) | on | |

In TF32, B-4 measured autotuning with NCHW inputs +4.7 % / +4.7 % on the CI scene and +3.4 % / +3.0 % on kitchen for vgg16_oxford, and +15.0 % / +18.5 % and +15.4 % / +16.7 % for alexnet_bvlc; neither option alone reached 3 % in both runs for vgg16_oxford, and no option did for the other six. Under the other precisions and in a deterministic run, the trainer gives NCHW inputs where B-4 measured them at least 3 % faster than the render's view at that precision and autotuning, in both runs (Q-273). vgg16_oxford's pattern is LPIPS-VGG's before revision 549o (Q-223). Without autotuning, cuDNN picks very slow kernels for alexnet_bvlc with channels-last inputs (131.9 ms against 16.0 at 1920×1080 in IEEE float32). SSIM's PyTorch fallback, which autotuning makes 7 to 11 % faster in both of B-4's runs, decides beside any network: on a CUDA device, an iteration whose SSIM takes the fallback autotunes whatever the network (the owner's answer, Q-273). `losses.ssim` takes it where fused-ssim doesn't import (Q-2) and for a window or σ other than the ones fused-ssim hardcodes (Q-42), the trainer's reading, provisional; with λ_ssim = 0 SSIM doesn't run. With fused-ssim installed and training's window, ꟻLIP's iterations leave the flag as the caller set it. Without a perceptual network, an iteration's only cuDNN convolutions are the SSIM fallback's when fused-ssim isn't installed, so it stays on, as it has been since #326.

The batch-first view of a render's [..., H, W, 3] is channels-last in memory, and every convolution follows that layout (source log 175). The NCHW copy is a private subclass of `PerceptualLoss` in `openftgs.train`, with a bfloat16 variant for alexnet_bvlc, whose `forward` copies the two images so that the batch-first views `PerceptualLoss.forward` takes of them are contiguous NCHW. `PerceptualLoss` is unchanged, so the evaluator's networks and the harness's keep the render's view. The copy is exact; the convolutions then take other kernels, whose results differ by rounding. B-4's NCHW cases start from an NCHW tensor, so they leave out the copy and its backward. Channels-last weights stay off, as §2.4 leaves them, and B-4 no longer measures them.

Autotuning must be on from a process's first convolution of a configuration to take effect: cuDNN caches the algorithm it picks and consults that cache whatever the flag says (source log 177). The perceptual network first runs inside a training iteration, so where the trainer autotunes, every configuration of the network is autotuned. Evaluation runs outside the iteration's flags, without autotuning; its ꟻLIP runs in IEEE float32 on the render's view. Autotuning chooses its algorithms at the first iteration, when the model is small, and their workspace must still fit when it has grown: B-4 measured up to 13.7 GiB peak allocated for vgg16_oxford with NCHW inputs and autotuning, and 12.4 GiB for alexnet_bvlc. LPIPS-VGG's measurements, the trainer's network until revision 549o, are kept in `docs/modules/cli/benchmarks.md` ("B-4 with LPIPS-VGG").

## Evaluation

`evaluate` renders the images of a data split through `eval.evaluate_images`, which goes through `render.render_request` one timestamp at a time, the path `ftgs render` and `ftgs eval` share (Q-57, Q-68). It builds the `Evaluator` once, on first use, and uses no generator, so an evaluation between steps never changes a run. With `out_dir` it writes METRICS.json through `eval.write_metrics`, in the report layout of Q-63, its header's `model` naming the trainer's iteration, with a null path and a null hash (Q-74, Q-136). A render that isn't finite raises `FloatingPointError` from `eval.evaluate_images`, naming up to five images, before the means are taken or anything is written, with or without `out_dir` (Q-129, Q-147).

# openftgs

openftgs is a clean-room Python implementation of FreeTimeGS (Wang et al., CVPR 2025), written from the paper and from the owner's spec in `docs/spec/openftgs-spec.md`. This glossary fixes the words used in the spec, the milestones, the questions file and commit messages. It says what each thing is; shapes, units, equations and file layouts belong to the spec, and the entry's closing reference points there.

Conventions: the spec's symbols (μx, μt, s, v, σ, σt, ρ, g, τ, t, u, S, Δt_f) and its normative tensor names (`means`, `times`, `log_durations`, `velocities`, `log_scales`, `quats`, `opacity_logits`, `sh0`, `shN`) are canonical. Identifiers fixed by the public API of spec §2.1 (`FreeTimeGaussians`, `num_gaussians`, `gaussian_ids`, `frame_dt`, `scene_scale_factor`, `test_camera`) keep their spelling in code even where this glossary avoids the word in prose. Where the spec and the paper name one thing differently, the spec's name is canonical and the paper's is listed under _Avoid_.

## Language

### Representation

**Primitive**:
One 4D Gaussian of the model, identified by its row index i, the same across all nine stored tensors; row order is deterministic, kept rows staying in order and new rows going last. §1.1, §1.3 Growth
_Avoid_: Gaussian, 4D Gaussian, splat, point, particle

**Quantity**:
One of a primitive's eight learnable properties from the paper: position, time, duration, velocity, scale, orientation, opacity and SH coefficients. §1.1
_Avoid_: attribute, parameter (ambiguous with the stored tensor), feature

**Stored tensor**:
One of the nine float32 tensors that hold the quantities in raw form, SH split into `sh0` and `shN`; their names are normative in checkpoints and the optimizer. §1.1
_Avoid_: raw parameter, attribute tensor, param (outside `params.safetensors`)

**Physical value**:
A quantity after activation, as the renderer and exchange files use it: σ = sigmoid(ô), a = exp(â), s = exp(ŝ) and R(q/‖q‖). §1.1
_Avoid_: activated value, decoded value, real value

**Moved position μx(t)**:
A primitive's position at query time t under its linear motion, as distinct from μx, its position at its own time center. E1
_Avoid_: current position, displaced mean, deformed position, `means_t` in prose

**Base opacity σ**:
A primitive's time-independent opacity, the sigmoid of its stored logit; the opacity that the sampling score and the dead test read. E3
_Avoid_: opacity (unqualified, where the effective opacity could be meant), peak opacity, static opacity

**Temporal opacity σt(t)**:
The unnormalized Gaussian in time, peaking at 1 at μt with standard deviation s, that scales a primitive's base opacity at time t. E4
_Avoid_: the paper's σ(t), temporal weight, temporal radial basis function, lifespan curve

**Effective opacity σ_eff(t)**:
The product σt(t)·σ that the rasterizer receives as a primitive's opacity at time t. E3
_Avoid_: space–time opacity (E3's name for the full σ(x, t), spatial factor included), time-modulated opacity, `opacities_t` in prose

**Duration s**:
The standard deviation of a primitive's temporal opacity, in normalized time, stored as ŝ = log s. E4, D-2
_Avoid_: temporal scale, lifespan, time scale, temporal extent

**Time center μt**:
The normalized time at which a primitive sits at μx and its temporal opacity peaks, stored in `times`. §1.1
_Avoid_: time (unqualified), a primitive's timestamp, temporal mean, "times" in prose

**Velocity v**:
A primitive's constant linear velocity in world units per unit of normalized time, stored raw. E1
_Avoid_: motion vector, displacement, flow

**SH DC band**:
The l = 0 spherical-harmonic coefficients, one RGB triple per primitive, held in `sh0` and set from the init point's color. §1.1, E2
_Avoid_: features_dc, base color, albedo

**SH higher bands**:
The l ≥ 1 coefficients held in `shN`, zero at initialization and empty at SH degree 0; kept apart from the DC band so each gets its own learning rate. §1.1
_Avoid_: features_rest, sh rest, view-dependent features

**SH degree L**:
The user parameter sh_degree, 0 to 3 with default 3, giving the highest band the model stores. D-3
_Avoid_: max degree, sh order, number of bands

**Active SH degree L_active(it)**:
The highest band that contributes to color at iteration it, rising from 0 by one band every 1,000 iterations up to L. E13
_Avoid_: current degree, SH warm-up, degree schedule

### Time

**Raw timestamp τ**:
An image's time in the dataset's own unit, float64, carried per image so unsynchronized cameras need no special case; the unit used at every file boundary. §1.4 Time, D-1
_Avoid_: time (unqualified), frame time, seconds (the unit, not the quantity)

**Normalized time t**:
Time mapped onto [0, 1] over the training timestamps, t = (τ − τ₀)/T_τ, in which every stored temporal quantity is expressed. E14
_Avoid_: scene time, scaled time, canonical time, normalized timestamp

**Training time span T_τ**:
τ₁ − τ₀, the distance between the smallest and largest training timestamps, which converts durations and velocities between raw and normalized units. E14
_Avoid_: sequence length (a frame count), time range, duration of the sequence

**Frame interval Δt_f**:
The typical gap between consecutive images of one training camera, in normalized time: the median over cameras of each camera's median gap. E14, D-46
_Avoid_: frame rate, time step, dt, "frame_dt" in prose

**Time normalizer**:
The pair (τ₀, τ₁) with its two conversions between τ and t, stored in every checkpoint and exchange file. §2.1 `TimeNormalizer`
_Avoid_: time scaler, time transform, time range

**Iteration it**:
The index of a training iteration, counting completed optimizer steps from 1 to T_it (30,000 by default). D-41
_Avoid_: step (reserved for Adam's per-tensor step counter), epoch, tick

**Training progress u**:
The schedule variable u = (it − 1)/(T_it − 1), 0 at the first iteration and 1 at the last. D-41, E12
_Avoid_: the paper's t for this quantity, progress fraction, normalized iteration

### Training

**Training iteration**:
The seven-step unit of training on one image: temporal slice, render, total loss, backpropagation with gradient statistics, one Adam step at that iteration's rates, and on event iterations the event. §1.3 One training iteration
_Avoid_: step, training step, batch

**Event**:
What happens at an iteration that is a multiple of 100 within the relocation window: growth while it ≤ 15,000, then relocation, then the reset of the gradient statistics, in that order; the period is K_reloc. §1.3 step 7, D-45
_Avoid_: densification step, refinement, strategy step, pruning step, the paper's N for the period

**Calendar**:
The table of iterations at which SH bands activate, growth and relocation events fall, regularization applies and the run ends; synthetic tests shorten every count in proportion to T_it. §1.3 Iteration calendar, §3.2
_Avoid_: schedule (reserved for learning rates), timeline, test calendar as a different thing

**Growth**:
The part of an event, in iterations 500 to 15,000, that adds primitives by cloning or splitting candidates; nothing is deleted. §1.3 Growth, D-18
_Avoid_: densification, densify, adaptive density control, refinement

**Candidate**:
A primitive whose accumulated gradient g exceeds τ_pos = 0.0002 at a growth event and so is cloned or split, subject to the cap. §1.3 Growth
_Avoid_: high-gradient Gaussian, densification candidate

**Clone**:
The growth of a candidate whose largest scale is at most 0.01·S: an exact copy of its nine stored tensors appended as a new row. §1.3 Growth, E17
_Avoid_: duplicate (gsplat's name), copy (reserved for relocation)

**Split**:
The growth of a larger candidate into two children drawn from its Gaussian with scales divided by 1.6, which replace it; the split is spatial only. E17, D-43
_Avoid_: 4D split, subdivide

**Cap N_max**:
The maximum primitive count, 3,000,000 by default and 500,000 in the † configuration; when candidates exceed the room left, the largest-g ones grow, ties to the lower row. D-44, D-39
_Avoid_: budget (reserved for initialization), max Gaussians, limit

**Relocation**:
The part of an event, in iterations 500 to 25,000, that moves every dead primitive onto a live target instead of deleting it. §1.3 Relocation
_Avoid_: pruning, deletion, respawn, resampling, MCMC relocation

**Dead set**:
The primitives whose base opacity is below 0.005, tested on the stored logit against ô_dead so that rounding cannot move the boundary. D-14
_Avoid_: dead Gaussians, prune mask, transparent set

**Live set**:
Every primitive not in the dead set; the population scored and drawn from in a relocation. §1.3 Relocation
_Avoid_: alive mask, survivors, kept set

**Target**:
A live primitive drawn with replacement, with probability proportional to its sampling score, to receive one or more dead primitives in a relocation. §1.3 Relocation step 3
_Avoid_: destination, parent (reserved for growth), donor

**Relocation move**:
The update applied once per target: with n copies in total, its opacity becomes 1 − (1 − σ)^(1/n) and its scales shrink so the stacked copies render like the original. E11, D-15
_Avoid_: MCMC split, opacity split, compute_relocation (gsplat's function name)

**Moved copy**:
A dead primitive after relocation: it carries its target's updated tensors and a position drawn from the target's Gaussian. §1.3 Relocation step 5, D-15
_Avoid_: copy (unqualified, which collides with clone), moved row, overwritten row, relocated Gaussian

**Sampling score ρ**:
The per-primitive score λg·ĝ + λo·σ, with λg = λo = 0.5, that weights the draw of relocation targets. E10
_Avoid_: the paper's s for this quantity, score, relocation score, importance

**Accumulated gradient g**:
The mean, over the iterations since the last event in which a primitive was visible, of the NDC-scaled norm of its 2D-mean gradient; it selects growth candidates and feeds the sampling score. E10, D-12
_Avoid_: the paper's ▽g, spatial gradient, positional gradient, view-space gradient, means2d gradient

**Normalized gradient ĝ**:
g divided by its 99th percentile over the live, at-least-once-visible primitives and capped at 1; identically 0 when that percentile is 0. E10, D-13
_Avoid_: normalized score, relative gradient, clipped gradient

**Gradient statistics**:
The per-primitive accumulators g_sum and count that produce g, kept on the GPU, reset after every event and inherited by grown rows from their parent. §2.1 `GradStats`, D-45
_Avoid_: densification stats, gradient accumulators, xyz_gradient_accum (3DGS's name)

**Count-control mode**:
The configured rule for how N evolves: "gradient" (growth as above), "score" (append 5 % drawn by ρ at each event, up to N_max) or "none" (N fixed, relocation alone moves capacity). §1.3 Count-control modes, D-18
_Avoid_: growth mode, strategy, densification strategy, MCMC mode

**Rendering loss L_render**:
The image term 0.8·L1 + 0.2·(1 − SSIM) + 0.01·LPIPS-VGG between the unclamped render and the training image. E7
_Avoid_: photometric loss, reconstruction loss, image loss (the L1 term alone)

**4D regularization L_reg**:
The mean over all N primitives of σ·sg[σt(t)] at the current image's time, weighted by λ_reg = 10⁻² for the whole run unless reg_until ends it. E8, E9, D-10
_Avoid_: temporal regularization, opacity regularization, early-stage regularizer, L_4d

**Total loss L**:
L_render + λ_reg·L_reg(t), the quantity backpropagated at every iteration. E9
_Avoid_: objective, overall loss, training loss

**Velocity annealing**:
The log-linear decay of the velocity learning rate from ηv,0 to ηv,1 over training progress, read as a product rather than the paper's printed sum. E12, D-19
_Avoid_: λt, velocity schedule, velocity decay

**Spatial scale S**:
1.1 times the largest distance from a training camera center to the centroid of the training camera centers; it scales the position and velocity rates and the clone/split boundary. D-27
_Avoid_: scene scale, scene extent, scene radius, cameras_extent (3DGS's name), n or N for anything but a count

**Ablation**:
A configured departure from the default method that reproduces one of the paper's §4.2 variants, such as "w/o 4D initialization" (zero initial velocities) or "w/o periodic relocation" (relocation off; 3DGS densification with pruning and opacity resets in its place). §1.3, §1.4
_Avoid_: variant, baseline, experiment

**† configuration**:
The default configuration with N_max = 500,000, reproducing the paper's "≤ 500k primitives" rows. D-39
_Avoid_: small model, capped model, lite

**Reference outcome**:
A number the paper reports (training time, FPS, model size, primitive count, PSNR) that serves as a target when calibrating, never as a setting. §1.5
_Avoid_: target metric, baseline number, spec value

### Initialization

**Init frame slot**:
A nominal time t_k = k·Δt_f for every k_stride-th k, admitting at most one image per training camera, the one within Δt_f/2 of t_k; a camera with none sits the slot out. §1.3 step 1, D-47
_Avoid_: keyframe, init time, frame index, slot (when the images or points are meant)

**Init frame**:
The images admitted to one init frame slot together with the points triangulated from them: the unit over which matching, the per-frame budget, the 3-NN scale estimate and the k-NN velocity operate. §1.3 steps 2–8
_Avoid_: initialization frame, keyframe, frame (unqualified)

**Camera pair**:
Two training cameras matched at an init frame: each camera with the training camera whose optical axis is angularly closest, every unordered pair matched once as (A, B) in manifest order. D-22
_Avoid_: image pair, stereo pair, view pair

**Match**:
One pixel correspondence between the two images of a camera pair, taken from RoMa's dense warp with certainty at least 0.5, up to 5,000 per pair drawn in proportion to certainty from the pair's own generator. D-22, D-56, D-58
_Avoid_: correspondence, keypoint, feature match, sample (RoMa's `sample()` is not used)

**Certainty**:
RoMa's raw per-pixel confidence that a warp is correct, used both as the admission threshold for matches and as their sampling weight. D-58
_Avoid_: confidence, score, probability

**Triangulation**:
The two-view linear DLT solve, in float64, that turns a match into a 3D point, kept when the point has positive depth in both views, reprojection error at most 2 px and a triangulation angle of at least 1°. §1.3 step 3, D-22
_Avoid_: reconstruction, lifting, SfM

**k-NN velocity**:
A point's initial velocity: the offset to its nearest neighbour among the next init frame's points divided by the time between the frames; a backward difference on the last frame, zero with a single frame. E15, D-23
_Avoid_: 4D initialization (the paper's ablation name for having it), scene flow, flow init

**Initial budget B**:
The number of points kept per frame of the sequence, 1,000 by default, so the start holds at most F·B points with F = round(1/Δt_f) + 1; an init frame keeps k_stride·B. D-24
_Avoid_: initial count, cap (reserved for N_max), seed count, "points_per_frame" in prose

**Stride k_stride**:
The spacing of init frame slots in frames, 1 by default; it also sets the initial duration s₀ = k_stride·Δt_f. D-21
_Avoid_: frame skip, subsampling, step

**Init points**:
The 4D point set training starts from: positions, colors and raw timestamps, with optional velocities, durations and frame ids; produced by `roma_init` or supplied by the user as a ftgs-init file. §2.1 `InitPoints`, §2.2
_Avoid_: point cloud, seed points, 4D points, initial Gaussians, SfM points

**Vendored RoMa**:
The copy of RoMa's inference code at commit 77f8d68 inside `openftgs._vendor.romatch`, with its OpenCV and xformers imports removed, that the `[init]` extra matches with. D-65
_Avoid_: romatch (the upstream package), RoMa package, the matcher

### Rendering

**Render backend**:
One of the two implementations of E6 behind `render()`: gsplat (CUDA, production) or reference (pure PyTorch, CPU, the readable oracle for tests). §2.1
_Avoid_: rasterizer (ambiguous with gsplat itself), renderer mode, engine

**Temporal slice**:
The fused per-primitive computation of a model at one time t, producing the per-time arrays, the E8 terms and the culling mask, with eager, compiled and optional CUDA implementations behind one signature. §2.4 Kernels, D-50
_Avoid_: deformation, per-time evaluation, `at_time` (the method), `GaussiansAtTime` (the return type)

**Per-time arrays**:
The derived arrays `means_t`, `opacities_t`, `quats_n`, `scales` and `sh` that the rasterizer consumes at a given time, never stored. §1.1
_Avoid_: deformed Gaussians, time-sliced parameters, instantaneous parameters

**Temporal index**:
The rendering-only structure that assigns primitives to buckets so a render at time t passes only its bucket to gsplat; images are unchanged because gsplat's own test still decides. D-55
_Avoid_: time index, temporal culling, bucket cache

**Bucket**:
One span of time, one frame interval wide, in the temporal index, listing every primitive whose effective opacity can reach 1/255 inside it, widened by a margin. D-55
_Avoid_: bin, time slot, window (reserved for SelfCap)

**Visible primitive**:
A primitive whose two screen radii are both positive in an image, which is what the gradient statistics count. E6
_Avoid_: active primitive, rendered Gaussian, in-view primitive, visibility mask (the tensor)

**Culling**:
The removal of primitives from a render before compositing: effective opacity below 1/255, depth outside [0.01, 10¹⁰], det Σ′ ≤ 0, or a screen box wholly off-image; the opacity cull happens inside gsplat's projection by default. E6, D-32, D-57
_Avoid_: pre-culling (one placement of the opacity cull), frustum culling (part of it), filtering, compaction (the explicit alternative)

**Packed mode**:
gsplat's rasterization layout in which only visible primitive–camera pairs carry data, each row of means2d labeled by `gaussian_ids`; the default until B-2 calibrates. D-57
_Avoid_: sparse mode, compact mode

**Background c_bg**:
The color composited behind the primitives, black by default and never random; input alpha channels are dropped against it. D-34
_Avoid_: bg, background randomization, clear color

**Render request**:
A ftgs-render-request file: items, each a camera and a raw timestamp, that `ftgs render` and the harness answer with a ftgs-renders file of rgb and alpha per item. §2.2
_Avoid_: view list, render job, query (reserved for the harness's forward queries)

### Data

**Scene manifest**:
The ftgs-scene JSON file describing a converted dataset: cameras in the continuous pixel convention, images with per-image raw timestamps and splits, optional masks, and metadata about how it was made. §2.2
_Avoid_: dataset config, transforms.json, scene file, SCENE.json (the CLI placeholder)

**Image record**:
One image of a scene manifest: its camera, raw timestamp, path, split and optional frame index. §2.1 `ImageRecord`
_Avoid_: frame (ambiguous with a time index), view, sample

**Camera**:
A pinhole camera as gsplat sees it: K in pixels under the continuous pixel convention, a 4×4 world-to-camera matrix with OpenCV axes, and an undistorted image size. §1.4 World and camera frames, D-28
_Avoid_: pose (the extrinsics alone), view, intrinsics (K alone), camera-to-world

**Continuous pixel convention**:
The convention in which the image spans [0, W] × [0, H] and pixel (u, v) is sampled at (u + 0.5, v + 0.5), so an OpenCV calibration gains +0.5 on cx and cy. §1.4
_Avoid_: half-pixel offset, pixel-center shift, integer convention

**Split**:
The label "train" or "test" on every image record; training, initialization and the spatial scale read only the train split. §2.1, D-66
_Avoid_: partition, fold, subset

**Test view**:
The camera held out for evaluation: cam00 on Neural3DV, 0015, 0007 or 0009 on SelfCap, configurable on ENeRF-Outdoor; its images form the test split. D-29
_Avoid_: held-out view, held-out camera, test camera (outside the converter's argument name), validation view, novel view (a camera not in the dataset at all)

**Multi-view video**:
The dataset object `MultiViewVideo`: the image records of a scene manifest with their time normalizer, spatial scale and frame interval, loading images lazily. §2.1
_Avoid_: dataset (unqualified), video, sequence, scene (the manifest)

**Frame cache**:
The uint8 copy of every training image, one safetensors tensor per camera keyed by the manifest's hash, placed on the GPU when it fits and memory-mapped otherwise. D-51
_Avoid_: image cache, dataloader, preload

**Prefetcher**:
The thread that copies the next images, in the sampler's known order, through pinned buffers to the GPU on a side stream. D-51
_Avoid_: dataloader worker, loader thread, pipeline

**Converter**:
A `ftgs convert` subcommand (n3dv, opencv-yaml, selfcap) that turns a released dataset into undistorted PNG frames and a scene manifest, decoding video with the user's ffmpeg. §1.4 Dataset protocols, D-48
_Avoid_: importer, preprocessor, loader, dataparser

**Undistortion**:
Removing lens distortion from input images so the manifest holds pinhole cameras: the package's own implementation of OpenCV's documented model for Neural3DV and ENeRF-Outdoor, COLMAP's image_undistorter for SelfCap. D-60, D-61
_Avoid_: rectification, calibration (the input), dewarping

**Resize ratio r**:
The factor applied to converted images by area averaging, with K scaled by the realized W′/W and H′/H; 0.5 for Neural3DV and the SelfCap dance and corgi scenes. D-33
_Avoid_: downsample factor, scale (reserved), resolution level

**SelfCap window**:
The 60-frame range of 0-based decode indices n, from the dataset card, that forms one FreeTimeGS scene of a longer SelfCap sequence. §1.4, D-59
_Avoid_: clip, segment, frame range, bucket

**Sync offset**:
A SelfCap camera's offset in seconds from `optimized/sync.json`, subtracted from the frame-index time n/60 to give the image's raw timestamp. D-59
_Avoid_: time shift, delay, latency

**Dynamic-region mask**:
A per-test-image mask of the moving content, binarized at 0.5 and supplied by the user, whose bounding box crops both render and ground truth for dynamic-region metrics; approximate when its background was estimated rather than captured. D-64, D-67
_Avoid_: foreground mask, matte, alpha matte, segmentation

**Fastest-10 frames**:
The 10 consecutive frames of dance1's window whose consecutive test-view ground-truth frames differ most, stored in the manifest for the ablations' "fastest motion" column. D-69
_Avoid_: fast segment, high-motion frames, motion window

### Files

**File format**:
One of the five versioned public formats, each carrying `format` and `version`: ftgs-scene (manifest, JSON), ftgs-init (init points, safetensors), ftgs-checkpoint (directory), ftgs-if (exchange model, safetensors) and ftgs-render-request with its ftgs-renders answer. Nothing is ever unpickled. §2.2
_Avoid_: pickle, .pt, PLY as an interface (it is export-only)

**Exchange model**:
The implementation-neutral content of a ftgs-if file: physical values in raw time units (position, time_center, duration, velocity, scale, rotation, opacity, sh) with the render conventions as metadata, so any implementation can import or export it. §2.2, §2.1 `ExchangeModel`
_Avoid_: FTGS-IF as a noun, interchange model, export file, model file (ambiguous with checkpoint)

**Checkpoint**:
A ftgs-checkpoint directory holding a trainer state in full, from which a resumed run continues bit-exactly on the reference backend. §2.2
_Avoid_: snapshot, save, .pt, model (when the exchange model is meant)

**Trainer state**:
The plain data a checkpoint holds: iteration, config, time normalizer, S, Δt_f, the nine stored tensors, Adam moments and step counters, gradient statistics, generator states and sampler state. §2.1 `TrainerState`
_Avoid_: state dict (PyTorch's generic term), session, run state

### Evaluation

**SSIM_k**:
E7's SSIM (11×11 Gaussian window, σ = 1.5, valid region) with data range k, so C₁ = (0.01k)² and C₂ = (0.03k)²; the paper reports SSIM₂ on SelfCap. E16, D-67
_Avoid_: ssim (unsubscripted, in results), MS-SSIM, structural_similarity

**DSSIM_k**:
The dissimilarity (1 − SSIM_k)/2, lower is better, as the paper's main tables report. E16, D-29, D-40
_Avoid_: D-SSIM loss (the loss term is 1 − SSIM), 1 − SSIM

**LPIPS-Alex**:
The LPIPS metric with the AlexNet backbone and inputs mapped to [−1, 1], reported for Neural3DV. E16, D-68
_Avoid_: lpips (unqualified), perceptual metric

**LPIPS-VGG**:
The LPIPS metric with the VGG backbone and inputs mapped to [−1, 1], reported for ENeRF-Outdoor and SelfCap; the same network is the perceptual term of the rendering loss. E16, D-68
_Avoid_: lpips (unqualified), LPIPS loss (when the metric is meant)

**lpips_vgg_raw**:
LPIPS-VGG computed on unmapped [0, 1] inputs, the 3DGS and gsplat convention, reported beside LPIPS-VGG until a comparison shows which the paper used. D-68
_Avoid_: raw LPIPS, unnormalized LPIPS

**Dynamic-region metric**:
Any metric computed after cropping render and ground truth to the dynamic-region mask's bounding box and blacking out pixels outside the mask in both; reported under `dynamic` beside the full-frame `entire` values. D-67
_Avoid_: masked metric, foreground metric, dynamic PSNR (unqualified)

### Differential harness

**Harness**:
`ftgs_harness`, the separate package that compares two implementations, A (this package) and B, through files, commands and metrics only, importing neither. §3.3
_Avoid_: test suite, conformance kit, benchmark (reserved for B-n)

**Adapter**:
The small executable each implementation ships, answering info, export, import, render, forward and train in the §2.2 formats plus the harness's own query and answer formats. §3.3 Adapter contract
_Avoid_: plugin, driver, wrapper, bridge

**Level**:
One of the harness's four stages: L0 boundary converters, L1 identical-model renders, L2 forward behavior on random inputs, L3 multi-seed statistics. §3.3
_Avoid_: stage, phase, tier (the tolerance class within L1)

**Tier**:
The L1 tolerance class chosen from both info replies: same rasterizer and constants, or different rasterizers. §3.3 L1
_Avoid_: level, tolerance mode, strictness

**Probe**:
One of the ten L1 scenes, P1 to P10, that isolates a single behavior by sweeping one variable so a mismatch points at one quantity. §3.3 L1
_Avoid_: test scene, fixture, case (any check's inputs), synthetic scene (reserved for S-n)

**Forward query**:
An L2 case: a ftgs-forward-queries file naming a query kind (motion, temporal_opacity, losses, lr_schedule, relocation_math and the others) with seeded inputs, answered by a ftgs-forward-answers file. §3.3 L2
_Avoid_: unit query, probe, request (reserved for render requests)

**Symptom**:
The report's description of a failed check in terms of inputs and outputs only: files, times, cameras, magnitude, sign, a pattern from the fixed vocabulary and the trend against the swept variable, never a function name or a suspected cause. §3.3 Report format
_Avoid_: diagnosis, root cause, bug, finding

**Not comparable**:
An adapter's verdict on a harness setting or query kind it cannot honor, reported rather than silently dropped. §3.3
_Avoid_: skipped, unsupported, ignored

**Null distribution**:
In L3, the A–A′ distances from a second independent seed set A′ of implementation A, against which the A–B distances are tested. §3.3 L3
_Avoid_: control run, baseline

### Clean-room process

**Owner**:
The person who wrote the spec, runs the sessions, sets the clean-room rules and decides every open question. §0, CLAUDE.md
_Avoid_: author (ambiguous with the paper's authors), maintainer, user, PI

**Implementer**:
The agent or person writing openftgs from the spec under the clean-room rules, in sessions the owner runs. CLAUDE.md
_Avoid_: developer, Claude, assistant, contributor

**Spec revision**:
The numbered version of the owner's document "FreeTimeGS — Clean-Room Specification & Architecture", exported to `docs/spec/openftgs-spec.md` with its SHA-256 (revision 549 at present); changes arrive only as a new export that replaces the file in one commit. docs/spec/README.md
_Avoid_: spec version, draft, edition, the spec's git history

**Provenance tag**:
The marker on every spec rule saying who fixed it and whether the implementer may change it: [P], [P→3DGS], [L: source] or [D-n]. §0
_Avoid_: citation, source tag, origin, label

**[P] tag**:
Stated in the paper, with section or equation; never changed. §0
_Avoid_: paper-fixed, hard requirement

**[P→3DGS] tag**:
Set where the paper says "same settings as 3DGS", with the value taken from 3DGS or gsplat defaults; changed only with a logged reason. §0
_Avoid_: 3DGS default, inherited setting

**[L: source] tag**:
Standard Gaussian-splatting practice taken from a named allowed source; changed only with a logged reason. §0
_Avoid_: literature value, convention

**Decision D-n**:
A numbered design decision in the spec's Appendix B filling a gap the paper leaves, with rationale, alternatives and detecting test; configurable, never changed silently, and this project's decision record. §0, Appendix B
_Avoid_: ADR, ambiguity-log row, assumption, choice

**Question Q-n**:
A numbered row in `docs/clean-room/questions.md` recording a provisional decision the implementer made where the spec is silent or ambiguous and the choice affects numbers, outputs or the public API; marked `# Q-n` in code and raised with the owner, who may turn it into a D-n. CLAUDE.md
_Avoid_: issue (the GitHub channel that carries it), TODO, open point, assumption

**Allowed source**:
Anything the implementer may read: the paper and project page, papers it cites, general Gaussian-splatting literature, dataset documentation and data files, and third-party software to the depth its §0 class permits. §0
_Avoid_: reference, dependency docs, whitelist

**Forbidden source**:
Anything from another FreeTimeGS implementation (the authors' code included), EasyVolcap in any form, any GPL or AGPL code, and the source code of software whose class allows documentation only. §0
_Avoid_: blacklist, upstream, the official repo

**Source log**:
The table of every source consulted and what was taken from it: Appendix A for the spec, `docs/clean-room/source-log.md` for the implementation. §0, CLAUDE.md
_Avoid_: references, bibliography, provenance log

**Incident**:
A source-log row recording that a forbidden source was met, that reading stopped and that nothing was taken, reported to the owner before any other work. CLAUDE.md
_Avoid_: violation, leak, contamination

**Clean-room guard**:
The PreToolUse hook `.claude/hooks/clean_room_guard.py` that blocks tool calls naming FreeTimeGS or EasyVolcap in searches, downloads, URLs and paths; a tripwire, not a permission. CLAUDE.md
_Avoid_: firewall, filter, the hook (unqualified)

**Milestone M-n**:
One of the nineteen bottom-up build steps M0 to M18 in `docs/clean-room/milestones.md`, each with its spec items, its tests and a review in a fresh session; the order is the spec author's suggestion, not part of the spec. milestones.md
_Avoid_: phase, sprint, epic, stage

**Unit test U-n**:
One of the thirty-eight §3.1 tests that pin an equation or convention with a numeric pass criterion, written from its spec description, not from the code; CPU tests use the reference backend. §3.1
_Avoid_: spec test, equation test, regression test

**Converter check C-n**:
One of the five §3.1 checks of a converter against real dataset geometry, run only when the dataset is present and never in CI. §3.1
_Avoid_: dataset test, integration test

**Synthetic scene S-n**:
One of the eleven §3.2 recovery tests that train on a scene with known ground truth and check what was recovered, not just that the code ran. §3.2
_Avoid_: toy scene, end-to-end test, smoke test (only S-10 and U-26's runs are that)

**Benchmark B-n**:
One of the eight §3.4 measurements of speed and memory run by `ftgs bench`, never of correctness; a budgeted one fails CI when more than 10 % worse than its baseline. §3.4
_Avoid_: perf test, profile, timing

**Calibrated threshold**:
A pass criterion or budget marked "calibrated" in the spec: set once from analytic expectations or the first reference run, then frozen; the first measured values go to questions.md as calibration data, and changing it needs a logged reason. §3.2, §3.4
_Avoid_: tuned threshold, empirical limit, magic number

**Exact optimization**:
A performance change that alters results only through floating-point rounding, on by default once a benchmark shows a gain. §2.4, D-49
_Avoid_: lossless, safe path, free speed-up

**Approximate optimization**:
A performance change that alters results beyond rounding: off by default, logged as a D-n, and required to pass L2 and L3 before it can become a default; TF32 in training-time convolutions is the one approximate default. §2.4, D-49
_Avoid_: lossy, fast path, mixed precision (one instance of it)

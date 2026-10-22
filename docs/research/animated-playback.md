# Playing a model as an animation: players and renderers

A documentation-only survey, written 2026-10-22 on tracker issue #352, of how a trained model can be watched like volumetric video: a free camera in 3D, play and pause, seeking in time. openftgs has no viewer and gets none: its command line is fixed by spec §2.1 (`convert`, `cache`, `init`, `train`, `render`, `eval`, `export`, `bench`), and viewing is left to external tools reading the files of §2.2. This document surveys those tools and the routes to a custom player, and recommends one per platform. It proposes no code and changes no spec item.

**Owner decisions behind the survey** (2026-10-22, in session):

- Two tiers. Splat playback, the model drawn as it renders, is the goal. The **centers tier**, the primitive centers drawn as points over time (what the request called an animated point cloud), is surveyed as a cheaper preview. This document uses "centers tier" for it throughout.
- Fidelity. "Looks right" is enough. A third-party renderer may differ from `ftgs render`; the gaps are recorded (section 2.6), not disqualifying.
- Platforms. Python on the training machine (CUDA available), a shareable web browser, a game engine or DCC tool, a native desktop application.
- Clean room. A candidate that advertises support for the paper's own released format, or for other 4D Gaussian formats, is read through its own documentation pages only, never its source; nothing about such a format is taken; every page is logged. The paper's released renderer and demos were never opened (spec Appendix A #2, D-6) and are not candidates.

**Words.** Glossary terms are used as `docs/GLOSSARY.md` spells them: primitive, quantity, physical value, moved position μx(t), effective opacity σ_eff(t), raw timestamp τ, normalized time t, training time span T_τ, frame interval Δt_f, camera, exchange model, checkpoint, render request. A **player** is the external program that draws the animation; a **viewer** is an existing program that draws static Gaussian-splat scenes. Third-party programs' own feature names appear only as named features.

**Clean-room record.** Every page, search, skipped search hit, screened-out candidate and general-knowledge claim about third-party software behind this document has a numbered row in `docs/clean-room/source-log.md`, from row 223 on (section 9).

## 1. Purpose and scope

The question is how to get from a trained model, a `ftgs-checkpoint` directory or a `ftgs-if` exchange file, to an interactive animation, without adding a viewer to openftgs. Three families of answers exist:

1. an existing program that already plays Gaussian-splat or point sequences, fed with files openftgs can write today;
2. an existing static viewer extended with the model's two time-dependent equations;
3. a renderer written for the purpose.

Sections 2 to 4 state what any player has to do, what it can read, and the three architectures the answers fall into; these parts rest on the spec and this repository alone. Section 5 is the survey of existing software, section 6 the custom routes, section 7 the recommendations, section 8 what openftgs would have to add for each route, as owner decisions rather than proposals.

## 2. What a player has to do

### 2.1 Load a model

The two public forms of a trained model are the exchange model (`ftgs-if`, one safetensors file of physical values in raw time units) and the checkpoint (a directory whose `params.safetensors` holds the nine stored tensors in raw form). Section 3 describes both. A player outside Python wants the exchange model: it needs no activation and carries its time span; the checkpoint is for Python players, which can call `FreeTimeGaussians.load` on either.

### 2.2 Evaluate the model at a time

The model's dependence on time is two equations (spec §1.1, "Scale, orientation and SH do not change with time; only position and opacity do"):

- the moved position μx(t) = μx + v·(t − μt) (E1), linear for every t, no clamping;
- the temporal opacity σt(t) = exp(−½ ((t − μt)/s)²) (E4), which multiplies the base opacity σ into the effective opacity σ_eff(t) = σt(t)·σ that the rasterizer receives (E3).

Scale, orientation and the SH coefficients are constant. The covariance (E5) and the color (E2, with the view direction from the camera center to the moved position) are what any static Gaussian-splat renderer already computes, so a static renderer becomes a player by adding four quantities per primitive (time center, duration, the three velocity components; opacity it already has) and one uniform, the current time, to its per-primitive stage. That hook is the survey's main discriminator.

In the exchange model's raw units (spec §1.4 "Time", E14) the same two lines read `position + velocity · (τ − time_center)` and `exp(−½ ((τ − time_center) / duration)²)`, since `time_center`, `duration` and `velocity` are stored as τ₀ + μt·T_τ, s·T_τ and v/T_τ. Renders are invariant under that unit change (§1.4, test U-4), so a player never needs normalized time.

A primitive whose effective opacity is below 1/255 is culled by `render()` (E6, D-32); a player may skip it too, and may precompute, per frame interval, which primitives can reach 1/255 at all (|t − μt| ≤ s·√(2 ln(255σ)), the bucket rule of D-55), so that a frame touches only its bucket.

### 2.3 Draw

**Splat tier.** Project each primitive with E6: covariance from E5, EWA projection, front-to-back alpha compositing by camera depth, background behind. Any Gaussian-splat renderer does this; the constants it uses are the fidelity gaps of section 2.6.

**Centers tier.** Draw the moved position of each primitive as a point, colored by the SH DC band alone (E2 at degree 0: C₀·`sh0` + 0.5, clamped at 0, with C₀ = 0.28209479177387814) and either culled or faded by its effective opacity. This needs no splatting and no sort, and any program that draws a colored point set per time step can do it; it shows where the primitives are and when they live, not what the render looks like.

### 2.4 Camera

The spec's camera (§1.4, D-28; `openftgs.types.Camera`) is a pinhole K in pixels under the continuous pixel convention plus a 4×4 world-to-camera matrix in OpenCV axes: camera x right, y down, z forward into the scene; x_cam = R_w·x + t_w; camera center o = −R_wᵀ·t_w; pixel (u, v) sampled at (u + 0.5, v + 0.5); no distortion; a right-handed world with no assumed up axis.

Interactive viewers mostly hold a camera-to-world pose in OpenGL axes (x right, y up, looking down −z). Converting between the two is a fixed change of camera axes, F = diag(1, −1, −1):

- world-to-camera, OpenGL axes: W_gl = F̃ · W_cv, with F̃ = diag(1, −1, −1, 1);
- camera-to-world, OpenGL axes: C_gl = W_cv⁻¹ · F̃, that is, the spec's camera-to-world matrix with its second and third columns negated;
- and back the same way, F̃ being its own inverse.

The vertical field of view follows from K as 2·atan(H / (2 f_y)); a principal point off the image center needs an off-center projection or a shift of the viewport; a viewer that puts pixel centers at integers takes c_x − 0.5 and c_y − 0.5 (§1.4 states the reverse conversion, from OpenCV calibrations to the spec). The checkpoint's scene scale S (`spatial_scale` in `manifest.json`, D-27) is a useful unit for orbit radius and movement speed; the exchange model carries no such number, so a player derives one from the positions' extent.

The training cameras of a scene (`ftgs-scene` manifest, §2.2) give natural starting poses and, interpolated, a camera path for offline rendering (section 4, route C).

### 2.5 Timeline

- **Span.** Both public forms carry τ₀ and τ₁ and the `time_unit` (§1.4, "Both are stored in every checkpoint and exchange file"). The converters set τ = k/fps seconds (k/30 for Neural3DV, k/fps for ENeRF-Outdoor and OpenCV YAML scenes, n/60 minus a per-camera offset for SelfCap; §1.4 and §2.1 `convert`), so with `time_unit` "s" a 1× playback advances τ at wall-clock speed.
- **Frames.** The exchange model carries no frame rate, frame count or frame interval. The checkpoint carries Δt_f (`frame_dt` in `manifest.json`), from which F = round(1/Δt_f) + 1 frames (§1.3 step 7). A player of exchange files takes a frame rate from the user or steps τ continuously.
- **Seek.** `render()` and the model accept any τ (§1.4, "Rendering accepts any τ"); between frames the motion and opacities interpolate by E1 and E4 themselves (test S-7 trains on even frames and tests on odd ones). Outside [τ₀, τ₁] the motion extrapolates linearly and `render()` warns; a player should mark that region or clamp its slider to the span.
- **Background.** Black unless the model was trained against another color; the exchange model's `background` (D-84, from its `render_conventions` block) says which.

### 2.6 Fidelity gaps against `ftgs render`

The owner accepted "looks right". The table says where a renderer that is not openftgs's own will differ from `ftgs render`, so a reader knows what a gap means when the survey marks one. No viewer documents these constants, so section 5 records a candidate's gaps as "unknown from documentation" unless its pages say otherwise; the structural gaps hold for every stock viewer.

| Convention of `ftgs render` | Spec | What a stock viewer typically does | Visible effect |
| --- | --- | --- | --- |
| Temporal opacity σt(t) and moved positions | E1, E3, E4 | None: a static scene | Without the two equations there is no animation at all; this is the gap every route must close |
| Rasterize mode `classic`, ε = 0.3 added to Σ′ | E6, D-6 | Varies; antialiased modes rescale opacity by the dilation | Slightly different softness of small primitives |
| Alpha clamp α_max = 0.999 | E6, D-31 | 0.99 in the 3DGS paper's formulation | Near-opaque primitives let a little light through |
| Skip α < 1/255; cull σ_eff < 1/255 | E6, D-32 | Similar thresholds are common, not guaranteed | Faint halos appear or vanish |
| Screen extent √(2 ln(255·σ_eff)) standard deviations, capped at 3.33 | E6, D-73 | Often a fixed 3σ | Footprint of faint primitives |
| Transmittance stop at 10⁻⁴, that primitive excluded | E6 | Varies | Deep layers of dense regions |
| Jacobian clamp at 0.15 of the image size beyond its edges, on J only | E6, D-73 | Varies | Primitives near the image border |
| Pixel centers at +0.5 | §1.4, D-28 | Integer pixel centers in GL-style pipelines | A half-pixel shift unless c_x, c_y are adjusted (section 2.4) |
| SH: gsplat 1.5.3's basis constants and band order, +0.5 offset, clamp at 0 only | E2, D-4 | 3DGS-style viewers use the same family of constants, not always the same signs or clamp | Wrong view-dependent tint when the basis differs; the DC band is unaffected |
| Output unclamped, no gamma conversion | E7, §1.4 "Images" | Clamp to [0, 1] and display as sRGB; some tone-map | Colors brighter than 1 saturate; a gamma curve applied twice darkens or lightens the whole image |
| Background composited as image + (1 − alpha)·c_bg, black by default | E6, D-34, D-84 | Configurable | Edges of the scene against the background |

The spec's own cross-rasterizer tier for "same look" is PSNR ≥ 45 dB and at most 0.5 % of pixels differing by more than 1/255 (§3.3 L1); only a renderer built on `render()` or gsplat 1.5.3 in classic mode meets it. No route in section 6 other than the first claims it.

## 3. The data a player can read today

### 3.1 The exchange model (`ftgs-if`)

One safetensors file of physical values in raw time units (§2.2; `docs/modules/io/file-formats.md`):

| Key | Dtype | Shape | Value |
| --- | --- | --- | --- |
| `position` | float32 | [N, 3] | μx |
| `time_center` | float64 | [N] | τ₀ + μt·T_τ |
| `duration` | float64 | [N] | s·T_τ |
| `velocity` | float32 | [N, 3] | v/T_τ, world units per unit of τ |
| `scale` | float32 | [N, 3] | exp(â) |
| `rotation` | float32 | [N, 4] | unit quaternion (w, x, y, z), w ≥ 0 |
| `opacity` | float32 | [N] | sigmoid(ô), in [10⁻⁷, 1 − 10⁻⁷] |
| `sh` | float32 | [N, (L+1)², 3] | SH coefficients in gsplat's basis order; color by E2 |

Metadata, all strings: `format` ("ftgs-if"), `version` ("1"), `sh_degree`, `time_unit`, `tau0`, `tau1`, `producer`, and `render_conventions`, a JSON block describing the producer's renderer (the constants of section 2.6 and the training background, D-73, D-84). No frame rate, frame count or Δt_f.

A safetensors file is simple to read outside Python: an 8-byte little-endian header length, a JSON header mapping each tensor name to its dtype, shape and byte range, then one flat buffer (source-log row 223). Every language with a JSON parser and typed arrays can load an exchange model in a few dozen lines, which matters for browser and native players.

Size per primitive at L = 3: 12 + 8 + 8 + 12 + 12 + 16 + 4 + 192 = 264 bytes, so a model of a million primitives is about 264 MB; at L = 0 it is 84 bytes per primitive.

### 3.2 The checkpoint (`ftgs-checkpoint`)

A directory (§2.2): `params.safetensors` with the nine stored tensors in raw form (`means`, `times`, `log_durations`, `velocities`, `log_scales`, `quats`, `opacity_logits`, `sh0`, `shN`, float32), the optimizer, statistics, generator and sampler files, and `manifest.json` with τ₀, τ₁, the scene scale S, Δt_f, the SH degree, the time unit and the full training configuration, whose `background` is the training background. A player that reads it directly must apply the activations of §1.1 (exp for scales and durations, sigmoid for opacity, normalization for quaternions) and E14's conversions to raw time; in Python, `FreeTimeGaussians.load` and `to_exchange` do that.

### 3.3 Per-time arrays and the PLY snapshot

In Python, `FreeTimeGaussians.at_time(t)` (§2.1; `docs/modules/model/at-time-and-exchange.md`) returns the per-time arrays at a normalized time: moved positions, effective opacities, normalized quaternions, scales and the SH coefficients cut to a degree, plus the culling mask `passes_cull`. `openftgs.io.write_ply` writes such arrays as one static PLY in the layout 3DGS viewers read (Q-18: `x y z nx ny nz f_dc_0..2 f_rest_*` channel-major, opacity as a logit, scales as logs, `rot_0..3`, normals zero). Both are library calls: `ftgs export` writes only `ftgs-if`, and `write_ply` does not drop culled rows, so a caller that wants smaller files filters by `passes_cull` first. The owner chose this static snapshot over a PLY with 4D extras (Q-18's alternatives), so a PLY sequence is openftgs's only route into viewers that read nothing but PLY.

Size per primitive per frame at L = 3: 62 float32 properties, 248 bytes. A sequence of F frames of N primitives, uncompressed and unfiltered, is 248·N·F bytes:

| N | F | PLY sequence | One exchange model | Centers only, baked positions (12 B) |
| --- | --- | --- | --- | --- |
| 100 k | 60 | 1.5 GB | 26 MB | 72 MB |
| 500 k | 300 | 37 GB | 132 MB | 1.8 GB |
| 1 M | 300 | 74 GB | 264 MB | 3.6 GB |

Filtering by `passes_cull` removes the primitives that are invisible at that frame, which can be most of them in a scene with much motion and short durations and few of them in a mostly static one; compressed splat formats shrink what remains (section 5.5, to be measured against their documentation). Even so, baked sequences stay one to two orders of magnitude above the single exchange model that live evaluation (section 4, architecture A) needs, and they lose the continuous time of section 2.5.

### 3.4 Offline frames

`ftgs render MODEL --request REQUEST.json --out RENDERS.safetensors` answers a render request (§2.2): items of an id, a raw timestamp and a camera, rendered to `rgb/<id>` and `alpha/<id>` float32 tensors, unclamped. `scripts/view_renders.py` is the one request writer in the repository (for a scene's test views) and turns renders into 8-bit PNGs by clamping to [0, 1]; a camera path along interpolated training cameras would be a variant of it. A video encoder then makes a file any player seeks in. The route is exact and non-interactive: the camera is fixed when the frames are rendered (section 4, architecture C).

## 4. Three architectures

**A. Live evaluation in the player.** The player loads one exchange model and evaluates E1 and E4 itself at the current τ, every frame. Costs: four extra quantities per primitive in GPU memory, two lines in the per-primitive stage, and a depth sort that must use the moved positions, so a renderer that sorted once from static positions now re-sorts each frame (a sort over N keys, with the moved positions computed in the sorter or on the GPU). Optional: the bucket rule of section 2.2 to touch only the primitives alive near τ. Benefits: one file, continuous time, no baking step, and the smallest footprint by one to two orders of magnitude (section 3.3). This is the architecture of every route in section 6 and of any viewer in section 5 that exposes a per-primitive attribute hook.

**B. Baked snapshots.** A Python step calls `at_time` and `write_ply` once per frame (or writes another static format) and a static viewer plays the files as a sequence: preloaded, streamed or swapped. Costs: the sizes of section 3.3, a conversion step per model, time quantized to the baked frames, and whatever per-file load and sort the viewer does at each swap. Benefits: works with viewers that have no extension hook at all, and with the many static formats their ecosystems read. The centers tier bakes cheaply (positions and colors per frame) and some timeline-native point programs play exactly that.

**C. Offline video.** A render request along a camera path at the frame times, `ftgs render`, PNGs, a video encoder. Costs: no free camera; a new path means a new render. Benefits: the exact look of `ftgs render`, play, pause and seek from any video player, and nothing to install. It is the baseline the interactive routes are measured against, and the right answer when the aim is a clip rather than exploration.

## 5. Survey of existing software

To be filled by the survey (section 5.1 Python and UI servers, 5.2 browser, 5.3 engines and DCC tools, 5.4 native, 5.5 static splat formats for baked sequences), one table per platform with the columns: license class · input formats · time or sequence support · per-primitive attribute and time-uniform hook · camera convention · fidelity gaps · tier served · effort to adapt · notes. Every entry rests on the candidate's license file and its own documentation pages, each with a source-log row.

## 6. Custom renderer routes, an effort ladder

Each route is a program outside openftgs that imports or reads what section 3 describes. Effort is a rough order of magnitude for one engineer; the survey's finalists per platform refine it.

**(i) A Python loop around `render()`, on the training machine.** A small program loads the model with `FreeTimeGaussians.load` (a checkpoint or an exchange model, on CUDA), receives a camera pose and a time from a browser-based UI library, builds the spec's camera from the pose (section 2.4), calls `render(model, camera, τ)` and streams the clamped image back; a time slider, play and pause, a speed control and the frame count from Δt_f or the user complete it. The look is exactly `ftgs render`'s: same backend, same constants, a slice per frame (`render()` is the per-frame call; `render_request`'s temporal index is for batches). Rendering is not the bottleneck at interactive sizes (§2.4 and B-7 target hundreds of frames per second for one camera at 1080p); encoding and transport of the image stream are. The centers tier is a point set of moved positions, colored by the DC band, pushed to the same UI. Effort: a few hundred lines; days. Needs a CUDA machine where the model lives, and a UI library whose license allows it (section 5.1).

**(ii) A browser renderer with the temporal attributes.** Take a permissively licensed WebGL or WebGPU Gaussian-splat renderer that lets a scene carry extra per-primitive attributes and lets the per-primitive stage be extended (the hook of section 2.2), parse the exchange model in JavaScript (section 3.1), add `time_center`, `duration` and `velocity` as attributes and τ as a uniform, write E1 and E4 into the per-primitive stage, make the depth sort read the moved positions, convert the camera (section 2.4), and attach a timeline. The look is approximate (section 2.6). Effort: days to a few weeks, depending on how much the renderer's hooks expose; the model's size on the wire (264 bytes per primitive at L = 3) calls for quantization or a reduced SH degree for sharing, which the static-format specifications of section 5.5 show how to do. Shareable as a link with no GPU server. The centers tier is a point renderer with the same two lines and no sort.

**(iii) A rasterizer from scratch.** A native (Rust or C++, Vulkan, Metal or wgpu) or WebGPU-compute implementation of E5 and E6 with the temporal equations, matching the spec's constants if wanted (section 2.6), with its own camera controls and timeline. Effort: weeks to months. The only route that could reach the spec's cross-rasterizer tier outside Python, and the one with the least reuse; not recommended unless a native application is a hard requirement and no permissive native viewer of section 5.4 offers a hook.

## 7. Recommendations

To be written after the survey, one per platform and one overall.

## 8. What openftgs would need

To be completed after the survey. Owner decisions, not proposals; each route's needs are listed against what exists:

- Routes reading the exchange model (A, i, ii) need nothing: `ftgs export --format ftgs-if` exists and `docs/modules/io/file-formats.md` describes the file.
- Route B needs the PLY snapshot per time from outside Python; `write_ply` is library-only and the CLI exports only `ftgs-if` (Q-18 deferred PLY's command-line form). Exposing it is a decision for the owner, recorded as a Q-n candidate on #352 if wanted.
- Route C needs a request writer for camera paths; `scripts/view_renders.py` shows the request's shape for test views.

## 9. Sources

Source-log rows 223 to (end of survey) in `docs/clean-room/source-log.md`.

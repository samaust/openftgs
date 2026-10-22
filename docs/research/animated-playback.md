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

**The project page's renderer.** The paper's project page links a renderer of its own, which spec D-6 and Appendix A #2 record as not consulted and which CLAUDE.md kept closed to the implementer. On 2026-10-22 the owner ruled in session that it may be read, documentation and source, as MIT software, except code it imports that the clean-room rules classify as not allowed (source-log row 270); the rule differs from spec §0, D-6 and Appendix A #2 until a new export carries it. The owner also relayed its credits. Of the nine projects named, GaussianSplats3D and antimatter15/splat were already in section 5.2; gsplat.tech is a web viewer demo whose license and source this survey did not check, so it is not evaluated; the 3DGS reference rasterizer and a fork of it are under the non-commercial license of row 237, documentation only, and are rasterizers rather than players; 4K4D is a paper of the same group, literature only, its code not sought; the framework that is never allowed stays unopened; and nvdiffrast is a mesh rasterizer under a research-only license, not a player (row 271). The renderer itself is evaluated in section 5.4.

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

**Centers tier.** Draw the moved position of each primitive as a point, colored by the SH DC band alone (E2 at degree 0: C₀·`sh0` + 0.5, clamped at 0, with C₀ = 0.28209479177387814) and either culled or faded by its effective opacity. This needs no splatting and no sort, and any program that draws a colored point set per time can do it; it shows where the primitives are and when they live, not what the render looks like.

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

Everything in this section comes from each candidate's license file and its own documentation pages, each with a source-log row (section 9); nothing was run, and no candidate's source code was read. A survey of this kind reports what documentation states, so "none documented" means the pages read do not describe the feature, not that it is absent.

**Columns.** *Reads*: the inputs its documentation names. *Time, as documented*: built-in playback or sequence support. *Hook*: the documented way to attach per-primitive data and a time input and to change centers and opacity per primitive (section 2.2). *Tier*: splats, centers or both. *Architecture*: the letter of section 4 the candidate serves. *Effort*: none (works as it is, perhaps after a baking step), small (days), medium (weeks), large (months), for one engineer.

**Fidelity.** No candidate documents the constants of section 2.6, so the fidelity-gaps column of the plan collapsed to "unknown from documentation" for every one and is left out of the tables; the structural gaps of section 2.6 hold for all of them. The pages did yield two concrete data: Spark draws each primitive's footprint to √8 ≈ 2.83 standard deviations, against the spec's opacity-aware extent capped at 3.33 (its advice to turn off the browser renderer's multisampling says nothing about D-6's rasterize mode); and gsplat's own viewer example clamps the output to [0, 1] before display, as any player must (E7).

### 5.1 Python and UI servers

| Candidate | License | Reads | Time, as documented | Hook | Tier | Architecture | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| viser | Apache-2.0 | arrays sent from Python | none built in; sliders, buttons, checkboxes and a hotkey command; its Record3D demo is a complete playback UI (timestep slider, play checkbox, FPS slider, space bar) | the splat handle's `set_gaussians(centers, covariances, rgbs, opacities)`, documented as the fast path for per-frame updates; point clouds per frame | both | A, with Python evaluating E1 and E4 | small |
| nerfview + `render()` | Apache-2.0 | a checkpoint or exchange model through openftgs | none built in; a render callback per camera change; a τ control added through viser | none needed: Python evaluates the model inside the callback | splats, exact | A, server-rendered | small |
| gsplat 1.5.3 viewer example | Apache-2.0 | its own `.pt` files | none | none | pattern only | — | — |
| rerun | MIT / Apache-2.0 | arrays logged per time, `.rrd` files | native: every log call sits on timelines; play, pause, step, loop, scrub, rate multiplier | none (no shader access); `Points3D` or the unstable `GaussianSplats3D` logged per frame | both | B, baked per step into memory or a file | small |
| ParaView | BSD-3-Clause | numbered files as a file series | native: Time Manager and VCR controls, Python-scriptable | none | centers | B | small (a point-PLY writer is needed) |
| Blender + Point Cloud Visualizer | Blender GPL; the add-on has no license file (0.9.x) and a paid 2.0+ version: documentation only | a numbered PLY sequence | native: preloaded sequence played on Blender's timeline | none | centers | B | small |
| Open3D, Polyscope | MIT; not checked | arrays | none; per-frame callbacks | none | centers | A by hand | not evaluated (general knowledge) |

**viser.** The client renders Gaussian splats in WebGL from arrays Python sends (`add_gaussian_splats`, experimental; the handle's `set_gaussians` updates all four arrays at once and may change the count). Python evaluates E1 and E4 per frame, builds covariances from E5 and DC colors from E2 at degree 0, and sends about 52 bytes per primitive per frame; the performance page warns against large per-frame payloads, so this path suits tens of thousands of primitives, not a million, and shows the DC band only. The centers tier is `add_point_cloud` per frame, or the Record3D pattern (one point set per frame added up front, visibility toggled by the slider). Cameras follow the COLMAP/OpenCV convention and the camera handle exposes a camera-to-world pose (wxyz, position) with the vertical field of view, so the spec's camera is its inverse with no axis change (section 2.4); the world is +Z up by default and settable.

**nerfview + `render()`.** nerfview wires a viser client to a Python callback that receives a camera-to-world matrix, a K from the viewer's field of view and size, and returns a uint8 image; gsplat 1.5.3's own viewer example inverts that matrix straight into gsplat's world-to-camera input with no axis flip, so the callback's camera is in OpenCV axes and `Camera(K, inverse(c2w), width, height)` is the spec's camera. A K built from a field of view puts the principal point at (W/2, H/2), which is already the continuous convention of §1.4. A τ slider and a play checkbox from viser's GUI, re-rendering on change and advancing τ by one frame interval per tick, complete the player; `render()` warns when τ leaves [τ₀, τ₁]. The look is `ftgs render`'s; the stream is JPEG over a websocket.

**rerun.** Every `rr.log` call carries the current time of each timeline (`rr.set_time` with a sequence index, a timestamp or a duration), and the viewer plays, pauses, steps, loops and scrubs with a rate multiplier. `Points3D` with per-point colors and radii, logged once per frame with the moved positions and DC colors, is the centers tier with a native timeline, about 16 bytes per primitive per frame, streamed live over gRPC or saved to an `.rrd` file. The `GaussianSplats3D` archetype (centers, linear scales, quaternions, RGBA colors with opacity as alpha, optional SH coefficients) is marked unstable and its page says nothing about time; logged per frame like any archetype it would be architecture B in memory, at about 224 bytes per primitive per frame at L = 3 (44 without SH), so hundreds of thousands of primitives over hundreds of frames exceed what a session should hold. The 3D view has first-person eye controls with WASDQE movement and an eye-up vector; the renderer's splat conventions are unknown.

**ParaView.** Numbered files are grouped into a file series and read as a time series, with a Time Manager, VCR controls and Python control of the animation scene. Points colored by a vertex array need PLY files of positions and RGB per frame, which openftgs does not write (`write_ply` writes the splat layout); the PLY reader and "Color By" rest on general knowledge, not on the pages read. No splats.

**Blender + Point Cloud Visualizer.** The 0.9.x add-on, unmaintained and for Blender 2.8x, preloads a numbered PLY sequence (positions, optional normals, RGB) and plays it on the timeline; a maintained 2.0+ version is sold separately with its own documentation, not read. Blender is GPL, so every Blender route is documentation only. Centers tier only.

### 5.2 Browser

| Candidate | License | Reads | Time, as documented | Hook | Tier | Architecture | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PlayCanvas engine | MIT | PLY, compressed PLY, SOG, streamed SOG, SPZ, GLB | none for splats (camera animation only) | extra per-splat texture streams (`addExtraStreams`), a work-buffer modifier (`modifySplatCenter`, `modifySplatRotationScale`, `modifySplatColor` with a vec4 color) in the Copy stage, uniforms by `setParameter`, GPU passes by `GSplatProcessor` | splats | A | small to medium |
| Spark (Three.js) | MIT | ply, sogs, spz, splat, ksplat, zip, rad | none; an `onFrame` callback and uniforms updatable every frame | dyno `objectModifiers` / `worldModifiers` over a per-splat struct (center, scales, quaternion, rgba); per-splat data through a second packed splat set or a data texture (building blocks, no example) | splats | A | medium |
| Chronosplat | GPL-3.0-or-later: a separate program, documentation only | a PLY sequence, converted to per-frame SOG plus a manifest | native timeline: play, pause, scrub, loop, fps override; bounded memory window | none | splats | B | none beyond baking |
| Babylon.js | Apache-2.0 | PLY, splat, spz, SOG, streamed SOG | none; parts move rigidly inside one sort; CPU `updateData` | vertex-stage plugin insertion points whose inputs the page does not describe | splats | A by code, undocumented | medium to large |
| SuperSplat | MIT | PLY and the formats above | camera animation; "animating splats" mentioned without detail | none | — | — | an editor and publisher, not a sequence player |
| GaussianSplats3D | MIT | ply, splat, ksplat | none | none; no longer developed, points to Spark | splats | — | not evaluated further |
| gsplat.js | MIT | splat (no SH), ply via a converter | none | none | splats | — | not evaluated further |
| splaTV | MIT | another 4D method's files | undocumented | — | — | — | not evaluated: no documentation; source not read |

**PlayCanvas.** The documented pieces of architecture A are all there. `format.addExtraStreams([{name, format}])` adds per-splat texture streams (RGBA8, RGBA16F, RGBA32F), read in shaders by `load{Stream}()`: two RGBA32F streams hold the time center, the duration and the velocity. `entity.gsplat.setWorkBufferModifier({glsl, wgsl})` defines `modifySplatCenter`, which applies E1 with a τ uniform set by `setParameter`, and `modifySplatColor`, whose vec4 color carries the opacity in its alpha (the pages never say so; an inference from "color and opacity" on the custom-shaders page), where E4 scales it. The work buffer is copied, sorted and then rendered, so a center moved in the Copy stage is sorted at its moved position (an inference from the documented order; the pages do not say when the copy re-runs as a uniform changes, and `GSplatProcessor.process()` is the documented explicit alternative: a GPU pass per frame writing a position stream and a color stream into a resource the pipeline then sorts and draws). Extra streams are not available on streamed SOG resources. Both WebGL2 (GLSL) and WebGPU (WGSL) are served. Loading an exchange model means a loader from safetensors into procedural splat data plus the two streams, or a conversion to PLY with a sidecar. The camera convention is not stated on the pages read; a GL-style camera converts as in section 2.4.

**Spark.** Modifiers are dyno shader programs over a per-splat struct with center, scales, quaternion and rgba, with uniforms (`dynoFloat` and the like) whose `.value` is assigned every frame and an `onFrame({mesh, time, deltaTime})` callback; a time uniform is therefore the application's, since the library offers no clock. Per-splat temporal quantities need a lookup by the splat's index, for which the standard library offers samplers, `texelFetch` and `readPackedSplat(packedSplats, index)` but no per-splat example. Sort order comes from GPU-computed distances over the generated splats, bucket-sorted in a worker and lagging a frame, and the generator contract asks for index-stable splats across frames, so moving splats are the designed case, though no page says in words that the sort reads modified centers. Formats are the broadest of the survey, with level-of-detail streaming; the camera is Three.js's (y up, looking down −z), and the loader example's 180° rotation about x is the conversion of section 2.4. Footprints reach √8 standard deviations (a documented fidelity gap against the 3.33 cap). Loading an exchange model means parsing safetensors in JavaScript into packed splats and a data texture.

**Chronosplat.** A static-file player of per-frame splat sequences on Three.js and Spark: its converter takes a PLY sequence "from any 3DGS exporter", emits per-frame SOG and a manifest, and offers frame decimation, SH reduction, opacity thresholding and floater filtering; the timeline plays, pauses, scrubs and loops with an fps override, frames stream through a bounded memory window, and it needs Node 22+, Python 3.10+ and WebGL2. Its author calls it barely tested. Under GPL-3.0-or-later it can only ever be used as a separate program, never adapted, and the survey took nothing from its source. Fed with `write_ply` frames (the 3DGS viewer layout it expects; untested), it is the no-code browser route for a short clip.

**Babylon.js** documents rigid motion of parts within one global sort and a CPU rewrite of splat data, but not what its vertex-stage plugin points receive or whether centers can be moved there; architecture A would mean reading its source, which Apache-2.0 allows but this survey did not do. The remaining rows are static viewers without hooks (GaussianSplats3D itself points to Spark), or, for splaTV, an undocumented viewer of another method's files whose source the owner's rule keeps closed.

### 5.3 Game engines and DCC tools

| Candidate | License | Reads | Time, as documented | Hook | Tier | Architecture | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unity Gaussian Splatting | MIT | PLY, SPZ | none | none documented; the shaders are MIT source | splats | A by modifying the package | medium to large; no significant development since 2023 |
| An Unreal plugin on a marketplace | unknown (listing not reachable, HTTP 403) | PLY sequences, per a search snippet | native, per the snippet | — | splats | B | not evaluated |
| Blender + KIRI 3DGS Render | GPL-2.0 and Blender GPL: documentation only | 3DGS PLY | proxy-based animation and baking of rigged splat sequences (experimental); exports animated PLY sequences; sequence import not documented | — | splats | B, import undocumented | not evaluated further |
| Blender + Point Cloud Visualizer | see 5.1 | PLY sequence | native | — | centers | B | small |
| Godot | — | — | — | — | — | — | not surveyed |

No engine or DCC route plays the model as openftgs writes it. Unity Gaussian Splatting renders PLY and SPZ on D3D12, Metal and Vulkan with no animation, sequence or per-splat customization documented; its MIT shaders could take E1 and E4, which is a modification of the package rather than a hook. The KIRI add-on brings splats into Blender as proxies that follow Blender's transforms and animation and exports animated PLY sequences, but documents no sequence import; Blender being GPL, every Blender route is documentation only.

### 5.4 Native

| Candidate | License | Reads | Time, as documented | Hook | Tier | Architecture | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Brush | Apache-2.0 | ply, compressed ply; a zip of splat files, shown as an animation | native for the zip | none documented | splats | B | none beyond baking |
| Splatapult | MIT | 3DGS PLY | none | none; C++ and OpenGL source to extend | splats, with an SfM-points toggle | A by code | large |
| A rasterizer from scratch | — | the exchange model | — | — | both | A | large (route iii) |

Brush, a trainer and viewer on WebGPU-class graphics for desktop, Android and recent Chrome, displays a zip of splat files as an animation, which makes it the native no-code route for a baked sequence (it also advertises a delta-frame PLY for two other dynamic methods, names only). Splatapult is a static OpenGL viewer with OpenXR support whose MIT source could be extended. Apple's platform splat support, named in a search snippet, is proprietary and was not evaluated.

### 5.5 Static splat formats for baked sequences

| Format | License of the tooling | Size against PLY | Keeps | Writer from Python |
| --- | --- | --- | --- | --- |
| PLY, 3DGS viewer layout | — | 1× (248 B per primitive at L = 3) | everything, float32 | `openftgs.io.write_ply` |
| compressed PLY (SuperSplat, gsplat) | MIT tools | not stated on the pages read | — | splat-transform |
| SPZ | MIT | about 10×, quantized: 24-bit positions, 8-bit log scales, three 10-bit quaternion components, 8-bit opacity and color, SH packed to 5 and 4 bits | SH degree 0 to 4 | spz's Python bindings, splat-transform |
| SOG | MIT tools | 15 to 20×, lossy; a streamed level-of-detail variant | SH optional (bands can be stripped) | splat-transform (Node) |
| KSPLAT | MIT | levels: 32-bit, 16-bit, SH to 8-bit | — | GaussianSplats3D's tools (no longer developed) |
| GLB with the Khronos splat extension | Khronos | about 1×, lossless | — | splat-transform |

None of these formats carries time: a sequence is a set of files plus whatever binds them, a manifest (Chronosplat), a zip (Brush), numbered names (ParaView, Point Cloud Visualizer) or timeline log calls (rerun). With SOG at 15 to 20× the sequences of section 3.3 shrink to about 4 to 5 GB for a million primitives over 300 frames and to under 100 MB for a hundred thousand over 60, before `passes_cull` filtering; the conversion step and the frame quantization of time remain.

## 6. Custom renderer routes, an effort ladder

Each route is a program outside openftgs that imports or reads what section 3 describes. Effort is a rough order of magnitude for one engineer; section 5 names the finalists.

**(i) A Python loop around `render()`, on the training machine.** A small program loads the model with `FreeTimeGaussians.load` (a checkpoint or an exchange model, on CUDA), receives a camera pose and a time from a browser-based UI library, builds the spec's camera from the pose (section 2.4), calls `render(model, camera, τ)` and streams the clamped image back; a time slider, play and pause, a speed control and the frame count from Δt_f or the user complete it. The look is exactly `ftgs render`'s: same backend, same constants, a slice per frame (`render()` is the per-frame call; `render_request`'s temporal index is for batches). Rendering is not the bottleneck at interactive sizes (§2.4 and B-7 target hundreds of frames per second for one camera at 1080p); encoding and transport of the image stream are. The centers tier is a point set of moved positions, colored by the DC band, pushed to the same UI. Effort: a few hundred lines; days. Needs a CUDA machine where the model lives. Finalists: viser with nerfview's callback pattern (section 5.1), whose cameras are already in the spec's convention.

**(ii) A browser renderer with the temporal attributes.** Take a permissively licensed WebGL or WebGPU Gaussian-splat renderer that lets a scene carry extra per-primitive data and lets the per-primitive stage be extended (the hook of section 2.2), parse the exchange model in JavaScript (section 3.1), add the time center, duration and velocity as per-primitive data and τ as a uniform, write E1 and E4 into the per-primitive stage, make the depth sort read the moved positions, convert the camera (section 2.4), and attach a timeline. The look is approximate (section 2.6). Effort: days to a few weeks, depending on how much the renderer's hooks expose; the model's size on the wire (264 bytes per primitive at L = 3) calls for quantization or a reduced SH degree for sharing, which the formats of section 5.5 show how to do. Shareable as a link with no GPU server. The centers tier is a point renderer with the same two lines and no sort. Finalists: PlayCanvas (extra streams, a pre-sort center modifier, a time uniform, GPU passes; WebGL2 and WebGPU) and Spark (per-frame uniforms and modifiers, the broadest formats; Three.js), both MIT (section 5.2).

**(iii) A rasterizer from scratch.** A native (Rust or C++, Vulkan, Metal or wgpu) or WebGPU-compute implementation of E5 and E6 with the temporal equations, matching the spec's constants if wanted (section 2.6), with its own camera controls and timeline. Effort: weeks to months. The only route that could reach the spec's cross-rasterizer tier outside Python, and the one with the least reuse; not recommended unless a native application is a hard requirement and extending Brush or Splatapult (section 5.4) is ruled out.

## 7. Recommendations

**Python on the training machine.** Build route (i): a viser client with a nerfview-style render callback around `render()`, a τ slider over [τ₀, τ₁], a play checkbox stepping one frame interval per tick, a speed control, and the scene's training cameras as preset poses. It is the only route with `ftgs render`'s exact look, it costs a few hundred lines, and viser's cameras are already in the spec's convention. For the centers tier in the same window, send the moved positions as a point set per frame (small models) or preload one point set per frame and toggle visibility (the Record3D pattern). To hand a result to someone without a GPU, log the centers tier per frame to a rerun `.rrd` file, whose viewer has the timeline built in; log splats to it only for small models, through an archetype marked unstable.

**Browser, shareable.** Build route (ii) on PlayCanvas first: its documentation names every piece needed (per-splat streams for the three temporal quantities, a center modifier that runs before the sort, a time uniform, a GPU pass as the explicit alternative) on WebGL2 and WebGPU. Spark is the second choice: per-frame uniforms and modifiers over a per-splat struct and the broadest format and streaming support, with the per-splat lookup to build from primitives. Both are MIT. Either route needs a JavaScript reader of the exchange model and a quantization step for sharing. For a short clip with no code, bake frames with `at_time` and `write_ply`, filtered by `passes_cull`, and play them in Chronosplat, a GPL program used as is.

**Game engines and DCC tools.** No documented hook or sequence import exists; Unity Gaussian Splatting (MIT) is the one engine where architecture A is feasible, by modifying its shaders, in a package with little recent development; Blender serves baked sequences only (splat import of sequences undocumented, the centers tier through Point Cloud Visualizer), and only through documentation, since Blender is GPL. Not a first target.

**Native.** Brush plays a zip of baked splat files as an animation with no code, size permitting; a native architecture A means route (iii) or extending Splatapult or Brush under their permissive licenses, a large effort. Only if a native application is required.

**Overall.** Route (i) now, since it is small and exact; route (ii) on PlayCanvas when a shareable player matters; baked sequences (Brush, Chronosplat, rerun) for short clips and inspection; architecture B at scale and route (iii) not at all.

## 8. What openftgs would need

Owner decisions, listed against what exists; none is proposed here.

- **Routes reading the exchange model (A, i, ii)** need nothing: `ftgs export --format ftgs-if` exists and `docs/modules/io/file-formats.md` describes the file. A note for third-party loaders on the SH basis (gsplat's order and signs, E2) and on the +0.5 pixel convention (§1.4) would spare them a guess; both are already stated in the spec.
- **Where route (i)'s program lives.** The viewer is out of scope for the package; whether a script like `scripts/view_renders.py` may hold it, or it lives in a separate repository, is the owner's call.
- **Baked routes (B)** need a per-time PLY export reachable from the command line, with `passes_cull` filtering, a frame range and a frame rate. The spec's CLI exports `ftgs-if` only and Q-18 kept PLY a library call, so this is a spec change, not a gap, and it is not recorded as a Q-n candidate. The centers tier needs a plain point PLY (positions and RGB), which nothing in openftgs writes. Conversions to SOG or SPZ are external MIT tools.
- **Route C** needs a request writer for camera paths, a variant of `scripts/view_renders.py`.

## 9. Sources

Source-log rows 223 to 271 of `docs/clean-room/source-log.md`: two general-knowledge rows for the conventions of sections 2.4 and 3.1 (223, 224), the license files (225 to 237, 251, 252, and within 254 to 259), the five searches (238 to 242), the documentation passes (243 to 261, 263, 265 to 268), two general-knowledge rows for screening (264, 269), the project page's renderer, not opened at first (262), the owner's rule that opens it and the credits the owner relayed (270), and the general-knowledge row that classifies the names in them (271).

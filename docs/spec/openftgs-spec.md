# FreeTimeGS — Clean-Room Specification & Architecture

Oct 5, 2026 · @Samuel Austin

## 0. Purpose, provenance and notation

This document specifies an Apache-2.0, clean-room Python implementation of FreeTimeGS (Wang et al., CVPR 2025) written from the paper alone. Together with the sources allowed by the clean-room rules below, it is all an implementer needs.

The paper text used is arXiv 2506.05348 v2, which carries the supplementary material as Appendices A–B. Equation numbers below follow that version. The CVF open-access PDF returned HTTP 403 to our fetcher, so it was not compared line by line (Appendix A).

**Clean-room rules.** Owner rule of 2026-10-07. Implementers may consult the paper, its supplementary material and project page; papers it cites; general Gaussian-splatting literature; and the documentation and data files of the datasets the paper evaluates on (Neural3DV, ENeRF-Outdoor, SelfCap). What they may read of third-party software depends on its class in the table below; software that fits two classes takes the stricter one. Never allowed: anything from another FreeTimeGS implementation, including the authors' own code and the EasyVolcap framework that the SelfCap card points to (source code, notebooks, forks or issue threads), and any code under GPL or AGPL. If a forbidden source turns up, implementers stop and report it. Every source consulted goes into Appendix A with what was taken from it.

| Class | Software | May read |
| --- | --- | --- |
| Imported or vendored by the package | PyTorch, gsplat, safetensors, lpips and the torchvision networks it loads, RoMa (`romatch`), fused-ssim, NumPy, Pillow, PyYAML, SciPy; the vendored RoMa code, including its copy of the DINOv2 layers, and what it imports (torchvision, einops, loguru, fused-local-corr) | documentation and source code |
| Replaced by the package's own code, even where tests still call it as a reference | OpenCV, scikit-image, torchmetrics | documentation only, so the replacement is written without seeing their code |
| Run as a separate program | ffmpeg, COLMAP, BackgroundMattingV2 | documentation only |
| Not imported by the package: transitive dependencies, and software evaluated but not adopted | e.g. the DINOv2 repository (RoMa's vendored copy of its layers is in the first class), kornia (imported only inside a vendored helper that the package never calls), decord, torchcodec | documentation and license files, plus packaging and build files (CI workflows, build scripts, CMake files) read only to establish licenses |

gsplat stays in the first class: the package imports it, and the reference backend was written while its source was already allowed. The owner's rule named only imported and replaced software; the owner confirmed the last two rows on 2026-10-07, and the same day allowed dataset documentation and data files and, for license checks only, the packaging and build files of software in the fourth class. Documentation tools that index whole repositories can return source code, so for software in the second to fourth classes implementers read the project's own documentation pages, never repository-indexed snippets (Appendix A, #36).

**Provenance tags.** Every rule carries one tag, so an implementer can tell what is fixed by the paper and what can be changed.

| Tag | Meaning | May the implementer change it? |
| --- | --- | --- |
| \[P §x, Eq. n\] | Stated in the paper | No |
| \[P→3DGS\] | Paper says "same settings as 3DGS"; value taken from 3DGS / gsplat defaults | Only with a logged reason |
| \[L: source\] | Standard Gaussian-splatting practice from an allowed source | Only with a logged reason |
| \[D-n\] | Design decision made here; rationale, alternatives and a detecting test are in Appendix B | Yes, via config, never silently |

**Notation.** The paper reuses s, N and t for different things; this document renames the second uses.

| Symbol here | Paper symbol | Meaning | Domain |
| --- | --- | --- | --- |
| i, N | i, N | primitive index, number of primitives | — |
| τ | — | raw timestamp in dataset units (e.g. seconds); T\_τ = τ₁ − τ₀ is the training time span | ℝ |
| t | t | normalized scene time, t = (τ − τ₀)/(τ₁ − τ₀) \[D-1\] | ℝ (training range \[0, 1\]) |
| μx | 𝛍x | primitive position at its own time | ℝ³ |
| μt | μt | primitive time center | ℝ |
| s | s (Eq. 4) | duration (std. dev. of temporal opacity) | ℝ>0 |
| v | 𝐯 | velocity, world units per unit of t | ℝ³ |
| q, R | R (via Σ) | orientation quaternion (w, x, y, z); R = R(q/‖q‖) | ℝ⁴, SO(3) |
| a, S | S | per-axis scales, S = diag(a) | ℝ³>0 |
| Σ | 𝚺 | 3D covariance R S Sᵀ Rᵀ | 3×3 SPD |
| σ | σ | base opacity | (0, 1) |
| σt(t) | σ(t) | temporal opacity | (0, 1\] |
| c\_lm, L | 𝐜lm, L | SH coefficients (RGB) and SH degree | ℝ³ |
| μx(t) | 𝛍x(t) | moved position at time t | ℝ³ |
| ρ | s (Eq. 7) | relocation sampling score | ℝ≥0 |
| g | ▽g | accumulated spatial gradient used by ρ | ℝ≥0 |
| u | t (Sec. 3.2, velocity schedule) | training progress, u = (it − 1)/(T\_it − 1), with it = 1 … T\_it the iteration index \[D-41\] | \[0, 1\] |
| ηv(u) | λt | velocity learning rate | ℝ>0 |
| K\_reloc | N (Sec. 3.2) | relocation period in iterations | ℕ |
| sg\[·\] | sg\[·\] | stop-gradient | — |

Stored (raw) parameters carry a hat: ô is the opacity logit, â = log a, ŝ = log s.

**Document map.** Part 1 is the specification, Part 2 the package architecture, with performance in §2.4, and Part 3 the test plan, with performance benchmarks in §3.4. Appendix A is the source log and Appendix B the ambiguity log.

**Revisions.** 2026-10-06: the project owner set D-3 (SH degree is a user parameter, 0–3, default 3), D-10 (4D regularization for the whole run) and D-18 (the primitive count grows). The affected sections, tests and log rows are updated, and D-42 to D-45 were added for the growth rules. Later the same day: Neural3DV frames are decoded by a user-installed ffmpeg (D-48); the clean-room rules now allow the RoMa, OpenCV, lpips and safetensors documentation; the frame interval is measured per camera (D-46); initialization handles unsynchronized cameras (D-47); and the API can resize the primitive set. Later that evening, at the owner's request, decord was evaluated and rejected, and a performance design was added: §2.4, benchmarks B-1 to B-8 in §3.4, D-49 to D-57 and U-28 to U-33. 2026-10-07: the owner chose RoMa's default model (D-58), a SelfCap converter now (D-59, D-60), OpenCV out of the core (D-61), the name openftgs (D-62) and new clean-room rules (§0); U-34 to U-37 and C-4 were added. Later that day the owner confirmed the two rows of the clean-room table that cover separate programs and software the package does not import, found no existing openftgs distribution on PyPI, and supplied hair-calib, yoga-calib, a corgi-release listing and its sync.json; these settled the SelfCap camera format and the missing background images, and gave the reading of the sync.json unit that C-5 checks (D-59, D-63, D-64). The owner then chose to vendor RoMa's inference code without its OpenCV import (D-65, U-38), so no install contains OpenCV and no LGPL sign-off is needed. The owner also confirmed that the hosted corgi-release archive itself fails gzip's CRC check (D-63 bullets in §1.4). An independent review by a fresh agent then reported 21 findings and no clean-room incident (Appendix A, #37). Each was checked against its source and fixed; among the changes, initialization uses training views only (D-66), evaluation follows Appendix B.1 (D-67), and the configuration schema is defined (§2.1). The owner kept D-10's whole-run default, now with a reg\_until setting, set the initial budget to 1,000 points per frame (D-24), and allowed dataset documentation and, for license checks, packaging files (§0). One documentation query returned scikit-image source code, from which nothing was taken (#36). A second independent review, of revision 455, reported 27 findings, again with no clean-room incident (Appendix A, #39). All were fixed: D-68 (LPIPS input range) and D-69 (fastest-motion frames) were added, and D-14 to D-16, D-44, D-45, D-47, D-49, D-54 and D-65 were revised. No owner decisions are open.

## 1. Specification

### 1.1 Per-primitive parameters

Each primitive has the paper's eight learnable quantities (position, time, duration, velocity, scale, orientation, opacity, SH coefficients) \[P §3.1\]. They are stored as nine float32 tensors; SH is split into DC and higher bands so each can have its own learning rate. The tensor names below are normative: checkpoints and the optimizer use them.

| Tensor | Paper quantity | Shape | Stored form | Activation → physical value | Initial value | Tag |
| --- | --- | --- | --- | --- | --- | --- |
| `means` | position μx | \[N, 3\] | raw, world units | identity | init point position | \[P §3.1\] |
| `times` | time μt | \[N\] | raw, normalized time | identity | (τᵢ − τ₀)/(τ₁ − τ₀) | \[P §3.1\], \[D-1\] |
| `log_durations` | duration s | \[N\] | ŝ = log s | s = exp(ŝ) | log s₀ (input value, else D-21) | \[P Eq. 4\], \[D-2\] |
| `velocities` | velocity v | \[N, 3\] | raw, world units per unit of t | identity | k-NN estimate, else 0 | \[P §3.1, §3.2\], \[D-23\] |
| `log_scales` | scale a | \[N, 3\] | â = log a | a = exp(â) | log of mean 3-NN distance | \[L: 3DGS §5.1\], \[D-25\] |
| `quats` | orientation q | \[N, 4\], order (w, x, y, z) | raw, unnormalized | R(q/‖q‖) | (1, 0, 0, 0) | \[L: 3DGS §4; gsplat conventions\]; init \[D-5\] |
| `opacity_logits` | opacity σ | \[N\] | ô = logit σ | σ = sigmoid(ô) | logit(0.1) | \[L: 3DGS §5.1\], \[D-5\] |
| `sh0` | c₀₀ | \[N, 1, 3\] | raw | color by E2 | (rgb − 0.5)/C₀ | \[L: gsplat\], \[D-4\] |
| `shN` | c\_lm, l ≥ 1 | \[N, (L+1)² − 1, 3\] | raw | color by E2 | 0 | \[D-3\] |

C₀ = 0.28209479177387814 (the l = 0 real SH constant). L is a user parameter, sh\_degree ∈ {0, 1, 2, 3}, default 3 \[D-3\]. At L = 3 a primitive holds 64 floats (256 bytes); at L = 0, shN has shape \[N, 0, 3\] and no optimizer group.

**Derived per-time values (never stored).** For a query time t the model produces the arrays the rasterizer consumes:

| Derived array | Shape | Definition |
| --- | --- | --- |
| `means_t` | \[N, 3\] | μx(t) = μx + v·(t − μt)  (E1) |
| `opacities_t` | \[N\] | σ·σt(t), with σt from E4  (E3) |
| `quats_n` | \[N, 4\] | q/‖q‖ |
| `scales` | \[N, 3\] | exp(â) |
| `sh` | \[N, (L+1)², 3\] | concat(`sh0`, `shN`) along dim 1 |

Scale, orientation and SH do not change with time; only position and opacity do \[P §3.1\].

**Validity rules, checked on load and after every relocation.** All tensors are finite. Every quaternion has norm > 1e-8. N is identical across all nine tensors. N never exceeds N\_max, and initialization stops with an error when the start budget F·B (D-24) would exceed it. N changes only at growth events (500 ≤ it ≤ 15,000) and only upward \[D-18, D-44\]; the one exception is the "w/o periodic relocation" ablation, which prunes (§1.3).

**Size evidence.** Table 7 of the paper reports model sizes that work out to about 124 bytes per primitive across all seven rows (if sizes are in MiB). That is 31 float32 values (for example SH degree 1 plus extras) or 62 float16 values; SH degree 3 stored in half precision (64 values, 128 bytes) is within 4%. The sizes therefore do not settle L, which is a user parameter (D-3).

### 1.2 Equations

The model, losses and schedules are fixed by equations E1–E17. E1–E4, E7, E8 and E10 restate the paper's Eqs. 1–7; the others fill in what the paper delegates to 3DGS or leaves implicit, and each carries its tag.

**E1 · Motion** \[P §3.1, Eq. 1\]

```latex
\mu_x(t) = \mu_x + v\,(t - \mu_t)
```

Linear for every t, including outside \[0, 1\]; no clamping. Jacobians used by tests: ∂μx(t)/∂μx = I, ∂μx(t)/∂v = (t − μt)·I, ∂μx(t)/∂μt = −v.

**E2 · Color** \[P §3.1, Eq. 2; +0.5 offset and clamp: L: gsplat 1.5.3, D-4\]

```latex
c_i = \max\Big(0,\; \sum_{l=0}^{L}\sum_{m=-l}^{l} c_{i,lm}\, Y_{lm}(d_i) + 0.5\Big),\qquad d_i = \frac{\mu_{x,i}(t) - o_{cam}}{\lVert \mu_{x,i}(t) - o_{cam} \rVert}
```

o\_cam = −R\_wᵀ t\_w is the camera center recovered from the world-to-camera matrix. Y\_lm is the real SH basis with gsplat's constants and band ordering. The view direction uses the moved position, as Eq. 2 states. The max is taken per channel. Only bands l ≤ L\_active(it) contribute during training (E13).

**E3 · Space–time opacity** \[P §3.1, Eq. 3\]

```latex
\sigma_i(x,t) = \sigma_{t,i}(t)\;\sigma_i\;\exp\!\Big(-\tfrac12\,(x-\mu_{x,i}(t))^\top \Sigma_i^{-1} (x-\mu_{x,i}(t))\Big)
```

σᵢ = sigmoid(ôᵢ) and Σᵢ comes from E5. The renderer receives the product σ\_eff,i(t) = σt,i(t)·σᵢ as the per-primitive opacity; the Gaussian factor is evaluated by splatting (E6).

**E4 · Temporal opacity** \[P §3.1, Eq. 4\]

```latex
\sigma_{t,i}(t) = \exp\!\Big(-\tfrac12\Big(\frac{t-\mu_{t,i}}{s_i}\Big)^2\Big),\qquad s_i = e^{\hat s_i}
```

Unnormalized: the peak is 1 at t = μt. Derivatives: ∂σt/∂μt = σt·(t − μt)/s² and ∂σt/∂ŝ = σt·((t − μt)/s)². μt also receives gradient through E1.

**E5 · Covariance** \[P §3.1, text under Eq. 3; L: 3DGS Eq. 6; quaternion convention: gsplat docs\]

```latex
\Sigma_i = R(q_i)\,S_i S_i^\top R(q_i)^\top,\qquad S_i = \mathrm{diag}\big(e^{\hat a_i}\big)
```

```latex
R(w,x,y,z) = \begin{bmatrix} 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2) \end{bmatrix}
```

q is normalized before use.

**E6 · Splatting and compositing** \[P Fig. 2 and §3.2 ("similar to 3DGS"); L: 3DGS Eqs. 3 and 5; constants: gsplat 1.5.3, D-6, D-31\]

For a camera with world-to-camera rotation R\_w, translation t\_w and intrinsics K, at pixel center p = (u + ½, v + ½):

```latex
x_c = R_w\,\mu_x(t) + t_w,\qquad \mu'_i = \Big(f_x \tfrac{x_c}{z_c} + c_x,\; f_y \tfrac{y_c}{z_c} + c_y\Big),\qquad \Sigma'_i = J\, R_w \Sigma_i R_w^\top J^\top + \epsilon I_2
```

```latex
\alpha_i(p) = \min\!\Big(\alpha_{max},\; \sigma_{eff,i}(t)\, \exp\!\big(-\tfrac12 (p-\mu'_i)^\top \Sigma'^{-1}_i (p-\mu'_i)\big)\Big)
```

```latex
C(p) = \sum_i c_i\, \alpha_i(p) \prod_{j<i} \big(1-\alpha_j(p)\big) \;+\; \Big(\prod_i \big(1-\alpha_i(p)\big)\Big)\, c_{bg}
```

Normative details, all matching gsplat 1.5.3 in classic mode:

- J is the EWA Jacobian of the perspective projection at x\_c, with x\_c/z\_c clamped to \[−(c\_x/f\_x + 0.15W/f\_x), (W − c\_x)/f\_x + 0.15W/f\_x\] and y\_c/z\_c likewise with H, f\_y, c\_y. The clamp applies to J only, not to μ′.
- ε = 0.3; α\_max = 0.999; a term is skipped when α < 1/255 or when its quadratic form ½(p − μ′)ᵀΣ′⁻¹(p − μ′) is negative (a numerical guard). The 3DGS paper clamps α at 0.99 instead (D-31).
- Primitives are composited front to back by camera-space depth z\_c.
- A pixel stops before the primitive that would bring transmittance to ≤ 10⁻⁴; that primitive is excluded.
- Primitives with z\_c < 0.01 or z\_c > 10¹⁰ are culled, as are primitives with σ\_eff < 1/255. The screen-space extent is √(2 ln(255·σ\_eff)) standard deviations, never more than 3.33. Radii are per axis, r\_x = ⌈e·√Σ′₀₀⌉ and r\_y = ⌈e·√Σ′₁₁⌉ with e that extent; a primitive whose det Σ′ ≤ 0, or whose box μ′ ± r lies wholly outside the image, gets zero radii. A primitive is visible in an image when both radii are positive.
- The background c\_bg is black \[D-34\].

**E7 · Rendering loss** \[P §3.2, Eq. 5; weights P §3.3\]

```latex
\mathcal L_{render} = \lambda_{img}\,\mathcal L_{img} + \lambda_{ssim}\,\mathcal L_{ssim} + \lambda_{perc}\,\mathcal L_{perc},\qquad (\lambda_{img},\lambda_{ssim},\lambda_{perc}) = (0.8,\;0.2,\;0.01)
```

- L\_img = mean |Î − I| over pixels and channels (L1) \[D-7\].
- L\_ssim = 1 − SSIM(Î, I) with an 11×11 Gaussian window (σ = 1.5), C₁ = 0.01², C₂ = 0.03², data range 1, per channel, no padding ("valid"), averaged over the map and channels \[D-8\].
- L\_perc = LPIPS with the VGG backbone, inputs mapped from \[0, 1\] to \[−1, 1\] \[D-9\].

Î is the render after background compositing and is not clamped: E2 clamps colors only at 0, so Î can exceed 1, as in gsplat 1.5.3's trainer. I is float32 RGB in \[0, 1\]. Metrics clamp Î to \[0, 1\] first (E16).

**E8 · 4D regularization** \[P §3.2, Eq. 6; weight P §3.3 and Table 5; "early stage": P §1\]

```latex
\mathcal L_{reg}(t) = \frac1N \sum_{i=1}^{N} \sigma_i \cdot \mathrm{sg}\big[\sigma_{t,i}(t)\big]
```

t is the timestamp of the current training image. The sum runs over all N primitives, visible or not, as Eq. 6 states \[P Eq. 6; D-11\]. The only nonzero gradient is ∂L\_reg/∂ôᵢ = (1/N)·σt,i(t)·σᵢ(1 − σᵢ).

**E9 · Total loss** \[weights P §3.3; composition D-37; schedule D-10\]

```latex
\mathcal L = \mathcal L_{render} + \lambda_{reg}\,\mathcal L_{reg}(t),\qquad \lambda_{reg} = 10^{-2}\ \text{at every iteration}
```

λ\_reg applies at every iteration by default. LossConfig.reg\_until ends it after a given iteration, which is how §1's "early stage" would be read as a cutoff; the paper gives no value for one (D-10).

**E10 · Relocation sampling score** \[P §3.2, Eq. 7; weights P §3.3; normalization D-12, D-13\]

```latex
\rho_i = \lambda_g\, \hat g_i + \lambda_o\, \sigma_i,\qquad \lambda_g = \lambda_o = 0.5,\qquad \hat g_i = \min\big(1,\; g_i / Q_{0.99}(g)\big)
```

gᵢ is the mean, over the iterations since the last relocation in which primitive i was visible, of ‖∂L/∂μ′ᵢ‖ with the 2D gradient rescaled to NDC units (x by W/2, y by H/2). Q₀.₉₉ is taken over live primitives that were visible at least once, interpolating linearly between order statistics (torch.quantile's default); if it is 0, ĝ ≡ 0. σᵢ is the base opacity, as in Eq. 7.

**E11 · Relocation move** \[L: 3DGS-MCMC Eq. 9; adoption D-15\]

For a live target j that receives n − 1 relocated primitives (n copies in total):

```latex
\sigma^{new}_j = 1 - (1-\sigma_j)^{1/n},\qquad a^{new}_j = a_j\,\frac{\sigma_j}{\sum_{i=1}^{n}\sum_{k=0}^{i-1}\binom{i-1}{k}\dfrac{(-1)^k\,(\sigma^{new}_j)^{k+1}}{\sqrt{k+1}}}
```

All n copies get σ\_new and a\_new plus the target's other tensors, including `times`, `log_durations` and `velocities`. n is clamped to \[1, 51\]. a\_new is computed in float64 from the unclamped σ\_new; only then is σ\_new clamped to \[0.005, 1 − 2⁻²³\], with the lower bound applied to the logit (D-14). gsplat 1.5.3 uses the same order. The n − 1 moved copies then take positions drawn from the target's Gaussian, μ\_j + R\_j diag(a\_j) z with z \~ N(0, I₃), as E17's split does \[D-15\].

**E12 · Velocity learning-rate annealing** \[P §3.2; product reading D-19; indexing D-41\]

```latex
\eta_v(u) = \eta_{v,0}^{\,1-u}\;\eta_{v,1}^{\,u},\qquad u = \frac{it-1}{T_{it}-1}\in[0,1]
```

The paper prints λ\_t = λ₀^(1−t) + λ₁^t. Read literally, the sum is about 1 at both ends (λ⁰ = 1 at u = 0 and at u = 1) and dips in between, so the rate would rise again at the end instead of annealing. The product form is the 3DGS log-linear schedule. Defaults: ηv,0 = ηx,0/Δt\_f and ηv,1 = ηx,1/Δt\_f \[D-19\].

**E13 · Other schedules** \[P→3DGS; L: 3DGS §5.1 and §7.1, 3DGS-MCMC §3.6; D-20; D-41\]

```latex
\eta_x(u) = \eta_{x,0}\Big(\frac{\eta_{x,1}}{\eta_{x,0}}\Big)^{u},\qquad \eta_{x,0} = 1.6\times10^{-4}\,S,\qquad \eta_{x,1} = 1.6\times10^{-6}\,S
```

```latex
L_{active}(it) = \min\big(\lfloor (it-1)/1000 \rfloor,\; L\big)
```

`times` uses the same exponential form, 1.6e-4 → 1.6e-6 in normalized time \[D-20\]. S is the spatial scale of D-27.

**E14 · Time normalization** \[D-1, D-46\]

```latex
t = \frac{\tau-\tau_0}{\tau_1-\tau_0},\qquad \Delta t_f = \operatorname{median}_{c}\Big(\operatorname{median}\big(\operatorname{diff}(\operatorname{sort}(t_{train,c}))\big)\Big)
```

τ₀ and τ₁ are the smallest and largest training timestamps; T\_τ = τ₁ − τ₀. Inputs given in raw units convert as μt = (τ\_c − τ₀)/T\_τ, s = d/T\_τ and v = v\_τ·T\_τ. Δt\_f is measured within each training camera c that has at least two images, so interleaved timestamps from unsynchronized cameras do not shrink it. Without such a camera it falls back to the pooled median of distinct timestamps, and to 1 for a single timestamp \[D-46\].

**E15 · Velocity initialization** \[P §3.2, "Initialization of our representation"; formula D-23\]

```latex
v_i = \frac{x^{(f')}_{\mathrm{nn}(i)} - x^{(f)}_i}{t_{f'} - t_f}
```

x^(f) are the points triangulated at frame f, f′ is the next initialization frame, and nn(i) is the nearest neighbour of xᵢ in x^(f′).

**E16 · Evaluation metrics** \[P §4, "Metrics"; DSSIM form derived from Tables 2, 3, 9 and 10; D-29, D-67\]

```latex
\mathrm{PSNR} = 10\log_{10}\frac{1}{\mathrm{MSE}(\hat I, I)},\qquad \mathrm{DSSIM}_k = \frac{1-\mathrm{SSIM}_k(\hat I, I)}{2},\quad k\in\{1,2\}
```

SSIM\_k is E7's SSIM (11×11 Gaussian window, σ = 1.5, valid region, mean over the map and channels) with data range k, so C₁ = (0.01k)² and C₂ = (0.03k)². Renders are clamped to \[0, 1\] first; each metric is computed per test image, averaged over a scene's test images, then over scenes \[D-67\]. Cross-check: SelfCap SSIM₂ 0.952 (Table 10) corresponds to DSSIM₂ 0.024 (Table 3), and ENeRF-Outdoor 0.846 (Table 9) to 0.077 (Table 2). The identity holds for every SelfCap row and for the ENeRF, 4K4D and Ours rows on ENeRF-Outdoor, but not for the 4DGS and STGS rows of Table 2 \[D-29, D-40\]. LPIPS uses AlexNet for Neural3DV and VGG for ENeRF-Outdoor and SelfCap, per the captions of Tables 6–11. Its inputs are mapped from \[0, 1\] to \[−1, 1\] for both backbones, and LPIPS-VGG is also reported with unmapped \[0, 1\] inputs, the 3DGS convention that gsplat's trainer follows, because the paper does not say which it used \[D-68\].

**E17 · Split during growth** \[L: 3DGS §5.2; adoption D-18, D-43\]

```latex
\mu_x^{(k)} = \mu_x + R(q)\,\mathrm{diag}(a)\,z_k,\quad z_k \sim \mathcal N(0, I_3),\qquad a^{(k)} = a/1.6,\qquad k = 1, 2
```

The two children replace their parent and copy its other tensors, including `times`, `log_durations` and `velocities`. A clone copies all nine tensors unchanged.

### 1.3 Initialization, relocation and optimization

Training starts from 4D points (matches → triangulation → k-NN velocities), runs 30,000 Adam iterations at one image per iteration, grows the primitive set by 3DGS clone and split between iterations 500 and 15,000, and relocates low-opacity primitives every 100 iterations instead of deleting them. There is no opacity reset and no pruning \[D-18, D-42\].

#### Initialization \[P §3.2, "Initialization of our representation"\]

1. Pick initialization frame slots: t\_k = k·Δt\_f for every k\_stride-th k, default k\_stride = 1 \[D-21\]. A slot holds, for each training camera that has an image within Δt\_f/2 of t\_k, that image (at most one per camera); a camera without one sits the slot out, which happens only near the ends of an unsynchronized sequence \[D-47\]. Initialization never reads a test-split image \[D-66\].
2. At each init frame, pair every training camera with the training camera whose optical axis is angularly closest (ties broken by center distance); a pair chosen by both of its cameras is matched once, as (A, B) with A first in the manifest's camera order \[D-22\]. Match each pair with RoMa \[P\], by default roma\_outdoor at 560 px upsampled to 864 px \[D-58\]. From match()'s dense two-way warp, keep pixels whose raw certainty is at least 0.5 and draw up to 5,000 of them without replacement, with probability proportional to certainty, from the pair's own generator \[D-22, D-56\]. RoMa's sample() is not used, because it draws from the global generator and sets every certainty above 0.05 to 1. RoMa's to\_pixel\_coordinates maps \[−1, 1\] onto \[0, W\], the continuous convention of §1.4, so matches need no half-pixel shift. RoMa refuses to start unless PyTorch's float32 matmul precision is "highest", so initialization runs with that setting. Tiny RoMa returns a one-way warp at input resolution with no bounds handling, so its matches whose target falls outside \[−1, 1\] are dropped \[D-58\].
3. Triangulate each match by two-view linear DLT. Keep points with positive depth in both views, reprojection error ≤ 2 px in both, and a triangulation angle ≥ 1°. Color = mean of the two source pixels \[D-22\].
4. Set μx to the point and μt to the mean normalized time of its two images, which is the slot time t\_k for synchronized rigs \[P; D-47\].
5. Set v by E15 against the next init frame; the last init frame uses the previous one. With a single init frame, v = 0 \[P; D-23\].
6. Set s₀ = k\_stride·Δt\_f. With slices spaced Δ apart and s = Δ, the summed temporal opacity of a static point is flat in t to a relative ripple of about 5·10⁻⁹ away from the sequence ends; at τ₀ and τ₁ it is about 70% of the interior value \[D-21\].
7. Keep at most k\_stride·B points of each init frame, a seeded uniform subset, so the start holds at most B points per frame of the sequence: F·B in all, with F = round(1/Δt\_f) + 1 frames (1 for a single timestamp). The default B = 1,000 gives about 300k points for 300 frames and 60k for 60 frames; initialization stops with an error if F·B exceeds N\_max \[D-24\].
8. Set scales isotropic: â = log(mean distance to the 3 nearest neighbours among the kept points of the same init frame), with the mean floored at 10⁻⁷ \[L: 3DGS §5.1, whose text gives the mean; gsplat's trainer uses the root mean square; D-25\].
9. Set opacity logit(0.1), quaternion (1, 0, 0, 0), `sh0` from the color, `shN` = 0 \[L; D-5\].

Points supplied through the public interface (§2.2) skip steps 1–6 and keep at most F·B points in total in step 7. Missing velocities become 0, which is the paper's "w/o 4D initialization" ablation \[P §4.2\]. Missing durations become s₀ = Δt\_f, with Δt\_f from the dataset (E14), and missing frame\_ids the index of the slot t\_k nearest to each point's time, which step 8 groups by. roma\_init writes velocities, durations and frame\_ids itself, so its files need neither fallback. With unsynchronized cameras, moving content triangulates with an error of up to |v| times the timestamp gap of the pair; training corrects small errors, and rigs with large offsets should supply initial points through the public interface instead \[D-47\].

#### One training iteration (it = 1 … T\_it, T\_it = 30,000) \[P §3.3\]

1. Take the next image from a seeded permutation of all training images, redrawn each epoch; batch size 1 \[D-26\].
2. Compute its normalized time t (E14) and L\_active (E13).
3. Build the per-time arrays (E1, E3, E4), cull σ\_eff < 1/255 \[D-32; inside gsplat by default, D-57\], render with gsplat (E6) and retain the gradient of the 2D means.
4. Evaluate the loss E9.
5. Backpropagate. For every primitive visible in this image (both screen radii > 0, E6), add its NDC-scaled 2D-mean gradient norm to g\_sum and 1 to count.
6. Set this iteration's learning rates (E12, E13) and take one Adam step on all nine tensors.
7. If it is a multiple of 100 with 500 ≤ it ≤ 25,000: grow (below) when it ≤ 15,000, then relocate, then reset g\_sum and count \[D-17, D-45\].

#### Growth \[D-18; L: 3DGS §5.2\]

At every event with it ≤ 15,000, before relocation:

1. Candidates are primitives whose mean NDC gradient g (E10) exceeds τ\_pos = 0.0002 \[L: 3DGS §5.2\].
2. If N plus the number of candidates would exceed N\_max (default 3,000,000), keep only the candidates with the largest g that fit, ties going to the lower row index \[D-44\].
3. A candidate whose largest scale is ≤ 0.01·S is cloned (an exact copy of all nine tensors, as gsplat's DefaultStrategy makes it; 3DGS §5.2 describes the clone as moved along the positional gradient); a larger one is split into two children by E17, which replace it \[L: 3DGS §5.2; gsplat DefaultStrategy\].
4. New rows start with zero Adam moments and inherit their parent's g\_sum and count, as gsplat's duplicate and split do, so the relocation later in the same event scores the grown region \[D-45\]. Model, optimizer and statistics are resized together (resize\_ in §2.1): kept rows stay in their old order and new rows go last, so row order is deterministic.
5. Nothing is deleted for low opacity or large size, and opacities are never reset; low-opacity primitives are relocated instead (below) \[D-42\].

Growth only adds, so the final count lies between the start (F·B, D-24) and N\_max, and τ\_pos sets where it ends: raising it yields fewer primitives. Table 7's PSNR-vs-N curve is checked at each run's final N, with τ\_pos sweeps at the default B and at B = 200 for final counts below the default start (D-18, D-39).

#### Relocation \[P §3.2, Eq. 7; §3.3\]

1. Dead set: primitives with base opacity σ < 0.005, tested on the stored logit as ô < ô\_dead = float32(logit 0.005) so that sigmoid rounding cannot move the boundary; live set: the rest \[D-14\]. If either set is empty, do nothing.
2. Score the live primitives with ρ (E10).
3. Draw one target per dead primitive from the live set, with replacement, P(j) ∝ ρ\_j, using the trainer's dedicated, checkpointed generator \[D-15\].
4. For each drawn target j, let n\_j = 1 + (times j was drawn). Apply E11 once per target, after all draws are made.
5. Overwrite every dead primitive's nine tensors with its target's updated values, then draw its position from the target's Gaussian with the relocation generator (E11) \[D-15\].
6. Zero the Adam moments (exp\_avg, exp\_avg\_sq) of every target row and every overwritten row; keep each tensor's step counter \[D-16\].

N is unchanged by relocation. A render taken just before and just after a relocation, at a time where the targets are near their temporal peak, should differ only slightly (test U-13).

#### Count-control modes \[D-18\]

- `growth = "gradient"` (default): the Growth rules above.
- `growth = "score"`: at each growth event, append max(0, min(N\_max, ⌊1.05·N⌋) − N) primitives drawn ∝ ρ and initialized by E11 \[L: 3DGS-MCMC §3.6\]. The count ends at N\_max.
- `growth = "none"`: N stays as initialized, and relocation alone moves capacity.
- Ablation "w/o periodic relocation" \[P §4.2\]: relocation off and 3DGS densification instead ("the same densify strategy as the 3DGS"): the Growth rules above, pruning of primitives with σ < 0.005 and, after iteration 3,000, of those whose largest scale exceeds 0.1·S, and an opacity reset to 0.01 every 3,000 iterations \[L: 3DGS §5.2; thresholds from gsplat DefaultStrategy\]. N is capped at the full run's final N by the largest-g rule of D-44. gsplat 1.5.3's DefaultStrategy cannot serve as the reference for the reset, because its reset condition never fires (an operator-precedence slip).

#### Optimizer and learning rates \[P §3.3: Adam "with the same settings as 3DGS"\]

Adam with β₁ = 0.9, β₂ = 0.999, ε = 10⁻¹⁵, no weight decay, one parameter group per tensor, dense updates \[P→3DGS; gsplat 1.5.3 trainer\].

| Tensor | LR at it = 1 | LR at it = T\_it | Schedule | Tag |
| --- | --- | --- | --- | --- |
| `means` | 1.6e-4 · S | 1.6e-6 · S | exponential (E13) | \[P→3DGS\] |
| `velocities` | 1.6e-4 · S / Δt\_f | 1.6e-6 · S / Δt\_f | annealed (E12) | form \[P\]; values \[D-19\] |
| `times` | 1.6e-4 | 1.6e-6 | exponential | \[D-20\] |
| `log_durations` | 5e-3 | 5e-3 | constant | \[D-20\] |
| `log_scales` | 5e-3 | 5e-3 | constant | \[P→3DGS\] |
| `quats` | 1e-3 | 1e-3 | constant | \[P→3DGS\] |
| `opacity_logits` | 5e-2 | 5e-2 | constant | \[P→3DGS\] |
| `sh0` | 2.5e-3 | 2.5e-3 | constant | \[P→3DGS\] |
| `shN` | 1.25e-4 | 1.25e-4 | constant | \[P→3DGS\] |

S = 1.1 × the largest distance from a training camera center to the mean of the training camera centers \[D-27\]. For a 300-frame sequence Δt\_f = 1/299, so the velocity rate starts at about 0.048·S per unit of t.

#### Iteration calendar for the default T\_it = 30,000

| Event | Iterations |
| --- | --- |
| SH bands active (L\_active = 0, 1, 2, 3) | 1–1,000 · 1,001–2,000 · 2,001–3,000 · 3,001–30,000 |
| 4D regularization on | 1–30,000 (whole run) |
| Growth (146 events) | 500, 600, …, 15,000 |
| Relocation (246 events) | 500, 600, …, 25,000 |
| Learning-rate decay | every iteration |
| Final checkpoint and evaluation | 30,000 |

### 1.4 Time, coordinate and camera conventions

Time is normalized to \[0, 1\] over the training timestamps, space is any right-handed world frame, and cameras follow gsplat: OpenCV axes, world-to-camera matrices, and pixel centers at half-integers. The paper states none of these, so all are decisions \[D-1, D-28\].

#### Time \[D-1\]

- Every image has its own timestamp τ (float64, any unit, e.g. seconds). Frame indices are optional metadata, so unsynchronized cameras are allowed.
- τ₀ and τ₁ are the smallest and largest training timestamps; T\_τ = τ₁ − τ₀. Both are stored in every checkpoint and exchange file.
- t = (τ − τ₀)/T\_τ (E14). If T\_τ = 0 (one frame), t ≡ 0, Δt\_f := 1 and velocities are frozen at 0.
- Rendering accepts any τ. Outside \[τ₀, τ₁\] the motion extrapolates linearly and the API emits a warning.
- Exchange files store temporal quantities in raw units (τ, duration d, velocity v\_τ), so implementations that normalize differently remain comparable (§2.2).
- Renders are invariant under (τ, μτ, d, v\_τ) → (aτ + b, aμτ + b, a·d, v\_τ/a) for any a > 0 (test U-4).

#### World and camera frames \[L: gsplat data conventions and kernels; D-28\]

- World: right-handed; units are those of the camera poses; no up axis is assumed.
- Camera axes: x right, y down, z forward into the scene.
- Extrinsics: a 4×4 world-to-camera matrix (gsplat `viewmats`), x\_cam = R\_w·x + t\_w; the camera center is o = −R\_wᵀ t\_w.
- Intrinsics: pinhole K = \[\[fx, 0, cx\], \[0, fy, cy\], \[0, 0, 1\]\] in pixels. The image spans \[0, W\] × \[0, H\] and pixel (u, v) is sampled at (u + 0.5, v + 0.5), as in gsplat's rasterization kernel.
- An OpenCV calibration (pixel centers at integers) converts with cx ← cx + 0.5 and cy ← cy + 0.5.
- Lens distortion is not modeled. Converters undistort images with the package's own implementation of OpenCV's documented distortion model and emit the undistorted K; SelfCap uses COLMAP instead, as its protocol prescribes (D-60, D-61).
- Resizing by ratio r: new size W′ = round(W·r) by H′ = round(H·r); fx and cx are multiplied by W′/W, and fy and cy by H′/H, which equal r when W·r and H·r are integers; pixels are area-averaged \[D-33\].

#### Images

- Float32 RGB in \[0, 1\], height × width × 3, taken from the files without gamma conversion; 8-bit inputs are divided by 255.
- Alpha channels are dropped and the background is black \[D-34\].

#### Dataset protocols \[P §4, "Datasets"\]

| Dataset | Frames used | Cameras | Resolution and resize | Test view | Camera files \[D-28\] |
| --- | --- | --- | --- | --- | --- |
| Neural3DV | first 300 at 30 FPS | 19–21 | 2704×2028, ratio 0.5 → 1352×1014 | cam00 ([dataset README](https://github.com/facebookresearch/Neural_3D_Video)) | `poses_bounds.npy`, LLFF layout |
| ENeRF-Outdoor | first 300 at 60 FPS | 18 | 1920×1080, ratio 1.0 | not stated; configurable \[D-29\] | OpenCV YAML intrinsics/extrinsics |
| SelfCap | 60-frame windows of longer 60 FPS sequences (dataset card) | 22–24 | 3840×2160, COLMAP undistortion then ratio 0.5; bike 1024×1024 per the card (1080×1080 in the paper), ratio 1.0 | dance 0015, corgi 0007, bike 0009 (dataset card) \[D-29\] | OpenCV-style YAML intri/extri plus sync.json (D-63); Hugging Face, non-commercial license |

Neural3DV converter \[D-28\]: each camera row of `poses_bounds.npy` holds a 3×5 matrix (camera-to-world rotation with columns down, right, back; translation; and H, W, focal) plus near/far bounds. The OpenCV rotation is \[col₁, col₀, −col₂\]; fx = fy = focal; cx = W/2, cy = H/2; then scale by the resize ratio. The column order is stated from general knowledge of the LLFF format and is verified by test C-1 (epipolar residuals), not taken on trust. Frames come from a user-installed ffmpeg (version 5.1 or later, the first release with -fps\_mode; general knowledge), called as a separate process; the package neither bundles nor links FFmpeg \[D-48\]. Per camera the converter runs ffmpeg -v error -i camXX.mp4 -map 0:v:0 -frames:v 300 -fps\_mode passthrough -f rawvideo -pix\_fmt rgb24 - and reads exactly W·H·3 bytes per frame from the pipe, W and H read first with ffprobe from the same FFmpeg installation, then resizes (D-33) and writes PNG. Frame k gets timestamp k/30 s. The ffmpeg path (--ffmpeg, default ffmpeg on PATH), its version line and the full command go into the manifest metadata; a missing ffmpeg or ffprobe stops the converter with exit code 2. With --hwaccel cuda the converter adds -hwaccel cuda before -i and records the decoder used in the manifest; --jobs sets how many cameras decode at once (§2.4).

ENeRF-Outdoor converter: per-camera K, distortion, rotation and translation are read with the package's OpenCV-style YAML reader (D-61); that file layout is general knowledge rather than a logged source, so key names are confirmed against the downloaded files during implementation and checked by test C-2. Videos given instead of image folders are decoded with the same ffmpeg call. Frame k gets timestamp k/fps s; --fps is required, 60 for ENeRF-Outdoor (§4 of the paper).

SelfCap converter \[D-59, D-60, D-61\]: the dataset is on Hugging Face (`zju3dv/SelfCap-Dataset`), and its card gives the FreeTimeGS evaluation protocol. Each FreeTimeGS scene is a 60-frame window of a longer released sequence:

| FreeTimeGS scene | Archive | Test camera | Frames n | Resize |
| --- | --- | --- | --- | --- |
| dance1 | hair-release | 0015 | 4120 ≤ n < 4180 | 0.5 |
| dance2 | hair-release | 0015 | 5530 ≤ n < 5590 | 0.5 |
| corgi1 | corgi-release | 0007 | 200 ≤ n < 260 | 0.5 |
| corgi2 | corgi-release | 0007 | 2950 ≤ n < 3010 | 0.5 |
| bike1 | bike-release | 0009 | 8900 ≤ n < 8960 | 1.0 |
| bike2 | bike-release | 0009 | 30020 ≤ n < 30080 | 1.0 |

- dance3 and dance4 are not released, so six of Table 3's eight scenes can be reproduced, and comparisons use Table 10's per-scene rows.
- Training cameras are all the others: 23 for the dance and corgi scenes and 21 for bike (the card lists 24, 24 and 22 cameras).
- Frames come from one MP4 per camera and are selected by their 0-based decode index n. The converter reads each video's width, height and frame count with ffprobe from the same FFmpeg installation, then runs ffmpeg -v error -i CAM.mp4 -map 0:v:0 -vf select='between(n\\,N0\\,N1-1)' -fps\_mode passthrough -frames:v N1−N0 -f rawvideo -pix\_fmt rgb24 - so only the window crosses the pipe; earlier frames are still decoded (D-48).
- Timestamps: τ = n/60 s minus the camera's offset from `optimized/sync.json` when the archive has one, because the card defines the actual time as the frame-index time minus sync.json. `--no-sync` ignores the file \[D-59\]. The offsets are read as seconds, since the card subtracts them from the frame-index time n/60; they span −0.013 to +0.012 for hair, −0.030 to +0.026 for yoga and −0.046 to +0.026 for corgi, up to 2.8 frames from the frame-index time at 60 FPS, and corgi's training cameras span 4.35 frames between them. C-5 checks this reading against the data.
- Undistortion follows the card's protocol: COLMAP's `image_undistorter` with `blank_pixels` = 0, run on the user's own COLMAP, then area downsampling for ratio 0.5. `--undistort internal` uses the package's own undistortion instead, which is not protocol-exact \[D-60\].
- Camera files, checked on hair-calib and yoga-calib and on the corgi listing: optimized/intri.yml and optimized/extri.yml in OpenCV-style YAML, with a names list of four-digit camera names. intri.yml holds K\_\<cam> (3×3), D\_\<cam> (5×1: k1, k2, p1, p2, k3, with k3 = 0 in both files), H\_\<cam> and W\_\<cam> (2160 and 3840), and in hair an identity ccm\_\<cam>. extri.yml holds R\_\<cam> (a Rodrigues vector), Rot\_\<cam> (the same rotation as a matrix, agreeing to 3·10⁻⁸), T\_\<cam> (metres), and in hair t\_\<cam> = 0, n\_\<cam> = 1 and f\_\<cam> = 20. Read as world-to-camera (x\_cam = Rot·x + T, OpenCV axes), every camera faces the middle of the rig (cosine ≥ 0.64), while under the camera-to-world reading none does, so Rot and T go straight into world\_to\_camera and K gets the +0.5 shift of §1.4. The package's PyYAML-based reader (D-61) parses both files. The converter stops if Rot and Rodrigues(R) differ by more than 10⁻⁶ or any ccm is not the identity \[D-63\].
- Calibration source: corgi and bike use optimized/ inside their release archives; hair (dance1, dance2) uses the separately published hair-calib archive (files dated 2025-02-12). --calib DIR overrides either \[D-63\].
- The archives also hold point clouds: dense ones every 1,000 frames, RealityCapture crops, sparse ones for every frame, and a set every 10 frames. FreeTimeGS initializes from RoMa, so they are unused by default; they can be supplied as initial points through the interface of §2.2. The card says they were made from multiview images for a model trained with no held-out view, so they may encode the test camera, and runs initialized from them are labeled \[D-66\]. Their file numbers are offset from the decode index: corgi's per-frame clouds run from 005200.ply to 008699.ply (3,500 files, the card's 3,500 frames), so decode index n is file 5200 + n.
- Integrity: the converter records each archive's SHA-256 in the manifest and fails on any ffmpeg decoding error inside the selected window. The hosted corgi-release archive fails gzip's CRC check (owner-verified on 2026-10-07 with a SHA-256 identical to Hugging Face's), so some bytes somewhere in it are wrong. corgi1 and corgi2 count only where their windows decode cleanly and the calibration passes the checks above, and their results carry a warning until the archive is fixed.
- The card gives the bike scenes as 1024×1024 and the paper as 1080×1080; the converter uses the actual video size.
- Dynamic-region metrics \[P App. B.1\] crop the render and the ground truth to the bounding box of the test image's dynamic mask and set pixels outside the mask to black in both; the box is taken per image, and soft masks are binarized at 0.5 \[D-67\]. They need one mask per test image (`--masks DIR`) and are skipped without them. The paper made them with BackgroundMattingV2 from ground-truth background images, but the corgi archive contains none (listing) and the card mentions none for any sequence, so its dynamic-region numbers cannot be rebuilt as published. A background estimated as the per-pixel temporal median of the test camera's whole video is an approximation users may run themselves, outside the package, and results based on it are labeled approximate \[D-64\].
- License: research and non-commercial use only, and modifications must stay open-source and non-commercial. Converted SelfCap data, masks and models trained on it are therefore never committed, bundled or used in CI.
- Ablations \[P §4.2\] run on dance1 and also report "10-frame with the fastest motion": the 10 consecutive frames of the window whose consecutive ground-truth frames of the test camera differ most (mean absolute difference), a range the converter stores in the manifest \[D-69\].

### 1.5 Hyperparameters

The paper fixes nine settings outright, and its "same settings as 3DGS" sentence imports the optimizer values. Everything else is a default chosen here. The three groups are kept in separate tables so defaults can be tuned without touching paper values.

**A · Stated by the paper**

| Setting | Value | Paper location |
| --- | --- | --- |
| Iterations | 30,000 for a 300-frame sequence | §3.3 |
| Optimizer | Adam, same settings as 3DGS | §3.3 |
| Loss weights λ\_img, λ\_ssim, λ\_perc | 0.8, 0.2, 0.01 | §3.3 |
| 4D regularization weight λ\_reg | 1e-2, printed 1e^{−2} (D-38); ablated: 0, 1e-3, 1e-2, 1e-1 | §3.3; Table 5 |
| Regularization timing | "in the early stage of optimization"; no cutoff, decay or schedule is given, and §3.3 states one constant weight | §1; §3.3 |
| Score weights λ\_g, λ\_o | 0.5, 0.5 | §3.3 |
| Relocation period | every 100 iterations | §3.3 |
| Velocity learning-rate schedule | printed as λ\_t = λ₀^(1−t) + λ₁^t, t from 0 to 1 over training; λ₀, λ₁ not given (read as a product: D-19) | §3.2 |
| Initialization | RoMa matches per frame, triangulation, k-NN velocity between two frames | §3.2 |

Reference outcomes, useful as targets but not settings: about 1 hour on an RTX 4090 for a 300-frame sequence (§3.3, which names no dataset). On Neural3DV, 125 MB, or 41 MB with at most 500k primitives (Table 6); these sizes match Table 7's rows at 1,060k and 347k primitives (Table 7 states the counts; the match is our inference). On SelfCap, 96 MB at 467 FPS, or 53 MB at 664 FPS with at most 500k primitives (Tables 3 and 8), which at Table 7's bytes per primitive is about 810k and 450k primitives (also our inference). On ENeRF-Outdoor, 454 FPS (Table 2); 450 FPS at 1080p (Fig. 1).

**B · Taken from 3DGS**: the optimizer values through "same settings" \[P→3DGS\], the other rows as standard practice \[L: 3DGS\]

| Setting | Value | Source |
| --- | --- | --- |
| Adam β₁, β₂, ε | 0.9, 0.999, 1e-15 | gsplat 1.5.3 trainer (3DGS replica) |
| Position learning rate | 1.6e-4·S → 1.6e-6·S, exponential | 3DGS §5.1; 3DGS-MCMC §3.6; gsplat |
| Scale, rotation, opacity rates | 5e-3, 1e-3, 5e-2 | gsplat 1.5.3 trainer |
| SH rates | 2.5e-3 (DC), 2.5e-3/20 (higher bands) | gsplat 1.5.3 trainer |
| SH schedule | one band added every 1,000 iterations | 3DGS §7.1 \[L\] |
| Activations | sigmoid opacity, exponential scale, normalized quaternion | 3DGS §4, §5.1 \[L\] |
| Scale initialization | mean distance to the 3 nearest points | 3DGS §5.1 \[L\] |
| Opacity initialization | 0.1 | gsplat 1.5.3 trainer \[L\] |

**C · Defaults chosen here** \[D\]

| Setting | Default | Decision |
| --- | --- | --- |
| Time normalization | \[0, 1\] over training timestamps | D-1 |
| Duration storage | log s | D-2 |
| SH degree L | user parameter sh\_degree ∈ {0, 1, 2, 3}, default 3 | D-3 |
| Image loss / SSIM loss / LPIPS backbone | L1 / 1 − SSIM (11×11, σ = 1.5, valid) / VGG | D-7, D-8, D-9 |
| Regularization window | whole run, constant λ\_reg = 1e-2 (reg\_until unset) | D-10 |
| Score gradient normalization | divide by 99th percentile, cap at 1 | D-12, D-13 |
| Dead threshold | base opacity < 0.005, tested as a float32 logit | D-14 |
| Relocation move | 3DGS-MCMC Eq. 9, copy other tensors, moved copies' positions drawn from the target's Gaussian | D-15 |
| Adam moments after relocation | zeroed for targets and moved rows | D-16 |
| Relocation window | iterations 500–25,000 | D-17 |
| Count control | grows by 3DGS clone/split (τ\_pos = 0.0002, clone/split boundary 0.01·S, split by E17) at iterations 500–15,000; no deletion, no opacity reset | D-18, D-42 to D-45 |
| Velocity rates ηv,0 → ηv,1 | 1.6e-4·S/Δt\_f → 1.6e-6·S/Δt\_f | D-19 |
| `times` / `log_durations` rates | 1.6e-4 → 1.6e-6 / 5e-3 constant | D-20 |
| Init frame stride, initial duration | 1, s₀ = stride·Δt\_f | D-21 |
| Matching and triangulation filters | 5,000 matches, certainty ≥ 0.5, ≤ 2 px, ≥ 1° | D-22 |
| Velocity matching | 1-NN, forward difference | D-23 |
| Initial budget B / growth cap N\_max | 1,000 points per frame (about 300k for 300 frames) / 3,000,000 (cap 500,000 for the † rows) | D-24, D-44, D-39 |
| Batch and sampling | 1 image, per-epoch permutation | D-26 |
| Spatial scale S | 1.1 × max distance of training cameras from their centroid | D-27 |
| Iterations for other sequence lengths | 30,000 | D-35 |
| Resolution warm-up | none | D-36 |
| Background | black | D-34 |
| Numeric precision | float32; float64 for raw timestamps | D-30 |
| Rasterizer constants | ε = 0.3, near 0.01, far 1e10, α\_max 0.999 (the 3DGS paper uses 0.99), skip α < 1/255, stop when transmittance ≤ 1e-4 (gsplat 1.5.3) | D-6, D-31 |

## 2. Package architecture

### 2.1 Module layout and public API

The package is one importable library, `openftgs`, with a CLI named `ftgs`. Modules depend only downward: CLI → trainer → training mechanics → model and renderer → data, init and I/O. The differential harness is a separate package that talks to implementations only through files and commands.

&#91;embedded content: package layers · 14 module groups\]

Each band imports only from the bands below it. The harness never imports the package; it works through the CLI and the file formats of §2.2.

| Module | Responsibility | Spec items |
| --- | --- | --- |
| `openftgs.types` | Dataclasses for public data: frozen `Camera`, `ImageRecord`, `InitPoints`, `TimeNormalizer` and `ExchangeModel`; `RenderOutput`; and `TrainerState`, the plain data a checkpoint holds | §1.4, §2.2 |
| `openftgs.config` | `TrainConfig` and sub-configs with the §1.5 defaults; every field records its provenance tag | §1.5 |
| `openftgs.motion` | Pure functions for E1, E4, effective opacity and E14; the fused temporal slice with eager, compiled and optional CUDA implementations (D-50) | E1, E3, E4, E14 |
| `openftgs.sh` | Real SH basis (reference), RGB ↔ DC conversion | E2 |
| `openftgs.model` | `FreeTimeGaussians`: the nine tensors, activations, `at_time()` | §1.1 |
| `openftgs.render` | `render()` with two backends: `gsplat` (CUDA, production) and `reference` (pure PyTorch, CPU, tests); the temporal index for rendering (D-55) | E2, E5, E6 |
| `openftgs.losses` | L1, SSIM, LPIPS wrapper, 4D regularization, total loss | E7–E9 |
| `openftgs.density` | Gradient statistics, growth (clone/split, E17), score, dead set, E11, `relocate()`, `grow()`, count-control modes | E10, E11, E17, §1.3 |
| `openftgs.optim` | Adam construction, schedules, row-wise moment reset, state I/O | E12, E13 |
| `openftgs.train` | `Trainer`: iteration loop, sampler, schedules, evaluation and checkpoint hooks | §1.3 |
| `openftgs.data` | Camera operations, `MultiViewVideo` dataset with lazy image loading, scene manifests, dataset converters (Neural3DV, OpenCV-style YAML, SelfCap), image resizing and undistortion and the YAML camera reader (D-61), frame cache and prefetcher (D-51) | §1.4 |
| `openftgs.init` | `InitPoints` → tensors, k-NN velocities, DLT triangulation, optional RoMa pipeline (D-58) on the vendored copy in openftgs.\_vendor.romatch (D-65) | §1.3, E15 |
| `openftgs.io` | Reads and writes full checkpoints (`TrainerState`), the FTGS-IF exchange format (`ExchangeModel`) and PLY files for viewers; plain data in and out, so it imports neither the model nor the trainer | §2.2 |
| `openftgs.eval` | PSNR, SSIM\_k, DSSIM\_k, LPIPS-Alex and LPIPS-VGG, dynamic-region crop and mask (App. B.1, D-67) | E16 |
| `openftgs.cli` | `ftgs convert · cache · init · train · render · eval · export · bench` | §2.2, §3.4 |
| `ftgs_harness` (separate package) | Black-box differential harness; never imports `openftgs` | §3.3 |

**Public API signatures.** Everything not listed here is private and may change between minor versions.

```python
# openftgs.types
@dataclass(frozen=True)
class Camera:
    K: Tensor                 # [3, 3] float32, continuous pixel convention (§1.4)
    world_to_camera: Tensor   # [4, 4] float32, OpenCV axes
    width: int
    height: int
    name: str = ""

@dataclass(frozen=True)
class ImageRecord:
    camera: Camera
    timestamp: float          # raw τ (float64)
    image_path: Path
    split: Literal["train", "test"]
    frame_index: int | None = None

@dataclass(frozen=True)
class InitPoints:
    positions: Tensor               # [N, 3] float32, world units
    colors: Tensor                  # [N, 3] float32 in [0, 1]
    timestamps: Tensor              # [N] float64, raw τ
    velocities: Tensor | None = None  # [N, 3] world units per unit of τ
    durations: Tensor | None = None   # [N] float64, raw τ units (std. dev. of E4)
    frame_ids: Tensor | None = None   # [N] int64; groups points for the 3-NN scale init

@dataclass(frozen=True)
class TimeNormalizer:
    tau0: float
    tau1: float
    def to_t(self, tau: float | Tensor) -> float | Tensor: ...
    def to_tau(self, t: float | Tensor) -> float | Tensor: ...

@dataclass
class RenderOutput:
    image: Tensor                  # [C, H, W, 3]
    alpha: Tensor                  # [C, H, W, 1]
    means2d: Tensor | None = None  # [C, N, 2] unpacked or [nnz, 2] packed; kept for gradient statistics
    visible: Tensor | None = None  # [C, N] bool, unpacked mode
    gaussian_ids: Tensor | None = None  # [nnz] int, packed mode: the primitive behind each row of means2d

@dataclass(frozen=True)
class ExchangeModel:               # FTGS-IF contents (§2.2): physical values, raw time units
    arrays: dict[str, Tensor]      # position, time_center, duration, velocity, scale, rotation, opacity, sh
    sh_degree: int
    tau0: float
    tau1: float
    time_unit: str
    producer: str

@dataclass
class TrainerState:                # plain tensors and dicts; a §2.2 checkpoint holds exactly this
    iteration: int
    config: dict                   # TrainConfig as JSON-ready values
    time: TimeNormalizer
    spatial_scale: float           # S (D-27)
    frame_dt: float                # Δt_f (E14)
    params: dict[str, Tensor]      # the nine stored tensors of §1.1
    optim: dict                    # per tensor: exp_avg, exp_avg_sq, step, schedule parameters
    stats: dict[str, Tensor]       # g_sum, count
    rng: dict                      # generator states
    sampler: dict                  # epoch, cursor, permutation
```

```python
# openftgs.model
class FreeTimeGaussians(torch.nn.Module):
    PARAM_NAMES = ("means", "times", "log_durations", "velocities",
                   "log_scales", "quats", "opacity_logits", "sh0", "shN")
    def __init__(self, params: Mapping[str, Tensor], *, sh_degree: int,
                 time: TimeNormalizer) -> None: ...
    @classmethod
    def from_init_points(cls, points: InitPoints, time: TimeNormalizer,
                         cfg: InitConfig, *, frame_dt: float, generator: torch.Generator) -> "FreeTimeGaussians": ...  # frame_dt gives F and missing durations
    @classmethod
    def from_exchange(cls, exchange: ExchangeModel, *, device: str = "cuda") -> "FreeTimeGaussians": ...
    @classmethod
    def load(cls, path: Path, *, device: str = "cuda") -> "FreeTimeGaussians": ...  # checkpoint directory or FTGS-IF file
    def to_exchange(self) -> ExchangeModel: ...
    @property
    def num_gaussians(self) -> int: ...
    def at_time(self, t: float, *, sh_degree: int | None = None) -> GaussiansAtTime: ...
    def physical(self) -> dict[str, Tensor]: ...   # activated values in raw time units
    def resize_(self, *, keep: Tensor | None = None,
                append: Mapping[str, Tensor] | None = None) -> None: ...  # kept rows keep their order; new rows go last
```

```python
# openftgs.render
def render(model: FreeTimeGaussians, cameras: Camera | Sequence[Camera], timestamp: float, *,
           time_units: Literal["raw", "normalized"] = "raw",
           sh_degree: int | None = None,
           background: Tensor | None = None,
           backend: Literal["gsplat", "reference"] = "gsplat",
           return_aux: bool = False) -> RenderOutput: ...

# openftgs.losses
def l1_loss(pred: Tensor, gt: Tensor) -> Tensor: ...
def ssim(pred: Tensor, gt: Tensor, *, window: int = 11, sigma: float = 1.5,
         data_range: float = 1.0, padding: Literal["valid", "same"] = "valid") -> Tensor: ...
class LPIPSLoss(torch.nn.Module):
    def __init__(self, net: Literal["vgg", "alex"] = "vgg") -> None: ...
def reg4d_loss(model: FreeTimeGaussians, t: float) -> Tensor: ...
def total_loss(pred: Tensor, gt: Tensor, model: FreeTimeGaussians, t: float, it: int,
               cfg: LossConfig) -> tuple[Tensor, dict[str, float]]: ...

# openftgs.density
class GradStats:
    def update(self, means2d_grad: Tensor, *, width: int, height: int,
               visible: Tensor | None = None, gaussian_ids: Tensor | None = None) -> None: ...  # visible in unpacked mode, gaussian_ids in packed mode; index_add_, no host sync
    def mean(self) -> Tensor: ...
    def reset(self) -> None: ...
    def resize_(self, *, keep: Tensor | None = None, append_from: Tensor | None = None) -> None: ...  # new rows copy their parents' statistics
def sampling_score(model: FreeTimeGaussians, stats: GradStats, cfg: RelocConfig) -> Tensor: ...
def grow(model: FreeTimeGaussians, optimizer: "FTGSOptimizer", stats: GradStats,
         cfg: GrowthConfig, *, spatial_scale: float, reloc: RelocConfig,
         generator: torch.Generator) -> GrowthReport: ...  # E17, D-18
def mcmc_split(opacity: Tensor, scales: Tensor, n: Tensor) -> tuple[Tensor, Tensor]: ...  # E11
def relocate(model: FreeTimeGaussians, optimizer: "FTGSOptimizer", stats: GradStats,
             cfg: RelocConfig, *, generator: torch.Generator) -> RelocationReport: ...

# openftgs.optim
class FTGSOptimizer:
    def set_iteration(self, it: int) -> dict[str, float]: ...   # applies E12/E13, returns rates
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def reset_rows(self, rows: Tensor) -> None: ...            # zero Adam moments of these rows
    def resize_(self, *, keep: Tensor | None = None, n_append: int = 0) -> None: ...  # new rows get zero moments
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
def build_optimizer(model: FreeTimeGaussians, cfg: OptimConfig, *,
                    spatial_scale: float, frame_dt: float) -> FTGSOptimizer: ...
```

```python
# openftgs.train
class Trainer:
    def __init__(self, cfg: TrainConfig, dataset: "MultiViewVideo",
                 init: InitPoints | FreeTimeGaussians, *, device: str = "cuda") -> None: ...  # the seed is cfg.seed
    @classmethod
    def from_checkpoint(cls, directory: Path, dataset: "MultiViewVideo", *,
                        device: str = "cuda") -> "Trainer": ...   # io.load_checkpoint, then load_state_dict
    @property
    def iteration(self) -> int: ...          # completed optimizer steps
    def step(self) -> StepLog: ...
    def train(self, until: int | None = None, callbacks: Sequence[Callback] = ()) -> None: ...
    def evaluate(self, split: str = "test", *, out_dir: Path | None = None) -> dict[str, float]: ...
    def state_dict(self) -> TrainerState: ...
    def load_state_dict(self, state: TrainerState) -> None: ...

# openftgs.data
class MultiViewVideo:
    records: list[ImageRecord]
    time: TimeNormalizer
    @classmethod
    def from_manifest(cls, path: Path) -> "MultiViewVideo": ...
    def load_image(self, index: int) -> Tensor: ...     # [H, W, 3] float32
def convert_n3dv(src: Path, dst: Path, *, frames: int = 300, resize: float = 0.5,
                 ffmpeg: str = "ffmpeg", hwaccel: Literal["none", "cuda"] = "none",
                 jobs: int = 0) -> Path: ...   # D-48; jobs 0 = one per four cores
def convert_opencv_yaml(src: Path, dst: Path, *, fps: float, frames: int, resize: float,
                        ffmpeg: str = "ffmpeg", hwaccel: Literal["none", "cuda"] = "none",
                        jobs: int = 0) -> Path: ...   # frame k at k/fps s; fps 60 for ENeRF-Outdoor

def convert_selfcap(src: Path, dst: Path, *,
                    scene: str, window: tuple[int, int] | None = None, test_camera: str | None = None,
                    undistort: Literal["colmap", "internal"] = "colmap", colmap: str = "colmap",
                    use_sync: bool = True, masks: Path | None = None, calib: Path | None = None,
                    ffmpeg: str = "ffmpeg", hwaccel: Literal["none", "cuda"] = "none",
                    jobs: int = 0) -> Path: ...   # D-59, D-60; dance1 … bike2 take window and test camera from the card, other names need both

# openftgs.init
def knn_velocities(points: Tensor, next_points: Tensor, dt: float) -> Tensor: ...    # E15
def triangulate_dlt(x1: Tensor, x2: Tensor, P1: Tensor, P2: Tensor) -> Tensor: ...
@dataclass
class RomaInitConfig:                        # D-21, D-22, D-24, D-56, D-58
    model: Literal["roma_outdoor", "tiny_roma_v1_outdoor"] = "roma_outdoor"
    coarse_res: int = 560
    upsample_res: int = 864
    upsample_preds: bool = True              # False: match at coarse_res only
    matches_per_pair: int = 5000
    min_certainty: float = 0.5               # on match()'s raw certainty
    stride: int = 1
    points_per_frame: int = 1_000            # B, D-24; the trainer applies InitConfig.points_per_frame again
    seed: int = 0
def roma_init(dataset: "MultiViewVideo", cfg: RomaInitConfig) -> InitPoints: ...  # extra [init]; training images only (D-66)

# openftgs.io: plain data in and out; imports neither model nor train
def save_checkpoint(state: TrainerState, directory: Path) -> Path: ...
def load_checkpoint(directory: Path) -> TrainerState: ...
def write_exchange(model: ExchangeModel, path: Path) -> None: ...
def read_exchange(path: Path) -> ExchangeModel: ...
```

```python
# openftgs.config: TrainConfig; its field names are the hyperparameter names the harness uses (§3.3)
@dataclass
class InitConfig:                                # §1.3 Initialization
    points_per_frame: int = 1_000                # B, D-24; F·B must not exceed GrowthConfig.n_max
    init_opacity: float = 0.1                    # D-5
    scale_knn: int = 3                           # L: 3DGS §5.1
    scale_floor: float = 1e-7                    # D-25

@dataclass
class LossConfig:                                # E7–E9
    lambda_img: float = 0.8                      # P §3.3
    lambda_ssim: float = 0.2                     # P §3.3
    lambda_perc: float = 0.01                    # P §3.3
    lambda_reg: float = 1e-2                     # P §3.3
    reg_until: int | None = None                 # D-10: last iteration with L_reg; None = whole run
    ssim_window: int = 11                        # D-8
    ssim_sigma: float = 1.5                      # D-8
    ssim_padding: Literal["valid", "same"] = "valid"    # D-8
    perc_net: Literal["vgg", "alex"] = "vgg"     # D-9

@dataclass
class GrowthConfig:                              # §1.3 Growth and count-control modes
    mode: Literal["gradient", "score", "none"] = "gradient"   # D-18
    start: int = 500                             # D-18
    stop: int = 15_000                           # D-18
    grad_threshold: float = 2e-4                 # τ_pos; L: 3DGS §5.2
    split_scale: float = 0.01                    # × S: larger primitives split, smaller ones clone; L: 3DGS §5.2
    n_max: int = 3_000_000                       # D-44; 500,000 for the † rows (D-39)
    score_rate: float = 0.05                     # mode "score"; L: 3DGS-MCMC §3.6

@dataclass
class RelocConfig:                               # E10, E11, §1.3 Relocation
    enabled: bool = True                         # False only in the "w/o periodic relocation" ablation
    every: int = 100                             # P §3.3; growth events share this period
    start: int = 500                             # D-17
    stop: int = 25_000                           # D-17
    lambda_g: float = 0.5                        # P §3.3
    lambda_o: float = 0.5                        # P §3.3
    grad_quantile: float = 0.99                  # D-13
    dead_opacity: float = 0.005                  # D-14; compared as a float32 logit
    reset_moved_moments: bool = True             # D-16
    copy_position: Literal["sample", "stack"] = "sample"   # D-15; "stack" leaves copies on the target (U-13)
    prune_opacity: float | None = None           # ablation only: 0.005 when relocation is off

@dataclass
class OptimConfig:                               # E12, E13, §1.3 Optimizer
    iterations: int = 30_000                     # P §3.3; D-35
    betas: tuple[float, float] = (0.9, 0.999)    # P→3DGS
    eps: float = 1e-15                           # P→3DGS
    means_lr: tuple[float, float] = (1.6e-4, 1.6e-6)       # × S; E13
    velocities_lr: tuple[float, float] = (1.6e-4, 1.6e-6)  # × S/Δt_f; E12, D-19
    times_lr: tuple[float, float] = (1.6e-4, 1.6e-6)       # D-20
    log_durations_lr: float = 5e-3               # D-20
    log_scales_lr: float = 5e-3                  # P→3DGS
    quats_lr: float = 1e-3                       # P→3DGS
    opacity_logits_lr: float = 5e-2              # P→3DGS
    sh0_lr: float = 2.5e-3                       # P→3DGS
    shN_lr: float = 1.25e-4                      # P→3DGS
    sh_interval: int = 1000                      # L: 3DGS §7.1

@dataclass
class TrainConfig:
    sh_degree: int = 3                           # D-3
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)   # D-34
    scene_scale_factor: float = 1.1              # D-27
    seed: int = 0
    init: InitConfig = field(default_factory=InitConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    reloc: RelocConfig = field(default_factory=RelocConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)       # §2.4
```

**CLI.** Exit code 0 means success, 2 invalid input or environment (a file that fails schema validation, a failed calibration or integrity check, or a missing ffmpeg or COLMAP), 3 a numerical failure (NaN or Inf).

```
ftgs convert {n3dv,opencv-yaml,selfcap} SRC OUT [--scene NAME] [--fps F] [--frames N] [--resize R] [--ffmpeg PATH] [--ffprobe PATH] [--hwaccel none|cuda] [--jobs J] [--undistort colmap|internal] [--colmap PATH] [--no-sync] [--masks DIR] [--calib DIR]
ftgs cache SCENE.json [--placement auto|gpu|mmap]
ftgs init roma SCENE.json OUT.safetensors [--stride K] [--points-per-frame B] [--seed S] [--model roma_outdoor|tiny_roma_v1_outdoor] [--no-upsample]
ftgs train SCENE.json --init INIT.safetensors --out RUN_DIR [--config CFG.json] [--seed S] [--resume CKPT_DIR [--force]] [--profile START:COUNT]
ftgs render MODEL --request REQUEST.json --out RENDERS.safetensors [--backend gsplat|reference]
ftgs eval MODEL SCENE.json --split test --out METRICS.json [--renders-dir DIR]
ftgs export CKPT_DIR --format ftgs-if --out MODEL.safetensors
ftgs bench {train-step,raster,slice,losses,adam,data,render,init,convert} SCENE.json --out BENCH.json
```

MODEL is either a checkpoint directory or an FTGS-IF file. The trainer reads one seed, TrainConfig.seed; ftgs train --seed overrides the config file's value, and --force resumes despite a changed scene manifest. ftgs eval writes METRICS.json with per-image and per-scene values of psnr, ssim1, ssim2, dssim1, dssim2, lpips\_alex, lpips\_vgg and lpips\_vgg\_raw (D-68) under entire, and under dynamic when the manifest has masks, with the mask source. Names: distribution and import package openftgs, CLI ftgs (owner choice, D-62). The README's first line states that this is an independent clean-room implementation of FreeTimeGS (Wang et al., CVPR 2025), not affiliated with its authors. The owner's pip check on 2026-10-07 found no existing distribution named openftgs; the name is held only once a first release is published.

### 2.2 Stable file interfaces

Five versioned file formats form the public boundary: scene manifest, init points, full checkpoint, exchange model and render request/output. All tensors live in safetensors files and all other state in JSON, so no file is ever unpickled. Safetensors metadata is a string-to-string map, so integers are written in decimal, floats in Python's shortest round-trip repr (exact for float64), and structured values as compact JSON. Every file carries `format` and `version`; readers reject an unknown major version and ignore unknown optional keys.

#### Scene manifest — multi-camera input with timestamps (`ftgs-scene` v1, JSON)

```json
{
  "format": "ftgs-scene", "version": 1, "time_unit": "s",
  "cameras": {
    "cam00": {"width": 1352, "height": 1014, "model": "pinhole",
              "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
              "world_to_camera": [[r11, r12, r13, t1], [r21, r22, r23, t2], [r31, r32, r33, t3], [0, 0, 0, 1]]}
  },
  "images": [
    {"camera": "cam00", "timestamp": 0.0, "frame_index": 0,
     "path": "images/cam00/000000.png", "split": "test"}
  ],
  "metadata": {"source": "n3dv/flame_steak", "resize": 0.5}
}
```

- Conventions are those of §1.4: OpenCV axes, world-to-camera matrices, K in the continuous pixel convention, undistorted images.
- `timestamp` is float64 in `time_unit` and is per image, so unsynchronized rigs need no special case.
- An image entry may override `K` and `world_to_camera` for moving or zooming cameras.
- Paths are relative to the manifest file.
- A test image may carry `mask`, the path of a mask image of the same size, for dynamic-region metrics; `metadata.mask_source` records how the masks were made, `metadata.masks_approximate` flags estimated backgrounds (D-64, D-67), and `metadata.fastest_10` holds dance1's \[n0, n0 + 10) range (D-69).

#### Initial points (`ftgs-init` v1, safetensors)

| Key | Dtype | Shape | Required | Meaning |
| --- | --- | --- | --- | --- |
| `positions` | float32 | \[N, 3\] | yes | world coordinates |
| `colors` | float32 | \[N, 3\] | yes | RGB in \[0, 1\] |
| `timestamps` | float64 | \[N\] | yes | raw τ of each point |
| `velocities` | float32 | \[N, 3\] | no (0) | world units per unit of τ |
| `durations` | float64 | \[N\] | no (D-21) | temporal std. dev. in units of τ, > 0 |
| `frame_ids` | int64 | \[N\] | no | groups points for the 3-NN scale init |

Metadata: `format`, `version`, `time_unit`, `producer`. Validation: all values finite, N ≥ 1, colors within \[0, 1\], durations positive.

#### Full checkpoint for exact resume (`ftgs-checkpoint` v1, directory)

| File | Contents |
| --- | --- |
| `manifest.json` | format, version, iteration, full `TrainConfig`, τ₀, τ₁, S, Δt\_f, SH degree, seed, scene-manifest SHA-256, versions of openftgs, torch, gsplat and CUDA, SHA-256 of every other file |
| `params.safetensors` | the nine stored tensors of §1.1, raw form, float32 |
| `optim.safetensors` | per tensor: `exp_avg`, `exp_avg_sq`, `step` |
| `optim.json` | per tensor: base rates, schedule parameters, β₁, β₂, ε |
| `stats.safetensors` | growth and relocation accumulators `g_sum` \[N\] and `count` \[N\] |
| `rng.safetensors` | torch CPU generator state, CUDA generator state per device, relocation generator, sampler generator |
| `rng.json` | Python `random` state and NumPy bit-generator state |
| `sampler.json` | epoch, cursor and the current epoch's permutation |

Contract: resuming from a checkpoint at iteration k and training to iteration m gives the same images, learning rates, relocation draws and parameters as an uninterrupted run. This is bit-exact on the reference backend on CPU. On gsplat it holds up to floating-point reordering, because the CUDA backward accumulates with atomic additions. Saves are atomic (write to a temporary directory, then rename) and also happen on SIGINT and SIGTERM. Resume refuses a scene manifest whose hash differs, unless forced.

#### Exchange model for other implementations (`ftgs-if` v1, safetensors)

The exchange file holds activated, physical values in raw time units, so it does not depend on any implementation's internal parameterization.

| Key | Dtype | Shape | Value |
| --- | --- | --- | --- |
| `position` | float32 | \[N, 3\] | μx |
| `time_center` | float64 | \[N\] | τ₀ + μt·T\_τ |
| `duration` | float64 | \[N\] | s·T\_τ |
| `velocity` | float32 | \[N, 3\] | v/T\_τ, world units per unit of τ |
| `scale` | float32 | \[N, 3\] | exp(â) |
| `rotation` | float32 | \[N, 4\] | unit quaternion (w, x, y, z) with w ≥ 0 |
| `opacity` | float32 | \[N\] | sigmoid(ô) computed in float64 and clamped to \[10⁻⁷, 1 − 10⁻⁷\], so saturated logits still validate |
| `sh` | float32 | \[N, (L+1)², 3\] | coefficients in gsplat's basis order; color by E2 |

Metadata: `format`, `version`, `sh_degree`, `time_unit`, `tau0`, `tau1`, `producer`, and `render_conventions` (JSON: α\_max, ε, near, far, α skip threshold, transmittance stop, pixel-center offset 0.5, background). Importers clamp opacity to \[10⁻⁷, 1 − 10⁻⁷\] before taking the logit.

#### Rendering at any camera and time (`ftgs-render-request` v1, JSON → `ftgs-renders` v1, safetensors)

```json
{
  "format": "ftgs-render-request", "version": 1, "time_unit": "s",
  "background": [0, 0, 0], "sh_degree": null,
  "items": [
    {"id": "r0000", "timestamp": 1.2345,
     "camera": {"width": 640, "height": 480, "K": [[...]], "world_to_camera": [[...]]}}
  ]
}
```

The output stores `rgb/<id>` as float32 \[H, W, 3\] without clamping to 1, and `alpha/<id>` as float32 \[H, W, 1\]. Its metadata records the producer and the SHA-256 of the request. `sh_degree: null` means the model's full degree. Python callers use `render()` (§2.1) directly with the same semantics.

### 2.3 Dependencies and licenses

Every dependency's own code is under a license compatible with Apache-2.0, permissive except for the file-level copyleft of MPL-2.0 (tqdm, hypothesis), and none is GPL or AGPL. OpenCV's binary wheels carry LGPL libraries: every wheel ships FFmpeg (LGPL-2.1), and the non-headless Linux wheels also ship Qt 5 (LGPL-3). Since the owner's choices of 2026-10-07 no install of openftgs depends on OpenCV: the core replaced it (D-61) and RoMa is vendored without it (D-65). It appears only in the test environment, as a reference oracle. ffmpeg itself is a program the user installs, not a dependency. "Verified" means the LICENSE file was read from the project's repository during this work (Appendix A, #8, #10–#12, #19, #20, #33). The other entries come from general knowledge and must be confirmed by the release-time license scan.

| Package | Version | Use | Scope | License | Checked |
| --- | --- | --- | --- | --- | --- |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | ==1.5.3 | CUDA rasterization | core | Apache-2.0 | verified at tag v1.5.3 |
|  ↳ numpy, jaxtyping, rich, ninja, typing\_extensions | gsplat's pins | gsplat runtime | core | BSD-3, MIT, MIT, Apache-2.0, PSF-2.0 | jaxtyping verified |
| torch | ≥2.1, CUDA build | tensors, autograd, Adam | core | BSD-3-Clause | knowledge |
| [safetensors](https://github.com/huggingface/safetensors) | ≥0.4 | checkpoint and exchange files | core | Apache-2.0 | verified |
| [lpips](https://github.com/richzhang/PerceptualSimilarity) | 0.1.4 | perceptual loss and LPIPS metric | core | BSD-2-Clause | verified |
|  ↳ torchvision, scipy, tqdm | lpips pins | lpips runtime | core | BSD-3, BSD-3, MPL-2.0 AND MIT | knowledge |
| PyYAML | ≥6.0 | reading OpenCV-style YAML camera files (D-61) | core | MIT | knowledge |
| Pillow | ≥10 | image I/O | core | MIT-CMU | knowledge |
| SciPy | ≥1.10 | exact k-NN on the CPU with cKDTree (D-56) | core | BSD-3-Clause | knowledge |
| [RoMa, vendored as openftgs.\_vendor.romatch](https://github.com/Parskatt/RoMa) | commit 77f8d68 | dense matches for 4D initialization (D-65) | extra `[init]` | MIT; its DINOv2 code Apache-2.0 | verified |
|  ↳ torchvision, einops, loguru | imported by the vendored code | RoMa runtime | extra `[init]` | BSD-3, MIT, MIT | knowledge; imports read from RoMa's source |
| [fused-local-corr](https://github.com/Parskatt/fused-local-corr) | ≥0.2.2 | optional RoMa speed-up | extra `[init]` | MIT | verified |
| [DINOv2](https://github.com/facebookresearch/dinov2) weights | via RoMa | RoMa backbone | extra `[init]` | Apache-2.0 | verified (repository) |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim) | any | faster SSIM loss | extra `[fast]` | MIT | verified |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | any | reference oracle for LPIPS in U-23 | dev only | Apache-2.0 | verified |
| pytest, hypothesis, ruff, mypy, pip-licenses, scikit-image, opencv-python-headless | latest | tests and CI; scikit-image and OpenCV only as reference oracles (U-9, U-34 to U-36) | dev only | MIT, MPL-2.0, MIT, MIT, MIT, BSD-3, Apache-2.0 with bundled LGPL libraries | knowledge |
| ffmpeg (user-installed program) | ≥5.1 | decoding input videos in the converters | external tool, called as a separate process | LGPL-2.1+ or GPL, depending on the user's build; never linked, imported or redistributed | owner decision (D-48) |
| COLMAP (user-installed program) | any version whose image\_undistorter has blank\_pixels | undistorting SelfCap as its protocol prescribes (D-60) | external tool, called as a separate process | new BSD for COLMAP itself (general knowledge); the user's build may include other licenses; never linked, imported or redistributed | follows the owner's choice to support SelfCap (D-60) |

Notes:

- MPL-2.0 (tqdm, hypothesis) is file-level copyleft. Depending on the unmodified packages is compatible with an Apache-2.0 release; nothing is vendored.
- `albumentations` was a RoMa training dependency and is not installed since the vendoring of D-65; the release scan still rejects any non-MIT package of that name.
- Pretrained weights (LPIPS backbones, RoMa, DINOv2) are downloaded at run time and never redistributed; their terms are recorded in the release checklist separately from the code licenses.
- [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) (MIT, verified) is only needed to build SelfCap dynamic-region masks; it is an external tool, not a dependency.
- Datasets are never bundled. Neural3DV is CC-BY-NC-4.0 ([LICENSE](https://github.com/facebookresearch/Neural_3D_Video)); tests use synthetic scenes only.
- CI runs `pip-licenses` on the resolved environment and fails on GPL or AGPL; LGPL needs manual sign-off. After D-61 and D-65 only the test environment installs OpenCV, and nothing from it is distributed. The release checklist still reviews the shared libraries bundled inside dependency wheels, not only their declared licenses, and checks the vendored directory's license files.
- gsplat is pinned to 1.5.3 because its current main branch clamps per-primitive alpha at 0.99 instead of 0.999, which changes renders of near-opaque primitives. Changing the pin requires re-running the render conformance tests (U-7, U-8).
- Resolved by the owner's choices of 2026-10-07 (D-61, D-65): OpenCV's README says to install only one of its four packages per environment, because all of them provide the same cv2 module. No install of openftgs contains OpenCV any more; the test environment installs opencv-python-headless alone as a reference oracle. Earlier, RoMa's opencv-python would have come in through `[init]`. The vendored RoMa code was written against torch ≥ 2.5.1 (RoMa's pyproject), which \[init\] therefore requires.
- Tiny RoMa (RomaInitConfig.model = "tiny\_roma\_v1\_outdoor") downloads and runs XFeat's code through torch.hub from verlab/accelerated\_features; that repository's license is checked before the option is used.
- SelfCap is licensed for non-commercial research only, and modifications must stay open-source and non-commercial (§1.4). Nothing derived from it enters the repository, the wheels or CI.
- RoMa and OpenCV: RoMa calls OpenCV only in its pose-estimation helpers (findEssentialMat, findFundamentalMat, recoverPose), one benchmark (findHomography) and a training data loader (imread); matching never does. Its utils module imports cv2 when loaded, however, and that module is on the import path of roma\_outdoor, so import romatch fails without some OpenCV package. Matching itself needs only torch, torchvision, NumPy, Pillow, einops and loguru, plus fused-local-corr on Linux unless use\_custom\_corr is off; RoMa's other declared dependencies (albumentations, h5py, kornia, matplotlib, poselib, timm, wandb) serve training, benchmarks and plots. The owner chose to vendor RoMa's inference code without that import (D-65), over an upstream change, a --no-deps install with a placeholder cv2 module, and keeping opencv-python in \[init\].

### 2.4 Performance architecture

The targets are the paper's reference outcomes (§1.5): about one hour of training for a 300-frame sequence on one RTX 4090 (§3.3 names no dataset; B-1 measures a Neural3DV scene), and 450 FPS at 1080p. Four rules apply to every hot path \[D-49\]:

1. Python only orchestrates. Work per primitive, pixel, match or point runs in compiled code: gsplat's CUDA kernels; PyTorch's kernels, which use cuBLAS, cuDNN and cuSOLVER on the GPU and, on the CPU, SIMD-vectorized loops, a BLAS/LAPACK library (MKL or OpenBLAS, depending on the PyTorch build) and OpenMP threads; and kernels generated by torch.compile (Triton on the GPU, vectorized C++ on the CPU). No Python loop runs over primitives, pixels, matches, points or frames in a hot path. The package ships no compiled code of its own unless B-1 justifies D-50's custom operator, and it links no BLAS directly.
2. Every optimization is either exact or approximate. Exact ones change results only through floating-point rounding and are on by default once a benchmark (§3.4) shows a gain. Approximate ones are off by default, each has a row in Appendix B, and each must pass the L2 and L3 equivalence checks before it can become a default. The one approximate default is TF32 in cuDNN convolutions during training, PyTorch's own default; it reaches the perceptual loss and, when fused-ssim is not installed, the SSIM loss's Gaussian filtering (D-52).
3. A training iteration makes no host–device synchronization except the three inside every gsplat 1.5.3 render: reading back the number of tile intersections (`Intersect.cpp`), in packed mode reading back the number of visible primitive–camera pairs (Projection.cpp), and inverting the view matrices for SH colors (torch.inverse, which synchronizes on CUDA). Loss values and statistics accumulate on the GPU and are read every 100 iterations, the period of the growth and relocation events, which synchronize anyway.
4. The reference backend stays plain PyTorch, because its job is to be readable and testable, not fast.

#### Where the time goes

| Stage | Runs | Cost driver | Plan | Kind |
| --- | --- | --- | --- | --- |
| Temporal slice: E1, E3, E4, activations and the E8 terms | every iteration, all N rows | memory traffic over up to 3M rows | one fused kernel via torch.compile with dynamic N; eager fallback \[D-50\] | exact |
| Culling σ\_eff < 1/255 (D-32) | every iteration | — | gsplat 1.5.3 already drops primitives with opacity below 1/255 during projection; explicit compaction only if B-2 shows a gain \[D-57\] | exact |
| Rasterization, forward and backward | every iteration | active primitives × tiles × pixels | gsplat's CUDA kernels; `packed` set by B-2, which gsplat documents as changing memory and speed, not results; `sparse_grad` off; `radius_clip` 0 \[D-57\] | exact |
| Perceptual loss (LPIPS-VGG) | every iteration | 13 convolution layers over five resolutions, the first two at full resolution, on two images, plus a backward pass | frozen weights, so the backward pass computes input gradients only; target features in a separate no-grad pass, so the backward pass covers the prediction only; channels-last layout; cuDNN autotuning; TF32 convolutions \[D-52\] | exact except TF32 |
| SSIM loss | every iteration | 11×11 Gaussian filtering | fused-ssim (`[fast]`) with valid padding; PyTorch fallback, whose Gaussian filtering runs as cuDNN convolutions and so uses TF32 in training (D-52) | exact with fused-ssim; TF32 in the fallback |
| Adam step | every iteration | about 64 floats per row, read and written several times | PyTorch's fused Adam, foreach as fallback \[D-53\] | exact |
| Gradient statistics, growth and relocation | every iteration; events every 100 | O(N) gathers and scatters | vectorized and kept on the GPU: masked sums in unpacked mode, scatter-adds through gaussian\_ids in packed mode, so no host synchronization | exact |
| Image loading | every iteration | 4–6 MB per image | uint8 frame cache, prefetch in the known sampling order, pinned staging buffers, copies on a side stream \[D-51\] | exact |
| RoMa matching | once, one pair per camera and init frame | network inference | GPU, one pair per forward pass with deterministic cuDNN; pairs spread across GPUs with per-pair seeds \[D-56\] | exact |
| Triangulation and k-NN | once | tens of millions of 4×4 systems; about 10⁵ points per frame | batched float64 solves on the GPU; exact tiled k-NN \[D-56\] | exact |
| Dataset conversion | once | decoding, resizing, PNG encoding | parallel ffmpeg processes; vectorized area resizing; PNG encoding in a process pool | exact |
| Rendering and evaluation | on demand | active primitives | inference mode, activations computed once per model, temporal index, cameras batched per time \[D-55\] | exact |

Rasterization and the perceptual loss should dominate an iteration. At 1352×1014 an image has 27 times the pixels of a 224×224 ImageNet image, and LPIPS-VGG runs its convolutions on two such images and back-propagates through one. B-1 measures the real split; until then these are estimates.

#### Kernels \[D-50\]

FreeTimeGS adds only elementwise work per primitive to 3DGS: motion (E1), temporal opacity (E4), effective opacity (E3), the activations and the per-row terms of E8. One fused function, `temporal_slice`, computes all of them and the culling mask, with three implementations behind one signature:

- `eager`: plain PyTorch, used by the reference backend and as the fallback;
- `compiled`: the same code through torch.compile with dynamic shapes, so growth events do not trigger recompilation;
- `cuda`: a custom operator, written only if B-1 shows the compiled slice taking more than 10% of an iteration.

U-28 holds all three to the same values and gradients. CUDA graphs are not used, because they forbid host synchronization and shape changes, and every gsplat render synchronizes. Folding the motion into gsplat's projection kernel would save one pass over N but fork gsplat's CUDA code, so it waits until B-1 shows it is worth it.

#### Precision \[D-52\]

Parameters, the temporal slice, rasterization, the regularizer and the optimizer use float32, as does the SSIM loss with fused-ssim. During training, cuDNN convolutions use TF32, PyTorch's default (`torch.backends.cudnn.allow_tf32`); the flag is global, so it reaches LPIPS-VGG and, without fused-ssim, the PyTorch SSIM fallback, whose Gaussian filter runs as a convolution. Matmul precision stays "highest", which RoMa requires, so TF32 never reaches matrix multiplications. Evaluation clears the cuDNN flag for its forward-only pass, so reported metrics are plain float32. bfloat16 LPIPS (`perc_precision = "bf16"`) is an approximate option. Triangulation uses float64.

#### Data pipeline \[D-51\]

PNG files stay the stable interface (§2.2). For training, a frame cache holds every training image as uint8: one safetensors file per camera with a \[frames, H, W, 3\] tensor, built by `ftgs cache` or on first use and keyed by the scene manifest's SHA-256. The cache lives on the GPU when it fits in half the GPU memory that is free after a warm-up iteration, minus what growth to N\_max will add. Otherwise it is memory-mapped, so the operating system keeps as much of it in RAM as it can, and frames are read one at a time with safetensors' `get_slice`. A 300-frame Neural3DV scene with 20 training cameras at 1352×1014 takes about 25 GB, so it is usually memory-mapped; SelfCap's 60 frames (about 9 GB) can stay on a large GPU.

The sampler's seeded order (D-26) is known ahead of time. A prefetch thread copies the next 8 images into a ring of pinned buffers and sends them to the GPU on a side stream; the training stream waits on an event. Conversion to float32 happens on the GPU. B-6 requires the training stream to wait for data less than 1% of the time.

#### Threads and processes

- One Python thread issues all GPU work. The GIL does not slow it down, because kernels run asynchronously.
- Prefetching uses one or two threads; file reads and tensor copies release the GIL.
- CPU math (the reference backend and the CPU k-NN) uses PyTorch's intra-op threads, set once at startup to the number of physical cores unless `OMP_NUM_THREADS` or `MKL_NUM_THREADS` is set.
- The converters run one ffmpeg process per camera, at most `--jobs` at a time (default: one per four cores), each with ffmpeg's own decoding threads, and encode PNG in a process pool. Pool workers set PyTorch to one thread so cores are not oversubscribed.

#### Several GPUs \[D-54\]

Independent work is spread across GPUs without changing results: RoMa pairs, evaluation frames, render requests and the seeds of an L3 run. Training on several GPUs means more than one image per step, which departs from the paper's one-image steps (D-26). It is therefore an approximate, opt-in mode (`images_per_step = b`) built on gsplat's `distributed=True` rasterization, which follows Grendel. It applies Grendel's scaling rule: learning rates times √b, Adam's β₁ and β₂ raised to the power b, and the iteration calendar counted in images. Grendel reports 3–4× speed-ups on four GPUs at b = 4 with no PSNR loss on static scenes. Global quantities then use all ranks: E8 averages over every rank's primitives; Q₀.₉₉, the relocation draws and the growth cap use gathered g and σ, decided on rank 0 and broadcast; and a step with b images uses the mean of L\_reg(t\_j) over them. The mode must pass L3 against single-GPU training before it is recommended.

#### Rendering at a given time \[D-55\]

Rendering a trained model computes the activations once and builds a temporal index. Time is split into buckets one training frame interval wide. Each bucket lists the primitives whose effective opacity can reach 1/255 inside it, that is |t − μt| ≤ s·√(2 ln(255σ)), widened by a small margin so the list is a superset of what gsplat keeps. A render at time t passes only its bucket to gsplat, whose own test then decides, so images are unchanged. Cameras that share a time render in one gsplat call. Training cannot use the index, because μt and s change at every step.

#### Initialization and conversion

RoMa runs on the GPU, and its pairs can be spread across GPUs. Each unordered camera pair is matched once, as (A, B) with A first in the manifest's camera order, in its own forward pass (batch size 1) with cuDNN autotuning off and deterministic algorithms on, whatever `PerfConfig` says. It draws its matches from a generator seeded with (seed, frame slot, A, B), so the result does not depend on how pairs are distributed or ordered, given the same GPU model and software. Triangulation solves batched 4×4 systems in float64 on the GPU. The k-NN searches of E15 and of the scale initialization are exact: tiled brute force on the GPU with explicit coordinate differences, or SciPy's cKDTree on the CPU. Ties go to the lower index; cKDTree does not document its tie order, so its candidates are re-sorted by (distance, index), with extra neighbours queried while the last two distances are equal. Distances are not computed as ‖a‖² + ‖b‖² − 2a·b, because in float32 that cancellation can swap near neighbors \[D-56\].

Decoding is a one-off and should not be the bottleneck of conversion: Neural3DV has about 6,000 frames. Most of the time is expected to go into PNG encoding, which runs in a process pool at a fast compression level (PNG is lossless, so pixels are unchanged). `--hwaccel cuda` asks the user's ffmpeg for hardware decoding when its build supports it; its frames must match CPU decoding to within one 8-bit level (U-27), and the manifest records which decoder ran.

#### Video decoding options (evaluated at the owner's request)

The package does not read video through OpenCV: D-48 already decodes with the user's ffmpeg. decord was evaluated as an in-process replacement and rejected.

| Option | What it adds to the environment | License of the decoder | GPU decoding | Maintenance | Verdict |
| --- | --- | --- | --- | --- | --- |
| User-installed ffmpeg, run as a separate process (D-48) | nothing | that of the user's build | `-hwaccel cuda`, if the user's build has it | active | kept |
| decord 0.6.0 | an Apache-2.0 library plus FFmpeg 4.1.6, which its release workflow builds and packs into the Linux and macOS wheels | built with `--enable-gpl --enable-nonfree` and libx264, so GPL; FFmpeg treats nonfree builds as unredistributable (general knowledge); `pip-licenses` would report only Apache-2.0 | only when built from source with CUDA; the PyPI wheels are CPU-only | last commit July 2022, last tag v0.6.0; the Linux wheel is built for CPython 3.6 and retagged py3 | rejected |
| torchcodec | a BSD-3-Clause library that loads the user's FFmpeg (major versions 4 to 9, shared libraries) at run time | that of the user's build | NVDEC with the CUDA wheels, the default on Linux; needs an FFmpeg built with NVDEC support, otherwise falls back to the CPU | active; version 0.17 needs torch ≥ 2.11 and Python 3.10–3.14 | candidate if in-process or GPU decoding is ever needed |
| OpenCV `VideoCapture` | FFmpeg, inside every OpenCV wheel | LGPL-2.1 | not in the pip wheels (general knowledge) | active | not needed |

decord's strength is fast random access into compressed video during training. This package decodes each video once into PNG files and the frame cache, so it gains nothing from that.

#### Settings

```python
# openftgs.config: TrainConfig.perf
@dataclass
class PerfConfig:
    slice_impl: Literal["auto", "eager", "compiled", "cuda"] = "auto"   # D-50
    packed: bool = True                      # gsplat default until B-2 calibrates (D-57)
    precull: Literal["internal", "compact"] = "internal"               # D-32, D-57
    frame_cache: Literal["auto", "gpu", "mmap", "off"] = "auto"       # D-51
    prefetch: int = 8
    log_every: int = 100
    perc_precision: Literal["tf32", "ieee", "bf16"] = "tf32"          # bf16 is approximate (D-52)
    adam: Literal["fused", "foreach", "selective"] = "fused"          # selective is approximate (D-53)
    images_per_step: int = 1                 # > 1 is approximate and uses several GPUs (D-54)
    deterministic: bool = False              # cuDNN autotuning off; deterministic kernels where PyTorch has them; RoMa initialization always runs this way (D-56)
```

## 3. Test plan

### 3.1 Unit tests derived from the equations

Thirty-eight unit tests pin every equation and convention, each with a numeric pass criterion. CPU tests use the pure-PyTorch reference backend and run on any machine; tests marked GPU compare it with gsplat. Gradient checks run in float64 on configurations kept away from the clamps and thresholds of E6, where gradients are not smooth.

| ID | Target | Test | Pass criterion |
| --- | --- | --- | --- |
| U-1 | E1 motion | μx(μt) = μx; μx(t₁) − μx(t₀) = v(t₁ − t₀); autograd Jacobians vs closed form; `gradcheck` | exact; rtol 1e-6; Jacobians I, (t − μt)I, −v |
| U-2 | E4 temporal opacity | values at μt, μt ± s, μt ± 2s; symmetry; monotone decay; ŝ = −30 and \|t − μt\|/s = 10⁴ | 1, e^−0.5, e^−2 to 1e-7; outputs and gradients finite, no NaN |
| U-3 | E3 effective opacity | `opacities_t` vs sigmoid(ô)·σt; gradient w.r.t. ô | 1e-7 |
| U-4 | E14 invariance | render after (τ, μτ, d, v\_τ) → (aτ + b, aμτ + b, a·d, v\_τ/a) for (a, b) ∈ {(2, 5), (10⁻³, −3), (10³, 0)} | max abs diff ≤ 1e-5 (reference), ≤ 1e-4 (GPU) |
| U-5 | E5 covariance | Σ symmetric positive definite; eigenvalues = a²; q, 3q, −q identical; R(q) equals the gsplat-documented matrix and is orthonormal | 1e-6 |
| U-6 | E2 color | degree 0 gives C₀c₀₀ + 0.5; degrees 1–3 vs an independent real-SH evaluation written from the published constants; direction taken from the moved mean (a primitive passing the camera flips the sign of the l = 1 response); negative sums clamp to 0; inactive bands contribute nothing | 1e-6 |
| U-7 | E6 rasterization (GPU) | 50 random scenes (N ≤ 64, 64×48) rendered by both backends; single primitive with σ\_eff = 1 centered on a pixel center; tiny Gaussian at (u + 0.5, v + 0.5); two stacked primitives with α = 0.999; primitive at z = 0.005 | RGB and alpha ≤ 1e-4; center alpha 0.999; peak at pixel (u, v); second layer excluded, pixel = 0.999·c₁ + 0.001·c\_bg; near primitive invisible |
| U-8 | culling (D-32) | render with and without pre-culling σ\_eff < 1/255 | images identical (bitwise on reference, ≤ 1e-6 GPU); culled primitives get exactly zero render gradient |
| U-9 | E7 losses | L1 on constant images; SSIM(I, I); SSIM vs scikit-image `structural_similarity` (Gaussian weights, σ = 1.5, population covariance, data range 1, valid region); LPIPS(I, I); \[0, 1\] → \[−1, 1\] mapping; weighted sum | analytic; L\_ssim = 0; 1e-4; 0; exact; 1e-6 |
| U-10 | E8 4D regularization | value = mean(σ·σt(t)); gradient w.r.t. ô = (1/N)σtσ(1 − σ); all other gradients; independence from the camera; applied at every iteration by default (E9) and at none after reg\_until when it is set | 1e-7; 1e-7 relative; exactly 0; identical; weight 1e-2 at it = 1 and at it = T\_it by default; 0 after reg\_until |
| U-11 | full forward | `gradcheck` (float64) of L1 + L\_reg through the reference renderer, 3 primitives, 8×8 image, all nine tensors | passes at eps 1e-6, atol 1e-5 |
| U-12 | gsplat gradients (GPU) | per-tensor gradients, gsplat vs reference, 20 random scenes | cosine ≥ 0.999 and relative L2 error ≤ 1e-3 |
| U-13 | E11 relocation math | n = 1 unchanged; n = 2 gives 1 − √(1 − σ); 1 − (1 − σ\_new)^n = σ; scale factor (computed in float64) vs an independent float64 evaluation of E11; clamp-active cases (σ\_new < 0.005, e.g. σ\_j = 0.02 with n = 5), where a\_new uses the unclamped σ\_new and σ\_new is clamped afterwards; vs gsplat `compute_relocation` (GPU, float32 kernel); render before vs after relocating 30 dead primitives onto 10 targets, at the targets' μt, with copy\_position = "stack" | 1e-10 relative for the first five (E11 runs in float64); 5e-4 relative against gsplat; PSNR ≥ 30 dB |
| U-14 | relocation procedure | ô = ô\_dead (the float32 logit of 0.005) counts as live; empty dead or live set is a no-op; 10⁵ draws over 5 fixed scores (χ² test); every moved row equals its target's updated row except its position, whose draws (10⁵ of them) match the target's Gaussian in mean and covariance; N unchanged; moments zero only on target and moved rows; step counters unchanged; statistics reset; same seed → same result | bitwise where stated; χ² p > 0.001; covariance within 1% |
| U-15 | E10 score | ĝ ∈ \[0, 1\]; ĝ = 1 above Q₀.₉₉; Q = 0 gives ρ = 0.5σ; never-visible primitives have g = 0 | exact |
| U-16 | gradient statistics | a 2D-mean gradient of (1, 0) on a 640×480 image accumulates 320; invisible primitives are not counted; the same through gaussian\_ids in packed mode, including a primitive seen by several cameras; no host synchronization (sync debug mode) | exact |
| U-17 | E12, E13 and the calendar | rates at it = 1, T\_it and the midpoint; ηv(1) ≠ ηv,0 + 1; rates never increase; L\_active steps at 1,000/1,001, 2,000/2,001, 3,000/3,001; growth exactly at 500, 600, …, 15,000; relocation exactly at 500, 600, …, 25,000 | 1e-12 relative; 146 growth and 246 relocation events |
| U-18 | E14 time normalizer | τ → t → τ round trip; Δt\_f with irregular timestamps; single-frame case; unsynchronized cameras give the same Δt\_f as synchronized ones (D-46); warning outside \[τ₀, τ₁\] | 1e-12; exact |
| U-19 | initialization | E15 on translated point sets (shift < half the point spacing); backward difference on the last frame; single frame gives v = 0; s₀ = stride·Δt\_f; per-frame 3-NN scales with the 1e-7 floor; noise-free DLT; filters drop points behind a camera or above 2 px error; seeded per-frame subsets of at most k\_stride·B points; frame-slot assignment with unsynchronized cameras (D-47); no test-split image is read (D-66); a pair chosen by both of its cameras is matched once; supplied unsynchronized points without durations or frame\_ids get s₀ = Δt\_f and nearest-slot groups; near the ends of an unsynchronized rig, a camera with no image within Δt\_f/2 of t\_k sits the slot out; scales come from the kept points after the per-frame subsample (D-25) | exact or 1e-6 |
| U-20 | camera conventions | known 3D point projects to the expected pixel in both backends; OpenCV → continuous shift of +0.5; resizing scales projections by r; Neural3DV converter reproduces a known OpenCV pose from a synthetic LLFF row | 1e-5 px; exact |
| U-21 | checkpoints | save/load round trip of every tensor, moment, step counter, generator state, sampler and statistic; 300 iterations straight vs 150 + resume + 150, under a test calendar with events every 50 iterations from 100 (growth up to 200, relocation up to 250), so events fall on both sides of the resume | bitwise (reference, CPU); on GPU, with no event in the window: identical image order and rates, each tensor within 1e-4 relative L2 error, and a PSNR of at least 50 dB between the two runs' renders |
| U-22 | exchange format | export → import → export; renders before vs after; quaternion sign canonical; a model with an opacity logit of 20 exports and validates; validator rejects NaN, durations ≤ 0, opacity outside (0, 1), wrong shapes, unknown major version | 1e-6 relative; 1e-6; exact; exported opacity 1 − 10⁻⁷; rejected with exit code 2 |
| U-23 | E16 metrics | PSNR at MSE 10⁻⁴ is 40 dB; DSSIM = (1 − SSIM)/2; DSSIM₂ < DSSIM₁ on a noisy pair; LPIPS backbone per dataset; SSIM\_k of two constant images a and b against its closed form (2ab + C₁)/(a² + b² + C₁), C₁ = (0.01k)²; a render above 1 is clamped before metrics; per-image averaging; dynamic-region crop and black fill on a synthetic mask (D-67); LPIPS with \[−1, 1\] inputs for both backbones, and the raw-input VGG variant, against torchmetrics' LPIPS with normalize=True and normalize=False (D-68) | 1e-6; exact for the crop and fill; LPIPS within 1e-4 |
| U-24 | determinism | two seeded 100-iteration runs on the reference backend | bitwise identical |
| U-25 | growth (E17, D-18, D-42 to D-45) | candidates are exactly those with g > 0.0002; clone vs split boundary at 0.01·S; 10⁵ split offsets have the parent's covariance; children have a/1.6 and copy the other tensors; the parent is removed; new rows have zero moments; with a full cap only the largest-g candidates grow, ties to the lower row index; new rows inherit their parent's g\_sum and count; N never decreases and stops growing after 15,000; growth runs before relocation in an event | exact; covariance within 1% |
| U-26 | SH degree parameter (D-3) | sh\_degree ∈ {0, 1, 2, 3} accepted, other values rejected; tensor shapes per L; L = 0 has an empty shN and no shN optimizer group; checkpoint and FTGS-IF round trips for each L; a 50-iteration smoke run for each L | exact; runs finish without NaN |
| U-27 | ffmpeg decoding (D-48) | a 10-frame test video made with ffmpeg's lavfi testsrc goes through the converter at resize 1.0; skipped when ffmpeg is absent; a run with a wrong --ffmpeg or --ffprobe path; the same video with --hwaccel cuda when the build supports it | 10 frames in order, pixels equal to ffmpeg's own PNG decode; exit code 2; hardware decoding within one 8-bit level of CPU decoding, with differing pixels counted |
| U-28 | temporal slice (D-50) | eager, compiled and, if built, CUDA implementations on random models with N from 1 to 10⁶, resized between calls as at growth events | values ≤ 1e-6 relative, gradients ≤ 1e-5 relative; at most 2 recompilations over 10 resizes |
| U-29 | synchronization policy (D-49) | 20 iterations away from any event, under torch.cuda.set\_sync\_debug\_mode("warn") | every synchronization warning comes from inside the gsplat render call |
| U-30 | frame cache and prefetch (D-51) | cache against PNG decoding for every image; prefetch order against sampler order; GPU and memory-mapped placements; resume in mid-epoch | bitwise equal; same order; same batches |
| U-31 | Adam kernels (D-53) | 100 steps of fused and foreach Adam on identical gradients; selective Adam with a fixed visibility mask; selective Adam with an all-true mask against Adam without bias correction, computed in float64 | parameters ≤ 1e-6 relative; with selective Adam, invisible rows and their moments unchanged bitwise; all-true mask within 1e-6 relative |
| U-32 | render temporal index (D-55) | renders with and without the index at 100 random times, including bucket edges and primitives exactly at the 1/255 bound | identical images (bitwise on the reference backend, ≤ 1e-6 on GPU) |
| U-33 | parallel initialization (D-56) | 1 against 3 shards (simulated on one GPU) and shuffled pair order, with D-56's deterministic RoMa settings; k-NN against float64 brute force on clustered points 10⁻⁴·S apart and far from the origin; duplicated points, whose ties go to the lower index | identical InitPoints, bitwise, on one GPU model and software stack; identical neighbors |
| U-34 | area resizing (D-33, D-61) | random images at ratios 0.5, 0.25, 0.37 and 1.0 against an independent float64 integration over pixel squares; ratio 0.5 against OpenCV's INTER\_AREA as a reference | ≤ 1e-12 in float64; within one 8-bit level of OpenCV |
| U-35 | undistortion (D-61) | source-position maps for 4-, 5-, 8- and 12-coefficient models against OpenCV's initUndistortRectifyMap; images against OpenCV's undistort; zero coefficients; a 14-coefficient (tilted) input | maps ≤ 1e-3 px; images of smooth test patterns (intensity gradients below 32 levels per pixel) within one 8-bit level away from the border; identity for zero coefficients; tilted input rejected with a clear error |
| U-36 | OpenCV-style YAML reader (D-61) | files written by OpenCV's FileStorage with the %YAML:1.0 header, opencv-matrix entries of several shapes and types, sequences of names and scalars | every value equal to what OpenCV reads back, bitwise |
| U-37 | SelfCap converter logic (D-59, D-60) | a generated mini-archive: 3 cameras, 12-frame videos, YAML calibration and a sync.json; window 4 ≤ n < 8 and test camera 0002, passed as window and test\_camera; once with --undistort internal and once with a stub COLMAP that records its input | frames 4–7 only, in order; τ = n/60 minus the offset (n/60 with --no-sync); test camera excluded from training; the COLMAP model gets FULL\_OPENCV and cx, cy + 0.5; exit code 2 without COLMAP unless --undistort internal, and exit code 2 when Rot disagrees with R or a ccm is not the identity |
| U-38 | vendored RoMa (D-65) | every vendored file's SHA-256 against VENDOR.json; importing the vendored RoMa in an environment with no OpenCV package and with xformers installed; a search of the vendored tree for romatch imports outside the vendored path, function-local ones included; on a GPU, in a separate CI job, match() on two fixed image pairs against upstream romatch at commit 77f8d68 with the same weights and inputs | hashes equal; the import succeeds and neither cv2 nor xformers enters sys.modules; no such import found; warps and certainties within 1e-5 on the same GPU and library versions |

Property-based variants (hypothesis) of U-1, U-2, U-5 and U-13 draw random shapes and values within the stated domains. scikit-image (BSD-3-Clause) is a test-only dependency for U-9.

**Dataset converter checks** (run when the dataset is available; not part of CI):

| ID | Target | Test | Pass criterion |
| --- | --- | --- | --- |
| C-1 | Neural3DV converter (D-28) | Sampson epipolar error of feature matches between neighbouring cameras at frame 0, with the converted cameras and with the rotation columns permuted | median ≤ 1 px at the training resolution; every permutation ≥ 10× worse |
| C-2 | ENeRF-Outdoor converter | same test after undistortion | median ≤ 1 px |
| C-3 | both converters | triangulate the matches and reproject into a third camera | median reprojection error ≤ 1.5 px |
| C-4 | SelfCap converter (D-59, D-60) | Sampson epipolar error of matches on the static background between neighbouring cameras at the first frame, after COLMAP undistortion; then the same with --undistort internal; and, needing only COLMAP, a synthetic image of Gaussian dots at known 3D points, rendered through a known distorted camera and undistorted by COLMAP | median ≤ 1 px at the training resolution in both modes; dot centroids within 0.1 px of where the read-back PINHOLE camera projects the dots (a wrong pixel-center convention gives about 0.5 px) |
| C-5 | SelfCap timing (D-59) | on corgi, for camera pairs whose offsets differ by more than one frame, match the moving subject between camera A at frame n and camera B at the frame nearest to A's corrected time; triangulate and reproject into a third camera; repeat with both cameras at frame n | the corrected pairing gives the lower median reprojection error |

### 3.2 Synthetic-scene recovery tests

Eleven synthetic scenes with known ground truth check that training recovers motion, timing and appearance, not just that the code runs. Scenes S-1 to S-8 are generated by a ground-truth FreeTimeGS model rendered with the reference backend (CPU) or gsplat (GPU). S-9 and S-11 use an independent analytic ray caster so the model is not graded only on data of its own form.

Shared setup: cameras on a ring or hemisphere looking at the origin, 64×64 to 128×96 pixels, black background, scene scale S ≈ 1. Schedules are shortened in proportion: every iteration count of the calendar (growth and relocation windows, the 100-iteration event period, the 1,000-iteration SH interval) is multiplied by T\_it/30,000 and rounded, with a minimum of 1. Growth is off (growth = "none") in S-1 to S-3, S-7 and S-8, whose criteria match primitives one to one. Ground-truth files and seeds are committed. A threshold marked "calibrated" is set once from analytic expectations plus a margin, then frozen; changing it needs a logged reason.

| ID | Scene and setup | What must be recovered | Pass criterion |
| --- | --- | --- | --- |
| S-1 | One static Gaussian, 8 ring cameras, 5 frames, s = 10; init position offset by 0.05·S, gray color; 2,000 iterations, relocation and regularization off | position, color, opacity | position error ≤ 1e-3·S; color ≤ 0.01; opacity ≤ 0.02; held-out view PSNR ≥ 40 dB |
| S-2 | 8 Gaussians moving linearly, \|v\| = 0.5–2 per unit t in random directions, s = 2 (always present), 12 hemisphere cameras, 30 frames; init at true trajectory points with v = 0; 5,000 iterations | trajectories and velocities | max over frames of ‖μx(t) − μx\_gt(t)‖ ≤ 0.01·S; velocity error ≤ 5% and ≤ 3°; held-out cameras PSNR ≥ 35 dB |
| S-3 | 12 Gaussians with s\_gt = 3 frames and centers spread over 30 frames; init with μt shifted by +2 frames and s = 1 frame | time centers and durations | \|μt − μt\_gt\| ≤ 0.5 frame; s within 25%; PSNR at held-out times ≥ 33 dB |
| S-4 | Points sampled on spheres moving between frames, displacement < half the sampling spacing | E15 initialization and its value | median velocity error ≤ 5%; with the same budget, zero-velocity init ends with lower PSNR than E15 init (direction of Table 4) |
| S-5 | S-2 plus 2,000 small random Gaussians; λ\_reg = 0 vs 10⁻² | effect of E8 | at the end of training the share of primitives with σ > 0.9 is lower with regularization; final PSNR not lower by more than 0.5 dB |
| S-6 | S-2 with half the primitives dead (σ = 10⁻³) at random places away from the objects; relocation on, growth off | E10, E11 and relocation | N constant; after each relocation event no primitive has σ < 0.005 except newly dead ones; PSNR with relocation ≥ PSNR without + 1 dB (calibrated) |
| S-7 | S-2 trained on even frames, tested at odd frames | interpolation at unseen times | odd-frame PSNR within 1 dB of even-frame PSNR |
| S-8 | S-2 with each camera offset by a random fraction of a frame (unsynchronized) | per-image timestamps | velocity error ≤ 1.2× the synchronized run's error |
| S-9 | Textured spheres moving linearly, rendered by an analytic ray caster (NumPy, in the test suite); 16 cameras, 20 frames, 128×96; init from surface samples, or RoMa when `[init]` is installed (GPU) | end-to-end quality on non-Gaussian data | held-out PSNR ≥ 28 dB and SSIM ≥ 0.9 (calibrated) |
| S-10 | Tiny scene, 200 iterations on CPU | training stability | mean loss of the last 20 iterations below that of the first 20; no NaN; checkpoint and resume succeed |
| S-11 | Textured spheres as in S-9, initialized from surface samples on one hemisphere only | coverage of surfaces with no initial points | with growth = "gradient", held-out PSNR on the uninitialized half within 1 dB of a fully initialized run (calibrated); the same metric is reported for growth = "none" started at the same final N, the missing points replaced by random low-opacity primitives |

Trajectories, not raw (μx, μt) pairs, are compared in S-2: shifting μt by δ and μx by v·δ leaves E1 unchanged, so the pair is not identifiable when the duration is long.

### 3.3 Black-box differential harness

The harness compares two implementations, A (this package) and B (any other), through files, commands and metrics only; it imports neither. It runs four levels, from converter checks to multi-seed statistics, and writes a report that describes every failure as an observable symptom.

&#91;embedded content: harness data flow · 2 adapters, 4 levels\]

Every arrow is a file on disk: the harness writes the cases, each adapter answers with files, and only the harness computes comparisons and metrics.

#### Adapter contract

Each implementation ships a small adapter executable, owned by that implementation's team. All data crosses the boundary in the §2.2 formats plus two harness formats (forward queries and answers).

| Command | Input | Output | Levels |
| --- | --- | --- | --- |
| `info` | — | JSON: name, version, supported query kinds, rasterizer constants | all |
| `export MODEL OUT` | native model | FTGS-IF file | L0, L1, L3 |
| `import IF OUT` | FTGS-IF file | native model | L0, L1 |
| `render MODEL REQUEST OUT` | native or FTGS-IF model, render request | `ftgs-renders` file | L0, L1, L3 |
| `forward QUERIES OUT` | `ftgs-forward-queries` (safetensors) | `ftgs-forward-answers` (safetensors) | L2 |
| `train SCENE INIT CONFIG SEED OUT_DIR` | scene manifest, init points, harness config, seed | native checkpoint and `train_log.jsonl` (iteration, loss terms, N) | L3 |

`harness_config.json` states hyperparameters in this spec's terms, the TrainConfig field names of §2.1 (for example growth.n\_max). Each adapter maps them to its native settings; a setting it cannot honor is reported as "not comparable", never silently dropped.

#### L0 · Boundary converters

Twenty FTGS-IF models (random and trained) go through import → export in each implementation. Fields must match within 1e-6 relative (opacity compared after the logit clamp, quaternions up to sign), every produced file must pass the schema validator, and renders before and after the round trip must agree within the L1 tolerance.

#### L1 · Identical model, near-exact renders

Both implementations load the same FTGS-IF file and render the same request of at least 200 items. Items cover training cameras, interpolated novel cameras, times on frames, between frames, at τ₀ and τ₁, and 10% outside the training range. Per item the harness computes the PSNR between A and B, max and mean |Δ|, the share of pixels with |Δ| > 1/255, and the alpha difference.

| Tier (from both `info` replies) | Pass |
| --- | --- |
| Same rasterizer and constants | PSNR(A, B) ≥ 60 dB and max \|Δ\| ≤ 1e-3 |
| Different rasterizers | PSNR(A, B) ≥ 45 dB and ≤ 0.5% of pixels with \|Δ\| > 1/255 |

Ten probe scenes isolate behaviors so that a mismatch points at one quantity:

| Probe | Content | Swept variable |
| --- | --- | --- |
| P1 | one static isotropic primitive at the image center | sub-pixel position |
| P2 | one moving primitive | time |
| P3 | one static primitive with a short duration | time |
| P4 | one near-opaque primitive | opacity 0.9 → 1 |
| P5 | one primitive with only l ≥ 1 SH energy | viewing direction |
| P6 | a primitive crossing the near plane over time | time |
| P7 | two overlapping primitives whose depth order swaps | time |
| P8 | a primitive with σ\_eff just above and below 1/255 | opacity |
| P9 | a large primitive partly off-screen | image position |
| P10 | any probe at times outside \[τ₀, τ₁\] | time |

#### L2 · Forward behavior on random inputs

The harness writes seeded random inputs, including edge values: s from 10⁻⁶ to 10², |t − μt|/s up to 10⁴, opacities near 0 and 1, view directions along the axes.

| Query kind | Inputs | Answer | Tolerance |
| --- | --- | --- | --- |
| `motion` | FTGS-IF arrays, K times | μx(t), \[K, N, 3\] | 1e-6 relative |
| `temporal_opacity` | FTGS-IF arrays, K times | σt, \[K, N\] | 1e-6 |
| `effective_opacity` | FTGS-IF arrays, K times | σ·σt, \[K, N\] | 1e-6 |
| `sh_color` | FTGS-IF arrays, camera centers, times | colors, \[K, N, 3\] | 1e-5 |
| `losses` | image pairs; FTGS-IF and t for L\_reg | L1, 1 − SSIM, LPIPS-VGG, L\_reg, total | 1e-5 relative; LPIPS 1e-3 |
| `lr_schedule` | iterations, S, Δt\_f | rate per tensor | 1e-9 relative |
| `relocation_math` | (σ, a, n) triples | σ\_new, a\_new | 1e-5 relative |
| `velocity_init` | point sets at two times | velocities | 1e-6 |
| `relocation_sampling` | fixed scores, dead count, 50 seeds | counts per target | χ² test, p > 0.001 |
| `grad_stats` | FTGS-IF arrays, one camera, t, ground-truth image, iteration | per-primitive NDC-scaled norm of the 2D-mean gradient of E9, and visibility, for one step (D-12) | 1e-3 relative |
| `relocation_score` | base opacities, gradient sums and visible counts | ρ (E10, D-13) | 1e-6 |
| `spatial_scale` | cameras with train/test labels | S (D-27) | 1e-9 relative |
| `growth_decisions` | g, scales, S, τ\_pos, N, N\_max | clone and split masks after the cap (D-18, D-44) | exact |

A query kind missing from an adapter's `info` reply is reported as "not comparable".

#### L3 · Multi-seed statistics

1. Scenes: S-9, plus real scenes the tester can access (e.g. two Neural3DV scenes, 300 frames, test view cam00).
2. A and B train from the same init file and harness config with seeds 0…K−1, K = 5 by default. A second independent seed set of A (A′) gives the null distribution.
3. Each run exports FTGS-IF and renders the held-out request. The harness computes every metric itself from the saved renders (E16: PSNR, SSIM₁, SSIM₂, DSSIM, LPIPS-Alex, LPIPS-VGG), so metric code cannot differ between A and B.
4. Image metrics per scene: Welch's t-test plus a TOST equivalence test with margins of ±0.25 dB PSNR, ±0.005 SSIM and ±0.01 LPIPS, Holm-corrected across metrics and scenes. The verdict is "equivalent" when TOST passes, "different" when Welch rejects, otherwise "inconclusive" with the observed seed spread and the K needed for 80% power.
5. Distributions from the exported models: opacity, log duration, speed ‖v‖, time centers, log of the largest scale, primitive count, and per-time PSNR on the held-out view. A–B distances (KS statistic, Wasserstein-1) are tested against the A–A′ distances by a seed permutation test; p < 0.01 flags "distribution differs".
6. Training traces: loss terms and N at iterations 1k, 7k, 15k and 30k, compared by median and spread.

#### Report format

The harness writes `report.json` (machine-readable), `report.md` (generated from it) and an `artifacts/` folder of diff images and plots. Exit code 0 means all checks passed, 1 at least one failed, 2 a harness or configuration error.

```json
{
  "format": "ftgs-harness-report", "version": 1,
  "run": {"started": "2026-10-06T09:00:00Z", "harness_version": "1.0.0", "config_sha256": "…"},
  "implementations": {"A": {"name": "…", "version": "…", "info": {}}, "B": {"name": "…", "version": "…", "info": {}}},
  "summary": {"L0": "pass", "L1": "fail", "L2": "pass", "L3": "inconclusive", "failed_checks": 3},
  "checks": [
    {
      "id": "L1.render.P2.r0007",
      "level": "L1",
      "status": "fail",
      "spec_refs": ["E1"],
      "inputs": {"model": "cases/P2.ftgs-if.safetensors", "request": "cases/P2.request.json", "items": ["r0007"]},
      "metric": "psnr_A_vs_B_db", "expected": ">= 60", "observed": 23.4,
      "symptom": "Both renders show the primitive, but B's copy sits 6.1 px left of A's at t = 0.83. The offset is 0 at the primitive's time center and grows linearly with |t - time_center| (7.3 px per unit of t). Alpha coverage agrees within 1e-4.",
      "localization": {"times": [0.83], "cameras": ["c03"], "pixel_bbox": [212, 140, 250, 176], "trend": "linear in |t - time_center|"},
      "repro": "ftgs-harness rerun L1.render.P2.r0007",
      "artifacts": ["artifacts/P2_r0007_A.png", "artifacts/P2_r0007_B.png", "artifacts/P2_r0007_absdiff.png"]
    }
  ]
}
```

Rules for the `symptom` field:

1. Describe only inputs and outputs: which files and items, which times and cameras, magnitude, sign, spatial pattern, and the trend against the swept variable.
2. Never name a function, variable, file or line of either implementation, and never state a suspected cause.
3. Include at least one number with its unit and the controlled variable it changes with.
4. Use the fixed pattern vocabulary: offset, scale change, blur, missing primitive, extra primitive, color shift, view-dependent color shift, temporal shift, temporal widening or narrowing, opacity saturation, border band, NaN/Inf, crash, schema violation.
5. Put spec items (E-n, D-n) only in `spec_refs`, taken from the check's definition, never inferred from code.
6. Give a one-command reproduction with the exact input files.

Trends are found automatically: the probes sweep one variable at a time, and the harness fits constant, linear and quadratic models to the discrepancy and reports the best fit with its coefficients. `report.md` lists failed checks first, grouped by pattern, each with its symptom sentence, numbers, thumbnails and reproduction command.

```
ftgs-harness run --a ADAPTER_A --b ADAPTER_B --config harness_config.json --levels L0,L1,L2,L3 --out RUN_DIR
ftgs-harness rerun CHECK_ID [--out RUN_DIR]
```

### 3.4 Performance benchmarks

Benchmarks measure speed and memory, not correctness: the equivalence of every optimized path is pinned by U-8 and U-28 to U-33. `ftgs bench` runs them and writes JSON with the GPU model, driver, CUDA, PyTorch and gsplat versions. CI runs B-2 to B-6 on one fixed GPU runner, on a synthetic scene the size of the Neural3DV setup (20 cameras, 300 frames, 1352×1014, 1M primitives), and fails when a budgeted figure gets more than 10% worse than its stored baseline. Datasets never enter CI (§2.3), so B-1, B-7 and B-8 run on real data before each release, and their results are stored with it. Budgets marked "calibrated" are set from the first reference run.

| ID | Measures | Setup | Budget or purpose |
| --- | --- | --- | --- |
| B-1 | time per iteration by stage (temporal slice, rasterization forward and backward, each loss, Adam, statistics, events, data wait) from CUDA events; peak memory | 300-frame Neural3DV scene at 1352×1014; N = 0.5M, 1M and 3M | full training ≤ 60 min on an RTX 4090 (paper §3.3) |
| B-2 | rasterization variants: `packed` on and off; D-32 compaction versus gsplat's internal culling | B-1 scene; 5%, 20% and 100% of primitives active | sets D-57's defaults; images identical (U-8) |
| B-3 | temporal-slice implementations | N from 10⁵ to 3·10⁶, with resizes as at growth events | sets D-50's choice; recompilations counted |
| B-4 | LPIPS-VGG in IEEE float32, TF32 and bfloat16, channels-last on and off; SSIM fused and in PyTorch | 1352×1014 and 1920×1080 | sets D-52's defaults |
| B-5 | Adam: foreach, fused and selective | N = 1M and 3M | fused no slower than foreach |
| B-6 | share of iteration time spent waiting for data, per cache placement | B-1 scene | < 1% |
| B-7 | rendering speed, one camera, with and without the temporal index | final B-1 model at 1920×1080, and SelfCap models, for which Table 3 reports 467 FPS | ≥ 450 FPS on an RTX 4090 (paper Fig. 1; calibrated) |
| B-8 | wall time of initialization (RoMa per pair and per GPU, triangulation, k-NN) and conversion (per camera, with and without hardware decoding) | one Neural3DV scene | reported; regression rule only |

`ftgs train --profile START:COUNT` records a PyTorch profiler trace of COUNT iterations from START. Each stage runs inside an NVTX range, so external GPU profilers show the same stage names.

## Appendix A — Source log

Every source consulted is listed below with what was taken from it; all access was on 2026-10-06 and 2026-10-07. No third-party FreeTimeGS implementation was searched for, opened or encountered, and one documentation query returned source code of a docs-only dependency (#36). Code links on the project page and gsplat's deformation-field module were deliberately left unopened, as were EasyVolcap and the extraction script that the SelfCap card links to.

| # | Source | What was taken |
| --- | --- | --- |
| 1 | Project doc "FreeTimeGS's paper" (this claude.ai project) | Links to the CVF paper and project page; instruction to also use arXiv and the supplementary |
| 2 | [FreeTimeGS project page](https://zju3dv.github.io/freetimegs/) | Authors, venue, abstract, pipeline summary; SelfCap is released by request form (it has since been published on Hugging Face, #25). Listed code links ("Renderer", "Framework") and demos were not opened |
| 3 | [FreeTimeGS, arXiv 2506.05348 v2 (HTML)](https://arxiv.org/html/2506.05348v2), read through the Hugging Face paper mirror `hf://papers/2506.05348/paper.md` | Full text including the supplementary (Appendices A–B): Eqs. 1–7; §1 early-stage regularization; §3.2 initialization and velocity annealing; §3.3 hyperparameters; §4 datasets, metrics, Tables 1–5 and ablations; Tables 6–11 (LPIPS backbones, SSIM₂, primitive counts, model sizes); dynamic-region evaluation |
| 4 | [CVF open-access PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.pdf) (user-provided) | Not read: the server answered HTTP 403. Equation numbering was not cross-checked against this version |
| 5 | arXiv abstract page; CVF HTML page | Not read: fetch permission prompts timed out. The CVF supplementary link was therefore not obtained; source #3 contains the supplementary |
| 6 | [3DGS, Kerbl et al. 2023 (arXiv 2308.04079)](https://arxiv.org/abs/2308.04079), via the Hugging Face mirror | Eqs. 3–7 (blending, Gaussian, projected covariance, Σ = RSSᵀRᵀ, loss with λ = 0.2); §5.1 sigmoid and exponential activations, 3-NN scale init, position-only exponential LR; §5.2 densification constants; §6 rasterizer; §7.1 resolution warm-up and one SH band per 1,000 iterations |
| 7 | [3DGS as MCMC, Kheradmand et al. 2024 (arXiv 2404.09591 v3)](https://arxiv.org/abs/2404.09591), via the Hugging Face mirror | Eq. 9 relocation; dead threshold 0.005; multinomial target choice applied after all draws; moment reset of targets; 5% growth; 500 warm-up iterations; 1.6e-4 → 1.6e-6 position rate. General literature, not cited by FreeTimeGS |
| 8 | [gsplat](https://github.com/nerfstudio-project/gsplat), tag v1.5.3 (git clone) | Apache-2.0 LICENSE; `rasterization()` signature and conventions; data-conventions doc (quaternion matrix, view matrix); SH direction (mean − camera center), +0.5 and clamp; kernels (pixel centers +0.5, α clamp 0.999, skip < 1/255, exclusive stop when transmittance ≤ 1e-4, opacity-aware 3.33σ extent, Jacobian clamp); MCMC strategy, relocate and sample\_add ops, `compute_relocation` kernel; default strategy (NDC gradient scaling, 3DGS defaults); example trainer (learning rates, Adam ε and betas, scene scale ×1.1, SH interval, init opacity 0.1, 3-NN scale init (root mean square of the three distances), SSIM "valid" padding, LPIPS settings (AlexNet inputs mapped to \[−1, 1\]; VGG fed \[0, 1\] images, described as the 3DGS repository's convention)); camera-centroid scene scale; dependency list; the trainer computes the loss on unclamped renders, clamps renders to \[0, 1\] before metrics and averages metrics per image (E7, D-67) |
| 9 | gsplat main branch at commit 512d366 (2026-09-19, version 1.6.0) | Screened for FreeTimeGS code: none (no file mentions it); `gsplat/contrib/dynamic` (deformation fields) not opened. Kernel constants (α clamp 0.99 here) motivated the 1.5.3 pin. Pure-PyTorch reference functions informed the reference backend |
| 10 | LICENSE files and packaging metadata only (metadata clones; no source code read): [RoMa](https://github.com/Parskatt/RoMa), [lpips](https://github.com/richzhang/PerceptualSimilarity), [fused-ssim](https://github.com/rahul-goel/fused-ssim), [torchmetrics](https://github.com/Lightning-AI/torchmetrics), [safetensors](https://github.com/huggingface/safetensors), [jaxtyping](https://github.com/patrick-kidger/jaxtyping), [DINOv2](https://github.com/facebookresearch/dinov2), [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2), [tyro](https://github.com/brentyi/tyro), [albumentations](https://github.com/albumentations-team/albumentations), [fused-local-corr](https://github.com/Parskatt/fused-local-corr) | Licenses in §2.3; RoMa's dependency list (requirements and pyproject); lpips 0.1.4 dependency list |
| 11 | [Neural 3D Video dataset README and LICENSE](https://github.com/facebookresearch/Neural_3D_Video) | cam00 is the held-out test view; poses use the NeRF/LLFF pose format; CC-BY-NC-4.0 |
| 12 | [NeRF repository README](https://github.com/bmild/nerf) (MIT) | Pose-format section: NeRF code uses OpenGL camera axes, LLFF tooling makes the poses. It does not give the column order of `poses_bounds.npy`, so D-28 relies on general knowledge plus test C-1 |
| 13 | PyTorch 2.13 documentation (via Context7): [Reproducibility notes](https://docs.pytorch.org/docs/2.13/notes/randomness.html), `torch.get_rng_state`, `torch.cuda.get_rng_state_all` / `set_rng_state_all` | RNG-state APIs for exact resume; deterministic-algorithm caveats |
| 14 | Cited works known through the paper's reference list only: STGS \[21\], 4DGS \[49\], SSIM \[43\], LPIPS \[52\], ENeRF \[23\], BGMv2 \[24\] | Identity of the metrics and baselines; their papers were not otherwise needed |
| 15 | [RoMa README](https://github.com/Parskatt/RoMa) (documentation; read after the owner allowed it) | API entry points (roma\_outdoor, match, sample, to\_pixel\_coordinates); default resolution 560×560 upsampled to 864×864; sample\_thresh; Tiny RoMa (tiny\_roma\_v1\_outdoor); MIT license, with DINOv2 under Apache-2.0 |
| 16 | [opencv-python README](https://github.com/opencv/opencv-python) (documentation) | Licensing section: packaging scripts MIT, OpenCV Apache-2.0, every wheel ships FFmpeg under LGPL-2.1, non-headless Linux wheels ship Qt 5 under LGPL-3. Installation section: install only one of the four OpenCV packages per environment, since all provide the cv2 module; headless packages avoid the X11/GUI dependency chain |
| 17 | Python package index | Not reachable from the workspace, so candidate package names were not checked; a recheck on 2026-10-07 got "no matching distribution" even for safetensors, so openftgs could not be checked from here; the owner's own check that day found no distribution named openftgs. PyPI pages for FreeTimeGS-like names were deliberately not opened, since they could show a third-party implementation |
| 18 | [RoMa paper, Edstedt et al., CVPR 2024 (arXiv 2305.15404)](https://arxiv.org/abs/2305.15404), cited as \[9\], via the Hugging Face mirror | §4.2: one model trained on MegaDepth is used for every evaluation except ScanNet-1500 (including indoor InLoc), and the final model was trained at 560×560; §4.3: balanced sampling of 10,000 matches for two-view geometry; §4.5: about 199 ms per pair at 560×560, batch 8, on an RTX 6000 |
| 19 | [decord repository](https://github.com/dmlc/decord), examined at the owner's request: README, LICENSE, tags, commit history, wheel workflow and build scripts (packaging files, allowed for license checks by the owner on 2026-10-07); no source code read | Apache-2.0; PyPI wheels are CPU-only and NVDEC decoding needs a source build; random-access VideoReader and VideoLoader; last tag v0.6.0, last commit 2022-07-19. The wheel workflow (.github/workflows/pypi.yml) and build scripts (tools/build\_manylinux2010.sh, tools/build\_macos\_10\_9.sh) configure FFmpeg 4.1.6 with --enable-gpl --enable-nonfree --enable-libx264 --enable-libvpx and bundle it into the wheels; the Linux wheel is built for CPython 3.6 and retagged py3; Windows wheels copy prebuilt FFmpeg 4.2.1 DLLs |
| 20 | [torchcodec README and LICENSE](https://github.com/meta-pytorch/torchcodec) and [documentation](https://meta-pytorch.org/torchcodec) (via Context7), read as PyTorch-ecosystem documentation while comparing decoders; the documentation search also returned one CMakeLists.txt comment, a build file of the kind the owner allowed for license checks on 2026-10-07 | BSD-3-Clause; uses the system's FFmpeg shared libraries, major versions 4 to 9 (built against non-GPL FFmpeg only at build time); CUDA wheels are the default on Linux; NVDEC needs an FFmpeg with NVDEC support and falls back to the CPU otherwise; VideoDecoder options num\_ffmpeg\_threads (default 1), seek\_mode, device; version 0.17 needs torch ≥ 2.11 and Python 3.10–3.14 |
| 21 | PyTorch 2.13 documentation (via Context7): Adam implementation notes, torch.compile and dynamic shapes, CUDA semantics (pinned memory, streams, CUDA graphs), TF32 notes, torch.cuda.set\_sync\_debug\_mode, threading environment variables, torch.cpu.get\_capabilities | Implementation speed fused > foreach > for-loop, with foreach the CUDA default; dynamic=True and mark\_dynamic; pinned memory enables asynchronous copies but pinning is expensive; CUDA graphs forbid host synchronization and dynamic shapes; TF32 is on by default for cuDNN convolutions and off for matmul; the sync debug mode is experimental and not exhaustive; MKL\_NUM\_THREADS takes precedence over OMP\_NUM\_THREADS |
| 22 | safetensors documentation (via Context7) | safe\_open loads lazily; get\_slice reads part of a tensor; files are memory-mapped by default |
| 23 | gsplat v1.5.3, performance material: rasterization() docstring notes, docs/source/tests/profile.rst, docs/batch.md, gsplat/optimizers/selective\_adam.py, the projection kernels, Projection.cpp, AdamCUDA.cu and Intersect.cpp | packed and sparse\_grad change memory and speed but not results; radius\_clip skips small primitives; batched rendering of many cameras or scenes; distributed=True follows Grendel; SelectiveAdam is a fused, visibility-masked Adam from Taming 3DGS; projection culls primitives with opacity below 1/255; the intersection count is read back to the host in every render, packed projection also reads back its visible count (Projection.cpp), and the SH path inverts the view matrices with torch.inverse (rendering.py), which synchronizes on CUDA (#40); SelectiveAdam's kernel applies no bias correction and never advances the step counter (AdamCUDA.cu, selective\_adam.py); profiling tables comparing packed and unpacked memory and speed |
| 24 | [Grendel: On Scaling Up 3D Gaussian Splatting Training (arXiv 2406.18533)](https://arxiv.org/abs/2406.18533), via the Hugging Face mirror (general Gaussian-splatting literature) | §4, Eqs. 1–2: learning rate × √(batch size), Adam β₁ and β₂ raised to the batch size; §5.2: 3–4× speed-up on 4 GPUs at batch 4 without PSNR loss on 13 static scenes, trained on the same number of images |
| 25 | [SelfCap dataset card and LICENSE](https://huggingface.co/datasets/zju3dv/SelfCap-Dataset) on Hugging Face (owner-provided link), read through the Hugging Face connector; file list only for the archives | FreeTimeGS protocol table (scene-to-archive mapping, test cameras, 60-frame windows, ratios; dance3 and dance4 unreleased); COLMAP undistortion with blank\_pixels=0, then INTER\_AREA at 0.5; camera conventions "follow EasyVolcap"; sync.json defines actual time as frame-index time minus the correction; cameras, frame counts and resolutions per sequence (bike 1024×1024); bundled point clouds; non-commercial, research-only license requiring open-source, non-commercial modifications. The EasyVolcap repository and the extraction script the card links to were not opened. The archives (binary, up to 23 GB) could not be read here, because the workspace shell cannot reach Hugging Face and the connector reads only text files |
| 26 | RoMa source code (romatch, commit 77f8d68): romatch/\_\_init\_\_.py, models/model\_zoo/\_\_init\_\_.py, models/matcher.py; allowed by the 2026-10-07 rule as a dependency the package imports | roma\_outdoor(device, coarse\_res=560, upsample\_res=864, amp\_dtype=float16, symmetric=True, upsample\_preds=True, ...); it refuses to build unless float32 matmul precision is "highest"; weights downloaded from GitHub releases and DINOv2 from Meta's server; tiny\_roma\_v1\_outdoor loads XFeat through torch.hub from verlab/accelerated\_features; match() returns a two-way warp and sigmoid certainty, zero outside the image; sample() thresholds at 0.05 (setting higher certainties to 1), draws with torch.multinomial on the global generator and balances by kernel density; to\_pixel\_coordinates maps \[−1, 1\] onto \[0, W\] |
| 27 | OpenCV 4.13 documentation (via Context7; documentation only) | undistort combines initUndistortRectifyMap and remap, fills pixels without a source with zeros, and defaults newCameraMatrix to cameraMatrix; distortion coefficients are ordered (k1, k2, p1, p2\[, k3\[, k4, k5, k6\[, s1, s2, s3, s4\[, τx, τy\]\]\]\]) with 4, 5, 8, 12 or 14 elements. The model's equations were not retrieved here (see #29); implementers take them from the calib3d documentation page |
| 28 | COLMAP documentation (via Context7; documentation only) | image\_undistorter command-line usage; PINHOLE and SIMPLE\_PINHOLE for undistorted images, OPENCV and FULL\_OPENCV for known intrinsics. The tutorial states COLMAP's image convention: the upper-left corner is at (0, 0) and the center of the first pixel at (0.5, 0.5), which D-60's +0.5 follows. The meaning of blank\_pixels was not found there and is general knowledge |
| 29 | OpenCV calib3d page and COLMAP command-line page, by direct fetch | Not read: the fetch permission prompts timed out |
| 30 | SelfCap calibration archives hair-calib.tar.gz and yoga-calib.tar.gz (owner-provided copies of the Hugging Face files; data only) | Each holds optimized/intri.yml, extri.yml and sync.json for 24 cameras named 0000–0023. Keys: names, K\_, D\_ (5×1, k3 = 0), H\_ and W\_ (2160, 3840), ccm\_ (identity; hair only); R\_ (Rodrigues), Rot\_, T\_, and t\_, n\_, f\_ (0, 1, 20; hair only). Rot equals Rodrigues(R) to 3·10⁻⁸. Under the world-to-camera reading every camera faces the rig's middle (cosine ≥ 0.64); under the camera-to-world reading none does (cosine ≤ −0.55). sync.json offsets span −0.013 to +0.012 (hair) and −0.030 to +0.026 (yoga). Both files parse with PyYAML after dropping the %YAML:1.0 line and registering opencv-matrix |
| 31 | Owner-provided listing of corgi-release.tar.gz and its optimized/sync.json | videos/0000–0023.mp4; dense\_pcds (every 1,000 frames), dense\_pcds\_rc\_bbox\_180k, pcds (every frame), pcds\_j10 (every 10 frames); optimized/intri.yml, extri.yml and sync.json; no background images. sync.json offsets span −0.046 to +0.026. The listing ended with a gzip CRC error; the owner then confirmed that the download's SHA-256 matches Hugging Face's and that gzip -t fails, so the hosted archive itself is damaged |
| 32 | RoMa source code, imports and OpenCV use (romatch at commit 77f8d68): import lines of models/\_\_init\_\_.py, model\_zoo/roma\_models.py, encoders.py, tiny.py, transformer/\_\_init\_\_.py and dinov2.py; utils/\_\_init\_\_.py, the head of utils/utils.py, utils/local\_correlation.py, part of roma\_models.py; a search of the package, pyproject.toml and requirements.txt for cv2 | cv2 appears only in utils/utils.py (recover\_pose, estimate\_pose, estimate\_pose\_uncalibrated: findEssentialMat, findFundamentalMat with USAC\_ACCURATE, recoverPose), benchmarks/hpatches\_sequences\_homog\_benchmark.py (findHomography) and datasets/scannet.py (imread). utils/utils.py imports cv2 at module level and is imported by encoders.py, matcher.py and transformer/\_\_init\_\_.py, so importing romatch requires cv2. The matching path imports torch, torchvision, numpy, PIL, einops and loguru; kornia is imported only inside a helper; local\_corr is imported inside local\_corr\_wrapper, used when use\_custom\_corr is true, which roma\_outdoor sets by default on Linux |
| 33 | RoMa source, files read for vendoring (commit 77f8d68): builder settings in model\_zoo/roma\_models.py, file headers of the DINOv2 layers, LICENSE, internal imports, line counts of the inference modules | The builders create the VGG encoder with pretrained=False, so no ImageNet weights are downloaded; run-time downloads are RoMa's weights from GitHub releases, DINOv2's from Meta's server and, for Tiny RoMa only, XFeat through torch.hub. The DINOv2 files carry Meta's copyright header and point to DINOv2's root LICENSE (Apache-2.0); RoMa's LICENSE is MIT, Copyright (c) 2023 Johan Edstedt. 14 absolute romatch imports, 12 at module level and 2 inside functions (matcher.py and tiny.py import tensor\_to\_pil when saving a visualization); about 3,800 lines in the inference modules. The training-only globals in romatch/\_\_init\_\_.py are used only by benchmarks, datasets, losses and checkpointing |
| 34 | lpips README in the PerceptualSimilarity repository (documentation; the package imports lpips) | Inputs are RGB images scaled to \[−1, +1\] (E7's L\_perc mapping) |
| 35 | safetensors README, via Context7 (the package imports safetensors) | The \_\_metadata\_\_ entry is a free-form string-to-string map; arbitrary JSON is not allowed and all values must be strings (§2.2) |
| 36 | Context7 query for scikit-image's structural\_similarity documentation, 2026-10-07 (incident) | The tool returned snippets of scikit-image's implementation file (src/\_skimage2/metrics/\_structural\_similarity.py): the C₁/C₂ formula from data\_range and the loop that averages SSIM over channels. scikit-image is in the docs-only class, so reading stopped and nothing was taken. The spec's SSIM (E7, D-8) predates the query and follows the SSIM paper's constants and gsplat's trainer; U-9's parameter names are general knowledge of scikit-image's documented API. Rule added in §0: docs-only software is read through its own documentation pages |
| 37 | Independent review of the spec at revision 320 by a fresh agent, 2026-10-07 | 21 findings (0 critical, 4 major, 14 minor, 3 nit) and no clean-room incident. The reviewer read the paper; gsplat at tag v1.5.3 only; RoMa at 77f8d68; the SelfCap card, files and listing; the 3DGS, 3DGS-MCMC, RoMa and Grendel papers; and COLMAP, safetensors and OpenCV documentation. It opened no FreeTimeGS implementation, EasyVolcap code, gsplat main branch, dynamic module or GPL code. Each finding was checked against its source (#38) before the spec changed |
| 38 | Sources re-read to check the review, 2026-10-07: gsplat v1.5.3 Projection.cpp, AdamCUDA.cu, optimizers/selective\_adam.py, RelocationCUDA.cu, relocation.py, strategy/ops.py, strategy/default.py, rendering.py and examples/simple\_trainer.py; RoMa 77f8d68 romatch imports, models/tiny.py, models/matcher.py and model\_zoo/\_\_init\_\_.py; the paper's Eq. 6, Tables 2, 6–8 and Appendix B.1; the SelfCap card and repository file list; the corgi listing; COLMAP's documentation website via Context7 | Packed projection reads its visible count back; SelectiveAdam has no bias correction; relocation computes the scale factor from the unclamped opacity, clamps afterwards and zeroes only the targets' moments; packed mode returns gaussian\_ids and \[nnz, 2\] means2d, and DefaultStrategy scatter-adds with index\_add\_; the trainer's loss uses unclamped renders; 14 romatch imports; Tiny RoMa's warp is one-way with no bounds handling; roma\_outdoor defaults to float16 autocast; Eq. 6 states 1/N over all N; Table 8 gives SelfCap 96 MB at 467 FPS and 53 MB at 664 FPS; B.1 gives the crop-and-fill rule; the card's point clouds come from multiview images for a model trained without held-out views, and the card mentions neither background images nor hair-calib; corgi's per-frame clouds are 005200–008699; COLMAP puts the first pixel's center at (0.5, 0.5) |
| 39 | Second independent review, of revision 455, by a fresh agent, 2026-10-07 | 27 findings (0 critical, 2 major, 18 minor, 7 nit) and no clean-room incident. The reviewer used the sources of #37 plus PyTorch's documentation website via Context7; for docs-only software it used documentation-website libraries only, and none for scikit-image. It opened no FreeTimeGS implementation, EasyVolcap code, gsplat main branch, dynamic module or GPL code, and read no source code of docs-only software. Each finding was checked against its source (#40) before the spec changed |
| 40 | Sources re-read to check the second review, 2026-10-07: gsplat v1.5.3 examples/simple\_trainer.py, rendering.py, cuda/csrc/ProjectionEWA3DGSFused.cu and strategy/default.py; RoMa 77f8d68 models/transformer/layers/attention.py and utils/utils.py; 3DGS-MCMC §3.4 via the Hugging Face mirror; PyTorch 2.13 documentation for torch.linalg.inv via Context7; the paper's §4.2; corgi's sync.json | The trainer feeds LPIPS-AlexNet inputs in \[−1, 1\] and LPIPS-VGG inputs in \[0, 1\], and takes 3-NN scales as a root mean square; the SH path calls torch.inverse on the view matrices, and torch.linalg.inv synchronizes on CUDA; projection zeroes both radii when det Σ′ ≤ 0 or the box is off-screen; DefaultStrategy prunes σ < 0.005 and, after 3,000 iterations, scales above 0.1·S, and its opacity-reset condition never fires; RoMa's DINOv2 attention uses xformers when installed, and a utils helper imports kornia; 3DGS-MCMC keeps the moved Gaussians' moments because its noise term dominates them; the ablations use dance1 and its 10 fastest frames; corgi's training cameras span 0.0725 s (4.35 frames) |

## Appendix B — Ambiguity log

Sixty-nine rows record decisions, sixty-eight of them on gaps in the paper; D-11 only confirms what Eq. 6 states, because implementations often vary it. D-3, D-10, D-18, D-24, D-48, D-58, D-61, D-62 and D-65 were set by the project owner, and D-59, D-60, D-63 and D-64 follow the owner's choice to support SelfCap now. None copies another implementation's behavior; each states its rationale, the alternatives, and the test that would expose a wrong choice. D-1 to D-29 and D-42 to D-47 concern the method itself; D-30 to D-41 and D-48 cover engineering, protocol and wording, and some of those (D-31, D-34, D-35, D-36) still change renders or training. D-58 to D-65 cover initialization, datasets, preprocessing, naming and vendoring, and D-66 to D-69 cover test-view isolation and evaluation details. D-49 to D-57 cover performance; they change results only through TF32 in training-time convolutions (D-52) and in opt-in modes.

| ID | Gap in the paper | Decision | Rationale | Alternatives | Detecting test |
| --- | --- | --- | --- | --- | --- |
| D-1 | Time unit and range of t, μt, s, v never stated | t = (τ − τ₀)/(τ₁ − τ₀) over training images; raw units at every file boundary | Ranges independent of sequence length; files stay comparable across implementations | seconds; frame indices; \[−1, 1\] | U-4; L2 `motion`, `temporal_opacity` |
| D-2 | Parameterization of μt, s, v | μt and v raw; s stored as log s | s must stay positive; log space matches how 3DGS stores scales | softplus(s); \|s\|; storing 1/s² | U-2; S-3 |
| D-3 | SH degree L (Eq. 2) | user parameter sh\_degree ∈ {0, 1, 2, 3}, default 3; one band enabled per 1,000 iterations up to L (owner decision, 2026-10-06) | Users trade model size for view-dependent detail; 3 is the 3DGS default | L = 1 (Table 7's \~124 bytes per primitive fit 31 float32 values, but SH degree 3 in float16 is within 4%); L = 0 | U-6; U-26; Neural3DV PSNR and size vs Table 6 for L ∈ {1, 3} |
| D-4 | Color offset, clamp and view direction | SH sum + 0.5, clamped at 0; direction from camera center to the moved mean | Matches the rasterizer used; direction as Eq. 2 states | sigmoid on the SH output; no clamp | U-6; L1 probe P5; "color shift ≈ 0.5" symptom |
| D-5 | Activations and initial opacity, scale, rotation | sigmoid opacity from 0.1; exp scales from 3-NN; identity rotation | 3DGS §5.1 (activations, 3-NN); gsplat trainer (opacity 0.1); identity rotation is our choice: deterministic, and irrelevant while scales are isotropic | initial opacity 0.5 (3DGS-MCMC); random rotations (gsplat trainer) | U-3; U-19; L3 opacity distribution |
| D-6 | Rasterizer and settings (the paper names none; the project page links a separate renderer, not consulted) | gsplat 1.5.3 classic mode, ε = 0.3, near 0.01, far 1e10 | Apache-2.0; 3DGS-equivalent math | antialiased mode; other rasterizers | U-7; L1 tiers |
| D-7 | "Image loss" undefined | L1 | 0.8/0.2 equals 3DGS's (1 − λ)L1 + λ·D-SSIM with λ = 0.2 | L2 | L2 `losses`; L3 PSNR |
| D-8 | Form of the SSIM loss | 1 − SSIM; 11×11 Gaussian, σ = 1.5; C₁ = 0.01², C₂ = 0.03²; valid padding | SSIM defaults; gsplat trainer padding | (1 − SSIM)/2; zero ("same") padding | U-9; L2 `losses` |
| D-9 | Perceptual-loss backbone | LPIPS-VGG, inputs in \[−1, 1\] | VGG features are the classic perceptual-loss choice | AlexNet; SqueezeNet | L2 `losses`; L3 LPIPS-VGG |
| D-10 | The intro applies the regularizer in an undefined "early stage" (§1), while §3.3 gives one constant weight and no schedule; whether that phrase means a cutoff is ambiguous | on for the whole run at constant λ\_reg = 10⁻² (owner decision, 2026-10-06, kept on 2026-10-07 after the review); LossConfig.reg\_until ends it earlier | §1's "early stage" gives no cutoff or decay, so reading it as one would mean inventing the value, and §3.3 gives one constant weight. The per-primitive strength already varies with temporal relevance through sg\[σt(t)\] | stop at 15,000; linear decay to 0; stop at 7,000 | S-5; L3 late-training opacity histogram and loss trace (a cutoff shows as a rise in mean opacity after it); whole run vs reg\_until = 15,000 on a Neural3DV scene and SelfCap dance1, compared with Tables 1, 3 and 5 |
| D-11 | None: Eq. 6 sums over all N primitives at the time of the current image. Recorded because restricting the sum to visible primitives is a common variant | all N primitives, mean over N, t of the current image | Eq. 6 sums i = 1…N | visible only; sum | U-10; L2 `losses` |
| D-12 | "Spatial gradient" in Eq. 7 | mean NDC-scaled norm of the 2D-mean gradient over visible iterations since the last relocation | The 3DGS densification signal; exposed by gsplat | 3D position gradient; absolute gradient; max | U-16; L2 `grad_stats` |
| D-13 | Eq. 7 adds a tiny gradient to an opacity in (0, 1) with equal weights | ĝ = min(1, g/Q₀.₉₉); σ = base opacity | Both terms on \[0, 1\]; quantile resists outliers | no normalization; divide by max; ranks; time-modulated opacity | U-15; L2 `relocation_score` |
| D-14 | Relocation threshold value | base opacity < 0.005, tested as ô < ô\_dead = float32(logit 0.005), with relocated logits clamped to ô\_dead | 3DGS prune threshold; 3DGS-MCMC dead threshold; comparing logits keeps the boundary exact, since sigmoid(logit 0.005) rounds to 0.0049999994 in float32 | 1/255; peak opacity over \[0, 1\] | U-14; S-6 |
| D-15 | How primitives are moved | multinomial ∝ ρ with replacement; copy target; 3DGS-MCMC Eq. 9 on opacity and scale; temporal tensors copied; each moved copy's position drawn from the target's Gaussian, as E17's split does | Matches "move … to the region with high sampling score"; Eq. 9 keeps the image nearly unchanged while copies are stacked. Stacked copies with equal α get identical position, scale and opacity gradients and, with D-16, identical moments, so they would never separate; 3DGS-MCMC separates them with a noise term that FreeTimeGS does not have | top-k targets; 3DGS clone; copies left stacked on the target (3DGS-MCMC without its noise); 3DGS-MCMC's position noise | U-13; U-14 (position draws); S-6; L2 `relocation_math` |
| D-16 | Optimizer state after a move | zero moments of targets and moved rows | A moved row's old moments belong to a dead primitive elsewhere and would push opacity down again. 3DGS-MCMC keeps them because its noise term dominates a dead Gaussian's moments, and FreeTimeGS has no such term. Reset rows take steps of up to about 6.6·lr in their first iterations, while the second moment rebuilds | targets only (3DGS-MCMC, gsplat) | U-14; L3 share of moved primitives that die again within 100 iterations |
| D-17 | Relocation start and end | 500 ≤ it ≤ 25,000 | 500-iteration warm-up (3DGS-MCMC §3.6); stop at 25,000 as gsplat MCMCStrategy's default, whose exclusive bounds give 244 events instead of 246 | whole run; stop at 15,000 | U-17; L3 N and relocation traces |
| D-18 | Growth and pruning besides relocation | The count grows (owner decision, 2026-10-06): 3DGS clone/split at iterations 500–15,000 with τ\_pos = 0.0002, clone/split boundary 0.01·S and split by E17; relocation instead of deletion; cap N\_max | §3.2: the regularizer "causes a dramatic increase in the number of Gaussian primitives needed". §4.2: without relocation "the model tend to use more gaussians with lower opacity", so the ablation had to "control the number of gaussians". Table 7's counts are irregular and the † model (≈ 347k, inferred) ends below its 500k cap, as a threshold-driven count would. Avoiding holes is not the reason: relocation also moves capacity under a fixed count (S-11 measures the difference) | fixed N with relocation only (the §4.2 wording); MCMC growth to a cap (ends exactly at the cap) | PSNR within ±0.3 dB of Table 7's curve, interpolated in log N at each run's final N, for runs spanning 70k to 2,042k (τ\_pos sweeps at B = 1,000, and at B = 200 below the default start); S-11; U-25; L3 N trace; ablation direction of Table 4 |
| D-19 | Velocity schedule printed as a sum; λ₀, λ₁ missing | product form; ηv = ηx/Δt\_f at both ends | The literal sum is ≈ 1 at both ends and dips between, so it never anneals; dividing by Δt\_f gives equal displacement per step one frame from μt | literal sum; ηv = ηx; constant | U-17; L2 `lr_schedule`; S-2 |
| D-20 | Rates for μt and log s | μt: 1.6e-4 → 1.6e-6; log s: 5e-3 constant | Temporal analogs of the position and log-scale rates | constant μt rate; frame-relative rates | S-3; L2 `lr_schedule` |
| D-21 | Initial duration | stride 1; s₀ = stride·Δt\_f | Summed temporal opacity of static content is flat (ripple ≈ 5e-9) away from the ends; ≈ 70% at τ₀ and τ₁ | whole sequence; half a frame | U-19; S-3; L3 duration distribution |
| D-22 | Camera pairs and filters for RoMa triangulation | nearest-axis pair per training camera, each unordered pair matched once; 5,000 matches; certainty ≥ 0.5; DLT; cheirality; ≤ 2 px; ≥ 1° | Standard two-view hygiene at bounded cost | all pairs; multi-view tracks with bundle adjustment | U-19; C-3 (L3 shares one init file, so A and B are unaffected) |
| D-23 | k and frame pairing for velocity k-NN | 1-NN to the next init frame; backward difference on the last; no outlier filter | "Point pairs" implies one partner per point. The paper takes the translation itself as the velocity; in normalized time that becomes translation/Δt (the literal reading would understate v by 1/Δt\_f, 299× for 300 frames) | k > 1 averaging; mutual NN; speed cap | U-19; S-4; L2 `velocity_init` |
| D-24 | Initial point budget (the paper gives none, and its matches run to millions of points) | at most B = 1,000 points per frame of the sequence (an init frame keeps k\_stride·B), as seeded uniform subsets: about 300k for 300 frames and 60k for 60 frames; initialization stops with an error if F·B exceeds N\_max (owner decision, 2026-10-07) | The count only grows (D-18), so the start is a floor and must sit below the smallest model to reproduce: Neural3DV † ≈ 347k, SelfCap † ≈ 450k and main ≈ 810k (sizes matched to Table 7). The earlier 1M start made all of them unreachable (found by the independent review, Appendix A #37). A per-frame budget keeps the initial density in time the same for every sequence length | 1M or 250k global budgets; every filtered match, with fewer matches per pair; voxel subsampling | U-19; L3 count; the default run's final N against about 1,060k (Neural3DV) and 810k (SelfCap) |
| D-25 | Neighbour set for the 3-NN scale init in 4D | same init frame only, among the points kept after the per-frame subsample; mean floored at 1e-7 | Cross-frame copies of static points sit at near-zero distance and would collapse scales; scales computed before the subsample would fit a spacing several times smaller than the kept points have | pooled frames; space-time distance; scales from the full set before subsampling | U-19 |
| D-26 | Image sampling | batch 1, seeded per-epoch permutation | 3DGS practice; reproducible | sample a time, then a camera; all cameras at one time | U-21; U-24 |
| D-27 | Spatial scale S for the position rate | 1.1 × max distance of training camera centers from their centroid | gsplat trainer definition, restricted to training cameras so test views never influence training (gsplat uses all cameras) | point-cloud extent | L2 `spatial_scale` |
| D-28 | Camera conventions; Neural3DV pose layout | OpenCV axes, world-to-camera, pixel centers at +0.5, undistorted pinhole; LLFF columns down, right, back | gsplat conventions; LLFF layout from general knowledge | OpenGL axes; integer pixel centers | U-20; C-1; C-2; L1 probe P1 (half-pixel offset) |
| D-29 | Test views (ENeRF-Outdoor, SelfCap), LPIPS backbones, DSSIM | Neural3DV cam00; SelfCap from the dataset card (0015, 0007, 0009); ENeRF-Outdoor configurable; LPIPS-Alex for Neural3DV, VGG otherwise; DSSIM = (1 − SSIM)/2 | Dataset README; table captions; (1 − SSIM)/2 matches every SelfCap row and the ENeRF, 4K4D and Ours rows on ENeRF-Outdoor (not 4DGS or STGS in Table 2) | 1 − SSIM | U-23 |
| D-30 | File formats and precision | float32 tensors, float64 timestamps; safetensors + JSON, no pickle | Safe loading; timestamp precision | torch.save; PLY | U-21; U-22 |
| D-31 | Alpha clamp differs across gsplat versions | pin 1.5.3 (α\_max 0.999); reference backend mirrors it | Latest release; 0.999 departs from the 3DGS paper's 0.99 (Appendix C), which gsplat's main branch also uses | α\_max = 0.99 once a gsplat release adopts it | U-7; L1 probe P4 |
| D-32 | Primitives below the alpha threshold | cull σ\_eff < 1/255, inside gsplat's projection by default (D-57) | Same output, less work | no pre-cull | U-8 |
| D-33 | Resizing method | area averaging: each output pixel is the mean of the input over its footprint, with input pixels as constant squares; separable; 8-bit output rounded half to even; K scaled by the realized ratios W′/W and H′/H (r when W·r and H·r are integers); implemented in the package (D-61) | Alias-free downsampling; SelfCap's protocol uses OpenCV's INTER\_AREA, which this should match on integer factors up to rounding (U-34) | bilinear; Lanczos | U-34; U-20 |
| D-34 | Background color | black, never random | Full-frame captures without masks | white; random | U-7; L1 border-band symptom |
| D-35 | Iterations for sequences other than 300 frames | 30,000 for any length | Only stated value | scale with frame count | L3 per-time PSNR; SelfCap scenes vs Table 10 |
| D-36 | 3DGS §7.1 resolution warm-up | not used | "Same settings" refers to the optimizer; gsplat's 3DGS replica does not warm up | ¼ resolution, doubled at 250 and 500 | L3 early loss traces |
| D-37 | Total loss never written | L\_render + λ\_reg·L\_reg (E9) | Only consistent reading of §3.3 | — | L2 `losses` |
| D-38 | "λ\_reg = 1e^{−2}" notation | 1 × 10⁻² | Table 5 sweeps decades 1e−3 to 1e−1 | — | — |
| D-39 | Mechanism behind "≤ 500k primitives" († rows) | † reproduction = the default configuration with N\_max = 500,000; the start (about 300k on Neural3DV, 60k on SelfCap) leaves room to grow | Compares quality at equal N; bytes per primitive depend on storage, so size is not compared | raise τ\_pos until N ≤ 500k | PSNR within 0.3 dB of Table 7's curve at the final N (32.97 dB at 347k, 32.94 dB at 618k); SelfCap † against Table 8 (27.27 dB) |
| D-40 | §4 says higher DSSIM means more similar; tables mark it lower-is-better | DSSIM is a dissimilarity, (1 − SSIM)/2 | Agrees with the rows listed under D-29 | — | U-23 |
| D-41 | Schedules use training progress t ∈ \[0, 1\] without indexing | it counts completed steps from 1; u = (it − 1)/(T\_it − 1) | First step uses η₀, last uses η₁ exactly | u = it/T\_it | U-17; L2 `lr_schedule` |
| D-42 | Opacity reset and deletion during growth | neither: no opacity reset and nothing deleted; low-opacity primitives are relocated | The 4D regularization already counters saturation; a reset would push many primitives toward the dead threshold just before relocation events; relocation is the paper's replacement for removing low-opacity primitives | 3DGS reset every 3,000 iterations; pruning of large primitives | U-25; L3 opacity histogram and N trace |
| D-43 | Temporal parameters of split or cloned children | copied unchanged; the split is spatial only (E17) | The paper defines no split; μt and s keep receiving their own gradients | also sample μt \~ N(μt, s²) and divide s by 1.6 (a 4D split) | U-25; S-3; L3 duration distribution |
| D-44 | Growth cap and candidate selection | N\_max = 3,000,000; when candidates exceed the room left, the largest-g ones grow, ties to the lower row index | Memory guard above Table 7's largest row (2,042k) | no cap; a random subset of candidates | U-25; L2 growth\_decisions |
| D-45 | Order of growth and relocation within one event | growth, then relocation, then statistics reset; new rows inherit their parent's statistics, so this event's relocation scores the grown region | Mirrors 3DGS's densify-then-prune, with relocation in pruning's place | relocation first | U-25 |
| D-46 | Frame interval Δt\_f with unsynchronized cameras (the paper's rigs are synchronized) | median over cameras of each camera's median gap between its own timestamps; pooled median as fallback; 1 for a single timestamp | Pooled timestamps of unsynchronized cameras interleave, which would shrink Δt\_f by up to the camera count, inflate the velocity rate (D-19) and shrink initial durations (D-21) | pooled median; nominal frame rate from metadata | U-18; S-8 |
| D-47 | Initialization with unsynchronized cameras | frame slots t\_k = k·Δt\_f holding each camera's image within Δt\_f/2 of t\_k, if it has one; a point's time is the mean of its two images' times | Keeps the per-frame pipeline and bounds the timing error to half a frame; cameras without such an image sit the slot out, as at the ends of corgi, whose training cameras span 4.35 frames | require synchronized rigs; interpolate matches in time | U-19; S-8 |
| D-48 | How Neural3DV videos are decoded | user-installed ffmpeg ≥ 5.1 called as a subprocess, raw RGB24 over a pipe, first 300 frames in order (owner decision, 2026-10-06) | Keeps FFmpeg out of the package; one resampling step (D-33) | OpenCV VideoCapture; PyAV | U-27; C-1 |
| D-49 | Performance policy; the paper reports training time and FPS but no implementation details | compiled code in every hot path; exact optimizations on by default after a benchmark; approximate ones opt-in and logged here, TF32 in training-time cuDNN convolutions being the one approximate default; no host synchronization in an iteration except the three inside gsplat's render (§2.4 rule 3) | Speed without silent changes to the method | optimize case by case; keep everything eager | U-29; B-1 |
| D-50 | Implementation of the per-primitive temporal work | one fused temporal\_slice through torch.compile with dynamic shapes, eager fallback; a custom CUDA operator only above 10% of an iteration; no CUDA graphs; no gsplat fork | Elementwise work fuses well without a build toolchain; gsplat synchronizes in every render | handwritten CUDA now; motion folded into gsplat's projection kernel | U-28; B-3 |
| D-51 | How images reach the GPU | uint8 frame cache, one safetensors file per camera, on the GPU when it fits and memory-mapped otherwise; prefetch in the seeded order through pinned buffers on a side stream; PNG stays the interface | Removes PNG decoding from the loop without changing the file formats | decode PNG every iteration; lossy JPEG with GPU decoding | U-30; B-6 |
| D-52 | Numeric precision | float32 throughout; TF32 for cuDNN convolutions in training (PyTorch's default), which reaches LPIPS and, without fused-ssim, the SSIM fallback; IEEE float32 for reported metrics; bfloat16 LPIPS opt-in; float64 triangulation | The perceptual term has weight 0.01, and reported numbers stay plain float32. cuDNN's TF32 flag is global, so it also reaches every other convolution in training | IEEE float32 everywhere; mixed precision everywhere | U-9; B-4; L2 losses |
| D-53 | Adam implementation | PyTorch's fused Adam, foreach as fallback; gsplat's SelectiveAdam opt-in | Same algorithm with fewer kernel launches. SelectiveAdam skips the moment decay and momentum steps of invisible rows, which Adam with "the same settings as 3DGS" performs. On visible rows gsplat 1.5.3's kernel also omits Adam's bias correction and never advances the step counter, so for a steady gradient its early steps are up to about 6.6 times larger | SelectiveAdam by default | U-31; B-5; L3 |
| D-54 | Several GPUs | exact sharding of initialization, evaluation, rendering and seeds; multi-GPU training opt-in, with Grendel's √b and β^b scaling, a calendar counted in images and global quantities computed over all ranks | The paper's steps use one image (D-26) | data-parallel replicas; no multi-GPU support | L3 against single-GPU runs |
| D-55 | Rendering at one time | buckets one frame interval wide listing the primitives that can reach σ\_eff ≥ 1/255, widened by a margin; gsplat's own test decides; rendering only | Exact, and the cost follows the active set | no index; sorting by μt | U-32; B-7 |
| D-56 | Parallel and numeric details of initialization | per-pair seeds, each unordered pair matched once in its own forward pass with deterministic cuDNN; float64 triangulation; exact k-NN by explicit differences or a KD-tree; ties to the lower index, with KD-tree candidates re-sorted by (distance, index) | Results independent of sharding and order; in float32, ‖a‖² + ‖b‖² − 2a·b can swap near neighbors | one global random stream; GEMM-based distances | U-33; B-8 |
| D-57 | gsplat flags and where D-32's cull happens | packed = True (gsplat's default) until B-2 calibrates; sparse\_grad False; radius\_clip 0; tile size 16; unsegmented sort; D-32's cull left to gsplat's projection unless B-2 shows compaction is faster | These flags leave results unchanged, except radius\_clip, which would drop small primitives; compaction adds a host synchronization in every iteration (indexing by a mask reads the kept count back), on top of gsplat's own, so it is used only if it pays for itself | gsplat's defaults unexamined; radius\_clip > 0 | U-8; B-2 |
| D-58 | RoMa model, resolution and match sampling (owner choice, 2026-10-07) | roma\_outdoor at RoMa's defaults (coarse 560, upsampled to 864); matching without upsampling and Tiny RoMa selectable; matches drawn by the package from match()'s raw certainty with the pair's generator, not by RoMa's sample(); use\_custom\_corr set to whether fused-local-corr can be imported, because RoMa's Linux default otherwise fails at match time, and recorded in the init file; with Tiny RoMa, whose warp is one-way and unclamped, matches whose target falls outside image B are dropped, and the 0.5 certainty threshold is not validated | Closest to "RoMa" as cited, and the RoMa paper uses its MegaDepth model for every benchmark except ScanNet. RoMa's sample() uses the global random generator and sets certainties above 0.05 to 1, which would void D-22's threshold | 560 px without upsampling; Tiny RoMa; roma\_indoor; RoMa's own sample() | U-19; U-33; B-8 |
| D-59 | SelfCap frame selection and timestamps (the card gives windows but no numbering base) | 0-based decode index n; τ = n/60 s minus the camera's sync.json offset when present, the offset read as seconds; --no-sync ignores it | The card defines the actual time this way, and per-image timestamps already handle unsynchronized cameras (D-46, D-47) | ignore sync.json; 1-based numbering | U-37; C-5; L3 per-time PSNR with and without sync |
| D-60 | Undistortion of SelfCap | the user's COLMAP image\_undistorter with blank\_pixels 0, as the card prescribes; cameras written as FULL\_OPENCV with cx, cy + 0.5 for COLMAP's half-integer pixel centers (COLMAP's documentation, Appendix A #28), and the PINHOLE result read back; --undistort internal as the alternative | Matches the published protocol, so per-scene results can be compared with Table 10 | internal undistortion by default; no undistortion | U-37; C-4 |
| D-61 | Replacing OpenCV in the core (owner choice, 2026-10-07) | own area resizing (D-33); own undistortion, in which each output pixel's source position comes from OpenCV's documented distortion model (4, 5, 8 or 12 coefficients; the tilted model is rejected), with K\_new = K, bilinear sampling and zeros outside; OpenCV-style YAML read with PyYAML after dropping the %YAML:1.0 line, with a constructor for opencv-matrix | Keeps LGPL binaries and the cv2 package clash out of the core; written from OpenCV's documentation only | opencv-python everywhere; headless core with RoMa installed by --no-deps | U-34; U-35; U-36; C-2 |
| D-62 | Distribution, import and CLI names (owner choice, 2026-10-07) | openftgs, openftgs and ftgs; the README's first line says the package is an independent clean-room implementation of FreeTimeGS, not affiliated with its authors | Marks an independent reimplementation, as OpenCLIP and OpenFold do, without reading as the authors' release | freetimegs; ftgs | the owner's pip check on 2026-10-07 found no existing distribution; the name is held only once a first release is published |
| D-63 | SelfCap calibration source and conventions (the card says only "follow EasyVolcap", which was left unopened) | hair from the separate hair-calib archive, corgi and bike from optimized/ in their release archives, --calib overriding; Rot and T read as world-to-camera with OpenCV axes; K shifted by +0.5; stop if Rot differs from Rodrigues(R) or a ccm is not the identity | The dataset repository holds hair's calibration as a separate archive, hair-calib (files dated 2025-02-12; the card does not mention it), so it is taken as current; which calibration the paper used is not stated. Under the world-to-camera reading every camera in both files faces the rig's middle, and under the other reading none does | hair-release's own calibration, if it has one; camera-to-world reading; applying ccm | U-37; C-4 |
| D-64 | Dynamic-region masks for SelfCap (the paper's background images are not released) | masks only from the user (--masks); dynamic-region metrics skipped without them; a temporal-median background of the test camera's video is documented as an approximation run outside the package, with results labeled approximate | The corgi archive contains no background images and the card mentions none, so the paper's masks cannot be rebuilt as published | drop dynamic-region metrics; estimate backgrounds inside the package | manifest records the mask source; metrics report labels approximate masks |
| D-65 | How RoMa is shipped (owner choice, 2026-10-07) | RoMa's inference modules at commit 77f8d68 (romatch/\_\_init\_\_.py; models/ with model\_zoo, matcher, encoders, tiny and the transformer with its DINOv2 layers; utils/ with \_\_init\_\_, utils, kde and local\_correlation; about 3,800 lines) are copied into openftgs/\_vendor/romatch. Benchmarks, datasets, losses, training and checkpointing are left out. The only changes: the 14 absolute romatch imports (12 at module level, 2 inside functions) point to the vendored path, cv2 is imported inside the three pose helpers instead of at module level, and the optional xformers import in the DINOv2 attention layer is removed, so attention always runs on PyTorch's scaled\_dot\_product\_attention. RoMa's MIT LICENSE and DINOv2's Apache-2.0 LICENSE sit next to the copy, Meta's file headers stay, and NOTICE names both. VENDOR.json records the commit, the patch and each file's SHA-256; updates happen only by re-running the vendoring script at a new commit and passing U-38 again | No install contains OpenCV, wandb, h5py, matplotlib or poselib, because matching needs only torch, torchvision, NumPy, Pillow, einops and loguru; the pinned copy also keeps initialization reproducible across RoMa releases | an upstream change making the import lazy; RoMa from PyPI with --no-deps and a placeholder cv2 module; RoMa from PyPI with opencv-python | U-38; U-19; B-8 |
| D-66 | Which images initialization may use (the paper does not say whether test views are excluded) | training-split images only: frame slots, camera pairs, matches and colors never read a test image; point clouds that may encode test views, such as SelfCap's archive clouds, are used only when supplied, and those runs are labeled | Test views must not influence training, as for S (D-27); the SelfCap card says its point clouds come from multiview images for a model trained with no held-out view | all cameras, as static-scene SfM pipelines often do | U-19 (no test image read); L3 test PSNR with and without the test camera in the initialization |
| D-67 | Evaluation details the paper leaves open: the SSIM used for metrics, clamping, averaging, and how Appendix B.1's dynamic-region crop treats renders and soft masks | metric SSIM = E7's SSIM (11×11 Gaussian, σ = 1.5, valid region, mean over the map and channels) with data range k, so C₁ = (0.01k)² and C₂ = (0.03k)²; renders clamped to \[0, 1\]; metrics per test image, averaged per scene, then over scenes; dynamic-region masks binarized at 0.5, one bounding box per test image, crop and black fill applied to both the render and the ground truth | The SSIM paper's constants, with the data range as the only change; clamping and per-image averaging as in gsplat's trainer; masking only the ground truth would charge the render for the background the dynamic-region metric is meant to exclude | a 7×7 uniform window; PSNR from pooled MSE; one box per sequence; masking the ground truth only | U-23; L3, which recomputes every metric from saved renders; SelfCap rows vs Table 10 |
| D-68 | Input range of the LPIPS metric (the paper names the backbones only) | inputs mapped from \[0, 1\] to \[−1, 1\] for both backbones, as the lpips README prescribes; LPIPS-VGG also reported with unmapped \[0, 1\] inputs, as lpips\_vgg\_raw | The mapping is the network's documented input range. gsplat's trainer feeds VGG \[0, 1\] images to match the 3DGS repository, and 3DGS-MCMC notes that published LPIPS values needed correcting, so both are reported until a comparison shows which convention the paper used | mapped inputs only; raw \[0, 1\] inputs for VGG only (the 3DGS and gsplat convention) | U-23; both variants against Tables 3 and 10 (SelfCap) and Tables 2 and 9 (ENeRF-Outdoor) |
| D-69 | How the ablations' "10-frame with the fastest motion" is chosen (§4.2 does not say) | within dance1's 60-frame window, the 10 consecutive frames whose consecutive ground-truth frames of the test camera differ most (mean absolute difference); the range goes into the manifest and the results | Uses only test-view ground truth, so training is unaffected, and anyone can recompute it from the files | optical-flow magnitude; motion of the reconstructed primitives | the reported range; Tables 4 and 5's "fastest" columns compared on it |

# The method

This page restates the method openftgs implements, from the owner's specification
({repo}`docs/spec/openftgs-spec.md`), which was written from the paper (Wang et al., CVPR 2025,
[arXiv 2506.05348](https://arxiv.org/abs/2506.05348) v2) alone. The parameters (§1.1), the notation
(§0) and the equations E1 to E17 (§1.2) are transcribed from the spec verbatim, each equation with
its label and its provenance tag; the training procedure (§1.3) is summarized. Where the two differ,
the spec wins. Nothing here comes from another implementation of the paper.

## Provenance tags

Every rule carries one tag, so a reader can tell what the paper fixes and what can be changed
(spec §0):

| Tag | Meaning | May the implementer change it? |
| --- | --- | --- |
| \[P §x, Eq. n\] | Stated in the paper | No |
| \[P→3DGS\] | Paper says "same settings as 3DGS"; value taken from 3DGS / gsplat defaults | Only with a logged reason |
| \[L: source\] | Standard Gaussian-splatting practice from an allowed source | Only with a logged reason |
| \[D-n\] | Design decision made here; rationale, alternatives and a detecting test are in Appendix B | Yes: via config where §2.1 defines a field for it, otherwise by a logged change; never silently \[D-72\] |

## Notation

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

## Per-primitive parameters

From spec §1.1.

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

**Validity rules, checked on load and after every relocation.** All tensors are finite. Every quaternion has norm > 1e-8. N is identical across all nine tensors. With a single timestamp (T\_τ = 0), every time center and every velocity is 0 \[D-71\]. N never exceeds N\_max, and initialization stops with an error when the start budget F·B (D-24) would exceed it. N changes only at growth events (500 ≤ it ≤ 15,000) and only upward \[D-18, D-44\]; the one exception is the "w/o periodic relocation" ablation, which prunes (§1.3).

**Size evidence.** Table 7 of the paper reports model sizes that work out to about 124 bytes per primitive across all seven rows (if sizes are in MiB). That is 31 float32 values (for example SH degree 1 plus extras) or 62 float16 values; SH degree 3 stored in half precision (64 values, 128 bytes) is within 4%. The sizes therefore do not settle L, which is a user parameter (D-3).

## Equations

The model, losses and schedules are fixed by equations E1–E17. E1–E4, E7, E8 and E10 restate the paper's Eqs. 1–7; the others fill in what the paper delegates to 3DGS or leaves implicit, and each carries its tag.

### E1 · Motion

Provenance: \[P §3.1, Eq. 1\]

```{math}
\mu_x(t) = \mu_x + v\,(t - \mu_t)
```

Linear for every t, including outside \[0, 1\]; no clamping. Jacobians used by tests: ∂μx(t)/∂μx = I, ∂μx(t)/∂v = (t − μt)·I, ∂μx(t)/∂μt = −v.

### E2 · Color

Provenance: \[P §3.1, Eq. 2; +0.5 offset and clamp: L: gsplat 1.5.3, D-4\]

```{math}
c_i = \max\Big(0,\; \sum_{l=0}^{L}\sum_{m=-l}^{l} c_{i,lm}\, Y_{lm}(d_i) + 0.5\Big),\qquad d_i = \frac{\mu_{x,i}(t) - o_{cam}}{\lVert \mu_{x,i}(t) - o_{cam} \rVert}
```

o\_cam = −R\_wᵀ t\_w is the camera center recovered from the world-to-camera matrix. Y\_lm is the real SH basis with gsplat's constants and band ordering. The view direction uses the moved position, as Eq. 2 states. The max is taken per channel. Only bands l ≤ L\_active(it) contribute during training (E13).

### E3 · Space–time opacity

Provenance: \[P §3.1, Eq. 3\]

```{math}
\sigma_i(x,t) = \sigma_{t,i}(t)\;\sigma_i\;\exp\!\Big(-\tfrac12\,(x-\mu_{x,i}(t))^\top \Sigma_i^{-1} (x-\mu_{x,i}(t))\Big)
```

σᵢ = sigmoid(ôᵢ) and Σᵢ comes from E5. The renderer receives the product σ\_eff,i(t) = σt,i(t)·σᵢ as the per-primitive opacity; the Gaussian factor is evaluated by splatting (E6).

### E4 · Temporal opacity

Provenance: \[P §3.1, Eq. 4\]

```{math}
\sigma_{t,i}(t) = \exp\!\Big(-\tfrac12\Big(\frac{t-\mu_{t,i}}{s_i}\Big)^2\Big),\qquad s_i = e^{\hat s_i}
```

Unnormalized: the peak is 1 at t = μt. Derivatives: ∂σt/∂μt = σt·(t − μt)/s² and ∂σt/∂ŝ = σt·((t − μt)/s)². μt also receives gradient through E1.

### E5 · Covariance

Provenance: \[P §3.1, text under Eq. 3; L: 3DGS Eq. 6; quaternion convention: gsplat docs\]

```{math}
\Sigma_i = R(q_i)\,S_i S_i^\top R(q_i)^\top,\qquad S_i = \mathrm{diag}\big(e^{\hat a_i}\big)
```

```{math}
R(w,x,y,z) = \begin{bmatrix} 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2) \end{bmatrix}
```

q is normalized before use.

### E6 · Splatting and compositing

Provenance: \[P Fig. 2 and §3.2 ("similar to 3DGS"); L: 3DGS Eqs. 3 and 5; constants: gsplat 1.5.3, D-6, D-31\]

For a camera with world-to-camera rotation R\_w, translation t\_w and intrinsics K, at pixel center p = (u + ½, v + ½):

```{math}
x_c = R_w\,\mu_x(t) + t_w,\qquad \mu'_i = \Big(f_x \tfrac{x_c}{z_c} + c_x,\; f_y \tfrac{y_c}{z_c} + c_y\Big),\qquad \Sigma'_i = J\, R_w \Sigma_i R_w^\top J^\top + \epsilon I_2
```

```{math}
\alpha_i(p) = \min\!\Big(\alpha_{max},\; \sigma_{eff,i}(t)\, \exp\!\big(-\tfrac12 (p-\mu'_i)^\top \Sigma'^{-1}_i (p-\mu'_i)\big)\Big)
```

```{math}
C(p) = \sum_i c_i\, \alpha_i(p) \prod_{j<i} \big(1-\alpha_j(p)\big) \;+\; \Big(\prod_i \big(1-\alpha_i(p)\big)\Big)\, c_{bg}
```

Normative details, all matching gsplat 1.5.3 in classic mode:

- J is the EWA Jacobian of the perspective projection at x\_c, with x\_c/z\_c clamped to \[−(c\_x/f\_x + 0.15W/f\_x), (W − c\_x)/f\_x + 0.15W/f\_x\] and y\_c/z\_c likewise with H, f\_y, c\_y. The clamp applies to J only, not to μ′.
- ε = 0.3; α\_max = 0.999; a term is skipped when α < 1/255 or when its quadratic form ½(p − μ′)ᵀΣ′⁻¹(p − μ′) is negative (a numerical guard). The 3DGS paper clamps α at 0.99 instead (D-31).
- Primitives are composited front to back by camera-space depth z\_c.
- A pixel stops before the primitive that would bring transmittance to ≤ 10⁻⁴; that primitive is excluded.
- Primitives with z\_c < 0.01 or z\_c > 10¹⁰ are culled, as are primitives with σ\_eff < 1/255. The screen-space extent is √(2 ln(255·σ\_eff)) standard deviations, never more than 3.33. Radii are per axis, r\_x = ⌈e·√Σ′₀₀⌉ and r\_y = ⌈e·√Σ′₁₁⌉ with e that extent; a primitive whose det Σ′ ≤ 0, or whose box μ′ ± r lies wholly outside the image, gets zero radii. A primitive is visible in an image when both radii are positive.
- The background c\_bg is black \[D-34\].

### E7 · Rendering loss

Provenance: \[P §3.2, Eq. 5; weights P §3.3\]

```{math}
\mathcal L_{render} = \lambda_{img}\,\mathcal L_{img} + \lambda_{ssim}\,\mathcal L_{ssim} + \lambda_{perc}\,\mathcal L_{perc},\qquad (\lambda_{img},\lambda_{ssim},\lambda_{perc}) = (0.8,\;0.2,\;0.01)
```

- L\_img = mean |Î − I| over pixels and channels (L1) \[D-7\].
- L\_ssim = 1 − SSIM(Î, I) with an 11×11 Gaussian window (σ = 1.5), C₁ = 0.01², C₂ = 0.03², data range 1, per channel, no padding ("valid"), averaged over the map and channels \[D-8\].
- L\_perc = LPIPS with the VGG backbone, inputs mapped from \[0, 1\] to \[−1, 1\] \[D-9\].

Î is the render after background compositing and is not clamped: E2 clamps colors only at 0, so Î can exceed 1, as in gsplat 1.5.3's trainer. I is float32 RGB in \[0, 1\]. Metrics clamp Î to \[0, 1\] first (E16).

### E8 · 4D regularization

Provenance: \[P §3.2, Eq. 6; weight P §3.3 and Table 5; "early stage": P §1\]

```{math}
\mathcal L_{reg}(t) = \frac1N \sum_{i=1}^{N} \sigma_i \cdot \mathrm{sg}\big[\sigma_{t,i}(t)\big]
```

t is the timestamp of the current training image. The sum runs over all N primitives, visible or not, as Eq. 6 states \[P Eq. 6; D-11\]. The only nonzero gradient is ∂L\_reg/∂ôᵢ = (1/N)·σt,i(t)·σᵢ(1 − σᵢ).

### E9 · Total loss

Provenance: \[weights P §3.3; composition D-37; schedule D-10\]

```{math}
\mathcal L = \mathcal L_{render} + \lambda_{reg}\,\mathcal L_{reg}(t),\qquad \lambda_{reg} = 10^{-2}\ \text{at every iteration}
```

λ\_reg applies at every iteration by default. LossConfig.reg\_until ends it after a given iteration, which is how §1's "early stage" would be read as a cutoff; the paper gives no value for one (D-10).

### E10 · Relocation sampling score

Provenance: \[P §3.2, Eq. 7; weights P §3.3; normalization D-12, D-13\]

```{math}
\rho_i = \lambda_g\, \hat g_i + \lambda_o\, \sigma_i,\qquad \lambda_g = \lambda_o = 0.5,\qquad \hat g_i = \min\big(1,\; g_i / Q_{0.99}(g)\big)
```

gᵢ is the mean, over the iterations since the last relocation in which primitive i was visible, of ‖∂L/∂μ′ᵢ‖ with the 2D gradient rescaled to NDC units (x by W/2, y by H/2). Q₀.₉₉ is taken over live primitives that were visible at least once, interpolating linearly between order statistics (torch.quantile's default); if it is 0, ĝ ≡ 0. σᵢ is the base opacity, as in Eq. 7.

### E11 · Relocation move

Provenance: \[L: 3DGS-MCMC Eq. 9; adoption D-15\]

For a live target j that receives n − 1 relocated primitives (n copies in total):

```{math}
\sigma^{new}_j = 1 - (1-\sigma_j)^{1/n},\qquad a^{new}_j = a_j\,\frac{\sigma_j}{\sum_{i=1}^{n}\sum_{k=0}^{i-1}\binom{i-1}{k}\dfrac{(-1)^k\,(\sigma^{new}_j)^{k+1}}{\sqrt{k+1}}}
```

All n copies get σ\_new and a\_new plus the target's other tensors, including `times`, `log_durations` and `velocities`. n is clamped to \[1, 51\]. a\_new is computed in float64 from the unclamped σ\_new; only then is σ\_new clamped to \[0.005, 1 − 2⁻²³\], with the lower bound applied to the logit (D-14). gsplat 1.5.3 uses the same order. The n − 1 moved copies then take positions drawn from the target's Gaussian, μ\_j + R\_j diag(a\_j) z with z \~ N(0, I₃), as E17's split does \[D-15\].

### E12 · Velocity learning-rate annealing

Provenance: \[P §3.2; product reading D-19; indexing D-41\]

```{math}
\eta_v(u) = \eta_{v,0}^{\,1-u}\;\eta_{v,1}^{\,u},\qquad u = \frac{it-1}{T_{it}-1}\in[0,1]
```

The paper prints λ\_t = λ₀^(1−t) + λ₁^t. Read literally, the sum is about 1 at both ends (λ⁰ = 1 at u = 0 and at u = 1) and dips in between, so the rate would rise again at the end instead of annealing. The product form is the 3DGS log-linear schedule. Defaults: ηv,0 = ηx,0/Δt\_f and ηv,1 = ηx,1/Δt\_f \[D-19\].

### E13 · Other schedules

Provenance: \[P→3DGS; L: 3DGS §5.1 and §7.1, 3DGS-MCMC §3.6; D-20; D-41\]

```{math}
\eta_x(u) = \eta_{x,0}\Big(\frac{\eta_{x,1}}{\eta_{x,0}}\Big)^{u},\qquad \eta_{x,0} = 1.6\times10^{-4}\,S,\qquad \eta_{x,1} = 1.6\times10^{-6}\,S
```

```{math}
L_{active}(it) = \min\big(\lfloor (it-1)/1000 \rfloor,\; L\big)
```

`times` uses the same exponential form, 1.6e-4 → 1.6e-6 in normalized time \[D-20\]. S is the spatial scale of D-27.

### E14 · Time normalization

Provenance: \[D-1, D-46\]

```{math}
t = \frac{\tau-\tau_0}{\tau_1-\tau_0},\qquad \Delta t_f = \operatorname{median}_{c}\Big(\operatorname{median}\big(\operatorname{diff}(\operatorname{sort}(t_{train,c}))\big)\Big)
```

τ₀ and τ₁ are the smallest and largest training timestamps; T\_τ = τ₁ − τ₀. Inputs given in raw units convert as μt = (τ\_c − τ₀)/T\_τ, s = d/T\_τ and v = v\_τ·T\_τ. With a single timestamp (T\_τ = 0), μt = 0 and durations and velocities convert with a time unit of 1 instead of T\_τ (s = d, v = v\_τ), consistent with Δt\_f := 1 (§1.4) \[D-70\]. Δt\_f is measured within each training camera c that has at least two images, so interleaved timestamps from unsynchronized cameras do not shrink it. Without such a camera it falls back to the pooled median of distinct timestamps, and to 1 for a single timestamp \[D-46\].

### E15 · Velocity initialization

Provenance: \[P §3.2, "Initialization of our representation"; formula D-23\]

```{math}
v_i = \frac{x^{(f')}_{\mathrm{nn}(i)} - x^{(f)}_i}{t_{f'} - t_f}
```

x^(f) are the points triangulated at frame f, f′ is the next initialization frame, and nn(i) is the nearest neighbour of xᵢ in x^(f′).

### E16 · Evaluation metrics

Provenance: \[P §4, "Metrics"; DSSIM form derived from Tables 2, 3, 9 and 10; D-29, D-67\]

```{math}
\mathrm{PSNR} = 10\log_{10}\frac{1}{\mathrm{MSE}(\hat I, I)},\qquad \mathrm{DSSIM}_k = \frac{1-\mathrm{SSIM}_k(\hat I, I)}{2},\quad k\in\{1,2\}
```

SSIM\_k is E7's SSIM (11×11 Gaussian window, σ = 1.5, valid region, mean over the map and channels) with data range k, so C₁ = (0.01k)² and C₂ = (0.03k)². Renders are clamped to \[0, 1\] first; each metric is computed per test image, averaged over a scene's test images, then over scenes \[D-67\]. Cross-check: SelfCap SSIM₂ 0.952 (Table 10) corresponds to DSSIM₂ 0.024 (Table 3), and ENeRF-Outdoor 0.846 (Table 9) to 0.077 (Table 2). The identity holds for every SelfCap row and for the ENeRF, 4K4D and Ours rows on ENeRF-Outdoor, but not for the 4DGS and STGS rows of Table 2 \[D-29, D-40\]. LPIPS uses AlexNet for Neural3DV and VGG for ENeRF-Outdoor and SelfCap, per the captions of Tables 6–11. Its inputs are mapped from \[0, 1\] to \[−1, 1\] for both backbones, and LPIPS-VGG is also reported with unmapped \[0, 1\] inputs, the 3DGS convention that gsplat's trainer follows, because the paper does not say which it used \[D-68\].

### E17 · Split during growth

Provenance: \[L: 3DGS §5.2; adoption D-18, D-43\]

```{math}
\mu_x^{(k)} = \mu_x + R(q)\,\mathrm{diag}(a)\,z_k,\quad z_k \sim \mathcal N(0, I_3),\qquad a^{(k)} = a/1.6,\qquad k = 1, 2
```

The two children replace their parent and copy its other tensors, including `times`, `log_durations` and `velocities`. A clone copies all nine tensors unchanged.

## Initialization, relocation and optimization

A summary of spec §1.3; the spec's numbered steps are normative. Training starts from 4D points,
runs 30,000 Adam iterations at one image per iteration, grows the primitive set between iterations
500 and 15,000, and relocates low-opacity primitives every 100 iterations instead of deleting them.
There is no opacity reset and no pruning \[D-18, D-42\].

### Initialization

From the training images only \[D-66\], at initialization frame slots spaced Δt_f apart \[D-21,
D-47\]:

- every training camera is paired with the one whose optical axis is closest, and each pair is
  matched densely with RoMa (by default `roma_outdoor`), keeping up to 5,000 matches with a
  certainty of at least 0.5 \[D-22, D-58\];
- each match is triangulated by two-view linear DLT and kept with positive depth, a reprojection
  error of at most 2 px in both views and a triangulation angle of at least 1° \[D-22\];
- each point takes its position, the mean normalized time of its two images, a velocity by E15
  against the next initialization frame \[D-23\], an initial duration s₀ = k_stride·Δt_f \[D-21\],
  isotropic scales from the mean distance to its 3 nearest neighbours \[D-25\], opacity 0.1,
  the identity quaternion and its color as `sh0` \[D-5\];
- the start holds at most B points per frame of the sequence, F·B in all, B = 1,000 by default
  \[D-24, D-74\].

Points supplied through the `ftgs-init` file (§2.2) skip matching and triangulation; missing
velocities become 0 and missing durations Δt_f.

:::{admonition} TODO
The spec gives no equation for the camera pairing, the certainty-weighted draw of matches, the
DLT triangulation or its filters, or the scale initialization; the steps above are prose in §1.3.
:::

### One training iteration

1. Take the next image of a seeded per-epoch permutation of the training images \[D-26\].
2. Compute its normalized time t (E14) and the active SH degree (E13).
3. Build the per-time arrays (E1, E3, E4), cull primitives below an effective opacity of 1/255, and
   render (E6).
4. Evaluate the total loss (E9).
5. Backpropagate, and accumulate each visible primitive's NDC-scaled 2D-mean gradient norm for the
   relocation score (E10).
6. Set this iteration's learning rates (E12, E13) and take one Adam step on all nine tensors.
7. At every multiple of 100 between 500 and 25,000: grow (up to iteration 15,000), then relocate,
   then reset the gradient statistics \[D-17, D-45\].

### Growth

Primitives whose mean NDC gradient exceeds τ_pos = 0.0002 are candidates; when N would exceed
N_max (3,000,000 by default), only the candidates with the largest gradients that fit are kept
\[D-44\]. A small candidate (largest scale at most 0.01·S) is cloned unchanged, a larger one is split
into two children by E17. New rows start with zero Adam moments and inherit their parent's
gradient statistics \[D-45\]. Nothing is deleted and opacities are never reset \[D-42\].

:::{admonition} TODO
The spec gives no equation for the candidate test, the cap on N or the clone; only the split has
one (E17).
:::

### Relocation

Primitives whose base opacity is below 0.005, tested on the stored logit, are dead \[D-14\]. Each
dead primitive draws a live target with probability proportional to the score ρ of E10, using the
trainer's own checkpointed generator \[D-15\]. Each target that receives copies has its opacity and
scales updated by E11, and every dead primitive takes its target's updated values, with a position
drawn from the target's Gaussian. The Adam moments of every touched row are zeroed and their step
counters kept \[D-16\]. N is unchanged.

:::{admonition} TODO
The spec gives no equation for the dead-set test, the draw of targets or the reset of the Adam
moments; the score (E10) and the move (E11) have one.
:::

### Count-control modes

`growth = "gradient"` (the default) follows the growth rules above; `"score"` appends primitives
drawn by ρ and initialized by E11 at each growth event, up to N_max; `"none"` keeps N as
initialized \[D-18\]. The paper's "w/o periodic relocation" ablation replaces relocation with
3DGS densification, with pruning and opacity resets (spec §1.3).

### Optimizer and learning rates

Adam with β₁ = 0.9, β₂ = 0.999, ε = 10⁻¹⁵, no weight decay, one parameter group per tensor
\[P→3DGS\]. The rates, from spec §1.3:

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

S = 1.1 × the largest distance from a training camera center to the mean of the training camera centers \[D-27\]. For a 300-frame sequence Δt\_f = 1/299, so the velocity rate starts at about 0.048·S per unit of t. With a single timestamp (T\_τ = 0), `times` and `velocities` take no updates (no parameter group, or a zero rate), so that they stay 0 (§1.4) \[D-71\].

### Iteration calendar

For the default T\_it = 30,000 (spec §1.3):

| Event | Iterations |
| --- | --- |
| SH bands active (L\_active = 0, 1, 2, 3) | 1–1,000 · 1,001–2,000 · 2,001–3,000 · 3,001–30,000 |
| 4D regularization on | 1–30,000 (whole run) |
| Growth (146 events) | 500, 600, …, 15,000 |
| Relocation (246 events) | 500, 600, …, 25,000 |
| Learning-rate decay | every iteration |
| Periodic checkpoints (ftgs train --checkpoint-every N) | N, 2N, … below 30,000; none by default \[D-83\] |
| Final checkpoint and evaluation | 30,000 |

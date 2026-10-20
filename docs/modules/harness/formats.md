# Harness formats

Everything crosses the boundary in the §2.2 formats plus the two harness formats below (§3.3 adapter contract). The harness reads and writes them with its own NumPy and safetensors code, written from §2.2, and validates every file an adapter produces; a file that fails is a check with the pattern "schema violation".

## `info`

An adapter's `info` prints one JSON object (Q-205):

- `name`, `version`;
- `query_kinds`: the L2 kinds it answers; a kind missing here is `not_comparable` in L2;
- `rasterizer`: `{name, version}` of the renderer that produced its pixels (openftgs: `openftgs-reference`, or `gsplat` with its version);
- `render_conventions`: at least the 12 entries of `ftgs-if`'s block (Q-35, D-73). A reply without one of them is a failed `info` check.

The L1 **tier** is "same rasterizer and constants" only when both `rasterizer` objects and every `render_conventions` entry match; otherwise it is "different rasterizers" (Q-206).

## `ftgs-forward-queries` and `ftgs-forward-answers`

One safetensors file per query kind (Q-207).

Queries: metadata `format` (`ftgs-forward-queries`), `version` (1), `kind`, `producer`, and `queries`, a JSON list of `{id, params}`. Tensors are keyed `<id>/<name>`. FTGS-IF arrays sit under `<id>/model/<ftgs-if key>` with §2.2's dtypes, and `params.model` holds `sh_degree`, `tau0`, `tau1` and `time_unit`. Times are raw τ in float64; images are [H, W, 3] float32.

Answers: metadata `format` (`ftgs-forward-answers`), `version`, `kind`, `producer`, `queries_sha256` and `answers`, a JSON list of `{id, status, message}` with status `ok`, `error` or `not_comparable`. Tensors are keyed `<id>/<name>`.

| Kind | Inputs | Answer |
| --- | --- | --- |
| motion | model, `times` [K] | `means_t` [K,N,3] |
| temporal_opacity | model, `times` [K] | `sigma_t` [K,N] |
| effective_opacity | model, `times` [K] | `opacity_eff` [K,N] |
| sh_color | model, `times` [K], `centers` [K,3]; `sh_degree` | `colors` [K,N,3] |
| losses | `pred`, `gt` [H,W,3]; model and `t` for L_reg; LossConfig fields, `iteration` | `l1`, `one_minus_ssim`, `lpips_vgg`, `reg`, `total` (float64 scalars) |
| lr_schedule | `iterations` [M]; OptimConfig fields, `spatial_scale`, `frame_dt` | `rates` [M,9] float64; tensor names in the metadata |
| relocation_math | `opacity` [k], `scales` [k,3], `n` [k] | `opacity_new`, `scales_new` (E11 with D-14's full clamp) |
| velocity_init | `points` [N,3], `next_points` [M,3], `dt` | `velocities` [N,3] |
| relocation_sampling | `scores` [N]; `count`, `seeds` (50) | `counts` [50,N] |
| grad_stats | model, `K`, `world_to_camera`, `t`, `gt`; `width`, `height`, `iteration`, a TrainConfig subset | `grad_norm` [N] (0 where invisible), `visible` [N] |
| relocation_score | `opacity`, `g_sum`, `count`; RelocConfig fields | `score` [N] |
| spatial_scale | `world_to_camera` [C,4,4], `train` [C]; `factor` | `S` |
| growth_decisions | `g` [N], `scales` [N,3]; `spatial_scale`, `grad_threshold`, `split_scale`, `n_max` | `clone`, `split` [N] bool |

## `not_comparable.json`

An adapter's `train` may write `OUT_DIR/not_comparable.json`, a JSON list of the dotted `harness_config.json` keys it could not honor. Each becomes a `not_comparable` check of L3, so no setting is dropped silently (§3.3, Q-202).

# The RoMa pipeline in `openftgs.init`

How `roma_init` meets §1.3 "Initialization" steps 1–7 and E15: init frame slots, camera pairs, dense matching, triangulation and the assembly of each init frame into init points. Steps 8 and 9, and the trainer's total cap F·B (D-74), stay in `FreeTimeGaussians.from_init_points` (`docs/modules/model/initialization.md`). The public names are Q-106's: `roma_init(dataset, cfg, *, devices, matcher)`, the `DenseMatcher` protocol, `roma_provenance`, `knn_velocities` and `triangulate_dlt`.

## Order of work

1. **Init frame slots** (D-21, D-47, D-66; Q-109). Only training records take part. The slots are every k_stride-th frame index of the sequence, F counted as in Q-21. An image joins a slot when its nearest frame is that slot's and it lies within Δt_f/2 of t_k; per camera the closest image wins. A camera with no such image sits the slot out.
2. **Camera pairs** (D-22; Q-108, Q-109). Within a slot, each camera's partner is the camera whose optical axis is angularly closest, read from each image's own camera; ties go to the center distance, then to the manifest's camera order. Each unordered pair is matched once, oriented (A, B) in the order of the manifest's `cameras` object.
3. **Matching** (D-58; Q-106, Q-110). The pair's two images go to a dense matcher: RoMa per `RomaInitConfig.model` by default, or the `matcher` supplied. Every match with a coordinate outside the image is dropped; of the rest, those whose raw certainty reaches the threshold are drawn in proportion to certainty, without replacement, with the pair's own generator, on the CPU.
4. **Triangulation** (D-22, D-56; Q-110). `triangulate_dlt` in float64; the depth, reprojection and angle filters of D-22; a color from the two source pixels; the mean raw τ of the two images as the timestamp. Positions are then rounded to float32, the values the init points store (Q-111).
5. **Assembly per init frame** (E15, D-23, D-24; Q-111). A slot's points are its pairs' points in canonical order. The per-init-frame cap k_stride·B keeps a seeded subset in that order; velocities come from E15 against the next slot's full point set (the previous one's on the last slot, none with a single slot); durations are s₀ in raw units; the frame ids are the slot indices.

## Seeds

Every random draw has a generator of its own, seeded from `RomaInitConfig.seed` and what the draw belongs to: the matches of a pair from (seed, slot, A, B), the subset of a slot from (seed, slot), by Q-72's hashing pattern (Q-110, Q-111). No draw depends on another's position in a stream, so the init points don't depend on the order in which pairs are matched.

## Shards

`devices` names one shard per entry, each with its own dense matcher; repeating an entry puts several shards on one GPU (Q-106). Pairs are listed in record order and taken by the shards from a shared queue; results are keyed by (slot, A, B) and assembled in canonical order, so how the pairs are spread over shards changes nothing (D-54, D-56, U-33). The weights are fetched once, before the shards start (Q-112).

## Determinism and precision

For the whole call, `roma_init` sets the float32 matmul precision RoMa requires, turns cuDNN autotuning off and deterministic algorithms on, and restores the previous settings afterwards (D-56, §2.4 "Precision"; Q-112). RoMa keeps its own float16 autocast. Triangulation, the filters and E15 run in float64; the k-NN of E15 is exact, ties to the lower index (D-56).

## The dense-matcher port

A dense matcher takes two images as `load_image` returns them and gives a warp in RoMa's normalized coordinates with a certainty per match (Q-106). The RoMa adapter imports the vendored copy (`docs/modules/init/vendored-roma.md`) only when it is built, so `openftgs.init` imports without the `[init]` extra. It hands RoMa PIL images and returns the whole two-way warp; `use_custom_corr` is on only when fused-local-corr imports (Q-112). The Tiny RoMa adapter loads XFeat from the commit whose license was checked (Q-116). Tests supply an analytic matcher that knows the scene's geometry, so U-19's clauses run on the CPU.

## Provenance

`roma_provenance(cfg)` gives the metadata `write_init_points` records: the settings, whether the fused local correlation was used, and the commits of the vendored RoMa and, for Tiny RoMa, of XFeat (D-58; Q-107).

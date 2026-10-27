# Quickstart

From multi-view video to a trained model, its metrics and renders at any time, on `kitchen`, a
scene of [openftgs-synth](https://github.com/samaust/openftgs-synth), the dataset openftgs is
validated on (D-79). The commands are the ones the project's own validation runs used, and the
times are theirs, on one RTX 4090 (24 GB) with an Intel i9-12900
({repo}`docs/clean-room/validation-runs.md`).

You need openftgs with the `[init]` extra, a CUDA build of PyTorch and ffmpeg 5.1 or later
({doc}`installation`).

## 1. Render the scene

openftgs-synth is published as its generator, a separate project under the GPL-3.0-or-later
license; the frames it renders are CC0-1.0. No rendered copy is available for download, so you
render `kitchen` yourself. The generator's
[README](https://github.com/samaust/openftgs-synth#readme) says how: it renders with Blender's
Cycles on an NVIDIA GPU, after a one-time download of its CC0 assets, and runs a scene's
configuration through build, render and package stages, then validates and checksums the
result. It can also time one frame first, to estimate the full render's time and disk use.

`kitchen` comes out in the Neural3DV layout: a folder `n3dv/kitchen` under the dataset's root
holding `poses_bounds.npy` and one video per camera, `cam00.mp4` to `cam19.mp4`. Below, `$SYNTH`
is the dataset's root.

## 2. Convert it into a scene manifest

```bash
ftgs convert n3dv $SYNTH/n3dv/kitchen kitchen --resize 1.0
```

This decodes the first 300 frames of each camera at 30 fps with ffmpeg and writes the frames
under `kitchen/images/` and the scene manifest `kitchen/scene.json` (§2.2): 20 cameras × 300
frames at 1352×1014, with `cam00` as the test view (§1.4). The scene is already at the size the
protocol trains at, hence `--resize 1.0` instead of Neural3DV's 0.5. About 41 s.

## 3. Cache the frames (optional)

```bash
ftgs cache kitchen/scene.json
```

Decodes every training frame once into a frame cache, keyed by the manifest's SHA-256; training
builds it on first use otherwise (D-51).

## 4. Initial points

```bash
ftgs init roma kitchen/scene.json kitchen-init.safetensors
```

At every frame, pairs each training camera with the one whose view is closest, matches each pair
with RoMa, triangulates the matches and estimates each point's velocity against the next frame
(§1.3): 300,000 points, 1,000 per frame. 30 min 26 s; the first run also downloads RoMa's
weights.

## 5. Train

```bash
ftgs train kitchen/scene.json --init kitchen-init.safetensors --out kitchen-run
```

30,000 iterations, with ꟻLIP as the perceptual loss at its calibrated weight (D-9, D-89). The
primitive count grows from the 300,000 initial points to its cap of 3,000,000, reached at
iteration 2,100. 14 min 22 s, the evaluation of the test view included, at a GPU memory peak of
23,533 MiB. The run directory holds `config.json`,
`train_log.jsonl`, `checkpoint/` and `METRICS.json`; Ctrl-C saves the checkpoint before stopping,
and `--resume kitchen-run/checkpoint` continues the run. `--checkpoint-every N` also saves every N
iterations (D-83).

## 6. Evaluate, export and render

```bash
ftgs eval kitchen-run/checkpoint kitchen/scene.json --split test --out METRICS.json
ftgs export kitchen-run/checkpoint --format ftgs-if --out kitchen.safetensors
ftgs render kitchen.safetensors --request request.json --out renders.safetensors
```

- `ftgs eval` scores the test view: PSNR, SSIM and DSSIM at data ranges 1 and 2, and ꟻLIP
  (E16), per image and per scene. `ftgs train` already wrote the same file into the run
  directory. The project's default run scored a PSNR of 28.32 dB and a ꟻLIP of 0.0900 on
  `cam00`.
- `ftgs export` writes the model as an exchange file (`ftgs-if`, §2.2): activated values in raw
  time units, which other implementations can read.
- `ftgs render` answers a render request: a JSON file (`ftgs-render-request`, §2.2) that lists
  cameras (size, `K`, `world_to_camera`) and timestamps, in the scene's time unit. Any time
  renders, outside the training range too, where the motion extrapolates linearly. The output
  holds `rgb/<id>` and `alpha/<id>` for each item.

```text
{
  "format": "ftgs-render-request", "version": 1, "time_unit": "s",
  "background": [0, 0, 0], "sh_degree": null,
  "items": [
    {"id": "r0000", "timestamp": 1.2345,
     "camera": {"width": 1352, "height": 1014, "K": [[...]], "world_to_camera": [[...]]}}
  ]
}
```

## On Neural3DV

A Neural3DV scene, for those who hold the dataset under its own license (CC-BY-NC-4.0), takes the
same commands, with the protocol's resize ratio: `ftgs convert n3dv SRC OUT`, whose default ratio
of 0.5 gives 1352×1014 frames ({doc}`datasets`). One caveat: the converter's reading of
`poses_bounds.npy` (D-28's LLFF column order) has not been checked against the real files.

:::{admonition} What stays unverified
:class: warning

This project validates on openftgs-synth, not on the datasets of the paper (D-79). So:

- D-28's LLFF column order, ENeRF-Outdoor's file layout and D-59's reading of SelfCap's sync
  offsets are unverified against the real files;
- no result is compared with the paper's tables.
:::

# Datasets

`ftgs convert` turns a multi-view video dataset into a scene manifest (`ftgs-scene`, §2.2) and
undistorted PNG frames, which every other command reads. It has three converters, one per input
layout (§1.4): `n3dv` for Neural3DV's, `opencv-yaml` for OpenCV-style YAML calibrations such as
ENeRF-Outdoor's, and `selfcap` for SelfCap's archives. openftgs ships no dataset (§2.3).

:::{admonition} What stays unverified
:class: warning

The three datasets the paper evaluates on are licensed for non-commercial research only, and
openftgs may later be used commercially, so this project downloads none of them; it validates on
openftgs-synth instead (D-79). The converters are specified for users who hold the datasets, and
checked end to end on openftgs-synth's scenes in all three layouts (G-1 to G-7), but:

- D-28's LLFF column order (how `poses_bounds.npy` is read),
- ENeRF-Outdoor's file layout, and
- D-59's reading of SelfCap's sync offsets

are unverified against the real files, and no result is compared with the paper's tables.
:::

## openftgs-synth

The dataset openftgs is validated on (D-79): the owner's rendered multi-view video dataset, under
CC0-1.0. Its generator, [openftgs-synth](https://github.com/samaust/openftgs-synth), is a
separate project under the GPL-3.0-or-later license, which renders the scenes with Blender from
CC0 assets; its README says how ({doc}`quickstart`).

It holds scenes in all three input layouts, H.264 and lossless videos, unsynchronized cameras
among them, and exact ground truth: cameras, per-frame times, undistorted frames, depth, dynamic
masks and background images. The project's validation runs used three scenes:

| Scene | Layout | Converted with | Images |
| --- | --- | --- | --- |
| `kitchen` | Neural3DV, `n3dv/kitchen` | `ftgs convert n3dv $SYNTH/n3dv/kitchen OUT --resize 1.0` | 20 cameras × 300 frames, 1352×1014, 30 fps; test view `cam00` |
| `courtyard` | OpenCV-style YAML, `enerf_outdoor/courtyard` | `ftgs convert opencv-yaml $SYNTH/enerf_outdoor/courtyard OUT --fps 60 --frames 300 --resize 1.0 --test-camera 0009` | 18 × 300, 1920×1080, 60 fps; test view `0009` |
| `studio` | SelfCap, `selfcap` | `ftgs convert selfcap $SYNTH/selfcap OUT --scene studio --window N0:N1 --test-camera 0007 --undistort internal --resize 1.0 --masks DIR` | 24 × 60, 1920×1080, 60 fps, unsynchronized; test view `0007` |

`$SYNTH` is the dataset's root. `studio` is a 60-frame window of a longer sequence, and its
`--masks` folder holds one mask per test image, made from the ground truth
({repo}`docs/clean-room/validation-runs.md`).

## Neural3DV

- **License:** CC-BY-NC-4.0, non-commercial
  ([the dataset's repository](https://github.com/facebookresearch/Neural_3D_Video)).
- **Protocol:** the first 300 frames at 30 FPS of 19 to 21 cameras, 2704×2028 resized by 0.5 to
  1352×1014; `cam00` is the test view (§1.4).
- **Layout:** SRC holds `poses_bounds.npy` and one video per camera, `camXX.mp4`. Row i of the
  array belongs to the i-th video in name order; each row holds a 3×5 matrix (the camera-to-world
  rotation, the translation, and H, W and the focal length) plus near and far bounds (D-28).
- **Convert:**

  ```bash
  ftgs convert n3dv SRC OUT
  ```

  `--frames` (300) and `--resize` (0.5) default to the protocol's values. Frame k gets the
  timestamp k/30 s.

## ENeRF-Outdoor

- **License:** its own terms, non-commercial (§1.4, D-79).
- **Protocol:** the first 300 frames at 60 FPS of 18 cameras at 1920×1080, ratio 1.0. The paper
  doesn't name the test view, so it is an option (D-29).
- **Layout:** SRC holds `intri.yml` and `extri.yml`, OpenCV-style YAML with each camera's K,
  distortion, rotation and translation, read with openftgs's own reader (D-61), and either
  `images/<camera>/` or `videos/<camera>.mp4`. The layout is that of the zju3dv family that
  SelfCap also uses; it is not confirmed against ENeRF-Outdoor's files.
- **Convert:**

  ```bash
  ftgs convert opencv-yaml SRC OUT --fps 60 --frames 300 --resize 1.0 --test-camera NAME
  ```

  `--fps`, `--frames` and `--resize` are required. Without `--test-camera` every image is a
  training image. Frame k gets the timestamp k/fps s, and frames are undistorted with openftgs's
  own implementation of OpenCV's distortion model.

## SelfCap

- **License:** research and non-commercial use only, and modifications must stay open-source and
  non-commercial. Nothing derived from SelfCap may enter the repository, the wheels or CI (§1.4,
  §2.3). The dataset is on Hugging Face, `zju3dv/SelfCap-Dataset`, and its card gives the
  protocol.
- **Protocol:** each scene is a 60-frame window of a longer 60 FPS sequence, undistorted with
  COLMAP and resized by 0.5 (bike at 1.0), with one test camera (D-59, D-60):

  | Scene | Archive | Test camera | Frames n | Resize |
  | --- | --- | --- | --- | --- |
  | dance1 | hair-release | 0015 | 4120 ≤ n < 4180 | 0.5 |
  | dance2 | hair-release | 0015 | 5530 ≤ n < 5590 | 0.5 |
  | corgi1 | corgi-release | 0007 | 200 ≤ n < 260 | 0.5 |
  | corgi2 | corgi-release | 0007 | 2950 ≤ n < 3010 | 0.5 |
  | bike1 | bike-release | 0009 | 8900 ≤ n < 8960 | 1.0 |
  | bike2 | bike-release | 0009 | 30020 ≤ n < 30080 | 1.0 |

- **Layout:** SRC is a folder of the archives as downloaded. The converter reads only each
  archive's videos and its calibration, `optimized/intri.yml` and `optimized/extri.yml`, with
  `optimized/sync.json`'s per-camera offsets when the archive has one (D-63). The dance scenes'
  calibration comes from the separately published hair-calib archive (`--calib DIR`).
- **Convert:**

  ```bash
  ftgs convert selfcap SRC OUT --scene dance1
  ```

  A scene of the table takes its window, test camera and ratio from it; another scene needs
  `--window N0:N1` and `--test-camera`. A frame's timestamp is n/60 s minus its camera's sync
  offset, read as seconds (`--no-sync` ignores the offsets). COLMAP must be installed
  (`--colmap PATH`); `--undistort internal` uses openftgs's own undistortion instead, which is
  not protocol-exact. `--masks DIR` supplies one mask per test image for the dynamic-region
  metrics (D-64, D-67).
- The hosted corgi-release archive fails gzip's CRC check; the conversion records it as a warning
  in the manifest, and corgi results carry it.

## What a conversion writes

OUT must be absent or empty. Frames go to `OUT/images/<camera>/<frame_index:06d>.png`, masks to
`OUT/masks/<camera>/`, and the manifest `OUT/scene.json` last. Its metadata records how the scene
was made: the converter, the resize ratio, the frames or window, ffmpeg's path and version, the
command run for each camera and the decoder that ran, and for SelfCap the archives' SHA-256s and
the undistortion. A conversion that fails removes what it wrote. `--jobs J` sets how many cameras
decode at once, and `--hwaccel cuda` decodes on the GPU, failing rather than falling back to the
CPU (D-81).

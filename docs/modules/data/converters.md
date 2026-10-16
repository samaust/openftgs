# Dataset converters in `openftgs.data`

How `convert_n3dv`, `convert_opencv_yaml` and `convert_selfcap` meet their §2.1 rows: each turns a released dataset into undistorted PNG frames and a scene manifest (§1.4 "Dataset protocols", §2.2). Their keyword additions (`ffprobe`, and `test_camera` on `convert_opencv_yaml`) are the owner's answer of Q-85.

## Inputs

- **Neural3DV** (D-28, Q-95): SRC holds `poses_bounds.npy` and `camXX.mp4`. Row i of the array belongs to the i-th video in name order, and the counts must match. Each video's size, read with ffprobe, must equal its row's H and W. cam00 is the test view.
- **OpenCV-style YAML** (D-61, Q-96): SRC holds `intri.yml` and `extri.yml` and either `images/<camera>/` or `videos/<camera>.mp4`. ENeRF-Outdoor's dataset page documents no layout, so this is the layout of the zju3dv family that SelfCap uses (D-63); C-2 would confirm it against the real files, and this project doesn't run C-2 (D-79). `test_camera` names the test view; without it every image is a training image.
- **SelfCap** (D-59, D-60, D-63, Q-86, Q-97): SRC is a folder of the archives as downloaded. Each archive used is read once, hashed while it streams, and only its videos and calibration files are extracted, into `OUT/.work/`. Archives are untrusted data: only regular files are extracted, to paths the converter builds itself. A gzip CRC failure after whole members (the hosted corgi archive has one) becomes a warning in the manifest, not a failure. The card's table gives each scene's window and test view; a scene outside the table needs both.

## Output

OUT must be absent or empty (Q-88). Frames go to `OUT/images/<camera>/<frame_index:06d>.png`, masks to `OUT/masks/<camera>/`, and `OUT/scene.json` is written last, through `io.write_scene_manifest`, and returned. `OUT/.work/` is removed whether the conversion succeeds or fails, and a conversion that fails removes everything it wrote: OUT was absent or empty when it started, so it is left as it was found, and the same command can simply run again (Q-88). Invalid input or environment raises `ValueError` (Q-26), which `ftgs convert` (M14) maps to exit code 2.

The manifest's metadata records how the scene was made (Q-94): the source and converter, the resize ratio, frames or window, ffmpeg's path, version line, the command per camera and the decoder that ran, and for SelfCap the archives' SHA-256s, the calibration source, whether sync offsets were applied, the undistortion mode, COLMAP's commands, warnings, the masks' source and, for dance1, the Fastest-10 range (D-69, Q-59). `metadata.source` for Neural3DV is "n3dv/<scene>", which `eval.lpips_backbone` reads (Q-62).

## Decoding with ffmpeg

Frames come from the user's ffmpeg, run as a separate program (D-48); the package never links FFmpeg. Width, height, pixel format and frame count come from ffprobe, and the converter reads exactly W·H·3 bytes per frame from the pipe. Up to `jobs` cameras decode at once (0 means one per four cores), and resizing, undistortion and PNG encoding run in a process pool whose workers use one PyTorch thread (§2.4). Outputs don't depend on `jobs`.

Anything ffmpeg prints at `-v error`, a nonzero exit or a short read fails the conversion (Q-93). SelfCap windows decode the frames before the window too, and ffmpeg's messages carry no frame number, so an error before the window also fails; the message says so.

`--hwaccel cuda` asks for CUDA decoding in a way that fails rather than silently falling back to the CPU, so the manifest's `decoder` says what ran (Q-92). The downloaded frames are converted back to the stream's own pixel format before the RGB conversion, so they go through the CPU path's conversion and come out identical to CPU decoding (G-6: no pixel differs).

## Undistortion and resizing

The YAML calibrations hold K in OpenCV's integer-centre convention. Frames are undistorted with that K (`undistort_image`, K_new = K), then K gains +0.5 on cx and cy, then frames and K are resized (`images-and-cameras.md`). Undistortion passes float64 values to the resizing, which rounds to 8 bits once (Q-90).

SelfCap's protocol undistorts with COLMAP instead (D-60, Q-97): the converter writes a text model with FULL_OPENCV cameras (cx, cy + 0.5 for COLMAP's pixel centres), runs `image_undistorter --blank_pixels 0`, converts COLMAP's binary output to text with `model_converter`, and reads the PINHOLE cameras back without any shift, since COLMAP's convention is the spec's. COLMAP changes the image size and principal point and keeps the focal lengths, so W, H and K come from the read-back camera. C-4's Gaussian-dot check confirms the convention to 0.026 px. COLMAP's resampling is softer than a bilinear remap into the same camera, which keeps G-4's frame bound open (Q-103). `undistort="internal"` uses the package's own undistortion instead, which is not protocol-exact.

## Timestamps and masks

Neural3DV frame k has τ = k/30; the YAML converter's frame k has τ = k/fps. SelfCap's `frame_index` is the decode index n, and τ = n/60 minus the camera's sync offset, read as seconds from `optimized/sync.json` (an object of camera name → offset; Q-101), unless `use_sync=False`.

`masks=DIR` supplies one mask per test image, `DIR/<n:06d>.png`, at the converted image's size, with a `DIR/masks.json` giving the masks' source and whether they are approximate (D-64, Q-87). A folder missing any test image's mask is refused, since §1.4 needs one per test image (Q-104). openftgs-synth ships true background images and dynamic masks, the ground truth that D-64 lacks for SelfCap.

## Checks

U-27 and U-37 run on clips that ffmpeg generates from its lavfi sources, with a stub COLMAP and with the real one. G-1 to G-7 compare the converters with openftgs-synth's ground truth, locally, where `$OPENFTGS_DATASETS/synth` exists (D-79). C-1 to C-5 are written from §3.1 and skip without the paper's datasets, apart from C-4's Gaussian-dot part, which needs only COLMAP.

## What stays unverified

The project validates on openftgs-synth, not on the paper's datasets (D-79). The converters are checked end to end by G-1 to G-7, but D-28's LLFF column order, ENeRF-Outdoor's file layout and D-59's reading of the sync offsets are only as good as the spec's assumptions; the release notes say so.

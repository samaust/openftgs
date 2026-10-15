# SPDX-License-Identifier: Apache-2.0
"""The dataset converters of openftgs.data (§1.4 "Dataset protocols", D-28, D-48, D-59 to D-61,
D-63, D-64; Q-85 to Q-101): U-27, U-37 and round trips on small generated datasets.

Videos are generated with the user's ffmpeg from its lavfi test sources, encoded losslessly
(libx264rgb, -qp 0) where pixels are compared, so ffmpeg's own PNG decode is the oracle for the
frames (U-27). Calibrations and poses are written by hand. Tests that need ffmpeg skip without
it, as U-27 says; CI installs none (#108, item 4).
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from openftgs.data import camera_from_llff, camera_from_opencv, convert_n3dv, convert_opencv_yaml
from openftgs.io import read_scene_manifest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH (D-48)",
)


def _video(
    path: Path,
    *,
    width: int = 64,
    height: int = 48,
    frames: int = 10,
    lossless: bool = True,
    smooth: bool = False,
) -> Path:
    """A lavfi testsrc clip; the lossless clips decode to exactly what ffmpeg's PNG decode
    gives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if lossless:
        codec = ["-c:v", "libx264rgb", "-qp", "0"]
    else:
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    pattern = "gradients" if smooth else "testsrc"  # gradients: smooth content (U-35's oracle)
    source = f"{pattern}=size={width}x{height}:rate=30"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-frames:v",
            str(frames),
            *codec,
            str(path),
        ],
        check=True,
    )
    return path


def _ffmpeg_pngs(video: Path, out: Path, frames: int) -> list[np.ndarray]:
    """U-27's oracle: ffmpeg's own decode of the first frames to PNG."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-frames:v",
            str(frames),
            "-fps_mode",
            "passthrough",
            str(out / "%03d.png"),
        ],
        check=True,
    )
    return [np.asarray(Image.open(out / f"{k + 1:03d}.png").convert("RGB")) for k in range(frames)]


def _llff_row(centre_x: float, *, width: int, height: int, focal: float) -> np.ndarray:
    """A poses_bounds.npy row (D-28) for a camera at (centre_x, 0, 0) looking down +z, with
    OpenCV axes x right, y down: the LLFF columns are down, right, back."""
    down, right, back = [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]
    matrix = np.array([down, right, back, [centre_x, 0.0, 0.0], [height, width, focal]]).T
    return np.concatenate([matrix.reshape(-1), [0.5, 50.0]])


def _n3dv_source(root: Path, cameras: int = 2, **video: Any) -> Path:
    width, height = video.get("width", 64), video.get("height", 48)
    rows = [_llff_row(0.25 * k, width=width, height=height, focal=50.0) for k in range(cameras)]
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "poses_bounds.npy", np.stack(rows))
    for k in range(cameras):
        _video(root / f"cam{k:02d}.mp4", **video)
    return root


@needs_ffmpeg
def test_u27_frames_come_through_in_order_and_equal_ffmpegs_decode(tmp_path: Path) -> None:
    """U-27, D-48: a 10-frame lavfi testsrc clip goes through the Neural3DV converter at resize
    1.0; its 10 frames come out in order, each equal to ffmpeg's own PNG decode."""
    src = _n3dv_source(tmp_path / "flame_steak")
    manifest_path = convert_n3dv(src, tmp_path / "out", frames=10, resize=1.0, jobs=1)
    assert manifest_path == tmp_path / "out" / "scene.json"
    manifest = read_scene_manifest(manifest_path)
    want = _ffmpeg_pngs(src / "cam01.mp4", tmp_path / "oracle", 10)
    got = [r for r in manifest.records if r.camera.name == "cam01"]
    assert [r.frame_index for r in got] == list(range(10))
    for record, frame in zip(got, want, strict=True):
        assert np.array_equal(np.asarray(Image.open(record.image_path)), frame)


@needs_ffmpeg
@pytest.mark.parametrize("which", ["ffmpeg", "ffprobe"])
def test_u27_a_wrong_ffmpeg_or_ffprobe_path_is_refused(tmp_path: Path, which: str) -> None:
    """U-27, D-48, Q-85: a wrong --ffmpeg or --ffprobe path stops the converter with a
    ValueError, which `ftgs convert` (M14) maps to exit code 2; nothing is left in OUT."""
    src = _n3dv_source(tmp_path / "scene")
    tools = {which: str(tmp_path / "missing" / which)}
    with pytest.raises(ValueError, match=which):
        convert_n3dv(src, tmp_path / "out", frames=2, resize=1.0, **tools)  # type: ignore[arg-type]
    assert not (tmp_path / "out" / "scene.json").exists()


@needs_ffmpeg
def test_n3dv_manifest_holds_the_converted_cameras_times_and_metadata(tmp_path: Path) -> None:
    """D-28, D-33, Q-94, Q-95: cameras from the LLFF rows, resized by the realized ratios;
    frame k at k/30 s; cam00 is the test view; metadata names the source, the ffmpeg run and
    the bounds."""
    src = _n3dv_source(tmp_path / "flame_steak", cameras=3, width=64, height=48, frames=4)
    manifest = read_scene_manifest(
        convert_n3dv(src, tmp_path / "out", frames=3, resize=0.5, jobs=2)
    )
    assert len(manifest.records) == 9
    by_camera: dict[str, list[float]] = {}
    for record in manifest.records:
        by_camera.setdefault(record.camera.name, []).append(record.timestamp)
        assert record.split == ("test" if record.camera.name == "cam00" else "train")
        assert (record.camera.width, record.camera.height) == (32, 24)
        assert np.asarray(Image.open(record.image_path)).shape == (24, 32, 3)
    assert by_camera == {name: [0.0, 1 / 30, 2 / 30] for name in ("cam00", "cam01", "cam02")}
    row = np.load(src / "poses_bounds.npy")[1]
    full = camera_from_llff(row, name="cam01")
    cam01 = next(r.camera for r in manifest.records if r.camera.name == "cam01")
    assert torch.equal(cam01.world_to_camera, full.world_to_camera)
    assert torch.equal(cam01.K, (full.K.double() * torch.tensor([[0.5], [0.5], [1.0]])).float())
    meta = manifest.metadata
    assert meta["source"] == "n3dv/flame_steak" and meta["converter"] == "n3dv"
    assert meta["resize"] == 0.5 and meta["fps"] == 30 and meta["frames"] == 3
    assert meta["ffmpeg"]["decoder"] == "cpu" and meta["ffmpeg"]["hwaccel"] == "none"
    assert meta["ffmpeg"]["version"].startswith("ffmpeg version")
    assert meta["ffmpeg"]["commands"]["cam00"][-1] == "-"
    assert meta["bounds"]["cam02"] == [0.5, 50.0]
    assert not (tmp_path / "out" / ".work").exists()


@needs_ffmpeg
def test_n3dv_outputs_do_not_depend_on_jobs(tmp_path: Path) -> None:
    """§2.4 rule 2: the number of parallel jobs changes no output byte."""
    src = _n3dv_source(tmp_path / "s", cameras=3, frames=3)
    one = convert_n3dv(src, tmp_path / "one", frames=3, resize=0.5, jobs=1)
    two = convert_n3dv(src, tmp_path / "two", frames=3, resize=0.5, jobs=3)
    files = sorted(p.relative_to(one.parent) for p in one.parent.rglob("*.png"))
    assert files == sorted(p.relative_to(two.parent) for p in two.parent.rglob("*.png"))
    for f in files:
        assert np.array_equal(
            np.asarray(Image.open(one.parent / f)), np.asarray(Image.open(two.parent / f))
        )
    a, b = json.loads(one.read_text()), json.loads(two.read_text())
    assert a["images"] == b["images"] and a["cameras"] == b["cameras"]


@needs_ffmpeg
@pytest.mark.parametrize("case", ["count", "size", "short", "not_empty"])
def test_n3dv_refuses_inconsistent_sources(tmp_path: Path, case: str) -> None:
    """Q-88, Q-95: a row count different from the video count, a video size different from its
    row's H and W, a video shorter than `frames`, and a non-empty OUT are ValueErrors."""
    src = _n3dv_source(tmp_path / "s", frames=3)
    frames = 3
    if case == "count":
        np.save(src / "poses_bounds.npy", np.load(src / "poses_bounds.npy")[:1])
    elif case == "size":
        _video(src / "cam01.mp4", width=32, height=48, frames=3)
    elif case == "short":
        frames = 4
    elif case == "not_empty":
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "keep.txt").write_text("x")
    with pytest.raises(ValueError):
        convert_n3dv(src, tmp_path / "out", frames=frames, resize=1.0)
    assert not (tmp_path / "out" / "scene.json").exists()


@needs_ffmpeg
@pytest.mark.gpu
def test_u27_hardware_decoding_is_within_one_level_of_cpu_decoding(tmp_path: Path) -> None:
    """U-27 (GPU), Q-92: with --hwaccel cuda (when the build lists it) the frames are within
    one 8-bit level of CPU decoding, the differing pixels counted; the manifest records the
    decoder that ran."""
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, check=True
    ).stdout.split()
    if "cuda" not in listed:
        pytest.skip("this ffmpeg build has no cuda hwaccel")
    src = _n3dv_source(tmp_path / "s", width=160, height=128, frames=10, lossless=False)
    cpu = read_scene_manifest(convert_n3dv(src, tmp_path / "cpu", frames=10, resize=1.0))
    gpu = read_scene_manifest(
        convert_n3dv(src, tmp_path / "gpu", frames=10, resize=1.0, hwaccel="cuda")
    )
    assert gpu.metadata["ffmpeg"]["decoder"] == "cuda"
    differing = 0
    for a, b in zip(cpu.records, gpu.records, strict=True):
        x = np.asarray(Image.open(a.image_path)).astype(int)
        y = np.asarray(Image.open(b.image_path)).astype(int)
        assert np.abs(x - y).max() <= 1
        differing += int((x != y).any(axis=-1).sum())
    print(f"U-27 hwaccel: {differing} differing pixels of {len(cpu.records) * 160 * 128}")


# OpenCV-style YAML converter (D-61, D-63; Q-85, Q-96).

_YAML_K = np.array([[60.0, 0.0, 31.5], [0.0, 60.0, 23.5], [0.0, 0.0, 1.0]])  # OpenCV convention
_YAML_D = np.array([[-0.12], [0.03], [0.0005], [-0.0004], [0.0]])


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(vector)
    return np.asarray(matrix)


def _write_calibration(
    root: Path,
    names: list[str],
    *,
    rot_error: float = 0.0,
    ccm: np.ndarray | None = None,
    distortion: np.ndarray = _YAML_D,
    sizes: tuple[int, int] | None = (64, 48),
) -> None:
    """intri.yml and extri.yml as OpenCV's FileStorage writes them, in the SelfCap key family
    (D-63): names, K_, D_, H_, W_, (ccm_), R_ (Rodrigues), Rot_, T_."""
    root.mkdir(parents=True, exist_ok=True)
    intri = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(str(root / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    intri.write("names", names)
    extri.write("names", names)
    for k, name in enumerate(names):
        intri.write(f"K_{name}", _YAML_K)
        intri.write(f"D_{name}", distortion)
        if sizes is not None:
            intri.write(f"W_{name}", sizes[0])
            intri.write(f"H_{name}", sizes[1])
        if ccm is not None:
            intri.write(f"ccm_{name}", ccm)
        vector = np.array([[0.0], [0.1 * k], [0.0]])
        rot = _rodrigues(vector)
        rot[0, 0] += rot_error
        extri.write(f"R_{name}", vector)
        extri.write(f"Rot_{name}", rot)
        extri.write(f"T_{name}", np.array([[0.2 * k], [0.0], [3.0]]))
    intri.release()
    extri.release()


def _yaml_source(
    root: Path, names: list[str], *, images: bool = False, frames: int = 4, **calibration: Any
) -> Path:
    _write_calibration(root, names, **calibration)
    for name in names:
        video = _video(root / "videos" / f"{name}.mp4", frames=frames, smooth=True)
        if images:
            pngs = _ffmpeg_pngs(video, root / "images" / name, frames)
            video.unlink()
            del pngs
    if images:
        (root / "videos").rmdir()
    return root


@needs_ffmpeg
@pytest.mark.parametrize("images", [False, True])
def test_opencv_yaml_undistorts_shifts_and_times_frames(tmp_path: Path, images: bool) -> None:
    """§1.4 ENeRF-Outdoor, D-61, D-63, Q-85, Q-96: frames from videos or image folders are
    undistorted with the file's K (OpenCV's convention), so they match OpenCV's undistort within
    one 8-bit level away from the border at resize 1.0; the manifest's K is the file's with
    +0.5 on cx and cy; R and T are world-to-camera; frame k at k/fps; the test camera is the
    test view."""
    src = _yaml_source(tmp_path / "actor1", ["0000", "0001"], images=images)
    manifest = read_scene_manifest(
        convert_opencv_yaml(
            src, tmp_path / "out", fps=60.0, frames=3, resize=1.0, test_camera="0001", jobs=1
        )
    )
    assert len(manifest.records) == 6
    for record in manifest.records:
        assert record.split == ("test" if record.camera.name == "0001" else "train")
        assert record.frame_index is not None
        assert record.timestamp == record.frame_index / 60.0
    record = next(r for r in manifest.records if r.camera.name == "0001" and r.frame_index == 2)
    want = camera_from_opencv(
        _YAML_K,
        _rodrigues(np.array([[0.0], [0.1], [0.0]])),
        np.array([0.2, 0.0, 3.0]),
        width=64,
        height=48,
        name="0001",
    )
    assert torch.equal(record.camera.K, want.K)
    torch.testing.assert_close(
        record.camera.world_to_camera, want.world_to_camera, rtol=0.0, atol=1e-7
    )
    frames = (
        _ffmpeg_pngs(src / "videos" / "0001.mp4", tmp_path / "o", 3)
        if not images
        else [np.asarray(Image.open(src / "images" / "0001" / f"{k:03d}.png")) for k in (1, 2, 3)]
    )
    oracle = cv2.undistort(frames[2], _YAML_K, _YAML_D)
    map_x, map_y = cv2.initUndistortRectifyMap(
        _YAML_K, _YAML_D, None, _YAML_K, (64, 48), cv2.CV_32FC1
    )
    inside = (map_x >= 1) & (map_x <= 62) & (map_y >= 1) & (map_y <= 46)
    got = np.asarray(Image.open(record.image_path)).astype(int)
    assert np.abs(got - oracle.astype(int))[inside].max() <= 1
    assert manifest.metadata["converter"] == "opencv-yaml"
    assert manifest.metadata["test_camera"] == "0001" and manifest.metadata["fps"] == 60.0
    assert ("ffmpeg" in manifest.metadata) is (not images)


@needs_ffmpeg
def test_opencv_yaml_without_a_test_camera_has_only_training_images(tmp_path: Path) -> None:
    """Q-85: without `test_camera` every image is a training image."""
    src = _yaml_source(tmp_path / "s", ["0000", "0001"], frames=2)
    manifest = read_scene_manifest(
        convert_opencv_yaml(src, tmp_path / "out", fps=30.0, frames=2, resize=0.5)
    )
    assert {r.split for r in manifest.records} == {"train"}
    assert {(r.camera.width, r.camera.height) for r in manifest.records} == {(32, 24)}


@needs_ffmpeg
@pytest.mark.parametrize("case", ["rot", "ccm", "size", "camera", "frames"])
def test_opencv_yaml_refuses_inconsistent_calibrations(tmp_path: Path, case: str) -> None:
    """D-63, Q-96: Rot and Rodrigues(R) differing by more than 1e-6, a ccm that isn't the
    identity, W_/H_ differing from the frames, an unknown test camera and too few frames are
    ValueErrors."""
    kwargs: dict[str, Any] = {}
    frames, test = 2, None
    if case == "rot":
        kwargs["rot_error"] = 1e-5
    elif case == "ccm":
        kwargs["ccm"] = np.diag([1.0, 1.0, 1.01])
    elif case == "size":
        kwargs["sizes"] = (64, 50)
    elif case == "camera":
        test = "0009"
    elif case == "frames":
        frames = 5
    src = _yaml_source(tmp_path / "s", ["0000", "0001"], frames=4, **kwargs)
    with pytest.raises(ValueError):
        convert_opencv_yaml(
            src, tmp_path / "out", fps=30.0, frames=frames, resize=1.0, test_camera=test
        )
    assert not (tmp_path / "out" / "scene.json").exists()


@needs_ffmpeg
def test_opencv_yaml_accepts_an_identity_ccm(tmp_path: Path) -> None:
    """D-63: an identity ccm is accepted."""
    src = _yaml_source(tmp_path / "s", ["0000", "0001"], frames=2, ccm=np.eye(3))
    assert convert_opencv_yaml(src, tmp_path / "out", fps=30.0, frames=1, resize=1.0).exists()

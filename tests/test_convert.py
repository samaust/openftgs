# SPDX-License-Identifier: Apache-2.0
"""The dataset converters of openftgs.data (§1.4 "Dataset protocols", D-28, D-48, D-59 to D-61,
D-63, D-64, D-80, D-81; Q-86 to Q-91, Q-93 to Q-101, Q-169): U-27, U-37 and round trips on
small generated datasets.

Videos are generated with the user's ffmpeg from its lavfi test sources, encoded losslessly
(libx264rgb, -qp 0) where pixels are compared, so ffmpeg's own PNG decode is the oracle for the
frames (U-27). Calibrations and poses are written by hand. Tests that need ffmpeg skip without
it, as U-27 says; CI installs none (#108, item 4).
"""

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pytest
import torch
from builders import (
    SELFCAP_OFFSETS,
    YAML_D,
    YAML_K,
    ffmpeg_video,
    n3dv_source,
    rodrigues,
    selfcap_archives,
    stub_colmap,
    write_opencv_calibration,
)
from PIL import Image

from openftgs import _stages
from openftgs.data import (
    camera_from_llff,
    camera_from_opencv,
    convert_n3dv,
    convert_opencv_yaml,
    convert_selfcap,
)
from openftgs.io import read_scene_manifest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH (D-48)",
)


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


@needs_ffmpeg
def test_u27_frames_come_through_in_order_and_equal_ffmpegs_decode(tmp_path: Path) -> None:
    """U-27, D-48: a 10-frame lavfi testsrc clip goes through the Neural3DV converter at resize
    1.0; its 10 frames come out in order, each equal to ffmpeg's own PNG decode."""
    src = n3dv_source(tmp_path / "flame_steak")
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
    """U-27, D-48, D-80: a wrong --ffmpeg or --ffprobe path stops the converter with a
    ValueError, which `ftgs convert` (M14) maps to exit code 2; nothing is left in OUT."""
    src = n3dv_source(tmp_path / "scene")
    tools = {which: str(tmp_path / "missing" / which)}
    with pytest.raises(ValueError, match=which):
        convert_n3dv(src, tmp_path / "out", frames=2, resize=1.0, **tools)  # type: ignore[arg-type]
    assert not (tmp_path / "out" / "scene.json").exists()


@needs_ffmpeg
def test_n3dv_manifest_holds_the_converted_cameras_times_and_metadata(tmp_path: Path) -> None:
    """D-28, D-33, Q-94, Q-95: cameras from the LLFF rows, resized by the realized ratios;
    frame k at k/30 s; cam00 is the test view; metadata names the source, the ffmpeg run and
    the bounds."""
    src = n3dv_source(tmp_path / "flame_steak", cameras=3, width=64, height=48, frames=4)
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
@pytest.mark.parametrize("typed", ["data/kitchen", "data/../data/kitchen"])
def test_n3dv_records_a_relative_srcs_videos_by_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: str
) -> None:
    """D-48, Q-169: a relative SRC is made absolute, normalized lexically, before the ffmpeg
    commands are built, so the full command the manifest records reads each video by its
    absolute path, whatever directory the conversion ran in."""
    n3dv_source(tmp_path / "data" / "kitchen", frames=3)
    monkeypatch.chdir(tmp_path)
    manifest = read_scene_manifest(
        convert_n3dv(Path(typed), tmp_path / "out", frames=2, resize=0.5, jobs=1)
    )
    commands = manifest.metadata["ffmpeg"]["commands"]
    for camera in ("cam00", "cam01"):
        command = commands[camera]
        video = Path.cwd() / "data" / "kitchen" / f"{camera}.mp4"
        assert command[command.index("-i") + 1] == str(video)
    assert manifest.metadata["source"] == "n3dv/kitchen"


@needs_ffmpeg
def test_n3dv_source_names_the_folder_when_src_is_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-62, Q-169: with SRC `.`, `metadata.source` is "n3dv/<the folder's name>", the form of
    §2.2's example manifest, not "n3dv/"."""
    monkeypatch.chdir(n3dv_source(tmp_path / "flame_steak", frames=2))
    manifest = read_scene_manifest(
        convert_n3dv(Path("."), tmp_path / "out", frames=1, resize=0.5, jobs=1)
    )
    assert manifest.metadata["source"] == "n3dv/flame_steak"


@needs_ffmpeg
def test_n3dv_source_keeps_the_name_of_a_symlinked_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-169: a relative SRC that names a symlink is made absolute without resolving the link,
    so the name it was given stays in `metadata.source` and in the recorded `-i` paths."""
    n3dv_source(tmp_path / "renders" / "take3", frames=2)
    (tmp_path / "flame_steak").symlink_to(tmp_path / "renders" / "take3", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    manifest = read_scene_manifest(
        convert_n3dv(Path("flame_steak"), tmp_path / "out", frames=1, resize=0.5)
    )
    assert manifest.metadata["source"] == "n3dv/flame_steak"
    command = manifest.metadata["ffmpeg"]["commands"]["cam01"]
    assert command[command.index("-i") + 1] == str(Path.cwd() / "flame_steak" / "cam01.mp4")


@needs_ffmpeg
def test_n3dv_outputs_do_not_depend_on_jobs(tmp_path: Path) -> None:
    """§2.4 rule 2: the number of parallel jobs changes no output byte."""
    src = n3dv_source(tmp_path / "s", cameras=3, frames=3)
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


def _process_pool_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The worker counts of every process pool started from here on, which still runs: a spy
    on the standard library's executor, not on openftgs."""
    real = concurrent.futures.ProcessPoolExecutor
    sizes: list[int] = []

    def spy(max_workers: int, *args: Any, **kwargs: Any) -> concurrent.futures.Executor:
        sizes.append(max_workers)
        return real(max_workers, *args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", spy)
    return sizes


@needs_ffmpeg
@pytest.mark.parametrize(
    ("variables", "budget"),
    [
        ({"OMP_NUM_THREADS": "2"}, 2),
        ({"MKL_NUM_THREADS": "2"}, 2),
        ({"OMP_NUM_THREADS": "5", "MKL_NUM_THREADS": "3"}, 3),
        ({}, None),
    ],
    ids=["omp", "mkl", "both", "unset"],
)
def test_the_encoding_pool_stays_within_the_thread_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], budget: int | None
) -> None:
    """Q-215, §2.4: the PNG pool starts at most the thread budget's processes, the smallest
    positive count of OMP_NUM_THREADS and MKL_NUM_THREADS, as PyTorch takes them; with neither
    set one per CPU, and never more than the frames. (Values that aren't positive counts are
    checked on the cache: ffprobe's own OpenMP runtime rejects them.)"""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    sizes = _process_pool_sizes(monkeypatch)
    src = n3dv_source(tmp_path / "s", cameras=3, frames=3)
    convert_n3dv(src, tmp_path / "out", frames=3, resize=0.5, jobs=2)
    frames = 9
    assert sizes == [min(budget or os.cpu_count() or 1, frames)]


@needs_ffmpeg
def test_n3dv_decoding_and_encoding_are_stages_per_camera(tmp_path: Path) -> None:
    """§3.4 B-8 (PRD #220): under a wall clock, every frame read from ffmpeg is the stage
    "convert.decode" in its camera's thread, and every frame processed and written in the pool
    is the stage "convert.encode", its duration measured in the pool's process; both are keyed
    by camera."""
    src = n3dv_source(tmp_path / "s", cameras=3, frames=3)
    with _stages.Clock("wall") as clock:
        convert_n3dv(src, tmp_path / "out", frames=3, resize=0.5, jobs=2)
    timings = clock.read()
    cameras = ["cam00", "cam01", "cam02"]
    for name in ("convert.decode", "convert.encode"):
        mine = [t for t in timings if t.name == name]
        assert sorted(str(t.key) for t in mine) == sorted(cameras * 3), name
        assert all(t.seconds > 0.0 for t in mine), name
    for camera in cameras:
        threads = {t.thread for t in timings if t.name == "convert.decode" and t.key == camera}
        assert len(threads) == 1 and threads != {threading.current_thread().name}
    assert {t.name for t in timings} == {"convert.decode", "convert.encode"}


@needs_ffmpeg
@pytest.mark.parametrize("case", ["count", "size", "short", "not_empty"])
def test_n3dv_refuses_inconsistent_sources(tmp_path: Path, case: str) -> None:
    """Q-88, Q-95: a row count different from the video count, a video size different from its
    row's H and W, a video shorter than `frames`, and a non-empty OUT are ValueErrors."""
    src = n3dv_source(tmp_path / "s", frames=3)
    frames = 3
    if case == "count":
        np.save(src / "poses_bounds.npy", np.load(src / "poses_bounds.npy")[:1])
    elif case == "size":
        ffmpeg_video(src / "cam01.mp4", width=32, height=48, frames=3)
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
    """U-27 (GPU), D-81: with --hwaccel cuda (when the build lists it) the frames are within
    one 8-bit level of CPU decoding, the differing pixels counted; the manifest records the
    decoder that ran. The clip is yuv420p H.264 rather than U-27's lossless RGB one, which
    CUDA can't decode (D-81 refuses gbrp)."""
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, check=True
    ).stdout.split()
    if "cuda" not in listed:
        pytest.skip("this ffmpeg build has no cuda hwaccel")
    src = n3dv_source(tmp_path / "s", width=160, height=128, frames=10, lossless=False)
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


# OpenCV-style YAML converter (D-61, D-63, D-80; Q-96).


def _yaml_source(
    root: Path, names: list[str], *, images: bool = False, frames: int = 4, **calibration: Any
) -> Path:
    write_opencv_calibration(root, names, **calibration)
    for name in names:
        video = ffmpeg_video(root / "videos" / f"{name}.mp4", frames=frames, smooth=True)
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
    """§1.4 ENeRF-Outdoor, D-61, D-63, D-80, Q-96: frames from videos or image folders are
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
        YAML_K,
        rodrigues(np.array([[0.0], [0.1], [0.0]])),
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
    oracle = cv2.undistort(frames[2], YAML_K, YAML_D)
    map_x, map_y = cv2.initUndistortRectifyMap(YAML_K, YAML_D, None, YAML_K, (64, 48), cv2.CV_32FC1)
    inside = (map_x >= 1) & (map_x <= 62) & (map_y >= 1) & (map_y <= 46)
    got = np.asarray(Image.open(record.image_path)).astype(int)
    assert np.abs(got - oracle.astype(int))[inside].max() <= 1
    assert manifest.metadata["converter"] == "opencv-yaml"
    assert manifest.metadata["test_camera"] == "0001" and manifest.metadata["fps"] == 60.0
    assert ("ffmpeg" in manifest.metadata) is (not images)


@needs_ffmpeg
def test_opencv_yaml_without_a_test_camera_has_only_training_images(tmp_path: Path) -> None:
    """D-80: without `test_camera` every image is a training image."""
    src = _yaml_source(tmp_path / "s", ["0000", "0001"], frames=2)
    manifest = read_scene_manifest(
        convert_opencv_yaml(src, tmp_path / "out", fps=30.0, frames=2, resize=0.5)
    )
    assert {r.split for r in manifest.records} == {"train"}
    assert {(r.camera.width, r.camera.height) for r in manifest.records} == {(32, 24)}


@needs_ffmpeg
def test_opencv_yaml_records_a_relative_srcs_videos_by_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§1.4 ("the same ffmpeg call"), Q-169: as for Neural3DV, a relative SRC is made absolute
    before the ffmpeg commands are built, so the recorded commands read the videos by absolute
    path, and `metadata.source` names SRC's folder."""
    _yaml_source(tmp_path / "data" / "actor1", ["0000", "0001"], frames=2)
    monkeypatch.chdir(tmp_path / "data")
    manifest = read_scene_manifest(
        convert_opencv_yaml(Path("actor1"), tmp_path / "out", fps=60.0, frames=1, resize=0.5)
    )
    for camera in ("0000", "0001"):
        command = manifest.metadata["ffmpeg"]["commands"][camera]
        video = Path.cwd() / "actor1" / "videos" / f"{camera}.mp4"
        assert command[command.index("-i") + 1] == str(video)
    assert manifest.metadata["source"] == "opencv-yaml/actor1"


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


# U-37 · SelfCap converter logic (D-59, D-60, D-63; Q-86, Q-87, Q-97, Q-101).


def _frames_at(video: Path, out: Path, indices: list[int]) -> list[np.ndarray]:
    """ffmpeg's own decode of the given decode indices, as the oracle for window selection."""
    all_frames = _ffmpeg_pngs(video, out, max(indices) + 1)
    return [all_frames[n] for n in indices]


@needs_ffmpeg
@pytest.mark.parametrize("undistort", ["internal", "colmap"])
def test_u37_selects_the_window_times_frames_and_holds_out_the_test_camera(
    tmp_path: Path, undistort: Literal["colmap", "internal"]
) -> None:
    """U-37, D-59, D-60, Q-86, Q-97: window 4 ≤ n < 8 and test camera 0002, passed explicitly
    for a scene outside the card's table: frames 4–7 only, in order (each the decode index's
    frame, resized by 0.5 as INTER_AREA would); τ = n/60 minus the camera's offset; the test
    camera excluded from training; with COLMAP, its model gets FULL_OPENCV with cx, cy + 0.5,
    and the read-back PINHOLE camera is used without a shift."""
    src = selfcap_archives(tmp_path, "mini")
    colmap, log = stub_colmap(tmp_path)
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="mini",
            window=(4, 8),
            test_camera="0002",
            undistort=undistort,
            colmap=str(colmap),
            jobs=2,
        )
    )
    assert len(manifest.records) == 12
    for record in manifest.records:
        assert record.frame_index is not None and record.frame_index in range(4, 8)
        assert record.timestamp == record.frame_index / 60 - SELFCAP_OFFSETS[record.camera.name]
        assert record.split == ("test" if record.camera.name == "0002" else "train")
        assert (record.camera.width, record.camera.height) == (32, 24)
    staging_video = tmp_path / "staging-mini" / "mini-release" / "videos" / "0001.mp4"
    expected = _frames_at(staging_video, tmp_path / "oracle", [4, 5, 6, 7])
    got = [r for r in manifest.records if r.camera.name == "0001"]
    assert [r.frame_index for r in got] == [4, 5, 6, 7]
    for record, frame in zip(got, expected, strict=True):
        want = cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA).astype(int)
        assert np.abs(np.asarray(Image.open(record.image_path)).astype(int) - want).max() <= 1
    full = camera_from_opencv(
        YAML_K,
        rodrigues(np.array([[0.0], [0.1], [0.0]])),
        np.array([0.2, 0.0, 3.0]),
        width=64,
        height=48,
    )
    camera = got[0].camera
    torch.testing.assert_close(
        camera.K, (full.K.double() * torch.tensor([[0.5], [0.5], [1.0]])).float()
    )
    meta = manifest.metadata
    assert meta["undistort"] == undistort and meta["sync"] is True
    assert meta["window"] == [4, 8] and meta["test_camera"] == "0002"
    archive = src / "mini-release.tar.gz"
    assert meta["archives"] == {archive.name: hashlib.sha256(archive.read_bytes()).hexdigest()}
    if undistort == "colmap":
        calls = [json.loads(line) for line in (log / "calls.jsonl").read_text().splitlines()]
        assert calls[0][0] == "image_undistorter" and calls[1][0] == "model_converter"
        assert calls[0][calls[0].index("--blank_pixels") + 1] == "0"
        rows = [
            line.split()
            for line in (log / "input_model" / "cameras.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ]
        assert {row[1] for row in rows} == {"FULL_OPENCV"} and len(rows) == 3
        params = [float(v) for v in rows[0][4:]]
        assert params[:4] == [60.0, 60.0, 32.0, 24.0]  # fx, fy, cx + 0.5, cy + 0.5
        assert len(params) == 12 and meta["colmap"]["version"].startswith("COLMAP stub")
    else:
        assert not log.exists() and "colmap" not in meta
    assert not (tmp_path / "out" / ".work").exists()


@needs_ffmpeg
def test_u37_no_sync_uses_the_frame_index_time(tmp_path: Path) -> None:
    """U-37, D-59: with use_sync=False, τ = n/60."""
    src = selfcap_archives(tmp_path, "mini")
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="mini",
            window=(4, 8),
            test_camera="0002",
            undistort="internal",
            use_sync=False,
        )
    )
    assert all(r.timestamp == r.frame_index / 60 for r in manifest.records)  # type: ignore[operator]
    assert manifest.metadata["sync"] is False


@needs_ffmpeg
def test_selfcap_dance_takes_its_calibration_from_hair_calib(tmp_path: Path) -> None:
    """§1.4, D-63, Q-86: dance1 reads hair-release for its videos and hair-calib for its
    calibration and sync offsets, and records both archives' SHA-256."""
    src = selfcap_archives(tmp_path, "hair", calib_archive=True)
    shutil.move(tmp_path / "src" / "hair-calib.tar.gz", src / "hair-calib.tar.gz")
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="dance1",
            window=(4, 8),
            test_camera="0002",
            undistort="internal",
        )
    )
    assert set(manifest.metadata["archives"]) == {"hair-release.tar.gz", "hair-calib.tar.gz"}
    assert manifest.metadata["calibration"] == "hair-calib.tar.gz"
    assert manifest.metadata["source"] == "selfcap/dance1"
    assert all(
        r.timestamp == r.frame_index / 60 - SELFCAP_OFFSETS[r.camera.name]  # type: ignore[operator]
        for r in manifest.records
    )
    assert "fastest_10" not in manifest.metadata  # a 4-frame window holds no 10 frames


@needs_ffmpeg
@pytest.mark.parametrize("case", ["no_colmap", "rot", "ccm", "window", "unknown_scene", "sync"])
def test_u37_refusals(tmp_path: Path, case: str) -> None:
    """U-37, D-60, D-63, Q-97, Q-101: without COLMAP (unless internal), when Rot disagrees with
    R, when a ccm is not the identity, a window beyond the video, a scene outside the table
    without window and test camera, and a sync.json missing a camera are ValueErrors (exit code
    2 at M14)."""
    kwargs: dict[str, Any] = {}
    if case == "rot":
        kwargs["rot_error"] = 1e-5
    elif case == "ccm":
        kwargs["ccm"] = np.diag([1.0, 2.0, 1.0])
    src = selfcap_archives(tmp_path, "mini", **kwargs)
    call: dict[str, Any] = {
        "scene": "mini",
        "window": (4, 8),
        "test_camera": "0002",
        "undistort": "internal",
    }
    if case == "no_colmap":
        call["undistort"], call["colmap"] = "colmap", str(tmp_path / "no" / "colmap")
    elif case == "window":
        call["window"] = (8, 13)
    elif case == "unknown_scene":
        call = {"scene": "mini"}
    elif case == "sync":
        staging = tmp_path / "staging-mini" / "calib"
        (staging / "sync.json").write_text(json.dumps({"0000": 0.0, "0001": 0.0}))
        with tarfile.open(src / "mini-release.tar.gz", "w:gz") as tar:
            tar.add(
                tmp_path / "staging-mini" / "mini-release" / "videos", arcname="mini-release/videos"
            )
            tar.add(staging, arcname="mini-release/optimized")
    with pytest.raises(ValueError):
        convert_selfcap(src, tmp_path / "out", **call)
    assert not (tmp_path / "out" / "scene.json").exists()


@needs_ffmpeg
@pytest.mark.parametrize("damage", ["crc", "truncated"])
def test_selfcap_records_a_bad_gzip_trailer_as_a_warning(tmp_path: Path, damage: str) -> None:
    """§1.4 Integrity, Q-86: an archive whose gzip trailer fails (a bad CRC, as the hosted corgi
    archive has, or a truncated trailer) after every member came out whole converts with a
    warning in the manifest; the hash is of the bytes as they are."""
    src = selfcap_archives(tmp_path, "mini")
    archive = src / "mini-release.tar.gz"
    data = bytearray(archive.read_bytes())
    if damage == "crc":
        data[-8] ^= 0xFF  # the CRC-32 of the gzip trailer
    else:
        del data[-4:]  # a truncated trailer: the size field cut off
    archive.write_bytes(bytes(data))
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="mini",
            window=(4, 8),
            test_camera="0002",
            undistort="internal",
        )
    )
    assert manifest.metadata["archives"]["mini-release.tar.gz"] == hashlib.sha256(data).hexdigest()
    assert len(manifest.metadata["warnings"]) == 1
    if damage == "crc":
        assert "CRC" in manifest.metadata["warnings"][0]


@needs_ffmpeg
def test_u37_masks_with_their_sidecar(tmp_path: Path) -> None:
    """D-64, Q-87: one mask per test image, DIR/<n:06d>.png at the converted size, with
    masks.json, gives each test image a mask and the two metadata keys; a missing mask is
    refused."""
    src = selfcap_archives(tmp_path, "mini")
    masks = tmp_path / "masks"
    masks.mkdir()
    for n in range(4, 8):
        Image.fromarray(np.full((24, 32), 255 * (n % 2), dtype=np.uint8), "L").save(
            masks / f"{n:06d}.png"
        )
    (masks / "masks.json").write_text(
        json.dumps({"mask_source": "test fixture", "masks_approximate": True})
    )
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="mini",
            window=(4, 8),
            test_camera="0002",
            undistort="internal",
            masks=masks,
        )
    )
    assert manifest.metadata["mask_source"] == "test fixture"
    assert manifest.metadata["masks_approximate"] is True
    for record, mask in zip(manifest.records, manifest.masks, strict=True):
        assert (mask is not None) is (record.split == "test")
    (masks / "000005.png").unlink()
    with pytest.raises(ValueError, match="000005"):
        convert_selfcap(
            src,
            tmp_path / "out2",
            scene="mini",
            window=(4, 8),
            test_camera="0002",
            undistort="internal",
            masks=masks,
        )


@needs_ffmpeg
@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs COLMAP on PATH (D-60)")
def test_u37_with_the_real_colmap(tmp_path: Path) -> None:
    """U-37 and D-60 against the user's COLMAP: with distortion, image_undistorter changes the
    size and keeps the focal lengths, and the converter uses the read-back PINHOLE camera, so the
    manifest's fx is the calibration's times the realized ratio; the run is recorded."""
    distortion = np.array([[-0.12], [0.03], [0.0005], [-0.0004], [0.0]])
    src = selfcap_archives(tmp_path, "mini", distortion=distortion)
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="mini",
            window=(4, 6),
            test_camera="0002",
            undistort="colmap",
        )
    )
    assert len(manifest.records) == 6
    for record in manifest.records:
        # COLMAP changes the size and keeps fx = 60; resizing by ½ rounds half to even (Q-89),
        # so COLMAP's width was 2W′ − 1, 2W′ or 2W′ + 1 and fx is 60·W′ over it.
        width = record.camera.width
        fx = record.camera.K[0, 0].item()
        assert any(
            fx == pytest.approx(60.0 * width / w, abs=1e-4)
            for w in (2 * width - 1, 2 * width, 2 * width + 1)
        )
        assert np.asarray(Image.open(record.image_path)).shape == (
            record.camera.height,
            record.camera.width,
            3,
        )
    assert manifest.metadata["colmap"]["version"].startswith("COLMAP")


@needs_ffmpeg
def test_u37_internal_undistortion_needs_no_colmap(tmp_path: Path) -> None:
    """U-37: with undistort='internal' a missing COLMAP is no error."""
    src = selfcap_archives(tmp_path, "mini")
    path = convert_selfcap(
        src,
        tmp_path / "out",
        scene="mini",
        window=(4, 6),
        test_camera="0002",
        undistort="internal",
        colmap=str(tmp_path / "no" / "colmap"),
    )
    assert "colmap" not in read_scene_manifest(path).metadata


@needs_ffmpeg
def test_opencv_yaml_outputs_do_not_depend_on_jobs(tmp_path: Path) -> None:
    """§2.4 rule 2: the number of parallel jobs changes no output byte of the YAML converter."""
    src = _yaml_source(tmp_path / "s", ["0000", "0001", "0002"], frames=3)
    one = convert_opencv_yaml(src, tmp_path / "one", fps=30.0, frames=3, resize=0.5, jobs=1)
    two = convert_opencv_yaml(src, tmp_path / "two", fps=30.0, frames=3, resize=0.5, jobs=3)
    files = sorted(p.relative_to(one.parent) for p in one.parent.rglob("*.png"))
    assert files and files == sorted(p.relative_to(two.parent) for p in two.parent.rglob("*.png"))
    for f in files:
        assert (one.parent / f).read_bytes() == (two.parent / f).read_bytes()
    assert json.loads(one.read_text())["images"] == json.loads(two.read_text())["images"]


@needs_ffmpeg
@pytest.mark.parametrize("start", ["absent", "empty"])
def test_a_failed_conversion_leaves_out_as_it_found_it(tmp_path: Path, start: str) -> None:
    """Q-88 (owner answer): a conversion that fails after writing frames removes everything it
    wrote, so OUT is absent or empty again and the same command can run again. A mask of the
    wrong size fails only once the frames exist."""
    src = selfcap_archives(tmp_path, "mini")
    masks = tmp_path / "masks"
    masks.mkdir()
    for n in range(4, 6):
        Image.fromarray(np.zeros((10, 10), dtype=np.uint8), "L").save(masks / f"{n:06d}.png")
    (masks / "masks.json").write_text(
        json.dumps({"mask_source": "test", "masks_approximate": False})
    )
    out = tmp_path / "out"
    if start == "empty":
        out.mkdir()
    call: dict[str, Any] = {
        "scene": "mini",
        "window": (4, 6),
        "test_camera": "0002",
        "undistort": "internal",
    }
    with pytest.raises(ValueError, match="mask size"):
        convert_selfcap(src, out, masks=masks, **call)
    if start == "absent":
        assert not out.exists()
    else:
        assert out.is_dir() and list(out.iterdir()) == []
    assert convert_selfcap(src, out, **call).is_file()  # the rerun works


@needs_ffmpeg
def test_selfcap_resize_overrides_the_default_ratio(tmp_path: Path) -> None:
    """D-80: `resize` sets the ratio; a scene outside the card's table is
    otherwise resized by 0.5. At 1.0 the 64×48 videos keep their size and the manifest records
    the ratio."""
    src = selfcap_archives(tmp_path, "mini")
    call: dict[str, Any] = {
        "scene": "mini",
        "window": (4, 6),
        "test_camera": "0002",
        "undistort": "internal",
    }
    full = read_scene_manifest(convert_selfcap(src, tmp_path / "full", resize=1.0, **call))
    half = read_scene_manifest(convert_selfcap(src, tmp_path / "half", **call))
    assert {(r.camera.width, r.camera.height) for r in full.records} == {(64, 48)}
    assert {(r.camera.width, r.camera.height) for r in half.records} == {(32, 24)}
    assert (full.metadata["resize"], half.metadata["resize"]) == (1.0, 0.5)
    with pytest.raises(ValueError):
        convert_selfcap(src, tmp_path / "bad", resize=0.0, **call)

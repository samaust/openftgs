# SPDX-License-Identifier: Apache-2.0
"""G-1 to G-7: the converters and the frame cache against openftgs-synth's ground truth (§1.4
"Validation data", §3.1, D-79; Q-100).

openftgs-synth (CC0-1.0, version 0.1.x) is the owner's rendered dataset, at
$OPENFTGS_DATASETS/synth; these checks run only where it is present, never in CI (#108). Its
ground truth is independent of this package: exact cameras and per-frame times in
`ground_truth/<scene>/cameras.json`, undistorted frames, masks. Its lossless videos decode
bit-exactly to the ground-truth frames, so frame comparisons use them; its released videos and
SelfCap archives are H.264. OpenCV (a test-only oracle) gives the floor of G-2 and G-4: what its
own `undistort` achieves on the same frames.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from openftgs.data import (
    FrameCache,
    MultiViewVideo,
    Prefetcher,
    convert_n3dv,
    convert_opencv_yaml,
    convert_selfcap,
    resize_area,
    resized_size,
)
from openftgs.io import read_scene_manifest
from openftgs.types import SceneManifest

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="needs ffmpeg and ffprobe on PATH (D-48)",
    ),
]


def _root() -> Path:
    """$OPENFTGS_DATASETS/synth, version 0.1.x, or a skip naming the folder (Q-100)."""
    root = Path(os.environ.get("OPENFTGS_DATASETS", "")) / "synth"
    if not (root / "VERSION").is_file():
        pytest.skip(f"openftgs-synth is not at {root}")
    version = (root / "VERSION").read_text().strip()
    if not version.startswith("0.1."):
        pytest.skip(f"openftgs-synth {version} is not 0.1.x")
    return root


@cache
def _checksums(root: Path) -> dict[str, str]:
    lines = (root / "SHA256SUMS").read_text().splitlines()
    return {line[66:]: line[:64] for line in lines if len(line) > 66}


def _verified(root: Path, *relative: str) -> None:
    """The files a check reads match the dataset's SHA256SUMS (D-79)."""
    sums = _checksums(root)
    for name in relative:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert sums[name] == digest, f"{name} doesn't match SHA256SUMS"


@cache
def _verified_folder(root: Path, folder: str) -> None:
    """Every file under `folder` matches SHA256SUMS."""
    _verified(
        root, *(str(p.relative_to(root)) for p in sorted((root / folder).rglob("*")) if p.is_file())
    )


def _ground_truth(root: Path, scene: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / "ground_truth" / scene / "cameras.json").read_text())
    return data


def _gt_camera(truth: dict[str, Any], layout: str, name: str) -> dict[str, Any]:
    return next(c for c in truth["cameras"] if c["layout_names"].get(layout) == name)


def _pinhole(root: Path, scene: str, camera: str, n: int) -> np.ndarray:
    path = root / "ground_truth" / scene / "pinhole" / camera / f"{n:06d}.png"
    return np.asarray(Image.open(path).convert("RGB"))


def _decode(video: Path, frames: int) -> list[np.ndarray]:
    """The first frames of a video, as ffmpeg decodes them to rgb24 (the distorted inputs)."""
    probe = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .split(",")
    )
    width, height = int(probe[0]), int(probe[1])
    data = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return list(np.frombuffer(data, np.uint8).reshape(frames, height, width, 3))


def _image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.float64)


def _records(manifest: SceneManifest, name: str) -> list[Any]:
    return [r for r in manifest.records if r.camera.name == name]


def _mean_error(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(diff[valid].mean() if valid is not None else diff.mean())


def _inside(K: np.ndarray, dist: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixels whose distorted source lies at least a pixel inside the image (U-35's interior)."""
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_32FC1)
    return (map_x >= 1) & (map_x <= width - 2) & (map_y >= 1) & (map_y <= height - 2)


# G-1 · Neural3DV converter (D-28, D-33, D-48).


@pytest.mark.parametrize(
    ("scene", "frames", "resize"),
    [("tiny", 12, 1.0), ("tiny", 12, 0.5), ("kitchen", 2, 1.0), ("kitchen", 2, 0.5)],
)
def test_g1_n3dv_matches_the_ground_truth(
    tmp_path: Path, scene: str, frames: int, resize: float
) -> None:
    """G-1: from the lossless videos, K and world-to-camera equal the ground truth's
    continuous-convention K (scaled by the realized ratios) and pose to float32 rounding; τ =
    k/30; cam00 is the test view; at ratio 1.0 frames are bitwise the ground truth's, at 0.5
    its area resizing."""
    root = _root()
    src = root / "lossless" / "n3dv" / scene
    _verified_folder(root, f"lossless/n3dv/{scene}")
    manifest = read_scene_manifest(
        convert_n3dv(src, tmp_path / "out", frames=frames, resize=resize, jobs=4)
    )
    truth = _ground_truth(root, scene)
    names = sorted({r.camera.name for r in manifest.records})
    for name in names:
        gt = _gt_camera(truth, "n3dv", name)
        records = _records(manifest, name)
        assert [r.frame_index for r in records] == list(range(frames))
        for record in records:
            assert record.timestamp == record.frame_index / 30
            assert record.split == ("test" if name == "cam00" else "train")
        camera = records[0].camera
        width, height = resized_size(gt["width"], gt["height"], resize)
        scale = torch.tensor(
            [[width / gt["width"]], [height / gt["height"]], [1.0]], dtype=torch.float64
        )
        K = (torch.tensor(gt["K_continuous"], dtype=torch.float64) * scale).float()
        torch.testing.assert_close(camera.K, K, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(
            camera.world_to_camera,
            torch.tensor(gt["world_to_camera"], dtype=torch.float64).float(),
            rtol=0.0,
            atol=1e-6,
        )
        for record in records:
            truth_frame = _pinhole(root, scene, gt["name"], record.frame_index)
            got = np.asarray(Image.open(record.image_path))
            want = truth_frame if resize == 1.0 else resize_area(truth_frame, width, height)
            assert np.array_equal(got, want)


def test_g1_n3dv_h264_error_is_recorded(tmp_path: Path) -> None:
    """G-1 (calibrated): the mean absolute difference between frames converted from the H.264
    videos and the ground truth, recorded for the Calibration data table."""
    root = _root()
    _verified_folder(root, "n3dv/tiny")
    manifest = read_scene_manifest(
        convert_n3dv(root / "n3dv" / "tiny", tmp_path / "out", frames=12, resize=1.0)
    )
    truth = _ground_truth(root, "tiny")
    errors = []
    for record in manifest.records:
        gt = _gt_camera(truth, "n3dv", record.camera.name)
        errors.append(
            _mean_error(
                _image(record.image_path),
                _pinhole(root, "tiny", gt["name"], int(record.frame_index or 0)),
            )
        )
    mean = float(np.mean(errors))
    print(f"G-1 H.264 mean absolute difference on tiny: {mean:.3f} levels")
    assert np.isfinite(mean)


# G-2 · OpenCV-YAML converter (D-61, D-63).


def _undistortion_check(
    got: np.ndarray, distorted: np.ndarray, truth: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> tuple[float, float, float]:
    """The converted frame's mean error to the ground truth inside U-35's interior, OpenCV's
    floor on the same frame, and the error against the ground truth with its principal point
    moved by 0.5 px, the ground truth translated by half a pixel on both axes (what a pixel-
    centre convention error would give)."""
    height, width = truth.shape[:2]
    inside = _inside(K, dist, width, height)
    inside[:2], inside[-2:], inside[:, :2], inside[:, -2:] = False, False, False, False
    ours = _mean_error(got, truth, inside)
    floor = _mean_error(cv2.undistort(distorted, K, dist), truth, inside)
    shifted = _mean_error(got, _translate(truth, 0.5, 0.5, (width, height))[0], inside)
    return ours, floor, shifted


@pytest.mark.parametrize(
    ("scene", "frames"), [("tiny", 12), ("tiny_d8", 12), ("tiny_d12", 12), ("courtyard", 2)]
)
def test_g2_opencv_yaml_matches_the_ground_truth(tmp_path: Path, scene: str, frames: int) -> None:
    """G-2: cameras equal the ground truth; mean absolute difference to the ground-truth frames
    at most OpenCV's undistort's plus 0.25 of an 8-bit level; at least 1.2× worse with the
    principal point moved by 0.5 px; courtyard's identity ccm is accepted."""
    root = _root()
    src = root / "lossless" / "enerf_outdoor" / scene
    _verified_folder(root, f"lossless/enerf_outdoor/{scene}")
    manifest = read_scene_manifest(
        convert_opencv_yaml(src, tmp_path / "out", fps=30.0, frames=frames, resize=1.0, jobs=4)
    )
    truth = _ground_truth(root, scene)
    names = sorted({r.camera.name for r in manifest.records})
    for name in names:  # every camera; frames for the first three
        gt = _gt_camera(truth, "enerf_outdoor", name)
        record = _records(manifest, name)[-1]
        torch.testing.assert_close(
            record.camera.K,
            torch.tensor(gt["K_continuous"], dtype=torch.float64).float(),
            rtol=1e-6,
            atol=0.0,
        )
        torch.testing.assert_close(
            record.camera.world_to_camera,
            torch.tensor(gt["world_to_camera"], dtype=torch.float64).float(),
            rtol=0.0,
            atol=1e-6,
        )
        if name not in names[:3]:
            continue
        K, dist = np.array(gt["K_opencv"]), np.array(gt["distortion"])
        distorted = _decode(src / "videos" / f"{name}.mp4", frames)[record.frame_index]
        truth_frame = _pinhole(root, scene, gt["name"], record.frame_index)
        ours, floor, shifted = _undistortion_check(
            np.asarray(Image.open(record.image_path)), distorted, truth_frame, K, dist
        )
        print(f"G-2 {scene} {name}: ours {ours:.3f}, OpenCV {floor:.3f}, shifted {shifted:.3f}")
        assert ours <= floor + 0.25
        assert shifted >= 1.2 * ours


# G-3, G-4 · SelfCap converter (D-59, D-60, D-63).

_SELFCAP = {"tiny_unsync": (0, 12), "studio": (0, 4)}


@pytest.fixture(scope="session")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder the G-checks share for one session: re-packed archives and conversions made
    once (#109, after the timing review)."""
    return tmp_path_factory.mktemp("synth-shared")


def _checked_cameras(root: Path, scene: str) -> list[str]:
    """The cameras G-3 and G-4 compare frames for: the first three and the test camera."""
    truth = _ground_truth(root, scene)
    names = sorted(str(c["layout_names"]["selfcap"]) for c in truth["cameras"])
    return sorted({*names[:3], _test_camera(truth)})


def _calibration_subset(source: Path, target: Path, names: list[str]) -> None:
    """intri.yml, extri.yml and sync.json of `source` restricted to `names`, written with
    OpenCV's FileStorage, as the dataset's own files are."""
    target.mkdir(parents=True, exist_ok=True)
    for file in ("intri.yml", "extri.yml"):
        reader = cv2.FileStorage(str(source / file), cv2.FILE_STORAGE_READ)
        writer = cv2.FileStorage(str(target / file), cv2.FILE_STORAGE_WRITE)
        for key in reader.root().keys():
            node = reader.getNode(key)
            if key == "names":
                writer.write("names", names)
            elif key.rsplit("_", 1)[-1] in names:
                if node.isInt():
                    writer.write(key, int(node.real()))
                elif node.isReal():
                    writer.write(key, node.real())
                else:
                    writer.write(key, node.mat())
        reader.release()
        writer.release()
    if (source / "sync.json").is_file():
        sync = json.loads((source / "sync.json").read_text())
        (target / "sync.json").write_text(json.dumps({n: sync[n] for n in names}))


def _repacked(root: Path, scene: str, shared: Path) -> Path:
    """The lossless videos of the checked cameras and their calibration, re-packed once per
    session as the scene's release archive (gzip level 0: the converter needs the format, not
    the compression), so frames can be compared bitwise-close (G-3, G-4)."""
    out = shared / f"repacked-{scene}"
    archive = out / f"{scene}-release.tar.gz"
    if archive.is_file():
        return out
    source = root / "lossless" / "selfcap" / scene
    _verified_folder(root, f"lossless/selfcap/{scene}")
    names = _checked_cameras(root, scene)
    _calibration_subset(source / "optimized", out / "optimized", names)
    partial = out / f".{archive.name}.tmp"
    with tarfile.open(partial, "w:gz", compresslevel=0) as tar:
        for name in names:
            tar.add(source / "videos" / f"{name}.mp4", arcname=f"{scene}-release/videos/{name}.mp4")
        tar.add(out / "optimized", arcname=f"{scene}-release/optimized")
    partial.rename(archive)
    return out


def _test_camera(truth: dict[str, Any]) -> str:
    return str(_gt_camera(truth, "selfcap", truth["test_camera"])["layout_names"]["selfcap"])


@pytest.mark.parametrize("scene", sorted(_SELFCAP))
def test_g3_selfcap_times_split_and_archive_hash(tmp_path: Path, scene: str) -> None:
    """G-3, from the dataset's own (H.264) archives: τ equals the ground truth's per-frame
    times to 1e-12, and n/60 without sync (on tiny_unsync); the test camera is held out; the
    recorded archive SHA-256 equals the dataset's checksum."""
    root = _root()
    truth = _ground_truth(root, scene)
    assert truth["fps"] == 60, "openftgs-synth 0.1.1 renders the SelfCap scenes at 60 fps"
    test = _test_camera(truth)
    window = _SELFCAP[scene]
    # Without sync τ is n/60 whatever the scene; the small scene checks it.
    for use_sync in (True, False) if scene == "tiny_unsync" else (True,):
        out = tmp_path / f"out-{use_sync}"
        manifest = read_scene_manifest(
            convert_selfcap(
                root / "selfcap",
                out,
                scene=scene,
                window=window,
                test_camera=test,
                undistort="internal",
                use_sync=use_sync,
                jobs=4,
            )
        )
        for record in manifest.records:
            gt = _gt_camera(truth, "selfcap", record.camera.name)
            n = int(record.frame_index or 0)
            expected = next(f["t"] for f in gt["frames"] if f["n"] == n) if use_sync else n / 60
            assert abs(record.timestamp - expected) <= 1e-12
            assert record.split == ("test" if record.camera.name == test else "train")
        name = f"{scene}-release.tar.gz"
        assert manifest.metadata["archives"] == {name: _checksums(root)[f"selfcap/{name}"]}


def _selfcap_frames_check(root: Path, scene: str, manifest: SceneManifest) -> None:
    """G-3's frame check: the converted frames against the ground truth, both area-resized by
    the scene's ratio, within OpenCV's floor plus 0.25 of a level (G-2's bound)."""
    truth = _ground_truth(root, scene)
    for name in sorted({r.camera.name for r in manifest.records})[:3]:
        gt = _gt_camera(truth, "selfcap", name)
        K, dist = np.array(gt["K_opencv"]), np.array(gt["distortion"])
        record = _records(manifest, name)[-1]
        n = int(record.frame_index or 0)
        video = root / "lossless" / "selfcap" / scene / "videos" / f"{name}.mp4"
        distorted = _decode(video, n + 1)[n]
        width, height = record.camera.width, record.camera.height
        want = resize_area(_pinhole(root, scene, gt["name"], n).astype(np.float64), width, height)
        floor = resize_area(cv2.undistort(distorted, K, dist).astype(np.float64), width, height)
        inside = _inside(K, dist, gt["width"], gt["height"]).astype(np.float64)
        valid = resize_area(inside, width, height) >= 1.0
        got = _image(record.image_path)
        shifted_full = _translate(
            _pinhole(root, scene, gt["name"], n), 0.5, 0.5, (gt["width"], gt["height"])
        )[0]
        shifted = _mean_error(got, resize_area(shifted_full, width, height), valid)
        ours, opencv = _mean_error(got, want, valid), _mean_error(floor, want, valid)
        print(f"G-3 {scene} {name}: ours {ours:.3f}, OpenCV {opencv:.3f}, shifted {shifted:.3f}")
        assert ours <= opencv + 0.25
        assert shifted >= 1.2 * ours


@pytest.mark.parametrize("scene", sorted(_SELFCAP))
def test_g3_selfcap_internal_frames_match_the_ground_truth(
    tmp_path: Path, shared: Path, scene: str
) -> None:
    """G-3, from the lossless videos re-packed as archives, undistort='internal': frames within
    G-2's bound of the ground truth (both area-resized by the scene's ratio)."""
    root = _root()
    truth = _ground_truth(root, scene)
    src = _repacked(root, scene, shared)
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene=scene,
            window=_SELFCAP[scene],
            test_camera=_test_camera(truth),
            undistort="internal",
            jobs=4,
        )
    )
    _selfcap_frames_check(root, scene, manifest)


def _translate(
    image: np.ndarray, dx: float, dy: float, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """`image` sampled bilinearly at (x + dx, y + dy) on a `size` = (width, height) grid, and
    the mask of samples inside the image."""
    width, height = size
    map_x, map_y = np.meshgrid(np.arange(width) + dx, np.arange(height) + dy)
    moved = cv2.remap(
        image.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
    )
    inside = (
        (map_x >= 0) & (map_x <= image.shape[1] - 1) & (map_y >= 0) & (map_y <= image.shape[0] - 1)
    )
    return moved.astype(np.float64), inside


_G4: dict[str, list[tuple[str, float, float, float]]] = {}


def _g4_errors(shared: Path, scene: str) -> list[tuple[str, float, float, float]]:
    """For three cameras of a COLMAP conversion: the converted frame's error to the ground
    truth translated by the principal points' difference, OpenCV's undistortion into COLMAP's
    camera scored the same way, and the error with the translation off by 0.5 px. Both sides
    are resized as the converter resizes and scored where the ground truth reaches. COLMAP's
    read-back fx, fy are checked to be the ground truth's on the way."""
    if scene in _G4:  # one conversion serves both G-4 tests
        return _G4[scene]
    root = _root()
    truth = _ground_truth(root, scene)
    src = _repacked(root, scene, shared)
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            shared / f"g4-{scene}",
            scene=scene,
            window=_SELFCAP[scene],
            test_camera=_test_camera(truth),
            undistort="colmap",
            jobs=4,
        )
    )
    results = []
    for name in sorted({r.camera.name for r in manifest.records})[:3]:
        gt = _gt_camera(truth, "selfcap", name)
        record = _records(manifest, name)[-1]
        n = int(record.frame_index or 0)
        K = record.camera.K.double().numpy()
        width, height = record.camera.width, record.camera.height
        # COLMAP's size before resizing by ½ (half to even, Q-89): 2W′ − 1, 2W′ or 2W′ + 1;
        # fx, fy are the truth's.
        sizes = [
            (wc, hc)
            for wc in (2 * width - 1, 2 * width, 2 * width + 1)
            for hc in (2 * height - 1, 2 * height, 2 * height + 1)
            if np.isclose(K[0, 0] * wc / width, gt["K_opencv"][0][0], rtol=1e-6)
            and np.isclose(K[1, 1] * hc / height, gt["K_opencv"][1][1], rtol=1e-6)
        ]
        assert sizes, f"{name}: the read-back fx, fy aren't the ground truth's"
        wc, hc = sizes[0]
        cx_c, cy_c = K[0, 2] * wc / width, K[1, 2] * hc / height  # continuous convention
        cx_g, cy_g = gt["K_continuous"][0][2], gt["K_continuous"][1][2]
        truth_frame = _pinhole(root, scene, gt["name"], n)
        video = root / "lossless" / "selfcap" / scene / "videos" / f"{name}.mp4"
        distorted = _decode(video, n + 1)[n]
        colmap_cv = np.array(
            [
                [gt["K_opencv"][0][0], 0, cx_c - 0.5],
                [0, gt["K_opencv"][1][1], cy_c - 0.5],
                [0, 0, 1],
            ]
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            np.array(gt["K_opencv"]),
            np.array(gt["distortion"]),
            None,
            colmap_cv,
            (wc, hc),
            cv2.CV_32FC1,
        )
        opencv = resize_area(
            cv2.remap(distorted, map_x, map_y, cv2.INTER_LINEAR).astype(np.float64), width, height
        )
        got = _image(record.image_path)
        errors, floors = [], []
        for offset in (0.0, 0.5):
            moved, inside = _translate(
                truth_frame, cx_g - cx_c + offset, cy_g - cy_c + offset, (wc, hc)
            )
            want = resize_area(moved, width, height)
            valid = resize_area(inside.astype(np.float64), width, height) >= 1.0
            valid[:2], valid[-2:], valid[:, :2], valid[:, -2:] = False, False, False, False
            errors.append(_mean_error(got, want, valid))
            floors.append(_mean_error(opencv, want, valid))
        print(
            f"G-4 {scene} {name}: ours {errors[0]:.3f}, OpenCV {floors[0]:.3f}, "
            f"off by 0.5 {errors[1]:.3f}"
        )
        results.append((name, errors[0], floors[0], errors[1]))
    _G4[scene] = results
    return results


@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs COLMAP on PATH (D-60)")
@pytest.mark.parametrize("scene", sorted(_SELFCAP))
def test_g4_selfcap_colmap_frames_line_up_with_the_read_back_camera(
    shared: Path, scene: str
) -> None:
    """G-4: COLMAP's read-back fx, fy equal the ground truth's, and the converted frames are at
    least 1.2× worse with the translation off by 0.5 px (C-4's pixel-centre check)."""
    for _, ours, _, off in _g4_errors(shared, scene):
        assert off >= 1.2 * ours


# Q-103: G-4's frame bound kept verbatim; COLMAP's resampling is softer than OpenCV's.
@pytest.mark.xfail(
    strict=True,
    reason="Q-103: COLMAP's warp is softer than OpenCV's bilinear remap into the same camera",
)
@pytest.mark.skipif(shutil.which("colmap") is None, reason="needs COLMAP on PATH (D-60)")
@pytest.mark.parametrize("scene", sorted(_SELFCAP))
def test_g4_selfcap_colmap_frames_within_g2s_bound(shared: Path, scene: str) -> None:
    """G-4: the converted frames equal the ground truth translated by the principal points'
    difference within G-2's bound: at most OpenCV's undistortion into COLMAP's camera, scored
    the same way, plus 0.25 of an 8-bit level."""
    for _, ours, opencv, _ in _g4_errors(shared, scene):
        assert ours <= opencv + 0.25


# G-5 · masks (D-64).


def test_g5_ground_truth_masks_reach_the_manifest(tmp_path: Path) -> None:
    """G-5: the test view's ground-truth masks, at the converted size, supplied with masks.json,
    give each test image a mask, metadata.mask_source and masks_approximate, and the dataset
    reads them back."""
    root = _root()
    scene = "tiny_unsync"
    _verified(root, f"selfcap/{scene}-release.tar.gz")
    truth = _ground_truth(root, scene)
    test = _test_camera(truth)
    gt = _gt_camera(truth, "selfcap", test)
    window = _SELFCAP[scene]
    width, height = resized_size(gt["width"], gt["height"], 0.5)
    masks = tmp_path / "masks"
    masks.mkdir()
    for n in range(*window):
        source = np.asarray(
            Image.open(root / "ground_truth" / scene / "masks" / gt["name"] / f"{n:06d}.png")
        )
        Image.fromarray(resize_area(source, width, height), "L").save(masks / f"{n:06d}.png")
    (masks / "masks.json").write_text(
        json.dumps({"mask_source": "openftgs-synth ground truth", "masks_approximate": False})
    )
    path = convert_selfcap(
        root / "selfcap",
        tmp_path / "out",
        scene=scene,
        window=window,
        test_camera=test,
        undistort="internal",
        masks=masks,
        jobs=4,
    )
    manifest = read_scene_manifest(path)
    assert manifest.metadata["mask_source"] == "openftgs-synth ground truth"
    assert manifest.metadata["masks_approximate"] is False
    video = MultiViewVideo(manifest)
    for index, record in enumerate(video.records):
        mask = video.load_mask(index)
        assert (mask is not None) is (record.split == "test")
        if mask is not None:
            assert mask.shape == (height, width) and float(mask.max()) > 0.0


# G-6 · hardware decoding (D-48).


@pytest.mark.gpu
def test_g6_hardware_decoding_matches_cpu_decoding(tmp_path: Path) -> None:
    """G-6 (GPU): 10 frames of courtyard's H.264 videos (three of its cameras) converted with
    --hwaccel cuda are within one 8-bit level of CPU decoding, with the differing pixels
    counted."""
    root = _root()
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, check=True
    ).stdout.split()
    if "cuda" not in listed:
        pytest.skip("this ffmpeg build has no cuda hwaccel")
    courtyard = root / "enerf_outdoor" / "courtyard"
    _verified_folder(root, "enerf_outdoor/courtyard")
    # Decoding is compared camera by camera: three cameras give the same check as eighteen.
    src = tmp_path / "courtyard"
    names = ["0000", "0001", "0002"]
    _yaml_subset(courtyard, src, names)
    cpu = read_scene_manifest(
        convert_opencv_yaml(src, tmp_path / "cpu", fps=60.0, frames=10, resize=1.0)
    )
    gpu = read_scene_manifest(
        convert_opencv_yaml(src, tmp_path / "gpu", fps=60.0, frames=10, resize=1.0, hwaccel="cuda")
    )
    assert gpu.metadata["ffmpeg"]["decoder"] == "cuda"
    differing = 0
    for a, b in zip(cpu.records, gpu.records, strict=True):
        x, y = _image(a.image_path), _image(b.image_path)
        assert np.abs(x - y).max() <= 1
        differing += int((x != y).any(axis=-1).sum())
    print(f"G-6: {differing} differing pixels of {len(cpu.records)} frames at 1920x1080")


def _yaml_subset(source: Path, target: Path, names: list[str]) -> None:
    """An opencv-yaml scene restricted to `names`: calibration rewritten, videos linked."""
    _calibration_subset(source, target, names)
    (target / "videos").mkdir()
    for name in names:
        (target / "videos" / f"{name}.mp4").symlink_to(source / "videos" / f"{name}.mp4")


# G-7 · frame cache (D-51).


def _kitchen(tmp_path: Path) -> MultiViewVideo:
    _verified_folder(_root(), "lossless/n3dv/kitchen")
    path = convert_n3dv(
        _root() / "lossless" / "n3dv" / "kitchen", tmp_path / "out", frames=2, resize=1.0, jobs=4
    )
    return MultiViewVideo.from_manifest(path)


def _training(video: MultiViewVideo) -> Iterator[int]:
    return (i for i, r in enumerate(video.records) if r.split == "train")


def test_g7_memory_mapped_cache_at_full_size(tmp_path: Path) -> None:
    """G-7: on a 2-frame kitchen conversion at full size (1352×1014), cache frames are bitwise
    PNG decoding, memory-mapped."""
    video = _kitchen(tmp_path)
    prefetcher = Prefetcher(FrameCache.open(video, placement="mmap"), device="cpu", depth=4)
    order = list(_training(video))
    prefetcher.schedule(order)
    for index in order:
        assert torch.equal(prefetcher.take(index), video.load_image(index))
    prefetcher.close()


@pytest.mark.gpu
def test_g7_gpu_cache_at_full_size(tmp_path: Path) -> None:
    """G-7 (GPU): the same frames from the cache in GPU memory, and through the CUDA
    prefetcher from the memory-mapped cache, are bitwise PNG decoding."""
    video = _kitchen(tmp_path)
    order = list(_training(video))
    on_gpu = Prefetcher(
        FrameCache.open(video, placement="gpu", device="cuda"), device="cuda", depth=0
    )
    mapped = Prefetcher(FrameCache.open(video, placement="mmap"), device="cuda", depth=4)
    mapped.schedule(order)
    for index in order:
        want = video.load_image(index).cuda()
        assert torch.equal(on_gpu.take(index), want)
        assert torch.equal(mapped.take(index), want)
    mapped.close()

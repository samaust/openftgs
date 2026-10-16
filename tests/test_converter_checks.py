# SPDX-License-Identifier: Apache-2.0
"""C-1 to C-5: the converters against the paper's datasets (§3.1 converter checks, D-28, D-59,
D-60; Q-100).

This project downloads none of the paper's datasets, whose licenses are non-commercial (D-79), so
these checks skip, naming the missing folder, unless the raw downloads sit under
$OPENFTGS_DATASETS: `n3dv/<scene>/`, `enerf_outdoor/<scene>/` and `selfcap/` (Q-100). C-4's
Gaussian-dot part needs only COLMAP and runs wherever COLMAP does.

Matches come from OpenCV's SIFT with Lowe's ratio test (a test-only oracle, §2.3); keypoints are
in OpenCV's pixel convention and gain +0.5 to meet the manifests' continuous convention (§1.4).
"""

import os
import shutil
import subprocess
import tarfile
from itertools import permutations
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pytest
from PIL import Image

from openftgs.data import convert_n3dv, convert_opencv_yaml, convert_selfcap
from openftgs.io import read_scene_manifest
from openftgs.types import Camera, SceneManifest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH (D-48)",
)
needs_colmap = pytest.mark.skipif(shutil.which("colmap") is None, reason="needs COLMAP (D-60)")
LOWE = 0.75  # Q-100


def _dataset(*parts: str) -> Path:
    """A raw dataset folder under $OPENFTGS_DATASETS, or a skip naming it (Q-100)."""
    root = Path(os.environ.get("OPENFTGS_DATASETS", ""))
    folder = root.joinpath(*parts)
    if not folder.is_dir() or not any(folder.iterdir()):
        pytest.skip(f"{folder} is empty: the paper's datasets aren't used by this project (D-79)")
    return folder


def _first_scene(folder: Path, marker: str) -> Path:
    scenes = sorted(p for p in folder.iterdir() if (p / marker).exists())
    if not scenes:
        pytest.skip(f"no scene with {marker} under {folder}")
    return scenes[0]


def _matches(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SIFT matches between two RGB images, in the continuous pixel convention."""
    sift = getattr(cv2, "SIFT_create")()  # noqa: B009 (absent from OpenCV 5's stubs)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < LOWE * n.distance]
    xa = np.array([ka[m.queryIdx].pt for m in good]) + 0.5
    xb = np.array([kb[m.trainIdx].pt for m in good]) + 0.5
    return xa, xb


def _parts(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w2c = camera.world_to_camera.double().numpy()
    return camera.K.double().numpy(), w2c[:3, :3], w2c[:3, 3]


def _fundamental(
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray, K2: np.ndarray, R2: np.ndarray, t2: np.ndarray
) -> np.ndarray:
    R = R2 @ R1.T
    t = t2 - R @ t1
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F: np.ndarray = np.linalg.inv(K2).T @ tx @ R @ np.linalg.inv(K1)
    return F


def _sampson(F: np.ndarray, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    a = np.c_[xa, np.ones(len(xa))]
    b = np.c_[xb, np.ones(len(xb))]
    Fa, Ftb = a @ F.T, b @ F
    numerator = np.einsum("ij,ij->i", b, Fa) ** 2
    denominator = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    errors: np.ndarray = np.sqrt(numerator / denominator)
    return errors


def _projection(camera: Camera) -> np.ndarray:
    K, R, t = _parts(camera)
    P: np.ndarray = K @ np.c_[R, t]
    return P


def _triangulate(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """A float64 DLT per point (init's triangulate_dlt is M13's)."""
    points = []
    for (u1, v1), (u2, v2) in zip(x1, x2, strict=True):
        A = np.stack(
            [u1 * P1[2] - P1[0], v1 * P1[2] - P1[1], u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]]
        )
        X = np.linalg.svd(A)[2][-1]
        points.append(X[:3] / X[3])
    return np.array(points)


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = np.c_[X, np.ones(len(X))] @ P.T
    pixels: np.ndarray = x[:, :2] / x[:, 2:]
    return pixels


def _frame(manifest: SceneManifest, camera: str, index: int = 0) -> tuple[Camera, np.ndarray]:
    record = next(r for r in manifest.records if r.camera.name == camera and r.frame_index == index)
    return record.camera, np.asarray(Image.open(record.image_path).convert("RGB"))


def _median_sampson(manifest: SceneManifest, a: str, b: str, index: int = 0) -> float:
    ca, ia = _frame(manifest, a, index)
    cb, ib = _frame(manifest, b, index)
    xa, xb = _matches(ia, ib)
    return float(np.median(_sampson(_fundamental(*_parts(ca), *_parts(cb)), xa, xb)))


def _neighbours(manifest: SceneManifest) -> list[tuple[str, str]]:
    names = sorted({r.camera.name for r in manifest.records})
    return list(zip(names, names[1:], strict=False))[:4]


def _reprojection(manifest: SceneManifest, a: str, b: str, c: str, index: int = 0) -> float:
    """C-3: matches of a with both b and c, triangulated from a and b, reprojected into c."""
    ca, ia = _frame(manifest, a, index)
    cb, ib = _frame(manifest, b, index)
    cc, ic = _frame(manifest, c, index)
    xab_a, xab_b = _matches(ia, ib)
    xac_a, xac_c = _matches(ia, ic)
    keys = {tuple(np.round(p, 2)): k for k, p in enumerate(xac_a)}
    pairs = [
        (i, keys[tuple(np.round(p, 2))])
        for i, p in enumerate(xab_a)
        if tuple(np.round(p, 2)) in keys
    ]
    if len(pairs) < 20:
        pytest.skip(f"only {len(pairs)} three-view tracks")
    i, j = (np.array(v) for v in zip(*pairs, strict=True))
    X = _triangulate(_projection(ca), _projection(cb), xab_a[i], xab_b[i])
    return float(np.median(np.linalg.norm(_project(_projection(cc), X) - xac_c[j], axis=1)))


# C-1 · Neural3DV (D-28).


@needs_ffmpeg
@pytest.mark.dataset
def test_c1_n3dv_epipolar_residuals_and_column_permutations(tmp_path: Path) -> None:
    """C-1: Sampson error of matches between neighbouring cameras at frame 0, median ≤ 1 px
    with the converted cameras; every permutation of the LLFF rotation columns ≥ 10× worse."""
    scene = _first_scene(_dataset("n3dv"), "poses_bounds.npy")
    manifest = read_scene_manifest(convert_n3dv(scene, tmp_path / "out", frames=1))
    poses = np.load(scene / "poses_bounds.npy")
    names = sorted({r.camera.name for r in manifest.records})
    for a, b in _neighbours(manifest):
        good = _median_sampson(manifest, a, b)
        assert good <= 1.0
        ca, ia = _frame(manifest, a)
        cb, ib = _frame(manifest, b)
        xa, xb = _matches(ia, ib)
        for order in permutations(range(3)):
            if order == (0, 1, 2):
                continue

            def pose(
                name: str, camera: Camera, order: tuple[int, ...] = order
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                m = poses[names.index(name)][:15].reshape(3, 5)
                c = m[:, list(order)]  # the columns permuted before D-28's [c1, c0, −c2]
                rotation = np.stack([c[:, 1], c[:, 0], -c[:, 2]])
                return camera.K.double().numpy(), rotation, -rotation @ m[:, 3]

            bad = float(np.median(_sampson(_fundamental(*pose(a, ca), *pose(b, cb)), xa, xb)))
            assert bad >= 10 * good, f"columns {order}: {bad} against {good}"


# C-2 · ENeRF-Outdoor; C-3 · both converters.


@needs_ffmpeg
@pytest.mark.dataset
def test_c2_enerf_outdoor_epipolar_residuals(tmp_path: Path) -> None:
    """C-2: the same test after undistortion: median Sampson error ≤ 1 px."""
    scene = _first_scene(_dataset("enerf_outdoor"), "intri.yml")
    manifest = read_scene_manifest(
        convert_opencv_yaml(scene, tmp_path / "out", fps=60.0, frames=1, resize=1.0)
    )
    for a, b in _neighbours(manifest):
        assert _median_sampson(manifest, a, b) <= 1.0


@needs_ffmpeg
@pytest.mark.dataset
@pytest.mark.parametrize("dataset", ["n3dv", "enerf_outdoor"])
def test_c3_triangulated_matches_reproject_into_a_third_camera(
    tmp_path: Path, dataset: str
) -> None:
    """C-3: matches triangulated from two cameras reproject into a third with a median error
    ≤ 1.5 px."""
    if dataset == "n3dv":
        scene = _first_scene(_dataset("n3dv"), "poses_bounds.npy")
        path = convert_n3dv(scene, tmp_path / "out", frames=1)
    else:
        scene = _first_scene(_dataset("enerf_outdoor"), "intri.yml")
        path = convert_opencv_yaml(scene, tmp_path / "out", fps=60.0, frames=1, resize=1.0)
    manifest = read_scene_manifest(path)
    names = sorted({r.camera.name for r in manifest.records})
    assert _reprojection(manifest, names[1], names[2], names[3]) <= 1.5


# C-4 · SelfCap undistortion (D-60).


def _selfcap_scene(folder: Path) -> str:
    for scene, archive in (("corgi1", "corgi"), ("dance1", "hair"), ("bike1", "bike")):
        if (folder / f"{archive}-release.tar.gz").is_file():
            return scene
    pytest.skip(f"no SelfCap release archive in {folder}")


@needs_ffmpeg
@needs_colmap
@pytest.mark.dataset
@pytest.mark.parametrize("undistort", ["colmap", "internal"])
def test_c4_selfcap_background_epipolar_residuals(
    tmp_path: Path, undistort: Literal["colmap", "internal"]
) -> None:
    """C-4: Sampson error of matches on the static background between neighbouring cameras at
    the first frame, after COLMAP undistortion and with undistort='internal': median ≤ 1 px.
    The background is where camera a's first two frames differ by at most 10 levels."""
    folder = _dataset("selfcap")
    scene = _selfcap_scene(folder)
    manifest = read_scene_manifest(
        convert_selfcap(folder, tmp_path / "out", scene=scene, undistort=undistort)
    )
    n0 = manifest.metadata["window"][0]
    for a, b in _neighbours(manifest):
        ca, ia = _frame(manifest, a, n0)
        _, ia1 = _frame(manifest, a, n0 + 1)
        cb, ib = _frame(manifest, b, n0)
        xa, xb = _matches(ia, ib)
        still = np.abs(ia.astype(int) - ia1.astype(int)).max(axis=-1) <= 10
        keep = still[
            np.clip(xa[:, 1].astype(int), 0, still.shape[0] - 1),
            np.clip(xa[:, 0].astype(int), 0, still.shape[1] - 1),
        ]
        errors = _sampson(_fundamental(*_parts(ca), *_parts(cb)), xa[keep], xb[keep])
        assert float(np.median(errors)) <= 1.0


@needs_ffmpeg
@needs_colmap
def test_c4_colmap_puts_gaussian_dots_where_the_pinhole_camera_projects_them(
    tmp_path: Path,
) -> None:
    """C-4, needing only COLMAP: an image of Gaussian dots at known 3D points, rendered through
    a known distorted camera and undistorted by COLMAP through the SelfCap converter: every dot's
    centroid lies within 0.1 px of where the read-back PINHOLE camera projects it (a wrong
    pixel-centre convention gives about 0.5 px; the scene is resized by ½, as any SelfCap scene
    outside the card's table is (Q-97), so that error would be about 0.25 px, still well above the
    bound)."""
    width, height, sigma = 640, 480, 1.6
    K = np.array([[520.0, 0.0, 319.5], [0.0, 520.0, 239.5], [0.0, 0.0, 1.0]])  # OpenCV
    dist = np.array([-0.18, 0.05, 0.0006, -0.0004, 0.0])
    rng = np.random.default_rng(4)
    grid = np.stack(
        np.meshgrid(np.linspace(-0.45, 0.45, 7), np.linspace(-0.3, 0.3, 5)), -1
    ).reshape(-1, 2)
    points = np.c_[grid * 4.0 + rng.uniform(-0.02, 0.02, grid.shape), np.full(len(grid), 4.0)]
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, dist)
    pixels = pixels.reshape(-1, 2)  # OpenCV convention: pixel centres at integers
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.zeros((height, width))
    for x, y in pixels:
        image += np.exp(-((u - x) ** 2 + (v - y) ** 2) / (2 * sigma**2))
    frame = np.clip(np.rint(np.repeat(image[..., None], 3, axis=-1) * 230), 0, 255).astype(np.uint8)
    staging = tmp_path / "staging" / "dots-release"
    (staging / "videos").mkdir(parents=True)
    (staging / "optimized").mkdir()
    for name in ("0000", "0001"):
        Image.fromarray(frame).save(tmp_path / "dot.png")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-loop",
                "1",
                "-i",
                str(tmp_path / "dot.png"),
                "-frames:v",
                "2",
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                str(staging / "videos" / f"{name}.mp4"),
            ],
            check=True,
        )
    intri = cv2.FileStorage(str(staging / "optimized" / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(str(staging / "optimized" / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    intri.write("names", ["0000", "0001"])
    for k, name in enumerate(("0000", "0001")):
        intri.write(f"K_{name}", K)
        intri.write(f"D_{name}", dist.reshape(5, 1))
        extri.write(f"R_{name}", np.zeros((3, 1)))
        extri.write(f"Rot_{name}", np.eye(3))
        extri.write(f"T_{name}", np.array([[0.1 * k], [0.0], [0.0]]))
    intri.release()
    extri.release()
    src = tmp_path / "src"
    src.mkdir()
    with tarfile.open(src / "dots-release.tar.gz", "w:gz") as tar:
        tar.add(staging, arcname="dots-release")
    manifest = read_scene_manifest(
        convert_selfcap(
            src,
            tmp_path / "out",
            scene="dots",
            window=(0, 1),
            test_camera="0001",
            undistort="colmap",
        )
    )
    camera, converted = _frame(manifest, "0000", 0)
    Kc = camera.K.double().numpy()  # the read-back PINHOLE camera, resized; continuous convention
    expected = (points @ Kc.T)[:, :2] / points[:, 2:]
    gray = converted.astype(np.float64).mean(axis=-1)
    errors = []
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    for ex, ey in expected:
        window = (np.abs(xx + 0.5 - ex) <= 4) & (np.abs(yy + 0.5 - ey) <= 4)
        weights = gray * window
        cx = float((weights * (xx + 0.5)).sum() / weights.sum())
        cy = float((weights * (yy + 0.5)).sum() / weights.sum())
        errors.append(np.hypot(cx - ex, cy - ey))
    print(f"C-4 dots: max centroid error {max(errors):.3f} px over {len(errors)} dots")
    assert max(errors) <= 0.1


# C-5 · SelfCap timing (D-59).


@needs_ffmpeg
@pytest.mark.dataset
def test_c5_corrected_pairing_reprojects_better(tmp_path: Path) -> None:
    """C-5, on corgi: for camera pairs whose sync offsets differ by more than one frame, the
    moving subject matched between camera A at frame n and camera B at the frame nearest A's
    corrected time, triangulated and reprojected into a third camera, gives a lower median
    reprojection error than pairing both cameras at frame n."""
    folder = _dataset("selfcap")
    if not (folder / "corgi-release.tar.gz").is_file():
        pytest.skip(f"no corgi-release.tar.gz in {folder}")
    manifest = read_scene_manifest(
        convert_selfcap(folder, tmp_path / "out", scene="corgi1", undistort="internal")
    )
    by_camera: dict[str, dict[int, Any]] = {}
    for record in manifest.records:
        by_camera.setdefault(record.camera.name, {})[record.frame_index] = record  # type: ignore[index]
    sync = {
        name: n / 60 - r.timestamp
        for name, frames in by_camera.items()
        for n, r in list(frames.items())[:1]
    }
    names = sorted(sync)
    n = manifest.metadata["window"][0] + 30
    pairs = [(a, b) for a in names for b in names if a < b and abs(sync[a] - sync[b]) * 60 > 1]
    if not pairs:
        pytest.skip("no camera pair whose offsets differ by more than one frame")
    a, b = pairs[0]
    c = next(x for x in names if x not in (a, b))
    corrected = round(n + (sync[b] - sync[a]) * 60)

    def error(nb: int) -> float:
        ra, rb, rc = (
            by_camera[a][n],
            by_camera[b][nb],
            by_camera[c][round(n + (sync[c] - sync[a]) * 60)],
        )
        ia, ib, ic = (np.asarray(Image.open(r.image_path).convert("RGB")) for r in (ra, rb, rc))
        moving = (
            np.abs(
                ia.astype(int)
                - np.asarray(Image.open(by_camera[a][n + 1].image_path).convert("RGB")).astype(int)
            ).max(axis=-1)
            > 10
        )
        xab_a, xab_b = _matches(ia, ib)
        xac_a, xac_c = _matches(ia, ic)
        keys = {tuple(np.round(p, 2)): k for k, p in enumerate(xac_a)}
        tracks = [
            (i, keys[tuple(np.round(p, 2))])
            for i, p in enumerate(xab_a)
            if tuple(np.round(p, 2)) in keys and moving[int(p[1]), int(p[0])]
        ]
        i, j = (np.array(v) for v in zip(*tracks, strict=True))
        X = _triangulate(_projection(ra.camera), _projection(rb.camera), xab_a[i], xab_b[i])
        return float(
            np.median(np.linalg.norm(_project(_projection(rc.camera), X) - xac_c[j], axis=1))
        )

    assert error(corrected) < error(n)

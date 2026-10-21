# SPDX-License-Identifier: Apache-2.0
"""The scripts that prepare M17's `studio` from openftgs-synth's ground truth (Q-219; #317):
`scripts/synth_window.py`, a window around the test view's fastest frames (D-69), and
`scripts/synth_masks.py`, the test view's ground-truth masks as `ftgs convert selfcap --masks`
takes them (Q-87, D-64).

Added tests, not §3 rows, on `tiny_unsync`, the dataset's small SelfCap-layout scene. Each
script runs as M17 runs it, in a process of its own.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.dataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCENE = "tiny_unsync"


def _root() -> Path:
    """$OPENFTGS_DATASETS/synth, version 0.1.x, or a skip naming the folder (Q-100)."""
    root = Path(os.environ.get("OPENFTGS_DATASETS", "")) / "synth"
    if not (root / "VERSION").is_file():
        pytest.skip(f"openftgs-synth is not at {root}")
    version = (root / "VERSION").read_text().strip()
    if not version.startswith("0.1."):
        pytest.skip(f"openftgs-synth {version} is not 0.1.x")
    return root


def _run(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _truth(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / "ground_truth" / SCENE / "cameras.json").read_text())
    return data


def _gt_name(truth: dict[str, Any], camera: str) -> str:
    return str(next(c["name"] for c in truth["cameras"] if c["layout_names"]["selfcap"] == camera))


def test_synth_window_holds_the_test_views_fastest_frames() -> None:
    """D-69, Q-219: the window of the given length is centred on the 10 consecutive frames whose
    consecutive ground-truth frames of the test view differ most (mean absolute difference over
    pixels and channels), clipped to the scene's frames, so it holds that range. The range is
    found here with NumPy, independently of `data.fastest_frames`."""
    root = _root()
    truth = _truth(root)
    camera = str(truth["test_camera"])
    frames = int(truth["num_frames"])
    folder = root / "ground_truth" / SCENE / "pinhole" / _gt_name(truth, camera)
    images = [
        np.asarray(Image.open(folder / f"{n:06d}.png").convert("RGB"), dtype=np.float64)
        for n in range(frames)
    ]
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(images, images[1:], strict=False)]
    sums = [sum(diffs[s : s + 9]) for s in range(frames - 9)]
    start = sums.index(max(sums))
    for length in (10, 11):
        done = _run(
            "synth_window.py", root / "ground_truth" / SCENE, "--camera", camera, "--length", length
        )
        assert done.returncode == 0, done.stderr
        report = json.loads(done.stdout)
        n0 = min(max(start + 5 - length // 2, 0), frames - length)
        assert report["window"] == [n0, n0 + length]
        assert report["fastest_10"] == [start, start + 10]
        assert report["camera"] == camera and report["frames"] == frames
        assert n0 <= start and start + 10 <= n0 + length


def test_synth_window_outside_10_to_the_scenes_frames_exits_2() -> None:
    """A window longer than the scene's frames, or shorter than D-69's 10, is exit 2."""
    root = _root()
    truth = _truth(root)
    camera = str(truth["test_camera"])
    for length in (9, int(truth["num_frames"]) + 1):
        done = _run(
            "synth_window.py", root / "ground_truth" / SCENE, "--camera", camera, "--length", length
        )
        assert done.returncode == 2 and "length" in done.stderr


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe on PATH (D-48)",
)
def test_synth_masks_are_the_ground_truths_and_reach_the_manifest_as_exact(tmp_path: Path) -> None:
    """Q-219, Q-87, D-64: for each decode index of the window, the test view's ground-truth mask
    unchanged (the same pixels, at the camera's full size), and masks.json with a mask_source
    naming the dataset's version and masks_approximate false; `ftgs convert selfcap
    --undistort internal --resize 1.0 --masks` takes the folder, and the manifest says the
    masks are not approximate."""
    from openftgs.cli import main
    from openftgs.io import read_scene_manifest

    root = _root()
    truth = _truth(root)
    camera = str(truth["test_camera"])
    gt = _gt_name(truth, camera)
    masks = tmp_path / "masks"
    done = _run(
        "synth_masks.py",
        root / "ground_truth" / SCENE,
        "--camera",
        camera,
        "--window",
        "2:9",
        masks,
    )
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in masks.iterdir()) == [f"{n:06d}.png" for n in range(2, 9)] + [
        "masks.json"
    ]
    for n in range(2, 9):
        ours = np.asarray(Image.open(masks / f"{n:06d}.png"))
        theirs = np.asarray(
            Image.open(root / "ground_truth" / SCENE / "masks" / gt / f"{n:06d}.png")
        )
        assert ours.dtype == np.uint8 and np.array_equal(ours, theirs)
    version = (root / "VERSION").read_text().strip()
    sidecar = json.loads((masks / "masks.json").read_text())
    assert sidecar == {
        "mask_source": f"openftgs-synth {version} ground_truth/{SCENE}/masks/{gt} (exact)",
        "masks_approximate": False,
    }
    out = tmp_path / "scene"
    argv = [
        "convert", "selfcap", root / "selfcap", out, "--scene", SCENE, "--window", "2:9",
        "--test-camera", camera, "--undistort", "internal", "--resize", "1.0", "--masks", masks,
    ]  # fmt: skip
    assert main([str(a) for a in argv]) == 0
    manifest = read_scene_manifest(out / "scene.json")
    assert manifest.metadata["masks_approximate"] is False
    assert manifest.metadata["mask_source"] == sidecar["mask_source"]


def test_synth_masks_refuse_a_missing_mask_or_a_used_folder(tmp_path: Path) -> None:
    """A window reaching past the ground truth's masks, or an output folder that already holds
    something, is exit 2 with nothing written."""
    root = _root()
    truth = _truth(root)
    camera = str(truth["test_camera"])
    scene = root / "ground_truth" / SCENE
    frames = int(truth["num_frames"])
    out = tmp_path / "past"
    done = _run("synth_masks.py", scene, "--camera", camera, "--window", f"0:{frames + 1}", out)
    assert done.returncode == 2 and "missing" in done.stderr
    assert not out.exists() or not any(out.iterdir())
    used = tmp_path / "used"
    used.mkdir()
    (used / "other.txt").write_text("x")
    done = _run("synth_masks.py", scene, "--camera", camera, "--window", "0:2", used)
    assert done.returncode == 2
    assert sorted(p.name for p in used.iterdir()) == ["other.txt"]

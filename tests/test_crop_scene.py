# SPDX-License-Identifier: Apache-2.0
"""`scripts/crop_scene.py`, a scene with a size per camera cropped from one with a single size
(#350; Q-238, Q-232).

Added tests, not §3 rows: on a tiny three-camera scene, camera k of the manifest's order loses k
columns and k rows, K and the pose stay, every image and mask is its source's top-left block, the
uncropped camera's files are copied byte for byte and the records carry over; an OUT that holds
something, or a camera too small to crop, is exit 2. The script runs as it is run, in a process
of its own, since its pool's workers import it by path.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from builders import pinhole
from PIL import Image

from openftgs.data import MultiViewVideo
from openftgs.io import read_scene_manifest, write_scene_manifest
from openftgs.types import ImageRecord

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "crop_scene.py"
WIDTH, HEIGHT = 9, 7


def _source(root: Path, *, width: int = WIDTH, height: int = HEIGHT) -> Path:
    """Three cameras, two frames each; cam00 is the test camera, with a mask per image."""
    rng = np.random.default_rng(0)
    records: list[ImageRecord] = []
    masks: list[Path | None] = []
    for c in range(3):
        w2c = torch.eye(4)
        w2c[0, 3] = 0.1 * c
        camera = pinhole(
            width, height, fx=11.0, fy=12.0, cx=4.5, cy=3.5, world_to_camera=w2c, name=f"cam{c:02d}"
        )
        split: Literal["train", "test"] = "test" if c == 0 else "train"
        for n in range(2):
            path = root / "images" / camera.name / f"{n:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            Image.fromarray(pixels, "RGB").save(path)
            records.append(ImageRecord(camera, n / 30.0, path, split, frame_index=n))
            if split == "test":
                mask = root / "masks" / camera.name / f"{n:06d}.png"
                mask.parent.mkdir(parents=True, exist_ok=True)
                values = rng.integers(0, 2, size=(height, width), dtype=np.uint8) * 255
                Image.fromarray(values, "L").save(mask)
                masks.append(mask)
            else:
                masks.append(None)
    manifest = root / "scene.json"
    write_scene_manifest(manifest, records, masks=masks, metadata={"source": "tiny"})
    return manifest


def _crop(scene: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(scene), str(out), "--jobs", "2"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image)


def test_each_camera_loses_its_place_in_the_camera_order(tmp_path: Path) -> None:
    """cam<k> is cropped by k pixels on the right and at the bottom, with K and the pose as the
    source's; every image and mask is its source's top-left block; cam00's files are the
    source's bytes; timestamps, splits and frame indices carry over; the scene loads with two
    training sizes, and its metadata names the source manifest and its SHA-256."""
    scene = _source(tmp_path / "source")
    out = tmp_path / "cropped"
    done = _crop(scene, out)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [str(out / "scene.json")]

    source = read_scene_manifest(scene)
    video = MultiViewVideo.from_manifest(out / "scene.json")
    assert video.manifest.camera_names == source.camera_names
    assert len(video.records) == len(source.records)
    for k, (old, new) in enumerate(zip(source.records, video.records, strict=True)):
        crop = int(new.camera.name[3:])
        assert (new.camera.width, new.camera.height) == (WIDTH - crop, HEIGHT - crop)
        assert torch.equal(new.camera.K, old.camera.K)
        assert torch.equal(new.camera.world_to_camera, old.camera.world_to_camera)
        assert (new.timestamp, new.split, new.frame_index) == (
            old.timestamp,
            old.split,
            old.frame_index,
        )
        assert new.image_path == out / old.image_path.relative_to(scene.parent)
        cropped = _pixels(new.image_path)
        assert cropped.shape == (HEIGHT - crop, WIDTH - crop, 3)
        assert np.array_equal(cropped, _pixels(old.image_path)[: HEIGHT - crop, : WIDTH - crop])
        assert torch.equal(video.load_image(k), torch.from_numpy(cropped).float() / 255.0)
        if crop == 0:
            assert new.image_path.read_bytes() == old.image_path.read_bytes()
    for old_mask, new_mask in zip(source.masks, video.manifest.masks, strict=True):
        assert (old_mask is None) == (new_mask is None)
        if old_mask is not None and new_mask is not None:
            assert np.array_equal(_pixels(new_mask), _pixels(old_mask))
    sizes = {(r.camera.width, r.camera.height) for r in video.records if r.split == "train"}
    assert sizes == {(WIDTH - 1, HEIGHT - 1), (WIDTH - 2, HEIGHT - 2)}
    derived = json.loads((out / "scene.json").read_text())["metadata"]["derived_from"]
    assert derived["sha256"] == source.sha256
    assert derived["manifest"] == str(source.path)


def test_refuses_a_full_folder_and_a_camera_too_small(tmp_path: Path) -> None:
    """An OUT that holds something is left alone, and a camera that would lose its last column
    or row is refused, both with exit 2."""
    scene = _source(tmp_path / "source")
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("x")
    refused = _crop(scene, full)
    assert refused.returncode == 2
    assert "absent or empty" in refused.stderr
    assert [p.name for p in full.iterdir()] == ["keep.txt"]

    narrow = _source(tmp_path / "narrow", width=2, height=7)
    too_small = _crop(narrow, tmp_path / "out")
    assert too_small.returncode == 2
    assert "can't lose 2 pixels" in too_small.stderr

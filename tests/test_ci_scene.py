# SPDX-License-Identifier: Apache-2.0
"""The GPU CI's scene generator, `scripts/ci_scene.py` (§3.4: "a synthetic scene the size of the
Neural3DV setup (20 cameras, 300 frames, 1352×1014 …)"; Q-178, Q-179, Q-T235.2; #235).

An added test, not a §3 row: the generator writes a valid scene, here with its size parameters
reduced. The script runs as the CI runs it, in a process of its own, since its pool's workers
import it by path.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from openftgs.data import MultiViewVideo
from openftgs.io import read_scene_manifest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_scene.py"
SMALL = (
    "--cameras",
    "3",
    "--frames",
    "4",
    "--width",
    "32",
    "--height",
    "24",
    "--short-frames",
    "2",
)


def _generate(out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(out), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_generator_writes_a_valid_small_scene(tmp_path: Path) -> None:
    """The scene and its short manifest read back as ftgs-scene files (§2.2): cam00 held out,
    frame k at k/30 s with frame index k (§1.4), 8-bit RGB PNGs of the given size, every camera
    looking at the same point with a 60° horizontal field of view and the principal point at the
    image centre; the short manifest is the first frames of the same PNGs (Q-178)."""
    out = tmp_path / "scene"
    done = _generate(out, *SMALL, "--jobs", "2")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [str(out / "scene.json"), str(out / "scene-2.json")]

    video = MultiViewVideo.from_manifest(out / "scene.json")
    records = video.records
    assert len(records) == 3 * 4
    assert video.manifest.camera_names == ("cam00", "cam01", "cam02")
    focal = 16.0 / math.tan(math.radians(30.0))
    for record in records:
        camera = record.camera
        assert record.split == ("test" if camera.name == "cam00" else "train")
        assert record.frame_index is not None
        assert record.timestamp == record.frame_index / 30.0
        assert (camera.width, camera.height) == (32, 24)
        np.testing.assert_allclose(
            camera.K.numpy(), [[focal, 0.0, 16.0], [0.0, focal, 12.0], [0.0, 0.0, 1.0]], rtol=1e-6
        )
        origin = camera.world_to_camera.double().numpy() @ [0.0, 0.0, 0.0, 1.0]
        np.testing.assert_allclose(origin[:2], [0.0, 0.0], atol=1e-5)  # on the optical axis
        assert origin[2] > 0  # in front of the camera
        with Image.open(record.image_path) as image:
            assert (image.format, image.mode, image.size) == ("PNG", "RGB", (32, 24))
    assert {r.frame_index for r in records} == {0, 1, 2, 3}
    w2c = [r.camera.world_to_camera.double().numpy() for r in records]
    centres = {tuple(np.round(-m[:3, :3].T @ m[:3, 3], 6)) for m in w2c}
    assert len(centres) == 3  # three distinct camera centres

    short = read_scene_manifest(out / "scene-2.json")
    expected = [(r.camera.name, r.timestamp, r.image_path) for r in records if r.timestamp < 0.05]
    assert [(r.camera.name, r.timestamp, r.image_path) for r in short.records] == expected


def test_generator_is_seeded_and_resumable(tmp_path: Path) -> None:
    """Every frame comes from its own seed, so one process and two write the same PNGs; a
    second run on the same folder keeps the PNGs and rewrites the same manifests; a folder of
    another generation is refused with exit 2."""
    one, two = tmp_path / "one", tmp_path / "two"
    assert _generate(one, *SMALL, "--jobs", "1").returncode == 0
    assert _generate(two, *SMALL, "--jobs", "2").returncode == 0
    pngs = sorted(p.relative_to(one) for p in one.rglob("*.png"))
    assert len(pngs) == 12
    assert all((one / p).read_bytes() == (two / p).read_bytes() for p in pngs)
    assert (one / "scene.json").read_bytes() == (two / "scene.json").read_bytes()

    manifest = (one / "scene.json").read_bytes()
    stamps = {p: (one / p).stat().st_mtime_ns for p in pngs}
    again = _generate(one, *SMALL, "--jobs", "1")
    assert again.returncode == 0, again.stderr
    assert (one / "scene.json").read_bytes() == manifest
    assert {p: (one / p).stat().st_mtime_ns for p in pngs} == stamps

    other = _generate(one, *SMALL[:-1], "3")  # --short-frames 3
    assert other.returncode == 2
    assert "remove it or choose another folder" in other.stderr

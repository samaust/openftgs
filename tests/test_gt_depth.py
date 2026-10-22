# SPDX-License-Identifier: Apache-2.0
"""`scripts/gt_depth.py`, expected depth and the geometry residual of a trained model against
openftgs-synth's ground-truth depth (#328; E2, E6, D-27; Q-226 to Q-231).

Added tests, not §3 rows: the EXR reader against files `exr_writer.py` writes (source log 212,
213) and against the dataset's files; one opaque primitive at a known camera-space z renders that
z as expected depth (E6 with a single term, Q-226); the residual of a primitive on a known surface
is 0 and its sign says in front or behind (Q-230); the edge exclusion (Q-228); the whole script on
a two-primitive scene with its ground truth (Q-229, Q-231), and its exit 2 when the scene and the
ground truth don't match.
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from builders import logit_of, model_from_rows, pinhole, write_scene
from exr_writer import write_exr, zip_block
from PIL import Image

from openftgs.config import TrainConfig, config_to_dict
from openftgs.io import save_checkpoint
from openftgs.types import Camera, TimeNormalizer, TrainerState

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gt_depth.py"
UNIT_TIME = TimeNormalizer(0.0, 1.0)


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("gt_depth", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("compression", [3, 2, 0])
def test_exr_reader_reads_back_written_files_exactly(tmp_path: Path, compression: int) -> None:
    """ZIP (16-line blocks, the last one short), ZIPS and uncompressed files of 37×21 depths,
    with zeros (no surface) and a smooth ramp, read back bit for bit."""
    g = np.random.default_rng(0)
    values = np.linspace(0.5, 9.0, 37 * 21, dtype=np.float32).reshape(21, 37)
    values[3:6, 4:9] = 0.0
    values[10] += g.normal(0.0, 0.01, 37).astype(np.float32)
    path = tmp_path / "depth.exr"
    write_exr(path, values, compression=compression)
    read = _module().read_exr(path)
    assert read.dtype == np.float32 and read.shape == (21, 37)
    assert np.array_equal(read, values)


def test_exr_reader_takes_a_rawzip_block(tmp_path: Path) -> None:
    """A ZIP block stored raw because compressing it didn't make it smaller is read as raw
    (the encoder's rule: whichever is smaller)."""
    values = np.random.default_rng(1).random((5, 7), dtype=np.float32)  # incompressible
    raw = values.astype("<f4").tobytes()
    assert zip_block(raw) == raw  # one block, stored raw
    path = tmp_path / "noise.exr"
    write_exr(path, values)
    assert np.array_equal(_module().read_exr(path), values)


def test_exr_reader_refuses_other_channels_and_tiles(tmp_path: Path) -> None:
    """A file whose channel isn't Y, and a tiled file, are refused with ValueError."""
    path = tmp_path / "rgb.exr"
    write_exr(path, np.zeros((2, 2), np.float32), channel="R")
    with pytest.raises(ValueError, match="channel"):
        _module().read_exr(path)
    write_exr(path, np.zeros((2, 2), np.float32))
    data = bytearray(path.read_bytes())
    data[5] |= 0x02  # version bit 9: single-part tiled
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="scanline"):
        _module().read_exr(path)


def test_exr_reader_refuses_a_truncated_file(tmp_path: Path) -> None:
    """A file cut short raises ValueError, not an error of the parser's own, so that the script
    exits 2."""
    path = tmp_path / "cut.exr"
    write_exr(path, np.ones((20, 9), np.float32))
    path.write_bytes(path.read_bytes()[:200])
    with pytest.raises(ValueError, match="EXR"):
        _module().read_exr(path)


@pytest.mark.dataset
def test_exr_reader_reads_ground_truth_depth() -> None:
    """openftgs-synth's depth files read as float32 at the camera's size, finite and never
    negative, in metres: indoors (kitchen) every pixel sees a surface a few metres away; the
    outdoor scene (courtyard) shows a little sky, depth 0, at the top of camera 0015's frame."""
    root = Path(os.environ["OPENFTGS_DATASETS"]) / "synth" / "ground_truth"
    read_exr = _module().read_exr
    kitchen = read_exr(root / "kitchen" / "depth" / "cam00" / "000000.exr")
    assert kitchen.shape == (1014, 1352) and kitchen.dtype == np.float32
    assert np.isfinite(kitchen).all() and (kitchen > 0).all() and kitchen.max() < 20
    courtyard = read_exr(root / "courtyard" / "depth" / "0015" / "000000.exr")
    assert courtyard.shape == (1080, 1920)
    assert np.isfinite(courtyard).all() and (courtyard >= 0).all()
    sky = np.nonzero(courtyard == 0)[0]
    assert sky.size > 0 and sky.max() < courtyard.shape[0] // 10


# Expected depth and the geometry residual on one primitive.


def _camera() -> Camera:
    return pinhole(48, 32, fx=40.0, fy=40.0, cx=24.0, cy=16.0, name="c")  # looking down +z


def _one_primitive(z: float, *, x: float = 0.0) -> Any:
    """An opaque primitive (σ near 1, always present) of scale 0.15 at (x, 0, z), whose colour
    isn't its depth."""
    row = {"mean": (x, 0.0, z), "scale": 0.15, "logit": logit_of(0.999), "sh0": (1.0, -0.5, 0.3)}
    return model_from_rows([row], time=UNIT_TIME)


@pytest.mark.parametrize("backend", ["reference", pytest.param("gsplat", marks=pytest.mark.gpu)])
def test_expected_depth_of_one_opaque_primitive_is_its_z(backend: str) -> None:
    """Q-226, E6 with one term: at the primitive's centre the expected depth equals its
    camera-space z, whatever its colour, on both backends."""
    device = "cuda" if backend == "gsplat" else "cpu"
    model = _one_primitive(2.5).to(device)
    out = _module().ExpectedDepthRenderer(model)(_camera(), 0.5, backend=backend)
    depth, alpha = out.expected_depth.cpu(), out.alpha.cpu()
    assert float(alpha[16, 24]) > 0.9
    assert float(depth[16, 24]) == pytest.approx(2.5, rel=1e-5)


def test_residual_is_zero_on_the_surface_and_signed_off_it() -> None:
    """Q-230: on a ground-truth plane at z = 3, a primitive on it has residual 0, one at z = 2.9
    −0.1 (in front), one at z = 3.2 +0.2 (behind); a centre whose pixel's neighbour has no
    surface is skipped, and so is one outside the image."""
    module = _module()
    camera = _camera()
    gt = np.full((32, 48), 3.0, dtype=np.float32)
    gt[:, 40:] = 0.0  # sky on the right
    means = torch.tensor(
        [[0.0, 0.0, 3.0], [0.1, 0.05, 2.9], [-0.2, 0.0, 3.2], [1.13, 0.0, 3.0], [5.0, 0.0, 3.0]]
    )
    r, rows, pixels = module.residuals(means, camera, gt)
    assert rows.tolist() == [0, 1, 2]  # the fourth lands on column 39, next to the sky
    assert pixels.shape == (3, 2)
    assert module.residuals(means[3:4], camera, np.full_like(gt, 3.0))[0].tolist() == [0.0]
    assert r.tolist() == pytest.approx([0.0, -0.1, 0.2], abs=1e-6)


def test_edge_exclusion_drops_pixels_next_to_jumps_and_sky() -> None:
    """Q-228: pixels within 1 px of a relative depth jump above 5 % or of a no-surface pixel are
    left out; a 4 % step and pixels further away are kept."""
    gt = np.full((9, 12), 2.0, dtype=np.float32)
    gt[:, 6:] = 2.5  # a 25 % jump between columns 5 and 6
    gt[0:3, 0:3] = 2.08  # a 4 % step
    gt[8, 11] = 0.0
    keep = _module().away_from_edges(gt)
    assert not keep[:, 5:7].any()
    assert keep[:, 3:5].all() and keep[0:7, 8:11].all()
    assert keep[0:2, 0:2].all() and keep[2, 2] and keep[3, 3]
    assert not keep[7, 10] and not keep[8, 11] and not keep[7, 11]


# The whole script.


def _ground_truth_scene(root: Path) -> tuple[Path, Path, Path]:
    """A checkpoint, scene and ground-truth folder: camera c sees a plane at z = 2.5 with its left
    half dynamic; the model holds an opaque primitive on the plane at the image centre (static)
    and one 0.1 in front of it on the left (dynamic). S = 1, so δ = 0.02."""
    camera = _camera()
    trainer = pinhole(48, 32, fx=40.0, fy=40.0, cx=24.0, cy=16.0, name="t")
    rows = [
        {"mean": (0.0, 0.0, 2.5), "scale": 0.15, "logit": logit_of(0.999), "sh0": (1.0, 0, 0)},
        {"mean": (-0.6, 0.0, 2.4), "scale": 0.15, "logit": logit_of(0.999), "sh0": (0, 1.0, 0)},
    ]
    model = model_from_rows(rows, time=UNIT_TIME)
    scene = write_scene(root / "scene", model, [(trainer, 0.5, "train"), (camera, 0.5, "test")])
    state = TrainerState(
        iteration=7,
        config=config_to_dict(TrainConfig(sh_degree=0)),
        time=UNIT_TIME,
        spatial_scale=1.0,
        frame_dt=0.1,
        params={k: v.detach() for k, v in model.named_parameters()},
        optim={},
        stats={},
        rng={},
        sampler={},
    )
    checkpoint = save_checkpoint(state, root / "checkpoint")
    gt = root / "gt"
    entry = {
        "name": "c",
        "width": 48,
        "height": 32,
        "K_continuous": camera.K.tolist(),
        "world_to_camera": camera.world_to_camera.tolist(),
        "frames": [{"n": 0, "t": 0.0}, {"n": 1, "t": 0.5}],
    }
    gt.mkdir()
    (gt / "cameras.json").write_text(json.dumps({"cameras": [entry]}))
    (gt / "depth" / "c").mkdir(parents=True)
    write_exr(gt / "depth" / "c" / "000001.exr", np.full((32, 48), 2.5, np.float32))
    mask = np.zeros((32, 48), np.uint8)
    mask[:, :24] = 255
    (gt / "masks" / "c").mkdir(parents=True)
    Image.fromarray(mask, "L").save(gt / "masks" / "c" / "000001.png")
    return checkpoint, scene, gt


def test_run_scores_each_test_frame_against_its_ground_truth(tmp_path: Path) -> None:
    """Q-231, Q-229: the test record is matched to frame n = 1 by its time; the primitive on the
    plane has residual 0 (static), the one in front −0.1 (dynamic), beyond δ at each of ½, 1 and
    2 times it; the pooled medians are bin centres; the plane has no edges; the expected depth
    lies between 2.4 and 2.5 wherever alpha ≥ 0.5."""
    checkpoint, scene, gt = _ground_truth_scene(tmp_path)
    out = tmp_path / "GT_DEPTH.json"
    module = _module()
    assert module.main([str(checkpoint), str(scene), str(gt), "--out", str(out),
                        "--backend", "reference"]) == 0  # fmt: skip
    result = json.loads(out.read_text())
    assert result["iteration"] == 7 and result["delta"] == pytest.approx(0.02)
    (frame,) = result["frames"]
    assert (frame["camera"], frame["n"], frame["timestamp"]) == ("c", 1, 0.5)
    residual = frame["residual"]
    assert residual["all"]["primitives"] == 2
    assert residual["static"]["median_abs"] == pytest.approx(0.0, abs=1e-6)
    assert residual["dynamic"]["median_abs"] == pytest.approx(0.1, abs=1e-6)
    assert residual["static"]["share_in_front"] == {"0.5": 0.0, "1.0": 0.0, "2.0": 0.0}
    assert residual["dynamic"]["share_in_front"] == {"0.5": 1.0, "1.0": 1.0, "2.0": 1.0}
    assert residual["all"]["share_in_front"]["1.0"] == pytest.approx(0.5)
    assert residual["all"]["share_behind"]["1.0"] == 0.0
    pooled = result["overall"]["residual"]
    assert pooled["static"]["median_abs"] == pytest.approx(0.00005)
    assert pooled["dynamic"]["median_abs"] == pytest.approx(0.1, abs=1e-4)  # a 0.1 mm bin
    depth = frame["expected_depth"]
    assert depth["edges"]["pixels"] == 0
    assert depth["all"]["pixels"] == depth["static"]["pixels"] + depth["dynamic"]["pixels"] > 0
    assert -0.1 - 1e-5 <= depth["all"]["bias"] <= 1e-5
    assert depth["static"]["median_abs"] == pytest.approx(0.0, abs=1e-5)
    assert result["overall"]["expected_depth"]["all"]["pixels"] == depth["all"]["pixels"]


@pytest.mark.parametrize("change", ["K", "pose", "time", "size", "camera"])
def test_run_is_exit_2_when_the_scene_doesnt_match_the_ground_truth(
    tmp_path: Path, change: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-231: a test record whose camera's K, pose or size differs from the ground truth's, whose
    time matches no frame, or whose camera the ground truth lacks, stops the run with exit 2."""
    checkpoint, scene, gt = _ground_truth_scene(tmp_path)
    cameras = json.loads((gt / "cameras.json").read_text())
    entry = cameras["cameras"][0]
    if change == "K":
        entry["K_continuous"][0][0] += 1e-3
    elif change == "pose":
        entry["world_to_camera"][0][3] += 1e-3
    elif change == "time":
        entry["frames"][1]["t"] = 0.5 + 1e-5
    elif change == "size":
        entry["width"] = 47
    else:
        entry["name"] = "d"
    (gt / "cameras.json").write_text(json.dumps(cameras))
    out = tmp_path / "GT_DEPTH.json"
    assert _module().main([str(checkpoint), str(scene), str(gt), "--out", str(out),
                           "--backend", "reference"]) == 2  # fmt: skip
    assert "gt_depth: error:" in capsys.readouterr().err and not out.exists()

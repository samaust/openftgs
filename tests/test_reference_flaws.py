# SPDX-License-Identifier: Apache-2.0
"""`scripts/reference_flaws.py`, the pixels of openftgs-synth's reference images no model could
match and each model's PSNR without them (#329; E16, D-67; Q-233 to Q-237).

Added tests, not §3 rows: a frame warped into its own camera with its own depth comes back
exactly on every pixel with a surface, and a texture linear on a plane comes back exactly in a
second camera (the warp's definition); the kept-pixel PSNR with no pixel flagged equals E16's
PSNR (Q-233); the clipping shares of an image with a known count at 255; the whole script on a
three-camera scene with planted flaws (Q-234 to Q-237), and its exit 2 when the scene and the
ground truth don't match.
"""

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from builders import look_at, model_from_rows, pinhole
from PIL import Image
from test_gt_depth import write_exr

from openftgs.config import TrainConfig, config_to_dict
from openftgs.eval import psnr
from openftgs.io import save_checkpoint
from openftgs.types import Camera, TimeNormalizer, TrainerState

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _module() -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))  # for its `import gt_depth`
    # Imported by name, so its dataclasses and the spawn pool's workers find it in sys.modules.
    return importlib.import_module("reference_flaws")


def _arrays(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    return camera.K.double().numpy(), camera.world_to_camera.double().numpy()


def test_warp_into_its_own_camera_reproduces_the_frame() -> None:
    """The warp's definition: back-projecting each pixel centre at its own depth and projecting
    it into the same camera lands on that centre, so every pixel with a surface comes back
    exactly; pixels without one (depth 0) are left out."""
    camera = look_at((1.0, -2.5, 1.2), (0.1, 0.2, 0.3), width=37, height=23, focal=31.7)
    K, w2c = _arrays(camera)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(23, 37, 3), dtype=np.uint8)
    depth = rng.uniform(0.5, 6.0, size=(23, 37)).astype(np.float32)
    depth[rng.random((23, 37)) < 0.2] = 0.0
    out = _module().warp(frame, depth, K, w2c, K, w2c)
    surface = depth > 0
    assert np.array_equal(out.inside, surface)
    assert np.array_equal(out.image[surface], frame[surface].astype(np.float64))
    xs, ys = out.pixel
    v, u = np.nonzero(surface)
    assert np.array_equal(xs[surface], u) and np.array_equal(ys[surface], v)
    np.testing.assert_allclose(out.z[surface], depth[surface], rtol=1e-12)


def test_warp_carries_a_plane_into_a_second_camera() -> None:
    """Two cameras 0.3 apart look at the plane z = 2 (world), whose colour is linear in world x
    and y; each frame holds that colour at its pixel centres. Bilinear sampling reproduces a
    linear function, so the second camera's frame warped into the first gives the first's
    colours wherever the point lands inside it, and the point's z there is 2."""
    w, h, f = 40, 30, 35.0
    target = pinhole(w, h, fx=f, fy=f, cx=20.0, cy=15.0)
    shifted = torch.eye(4)
    shifted[0, 3] = -0.3  # camera centre at x = 0.3
    source = pinhole(w, h, fx=f, fy=f, cx=20.0, cy=15.0, world_to_camera=shifted)

    def frame(centre_x: float) -> np.ndarray:
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        x = (u - 20.0) / f * 2.0 + centre_x
        y = (v - 15.0) / f * 2.0
        return np.stack([10.0 + 50.0 * x, 100.0 + 40.0 * y, 30.0 + 20.0 * x + 10.0 * y], -1)

    depth = np.full((h, w), 2.0, np.float32)
    offset = -float(shifted[0, 3])  # 0.3 as float32 stores it, as the camera has it
    out = _module().warp(frame(offset), depth, *_arrays(target), *_arrays(source))
    inside = out.inside
    assert inside.sum() == h * (w - 6) and not inside[:, :6].any()  # 0.3 m = 5.25 px
    np.testing.assert_allclose(out.image[inside], frame(0.0)[inside], atol=1e-9)
    np.testing.assert_allclose(out.z[inside], 2.0, atol=1e-6)


def test_clipped_counts_the_pixels_at_255() -> None:
    """A 10×8 image with 7 pixels at 255 in one channel only and 3 at 255 in all three."""
    image = np.full((8, 10, 3), 200, np.uint8)
    for k in range(7):
        image[k, k, k % 3] = 255
    image[7, 7:10] = 255
    any_255, all_255 = _module().clipped(image)
    assert any_255.sum() == 10 and all_255.sum() == 3
    assert all_255[7, 7:10].all() and not all_255[:7].any()


def test_kept_psnr_with_no_pixel_flagged_is_e16s_psnr() -> None:
    """Q-233: with every pixel kept, the kept-pixel PSNR equals E16's (the render clamped to
    [0, 1] first, here above 1 in places); on a subset it is the PSNR of that subset's MSE over
    its three channels; with nothing kept there is none."""
    gen = torch.Generator().manual_seed(1)
    pred = torch.rand(9, 11, 3, generator=gen) * 1.3
    gt = torch.rand(9, 11, 3, generator=gen)
    kept_psnr = _module().kept_psnr
    everything = torch.ones(9, 11, dtype=torch.bool)
    assert kept_psnr(pred, gt, everything) == pytest.approx(float(psnr(pred, gt)), abs=1e-9)
    keep = torch.zeros(9, 11, dtype=torch.bool)
    keep[2:5, 3:8] = True
    diff = pred.double().clamp(0, 1)[2:5, 3:8] - gt.double()[2:5, 3:8]
    expected = -10.0 * math.log10(float(diff.square().mean()))
    assert kept_psnr(pred, gt, keep) == pytest.approx(expected, abs=1e-9)
    assert kept_psnr(pred, gt, torch.zeros_like(keep)) is None
    assert kept_psnr(gt, gt, keep) == math.inf


# A three-camera scene: cameras a, b, c on the x axis 0.2 apart, looking down +z at a plane at
# depth 2 with one colour everywhere; b is the test camera, c's frames are 4 ms late (Q-237).
_W, _H, _F = 40, 30, 40.0
_BASE = (100, 120, 140)
_FRAMES = 5
_CLIP = (slice(10, 14), slice(10, 14))  # b, frame 0: at 255, in no other camera or frame
_FLICKER = (slice(20, 24), slice(25, 29))  # b, frame 2: 5 levels brighter
_MOVER = (slice(0, 5), slice(0, 5))  # b, frame 4: dynamic


def _camera(name: str, x: float) -> Camera:
    w2c = torch.eye(4)
    w2c[0, 3] = -x
    return pinhole(_W, _H, fx=_F, fy=_F, cx=_W / 2, cy=_H / 2, world_to_camera=w2c, name=name)


def _flawed_scene(root: Path) -> tuple[Path, Path, Path]:
    """The manifest, the ground-truth folder and a checkpoint of one primitive in front of b."""
    cameras = {"a": (_camera("a", -0.2), 0.0), "b": (_camera("b", 0.0), 0.0)}
    cameras["c"] = (_camera("c", 0.2), 0.004)
    gt, images = root / "gt", root / "scene"
    entries, gt_cameras = [], []
    for name, (camera, offset) in cameras.items():
        frames = []
        for n in range(_FRAMES):
            t = n / 30 + offset
            frames.append({"n": n, "t": t})
            pixels = np.empty((_H, _W, 3), np.uint8)
            pixels[:] = _BASE
            mask = np.zeros((_H, _W), np.uint8)
            if name == "b" and n == 0:
                pixels[_CLIP] = 255
            if name == "b" and n == 2:
                pixels[_FLICKER] += 5
            if name == "b" and n == 4:
                mask[_MOVER] = 255
            path = f"images/{name}/{n:06d}.png"
            (images / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels, "RGB").save(images / path)
            split = "test" if name == "b" else "train"
            entries.append(
                {"camera": name, "timestamp": t, "frame_index": n, "path": path, "split": split}
            )
            for kind in ("depth", "masks"):
                (gt / kind / name).mkdir(parents=True, exist_ok=True)
            write_exr(gt / "depth" / name / f"{n:06d}.exr", np.full((_H, _W), 2.0, np.float32))
            Image.fromarray(mask, "L").save(gt / "masks" / name / f"{n:06d}.png")
        gt_cameras.append(
            {
                "name": name,
                "width": _W,
                "height": _H,
                "K_continuous": camera.K.tolist(),
                "world_to_camera": camera.world_to_camera.tolist(),
                "frames": frames,
            }
        )
    (gt / "cameras.json").write_text(json.dumps({"cameras": gt_cameras}))
    manifest = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": {
            name: {
                "width": _W,
                "height": _H,
                "model": "pinhole",
                "K": camera.K.tolist(),
                "world_to_camera": camera.world_to_camera.tolist(),
            }
            for name, (camera, _) in cameras.items()
        },
        "images": entries,
        "metadata": {"source": "synthetic"},
    }
    (images / "scene.json").write_text(json.dumps(manifest))
    time = TimeNormalizer(0.0, 0.2)
    row = {"mean": (0.0, 0.0, 2.0), "scale": 0.3, "logit": 2.0, "sh0": (0.0, 0.2, 0.4)}
    model = model_from_rows([row], time=time)
    state = TrainerState(
        iteration=3,
        config=config_to_dict(TrainConfig(sh_degree=0)),
        time=time,
        spatial_scale=1.0,
        frame_dt=1 / 6,
        params={k: v.detach() for k, v in model.named_parameters()},
        optim={},
        stats={},
        rng={},
        sampler={},
    )
    checkpoint = save_checkpoint(state, root / "checkpoint")
    return images / "scene.json", gt, checkpoint


def _flags(out: Path, n: int) -> np.ndarray:
    return np.asarray(Image.open(out / "flags" / "b" / f"{n:06d}.png"))


def test_scene_flags_the_planted_flaws(tmp_path: Path) -> None:
    """Q-234 to Q-237 on the three-camera scene: b's neighbours are a and c; b's clipped patch
    at frame 0 is flagged clipped, cross-camera (against a at the same instant and c at its
    nearest frame, static pixels only) and flicker; frame 2's 5-level patch is flagged as
    flicker at every scale (it changes against both adjacent frames) and as cross-camera at
    none (5 levels isn't above ½ × 10); the frames next to it aren't flagged, since each
    matches its other neighbour; frame 4's mover is left out of the static pixels, dilated by
    1 px."""
    scene, gt, _ = _flawed_scene(tmp_path)
    module = _module()
    out = tmp_path / "reference"
    assert module.main(["scene", str(scene), str(gt), "--out", str(out)]) == 0
    result = json.loads((out / "REFERENCE.json").read_text())
    assert result["neighbours"]["b"] == ["a", "c"] and result["test_cameras"] == ["b"]
    index = json.loads((out / "flags" / "index.json").read_text())
    assert [(e["camera"], e["n"]) for e in index] == [("b", n) for n in range(_FRAMES)]
    bits = module.FLAG_BITS
    union = module.union_bits(1.0)
    planted = {0: _CLIP, 2: _FLICKER}
    for n in range(_FRAMES):
        flags = _flags(out, n)
        rest = np.ones((_H, _W), bool)
        if n in planted:
            rest[planted[n]] = False
        assert not (flags[rest] & union).any(), n
    every = (bits["clipped"] | bits["cross"] | bits["flicker"]) & union
    assert ((_flags(out, 0)[_CLIP] & every) == every).all()
    flicker_bits = bits["flicker"] | bits["flicker_0.5"] | bits["flicker_2.0"]
    assert (_flags(out, 2)[_FLICKER] == flicker_bits).all()
    frames = result["cameras"]["b"]["frames"]
    assert frames[0]["clipped"] == {"any": 16 / (_W * _H), "all": 16 / (_W * _H)}
    assert frames[0]["flagged"]["clipped"] == 16 / (_W * _H)
    from_a, from_c = frames[4]["cross"]["a"], frames[4]["cross"]["c"]
    assert from_a["same_instant"] and from_a["source_n"] == 4
    assert from_a["regions"]["all"]["mse"] == 0.0 and from_a["regions"]["all"]["psnr"] is None
    assert from_a["regions"]["all"]["pixels"] == _H * (_W - 4)  # 0.2 m = 4 px
    assert from_a["regions"]["all"]["share_of_surface"] == pytest.approx((_W - 4) / _W)
    assert from_a["regions"]["dynamic"]["pixels"] == 36  # the mover, dilated by 1 px
    assert not from_c["same_instant"] and from_c["source_n"] == 4
    assert from_c["dt"] == pytest.approx(0.004)
    assert from_c["regions"]["all"] is None and from_c["regions"]["dynamic"] is None
    # Q-237: c sees columns 4 and up, so 2 columns of the dilated mover are left out
    assert from_c["regions"]["static"]["pixels"] == _H * (_W - 4) - 2 * 6
    assert frames[1]["flicker"]["static"]["share_above"]["1.0"] == 16 / (_W * _H)
    assert frames[3]["flicker"]["next_n"] == 4 and frames[3]["flicker"]["dynamic"]["pixels"] == 36
    assert frames[4]["flicker"] is None
    clipped = result["cameras"]["b"]["pooled"]["clipped"]["any"]
    assert clipped == pytest.approx(16 / (_W * _H) / _FRAMES)
    maps = np.load(out / "maps" / "b.npz")
    assert maps["clipped_any_share"][_CLIP] == pytest.approx(1 / _FRAMES)
    assert (out / "maps" / "b_flicker.png").exists()


def test_run_reports_psnr_with_and_without_the_flagged_pixels(tmp_path: Path) -> None:
    """Q-233 at the script's level: each test frame's PSNR is E16's, and wherever a flag marks
    no pixel, the kept-pixel PSNR equals it; frame 0's clipped share is its 16 pixels."""
    scene, gt, checkpoint = _flawed_scene(tmp_path)
    module = _module()
    flags = tmp_path / "reference"
    assert module.main(["scene", str(scene), str(gt), "--out", str(flags)]) == 0
    out = tmp_path / "REFERENCE_PSNR.json"
    assert module.main(["run", str(checkpoint), str(scene), str(flags / "flags"),
                        "--out", str(out), "--backend", "reference"]) == 0  # fmt: skip
    result = json.loads(out.read_text())
    assert result["iteration"] == 3 and len(result["frames"]) == _FRAMES
    for frame in result["frames"]:
        for name in ("clipped", "cross", "flicker", "union"):
            if frame[name]["share"] == 0.0:
                assert frame[name]["kept_psnr"] == pytest.approx(frame["psnr"], abs=1e-9)
                assert frame[name]["flagged_psnr"] is None
    first = result["frames"][0]
    assert first["clipped"]["share"] == 16 / (_W * _H)
    assert first["clipped"]["kept_psnr"] > first["psnr"]  # the clipped patch is far off
    overall = result["overall"]
    assert overall["frames"] == _FRAMES
    assert overall["psnr"] == pytest.approx(np.mean([f["psnr"] for f in result["frames"]]))


def test_scene_is_exit_2_when_the_scene_doesnt_match_the_ground_truth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-231 as `gt_depth.py` applies it: a camera whose K differs from the ground truth's."""
    scene, gt, _ = _flawed_scene(tmp_path)
    cameras = json.loads((gt / "cameras.json").read_text())
    cameras["cameras"][1]["K_continuous"][0][0] += 1e-3
    (gt / "cameras.json").write_text(json.dumps(cameras))
    out = tmp_path / "reference"
    assert _module().main(["scene", str(scene), str(gt), "--out", str(out)]) == 2
    assert "reference_flaws: error:" in capsys.readouterr().err
    assert not (out / "REFERENCE.json").exists()

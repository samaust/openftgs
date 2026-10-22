# SPDX-License-Identifier: Apache-2.0
"""`scripts/gt_depth.py`, expected depth and the geometry residual of a trained model against
openftgs-synth's ground-truth depth (#328; E2, E6, D-27; Q-226 to Q-231).

Added tests, not §3 rows: the EXR reader against files written here from the OpenEXR file-format
documentation (source log); one ground-truth file read; one opaque primitive at a known
camera-space z renders that z as expected depth (E6 with a single term, Q-226); the residual of a
primitive on a known surface is 0 and its sign says in front or behind (Q-229, Q-230); the edge
exclusion (Q-228).
"""

import importlib.util
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from builders import logit_of, model_from_rows, pinhole

from openftgs.types import Camera, TimeNormalizer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gt_depth.py"
UNIT_TIME = TimeNormalizer(0.0, 1.0)


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("gt_depth", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A writer for single-part scanline EXR files with one FLOAT channel, from the OpenEXR file
# layout documentation (magic, version, attributes, line offset table, scan line blocks) and
# ZIP's byte split and delta step as the encoder applies them (source log).


def _attribute(name: str, type_name: str, value: bytes) -> bytes:
    return (
        name.encode() + b"\0" + type_name.encode() + b"\0" + struct.pack("<i", len(value)) + value
    )


def _zip_block(raw: bytes) -> bytes:
    """The ZIP transform: even bytes then odd bytes, then each byte stored as its difference
    from the previous one plus 128 (mod 256), then zlib; the raw bytes when that isn't
    smaller."""
    split = raw[0::2] + raw[1::2]
    deltas = bytes(
        [split[0]] + [(split[i] - split[i - 1] + 128) % 256 for i in range(1, len(split))]
    )
    packed = zlib.compress(deltas)
    return packed if len(packed) < len(raw) else raw


def write_exr(path: Path, values: np.ndarray, *, compression: int = 3, channel: str = "Y") -> None:
    height, width = values.shape
    lines = {0: 1, 2: 1, 3: 16}[compression]  # NO_COMPRESSION, ZIPS, ZIP
    chlist = channel.encode() + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) + b"\0"
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    header = b"\x76\x2f\x31\x01" + struct.pack("<i", 2)
    header += _attribute("channels", "chlist", chlist)
    header += _attribute("compression", "compression", bytes([compression]))
    header += _attribute("dataWindow", "box2i", box)
    header += _attribute("displayWindow", "box2i", box)
    header += _attribute("lineOrder", "lineOrder", b"\0")
    header += _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"
    blocks = []
    for y in range(0, height, lines):
        raw = values[y : y + lines].astype("<f4").tobytes()
        data = raw if compression == 0 else _zip_block(raw)
        blocks.append(struct.pack("<ii", y, len(data)) + data)
    offset = len(header) + 8 * len(blocks)
    table = b""
    for block in blocks:
        table += struct.pack("<Q", offset)
        offset += len(block)
    path.write_bytes(header + table + b"".join(blocks))


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


def test_exr_reader_takes_a_raw_zip_block(tmp_path: Path) -> None:
    """A ZIP block stored raw because compressing it didn't make it smaller is read as raw
    (the encoder's rule: whichever is smaller)."""
    values = np.random.default_rng(1).random((5, 7), dtype=np.float32)  # incompressible
    raw = values.astype("<f4").tobytes()
    assert _zip_block(raw) == raw  # one block, stored raw
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


@pytest.mark.dataset
def test_exr_reader_reads_ground_truth_depth() -> None:
    """openftgs-synth's depth files read as float32 at the camera's size, finite and never
    negative, in metres: indoors (kitchen) every pixel sees a surface a few metres away; the
    outdoor scene (courtyard) shows a little sky, depth 0, in camera 0015."""
    root = Path(os.environ["OPENFTGS_DATASETS"]) / "synth" / "ground_truth"
    read_exr = _module().read_exr
    kitchen = read_exr(root / "kitchen" / "depth" / "cam00" / "000000.exr")
    assert kitchen.shape == (1014, 1352) and kitchen.dtype == np.float32
    assert np.isfinite(kitchen).all() and (kitchen > 0).all() and kitchen.max() < 20
    courtyard = read_exr(root / "courtyard" / "depth" / "0015" / "000000.exr")
    assert courtyard.shape == (1080, 1920)
    assert np.isfinite(courtyard).all() and (courtyard >= 0).all()
    assert 0 < (courtyard == 0).mean() < 1


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
    out = _module().depth_render(model, _camera(), 0.5, backend=backend)
    depth, alpha = out.depth.cpu(), out.alpha.cpu()
    assert float(alpha[16, 24]) > 0.9
    assert float(depth[16, 24]) == pytest.approx(2.5, rel=1e-5)


def test_residual_is_zero_on_the_surface_and_signed_off_it() -> None:
    """Q-229, Q-230: on a ground-truth plane at z = 3, a primitive on it has residual 0, one at
    z = 2.9 −0.1 (in front), one at z = 3.2 +0.2 (behind); a centre whose pixel or neighbour has
    no surface is skipped, and so is one outside the image."""
    module = _module()
    camera = _camera()
    gt = np.full((32, 48), 3.0, dtype=np.float32)
    gt[:, 40:] = 0.0  # sky on the right
    means = torch.tensor(
        [[0.0, 0.0, 3.0], [0.1, 0.05, 2.9], [-0.2, 0.0, 3.2], [1.2, 0.0, 3.0], [5.0, 0.0, 3.0]]
    )
    r, rows, pixels = module.residuals(means, camera, gt)
    assert rows.tolist() == [0, 1, 2]  # the fourth projects next to the sky, the fifth off-image
    assert r.tolist() == pytest.approx([0.0, -0.1, 0.2], abs=1e-6)
    assert pixels.shape == (3, 2)


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

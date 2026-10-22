# SPDX-License-Identifier: Apache-2.0
"""EXR files for the tests of the scripts that read openftgs-synth's ground-truth depth
(`scripts/gt_depth.py`, `scripts/reference_flaws.py`): a writer of single-part scanline files
with one FLOAT channel, uncompressed, ZIPS or ZIP."""

import struct
import zlib
from pathlib import Path

import numpy as np

# A writer for single-part scanline EXR files with one FLOAT channel, from the OpenEXR file
# layout documentation (magic, version, attributes, line offset table, scan line blocks) and
# ZIP's byte split and delta step as OpenEXR's src/lib/OpenEXRCore/internal_zip.c at commit
# 6a75faa5f35786c5a4f6f3a2cdcfcc26496b3f54 applies them (BSD-3-Clause, Copyright Contributors to
# the OpenEXR Project), described there and written here anew (source log 212, 213).


def _attribute(name: str, type_name: str, value: bytes) -> bytes:
    return (
        name.encode() + b"\0" + type_name.encode() + b"\0" + struct.pack("<i", len(value)) + value
    )


def zip_block(raw: bytes) -> bytes:
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
        data = raw if compression == 0 else zip_block(raw)
        blocks.append(struct.pack("<ii", y, len(data)) + data)
    offset = len(header) + 8 * len(blocks)
    table = b""
    for block in blocks:
        table += struct.pack("<Q", offset)
        offset += len(block)
    path.write_bytes(header + table + b"".join(blocks))

# SPDX-License-Identifier: Apache-2.0
"""A window of an openftgs-synth scene around its test view's fastest frames (Q-219; #317).

    python scripts/synth_window.py GT_SCENE_DIR --camera NAME [--length L] [--layout LAYOUT]

GT_SCENE_DIR is `ground_truth/<scene>/` of openftgs-synth; NAME is the test view's camera name
in LAYOUT (default `selfcap`), mapped to the ground truth's camera through `cameras.json`'s
`layout_names`. The script reads that camera's ground-truth frames, `pinhole/<cam>/<n:06d>.png`
for every decode index n, finds D-69's 10 consecutive frames whose consecutive frames differ
most with `openftgs.data.fastest_frames`, and writes to stdout, as JSON, the window of L frames
(default 60) centred on them and clipped to the scene's frames:

    {"camera": NAME, "frames": F, "fastest_10": [s, s + 10], "window": [n0, n0 + L]}

with n0 = min(max(s + 5 − ⌊L/2⌋, 0), F − L), so the window holds the range. M17 converts
`studio` on this window (`ftgs convert selfcap --window n0:n1`). Exit 0 once written, 2 for
invalid arguments, a camera the scene doesn't have or a missing frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from openftgs.data import fastest_frames

_FASTEST = 10  # D-69: "10-frame with the fastest motion"


def _gt_camera(truth: dict[str, Any], layout: str, name: str) -> dict[str, Any]:
    for camera in truth["cameras"]:
        if camera["layout_names"].get(layout) == name:
            return dict(camera)
    raise ValueError(f"the scene has no {layout} camera {name!r}")


def window(scene: Path, camera: str, length: int, layout: str = "selfcap") -> dict[str, Any]:
    """The window and the fastest range, as the script writes them."""
    truth = json.loads((scene / "cameras.json").read_text())
    frames = int(truth["num_frames"])
    if not _FASTEST <= length <= frames:
        raise ValueError(f"length must be between {_FASTEST} and the scene's {frames} frames")
    folder = scene / "pinhole" / _gt_camera(truth, layout, camera)["name"]

    def images() -> Iterator[torch.Tensor]:
        for n in range(frames):
            path = folder / f"{n:06d}.png"
            if not path.is_file():
                raise ValueError(f"missing ground-truth frame {path}")
            yield torch.from_numpy(np.asarray(Image.open(path).convert("RGB")))

    start = fastest_frames(images(), count=_FASTEST)
    n0 = min(max(start + _FASTEST // 2 - length // 2, 0), frames - length)
    return {
        "camera": camera,
        "frames": frames,
        "fastest_10": [start, start + _FASTEST],
        "window": [n0, n0 + length],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="a window around an openftgs-synth test view's fastest frames (D-69)"
    )
    parser.add_argument("scene", type=Path, help="ground_truth/<scene>/ of openftgs-synth")
    parser.add_argument("--camera", required=True, help="the test view's camera name")
    parser.add_argument("--length", type=int, default=60, help="the window's frames (60)")
    parser.add_argument("--layout", default="selfcap", help="the layout NAME is in (selfcap)")
    args = parser.parse_args(argv)
    try:
        report = window(args.scene, args.camera, args.length, args.layout)
    except (OSError, ValueError, KeyError) as error:
        print(f"synth_window: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

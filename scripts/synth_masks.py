# SPDX-License-Identifier: Apache-2.0
"""An openftgs-synth test view's ground-truth masks, as `ftgs convert selfcap --masks` takes
them (Q-87, D-64; Q-219; #317).

    python scripts/synth_masks.py GT_SCENE_DIR --camera NAME --window N0:N1 OUT [--layout LAYOUT]

GT_SCENE_DIR is `ground_truth/<scene>/` of openftgs-synth; NAME is the test view's camera name
in LAYOUT (default `selfcap`), mapped to the ground truth's camera through `cameras.json`'s
`layout_names`. Writes, in the new or empty folder OUT:

- `<n:06d>.png` for each decode index n in [N0, N1): the ground truth's
  `masks/<cam>/<n:06d>.png`, copied byte for byte, so the masks stay exact. They line up with
  the converted test images only where those are in the ground truth's pinhole camera at full
  size, that is with `--undistort internal --resize 1.0` (Q-219);
- `masks.json`: `{"mask_source": "openftgs-synth <VERSION> ground_truth/<scene>/masks/<cam>
  (exact)", "masks_approximate": false}`, the dataset's version read from its `VERSION`.

Every mask is checked first (present, 8-bit or 1-bit, the camera's width and height), so a
failure writes nothing. Exit 0 once written, 2 for invalid arguments, a camera the scene
doesn't have, a missing or invalid mask, or an OUT that holds something.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

_MODES = ("L", "1")  # the converter's mask modes (Q-87)


def _gt_camera(truth: dict[str, Any], layout: str, name: str) -> dict[str, Any]:
    for camera in truth["cameras"]:
        if camera["layout_names"].get(layout) == name:
            return dict(camera)
    raise ValueError(f"the scene has no {layout} camera {name!r}")


def _window(text: str) -> tuple[int, int]:
    first, sep, last = text.partition(":")
    if not sep or not first.isdigit() or not last.isdigit() or int(first) >= int(last):
        raise argparse.ArgumentTypeError(f"window must be N0:N1 with 0 <= N0 < N1, got {text!r}")
    return int(first), int(last)


def write_masks(
    scene: Path, camera: str, window: tuple[int, int], out: Path, layout: str = "selfcap"
) -> None:
    """Checks the window's masks, then copies them and writes masks.json into `out`."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out} holds something; give a new or empty folder")
    truth = json.loads((scene / "cameras.json").read_text())
    gt = _gt_camera(truth, layout, camera)
    size = (int(gt["width"]), int(gt["height"]))
    folder = scene / "masks" / gt["name"]
    sources = []
    for n in range(*window):
        path = folder / f"{n:06d}.png"
        if not path.is_file():
            raise ValueError(f"missing ground-truth mask {path}")
        with Image.open(path) as image:
            if image.mode not in _MODES or image.size != size:
                raise ValueError(
                    f"{path}: a mask must be 8-bit or 1-bit at {size}, got {image.mode} at "
                    f"{image.size}"
                )
        sources.append(path)
    version = (scene.parent.parent / "VERSION").read_text().strip()
    sidecar = {
        "mask_source": (
            f"openftgs-synth {version} ground_truth/{scene.name}/masks/{gt['name']} (exact)"
        ),
        "masks_approximate": False,  # Q-219: copied from the ground truth, so exact
    }
    out.mkdir(parents=True, exist_ok=True)
    for path in sources:
        shutil.copyfile(path, out / path.name)
    (out / "masks.json").write_text(json.dumps(sidecar, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="an openftgs-synth test view's ground-truth masks for ftgs convert --masks"
    )
    parser.add_argument("scene", type=Path, help="ground_truth/<scene>/ of openftgs-synth")
    parser.add_argument("--camera", required=True, help="the test view's camera name")
    parser.add_argument("--window", required=True, type=_window, help="decode indices N0:N1")
    parser.add_argument("out", type=Path, help="a new or empty folder")
    parser.add_argument("--layout", default="selfcap", help="the layout NAME is in (selfcap)")
    args = parser.parse_args(argv)
    try:
        write_masks(args.scene, args.camera, args.window, args.out, args.layout)
    except (OSError, ValueError, KeyError) as error:
        print(f"synth_masks: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

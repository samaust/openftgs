# SPDX-License-Identifier: Apache-2.0
"""A scene whose cameras each have their own image size, derived from one whose cameras share
one, by cropping (#350; Q-238, Q-232; D-51).

    python scripts/crop_scene.py SCENE.json OUT [--jobs J]

The k-th camera of SCENE.json, in the manifest's camera order (Q-108), loses k columns on the
right and k rows at the bottom of every image: on a Neural3DV-style scene (`cam00` to `cam19`)
camera `cam<i>` is cropped by i pixels, and `cam00`, the test camera, is unchanged. K and the
pose stay as they are, since the principal point doesn't move when only the right and bottom
edges go, and pixels are copied, not resampled: every image is its source's top-left block.
Masks are cropped the same way. On kitchen60 (20 cameras at 1352×1014) that gives 19 training
sizes, from 1351×1013 down to 1333×995, as SelfCap's converter gives each camera the size
COLMAP sets for it.

Writes, under the folder OUT, which must be absent or empty:

- `images/...` and `masks/...`: the PNGs at the paths the source manifest gives them relative to
  its folder, cropped by J processes (default: one per CPU) at the converters' fast PNG level;
  the images of the uncropped camera are copied byte for byte;
- `scene.json`: the ftgs-scene manifest (§2.2), written with `openftgs.io.write_scene_manifest`,
  with the source's records, timestamps, splits and frame indices, and metadata naming the
  source manifest, its SHA-256 and the crop.

Exit 0 once the scene is written and reads back with `MultiViewVideo.from_manifest`; 2 for an
input that can't be read, a camera too small to crop, or an OUT that holds something.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

from openftgs.data import MultiViewVideo
from openftgs.io import read_scene_manifest, write_scene_manifest
from openftgs.types import Camera

_PNG_LEVEL = 1  # as the converters' (openftgs.data, §2.4): PNG is lossless, so pixels don't move


def crops(camera_names: tuple[str, ...]) -> dict[str, int]:
    """The pixels each camera loses on the right and at the bottom: its place in the manifest's
    camera order."""
    return {name: k for k, name in enumerate(camera_names)}


def cropped_camera(camera: Camera, crop: int) -> Camera:
    """`camera` with `crop` fewer columns and rows; K and the pose unchanged."""
    width, height = camera.width - crop, camera.height - crop
    if width < 1 or height < 1:
        raise ValueError(
            f"camera {camera.name!r}: {camera.width}×{camera.height} can't lose {crop} pixels"
        )
    return dataclasses.replace(camera, width=width, height=height)


def _crop_file(source: str, target: str, width: int, height: int) -> None:
    """One PNG: copied when nothing is cropped, otherwise its top-left width × height block,
    written to a temporary sibling and renamed."""
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    temporary = f"{target}.tmp-{os.getpid()}"
    with Image.open(source) as image:
        if image.size == (width, height):
            shutil.copyfile(source, temporary)
        else:
            image.crop((0, 0, width, height)).save(
                temporary, format="PNG", compress_level=_PNG_LEVEL
            )
    os.replace(temporary, target)


def crop_scene(scene: Path, out: Path, *, jobs: int = 0) -> Path:
    """Write the cropped scene of `scene` under `out` and return its manifest's path."""
    manifest = read_scene_manifest(scene)
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} must be absent or empty")
    root = manifest.path.parent
    amount = crops(manifest.camera_names)
    cameras: dict[str, Camera] = {}
    records = []
    masks: list[Path | None] = []
    work: list[tuple[str, str, int, int]] = []
    for record, mask in zip(manifest.records, manifest.masks, strict=True):
        name = record.camera.name
        if name not in cameras:
            cameras[name] = cropped_camera(record.camera, amount[name])
        camera = cameras[name]
        image = out / record.image_path.relative_to(root)
        work.append((str(record.image_path), str(image), camera.width, camera.height))
        records.append(dataclasses.replace(record, camera=camera, image_path=image))
        if mask is None:
            masks.append(None)
        else:
            target = out / mask.relative_to(root)
            work.append((str(mask), str(target), camera.width, camera.height))
            masks.append(target)
    out.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs or os.cpu_count()) as pool:
        for done in concurrent.futures.as_completed(
            [pool.submit(_crop_file, *item) for item in work]
        ):
            done.result()
    metadata = {
        "source": manifest.metadata.get("source"),
        "producer": "scripts/crop_scene.py",
        "derived_from": {
            "manifest": str(manifest.path),
            "sha256": manifest.sha256,
            "crop": (
                "camera k of the manifest's camera order loses k columns on the right and k rows "
                "at the bottom; K and the pose unchanged; pixels copied"
            ),
        },
    }
    path = out / "scene.json"
    write_scene_manifest(
        path, records, time_unit=manifest.time_unit, masks=masks, metadata=metadata
    )
    MultiViewVideo.from_manifest(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="a scene cropped to a size per camera (#350)")
    parser.add_argument("scene", type=Path, metavar="SCENE.json")
    parser.add_argument("out", type=Path, metavar="OUT")
    parser.add_argument("--jobs", type=int, default=0, help="processes; 0: one per CPU")
    args = parser.parse_args(argv)
    if args.jobs < 0:
        parser.error(f"--jobs must be at least 0, got {args.jobs}")
    try:
        path = crop_scene(args.scene, args.out, jobs=args.jobs)
    except (OSError, ValueError) as err:
        print(f"crop_scene: {err}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""The test-view renders of a trained model, as PNGs, through the public CLI (#327; §2.2
"Rendering at any camera and time", E16, D-84; Q-56, Q-125, Q-137, Q-224).

    python scripts/view_renders.py request MODEL SCENE.json --out REQUEST.json
    python scripts/view_renders.py run MODEL SCENE.json DIR [--compare METRICS.json]
                                       [--backend gsplat|reference]

`request` writes a ftgs-render-request with one item per test record, in the manifest's order:
the record's timestamp, its camera's size, K and world_to_camera, the scene's time unit,
`sh_degree: null` and, as background, the one MODEL was trained against (D-84: a checkpoint's
configuration, or what an exchange file records). Its items are what `ftgs eval --renders-dir`
matches the records to (Q-137); the package has no request writer (Q-56).

`run` writes DIR/request.json, then runs `ftgs render` and `ftgs eval --renders-dir` as
processes of their own, writing DIR/renders.safetensors and DIR/METRICS.json. It writes each
render as an 8-bit RGB PNG, clamped to [0, 1] as E16 clamps it and rounded, at its record's image
path relative to the scene (with a .png suffix) under DIR; the PNGs are for looking at, since
they can't reproduce the metrics (Q-137). It then deletes the float32 renders.safetensors.
With `--compare`, it writes DIR/COMPARISON.json: per block of scene means (`entire`, and
`dynamic` where both files have it), each metric's difference from the given METRICS.json
(this run's minus that one; null where either value is null), the largest per-image PSNR
difference, and as `findings` every PSNR difference above 0.01 dB and every other metric's
difference above 0.001 (#327, Q-224). The backend is gsplat unless `--backend reference`; on a
machine without CUDA `ftgs render` refuses gsplat rather than falling back (Q-125).

Exit 0 once written; 2 for an input that can't be read or matched; a failing ftgs command's own
exit code otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openftgs.types import ImageRecord, SceneManifest

# Q-224: larger differences from the run's METRICS.json are findings for the owner: #327 names
# 0.01 dB for PSNR and "not small" for SSIM and LPIPS, whose values lie in [0, 1].
_PSNR_FINDING_DB = 0.01
_UNIT_FINDING = 1e-3
_BLOCKS = ("entire", "dynamic")  # fastest_10 is out of #327's scope: no M17 manifest names it


class InputError(Exception):
    pass


class CommandFailed(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"exit code {code}")
        self.code = code


def _background(model: Path) -> list[float]:
    """D-84: the background MODEL was trained against."""
    from openftgs.config import config_from_dict
    from openftgs.io import load_checkpoint, read_exchange

    if model.is_dir():
        return list(config_from_dict(load_checkpoint(model).config).background)
    return list(read_exchange(model).background)


def _test_records(scene: Path) -> tuple[SceneManifest, list[ImageRecord]]:
    from openftgs.io import read_scene_manifest

    manifest = read_scene_manifest(scene)
    records = [r for r in manifest.records if r.split == "test"]
    if not records:
        raise InputError(f"the test split of {scene} holds no images")
    return manifest, records


def write_request(model: Path, scene: Path, out: Path) -> None:
    try:
        manifest, records = _test_records(scene)
        background = _background(model)
    except (OSError, ValueError) as err:
        raise InputError(str(err)) from None
    # Q-56: the request's keys; Q-137: eval matches each record to the item with its exact
    # timestamp, size, K and pose
    body = {
        "format": "ftgs-render-request",
        "version": 1,
        "time_unit": manifest.time_unit,
        "background": background,
        "sh_degree": None,
        "items": [
            {
                "id": f"r{k:05d}",
                "timestamp": record.timestamp,  # float64; json writes its shortest round trip
                "camera": {
                    "width": record.camera.width,
                    "height": record.camera.height,
                    "K": record.camera.K.tolist(),
                    "world_to_camera": record.camera.world_to_camera.tolist(),
                },
            }
            for k, record in enumerate(records)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1), encoding="utf-8")


def write_pngs(directory: Path, scene: Path) -> int:
    """The PNGs of DIR's renders, which answer DIR's request.json; the number written."""
    import torch
    from PIL import Image

    from openftgs.io import read_render_request, read_renders

    request_path = directory / "request.json"
    try:
        manifest, records = _test_records(scene)
        sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
        request = read_render_request(request_path)
        # Q-56: renders recorded for another request are refused
        renders = read_renders(directory / "renders.safetensors", request_sha256=sha)
    except (OSError, ValueError) as err:
        raise InputError(str(err)) from None
    if len(request.items) != len(records):
        raise InputError(
            f"{request_path} has {len(request.items)} items for {len(records)} test records"
        )
    root = manifest.path.parent.resolve()
    for item, record in zip(request.items, records, strict=True):  # write_request's order
        try:
            name = record.image_path.resolve().relative_to(root).with_suffix(".png")
        except ValueError:
            raise InputError(f"{record.image_path} isn't under {root}") from None
        if item.timestamp != record.timestamp:
            raise InputError(f"{request_path}: item {item.id} isn't the render of {name}")
        rgb = renders.rgb.get(item.id)
        if rgb is None or tuple(rgb.shape) != (record.camera.height, record.camera.width, 3):
            raise InputError(f"{directory}: no render of {name}")
        pixels = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, "RGB").save(target, format="PNG")
    return len(records)


def _difference(ours: Any, theirs: Any) -> float | None:
    """ours − theirs; None where METRICS.json holds null for either (a non-finite value)."""
    return None if ours is None or theirs is None else float(ours) - float(theirs)


def compare(ours: dict[str, Any], theirs: dict[str, Any]) -> dict[str, Any]:
    """#327, Q-224: this run's metrics minus the given METRICS.json's."""
    blocks: dict[str, dict[str, float | None]] = {}
    largest: dict[str, float] = {}
    findings: list[str] = []
    for block in _BLOCKS:
        if block not in ours or block not in theirs:
            continue
        our_means, their_means = ours[block]["scene"], theirs[block]["scene"]
        blocks[block] = {
            name: _difference(our_means[name], their_means[name])
            for name in their_means
            if name in our_means
        }
        for name, delta in blocks[block].items():
            limit = _PSNR_FINDING_DB if name == "psnr" else _UNIT_FINDING
            if delta is not None and abs(delta) > limit:
                findings.append(f"{block}: scene {name} differs by {delta:+.4g}")
        their_images = {row["path"]: row for row in theirs[block].get("images", [])}
        image_psnr_deltas = [
            delta
            for row in ours[block].get("images", [])
            if row["path"] in their_images
            and (delta := _difference(row["psnr"], their_images[row["path"]]["psnr"])) is not None
        ]
        if image_psnr_deltas:
            largest[block] = max(image_psnr_deltas, key=abs)
            if abs(largest[block]) > _PSNR_FINDING_DB:
                findings.append(f"{block}: an image's PSNR differs by {largest[block]:+.4g} dB")
    return {"blocks": blocks, "largest_image_psnr_difference": largest, "findings": findings}


def _ftgs(*args: Path | str) -> None:
    """One public `ftgs` command, in a process of its own."""
    done = subprocess.run([sys.executable, "-m", "openftgs.cli", *map(str, args)], check=False)
    if done.returncode != 0:
        raise CommandFailed(done.returncode)


def run(
    model: Path, scene: Path, directory: Path, *, compare_with: Path | None, backend: str
) -> int:
    """#327's steps for one model; the number of PNGs written."""
    request, renders = directory / "request.json", directory / "renders.safetensors"
    write_request(model, scene, request)
    # Q-125: gsplat needs CUDA, and ftgs render refuses it without
    _ftgs("render", model, "--request", request, "--out", renders, "--backend", backend)
    metrics = directory / "METRICS.json"
    _ftgs("eval", model, scene, "--renders-dir", directory, "--out", metrics)
    count = write_pngs(directory, scene)
    renders.unlink()
    if compare_with is not None:
        try:
            theirs = json.loads(compare_with.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise InputError(f"--compare {compare_with}: {err}") from None
        comparison = compare(json.loads(metrics.read_text(encoding="utf-8")), theirs)
        comparison["compared_with"] = str(compare_with)
        out = directory / "COMPARISON.json"
        out.write_text(json.dumps(comparison, indent=1), encoding="utf-8")
        for finding in comparison["findings"]:
            print(f"view_renders: finding: {finding}", file=sys.stderr)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the test-view renders of a model, as PNGs")
    commands = parser.add_subparsers(dest="command", required=True)
    request_parser = commands.add_parser("request", help="write the test split's render request")
    run_parser = commands.add_parser("run", help="request, ftgs render, ftgs eval, PNGs")
    for sub in (request_parser, run_parser):
        sub.add_argument("model", type=Path, metavar="MODEL")
        sub.add_argument("scene", type=Path, metavar="SCENE.json")
    request_parser.add_argument("--out", type=Path, required=True, metavar="REQUEST.json")
    run_parser.add_argument("dir", type=Path, metavar="DIR")
    run_parser.add_argument("--compare", type=Path, metavar="METRICS.json")
    run_parser.add_argument("--backend", choices=("gsplat", "reference"), default="gsplat")
    args = parser.parse_args(argv)
    try:
        if args.command == "request":
            write_request(args.model, args.scene, args.out)
        else:
            count = run(
                args.model, args.scene, args.dir, compare_with=args.compare, backend=args.backend
            )
            print(f"view_renders: {count} PNGs in {args.dir}")
    except InputError as err:
        print(f"view_renders: error: {err}", file=sys.stderr)
        return 2
    except CommandFailed as failure:
        return failure.code
    return 0


if __name__ == "__main__":
    sys.exit(main())

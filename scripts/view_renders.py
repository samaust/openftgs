# SPDX-License-Identifier: Apache-2.0
"""The test-view renders of a trained model, as PNGs, through the public CLI (#327; §2.2
"Rendering at any camera and time", E16, D-84; Q-56, Q-137).

    python scripts/view_renders.py request MODEL SCENE.json --out REQUEST.json [--split S]
    python scripts/view_renders.py pngs DIR SCENE.json [--split S]
    python scripts/view_renders.py run MODEL SCENE.json DIR [--split S] [--compare METRICS.json]
                                       [--backend gsplat|reference] [--keep-renders]

`request` writes a ftgs-render-request with one item per record of the split (default test),
in the manifest's order: the record's timestamp, its camera's size, K and world_to_camera, the
scene's time unit, `sh_degree: null` and, as background, the one MODEL was trained against
(D-84: a checkpoint's configuration, or what an exchange file records). Its items are what
`ftgs eval --renders-dir` matches the records to (Q-137); the package has no request writer
(Q-56).

`pngs` writes each render of DIR (`request.json` and `renders.safetensors`) as an 8-bit RGB PNG,
clamped to [0, 1] as E16 clamps it and rounded, at its record's image path relative to the
scene, under DIR. The PNGs are for looking at: they can't reproduce the metrics (Q-137).

`run` does both around `ftgs render` and `ftgs eval --renders-dir`, run as processes of their
own, writing DIR/request.json, DIR/METRICS.json and the PNGs, then deletes DIR's float32
renders.safetensors unless `--keep-renders`. With `--compare`, it writes DIR/COMPARISON.json:
per block of scene means (`entire`, and `dynamic` and `fastest_10` where both files have them),
each metric's difference from the given METRICS.json (this run's minus that one), the largest
per-image PSNR difference, and as `findings` every PSNR difference above 0.01 dB (#327). The
backend is gsplat where PyTorch sees CUDA, as `ftgs eval` chooses (Q-125), else reference.

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
from typing import Any

_PSNR_FINDING_DB = 0.01  # #327: a larger PSNR difference from the run's METRICS.json is reported
_BLOCKS = ("entire", "dynamic", "fastest_10")


class _InputError(Exception):
    pass


def _background(model: Path) -> list[float]:
    """D-84: the background MODEL was trained against."""
    from openftgs.config import config_from_dict
    from openftgs.io import load_checkpoint, read_exchange

    if model.is_dir():
        return list(config_from_dict(load_checkpoint(model).config).background)
    return list(read_exchange(model).background)


def _records(scene: Path, split: str) -> tuple[Any, list[Any]]:
    from openftgs.io import read_scene_manifest

    manifest = read_scene_manifest(scene)
    records = [r for r in manifest.records if r.split == split]
    if not records:
        raise _InputError(f"the {split} split of {scene} holds no images")
    return manifest, records


def write_request(model: Path, scene: Path, out: Path, split: str) -> None:
    try:
        manifest, records = _records(scene, split)
        background = _background(model)
    except (OSError, ValueError) as err:
        raise _InputError(str(err)) from None
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


def write_pngs(directory: Path, scene: Path, split: str) -> int:
    """The PNGs of DIR's renders; the number written."""
    import torch
    from PIL import Image

    from openftgs.io import read_render_request, read_renders

    try:
        manifest, records = _records(scene, split)
        request_path = directory / "request.json"
        sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
        request = read_render_request(request_path)
        renders = read_renders(directory / "renders.safetensors", request_sha256=sha)
    except (OSError, ValueError) as err:
        raise _InputError(str(err)) from None
    if len(request.items) != len(records):
        raise _InputError(
            f"{request_path} has {len(request.items)} items for {len(records)} {split} records"
        )
    root = manifest.path.parent
    for item, record in zip(request.items, records, strict=True):
        name = record.image_path.relative_to(root)
        if item.timestamp != record.timestamp:
            raise _InputError(f"{request_path}: item {item.id} isn't the render of {name}")
        rgb = renders.rgb.get(item.id)
        if rgb is None or tuple(rgb.shape) != (record.camera.height, record.camera.width, 3):
            raise _InputError(f"{directory}: no render of {name}")
        pixels = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, "RGB").save(target)
    return len(records)


def _compare(ours: dict[str, Any], theirs: dict[str, Any]) -> dict[str, Any]:
    """#327: this run's metrics minus the given METRICS.json's."""
    blocks: dict[str, dict[str, float]] = {}
    largest: dict[str, float] = {}
    findings: list[str] = []
    for block in _BLOCKS:
        if block not in ours or block not in theirs:
            continue
        a, b = ours[block]["scene"], theirs[block]["scene"]
        blocks[block] = {name: a[name] - b[name] for name in b if name in a}
        delta = blocks[block].get("psnr")
        if delta is not None and abs(delta) > _PSNR_FINDING_DB:
            findings.append(f"{block}: scene PSNR differs by {delta:+.4f} dB")
        theirs_by_path = {row["path"]: row for row in theirs[block].get("images", [])}
        rows = [
            row["psnr"] - theirs_by_path[row["path"]]["psnr"]
            for row in ours[block].get("images", [])
            if row["path"] in theirs_by_path
        ]
        if rows:
            largest[block] = max(rows, key=abs)
            if abs(largest[block]) > _PSNR_FINDING_DB:
                findings.append(f"{block}: an image's PSNR differs by {largest[block]:+.4f} dB")
    return {"blocks": blocks, "largest_image_psnr_difference": largest, "findings": findings}


def _ftgs(*args: Path | str) -> None:
    """One public `ftgs` command, in a process of its own; its failure ends the run."""
    done = subprocess.run([sys.executable, "-m", "openftgs.cli", *map(str, args)], check=False)
    if done.returncode != 0:
        raise SystemExit(done.returncode)


def run(args: argparse.Namespace) -> None:
    import torch

    backend = args.backend or ("gsplat" if torch.cuda.is_available() else "reference")
    directory: Path = args.dir
    request, renders = directory / "request.json", directory / "renders.safetensors"
    write_request(args.model, args.scene, request, args.split)
    _ftgs("render", args.model, "--request", request, "--out", renders, "--backend", backend)
    metrics = directory / "METRICS.json"
    _ftgs("eval", args.model, args.scene, "--split", args.split,
          "--renders-dir", directory, "--out", metrics)  # fmt: skip
    count = write_pngs(directory, args.scene, args.split)
    if args.compare is not None:
        try:
            theirs = json.loads(args.compare.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise _InputError(f"--compare {args.compare}: {err}") from None
        comparison = _compare(json.loads(metrics.read_text(encoding="utf-8")), theirs)
        comparison["compared_with"] = str(args.compare)
        (directory / "COMPARISON.json").write_text(json.dumps(comparison, indent=1))
        for finding in comparison["findings"]:
            print(f"view_renders: finding: {finding}", file=sys.stderr)
    if not args.keep_renders:
        renders.unlink()
    print(f"view_renders: {count} PNGs in {directory}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the test-view renders of a model, as PNGs")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("request", help="write the render request of a scene's split")
    p.add_argument("model", type=Path, metavar="MODEL")
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--out", type=Path, required=True, metavar="REQUEST.json")
    p.add_argument("--split", choices=("test", "train"), default="test")
    p = commands.add_parser("pngs", help="write DIR's renders as PNGs")
    p.add_argument("dir", type=Path, metavar="DIR")
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("--split", choices=("test", "train"), default="test")
    p = commands.add_parser("run", help="request, ftgs render, ftgs eval --renders-dir, PNGs")
    p.add_argument("model", type=Path, metavar="MODEL")
    p.add_argument("scene", type=Path, metavar="SCENE.json")
    p.add_argument("dir", type=Path, metavar="DIR")
    p.add_argument("--split", choices=("test", "train"), default="test")
    p.add_argument("--compare", type=Path, metavar="METRICS.json")
    p.add_argument("--backend", choices=("gsplat", "reference"))
    p.add_argument("--keep-renders", dest="keep_renders", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "request":
            write_request(args.model, args.scene, args.out, args.split)
        elif args.command == "pngs":
            print(f"view_renders: {write_pngs(args.dir, args.scene, args.split)} PNGs")
        else:
            run(args)
    except _InputError as err:
        print(f"view_renders: error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

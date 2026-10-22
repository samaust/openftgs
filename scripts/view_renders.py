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
With `--compare`, it writes DIR/COMPARISON.json: per block of scene means (`entire` and
`dynamic`), each metric's difference from the given METRICS.json (this run's minus that one;
null where either value is null), per block and metric the largest per-image difference, and as
`findings` every scene mean or image value differing by more than 0.01 dB in PSNR or 10⁻⁴ in
any other metric, every value null on one side only, and every block, metric and image path in
one file only (#327, Q-224). The backend is gsplat unless `--backend reference`; on a
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
# 0.01 dB for PSNR, and 10⁻⁴ puts the other metrics, whose values lie in [0, 1], on its footing.
_PSNR_FINDING_DB = 0.01
_UNIT_FINDING = 1e-4
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


def _side(ours: bool) -> str:
    return "this run's" if ours else "the given"


def _one_sided(what: str, theirs: set[str], ours: set[str], findings: list[str]) -> None:
    """Q-224: a finding for each name in one file only."""
    for name in sorted(theirs ^ ours):
        findings.append(f"{what} {name} is only in {_side(name in ours)} METRICS.json")


def _compare_values(
    where: str, ours: dict[str, Any], theirs: dict[str, Any], findings: list[str]
) -> dict[str, float | None]:
    """The differences of the metrics both sides name; Q-224's findings among them."""
    _one_sided(f"{where} metric", set(theirs), set(ours), findings)
    deltas: dict[str, float | None] = {}
    for name in theirs:
        if name not in ours:
            continue
        deltas[name] = delta = _difference(ours[name], theirs[name])
        limit = _PSNR_FINDING_DB if name == "psnr" else _UNIT_FINDING
        if delta is not None and abs(delta) > limit:
            findings.append(f"{where} {name} differs by {delta:+.4g}")
        elif delta is None and (ours[name] is None) != (theirs[name] is None):
            findings.append(f"{where} {name} is null in {_side(ours[name] is None)} METRICS.json")
    return deltas


def compare(ours: dict[str, Any], theirs: dict[str, Any]) -> dict[str, Any]:
    """#327, Q-224: this run's metrics minus the given METRICS.json's."""
    blocks: dict[str, dict[str, float | None]] = {}
    largest: dict[str, dict[str, float]] = {}
    findings: list[str] = []
    their_blocks, our_blocks = ({b for b in _BLOCKS if b in body} for body in (theirs, ours))
    _one_sided("block", their_blocks, our_blocks, findings)
    for block in _BLOCKS:
        if block not in ours or block not in theirs:
            continue
        our_means, their_means = ours[block]["scene"], theirs[block]["scene"]
        blocks[block] = _compare_values(f"{block}: scene", our_means, their_means, findings)
        metrics = list(dict.fromkeys([*their_means, *our_means]))  # a row's other keys aren't
        our_images = {row["path"]: row for row in ours[block].get("images", [])}
        their_images = {row["path"]: row for row in theirs[block].get("images", [])}
        _one_sided(f"{block}: image", set(their_images), set(our_images), findings)
        image_deltas: dict[str, list[float]] = {name: [] for name in metrics}
        for path, their_row in their_images.items():
            if path not in our_images:
                continue
            our_row = our_images[path]
            row_deltas = _compare_values(
                f"{block}: {path}",
                {name: our_row[name] for name in metrics if name in our_row},
                {name: their_row[name] for name in metrics if name in their_row},
                findings,
            )
            for name, delta in row_deltas.items():
                if delta is not None:
                    image_deltas[name].append(delta)
        largest[block] = {
            name: max(deltas, key=abs) for name, deltas in image_deltas.items() if deltas
        }
    return {
        "blocks": blocks,
        "largest_image_difference": largest,
        "largest_image_psnr_difference": {
            block: values["psnr"] for block, values in largest.items() if "psnr" in values
        },
        "findings": findings,
    }


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

# SPDX-License-Identifier: Apache-2.0
"""How much of a trained model's test-view error belongs to openftgs-synth's reference images:
the pixels of the converted frames no model could match, and each model's PSNR without them
(#329, from #318; E16, D-64, D-67; Q-233 to Q-237).

    python scripts/reference_flaws.py scene SCENE.json GT_DIR --out DIR [--workers N]
    python scripts/reference_flaws.py run CHECKPOINT SCENE.json FLAGS_DIR --out RESULT.json
                                      [--backend gsplat|reference]

The reference is the converted frames of SCENE.json, the images the models were trained on and
E16 scores against. GT_DIR is the scene's `ground_truth/<scene>` folder, which supplies the
cameras (`cameras.json`), the depth (`depth/<cam>/<n:06d>.exr`) and the dynamic masks
(`masks/<cam>/<n:06d>.png`). Every record of SCENE.json is matched to its camera and decode
index n there and its camera checked against the ground truth's, as `gt_depth.py` does
(Q-231); a mismatch is exit 2.

`scene` measures, for every camera of the scene and every frame:
1. **Cross-camera consistency**: the frame of each of the camera's two nearest cameras (Q-235)
   is warped into its view with the ground-truth depth and poses, and compared over the pixels
   both see. A pixel of the target camera is back-projected from its centre at its depth,
   projected into the other camera and sampled there bilinearly; it is compared when the point's
   z lies within 1 % of the other camera's depth at the pixel it lands in (occlusion, Q-234) and
   neither pixel lies within 1 px of a depth edge (Q-228's rule). Cameras whose frames lie more
   than 1 µs apart (studio's) are compared on static pixels only, at the other camera's frame
   nearest in time (Q-237). Reported as PSNR, the mean |difference| in levels and the share
   above ½, 1 and 2 times 10 levels in any channel, on all, static and dynamic pixels.
2. **Clipping**: the share of pixels at 255 in any channel and in all three.
3. **Flicker**: the change from frame to frame of each camera, on static pixels and inside the
   ground truth's dynamic masks apart. Static means outside both frames' masks dilated by 1 px,
   since the masks are point-sampled and the frames box-filtered (Q-236). Static pixels still
   change with moving shadows and indirect light, which the ground truth's dynamic masks don't
   cover.

A pixel is flagged (Q-236) when it is clipped in any channel; when its smallest difference over
the cameras that see it exceeds 10 levels in any channel (cross-camera); or when, on static
pixels, its smallest change to the adjacent frames exceeds 2 levels in any channel (flicker).
The smallest, so that a flaw of the other camera or frame doesn't flag this one. Flags are also
made at ½ and 2 times both thresholds.

DIR receives `REFERENCE.json` (every camera's figures per frame and pooled, and the scene's),
`maps/<cam>.npz` (per pixel, over the frames: the shares clipped and flagged, and the mean
|difference| against each neighbour and to the next frame) with `maps/<cam>_<flag>.png` (the
share of frames flagged, 255 = every frame), and for each test image `flags/<cam>/<n:06d>.png`,
a bit field (`FLAG_BITS`) indexed by `flags/index.json`.

`run` renders each test record of SCENE.json from CHECKPOINT through `render()` and reports, per
frame and averaged over frames as E16 does, its PSNR as E16 computes it, and the PSNR over the
pixels left once those flagged in FLAGS_DIR are removed (Q-233), for each flag alone, for their
union, and for the union at ½ and 2 times the thresholds; with the PSNR over the flagged pixels
and their share. SSIM and LPIPS have no kept-pixel form.

These are diagnostics, not pass criteria: the spec defines no metric for them. Exit 0 once
written; 2 for an input that can't be read or matched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gt_depth
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch

_Backend = Literal["gsplat", "reference"]

_NEIGHBOURS = 2  # Q-235
_OCCLUSION = 0.01  # Q-234: relative
_CROSS_LEVELS = 10.0  # Q-236
_FLICKER_LEVELS = 2.0  # Q-236
_SCALES = (0.5, 1.0, 2.0)  # Q-236's check: the thresholds at ½, 1 and 2 times
_SAME_INSTANT_S = 1e-6  # Q-237: frames closer in time are the same instant
_MATCH_S = 1e-6  # Q-231: a record's timestamp against the flag index's
_SNAP_PX = 1e-6  # a sample this close to a pixel centre is taken at it
_REGIONS = ("all", "static", "dynamic")

FLAG_BITS: dict[str, int] = {
    "clipped": 1,
    "cross": 2,
    "flicker": 4,
    "cross_0.5": 8,
    "cross_2.0": 16,
    "flicker_0.5": 32,
    "flicker_2.0": 64,
}


def _bit(kind: str, scale: float) -> int:
    return FLAG_BITS[kind if scale == 1.0 else f"{kind}_{scale}"]


class InputError(Exception):
    pass


# -- Measurements ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Warp:
    image: np.ndarray  # [H, W, 3] float64, the source sampled at each target pixel; 0 elsewhere
    inside: np.ndarray  # [H, W] bool: a target surface pixel whose point lands inside the source
    z: np.ndarray  # [H, W] the point's z in the source camera
    pixel: tuple[np.ndarray, np.ndarray]  # (x, y) of the source pixel the point lands in


def warp(
    source: np.ndarray,
    target_depth: np.ndarray,
    target_K: np.ndarray,
    target_w2c: np.ndarray,
    source_K: np.ndarray,
    source_w2c: np.ndarray,
) -> Warp:
    """The source frame [H', W', C] seen from the target camera: each target pixel with a surface
    (depth > 0) back-projected from its centre at its planar depth, projected into the source
    camera and sampled bilinearly there (§1.4's continuous convention: pixel (u, v) spans
    [u, u + 1)). In float64; a sample within 10⁻⁶ px of a pixel centre is taken at it, so a
    frame warped into its own camera comes back unchanged."""
    h, w = target_depth.shape
    sh, sw = source.shape[:2]
    depth = target_depth.astype(np.float64)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    k = np.linalg.inv(np.asarray(target_K, dtype=np.float64))
    ray = [k[i, 0] * u + k[i, 1] * v + k[i, 2] for i in range(3)]
    point = [r * depth / ray[2] for r in ray]  # planar depth: the camera-space z
    m = np.asarray(source_w2c, np.float64) @ np.linalg.inv(np.asarray(target_w2c, np.float64))
    y = [m[i, 0] * point[0] + m[i, 1] * point[1] + m[i, 2] * point[2] + m[i, 3] for i in range(3)]
    ks = np.asarray(source_K, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = (ks[0, 0] * y[0] + ks[0, 1] * y[1] + ks[0, 2] * y[2]) / y[2]
        py = (ks[1, 0] * y[0] + ks[1, 1] * y[1] + ks[1, 2] * y[2]) / y[2]
    x, yy = px - 0.5, py - 0.5  # index coordinates: pixel centres at integers
    for c in (x, yy):
        nearest = np.round(c)
        with np.errstate(invalid="ignore"):  # non-finite where there is no surface
            snap = np.abs(c - nearest) < _SNAP_PX
        c[snap] = nearest[snap]
    inside = (depth > 0) & (y[2] > 0) & (x >= 0) & (x <= sw - 1) & (yy >= 0) & (yy <= sh - 1)
    x, yy = np.where(inside, x, 0.0), np.where(inside, yy, 0.0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(yy).astype(np.int64)
    fx, fy = (x - x0)[..., None], (yy - y0)[..., None]
    x1, y1 = np.minimum(x0 + 1, sw - 1), np.minimum(y0 + 1, sh - 1)
    src = source.astype(np.float64)
    image = (1 - fy) * ((1 - fx) * src[y0, x0] + fx * src[y0, x1]) + fy * (
        (1 - fx) * src[y1, x0] + fx * src[y1, x1]
    )
    image[~inside] = 0.0
    pixel = (
        np.clip(np.floor(x + 0.5), 0, sw - 1).astype(np.int64),
        np.clip(np.floor(yy + 0.5), 0, sh - 1).astype(np.int64),
    )
    return Warp(image, inside, y[2], pixel)


def clipped(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pixels of an 8-bit image at 255 in any channel and in all three."""
    full = image >= 255
    return full.any(axis=-1), full.all(axis=-1)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """A boolean mask grown by 1 px (its 3×3 neighbourhood)."""
    padded = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def kept_psnr(pred: torch.Tensor, gt: torch.Tensor, keep: torch.Tensor) -> float | None:
    """Q-233: PSNR = 10·log10(1/MSE) over the kept pixels of one image [H, W, 3] only, all three
    channels of each, the render clamped to [0, 1] first as E16 does; None without kept pixels,
    +inf for an MSE of 0."""
    count = int(keep.sum())
    if count == 0:
        return None
    diff = pred.double().clamp(0.0, 1.0) - gt.double()
    mse = float(diff.square()[keep].sum()) / (3 * count)
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def _psnr_of(mse: float | None) -> float | None:
    """For JSON: None when there is nothing to compare or the MSE is 0 (see the `mse` beside)."""
    if mse is None or mse <= 0.0:
        return None
    return -10.0 * math.log10(mse)


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


@dataclass
class _Pool:
    """Sums over pixels of a comparison in levels: pixels, Σ squared and absolute differences
    over channels, the pixels above each threshold, and each frame's PSNR."""

    levels: float
    pixels: int = 0
    sum_sq: float = 0.0
    sum_abs: float = 0.0
    above: dict[float, int] = field(default_factory=dict)
    frame_psnr: list[float] = field(default_factory=list)

    def figures(self, diff: np.ndarray, surface: int | None) -> dict[str, Any]:
        """One frame's figures from diff [M, 3] in levels, added to the pool."""
        n = int(diff.shape[0])
        worst = diff.max(axis=1) if n else diff[:, 0]
        sq, ab = float(np.square(diff).sum()), float(diff.sum())
        above = {s: int((worst > s * self.levels).sum()) for s in _SCALES}
        self.pixels += n
        self.sum_sq += sq
        self.sum_abs += ab
        for s, c in above.items():
            self.above[s] = self.above.get(s, 0) + c
        mse = sq / (3 * n) / 255.0**2 if n else None
        psnr = _psnr_of(mse)
        if psnr is not None:
            self.frame_psnr.append(psnr)
        return {
            "pixels": n,
            "share_of_surface": n / surface if surface else None,
            "mse": mse,
            "psnr": psnr,
            "mean_abs": ab / (3 * n) if n else None,
            "share_above": {str(s): c / n if n else None for s, c in above.items()},
        }

    def pooled(self) -> dict[str, Any]:
        n = self.pixels
        mse = self.sum_sq / (3 * n) / 255.0**2 if n else None
        return {
            "pixels": n,
            "mse": mse,
            "psnr": _psnr_of(mse),
            "mean_frame_psnr": float(np.mean(self.frame_psnr)) if self.frame_psnr else None,
            "frames": len(self.frame_psnr),
            "mean_abs": self.sum_abs / (3 * n) if n else None,
            "share_above": {str(s): self.above.get(s, 0) / n if n else None for s in _SCALES},
        }


# -- The scene's frames ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    n: int
    timestamp: float
    image: Path
    depth: Path
    mask: Path
    test: bool


@dataclass(frozen=True)
class _View:
    name: str
    K: np.ndarray
    w2c: np.ndarray
    frames: tuple[_Frame, ...]  # by timestamp

    @property
    def centre(self) -> np.ndarray:
        rotation, t = self.w2c[:3, :3], self.w2c[:3, 3]
        return np.asarray(-rotation.T @ t)


@dataclass
class _Loaded:
    image: np.ndarray  # uint8 [H, W, 3]
    dynamic: np.ndarray  # bool [H, W], undilated
    depth: np.ndarray | None = None
    away: np.ndarray | None = None  # away from depth edges (Q-228)


def _load(frame: _Frame, *, depth: bool) -> _Loaded:
    try:
        with Image.open(frame.image) as image:
            pixels = np.asarray(image.convert("RGB"))
        with Image.open(frame.mask) as image:
            dynamic = np.asarray(image.convert("L")) > 127
        loaded = _Loaded(pixels, dynamic)
        if depth:
            loaded.depth = gt_depth.read_exr(frame.depth)
            loaded.away = gt_depth.away_from_edges(loaded.depth)
    except (OSError, ValueError) as err:
        raise InputError(str(err)) from None
    sizes = {pixels.shape[:2], dynamic.shape}
    if loaded.depth is not None:
        sizes.add(loaded.depth.shape)
    if len(sizes) != 1:
        raise InputError(f"{frame.image}: the frame, mask and depth sizes differ")
    return loaded


def _nearest(view: _View, timestamp: float) -> _Frame:
    """Q-237: the frame of a camera nearest in time (the earlier one on a tie)."""
    return min(view.frames, key=lambda f: (abs(f.timestamp - timestamp), f.timestamp))


def _views(scene: Path, gt_dir: Path) -> tuple[list[_View], dict[str, list[str]]]:
    """The scene's cameras with their frames, each matched to the ground truth (Q-231), and each
    camera's two nearest cameras by centre (Q-235)."""
    from openftgs.io import read_scene_manifest

    try:
        manifest = read_scene_manifest(scene)
        cameras = {
            c["name"]: c for c in json.loads((gt_dir / "cameras.json").read_text())["cameras"]
        }
        by_camera: dict[str, list[_Frame]] = {}
        meta: dict[str, Any] = {}
        for record in manifest.records:
            name, n = gt_depth.ground_truth_frame(record, cameras)
            frame = _Frame(
                n,
                record.timestamp,
                record.image_path,
                gt_dir / "depth" / name / f"{n:06d}.exr",
                gt_dir / "masks" / name / f"{n:06d}.png",
                record.split == "test",
            )
            by_camera.setdefault(name, []).append(frame)
            meta[name] = record.camera
    except gt_depth.InputError as err:
        raise InputError(str(err)) from None
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise InputError(f"{type(err).__name__}: {err}") from None
    views = [
        _View(
            name,
            meta[name].K.double().numpy(),
            meta[name].world_to_camera.double().numpy(),
            tuple(sorted(frames, key=lambda f: f.timestamp)),
        )
        for name, frames in by_camera.items()
    ]
    if len(views) <= _NEIGHBOURS:
        raise InputError(f"{scene}: the cross-camera comparison needs {_NEIGHBOURS + 1} cameras")
    neighbours = {}
    for view in views:
        others = sorted(
            (float(np.linalg.norm(o.centre - view.centre)), o.name)
            for o in views
            if o.name != view.name
        )
        neighbours[view.name] = [name for _, name in others[:_NEIGHBOURS]]
    return views, neighbours


# -- One camera's measurements --------------------------------------------------------------------


def _cross(
    target: _Loaded, view: _View, frame: _Frame, other: _View, pools: dict[str, _Pool]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Item 1 for one target frame and one neighbour: its figures, and per target pixel the
    largest channel difference in levels where compared (inf elsewhere) and whether compared."""
    source_frame = _nearest(other, frame.timestamp)
    source = _load(source_frame, depth=True)
    assert target.depth is not None and target.away is not None
    assert source.depth is not None and source.away is not None
    w = warp(source.image, target.depth, view.K, view.w2c, other.K, other.w2c)
    xs, ys = w.pixel
    other_depth = source.depth[ys, xs].astype(np.float64)
    seen = (  # Q-234, Q-228
        w.inside
        & target.away
        & source.away[ys, xs]
        & (np.abs(w.z - other_depth) <= _OCCLUSION * other_depth)
    )
    static = ~_dilate(target.dynamic) & ~_dilate(source.dynamic)[ys, xs]  # Q-236, Q-237
    dt = source_frame.timestamp - frame.timestamp
    same_instant = abs(dt) <= _SAME_INSTANT_S
    compared = seen if same_instant else seen & static  # Q-237
    diff = np.abs(w.image - target.image.astype(np.float64))
    surface = int((target.depth > 0).sum())
    regions: dict[str, Any] = {}
    for region in _REGIONS:
        if not same_instant and region != "static":
            regions[region] = None
            continue
        select = {"all": compared, "static": compared & static, "dynamic": compared & ~static}
        regions[region] = pools[region].figures(diff[select[region]], surface)
    worst = np.where(compared, diff.max(axis=-1), np.inf)
    figures = {
        "source_n": source_frame.n,
        "dt": dt,
        "same_instant": same_instant,
        "surface_pixels": surface,
        "regions": regions,
    }
    return figures, worst, compared


def _flicker_pair(
    a: _Loaded, b: _Loaded, pools: dict[str, _Pool]
) -> tuple[dict[str, Any], np.ndarray]:
    """Item 3 for frames k and k + 1: figures on static pixels (outside both masks dilated by
    1 px) and inside them, and per pixel the largest channel change in levels where static (inf
    elsewhere)."""
    diff = np.abs(a.image.astype(np.float64) - b.image.astype(np.float64))
    static = ~_dilate(a.dynamic | b.dynamic)  # Q-236
    regions = {
        "static": pools["static"].figures(diff[static], None),
        "dynamic": pools["dynamic"].figures(diff[~static], None),
    }
    return regions, np.where(static, diff.max(axis=-1), np.inf)


def _share(mask: np.ndarray) -> float:
    return float(mask.mean())


def _camera_job(job: tuple[_View, list[_View], str | None]) -> dict[str, Any]:
    """Items 1 to 3 for every frame of one camera; writes its maps and, for its test frames, the
    flag images under `out`."""
    view, others, out_name = job
    out = Path(out_name) if out_name is not None else None
    cross_pools = {o.name: {r: _Pool(_CROSS_LEVELS) for r in _REGIONS} for o in others}
    flicker_pools = {r: _Pool(_FLICKER_LEVELS) for r in ("static", "dynamic")}
    frames_out: list[dict[str, Any]] = []
    sums: dict[str, np.ndarray] = {}
    previous: np.ndarray | None = None  # the static change from frame k − 1 to k
    current = _load(view.frames[0], depth=True)
    for i, frame in enumerate(view.frames):
        nxt = _load(view.frames[i + 1], depth=True) if i + 1 < len(view.frames) else None
        record: dict[str, Any] = {"n": frame.n, "timestamp": frame.timestamp, "test": frame.test}
        any_255, all_255 = clipped(current.image)
        record["clipped"] = {"any": _share(any_255), "all": _share(all_255)}
        record["cross"] = {}
        smallest = np.full(any_255.shape, np.inf)
        for other in others:
            figures, worst, compared = _cross(current, view, frame, other, cross_pools[other.name])
            record["cross"][other.name] = figures
            smallest = np.minimum(smallest, worst)
            _accumulate(sums, f"cross_{other.name}_sum_abs", np.where(compared, worst, 0.0))
            _accumulate(sums, f"cross_{other.name}_compared", compared)
        change = np.full(any_255.shape, np.inf)
        if previous is not None:
            change = np.minimum(change, previous)
        if nxt is not None:
            record["flicker"], forward = _flicker_pair(current, nxt, flicker_pools)
            record["flicker"]["next_n"] = view.frames[i + 1].n
            change = np.minimum(change, forward)
            _accumulate(sums, "flicker_sum_abs", np.where(np.isfinite(forward), forward, 0.0))
            _accumulate(sums, "flicker_static", np.isfinite(forward))
            previous = forward
        else:
            record["flicker"] = None
        flags = _flags(any_255, smallest, change)
        record["flagged"] = _flag_shares(flags)
        for name, mask in (("clipped_any", any_255), ("clipped_all", all_255)):
            _accumulate(sums, name, mask)
        for kind in ("cross", "flicker"):
            _accumulate(sums, f"{kind}_flagged", (flags & FLAG_BITS[kind]) > 0)
        if frame.test and out is not None:
            path = out / "flags" / view.name / f"{frame.n:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(flags, "L").save(path)
        frames_out.append(record)
        if nxt is not None:
            current = nxt
    count = len(view.frames)
    if out is not None:
        _write_maps(out, view.name, [o.name for o in others], sums, count)
    return {
        "camera": view.name,
        "neighbours": [o.name for o in others],
        "pooled": {
            "clipped": {
                k: float(np.mean([f["clipped"][k] for f in frames_out])) for k in ("any", "all")
            },
            "cross": {
                name: {r: p.pooled() for r, p in pools.items()}
                for name, pools in cross_pools.items()
            },
            "flicker": {r: p.pooled() for r, p in flicker_pools.items()},
            "flagged": _mean_flag_shares([f["flagged"] for f in frames_out]),
        },
        "frames": frames_out,
        "_pools": {"cross": cross_pools, "flicker": flicker_pools},
    }


def _flags(any_255: np.ndarray, smallest: np.ndarray, change: np.ndarray) -> np.ndarray:
    """Q-236: the bit field of one frame from its clipped pixels, its smallest cross-camera
    difference and its smallest static change to the adjacent frames (inf where not compared,
    which flags nothing)."""
    flags = np.where(any_255, FLAG_BITS["clipped"], 0).astype(np.uint8)
    for s in _SCALES:
        cross = np.isfinite(smallest) & (smallest > s * _CROSS_LEVELS)
        flicker = np.isfinite(change) & (change > s * _FLICKER_LEVELS)
        flags |= np.where(cross, _bit("cross", s), 0).astype(np.uint8)
        flags |= np.where(flicker, _bit("flicker", s), 0).astype(np.uint8)
    return flags


def _accumulate(sums: dict[str, np.ndarray], name: str, values: np.ndarray) -> None:
    v = values.astype(np.float64)
    if name in sums:
        sums[name] += v
    else:
        sums[name] = v.copy()


def _flag_shares(flags: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"clipped": _share((flags & FLAG_BITS["clipped"]) > 0)}
    for kind in ("cross", "flicker"):
        out[kind] = {str(s): _share((flags & _bit(kind, s)) > 0) for s in _SCALES}
    out["union"] = {str(s): _share((flags & union_bits(s)) > 0) for s in _SCALES}
    return out


def union_bits(scale: float) -> int:
    """The bits of the union of the three flags at a scale of the thresholds."""
    return FLAG_BITS["clipped"] | _bit("cross", scale) | _bit("flicker", scale)


def _mean_flag_shares(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"clipped": float(np.mean([r["clipped"] for r in rows]))}
    for kind in ("cross", "flicker", "union"):
        out[kind] = {str(s): float(np.mean([r[kind][str(s)] for r in rows])) for s in _SCALES}
    return out


def _write_maps(
    out: Path, name: str, others: list[str], sums: dict[str, np.ndarray], count: int
) -> None:
    """Per pixel over the camera's frames: the shares clipped and flagged, and the mean
    |difference| (largest channel, levels) against each neighbour and to the next frame where
    compared (NaN where never)."""
    maps: dict[str, np.ndarray] = {
        "clipped_any_share": sums["clipped_any"] / count,
        "clipped_all_share": sums["clipped_all"] / count,
        "cross_flagged_share": sums["cross_flagged"] / count,
        "flicker_flagged_share": sums["flicker_flagged"] / count,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        for other in others:
            seen = sums[f"cross_{other}_compared"]
            maps[f"cross_{other}_mean_abs"] = sums[f"cross_{other}_sum_abs"] / seen
            maps[f"cross_{other}_compared_share"] = seen / count
        if "flicker_static" in sums:
            static = sums["flicker_static"]
            maps["flicker_static_mean_abs"] = sums["flicker_sum_abs"] / static
    (out / "maps").mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {k: v.astype(np.float32) for k, v in maps.items()}
    np.savez_compressed(out / "maps" / f"{name}.npz", **arrays)
    for flag, key in (
        ("clipped", "clipped_any_share"),
        ("cross", "cross_flagged_share"),
        ("flicker", "flicker_flagged_share"),
    ):
        image = np.rint(np.clip(maps[key], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(image, "L").save(out / "maps" / f"{name}_{flag}.png")


# -- The scene ------------------------------------------------------------------------------------


def scene_report(
    scene: Path, gt_dir: Path, out: Path | None, *, workers: int = 1
) -> dict[str, Any]:
    """Items 1 to 3 on every camera of a scene (see the module's docstring)."""
    views, neighbours = _views(scene, gt_dir)
    by_name = {v.name: v for v in views}
    jobs = [
        (v, [by_name[n] for n in neighbours[v.name]], None if out is None else str(out))
        for v in views
    ]
    if workers > 1:
        import multiprocessing

        # Each worker is one thread: the ledger's budget is the number of workers. Spawned
        # workers take the environment at start, so it is restored once they are running.
        thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        saved = {var: os.environ.get(var) for var in thread_vars}
        os.environ.update(dict.fromkeys(thread_vars, "1"))
        try:
            pool = multiprocessing.get_context("spawn").Pool(workers)
        finally:
            for var, value in saved.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
        with pool:
            cameras = pool.map(_camera_job, jobs, chunksize=1)
    else:
        cameras = [_camera_job(job) for job in jobs]
    test_cameras = [v.name for v in views if any(f.test for f in v.frames)]
    overall: dict[str, Any] = {}
    for label, names in (("all_cameras", [v.name for v in views]), ("test_cameras", test_cameras)):
        chosen = [c for c in cameras if c["camera"] in names]
        cross = {r: _Pool(_CROSS_LEVELS) for r in _REGIONS}
        flicker = {r: _Pool(_FLICKER_LEVELS) for r in ("static", "dynamic")}
        for c in chosen:
            for pools in c["_pools"]["cross"].values():
                for r, p in pools.items():
                    _merge(cross[r], p)
            for r, p in c["_pools"]["flicker"].items():
                _merge(flicker[r], p)
        overall[label] = {
            "cross": {r: p.pooled() for r, p in cross.items()},
            "flicker": {r: p.pooled() for r, p in flicker.items()},
            "clipped": {
                k: float(np.mean([c["pooled"]["clipped"][k] for c in chosen]))
                for k in ("any", "all")
            },
        }
    for c in cameras:
        del c["_pools"]
    index = [
        {"camera": v.name, "n": f.n, "timestamp": f.timestamp, "path": f"{v.name}/{f.n:06d}.png"}
        for v in views
        for f in v.frames
        if f.test
    ]
    if out is not None:
        (out / "flags").mkdir(parents=True, exist_ok=True)
        (out / "flags" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return {
        "scene": str(scene),
        "ground_truth": str(gt_dir),
        "thresholds": {
            "cross_levels": _CROSS_LEVELS,
            "flicker_levels": _FLICKER_LEVELS,
            "scales": list(_SCALES),
            "occlusion": _OCCLUSION,
            "same_instant_s": _SAME_INSTANT_S,
        },
        "flag_bits": FLAG_BITS,
        "neighbours": neighbours,
        "test_cameras": test_cameras,
        "test_frames": len(index),
        "overall": overall,
        "cameras": {c["camera"]: c for c in cameras},
    }


def _merge(into: _Pool, other: _Pool) -> None:
    into.pixels += other.pixels
    into.sum_sq += other.sum_sq
    into.sum_abs += other.sum_abs
    for s, c in other.above.items():
        into.above[s] = into.above.get(s, 0) + c
    into.frame_psnr.extend(other.frame_psnr)


# -- A trained model's PSNR without the flagged pixels --------------------------------------------


def _flag_sets(flags: np.ndarray) -> dict[str, np.ndarray]:
    """Item 4's pixel sets: each flag alone, their union, and the union at ½ and 2 times."""
    return {
        "clipped": (flags & FLAG_BITS["clipped"]) > 0,
        "cross": (flags & FLAG_BITS["cross"]) > 0,
        "flicker": (flags & FLAG_BITS["flicker"]) > 0,
        "union": (flags & union_bits(1.0)) > 0,
        "union_0.5": (flags & union_bits(0.5)) > 0,
        "union_2.0": (flags & union_bits(2.0)) > 0,
    }


def run_report(
    checkpoint: Path, scene: Path, flags_dir: Path, *, backend: _Backend
) -> dict[str, Any]:
    """Item 4 for one trained model (see the module's docstring)."""
    import torch

    from openftgs.config import config_from_dict
    from openftgs.eval import psnr
    from openftgs.io import load_checkpoint, read_scene_manifest
    from openftgs.model import FreeTimeGaussians
    from openftgs.render import render

    try:
        manifest = read_scene_manifest(scene)
        index = json.loads((flags_dir / "index.json").read_text())
        state = load_checkpoint(checkpoint)
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise InputError(f"{type(err).__name__}: {err}") from None
    records = [r for r in manifest.records if r.split == "test"]
    if not records:
        raise InputError(f"the test split of {scene} holds no images")
    device = torch.device("cuda" if backend == "gsplat" else "cpu")
    config = config_from_dict(state.config)
    # E16's Î is the render composited over the run's background (D-34), as evaluate() renders.
    background = torch.tensor(config.background, dtype=torch.float32, device=device)
    model = FreeTimeGaussians(
        {k: v.to(device) for k, v in state.params.items()},
        sh_degree=config.sh_degree,
        time=state.time,
        time_unit=state.time_unit,
    )
    model.requires_grad_(False)
    sets = tuple(_flag_sets(np.zeros((1, 1), np.uint8)))
    sums: dict[str, dict[str, list[float]]] = {
        name: {"kept": [], "flagged": [], "share": []} for name in sets
    }
    full: list[float] = []
    frames: list[dict[str, Any]] = []
    for record in records:
        hits = [
            e
            for e in index
            if e["camera"] == record.camera.name
            and abs(e["timestamp"] - record.timestamp) <= _MATCH_S
        ]
        if len(hits) != 1:
            raise InputError(
                f"camera {record.camera.name}: {len(hits)} flag images at t = {record.timestamp}"
            )
        try:
            with Image.open(flags_dir / hits[0]["path"]) as image:
                flags = np.asarray(image)
            with Image.open(record.image_path) as image:
                pixels = np.asarray(image.convert("RGB"))
        except OSError as err:
            raise InputError(str(err)) from None
        if flags.shape != pixels.shape[:2]:
            raise InputError(f"{hits[0]['path']}: the flags and the frame sizes differ")
        gt = torch.from_numpy(pixels.astype(np.float32)).to(device) / 255.0
        with torch.no_grad():
            pred = render(
                model, record.camera, record.timestamp, background=background, backend=backend
            ).image[0]
        value = float(psnr(pred, gt))  # E16
        full.append(value)
        frame: dict[str, Any] = {
            "camera": record.camera.name,
            "n": hits[0]["n"],
            "timestamp": record.timestamp,
            "psnr": _finite(value),
        }
        for name, flagged in _flag_sets(flags).items():
            on = torch.from_numpy(flagged).to(device)
            share = float(flagged.mean())
            kept = _finite(kept_psnr(pred, gt, ~on))  # Q-233
            on_flagged = _finite(kept_psnr(pred, gt, on))
            frame[name] = {"share": share, "kept_psnr": kept, "flagged_psnr": on_flagged}
            sums[name]["share"].append(share)
            for figure, slot in ((kept, "kept"), (on_flagged, "flagged")):
                if figure is not None:
                    sums[name][slot].append(figure)
        frames.append(frame)
    overall: dict[str, Any] = {"psnr": float(np.mean(full)), "frames": len(full)}
    for name, values in sums.items():
        overall[name] = {
            "share": float(np.mean(values["share"])),
            "kept_psnr": float(np.mean(values["kept"])) if values["kept"] else None,
            "kept_frames": len(values["kept"]),
            "flagged_psnr": float(np.mean(values["flagged"])) if values["flagged"] else None,
            "flagged_frames": len(values["flagged"]),
        }
    return {
        "checkpoint": str(checkpoint),
        "iteration": state.iteration,
        "scene": str(scene),
        "flags": str(flags_dir),
        "backend": backend,
        "overall": overall,
        "frames": frames,
    }


# -- Command line ---------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the reference images' flaws (#329)")
    commands = parser.add_subparsers(dest="command", required=True)
    scene = commands.add_parser("scene", help="items 1 to 3 on a scene's converted frames")
    scene.add_argument("scene", type=Path, metavar="SCENE.json")
    scene.add_argument("gt_dir", type=Path, metavar="GT_DIR")
    scene.add_argument("--out", type=Path, required=True, metavar="DIR")
    scene.add_argument("--workers", type=int, default=1, metavar="N")
    run = commands.add_parser("run", help="item 4: a trained model's PSNR without flagged pixels")
    run.add_argument("checkpoint", type=Path, metavar="CHECKPOINT")
    run.add_argument("scene", type=Path, metavar="SCENE.json")
    run.add_argument("flags", type=Path, metavar="FLAGS_DIR")
    run.add_argument("--out", type=Path, required=True, metavar="RESULT.json")
    run.add_argument("--backend", choices=("gsplat", "reference"), default="gsplat")
    args = parser.parse_args(argv)
    try:
        if args.command == "scene":
            if args.workers < 1:
                parser.error("--workers must be at least 1")
            result = scene_report(args.scene, args.gt_dir, args.out, workers=args.workers)
            path = args.out / "REFERENCE.json"
        else:
            result = run_report(args.checkpoint, args.scene, args.flags, backend=args.backend)
            path = args.out
    except InputError as err:
        print(f"reference_flaws: error: {err}", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.command == "scene":
        figures = result["overall"]["test_cameras"]
        cross = figures["cross"]["all"]["psnr"] or figures["cross"]["static"]["psnr"]
        print(
            f"reference_flaws: test views' cross-camera PSNR {cross}, clipped (any channel) "
            f"{figures['clipped']['any']:.4f}; {len(result['cameras'])} cameras"
        )
    else:
        overall = result["overall"]
        print(
            f"reference_flaws: PSNR {overall['psnr']:.3f}, kept-pixel PSNR "
            f"{overall['union']['kept_psnr']} without {overall['union']['share']:.4f} of the "
            f"pixels; {overall['frames']} frames"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

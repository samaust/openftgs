# SPDX-License-Identifier: Apache-2.0
"""Expected depth and the geometry residual of a trained model against openftgs-synth's
ground-truth depth, per test frame (#328, from #318; E2, E6, D-27; Q-226 to Q-231).

    python scripts/gt_depth.py CHECKPOINT SCENE.json GT_DIR --out RESULT.json
                               [--backend gsplat|reference]

GT_DIR is the scene's `ground_truth/<scene>` folder: `cameras.json`, `depth/<cam>/<n:06d>.exr`
(planar z in metres at the pixel centre, 0 for no surface) and `masks/<cam>/<n:06d>.png` (255
where a dynamic object is visible). Each test record of SCENE.json is matched to its camera and
decode index n there, and its camera checked against the ground truth's (Q-231); a mismatch is
exit 2, for the owner to look at.

For each test frame, at the record's time:
- **expected depth** (Q-226): the model rendered through `render()` with SH degree 0 and each
  primitive's colour set to its camera-space depth z, so that image ÷ alpha = Σ wᵢ zᵢ / Σ wᵢ
  (E2, E6). It is compared with the ground truth over pixels with a surface, alpha ≥ 0.5
  (Q-227) and away from depth edges (Q-228), on all of them, on static pixels and on dynamic
  pixels: the number of pixels, their share of the surface pixels, and the mean, median and
  root mean square of |error| and the mean signed error, in metres (Q-231);
- **geometry residual**: for every primitive the render reports visible with σ_eff ≥ 1/255,
  r = its centre's camera-space z − the ground-truth depth at the pixel its centre projects to
  (Q-230), negative in front of the surface. Weighted by σ_eff: the median and 90th percentile
  of |r|, and the shares with r < −δ and r > +δ, δ = 0.02·S with S the checkpoint's spatial
  scale (D-27, Q-229), split by whether that pixel is static or dynamic.

RESULT.json holds every frame's figures and the same over all frames, whose quantiles come from
histograms of 0.1 mm bins (Q-231). These are diagnostics, not pass criteria: the spec defines no
metric for them.

Exit 0 once written; 2 for an input that can't be read or matched.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from openftgs.model import FreeTimeGaussians
    from openftgs.types import Camera

ALPHA_MIN = 0.5  # Q-227
EDGE_JUMP = 0.05  # Q-228: a relative depth jump above it, within 1 px, marks an edge
DELTA_OF_S = 0.02  # Q-229: δ = 0.02·S
OPACITY_MIN = 1.0 / 255.0  # D-32's cull, which the render applies too
# Q-231: matching a record to the ground truth
TIME_TOLERANCE_S = 1e-6
POSE_TOLERANCE = 1e-5
K_TOLERANCE = 1e-6  # relative
BIN_M = 1e-4  # Q-231: the pooled histograms' bins
BINS = 200_000  # 20 m; larger values land in the last bin


class InputError(Exception):
    pass


# -- EXR ------------------------------------------------------------------------------------------

# The layout of a single-part scanline file (magic, version flags, attributes, the line offset
# table, scan line blocks, channels in alphabetical order within a scan line) follows the OpenEXR
# file layout documentation; ZIP's reconstruction after inflating (undo the byte deltas, then
# interleave the two halves) follows OpenEXR's src/lib/OpenEXRCore/internal_zip.c at commit
# 6a75faa5f35786c5a4f6f3a2cdcfcc26496b3f54 (BSD-3-Clause, Copyright Contributors to the OpenEXR
# Project), described there and written here anew with NumPy (source log).
_MAGIC = b"\x76\x2f\x31\x01"
_NOT_SCANLINE = 0x200 | 0x800 | 0x1000  # tiled, deep, multi-part
_LINES_PER_BLOCK = {0: 1, 2: 1, 3: 16}  # NO_COMPRESSION, ZIPS, ZIP
_FLOAT = 2


def _cstring(data: bytes, at: int) -> tuple[str, int]:
    end = data.index(b"\0", at)
    return data[at:end].decode("ascii"), end + 1


def _unzip(packed: bytes, size: int) -> bytes:
    if len(packed) == size:  # stored raw: compressing didn't make it smaller
        return packed
    deltas = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
    if deltas.size != size:
        raise ValueError(f"a ZIP block inflates to {deltas.size} bytes, expected {size}")
    # Undo the deltas: byte i = byte i − 1 + stored i − 128 (mod 256), the first one as stored.
    steps = deltas.astype(np.int64) - 128
    steps[0] = deltas[0]
    split = (np.cumsum(steps) % 256).astype(np.uint8)
    # Interleave: the first ⌈n/2⌉ bytes are the even positions, the rest the odd ones.
    out = np.empty(size, dtype=np.uint8)
    half = (size + 1) // 2
    out[0::2], out[1::2] = split[:half], split[half:]
    return out.tobytes()


def read_exr(path: Path) -> np.ndarray:
    """The single FLOAT channel "Y" of a scanline EXR file (uncompressed, ZIPS or ZIP) as
    float32 [H, W]; ValueError for any other kind of file."""
    data = Path(path).read_bytes()
    if data[:4] != _MAGIC:
        raise ValueError(f"{path}: not an EXR file")
    (version,) = struct.unpack_from("<i", data, 4)
    if version & 0xFF != 2 or version & _NOT_SCANLINE:
        raise ValueError(f"{path}: not a single-part scanline EXR file")
    attributes: dict[str, tuple[str, bytes]] = {}
    at = 8
    while data[at] != 0:
        name, at = _cstring(data, at)
        type_name, at = _cstring(data, at)
        (size,) = struct.unpack_from("<i", data, at)
        attributes[name] = (type_name, data[at + 4 : at + 4 + size])
        at += 4 + size
    at += 1  # the header's terminating null byte
    channels = attributes["channels"][1]
    name, c = _cstring(channels, 0)
    pixel_type, _, x_sampling, y_sampling = struct.unpack_from("<iB3xii", channels, c)
    if name != "Y" or channels[c + 16 :] != b"\0" or pixel_type != _FLOAT:
        raise ValueError(f"{path}: expected one FLOAT channel Y")
    if (x_sampling, y_sampling) != (1, 1):
        raise ValueError(f"{path}: subsampled channels aren't read")
    compression = attributes["compression"][1][0]
    if compression not in _LINES_PER_BLOCK:
        raise ValueError(f"{path}: compression {compression} isn't read")
    x0, y0, x1, y1 = struct.unpack("<iiii", attributes["dataWindow"][1])
    width, height = x1 - x0 + 1, y1 - y0 + 1
    lines = _LINES_PER_BLOCK[compression]
    blocks = -(-height // lines)
    offsets = struct.unpack_from(f"<{blocks}Q", data, at)
    out = np.empty((height, width), dtype=np.float32)
    for offset in offsets:
        y, size = struct.unpack_from("<ii", data, offset)
        rows = min(lines, y1 + 1 - y)
        raw = data[offset + 8 : offset + 8 + size]
        expected = rows * width * 4
        pixels = raw if compression == 0 else _unzip(raw, expected)
        if len(pixels) != expected:
            raise ValueError(f"{path}: the block at y = {y} holds {len(pixels)} bytes")
        out[y - y0 : y - y0 + rows] = np.frombuffer(pixels, dtype="<f4").reshape(rows, width)
    return out


# -- Expected depth and the residual --------------------------------------------------------------


@dataclass(frozen=True)
class DepthRender:
    depth: torch.Tensor  # [H, W] expected depth, 0 where alpha is 0
    alpha: torch.Tensor  # [H, W]
    means: torch.Tensor  # [N, 3] μ(t), world
    opacities: torch.Tensor  # [N] σ_eff(t)
    visible: torch.Tensor  # [N] bool, as the render reports it


class DepthRenderer:
    """Q-226: a copy of a model with SH degree 0, whose sh0 is set per camera and time so that
    each primitive's colour is its camera-space depth."""

    def __init__(self, model: FreeTimeGaussians) -> None:
        import torch

        from openftgs.model import FreeTimeGaussians

        params = {
            name: getattr(model, name).detach()
            for name in ("means", "times", "log_durations", "velocities", "log_scales")
            + ("quats", "opacity_logits", "sh0")
        }
        params["shN"] = params["sh0"].new_zeros((model.num_gaussians, 0, 3))
        self.model = FreeTimeGaussians(
            params, sh_degree=0, time=model.time, time_unit=model.time_unit
        )
        self.model.requires_grad_(False)
        self._torch = torch

    def __call__(self, camera: Camera, tau: float, *, backend: str) -> DepthRender:
        from openftgs.render import render
        from openftgs.sh import C0

        torch = self._torch
        with torch.no_grad():
            g = self.model.at_time(self.model.time.to_t(tau))
            w2c = camera.world_to_camera.to(g.means_t)
            z = g.means_t @ w2c[2, :3] + w2c[2, 3]
            # Q-226: colour = C0·sh0 + 0.5 = z (E2, which clamps only at 0)
            self.model.sh0.copy_(((z - 0.5) / C0)[:, None, None].expand(-1, 1, 3))
            out = render(self.model, camera, tau, backend=backend, return_aux=True)  # type: ignore[arg-type]
        image, alpha = out.image[0, ..., 0], out.alpha[0, ..., 0]
        depth = torch.where(alpha > 0, image / alpha.clamp_min(1e-12), torch.zeros_like(image))
        assert out.visible is not None
        return DepthRender(depth, alpha, g.means_t, g.opacities_t, out.visible[0])


def depth_render(
    model: FreeTimeGaussians, camera: Camera, tau: float, *, backend: str
) -> DepthRender:
    return DepthRenderer(model)(camera, tau, backend=backend)


def _neighbourhood(values: np.ndarray, fill: float) -> list[np.ndarray]:
    """The 3×3 neighbours of every pixel (itself included), `fill` beyond the image."""
    padded = np.pad(values, 1, constant_values=fill)
    h, w = values.shape
    return [
        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ]


def _surface_around(gt: np.ndarray) -> np.ndarray:
    """Pixels whose 3×3 neighbourhood has a surface everywhere inside the image."""
    keep = np.ones(gt.shape, dtype=bool)
    for nb in _neighbourhood(gt, np.inf):
        keep &= nb > 0
    return keep


def away_from_edges(gt: np.ndarray) -> np.ndarray:
    """Q-228: pixels with a surface and no relative depth jump above 5 %, nor a no-surface pixel,
    within 1 px (their 3×3 neighbourhood)."""
    keep = _surface_around(gt)
    safe = np.where(gt > 0, gt, 1.0)
    for nb in _neighbourhood(gt, np.nan):
        jump = np.abs(nb - gt) / safe
        keep &= ~(jump > EDGE_JUMP)  # NaN beyond the image compares False
    return keep


def residuals(
    means: torch.Tensor, camera: Camera, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The geometry residual r = z − D of each centre (world, [M, 3]) whose nearest pixel and its
    3×3 neighbours have a ground-truth surface (Q-230): r, the rows kept and their pixels
    (u, v)."""
    w2c = camera.world_to_camera.double().cpu().numpy()
    K = camera.K.double().cpu().numpy()
    x = means.detach().double().cpu().numpy() @ w2c[:3, :3].T + w2c[:3, 3]
    z = x[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (x @ K.T)[:, :2] / z[:, None]
    u, v = np.floor(uv[:, 0]), np.floor(uv[:, 1])  # §1.4: pixel (u, v) spans [u, u + 1)
    h, w = gt.shape
    inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    rows = np.flatnonzero(inside)
    ui, vi = u[rows].astype(np.int64), v[rows].astype(np.int64)
    ok = _surface_around(gt)[vi, ui]
    rows, ui, vi = rows[ok], ui[ok], vi[ok]
    return z[rows] - gt[vi, ui].astype(np.float64), rows, np.stack([ui, vi], axis=1)


# -- Statistics -----------------------------------------------------------------------------------


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    if values.size == 0 or weights.sum() <= 0:
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


class _Pool:
    """Q-231: figures over all frames, quantiles from a histogram of |value| in 0.1 mm bins."""

    def __init__(self) -> None:
        self.hist = np.zeros(BINS)
        self.weight = 0.0
        self.count = 0
        self.sum_abs = self.sum_sq = self.sum = 0.0
        self.front = self.behind = 0.0

    def add(self, values: np.ndarray, weights: np.ndarray, delta: float | None = None) -> None:
        bins = np.minimum((np.abs(values) / BIN_M).astype(np.int64), BINS - 1)
        self.hist += np.bincount(bins, weights=weights, minlength=BINS)
        self.weight += float(weights.sum())
        self.count += int(values.size)
        self.sum_abs += float((weights * np.abs(values)).sum())
        self.sum_sq += float((weights * values**2).sum())
        self.sum += float((weights * values).sum())
        if delta is not None:
            self.front += float(weights[values < -delta].sum())
            self.behind += float(weights[values > delta].sum())

    def quantile(self, q: float) -> float | None:
        if self.weight <= 0:
            return None
        k = int(np.searchsorted(np.cumsum(self.hist), q * self.weight))
        return (k + 0.5) * BIN_M


def _depth_figures(error: np.ndarray, surface: int) -> dict[str, Any]:
    """Q-231: an expected-depth comparison over its kept pixels, in metres."""
    n = int(error.size)
    if n == 0:
        return _no_pixels(surface)
    a = np.abs(error)
    return {
        "pixels": n,
        "share_of_surface": n / surface if surface else None,
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
    }


def _no_pixels(surface: int) -> dict[str, Any]:
    empty = dict.fromkeys(("mean_abs", "median_abs", "rmse", "bias"))
    return {"pixels": 0, "share_of_surface": 0.0 if surface else None, **empty}


def _pooled_depth(pool: _Pool, surface: int) -> dict[str, Any]:
    if pool.count == 0:
        return _no_pixels(surface)
    return {
        "pixels": pool.count,
        "share_of_surface": pool.count / surface if surface else None,
        "mean_abs": pool.sum_abs / pool.weight,
        "median_abs": pool.quantile(0.5),
        "rmse": float(np.sqrt(pool.sum_sq / pool.weight)),
        "bias": pool.sum / pool.weight,
    }


def _residual_figures(r: np.ndarray, w: np.ndarray, delta: float) -> dict[str, Any]:
    total = float(w.sum())
    a = np.abs(r)
    return {
        "primitives": int(r.size),
        "weight": total,
        "median_abs": _weighted_quantile(a, w, 0.5),
        "p90_abs": _weighted_quantile(a, w, 0.9),
        "share_in_front": float(w[r < -delta].sum()) / total if total > 0 else None,
        "share_behind": float(w[r > delta].sum()) / total if total > 0 else None,
    }


def _pooled_residual(pool: _Pool) -> dict[str, Any]:
    total = pool.weight
    return {
        "primitives": pool.count,
        "weight": total,
        "median_abs": pool.quantile(0.5),
        "p90_abs": pool.quantile(0.9),
        "share_in_front": pool.front / total if total > 0 else None,
        "share_behind": pool.behind / total if total > 0 else None,
    }


# -- Matching the scene to the ground truth -------------------------------------------------------


def _ground_truth_frame(record: Any, cameras: dict[str, Any]) -> tuple[str, int]:
    """Q-231: the ground-truth camera and decode index n of a test record, after checking its
    camera against the ground truth's."""
    cam = record.camera
    gt = cameras.get(cam.name)
    if gt is None:
        raise InputError(f"camera {cam.name!r} isn't in the ground truth")
    if (cam.width, cam.height) != (gt["width"], gt["height"]):
        raise InputError(
            f"camera {cam.name}: {cam.width}×{cam.height} isn't the ground truth's size"
        )
    K, gt_K = cam.K.double().numpy(), np.array(gt["K_continuous"])
    if not np.allclose(K, gt_K, rtol=K_TOLERANCE, atol=K_TOLERANCE):
        raise InputError(f"camera {cam.name}: K differs from the ground truth's K_continuous")
    pose = cam.world_to_camera.double().numpy()
    if np.abs(pose - np.array(gt["world_to_camera"])).max() > POSE_TOLERANCE:
        raise InputError(f"camera {cam.name}: the pose differs from the ground truth's")
    hits = [f["n"] for f in gt["frames"] if abs(f["t"] - record.timestamp) <= TIME_TOLERANCE_S]
    if len(hits) != 1:
        raise InputError(f"camera {cam.name}: {len(hits)} frames at t = {record.timestamp}")
    return cam.name, int(hits[0])


def run(checkpoint: Path, scene: Path, gt_dir: Path, *, backend: str) -> dict[str, Any]:
    from PIL import Image

    from openftgs.config import config_from_dict
    from openftgs.io import load_checkpoint, read_scene_manifest
    from openftgs.model import FreeTimeGaussians

    try:
        manifest = read_scene_manifest(scene)
        cameras = {
            c["name"]: c for c in json.loads((gt_dir / "cameras.json").read_text())["cameras"]
        }
        state = load_checkpoint(checkpoint)
    except (OSError, ValueError, KeyError) as err:
        raise InputError(str(err)) from None
    records = [r for r in manifest.records if r.split == "test"]
    if not records:
        raise InputError(f"the test split of {scene} holds no images")
    frames = [_ground_truth_frame(r, cameras) for r in records]
    device = "cuda" if backend == "gsplat" else "cpu"
    model = FreeTimeGaussians(
        {k: v.to(device) for k, v in state.params.items()},
        sh_degree=config_from_dict(state.config).sh_degree,
        time=state.time,
        time_unit=state.time_unit,
    )
    renderer = DepthRenderer(model)
    delta = DELTA_OF_S * state.spatial_scale  # Q-229
    regions = ("all", "static", "dynamic")
    depth_pools = {k: _Pool() for k in regions}
    residual_pools = {k: _Pool() for k in regions}
    surface_totals = dict.fromkeys(regions, 0)
    rows_out = []
    for record, (name, n) in zip(records, frames, strict=True):
        try:
            gt = read_exr(gt_dir / "depth" / name / f"{n:06d}.exr")
            with Image.open(gt_dir / "masks" / name / f"{n:06d}.png") as image:
                dynamic = np.asarray(image.convert("L")) > 127
        except (OSError, ValueError) as err:
            raise InputError(str(err)) from None
        out = renderer(record.camera, record.timestamp, backend=backend)
        depth, alpha = out.depth.double().cpu().numpy(), out.alpha.cpu().numpy()
        kept = away_from_edges(gt) & (alpha >= ALPHA_MIN)  # Q-227, Q-228
        error = depth - gt
        masks = {"all": np.ones_like(dynamic), "static": ~dynamic, "dynamic": dynamic}
        frame: dict[str, Any] = {"camera": name, "n": n, "timestamp": record.timestamp}
        frame["depth"] = {}
        for region, mask in masks.items():
            surface = int(((gt > 0) & mask).sum())
            e = error[kept & mask]
            frame["depth"][region] = _depth_figures(e, surface)
            depth_pools[region].add(e, np.ones_like(e))
            surface_totals[region] += surface
        chosen = (out.visible & (out.opacities >= OPACITY_MIN)).nonzero().squeeze(1)
        r, rows, pixels = residuals(out.means[chosen], record.camera, gt)
        w = out.opacities[chosen].double().cpu().numpy()[rows]
        at_dynamic = dynamic[pixels[:, 1], pixels[:, 0]]
        frame["residual"] = {}
        for region, select in (("all", np.ones_like(at_dynamic)), ("static", ~at_dynamic),
                               ("dynamic", at_dynamic)):  # fmt: skip
            frame["residual"][region] = _residual_figures(r[select], w[select], delta)
            residual_pools[region].add(r[select], w[select], delta)
        rows_out.append(frame)
    return {
        "checkpoint": str(checkpoint),
        "iteration": state.iteration,
        "scene": str(scene),
        "ground_truth": str(gt_dir),
        "backend": backend,
        "alpha_min": ALPHA_MIN,
        "edge_jump": EDGE_JUMP,
        "spatial_scale": state.spatial_scale,
        "delta": delta,
        "overall": {
            "depth": {k: _pooled_depth(depth_pools[k], surface_totals[k]) for k in regions},
            "residual": {k: _pooled_residual(residual_pools[k]) for k in regions},
        },
        "frames": rows_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="expected depth and the geometry residual")
    parser.add_argument("checkpoint", type=Path, metavar="CHECKPOINT")
    parser.add_argument("scene", type=Path, metavar="SCENE.json")
    parser.add_argument("gt_dir", type=Path, metavar="GT_DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="RESULT.json")
    parser.add_argument("--backend", choices=("gsplat", "reference"), default="gsplat")
    args = parser.parse_args(argv)
    try:
        result = run(args.checkpoint, args.scene, args.gt_dir, backend=args.backend)
    except InputError as err:
        print(f"gt_depth: error: {err}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    overall = result["overall"]
    print(
        f"gt_depth: median |depth error| {overall['depth']['all']['median_abs']} m, "
        f"median |r| {overall['residual']['all']['median_abs']} m, {len(result['frames'])} frames"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""Camera rigs and request items fitted to a model, for L0 and L1's cases (§3.3; Q-209)."""

import numpy as np

from ftgs_harness.cameras import Cam, intrinsics, look_at
from ftgs_harness.formats import Exchange, RequestItem
from ftgs_harness.rendering import CaseItem

__all__ = ["Rig", "fit_rig", "frames", "main_items"]

FRAMES = 20  # §3.3 L1's rig: 20 frames (Q-209)


class Rig:
    """`count` "training" cameras on a horizontal circle around a model, looking at its centre,
    and novel cameras between them."""

    def __init__(
        self, center: np.ndarray, radius: float, count: int, width: int, height: int
    ) -> None:
        self.center, self.radius, self.count = center, radius, count
        self.width, self.height = width, height
        self.focal = 0.8 * width

    def camera(self, position: float, name: str) -> Cam:
        """The camera at `position` along the circle, in units of the spacing (k is camera k)."""
        angle = 2.0 * np.pi * position / self.count
        eye = self.center + np.array(
            [self.radius * np.cos(angle), self.radius * np.sin(angle), 0.35 * self.radius]
        )
        return Cam(
            name,
            intrinsics(self.width, self.height, self.focal),
            look_at(eye, self.center),
            self.width,
            self.height,
        )

    def training(self) -> list[Cam]:
        return [self.camera(k, f"c{k:02d}") for k in range(self.count)]


def fit_rig(model: Exchange, *, count: int = 8, width: int = 128, height: int = 96) -> Rig:
    """A rig around the model: centred on the median position, at 3× the 90th-percentile
    distance of the positions from it (at least 1)."""
    positions = model.arrays["position"].astype(np.float64)
    center = np.median(positions, axis=0)
    distance = float(np.quantile(np.linalg.norm(positions - center, axis=1), 0.9))
    return Rig(center, 3.0 * max(distance, 1.0 / 3.0), count, width, height)


def frames(model: Exchange) -> np.ndarray:
    """The rig's frame times: FRAMES evenly over [τ₀, τ₁] (one when τ₀ = τ₁)."""
    if model.tau1 == model.tau0:
        return np.array([model.tau0])
    return np.linspace(model.tau0, model.tau1, FRAMES)


def _item(item_id: str, cam: Cam, tau: float) -> CaseItem:
    return CaseItem(
        RequestItem(item_id, float(tau), cam.K, cam.world_to_camera, cam.width, cam.height),
        cam.name,
    )


def main_items(model: Exchange, rig: Rig, count: int, rng: np.random.Generator) -> list[CaseItem]:
    """§3.3 L1's request of `count` items: training cameras on frames (30 %), between frames
    (15 %) and at random times (10 %), novel cameras between training ones (25 %), at τ₀ and τ₁
    (10 %), and 10 % outside [τ₀, τ₁] (Q-209)."""
    times = frames(model)
    span = model.tau1 - model.tau0
    step = span / (len(times) - 1) if len(times) > 1 else 1.0
    n_out = int(np.ceil(0.1 * count))
    n_ends = int(round(0.1 * count))
    n_novel = int(round(0.25 * count))
    n_between = int(round(0.15 * count))
    n_frames = int(round(0.3 * count))
    n_random = max(count - n_out - n_ends - n_novel - n_between - n_frames, 0)
    plan: list[tuple[str, float, float]] = []  # (kind, camera position, τ)
    cams = rig.count
    for _ in range(n_frames):
        plan.append(("train", int(rng.integers(cams)), float(rng.choice(times))))
    for _ in range(n_between):
        k = int(rng.integers(max(len(times) - 1, 1)))
        plan.append(
            ("train", int(rng.integers(cams)), float(times[k] + rng.uniform(0.2, 0.8) * step))
        )
    for _ in range(n_random):
        plan.append(("train", int(rng.integers(cams)), float(rng.uniform(model.tau0, model.tau1))))
    for _ in range(n_novel):
        position = int(rng.integers(cams)) + float(rng.uniform(0.2, 0.8))
        plan.append(("novel", position, float(rng.uniform(model.tau0, model.tau1))))
    for k in range(n_ends):
        plan.append(("train", int(rng.integers(cams)), model.tau1 if k % 2 else model.tau0))
    for k in range(n_out):
        reach = rng.uniform(0.05, 0.3) * max(span, 1.0)
        plan.append(
            ("train", int(rng.integers(cams)), model.tau1 + reach if k % 2 else model.tau0 - reach)
        )
    out = []
    for i, (kind, position, tau) in enumerate(plan):
        name = f"c{int(position):02d}" if kind == "train" else f"n{position:.2f}"
        out.append(_item(f"r{i:04d}", rig.camera(position, name), tau))
    return out


def small_items(model: Exchange, rig: Rig) -> list[CaseItem]:
    """L0's 4-item request: two training cameras, each at two frame times."""
    times = frames(model)
    picks = [
        (0, times[0]),
        (0, times[len(times) // 2]),
        (rig.count // 2, times[-1]),
        (rig.count // 2, times[len(times) // 3]),
    ]
    return [_item(f"r{i:04d}", rig.camera(k, f"c{k:02d}"), t) for i, (k, t) in enumerate(picks)]

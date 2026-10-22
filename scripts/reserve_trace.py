# SPDX-License-Identifier: Apache-2.0
"""What PyTorch's allocator holds after every iteration of a training run's first epoch, and
whether frame cache "auto" reads a settled reserve at its decision point (#350; Q-238, Q-232,
Q-225; D-51).

    python scripts/reserve_trace.py SCENE.json --init INIT --out TRACE.json [--iterations N]

A trainer of the default configuration, with only the frame cache set to "off" so that no cache
moves the allocator, trains on SCENE.json from INIT on CUDA, cuDNN autotuning on as in training
(§2.4 rule 2, Q-173), for N iterations (default: one epoch, one iteration per training image,
D-26). After each iteration, once `step()` has returned, as B-6's probe reads them (Q-189), without
a synchronization:

- `free_bytes`: `torch.cuda.mem_get_info`'s free memory;
- `reserved_bytes`, `allocated_bytes` and the reserve, reserved − allocated (Q-225);
- `num_device_free`: `torch.cuda.memory_stats`' count of the blocks the allocator has returned to
  the driver, so far in the process;
- whether the iteration was the first at its image size (W, H), counted over training images as
  "auto" counts them (Q-232); with the image, its camera, N after the iteration
  (`num_gaussians`) and whether it ran a growth, relocation or pruning event (`event`).

The summary (`summary`) gives k, the iteration at which the last training size first appears;
the reserve after iteration k + 1, where "auto" decides (Q-232); the steady reserve, the median
over the iterations after k + 1, and the smallest of them; and how many first-at-a-size
iterations returned blocks to the driver (`num_device_free` rose across them). The trace passes
when the reserve at the decision is at least 90 % of the steady reserve: Q-232's rule then reads
a settled allocator (#350).

TRACE.json holds the setup, the summary and every iteration's readings. This is a diagnostic,
not a pass criterion of the spec: exit 0 once written, pass or fail; 2 for an input that can't
be read or a trace too short to reach k + 1 and an iteration after it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SETTLED = 0.9  # Q-238 (#350): the decision's reserve against the steady reserve's median


def summary(
    rows: Sequence[dict[str, Any]], *, training_sizes: int, frees_before: int
) -> dict[str, Any]:
    """The trace's figures from its per-iteration `rows` (iterations 1, 2, … in order), given
    the scene's number of distinct training sizes and `num_device_free` before iteration 1. A
    trace in which some size never comes, or that ends before an iteration after k + 1, is a
    ValueError."""
    firsts = [r["iteration"] for r in rows if r["first_at_size"]]
    if len(firsts) != training_sizes:
        raise ValueError(
            f"the trace reaches {len(firsts)} of the {training_sizes} training image sizes"
        )
    k = max(firsts)
    by_iteration = {r["iteration"]: r for r in rows}
    later = [_reserve(r) for r in rows if r["iteration"] > k + 1]
    if k + 1 not in by_iteration or not later:
        raise ValueError(f"the trace ends before an iteration after k + 1 = {k + 1}")
    decision = by_iteration[k + 1]
    steady = statistics.median(later)
    previous = frees_before
    returned, returned_elsewhere = 0, 0
    for row in rows:
        if row["num_device_free"] > previous:
            if row["first_at_size"]:
                returned += 1
            else:
                returned_elsewhere += 1
        previous = row["num_device_free"]
    ratio = _reserve(decision) / steady if steady > 0 else float("inf")
    return {
        "training_sizes": training_sizes,
        "k": k,
        "decision_iteration": k + 1,
        "decision_reserve_bytes": _reserve(decision),
        "decision_free_bytes": decision["free_bytes"],
        "decision_reserved_bytes": decision["reserved_bytes"],
        "decision_allocated_bytes": decision["allocated_bytes"],
        "steady_reserve_bytes": steady,
        "smallest_later_reserve_bytes": min(later),
        "later_iterations": len(later),
        "ratio": ratio,
        "threshold": _SETTLED,
        "passes": ratio >= _SETTLED,
        "firsts_returning_blocks": returned,
        "others_returning_blocks": returned_elsewhere,
    }


def _reserve(row: dict[str, Any]) -> int:
    return int(row["reserved_bytes"]) - int(row["allocated_bytes"])


def _trace(scene: Path, init: Path, iterations: int | None) -> dict[str, Any]:
    import torch

    from openftgs.config import TrainConfig
    from openftgs.data import MultiViewVideo
    from openftgs.io import read_init_points
    from openftgs.train import Trainer

    video = MultiViewVideo.from_manifest(scene)
    points = read_init_points(init, time_unit=video.manifest.time_unit)
    training = [r for r in video.records if r.split == "train"]
    sizes = {(r.camera.width, r.camera.height) for r in training}
    count = len(training) if iterations is None else iterations
    cfg = TrainConfig()
    cfg = dataclasses.replace(cfg, perf=dataclasses.replace(cfg.perf, frame_cache="off"))
    trainer = Trainer(cfg, video, points, device="cuda")
    frees_before = int(torch.cuda.memory_stats()["num_device_free"])
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for _ in range(count):
        log = trainer.step()
        free, _total = torch.cuda.mem_get_info()
        reserved, allocated = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
        frees = int(torch.cuda.memory_stats()["num_device_free"])
        camera = video.records[log.image_index].camera
        size = (camera.width, camera.height)
        rows.append(
            {
                "iteration": log.iteration,
                "image_index": log.image_index,
                "camera": camera.name,
                "width": camera.width,
                "height": camera.height,
                "first_at_size": size not in seen,
                "free_bytes": free,
                "reserved_bytes": reserved,
                "allocated_bytes": allocated,
                "reserve_bytes": reserved - allocated,
                "num_device_free": frees,
                "num_gaussians": log.num_gaussians,
                "event": log.growth is not None
                or log.relocation is not None
                or log.pruned is not None,
            }
        )
        seen.add(size)
    setup = {
        "scene": str(scene),
        "scene_sha256": video.manifest.sha256,
        "init": str(init),
        "init_points": int(points.positions.shape[0]),
        "configuration": 'the default TrainConfig with perf.frame_cache = "off"',
        "training_images": len(training),
        "iterations": count,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "num_device_free_before": frees_before,
    }
    return {
        "setup": setup,
        "summary": summary(rows, training_sizes=len(sizes), frees_before=frees_before),
        "iterations": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the allocator over a first epoch (#350)")
    parser.add_argument("scene", type=Path, metavar="SCENE.json")
    parser.add_argument("--init", type=Path, required=True, metavar="INIT")
    parser.add_argument("--out", type=Path, required=True, metavar="TRACE.json")
    parser.add_argument("--iterations", type=int, default=None, help="default: one epoch")
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations < 1:
        parser.error(f"--iterations must be at least 1, got {args.iterations}")
    try:
        trace = _trace(args.scene, args.init, args.iterations)
    except (OSError, ValueError) as err:
        print(f"reserve_trace: {err}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(trace, indent=1) + "\n")
    print(json.dumps(trace["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

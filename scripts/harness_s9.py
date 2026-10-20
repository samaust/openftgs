# SPDX-License-Identifier: Apache-2.0
"""S-9's scene for the differential harness's L3 (§3.2 S-9, §3.3 L3; Q-203; #292).

    python scripts/harness_s9.py OUT [--k K] [--gpus 0,1] [--seed S] [--trained N]

Writes, under the folder OUT, with the test suite's own S-9 helpers (tests/synthetic/raycast.py
and tests/synthetic/data/seeds.json), so the harness gets the scene S-9 trains on without a copy
of the ray caster (owner decision, 2026-10-20):

- `scene/`: the ray-cast PNGs, `scene/scene.json` (every image, the held-out cameras as the test
  split) and `scene/train.json` (the training images only);
- `init.safetensors`: the surface samples, 300 per sphere and frame (Q-140);
- `harness_config.s9.json`: S-9's configuration, 7,000 iterations on §3.2's shortened calendar
  without the perceptual loss (Q-140), without `seed`, which the harness sets per run;
- `trained/`: N short trainings of S-9 (700 iterations on the same shortened calendar, seeds
  0 to N − 1; 2 by default) through `ftgs train` and `ftgs export`, the trained models L0 sends
  through the converters with its random ones (§3.3 L0: "random and trained"; Q-209). They train
  on the GPU when there is one; on the CPU they take about 10 min each;
- `settings.s9.json`: run settings naming the scene, with `train_manifest` the training-only
  manifest so `ftgs train` doesn't evaluate at its end, K (5 by default), the GPUs and the
  trained models.

Then: ftgs-harness run --a "ftgs-adapter --backend gsplat" --b "…" --config
OUT/harness_config.s9.json --settings OUT/settings.s9.json --levels L0,L1,L2,L3 --out RUN_DIR.
Exit 0 once written, 2 for invalid arguments or an OUT that holds something.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ITERATIONS = 7_000  # Q-140: S-9's run
TRAINED_ITERATIONS = 700  # L0's trained models: a tenth of S-9's run
PER_SPHERE = 300  # Q-140: lattice points per sphere and frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--k", type=int, default=5, help="seeds per implementation (§3.3: 5)")
    parser.add_argument("--gpus", default="", help="comma-separated GPU ids for L3's runs")
    parser.add_argument("--seed", type=int, default=0, help="the harness's case seed")
    parser.add_argument("--trained", type=int, default=2, help="trained models for L0")
    args = parser.parse_args(argv)
    if args.k < 2:
        parser.error("--k must be at least 2")
    if args.trained < 0:
        parser.error("--trained must be at least 0")
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        print(f"harness_s9: {out} must be absent or empty", file=sys.stderr)
        return 2
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "synthetic")]
    import raycast
    from synthetic_scenes import load_seeds, synthetic_config

    from openftgs.cli import main as ftgs
    from openftgs.config import TrainConfig, config_to_dict
    from openftgs.io import write_init_points

    seeds = load_seeds()["s9"]
    out.mkdir(parents=True, exist_ok=True)
    scene = raycast.write_scene(out / "scene", seeds["spheres"])
    points = raycast.surface_samples(scene.spheres, seeds["lattice"], PER_SPHERE)
    write_init_points(points, out / "init.safetensors", time_unit="s", producer="harness_s9 (S-9)")
    config = config_to_dict(synthetic_config(ITERATIONS, TrainConfig()))
    del config["seed"]  # Q-203: the harness gives every run its own
    (out / "harness_config.s9.json").write_text(json.dumps(config, indent=2) + "\n")
    trained = []
    if args.trained:
        short = out / "trained" / "config.json"
        short.parent.mkdir()
        short.write_text(
            json.dumps(config_to_dict(synthetic_config(TRAINED_ITERATIONS, TrainConfig())))
        )
        for seed in range(args.trained):
            run = out / "trained" / f"run{seed}"
            model = out / "trained" / f"model{seed}.ftgs-if.safetensors"
            train = [str(scene.train_manifest), "--init", str(out / "init.safetensors")]
            train += ["--config", str(short), "--seed", str(seed), "--out", str(run)]
            if ftgs(["train", *train]) != 0:
                return 2
            if ftgs(
                ["export", str(run / "checkpoint"), "--format", "ftgs-if", "--out", str(model)]
            ):
                return 2
            trained.append(str(model.relative_to(out)))
    settings = {
        "seed": args.seed,
        "k": args.k,
        "gpus": [g for g in args.gpus.split(",") if g],
        "l0": {"trained_models": trained},
        "l3": {
            "scenes": [
                {
                    "name": "s9",
                    "manifest": str(scene.manifest.relative_to(out)),
                    "train_manifest": str(scene.train_manifest.relative_to(out)),
                    "init": "init.safetensors",
                }
            ]
        },
    }
    (out / "settings.s9.json").write_text(json.dumps(settings, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

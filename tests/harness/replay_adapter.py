# SPDX-License-Identifier: Apache-2.0
"""A replay adapter for the harness's L3 tests (ticket #291): `train` copies a pre-made run
directory instead of training, and records the CUDA_VISIBLE_DEVICES it was given; every other
command goes to openftgs's adapter.

    python replay_adapter.py --source DIR [--not-comparable KEY …] <command> <args…>

DIR holds run0, run1, …; seed s replays run(s mod count).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ADAPTER = str(Path(sys.executable).parent / "ftgs-adapter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--not-comparable", action="append", default=[])
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()
    if opts.command != "train":
        return subprocess.run(
            [ADAPTER, "--backend", "reference", opts.command, *opts.args], check=False
        ).returncode
    _scene, _init, _config, seed, out = opts.args
    runs = sorted(p for p in opts.source.iterdir() if p.name.startswith("run"))
    shutil.copytree(runs[int(seed) % len(runs)], out)
    (Path(out) / "env.json").write_text(
        json.dumps({"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")})
    )
    if opts.not_comparable:
        (Path(out) / "not_comparable.json").write_text(json.dumps(opts.not_comparable))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""Write the committed ground truth of the §3.2 synthetic scenes (`data/*.safetensors`) from
the committed seeds (`data/seeds.json`). Run from the repository root:

    python tests/synthetic/make_ground_truth.py

A test checks that the committed files are what this script writes, so the seeds and the files
can't drift apart.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent)]

from synthetic_scenes import DATA, GROUND_TRUTH_FILES, generate_ground_truth, load_seeds

from openftgs.io import write_exchange


def main() -> None:
    for scene, model in generate_ground_truth(load_seeds()).items():
        path = DATA / GROUND_TRUTH_FILES[scene]
        write_exchange(model, path)
        print(f"{path}: {model.arrays['position'].shape[0]} Gaussians")


if __name__ == "__main__":
    main()

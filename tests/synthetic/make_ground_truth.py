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

import torch
from synthetic_scenes import DATA, GROUND_TRUTH_FILES, generate_ground_truth, load_seeds

from openftgs.io import read_exchange, write_exchange
from openftgs.types import ExchangeModel


def _same(path: Path, model: ExchangeModel) -> bool:
    """Whether `path` already holds `model`: safetensors orders the header's metadata
    differently from one write to the next, so an unchanged file is left as committed."""
    if not path.is_file():
        return False
    old = read_exchange(path)
    return (
        (old.sh_degree, old.tau0, old.tau1, old.time_unit, old.producer)
        == (model.sh_degree, model.tau0, model.tau1, model.time_unit, model.producer)
        and old.arrays.keys() == model.arrays.keys()
        and all(torch.equal(old.arrays[k], v) for k, v in model.arrays.items())
    )


def main() -> None:
    for scene, model in generate_ground_truth(load_seeds()).items():
        path = DATA / GROUND_TRUTH_FILES[scene]
        if _same(path, model):
            print(f"{path}: unchanged")
            continue
        write_exchange(model, path)
        print(f"{path}: {model.arrays['position'].shape[0]} Gaussians")


if __name__ == "__main__":
    main()

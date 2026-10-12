# SPDX-License-Identifier: Apache-2.0
"""Marker handling for the test suite (§3 introduction; PRD #1).

Tests not marked ``gpu`` get the ``cpu`` marker and run on the reference backend.
``gpu``, ``dataset`` and ``weights`` tests are skipped when their resource is absent.
"""

import os
from pathlib import Path

import pytest


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    datasets = os.environ.get("OPENFTGS_DATASETS")
    have_datasets = datasets is not None and Path(datasets).is_dir()
    allow_downloads = os.environ.get("OPENFTGS_ALLOW_DOWNLOADS") == "1"
    have_cuda: bool | None = None
    for item in items:
        if item.get_closest_marker("gpu") is None:
            item.add_marker(pytest.mark.cpu)
        else:
            if have_cuda is None:
                have_cuda = _cuda_available()
            if not have_cuda:
                item.add_marker(pytest.mark.skip(reason="needs a CUDA device"))
        if item.get_closest_marker("dataset") is not None and not have_datasets:
            item.add_marker(pytest.mark.skip(reason="OPENFTGS_DATASETS is not a directory"))
        if item.get_closest_marker("weights") is not None and not allow_downloads:
            item.add_marker(pytest.mark.skip(reason="OPENFTGS_ALLOW_DOWNLOADS is not 1"))

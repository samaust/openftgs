# SPDX-License-Identifier: Apache-2.0
"""Marker handling for the test suite (§3 introduction; PRD #1).

Tests not marked ``gpu`` get the ``cpu`` marker and run on the reference backend.
``gpu``, ``dataset`` and ``weights`` tests are skipped when their resource is absent.
A session that runs ``gpu`` tests opens one CPU+CUDA profiler session before any test
(Q-200).
"""

import os
from pathlib import Path

import pytest


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Q-7: the dataset directory and the download permission come from the environment.
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


@pytest.fixture(scope="session", autouse=True)
def _cuda_profiler_warm_up(request: pytest.FixtureRequest) -> None:
    """Q-200: with PyTorch 2.14.1, a CPU+CUDA profiler session that follows a CPU-only one in
    the same process records no GPU side (no kernels, no GPU-side copies of the ranges), while
    one CPU+CUDA session first keeps every later one whole. So a session that runs a ``gpu``
    test opens an empty one before its first test; sessions without one are left alone. The
    selected items are read here, after ``-m`` has deselected its tests."""
    if not any(
        item.get_closest_marker("gpu") is not None and item.get_closest_marker("skip") is None
        for item in request.session.items
    ):
        return
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities):
        pass

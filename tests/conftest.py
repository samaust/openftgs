# SPDX-License-Identifier: Apache-2.0
"""Marker handling and parallel runs for the test suite (§3 introduction; PRD #1; Q-214).

Tests not marked ``gpu`` get the ``cpu`` marker and run on the reference backend.
``gpu``, ``dataset`` and ``weights`` tests are skipped when their resource is absent.
A session that runs ``gpu`` tests opens one CPU+CUDA profiler session before any test
(Q-200).

Under pytest-xdist (``-n``), the run's thread budget is divided among its workers (Q-214).
"""

import os
from pathlib import Path

import pytest

# Q-214: the variables machine.py sets for a budget, set again in each worker to its share.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCHINDUCTOR_COMPILE_THREADS",
)
# Q-214: the threads `-n auto` gives each worker; two ran the CPU suite fastest on the i9-12900
# (8 workers × 2 against 16 × 1 and 4 × 4).
_WORKER_THREADS = 2


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _budget() -> int:
    """Q-214: the run's CPU threads: the smallest positive count of OMP_NUM_THREADS and
    MKL_NUM_THREADS (machine.py's `--threads`, a subagent's budget), as `openftgs.data` takes it
    (Q-215), otherwise the physical cores this process may run on, the count ftgs itself takes
    (§2.4, Q-176). Not imported from `openftgs.data`, whose NumPy would read the BLAS variable
    before a worker sets its share."""
    values = (os.environ.get(name, "") for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    counts = [int(value) for value in values if value.isdigit() and int(value) > 0]
    if counts:
        return min(counts)
    from openftgs.cli import _physical_cores

    return _physical_cores()


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """Q-214: `-n auto` starts a worker per two threads of the budget, not one per logical CPU."""
    return max(1, _budget() // _WORKER_THREADS)


def pytest_cmdline_main(config: pytest.Config) -> None:
    """Q-214: more workers than the budget's threads would oversubscribe the cores it holds."""
    if hasattr(config, "workerinput") or config.getoption("dist", "no") == "no":
        return
    workers = len(config.getoption("tx", None) or [])
    if workers > _budget():
        raise pytest.UsageError(
            f"-n {workers} is more workers than the budget's {_budget()} threads "
            "(OMP_NUM_THREADS, or the physical cores); use -n auto"
        )


def pytest_configure(config: pytest.Config) -> None:
    """Q-214: each xdist worker runs at its share of the budget, through the variables that its
    subprocesses inherit and that keep ftgs's `main()` from raising the count to every core."""
    if not config.pluginmanager.hasplugin("xdist"):
        config.addinivalue_line("markers", "xdist_group(name): tests that share one xdist worker")
    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return
    threads = str(max(1, _budget() // int(workerinput["workercount"])))
    for name in _THREAD_VARIABLES:
        os.environ[name] = threads
    import torch  # read the variables at import; set again in case a plugin imported it first

    torch.set_num_threads(int(threads))


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
            # Q-214: under --dist loadgroup, GPU tests in no other group take turns on one worker.
            if item.get_closest_marker("xdist_group") is None:
                item.add_marker(pytest.mark.xdist_group("gpu"))
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

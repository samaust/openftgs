# SPDX-License-Identifier: Apache-2.0
"""Environment check for the CUDA build of torch (§2.3 torch row; Q-3; PRD #1).

Added test, not a §3 row: on a machine with an NVIDIA GPU, torch must be a CUDA
build that sees it; otherwise every gpu-marked test would skip silently.
"""

import shutil
import subprocess

import pytest


def _nvidia_gpu_present() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    listing = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, check=False)
    return listing.returncode == 0 and "GPU" in listing.stdout


def test_cuda_torch_on_gpu_machine() -> None:
    if not _nvidia_gpu_present():
        pytest.skip("no NVIDIA GPU on this machine")
    import torch

    assert torch.version.cuda is not None, "torch is a CPU build on a machine with a GPU"
    assert torch.cuda.is_available()

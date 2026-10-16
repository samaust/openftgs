# SPDX-License-Identifier: Apache-2.0
"""U-38: the vendored RoMa in ``openftgs._vendor.romatch`` (D-65; #134, #137).

D-65 as Q-113 and Q-114 amend it: RoMa's 22 inference files at commit 77f8d68, with the
absolute ``romatch`` imports pointing to the vendored path, no OpenCV import, and no
xformers import in the DINOv2 layers. The comparison of ``match()`` with upstream romatch
at 77f8d68 runs on a GPU with downloaded weights, when upstream is importable (Q-117;
``docs/modules/init/vendored-roma.md`` says how).
"""

import ast
import contextlib
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from builders import TexturedPlane, look_at
from PIL import Image

import openftgs

VENDORED = Path(openftgs.__file__).resolve().parent / "_vendor" / "romatch"


def _vendored_sources() -> list[Path]:
    return sorted(VENDORED.rglob("*.py"))


def _absolute_romatch_imports(path: Path) -> list[int]:
    """Line numbers of imports of ``romatch`` itself, at any depth in the file."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names = [node.module]
        else:
            continue
        if any(name == "romatch" or name.startswith("romatch.") for name in names):
            lines.append(node.lineno)
    return lines


def _vendor_json() -> dict[str, Any]:
    record = json.loads((VENDORED / "VENDOR.json").read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    return record


def test_u38_every_vendored_file_has_the_sha256_of_vendor_json() -> None:
    record = _vendor_json()
    assert record["commit"] == "77f8d68803526dcddfd9b7a46bc76125bdc25f15"  # D-65
    listed = {entry["path"]: entry["sha256"] for entry in record["files"]}
    on_disk = {path.relative_to(VENDORED).as_posix() for path in _vendored_sources()}
    assert len(listed) == 22
    assert set(listed) == on_disk
    actual = {path: hashlib.sha256((VENDORED / path).read_bytes()).hexdigest() for path in listed}
    assert actual == listed


def test_u38_license_files_have_the_sha256_of_vendor_json() -> None:
    # Q-115: RoMa's license at the copy's root, the others in LICENSES beside it.
    record = _vendor_json()
    listed = {entry["path"]: entry["sha256"] for entry in record["licenses"]}
    assert {entry["project"] for entry in record["licenses"]} == {
        "RoMa",
        "DINOv2",
        "DINO",
        "timm",
        "LoFTR",
        "Patch2Pix",
        "Kornia",
    }
    assert "LICENSE" in listed
    on_disk = {path.relative_to(VENDORED).as_posix() for path in (VENDORED / "LICENSES").iterdir()}
    assert set(listed) == on_disk | {"LICENSE"}
    actual = {path: hashlib.sha256((VENDORED / path).read_bytes()).hexdigest() for path in listed}
    assert actual == listed


# Runs in a subprocess, since other tests import cv2 into the pytest process (Q-117).
_IMPORT_WITHOUT_CV2 = """
import importlib.abc
import json
import sys


class _BlockCv2(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("cv2 is blocked by U-38's import hook")
        return None


assert "cv2" not in sys.modules
sys.meta_path.insert(0, _BlockCv2())
import openftgs._vendor.romatch

print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in ("cv2", "xformers"))))
"""

# A stand-in for an installed xformers: importable, with the names the DINOv2 layers ask
# for, and recording any import in a marker file (Q-117).
_STUB_XFORMERS = """
import os
import pathlib

pathlib.Path(os.environ["U38_XFORMERS_MARKER"]).write_text("imported")
"""
_STUB_XFORMERS_OPS = """
fmha = None


def memory_efficient_attention(*args, **kwargs):
    raise NotImplementedError


def unbind(*args, **kwargs):
    raise NotImplementedError


def scaled_index_add(*args, **kwargs):
    raise NotImplementedError


def index_select_cat(*args, **kwargs):
    raise NotImplementedError


class SwiGLU:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
"""


def test_u38_imports_without_cv2_and_never_imports_xformers(tmp_path: Path) -> None:
    stub = tmp_path / "site" / "xformers"
    (stub / "ops").mkdir(parents=True)
    (stub / "__init__.py").write_text(_STUB_XFORMERS, encoding="utf-8")
    (stub / "ops" / "__init__.py").write_text(_STUB_XFORMERS_OPS, encoding="utf-8")
    marker = tmp_path / "xformers-imported"
    source_root = Path(openftgs.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path / "site"), str(source_root)])
    env["U38_XFORMERS_MARKER"] = str(marker)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WITHOUT_CV2],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == []
    assert not marker.exists()


def test_u38_no_absolute_romatch_import_in_the_vendored_tree() -> None:
    sources = _vendored_sources()
    assert len(sources) == 22  # D-65's inference files (source log row 116)
    found = {
        str(path.relative_to(VENDORED)): lines
        for path in sources
        if (lines := _absolute_romatch_imports(path))
    }
    assert found == {}


# U-38's comparison with upstream (GPU, weights; Q-117).


def _git_blob_id(data: bytes) -> str:
    """The id git gives a file's bytes: SHA-1 of "blob <size>\\0" and the bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _upstream_zoo() -> Any:
    """Upstream romatch's model zoo, or a skip unless the importable romatch holds the files
    of 77f8d68 that the copy was made from (their git blob ids, as VENDOR.json records them;
    only paths are reported, never contents)."""
    upstream = pytest.importorskip("romatch", reason="upstream romatch is not importable (Q-117)")
    root = Path(upstream.__file__).resolve().parents[1]
    differing = [
        entry["upstream_path"]
        for entry in _vendor_json()["files"]
        if not (root / entry["upstream_path"]).is_file()
        or _git_blob_id((root / entry["upstream_path"]).read_bytes()) != entry["upstream_blob"]
    ]
    if differing:
        pytest.skip(f"the importable romatch isn't 77f8d68: {differing} differ")
    return importlib.import_module("romatch.models.model_zoo")


@contextlib.contextmanager
def _strict() -> Iterator[None]:
    """The flags `roma_init` runs RoMa under (Q-112), restored afterwards."""
    cudnn = torch.backends.cudnn
    saved = (
        torch.get_float32_matmul_precision(),
        cudnn.benchmark,
        cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.set_float32_matmul_precision("highest")
        cudnn.benchmark, cudnn.deterministic = False, True
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.set_float32_matmul_precision(saved[0])
        cudnn.benchmark, cudnn.deterministic = saved[1], saved[2]
        torch.use_deterministic_algorithms(saved[3], warn_only=saved[4])


def _pair(seed: int) -> tuple[Image.Image, Image.Image]:
    """A procedurally generated image pair: a textured plane (seeded) seen from two cameras.
    Pair 0 is a small baseline at one size; pair 1 a wider one, at two sizes."""
    plane = TexturedPlane(seed=100 + seed)
    views = [((-8.0, 0.3), (320, 256)), ((8.0, 0.1), (320, 256))]
    if seed == 1:
        views = [((-20.0, 0.6), (384, 288)), ((15.0, -0.2), (256, 192))]
    images = []
    for (yaw, height), (width, rows) in views:
        theta = math.radians(yaw)
        eye = (3.0 * math.sin(theta), -3.0 * math.cos(theta), height)
        camera = look_at(eye, width=width, height=rows, focal=0.875 * width)
        images.append(Image.fromarray(plane.render(camera, 0.0).numpy(), "RGB"))
    return images[0], images[1]


@pytest.mark.gpu
@pytest.mark.weights
def test_u38_vendored_match_agrees_with_upstream_romatch(
    record_property: Callable[[str, object], None],
) -> None:
    """U-38 (GPU, weights): on two procedurally generated, seeded image pairs, the vendored and
    upstream (77f8d68) roma_outdoor, built from the same weights with the same settings, give
    `match()` warps and certainties within 1e-5. Both run under Q-112's flags, with
    `use_custom_corr` off (Q-6)."""
    upstream_zoo = _upstream_zoo()
    vendored_zoo = importlib.import_module("openftgs._vendor.romatch.models.model_zoo")
    urls = vendored_zoo.weight_urls
    assert upstream_zoo.weight_urls == urls
    weights = torch.hub.load_state_dict_from_url(urls["romatch"]["outdoor"], map_location="cpu")
    dinov2 = torch.hub.load_state_dict_from_url(urls["dinov2"], map_location="cpu")
    device = torch.device("cuda:0")
    differences: dict[str, float] = {}
    with _strict():
        models = [
            zoo.roma_outdoor(
                device,
                weights=dict(weights),
                dinov2_weights=dict(dinov2),
                coarse_res=560,
                upsample_res=864,
                use_custom_corr=False,
            )
            for zoo in (vendored_zoo, upstream_zoo)
        ]
        for seed in (0, 1):
            image_a, image_b = _pair(seed)
            (warp, certainty), (warp_up, certainty_up) = (
                model.match(image_a, image_b, device=device) for model in models
            )
            differences[f"warp {seed}"] = float((warp - warp_up).abs().max())
            differences[f"certainty {seed}"] = float((certainty - certainty_up).abs().max())
            torch.testing.assert_close(warp, warp_up, rtol=0.0, atol=1e-5)
            torch.testing.assert_close(certainty, certainty_up, rtol=0.0, atol=1e-5)
            assert float(certainty.max()) > 0.5  # the pair is matched, not blank
    record_property("max_abs_differences", json.dumps(differences))
    print(f"U-38: largest differences {differences}")

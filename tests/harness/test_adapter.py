# SPDX-License-Identifier: Apache-2.0
"""`ftgs-adapter`, openftgs's adapter for the differential harness (§3.3 adapter contract,
D-73; Q-202, Q-205), run as the harness runs it: one process per command."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from builders import exchange_arrays, random_model, ring, write_scene

from openftgs.config import LossConfig, PerfConfig, TrainConfig, config_to_dict
from openftgs.io import (
    load_checkpoint,
    read_exchange,
    read_renders,
    render_conventions,
    write_exchange,
    write_init_points,
)
from openftgs.types import ExchangeModel, InitPoints, TimeNormalizer

ADAPTER = str(Path(sys.executable).parent / "ftgs-adapter")
FTGS = str(Path(sys.executable).parent / "ftgs")


def _adapter(*args: object, backend: str = "reference") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ADAPTER, "--backend", backend, *(str(a) for a in args)],
        capture_output=True,
        text=True,
        check=False,
    )


def _exchange(path: Path, *, n: int = 12, sh_degree: int = 2, seed: int = 0) -> Path:
    arrays = exchange_arrays(n, sh_degree, seed)
    arrays["position"] = arrays["position"] * 0.3  # inside the request's view
    arrays["scale"] = arrays["scale"] * 0.2
    write_exchange(
        ExchangeModel(arrays, sh_degree, 2.0, 12.0, "s", "test", background=(0.0, 0.2, 1.0)), path
    )
    return path


def _request(path: Path) -> Path:
    items = []
    for k, cam in enumerate(ring(3, radius=4.0, elevation=1.0, width_px=24, height_px=16)):
        items.append(
            {
                "id": f"r{k}",
                "timestamp": 2.0 + 3.0 * k,
                "camera": {
                    "width": cam.width,
                    "height": cam.height,
                    "K": cam.K.tolist(),
                    "world_to_camera": cam.world_to_camera.tolist(),
                },
            }
        )
    path.write_text(
        json.dumps(
            {"format": "ftgs-render-request", "version": 1, "time_unit": "s", "items": items}
        )
    )
    return path


@pytest.mark.parametrize(
    ("backend", "rasterizer"), [("reference", "openftgs-reference"), ("gsplat", "gsplat")]
)
def test_info_reports_the_render_conventions_and_the_rasterizer(
    backend: str, rasterizer: str
) -> None:
    """D-73, Q-205: the 12 render_conventions entries (Q-35), black (D-34), and the rasterizer
    that produces this backend's pixels."""
    done = _adapter("info", backend=backend)
    assert done.returncode == 0, done.stderr
    info = json.loads(done.stdout)
    assert info["name"] == "openftgs" and isinstance(info["version"], str)
    assert info["render_conventions"] == render_conventions()
    assert info["rasterizer"]["name"] == rasterizer
    assert isinstance(info["query_kinds"], list)


def test_import_then_export_round_trips_an_exchange_file(tmp_path: Path) -> None:
    """§3.3 L0: through the native directory and back, every field within 1e-6 relative
    (opacity after the clamp, quaternions up to sign), the background exact (D-84)."""
    source = _exchange(tmp_path / "in.safetensors")
    assert _adapter("import", source, tmp_path / "native").returncode == 0
    assert (tmp_path / "native" / "params.safetensors").is_file()
    done = _adapter("export", tmp_path / "native", tmp_path / "out.safetensors")
    assert done.returncode == 0, done.stderr
    a, b = read_exchange(source), read_exchange(tmp_path / "out.safetensors")
    assert (b.sh_degree, b.tau0, b.tau1, b.background) == (2, 2.0, 12.0, (0.0, 0.2, 1.0))
    for key in a.arrays:
        x, y = a.arrays[key].double(), b.arrays[key].double()
        if key == "rotation":
            y = torch.where((x * y).sum(-1, keepdim=True) < 0, -y, y)
        torch.testing.assert_close(y, x, rtol=1e-6, atol=1e-6 * float(x.abs().max()), msg=key)


def test_render_equals_ftgs_render_for_each_kind_of_model(tmp_path: Path) -> None:
    """Q-202: the adapter's render of an FTGS-IF file is `ftgs render`'s, bit for bit; its
    render of the imported native directory is the same images up to the round trip."""
    source = _exchange(tmp_path / "m.safetensors")
    request = _request(tmp_path / "req.json")
    out = tmp_path / "ftgs.safetensors"
    subprocess.run(
        [FTGS, "render", source, "--request", request, "--out", out, "--backend", "reference"],
        check=True,
        capture_output=True,
    )
    assert _adapter("render", source, request, tmp_path / "a.safetensors").returncode == 0
    assert _adapter("import", source, tmp_path / "native").returncode == 0
    assert (
        _adapter("render", tmp_path / "native", request, tmp_path / "n.safetensors").returncode == 0
    )
    want, got = read_renders(out), read_renders(tmp_path / "a.safetensors")
    native = read_renders(tmp_path / "n.safetensors")
    for item in want.rgb:
        assert torch.equal(got.rgb[item], want.rgb[item])
        assert torch.equal(got.alpha[item], want.alpha[item])
        torch.testing.assert_close(native.rgb[item], want.rgb[item], rtol=0, atol=1e-5)


def test_a_bad_input_is_exit_2(tmp_path: Path) -> None:
    (tmp_path / "bad.safetensors").write_bytes(b"not a file")
    assert _adapter("import", tmp_path / "bad.safetensors", tmp_path / "n").returncode == 2


def _train_inputs(root: Path) -> dict[str, Path]:
    time = TimeNormalizer(0.0, 2.0 / 30.0)
    cams = ring(3, radius=4.0, elevation=1.0, width_px=24, height_px=16, focal=22.0)
    gt = random_model(12, seed=0, sh_degree=0, time=time, spread=0.6, log_scale=(-2.5, -1.5))
    scene = write_scene(
        root / "scene", gt, [(c, k / 30.0, "train") for k in range(3) for c in cams]
    )
    g = torch.Generator().manual_seed(1)
    points = InitPoints(
        positions=(torch.rand(20, 3, generator=g) - 0.5),
        colors=torch.full((20, 3), 0.5),
        timestamps=torch.rand(20, generator=g, dtype=torch.float64) * (2.0 / 30.0),
    )
    write_init_points(points, root / "init.safetensors", time_unit="s", producer="test")
    cfg = TrainConfig(sh_degree=0, loss=LossConfig(lambda_perc=0.0), perf=PerfConfig(log_every=2))
    data: dict[str, Any] = config_to_dict(cfg)
    data["optim"]["iterations"] = 4
    del data["seed"]
    data["growth"]["not_a_field"] = 1  # a setting openftgs can't honor
    data["elsewhere"] = {"x": 1}
    (root / "harness_config.json").write_text(json.dumps(data))
    return {
        "scene": scene,
        "init": root / "init.safetensors",
        "config": root / "harness_config.json",
    }


def test_train_delegates_to_ftgs_train_and_reports_settings_it_cannot_honor(
    tmp_path: Path,
) -> None:
    """Q-202, §3.3: train writes the native checkpoint and train_log.jsonl (iteration, loss
    terms, N); config keys TrainConfig doesn't have are listed in not_comparable.json and the
    run goes on without them; the seed is the one given."""
    inputs = _train_inputs(tmp_path)
    out = tmp_path / "run"
    done = _adapter("train", inputs["scene"], inputs["init"], inputs["config"], 7, out)
    assert done.returncode == 0, done.stderr
    assert (out / "checkpoint").is_dir()
    rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert rows[-1]["iteration"] == 4 and "num_gaussians" in rows[-1] and rows[-1]["losses"]
    assert json.loads((out / "not_comparable.json").read_text()) == [
        "growth.not_a_field",
        "elsewhere",
    ]
    assert load_checkpoint(out / "checkpoint").config["seed"] == 7
    exported = _adapter("export", out, tmp_path / "trained.safetensors")
    assert exported.returncode == 0, exported.stderr


def test_the_console_scripts() -> None:
    """Q-202, Q-203: `ftgs-adapter` is openftgs's second console script, beside `ftgs`, and
    `ftgs-harness` the harness's."""
    import importlib.metadata

    scripts = {e.name: e.value for e in importlib.metadata.entry_points(group="console_scripts")}
    assert scripts["ftgs-adapter"] == "openftgs.cli._adapter:main"
    assert scripts["ftgs-harness"] == "ftgs_harness.cli:main"
    assert scripts["ftgs"] == "openftgs.cli:main"

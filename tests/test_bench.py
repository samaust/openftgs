# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench` (§2.1 CLI line and exit codes; §3.4): the subcommands and their B-n, the exit
codes and BENCH.json (Q-166), `--model` on `train-step` and `render` (Q-167), B-2 (`bench
raster`; D-57, U-8, Q-T227.1), B-3 (`bench slice`; D-50, Q-13), and the seeded model
synthesizer of the benchmark setups (Q-168).

The tests call `main(argv)` in process. CPU tests make PyTorch see no CUDA device, as the CLI
tests do (Q-125); the benchmarks themselves run only on a GPU.
"""

import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from builders import look_at, random_model, ring, write_scene

from openftgs.cli import main
from openftgs.io import write_exchange
from openftgs.types import TimeNormalizer

TIME = TimeNormalizer(0.0, 2.0 / 30.0)  # three frames at 30 FPS
SUBCOMMANDS = (
    "train-step",
    "raster",
    "slice",
    "losses",
    "adam",
    "data",
    "render",
    "init",
    "convert",
)


@pytest.fixture(autouse=True)
def _cpu_unless_gpu(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-125: the CLI picks CUDA when PyTorch sees it; CPU tests make it see none."""
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _ftgs(*args: object) -> int:
    return main([str(a) for a in args])


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four 48×32 ring cameras, three frames each, around a small random model."""
    root = tmp_path_factory.mktemp("scene")
    cams = ring(4, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    model = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.6)
    images = [(cam, k / 30.0, "train") for k in range(3) for cam in cams]
    return write_scene(root, model, images)


@pytest.fixture(scope="module")
def forward_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A forward-facing rig shaped like Neural3DV's (and openftgs-synth's `kitchen`, source log
    163): six 135×101 cameras about 1.5 units apart in a plane, with kitchen's field of view,
    their axes converging roughly 2.6 units ahead, so the centroid of the camera centres lies far
    from the content."""
    root = tmp_path_factory.mktemp("forward")
    cams = [
        look_at(
            (x, 0.0, z),
            (0.1 * x, 2.6, -0.1 * z),
            width=135,
            height=101,
            focal=117.0,
            name=f"cam{k:02d}",
        )
        for k, (x, z) in enumerate(
            [(-0.75, -0.2), (-0.25, 0.2), (0.25, -0.2), (0.75, 0.2), (-0.5, 0.0), (0.5, 0.0)]
        )
    ]
    model = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.3)
    images = [(cam, k / 30.0, "train") for k in range(2) for cam in cams]
    return write_scene(root, model, images)


def test_bench_without_cuda_exits_2_except_convert(
    scene: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-166 (§2.1 exit codes): every benchmark but `convert` needs a CUDA device, and without
    one it is exit 2, an invalid environment, with nothing written."""
    out = tmp_path / "bench.json"
    for subcommand in SUBCOMMANDS:
        if subcommand == "convert":
            continue
        assert _ftgs("bench", subcommand, scene, "--out", out) == 2, subcommand
        assert "CUDA" in capsys.readouterr().err, subcommand
        assert not out.exists()


def test_bench_with_an_invalid_scene_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-166 (§2.1 exit codes): a scene manifest that is missing or fails validation is exit 2
    for every subcommand, `convert` included, with nothing written."""
    out = tmp_path / "bench.json"
    broken = tmp_path / "broken.json"
    broken.write_text('{"format": "ftgs-scene", "version": 1}')
    for subcommand in SUBCOMMANDS:
        for manifest in (tmp_path / "absent.json", broken):
            assert _ftgs("bench", subcommand, manifest, "--out", out) == 2, subcommand
            assert f"scene manifest {manifest}" in capsys.readouterr().err
            assert not out.exists()


def test_bench_model_is_only_for_train_step_and_render(
    scene: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-167: `--model MODEL` belongs to `train-step` and `render` only; given to another
    subcommand it is a usage error, exit 2, with nothing written."""
    out = tmp_path / "bench.json"
    model = tmp_path / "model.safetensors"
    for subcommand in SUBCOMMANDS:
        if subcommand in ("train-step", "render"):
            continue
        assert _ftgs("bench", subcommand, scene, "--out", out, "--model", model) == 2
        assert "--model" in capsys.readouterr().err, subcommand
        assert not out.exists()


@pytest.mark.gpu
def test_bench_train_step_and_render_take_a_model(
    scene: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-167: `train-step` and `render` take `--model MODEL`, an FTGS-IF file or a checkpoint
    directory, read as `ftgs render` reads one: an unreadable MODEL is exit 2. Until B-1 (#226)
    and B-7 (#231) land, a valid one reaches Q-130's message: exit 1, nothing written."""
    out = tmp_path / "bench.json"
    model = tmp_path / "model.safetensors"
    write_exchange(random_model(32, seed=1, sh_degree=1, time=TIME).to_exchange(), model)
    for subcommand in ("train-step", "render"):
        assert _ftgs("bench", subcommand, scene, "--out", out, "--model", model) == 1
        assert "comes with #" in capsys.readouterr().err
        absent = tmp_path / "absent.safetensors"
        assert _ftgs("bench", subcommand, scene, "--out", out, "--model", absent) == 2
        assert f"model {absent}" in capsys.readouterr().err
        assert not out.exists()


# The model synthesizer of the benchmark setups (Q-168, Q-181): private tests of
# `openftgs.cli._bench._synthesize`, which B-1 to B-7 share.

PARAM_NAMES = (
    "means",
    "times",
    "log_durations",
    "velocities",
    "log_scales",
    "quats",
    "opacity_logits",
    "sh0",
    "shN",
)


def test__synthesize_gives_n_rows_at_sh_degree_3_seeded(scene: Path) -> None:
    """Q-168: exactly N rows at SH degree 3, the same for the same seed and different for
    another; σ within [0.3, 0.95]."""
    from openftgs.cli._bench import _synthesize, _volume
    from openftgs.data import MultiViewVideo

    volume = _volume(MultiViewVideo.from_manifest(scene))
    a = _synthesize(volume, 5000, active=0.2, seed=3, device="cpu")
    b = _synthesize(volume, 5000, active=0.2, seed=3, device="cpu")
    c = _synthesize(volume, 5000, active=0.2, seed=4, device="cpu")
    assert a.num_gaussians == 5000 and a.sh_degree == 3
    for name in PARAM_NAMES:
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0.0, atol=0.0)
    assert not torch.equal(a.means, c.means)
    sigma = torch.sigmoid(a.opacity_logits.detach())
    assert float(sigma.min()) >= 0.3 - 1e-6 and float(sigma.max()) <= 0.95 + 1e-6


@pytest.mark.parametrize("active", [0.05, 0.2, 1.0])
def test__synthesize_activates_the_fraction_asked_at_every_time(scene: Path, active: float) -> None:
    """Q-168: "durations set for the target active fraction": at any time in [0, 1], the share
    of rows that pass the opacity cull (σ_eff ≥ 1/255, D-32) is the fraction asked, within
    0.01 at 50,000 rows (binomial spread at most 0.0023), and every row at 100 %."""
    from openftgs.cli._bench import _synthesize, _volume
    from openftgs.data import MultiViewVideo

    volume = _volume(MultiViewVideo.from_manifest(scene))
    model = _synthesize(volume, 50_000, active=active, seed=0, device="cpu")
    for t in (0.0, 0.13, 0.5, 0.91, 1.0):
        with torch.no_grad():
            share = float(model.at_time(t).passes_cull.float().mean())
        if active == 1.0:
            assert share == 1.0, t
        else:
            assert abs(share - active) <= 0.01, (t, share)


@pytest.mark.parametrize("rig", ["ring", "forward"])
def test__synthesize_puts_most_rows_in_view_of_every_camera(
    rig: str, scene: Path, forward_scene: Path
) -> None:
    """Q-168, Q-181: the model lies "around the scene's centre", in view: for every training
    camera, at least 70 % of the rows project in front of it and inside its image at t = ½ (the
    32 spheres hold three quarters of the rows), on a ring rig and on a forward-facing one."""
    from openftgs.cli._bench import _synthesize, _volume
    from openftgs.data import MultiViewVideo

    video = MultiViewVideo.from_manifest(scene if rig == "ring" else forward_scene)
    model = _synthesize(_volume(video), 20_000, active=0.2, seed=1, device="cpu")
    with torch.no_grad():
        means = model.at_time(0.5).means_t.double()
    cameras = {r.camera.name: r.camera for r in video.records if r.split == "train"}
    for camera in cameras.values():
        w2c, K = camera.world_to_camera.double(), camera.K.double()
        x = means @ w2c[:3, :3].T + w2c[:3, 3]
        z = x[:, 2]
        u = K[0, 0] * x[:, 0] / z + K[0, 2]
        v = K[1, 1] * x[:, 1] / z + K[1, 2]
        inside = (z > 0) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        assert float(inside.double().mean()) >= 0.7, camera.name


# B-2's camera, times and figures (§3.4; Q-T227.1): private tests of `openftgs.cli._bench`.


@pytest.fixture(scope="module")
def held_out_first(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three 24×16 ring cameras with 25 frames each, written frame by frame, the first camera
    held out for testing."""
    root = tmp_path_factory.mktemp("held-out-first")
    cams = ring(3)
    model = random_model(8, seed=0, sh_degree=0)
    images = [
        (cam, k / 30.0, "test" if c == 0 else "train")
        for k in range(25)
        for c, cam in enumerate(cams)
    ]
    return write_scene(root, model, images)


def test__raster_frames_takes_the_first_training_camera_at_ten_spread_frames(
    held_out_first: Path, scene: Path
) -> None:
    """Q-T227.1: "one training camera at 10 fixed times" is the first training camera in the
    manifest's order, at 10 of its frames evenly spaced over them in time, round(k·(F − 1)/9)
    for k = 0…9: with 25 frames, 0, 3, 5, 8, 11, 13, 16, 19, 21 and 24; with 3, each repeats."""
    from openftgs.cli._bench import _raster_frames
    from openftgs.data import MultiViewVideo

    video = MultiViewVideo.from_manifest(held_out_first)
    records = [video.records[i] for i in _raster_frames(video)]
    assert {r.camera.name for r in records} == {"ring01"}
    assert all(r.split == "train" for r in records)
    frames = [round(r.timestamp * 30.0) for r in records]
    assert frames == [0, 3, 5, 8, 11, 13, 16, 19, 21, 24]

    video = MultiViewVideo.from_manifest(scene)
    records = [video.records[i] for i in _raster_frames(video)]
    assert {r.camera.name for r in records} == {"ring00"}
    assert [round(r.timestamp * 30.0) for r in records] == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test__raster_figures_pairs_each_calls_stages_and_takes_medians() -> None:
    """Q-T227.1: per call, the whole span from the slice's start to the backward's end, the
    rasterization from `raster.forward`'s start to the `backward.raster_done` point, its forward
    (`raster.forward`) and its backward (`backward.losses_done` to `backward.raster_done`), and
    the backward; the figures are the medians over the calls. Three hand-built calls 10 s apart:
    forward 2, 3, 4 s; backward 5, 4, 6 s; rasterization backward 3, 1, 2 s."""
    from openftgs._stages import Timing
    from openftgs.cli._bench import _raster_figures

    timings: list[Timing] = []

    def add(name: str, start: float, end: float, thread: str = "MainThread") -> None:
        timings.append(Timing(name, None, thread, len(timings), start, end))

    for k, (forward, backward, raster_backward) in enumerate([(2, 5, 3), (3, 4, 1), (4, 6, 2)]):
        o = 10.0 * k
        add("slice", o, o + 1.0)
        add("slice.kernel", o, o + 0.8)
        add("raster.forward", o + 1.0, o + 1.0 + forward)
        add("raster.forward.projection", o + 1.0, o + 1.5)
        add("loss.l1", o + 1.0 + forward, o + 1.1 + forward)
        start = o + 1.1 + forward
        add("backward", start, start + backward)
        add("backward.losses_done", start + 0.2, start + 0.2, "autograd")
        done = start + 0.2 + raster_backward
        add("backward.raster_done", done, done, "autograd")
    figures = _raster_figures(timings)
    assert figures == {
        "forward_backward_s": pytest.approx(8.1),  # 1 + forward + 0.1 + backward: 8.1, 8.1, 11.1
        "raster_s": pytest.approx(5.3),  # forward + 0.3 + rasterization backward: 5.3, 4.3, 6.3
        "raster_forward_s": pytest.approx(3.0),
        "raster_backward_s": pytest.approx(2.0),
        "backward_s": pytest.approx(5.0),
        "repetitions": 3,
    }


# BENCH.json (Q-166) and B-3, `ftgs bench slice` (§3.4, D-50; Q-13, Q-182), run once on the
# GPU for the tests below: about half a minute, most of it compiling the slice.


@pytest.fixture(scope="module")
def slice_bench(scene: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("bench") / "results" / "BENCH.json"
    assert _ftgs("bench", "slice", scene, "--out", out) == 0
    assert not list(out.parent.glob(".*")), "the temporary file is left behind (Q-126)"
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return body


@pytest.mark.gpu
def test_bench_json_layout_and_environment(slice_bench: dict[str, Any], scene: Path) -> None:
    """Q-166, §3.4: BENCH.json holds its format and version, the benchmark and the command, the
    environment (§3.4's GPU model, driver, CUDA, PyTorch and gsplat versions, with cuDNN,
    fused-ssim, Python and the commit), the scene, the date, the setup, the cases and the
    budgets."""
    body = slice_bench
    assert set(body) == {
        "format",
        "version",
        "benchmark",
        "command",
        "environment",
        "scene",
        "date",
        "setup",
        "cases",
        "budgets",
    }
    assert (body["format"], body["version"], body["benchmark"]) == ("ftgs-bench", 1, "B-3")
    assert body["command"][:3] == ["ftgs", "bench", "slice"]
    env = body["environment"]
    assert env["gpu"] == {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()}
    assert isinstance(env["driver"], str) and env["driver"]  # nvidia-smi is on this machine
    assert env["cuda"] == torch.version.cuda
    assert env["cudnn"] == torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    assert env["torch"] == torch.__version__
    assert env["gsplat"] == importlib.metadata.version("gsplat")
    assert env["python"] == platform.python_version()
    assert env["fused_ssim"] is None or isinstance(env["fused_ssim"], str)
    assert env["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", env["commit"])
    assert env["dirty"] is None or isinstance(env["dirty"], bool)
    assert body["scene"] == {
        "path": str(scene.resolve()),
        "sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "cameras": 4,
        "frames": 3,
        "width": 48,
        "height": 32,
    }
    assert datetime.fromisoformat(body["date"]).tzinfo is not None
    for case in body["cases"]:
        assert set(case) == {"case", "figures"}
        assert isinstance(case["case"], dict) and isinstance(case["figures"], dict)
    for budget in body["budgets"]:
        assert set(budget) == {"case", "figure", "limit", "value", "pass", "kind"}


@pytest.mark.gpu
def test_b3_times_both_slices_at_every_size_and_counts_recompilations(
    slice_bench: dict[str, Any],
) -> None:
    """B-3 (§3.4; D-50, Q-13, Q-182): the eager and compiled temporal slices, forward and
    backward, at N from 10⁵ to 3·10⁶ over 10 resizes: at each size the first call's time and the
    medians of 50 calls; then the recompilations over the resizes, counted by dynamo's unique
    graphs (Q-13) and set against U-28's limit of 2 as a budget entry, which doesn't change the
    exit code (Q-166)."""
    setup = slice_bench["setup"]
    sizes = setup["sizes"]
    assert sizes[0] == 100_000 and sizes[-1] == 3_000_000 and len(sizes) == 11
    assert sizes == sorted(set(sizes)) and setup["resizes"] == 10
    assert setup["repetitions"] == 50
    cases = {
        (c["case"].get("impl"), c["case"].get("n")): c["figures"] for c in slice_bench["cases"]
    }
    for impl in ("eager", "compiled"):
        for n in sizes:
            figures = cases[(impl, n)]
            assert figures["repetitions"] == 50
            for name in ("forward_backward_s", "forward_s", "backward_s", "first_call_s"):
                assert 0.0 < figures[name] < 60.0, (impl, n, name)
    compiled = cases[("compiled", sizes[0])]
    assert compiled["first_call_s"] > compiled["forward_backward_s"]  # compilation included
    resized = [c for c in slice_bench["cases"] if c["case"] == {"impl": "compiled", "resizes": 10}]
    assert len(resized) == 1
    recompilations = resized[0]["figures"]["recompilations"]
    assert recompilations == resized[0]["figures"]["unique_graphs"] - 1 >= 0
    assert slice_bench["budgets"] == [
        {
            "case": {"impl": "compiled", "resizes": 10},
            "figure": "recompilations",
            "limit": 2,
            "value": recompilations,
            "pass": recompilations <= 2,
            "kind": "limit",
        }
    ]


# B-2, `ftgs bench raster` (§3.4, D-32, D-57; U-8, Q-T227.1), run once on the GPU for the tests
# below.

SIZES = (1_000_000, 3_000_000)  # #227: N = 1M and 3M
ACTIVE = (0.05, 0.2, 1.0)  # #227: 5 %, 20 % and 100 % of rows active
VARIANTS = (  # packed × precull, D-57's default first
    (True, "internal"),
    (True, "compact"),
    (False, "internal"),
    (False, "compact"),
)


@pytest.fixture(scope="module")
def raster_bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """B-2 on four 384×256 ring cameras, three frames each: about half a minute. On `scene`'s
    48×32 images, six tiles hold the millions of rows, so gsplat rasterizes them in six thread
    blocks and the run takes over two minutes."""
    root = tmp_path_factory.mktemp("raster")
    cams = ring(4, radius=4.0, elevation=1.0, width_px=384, height_px=256, focal=352.0)
    model = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.6)
    scene = write_scene(root, model, [(cam, k / 30.0, "train") for k in range(3) for cam in cams])
    out = root / "B-2.json"
    assert _ftgs("bench", "raster", scene, "--out", out) == 0
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return body


@pytest.mark.gpu
def test_b2_times_every_variant_at_each_size_and_active_fraction(
    raster_bench: dict[str, Any],
) -> None:
    """B-2 (§3.4; D-57, Q-T227.1): `render` forward plus an L1 backward for every `packed` ×
    `precull` combination, on synthesized models at N = 1M and 3M with 5 %, 20 % and 100 % of
    rows active, from one training camera at 10 fixed times: per case the medians of 50 calls,
    the first call and the peak memory; the rasterization is part of the whole call."""
    body = raster_bench
    assert body["benchmark"] == "B-2" and body["command"][:3] == ["ftgs", "bench", "raster"]
    setup = body["setup"]
    assert setup["sizes"] == list(SIZES) and setup["active_fractions"] == list(ACTIVE)
    assert setup["variants"] == [{"packed": p, "precull": c} for p, c in VARIANTS]
    assert setup["camera"] == "ring00" and len(setup["timestamps"]) == 10
    assert setup["repetitions"] == 50
    timed = [c for c in body["cases"] if "packed" in c["case"]]
    keys = [
        (c["case"]["n"], c["case"]["active"], c["case"]["packed"], c["case"]["precull"])
        for c in timed
    ]
    assert sorted(keys) == sorted((n, a, p, c) for n in SIZES for a in ACTIVE for p, c in VARIANTS)
    for case in timed:
        figures = case["figures"]
        assert figures["repetitions"] == 50, case["case"]
        for name in (
            "forward_backward_s",
            "raster_s",
            "raster_forward_s",
            "raster_backward_s",
            "backward_s",
            "first_call_s",
        ):
            assert 0.0 < figures[name] < 60.0, (case["case"], name)
        assert figures["raster_s"] < figures["forward_backward_s"], case["case"]
        assert figures["peak_memory_bytes"] > 0, case["case"]


@pytest.mark.gpu
def test_b2_counts_tile_intersections_per_time(raster_bench: dict[str, Any]) -> None:
    """B-2 (§3.4; Q-T227.1): per model and time, the rows active, the rows visible in the camera
    and gsplat's tile intersections, which every variant shares; more rows active at a time give
    more of each, and the active share is near the fraction asked. The memory resident before
    the variants run is the model's and the targets' alone: about the same at every active
    fraction of one N, and more at 3M than at 1M by at least the 2M rows' 64 float32 values."""
    counted = {
        (c["case"]["n"], c["case"]["active"]): c["figures"]
        for c in raster_bench["cases"]
        if "packed" not in c["case"]
    }
    assert sorted(counted) == sorted((n, a) for n in SIZES for a in ACTIVE)
    for (n, active), figures in counted.items():
        for name in ("active_rows", "visible_rows", "tile_intersections"):
            assert len(figures[name]) == 10 and all(isinstance(v, int) for v in figures[name])
        for rows in figures["active_rows"]:
            assert abs(rows / n - active) <= 0.01, (n, active)
        assert all(
            0 < v <= a for v, a in zip(figures["visible_rows"], figures["active_rows"], strict=True)
        )
        assert figures["resident_memory_bytes"] > 0
    for n in SIZES:
        for fewer, more in zip(ACTIVE, ACTIVE[1:], strict=False):
            for name in ("visible_rows", "tile_intersections"):
                assert sum(counted[(n, fewer)][name]) < sum(counted[(n, more)][name]), (n, name)
        # The same up to the caching allocator's rounding of blocks, well under a tenth of the
        # model; a model still held from before would add at least a 1M-row model.
        resident = [counted[(n, active)]["resident_memory_bytes"] for active in ACTIVE]
        assert max(resident) - min(resident) <= n * 64 * 4 // 10, (n, resident)
    small, large = (counted[(n, ACTIVE[0])]["resident_memory_bytes"] for n in SIZES)
    assert large - small >= (SIZES[1] - SIZES[0]) * 64 * 4  # §1.1 at SH degree 3 (Q-168)


@pytest.mark.gpu
def test_b2_spot_checks_each_image_against_packed_internal(raster_bench: dict[str, Any]) -> None:
    """B-2 (§3.4 "images identical (U-8)"; D-57, Q-T227.1): at every time, each variant's image
    against the packed/internal one (D-57's default), its largest difference and the pixels over
    1e-6, held to 1e-6 by a `limit` entry (#227); and each compacted image against the same
    layout's internal one, U-8 itself, within 1e-6 and held to it by a `limit` entry too.
    Neither entry changes the exit code (Q-166)."""
    timed = [c for c in raster_bench["cases"] if "packed" in c["case"]]
    others = [c for c in timed if (c["case"]["packed"], c["case"]["precull"]) != VARIANTS[0]]
    assert len(others) == 18
    expected = []
    for case in others:
        figures = case["figures"]
        names = ["image_max_abs_diff"]
        if case["case"]["precull"] == "compact":
            names.append("precull_max_abs_diff")
            assert figures["precull_max_abs_diff"] <= 1e-6, case["case"]  # U-8
        for name in names:
            value = figures[name]
            expected.append(
                {
                    "case": case["case"],
                    "figure": name,
                    "limit": 1e-6,
                    "value": value,
                    "pass": value <= 1e-6,
                    "kind": "limit",
                }
            )
        over = figures["image_pixels_over_limit"]
        assert isinstance(over, int) and (over == 0) == (figures["image_max_abs_diff"] <= 1e-6)
    assert raster_bench["budgets"] == expected


@pytest.mark.gpu
def test_bench_slice_refuses_a_single_timestamp_scene_and_an_unwritable_out(
    tmp_path: Path, scene: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-166, Q-181: a scene with a single timestamp leaves no row of a synthesized model
    inactive (D-71), so it is exit 2; so is a BENCH.json whose folder can't be made, found
    before the benchmark runs (Q-142)."""
    cams = ring(4, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    model = random_model(8, seed=0, sh_degree=0)
    single = write_scene(tmp_path / "single", model, [(cam, 0.0, "train") for cam in cams])
    out = tmp_path / "bench.json"
    assert _ftgs("bench", "slice", single, "--out", out) == 2
    assert "more than one timestamp" in capsys.readouterr().err
    assert not out.exists()
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    assert _ftgs("bench", "slice", scene, "--out", blocker / "bench.json") == 2
    assert "cannot write BENCH.json" in capsys.readouterr().err

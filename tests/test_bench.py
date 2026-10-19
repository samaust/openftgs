# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench` (§2.1 CLI line and exit codes; §3.4): the subcommands and their B-n, the exit
codes and BENCH.json (Q-166), `--model` on `train-step` and `render` (Q-167), B-1 (`bench
train-step`; D-49, D-50, Q-185), B-2 (`bench raster`; D-57, U-8, Q-187), B-3 (`bench slice`;
D-50, Q-13), B-6 (`bench data`; D-51, Q-99, Q-178, Q-T232.1), and the seeded model synthesizer
of the benchmark setups (Q-168).

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
from typing import Any, Literal

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
    directory, read as `ftgs render` reads one: an unreadable MODEL is exit 2. Until B-7 (#231)
    lands, a valid one reaches Q-130's message there: exit 1, nothing written. B-1 with a model
    is `test_b1_takes_a_model`; one of another scene's time normalizer can't be trained on this
    scene, exit 2 (#226)."""
    out = tmp_path / "bench.json"
    model = tmp_path / "model.safetensors"
    write_exchange(random_model(32, seed=1, sh_degree=1, time=TIME).to_exchange(), model)
    assert _ftgs("bench", "render", scene, "--out", out, "--model", model) == 1
    assert "comes with #" in capsys.readouterr().err
    other = tmp_path / "other.safetensors"
    later = TimeNormalizer(1.0, 2.0)
    write_exchange(random_model(32, seed=1, sh_degree=3, time=later).to_exchange(), other)
    assert _ftgs("bench", "train-step", scene, "--out", out, "--model", other) == 2
    assert f"model {other}" in capsys.readouterr().err
    assert not out.exists()
    for subcommand in ("train-step", "render"):
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


# B-2's camera, times and figures (§3.4; Q-187): private tests of `openftgs.cli._bench`.


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
    """Q-187: "one training camera at 10 fixed times" is the first training camera in the
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
    """Q-187: per call, the whole span from the slice's start to the backward's end, the
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


# B-2, `ftgs bench raster` (§3.4, D-32, D-57; U-8, Q-187), run once on the GPU for the tests
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
    """B-2 (§3.4; D-57, Q-187): `render` forward plus an L1 backward for every `packed` ×
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
    """B-2 (§3.4; Q-187): per model and time, the rows active, the rows visible in the camera
    and gsplat's tile intersections, which every variant shares; more rows active at a time give
    more of each, and the active share is near the fraction asked. The memory resident before
    the variants run is the model's and the targets' alone: about the same at every active
    fraction of one N, and more at 3M than at 1M by about the 2M rows' 64 float32 values."""
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
    added = (SIZES[1] - SIZES[0]) * 64 * 4  # §1.1 at SH degree 3 (Q-168)
    assert abs((large - small) - added) <= added // 10, (small, large)


@pytest.mark.gpu
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
def test__raster_call_synchronizes_only_inside_gsplat(scene: Path, packed: bool) -> None:
    """Q-187, §2.4 rule 3: a B-2 call given the scene's camera, held on the host, times what a
    training iteration runs, so with the opacity cull inside gsplat's projection it synchronizes
    only inside gsplat's own calls, as `Trainer.step` does with its cameras on the device: every
    synchronization torch's debug mode reports comes from gsplat's files."""
    import warnings

    from openftgs.cli._bench import _raster_call, _raster_frames, _synthesize, _volume
    from openftgs.data import MultiViewVideo

    video = MultiViewVideo.from_manifest(scene)
    record = video.records[_raster_frames(video)[0]]
    assert record.camera.K.device.type == "cpu"
    model = _synthesize(_volume(video), 2000, active=0.2, seed=0, device="cuda")
    model.slice_impl = "eager"
    target = torch.zeros(1, record.camera.height, record.camera.width, 3, device="cuda")
    call = _raster_call(model, record.camera, [(record.timestamp, target)], (packed, "internal"))
    call()
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            call()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    sites = {
        f"{r.filename}:{r.lineno}"
        for r in records
        if "synchronizing CUDA operation" in str(r.message)
    }
    assert all("/gsplat/" in site for site in sites), sites


@pytest.mark.gpu
def test_b2_spot_checks_each_image_against_packed_internal(raster_bench: dict[str, Any]) -> None:
    """B-2 (§3.4 "images identical (U-8)"; D-57, Q-187): at every time, each variant's image
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


# B-1, `ftgs bench train-step` (§3.4, D-49, D-50; Q-168, Q-185): private tests of its
# configuration, of the summary of the stage clock's records, and of the training bound.


def test__train_step_config_starts_events_at_100_with_2_percent_headroom() -> None:
    """#226: the default configuration, LPIPS-VGG included, except that growth and relocation
    start at iteration 100, n_max = ⌈1.02·N⌉ and sh_interval = 1; so the timed iterations 121 to
    420 hold exactly the events at 200, 300 and 400, all at SH degree 3."""
    from openftgs.cli._bench import _train_step_config
    from openftgs.config import TrainConfig
    from openftgs.train import calendar_entry

    cfg = _train_step_config(500_000)
    default = TrainConfig()
    assert (cfg.growth.start, cfg.reloc.start, cfg.optim.sh_interval) == (100, 100, 1)
    assert cfg.growth.n_max == 510_000
    assert _train_step_config(3_000_000).growth.n_max == 3_060_000
    assert _train_step_config(101).growth.n_max == 104  # ⌈103.02⌉
    assert _train_step_config(100).growth.n_max == 102  # exactly 102: no rounding up
    assert (cfg.loss.lambda_perc, cfg.loss.perc_net) == (0.01, "vgg")
    assert cfg.sh_degree == 3 and _train_step_config(10, sh_degree=1).sh_degree == 1
    # Everything else is the default.
    assert cfg.loss == default.loss and cfg.perf == default.perf and cfg.init == default.init
    assert cfg.growth.stop == default.growth.stop and cfg.reloc.stop == default.reloc.stop
    assert cfg.optim.iterations == default.optim.iterations
    entries = {it: calendar_entry(cfg, it) for it in range(121, 421)}
    assert [it for it, e in entries.items() if e.event] == [200, 300, 400]
    assert all(e.grow and e.relocate for it, e in entries.items() if e.event)
    assert {e.sh_degree for e in entries.values()} == {3}
    assert calendar_entry(cfg, 100).event and not calendar_entry(cfg, 99).event


def test__stage_figures_split_iterations_and_backward_and_skip_events() -> None:
    """§3.4 B-1, #226: per stage, the mean and median over the iterations without an event,
    and each one's share of an iteration (the ratio of the means); the backward stage split
    into three parts by its two points, the last being the temporal slice's backward. Records
    come in the order they began, the points from the autograd thread among them."""
    from openftgs._stages import Timing
    from openftgs.cli._bench import _iteration_parts, _stage_figures

    def rec(index: int, name: str, start: float, end: float, thread: str = "main") -> Timing:
        return Timing(name, None, thread, index, start, end)

    timings = [
        # Iteration 1, no event: 10 s, data 1, slice kernel 1, backward 4 split 1 / 2 / 1.
        rec(0, "iteration", 0.0, 10.0),
        rec(1, "data", 0.0, 1.0),
        rec(2, "slice.kernel", 1.0, 2.0),
        rec(3, "backward", 5.0, 9.0),
        rec(4, "backward.losses_done", 6.0, 6.0, "autograd"),
        rec(5, "backward.raster_done", 8.0, 8.0, "autograd"),
        # Iteration 2, an event: 20 s, of which the event 8.
        rec(6, "iteration", 10.0, 30.0),
        rec(7, "data", 10.0, 11.0),
        rec(8, "slice.kernel", 11.0, 12.0),
        rec(9, "backward", 13.0, 17.0),
        rec(10, "backward.losses_done", 14.0, 14.0, "autograd"),
        rec(11, "backward.raster_done", 16.0, 16.0, "autograd"),
        rec(12, "events", 20.0, 28.0),
        # Iteration 3, no event: 14 s, data 3, slice kernel 1, backward 6 split 1 / 4 / 1.
        rec(13, "iteration", 30.0, 44.0),
        rec(14, "data", 30.0, 33.0),
        rec(15, "slice.kernel", 33.0, 34.0),
        rec(16, "backward", 36.0, 42.0),
        rec(17, "backward.losses_done", 37.0, 37.0, "autograd"),
        rec(18, "backward.raster_done", 41.0, 41.0, "autograd"),
    ]
    parts = _iteration_parts(list(reversed(timings)))  # sorted by index, whatever the order
    assert len(parts) == 3
    assert parts[1]["events"] == 8.0 and parts[1]["iteration"] == 20.0
    assert parts[2]["backward.raster"] == 4.0 and parts[2]["backward.slice"] == 1.0
    figures = _stage_figures(parts, events=[False, True, False])
    assert set(figures) == {
        "iteration",
        "data",
        "slice.kernel",
        "backward",
        "backward.losses",
        "backward.raster",
        "backward.slice",
    }
    assert figures["iteration"] == {"mean_s": 12.0, "median_s": 12.0, "share": 1.0, "count": 2}
    assert figures["data"] == {"mean_s": 2.0, "median_s": 2.0, "share": 2.0 / 12.0, "count": 2}
    assert figures["backward.losses"]["mean_s"] == 1.0
    assert figures["backward.raster"] == {
        "mean_s": 3.0,
        "median_s": 3.0,
        "share": 3.0 / 12.0,
        "count": 2,
    }
    assert figures["backward.slice"]["share"] == 1.0 / 12.0


def test__training_bound_counts_the_default_calendars_iterations_and_events() -> None:
    """Q-168's bound for B-1: T_it × the iteration time + the default calendar's number of
    events × the event time. By default T_it = 30,000, and the events are the relocation events
    every 100 iterations from 500 to 25,000, 246 of them, growth's (500 to 15,000) falling on
    the same iterations (§1.3 Iteration calendar)."""
    from openftgs.cli._bench import _training_bound

    assert _training_bound(0.1, 0.5) == (30_000 * 0.1 + 246 * 0.5, 30_000, 246)


# B-1 run on the small scene with reduced sizes (#226: "a test on a small scene, with reduced
# sizes, checks the layout"): two synthesized models, and once with `--model`. The default
# configuration holds LPIPS-VGG, so these need its weights.

TRAIN_STEP_CASES = ((300, 0.2), (400, 0.05))
TRAINING_STAGES = (
    "iteration",
    "data",
    "slice",
    "slice.kernel",
    "slice.sh_concat",
    "raster.forward",
    "raster.forward.projection",
    "raster.forward.sh",
    "raster.forward.isect",
    "raster.forward.rasterize",
    "loss.l1",
    "loss.ssim",
    "loss.perceptual",
    "loss.reg",
    "backward",
    "statistics",
    "adam",
    "events",
    "log",
)
BACKWARD_PARTS = ("backward.losses", "backward.raster", "backward.slice")


@pytest.fixture(scope="module")
def train_step_bench(scene: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("bench") / "B-1.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("openftgs.cli._bench._TRAIN_STEP_CASES", TRAIN_STEP_CASES)
        assert _ftgs("bench", "train-step", scene, "--out", out) == 0
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return body


@pytest.fixture(scope="module")
def train_step_model_bench(scene: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("bench-model")
    model = root / "model.safetensors"
    write_exchange(random_model(64, seed=2, sh_degree=3, time=TIME).to_exchange(), model)
    out = root / "B-1.json"
    assert _ftgs("bench", "train-step", scene, "--out", out, "--model", model) == 0
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return body


def _cases_by(body: dict[str, Any]) -> dict[tuple[tuple[str, Any], ...], dict[str, Any]]:
    return {tuple(sorted(c["case"].items())): c["figures"] for c in body["cases"]}


@pytest.mark.gpu
@pytest.mark.weights
def test_b1_times_every_stage_and_bounds_a_training_run(train_step_bench: dict[str, Any]) -> None:
    """§3.4 B-1 (#226, Q-185): per case, every stage of a training iteration and the three
    parts of its backward, timed over iterations 121 to 420 by the stage clock, with the mean
    and median over the 297 iterations without an event and its share of an iteration; the
    events at 200, 300 and 400; whole iterations 421 to 720 without the clock; peak memory, the
    active and visible fractions, N at the start and at the end; the cost of copying the
    relocation scores to the CPU (Q-72 note 5); and the bound on a training run (Q-168) and
    D-50's 10 % as budget entries, which don't change the exit code (Q-166)."""
    body = train_step_bench
    assert body["benchmark"] == "B-1"
    assert body["command"][:3] == ["ftgs", "bench", "train-step"]
    setup = body["setup"]
    assert setup["timed"] == [121, 420] and setup["unclocked"] == [421, 720]
    assert setup["events"] == [200, 300, 400] and setup["warmup"] == 120
    assert setup["cases"] == [list(c) for c in TRAIN_STEP_CASES]
    assert (setup["training_iterations"], setup["default_events"]) == (30_000, 246)
    cases = _cases_by(body)
    for n, active in TRAIN_STEP_CASES:
        for name in TRAINING_STAGES + BACKWARD_PARTS:
            figures = cases[(("active", active), ("n", n), ("stage", name))]
            assert set(figures) == {"mean_s", "median_s", "share", "count"}, name
            assert figures["count"] == 297, name
            assert figures["mean_s"] > 0.0 and figures["median_s"] > 0.0, name
            assert 0.0 < figures["share"] <= 1.0 + 1e-9, name
        for it in (200, 300, 400):
            event = cases[(("active", active), ("iteration", it), ("n", n))]
            assert event["events_s"] > 0.0 and event["iteration_s"] > event["events_s"]
        summary = cases[(("active", active), ("n", n))]
        assert summary["n_start"] == n
        assert n <= summary["n_timed_start"] <= summary["n_end"] <= -(-102 * n // 100)
        for name in ("active_fraction", "visible_fraction", "slice_share"):
            assert 0.0 <= summary[name] <= 1.0, name
        assert summary["visible_fraction"] <= summary["active_fraction"] + 1e-9
        assert summary["peak_reserved_bytes"] >= summary["peak_allocated_bytes"] > 0
        assert summary["iterations"] == summary["unclocked_iterations"] == 297
        for name in ("iteration_unclocked_mean_s", "iteration_unclocked_median_s"):
            assert summary[name] > 0.0, name
        assert summary["clock_overhead_s"] == pytest.approx(
            cases[(("active", active), ("n", n), ("stage", "iteration"))]["mean_s"]
            - summary["iteration_unclocked_mean_s"]
        )
        slice_s = (
            cases[(("active", active), ("n", n), ("stage", "slice.kernel"))]["mean_s"]
            + cases[(("active", active), ("n", n), ("stage", "backward.slice"))]["mean_s"]
        )
        iteration_s = cases[(("active", active), ("n", n), ("stage", "iteration"))]["mean_s"]
        assert summary["slice_share"] == pytest.approx(slice_s / iteration_s)
        assert summary["score_copy_s"] > 0.0 and summary["events_s"] > 0.0
        assert summary["training_bound_s"] == pytest.approx(
            30_000 * summary["iteration_unclocked_mean_s"] + 246 * summary["event_extra_s"]
        )
    largest = max(TRAIN_STEP_CASES)  # the largest N
    kinds = {b["kind"]: b for b in body["budgets"]}
    assert set(kinds) == {"bound", "limit"} and len(body["budgets"]) == 2
    bound = kinds["bound"]
    assert bound["case"] == {"n": largest[0], "active": largest[1]}
    assert bound["figure"] == "training_bound_s" and bound["limit"] == 3600
    assert bound["value"] == cases[(("active", largest[1]), ("n", largest[0]))]["training_bound_s"]
    assert bound["pass"] == (bound["value"] <= 3600)
    share = kinds["limit"]
    assert share["figure"] == "slice_share" and share["limit"] == 0.1
    shares = {(n, a): cases[(("active", a), ("n", n))]["slice_share"] for n, a in TRAIN_STEP_CASES}
    assert share["value"] == max(shares.values())
    assert shares[(share["case"]["n"], share["case"]["active"])] == share["value"]
    assert share["pass"] == (share["value"] <= 0.1)


@pytest.mark.gpu
@pytest.mark.weights
def test_b1_takes_a_model(train_step_model_bench: dict[str, Any]) -> None:
    """Q-167, #226: with `--model`, B-1 times that model, at its SH degree, as its one case,
    which carries the bound."""
    body = train_step_model_bench
    assert body["command"][-2] == "--model"
    assert body["setup"]["cases"] == [[64, None]]
    cases = _cases_by(body)
    summary = cases[(("model", "given"), ("n", 64))]
    assert summary["n_start"] == 64
    assert cases[(("model", "given"), ("n", 64), ("stage", "loss.perceptual"))]["count"] == 297
    bound = next(b for b in body["budgets"] if b["kind"] == "bound")
    assert bound["case"] == {"model": "given", "n": 64}


# B-6, `ftgs bench data` (§3.4, D-51; Q-99, Q-178, Q-T232.1): private tests of its
# configuration, of the figures of its two windows, of Q-99's rule and of its budgets.


def test__data_config_is_b1s_with_the_case_placement() -> None:
    """#232: B-6 runs B-1's configuration (#226) with each case's frame-cache placement and
    prefetch depth, and nothing else changed; no depth keeps the default."""
    import dataclasses

    from openftgs.cli._bench import _data_config, _train_step_config
    from openftgs.config import PerfConfig

    b1 = _train_step_config(1_000_000)
    cases: tuple[tuple[Literal["gpu", "mmap", "off"], int | None], ...] = (
        ("gpu", None),
        ("mmap", 8),
        ("mmap", 0),
        ("off", None),
    )
    for placement, prefetch in cases:
        cfg = _data_config(1_000_000, placement, prefetch)
        want = PerfConfig().prefetch if prefetch is None else prefetch
        assert (cfg.perf.frame_cache, cfg.perf.prefetch) == (placement, want)
        assert dataclasses.replace(cfg, perf=b1.perf) == b1


def test__auto_estimate_applies_q99s_rule_and_measures_its_margin() -> None:
    """Q-99 (D-51): "auto" puts the cache on the GPU when it fits in half the GPU memory free
    after a warm-up iteration, minus what growth to N_max adds, the boundary included. Q-99's
    note (Q-T232.1): what a run with the cache on the GPU then took beyond the cache and the
    growth, from its peak reserved memory and the memory allocated after that iteration,
    against the half of the free memory the rule leaves for it."""
    import dataclasses

    from openftgs.cli._bench import _AutoEstimate

    gib = 2**30
    rule = _AutoEstimate(
        free=10 * gib, growth=gib, cache=4 * gib, allocated=3 * gib, reserved=5 * gib
    )
    assert rule.limit == 4 * gib and rule.fits
    assert not dataclasses.replace(rule, cache=4 * gib + 1).fits
    assert rule.figures() == {
        "auto_free_bytes": 10 * gib,
        "auto_growth_bytes": gib,
        "auto_allocated_bytes": 3 * gib,
        "auto_reserved_bytes": 5 * gib,
        "cache_bytes": 4 * gib,
        "auto_limit_bytes": 4 * gib,
    }
    # 11 GiB reserved at the peak less the 3 allocated after the warm-up iteration is 8 GiB, of
    # which the cache takes 4 and growth 1: 3 GiB beyond the estimate, of the 5 the rule leaves.
    assert rule.margin(peak_reserved=11 * gib) == {
        "beyond_estimate_bytes": 3 * gib,
        "margin_used": 0.6,
    }


def test__data_figures_share_the_data_wait_over_iterations_without_an_event() -> None:
    """§3.4 B-6 (Q-T232.1): the figure is the data stage's CUDA-event time over the iteration's,
    a ratio of means over the timed iterations without an event, as B-1's shares (Q-185); the
    wall clock's window gives the host time of both, a diagnostic."""
    from openftgs.cli._bench import _data_figures

    timed = [
        {"iteration": 10.0, "data": 1.0},
        {"iteration": 20.0, "data": 5.0, "events": 8.0},  # an event: left out
        {"iteration": 14.0, "data": 3.0},
    ]
    host = [{"iteration": 0.2, "data": 0.01}, {"iteration": 0.3, "data": 0.03}]
    figures = _data_figures(timed, [False, True, False], host, [False, False])
    assert figures == {
        "data_share": 2.0 / 12.0,
        "data_mean_s": 2.0,
        "data_median_s": 2.0,
        "iteration_mean_s": 12.0,
        "iterations": 2,
        "data_host_mean_s": pytest.approx(0.02),
        "iteration_host_mean_s": pytest.approx(0.25),
        "host_iterations": 2,
    }


def test__data_budgets_hold_the_cache_placements_under_1_percent() -> None:
    """§3.4 B-6, D-51 (#232): the data wait's share is under 1 % for the cache's placements,
    "gpu" and "mmap" at either prefetch depth, at B-6's N; decoding PNGs ("off") and Q-99's
    case at 3M rows have no budget. A failed budget is recorded, not an error (Q-166)."""
    from openftgs.cli._bench import _data_budgets

    def case(placement: str, n: int, share: float, **extra: int) -> dict[str, Any]:
        return {
            "case": {"n": n, "placement": placement, **extra},
            "figures": {"data_share": share, "data_mean_s": 0.001},
        }

    cases = [
        case("gpu", 1_000, 0.001),
        case("mmap", 1_000, 0.002, prefetch=8),
        case("mmap", 1_000, 0.02, prefetch=0),
        case("off", 1_000, 0.3),
        case("gpu", 3_000, 0.001),
    ]
    assert _data_budgets(cases, n=1_000) == [
        {
            "case": c["case"],
            "figure": "data_share",
            "limit": 0.01,
            "value": c["figures"]["data_share"],
            "pass": passed,
            "kind": "budget",
        }
        for c, passed in zip(cases[:3], (True, True, False), strict=True)
    ]


# B-6 run on the small scene with reduced sizes, as B-1's tests (#226): every placement at 300
# rows and the GPU cache at 400 for Q-99's note; then with a cache too large for "auto" to place
# on the GPU. The default configuration holds LPIPS-VGG, so these need its weights.

DATA_N, MEMORY_N = 300, 400


@pytest.fixture(scope="module")
def data_bench(scene: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("bench") / "B-6.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("openftgs.cli._bench._DATA_N", DATA_N)
        patch.setattr("openftgs.cli._bench._MEMORY_N", MEMORY_N)
        assert _ftgs("bench", "data", scene, "--out", out) == 0
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return body


@pytest.mark.gpu
@pytest.mark.weights
def test_b6_shares_the_data_wait_per_placement(data_bench: dict[str, Any]) -> None:
    """§3.4 B-6, D-51 (#232, Q-T232.1): with B-1's configuration, the data stage's share of an
    iteration, from the stage clock's CUDA events over iterations 121 to 320 (the 198 without an
    event), and its host time over 321 to 420 under the wall clock (the 99 without an event),
    for the GPU cache, the memory-mapped cache at prefetch depths 8 and 0 and PNG decoding; then
    the GPU cache at a larger N with peak memory against what "auto" estimated (Q-99's note).
    The cache's placements are held to 1 % as budget entries, which don't change the exit
    code (Q-166)."""
    body = data_bench
    assert body["benchmark"] == "B-6"
    assert body["command"][:3] == ["ftgs", "bench", "data"]
    setup = body["setup"]
    assert setup["warmup"] == 120 and setup["timed"] == [121, 320] and setup["host"] == [321, 420]
    assert setup["events"] == [200, 300, 400] and setup["share_limit"] == 0.01
    assert setup["skipped"] == []
    want = [
        {"n": DATA_N, "placement": "gpu"},
        {"n": DATA_N, "placement": "mmap", "prefetch": 8},
        {"n": DATA_N, "placement": "mmap", "prefetch": 0},
        {"n": DATA_N, "placement": "off"},
        {"n": MEMORY_N, "placement": "gpu"},
    ]
    assert [c["case"] for c in body["cases"]] == want
    for entry in body["cases"]:
        case, figures = entry["case"], entry["figures"]
        assert figures["iterations"] == 198 and figures["host_iterations"] == 99, case
        assert 0.0 < figures["data_share"] < 1.0, case
        assert figures["data_share"] == pytest.approx(
            figures["data_mean_s"] / figures["iteration_mean_s"]
        )
        for name in ("data_median_s", "data_host_mean_s", "iteration_host_mean_s"):
            assert figures[name] > 0.0, (case, name)
        assert figures["peak_reserved_bytes"] >= figures["peak_allocated_bytes"] > 0, case
        if case["placement"] != "gpu":
            assert "auto_free_bytes" not in figures, case
            continue
        # Q-99: "auto" would have placed the cache on the GPU, which then holds it.
        assert figures["cache_bytes"] <= figures["auto_limit_bytes"], case
        assert figures["auto_limit_bytes"] == pytest.approx(
            0.5 * figures["auto_free_bytes"] - figures["auto_growth_bytes"]
        )
        assert figures["auto_growth_bytes"] > 0  # n_max = ⌈1.02·N⌉ > N
        assert figures["peak_allocated_bytes"] >= figures["cache_bytes"], case
        assert figures["beyond_estimate_bytes"] == (
            figures["peak_reserved_bytes"]
            - figures["auto_allocated_bytes"]
            - figures["cache_bytes"]
            - figures["auto_growth_bytes"]
        )
        assert figures["margin_used"] == pytest.approx(
            figures["beyond_estimate_bytes"] / (0.5 * figures["auto_free_bytes"])
        )
    budgets = body["budgets"]
    assert [b["case"] for b in budgets] == want[:3]
    for budget, entry in zip(budgets, body["cases"], strict=False):
        assert (budget["figure"], budget["limit"], budget["kind"]) == ("data_share", 0.01, "budget")
        assert budget["value"] == entry["figures"]["data_share"]
        assert budget["pass"] == (budget["value"] <= 0.01)


@pytest.mark.gpu
@pytest.mark.weights
def test_b6_skips_the_gpu_placement_where_auto_would_not_choose_it(
    scene: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-178, Q-99 (Q-T232.1): the GPU placement is measured only where "auto" would place the
    cache there after a warm-up iteration; a cache too large for that, as kitchen's on a 24 GiB
    GPU, is skipped at either N, with what "auto" estimated, and gets no case or budget."""
    from openftgs.data import FrameCache

    monkeypatch.setattr("openftgs.cli._bench._DATA_N", DATA_N)
    monkeypatch.setattr("openftgs.cli._bench._MEMORY_N", MEMORY_N)
    monkeypatch.setattr("openftgs.cli._bench._DATA_PLACEMENTS", (("gpu", None), ("mmap", 8)))
    monkeypatch.setattr(FrameCache, "size_of", staticmethod(lambda dataset: 2**50))
    out = tmp_path / "B-6.json"
    assert _ftgs("bench", "data", scene, "--out", out) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    skipped = body["setup"]["skipped"]
    assert [s["case"] for s in skipped] == [
        {"n": DATA_N, "placement": "gpu"},
        {"n": MEMORY_N, "placement": "gpu"},
    ]
    for entry in skipped:
        assert "auto" in entry["reason"]
        estimate = entry["estimate"]
        assert estimate["cache_bytes"] == 2**50 > estimate["auto_limit_bytes"]
    want = {"n": DATA_N, "placement": "mmap", "prefetch": 8}
    assert [c["case"] for c in body["cases"]] == [want]
    assert [b["case"] for b in body["budgets"]] == [want]

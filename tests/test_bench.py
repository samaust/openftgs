# SPDX-License-Identifier: Apache-2.0
"""`ftgs bench` (§2.1 CLI line and exit codes; §3.4): the subcommands and their B-n, the exit
codes and BENCH.json (Q-166), `--model` on `train-step` and `render` (Q-167), B-3 (`bench
slice`; D-50, Q-13), B-7 (`bench render`; D-55, Q-174, Q-T231.1, Q-T231.2), and the seeded
model synthesizer of the benchmark setups (Q-168).

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
    lands, a valid one reaches Q-130's message for `train-step`: exit 1, nothing written (B-7
    with a model: `test_b7_times_the_model_given`)."""
    out = tmp_path / "bench.json"
    model = tmp_path / "model.safetensors"
    write_exchange(random_model(32, seed=1, sh_degree=1, time=TIME).to_exchange(), model)
    assert _ftgs("bench", "train-step", scene, "--out", out, "--model", model) == 1
    assert "comes with #" in capsys.readouterr().err
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


# B-7's setup (Q-168, Q-T231.1) and its rule for T (Q-T231.2): private tests of the helpers of
# `openftgs.cli._bench._render`.


@pytest.fixture(scope="module")
def tested_scene(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The four ring cameras of `scene`, three frames each, the third camera's images in the
    test split."""
    root = tmp_path_factory.mktemp("tested")
    cams = ring(4, radius=4.0, elevation=1.0, width_px=48, height_px=32, focal=44.0)
    model = random_model(16, seed=0, sh_degree=1, time=TIME, spread=0.6)
    images = [
        (cam, k / 30.0, "test" if c == 2 else "train")
        for k in range(3)
        for c, cam in enumerate(cams)
    ]
    return write_scene(root, model, images)


def test__b7_camera_is_the_test_camera_at_1920_by_1080(scene: Path, tested_scene: Path) -> None:
    """Q-168, Q-T231.1: B-7's camera has the pose of the first camera in the manifest's order
    that has test images (the first camera when none has), fx and fy scaled by 1080/H (here
    44 · 1080/32 = 1485), and the principal point at the centre of 1920×1080."""
    from openftgs.cli._bench import _b7_camera
    from openftgs.data import MultiViewVideo

    expected_k = torch.tensor([[1485.0, 0.0, 960.0], [0.0, 1485.0, 540.0], [0.0, 0.0, 1.0]])
    for path, pose in ((tested_scene, "ring02"), (scene, "ring00")):
        video = MultiViewVideo.from_manifest(path)
        source = next(r.camera for r in video.records if r.camera.name == pose)
        camera = _b7_camera(video)
        assert (camera.width, camera.height, camera.name) == (1920, 1080, pose)
        torch.testing.assert_close(camera.K, expected_k, rtol=1e-6, atol=0.0)
        torch.testing.assert_close(
            camera.world_to_camera, source.world_to_camera, rtol=0.0, atol=0.0
        )


def test__b7_times_spread_a_request_of_k_times_over_the_scene() -> None:
    """Q-T231.1: a request of k times takes the timestamps round(j(F − 1)/(k − 1)), rounded half
    up, j = 0 … k − 1, and the middle one for k = 1; the swept sizes are 1, 2, 3, 4, 6, 8, 12,
    16, 24, 32, 48, 64 and 128 below F, then F."""
    from openftgs.cli._bench import _request_sizes, _spread

    assert _spread(300, 1) == [149]
    assert _spread(300, 2) == [0, 299]
    assert _spread(300, 3) == [0, 150, 299]  # 149.5 rounds up
    assert _spread(300, 4) == [0, 100, 199, 299]
    assert _spread(5, 5) == [0, 1, 2, 3, 4]
    assert _request_sizes(300) == [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 300]
    assert _request_sizes(3) == [1, 2, 3]
    assert _request_sizes(1) == [1]


def _rows_model(mu: list[float], reach: list[float], sigma: list[float]) -> Any:
    """A model of one row per entry at the origin, at rest, with time centre μt, opacity σ and
    the duration s for which E3's σ·σt(t) reaches 1/255 exactly at |t − μt| = reach (E4), over
    the unit time span."""
    import math

    from openftgs.model import FreeTimeGaussians

    n = len(mu)
    spread = [math.sqrt(2.0 * math.log(255.0 * s)) if 255.0 * s > 1.0 else 1.0 for s in sigma]
    duration = [r / f for r, f in zip(reach, spread, strict=True)]
    params = {
        "means": torch.zeros(n, 3),
        "times": torch.tensor(mu),
        "log_durations": torch.log(torch.tensor(duration)),
        "velocities": torch.zeros(n, 3),
        "log_scales": torch.full((n, 3), -3.0),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        "opacity_logits": torch.logit(torch.tensor(sigma, dtype=torch.float64)).float(),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 0, 3),
    }
    return FreeTimeGaussians(params, sh_degree=0, time=TimeNormalizer(0.0, 1.0), time_unit="s")


def test__index_figures_reports_the_buckets_the_long_list_and_the_active_rows() -> None:
    """Q-T231.1, D-55: with buckets 0.1 wide, a row of reach 0.02 at μt = 0.25 lies in bucket 2
    alone, one of reach 0.06 at 0.75 in buckets 6 to 8, one of reach 0.5 at 0.5 covers all 11
    buckets of [0, 1] and goes to the long list, and one with 255σ < 1 is never listed. At the
    times 0.25 and 0.75, each render is passed its bucket's row and the long row: 2 of 4 rows,
    1 of them the bucket's own; the long list holds 1 of 4; and 2 of 4 rows pass the opacity
    cull at each time."""
    from openftgs.cli._bench import _index_figures

    model = _rows_model(
        mu=[0.25, 0.5, 0.5, 0.75], reach=[0.02, 0.5, 0.3, 0.06], sigma=[0.5, 0.5, 0.003, 0.5]
    )
    figures = _index_figures(model, 0.1, [0.25, 0.75])
    assert figures == {
        "bucket_width": 0.1,
        "passed_share_mean": 0.5,
        "bucket_share_mean": 0.25,
        "long_share": 0.25,
        "active_share_mean": 0.5,
    }


def test__break_even_is_the_smallest_size_from_which_the_index_pays() -> None:
    """Q-T231.2: the break-even is the smallest request size at which the request with the
    index takes no longer than without it, there and at every larger size; none when the index
    is slower at the largest size."""
    from openftgs.cli._bench import _break_even

    # (with the index, without it), in seconds
    slower_at_8 = {1: (5.0, 3.0), 2: (6.0, 6.0), 4: (7.0, 9.0), 8: (9.0, 8.0), 16: (1.0, 2.0)}
    assert _break_even(slower_at_8) == 16
    assert _break_even({1: (5.0, 3.0), 2: (5.0, 6.0), 4: (6.0, 9.0)}) == 2
    assert _break_even({1: (1.0, 3.0), 3: (2.0, 3.0)}) == 1
    assert _break_even({1: (5.0, 3.0), 3: (9.0, 8.0)}) is None


def test__threshold_is_the_largest_break_even_and_at_least_2() -> None:
    """Q-T231.2: T is the largest break-even over the models that have one, and at least 2;
    None when no model has one."""
    from openftgs.cli._bench import _threshold

    assert _threshold([4, None, 8]) == 8
    assert _threshold([1, None, 1]) == 2
    assert _threshold([None, None]) is None


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


# B-7, `ftgs bench render` (§3.4, D-55; Q-174, Q-T231.1, Q-T231.2), run once on the GPU for the
# tests below on `tested_scene`: three distinct training times, so requests of 1, 2 and 3 times.
# It builds the LPIPS networks for its evaluation figures, hence `weights`.

B7_MODELS = [(n, active) for n in (1_000_000, 3_000_000) for active in (0.01, 0.05, 0.2)]


@pytest.fixture(scope="module")
def render_bench(tested_scene: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("bench") / "B-7.json"
    assert _ftgs("bench", "render", tested_scene, "--out", out) == 0
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return body


def _b7_cases(body: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """B-7's cases by their keys' values in a fixed order, absent keys as None."""
    keys = ("n", "active", "times", "index", "outputs", "evaluation", "threshold")
    return {tuple(c["case"].get(k) for k in keys): c["figures"] for c in body["cases"]}


@pytest.mark.gpu
@pytest.mark.weights
def test_b7_renders_every_training_time_with_and_without_the_index_on_both_devices(
    render_bench: dict[str, Any],
) -> None:
    """B-7 (§3.4; #231, Q-T231.1): on synthesized models at N = 1M and 3M with 1, 5 and 20 % of
    rows active, `render_request` from one 1920×1080 camera at every distinct training time
    (3 here), with the index on and off and the outputs on the GPU and on the CPU, and with the
    outputs on the GPU at requests of fewer times; FPS is renders over the request's median time
    over 10 calls; buckets one frame interval wide (D-55: here 1/30 s over a span of 2/30 s)."""
    body = render_bench
    assert body["benchmark"] == "B-7" and body["command"][:3] == ["ftgs", "bench", "render"]
    setup = body["setup"]
    assert setup["camera"]["width"] == 1920 and setup["camera"]["height"] == 1080
    assert setup["camera"]["pose"] == "ring02"
    assert (setup["times"], setup["request_sizes"], setup["repetitions"]) == (3, [1, 2, 3], 10)
    assert setup["frame_dt"] == pytest.approx(0.5)
    cases = _b7_cases(body)
    for n, active in B7_MODELS:
        for k in (1, 2, 3):
            for outputs in ("cuda", "cpu") if k == 3 else ("cuda",):
                for index in (True, False):
                    figures = cases[(n, active, k, index, outputs, None, None)]
                    assert figures["renders"] == k and figures["repetitions"] == 10
                    assert 0.0 < figures["request_s"] < 60.0 and figures["first_call_s"] > 0.0
                    assert figures["fps"] == pytest.approx(k / figures["request_s"])
                    assert figures["render_s"] == pytest.approx(figures["request_s"] / k)
        assert (n, active, 1, True, "cpu", None, None) not in cases


@pytest.mark.gpu
@pytest.mark.weights
def test_b7_reports_the_index_build_and_its_buckets(render_bench: dict[str, Any]) -> None:
    """B-7 (#231, Q-T231.1, D-55): per model, the index's build time and, at the 3 times, the
    rows passed to a render (a superset of those that pass the opacity cull, U-32), the bucket's
    own and the long rows, each over N: passed = bucket + long; and the model's break-even."""
    cases = _b7_cases(render_bench)
    for n, active in B7_MODELS:
        figures = cases[(n, active, None, None, None, None, None)]
        assert figures["build_s"] > 0.0 and figures["build_first_call_s"] > 0.0
        assert figures["bucket_width"] == pytest.approx(0.5)
        for name in ("passed_share_mean", "bucket_share_mean", "long_share", "active_share_mean"):
            assert 0.0 <= figures[name] <= 1.0, name
        assert figures["passed_share_mean"] == pytest.approx(
            figures["bucket_share_mean"] + figures["long_share"]
        )
        assert figures["passed_share_mean"] >= figures["active_share_mean"]
        assert abs(figures["active_share_mean"] - active) <= 0.01  # Q-181's synthesizer
        assert figures["break_even_times"] in (None, 1, 2, 3)


@pytest.mark.gpu
@pytest.mark.weights
def test_b7_holds_the_fps_budget_and_sets_the_threshold(render_bench: dict[str, Any]) -> None:
    """B-7 (§3.4: ≥ 450 FPS; Q-180, Q-T231.2): a budget entry per model for the requests at
    every training time with the outputs on the GPU, index on and off, which never changes the
    exit code; and T, the largest break-even over the models that have one and at least 2, or
    None when none has."""
    body = render_bench
    cases = _b7_cases(body)
    assert len(body["budgets"]) == 2 * len(B7_MODELS)
    for budget in body["budgets"]:
        case = budget["case"]
        assert (case["times"], case["outputs"]) == (3, "cuda")
        figures = cases[(case["n"], case["active"], 3, case["index"], "cuda", None, None)]
        assert budget == {
            "case": case,
            "figure": "fps",
            "limit": 450.0,
            "value": figures["fps"],
            "pass": figures["fps"] >= 450.0,
            "kind": "budget",
        }
    found = [cases[(n, a, None, None, None, None, None)]["break_even_times"] for n, a in B7_MODELS]
    known = [k for k in found if k is not None]
    threshold = cases[(None, None, None, None, None, None, "T")]["distinct_times"]
    assert threshold == (max([*known, 2]) if known else None)


@pytest.mark.gpu
@pytest.mark.weights
def test_b7_times_the_share_of_evaluation_spent_rendering(render_bench: dict[str, Any]) -> None:
    """PRD #220's rule for `evaluate_images` (Q-68, Q-T231.1): per model, the evaluation of
    the test split (camera ring02 at 3 times) against the render requests it sends, one per
    timestamp; the share is the second over the first."""
    cases = _b7_cases(render_bench)
    assert render_bench["setup"]["evaluation_split"] == "test"
    for n, active in B7_MODELS:
        figures = cases[(n, active, None, None, None, "test", None)]
        assert (figures["images"], figures["requests"]) == (3, 3)
        assert 0.0 < figures["render_s"] < figures["evaluation_s"]
        assert figures["render_share"] == pytest.approx(
            figures["render_s"] / figures["evaluation_s"]
        )


@pytest.mark.gpu
@pytest.mark.weights
def test_b7_times_the_model_given(tested_scene: Path, tmp_path: Path) -> None:
    """Q-167, Q-T231.1: with `--model`, B-7 times that model alone, its N in the cases and no
    active fraction; an exchange model carries no Δt_f, so the buckets take the scene's."""
    model = tmp_path / "model.safetensors"
    write_exchange(random_model(32, seed=1, sh_degree=1, time=TIME).to_exchange(), model)
    out = tmp_path / "B-7.json"
    assert _ftgs("bench", "render", tested_scene, "--out", out, "--model", model) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["command"][-2:] == ["--model", str(model)]
    assert body["setup"]["model"] == f"given: {model}"
    assert body["setup"]["frame_dt"] == pytest.approx(0.5)
    models = {(c["case"].get("n"), c["case"].get("active")) for c in body["cases"]}
    assert models == {(32, None), (None, None)}  # the threshold case has neither
    assert len(body["budgets"]) == 2


@pytest.mark.gpu
def test_bench_render_without_the_lpips_weights_exits_2(
    tested_scene: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-166, Q-142, Q-T231.1: B-7's evaluation figures need the LPIPS networks, built as
    `ftgs eval` builds them before anything is measured; when their weights can't be had (an
    OSError of the download) it is exit 2, an invalid environment, with nothing written."""
    import openftgs.eval

    def offline(self: Any, **kwargs: Any) -> None:
        raise OSError("no network")

    monkeypatch.setattr(openftgs.eval.Evaluator, "__init__", offline)
    out = tmp_path / "B-7.json"
    assert _ftgs("bench", "render", tested_scene, "--out", out) == 2
    assert "LPIPS networks" in capsys.readouterr().err
    assert not out.exists()

# SPDX-License-Identifier: Apache-2.0
"""`ftgs-adapter forward`: the thirteen L2 query kinds of §3.3 (Q-207), each answered by the
package's own functions. Expected values come from the equations, written here in float64 from
the FTGS-IF arrays (E1 to E4, E11, E12, E13, E15, D-13, D-27, D-44), or, for the kinds whose
answer is the trainer's own computation (losses, grad_stats), from the direct library calls."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.stats
import torch
from builders import exchange_arrays, ring
from real_sh import real_sh

from ftgs_harness.formats import (
    Answer,
    Exchange,
    Query,
    model_tensors,
    read_answers,
    write_queries,
)
from openftgs.cli import _adapter
from openftgs.config import LossConfig, TrainConfig, config_to_dict
from openftgs.density import GradStats
from openftgs.losses import l1_loss, reg4d_loss, ssim, total_loss
from openftgs.model import FreeTimeGaussians
from openftgs.render import render
from openftgs.train import calendar_entry
from openftgs.types import Camera, ExchangeModel

TAU0, TAU1 = 2.0, 12.0


def _exchange(n: int = 8, sh_degree: int = 2, seed: int = 0) -> Exchange:
    arrays = {k: v.numpy() for k, v in exchange_arrays(n, sh_degree, seed).items()}
    arrays["position"] = arrays["position"] * np.float32(0.3)
    arrays["scale"] = arrays["scale"] * np.float32(0.2)
    return Exchange(arrays, sh_degree, TAU0, TAU1)


def _ask(
    tmp_path: Path, kind: str, params: dict[str, Any], tensors: dict[str, np.ndarray]
) -> Answer:
    """One query of `kind` through `ftgs-adapter forward`, in process."""
    queries = tmp_path / f"{kind}.q.safetensors"
    sha = write_queries(queries, kind, [Query("q0", params, tensors)])
    answers = tmp_path / f"{kind}.a.safetensors"
    assert _adapter.main(["--backend", "gsplat", "forward", str(queries), str(answers)]) == 0
    answer = read_answers(answers, kind=kind, queries_sha256=sha)["q0"]
    assert answer.status == "ok", answer.message
    return answer


def _with_model(ex: Exchange, **params: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tensors, model_params = model_tensors(ex)
    return {"model": model_params, **params}, tensors


TIMES = np.array([TAU0, 4.25, 7.0, TAU1, TAU1 + 3.0, TAU0 - 1.5])  # in and outside [τ₀, τ₁]


def _raw_motion(ex: Exchange) -> tuple[np.ndarray, np.ndarray]:
    a = {k: v.astype(np.float64) for k, v in ex.arrays.items()}
    dt = TIMES[:, None] - a["time_center"][None]  # [K, N], raw units
    means = a["position"][None] + a["velocity"][None] * dt[..., None]  # E1
    sigma_t = np.exp(-0.5 * (dt / a["duration"][None]) ** 2)  # E4
    return means, sigma_t


def test_info_lists_all_thirteen_query_kinds(capsys: pytest.CaptureFixture[str]) -> None:
    """§3.3 L2: openftgs answers every kind (Q-205)."""
    from ftgs_harness.info import QUERY_KINDS

    assert sorted(_adapter._info("gsplat")["query_kinds"]) == sorted(QUERY_KINDS)


def test_motion_and_temporal_and_effective_opacity(tmp_path: Path) -> None:
    """E1, E4, E3 in raw time: μx(τ) = position + velocity·(τ − time_center); σt =
    exp(−½((τ − time_center)/duration)²); σ·σt; inside and outside [τ₀, τ₁]. Tolerances §3.3's
    (1e-6 relative for motion; 1e-6 for the opacities)."""
    ex = _exchange()
    params, tensors = _with_model(ex)
    tensors["times"] = TIMES
    means, sigma_t = _raw_motion(ex)
    got = _ask(tmp_path, "motion", params, tensors).tensors["means_t"]
    np.testing.assert_allclose(got, means, rtol=1e-6, atol=1e-6 * np.abs(means).max())
    got = _ask(tmp_path, "temporal_opacity", params, tensors).tensors["sigma_t"]
    np.testing.assert_allclose(got, sigma_t, rtol=0, atol=1e-6)
    got = _ask(tmp_path, "effective_opacity", params, tensors).tensors["opacity_eff"]
    opacity = ex.arrays["opacity"].astype(np.float64)[None]
    np.testing.assert_allclose(got, opacity * sigma_t, rtol=0, atol=1e-6)


@pytest.mark.parametrize("degree", [None, 0, 1])
def test_sh_color_is_e2_at_the_moved_mean(tmp_path: Path, degree: int | None) -> None:
    """E2: max(0, Σ c_lm Y_lm(d) + 0.5), d from the camera centre to μx(τ), with `sh_degree`
    bands (null: the model's); tolerance 1e-5."""
    ex = _exchange()
    params, tensors = _with_model(ex, sh_degree=degree)
    tensors["times"] = TIMES[:4]
    centers = np.array([[4.0, 0.0, 1.0], [0.0, -3.0, 2.0], [1.0, 1.0, -5.0], [-2.0, 2.0, 2.0]])
    tensors["centers"] = centers
    means, _ = _raw_motion(ex)
    use = ex.sh_degree if degree is None else degree
    coeffs = ex.arrays["sh"].astype(np.float64)[:, : (use + 1) ** 2]
    expected = np.stack(
        [
            np.maximum((coeffs * real_sh(means[k] - centers[k], use)[..., None]).sum(1) + 0.5, 0.0)
            for k in range(4)
        ]
    )
    got = _ask(tmp_path, "sh_color", params, tensors).tensors["colors"]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


def test_losses_are_the_trainers_terms(tmp_path: Path) -> None:
    """E7 to E9 without LPIPS (no download): L1, 1 − SSIM, L_reg at the query's time and the
    total, against the direct calls; L1 also against its definition."""
    ex = _exchange()
    g = np.random.default_rng(3)
    pred = g.random((20, 24, 3), dtype=np.float32)
    gt = g.random((20, 24, 3), dtype=np.float32)
    params, tensors = _with_model(ex, lpips=False, iteration=5, loss={"lambda_reg": 0.02})
    tensors.update({"pred": pred, "gt": gt, "t": np.array(6.0)})
    got = _ask(tmp_path, "losses", params, tensors).tensors
    assert set(got) == {"l1", "one_minus_ssim", "reg", "total"}
    assert float(got["l1"]) == pytest.approx(float(np.abs(pred - gt).mean()), rel=1e-6)
    model = FreeTimeGaussians.from_exchange(_openftgs(ex), device="cpu")
    t = (6.0 - TAU0) / (TAU1 - TAU0)
    p, q = torch.from_numpy(pred)[None], torch.from_numpy(gt)[None]
    cfg = LossConfig(lambda_perc=0.0, lambda_reg=0.02)
    total, _ = total_loss(p, q, model, t, 5, cfg)
    assert float(got["one_minus_ssim"]) == pytest.approx(1.0 - float(ssim(p, q)), rel=1e-5)
    assert float(got["reg"]) == pytest.approx(float(reg4d_loss(model, t).detach()), rel=1e-5)
    assert float(got["total"]) == pytest.approx(float(total.detach()), rel=1e-5)
    assert float(l1_loss(p, q)) == pytest.approx(float(got["l1"]), rel=1e-6)


@pytest.mark.weights
def test_losses_with_lpips_vgg(tmp_path: Path) -> None:
    """E7, D-9: with `lpips` the answer adds LPIPS-VGG on inputs mapped to [−1, 1], and the
    total includes λ_perc times it; against the direct calls (§3.3 tolerance 1e-3)."""
    from openftgs.losses import LPIPSLoss

    ex = _exchange()
    g = np.random.default_rng(4)
    pred = g.random((40, 48, 3), dtype=np.float32)
    gt = g.random((40, 48, 3), dtype=np.float32)
    params, tensors = _with_model(ex, lpips=True, iteration=1)
    tensors.update({"pred": pred, "gt": gt, "t": np.array(3.0)})
    got = _ask(tmp_path, "losses", params, tensors).tensors
    p, q = torch.from_numpy(pred)[None], torch.from_numpy(gt)[None]
    vgg = LPIPSLoss("vgg")
    assert float(got["lpips_vgg"]) == pytest.approx(float(vgg(p, q).detach()), abs=1e-3)
    model = FreeTimeGaussians.from_exchange(_openftgs(ex), device="cpu")
    total, _ = total_loss(p, q, model, 0.1, 1, LossConfig(), perceptual=vgg)
    assert float(got["total"]) == pytest.approx(float(total.detach()), rel=1e-5)


def _openftgs(ex: Exchange) -> ExchangeModel:
    arrays = {k: torch.from_numpy(v.copy()) for k, v in ex.arrays.items()}
    return ExchangeModel(arrays, ex.sh_degree, ex.tau0, ex.tau1, ex.time_unit, "test")


def test_lr_schedule_follows_e12_and_e13(tmp_path: Path) -> None:
    """E12, E13 and §1.3's table, columns in its order: at it = 1 the start rates, at T_it the
    end ones, log-linear between; velocities' over Δt_f (D-19); the constants constant."""
    s, dt, t_it = 2.5, 0.05, 101
    its = np.array([1, 51, t_it], dtype=np.int64)
    got = _ask(
        tmp_path,
        "lr_schedule",
        {"optim": {"iterations": t_it}, "spatial_scale": s, "frame_dt": dt},
        {"iterations": its},
    ).tensors["rates"]
    u = (its - 1) / (t_it - 1)
    means = 1.6e-4 * s * (1e-2) ** u
    expected = np.stack(
        [means, means / dt, 1.6e-4 * (1e-2) ** u]
        + [np.full(3, r) for r in (5e-3, 5e-3, 1e-3, 5e-2, 2.5e-3, 1.25e-4)],
        axis=1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def _e11(sigma: float, a: float, n: int) -> tuple[float, float]:
    """E11 in float64, written from its formula, with D-14's clamp of σ_new."""
    new = 1.0 - (1.0 - sigma) ** (1.0 / n)
    denominator = sum(
        math.comb(i - 1, k) * (-1) ** k * new ** (k + 1) / math.sqrt(k + 1)
        for i in range(1, n + 1)
        for k in range(i)
    )
    return min(max(new, 0.005), 1.0 - 2.0**-23), a * sigma / denominator


def test_relocation_math_is_e11_with_the_whole_clamp(tmp_path: Path) -> None:
    """E11, D-14: σ_new = 1 − (1 − σ)^(1/n) clamped to [0.005, 1 − 2⁻²³] (the lower bound too,
    Q-50), a_new from the unclamped σ_new; tolerance 1e-5 relative."""
    cases = [(0.75, 0.3, 2), (0.5, 1.0, 1), (0.9, 0.05, 7), (0.01, 0.2, 51), (0.999, 2.0, 3)]
    sigma = np.array([c[0] for c in cases])
    scales = np.array([[c[1], 2 * c[1], 0.5 * c[1]] for c in cases])
    n = np.array([c[2] for c in cases], dtype=np.int64)
    got = _ask(tmp_path, "relocation_math", {}, {"opacity": sigma, "scales": scales, "n": n})
    for row, (s, a, k) in enumerate(cases):
        new, a_new = _e11(s, a, k)
        assert float(got.tensors["opacity_new"][row]) == pytest.approx(new, rel=1e-5)
        assert float(got.tensors["scales_new"][row, 0]) == pytest.approx(a_new, rel=1e-5)
    assert float(got.tensors["opacity_new"][3]) == 0.005  # the lower bound binds (σ = 0.01, n = 51)


def test_velocity_init_is_e15(tmp_path: Path) -> None:
    """E15: (x'_nn(i) − x_i)/dt, by brute force; also the backward difference (dt < 0)."""
    g = np.random.default_rng(5)
    points = g.random((30, 3), dtype=np.float32)
    nxt = g.random((40, 3), dtype=np.float32)
    for dt in (0.1, -0.1):
        got = _ask(
            tmp_path,
            "velocity_init",
            {},
            {"points": points, "next_points": nxt, "dt": np.array(dt)},
        ).tensors["velocities"]
        d = np.linalg.norm(points[:, None].astype(np.float64) - nxt[None], axis=-1)
        expected = (nxt[d.argmin(1)].astype(np.float64) - points) / dt
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_relocation_sampling_draws_in_proportion_to_the_scores(tmp_path: Path) -> None:
    """D-15: 50 seeds × 200 draws from 20 live targets, pooled: χ² against the scores,
    p > 0.001 (§3.3); zero-score rows never drawn; one seed is the same draw twice."""
    scores = np.concatenate([np.linspace(0.1, 1.0, 20), np.zeros(5)]).astype(np.float32)
    params = {"count": 200, "seeds": list(range(50))}
    counts = _ask(tmp_path, "relocation_sampling", params, {"scores": scores}).tensors["counts"]
    assert counts.shape == (50, 25) and (counts.sum(1) == 200).all() and (counts[:, 20:] == 0).all()
    pooled = counts.sum(0)[:20]
    expected = scores[:20] / scores.sum() * pooled.sum()
    assert scipy.stats.chisquare(pooled, expected).pvalue > 0.001
    again = _ask(tmp_path, "relocation_sampling", params, {"scores": scores}).tensors["counts"]
    assert np.array_equal(again, counts)


def test_grad_stats_is_one_training_steps_statistic(tmp_path: Path) -> None:
    """D-12: the NDC-scaled 2D-mean gradient norm of E9 for one step, and visibility, as the
    trainer computes them on the reference backend."""
    ex = _exchange(n=10, sh_degree=1)
    ex.arrays["duration"] = np.full(10, 50.0)  # every primitive present at τ = 5
    cam = ring(1, radius=4.0, elevation=1.0, width_px=24, height_px=16, focal=20.0)[0]
    gt = np.random.default_rng(1).random((16, 24, 3), dtype=np.float32)
    cfg = TrainConfig(sh_degree=1, loss=LossConfig(lambda_perc=0.0))
    config = config_to_dict(cfg)
    params, tensors = _with_model(ex, width=24, height=16, iteration=1500, config=config)
    tensors.update(
        {
            "K": cam.K.double().numpy(),
            "world_to_camera": cam.world_to_camera.double().numpy(),
            "t": np.array(5.0),
            "gt": gt,
        }
    )
    got = _ask(tmp_path, "grad_stats", params, tensors).tensors
    model = FreeTimeGaussians.from_exchange(_openftgs(ex), device="cpu")
    t = (5.0 - TAU0) / (TAU1 - TAU0)
    camera = Camera(K=cam.K, world_to_camera=cam.world_to_camera, width=24, height=16)
    out = render(
        model,
        camera,
        t,
        time_units="normalized",
        sh_degree=calendar_entry(cfg, 1500).sh_degree,
        backend="reference",
        return_aux=True,
        packed=False,
    )
    loss, _ = total_loss(out.image, torch.from_numpy(gt)[None], model, t, 1500, cfg.loss)
    torch.autograd.backward(loss)
    stats = GradStats(10)
    assert out.means2d is not None and out.means2d.grad is not None and out.visible is not None
    stats.update(out.means2d.grad, width=24, height=16, visible=out.visible)
    assert got["visible"].any()
    assert np.array_equal(got["visible"], out.visible[0].numpy())
    np.testing.assert_allclose(got["grad_norm"], stats.g_sum.numpy(), rtol=1e-6, atol=0)


def test_relocation_score_is_e10(tmp_path: Path) -> None:
    """E10, D-13 with λ_g = λ_o = 0.5: ρ = 0.5·min(1, g/Q₀.₉₉) + 0.5·σ, g = g_sum/count, Q over
    the live rows seen at least once. Here every seen row has g = 2, so ĝ = 1 where seen and 0
    where not, and the dead row (σ = 0.001) doesn't count toward Q."""
    sigma = np.array([0.2, 0.6, 0.9, 0.001, 0.4], dtype=np.float32)
    g_sum = np.array([4.0, 2.0, 6.0, 1000.0, 0.0], dtype=np.float32)
    count = np.array([2, 1, 3, 1, 0], dtype=np.int32)
    got = _ask(
        tmp_path, "relocation_score", {}, {"opacity": sigma, "g_sum": g_sum, "count": count}
    ).tensors["score"]
    seen = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(got, 0.5 * seen + 0.5 * sigma, rtol=0, atol=1e-6)


def test_spatial_scale_is_d27_over_the_training_cameras(tmp_path: Path) -> None:
    """D-27: 1.1 × the largest distance of a training centre from their centroid; a test camera
    far away is left out. Four centres at distance 2: S = 2.2."""
    w2c = []
    for center in [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (100, 0, 0)]:
        m = np.eye(4)
        m[:3, 3] = -np.array(center, dtype=np.float64)
        w2c.append(m)
    train = np.array([True, True, True, True, False])
    got = _ask(
        tmp_path,
        "spatial_scale",
        {"factor": 1.1},
        {"world_to_camera": np.stack(w2c), "train": train},
    ).tensors["S"]
    assert float(got) == pytest.approx(2.2, rel=1e-9)


def test_growth_decisions_apply_the_cap_and_the_clone_split_boundary(tmp_path: Path) -> None:
    """D-18, D-44: g = (3, 5, 1, 5, 3, 0) above τ_pos = 0.5, N = 6, N_max = 9: room for three,
    rows 1 and 3 then 0; row 3 is large (above 0.01·S = 1), so split, the others cloned."""
    g = np.array([3.0, 5.0, 1.0, 5.0, 3.0, 0.0], dtype=np.float32)
    scales = np.full((6, 3), 0.1, dtype=np.float32)
    scales[3, 2] = 1.5
    params = {"spatial_scale": 100.0, "grad_threshold": 0.5, "split_scale": 0.01, "n_max": 9}
    got = _ask(tmp_path, "growth_decisions", params, {"g": g, "scales": scales}).tensors
    assert got["clone"].tolist() == [True, True, False, False, False, False]
    assert got["split"].tolist() == [False, False, False, True, False, False]


def test_a_query_that_fails_is_an_error_answer_and_the_others_are_answered(
    tmp_path: Path,
) -> None:
    """Q-207: an `error` answer for the query that can't be answered; the rest still are."""
    good = Query("q0", {}, {"g": np.zeros(2, np.float32), "scales": np.ones((2, 3), np.float32)})
    good.params = {"spatial_scale": 1.0, "grad_threshold": 0.5, "split_scale": 0.01, "n_max": 9}
    bad = Query("q1", {"spatial_scale": 1.0}, {})
    queries = tmp_path / "q.safetensors"
    sha = write_queries(queries, "growth_decisions", [good, bad])
    out = tmp_path / "a.safetensors"
    assert _adapter.main(["--backend", "gsplat", "forward", str(queries), str(out)]) == 0
    answers = read_answers(out, kind="growth_decisions", queries_sha256=sha)
    assert answers["q0"].status == "ok" and answers["q1"].status == "error"
    unknown = tmp_path / "u.safetensors"
    sha = write_queries(unknown, "no_such_kind", [Query("q0", {}, {})])
    assert _adapter.main(["--backend", "gsplat", "forward", str(unknown), str(out)]) == 0
    assert (
        read_answers(out, kind="no_such_kind", queries_sha256=sha)["q0"].status == "not_comparable"
    )

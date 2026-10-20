# SPDX-License-Identifier: Apache-2.0
"""The harness's own §2.2 readers, writers and validators, and the forward formats (§2.2,
§3.3 adapter contract; D-71, D-84; Q-207). openftgs's own I/O is the other side of every
round trip: the two were written separately from §2.2, so each checks the other."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from builders import BAD_EXCHANGE_CASES, bad_exchange, exchange_arrays
from safetensors.torch import save_file

from ftgs_harness.formats import (
    Answer,
    Exchange,
    FormatError,
    Query,
    RequestItem,
    model_from_tensors,
    model_tensors,
    read_answers,
    read_exchange,
    read_queries,
    read_renders,
    read_scene,
    write_answers,
    write_exchange,
    write_queries,
    write_request,
)
from openftgs import io
from openftgs.types import ExchangeModel, Renders


def _openftgs_model(n: int = 6, sh_degree: int = 3) -> ExchangeModel:
    return ExchangeModel(
        exchange_arrays(n, sh_degree), sh_degree, 2.0, 12.0, "s", "test", background=(1.0, 0.5, 0.0)
    )


def test_reads_what_openftgs_writes(tmp_path: Path) -> None:
    """§2.2: every key and the metadata, bit for bit, the background included (D-84)."""
    model = _openftgs_model()
    io.write_exchange(model, tmp_path / "m.safetensors")
    ex = read_exchange(tmp_path / "m.safetensors")
    assert (ex.sh_degree, ex.tau0, ex.tau1, ex.time_unit) == (3, 2.0, 12.0, "s")
    for key, value in model.arrays.items():
        assert np.array_equal(ex.arrays[key], value.numpy()), key
    assert ex.render_conventions is not None
    assert ex.render_conventions["background"] == [1.0, 0.5, 0.0]


def test_openftgs_reads_what_the_harness_writes(tmp_path: Path) -> None:
    model = _openftgs_model(sh_degree=1)
    ex = Exchange(
        {k: v.numpy() for k, v in model.arrays.items()},
        1,
        2.0,
        12.0,
        render_conventions={"background": [0.0, 0.0, 1.0]},
    )
    write_exchange(tmp_path / "m.safetensors", ex)
    back = io.read_exchange(tmp_path / "m.safetensors")
    assert back.sh_degree == 1 and back.background == (0.0, 0.0, 1.0)
    for key, value in model.arrays.items():
        assert torch.equal(back.arrays[key], value), key


@pytest.mark.parametrize("case", BAD_EXCHANGE_CASES)
def test_rejects_every_file_openftgs_rejects(tmp_path: Path, case: str) -> None:
    """§2.2 validation: each malformed file openftgs's reader refuses, the harness refuses too."""
    arrays, meta = bad_exchange(case)
    path = tmp_path / "bad.safetensors"
    save_file({k: v.contiguous() for k, v in arrays.items()}, str(path), meta)
    with pytest.raises(ValueError):
        io.read_exchange(path)
    with pytest.raises(FormatError):
        read_exchange(path)


@pytest.mark.parametrize("case", ["w < 0", "not unit", "single timestamp with a velocity"])
def test_rejects_rotations_and_single_timestamps_outside_section_2_2(
    tmp_path: Path, case: str
) -> None:
    """§2.2: unit quaternions with w ≥ 0; D-71: with τ₀ = τ₁ every time_center is τ₀ and every
    velocity 0."""
    arrays = {k: v.numpy().copy() for k, v in exchange_arrays().items()}
    tau0, tau1 = 2.0, 12.0
    if case == "w < 0":
        arrays["rotation"][0] *= -1.0
    elif case == "not unit":
        arrays["rotation"][0] *= 1.01
    else:
        tau1 = tau0
        arrays["time_center"][:] = tau0
    with pytest.raises(FormatError):
        write_exchange(tmp_path / "m.safetensors", Exchange(arrays, 3, tau0, tau1))


def _item(item_id: str, w: int = 8, h: int = 6) -> RequestItem:
    K = np.array([[10.0, 0.0, w / 2], [0.0, 10.0, h / 2], [0.0, 0.0, 1.0]])
    w2c = np.eye(4)
    w2c[2, 3] = 4.0
    return RequestItem(item_id, 2.5, K, w2c, w, h)


def test_requests_and_renders_cross_openftgs(tmp_path: Path) -> None:
    """§2.2: openftgs parses the harness's request; the harness reads openftgs's renders for
    it, checks the request's SHA-256 and each item's size."""
    items = [_item("r0000"), _item("r0001", 4, 3)]
    sha = write_request(tmp_path / "req.json", items, background=(0.0, 1.0, 0.0))
    request = io.read_render_request(tmp_path / "req.json")
    assert [i.id for i in request.items] == ["r0000", "r0001"]
    assert request.background == (0.0, 1.0, 0.0)
    assert request.items[1].camera.width == 4
    renders = Renders(
        rgb={i.id: torch.rand(i.height, i.width, 3) for i in items},
        alpha={i.id: torch.rand(i.height, i.width, 1) for i in items},
    )
    io.write_renders(renders, tmp_path / "out.safetensors", producer="t", request_sha256=sha)
    got = read_renders(tmp_path / "out.safetensors", request_sha256=sha, items=items)
    assert np.array_equal(got["r0001"][0], renders.rgb["r0001"].numpy())
    with pytest.raises(FormatError, match="made for request"):
        read_renders(tmp_path / "out.safetensors", request_sha256="0" * 64, items=items)
    with pytest.raises(FormatError, match="rgb/r0000"):
        read_renders(tmp_path / "out.safetensors", request_sha256=sha, items=[_item("r0000", 9)])


def test_scene_manifest_with_a_per_image_pose(tmp_path: Path) -> None:
    """§2.2: an image may override its camera's K and world_to_camera; paths are relative."""
    cam = {"width": 4, "height": 3, "K": np.eye(3).tolist(), "world_to_camera": np.eye(4).tolist()}
    pose = np.eye(4)
    pose[0, 3] = 1.0
    body = {
        "format": "ftgs-scene",
        "version": 1,
        "time_unit": "s",
        "cameras": {"c": cam},
        "images": [
            {"camera": "c", "timestamp": 0.0, "path": "a.png", "split": "train"},
            {
                "camera": "c",
                "timestamp": 1.0,
                "path": "b.png",
                "split": "test",
                "world_to_camera": pose.tolist(),
            },
        ],
    }
    (tmp_path / "scene.json").write_text(json.dumps(body))
    scene = read_scene(tmp_path / "scene.json")
    assert [i.split for i in scene.images] == ["train", "test"]
    assert scene.images[1].path == tmp_path / "b.png"
    assert np.array_equal(scene.images[1].world_to_camera, pose)


def test_forward_queries_and_answers_round_trip(tmp_path: Path) -> None:
    """Q-207: one file per kind; tensors under <id>/<name>, the model under <id>/model/…, and
    the answers tied to the queries file by its SHA-256."""
    ex = read_exchange_from(_openftgs_model(), tmp_path)
    model, params = model_tensors(ex)
    times = np.array([2.0, 7.5], dtype=np.float64)
    queries = [Query("q0000", {"model": params}, {**model, "times": times})]
    sha = write_queries(tmp_path / "q.safetensors", "motion", queries)
    kind, back = read_queries(tmp_path / "q.safetensors")
    assert kind == "motion" and back[0].params == {"model": params}
    assert np.array_equal(back[0].tensors["times"], times)
    again = model_from_tensors(back[0].tensors, back[0].params["model"])
    assert np.array_equal(again.arrays["sh"], ex.arrays["sh"])
    answers = [
        Answer("q0000", "ok", "", {"means_t": np.zeros((2, 6, 3), np.float32)}),
        Answer("q0001", "not_comparable", "unknown", {}),
    ]
    write_answers(tmp_path / "a.safetensors", "motion", sha, answers)
    got = read_answers(tmp_path / "a.safetensors", kind="motion", queries_sha256=sha)
    assert got["q0000"].tensors["means_t"].shape == (2, 6, 3)
    assert got["q0001"].status == "not_comparable"
    with pytest.raises(FormatError):
        read_answers(tmp_path / "a.safetensors", kind="motion", queries_sha256="x")
    with pytest.raises(FormatError):
        read_answers(tmp_path / "a.safetensors", kind="losses", queries_sha256=sha)


def read_exchange_from(model: ExchangeModel, tmp_path: Path) -> Exchange:
    io.write_exchange(model, tmp_path / "src.safetensors")
    return read_exchange(tmp_path / "src.safetensors")

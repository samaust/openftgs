# SPDX-License-Identifier: Apache-2.0
"""The fault-injecting wrapper adapter (ticket #284): with no fault its outputs are the wrapped
adapter's, tensor for tensor with the same metadata; each fault changes only what it names."""

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from adapter_cases import exchange_file, request_file
from safetensors import safe_open

HERE = Path(__file__).parent
ADAPTER = str(Path(sys.executable).parent / "ftgs-adapter")
INNER = f"{ADAPTER} --backend reference"


def _wrapped(*args: object, faults: tuple[str, ...] = ()) -> int:
    argv = [sys.executable, str(HERE / "fault_adapter.py"), "--inner", INNER]
    for fault in faults:
        argv += ["--fault", fault]
    return subprocess.run([*argv, *(str(a) for a in args)], check=False).returncode


def _plain(*args: object) -> int:
    return subprocess.run([*shlex.split(INNER), *(str(a) for a in args)], check=False).returncode


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with safe_open(str(path), framework="np") as f:
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="np") as f:
        return dict(f.metadata() or {})


def test_without_a_fault_the_outputs_are_the_wrapped_adapters(tmp_path: Path) -> None:
    model = exchange_file(tmp_path / "m.safetensors")
    request = request_file(tmp_path / "req.json")
    assert _wrapped("render", model, request, tmp_path / "w.safetensors") == 0
    assert _plain("render", model, request, tmp_path / "p.safetensors") == 0
    # safetensors keeps its metadata in a hash map, so equal files can differ in their bytes.
    w, p = _arrays(tmp_path / "w.safetensors"), _arrays(tmp_path / "p.safetensors")
    assert w.keys() == p.keys() and all(np.array_equal(w[k], p[k]) for k in w)
    assert _metadata(tmp_path / "w.safetensors") == _metadata(tmp_path / "p.safetensors")


@pytest.mark.parametrize(
    ("fault", "changed"),
    [
        ("velocity=1.1", "velocity"),
        ("duration=1.2", "duration"),
        ("dc=0.05", "sh"),
        ("position_zero=0.5", "position"),
    ],
)
def test_a_model_fault_changes_only_its_field(tmp_path: Path, fault: str, changed: str) -> None:
    model = exchange_file(tmp_path / "m.safetensors")
    assert _wrapped("import", model, tmp_path / "w", faults=(fault,)) == 0
    assert _plain("import", model, tmp_path / "p") == 0
    assert _plain("export", tmp_path / "w", tmp_path / "w.safetensors") == 0
    assert _plain("export", tmp_path / "p", tmp_path / "p.safetensors") == 0
    w, p = _arrays(tmp_path / "w.safetensors"), _arrays(tmp_path / "p.safetensors")
    differ = sorted(k for k in w if not np.array_equal(w[k], p[k]))
    assert differ == [changed]


def test_output_faults_change_what_they_name(tmp_path: Path) -> None:
    model = exchange_file(tmp_path / "m.safetensors")
    assert _wrapped("export", model, tmp_path / "x.safetensors", faults=("drop_key=velocity",)) == 0
    assert "velocity" not in _arrays(tmp_path / "x.safetensors")
    request = request_file(tmp_path / "req.json")
    assert _wrapped("render", model, request, tmp_path / "n.safetensors", faults=("nan",)) == 0
    rgb = _arrays(tmp_path / "n.safetensors")["rgb/r0"]
    assert np.isnan(rgb).sum() == 3 and np.isnan(rgb[rgb.shape[0] // 2, rgb.shape[1] // 2]).all()
    assert _wrapped("render", model, request, tmp_path / "c.safetensors", faults=("cx=0.5",)) == 0
    with safe_open(str(tmp_path / "c.safetensors"), framework="np") as f:
        sha = (f.metadata() or {})["request_sha256"]
    assert sha == hashlib.sha256(request.read_bytes()).hexdigest()
    assert _wrapped("info", faults=("exit=5",)) == 5


def test_forward_faults_drop_answer_tensors_and_answer_without_lpips(tmp_path: Path) -> None:
    """`lpips_off` answers `losses` as if its queries said `lpips` false, so no weights are
    needed, and files the answers under the original queries; `drop_answer` removes the
    tensors it names from every answer."""
    from ftgs_harness.formats import file_sha256, read_answers, write_queries
    from ftgs_harness.queries import generate

    queries = tmp_path / "losses.queries.safetensors"
    batch = generate("losses", 1, seed=0, config={"loss": {"lambda_perc": 0.0}})
    assert batch and all(q.params["lpips"] for q in batch)
    write_queries(queries, "losses", batch)
    out = tmp_path / "losses.answers.safetensors"
    faults = ("lpips_off", "drop_answer=l1,reg")
    assert _wrapped("forward", queries, out, faults=faults) == 0
    answers = read_answers(out, kind="losses", queries_sha256=file_sha256(queries))
    assert answers.keys() == {q.id for q in batch}
    assert all(a.status == "ok" for a in answers.values())
    assert all(set(a.tensors) == {"one_minus_ssim", "total"} for a in answers.values())


def test_info_faults(tmp_path: Path) -> None:
    argv = [sys.executable, str(HERE / "fault_adapter.py"), "--inner", INNER]
    done = subprocess.run(
        [*argv, "--fault", "no_kind=motion", "--fault", "drop_convention=near", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(done.stdout)
    assert "motion" not in info["query_kinds"] and len(info["query_kinds"]) == 12
    assert "near" not in info["render_conventions"]

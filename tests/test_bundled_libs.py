# SPDX-License-Identifier: Apache-2.0
"""The inventory of shared libraries bundled in dependency wheels (§2.3 notes; Q-4; #366).

Added tests, not §3 rows. The seam is the maintainer script's classifier (owner, #355): given a
distribution's file list, as its RECORD gives it, it tells the distribution's own extension
modules apart from the libraries bundled into its wheel (auditwheel's `<name>.libs/`,
`torch/lib`, `nvidia/*/lib`). A shared library in neither group is reported as unrecognised, so
the real inventory run exposes wheel layouts the classifier hasn't seen.
"""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bundled_libs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bundled_libs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


libs = _load()


def test_own_extension_modules_are_told_apart_from_bundled_libraries() -> None:
    files = [
        "numpy/__init__.py",
        "numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so",
        "numpy/linalg/_umath_linalg.cpython-311-x86_64-linux-gnu.so",
        "numpy.libs/libscipy_openblas64_-ff651d7f.so",
        "numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0",
        "pillow.libs/libjpeg-8a13c6e0.so.62.4.0",
        "torch/_C.cpython-314-x86_64-linux-gnu.so",
        "torch/lib/libtorch_cpu.so",
        "torch/lib/libgomp-a34b3233.so.1",
        "nvidia/cu13/lib/libcublas.so.13",
        "nvidia/cudnn/lib/libcudnn.so.9",
        "numpy-2.3.4.dist-info/RECORD",
        "numpy-2.3.4.dist-info/LICENSE.txt",
    ]
    result = libs.classify(files)
    assert result.extensions == [
        "numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so",
        "numpy/linalg/_umath_linalg.cpython-311-x86_64-linux-gnu.so",
        "torch/_C.cpython-314-x86_64-linux-gnu.so",
    ]
    assert result.bundled == [
        "numpy.libs/libscipy_openblas64_-ff651d7f.so",
        "numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0",
        "pillow.libs/libjpeg-8a13c6e0.so.62.4.0",
        "torch/lib/libtorch_cpu.so",
        "torch/lib/libgomp-a34b3233.so.1",
        "nvidia/cu13/lib/libcublas.so.13",
        "nvidia/cudnn/lib/libcudnn.so.9",
    ]
    assert result.unrecognised == []


def test_triton_backend_libraries_are_bundled() -> None:
    """Exposed by the first inventory run (#366): triton's wheel carries NVIDIA's CUPTI under
    `triton/backends/<backend>/lib/`."""
    path = "triton/backends/nvidia/lib/cupti/libcupti.so.12"
    assert libs.classify([path]).bundled == [path]


@pytest.mark.parametrize(
    "path",
    [
        "triton/_C/libtriton.so",
        "somepkg/lib/libfoo.so.1",
        "somepkg/vendor/libbar.dylib",
    ],
)
def test_shared_library_in_an_unknown_layout_is_unrecognised(path: str) -> None:
    result = libs.classify(["somepkg/__init__.py", path])
    assert result.unrecognised == [path]
    assert result.extensions == [] and result.bundled == []


@pytest.mark.parametrize(
    "path",
    [
        "torch/lib.py",
        "pkg/libs/readme.txt",
        "pkg/sonar.py",
        "pkg/data.so.txt",
        "nvidia/__init__.py",
    ],
)
def test_files_that_are_not_shared_libraries_are_left_out(path: str) -> None:
    result = libs.classify([path])
    assert result.extensions == result.bundled == result.unrecognised == []


def test_license_notices_are_the_metadata_files_and_the_notices_beside_the_libraries() -> None:
    files = [
        "numpy/__init__.py",
        "numpy-2.3.4.dist-info/METADATA",
        "numpy-2.3.4.dist-info/licenses/LICENSE.txt",
        "numpy-2.3.4.dist-info/licenses/numpy/_core/src/umath/svml/LICENSE",
        "cv2/LICENSE-3RD-PARTY.txt",
        "cv2/licenses/ffmpeg-COPYING.LGPLv2.1.txt",
        "pkg-1.0.dist-info/COPYING",
        "pkg/notice_board.py",
        "pkg/license_check.py",
    ]
    assert libs.notices(files) == [
        "numpy-2.3.4.dist-info/licenses/LICENSE.txt",
        "numpy-2.3.4.dist-info/licenses/numpy/_core/src/umath/svml/LICENSE",
        "cv2/LICENSE-3RD-PARTY.txt",
        "cv2/licenses/ffmpeg-COPYING.LGPLv2.1.txt",
        "pkg-1.0.dist-info/COPYING",
    ]


class _Metadata:
    def __init__(self, headers: dict[str, list[str]]) -> None:
        self._headers = headers

    def get_all(self, name: str, failobj: None = None) -> list[str] | None:
        return self._headers.get(name, failobj)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"License-Expression": ["BSD-3-Clause"], "License": ["BSD"]}, "BSD-3-Clause"),
        ({"License": ["MIT License"]}, "MIT License"),
        (
            {"License": ["Copyright (c) 2005, NumPy Developers.\nAll rights reserved."]},
            "Copyright (c) 2005, NumPy Developers.",
        ),
        (
            {"License": ["BSD"], "Classifier": ["License :: OSI Approved :: BSD License"]},
            "BSD License",
        ),
        (
            {
                "License": ["UNKNOWN"],
                "Classifier": [
                    "License :: OSI Approved :: BSD License",
                    "Programming Language :: Python",
                    "License :: OSI Approved :: MIT License",
                ],
            },
            "BSD License; MIT License",
        ),
        ({}, "UNKNOWN"),
    ],
)
def test_declared_license_takes_the_expression_then_the_classifiers_then_the_field(
    headers: dict[str, list[str]], expected: str
) -> None:
    """The precedence of pip-licenses' `--from=mixed`, which the license scan uses; a License
    field holding a whole license text is reported by its first line."""
    assert libs.declared_license(_Metadata(headers)) == expected


def _wheel(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            "pkg-1.0.dist-info/METADATA", "Name: pkg\nVersion: 1.0\nLicense-Expression: MIT\n"
        )
        for name in names:
            wheel.writestr(name, "")
    return path


def test_main_inventories_a_wheel_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wheel = _wheel(
        tmp_path / "pkg-1.0-cp314-cp314-linux_x86_64.whl",
        ["pkg/_ext.cpython-314-x86_64-linux-gnu.so", "pkg.libs/libz-1a2b.so.1", "pkg/LICENSE"],
    )
    assert libs.main(["--json", "--wheel", str(wheel)]) == 0
    (entry,) = json.loads(capsys.readouterr().out)
    assert entry["name"] == "pkg" and entry["version"] == "1.0" and entry["license"] == "MIT"
    assert entry["extensions"] == ["pkg/_ext.cpython-314-x86_64-linux-gnu.so"]
    assert entry["bundled"] == ["pkg.libs/libz-1a2b.so.1"]
    assert entry["unrecognised"] == []
    assert entry["notices"] == ["pkg/LICENSE"]


def test_main_says_when_a_wheel_holds_no_shared_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _wheel(tmp_path / "pkg-1.0-py3-none-any.whl", ["pkg/__init__.py"])
    assert libs.main(["--wheel", str(wheel)]) == 0
    assert "pkg 1.0: no shared library" in capsys.readouterr().out

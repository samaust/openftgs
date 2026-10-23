# Release checklist

M18's record of the release-time checks that spec §2.3's notes and D-79 call for, for openftgs 0.1.0 (PRD #355): the license scan and its sign-offs, the libraries bundled in dependency wheels, the vendored license files, the pretrained weights' terms, the absence of dataset material, and the source-log audit. Each section gives the commands run, the commit and environment they ran on, and their results; a sign-off is the owner's, given in the build session, and is recorded with its date.

## License scan

`scripts/license_scan.py` (Q-4) run with each environment's interpreter, from a checkout of `ticket-366-license-scan` at 0cf7653 (the scan script and `scripts/license_signoff.json` as on `prd-355-release` at 7e1faf0), on 2026-10-22:

```
<env>/bin/python scripts/license_scan.py
```

The environments were built fresh under `.scratch/prd-355-release/` (gitignored) with uv 0.11.32, following the install step of `.github/workflows/ci.yml`'s licenses job: `uv venv --seed`, then CPU torch and torchvision from `https://download.pytorch.org/whl/cpu`, then `-e ".[dev,init]" -e "harness[lpips]"` from the checkout. The CUDA environment installs `torch==2.14.1` and torchvision from PyPI instead, whose Linux wheel is the CUDA 13.0 build: its metadata version is `2.14.1`, and it reports `2.14.1+cu130` at run time (`torch.version.cuda` 13.0). CI installs with pip rather than uv; both take the newest versions the two indexes offer. pip-licenses 5.5.5 ran in all three, and it leaves pip, setuptools and wheel out of its count, which is why the 3.11 environment, seeded with wheel too, counts the same as the 3.14 one.

| Environment | Python | torch | Distributions | Result |
| --- | --- | --- | --- | --- |
| `venv-ci-314` (CI's) | 3.14.8 | 2.14.1+cpu | 63 | Passed (exit 0) |
| `venv-ci-311` (CI's) | 3.11.17 | 2.14.1+cpu | 63 | Passed (exit 0) |
| `venv-cuda-314` (CUDA) | 3.14.8 | 2.14.1 (cu130) | 82 | **Failed (exit 1)**: 16 rows need the owner's sign-off |

In all three, gsplat 1.5.3 (`UNKNOWN`) matched its existing sign-off, and no row is GPL, AGPL or non-commercial. The two CI environments differ only in versions (numpy 2.5.2 and 2.4.6, scipy 1.18.1 and 1.17.1, contourpy, tifffile); the CUDA environment adds cuda-bindings 13.4.4 and cuda-pathfinder 1.8.4 (both Apache-2.0, which pass), triton 3.8.0 (MIT) and the 16 rows below, and resolves newer numpy (2.5.4) and Pillow (13.0.0) than the PyTorch CPU index serves.

**Needing the owner's sign-off** (CUDA environment only; none is signed off yet). None comes from openftgs's own requirements: torch's Linux wheel requires them (its `Requires-Dist`, `platform_system == "Linux"`), the user installs them with torch, and openftgs never redistributes them. The last column is the license notice the wheel itself ships (Libraries bundled in dependency wheels, below).

| Package | Version | Reported license | Why it is installed | The wheel's notice |
| --- | --- | --- | --- | --- |
| cuda-toolkit | 13.0.3.0 | `UNKNOWN` | torch requires `cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3`; a meta-package ("CUDA Toolkit meta-package") with no files besides its metadata, whose extras pull in the eleven wheels marked "cuda-toolkit" below | none |
| nvidia-cublas | 13.1.1.3 | `LicenseRef-NVIDIA-Proprietary` | cuda-toolkit `[cublas]`; also nvidia-cudnn-cu13 and nvidia-cusolver | CUDA Toolkit EULA |
| nvidia-cuda-cupti | 13.0.85 | `Other/Proprietary License` | cuda-toolkit `[cupti]` | CUDA Toolkit EULA |
| nvidia-cuda-nvrtc | 13.0.88 | `Other/Proprietary License` | cuda-toolkit `[nvrtc]` and `[cublas]` | CUDA Toolkit EULA |
| nvidia-cuda-runtime | 13.0.96 | `LicenseRef-NVIDIA-Proprietary` | cuda-toolkit `[cudart]` | CUDA Toolkit EULA |
| nvidia-cufft | 12.0.0.61 | `Other/Proprietary License` | cuda-toolkit `[cufft]` | CUDA Toolkit EULA |
| nvidia-cufile | 1.15.1.6 | `Other/Proprietary License` | cuda-toolkit `[cufile]` | CUDA Toolkit EULA |
| nvidia-curand | 10.4.0.35 | `Other/Proprietary License` | cuda-toolkit `[curand]` | CUDA Toolkit EULA |
| nvidia-cusolver | 12.0.4.66 | `Other/Proprietary License` | cuda-toolkit `[cusolver]` | CUDA Toolkit EULA |
| nvidia-cusparse | 12.6.3.3 | `Other/Proprietary License` | cuda-toolkit `[cusparse]` and `[cusolver]` | CUDA Toolkit EULA |
| nvidia-nvjitlink | 13.4.92 | `LicenseRef-NVIDIA-Proprietary` | cuda-toolkit `[nvjitlink]`, `[cufft]`, `[cusolver]` and `[cusparse]` | CUDA Toolkit EULA |
| nvidia-nvtx | 13.0.85 | `Other/Proprietary License` | cuda-toolkit `[nvtx]` | Apache-2.0 |
| nvidia-cudnn-cu13 | 9.24.0.43 | `LicenseRef-NVIDIA-Proprietary` | torch directly (`==9.24.0.43`) | NVIDIA SDK license agreement |
| nvidia-cusparselt-cu13 | 0.8.1 | `NVIDIA Proprietary Software` | torch directly (`==0.8.1`) | NVIDIA SDK license agreement |
| nvidia-nccl-cu13 | 2.30.7 | `LicenseRef-NVIDIA-Proprietary` | torch directly (`==2.30.7`) | BSD-3-Clause (NVIDIA) |
| nvidia-nvshmem-cu13 | 3.4.5 | `LicenseRef-NVIDIA-Proprietary` | torch directly (`==3.4.5`) | CUDA Toolkit EULA |

Each sign-off becomes an entry of `scripts/license_signoff.json` keyed on the package, the version (exact, or a pattern the owner chooses) and the reported license string above, after which the CUDA scan is rerun here. Sign-offs: awaiting the owner (2026-10-22).

## Libraries bundled in dependency wheels

`scripts/bundled_libs.py` (#366) walks the installed distributions' RECORDs and sorts each one's shared libraries into its own extension modules (CPython's `.cpython-*.so`, `.abi3.so` and `.pyd` names), the libraries bundled into its wheel (`<name>.libs/`, `torch/lib/`, `nvidia/<component>/lib/`, `triton/backends/<backend>/lib/`) and unrecognised layouts; for each distribution with a bundled or unrecognised library it reports the version, the declared license and the license notices the wheel ships. Run on 2026-10-22, from the same checkout, in the same three environments:

```
<env>/bin/python scripts/bundled_libs.py [--json]
```

The notices were then read, each distribution's own LICENSE first (source-log rows T366.2 to T366.9). No library's source was opened. Where a wheel ships no notice for a library it bundles, the library's license is taken from another installed wheel's notice for the same library, and the row says so.

**openftgs's own wheel holds no shared library.** Built through uvx as Q-241 says (`uvx --from build pyproject-build --wheel --outdir .scratch/prd-355-release/wheel-366 .`, build 1.6.1), then `python scripts/bundled_libs.py --wheel .scratch/prd-355-release/wheel-366/openftgs-0.1.0.dev0-py3-none-any.whl` printed `openftgs 0.1.0.dev0: no shared library`.

**Inventory.** Versions are those of `venv-ci-314`, `venv-ci-311` and `venv-cuda-314` where they differ; the layouts are the same in all three, except that the CUDA builds of torch and torchvision bundle more, and triton and the NVIDIA wheels are in the CUDA environment only.

| Distribution (declared license) | Bundled libraries | Their license, and where it is stated | Outcome |
| --- | --- | --- | --- |
| numpy 2.5.2, 2.4.6, 2.5.4 (`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`) | `numpy.libs/`: libscipy_openblas64_ | OpenBLAS, BSD-3-Clause, with LAPACK inside it, BSD-3-Clause-Open-MPI (numpy's LICENSE.txt) | Pass |
| | libgfortran | GCC runtime library, GPL-3.0-or-later WITH GCC-exception-3.1 (numpy's LICENSE.txt) | **Sign-off** (runtime exception) |
| | libquadmath | LGPL-2.1-or-later (numpy's LICENSE.txt) | **Sign-off** (LGPL) |
| scipy 1.18.1, 1.17.1 (`BSD License`) | `scipy.libs/`: libscipy_openblas | OpenBLAS and LAPACK, as numpy's (scipy's LICENSE.txt) | Pass |
| | libgfortran (two copies) | GPL-3.0-or-later WITH GCC-exception-3.1 (scipy's LICENSE.txt) | **Sign-off** (runtime exception) |
| | libquadmath (two copies) | LGPL-2.1-or-later (scipy's LICENSE.txt) | **Sign-off** (LGPL) |
| opencv-python-headless 5.0.0.93 (`Apache Software License`; the packaging's LICENSE.txt is MIT) | `opencv_python_headless.libs/`: libavcodec, libavformat, libavutil, libswresample, libswscale | FFmpeg, LGPL-2.1 (LICENSE-3RD-PARTY.txt) | **Sign-off** (LGPL) |
| | libgfortran | No section in opencv's notice; GPL-3.0-or-later WITH GCC-exception-3.1 by numpy's notice for the same library | **Sign-off** (runtime exception) |
| | libquadmath | No section in opencv's notice; LGPL-2.1-or-later by numpy's notice | **Sign-off** (LGPL) |
| | libopenblasp | No section in opencv's notice; OpenBLAS, BSD-3-Clause, by numpy's notice | Pass |
| | libvpx, libaom, libavif, libpng16 | BSD-3-Clause, BSD-2-Clause, BSD-2-Clause and the libpng license (LICENSE-3RD-PARTY.txt) | Pass |
| | libdrm | No notice in any installed wheel | **Owner's call** (unknown) |
| pillow 12.3.0, 13.0.0 (`MIT-CMU`) | `pillow.libs/`: libavif, libbrotlicommon, libbrotlidec, libfreetype, libharfbuzz, libjpeg, liblcms2, liblzma, libopenjp2, libpng16, libsharpyuv, libtiff, libwebp, libwebpdemux, libwebpmux, libXau, libxcb, libzstd | Each has a section in Pillow's LICENSE (libsharpyuv under libwebp's), all permissive: MIT or MIT-style (brotli, HarfBuzz's "Old MIT", Little CMS, libXau, libxcb), BSD (libavif, libwebp, OpenJPEG, zstd), the IJG terms (libjpeg), the libpng license, the libtiff license; FreeType is offered under the FreeType License or GPL-2.0, and the FreeType License is taken; liblzma is in the public domain | Pass |
| torch 2.14.1+cpu, 2.14.1 (`Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`) | `torch/lib/`: libtorch, libtorch_cpu, libtorch_python, libtorch_global_deps, libc10, libshm, and test libraries (CPU build); plus libtorch_cuda, libtorch_cuda_linalg, libtorch_nvshmem, libc10_cuda, libcaffe2_nvrtc (CUDA build) | torch's own builds, under torch's LICENSE | Pass |
| | libgomp.so.1 | No notice in torch's wheel. GCC's OpenMP runtime, by its `GOMP_1.0` to `GOMP_5.0` symbol versions and its "gcc 8.5.0" build note; GCC licenses it GPL-3.0-or-later WITH GCC-exception-3.1, as numpy's notice states for its sibling libgfortran | **Sign-off** (runtime exception) |
| torchvision 0.29.1+cpu, 0.29.1 (`BSD`) | `torchvision.libs/`: libjpeg, libpng16, libsharpyuv, libwebp, libz | torchvision's wheel ships only its own BSD-3-Clause LICENSE; libjpeg-turbo, libpng, libwebp (with libsharpyuv) and zlib are permissive by Pillow's sections for the same libraries | Pass |
| | libcudart, libnvjpeg (CUDA build only) | No notice in torchvision's wheel. NVIDIA's CUDA runtime and nvJPEG: both are listed in Attachment A of the CUDA Toolkit EULA (the notice of nvidia-cuda-runtime) as distributable portions | **Owner's call** (proprietary, inside a BSD wheel) |
| triton 3.8.0 (`MIT`; CUDA environment) | `triton/backends/nvidia/lib/cupti/` and `cupti-blackwell/`: libcupti, libnvperf_host, libnvperf_target, libpcsamplingutil, libcheckpoint | No notice in triton's wheel, whose only notice is its MIT LICENSE. NVIDIA's CUPTI: listed in Attachment A of the CUDA Toolkit EULA as distributable | **Owner's call** (proprietary, inside an MIT wheel) |
| The 15 NVIDIA runtime wheels (CUDA environment) | `nvidia/cu13/lib/`, `nvidia/cudnn/lib/`, `nvidia/cusparselt/lib/`, `nvidia/nccl/lib/`, `nvidia/nvshmem/lib/`: the wheels' own libraries | Their own notices (License scan, above) | Covered by the license scan's sign-offs |

**Unrecognised layouts**, resolved by hand: `torchvision/_C_stable.so` and `torchvision/image_stable.so` are torchvision's own operator libraries, and `triton/_C/libtriton.so`, `triton/_C/libproton.so`, `triton/plugins/libMLIRDialectPlugin.so`, `triton/plugins/libTritonPluginsTestLib.so` and `triton/instrumentation/libGPUInstrumentationTestLib.so` are triton's own, under each project's license; the classifier leaves them unrecognised because they don't carry CPython's extension suffixes. Outside the shared-library inventory, triton also ships NVIDIA's executables `ptxas`, `ptxas-blackwell`, `cuobjdump` and `nvdisasm` under `triton/backends/nvidia/bin/`.

**Needing the owner's sign-off** (LGPL, and GPL with a runtime exception; the ticket's rule). openftgs bundles none of them: they arrive with the user's install of numpy, scipy, opencv-python-headless (`[dev]` only, the test oracle) and torch.

| Library | License | Bundled by (versions) | Environments |
| --- | --- | --- | --- |
| libquadmath | LGPL-2.1-or-later | numpy (2.5.2, 2.4.6, 2.5.4), scipy (1.18.1, 1.17.1), opencv-python-headless 5.0.0.93 | all three |
| FFmpeg (libavcodec, libavformat, libavutil, libswresample, libswscale) | LGPL-2.1 | opencv-python-headless 5.0.0.93 | all three |
| libgfortran | GPL-3.0-or-later WITH GCC-exception-3.1 | numpy, scipy, opencv-python-headless (versions as above) | all three |
| libgomp | GPL-3.0-or-later WITH GCC-exception-3.1 | torch 2.14.1+cpu and 2.14.1 | all three |

**For the owner's decision** (not in the ticket's sign-off rule, but proprietary or unknown, which Q-4 sends to a sign-off when it is a distribution's declared license): NVIDIA's libcudart and libnvjpeg inside torchvision 0.29.1's CUDA wheel; NVIDIA's CUPTI inside triton 3.8.0; libdrm inside opencv-python-headless 5.0.0.93, for which no installed wheel ships a notice.

Sign-offs: awaiting the owner (2026-10-22).

## The vendored RoMa

To be written by #367.

## Pretrained weights' terms

To be written by #367.

## No dataset material

To be written by #367.

## Source-log audit

To be written by #368.

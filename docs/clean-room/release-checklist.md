# Release checklist

M18's record of the release-time checks that spec §2.3's notes and D-79 call for, for openftgs 0.1.0 (PRD #355): the license scan and its sign-offs, the libraries bundled in dependency wheels, the vendored license files, the pretrained weights' terms, the absence of dataset material, and the source-log audit. Each section gives the commands run, the commit and environment they ran on, and their results; a sign-off is the owner's, given in the build session, and is recorded with its date.

## License scan

`scripts/license_scan.py` (Q-4) run with each environment's interpreter, from a checkout of `ticket-366-license-scan` at 0cf7653 (the scan script and `scripts/license_signoff.json` as on `prd-355-release` at 7e1faf0), on 2026-10-22:

```
<env>/bin/python scripts/license_scan.py
```

The environments were built fresh under `.scratch/prd-355-release/` (gitignored) with uv 0.11.32, following the install step of `.github/workflows/ci.yml`'s licenses job: `uv venv --seed`, then CPU torch and torchvision from `https://download.pytorch.org/whl/cpu`, then `-e ".[dev,init]" -e "harness[lpips]"` from the checkout. The CUDA environment installs `torch==2.14.1` and torchvision from PyPI instead, whose Linux wheel is the CUDA 13.0 build: its metadata version is `2.14.1`, and it reports `2.14.1+cu130` at run time (`torch.version.cuda` 13.0). CI installs with pip rather than uv; both take the newest versions the two indexes offer. pip-licenses 5.5.5 ran in all three. It leaves out system packages, itself and its prettytable and wcwidth dependencies; pip, setuptools and wheel appear in none of its rows, which is why the 3.11 environment, seeded with wheel too, counts the same as the 3.14 one.

| Environment | Python | torch | Distributions | Result |
| --- | --- | --- | --- | --- |
| `venv-ci-314` (CI's) | 3.14.8 | 2.14.1+cpu | 63 | Passed (exit 0) |
| `venv-ci-311` (CI's) | 3.11.17 | 2.14.1+cpu | 63 | Passed (exit 0) |
| `venv-cuda-314` (CUDA) | 3.14.8 | 2.14.1 (cu130) | 82 | Failed (exit 1): 16 rows needed the owner's sign-off; passed after it (below) |

In all three, gsplat 1.5.3 (`UNKNOWN`) matched its existing sign-off, and no row is GPL, AGPL or non-commercial. The two CI environments differ only in versions (numpy 2.5.2 and 2.4.6, scipy 1.18.1 and 1.17.1, contourpy, tifffile); the CUDA environment adds cuda-bindings 13.4.4 and cuda-pathfinder 1.8.4 (both Apache-2.0, which pass), triton 3.8.0 (MIT) and the 16 rows below, and resolves newer numpy (2.5.4) and Pillow (13.0.0) than the PyTorch CPU index serves.

**Needing the owner's sign-off** (CUDA environment only; signed off on 2026-10-23, below). None comes from openftgs's own requirements: torch's Linux wheel requires them (its `Requires-Dist`, `platform_system == "Linux"`), the user installs them with torch, and openftgs never redistributes them. The last column is the license notice the wheel itself ships (Libraries bundled in dependency wheels, below).

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

**Sign-off, 2026-10-23.** The owner, in the build session, on the 16 rows above: "Yes." Each is an entry of `scripts/license_signoff.json` keyed on the package, its exact version and the reported license string above, with the wheel's notice as `actual`, the reason "Installed by torch's Linux CUDA 13.0 wheel, not by openftgs; installed by the user and never redistributed (Q-4)", source-log row 331 as its source, and the date 2026-10-23. A change of version or license string voids it.

**Rerun after the sign-offs**, on 2026-10-23, from `ticket-366-license-scan` with `prd-355-release` at 908cdf1 merged in and the 16 entries added, in the same environments (whose packages are unchanged: the merge changed no dependency, and the licenses job still installs `.[dev,init]` and `harness[lpips]`); the system's Python 3.14 is now 3.14.9, which the 3.14 environments run on:

| Environment | Python | Distributions | Result |
| --- | --- | --- | --- |
| `venv-ci-314` | 3.14.9 | 63 | Passed (exit 0); gsplat signed off; the 16 new entries reported as stale, a warning only, since CI's CPU torch installs no NVIDIA wheel |
| `venv-ci-311` | 3.11.17 | 63 | Passed (exit 0); as `venv-ci-314` |
| `venv-cuda-314` | 3.14.9 | 82 | **Passed (exit 0)**: gsplat and the 16 rows signed off, nothing stale (Q-4's detecting test) |

**Rerun on the merged tree (#432), 2026-10-27.** `main` at 0efc6b2 (PRD #380 and its follow-ups) merged into `ticket-432-after-380` (9dc5ce30), the scan run from that branch at 747d6e7. The environments were built again from it, as above, in new folders: `uv venv --seed`, CPU torch and torchvision from the CPU index (torch 2.14.1 and torchvision from PyPI for the CUDA one), then `-e ".[dev,init]" -e "harness[torch]"`, the licenses job's install since Q-254 and Q-260 (`harness[lpips]` is gone). Revision 549o's §2.3 and #393 removed lpips from the core, torchmetrics from `[dev]` and tqdm with lpips, and set the scan's expected sets: against the environments above, each loses lpips 0.1.4, torchmetrics, its lightning-utilities and tqdm, and gains nothing. The other changes are newer releases of six permissive or documentation-only packages (fonttools, hypothesis, pytest, ruff, wadler-lindig, wcwidth); no signed-off package changed version or license string, so no sign-off is voided (Q-4).

| Environment | Python | torch | Distributions | Result |
| --- | --- | --- | --- | --- |
| `venv-ci-314-432` (CI's) | 3.14.9 | 2.14.1+cpu | 59 | Passed (exit 0); gsplat signed off; the 16 CUDA entries reported as stale, a warning only |
| `venv-ci-311-432` (CI's) | 3.11.17 | 2.14.1+cpu | 59 | Passed (exit 0); as `venv-ci-314-432` |
| `venv-cuda-314-432` (CUDA) | 3.14.9 | 2.14.1 (cu130) | 78 | **Passed (exit 0)**: gsplat and the 16 rows signed off, nothing stale |

The code openftgs now carries from other projects is covered by NOTICE, not by the scan, which sees installed distributions only: the vendored ꟻLIP loss (`openftgs._vendor.flip`, BSD-3-Clause, with its license file and VENDOR.json; D-65, D-89) and the HaarPSI adapted from piq 0.8.0 in `openftgs.losses` (Apache-2.0, attribution in the code; §2.3). Both are named in NOTICE, checked under "The vendored ꟻLIP" below.

## Libraries bundled in dependency wheels

`scripts/bundled_libs.py` (#366) walks the installed distributions' RECORDs and sorts each one's shared libraries into its own extension modules (CPython's `.cpython-*.so`, `.abi3.so` and `.pyd` names), the libraries bundled into its wheel (`<name>.libs/`, `torch/lib/`, `nvidia/<component>/lib/`, `triton/backends/<backend>/lib/`) and unrecognised layouts; for each distribution with a bundled or unrecognised library it reports the version, the declared license and the license notices the wheel ships. Run on 2026-10-22, from the same checkout, in the same three environments:

```
<env>/bin/python scripts/bundled_libs.py [--json]
```

The notices were then read, each distribution's own LICENSE first (source-log rows 324 to 331). No library's source was opened. Where a wheel ships no notice for a library it bundles, the library's license is taken from another installed wheel's notice for the same library, and the row says so.

**openftgs's own wheel holds no shared library.** Built through uvx as Q-241 says (`uvx --from build pyproject-build --wheel --outdir .scratch/prd-355-release/wheel-366 .`, build 1.6.1), then `python scripts/bundled_libs.py --wheel .scratch/prd-355-release/wheel-366/openftgs-0.1.0.dev0-py3-none-any.whl` printed `openftgs 0.1.0.dev0: no shared library`.

**Inventory.** Versions are those of `venv-ci-314`, `venv-ci-311` and `venv-cuda-314` where they differ; the layouts are the same in all three, except that the CUDA builds of torch and torchvision bundle more, and triton and the NVIDIA wheels are in the CUDA environment only.

| Distribution (declared license) | Bundled libraries | Their license, and where it is stated | Outcome |
| --- | --- | --- | --- |
| numpy 2.5.2, 2.4.6, 2.5.4 (`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`) | `numpy.libs/`: libscipy_openblas64_ | OpenBLAS, BSD-3-Clause, with LAPACK inside it, BSD-3-Clause-Open-MPI (numpy's LICENSE.txt) | Pass |
| | libgfortran | GCC runtime library, GPL-3.0-or-later WITH GCC-exception-3.1 (numpy's LICENSE.txt) | **Signed off** (runtime exception; 2026-10-23) |
| | libquadmath | LGPL-2.1-or-later (numpy's LICENSE.txt) | **Signed off** (LGPL; 2026-10-23) |
| scipy 1.18.1, 1.17.1 (`BSD License`) | `scipy.libs/`: libscipy_openblas | OpenBLAS and LAPACK, as numpy's (scipy's LICENSE.txt) | Pass |
| | libgfortran (two copies) | GPL-3.0-or-later WITH GCC-exception-3.1 (scipy's LICENSE.txt) | **Signed off** (runtime exception; 2026-10-23) |
| | libquadmath (two copies) | LGPL-2.1-or-later (scipy's LICENSE.txt) | **Signed off** (LGPL; 2026-10-23) |
| opencv-python-headless 5.0.0.93 (`Apache Software License`; the packaging's LICENSE.txt is MIT) | `opencv_python_headless.libs/`: libavcodec, libavformat, libavutil, libswresample, libswscale | FFmpeg, LGPL-2.1 (LICENSE-3RD-PARTY.txt) | **Signed off** (LGPL; 2026-10-23) |
| | libgfortran | No section in opencv's notice; GPL-3.0-or-later WITH GCC-exception-3.1 by numpy's notice for the same library | **Signed off** (runtime exception; 2026-10-23) |
| | libquadmath | No section in opencv's notice; LGPL-2.1-or-later by numpy's notice | **Signed off** (LGPL; 2026-10-23) |
| | libopenblasp | No section in opencv's notice; OpenBLAS, BSD-3-Clause, by numpy's notice | Pass |
| | libvpx, libaom, libavif, libpng16 | BSD-3-Clause, BSD-2-Clause, BSD-2-Clause and the libpng license (LICENSE-3RD-PARTY.txt) | Pass |
| | libdrm | No notice in any installed wheel; libdrm's own release 2.4.134 ships no license file, and its README states no license (source-log row 333). Debian's copyright file for libdrm: Expat for `Files: *`, the core library bundled here ("Debian copyright file", the owner's decision of 2026-10-27; source-log row T432.1) | Pass (Expat; 2026-10-27, below) |
| pillow 12.3.0, 13.0.0 (`MIT-CMU`) | `pillow.libs/`: libavif, libbrotlicommon, libbrotlidec, libfreetype, libharfbuzz, libjpeg, liblcms2, liblzma, libopenjp2, libpng16, libsharpyuv, libtiff, libwebp, libwebpdemux, libwebpmux, libXau, libxcb, libzstd | Each has a section in Pillow's LICENSE (libsharpyuv under libwebp's), all permissive: MIT or MIT-style (brotli, HarfBuzz's "Old MIT", Little CMS, libXau, libxcb), BSD (libavif, libwebp, OpenJPEG, zstd), the IJG terms (libjpeg), the libpng license, the libtiff license; FreeType is offered under the FreeType License or GPL-2.0, and the FreeType License is taken; liblzma is in the public domain | Pass |
| torch 2.14.1+cpu, 2.14.1 (`Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`) | `torch/lib/`: libtorch, libtorch_cpu, libtorch_python, libtorch_global_deps, libc10, libshm, and test libraries (CPU build); plus libtorch_cuda, libtorch_cuda_linalg, libtorch_nvshmem, libc10_cuda, libcaffe2_nvrtc (CUDA build) | torch's own builds, under torch's LICENSE | Pass |
| | libgomp.so.1 | No notice in torch's wheel. GCC's OpenMP runtime, by its `GOMP_1.0` to `GOMP_5.0` symbol versions and its "gcc 8.5.0" build note; GCC licenses it GPL-3.0-or-later WITH GCC-exception-3.1, as numpy's notice states for its sibling libgfortran | **Signed off** (runtime exception; 2026-10-23) |
| torchvision 0.29.1+cpu, 0.29.1 (`BSD`) | `torchvision.libs/`: libjpeg, libpng16, libsharpyuv, libwebp, libz | torchvision's wheel ships only its own BSD-3-Clause LICENSE; libjpeg-turbo, libpng, libwebp (with libsharpyuv) and zlib are permissive by Pillow's sections for the same libraries | Pass |
| | libcudart, libnvjpeg (CUDA build only) | No notice in torchvision's wheel. NVIDIA's CUDA runtime and nvJPEG: both are listed in Attachment A of the CUDA Toolkit EULA (the notice of nvidia-cuda-runtime) as distributable portions | **Signed off** (proprietary, inside a BSD wheel; 2026-10-23) |
| triton 3.8.0 (`MIT`; CUDA environment) | `triton/backends/nvidia/lib/cupti/` and `cupti-blackwell/`: libcupti, libnvperf_host, libnvperf_target, libpcsamplingutil, libcheckpoint | No notice in triton's wheel, whose only notice is its MIT LICENSE. NVIDIA's CUPTI: listed in Attachment A of the CUDA Toolkit EULA as distributable | **Signed off** (proprietary, inside an MIT wheel; 2026-10-23) |
| The 15 NVIDIA runtime wheels (CUDA environment) | `nvidia/cu13/lib/`, `nvidia/cudnn/lib/`, `nvidia/cusparselt/lib/`, `nvidia/nccl/lib/`, `nvidia/nvshmem/lib/`: the wheels' own libraries | Their own notices (License scan, above) | Covered by the license scan's sign-offs |

**Unrecognised layouts**, resolved by hand: `torchvision/_C_stable.so` and `torchvision/image_stable.so` are torchvision's own operator libraries, and `triton/_C/libtriton.so`, `triton/_C/libproton.so`, `triton/plugins/libMLIRDialectPlugin.so`, `triton/plugins/libTritonPluginsTestLib.so` and `triton/instrumentation/libGPUInstrumentationTestLib.so` are triton's own, under each project's license; the classifier leaves them unrecognised because they don't carry CPython's extension suffixes. Outside the shared-library inventory, triton also ships NVIDIA's executables `ptxas`, `ptxas-blackwell`, `cuobjdump` and `nvdisasm` under `triton/backends/nvidia/bin/`.

**Rerun on the merged tree (#432), 2026-10-27.** `scripts/bundled_libs.py` in the three environments of the license scan's rerun (`venv-ci-314-432`, `venv-ci-311-432`, `venv-cuda-314-432`), from 747d6e7: the same distributions bundle the same libraries at the same versions as in the inventory above (numpy, scipy, opencv-python-headless, pillow, torch, torchvision, triton and the NVIDIA wheels); lpips, torchmetrics and tqdm, now gone, bundled none. So every sign-off of this section stands, and libdrm is settled below. openftgs's own wheel, built from 747d6e7 (The sdist and the wheel, below), still holds no shared library: `python scripts/bundled_libs.py --wheel <s>/t432-build/openftgs-0.1.0.dev0-py3-none-any.whl` printed `openftgs 0.1.0.dev0: no shared library`; the vendored ꟻLIP is Python.

**Signed off by the owner** (LGPL, and GPL with a runtime exception; the ticket's rule). openftgs bundles none of them: they arrive with the user's install of numpy, scipy, opencv-python-headless (`[dev]` only, the test oracle) and torch.

| Library | License | Bundled by (versions) | Environments |
| --- | --- | --- | --- |
| libquadmath | LGPL-2.1-or-later | numpy (2.5.2, 2.4.6, 2.5.4), scipy (1.18.1, 1.17.1), opencv-python-headless 5.0.0.93 | all three |
| FFmpeg (libavcodec, libavformat, libavutil, libswresample, libswscale) | LGPL-2.1 | opencv-python-headless 5.0.0.93 | all three |
| libgfortran | GPL-3.0-or-later WITH GCC-exception-3.1 | numpy, scipy, opencv-python-headless (versions as above) | all three |
| libgomp | GPL-3.0-or-later WITH GCC-exception-3.1 | torch 2.14.1+cpu and 2.14.1 | all three |

**Put to the owner as well** (not in the ticket's sign-off rule, but proprietary or unknown, which Q-4 sends to a sign-off when it is a distribution's declared license): NVIDIA's libcudart and libnvjpeg inside torchvision 0.29.1's CUDA wheel; NVIDIA's CUPTI inside triton 3.8.0; libdrm inside opencv-python-headless 5.0.0.93, for which no installed wheel ships a notice.

**Sign-off, 2026-10-23.** The owner, in the build session, on the four libraries of the table above: "Yes." On the three items put to them: "(a) for the NVIDIA ones and (c) for libdrm", that is, NVIDIA's libcudart and libnvjpeg in torchvision's CUDA wheel and NVIDIA's CUPTI in triton are signed off like the proprietary rows of the license scan, and libdrm's license is to be read on its own project first, passing with no sign-off if it is permissive.

**libdrm, 2026-10-23: open** (settled on 2026-10-27, below). Its project's GitLab (`gitlab.freedesktop.org/mesa/drm`) refused the automated request, so its release tarball `https://dri.freedesktop.org/libdrm/libdrm-2.4.134.tar.xz`, the newest, was downloaded into the session's scratchpad and its file names listed: it has no LICENSE, COPYING or other license file. Its `README.rst`, the only top-level documentation, states no license. Nothing else of it was opened, since its license is stated only inside its source files and build files, which the clean-room rules don't open for a project whose license isn't established (source-log row 333). The owner decides how it is established.

**libdrm, 2026-10-27: passes (Expat).** The owner's decision, in the build session: settle it from Debian's machine-readable copyright file for the libdrm package ("Debian copyright file", 2026-10-27), distribution documentation, not libdrm's source. Only that file was fetched, `https://metadata.ftp-master.debian.org/changelogs//main/libd/libdrm/unstable_copyright` (source-log row T432.1). In copyright format 1.0 (DEP-5), with `Upstream-Name: libdrm` and the header's `License: Expat`, it gives `Files: *` and every per-driver stanza (amdgpu, android, etnaviv, exynos, freedreno, include, intel, nouveau, omap, radeon, tegra, tests, vc4) the Expat (MIT) license; one stanza, `Files: etnaviv/etnaviv_drm.h`, a header of the etnaviv driver library, is GPL-2. The library opencv-python-headless bundles is the core `libdrm.so.2` (`opencv_python_headless.libs/libdrm-b0291a67.so.2.4.0`), whose files fall under `Files: *`, not one of the per-driver libraries such as `libdrm_etnaviv`. Under #366's rule, a permissive license passes with no sign-off: libdrm's row passes. The GPL-2 header is reported to the owner with this result, since the rule's "permissive" is read here for the core library only.

## The vendored RoMa

D-65; §2.3 notes ("checks the vendored directory's license files"); Q-113, Q-115 and Q-118, whose detecting test is this audit. Run on 2026-10-22 by ticket #367, on commit 7e1faf0 of `prd-355-release`, with the project's `.venv` (Python 3.14.8) and git 2.43.0.

**The upstream clone.** A blobless clone of RoMa (MIT), `git clone --filter=blob:none --no-checkout https://github.com/Parskatt/RoMa <clone>`, into a new, empty folder, `<clone>` = `.scratch/t367-roma/RoMa` inside the ticket's worktree (gitignored; the session's tooling refused git commands on the main checkout's `.scratch/`). Its HEAD is still 77f8d68803526dcddfd9b7a46bc76125bdc25f15 (2026-01-23). Its `LICENSE` was read before anything else: MIT, Copyright (c) 2023 Johan Edstedt, git blob ca95157, the blob of source log row 106. Source log row 278.

**The vendoring script's check mode.** Run from the worktree, not from the clone, with the clone's path as an argument:

```
.venv/bin/python -I scripts/vendor_roma.py --source <clone> --check
```

It rebuilds the copy in a temporary folder from the 22 inference files at 77f8d68 (each blob id checked against the commit's tree) and the eight license texts, fetched again at their pinned commits and checked against their SHA-256 (source log row 279), and compares every file with `src/openftgs/_vendor/romatch`. Result, exit 0:

```
22 files (3798 upstream lines), 10 changed; romatch imports rewritten: 12 at module level, 2 in RegressionMatcher.visualize_warp, TinyRoMa.visualize_warp; 8 license files
check: the copy and VENDOR.json match the rebuild
```

So the vendored copy, its eight license files and `VENDOR.json` are identical, byte for byte, to a fresh rebuild: Q-113's cut of the pose helpers and Q-115's modification notices included, and InLoc's text extracted from its README as Q-118 says (SHA-256 b35cbfd4… of the README, 95970e9a… of the text).

**NOTICE against VENDOR.json.** For each entry of `VENDOR.json`'s `licenses`, the file's SHA-256 was recomputed and NOTICE was searched for the project's name and for the file's path in parentheses:

```python
# .venv/bin/python -I, from the repository root
import hashlib, json, pathlib
root = pathlib.Path("src/openftgs/_vendor/romatch")
record = json.loads((root / "VENDOR.json").read_text())
notice = pathlib.Path("NOTICE").read_text()
for e in record["licenses"]:
    sha = hashlib.sha256((root / e["path"]).read_bytes()).hexdigest()
    named = f"({e['path']})" in notice and e["project"] in notice
    print(e["path"], e["project"], e["license"], named, sha == e["sha256"])
print(record["commit"] in notice)
```

| License file | Project | License | Named in NOTICE | SHA-256 as recorded |
| --- | --- | --- | --- | --- |
| `LICENSE` | RoMa | MIT | yes | yes |
| `LICENSES/DINOv2.txt` | DINOv2 | Apache-2.0 | yes | yes |
| `LICENSES/DINO.txt` | DINO | Apache-2.0 | yes | yes |
| `LICENSES/timm.txt` | timm | Apache-2.0 | yes | yes |
| `LICENSES/LoFTR.txt` | LoFTR | Apache-2.0 | yes | yes |
| `LICENSES/Patch2Pix.txt` | Patch2Pix | MIT | yes | yes |
| `LICENSES/Kornia.txt` | Kornia | Apache-2.0 | yes | yes |
| `LICENSES/InLoc.txt` | InLoc | MIT | yes | yes |

NOTICE also names RoMa's commit, 77f8d68803526dcddfd9b7a46bc76125bdc25f15, as `VENDOR.json` does. NOTICE names every license text that `VENDOR.json` records, and no other.

**The license files ship.** The wheel built for "No dataset material" below holds `openftgs/_vendor/romatch/LICENSE`, the seven files of `openftgs/_vendor/romatch/LICENSES/` and `openftgs/_vendor/romatch/VENDOR.json` (pyproject's package data), and `LICENSE` and `NOTICE` under `openftgs-0.1.0.dev0.dist-info/licenses/`.

**U-38**, CPU part, on the same commit: `tests/test_vendored_roma.py` and `tests/test_skeleton.py` with `-m "not gpu and not weights"`, 56 passed, 1 deselected (U-38's GPU comparison with upstream romatch, `test_u38_vendored_match_agrees_with_upstream_romatch`, which this ticket's CPU budget doesn't run).

## The vendored ꟻLIP

D-65, D-89; §2.3's ꟻLIP row (BSD-3-Clause, with its header and license file); Q-252 (the reference copy beside the tests) and Q-254 (the harness's own copy). Added by #432 on 2026-10-27, on `ticket-432-after-380` at 747d6e7 (`main` at 0efc6b2 merged in), with the project's `.venv` (Python 3.14.9). The check is RoMa's, above, with the reference copy in place of an upstream clone: `tests/reference/flip` holds NVIDIA's `src/pytorch/flip_loss.py` and `LICENSE` at commit d3ab02a, unchanged, SHA-256 bf2b83ae… and 13b95507…, the values of source-log rows 425 and 444, recomputed for this check.

**The vendoring script's check mode**, for openftgs's copy and for the harness's:

```
.venv/bin/python -I scripts/vendor_flip.py --check
.venv/bin/python -I scripts/vendor_flip.py --target harness --check
```

Each rebuilds its copy in a temporary folder from the reference copy and compares every file with the vendored one. Results, exit 0:

```
flip_loss.py: 716 -> 770 lines, 25 edits, sha256 789799598c4d18bc
check: the copy and VENDOR.json match the rebuild
flip_loss.py: 716 -> 720 lines, 18 edits, sha256 4881e64009e43496
check: the copy and VENDOR.json match the rebuild
```

So `src/openftgs/_vendor/flip` (`flip_loss.py` with its marked edits, `LICENSE` and `VENDOR.json`) and `harness/src/ftgs_harness/_vendor/flip` are identical, byte for byte, to fresh rebuilds; the packages' own `__init__.py` files are theirs.

**NOTICE against VENDOR.json**, with the script of "The vendored RoMa" run over the three `VENDOR.json` files (RoMa's, ꟻLIP's and the harness's ꟻLIP), each file's SHA-256 recomputed:

| Copy | License file | Project | License | Named in NOTICE | SHA-256 as recorded |
| --- | --- | --- | --- | --- | --- |
| `openftgs/_vendor/flip` | `LICENSE` | ꟻLIP | BSD-3-Clause | yes | yes |
| `ftgs_harness/_vendor/flip` | `LICENSE` | ꟻLIP | BSD-3-Clause | yes | yes |

NOTICE names ꟻLIP's repository and its commit, d3ab02a4425eb167d8a4021b382a2bafc6e6c838, as both `VENDOR.json` files do, and `flip_loss.py` matches its recorded SHA-256 in both copies. NOTICE also names piq 0.8.0 for the HaarPSI that `openftgs.losses` adapts (Apache-2.0; `_haarpsi_constants`, `_similarity`, `_haarpsi`), and the trunk networks' weights (below). RoMa's eight license files still match, as above.

**The license files ship.** The wheel built for "No dataset material" below holds `openftgs/_vendor/flip/LICENSE`, `VENDOR.json`, `flip_loss.py` and `__init__.py`. The harness, a separate distribution this release doesn't publish (Q-5), carries its own copy.

## Pretrained weights' terms

§2.3 notes: pretrained weights are downloaded at run time and never redistributed, and their terms are recorded here, apart from the code licenses; XFeat's license is checked before `RomaInitConfig.model = "tiny_roma_v1_outdoor"` is used. Q-7 (CI never downloads weights: `weights` tests run only with `OPENFTGS_ALLOW_DOWNLOADS=1`), Q-116, Q-123 and Q-198 (the files `roma_init` loads). Recorded on 2026-10-22 by ticket #367 from each project's own statement of its terms, every read logged (source log rows 278 and 280 to 285), and updated by #432 on 2026-10-27 for revision 549o: LPIPS's linear layers and torchvision's AlexNet and VGG16 ImageNet trunks left with lpips (D-9, #393), and the six trunk perceptual networks' weights came in, their terms from the publishers' statements the research logged (source-log rows 340, 348, 367, 369, 419, 421 and 423; URLs, sizes and SHA-256 in rows 424 and 438 to 443; D-89's weights column). ꟻLIP, the default perceptual network and the perceptual metric, and HaarPSI have no weights. The files openftgs fetches are those its code names: `openftgs._weights`'s six trunk files, for `openftgs.losses.PerceptualLoss` (Q-258), and the vendored `model_zoo.weight_urls` through `openftgs.init`'s `_weight_urls` (Q-198), which never asks for `roma_indoor.pth`.

| Weight | Where it comes from | When it is fetched | Terms, from the project's own statement | Redistributed by openftgs |
| --- | --- | --- | --- | --- |
| Oxford's VGG16, `VGG_ILSVRC_16_layers.caffemodel` (553,432,081 bytes, SHA-256 a6196bc4…, row 424) | http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel, the Visual Geometry Group's own host (row 419) | downloaded by `openftgs._weights` into torch.hub's checkpoint cache, when not cached, the first time `PerceptualLoss("vgg16_oxford")` is built (training with `loss.perc_net = "vgg16_oxford"`), checked against its byte count and SHA-256 before every use, and its convolutions converted once into `openftgs-vgg16_oxford.safetensors` beside it (Q-258) | the VGG model page: the models are released under the Creative Commons Attribution License, CC BY 4.0, with attribution by citing the technical report (Simonyan and Zisserman, arXiv:1409.1556) (rows 340, 419). NOTICE gives the attribution | no |
| BVLC's AlexNet, `bvlc_alexnet.caffemodel` (243,862,414 bytes, SHA-256 4bff209a…, row 424) | http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel, the URL of Caffe's model readme (row 421) | as VGG16's, for `"alexnet_bvlc"`, converted into `openftgs-alexnet_bvlc.safetensors` | Caffe's model readme for the BVLC AlexNet: released for unrestricted use (`license: unrestricted`), no named license (rows 348, 421); Caffe's repository is BSD-2-Clause | no |
| DINOv2 ViT-S/14 and ViT-B/14, `dinov2_vits14_pretrain.pth` and `dinov2_vitb14_pretrain.pth` (88,283,115 and 346,378,731 bytes, SHA-256 b938bf1b… and 0b8b82f8…, row 424) | https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth and …/dinov2_vitb14/dinov2_vitb14_pretrain.pth, the URLs DINOv2's README gives (rows 423, 440) | downloaded by `openftgs._weights`, as above, the first time `"dinov2_s14"` or `"dinov2_b14"` is built (they need `[init]`, Q-248), checked before every use, loaded as a state dict | DINOv2's README: "DINOv2 code and model weights are released under the Apache License 2.0" (row 367); the XRay-DINO and Cell-DINO weights, under the FAIR Noncommercial Research License, are never fetched | no |
| DINO ViT-S/16 and ViT-B/16, `dino_deitsmall16_pretrain.pth` and `dino_vitbase16_pretrain.pth` (86,710,517 and 343,242,485 bytes, SHA-256 1566d504… and bf34ad0f…, row 424) | https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth and …/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth (rows 423, 440) | as DINOv2's, for `"dino_s16"` and `"dino_b16"` | DINO's README puts "this repository" under Apache 2.0 and doesn't name the pretrained weights, which it hosts and links (row 369); D-89 records them as "Meta, the repository's Apache-2.0", the owner's decision on #373 (row 424) | no |
| RoMa outdoor, `roma_outdoor.pth` (445,647,516 bytes, SHA-256 c7a45c80…, source log row 132) | https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth, a release of RoMa's author's storage repository | downloaded by `roma_init`'s default matcher (`ftgs init roma`, model `roma_outdoor`) through `torch.hub.load_state_dict_from_url` into torch.hub's checkpoint cache, when not cached | RoMa's README (77f8d68), "License": all its code except DINOv2 is MIT, DINOv2 Apache-2.0; it states nothing about the weights. The repository that hosts them, Parskatt/storage at 6fed440, has an MIT `LICENSE`, Copyright (c) 2022 Johan Edstedt, and no README | no |
| DINOv2 ViT-L/14, `dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes, SHA-256 d5383ea8…, source log row 132) | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth, the URL DINOv2's README gives for the ViT-L/14 backbone | with RoMa outdoor, as RoMa's backbone | DINOv2's README (7764ea0), "License": "DINOv2 code and model weights are released under the Apache License 2.0"; its `LICENSE` is the Apache-2.0 text vendored as `LICENSES/DINOv2.txt`. The same README puts the XRay-DINO and Cell-DINO weights under the FAIR Noncommercial Research License; openftgs fetches neither | no |
| Tiny RoMa v1 outdoor, `tiny_roma_v1_outdoor.pth` (11,432,107 bytes, SHA-256 3e2ac348…, source log row 135) | https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth | downloaded by `roma_init`'s default matcher when `RomaInitConfig.model = "tiny_roma_v1_outdoor"`, as RoMa outdoor's; it supplies every XFeat weight Tiny RoMa keeps (Q-123) | as RoMa outdoor's: RoMa's README is silent on weights; Parskatt/storage is MIT, Copyright (c) 2022 Johan Edstedt | no |
| XFeat, through torch.hub | the archive https://github.com/verlab/accelerated_features/zipball/e92685f57f8318b18725c5c8c0bd28c7fe188d9a, loaded with `pretrained=False` (Q-116, Q-123) | downloaded into torch.hub's cache when Tiny RoMa is selected, for XFeat's code | XFeat's `LICENSE` at e92685f: Apache-2.0 (SHA-256 c71d239d…, source log row 107, fetched again with the same hash); its README's "License" section points to it. openftgs loads no XFeat weights of its own: they come from Tiny RoMa's checkpoint (Q-123). The archive does carry the repository's `weights/xfeat.pt` and `weights/xfeat-lighterglue.pt` (names in torch.hub's cache, source log row 285), under the same Apache-2.0, unused | no |

**XFeat's terms are permissive** (Apache-2.0, for its code and the repository's files), so the owner's decision the ticket reserves for a non-permissive result doesn't arise: `RomaInitConfig.model = "tiny_roma_v1_outdoor"` stays as it is.

**Never redistributed.** openftgs's wheel and sdist hold no weight file: no `.pth`, `.pt`, `.ckpt`, `.bin` or `.safetensors` in either (the listings of "No dataset material" below). The weights reach a user's machine only from the projects' own hosts, at run time.

**For the owner's attention (no sign-off asked).** Every weight openftgs fetches now has a publisher's statement of its terms. Two are weaker than a named license, as D-89 records them: BVLC's AlexNet is released "for unrestricted use" with no license named, and DINO's README puts "this repository" under Apache 2.0 without naming the weights it hosts. Oxford's VGG16 asks for attribution under CC BY 4.0, which NOTICE gives. The release notes' account of the weights' terms should say so. History: until revision 549o the perceptual loss and metric were LPIPS, whose torchvision trunks had no affirmative grant; the owner declined to sign them off (2026-10-23) and chose ꟻLIP, which has no weights, after PRD #373's and #374's research (D-9, 2026-10-25).

## No dataset material

§2.3 notes: datasets are never bundled; SelfCap is licensed for non-commercial research only, and nothing derived from it enters the repository, the wheels or CI; Neural3DV is CC-BY-NC-4.0 (D-79). CLAUDE.md: `docs/paper/` and datasets are never committed. Checked on 2026-10-22 by ticket #367 on commit 7e1faf0 of `prd-355-release`, with git 2.43.0, in the tracked tree, the whole history, and a freshly built sdist and wheel. Every search root is inside the repository, out of `.venv` and `.scratch`; helper scripts and outputs went to the session's scratchpad, written `<s>` below.

### The tracked tree

253 tracked files (`git ls-files`).

**By name.** `git ls-files | grep -i -E '<media and data extensions>|<dataset and scene names>|docs/paper|(^|/)(data|datasets|runs)/'`, with the extensions `mp4 avi mov mkv webm png jpg jpeg bmp tif tiff exr npy npz ply tar gz tgz zip 7z pth pt ckpt bin h5 pdf yml yaml safetensors svg`, the names `poses_bounds intri extri sync.json selfcap n3dv neural enerf corgi hair yoga bike dance coffee spinach beef salmon steak actor`. Hits, each accounted for:

| Hit | What it is |
| --- | --- |
| `.github/workflows/ci.yml`, `gpu-bench.yml` | CI workflows |
| `docs/coverage.svg` | the coverage badge (Q-217) |
| `docs/modules/data/*.md` (4) | module docs of `openftgs.data` |
| `tests/synthetic/data/s{1,2,3,4,6}_gt.safetensors`, `seeds.json` | the S-n fixtures (below) |

No `docs/paper/`, `data/`, `datasets/` or `runs/` path, and no image, video, NumPy, point-cloud, archive or checkpoint file.

**By content.** The only binary files are the five S-n fixtures: every other tracked file is UTF-8 text without a NUL byte (the history check below applies the same test to every blob). The largest tracked files are `docs/clean-room/questions.md` (587,931 bytes), the spec, the source log and source and test modules. `git grep -I -i -c -E 'selfcap|neural.?3d.?v|n3dv|enerf|coffee_martini|cook_spinach|cut_roasted_beef|flame_salmon|flame_steak|sear_steak|hair-calib|yoga-calib|corgi'` names the datasets in 37 files, every hit a name or prose, not data:

- the spec and `docs/spec/README.md`, CLAUDE.md, `docs/clean-room/` (questions, source log, milestones, validation runs), `docs/GLOSSARY.md`, `docs/modules/`, `docs/research/animated-playback.md`, `docs/agents/clean-room.md`, `.claude/skills/tdd/mocking.md` and `README.md`: the owner's and the project's prose;
- `src/openftgs/data.py`, `eval.py`, `cli/__init__.py` and `cli/_bench.py`: the converters and their names, written from §1.4 and the datasets' documentation (D-28, D-59 to D-64); `_bench.py`'s `selfcap/studio` and `n3dv/kitchen` are openftgs-synth scenes (CC0-1.0, Q-239);
- `scripts/ci_scene.py`, `crop_scene.py`, `synth_masks.py` and `synth_window.py`: the CI's procedural scene, sized after Neural3DV's setup (§3.4), and tools for converted openftgs-synth scenes;
- the tests: scene names as strings (`n3dv/flame_steak` in `test_eval.py` and `test_io.py`), and mini-archives and calibration files in the datasets' layouts that `tests/builders.py` and the tests generate in pytest's `tmp_path`, with invented values (`SELFCAP_OFFSETS = {"0000": 0.004, "0001": -0.0125, "0002": 0.02}`; `test_data_images.py`'s hand-written `intri.yml` with round numbers, and the 2160 × 3840 image size the spec's prose gives). C-1 to C-5 in `tests/test_converter_checks.py` read the datasets only from `$OPENFTGS_DATASETS` when the raw downloads are there, and skip otherwise (Q-7, Q-100, D-79).

`git grep -n -E '%YAML|opencv-matrix'` finds OpenCV-style YAML only in `data.py`'s reader, its module doc and `test_data_images.py`'s generated files.

**The benchmark baselines** (`benchmarks/baselines/nvidia-geforce-rtx-4090/*.json`, 6 files) are BENCH.json timing records of the CI's procedural scene (`scripts/ci_scene.py`), with no image data (Q-196).

### The whole history

Every ref (`git for-each-ref`: 36 local branches, `main` and the worktrees' branches among them, and 2 remote-tracking refs), 1,325 commits (`git rev-list --all`):

```
git rev-list --all --objects > <s>/objects.txt                       # 9,098 objects
git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' < <s>/objects.txt > <s>/sizes.txt   # 2,706 blobs
awk '$2=="blob"{print $1}' <s>/sizes.txt > <s>/blobs.txt
git cat-file --batch < <s>/blobs.txt > <s>/blobs.bin                  # the 2,706 blobs, read by the two scripts below
git log --all --name-only --format= | sort -u                         # every path ever committed
git log --all --oneline -- docs/paper                                 # empty
git log --all --oneline -- 'data/*' 'datasets/*' 'runs/*'             # empty
```

The two scripts (run with `/usr/bin/python3 -I`, Python 3.12.3) walk `blobs.bin` record by record, a header line `<oid> blob <size>` and then `size` bytes, and name each blob by its path in `sizes.txt`:

```python
binary = b"\0" in body or not _decodes_as_utf8(body)                 # the first: binary blobs
hits = Counter(path[oid] for oid, body in blobs if re.search(pattern, body, re.I))   # the second, per pattern
```

- **Paths.** 273 paths were ever committed. The name search of the tracked tree, run on them, adds only the retired S-5 fixture `tests/synthetic/data/s5_gt.safetensors` (removed by e9c33df, #216) and the skills' `agents/openai.yaml` files (agent configuration, removed by 237c8b3). `docs/paper/`, `data/`, `datasets/` and `runs/` were never committed.
- **Binary blobs.** A script that reads `blobs.bin` and flags every blob holding a NUL byte or not decoding as UTF-8 finds 6 binary blobs and 2,700 text blobs. The six are the S-n fixtures: `s1_gt` (1,116 bytes), `s2_gt` (1,712), `s3_gt` (2,056), `s4_gt` (21,224), `s6_gt` (3,072) and the retired `s5_gt` (169,776). Every blob above 50 KB is a version of a text file (questions, source log, glossary, spec, module docs, source and test modules) or `s5_gt`.
- **Data signatures.** A second script, a case-insensitive regular-expression search of every blob, finds no PNG (`\x89PNG`), NumPy (`\x93NUMPY`), MP4 (`ftyp(isom|mp4)`), PLY (`ply\nformat`) or `base64,` content. `%YAML` and `opencv-matrix` occur only in versions of `questions.md`, `source-log.md`, the spec, `docs/modules/data/images-and-cameras.md`, `data.py` and `test_data_images.py`; `poses_bounds` only as a name, in the same kinds of files and the converter tests.

### The sdist and the wheel

Built through `uvx`, as Q-241 says, from the worktree into the scratchpad, never the repository's `dist/`. The project has no MANIFEST.in, so, before building, empty placeholder files were put at the ignored paths that hold the paper and data in a working checkout, `docs/paper/placeholder.pdf`, `data/placeholder.txt`, `datasets/placeholder.txt` and `runs/placeholder.txt`, beside the worktree's untracked RoMa clone (`.scratch/`); they were removed after the build, with the build's `src/openftgs.egg-info`.

```
uvx --from build pyproject-build --installer uv --outdir <s>/t367-build <worktree>
```

uv 0.11.32, build 1.6.1, setuptools 84.2.1 in the isolated build environments (source log row 286). Built: `openftgs-0.1.0.dev0.tar.gz` (563,382 bytes, SHA-256 0f175d34a7860c1ec211ba92cd4059a7de9ea7b4b45e3ee418e3537b8a828e2d) and, from it, `openftgs-0.1.0.dev0-py3-none-any.whl` (276,732 bytes, SHA-256 ac53bb18ad9c4cdf94671f77d3b6971853873cb0b8ab20f0b1a16d125fbfad7d). Listed with `tar tzf` and `python -m zipfile -l`, and compared with `git ls-files`:

- **sdist**, 109 files: `LICENSE`, `NOTICE`, `README.md`, `pyproject.toml`, the package under `src/openftgs/` (the vendored RoMa's license files and `VENDOR.json` included), and setuptools' default `tests/test*.py` (42 test modules, without `conftest.py`, `builders.py`, `tests/synthetic/` or `tests/harness/`). Every file is a tracked file except setuptools' generated `PKG-INFO`, `setup.cfg` and `src/openftgs.egg-info/*` (6).
- **wheel**, 62 files: the package under `openftgs/`, the vendored RoMa's license files and `VENDOR.json` included, and the 7 files of the `.dist-info`, with `licenses/LICENSE` and `licenses/NOTICE`. Its 55 package files are the sdist's 55 files under `src/openftgs/`, each a tracked file.
- **Neither** holds a placeholder, a `docs/` path, a `data/`, `datasets/` or `runs/` path, a `__pycache__` or `.pyc` file, any file with the media, data, archive, checkpoint or YAML extensions of the name search, or a file named after a dataset or scene. The dataset names inside them are those of the tracked files they copy (the converters and the test modules, above).

**Rebuilt on the merged tree (#432), 2026-10-27**, from `ticket-432-after-380` at 747d6e7 (`main` at 0efc6b2 merged in, 295 tracked files), the same way: the same four placeholders put in place and removed after the build, with `src/openftgs.egg-info` and setuptools' `build/`; `uvx --from build pyproject-build --installer uv --outdir <s>/t432-build <worktree>`, uv 0.11.32, build 1.6.1. Built: `openftgs-0.1.0.dev0.tar.gz` (699,552 bytes, SHA-256 95df83adeeb1360250a26bb95f891e2f69dafa22aba79237d7e196c2ac4b32de) and, from it, `openftgs-0.1.0.dev0-py3-none-any.whl` (358,501 bytes, SHA-256 67c40811897bcd7681f36c3d6a5a9f3d1f5742b72722d56b1e61d0bddf5ad1a6), listed with Python's `tarfile` and `zipfile` and compared with `git ls-files`:

- **sdist**, 124 files: `LICENSE`, `NOTICE`, `README.md`, `pyproject.toml`, the package's 62 files under `src/openftgs/` (now with the vendored ꟻLIP's `flip_loss.py`, `LICENSE`, `VENDOR.json` and `__init__.py`, and `_caffe_trunks.py`, `_legacy.py` and `_weights.py`), and setuptools' default `tests/test*.py` (50 test modules; not `tests/reference/flip`, `conftest.py`, `builders.py`, `tests/synthetic/` or `tests/harness/`). Every file is a tracked file except setuptools' generated `PKG-INFO`, `setup.cfg` and the six files of `src/openftgs.egg-info/`.
- **wheel**, 69 files: the same 62 package files under `openftgs/`, and the 7 files of the `.dist-info`, with `licenses/LICENSE` and `licenses/NOTICE`.
- **Neither** holds a placeholder, a `docs/`, `data/`, `datasets/` or `runs/` path, a `__pycache__` or `.pyc` file, or any file with the media, data, archive, checkpoint (`.pth`, `.pt`, `.ckpt`, `.bin`, `.safetensors`, `.caffemodel`) or YAML extensions of the name search.

### The S-n fixtures' provenance

The only data files in the repository and its history. `tests/synthetic/data/s1_gt.safetensors`, `s2_gt`, `s3_gt`, `s4_gt` and `s6_gt` are ftgs-if exchange models (§2.2; 1, 8, 12, 240 and 24 Gaussians, SH degree 0) of §3.2's synthetic scenes, written by `tests/synthetic/make_ground_truth.py` with `openftgs.io.write_exchange` from the seeds in `tests/synthetic/data/seeds.json` and `synthetic_scenes.py`'s generators; their metadata's producer is "openftgs tests (M11 ground truth)", or M13 for `s4_gt`. `test_committed_ground_truth_reproduces_from_its_seeds` (`tests/synthetic/test_synthetic_recovery.py`) checks that the committed files are what the seeds give, and passed on this commit. SHA-256: s1 39ece2ad…, s2 8b4927ec…, s3 bfef9f4b…, s4 888a5770…, s6 fb0efad4…, seeds.json 2063ebb8…. They derive from no dataset. The other S-n scenes are generated at test time: S-7 and S-8 from S-2's ground truth and their seeds, S-5, S-9 and S-11 by the analytic ray caster (`tests/synthetic/raycast.py`) from their seeds, S-10 from a tiny scene built in the test; the retired `s5_gt` was a generated exchange model like the others.

### The spec's observations of SelfCap's calibration files

The spec records what the owner saw in SelfCap's files: §1.4's SelfCap protocol ("Camera files, checked on hair-calib and yoga-calib and on the corgi listing", the sync.json offset ranges, the calibration source, the point clouds' numbering, the corgi archive's CRC failure), D-59, D-63 and D-64, and Appendix A rows 30, 31 and 38. They are the owner's text, not data: file and key names, matrix shapes, image sizes, value ranges and agreement bounds (offsets from −0.046 to +0.026 s, Rot against Rodrigues(R) to 3·10⁻⁸, cosines ≥ 0.64), counts and dates, written as prose. Beyond the few values that prose quotes (the 2160 × 3840 size, k3 = 0, hair's t\_ = 0, n\_ = 1 and f\_ = 20, the ranges and bounds), the checks above found no calibration matrix, frame, mask or point cloud of SelfCap in the repository, its history or the built packages. `questions.md`, the module docs and the tests repeat that prose, or build files in that layout with invented values (above). The owner's copies of hair-calib and yoga-calib, and the corgi listing, were never committed.

## Source-log audit

Ticket #368, 2026-10-22, on the tree of `prd-355-release` at 7e1faf0 (the repository's own files only; nothing outside the repository was opened). Every search read tracked files only, through `git grep` or `git ls-files` fed to a Python reader, so `.venv`, `.scratch` and every other ignored path stayed out (owner rule, 2026-10-12). The records matched against are `docs/clean-room/source-log.md`, rows 1 to 275, and spec Appendix A, #1 to #44.

**Scope.** `src/` (the vendored RoMa included), `scripts/`, `tests/`, `harness/`, `docs/modules/`, `README.md`, `NOTICE` and both `pyproject.toml` files, as the ticket lists them, plus `.github/workflows/` and `benchmarks/`.

**Searches.**
- URLs: `git grep -n -o -E "https?://[^ )\"'>\`<,]+"` over the scope, and a search for hosting sites named without a scheme (`github.com`, `huggingface.co`, `pypi.org`, `arxiv.org`, the project page's host).
- Projects: an AST walk of every `.py` file under `src`, `scripts`, `tests`, `harness` and `docs` for top-level imports outside the standard library and the two first-party packages; the dependency tables of both `pyproject.toml` files; the programs the code runs; the actions the workflows use; project names in the module docs, the README and NOTICE; literature cited in comments and docstrings.
- Attribution: `git grep -n -i -E "adapted|adapts|ported|borrowed|taken from|copied|based on|inspired by|derived from|courtesy|credit"`, `copyright`, SPDX lines other than Apache-2.0, third-party file paths in comments (`gsplat/…`, `torch/…`), and every comment that cites the source log.
- Forbidden names: the clean-room guard's own pattern (`NAME` in `.claude/hooks/clean_room_guard.py`) over every tracked file, with the names the records keep closed or documentation-only: the 3DGS reference rasterizer's repository and organization, gsplat's `contrib/dynamic`, the project page's renderer (`fast-gaussian-rasterization`, `fast_gauss`, its owner's account), 4K4D, nvdiffrast, BlenderProc, plyfile and the authors' group's GitHub organization. Over the scope, also the implementations of other dynamic-scene methods (4DGS, STGS, Spacetime Gaussians, 4D-GS, Deformable 3DGS, HexPlane, K-Planes, Dynamic 3DGS) and tqdm: no hit.

### URLs

71 occurrences, 49 distinct URLs, in 26 files. Each distinct URL has a row, an Appendix A entry, or a reason it needs none.

| Distinct URLs | Where | Record |
| --- | --- | --- |
| RoMa's repository and its `LICENSE` at 77f8d68 (2) | NOTICE, `scripts/vendor_roma.py`, `VENDOR.json` | Appendix A #10, #15, #26, #33; rows 106, 116 |
| DINOv2's repository and `LICENSE` at 7764ea0 (2) | NOTICE, `VENDOR.json` | Appendix A #10; row 108 |
| DINO's repository and `LICENSE` at 7c446df (2) | NOTICE, `VENDOR.json` | row 111 |
| timm's repository and `LICENSE` at 637f819 (2) | NOTICE, `VENDOR.json` | row 112 |
| Patch2Pix's repository and `LICENSE` at 0777495 (2) | NOTICE, `VENDOR.json`, vendored `utils/utils.py` | row 109 |
| LoFTR's repository and `LICENSE` at 94e98b6 (2) | NOTICE, `VENDOR.json`, `scripts/vendor_roma.py` | row 110 |
| Kornia's repository and `LICENSE` at 542752e (2) | NOTICE, `VENDOR.json` | row 113 |
| InLoc's demo repository, its README at b4c42de and that path's prefix (3) | NOTICE, `VENDOR.json`, `tests/test_vendored_roma.py:99` | row 139 (Q-118) |
| `https://raw.githubusercontent.com`, the prefix of the license URLs (1) | `scripts/vendor_roma.py:134` | rows 106–113, 139 |
| RoMa outdoor's and Tiny RoMa's weights, DINOv2 ViT-L/14's weights (3) | vendored `models/model_zoo/__init__.py`, `models/encoders.py` | rows 132, 135; Appendix A #26, #33 |
| fused-ssim's repository and commit a7c48d6 (2) | `README.md:37, 40`, `src/openftgs/losses.py:56` | rows 45, 73; Appendix A #10 |
| fused-local-corr's repository and commit 689f2f7 (2) | `README.md:30, 33` | row 3; Appendix A #10 |
| PyTorch's CPU wheel index and its install page (2) | `.github/workflows/ci.yml:33`, `README.md:26` | new row 277 |
| The CI workflow's page and status badge (2) | `README.md:3` | the project's own repository; the badge's URL scheme in row 277 |
| Links carried in RoMa's own files: `RoMaV2/issues/35` (twice), DINO's `vision_transformer.py` (main and master) and `issues/8`, timm's `vision_transformer.py`, `patch_embed.py`, `mlp.py`, `drop.py` and the `#L103-L110` link, LoFTR's `geometry.py`, and the `roma_indoor.pth` weights (11) | vendored RoMa | none needed: upstream text vendored unchanged with RoMa at 77f8d68 (D-65; `VENDOR.json` records each file's hash and patch, and these lines are in no patch), whose reading is Appendix A #26, #33 and rows 116, 120; none of these pages was opened, and `roma_indoor.pth` is never downloaded, since `openftgs.init` builds only RoMa outdoor and Tiny RoMa |
| `http://www.apache.org/licenses/` and `…/LICENSE-2.0` (2) | `harness/LICENSE`, the vendored `LICENSES/` | none needed: part of the Apache-2.0 text |
| The SVG namespace, `http://www.w3.org/2000/svg`, as written and in ElementTree's `{…}` form (2) | `scripts/coverage_local.py:58`, `tests/test_coverage_local.py:30` | none needed: an XML namespace name, never fetched |
| Placeholders in the guard's tests: `example.com/page`, `example.com/{NAME}.tar.gz`, `github.com/someone/…`, `github.com/x/…`, and the project page's address built from the assembled name (5) | `tests/test_clean_room_guard.py:79, 99, 100, 133, 134` | none needed: inputs to the guard, never fetched; the last is the project page, an allowed source (Appendix A #2) |

The one scheme-less hit, "the zju3dv family" of YAML layouts in `docs/modules/data/converters.md:8`, is D-63's reading of the SelfCap files (Appendix A #25, #30). `docs/modules/`, `harness/src/` and `benchmarks/` hold no URL.

### Upstream projects

| Project | Where | Record |
| --- | --- | --- |
| PyTorch (and torch.hub, NVTX, the profiler) | package, scripts, tests, harness | Appendix A #13, #21; rows 8, 15, 17–19, 50, 131, 153, 161, 175–178 |
| gsplat 1.5.3 | `render`, `optim`, tests | Appendix A #8, #9, #23, #38, #40; rows 6, 10, 13, 22, 27–31, 55–57, 158, 184 |
| safetensors | `io`, harness, tests | Appendix A #22, #35; rows 11, 148, 194 |
| ꟻLIP at d3ab02a (BSD-3-Clause), vendored with marked changes as `openftgs._vendor.flip`, its unmodified reference in `tests/reference/flip` (Q-252) and the harness's copy (Q-254), built by `scripts/vendor_flip.py` (#432) | `losses`, `eval`, harness, NOTICE | rows 425, 444, 445; research rows 361 to 365 |
| piq 0.8.0's HaarPSI (Apache-2.0), adapted in `openftgs.losses`, not imported (#432) | `losses`, NOTICE | rows 430, 434 (research), 446 to 448 |
| Caffe's `caffe.proto` field numbers, for the wire-format reader of `openftgs._weights` (BSD-2-Clause) (#432) | `_weights.py` | rows 422, 441 |
| The six trunk networks' weights: Oxford's VGG16 (CC BY 4.0), BVLC's AlexNet, DINOv2 ViT-S/14 and ViT-B/14, DINO ViT-S/16 and ViT-B/16 (#432; Pretrained weights' terms, above) | `_weights.py`, `_caffe_trunks.py`, `losses`, NOTICE | rows 424, 438 to 443 |
| PyYAML; Pillow; SciPy; NumPy | package, scripts, tests | rows 88, 89; rows 74, 183; rows 7, 61, 127, 194; rows 15, 20, 102, 194 |
| fused-ssim (optional) | `losses`, `train` | rows 45, 73 |
| RoMa at 77f8d68, vendored, with what it imports: torchvision, einops, loguru, kornia (inside one helper), xformers (optional), fused-local-corr (`local_corr`) | `src/openftgs/_vendor/romatch`, `openftgs.init` | Appendix A #10, #15, #26, #32, #33; rows 106–126, 139, 156; fused-local-corr row 3 |
| DINOv2, DINO, timm, Patch2Pix, LoFTR, Kornia, InLoc, credited in RoMa's files | NOTICE, `VENDOR.json`, the vendored `LICENSES/` | rows 106–113, 139 (Q-115, Q-118) |
| XFeat through torch.hub (Tiny RoMa), and the XFeat paper named in RoMa's `models/tiny.py` | `openftgs.init`, tests | Appendix A #26; rows 107, 136, 138 |
| pytest, pytest-xdist, pytest-cov and coverage | tests, `pyproject.toml`, CI | rows 79, 196–201 |
| hypothesis (MPL-2.0, documentation only) | tests | row 7 |
| ruff, mypy | `pyproject.toml`, CI | rows 76, 82, 84, 275 |
| pip-licenses | `scripts/license_scan.py`, CI | rows 1, 5 |
| scikit-image and opencv-python-headless, the tests' oracles (torchmetrics, an oracle until revision 549o, left with #393; rows 66, 68, 69) | tests | row 46; rows 89, 188 |
| matplotlib | harness | row 194 |
| setuptools as the build backend; GitHub Actions | both `pyproject.toml` files; `.github/workflows/` | new row 277; row 182 (`gpu-bench.yml`) and 277 (`ci.yml`) |
| ffmpeg and ffprobe; COLMAP; nvidia-smi; git | `openftgs.data`, `openftgs.cli`, scripts | rows 95, 101, 179, 180; Appendix A #28, rows 92, 100; row 162; row 164 and rows 115, 116 |
| OpenEXR's file layout and ZIP step; the PLY format; OpenCV's FileStorage YAML | `scripts/gt_depth.py`, `tests/exr_writer.py`; `openftgs.io`; `openftgs.data` | rows 212, 213; rows 12, 13; row 89 |
| Neural3DV and its LLFF poses; ENeRF-Outdoor; SelfCap; openftgs-synth | converters, scripts, tests | Appendix A #11, #12; row 97; Appendix A #25, #30, #31, rows 98, 99; Appendix A #44, rows 88, 133, 163, 203, 214, 274 |
| The FreeTimeGS paper; 3DGS (its EWA projection); 3DGS-MCMC; SSIM (Wang et al. 2004) | the package, docstrings and comments | Appendix A #3; #6; #7; #14 |

Named without anything taken: Nsight Systems in `docs/modules/cli/benchmarks.md:92`, as a profiler that shows the NVTX ranges; BackgroundMattingV2 as the `"bgmv2"` mask source in `tests/test_eval.py:496` (Appendix A #10, #14). Neither needs a row.

### Attribution comments

| Comment | Source | Record |
| --- | --- | --- |
| `src/openftgs/render.py:909`, "Adapted from gsplat 1.5.3, gsplat/rendering.py, rasterization()", Apache-2.0 | gsplat 1.5.3 | row 27 (Q-37); row 28 found no NOTICE file at v1.5.3, so NOTICE has no gsplat entry |
| `src/openftgs/optim.py:191`, `torch/optim/adam.py`, "read, not copied" | PyTorch 2.14.1 | row 50 |
| `scripts/gt_depth.py:76-82` and `tests/exr_writer.py:12-16`, OpenEXR's file layout and `internal_zip.c` at 6a75faa, "written here anew" | OpenEXR (BSD-3-Clause) | rows 212, 213; `gt_depth.py` says "(source log)" without the numbers, and its content matches those two rows |
| `scripts/vendor_roma.py:188`, the approach credited in RoMa's `utils/utils.py` | InLoc | row 139 (Q-118) |
| The vendored files' own credits: Meta's headers on the DINOv2 layers and their references to DINO and timm, LoFTR's "adapted from here" and the Patch2Pix and InLoc credits in `utils/utils.py` | RoMa's upstream text | NOTICE, `VENDOR.json`, the vendored `LICENSES/`; rows 106–113, 139 (D-65, Q-115, Q-118) |

No SPDX line in the scope names a license other than Apache-2.0; the vendored files keep their own headers (D-65). The code cites the source log on 35 lines of `src/`, `scripts/` and `tests/`, and the module docs 18 times; each cited row was checked against what the comment says it holds, and all match.

### Forbidden-name search

Over every tracked file at 7e1faf0, before this section and rows 276 and 277 were written:

| Hits | Files | What | Classification |
| --- | --- | --- | --- |
| 312 | 50 | `FreeTimeGaussians` | the package's own class (§2.1); in the scope, 275 hits in 43 files; the source log's only hit of the guard's pattern is this name, in row 32 |
| 60 | 11 | the paper's name | the work's own subject: in the scope only D-62's statement (`README.md:5`, `pyproject.toml:9`, `src/openftgs/__init__.py:2`, `src/openftgs/cli/__init__.py:910`) and §1.4's "FreeTimeGS scene" (`src/openftgs/data.py:1251`); elsewhere the spec, CLAUDE.md, the glossary, `docs/spec/README.md`, `milestones.md` and the guard |
| 23 | 4 | the never-allowed framework | rule text only: CLAUDE.md, the guard, the glossary and the spec (§0, Appendix A's introduction, #25, #37, #39, D-63); none in the scope |
| 5 | 3 | the guard's pattern quoted | the guard's comment and pattern, the `tdd` skill's file-name rule, CLAUDE.md |
| 17 | 11 | the authors' group's GitHub organization | allowed sources only: the project page (Appendix A #2; the guard's one allowed URL and its test), the SelfCap dataset (#25, row 98), ENeRF's dataset documentation (row 97), LoFTR (Apache-2.0, credited by RoMa; rows 110, 115) and the YAML family of §1.4 and Q-96; never the authors' code |
| 14 | 3 | the project page's renderer | allowed under the owner rule of 2026-10-22: CLAUDE.md, rows 270, 272 and 273, and `docs/research/animated-playback.md` §5.4; none in the scope |
| 5 | 3 | gsplat's `contrib/dynamic` | statements that it stayed unopened: CLAUDE.md, rows 154, 158 and 245, Appendix A #9 |
| 3 | 1 | the 3DGS reference repository and organization | row 237, its license file only (non-commercial, documentation only) |
| 4 | 2 | BlenderProc | row 87 and Appendix A #43, license files only (GPL-3.0) |
| 4 | 3 | 4K4D | a paper of the project page's group, named only (row 271, the research doc) and as a row label of the paper's tables (spec §1.2, D-29) |
| 2 | 2 | nvdiffrast | general knowledge only (row 271, the research doc) |
| 1 | 1 | plyfile | row 12: neither used nor read |
| 10 | 4 | "Gaussian-splatting" | false positives of the search's pattern: "general Gaussian-splatting literature" and "practice" (CLAUDE.md, the spec, the glossary), rows 12 and 240, and a trade-press article's address in row 251 |

`tests/test_clean_room_guard.py` assembles the names it tests from two string halves, so its source holds no hit. No row of the source log names a forbidden source as read: the rows at the boundary are 12 (plyfile, not read), 87 and 237 (license files only), 154, 158 and 245 (gsplat's dynamic module, not opened), 262 (the renderer, not opened under the rule then in force), 270 to 273 (the renderer, under the owner rule of 2026-10-22) and 271 (general knowledge). Row 273's sighting of the framework's import in one file's header is not an incident (PRD #355, owner decision 1; row 276), and nothing of the renderer is in the scope. The source log has no row marked **Incident**. The check that no tracked path holds the paper's or the framework's name, a `git ls-files` piped into `grep`, was blocked by the guard as a search it couldn't confirm stays inside the repository, and was not retried (reported to the run's lead).

### New rows

- **276**: the owner's decision on row 273, in the owner's words (housekeeping, below).
- **277**: the general knowledge no earlier row recorded: PyTorch's CPU wheel index and install page, GitHub Actions as `ci.yml` uses it and the workflow badge's URL, the PEP 517, 621 and 639 packaging of both `pyproject.toml` files, `pip wheel` in CI, and the flat badge's look in `scripts/coverage_local.py` (Q-217).

Nothing else in the scope needed a row: every other URL, project and attribution matches a row or Appendix A, or is one of the kinds in the URL table that needs none.

### Housekeeping (#368)

- Q-239 and Q-240 moved from inside the Calibration data table into the Q-n table, after Q-238 and before Q-241 (PRD #355, owner decision 6; commit 071788e). The two rows are 4,332 bytes with SHA-256 `926d2b7637c513c198f1fc4d0f7730626afaf4f6b67269d1fcdae33764b9b4a8` before and after; `questions.md` keeps its size and its lines, and with the two rows taken out the old and new files are identical. The Calibration data table now holds only its own rows.
- Row 276 records the owner's decision on row 273 (commit 6f42859).
- The glossary's "Spec revision" entry already read revision 549n, word for word as the PRD gives it (5288b72); it was checked, not changed.

### Rerun over rows 334 to 467 (#432)

Ticket #432, 2026-10-27, on `ticket-432-after-380` at 747d6e7: `main` at 0efc6b2 (PRD #380 and its follow-ups, #396 to #431) merged into `prd-355-release`. The records matched against are source-log rows 334 to 467, `main`'s, word for word the same on this tree, and spec Appendix A #45, which points at rows 334 to 436 (#373's and #374's research); rows 437 to 467 are #380's build rows, after that export. The branch's own rows 275 to 333 (#356 to #369) are recorded by the sections of this checklist that cite them and were not audited again here. The searches are those above, run on tracked files only (`git grep`, `git ls-files`, `main`'s tree through `git archive`); the vendored RoMa, `scripts/vendor_roma.py` and its test are unchanged since 7e1faf0.

**URLs.** 29 distinct URLs in the scope are new since the table above: 22 from `main`'s work and 7 from the documentation site. Each has a row, an Appendix A entry, or a reason it needs none:

| Distinct URLs | Where | Record |
| --- | --- | --- |
| ꟻLIP's repository, its `LICENSE` and `src/pytorch/flip_loss.py` at d3ab02a, and `scripts/vendor_flip.py`'s template of the raw address (4) | NOTICE, both ꟻLIP copies' `__init__.py` and `VENDOR.json`, `scripts/vendor_flip.py` | rows 361 to 363, 425, 444; the script only writes the address into `VENDOR.json` and opens no connection |
| Links in NVIDIA's own text: three NVIDIA Research publication pages, a tone-mapping page, an ACES curve article, an sRGB–CIEXYZ conversion page (6) | the three copies of `flip_loss.py` | none needed: upstream text carried unchanged, byte-identical in all three copies and in no edit of `VENDOR.json`; none was opened |
| piq's repository (1) | NOTICE, `src/openftgs/losses.py` | rows 357, 359, 446 |
| The Apache-2.0 license's https address (1) | NOTICE | none needed: the license's own address |
| Oxford VGG's model page; CC BY 4.0's deed (2) | NOTICE | rows 340, 419; row 450 |
| The six trunk weights' URLs, Oxford's and BVLC's Caffe files and the four DINO checkpoints (6) | `src/openftgs/_weights.py`, `tests/test_weights.py` | rows 419, 421, 440 (the URLs), 424 (sizes and SHA-256, which match the code), 443 |
| The test's local server and an unreachable address (2) | `tests/test_weights.py` | none needed: `127.0.0.1`, never outside the machine |
| The Read the Docs badge; the site; the project's repository (3) | `README.md`, `pyproject.toml`, `docs/index.md`, `tests/test_docs_and_ci_config.py` | row 315; the project's own site (Q-244) and repository |
| openftgs-synth's repository and its README anchor (2) | `docs/datasets.md`, `docs/quickstart.md` | row 307 |
| Neural3DV's repository (1) | `docs/datasets.md` | Appendix A #11 |
| The paper on arXiv (1) | `docs/index.md`, `docs/method.md` | Appendix A #3; row 309 |

PyTorch's CPU index and install page, fused-local-corr and fused-ssim now also appear in the docs install and `docs/installation.md` (rows 3, 45, 73, 277, 314, 321). `docs/conf.py`'s intersphinx inventories are row 295.

**Upstream projects.** New: NVIDIA's ꟻLIP, piq's HaarPSI, Caffe's field numbers and the six weights (the table above); #374's prototype `converters.py` and `candidates.py`, the owner's uncommitted Apache-2.0 research code read in place (rows 438, 439, 449, 452; the allowance recorded in row 463); lpips as a source of facts only, its ε and its uncalibrated distance (rows 343, 349, 350, 435); PyTorch's torch.hub, safetensors' errors and protobuf's wire format as general knowledge (rows 441, 454, 456, 457, 465, 466); tools used for their output (rows 437, 459 to 461). Gone: lpips 0.1.4 with torchvision's trunks, and torchmetrics. An AST walk of every tracked Python file outside the vendored RoMa finds no import of lpips, torchmetrics, piq or torchvision; their remaining mentions are the ε's and distance's source in comments and docstrings, the handling of the retired `perc_net` names (Q-247: `_legacy.py`, `config`, `io`, `eval`, `train`, `cli/_bench.py`), and the checks that they are absent (`tests/test_skeleton.py`, `ci.yml`'s wheel check, the harness's settings and tests).

**Attribution comments.** New: `_caffe_trunks.py` and `_weights.py` (adapted from the prototype; `caffe.proto`), `losses.py` (HaarPSI from piq 0.8.0 between its start and end markers; the DINO networks from the prototype), the vendored ꟻLIP's modification notices and package docstrings, and NOTICE's ꟻLIP, piq and VGG16 sections. The only SPDX lines other than Apache-2.0 are NVIDIA's `BSD-3-Clause` in the three ꟻLIP copies. The code now cites the source log on 81 lines of `src/`, `scripts/` and `tests/` (15 of them `scripts/license_signoff.json`'s entries, row 331), 29 of them rows 334 or later, and the module docs on 24 lines, 9 of them rows 334 or later. Every comment citing a row of 334 or later was checked against the row, and all match but one, in part: `_caffe_trunks.py`'s comment on BVLC's mean pixel cites row 426 for the means "to eight decimals", which row 426 records to three; the eight-decimal values are in row 449.

**Rows 334 to 467.** Rows 334 to 436 are #373's and #374's research (Appendix A #45): the screened-out candidates, the papers the owner accepted after the fact (row 436), torchvision's terms for LPIPS's trunks, and the reads behind the code that stayed (ꟻLIP, piq, Caffe, the VGG, AlexNet, DINOv2 and DINO terms). Rows 437 to 467 are #380's build: the trunk weights (438 to 443), the vendored ꟻLIP (444, 445), HaarPSI (446 to 448), the Caffe trunks (449 to 451), the DINO networks (452 to 454), the trainer, the benchmark, the dependency removal and the reviews (455 to 467). No row is marked **Incident**, and none records reading the source code of a copyleft, non-commercial or unlicensed project: the closed-class projects the research met (pyiqa, PieAPP, Watson-DFT, LASI, DINOv3, ST-LPIPS, R-LPIPS; GPL and LGPL system tools in row 437) were read as license files, READMEs, papers or tool output only.

**Forbidden names**, over rows 334 to 467 and the 105 files `main` changed since 7e70ed7, and the site's pages: the guard's pattern finds the package's own class, the paper's name as the project's subject and rule text, and the framework named only in rule text and D-63 ("left unopened"); the other closed names of the search above appear only in CLAUDE.md and the spec, as before. The closed-class candidates appear only in rows 358 to 436 and in `docs/research/perceptual-alternatives.md`, as survey entries with their licenses. tqdm appears only where it is removed or checked absent. The guard blocked no search.

**For the owner's attention** (findings, nothing changed):
- the harness's copy of ꟻLIP (`harness/src/ftgs_harness/_vendor/flip`, Q-254, #391) has no source-log row of its own (row 444 names `openftgs._vendor.flip`; the copy was made by the same script from the same reference copy, with no new outside source), NOTICE names openftgs's copy and `tests/reference/flip` but not the harness's, and the harness's wheel ships ꟻLIP's BSD-3-Clause `LICENSE` as package data with no NOTICE of its own; the harness is a separate distribution this release doesn't publish (Q-5);
- row 420 read the README and the deploy prototxt of a gist without a license file, for the VGG16 model's layer names, facts about the model, nothing copied;
- the comment on BVLC's mean pixel above.

**New rows**: T432.1 (Debian's copyright file for libdrm, License scan above) and T432.2 (the tools run for #432's checks).

**Housekeeping (#432).** At the merge of `main` (9dc5ce30): the glossary's "Spec revision" entry now reads revision 549o (revision 549, amended fifteen times in the repository in the owner's name, as 549a to 549o); Q-239 and Q-240 stay after Q-238, unchanged, where #368 moved them, and are not repeated inside `main`'s Calibration data table; every Q-n and source-log row of both sides is kept, `main`'s numbers final, with no renumbering.

## Benchmark baselines

§3.4's CI paragraph (a budgeted figure more than 10 % worse than its stored baseline fails); Q-170, Q-196, Q-245, Q-253. Q-196's refresh is #369's (`docs/modules/cli/benchmarks.md`, "The refresh of M18"; Calibration data, 2026-10-23). Added by #432 on 2026-10-27.

**The stored files after the merge of `main`** (9dc5ce30; Calibration data, 2026-10-27). `B-2-raster.json`, `B-3-slice.json` and `B-5-adam.json` keep #369's per-case medians of three clean runs. `B-4-losses.json`, `B-6-data.json` and `B-6-data-60.json` are `main`'s: `main` recorded them again with ꟻLIP after #369's runs, the two B-6 files in #394's recorded run and B-4 in #427's refresh, after #394's and #380's review's. #369's medians of those three, measured with LPIPS-VGG, are superseded.

**The comparison on the merged tree.** #423 changed the gsplat path's SH gather after #369's runs (Q-274), so the GPU benchmark job's steps (`.github/workflows/gpu-bench.yml`: B-2, B-3, B-4, B-5, B-6 on the CI scene and B-6 on its 60-frame conversion) run once on the merged tree, alone on a quiet machine, and `scripts/bench_compare.py` compares the six files with the stored baselines. Pending: the run needs the machine to itself, and its result is added here and reported to the owner. Replacing a baseline is the owner's call (Q-196).

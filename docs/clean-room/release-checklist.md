# Release checklist

M18's record of the release-time checks that spec §2.3's notes and D-79 call for, for openftgs 0.1.0 (PRD #355): the license scan and its sign-offs, the libraries bundled in dependency wheels, the vendored license files, the pretrained weights' terms, the absence of dataset material, and the source-log audit. Each section gives the commands run, the commit and environment they ran on, and their results; a sign-off is the owner's, given in the build session, and is recorded with its date.

## License scan

To be written by #366.

## Libraries bundled in dependency wheels

To be written by #366.

## The vendored RoMa

To be written by #367.

## Pretrained weights' terms

To be written by #367.

## No dataset material

To be written by #367.

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
| PyTorch's CPU wheel index and its install page (2) | `.github/workflows/ci.yml:33`, `README.md:26` | new row T368.2 |
| The CI workflow's page and status badge (2) | `README.md:3` | the project's own repository; the badge's URL scheme in T368.2 |
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
| lpips 0.1.4, with torchvision's AlexNet and VGG trunks | `losses`, `eval`, harness | Appendix A #10, #34; rows 43, 44, 67, 174 |
| PyYAML; Pillow; SciPy; NumPy | package, scripts, tests | rows 88, 89; rows 74, 183; rows 7, 61, 127, 194; rows 15, 20, 102, 194 |
| fused-ssim (optional) | `losses`, `train` | rows 45, 73 |
| RoMa at 77f8d68, vendored, with what it imports: torchvision, einops, loguru, kornia (inside one helper), xformers (optional), fused-local-corr (`local_corr`) | `src/openftgs/_vendor/romatch`, `openftgs.init` | Appendix A #10, #15, #26, #32, #33; rows 106–126, 139, 156; fused-local-corr row 3 |
| DINOv2, DINO, timm, Patch2Pix, LoFTR, Kornia, InLoc, credited in RoMa's files | NOTICE, `VENDOR.json`, the vendored `LICENSES/` | rows 106–113, 139 (Q-115, Q-118) |
| XFeat through torch.hub (Tiny RoMa), and the XFeat paper named in RoMa's `models/tiny.py` | `openftgs.init`, tests | Appendix A #26; rows 107, 136, 138 |
| pytest, pytest-xdist, pytest-cov and coverage | tests, `pyproject.toml`, CI | rows 79, 196–201 |
| hypothesis (MPL-2.0, documentation only) | tests | row 7 |
| ruff, mypy | `pyproject.toml`, CI | rows 76, 82, 84, 275 |
| pip-licenses | `scripts/license_scan.py`, CI | rows 1, 5 |
| scikit-image, opencv-python-headless and torchmetrics, the tests' oracles | tests | row 46; rows 89, 188; rows 66, 68, 69 |
| matplotlib | harness | row 194 |
| setuptools as the build backend; GitHub Actions | both `pyproject.toml` files; `.github/workflows/` | new row T368.2; row 182 (`gpu-bench.yml`) and T368.2 (`ci.yml`) |
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

Over every tracked file at 7e1faf0, before this section and rows T368.1 and T368.2 were written:

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

`tests/test_clean_room_guard.py` assembles the names it tests from two string halves, so its source holds no hit. No row of the source log names a forbidden source as read: the rows at the boundary are 12 (plyfile, not read), 87 and 237 (license files only), 154, 158 and 245 (gsplat's dynamic module, not opened), 262 (the renderer, not opened under the rule then in force), 270 to 273 (the renderer, under the owner rule of 2026-10-22) and 271 (general knowledge). Row 273's sighting of the framework's import in one file's header is not an incident (PRD #355, owner decision 1; row T368.1), and nothing of the renderer is in the scope. The source log has no row marked **Incident**. The check that no tracked path holds the paper's or the framework's name, a `git ls-files` piped into `grep`, was blocked by the guard as a search it couldn't confirm stays inside the repository, and was not retried (reported to the run's lead).

### New rows

- **T368.1**: the owner's decision on row 273, in the owner's words (housekeeping, below).
- **T368.2**: the general knowledge no earlier row recorded: PyTorch's CPU wheel index and install page, GitHub Actions as `ci.yml` uses it and the workflow badge's URL, the PEP 517, 621 and 639 packaging of both `pyproject.toml` files, `pip wheel` in CI, and the flat badge's look in `scripts/coverage_local.py` (Q-217).

Nothing else in the scope needed a row: every other URL, project and attribution matches a row or Appendix A, or is one of the kinds in the URL table that needs none.

### Housekeeping (#368)

- Q-239 and Q-240 moved from inside the Calibration data table into the Q-n table, after Q-238 and before Q-241 (PRD #355, owner decision 6; commit 071788e). The two rows are 4,332 bytes with SHA-256 `926d2b7637c513c198f1fc4d0f7730626afaf4f6b67269d1fcdae33764b9b4a8` before and after; `questions.md` keeps its size and its lines, and with the two rows taken out the old and new files are identical. The Calibration data table now holds only its own rows.
- Row T368.1 records the owner's decision on row 273 (commit 6f42859).
- The glossary's "Spec revision" entry already read revision 549n, word for word as the PRD gives it (5288b72); it was checked, not changed.

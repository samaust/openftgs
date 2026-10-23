# Release checklist

M18's record of the release-time checks that spec §2.3's notes and D-79 call for, for openftgs 0.1.0 (PRD #355): the license scan and its sign-offs, the libraries bundled in dependency wheels, the vendored license files, the pretrained weights' terms, the absence of dataset material, and the source-log audit. Each section gives the commands run, the commit and environment they ran on, and their results; a sign-off is the owner's, given in the build session, and is recorded with its date.

## License scan

To be written by #366.

## Libraries bundled in dependency wheels

To be written by #366.

## The vendored RoMa

D-65; §2.3 notes ("checks the vendored directory's license files"); Q-113, Q-115 and Q-118, whose detecting test is this audit. Run on 2026-10-22 by ticket #367, on commit 7e1faf0 of `prd-355-release`, with the project's `.venv` (Python 3.14.8) and git 2.43.0.

**The upstream clone.** A blobless clone of RoMa (MIT), `git clone --filter=blob:none --no-checkout https://github.com/Parskatt/RoMa <clone>`, into a new, empty folder, `<clone>` = `.scratch/t367-roma/RoMa` inside the ticket's worktree (gitignored; the session's tooling refused git commands on the main checkout's `.scratch/`). Its HEAD is still 77f8d68803526dcddfd9b7a46bc76125bdc25f15 (2026-01-23). Its `LICENSE` was read before anything else: MIT, Copyright (c) 2023 Johan Edstedt, git blob ca95157, the blob of source log row 106. Source log row T367.1.

**The vendoring script's check mode.** Run from the worktree, not from the clone, with the clone's path as an argument:

```
.venv/bin/python -I scripts/vendor_roma.py --source <clone> --check
```

It rebuilds the copy in a temporary folder from the 22 inference files at 77f8d68 (each blob id checked against the commit's tree) and the eight license texts, fetched again at their pinned commits and checked against their SHA-256 (source log row T367.2), and compares every file with `src/openftgs/_vendor/romatch`. Result, exit 0:

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

## Pretrained weights' terms

§2.3 notes: pretrained weights are downloaded at run time and never redistributed, and their terms are recorded here, apart from the code licenses; XFeat's license is checked before `RomaInitConfig.model = "tiny_roma_v1_outdoor"` is used. Q-7 (CI never downloads weights: `weights` tests run only with `OPENFTGS_ALLOW_DOWNLOADS=1`), Q-116, Q-123 and Q-198 (the files `roma_init` loads). Recorded on 2026-10-22 by ticket #367 from each project's own statement of its terms, every read logged (source log rows T367.1 and T367.3 to T367.8). The files openftgs fetches are those its code names: lpips's `LPIPS(net=…)` in `openftgs.losses.LPIPSLoss`, and the vendored `model_zoo.weight_urls` through `openftgs.init`'s `_weight_urls` (Q-198), which never asks for `roma_indoor.pth`.

| Weight | Where it comes from | When it is fetched | Terms, from the project's own statement | Redistributed by openftgs |
| --- | --- | --- | --- | --- |
| LPIPS linear layers, `lpips/weights/v0.1/alex.pth` and `vgg.pth` | inside the lpips 0.1.4 wheel from PyPI, a core dependency (§2.3) | not downloaded at run time: installed with lpips, and loaded when an LPIPS network is built (training's perceptual loss when λ_perc > 0, `ftgs eval`'s LPIPS metrics) | the lpips wheel's `LICENSE`: BSD-2-Clause, Copyright (c) 2018 Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang; its README (the wheel's METADATA) states no separate terms for the weights, which it ships as part of the package | no: openftgs depends on lpips and ships no weight file |
| AlexNet ImageNet trunk, `alexnet-owt-7be5be79.pth` | https://download.pytorch.org/models/alexnet-owt-7be5be79.pth, torchvision's `AlexNet_Weights.IMAGENET1K_V1` (torchvision 0.29.1) | downloaded by torchvision into torch.hub's checkpoint cache, when not cached, whenever lpips builds an AlexNet trunk (lpips 0.1.4 passes `pretrained=True`): by `openftgs.eval.Evaluator`, which builds both backbones, so at the first `ftgs eval`, the trainer's evaluation or `ftgs bench render`, whatever backbone the scene's metrics report; or by training's perceptual loss when λ_perc > 0 and `perc_net` is `alex` | torchvision's README, "Pre-trained Model License": the pre-trained models "may have their own licenses or terms and conditions derived from the dataset used for training", and it is the user's responsibility to determine whether they may use them (only SWAG models are named, as CC-BY-NC-4.0; these are not SWAG). torchvision's BSD-3-Clause license covers its code. No affirmative grant for these weights: their terms follow ImageNet-1k, their training data | no |
| VGG16 ImageNet trunk, `vgg16-397923af.pth` | https://download.pytorch.org/models/vgg16-397923af.pth, torchvision's `VGG16_Weights.IMAGENET1K_V1` | as AlexNet's: by the `Evaluator`, with it, or by training's perceptual loss when `perc_net` is `vgg` | as AlexNet's: torchvision's disclaimer, no affirmative grant, ImageNet-1k training data | no |
| RoMa outdoor, `roma_outdoor.pth` (445,647,516 bytes, SHA-256 c7a45c80…, source log row 132) | https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth, a release of RoMa's author's storage repository | downloaded by `roma_init`'s default matcher (`ftgs init roma`, model `roma_outdoor`) through `torch.hub.load_state_dict_from_url` into torch.hub's checkpoint cache, when not cached | RoMa's README (77f8d68), "License": all its code except DINOv2 is MIT, DINOv2 Apache-2.0; it states nothing about the weights. The repository that hosts them, Parskatt/storage at 6fed440, has an MIT `LICENSE`, Copyright (c) 2022 Johan Edstedt, and no README | no |
| DINOv2 ViT-L/14, `dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes, SHA-256 d5383ea8…, source log row 132) | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth, the URL DINOv2's README gives for the ViT-L/14 backbone | with RoMa outdoor, as RoMa's backbone | DINOv2's README (7764ea0), "License": "DINOv2 code and model weights are released under the Apache License 2.0"; its `LICENSE` is the Apache-2.0 text vendored as `LICENSES/DINOv2.txt`. The same README puts the XRay-DINO and Cell-DINO weights under the FAIR Noncommercial Research License; openftgs fetches neither | no |
| Tiny RoMa v1 outdoor, `tiny_roma_v1_outdoor.pth` (11,432,107 bytes, SHA-256 3e2ac348…, source log row 135) | https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth | downloaded by `roma_init`'s default matcher when `RomaInitConfig.model = "tiny_roma_v1_outdoor"`, as RoMa outdoor's; it supplies every XFeat weight Tiny RoMa keeps (Q-123) | as RoMa outdoor's: RoMa's README is silent on weights; Parskatt/storage is MIT, Copyright (c) 2022 Johan Edstedt | no |
| XFeat, through torch.hub | the archive https://github.com/verlab/accelerated_features/zipball/e92685f57f8318b18725c5c8c0bd28c7fe188d9a, loaded with `pretrained=False` (Q-116, Q-123) | downloaded into torch.hub's cache when Tiny RoMa is selected, for XFeat's code | XFeat's `LICENSE` at e92685f: Apache-2.0 (SHA-256 c71d239d…, source log row 107, fetched again with the same hash); its README's "License" section points to it. openftgs loads no XFeat weights of its own: they come from Tiny RoMa's checkpoint (Q-123). The archive does carry the repository's `weights/xfeat.pt` and `weights/xfeat-lighterglue.pt` (names in torch.hub's cache, source log row T367.8), under the same Apache-2.0, unused | no |

**XFeat's terms are permissive** (Apache-2.0, for its code and the repository's files), so the owner's decision the ticket reserves for a non-permissive result doesn't arise: `RomaInitConfig.model = "tiny_roma_v1_outdoor"` stays as it is.

**Never redistributed.** openftgs's wheel and sdist hold no weight file: no `.pth`, `.pt`, `.ckpt`, `.bin` or `.safetensors` in either (the listings of "No dataset material" below). The weights reach a user's machine only from the projects' own hosts, at run time, or with the lpips wheel from PyPI.

**For the owner's attention (no sign-off asked).** Only the AlexNet and VGG16 trunks have no affirmative grant: torchvision's statement is a disclaimer that leaves their terms to their training data. The release notes' account of the weights' terms should say so.

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

uv 0.11.32, build 1.6.1, setuptools 84.2.1 in the isolated build environments (source log row T367.9). Built: `openftgs-0.1.0.dev0.tar.gz` (563,382 bytes, SHA-256 0f175d34a7860c1ec211ba92cd4059a7de9ea7b4b45e3ee418e3537b8a828e2d) and, from it, `openftgs-0.1.0.dev0-py3-none-any.whl` (276,732 bytes, SHA-256 ac53bb18ad9c4cdf94671f77d3b6971853873cb0b8ab20f0b1a16d125fbfad7d). Listed with `tar tzf` and `python -m zipfile -l`, and compared with `git ls-files`:

- **sdist**, 109 files: `LICENSE`, `NOTICE`, `README.md`, `pyproject.toml`, the package under `src/openftgs/` (the vendored RoMa's license files and `VENDOR.json` included), and setuptools' default `tests/test*.py` (42 test modules, without `conftest.py`, `builders.py`, `tests/synthetic/` or `tests/harness/`). Every file is a tracked file except setuptools' generated `PKG-INFO`, `setup.cfg` and `src/openftgs.egg-info/*` (6).
- **wheel**, 62 files: the package under `openftgs/`, the vendored RoMa's license files and `VENDOR.json` included, and the 7 files of the `.dist-info`, with `licenses/LICENSE` and `licenses/NOTICE`. Its 55 package files are the sdist's 55 files under `src/openftgs/`, each a tracked file.
- **Neither** holds a placeholder, a `docs/` path, a `data/`, `datasets/` or `runs/` path, a `__pycache__` or `.pyc` file, any file with the media, data, archive, checkpoint or YAML extensions of the name search, or a file named after a dataset or scene. The dataset names inside them are those of the tracked files they copy (the converters and the test modules, above).

### The S-n fixtures' provenance

The only data files in the repository and its history. `tests/synthetic/data/s1_gt.safetensors`, `s2_gt`, `s3_gt`, `s4_gt` and `s6_gt` are ftgs-if exchange models (§2.2; 1, 8, 12, 240 and 24 Gaussians, SH degree 0) of §3.2's synthetic scenes, written by `tests/synthetic/make_ground_truth.py` with `openftgs.io.write_exchange` from the seeds in `tests/synthetic/data/seeds.json` and `synthetic_scenes.py`'s generators; their metadata's producer is "openftgs tests (M11 ground truth)", or M13 for `s4_gt`. `test_committed_ground_truth_reproduces_from_its_seeds` (`tests/synthetic/test_synthetic_recovery.py`) checks that the committed files are what the seeds give, and passed on this commit. SHA-256: s1 39ece2ad…, s2 8b4927ec…, s3 bfef9f4b…, s4 888a5770…, s6 fb0efad4…, seeds.json 2063ebb8…. They derive from no dataset. The other S-n scenes are generated at test time: S-7 and S-8 from S-2's ground truth and their seeds, S-5, S-9 and S-11 by the analytic ray caster (`tests/synthetic/raycast.py`) from their seeds, S-10 from a tiny scene built in the test; the retired `s5_gt` was a generated exchange model like the others.

### The spec's observations of SelfCap's calibration files

The spec records what the owner saw in SelfCap's files: §1.4's SelfCap protocol ("Camera files, checked on hair-calib and yoga-calib and on the corgi listing", the sync.json offset ranges, the calibration source, the point clouds' numbering, the corgi archive's CRC failure), D-59, D-63 and D-64, and Appendix A rows 30, 31 and 38. They are the owner's text, not data: file and key names, matrix shapes, image sizes, value ranges and agreement bounds (offsets from −0.046 to +0.026 s, Rot against Rodrigues(R) to 3·10⁻⁸, cosines ≥ 0.64), counts and dates, written as prose. Beyond the few values that prose quotes (the 2160 × 3840 size, k3 = 0, hair's t\_ = 0, n\_ = 1 and f\_ = 20, the ranges and bounds), the checks above found no calibration matrix, frame, mask or point cloud of SelfCap in the repository, its history or the built packages. `questions.md`, the module docs and the tests repeat that prose, or build files in that layout with invented values (above). The owner's copies of hair-calib and yoga-calib, and the corgi listing, were never committed.

## Source-log audit

To be written by #368.

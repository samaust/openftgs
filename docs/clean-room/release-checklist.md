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
| AlexNet ImageNet trunk, `alexnet-owt-7be5be79.pth` | https://download.pytorch.org/models/alexnet-owt-7be5be79.pth, torchvision's `AlexNet_Weights.IMAGENET1K_V1` (torchvision 0.29.1) | downloaded by torchvision into torch.hub's checkpoint cache the first time an LPIPS-Alex network is built (lpips 0.1.4 passes `pretrained=True`) | torchvision's README, "Pre-trained Model License": the pre-trained models "may have their own licenses or terms and conditions derived from the dataset used for training", and it is the user's responsibility to determine whether they may use them (only SWAG models are named, as CC-BY-NC-4.0; these are not SWAG). torchvision's BSD-3-Clause license covers its code. No affirmative grant for these weights: their terms follow ImageNet-1k, their training data | no |
| VGG16 ImageNet trunk, `vgg16-397923af.pth` | https://download.pytorch.org/models/vgg16-397923af.pth, torchvision's `VGG16_Weights.IMAGENET1K_V1` | as AlexNet's, the first time an LPIPS-VGG network is built | as AlexNet's: torchvision's disclaimer, no affirmative grant, ImageNet-1k training data | no |
| RoMa outdoor, `roma_outdoor.pth` (445,647,516 bytes, SHA-256 c7a45c80…, source log row 132) | https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth, a release of RoMa's author's storage repository | downloaded by `roma_init`'s default matcher (`ftgs init roma`, model `roma_outdoor`) through `torch.hub.load_state_dict_from_url` into torch.hub's checkpoint cache, when not cached | RoMa's README (77f8d68), "License": all its code except DINOv2 is MIT, DINOv2 Apache-2.0; it states nothing about the weights. The repository that hosts them, Parskatt/storage at 6fed440, has an MIT `LICENSE`, Copyright (c) 2022 Johan Edstedt, and no README | no |
| DINOv2 ViT-L/14, `dinov2_vitl14_pretrain.pth` (1,217,586,395 bytes, SHA-256 d5383ea8…, source log row 132) | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth, the URL DINOv2's README gives for the ViT-L/14 backbone | with RoMa outdoor, as RoMa's backbone | DINOv2's README (7764ea0), "License": "DINOv2 code and model weights are released under the Apache License 2.0"; its `LICENSE` is the Apache-2.0 text vendored as `LICENSES/DINOv2.txt`. The same README puts the XRay-DINO and Cell-DINO weights under the FAIR Noncommercial Research License; openftgs fetches neither | no |
| Tiny RoMa v1 outdoor, `tiny_roma_v1_outdoor.pth` (11,432,107 bytes, SHA-256 3e2ac348…, source log row 135) | https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth | downloaded by `roma_init`'s default matcher when `RomaInitConfig.model = "tiny_roma_v1_outdoor"`, as RoMa outdoor's; it supplies every XFeat weight Tiny RoMa keeps (Q-123) | as RoMa outdoor's: RoMa's README is silent on weights; Parskatt/storage is MIT, Copyright (c) 2022 Johan Edstedt | no |
| XFeat, through torch.hub | the archive https://github.com/verlab/accelerated_features/zipball/e92685f57f8318b18725c5c8c0bd28c7fe188d9a, loaded with `pretrained=False` (Q-116, Q-123) | downloaded into torch.hub's cache when Tiny RoMa is selected, for XFeat's code | XFeat's `LICENSE` at e92685f: Apache-2.0 (SHA-256 c71d239d…, source log row 107, fetched again with the same hash); its README's "License" section points to it. openftgs loads no XFeat weights of its own: they come from Tiny RoMa's checkpoint (Q-123). The archive does carry the repository's `weights/xfeat.pt` and `weights/xfeat-lighterglue.pt` (names in torch.hub's cache, source log row T367.8), under the same Apache-2.0, unused | no |

**XFeat's terms are permissive** (Apache-2.0, for its code and the repository's files), so the owner's decision the ticket reserves for a non-permissive result doesn't arise: `RomaInitConfig.model = "tiny_roma_v1_outdoor"` stays as it is.

**Never redistributed.** openftgs's wheel and sdist hold no weight file: no `.pth`, `.pt`, `.ckpt`, `.bin` or `.safetensors` in either (the listings of "No dataset material" below). The weights reach a user's machine only from the projects' own hosts, at run time, or with the lpips wheel from PyPI.

**For the owner's attention (no sign-off asked).** Only the AlexNet and VGG16 trunks have no affirmative grant: torchvision's statement is a disclaimer that leaves their terms to their training data. The release notes' account of the weights' terms should say so.

## No dataset material

To be written by #367.

## Source-log audit

To be written by #368.

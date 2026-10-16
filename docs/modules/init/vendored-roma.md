# The vendored RoMa in `openftgs._vendor.romatch`

How D-65 is met, as Q-113, Q-114 and Q-115 amend it: RoMa's inference code at commit 77f8d68, copied into the package so that no install needs OpenCV, xformers or RoMa's training dependencies, and so that matching stays reproducible across RoMa releases.

## The file set

The 22 inference files of D-65: the package's `__init__`, the models (model zoo, matcher, encoders, Tiny RoMa, the transformer with its DINOv2 layers) and the utilities (`utils`, `kde`, `local_correlation`). Benchmarks, datasets, losses, training and checkpointing are left out, and so is `utils/transforms.py`, which nothing in the set imports.

## The patch

The vendoring script, `scripts/vendor_roma.py`, builds the copy from a git clone holding the commit (a blobless clone will do), checking each file's git blob id against the commit's tree; nothing in the copy is edited by hand. It locates every change with Python's ast and makes these, and no others:

- the 14 absolute `romatch` imports, 12 at module level and 2 inside functions, point to `openftgs._vendor.romatch` (D-65);
- `utils/utils.py` loses its module-level cv2 import and its three pose-estimation helpers, removed by name with the lines between them, which include a credit to an LGPL-2.1 project; `utils/__init__.py` stops importing them (Q-113);
- the three DINOv2 layers that import xformers keep only the fallback branch of that import, so attention always runs on PyTorch's `scaled_dot_product_attention` (Q-114);
- each modified file gets one line saying it was modified, pointing to VENDOR.json, after the file's leading comment lines (Meta's header in the DINOv2 layers) or else at line 1 (Q-115).

The script prints file names, counts, function names and hashes, never source text and never a diff. `python scripts/vendor_roma.py --source CLONE` rewrites the copy; with `--check` it rebuilds the copy in a temporary folder and compares it with the committed one file by file, VENDOR.json included. Both fetch the license texts, so they need network. A reviewer checks the hashes and the patch descriptions, not the upstream text of what was removed.

No one reads `romatch/utils/utils.py` upstream above line 77 (`def unnormalize_coords`): the lines above it hold the code credited to the LGPL-2.1 project, and anything there is found by name with ast.

## VENDOR.json and the license files

VENDOR.json records the upstream URL and commit, the script, each license file with its source and SHA-256, and per vendored file its upstream git blob id, the SHA-256 of the vendored bytes and its patch as a plain description. RoMa's MIT license sits at the copy's root as `LICENSE`; a `LICENSES` folder beside it holds the texts of DINOv2, DINO, timm, LoFTR, Patch2Pix and Kornia, one `<project>.txt` each, which the script fetches from raw.githubusercontent.com at pinned commits and checks against the SHA-256 of source log rows 106 and 108–113. NOTICE names them all (Q-115). The files ship in the wheel, and CI's wheel job checks for them.

## U-38

U-38's CPU tests, in `tests/test_vendored_roma.py`, check the hashes against VENDOR.json, search the copy for absolute `romatch` imports, and import the copy in a subprocess with cv2 blocked and a stub xformers on the path (Q-117). Its comparison with upstream runs locally, on a GPU with the weights downloaded, against upstream romatch at 77f8d68 made importable beside the copy: the vendored and upstream `roma_outdoor`, built from the same weights with `use_custom_corr` off, match two seeded textured-plane pairs under the flags `roma_init` sets, and their warps and certainties must agree within 1e-5 (they were bitwise equal on the RTX 4090, #137).

The test is skipped unless `import romatch` works and the files it finds have the upstream git blob ids VENDOR.json records, so another RoMa version can't stand in. Nothing needs installing; upstream's own package can go on pytest's path for that run only, from a clone of RoMa (a blobless one will do, and upstream needs OpenCV, which the dev environment has):

```sh
git -C CLONE archive -o romatch-77f8d68.tar 77f8d68 romatch
mkdir UPSTREAM && tar -xf romatch-77f8d68.tar -C UPSTREAM
OPENFTGS_ALLOW_DOWNLOADS=1 pytest -o "pythonpath=src UPSTREAM" tests/test_vendored_roma.py
```

This copies upstream's `utils/utils.py` whole, the lines above 77 included, so nobody opens that copy either; the test only hashes its bytes.

## Updating

Only by re-running the script at a new commit, after the attribution scan and license checks that this copy had (source log rows 106–120), and passing U-38 again (D-65).

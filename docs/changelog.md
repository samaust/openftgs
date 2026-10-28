# Changelog

Every release of openftgs, newest first. Versions follow the package's metadata; each release is
tagged `v<version>` in the repository.

## 0.1.0

The first release of openftgs, an independent clean-room implementation of FreeTimeGS (Wang et
al., CVPR 2025), not affiliated with its authors. It implements revision 549o of the owner's
specification, milestones M0 to M18 of its plan. Its development status is alpha.

### What the release contains

- **The library**, the import package `openftgs`, with its 15 public modules in the bands of
  spec §2.1: `types`, `config`, `io`, `data` and `init`; `motion`, `sh`, `model` and `render`;
  `optim`, `losses` and `density`; `eval` and `train`; and `cli`. The {doc}`api/index` documents
  their public API; everything else is private.
- **The model and its training:** Gaussian primitives that each live at their own time and
  position, move with their own velocity and fade with a temporal opacity, rendered by a
  reference backend in PyTorch or by gsplat 1.5.3's stage functions; the losses of E7, the
  optimizer and its schedules, gradient statistics with growth and relocation, and checkpoints
  that resume. {doc}`method` restates the method from the specification.
- **The command line `ftgs`:** `convert` (Neural3DV's `poses_bounds.npy`, OpenCV-style YAML and
  SelfCap-style archives into a scene manifest), `cache`, `init roma`, `train`, `render`, `eval`,
  `export` and `bench`, each with its help text ({doc}`cli`). The second executable,
  `ftgs-adapter`, serves the differential harness.
- **Initialization** from RoMa's dense matches, with a vendored copy of RoMa's inference code
  (MIT, with DINOv2's layers under Apache-2.0), in the `[init]` extra, which brings torchvision,
  einops and loguru, and tqdm for the code of XFeat that Tiny RoMa loads through torch.hub.
- **Evaluation:** PSNR, SSIM, DSSIM and ꟻLIP on a scene's test views, written to METRICS.json,
  with the dynamic-region and fastest-frames metrics where a scene provides them.
- **The perceptual network:** ꟻLIP (LDR) is the default perceptual network and the reported
  perceptual metric, and D-89's seven alternatives are selectable as `loss.perc_net`, each with
  its calibrated weight: `haarpsi`, `vgg16_oxford`, `alexnet_bvlc`, `dinov2_s14`, `dinov2_b14`,
  `dino_s16` and `dino_b16`. ꟻLIP is a vendored copy of NVIDIA's PyTorch loss (BSD-3-Clause), and
  HaarPSI is adapted from piq 0.8.0 (Apache-2.0).
- **This documentation site**, with the installation, a quickstart on openftgs-synth's
  `kitchen`, the datasets, the method, the command line and the API reference.
- **NOTICE** names the third-party code the package includes: the vendored RoMa and the code it
  credits, ꟻLIP (BSD-3-Clause) and piq's HaarPSI (Apache-2.0), and gives the attribution that
  Oxford's VGG16 weights ask for under CC BY 4.0.

The differential harness, the distribution `ftgs-harness` in the repository's `harness/` folder,
is not published (Q-5).

### Validation data (D-79)

The three datasets the paper evaluates on are licensed for non-commercial research only, and
openftgs may later be used commercially, so this project uses none of them. It validates on
openftgs-synth, the owner's CC0-1.0 rendered multi-view video dataset, with scenes in the three
input layouts the converters read and exact ground truth. The converters stay as specified, for
users who hold the paper's datasets. Therefore:

- The project validated on openftgs-synth.
- D-28's LLFF column order, ENeRF-Outdoor's file layout and D-59's sync-offset reading are
  unverified against the real files.
- No result is compared with the paper's tables.

The validation runs, with their commands and figures, are recorded in
{repo}`docs/clean-room/validation-runs.md`. On `kitchen`, the default training took 14 min 22 s
on one RTX 4090, its test-view evaluation included, and scored a PSNR of 28.32 dB and a ꟻLIP of
0.0900 on its test camera, `cam00`.

### Known limitations

- **No LPIPS.** openftgs reports no LPIPS, so its perceptual figures can't be compared with the
  paper's perceptual columns (E16). ꟻLIP takes LPIPS's place in the loss and the metrics since
  revision 549o.
- **Checkpoints saved before revision 549o** can be read, by `ftgs eval`, `ftgs render`,
  `ftgs export` and `ftgs bench render --model`, with their legacy `perc_net` and `lambda_perc`
  replaced by 549o's defaults in memory, but training can't resume from them (Q-247).
- **METRICS.json is version 2** (Q-256): psnr, ssim1, ssim2, dssim1, dssim2 and flip, with no
  LPIPS keys. Files of version 1, written before 549o, are still read by the repository's
  run-summary script and render viewer.
- **Training needs a CUDA build of PyTorch.** On Linux x86_64 PyPI's default torch wheel is one;
  for another CUDA version, or on Windows, install torch from the PyTorch index first (Q-3).
  openftgs is developed and validated on Linux with one RTX 4090, and CI runs the CPU suite on
  Linux only.
- **Two optional speed-ups are installed by hand.** fused-ssim is not on PyPI, so the `[fast]`
  extra is empty (Q-2); RoMa's fused-local-corr is not in `[init]` and is built from source
  against the installed torch (D-85). {doc}`installation` gives both commands. Without them the
  same computations run in plain PyTorch.
- **ffmpeg and COLMAP are separate programs** the user installs (D-48, D-60); openftgs neither
  bundles nor links them.

### Pretrained weights' terms

openftgs includes no pretrained weights and never redistributes them: each is downloaded at run
time from its publisher, into torch.hub's cache, only when a feature needs it. ꟻLIP and HaarPSI
have no weights, so the default configuration downloads nothing for its perceptual network. The
terms below are each publisher's own statement, as the release checklist,
{repo}`docs/clean-room/release-checklist.md`, records them.

| Weight | Fetched when | Terms |
| --- | --- | --- |
| Oxford's VGG16 (the Visual Geometry Group's Caffe release) | `loss.perc_net = "vgg16_oxford"` | Creative Commons Attribution 4.0 (CC BY 4.0), with attribution by citing Simonyan and Zisserman, arXiv:1409.1556; NOTICE gives it |
| BVLC's AlexNet (Caffe's model zoo) | `loss.perc_net = "alexnet_bvlc"` | released "for unrestricted use", with no license named |
| DINOv2 ViT-S/14 and ViT-B/14 | `loss.perc_net = "dinov2_s14"` or `"dinov2_b14"` (needs `[init]`) | DINOv2's README: code and model weights under the Apache License 2.0 |
| DINO ViT-S/16 and ViT-B/16 | `loss.perc_net = "dino_s16"` or `"dino_b16"` (needs `[init]`) | DINO's README puts its repository under the Apache License 2.0 and doesn't name the weights it hosts |
| RoMa outdoor, with its DINOv2 ViT-L/14 backbone | `ftgs init roma` with the default model | RoMa's README states nothing about its weights; the repository hosting them is MIT; DINOv2's weights are under the Apache License 2.0 |
| Tiny RoMa v1 outdoor, and XFeat's code through torch.hub | `RomaInitConfig.model = "tiny_roma_v1_outdoor"` | as RoMa outdoor's; XFeat's repository is under the Apache License 2.0 |

Two of these statements are weaker than a named license: BVLC's AlexNet names none, and DINO's
README doesn't name the weights. Check that each fits your use before you select it.

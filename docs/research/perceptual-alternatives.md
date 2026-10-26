# Perceptual loss and metric: alternatives to LPIPS with licensed weights

A research document, written from 2026-10-23 on PRD #373 (which absorbed issue #372), on replacing LPIPS in openftgs's two uses of it, the training loss and the evaluation metric, with something whose pretrained weights carry an explicit permissive grant from their publisher, or on keeping LPIPS. This version holds phase 1, the desk survey (sections 1 to 4), the test plan of PRD #374 (section 5), which replaced #373's phases 2 and 3 after the owner's checkpoint 1, and its results (section 6); sections 7 to 9 (the recommendation, what adopting it would change, the sources) are #373's phase 4. The document proposes no code and changes no spec item: adopting a replacement is a spec change the owner makes (E7, E16, D-9, D-29, D-68 and §2.1's `perc_net`), followed by its own implementation issue.

**Owner decisions behind the work** (2026-10-23, in session; PRD #373):

- Scope: both uses, the training loss (E7's L_perc, D-9) and the evaluation metric (E16, D-29, D-68).
- The bar: the weights' own publisher states a permissive license that allows commercial use, as DINOv2's README does for its weights. The training data is recorded, not judged.
- Depth: a survey, a test plan, and the tests themselves (training runs on openftgs-synth's `kitchen`).
- Timing: now, before #370 and step 7 of PRD #355, which wait for the owner's decision.

**Words.** A **trunk** is the pretrained network whose features a perceptual distance compares (LPIPS's VGG16 or AlexNet). **Calibrated** means per-channel weights learned on human judgments sit on top of the trunk (LPIPS's linear layers, "lin"); **uncalibrated** means every channel counts equally (lpips's `lpips=False`). The **publisher** of a set of weights is whoever trained and released those files; a **porter** converted someone else's files to another framework. A **grant** is a license statement that covers the weights themselves, not only a repository's code. BAPPS 2AFC is the two-alternative forced-choice agreement with human judgments on the LPIPS paper's dataset (human ceiling 73.9 %); SRCC is the Spearman rank correlation with mean opinion scores on TID2013, KADID-10k or PIPAL. A **candidate** is a loss or metric under test, not the glossary's growth candidate; **checkpoints 1 and 2** are PRD #373's review points with the owner, not ftgs checkpoints.

**Clean-room record.** Every page, license file, model card, search, skipped hit and general-knowledge claim behind this document has a row in `docs/clean-room/source-log.md`, with temporary IDs T373.1 onward, numbered at the merge into `main`. Each project's license file was read before anything else of it, except where its row says otherwise: a README read just before its MIT file (T373.20), and repository pages read for their sidebar license and README where no license file sat at the usual path (T373.54, T373.55, T373.60, T373.61, T373.75); a README is documentation, which every license class allows. Source code was read only under permissive licenses: in the survey for identifiers and provenance comments (T373.6, T373.16, T373.17), and in the tests for Caffe's field numbers (T373.89), piq's signatures and conventions (T373.97, T373.101) and ꟻLIP's PyTorch loss, copied unchanged with its header into the uncommitted prototype (T373.92); pyiqa (non-commercial, T373.25), PieAPP (non-commercial, T373.54), and the unlicensed Watson-DFT and LASI repositories (T373.55, T373.75) were read through their documentation only. No search named the paper's method; no hit looked like an implementation of it; nothing was taken from the paper's own code links. The image-quality papers drawn on here (the rows marked image-quality or perceptual-metric literature) are neither papers the paper cites nor Gaussian-splatting literature, the classes CLAUDE.md allows; they were read at PRDs #373's and #374's direction, and the owner accepted them after the fact on 2026-10-25, at the merge of #374 (T373.103).

## 1. Scope and the owner's decisions

The release checklist's "Pretrained weights' terms" (#367, on `prd-355-release`) found that LPIPS's own linear layers are BSD-2-Clause, inside the lpips wheel, while its ImageNet trunks, torchvision's `vgg16-397923af.pth` and `alexnet-owt-7be5be79.pth`, have no affirmative grant: torchvision's README disclaims and leaves their terms to their training data (re-read for this survey, T373.2). Every other weight openftgs fetches (RoMa, DINOv2, Tiny RoMa, XFeat) has a publisher's permissive statement. The perceptual network is thus the one grey area over commercial use, in two places:

- **The loss.** E7's L_perc, λ_perc = 0.01 [P §3.3], LPIPS-VGG on inputs mapped to [−1, 1] [D-9], built by the trainer whenever λ_perc > 0.
- **The metric.** E16's LPIPS-Alex and LPIPS-VGG [P §4], with the unmapped `lpips_vgg_raw` [D-68], built by `openftgs.eval.Evaluator` for every evaluation, whatever the scene.

The survey screens candidates for both uses against the owner's bar first and for fitness second, in the PRD's order: LPIPS with re-licensed trunks (section 3.1), weight-free measures (3.2), trunks whose weights carry the publisher's grant (3.3), learned full-reference metrics (3.4), with the evidence of fitness collected in the human-alignment column and in section 3.5. Section 4 names the finalists; section 5 is the test plan that would measure them on kitchen.

## 2. How LPIPS is used and what the spec fixes

### 2.1 The spec items

| Item | What it fixes | Tag |
| --- | --- | --- |
| E7 | L_render = 0.8·L1 + 0.2·(1 − SSIM) + 0.01·L_perc; L_perc is LPIPS-VGG on inputs mapped from [0, 1] to [−1, 1]; Î is not clamped (it can exceed 1) | [P §3.2, Eq. 5; weights P §3.3]; D-9 |
| D-9 | LPIPS-VGG as the loss backbone, alternatives AlexNet and SqueezeNet | decision |
| E16 | PSNR, SSIM_k, DSSIM_k, and LPIPS: AlexNet for Neural3DV, VGG for ENeRF-Outdoor and SelfCap (the captions of the paper's Tables 6–11) | [P §4] |
| D-29 | LPIPS-Alex for Neural3DV, VGG otherwise | decision |
| D-67 | renders clamped to [0, 1] before metrics; per-image means, then per scene | decision |
| D-68 | both backbones on [−1, 1] inputs, and LPIPS-VGG also on unmapped [0, 1] inputs as `lpips_vgg_raw` | decision |
| §2.1 | `LPIPSLoss(net: Literal["vgg", "alex"] = "vgg")`; `total_loss(…, perceptual: LPIPSLoss \| None)`; `LossConfig.perc_net: Literal["vgg", "alex"] = "vgg"` | signatures |
| §2.3 | lpips 0.1.4 (BSD-2-Clause) and the torchvision networks it loads, core; torchmetrics' LPIPS as U-23's oracle, dev only; pretrained weights downloaded at run time, never redistributed, their terms recorded apart from the code licenses | table and note |
| §2.4, D-52 | the perceptual loss's row of "Where the time goes"; TF32 convolutions in training, bfloat16 LPIPS opt-in, IEEE float32 for reported metrics | design |
| U-9, U-23 | LPIPS(I, I) = 0, the [−1, 1] mapping; the LPIPS backbone per dataset and both input ranges against torchmetrics | tests |

E7's weights and E16's metric list are [P]: the paper fixes LPIPS by name in both, so any replacement departs from the paper and from comparability with its tables, which is why adopting one is the owner's spec change and not a Q-n gap.

### 2.2 The code

- **`openftgs.losses.LPIPSLoss`** wraps lpips 0.1.4's `LPIPS(net=…)`, frozen and kept in evaluation mode (the linear layers hold dropout; Q-41), and takes `forward(pred, gt, *, map_inputs=True)` on `[..., H, W, 3]` float images of one shape, dtype and device, viewed batch-first, mapped to [−1, 1] unless `map_inputs=False` (Q-58), and returns the batch mean. This is the contract a candidate must meet (Q-41).
- **`openftgs.losses.total_loss`** adds `λ_perc · perceptual(pred, gt)` when λ_perc > 0, and checks that the network's private `_backbone` equals `cfg.perc_net` (Q-41). A candidate injected for a test run (section 5) must therefore carry `_backbone` set to the `net` it was built with.
- **`openftgs.train`.** `Trainer.__init__` builds the network once, when λ_perc > 0, through the private factory `_perceptual_network(cfg.perf)(cfg.loss.perc_net)`: a bfloat16 subclass under `perc_precision = "bf16"` (Q-172), an NCHW-input subclass where B-4 measured it faster (Q-223), or plain `LPIPSLoss`. This factory is the seam PRD #373 approved for injecting a candidate without changing the package.
- **`openftgs.eval.Evaluator`** always builds `LPIPSLoss("alex")` and `LPIPSLoss("vgg")` and reports `lpips_alex`, `lpips_vgg` and `lpips_vgg_raw` on renders clamped to [0, 1]; `lpips_backbone(source)` picks the headline one per dataset (D-29). The harness's L3 computes its own LPIPS through its `[lpips]` extra (Q-201).

### 2.3 What LPIPS costs today

B-4's second run (Calibration data, #326) times LPIPS-VGG's forward on both images plus the prediction's backward, on the adopted path (TF32, NCHW inputs, cuDNN autotuning), at **92.52 ms at 1352×1014 and 138.81 ms at 1920×1080** on the RTX 4090; fused-ssim's SSIM is 0.41 and 0.66 ms, PyTorch's SSIM path 8.54 and 13.22 ms. B-1 on kitchen's trained model puts the perceptual loss at about 82 % of a 118.3 ms iteration (forward 44.2 %, the losses' backward 37.7 %). Two consequences for the survey: a cheaper loss shortens training several-fold, and a run without the perceptual term (λ_perc = 0) is about a fifth of an LPIPS run's iteration time once the model is grown. These are the anchors of every cost estimate below.

## 3. The screening table

One table, split by family, with the same columns throughout. Each cell's source is the source-log row cited in it; "est." marks a cost estimated by the method of section 3.6, not measured. Costs are the forward on both images plus the prediction's backward, at 1352×1014 / 1920×1080, on the RTX 4090, against LPIPS-VGG's measured 92.5 / 138.8 ms. **Bar** is the owner's bar on the weights; **fit** is fitness for the two uses.

### 3.1 LPIPS and its trunks

**Where torchvision's trunks came from.** Both files were trained by torchvision itself. The VGG16 page and the model source say `IMAGENET1K_V1`, the file `vgg16-397923af.pth` that lpips loads, was trained from scratch with torchvision's own simplified recipe, and its recipe link is torchvision's classification references, whose AlexNet and VGG section gives the training command (learning rate 10⁻², 90 epochs on 8 GPUs) and lists these models among those trained, not ported (T373.3, T373.5, T373.6). The AlexNet file is the "One weird trick" variant, hence `owt`, trained by the same references (T373.4). Neither is a conversion of an original release, so no original license passes to them, and torchvision grants none (T373.2). A second VGG16 file, `IMAGENET1K_FEATURES`, was ported by torchvision from ssd.pytorch, whose README gives the file's URL but not who made it or from what (T373.6, T373.8, T373.9); its Caffe-style input standardization suggests a Caffe origin, but no publisher documents the chain, so it fails the bar too. LPIPS-Squeeze's trunk is no better: torchvision's SqueezeNet weights were trained in PyTorch by the contributor of its pull request (T373.76, T373.78).

**Whether explicitly licensed originals exist.** Yes, for both architectures: Oxford's Visual Geometry Group releases the original VGG16 and VGG19 (Caffe and MatConvNet files) under Creative Commons Attribution 4.0, with attribution by citing the technical report (T373.7), and Keras's porter states that Keras's VGG16 weights are a port of exactly that release (T373.20). Caffe's BVLC AlexNet, a replication of the original grouped AlexNet, is stated by its publisher to be released for use without restriction, with no named license (T373.15). Neither is the trunk LPIPS was calibrated on.

**Whether LPIPS's linear layers stay valid on such a trunk.** No. lpips builds its trunks with torchvision's constructors and `pretrained=True` (T373.16), so its v0.1 linear layers were learned on torchvision's own weights; they are one non-negative weight per channel of each tapped layer, on channel-normalized features (T373.13), and an independently trained trunk orders and scales its channels arbitrarily, so the weights mean nothing on it (T373.82). The Caffe files also expect BGR images in 0–255 with the mean subtracted, where lpips's scaling layer produces torchvision's standardization (T373.17, T373.82). Two ways remain: use the licensed trunk **uncalibrated**, which the LPIPS paper measured at 67.0 % 2AFC for VGG and 68.9 % for AlexNet on torchvision's trunks, against 69.2 % and 69.8 % calibrated (T373.13), or **recalibrate** the linear layers on BAPPS, whose terms lpips's README does not state (T373.11), which would make openftgs the publisher of new weights. How far an uncalibrated score moves from LPIPS cannot be stated from documents: it is a different distance on a different scale, and the metric study (section 5.8) measures its agreement instead.

| Candidate | Code license | Weights: publisher's statement | Training data | Differentiable | Parameters; cost | Human alignment | New dependency | Maintenance | Fit to `LPIPSLoss` | Bar; fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LPIPS-VGG v0.1 (incumbent loss and metric) | BSD-2-Clause (T373.10) | lin layers BSD-2 in the wheel; trunk `vgg16-397923af.pth`: no grant, torchvision disclaims (T373.2, T373.3) | ImageNet-1k (trunk); BAPPS 2AFC train (lin) (T373.13) | yes | 138 M (14.7 M in the convolutions used) + 1,472 (T373.3, T373.13, T373.84); measured 92.5 / 138.8 ms | BAPPS 2AFC 69.2 (T373.13); SRCC TID2013 0.67, KADID 0.72, PIPAL 0.57 in piq's table (T373.26), 0.838 / 0.883 in DreamSim's protocol (T373.52); best with DISTS as an optimization loss (T373.27) | none (§2.3) | last release 2021-08 (T373.81) | is the contract | **fails** (trunk) |
| LPIPS-Alex v0.1 (metric for Neural3DV) | BSD-2-Clause (T373.10) | trunk `alexnet-owt-7be5be79.pth`: no grant (T373.2, T373.4) | ImageNet-1k; BAPPS (T373.13) | yes | 61 M (2.5 M convolutional) (T373.4, T373.84); est. under a tenth of LPIPS-VGG at both sizes (under 9 / 14 ms) | BAPPS 69.8 (T373.13) | none | as above | is the contract | **fails** (trunk) |
| LPIPS-Squeeze v0.1 | BSD-2-Clause | trunk trained by a torchvision contributor, no grant (T373.76, T373.78) | ImageNet-1k; BAPPS | yes | 1.24 M (T373.76); est. under a tenth at both sizes | BAPPS 70.0 (T373.13) | none | as above | `net="squeeze"` is outside §2.1's `Literal` | **fails** (trunk) |
| torchvision VGG16 `IMAGENET1K_FEATURES` | BSD-3-Clause (T373.1) | ported from ssd.pytorch, whose README documents no origin or terms (T373.6, T373.9) | not stated (T373.3) | yes | 138 M; ≈ LPIPS-VGG, 92 / 139 ms | none: lin invalid on it (T373.82) | none | frozen file | Caffe-style standardization (T373.3) | **fails** (no publisher statement) |
| VGG16, Oxford's release, uncalibrated | conversion code would be openftgs's own | **CC BY 4.0**, attribution by citing the report (T373.7); Keras's port says the same (T373.20) | ImageNet (ILSVRC) (T373.7) | yes | 138 M (14.7 M used); ≈ LPIPS-VGG, 92 / 139 ms (same layers; T373.84) | uncalibrated VGG on torchvision's sibling trunk: BAPPS 67.0 (T373.13); untested on this trunk | a converter from Caffe, MatConvNet `.mat` (SciPy, if the file is not MAT 7.3) or Keras HDF5 (h5py) (T373.69, T373.84) | a frozen 2014 release (T373.7) | BGR, 0–255, mean subtracted: a fixed map of [0, 1] RGB (T373.82) | **passes** (if CC BY's attribution counts as permissive); **finalist** |
| VGG16, Oxford's release, recalibrated on BAPPS | openftgs's own | trunk CC BY 4.0; the new lin layers would be openftgs's | ImageNet; BAPPS, terms unstated (T373.11) | yes | as above | expected near LPIPS-VGG's 69.2 (T373.13); unmeasured | as above, plus BAPPS (6.6 GB) and a training step | as the trunk | as above | bar passes on the trunk, but BAPPS's terms are unknown and openftgs would publish weights against §2.3's note: **owner's call**, not a finalist |
| AlexNet, BVLC's Caffe release, uncalibrated | openftgs's own conversion | **use without restriction**, no named license (T373.15) | ImageNet (validation set named only) (T373.15) | yes | about 61 M (about 2.3 M convolutional, T373.84); est. under a tenth of LPIPS-VGG at both sizes | uncalibrated AlexNet on torchvision's sibling: BAPPS 68.9 (T373.13); untested on this trunk | a Caffe converter (protobuf) | a frozen Caffe-era release | BGR, 0–255, mean subtracted; grouped convolutions and LRN (T373.15, T373.82) | **passes** (if "without restriction" counts as a grant); metric **finalist**, beside the VGG one |
| SqueezeNet, the original release | openftgs's own conversion; the repository's BSD file (T373.77) | the repository's BSD file names software only; the model files sit in the repository (T373.77) | ImageNet (general knowledge) | yes | 1.2 M; est. under a tenth at both sizes | not sought: borderline on the bar, and the LPIPS paper's figures are for torchvision's trunk (T373.13) | a Caffe converter | a frozen release | Caffe convention (T373.82) | **borderline** (a repository grant, not one naming the weights) |

### 3.2 Weight-free losses and metrics

None of these has learned weights, so all clear the bar; NLPD's few constants were fitted by its authors: on undistorted natural images, unprinted, in its 2016 paper, and to human ratings, printed, in its 2017 version (T373.83, T373.98, T373.99). The question is fitness. Ding et al. trained networks with eleven of these models as the loss and had people rank the results (T373.27): MAE and MS-SSIM led for denoising, LPIPS and DISTS for deblurring, super-resolution and compression, and the gradient-based models (FSIM, VSI, GMSD) produced ringing, VIF over-enhanced contrast, NLPD and CW-SSIM blur, and the grayscale models applied per RGB channel (MS-SSIM, NLPD) colour shifts. As metrics, the gradient and wavelet models (HaarPSI, MDSI, VSI) top TID2013 and KADID-10k in piq's table, above LPIPS-VGG, while on BAPPS, whose distortions include real algorithms' outputs, the hand-crafted ones trail deep features (SSIM 63.1, FSIMc 63.8 against 67 to 70; MS-SSIM 61.2 on its traditional subset against LASI's 73.8) (T373.13, T373.26, T373.74).

| Candidate | Code license | Weights | Training data | Differentiable | Parameters; cost | Human alignment | New dependency | Maintenance | Fit to `LPIPSLoss` | Bar; fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 + D-SSIM only (λ_perc = 0) | openftgs | none | none | yes | 0; saves the 92.5 / 138.8 ms | the 3DGS and 3DGS-MCMC recipe (T373.66, T373.67); E7 without its [P] term | none | openftgs's own | no network: `LossConfig.lambda_perc = 0` | passes; **run B** (a candidate, not a module) |
| MS-SSIM | piq Apache-2.0 (T373.24), torchmetrics Apache-2.0 (§2.3), or openftgs's own on `losses.ssim` | none | none | yes | 0; est. 11 / 18 ms on the PyTorch path (1.3× its SSIM, T373.84) | SRCC 0.80 / 0.80 / 0.55 (T373.26); BAPPS 61.2 traditional, 79.3 CNN (T373.74); as a loss about MAE, with blur and per-channel colour shifts (T373.27) | none if written on `losses.ssim` | own code, or piq (last released 2023-07, T373.70) | fits; negative contrast terms under fractional exponents can give NaN gradients (a candidate check, section 5.4) | passes; reserve (it overlaps E7's D-SSIM) |
| GMSD, MS-GMSD | piq (T373.24) | none | none | yes | 0; est. < 20 / < 30 ms | SRCC 0.80 / 0.85 / 0.58 and 0.81 / 0.85 / 0.59 (T373.26); as a loss: ringing, discards luminance (T373.27) | piq or own | piq last released 2023-07 (T373.70) | grayscale | passes; fails fit as a loss |
| MDSI | piq | none | none | yes | 0; est. < 20 / < 30 ms | SRCC 0.89 / 0.89 / 0.59 (T373.26); not in Ding's study; gradient-based like GMSD | piq or own | as piq | fits | passes; metric reserve |
| HaarPSI | piq | none | none | yes | 0; est. < 20 / < 30 ms | SRCC 0.87 / 0.89 / 0.59 (T373.26); not in Ding's study; no BAPPS figure found | piq or own | as piq | fits | passes; metric reserve, computed in the metric study |
| FSIM (FSIMc) | piq | none | none | yes | 0; est. 20–60 / 30–90 ms | SRCC 0.80 / 0.83 / 0.59 (T373.26); BAPPS FSIMc 63.8 (T373.13); as a loss: ringing, unstable (T373.27) | piq or own | as piq | fits | passes; fails fit as a loss |
| VSI | piq | none | none | yes | 0; est. 20–60 / 30–90 ms | SRCC 0.90 / 0.88 / 0.54 (T373.26); as a loss: ringing, unstable (T373.27) | piq or own | as piq | fits | passes; fails fit as a loss |
| VIF (VIFp) | piq | none | none | yes | 0; est. 20–60 / 30–90 ms | SRCC 0.61 / 0.65 / 0.50 (T373.26); as a loss: over-enhanced contrast (T373.27) | piq or own | as piq | fits | passes; fails fit |
| NLPD | own, from the paper (not in piq, T373.26) | constants printed in the 2017 version (T373.99) | undistorted natural images (2016, T373.98) or human ratings (2017, T373.99) | yes | a few constants; est. 10–30 / 15–45 ms | as a loss: blur and colour shifts (T373.27) | own code | own code | grayscale per channel | passes; fails fit as a loss |
| CW-SSIM | own (not in piq) | none | none | yes | 0; est. 30–80 / 45–120 ms (complex steerable pyramid) | as a loss: blur (T373.27) | own code | own code | grayscale | passes; fails fit |
| ꟻLIP (LDR) | BSD-3-Clause, NVIDIA (T373.28) | none | none | yes: a PyTorch loss module ships in the repository (T373.29, T373.30) | 0; est. 10–30 / 15–45 ms | designed for rendered images against a reference; its user study had people rate error maps against the differences they saw when flipping, where it beat the other metrics on average (T373.32); no TID2013, KADID or BAPPS figure found | `flip-evaluator` 1.7 (no cp314 wheel seen, and the pip package is the API, not the loss) or the repository's BSD-3 PyTorch file vendored (T373.30, T373.71) | active, 1.7 of 2025 (T373.29, T373.71) | sRGB in [0, 1]: Î above 1 needs a decision; a pixels-per-degree parameter (T373.30) | passes; **finalist** for the metric, reserve for the loss |
| LASI | its JAX code has no license: own PyTorch code from the paper (T373.74, T373.75) | none: weights are solved per image | none | yes, end to end (T373.74) | 0; MS-SSIM class per its authors, a 12×12 solve per pixel (T373.74): est. 10–40 / 15–60 ms, to be measured | BAPPS 2AFC 73.8 / 81.4 / 70.2 / 60.5 / 62.1 / 62.3 against uncalibrated LPIPS 73.3 / 83.1 / 71.7 / 60.7 / 65.0 / 62.7 and calibrated 79.3 / 83.7 / 81.4 / 61.2 / 65.6 / 63.3 (T373.74); untested as a loss | own code | 2023 paper | fits | passes; metric **finalist** |

piq also lists IW-SSIM, SR-SIM and DSS (T373.26); they are variants of the families above and were not screened further.

### 3.3 Trunks whose weights carry the publisher's grant

No calibrated distance with licensed weights was found on any of these trunks; each would be used **uncalibrated**, like the licensed VGG16 of section 3.1. Kumar et al. found that uncalibrated distances on modern, more accurate classifiers agree less with BAPPS than AlexNet's (ResNet-18 68.5, ViT-B/8 67.6, EfficientNet-B5 67.2, against 68.9), that the best layers are early ones, and that a linear layer narrows the gap (T373.68); a training-free study of DINO-family features found DINO v1's intermediate features best, ahead of DINOv2's and of LPIPS on PIPAL (T373.73). The image–text models (CLIP, OpenCLIP, SigLIP) are trained for semantics at 224×224, a poor match for a full-resolution reconstruction loss.

| Candidate | Code license | Weights: publisher's statement | Training data | Differentiable | Parameters; cost | Human alignment | New dependency | Maintenance | Fit to `LPIPSLoss` | Bar; fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DINOv2 ViT-S/14 (also B/14), uncalibrated multi-block distance | Apache-2.0 (T373.33); the layers are vendored with RoMa (D-65) | **Apache-2.0 for code and weights**, in the README (T373.34) | 142 M images without labels (T373.34) | yes | 21 M (B/14 86 M) (T373.34); S/14 est. 1.4× / 2× LPIPS-VGG at full resolution (about 130 / 280 ms), about 0.15× at half resolution; B/14 est. 3.5× at full resolution (T373.84) | training-free DINOv2 ViT-B: SRCC TID2013 0.722, PIPAL 0.573 (T373.73); DreamSim's LoRA-tuned DINOv2 branch NIGHTS 95.0 % (T373.51); BAPPS not found | none: the vendored builders (`vit_small`, patch 14) and `dl.fbaipublicfiles.com` weights | Meta, current | crop or resize to a multiple of 14 (1344×1008), ImageNet standardization, position embeddings interpolated by the vendored code (T373.84) | **passes**; loss **finalist** |
| DINO (v1) ViT-S/16, S/8, B/16 | Apache-2.0 (T373.35) | the README licenses "this repository"; the weights, on Meta's file host, are not named (T373.36) | ImageNet, self-supervised (T373.36) | yes | 21–85 M (T373.36); S/16 est. 0.9× LPIPS-VGG at full resolution; S/8 far more (T373.84) | best training-free DINO-family features: ViT-B SRCC TID2013 0.786, PIPAL 0.637, above LPIPS's 0.588 (T373.73) | none (DINO's license is already among the vendored texts) | not checked | patch 16 or 8 | **borderline** (repository grant only): the owner may count it, as the release checklist did for RoMa's storage repository |
| DINOv3 ViT-S/16, ConvNeXt-T and others | DINOv3 License (T373.37, T373.38) | the custom "DINOv3 License": commercial use allowed with conditions that pass downstream, use restrictions, termination, and gated access (T373.38) | not sought: fails the bar | yes | 21 M and up (T373.37); cost not estimated: screened out on the bar | not sought: fails the bar | download needs an access request | current | not assessed: screened out on the bar | **fails as stated** (custom, not a standard permissive license; the owner's call) |
| OpenAI CLIP | MIT, code only (T373.39) | none for the weights; any deployed use, commercial or not, out of scope (T373.40) | crawled image–text pairs (T373.40) | yes | not estimated: screened out on the bar | not sought: fails the bar | `clip` | not checked: fails the bar | 224×224, semantic | **fails** |
| OpenCLIP ViT-B/32 (LAION-2B) | MIT (T373.41) | model card's license `mit`, but its text puts any deployed use out of scope (T373.42) | LAION-2B, uncurated web data (T373.42) | yes | about 0.2 B (T373.42); est. 0.35× at full resolution, designed for 224 (T373.84) | DreamSim's LoRA-tuned branch NIGHTS 95.3 % (T373.51); uncalibrated low-level not found | open-clip-torch | not checked | 224×224, semantic | passes formally (with the scope caveat); fails fit |
| SigLIP B/16 | big_vision Apache-2.0 (T373.85) | model card's license `apache-2.0` under Google's organization (T373.43) | WebLI (T373.43) | yes | about 0.2 B (T373.43); est. 2.3× at full resolution (T373.84) | none at low level read | transformers or big_vision | not checked | 224×224, semantic | passes; fails fit |
| ConvNeXt-T (v1) | MIT (T373.44) | the README licenses "this project"; weights on Meta's host not named (T373.47) | ImageNet-1k or 22k (T373.47) | yes | 28 M, 4.5 GFLOPs at 224 (T373.47); est. 0.3× LPIPS-VGG (T373.84) | none read; modern accurate classifiers score lower uncalibrated (T373.68) | timm or own definition | not checked | stride-4 stem: no full-resolution features (T373.84) | **borderline** (repository grant only) |
| ConvNeXt V2 | MIT and CC BY-NC 4.0 (T373.45) | ImageNet models CC-BY-NC (T373.48) | ImageNet (T373.48) | yes | not estimated: screened out on the bar | not sought: fails the bar | not assessed | not checked | not assessed | **fails** |
| timm's own ImageNet weights | Apache-2.0 (T373.46) | the author asks users to assume ImageNet's non-commercial terms (T373.49) | ImageNet (T373.49) | yes | not estimated: screened out on the bar | not sought: fails the bar | timm | not checked | not assessed | **fails** |
| torchvision's other ImageNet weights | BSD-3-Clause (T373.1) | the same disclaimer as the LPIPS trunks (T373.2) | ImageNet | yes | not estimated: screened out on the bar | not sought: fails the bar | none | as torchvision | not assessed | **fails** |
| ResNet-50, Kaiming He's Caffe release (and Keras's port) | MIT, naming the software (T373.22) | the README names no license for the models (T373.23); Keras's porter calls them MIT (T373.20) | ImageNet | yes | 25.6 M (T373.84); est. 0.3× | uncalibrated ResNet-18 68.5 (T373.68); ResNet-50 not read | a Caffe or HDF5 converter | frozen 2016 release | stride-2 stem | **borderline** (repository grant, read as covering the models by the porter only) |

### 3.4 Learned full-reference metrics

| Candidate | Code license | Weights: publisher's statement | Training data | Differentiable | Parameters; cost | Human alignment | New dependency | Maintenance | Fit to `LPIPSLoss` | Bar; fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamSim (ensemble) | MIT (T373.50) | its own fine-tuned weights: no stated license; members include OpenAI CLIP (T373.40, T373.51) | NIGHTS, diffusion-generated triplets, no terms stated (T373.51, T373.52) | yes (T373.51) | three ViT-B at 224² by design, est. about 10 ms; resizing loses the detail a render loss needs (T373.52, T373.84) | NIGHTS 96.2 %; SRCC TID2013 0.814, KADID 0.868 (LPIPS 0.838 / 0.883 in the same table); targets mid-level, semantic similarity (T373.52) | dreamsim with open-clip-torch, peft, timm, transformers (T373.79) | 0.2.1 of 2024-10 (T373.79) | 224×224 | **fails** |
| DreamSim, DINOv2 ViT-B/14 branch | MIT | trunk Apache-2.0 (T373.34); the LoRA weights on top unlicensed (T373.51) | as above | yes | as above | NIGHTS 95.0 % (T373.51) | as above | as above | 224×224 | **fails** (head unlicensed); fails fit |
| DISTS | MIT (T373.53) | α and β of the authors on an ImageNet VGG16 whose source the paper does not name (T373.57, T373.59) | ImageNet; KADID-10k and DTD textures (T373.59) | yes (T373.57) | 14.7 M + about 2,950; ≈ LPIPS-VGG at full resolution, shorter side 256 by design (T373.59) | SRCC LIVE 0.954, CSIQ 0.929, TID2013 0.830 (T373.59); BAPPS 65.1 (T373.65); best with LPIPS as a loss (T373.27) | dists-pytorch 0.1 of 2020 (T373.80) | stale | shorter side 256 | **fails** as published; on Oxford's VGG16 its α and β would need retraining on KADID |
| A-DISTS | no repository found; own code from the paper (T373.64) | no human-score training: only a two-parameter texture model fitted to hand-labelled patches (T373.65) | the trunk's | yes | as DISTS, 21×21 windows (T373.65) | SRCC TID2013 0.836, KADID 0.890; BAPPS 64.2 (T373.65) | own code | 2021 paper | shorter side 256 | passes only rebuilt on Oxford's VGG16; reserve |
| PieAPP | non-commercial BSD variant (T373.54) | weights by download scripts, no terms (T373.54) | its own pairwise-preference dataset | yes | 64×64 patches at stride 6: heavy at full resolution (T373.54) | SRCC 0.84 / 0.87 / 0.70 (T373.26); fails as a loss (T373.27) | not assessed: fails the bar | 2018 | patches | **fails** |
| Watson-DFT (and DCT, Deeploss) | no license (T373.55) | in the repository, no grant (T373.55) | BAPPS 2AFC (T373.55) | yes | a handful of parameters; cheap (FFT; not estimated further, screened out on the bar) | not sought: fails the bar (its README gives no figures, T373.55) | not assessed: fails the bar | 2020 | fits | **fails** as published |
| ST-LPIPS | BSD-2-Clause per its page (T373.60) | the repository's license only; the trunks' origin undocumented (T373.60, T373.63) | BAPPS (T373.63) | yes | ≈ LPIPS-Alex or -VGG | BAPPS 70.39 at 256² (T373.63); PIPAL 0.631 (T373.73) | stlpips_pytorch | not checked | fits | **fails** (no publisher statement on the trunks) |
| E-LPIPS | BSD-2-Clause (T373.56) | torchvision's trunks (T373.58) | ImageNet; BAPPS | yes | LPIPS × the number of sampled transformations (T373.58) | robust variant of LPIPS (T373.58) | TensorFlow 1.12 | old | not assessed: fails the bar | **fails** (trunk) |
| R-LPIPS | BSD-2-Clause per its page (T373.61) | AlexNet trunk of unnamed source; adversarially tuned linear layer (T373.61) | BAPPS | yes | ≈ LPIPS-Alex | not sought: fails the bar | not assessed: fails the bar | not checked | fits | **fails** (trunk) |

### 3.5 Summary of the screening

| Family | Candidates | Pass the bar | Borderline (owner's call) | Fail |
| --- | --- | --- | --- | --- |
| LPIPS and its trunks (3.1) | 8 | 3: Oxford's VGG16 uncalibrated; Oxford's VGG16 recalibrated (BAPPS's terms then open); BVLC's AlexNet uncalibrated | 1: the original SqueezeNet | 4: LPIPS-VGG, -Alex, -Squeeze on torchvision's trunks, and torchvision's ported VGG16 features |
| Weight-free (3.2) | 12 | 12 | 0 | 0 (fitness rules out GMSD, FSIM, VSI, VIF, NLPD and CW-SSIM as losses) |
| Licensed trunks (3.3) | 11 | 3: DINOv2; OpenCLIP (with its scope caveat); SigLIP | 3: DINO v1, ConvNeXt v1, Kaiming He's ResNet | 5: DINOv3 (custom license, the owner's call), OpenAI CLIP, ConvNeXt V2, timm's and torchvision's other weights |
| Learned metrics (3.4) | 9 | 1, rebuilt: A-DISTS on Oxford's VGG16 | 0 | 8: DreamSim and its DINOv2 branch (unlicensed heads, CLIP), DISTS (trunk), PieAPP (non-commercial), Watson-DFT (unlicensed), ST-LPIPS, E-LPIPS and R-LPIPS (trunks) |
| **All** | **40** | **19** | **4** | **17** |

Why the 17 fail, in short: torchvision's or timm's ImageNet weights, which their publishers do not license (6: LPIPS-VGG, -Alex and -Squeeze, E-LPIPS, timm's and torchvision's other weights); a trunk whose origin no publisher documents (4: torchvision's ported VGG16 features, DISTS, ST-LPIPS, R-LPIPS); a non-commercial or custom license (3: PieAPP, ConvNeXt V2, DINOv3); weights with no license at all (3: Watson-DFT and DreamSim's two fine-tuned heads); a model card that keeps deployment out of scope with no grant (1: OpenAI CLIP). Of the 19 that pass, fitness narrows the loss to a VGG-type feature distance, a DINOv2 feature distance, the weight-free ꟻLIP and MS-SSIM, or no perceptual term at all; the image–text trunks are semantic and 224-pixel, and the gradient-based weight-free measures make ringing as losses.

**A thin spot.** The evidence on the losses used across the Gaussian-splatting literature rests on two papers here, 3DGS and 3DGS-MCMC, both L1 and D-SSIM with no perceptual term (T373.66, T373.67). The dynamic-scene baselines the paper cites were not fetched, to keep searches away from where implementations of the paper turn up; the owner may name their arXiv identifiers if that cell should be thicker.

### 3.6 How the costs were estimated

Costs not measured are arithmetic (T373.84): a convolutional trunk's multiply-adds at 224² (the publisher's figure) scaled by the pixel count (×27.32 to 1352×1014, ×41.33 to 1920×1080) and divided by VGG16's convolutional 15.4 G, times LPIPS-VGG's measured 92.5 / 138.8 ms; a ViT's linear layers at about its parameter count per token and its attention at 2·n²·d per block, with n = ⌊H/p⌋·⌊W/p⌋ (6,912 tokens for ViT-S/14 at 1344×1008, 10,549 at 1918×1078). Kernel efficiency differs (depthwise convolutions and float32 attention run below the dense convolutions' throughput), so the estimates rank candidates; the candidate checks (section 5.4) measure them as B-4 measures LPIPS.

## 4. The finalists

The finalists below were proposed for checkpoint 1. The owner's decisions there (section 5.1) replaced them: PRD #374 runs every candidate that passed in full, so this section records the survey's ranking, not the run list.

### 4.1 For the loss

1. **VGG16-CC: LPIPS's distance on Oxford's VGG16, uncalibrated.**
   - *What:* the five layers LPIPS taps (relu1_2 to relu5_3), channel-normalized, squared differences averaged over channels and space and summed over layers, every channel weighted equally (lpips's `lpips=False` form, T373.17), on Oxford's original VGG16 converted once to PyTorch.
   - *Code license:* openftgs's own (the converter and the distance).
   - *Weights' publisher statement:* Oxford VGG, Creative Commons Attribution 4.0, attribution by citing Simonyan and Zisserman's report: https://www.robots.ox.ac.uk/~vgg/research/very_deep/ (T373.7).
   - *Cost:* the same convolutions as LPIPS-VGG, so ≈ 92.5 / 138.8 ms; per-iteration cost equal to run A's.
   - *Human alignment:* the uncalibrated VGG distance scored 67.0 % on BAPPS 2AFC on torchvision's sibling trunk, against 69.2 % calibrated (T373.13); untested on Oxford's trunk; VGG features are the classic perceptual loss, which lpips's README calls closer to it than AlexNet when optimizing (T373.11).
   - *Why:* the smallest change from E7's term: same architecture, layers and cost, a publisher's grant on the weights, and no calibration data.
   - *Open:* which file to convert (the Caffe model, MatConvNet's `.mat`, which SciPy reads only if it is not MAT 7.3, or Keras's HDF5 port) is settled by the candidate checks (section 5.4); the converted weights cannot ship in the wheel (§2.3's note), so a user's first run would download Oxford's file and convert it.
2. **DINOv2-S: an uncalibrated distance on DINOv2 ViT-S/14 features.**
   - *What:* patch tokens of four evenly spaced blocks (for example 3, 6, 9 and 12), channel-normalized, squared differences averaged as in LPIPS, on the image cropped to a multiple of 14 and standardized as DINOv2 expects. **Default: half resolution** (the render and the target downsampled by 2 before the trunk, about 676×507, cropped to 672×504), since full resolution fails the cost criterion by the estimate below and kitchen's default run already peaks at 23.5 GB of the card's 24; full resolution replaces it only if the candidate checks measure it no slower than LPIPS-VGG and within the memory headroom. L1 and D-SSIM keep the full-resolution detail either way.
   - *Code license:* the vendored DINOv2 layers, Apache-2.0 (D-65), and openftgs's distance.
   - *Weights' publisher statement:* Meta, DINOv2 code and weights under Apache-2.0: https://github.com/facebookresearch/dinov2 (README, "License") (T373.34); file `dinov2_vits14_pretrain.pth` on `dl.fbaipublicfiles.com`.
   - *Cost:* est. 1.4× / 2× LPIPS-VGG at full resolution, about 0.15× at half resolution (section 3.6): it meets "no higher than LPIPS-VGG" only at reduced resolution unless the candidate checks measure better.
   - *Human alignment:* training-free DINOv2 ViT-B features score SRCC 0.722 on TID2013 and 0.573 on PIPAL, below DINO v1's 0.786 and 0.637 (T373.73); no BAPPS figure; self-supervised features generally track human judgments (T373.13).
   - *Why:* the only modern trunk with an explicit grant on its weights that is already in the dependency tree, with no new package.
3. **ꟻLIP (LDR) as a loss, the weight-free reserve.**
   - *What:* NVIDIA's rendering-oriented difference measure, its PyTorch loss, on sRGB inputs in [0, 1].
   - *Code license:* BSD-3-Clause, NVIDIA (T373.28).
   - *Weights:* none.
   - *Cost:* est. 10–30 ms, a third to a tenth of LPIPS-VGG.
   - *Human alignment:* its authors' user study on rendered images (T373.32); no BAPPS or TID figure; not among Ding et al.'s losses.
   - *Why:* a weight-free candidate designed for exactly this comparison (renders against a reference), cheap, maintained, and permissively licensed; it replaces D if DINOv2 cannot meet the cost criterion, or runs as an optional sixth run.

Run B, λ_perc = 0, is itself the weight-free answer for the loss: the 3DGS recipe (T373.66, T373.67), and the cheapest by far. MS-SSIM stays in reserve: it overlaps E7's D-SSIM and behaved like MAE as a loss (T373.27).

### 4.2 For the metric

1. **VGG16-CC and AlexNet-CC, uncalibrated, as the replacements of LPIPS-VGG and LPIPS-Alex.** The loss finalist's module serves the VGG metric; the same distance on BVLC's AlexNet (use without restriction, https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet, T373.15) replaces D-29's LPIPS-Alex for Neural3DV. Code openftgs's own; cost ≈ LPIPS-VGG and under a tenth of it; alignment 67.0 % and 68.9 % BAPPS 2AFC uncalibrated on the sibling trunks (T373.13). Closest to E16's intent ("a deep-feature distance with these two backbones").
2. **LASI.** Weight-free, so nothing to license, and on BAPPS level with uncalibrated LPIPS and well above MS-SSIM (73.8 / 81.4 / 70.2 against 73.3 / 83.1 / 71.7 and 61.2 / 79.3 / 64.6 on the traditional, CNN and super-resolution subsets) (T373.74). Its reference code is unlicensed JAX (T373.75), so openftgs would write it from the paper; cost of the MS-SSIM class per its authors, to be measured.
3. **ꟻLIP (LDR).** Weight-free, BSD-3-Clause (T373.28), designed for rendered images against ground truth and validated by its authors on that task (T373.32), with error maps that serve the owner's visual comparison; no standard-benchmark figures, so it is weaker on the first metric criterion.

The metric study also computes, at little cost, HaarPSI and MDSI (the best TID2013 and KADID-10k figures in piq's table, T373.26), MS-SSIM, DINOv2-S, and LPIPS-Alex and -VGG as measurements, so that the ranking of section 5 does not rest on the finalists alone.

### 4.3 What a finalist would add to §2.3

Owner decisions, listed for adoption, not for the tests (the tests install into a scratch venv only):

- **VGG16-CC and AlexNet-CC:** no Python package if Oxford's MatConvNet file reads with SciPy (already core); otherwise h5py (BSD-3-Clause) for Keras's HDF5 port, or protobuf (BSD-3-Clause) with Caffe's BSD-2 `caffe.proto` for the Caffe files. Either way, a run-time download from Oxford's or Caffe's host (availability checked with the candidate checks), a conversion step, a cache, and an attribution line for CC BY 4.0 in the documentation and NOTICE.
- **DINOv2-S:** none: the vendored builders and Meta's weights host, as RoMa's DINOv2 ViT-L/14 already uses.
- **ꟻLIP:** `flip-evaluator` (BSD-3-Clause; no cp314 wheel seen, T373.71) does not carry the PyTorch loss; vendoring the repository's BSD-3 PyTorch file (as D-65 vendors RoMa) or writing it from the paper adds no package.
- **LASI:** none (openftgs's own code).
- **HaarPSI, MDSI, MS-SSIM (if chosen):** piq (Apache-2.0, last released 2023, requires torchvision, T373.70) or openftgs's own code.

If both uses leave LPIPS, §2.3's lpips row and its torchvision dependency go from the core, and torchmetrics' LPIPS oracle (U-23) with them.

## 5. The test plan (PRD #374)

PRD #374, opened on 2026-10-23 after the owner's checkpoint 1, replaces PRD #373's phases 2 and 3 (the test plan and the tests); the draft that stood here for checkpoint 1 is superseded. This section is the plan as declared before any result, apart from the secondary analysis section 5.7 marks as added at A's gate: it was committed while run A's first seed was still training, before anything had been computed on a render. Section 6 reports what ran against it, and any departure from it.

### 5.1 The owner's decisions at checkpoint 1 (2026-10-23)

1. **The original trunks meet the bar:** Oxford's VGG16 (CC BY 4.0, with attribution) and BVLC's AlexNet ("unrestricted use").
2. **A repository-wide license counts as the publisher's grant for DINO v1 only.** ConvNeXt v1, Kaiming He's ResNet-50 and the original SqueezeNet stay out.
3. **Seven measures stay out of the loss runs:** GMSD and MS-GMSD, FSIM, VSI, VIF, NLPD, CW-SSIM and MS-SSIM. They are still computed as metrics.
4. **DINOv3 and a VGG16 recalibrated on BAPPS stay out.**
5. **Every loss candidate gets full runs at three seeds,** not a short-run screen or three finalists: section 4's finalists rest on indirect evidence, so every candidate that passed is tested in full, except those excluded for a reason of their own.
6. **Disk space is limited:** each run's large files are deleted as soon as its results are kept (section 5.10).
7. **The plan is its own PRD,** #374, related to #373. Phase 4 of #373 (sections 7 to 9 and the recommendation) stays there.

Excluded, with their reasons: OpenCLIP and SigLIP (semantic models built for 224×224 images, unfit for a full-resolution loss or metric); the seven of decision 3, as losses; DINOv3 and a recalibrated VGG16 (decision 4); ConvNeXt v1, ResNet-50 and the original SqueezeNet (decision 2); the 17 candidates that failed the bar (section 3.5). Left to the owner's veto: MDSI and HaarPSI are gradient-based, like the measures Ding et al. found to cause ringing as losses, though Ding did not test these two (T373.27); DINO v1 ViT-S/8 is left out by the cost rule (section 3.6).

### 5.2 Setup, held fixed

- **Code:** branch `docs-perceptual-alternatives` at dfb5076, whose `src/` is `main`'s 7e70ed7. Since validation run 20 (f6047c1, #326), `src/` has changed only in where the frame cache's "auto" places the frames (#325 and its follow-ups #341 and #348) and in the benchmarks (#351), not in the training arithmetic.
- **Environment:** a scratch venv, `.scratch/perceptual/venv-perc-314` in this worktree, made with uv from the shared `.venv`'s frozen pins (Python 3.14.9, torch 2.14.1+cu130, torchvision 0.29.1, lpips 0.1.4, gsplat 1.5.3, fused-ssim at a7c48d6, every pin identical), with openftgs editable from the worktree. The candidates' packages go only there, each license read first. The shared `.venv` is fingerprinted (its files' paths, sizes and modification times) before and after.
- **Data:** kitchen, M15's conversion at `<m15-runs>/kitchen` with its frame cache (20 cameras × 300 frames at 1352×1014, cam00 the test view), M17's init points `<m17-runs>/kitchen/init.safetensors`, the default configuration, seeds 0, 1 and 2. All read only, fingerprinted likewise before and after.
- **Run folders:** `/home/auss/datasets/openftgs-m18-perceptual/`, outside the repository: `runs/<config>-s<seed>/` while a run is live, `kept/<config>-s<seed>/` for what stays, `kept/common/` for the fixed inputs (the test request, the crop windows, the checks, the calibration), `weights/` for the downloaded and converted weights.
- **Prototype code,** not committed, archived under `kept/code/` at the end: `common.py`, `configs.py`, `make_common.py` (the test request and the crop windows), `train_with.py` (the seam), `pipeline.py` (one run's stages), `candidates.py` (the loss and metric modules), `converters.py` (the Caffe weights), `checks.py`, `calibrate.py`, `metric_runner.py`, `crops.py`, `verify.py`, `study.py` (the statistics and the metric study) and `sheet.py` (the blind comparison). Each run's `candidate.json` records the code's SHA-256.
- **The seam,** approved with #373's plan: `train_with.py` replaces `openftgs.train._perceptual_network` with a factory returning the candidate's class, then calls `openftgs.cli.main(["train", …])`, so each run is `ftgs train`'s own. The class takes `net`, sets `_backbone = net`, freezes its parameters, stays in evaluation mode, and meets `LPIPSLoss`'s `forward(pred, gt, *, map_inputs=True)` on `[..., H, W, 3]` floats, Î unclamped (Q-41). Every run's `config.json` therefore says `perc_net: vgg`; the `candidate.json` beside it names the real candidate, its λ, its resolution, its seed, the commit and the code's digest. A and B run without the seam, B with the configuration file `{"loss": {"lambda_perc": 0.0}}`. The trainer's channels-last conversion of the perceptual network (`_CHANNELS_LAST_WEIGHTS`, Q-173) is off at this commit, so no candidate meets it and the ViTs need no check for it. Under the seam each candidate runs with the training path's flags: TF32 convolutions, IEEE float32 matrix products, cuDNN autotuning (D-52, Q-73, Q-173).

### 5.3 The loss configurations: 13 × seeds 0, 1 and 2, so 39 full runs

| Run | Candidate | Weights and source | Resolution of the trunk |
| --- | --- | --- | --- |
| A | LPIPS-VGG, the reference | torchvision's VGG16 (the current loss) | full |
| B | λ_perc = 0: L1 + D-SSIM only | none | — |
| C1 | VGG16-CC: LPIPS's distance uncalibrated, five layers, equal channel weights | Oxford's VGG16, CC BY 4.0 | full |
| C2 | AlexNet-CC, the same distance | BVLC's AlexNet, unrestricted use | full |
| C3 | A-DISTS on Oxford's VGG16, own code from the paper | Oxford's VGG16 | full, shorter side as its paper fixes |
| D1 | DINOv2 ViT-S/14, a distance over four blocks | Meta, Apache-2.0 (vendored layers, D-65) | half; full only if the checks fit it |
| D2 | DINOv2 ViT-B/14 | Meta, Apache-2.0 | half |
| D3 | DINO v1 ViT-S/16 | Meta, Apache-2.0 repository grant (decision 2) | full if the checks fit it, else half |
| D4 | DINO v1 ViT-B/16 | as D3 | half |
| F1 | ꟻLIP (LDR) | NVIDIA, BSD-3-Clause; no weights | full |
| F2 | LASI, own code from the paper | no weights | full |
| F3 | MDSI | piq, Apache-2.0; no weights | full |
| F4 | HaarPSI | piq, Apache-2.0; no weights | full |

How each is built:

- **C1, VGG16-CC.** Oxford's VGG16 release, converted once to PyTorch. The image, RGB with Î unclamped, is mapped to the release's convention (BGR, times 255, minus the per-channel mean it documents); features after relu1_2, relu2_2, relu3_3, relu4_3 and relu5_3; each feature vector normalized to unit length over channels (ε = 10⁻¹⁰, as lpips); squared differences summed over channels, averaged over space, summed over the five layers. This is lpips's `lpips=False` distance (T373.17) on Oxford's weights.
- **C2, AlexNet-CC.** The same distance on BVLC's AlexNet, after relu1 to relu5 (before the response normalization and pooling), with the per-channel mean of the release's mean image.
- **C3, A-DISTS.** Written from its paper (T373.65) on Oxford's VGG16, the input resized so that its shorter side is 256, as the paper fixes.
- **D1 to D4.** The patch tokens after blocks 3, 6, 9 and 12 of the twelve, each token's vector normalized to unit length, squared differences summed over channels and averaged over tokens, summed over the four blocks; the image standardized with ImageNet's mean and standard deviation, downsampled by 2 (bilinear, antialiased) at half resolution, centre-cropped to a multiple of the patch size, the position embeddings interpolated by the model's own code. DINOv2 through the vendored layers (D-65, T373.33). DINO v1 through the same builder if DINO's weights load into it strictly; otherwise through DINO's Apache-2.0 `vision_transformer.py`, adapted with its attribution.
- **F1, ꟻLIP.** NVIDIA's `LDRFLIPLoss`, the repository's BSD-3-Clause PyTorch file copied in with its header, at its default pixels per degree.
- **F2, LASI.** Written from its paper (T373.74); its JAX code, under no license (T373.75), is never read.
- **F3, F4.** piq's MDSI (a distance) and 1 − HaarPSI (a similarity), at piq's defaults with a data range of 1.

The from-paper implementations (LASI, A-DISTS, and NLPD and CW-SSIM as metrics) are validated by the sanity checks only and are labelled "own implementation, not validated against published figures" wherever their numbers appear. Validating them against TID2013, KADID-10k or BAPPS would mean downloading datasets outside CLAUDE.md's allowed sources; if one of them wins, that validation becomes a precondition of adopting it.

### 5.4 The candidate checks

Before any candidate trains, each of C1 to F4 is checked on its own:

- **Sanity,** on kitchen's ground-truth frame 0 of cam01, cam05, cam10 and cam15 at 1352×1014, under four families of distortion: Gaussian noise of σ ∈ {1, 2, 4, 8, 16}/255 (seeded, clipped to [0, 1]); Gaussian blur of σ ∈ {0.5, 1, 2, 4} px; JPEG at quality 90, 70, 50, 30 and 10 (Pillow 12.3.0); sub-pixel shifts of 0.25, 0.5 and 1 px (bilinear, edges replicated). A candidate passes when:
  1. on identical images it gives 0, or at most 10⁻⁶ of its value at the strongest noise;
  2. its mean over the four images increases at every step of every family;
  3. its gradient with respect to the prediction is finite at the identity, at every distortion, and on a prediction with a 256×256 patch raised by 0.3, above 1 where the image is bright (E7's unclamped Î).

  Symmetry is recorded, not required.
- **Cost,** by B-4's method (§3.4, Q-190): the forward on both images plus the prediction's backward, the median of 30 calls after the first call and 10 warm-ups, at 1352×1014 and 1920×1080, under the training path's flags (section 5.2). LPIPS-VGG on the trainer's adopted path (NCHW inputs, TF32, autotuning) is timed in the same process as the reference. The machine runs nothing else (`machine.py run --exclusive --wait`, with `uptime` recorded).
- **Memory,** on top of the trained kitchen model: run 20's model (N = 3,000,000) on the GPU with its gradients and its Adam moments allocated, one training view at 1352×1014 rendered with gsplat, then L1, SSIM, the candidate and the backward; torch's peak allocated memory, for each candidate and for LPIPS-VGG in the same process.
- **Drops,** as the PRD sets them: a candidate is dropped when it fails a sanity check, when its cost at 1352×1014 (kitchen's training size, the size every run trains at) exceeds LPIPS-VGG's, or when it doesn't fit even at half resolution. "Fits" means a peak at most 800 MiB above LPIPS-VGG's: run 20 left 1,072 MiB of the card's 24,564 free at its peak of 23,492 MiB, less a quarter for fragmentation. D1 and D3 run at full resolution when full resolution passes both the cost and the memory rule, else at half; D2 and D4 at half. The cost at 1920×1080 is recorded for an adoption decision, not used to drop. A dropped candidate's numbers are recorded in section 6.

### 5.5 λ calibration

λ_perc = 0.01 was tuned for LPIPS. For each candidate c,

λ_c = 0.01 · median ‖∇_Î LPIPS-VGG(Î, I)‖₂ / ‖∇_Î L_c(Î, I)‖₂,

the median over 16 renders: frame 0 of the first eight training cameras (cam01 to cam08) at 1352×1014, rendered from run 20's trained model and from the initialization (the model the trainer builds from `init.safetensors` before its first step). Each candidate is calibrated at its configuration's resolution, its resizing inside the gradient, under the training path's flags. Both models' medians are recorded; a factor above 2 between them is reported in the interim report.

### 5.6 The runs

- **Per run,** the pipeline's stages: train the default 30,000 iterations, writing only the final checkpoint; `ftgs render` of the test request (cam00 × 300 frames); `ftgs eval --renders-dir` on those renders, so the reference metrics score the very renders the candidates score (the trainer's own end-of-run METRICS.json is kept too); `scripts/run_summary.py`; `scripts/reference_flaws.py run` (REFERENCE_PSNR.json); every metric of section 5.8 per image; the crop bank (section 5.9); the kept files; the check; the deletion (section 5.10). Each step is logged as M17's were: the command, the commit, `uptime`, `/usr/bin/time -v`, the GPU's memory sampled once a second, and torch's allocator peaks.
- **Order:** by seed. Every configuration at seed 0 first, A first of all; then seed 1; then seed 2.
- **A's gate:** A at seed 0 must reproduce validation run 20, PSNR on cam00 within 0.05 dB of 29.752 dB (the recorded run-to-run spread, validation runs 2 and 5). If it doesn't, the owner hears before any other run.
- **The interim report,** after seed 0: the table of every configuration's metrics and the crop sheet go to the owner. It doesn't stop the runs; the owner may drop a configuration from seeds 1 and 2.

### 5.7 The statistics, declared before any result

- **The values:** per run, the scene means over cam00's 300 test images in `ftgs eval`'s METRICS.json (E16, D-67): PSNR and SSIM₁ are tested; SSIM₂, LPIPS-Alex and LPIPS-VGG are reported as measurements.
- **The comparisons:** every configuration other than A against A, on PSNR and on SSIM₁, each with its three per-seed values against A's three.
- **"No worse than LPIPS-VGG":** TOST, two one-sided Welch tests (the harness's `tost_p`), at the owner's margins, ±0.1 dB PSNR and ±0.002 SSIM₁.
- **"Better":** Welch's test in the favourable direction, the configuration above A: the harness's two-sided `welch_p` halved when the difference is favourable, 1 minus that otherwise.
- **Correction:** Holm, one family per test and margin, across every compared configuration and both metrics (m = 2 × the configurations compared, 24 if none is dropped), as the harness's L3 corrects (§3.3); α = 0.05.
- **Verdicts,** per configuration and metric: "equivalent" when TOST passes after Holm; "better" when the favourable Welch test passes after Holm; "inconclusive" otherwise. Beside every verdict: the mean difference, its Welch 95 % confidence interval, and the seeds a side that `k_for_power` gives for 80 % power at the margin. A configuration clearly below A shows in its interval, though no test of "worse" is declared.
- **Also reported:** the same tests at L3's margins, ±0.25 dB PSNR and ±0.005 SSIM₁.
- **The caveat:** three seeds a side give little power. Wherever a test fails to conclude, the document says so, with the seeds it would have needed.
- **Added after A's first seed, before any candidate result** (2026-10-23, the owner's decision at A's gate, #374): a secondary analysis on PSNR outside #329's flagged pixels, `REFERENCE_PSNR.json`'s `overall.union.kept_psnr` (Q-233): cam00's PSNR over the pixels that none of the three flags (clipped, cross-camera, flicker) marks at its default threshold. It gets the same tests (TOST at the owner's ±0.1 dB and L3's ±0.25 dB, the favourable Welch test) in Holm families of its own; the declared tests stay primary. The reason: A at seed 0 gave 27.32 dB against run 20's 29.75 dB with matching training traces, and the gap lies in a strip of cam00 the training cameras barely cover, mostly inside the clipped flags (section 6).

### 5.8 The metric study

Every metric candidate runs on every run's 300 test renders, clamped to [0, 1] (D-67), against the converted frames, per image; a run's value is the mean over its images. The candidates: VGG16-CC, AlexNet-CC, A-DISTS (on Oxford's VGG16), DINOv2-S and -B, DINO v1-S and -B, ꟻLIP, LASI, MDSI, HaarPSI, MS-SSIM, GMSD, MS-GMSD, FSIM, VSI, VIF, NLPD and CW-SSIM; for reference, PSNR, SSIM₁, SSIM₂, LPIPS-Alex and LPIPS-VGG, from `ftgs eval`, as measurements only.

**The input conventions,** fixed before anything is computed:

| Metric | Implementation | Input | Resolution | Parameters | Better |
| --- | --- | --- | --- | --- | --- |
| PSNR, SSIM₁, SSIM₂, LPIPS-Alex, LPIPS-VGG | `ftgs eval` | RGB in [0, 1]; LPIPS mapped to [−1, 1] (D-68) | full | the spec's (E16) | PSNR, SSIM higher; LPIPS lower |
| VGG16-CC | own, Oxford's VGG16 | RGB in [0, 1], mapped to the release's BGR, 0–255, mean subtracted | full | five layers, unit-normalized, equal weights | lower |
| AlexNet-CC | own, BVLC's AlexNet | as VGG16-CC | full | relu1 to relu5 | lower |
| A-DISTS | own from its paper, Oxford's VGG16 | as its paper | shorter side 256 | the paper's | lower |
| DINOv2-S, DINOv2-B, DINO-S, DINO-B | own distance, the publisher's weights | RGB standardized with ImageNet's mean and deviation | the loss configuration's (section 5.4) | four blocks | lower |
| ꟻLIP | NVIDIA's PyTorch file | sRGB in [0, 1] | full | the default pixels per degree | lower |
| LASI | own from its paper | as its paper | full | the paper's neighbourhood and decay | lower |
| MDSI | piq | RGB in [0, 1], data range 1 | piq's own size-dependent downsampling | piq's defaults | lower |
| HaarPSI | piq | as MDSI | as MDSI | piq's defaults | higher |
| MS-SSIM | piq | as MDSI | full | 11-tap Gaussian, σ 1.5, five scales, the published weights | higher |
| GMSD, MS-GMSD | piq | as MDSI (piq takes the luminance) | as MDSI | piq's defaults | lower |
| FSIM | piq, chromatic | as MDSI | as MDSI | piq's defaults | higher |
| VSI | piq | as MDSI | as MDSI | piq's defaults | higher |
| VIF | piq's VIFp | as MDSI (luminance) | full | piq's defaults | higher |
| NLPD | own from its paper | as its paper | full | the paper's constants | lower |
| CW-SSIM | own from its paper | luminance in [0, 1] | full | the paper's | higher |

**Agreement with PSNR and SSIM₁:** per image, Spearman's ρ of each metric with PSNR and with SSIM₁, pooled over every run and as the median within runs; across runs, Spearman's ρ and Kendall's τ between each metric's run means and PSNR's and SSIM₁'s. Signs are turned so that a positive value means agreement.

**The owner's blind comparison:**

- **The crops' values:** every metric on every kept crop, the 8-bit crop against the ground truth's.
- **The pairs:** for each crop window and each pair of runs of different configurations at the same seed, the disagreement d = min(k, M − k)/M, where k of the M candidate metrics prefer the first run's crop. Triplets are taken greedily by d, ties in a seeded random order, at most one per window, and each configuration in at most 13 triplets (2 × 80/13, rounded up): 80 triplets.
- **The sheet:** one numbered PNG per triplet, 01 to 80: a candidate crop, the ground truth in the middle, the other candidate crop. Which side holds which run is drawn with a fixed seed; the key stays in `kept/blind/key.json`, unseen by the owner until the answers are in.
- **The answer:** one string of 80 characters, `L` or `R` for the crop closer to the ground truth, `=` for no preference.
- **The results:** each metric's agreement with the owner's choices, ties left out, with its Clopper–Pearson 95 % interval, beside its rank correlations with PSNR and SSIM₁. The same choices judge the losses: each configuration's share of the triplets it won, with its interval, beside PSNR and SSIM.

### 5.9 The crop bank

`kept/common/crop-windows.json`, made from the ground truth alone before the first run (`make_common.py`): 30 frames of cam00 (5, 15, …, 295) and four 256×256 windows in each, on a 32-pixel grid: the two of most temporal change |I_t − I_{t−1}|, where reconstructions of a dynamic scene differ, then the two of most gradient energy, none overlapping; 120 windows. Each run's crops are cut from its renders clamped to [0, 1] (D-67) and written as 8-bit PNG; the same windows of the ground truth are cut once.

### 5.10 The disk rule

- **Where:** everything under `/home/auss/datasets/openftgs-m18-perceptual/`. Read only, never written or deleted: the kitchen scene and its frame cache, kitchen's init points, run 20, the dataset, and every other folder.
- **Per run, in this order:** train, writing only the final checkpoint; evaluate; render; compute every metric per image; cut the crop bank; write the kept files (`config.json`, `candidate.json`, the two METRICS.json, `SUMMARY.json`, `REFERENCE_PSNR.json`, `train_log.jsonl`, the per-image table, the crop bank, the logs with the command, commit, wall time and GPU memory); check that every kept file exists and parses, that the per-image table has every metric of every image and is finite, and that every crop decodes. Only then are that run's checkpoint and renders deleted, each path checked to lie inside the run's own folder first.
- **If a check fails,** nothing is deleted: the run stops and is reported.
- **Sizes:** about 2.2 GB of checkpoint and 6.6 GB of float32 renders per run while it is live; about 20 to 30 MB kept per run; the venv and the weights about 10 GB.
- **The trade-off:** no trained model is kept. A later check that needs one retrains its run, 15 to 60 minutes.

### 5.11 Budget

- **GPU:** about 28 hours: the checks 1 h, the calibration 0.5 h, the 39 runs about 22.5 h plus about 3.5 h to evaluate, render and score them, the metric study 1.5 h. PRD #355's #370 and step 7 wait for it.
- **Disk:** about 9 GB at a time while a run is live, about 12 GB kept with the venv and the weights.
- **The owner's time:** the interim report, the blind comparison (about 45 minutes), and #373's checkpoint 2.

## 6. Results

Everything below was measured on the RTX 4090 at ba243ec and after (the code's `src/` unchanged throughout), in the scratch venv of section 5.2, with the prototype code archived under `kept/code/`. The from-paper implementations (LASI, NLPD, CW-SSIM) are labelled "own implementation, not validated against published figures" wherever their numbers appear.

### 6.1 What changed from the plan

- **C3, A-DISTS, dropped before any run:** its paper prints neither the five fitted weights and biases of its texture probability (eq. 7) nor its constant *c*, fitted on the authors' own hand-labelled patches (T373.95), and #373's search found no code repository (T373.64). It leaves the loss runs and the metric study. The decision, recorded provisionally on #374 with three alternatives, was confirmed by the owner on 2026-10-25.
- **A's gate failed; the owner accepted A at seed 0 as the reference** (section 6.2), and asked for a secondary analysis on the kept-pixel PSNR, declared before any candidate result (section 5.7; 5ad93c7).
- **Two rulings on the checks** (the owner, 2026-10-23, in session): the shift family is judged by 0.5 px and 1 px each scoring above 0.25 px, the step where blur and displacement grow together, since LPIPS-VGG itself scores the exact 1 px shift below the bilinear 0.5 px one; and ꟻLIP's and HaarPSI's values at the identity (a stabilizer constant and float32 rounding) pass, departures from the 10⁻⁶ threshold section 5.4 declared.
- **LASI's prototype was fixed during the checks:** under the training path's flags its windowed sums ran in TF32 and cuDNN's autotuning picked FFT convolutions, which broke Ā's positive definiteness; it now computes in IEEE float32 without autotuning whatever the caller's flags, and its first sanity numbers were discarded (the run had imported the old code).
- **Î above 1:** ꟻLIP's own loss clamps both images to [0, 1] (T373.92), and the piq-based candidates (F4, and MDSI before its drop) clamp Î to [0, 1] first because piq asserts its inputs lie there (T373.97); pixels above 1 get no gradient from those terms, while L1 and D-SSIM still see them. The other candidates take Î unclamped.
- **DINO v1 loaded into the vendored DINOv2 builder** (`vit_small` and `vit_base`, patch 16, no layer scale): every key of DINO's files loaded, the only key missing being DINOv2's mask token, unused without masks. DINO's `vision_transformer.py` was neither needed nor read.
- **The own-code metrics' decisions,** made where their papers are silent or for cost:
  - *LASI:* the sum over earlier values truncated where ω^ℓ < 10⁻⁶; the pseudo-inverse replaced by a ridge solve, ε = 10⁻⁵ tr(Ā)/N, raised a hundredfold up to twice where a factorization fails; IEEE float32 without cuDNN autotuning. Checked against a brute-force solve of eq. 1 on a 48×64 image, within 3·10⁻⁵ relative.
  - *NLPD:* the 2017 version's parameters (T373.99) with the 2016 paper's six scales (T373.98), on Rec. 601 luma turned into luminance on the 2017 paper's display (5 to 300 cd/m², gamma 2.2).
  - *CW-SSIM:* only the finest bandpass level (π/4 to π, the highpass residual above π/2 split off); 16 orientations of cos^(K−1) on one half-plane; K = 0 with 0/0 counted as 1; the orientation-averaged map weighted by a centred Gaussian of a quarter of the image's size; on Rec. 601 luma (T373.100).

  All three are validated by section 5.4's sanity checks only (section 6.4).
- **Precision of the metrics:** every candidate metric ran in IEEE float32 with TF32 off, D-52's convention for reported metrics, as `ftgs eval`'s do; section 5.8's table didn't state it.
- **The blind comparison** used the cap section 5.8's formula gives for 10 configurations, 16, not the 13 written for 13, and filled 79 triplets. Its sheets were numbered in selection order, most-disputed first, with only the sides drawn at random; #374 asked for the order randomized too (section 6.9).
- **Section 5 against #374's numbers:** the renders are float32 files of about 6.6 GB per run, not #374's 1 GB, so about 9 GB were live at a time, not 3.5. Each run rendered first, then evaluated those renders (`ftgs eval --renders-dir`), as section 5.6 says, where section 5.10 and #374 list evaluate before render.
- **Folders and superseded files:** besides `runs/`, `kept/` and `weights/`, the research wrote `downloads/` (the publishers' files, T373.91), `logs/` and `investigation/` (section 6.2). Under `kept/`, three superseded outputs were replaced: LASI's first sanity file, made with the code before its fix; the first timing file of LPIPS-VGG, whose 94.0 ms stays in `logs/checks.txt`; and a first set of 64 blind sheets made with the cap of 13, replaced before the owner saw any. Nothing outside the research folder was written or deleted.
- **Order and the interim report:** the runs went by seed, A first at each. B at seed 0 ran during the checks, before the calibration it doesn't need. The interim report went to the owner on #374 after seed 0, and no configuration was dropped from seeds 1 and 2.
- **Two additions to the plan:** the classification smoke check of the converted trunks (section 6.3) and the investigation of A's gate (section 6.2).
- **Dropped by the checks:** F2 (LASI) and F3 (MDSI); D1 and D3 run at half resolution (section 6.4). The loss runs are therefore 10 configurations at three seeds, 30 runs: A, B, C1, C2, D1 to D4, F1 and F4.

### 6.2 A's gate

| Run | Code | PSNR on cam00 | Kept-pixel PSNR | SSIM₁ | LPIPS-Alex |
| --- | --- | --- | --- | --- | --- |
| Validation run 2, seed 0 | 30e4885 | 27.33 | 29.57 | | |
| Validation run 5, seed 1 | 30b82be | 27.28 | 29.79 | | 0.1894 |
| Validation run 20, seed 0 | f6047c1 (#326) | 29.75 | 30.88 | 0.8768 | 0.1850 |
| A, seed 0 | ba243ec | 27.32 | 29.66 | 0.8755 | 0.1915 |

- **Training matches:** A-s0's and run 20's `train_log.jsonl` agree to about 0.1 % at every logged iteration (total loss 0.03342 for both at iteration 30,000); their N differs from iteration 500 on (337,078 against 337,072), GPU nondeterminism at the same seed.
- **The evaluation hasn't changed:** `ftgs eval` at ba243ec re-scores run 20's own checkpoint at 29.752 dB, its recorded value.
- **Where the gap lies:** run 20 leads by 2.44 dB on average, on 283 of the 300 frames by more than 1 dB; on 10 frames, nearly all of A-s0's extra squared error lies in the right-edge strip of cam00's upper half, a blown-out wall the training cameras barely cover, where every model leaves dark smears and run 20's are faint. Inside #329's clipped flags A-s0 scores 21.7 dB, run 20 27.3 dB.
- **Reading:** three of four default kitchen runs land at 27.28 to 27.33 dB and run 20 is the outlier; cam00's PSNR carries a run-to-run component of about 2.4 dB from that strip; on the kept pixels run 20 still leads the others by 1.1 to 1.3 dB. The owner accepted A-s0 as reproducing the default (#374, 2026-10-23).

### 6.3 The converted trunks

`converters.py` read both `.caffemodel` files with its own wire-format reader (T373.89): Oxford's VGG16 in the legacy `layers` format, 13 convolutions and three fully connected layers; BVLC's AlexNet, 5 convolutions with the grouped conv2, conv4 and conv5 (48 and 192 input channels). As a smoke check of the channel order and the mean, both classified three kitchen frames (a 224 or 227 centre crop of the shorter side at 256) beside torchvision's own trunks: Oxford's VGG16 gives "dining table" first on all three (0.66, 0.55, 0.75), as torchvision's VGG16 does on two; BVLC's AlexNet agrees with torchvision's AlexNet on the first frame ("dining table", then "pool table") and, like it, spreads its probability on the other two (`kept/common/smoke-trunks.txt`).

### 6.4 The candidate checks

Sanity on four ground-truth frames (cam01, cam05, cam10, cam15 at 1352×1014); cost by B-4's method with the machine to itself, LPIPS-VGG on the trainer's path in the same process (94.0 ms both times it was timed); memory as torch's peak allocation of one training step on run 20's model (N = 3,000,000, its Adam moments resident, 1.43 GiB), LPIPS-VGG's 16.07 GiB the reference.

| Candidate | Sanity | ms at 1352×1014 | ms at 1920×1080 | Peak memory, GiB | Verdict |
| --- | --- | --- | --- | --- | --- |
| LPIPS-VGG (reference) | fails the 0.5 → 1 px shift step | 94.0 | 140.7 | 16.07 | the reference |
| VGG16-CC (C1) | passes | 92.9 | 138.8 | 10.71 | runs, full resolution |
| AlexNet-CC (C2) | passes under the shift ruling (fails the 0.5 → 1 px step) | 8.3 | 14.2 | 14.81 | runs, full resolution |
| DINOv2-S/14 (D1), full | passes | 275.3 | 551.4 | 3.93 | fails the cost rule |
| DINOv2-S/14 (D1), half | passes | 31.1 | 45.4 | 3.54 | runs, half resolution |
| DINOv2-B/14 (D2), half | passes | 61.9 | 105.2 | 3.78 | runs, half resolution |
| DINO-S/16 (D3), full | passes | 153.0 | 326.1 | 3.64 | fails the cost rule |
| DINO-S/16 (D3), half | passes | 17.1 | 34.4 | 3.53 | runs, half resolution |
| DINO-B/16 (D4), half | passes | 36.6 | 69.4 | 3.77 | runs, half resolution |
| ꟻLIP (F1) | passes under the identity ruling (3.2·10⁻⁷ at the identity, 2.4·10⁻⁶ of its strongest-noise value) | 11.6 | 17.3 | 3.54 | runs |
| LASI (F2), own implementation | fails: blur peaks at σ = 1 px then falls (0.635, 0.629, 0.615), the 1 px shift scores below 0.25 px (0.101 against 0.291) | 1,379.5 | 1,991.4 | 19.33 | dropped (sanity and cost; its memory, measured at full resolution only, also exceeds the reference's) |
| MDSI (F3) | fails: non-finite gradients at the identity and under every distortion | 2.4 | 2.7 | 3.45 | dropped |
| HaarPSI (F4) | passes under the identity ruling (up to 2.5·10⁻⁵ of its scale in float32, 10⁻¹⁵ in float64) | 2.5 | 2.9 | 3.45 | runs |
| NLPD, own implementation, metric only | passes the value checks; its gradient at the identity isn't finite (a fractional power of 0), which a metric doesn't need | | | | metric |
| CW-SSIM, own implementation, metric only | passes | | | | metric |

NLPD and CW-SSIM were sanity-checked after the runs, at the review of #374; neither result changes a metric value. Symmetry, recorded as declared: every candidate gives the same value with its arguments swapped, to float rounding (CW-SSIM 3.6·10⁻⁷ relative), except MDSI (0.19 %). The peaks include cuDNN autotuning's trial workspaces, which is why LPIPS-VGG's and AlexNet-CC's exceed VGG16-CC's; every candidate but LASI stays below LPIPS-VGG's. The per-candidate files are `kept/common/checks/{sanity,timing}-<candidate>.json` and `memory.json`, with `uptime` beside each timing (load averages 0.3 and 1.7, the second just after run B's end, nothing else on the machine).

### 6.5 λ calibration

| Configuration | λ | Median ratio, trained model | Median ratio, initialization | Their quotient |
| --- | --- | --- | --- | --- |
| C1, VGG16-CC | 0.00164 | 0.136 | 0.206 | 0.66 |
| C2, AlexNet-CC | 0.00408 | 0.309 | 0.518 | 0.60 |
| D1, DINOv2-S/14, half | 0.00185 | 0.353 | 0.103 | **3.44** |
| D2, DINOv2-B/14, half | 0.00204 | 0.219 | 0.199 | 1.10 |
| D3, DINO-S/16, half | 0.00358 | 0.497 | 0.262 | 1.90 |
| D4, DINO-B/16, half | 0.00484 | 0.490 | 0.475 | 1.03 |
| F1, ꟻLIP | 0.415 | 10.27 | 75.05 | **0.14** |
| F4, HaarPSI | 0.0617 | 2.33 | 10.08 | **0.23** |

λ is 0.01 times the median over all 16 renders, as declared. D1, F1 and F4 differ by more than a factor of 2 between the trained model and the initialization; the declared rule reports this, and the runs use the pooled median (`kept/common/calibration.json` holds every ratio).

### 6.6 The runs

All 30 runs completed the 30,000 iterations; every one reached N = 3,000,000, between iterations 2,100 (F1) and 3,800, and kept its files, passed the check of section 5.10, and had its checkpoint and renders deleted. Means ± sample standard deviation over seeds 0, 1 and 2; training time is the wall clock of `ftgs train`; "torch peak" the allocator's peak during training, "GPU peak" the largest value `nvidia-smi` sampled, which the frame cache's "auto" placement fills to near the card's 24,564 MiB in every run.

| Configuration | λ | PSNR per seed (0, 1, 2) | PSNR | Kept-pixel PSNR | SSIM₁ | SSIM₂ | LPIPS-Alex | LPIPS-VGG | Training, min | torch peak, GiB | GPU peak, MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A, LPIPS-VGG | 0.01 | 27.32, 26.53, 28.70 | 27.52 ± 1.10 | 29.72 ± 0.52 | 0.8750 ± 0.0005 | 0.9423 ± 0.0004 | 0.1896 ± 0.0018 | 0.2423 ± 0.0010 | 56.9 ± 1.0 | 13.96 | 23318 |
| B, λ = 0 | 0 | 28.30, 25.43, 27.85 | 27.20 ± 1.54 | 29.62 ± 1.48 | 0.8748 ± 0.0023 | 0.9421 ± 0.0013 | 0.2012 ± 0.0032 | 0.2737 ± 0.0018 | 9.6 ± 0.0 | 5.51 | 23661 |
| C1, VGG16-CC | 0.00164 | 27.70, 28.09, 29.05 | 28.28 ± 0.70 | 30.41 ± 0.63 | 0.8763 ± 0.0001 | 0.9432 ± 0.0002 | 0.1864 ± 0.0008 | 0.2534 ± 0.0003 | 55.6 ± 0.0 | 13.95 | 23853 |
| C2, AlexNet-CC | 0.00408 | 26.88, 26.80, 28.27 | 27.31 ± 0.83 | 29.33 ± 0.58 | 0.8722 ± 0.0006 | 0.9408 ± 0.0003 | 0.1806 ± 0.0002 | 0.2675 ± 0.0000 | 13.9 ± 0.0 | 12.70 | 23433 |
| D1, DINOv2-S/14 ½ | 0.00185 | 28.60, 26.85, 28.48 | 27.97 ± 0.98 | 29.25 ± 0.94 | 0.8774 ± 0.0010 | 0.9434 ± 0.0006 | 0.1913 ± 0.0023 | 0.2648 ± 0.0026 | 25.3 ± 0.0 | 5.60 | 22873 |
| D2, DINOv2-B/14 ½ | 0.00204 | 28.52, 26.48, 26.90 | 27.30 ± 1.08 | 29.13 ± 1.29 | 0.8769 ± 0.0016 | 0.9432 ± 0.0011 | 0.1925 ± 0.0029 | 0.2642 ± 0.0026 | 40.9 ± 0.0 | 5.84 | 24016 |
| D3, DINO-S/16 ½ | 0.00358 | 26.50, 27.28, 27.65 | 27.14 ± 0.59 | 29.31 ± 0.45 | 0.8756 ± 0.0000 | 0.9427 ± 0.0001 | 0.1911 ± 0.0009 | 0.2660 ± 0.0003 | 18.3 ± 0.0 | 5.60 | 23875 |
| D4, DINO-B/16 ½ | 0.00484 | 28.97, 29.35, 28.27 | 28.86 ± 0.55 | 30.84 ± 0.16 | 0.8770 ± 0.0001 | 0.9435 ± 0.0001 | 0.1896 ± 0.0012 | 0.2655 ± 0.0010 | 28.7 ± 0.0 | 5.83 | 23635 |
| F1, ꟻLIP | 0.415 | 28.38, 27.01, 26.58 | 27.32 ± 0.94 | 29.48 ± 0.60 | 0.8777 ± 0.0011 | 0.9435 ± 0.0007 | 0.1946 ± 0.0017 | 0.2710 ± 0.0010 | 15.7 ± 0.0 | 5.52 | 23940 |
| F4, HaarPSI | 0.0617 | 27.66, 25.24, 26.52 | 26.47 ± 1.21 | 28.90 ± 0.62 | 0.8721 ± 0.0017 | 0.9404 ± 0.0010 | 0.2078 ± 0.0017 | 0.2801 ± 0.0014 | 10.9 ± 0.0 | 5.51 | 23965 |

- **The spread between seeds is large.** cam00's PSNR moves by 0.55 to 1.54 dB (standard deviation) within one configuration, mostly from the strip of section 6.2; the kept-pixel PSNR by 0.16 to 1.48 dB; SSIM₁ by at most 0.0023.
- **Descriptively,** D4 has the highest mean PSNR and kept-pixel PSNR (28.86 and 30.84 dB, the latter the steadiest of all), then C1 (28.28 and 30.41 dB); A sits at 27.52 and 29.72 dB, B at 27.20 and 29.62 dB, F4 lowest (26.47 and 28.90 dB). SSIM₁ spans 0.8721 to 0.8777.
- **Each loss leans its own metric:** A has the lowest LPIPS-VGG (0.2423) and C2, trained on an AlexNet distance, the lowest LPIPS-Alex (0.1806); section 6.8 shows the same for the candidates' own distances.
- **Cost:** A trains in 56.9 minutes and C1 in 55.6, the same convolutions; the others in 9.6 (B) to 40.9 minutes (D2). The candidates other than C1 and C2 need 5.5 to 5.8 GiB at the allocator's peak, against 12.7 to 14.0 for the VGG and AlexNet trunks. Training times are steady to a tenth of a minute, A's apart (56.9 ± 1.0 minutes, its first seed the slowest).

### 6.7 The statistics

As declared (section 5.7): each configuration's three per-seed values against A's three, Holm across the 18 comparisons of a family (9 configurations × 2 metrics), α = 0.05. "Better" is the one-sided Welch test; its Holm-adjusted p is 1.000 wherever the difference points the other way.

| Configuration | Metric | Difference from A | Welch 95 % interval | TOST ±owner, Holm | TOST ±L3, Holm | Better, Holm | Verdict at the owner's margin | Verdict at L3's | Seeds a side for 80 % power (owner, L3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | PSNR | -0.321 | [-3.493, +2.851] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 3076, 493 |
| B | SSIM₁ | -0.0001 | [-0.0055, +0.0052] | 1.000 | 0.376 | 1.000 | inconclusive | inconclusive | 13, 3 |
| C1 | PSNR | +0.765 | [-1.475, +3.005] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1446, 232 |
| C1 | SSIM₁ | +0.0014 | [+0.0001, +0.0026] | 1.000 | 0.050 | 0.320 | inconclusive | equivalent | 2, 2 |
| C2 | PSNR | -0.202 | [-2.472, +2.068] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1620, 260 |
| C2 | SSIM₁ | -0.0027 | [-0.0040, -0.0015] | 1.000 | 0.063 | 1.000 | inconclusive | inconclusive | 2, 2 |
| D1 | PSNR | +0.458 | [-1.911, +2.828] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1854, 298 |
| D1 | SSIM₁ | +0.0025 | [+0.0004, +0.0045] | 1.000 | 0.208 | 0.269 | inconclusive | inconclusive | 4, 2 |
| D2 | PSNR | -0.217 | [-2.684, +2.251] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 2030, 326 |
| D2 | SSIM₁ | +0.0020 | [-0.0017, +0.0056] | 1.000 | 0.398 | 1.000 | inconclusive | inconclusive | 8, 2 |
| D3 | PSNR | -0.375 | [-2.636, +1.885] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1329, 214 |
| D3 | SSIM₁ | +0.0006 | [-0.0007, +0.0019] | 0.398 | 0.042 | 1.000 | inconclusive | equivalent | 2, 2 |
| D4 | PSNR | +1.346 | [-0.932, +3.624] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1290, 208 |
| D4 | SSIM₁ | +0.0020 | [+0.0008, +0.0033] | 1.000 | 0.063 | 0.159 | inconclusive | inconclusive | 2, 2 |
| F1 | PSNR | -0.195 | [-2.533, +2.142] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1789, 287 |
| F1 | SSIM₁ | +0.0028 | [+0.0004, +0.0051] | 1.000 | 0.376 | 0.271 | inconclusive | inconclusive | 5, 2 |
| F4 | PSNR | -1.042 | [-3.674, +1.590] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 2292, 368 |
| F4 | SSIM₁ | -0.0029 | [-0.0066, +0.0008] | 1.000 | 0.752 | 1.000 | inconclusive | inconclusive | 8, 2 |

The secondary analysis on the kept-pixel PSNR (declared at A's gate), Holm across its 9 comparisons:

| Configuration | Metric | Difference from A | Welch 95 % interval | TOST ±owner, Holm | TOST ±L3, Holm | Better, Holm | Verdict at the owner's margin | Verdict at L3's | Seeds a side for 80 % power (owner, L3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | kept-pixel PSNR | -0.102 | [-3.351, +3.148] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 2112, 339 |
| C1 | kept-pixel PSNR | +0.691 | [-0.632, +2.013] | 1.000 | 1.000 | 0.873 | inconclusive | inconclusive | 570, 92 |
| C2 | kept-pixel PSNR | -0.391 | [-1.649, +0.866] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 524, 85 |
| D1 | kept-pixel PSNR | -0.468 | [-2.405, +1.469] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 996, 160 |
| D2 | kept-pixel PSNR | -0.594 | [-3.359, +2.171] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 1658, 266 |
| D3 | kept-pixel PSNR | -0.412 | [-1.524, +0.700] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 406, 66 |
| D4 | kept-pixel PSNR | +1.118 | [-0.053, +2.288] | 1.000 | 1.000 | 0.248 | inconclusive | inconclusive | 257, 42 |
| F1 | kept-pixel PSNR | -0.239 | [-1.529, +1.052] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 547, 89 |
| F4 | kept-pixel PSNR | -0.822 | [-2.133, +0.489] | 1.000 | 1.000 | 1.000 | inconclusive | inconclusive | 562, 91 |

- **At the owner's margins nothing concludes:** no configuration is shown equivalent to A or better than it, on PSNR, SSIM₁ or the kept-pixel PSNR. The seeds column is the harness's `k_for_power`, as L3 reports it: the seeds a side for 80 % power if the true difference were 0, at α = 0.05 per test, without Holm. It is a floor, which is why SSIM₁ rows reading 2 are still inconclusive at 3: Holm across 18 tests and the observed differences ask for more, and where a difference already lies outside the margin (C2's SSIM₁, −0.0027 against ±0.002) no number of seeds would make TOST pass if that difference is real. With A's PSNR varying by 1.10 dB between seeds, the floor at ±0.1 dB is 1,290 to 3,076 seeds a side.
- **At L3's margins,** C1's and D3's SSIM₁ are equivalent to A's (Holm-adjusted p 0.0499 and 0.0417); nothing else concludes.
- **No configuration is better than A after Holm.** The largest differences are D4's +1.35 dB PSNR (interval −0.93 to +3.62) and +1.12 dB kept-pixel PSNR (−0.05 to +2.29), and C1's +0.77 and +0.69 dB, all inside intervals that include 0. On SSIM₁ the 95 % intervals of C1, D1, D4 and F1 lie above 0 (C1's +0.0001 to +0.0026) and C2's below (−0.0040 to −0.0015), uncorrected; after Holm none of them is a declared verdict.
- **The caveat holds everywhere:** three seeds a side give little power on this view; every "inconclusive" here means that, not equivalence.

### 6.8 The metric study

Every candidate metric on every run's 300 clamped test renders, in IEEE float32 (section 6.1). Correlations have their signs turned so that positive means agreement (a distance that falls as PSNR rises counts as positive). "Pooled" is over all 9,000 images, "within runs" the median over the 30 runs of the per-run correlation, "run means" over the 30 runs' means. The last column is section 6.9's.

| Metric | Better | Per image, with PSNR: pooled, within runs | Per image, with SSIM₁: pooled, within runs | Run means, with PSNR: ρ, τ | Run means, with SSIM₁: ρ, τ | Agreement with the owner (n = 35) |
| --- | --- | --- | --- | --- | --- | --- |
| VGG16-CC | lower | +0.35, +0.35 | +0.58, +0.79 | +0.34, +0.25 | +0.45, +0.34 | 17/35, 0.49 [0.31, 0.66] |
| AlexNet-CC | lower | +0.47, +0.61 | +0.59, +0.91 | +0.49, +0.35 | +0.33, +0.27 | 19/35, 0.54 [0.37, 0.71] |
| ꟻLIP | lower | +0.57, +0.64 | +0.77, +0.84 | +0.55, +0.42 | +0.80, +0.61 | 13/35, 0.37 [0.21, 0.55] |
| LASI (own) | lower | +0.26, +0.23 | +0.48, +0.68 | +0.39, +0.28 | +0.04, +0.02 | 21/35, 0.60 [0.42, 0.76] |
| MDSI | lower | +0.62, +0.62 | +0.68, +0.64 | +0.66, +0.49 | +0.82, +0.62 | 21/35, 0.60 [0.42, 0.76] |
| DINOv2-S/14 ½ | lower | +0.34, +0.29 | +0.55, +0.57 | +0.34, +0.22 | +0.67, +0.50 | 16/35, 0.46 [0.29, 0.63] |
| DINOv2-B/14 ½ | lower | +0.27, +0.29 | +0.57, +0.62 | +0.26, +0.17 | +0.69, +0.52 | 15/35, 0.43 [0.26, 0.61] |
| DINO-S/16 ½ | lower | +0.46, +0.42 | +0.54, +0.60 | +0.48, +0.35 | +0.65, +0.44 | 19/35, 0.54 [0.37, 0.71] |
| DINO-B/16 ½ | lower | +0.50, +0.43 | +0.58, +0.62 | +0.49, +0.37 | +0.68, +0.46 | 20/35, 0.57 [0.39, 0.74] |
| HaarPSI | higher | +0.67, +0.70 | +0.68, +0.72 | +0.68, +0.49 | +0.72, +0.54 | 17/35, 0.49 [0.31, 0.66] |
| MS-SSIM | higher | +0.59, +0.62 | +0.80, +0.83 | +0.60, +0.47 | +0.86, +0.67 | 17/35, 0.49 [0.31, 0.66] |
| GMSD | lower | +0.62, +0.62 | +0.71, +0.75 | +0.73, +0.51 | +0.67, +0.49 | 16/35, 0.46 [0.29, 0.63] |
| MS-GMSD | lower | +0.75, +0.75 | +0.67, +0.71 | +0.79, +0.59 | +0.67, +0.48 | 16/35, 0.46 [0.29, 0.63] |
| FSIM | higher | +0.87, +0.84 | +0.53, +0.52 | +0.86, +0.69 | +0.67, +0.49 | 19/35, 0.54 [0.37, 0.71] |
| VSI | higher | +0.67, +0.71 | +0.68, +0.68 | +0.68, +0.50 | +0.79, +0.60 | 17/35, 0.49 [0.31, 0.66] |
| VIF | higher | +0.15, +0.25 | +0.85, +0.87 | +0.22, +0.15 | +0.87, +0.69 | 19/35, 0.54 [0.37, 0.71] |
| NLPD (own) | lower | +0.78, +0.78 | +0.70, +0.72 | +0.78, +0.61 | +0.83, +0.66 | 14/35, 0.40 [0.24, 0.58] |
| CW-SSIM (own) | higher | +0.17, +0.08 | +0.50, +0.56 | +0.35, +0.23 | +0.22, +0.17 | 17/35, 0.49 [0.31, 0.66] |
| LPIPS-Alex | lower | +0.35, +0.39 | +0.55, +0.84 | +0.41, +0.30 | +0.07, +0.04 | 18/35, 0.51 [0.34, 0.69] |
| LPIPS-VGG | lower | +0.31, +0.32 | +0.47, +0.72 | +0.36, +0.26 | +0.28, +0.19 | 19/35, 0.54 [0.37, 0.71] |
| PSNR | higher | | | | | 13/35, 0.37 [0.21, 0.55] |
| SSIM₁ | higher | | | | | 17/35, 0.49 [0.31, 0.66] |

Each configuration's mean, over its three runs, of each candidate metric:

| Configuration | VGG16-CC | AlexNet-CC | DINOv2-S/14 ½ | DINOv2-B/14 ½ | DINO-S/16 ½ | DINO-B/16 ½ | ꟻLIP | LASI (own) | MDSI | HaarPSI | MS-SSIM | GMSD | MS-GMSD | FSIM | VSI | VIF | NLPD (own) | CW-SSIM (own) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1.674 | 0.8647 | 0.5952 | 0.7659 | 0.6377 | 0.6163 | 0.1001 | 0.4507 | 0.2807 | 0.7915 | 0.9504 | 0.07659 | 0.08341 | 0.9614 | 0.9908 | 0.4767 | 0.1557 | 0.5597 |
| B | 1.813 | 0.9174 | 0.6289 | 0.7953 | 0.6643 | 0.6323 | 0.09897 | 0.4573 | 0.2816 | 0.7781 | 0.9503 | 0.08057 | 0.08799 | 0.958 | 0.9903 | 0.479 | 0.1581 | 0.5509 |
| C1 | 1.546 | 0.8389 | 0.5625 | 0.7317 | 0.6084 | 0.5834 | 0.09938 | 0.4485 | 0.2773 | 0.793 | 0.9519 | 0.07517 | 0.08173 | 0.9637 | 0.991 | 0.4789 | 0.1518 | 0.5627 |
| C2 | 1.779 | 0.7754 | 0.606 | 0.771 | 0.6344 | 0.6106 | 0.1015 | 0.4469 | 0.2804 | 0.7893 | 0.9495 | 0.0772 | 0.08412 | 0.9612 | 0.9907 | 0.4694 | 0.1563 | 0.5561 |
| D1 | 1.716 | 0.8485 | 0.5346 | 0.7046 | 0.5935 | 0.5725 | 0.09758 | 0.4516 | 0.2769 | 0.7991 | 0.9525 | 0.07514 | 0.08138 | 0.9657 | 0.9914 | 0.4822 | 0.1512 | 0.5569 |
| D2 | 1.706 | 0.8477 | 0.5303 | 0.6819 | 0.599 | 0.578 | 0.09832 | 0.4525 | 0.2777 | 0.7952 | 0.9514 | 0.07596 | 0.08276 | 0.9634 | 0.9911 | 0.4822 | 0.1546 | 0.5574 |
| D3 | 1.727 | 0.8599 | 0.5706 | 0.7353 | 0.5619 | 0.544 | 0.1006 | 0.452 | 0.2809 | 0.7861 | 0.95 | 0.07785 | 0.08553 | 0.9599 | 0.9904 | 0.4799 | 0.1573 | 0.557 |
| D4 | 1.733 | 0.8325 | 0.5705 | 0.7445 | 0.5449 | 0.5034 | 0.09668 | 0.451 | 0.2741 | 0.8012 | 0.9525 | 0.07397 | 0.07985 | 0.9678 | 0.9916 | 0.4809 | 0.1497 | 0.5564 |
| F1 | 1.729 | 0.8551 | 0.5893 | 0.7415 | 0.6137 | 0.5846 | 0.09404 | 0.4594 | 0.2736 | 0.7992 | 0.9547 | 0.07481 | 0.08218 | 0.9626 | 0.9917 | 0.4884 | 0.1503 | 0.5476 |
| F4 | 1.847 | 0.9394 | 0.6558 | 0.8248 | 0.6985 | 0.6627 | 0.1058 | 0.4646 | 0.2867 | 0.7816 | 0.9471 | 0.07759 | 0.08687 | 0.9542 | 0.9898 | 0.4748 | 0.1622 | 0.5368 |
| better | lower | lower | lower | lower | lower | lower | lower | lower | lower | higher | higher | lower | lower | higher | higher | higher | lower | higher |

- **Across runs,** FSIM, NLPD, MS-GMSD, GMSD, MDSI, MS-SSIM, VSI and HaarPSI order the 30 runs most like PSNR (Spearman 0.60 to 0.86) and also like SSIM₁ (0.67 to 0.86); VIF follows SSIM₁ (0.87) but not PSNR (0.22).
- **The deep distances follow the run ranking less:** the DINO distances 0.26 to 0.49 with PSNR and 0.65 to 0.69 with SSIM₁, VGG16-CC and AlexNet-CC 0.34 and 0.49 with PSNR, LPIPS-Alex and LPIPS-VGG 0.41 and 0.36 with PSNR and 0.07 and 0.28 with SSIM₁. Within a run, frame to frame, AlexNet-CC and LPIPS-Alex do follow SSIM₁ closely (0.91 and 0.84).
- **A trained-on distance flatters its own run:** C1 has the lowest VGG16-CC, C2 the lowest AlexNet-CC, D2 the lowest DINOv2-B/14, D4 the lowest DINO-B/16 and F1 the lowest ꟻLIP; D1 and D3 come second on their own distances, behind D2 and D4; F4 does not lead on HaarPSI.
- **LASI and CW-SSIM** (both own implementations, not validated against published figures) track neither PSNR nor SSIM₁ across runs (0.39 and 0.35 with PSNR, 0.04 and 0.22 with SSIM₁).
- MDSI and LASI ran as metrics though dropped as losses.

### 6.9 The blind comparison

79 triplets (`kept/blind/sheets/01.png` to `79.png`), chosen as section 5.8 declared from the crops of all 30 runs: for each window and pair of runs of different configurations at the same seed, the share of the 18 candidate metrics that disagree; the most-disputed pairs first, one per window, numbered in that order; only the sides were drawn at random (seed 374), where #374 asked for the order randomized too, so an order effect such as fatigue toward the end would fall on the least-disputed pairs. The cap per configuration was written as 13 (2 × 80/13) for 13 configurations; with 10 it was applied as the formula gives, 16, which filled 79 triplets (disagreement 0.44 to 0.50, 49 motion and 30 detail windows, 29, 18 and 32 at seeds 0, 1 and 2). The owner answered on 2026-10-24, without the key: 13 left, 22 right, 44 ties (`kept/blind/scores.json`).

| Configuration | Won / decisive | Share | Clopper–Pearson 95 % | Ties it appeared in |
| --- | --- | --- | --- | --- |
| A, LPIPS-VGG | 4 / 10 | 0.40 | [0.12, 0.74] | 6 |
| B, λ = 0 | 4 / 10 | 0.40 | [0.12, 0.74] | 6 |
| C1, VGG16-CC | 5 / 7 | 0.71 | [0.29, 0.96] | 9 |
| C2, AlexNet-CC | 3 / 4 | 0.75 | [0.19, 0.99] | 12 |
| D1, DINOv2-S/14 ½ | 4 / 7 | 0.57 | [0.18, 0.90] | 9 |
| D2, DINOv2-B/14 ½ | 5 / 6 | 0.83 | [0.36, 1.00] | 10 |
| D3, DINO-S/16 ½ | 4 / 6 | 0.67 | [0.22, 0.96] | 10 |
| D4, DINO-B/16 ½ | 2 / 9 | 0.22 | [0.03, 0.60] | 7 |
| F1, ꟻLIP | 3 / 8 | 0.38 | [0.09, 0.76] | 8 |
| F4, HaarPSI | 1 / 3 | 0.33 | [0.01, 0.91] | 11 |

- **The owner preferred neither crop in 44 of 79 triplets,** and the 35 decisive choices split without a clear winner: every configuration's 95 % interval includes one half. D2, C2 and C1 won most of theirs (5 of 6, 3 of 4, 5 of 7); D4, the PSNR leader, won 2 of 9.
- **No metric agrees with the owner better than chance:** on the 35 decisive triplets the shares run from 0.37 (PSNR, ꟻLIP) to 0.60 (LASI, MDSI), every interval including 0.5; LPIPS-Alex 0.51, LPIPS-VGG 0.54, SSIM₁ 0.49 (table of section 6.8).
- **What this design can and can't show:** the pairs were chosen where the metrics disagree most, so on them every metric is near a coin toss by construction, and the renders differ least; the comparison measures which metric breaks those ties as a person does, and with 35 decisive answers it can't separate them. It does show that, where the metrics disagree, the owner mostly saw no difference between configurations.

### 6.10 Budget, disk and what stayed unchanged

- **GPU:** the 30 runs' pipelines took 17.2 hours (training, render, evaluation, every metric, crops), the checks about 20 minutes, the calibration 20 seconds, the crop metrics 4.7 minutes and the gate's investigation about 5 minutes: about 17.8 hours against the 28 estimated, since C3, F2 and F3 were dropped and most candidates train faster than A. LASI took 263 of each run's 330 seconds of metrics.
- **Disk:** at most about 9 GB at a time; kept, 380 MB (every run's files, the crops, the checks, the calibration, the study, the blind sheets and key, and the prototype code under `kept/code/`), beside 1.6 GB of downloads, 66 MB of converted weights, 0.65 GB of the gate's investigation (`investigation/`: run 20's re-evaluation and 10 frames from three models) and the 5.9 GB scratch venv.
- **The branch:** only `docs/research/perceptual-alternatives.md` and `docs/clean-room/source-log.md` differ from `main`, and `tests/test_skeleton.py` passes (51 passed, CPU-marked; no GPU-marked test in it), run with the worktree's openftgs and harness in the scratch venv.
- **Unchanged:** the shared `.venv` (its files' paths, sizes and modification times, and `pip freeze`) and the read-only inputs (kitchen, its frame cache, M17's kitchen runs and the dataset), fingerprinted before the first run and after the last, match. The fingerprints and both freezes are kept in `kept/common/`; the scratch venv is deleted with the worktree and can be rebuilt from `kept/common/scratch-freeze.txt` (uv from the same pins, with openftgs and the harness editable).

### 6.11 What the tests show, for #373's recommendation

- **The loss:** no candidate is shown no worse than LPIPS-VGG at the owner's margins, and none is shown worse or better: on kitchen's cam00, three seeds can't resolve differences of the size the margins ask for. Descriptively, D4 (DINO-B/16 at half resolution) and C1 (Oxford's VGG16, uncalibrated) have the highest means of PSNR and of kept-pixel PSNR, D4 the steadiest; C2's SSIM₁ interval lies below A's, uncorrected. Every candidate but C1 trains 1.4 (D2) to 6 times faster than A (C1 matches A's cost), and B, without a perceptual term, is the fastest and not distinguishable from A either.
- **The owner's eye** saw no difference in most of the disputed crops and preferred no configuration clearly.
- **The metric:** the weight-free FSIM, NLPD, MS-GMSD, GMSD, MDSI, MS-SSIM, VSI and HaarPSI agree best with PSNR and SSIM₁ across runs; the deep-feature distances, LPIPS included, agree least, and most candidate distances flatter the run trained on them (section 6.8). No metric agreed with the owner's choices better than chance on the disputed crops.
- **Limits:** one scene, one test view whose PSNR is dominated by a strip the training cameras barely cover, three seeds, own implementations of LASI, NLPD and CW-SSIM validated by sanity checks only, and a blind comparison of 35 decisive answers on the hardest pairs.

## 7. to 9.

The recommendation, what adopting it would change, and the sources: #373's phase 4, after the results.

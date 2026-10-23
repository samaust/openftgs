# Perceptual loss and metric: alternatives to LPIPS with licensed weights

A research document, written from 2026-10-23 on PRD #373 (which absorbed issue #372), on replacing LPIPS in openftgs's two uses of it, the training loss and the evaluation metric, with something whose pretrained weights carry an explicit permissive grant from their publisher, or on keeping LPIPS. This version holds phase 1, the desk survey (sections 1 to 4), and a draft of phase 2, the test plan (section 5), for the owner's checkpoint 1. Sections 6 to 9 (results, recommendation, what adopting it would change, sources) follow the tests. The document proposes no code and changes no spec item: adopting a replacement is a spec change the owner makes (E7, E16, D-9, D-29, D-68 and §2.1's `perc_net`), followed by its own implementation issue.

**Owner decisions behind the work** (2026-10-23, in session; PRD #373):

- Scope: both uses, the training loss (E7's L_perc, D-9) and the evaluation metric (E16, D-29, D-68).
- The bar: the weights' own publisher states a permissive license that allows commercial use, as DINOv2's README does for its weights. The training data is recorded, not judged.
- Depth: a survey, a test plan, and the tests themselves (training runs on openftgs-synth's `kitchen`).
- Timing: now, before #370 and step 7 of PRD #355, which wait for the owner's decision.

**Words.** A **trunk** is the pretrained network whose features a perceptual distance compares (LPIPS's VGG16 or AlexNet). **Calibrated** means per-channel weights learned on human judgments sit on top of the trunk (LPIPS's linear layers, "lin"); **uncalibrated** means every channel counts equally (lpips's `lpips=False`). The **publisher** of a set of weights is whoever trained and released those files; a **porter** converted someone else's files to another framework. A **grant** is a license statement that covers the weights themselves, not only a repository's code. BAPPS 2AFC is the two-alternative forced-choice agreement with human judgments on the LPIPS paper's dataset (human ceiling 73.9 %); SRCC is the Spearman rank correlation with mean opinion scores on TID2013, KADID-10k or PIPAL.

**Clean-room record.** Every page, license file, model card, search, skipped hit and general-knowledge claim behind this document has a row in `docs/clean-room/source-log.md`, with temporary IDs T373.1 onward, numbered at the merge into `main`. Each project's license file was read before anything else of it, except where its row says otherwise: a README read just before its MIT file (T373.20), and repository pages read for their sidebar license and README where no license file sat at the usual path (T373.54, T373.55, T373.60, T373.61, T373.75); a README is documentation, which every license class allows. Source code was read only under permissive licenses, and only for identifiers and provenance comments (T373.6, T373.16, T373.17); pyiqa (non-commercial, T373.25), PieAPP (non-commercial, T373.54), and the unlicensed Watson-DFT and LASI repositories (T373.55, T373.75) were read through their documentation only. No search named the paper's method; no hit looked like an implementation of it; nothing was taken from the paper's own code links.

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

**Whether LPIPS's linear layers stay valid on such a trunk.** No. lpips builds its trunks with torchvision's constructors and `pretrained=True` (T373.16), so its v0.1 linear layers were learned on torchvision's own weights; they are one non-negative weight per channel of each tapped layer, on channel-normalized features (T373.13), and an independently trained trunk orders and scales its channels arbitrarily, so the weights mean nothing on it (T373.82). The Caffe files also expect BGR images in 0–255 with the mean subtracted, where lpips's scaling layer produces torchvision's standardization (T373.17, T373.82). Two ways remain: use the licensed trunk **uncalibrated**, which the LPIPS paper measured at 67.0 % 2AFC for VGG and 68.9 % for AlexNet on torchvision's trunks, against 69.2 % and 69.8 % calibrated (T373.13), or **recalibrate** the linear layers on BAPPS, whose terms lpips's README does not state (T373.11), which would make openftgs the publisher of new weights. How far an uncalibrated score moves from LPIPS cannot be stated from documents: it is a different distance on a different scale, and 3c measures its agreement instead.

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

None of these has learned weights, so all clear the bar; NLPD's few constants were fitted to human ratings by its authors and are published with it (T373.83). The question is fitness. Ding et al. trained networks with eleven of these models as the loss and had people rank the results (T373.27): MAE and MS-SSIM led for denoising, LPIPS and DISTS for deblurring, super-resolution and compression, and the gradient-based models (FSIM, VSI, GMSD) produced ringing, VIF over-enhanced contrast, NLPD and CW-SSIM blur, and the grayscale models applied per RGB channel (MS-SSIM, NLPD) colour shifts. As metrics, the gradient and wavelet models (HaarPSI, MDSI, VSI) top TID2013 and KADID-10k in piq's table, above LPIPS-VGG, while on BAPPS, whose distortions include real algorithms' outputs, the hand-crafted ones trail deep features (SSIM 63.1, FSIMc 63.8 against 67 to 70; MS-SSIM 61.2 on its traditional subset against LASI's 73.8) (T373.13, T373.26, T373.74).

| Candidate | Code license | Weights | Training data | Differentiable | Parameters; cost | Human alignment | New dependency | Maintenance | Fit to `LPIPSLoss` | Bar; fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 + D-SSIM only (λ_perc = 0) | openftgs | none | none | yes | 0; saves the 92.5 / 138.8 ms | the 3DGS and 3DGS-MCMC recipe (T373.66, T373.67); E7 without its [P] term | none | openftgs's own | no network: `LossConfig.lambda_perc = 0` | passes; **run B** (a candidate, not a module) |
| MS-SSIM | piq Apache-2.0 (T373.24), torchmetrics Apache-2.0 (§2.3), or openftgs's own on `losses.ssim` | none | none | yes | 0; est. 11 / 18 ms on the PyTorch path (1.3× its SSIM, T373.84) | SRCC 0.80 / 0.80 / 0.55 (T373.26); BAPPS 61.2 traditional, 79.3 CNN (T373.74); as a loss about MAE, with blur and per-channel colour shifts (T373.27) | none if written on `losses.ssim` | own code, or piq (last released 2023-07, T373.70) | fits; negative contrast terms under fractional exponents can give NaN gradients (a 3a check) | passes; reserve (it overlaps E7's D-SSIM) |
| GMSD, MS-GMSD | piq (T373.24) | none | none | yes | 0; est. < 20 / < 30 ms | SRCC 0.80 / 0.85 / 0.58 and 0.81 / 0.85 / 0.59 (T373.26); as a loss: ringing, discards luminance (T373.27) | piq or own | piq last released 2023-07 (T373.70) | grayscale | passes; fails fit as a loss |
| MDSI | piq | none | none | yes | 0; est. < 20 / < 30 ms | SRCC 0.89 / 0.89 / 0.59 (T373.26); not in Ding's study; gradient-based like GMSD | piq or own | as piq | fits | passes; metric reserve |
| HaarPSI | piq | none | none | yes | 0; est. < 20 / < 30 ms | SRCC 0.87 / 0.89 / 0.59 (T373.26); not in Ding's study; no BAPPS figure found | piq or own | as piq | fits | passes; metric reserve, computed in 3c |
| FSIM (FSIMc) | piq | none | none | yes | 0; est. 20–60 / 30–90 ms | SRCC 0.80 / 0.83 / 0.59 (T373.26); BAPPS FSIMc 63.8 (T373.13); as a loss: ringing, unstable (T373.27) | piq or own | as piq | fits | passes; fails fit as a loss |
| VSI | piq | none | none | yes | 0; est. 20–60 / 30–90 ms | SRCC 0.90 / 0.88 / 0.54 (T373.26); as a loss: ringing, unstable (T373.27) | piq or own | as piq | fits | passes; fails fit as a loss |
| VIF (VIFp) | piq | none | none | yes | 0; est. 20–60 / 30–90 ms | SRCC 0.61 / 0.65 / 0.50 (T373.26); as a loss: over-enhanced contrast (T373.27) | piq or own | as piq | fits | passes; fails fit |
| NLPD | own, from the paper (not in piq, T373.26) | published constants, fitted to ratings by the authors (T373.83) | the authors' fit of those constants to human ratings (T373.83) | yes | a few constants; est. 10–30 / 15–45 ms | as a loss: blur and colour shifts (T373.27) | own code | own code | grayscale per channel | passes; fails fit as a loss |
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

Costs not measured are arithmetic (T373.84): a convolutional trunk's multiply-adds at 224² (the publisher's figure) scaled by the pixel count (×27.32 to 1352×1014, ×41.33 to 1920×1080) and divided by VGG16's convolutional 15.4 G, times LPIPS-VGG's measured 92.5 / 138.8 ms; a ViT's linear layers at about its parameter count per token and its attention at 2·n²·d per block, with n = ⌊H/p⌋·⌊W/p⌋ (6,912 tokens for ViT-S/14 at 1344×1008, 10,549 at 1918×1078). Kernel efficiency differs (depthwise convolutions and float32 attention run below the dense convolutions' throughput), so the estimates rank candidates; 3a measures the finalists as B-4 measures LPIPS.

## 4. The finalists

### 4.1 For the loss

1. **VGG16-CC: LPIPS's distance on Oxford's VGG16, uncalibrated.**
   - *What:* the five layers LPIPS taps (relu1_2 to relu5_3), channel-normalized, squared differences averaged over channels and space and summed over layers, every channel weighted equally (lpips's `lpips=False` form, T373.17), on Oxford's original VGG16 converted once to PyTorch.
   - *Code license:* openftgs's own (the converter and the distance).
   - *Weights' publisher statement:* Oxford VGG, Creative Commons Attribution 4.0, attribution by citing Simonyan and Zisserman's report: https://www.robots.ox.ac.uk/~vgg/research/very_deep/ (T373.7).
   - *Cost:* the same convolutions as LPIPS-VGG, so ≈ 92.5 / 138.8 ms; per-iteration cost equal to the baseline's.
   - *Human alignment:* the uncalibrated VGG distance scored 67.0 % on BAPPS 2AFC on torchvision's sibling trunk, against 69.2 % calibrated (T373.13); untested on Oxford's trunk; VGG features are the classic perceptual loss, which lpips's README calls closer to it than AlexNet when optimizing (T373.11).
   - *Why:* the smallest change from E7's term: same architecture, layers and cost, a publisher's grant on the weights, and no calibration data.
   - *Open:* which file to convert (the Caffe model, MatConvNet's `.mat`, which SciPy reads only if it is not MAT 7.3, or Keras's HDF5 port) is settled in 3a; the converted weights cannot ship in the wheel (§2.3's note), so a user's first run would download Oxford's file and convert it.
2. **DINOv2-S: an uncalibrated distance on DINOv2 ViT-S/14 features.**
   - *What:* patch tokens of four evenly spaced blocks (for example 3, 6, 9 and 12), channel-normalized, squared differences averaged as in LPIPS, on the image cropped to a multiple of 14 and standardized as DINOv2 expects. **Default: half resolution** (the render and the target downsampled by 2 before the trunk, about 676×507, cropped to 672×504), since full resolution fails the cost criterion by the estimate below and kitchen's default run already peaks at 23.5 GB of the card's 24; full resolution replaces it only if 3a measures it no slower than LPIPS-VGG and within the memory headroom. L1 and D-SSIM keep the full-resolution detail either way.
   - *Code license:* the vendored DINOv2 layers, Apache-2.0 (D-65), and openftgs's distance.
   - *Weights' publisher statement:* Meta, DINOv2 code and weights under Apache-2.0: https://github.com/facebookresearch/dinov2 (README, "License") (T373.34); file `dinov2_vits14_pretrain.pth` on `dl.fbaipublicfiles.com`.
   - *Cost:* est. 1.4× / 2× LPIPS-VGG at full resolution, about 0.15× at half resolution (section 3.6): it meets "no higher than LPIPS-VGG" only at reduced resolution unless 3a measures better.
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

The metric study (3c) also computes, at little cost, HaarPSI and MDSI (the best TID2013 and KADID-10k figures in piq's table, T373.26), MS-SSIM, DINOv2-S, and LPIPS-Alex and -VGG as measurements, so that the ranking of section 5 does not rest on the finalists alone.

### 4.3 What a finalist would add to §2.3

Owner decisions, listed for adoption, not for the tests (the tests install into a scratch venv only):

- **VGG16-CC and AlexNet-CC:** no Python package if Oxford's MatConvNet file reads with SciPy (already core); otherwise h5py (BSD-3-Clause) for Keras's HDF5 port, or protobuf (BSD-3-Clause) with Caffe's BSD-2 `caffe.proto` for the Caffe files. Either way, a run-time download from Oxford's or Caffe's host (availability to check in 3a), a conversion step, a cache, and an attribution line for CC BY 4.0 in the documentation and NOTICE.
- **DINOv2-S:** none: the vendored builders and Meta's weights host, as RoMa's DINOv2 ViT-L/14 already uses.
- **ꟻLIP:** `flip-evaluator` (BSD-3-Clause; no cp314 wheel seen, T373.71) does not carry the PyTorch loss; vendoring the repository's BSD-3 PyTorch file (as D-65 vendors RoMa) or writing it from the paper adds no package.
- **LASI:** none (openftgs's own code).
- **HaarPSI, MDSI, MS-SSIM (if chosen):** piq (Apache-2.0, last released 2023, requires torchvision, T373.70) or openftgs's own code.

If both uses leave LPIPS, §2.3's lpips row and its torchvision dependency go from the core, and torchmetrics' LPIPS oracle (U-23) with them.

## 5. The test plan (draft for checkpoint 1)

Nothing in this section has run. It follows PRD #373's phases 2 and 3; the owner confirms or edits the finalists and the criteria before any GPU time is spent.

### 5.1 Setup, held fixed

- **Code:** `main` at the research commit (7e70ed7 or the branch's base when the runs start), openftgs installed editable from this worktree into a scratch venv, `.scratch/perceptual/venv-perc-314` (Python 3.14, the CUDA build of PyTorch the shared `.venv` pins); candidate packages (piq, the ꟻLIP loss file, a converter's dependency) only there, each license read first; the shared `.venv` untouched.
- **Data:** kitchen, M15's conversion at `<m15-runs>/kitchen` (20 cameras × 300 frames at 1352×1014, cam00 held out), its frame cache built beforehand, M17's init points `<m17-runs>/kitchen/init.safetensors`, the default configuration, seed 0 (Q-220).
- **Runs** under `/home/auss/datasets/openftgs-m18-perceptual/<run>/`, outside the repository; logs with the command, commit, `uptime`, `/usr/bin/time -v` and GPU samples, as M17's.
- **Prototype code** in `.scratch/perceptual/` (not committed): `candidates.py` (the candidate modules), `checks.py` (3a), `calibrate.py`, `train_with.py` (the seam below), `metric_study.py`, `sheet.py`. The document will record every command, the patch point and the versions.
- **The seam** (approved with the plan): `train_with.py --loss NAME --lambda λ` replaces `openftgs.train._perceptual_network` with a factory returning a class whose constructor takes `net`, sets `_backbone = net` (so `total_loss`'s check passes for `perc_net = "vgg"`), freezes its parameters, stays in evaluation mode, and whose `forward(pred, gt, *, map_inputs=True)` meets `LPIPSLoss`'s contract; it then runs the trainer as `ftgs train` does, with `LossConfig.lambda_perc` set to λ by a configuration file. It catches a candidate's effect on a real run; it misses what a production implementation would add (precision modes, channels-last, caching).

### 5.2 3a. The candidates on their own

- **Sanity,** on kitchen's ground-truth frame 0 of cams 01, 05, 10 and 15 at full resolution: Gaussian noise σ ∈ {1, 2, 4, 8, 16}/255; Gaussian blur σ ∈ {0.5, 1, 2, 4} px; JPEG at quality {90, 70, 50, 30, 10} (Pillow); sub-pixel shifts of {0.25, 0.5, 1} px by bilinear resampling. Each candidate must give 0 (within 10⁻⁶ of its scale) on identical images, grow monotonically along each family, and have finite gradients with respect to the prediction, including at the identity and on a prediction with values above 1 (E7's unclamped Î). Symmetry is recorded, not required (ꟻLIP's HDR form is asymmetric).
- **Timing,** as B-4 times LPIPS (Q-190): the forward on both images plus the prediction's backward, the median of 30 calls after the first call and 10 warm-ups, in the training path's precision (TF32, autotuning as adopted), at 1352×1014 and 1920×1080, with the machine to itself (`.claude/skills/implement-without-subagents/scripts/machine.py run --exclusive`, as B-4's second run), LPIPS-VGG measured in the same session as the reference; also peak allocated memory.
- **Conversion** (VGG16-CC, AlexNet-CC): download Oxford's and BVLC's files, record their size and SHA-256, convert, and check the converted VGG16's ImageNet top-1 on a handful of validation images is sane (a smoke check, not a benchmark).

### 5.3 λ calibration

λ_perc = 0.01 was tuned for LPIPS. For each loss finalist c, λ_c = 0.01 · median_i ‖∇_Î LPIPS-VGG(Î_i, I_i)‖₂ / ‖∇_Î L_c(Î_i, I_i)‖₂ over 8 training images (frame 0 of the first 8 training cameras), with Î_i rendered from two models: M17's trained kitchen model at #326 (row 20's checkpoint, `<m17-runs>/kitchen/default-326`) and the initialization (iteration 1). The trained-model value is used, since most of training is near convergence; both values and their spread are recorded, and if they differ by more than a factor of 2 the owner is told before the runs.

### 5.4 3b. The training runs

| Run | Loss | λ_perc | Estimated wall time |
| --- | --- | --- | --- |
| A | LPIPS-VGG (the baseline) | 0.01 | 57 min (rows 2 and 20 of the validation runs: 58 min 18 s and 56 min 20 s) |
| B | L1 + D-SSIM only | 0 | about 15–20 min: the perceptual loss is 82 % of a grown iteration (B-1), so the rest is about 21 ms of 118 |
| C | VGG16-CC | λ_C from 5.3 | about 57 min (the same convolutions) |
| D | DINOv2-S, at half resolution by default (section 4.1) | λ_D from 5.3, calibrated at the resolution the run uses | about 25 min at half resolution; 60–90 min if 3a clears full resolution (est.) |
| E (optional) | the better of C and D | λ at 3× or ⅓× its calibrated value, on the side the owner picks | as C or D |
| F (optional) | ꟻLIP | λ_F from 5.3 | about 20–25 min (est.) |

Each run is `ftgs train $K/scene.json --init $M/kitchen/init.safetensors --out $R/<run>` (A, and B with a configuration file `{"loss": {"lambda_perc": 0.0}}`), or `train_with.py` with the same arguments (C to F); then `ftgs render` on the held-out request (cam00 × 300 frames) and `ftgs eval`. Recorded per run: command, commit, configuration, seed, wall time, peak GPU memory, the N trace, PSNR, SSIM₁, SSIM₂, LPIPS-Alex and LPIPS-VGG (measurements only, for continuity with M17), and every candidate metric of 3c.

**Acceptance of the baseline:** A must reproduce row 20's kitchen numbers (PSNR 29.75 dB, LPIPS-Alex 0.1850) within the recorded run-to-run spread (0.05 dB, rows 2 and 5); if the code since #326 has moved them, that is reported before the finalists are judged.

### 5.5 3c. The metric study and the owner's comparison

- **Agreement:** every candidate metric (VGG16-CC, AlexNet-CC, LASI, ꟻLIP, HaarPSI, MDSI, MS-SSIM, DINOv2-S, and LPIPS-Alex and -VGG as references) on the 300 test renders of every run: the per-run mean and ranking, and the per-image Spearman correlation with PSNR, SSIM₁ and LPIPS-VGG, pooled over runs and within each run.
- **Comparison sheet:** the same frames (a static, a moving and the fastest-motion frame of cam00) and three crops per frame from every run, side by side with the ground truth, as PNGs under the run directory, so the owner's eye is the human judgment and the conclusion does not rest on LPIPS.

### 5.6 Proposed decision criteria

For the owner to accept or change at checkpoint 1.

- **A loss** is acceptable when, against A: PSNR at least A's minus 0.1 dB; SSIM₁ at least A's minus 0.002; visibly no worse in the owner's side-by-side crops; per-iteration cost no higher than LPIPS-VGG's (3a's timing).
- **A metric** is acceptable when: its published human alignment is at least comparable to LPIPS's (BAPPS 2AFC within about 2 points of LPIPS-VGG's 69.2, or SRCC on TID2013 and KADID-10k at least LPIPS-VGG's in the same source); it orders the runs as the owner's visual ranking and PSNR and SSIM do; and its weights clear the bar.
- A single run per configuration resolves differences above the 0.05 dB seed spread only; a difference within it is reported as no difference.

### 5.7 GPU hours

| Part | Estimate |
| --- | --- |
| 3a checks and timings (with conversions) | 0.5 h |
| λ calibration | 0.2 h |
| A, B, C, D | 1.0 + 0.3 + 1.0 + 0.4 h (D at half resolution; 1.0–1.5 h at full) |
| Render and eval of each run (4 × about 10 min) | 0.7 h |
| 3c metric study and sheet | 0.3 h |
| **Core total** | **about 4.5 h (5 to 5.5 h with D at full resolution)** |
| Optional E and F | + 0.5 to 1.0 h and + 0.5 h (with their render and eval), **about 5.5 to 7 h in all** |

The PRD's estimate of about an hour per run holds for A and C; B, F and a half-resolution D are much shorter.

## 6. to 9.

Results, recommendation, what adopting it would change, and the sources: after the tests.

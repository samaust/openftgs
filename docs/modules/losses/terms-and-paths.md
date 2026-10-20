# Loss terms and their paths in `openftgs.losses`

How `openftgs.losses` meets its §2.1 row for E7, E8 and E9: the three terms of the rendering loss L_render, the 4D regularization L_reg and the total loss L. The input layout, the API details §2.1 leaves open and the SSIM paths are the owner's answers recorded in `docs/clean-room/questions.md` (Q-41, Q-42). The signature of `total_loss` (D-77) and U-11's reading of the stop-gradient (D-78) entered the spec in revision 549c.

## Input layout

Every image term takes `pred` and `gt` in §1.4's layout, height × width × 3, with any leading dimensions. A single training image and `RenderOutput.image`, which has one leading camera dimension, both fit. The leading dimensions flatten into a batch, and every term averages over the batch, the pixels and the channels (Q-41). One private helper checks the two images and turns them into the batch-first, channel-second view that the convolutions need. On a contiguous channels-last tensor that view is already in channels-last memory format, so nothing is copied; only the fused-ssim path copies, because fused-ssim reads contiguous batch-first tensors. No term clamps the render: E2 clamps colors only at zero, so Î can exceed one, and only the metrics clamp (E7, E16).

## The two SSIM paths

`ssim` returns SSIM itself; the loss term is one minus it. Its constants scale with the data range, as the glossary's SSIM_k does, so evaluation can ask for SSIM₂ with the same function (D-67).

There are two paths behind the one function (§2.4 "Where the time goes", D-52):

- **fused-ssim**, the optional `[fast]` speed-up (Q-2). It hardcodes E7's window, σ, constants and data range, runs on CUDA in float32, and differentiates only its first image. `ssim` uses it only when all of that holds and the target image doesn't require a gradient (Q-42). The module imports it once, lazily, and treats an import failure as "not installed", without a warning. The installed package reports only its version, not the commit the tests ran with (Q-2), so the commit pin is a documented install instruction, not something the code checks.
- **The PyTorch fallback**, used everywhere else: on the CPU, in float64, with other constants, or when the target needs a gradient. It filters the two images, their squares and their product with the separable Gaussian window in one grouped convolution, and forms SSIM from population moments (D-8). It is written from the SSIM definition, not from either library, and its numerator and denominator are built symmetrically, so an image compared with itself gives exactly SSIM's maximum (U-9).

The two paths agree in value. "same" means zero padding in both, and fused-ssim's "valid" is a crop of its zero-padded map, which equals unpadded filtering. Their gradients can differ only where fused-ssim clamps a pixel's SSIM to its mathematical range, a numerical guard the fallback doesn't need (Q-42). On the GPU the fallback's convolution runs in TF32 during training, PyTorch's default, which is D-52's one approximate default; tests that compare the paths clear the flag.

B-4 (#228, Calibration data) measured what fused-ssim gains: its forward and the prediction's backward take 0.41 ms at 1352×1014 and 0.66 ms at 1920×1080, against 8.5 and 13.2 ms for the PyTorch fallback in TF32, about 95 % less, so it stays in use whenever it imports and Q-42 needs no second look.

## The LPIPS wrapper

`LPIPSLoss` wraps the lpips package's network for the backbone `LossConfig.perc_net` names (D-9). It maps its inputs from the image range to the network's documented input range (D-9, D-68) and returns the mean over the batch. lpips freezes only its trunk and puts dropout in its linear layers, so the wrapper keeps the network private, freezes every parameter and keeps the network in evaluation mode even when a parent module calls `.train()` (Q-41). With the weights frozen and a target that doesn't require a gradient, autograd records nothing on the target's branch, which is the separate no-grad target pass §2.4 asks for, without reimplementing lpips.

The network's convolutions already run channels-last: the batch-first view of a [..., H, W, 3] render is channels-last in memory, and every layer keeps that layout on the CPU and on CUDA (source log 175, 176). B-4 (#228, Calibration data) measured the rest of §2.4's plan for this loss: channels-last weights gain 1.0 % and 2.9 % at 1352×1014 and 1920×1080, and cuDNN autotuning 4.3 % and −1.0 %, so neither reaches Q-173's 3 % at both sizes and both stay off; the weights stay NCHW. TF32 is 12 % and 17 % faster than IEEE float32 (D-52's default), and bf16 30 % faster than TF32, with its value within 2.2·10⁻³ relative but its gradient about 18 % off in L2 (`perc_precision = "bf16"`, opt-in; the trainer's private subclass, `docs/modules/train/trainer.md`).

`forward` takes a keyword-only `map_inputs`, on by default. Off, it passes the images to the network unmapped: that is the lpips_vgg_raw metric of D-68, which `openftgs.eval` computes with the same VGG network (Q-58). The rendering loss always maps (D-9).

## The 4D regularization

`reg4d_loss` is the mean of the per-row terms of E8 that `openftgs.motion.reg4d_terms` computes (Q-12), at the training image's time in normalized time, never a raw timestamp (§1.4, Q-41); `total_loss` passes its `t` on unchanged. The stop-gradient on the temporal opacity therefore lives in one place, the motion module's private helper, which the temporal slice shares. The mean runs over all N rows, visible or not (D-11), and an empty model gives zero rather than NaN (Q-41). The function reads the model's parameters when it is called, so a test can substitute them with `torch.func.functional_call` (U-11). Because the stop-gradient leaves L_reg's value moving with μt and ŝ while its gradient through them is zero, U-11's gradient check evaluates L_reg with only the opacity logits substituted (D-78). The terms are recomputed eagerly, one elementwise pass over N, because `render` doesn't hand the trainer its slice. B-1 measured that pass's forward (`loss.reg`) on kitchen at 0.04 to 0.13 ms, 0.04 to 0.11 % of an iteration from 0.5M to 3M rows (#226; its backward is interleaved with the other losses' and can't be timed apart), far under the 5 % at which PRD #220 asks for a change, so it stays.

## The total loss

`total_loss` composes E7 and E9 (D-37) from `LossConfig`:

- The perceptual network is passed in as the keyword `perceptual`, built once by the trainer (D-77).
- The returned dict holds detached zero-dimensional tensors on the images' device, so computing the loss never synchronizes with the host (§2.4 rule 3, D-77). The trainer reads them at its logging period.
- A term whose λ is zero is not computed and has no key for the whole run, which saves the perceptual pass in λ_perc = 0 ablations (Q-41).
- After `reg_until`, which names the last iteration with L_reg (D-10), L_reg leaves the total but is still reported under its key, computed without a gradient. The keys therefore stay fixed for a run, and the loss trace D-10 names shows L_reg after a cutoff (Q-41, owner's answer).
- The iteration index is counted as D-41 counts it.
- A perceptual network built for another backbone than `LossConfig.perc_net` is rejected (D-9, Q-41).

# Metrics in `openftgs.eval`

How `openftgs.eval` meets its §2.1 row for E16: the image metrics, the dynamic-region crop and mask of App. B.1, the averaging of D-67, and the metrics report that `ftgs eval` writes. The spec gives the module no signatures; its public API is Q-60's. `eval` imports only `losses` and `types`. It never renders: renders come from `render.render_request` (Q-57), so the metrics stay pure tensor functions that run on the CPU.

## The metrics

Every metric takes images in the layout of the losses (Q-41) and returns one value per image. The render is clamped to [0, 1] first (D-67); the ground truth is used as loaded (§1.4).

- **PSNR** takes its MSE in float64, so the value doesn't depend on the order of the reduction (Q-60). A perfect render has an infinite PSNR, which the report writes as null (Q-63).
- **SSIM_k** is the loss's SSIM with data range k, called once per image through the public `losses.ssim`, so the metric and the loss share one definition (E7, E16, D-67). k is 1 or 2. **DSSIM_k** is derived from it (D-29, D-40).
- **LPIPS-Alex** and **LPIPS-VGG** run the loss's `LPIPSLoss` wrapper, one network per backbone, built once by `Evaluator`. **lpips_vgg_raw** runs the same VGG network with `map_inputs=False` (D-68, Q-58).

Evaluation is a forward pass with plain float32 convolutions (D-52): every metric runs under inference mode with cuDNN's TF32 flag cleared, and the flag is restored afterwards. On CUDA float32, SSIM₁ may take the fused-ssim path, which agrees with the fallback in value (Q-42).

## The dynamic region

`dynamic_region` turns a render, its ground truth and the test image's mask into the pair that the dynamic-region metrics compare (App. B.1, D-67). The mask is binarized at 0.5, a pixel counting as dynamic at 0.5 and above. Both images are clamped and filled with black outside the mask, then cropped to the mask's bounding box, the same box for both (Q-61).

A mask with no dynamic pixel, or a box too small for every metric to be defined, gives no pair and a warning, and the report counts that image as skipped (Q-61). The minimum side comes from the largest window of the metrics, LPIPS-Alex's.

## Averaging and the backbone

Each metric is computed per test image, averaged over a scene's test images, then over scenes, never pooled (D-67). `average_metrics` serves both levels. `lpips_backbone` picks the backbone that the paper reports for a dataset (D-29). It reads it from the scene manifest's `metadata.source`: Neural3DV is AlexNet, and every other dataset, including synthetic scenes, is VGG (Q-62). The report carries all three LPIPS values in either case; the backbone only says which one the paper's tables compare to.

## The metrics report

`metrics_report` builds the body of METRICS.json from per-image results and the scene manifest's `metadata` (Q-63). It holds:

- the full-frame metrics under `entire`;
- the dynamic-region metrics under `dynamic`, when any test image is masked, with the mask source and the approximate flag of D-64;
- under `fastest_10`, the means over the Fastest-10 frames, when the manifest names them (D-69).

M14's `ftgs eval` adds the header (model, scene and producer) and writes the file without NaN or Infinity, as every JSON file of the package is written (Q-25).

## The Fastest-10 frames

The selection itself is not in `eval`. D-69 has the converter store the range in the manifest, and the converter is in `data`, which may not import `eval`, so the selection is `openftgs.data.fastest_frames` (Q-59). It streams the test camera's frames and returns the start of the window. M12's converter maps that start to the manifest's `frame_index` values, and the report selects the test images by `frame_index`.

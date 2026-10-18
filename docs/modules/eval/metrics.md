# Metrics in `openftgs.eval`

How `openftgs.eval` meets its §2.1 row for E16: the image metrics, the dynamic-region crop and mask of App. B.1, the averaging of D-67, and the metrics report that `ftgs eval` writes. The spec gives the module no signatures; its public API is Q-60's, with `check_nan_metrics` (Q-148). `eval` imports only `losses`. It never renders: renders come from `render.render_request` (Q-57), so the metrics stay pure tensor functions that run on the CPU.

## The metrics

Every metric takes images in the layout of the losses (Q-41) and returns one value per image. The render is clamped to [0, 1] first (D-67); the ground truth is used as loaded (§1.4).

- **PSNR** takes its MSE in float64, so the value doesn't depend on the order of the reduction (Q-60). A perfect render has an infinite PSNR, which the report writes as null (Q-63).
- **SSIM_k** is the loss's SSIM with data range k, called once per image through the public `losses.ssim`, so the metric and the loss share one definition (E7, E16, D-67). k is 1 or 2. **DSSIM_k** is derived from it (D-29, D-40).
- **LPIPS-Alex** and **LPIPS-VGG** run the loss's `LPIPSLoss` wrapper, one network per backbone, built once by `Evaluator`. **lpips_vgg_raw** runs the same VGG network with `map_inputs=False` (D-68, Q-58).

Evaluation is a forward pass with plain float32 convolutions (D-52): every metric runs under inference mode with cuDNN's TF32 flag cleared, and the flag is restored afterwards. `Evaluator` computes on a float32 copy of the pair, the LPIPS networks' dtype, and derives DSSIM_k from the SSIM_k it has just computed (Q-60). On CUDA float32, SSIM₁ may take the fused-ssim path, which agrees with the fallback in value (Q-42).

## The dynamic region

`dynamic_region` turns a render, its ground truth and the test image's mask into the pair that the dynamic-region metrics compare (App. B.1, D-67). The mask is binarized at 0.5, a pixel counting as dynamic at 0.5 and above. The render is clamped, as for every metric; then both images are filled with black outside the mask and cropped to the mask's bounding box, the same box for both (Q-61).

The box is the bounding box of every dynamic pixel, as D-67 defines it, so a single stray mask pixel far from the subject enlarges the crop with black-on-black pixels, which raise PSNR and SSIM and lower LPIPS; masks are the user's to supply and clean (D-64), and the package doesn't filter them. A mask with no dynamic pixel, or a box too small for every metric to be defined, gives no pair and a warning, and the report counts that image as skipped (Q-61). The minimum side comes from the largest window of the metrics, LPIPS-Alex's. Masks come in as bool, or floating in [0, 1]: an integer mask is refused, so whoever reads a mask file (M10's manifest reader, M14's `ftgs eval`) divides an 8-bit mask by 255 first, as §1.4 does for every 8-bit input (Q-61).

## Averaging and the backbone

Each metric is computed per test image, averaged over a scene's test images, then over scenes, never pooled (D-67). `average_metrics` serves both levels. `lpips_backbone` picks the backbone that the paper reports for a dataset (D-29). It reads it from the scene manifest's `metadata.source`: Neural3DV is AlexNet, and every other dataset, including synthetic scenes, is VGG (Q-62). The report carries all three LPIPS values in either case; the backbone only says which one the paper's tables compare to. The match is exact and case-sensitive, so M12's Neural3DV converter writes the source as "n3dv/<scene>", as in §2.2's example (Q-62).

## The metrics report

`metrics_report` builds the body of METRICS.json from per-image results and the scene manifest's `metadata` (Q-63). It holds:

- the full-frame metrics under `entire`;
- the dynamic-region metrics under `dynamic`, when any test image is masked, with the mask source and the approximate flag of D-64;
- under `fastest_10`, the means over the Fastest-10 frames, when the manifest names them (D-69).

`write_metrics` adds the header and writes the file without NaN or Infinity, as every JSON file of the package is written (Q-25). It replaces an existing file atomically: a temporary sibling is renamed over it, so a write that fails partway leaves the old file (Q-126, Q-146). It is the one writer of METRICS.json, for `Trainer.evaluate` and `ftgs eval` alike (Q-136). The header holds `producer`; `model`, as {`path`, `iteration`}, either of which may be null (a trainer's model has no file, an exchange model no iteration); and the scene manifest as `manifest` ({`path`, `sha256`}), because `scene` already holds the per-scene means (Q-63). When `ftgs eval` scores supplied renders, a `renders` entry ({`dir`, `request_sha256`, `producer`}) says what was scored (Q-137).

The `dynamic` block counts under `missing` the images of the split that have no mask, 0 for a full set, so a partial set is visible in the report (Q-135); `ftgs eval` and `ftgs train` refuse a partial set before doing any work.

A null in the report means "not defined here": a perfect render's PSNR, a mean over no images, a dynamic region that was skipped (Q-61). It never stands for a failed render: `check_nan_metrics` raises `FloatingPointError`, naming up to five images, when a metric of an image's entire frame or dynamic region is NaN, which only a render that isn't finite gives, and `Trainer.evaluate` and `ftgs eval` call it before anything is written (Q-129, Q-147, Q-148).

## The Fastest-10 frames

The selection itself is not in `eval`. D-69 has the converter store the range in the manifest, and the converter is in `data`, which may not import `eval`, so the selection is `openftgs.data.fastest_frames` (Q-59). It streams the test camera's frames and returns the start of the window. M12's converter scores the test camera's frames as it stores them, after undistortion and resizing, so the range can be recomputed from the files (D-69), and maps the start to the manifest's `frame_index` values, by which the report selects the test images (Q-59, Q-63).

## Evaluating a model on a scene

`evaluate_images` evaluates a model on the images of one data split of a multi-view video, for `Trainer.evaluate` now and `ftgs eval` in M14 (Q-68). It renders through `render.render_request`, one request per timestamp, so cameras that share a time render in one call (D-55) and only one time's renders are held at once; it never calls the render backends itself. Each image is evaluated by the `Evaluator` the caller passes, which holds the LPIPS networks, with the test image's mask when it has one. The results come back as `ImageMetrics` in the manifest's order, each with its path relative to the manifest, as the metrics report lists them. Given `renders`, a prediction for every record of the split by record index, it renders nothing and scores those instead; `ftgs eval --renders-dir` uses it, so supplied renders and the model's own go through the same evaluator and report (Q-137).

# Image processing and cameras in `openftgs.data`

The package's own replacements for OpenCV in the core (D-61): area resizing, undistortion and the OpenCV-style YAML reader, and the camera helpers the converters build cameras with. Their signatures are Q-84's. OpenCV appears only in the tests, as an oracle (§2.3).

## Conventions, in order

A calibration in OpenCV's integer-centre convention is used as it is to undistort, then converted to the continuous pixel convention by adding 0.5 to cx and cy (§1.4, `camera_from_opencv`), then resized. Undistorting after the shift would move every source position by half a pixel; U-35's map tolerance would catch it. Neural3DV's poses carry cx = W/2, cy = H/2, already the image centre in the continuous convention, so `camera_from_llff` adds nothing (D-28). COLMAP's read-back cameras are in the continuous convention too (`converters.md`).

## Area resizing

`resized_size` gives W′ = round(W·r) and H′ = round(H·r), an exact half rounding up, at least 1 (Q-89). `resize_area` treats every input pixel as a constant square and gives each output pixel the mean over its footprint, the footprint taken from the realized ratios W/W′ and H/H′ (D-33). It is separable, with banded weights in float64. uint8 input gives uint8 output rounded half to even; floating input gives float64. `resize_camera` scales fx and cx by W′/W and fy and cy by H′/H.

## Undistortion

`undistortion_map(K, dist, width, height)` gives, for every output pixel, the source position in the distorted image, by the per-pixel steps of OpenCV's documented `initUndistortRectifyMap` with R = I and K_new = K: normalize through K, apply the distortion model (radial k1–k6, tangential p1, p2, thin-prism s1–s4), map back through K. It accepts 0, 4, 5, 8 or 12 coefficients; 14 coefficients (the tilted model) is refused, as D-61 says. `undistort_image` samples the image bilinearly at those positions, with pixels outside the image counting as 0. Zero coefficients return the input unchanged.

## The YAML reader

`read_opencv_yaml` reads a file written by OpenCV's FileStorage with PyYAML (Q-91). A `%YAML:1.0` first line, which isn't valid YAML, is dropped; OpenCV 5's `%YAML 1.2` is kept. An `!!opencv-matrix` becomes a NumPy array of its rows, columns and channels, in the dtype its `dt` names, so values equal what OpenCV itself reads back. `.Nan`, `.Inf` and `-.Inf` become floats, which PyYAML would otherwise leave as strings.

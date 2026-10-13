# The multi-view video in `openftgs.data`

How `MultiViewVideo` meets its §2.1 row: the image records of a scene manifest, with their time normalizer, spatial scale S, frame interval Δt_f, and lazily loaded images and test-image masks. The members beyond §2.1's four are a provisional decision (Q-65).

## From the file to the dataset

The scene manifest is parsed in `openftgs.io` (owner answer, Q-64). `read_scene_manifest` returns a plain `SceneManifest` with the records in file order, the mask paths aligned with them, the metadata, the time unit and the SHA-256 of the file's bytes. `MultiViewVideo(manifest)` builds the dataset from it, and `from_manifest(path)` does both. The constructor checks that every image and mask file exists, so a missing file fails before training starts, not hours into it.

## Time, Δt_f and S

All three are read from the training images only (data split "train"), so the test view never influences training (D-27, D-66):

- the time normalizer spans the smallest and largest training timestamps (§1.4 "Time");
- Δt_f comes from `types.frame_interval` with the training timestamps grouped by camera name (E14, D-46, Q-8). Records are grouped by name, never by comparing `Camera` objects, whose generated `==` compares tensors (`docs/modules/types/time.md`);
- S is `spatial_scale(factor)`. The factor is `TrainConfig.scene_scale_factor`, which the dataset never sees, so the caller passes it (D-27, Q-65). The centers are those of the distinct training poses, computed in float64 as E2's camera center; a pose repeated by every frame of a camera counts once, and an image's own pose override counts as its own center. With fewer than two distinct centers S would be 0, which the optimizer would reject later, so `spatial_scale` raises at once.

## Images and masks

`load_image` decodes the file at every call (the frame cache of D-51 comes in M12). It accepts 8-bit RGB and RGBA files, drops the alpha channel as §1.4 says, checks the size against the record's camera, and returns float32 values divided by 255. Other modes are refused rather than converted, since a conversion (palette, grayscale, 16-bit) would be a choice the spec doesn't make (Q-65).

`load_mask` returns None for an image without a mask. A mask is a bilevel or 8-bit grayscale image of the image's size; 8-bit values are divided by 255, as Q-61 asks of mask readers, so `eval.dynamic_region` sees values in [0, 1]. Only test images carry masks (§2.2).

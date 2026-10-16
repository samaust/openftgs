# File formats in `openftgs.io`

How `openftgs.io` meets its §2.1 row: plain data in and out for the `ftgs-checkpoint`, `ftgs-if` and `ftgs-init` formats of §2.2, and a PLY snapshot for viewers. It imports neither `model` nor `train`, so it keeps its own copy of the nine tensor names, which a test pins to `FreeTimeGaussians.PARAM_NAMES`. Every rejection is a `ValueError` (Q-26).

## Checkpoint directory

A checkpoint holds exactly the files of the §2.2 table (Q-25):

- `manifest.json` carries the scalar fields of `TrainerState`, the configuration, the scene hash and time unit (Q-17), the library versions and the SHA-256 of every other file;
- `params.safetensors` and `stats.safetensors` hold tensors only;
- the optimizer and generator states are split by one generic rule: every tensor, at any depth of the nested dict, goes to the `.safetensors` file under its key path joined with `/`, and everything else stays in the `.json` file; loading merges them back;
- `sampler.json` is JSON only.

Tensors are looked for in nested dicts only: a tensor inside a list is refused, so per-device states go in a dict keyed by device. The safetensors files carry the format and version in their metadata; the JSON files are covered by the manifest's. A save also checks the stored-tensor dtype and the length of the statistics.

JSON is the round trip's limit: tuples come back as lists, and non-finite floats are refused. Loading checks the format, the major version, that every listed file is there and every hash before it builds the `TrainerState`, whose tensors are on the CPU; files the manifest doesn't list are ignored. It also applies the §1.1 validity rule on N: the stored configuration must be valid and N within its cap N_max, so every reader of a checkpoint gets the check.

A save is atomic: the files go into a temporary sibling directory, which replaces the target by renames, the old directory being moved aside first and removed last. It never replaces a directory that isn't a checkpoint, unless that directory is empty. POSIX can't swap two directories in one rename, so a kill between the two renames leaves the old checkpoint set aside next to the target; the next save of that target restores the most recently modified one and clears the leftovers (Q-25). Two processes saving to one target at once are not supported.

## Exchange and init-point files

`ftgs-if` and `ftgs-init` files are validated on write and on read by the same functions that `openftgs.model` calls on in-memory data (Q-22): exact dtypes and shapes, finite values, and the ranges of §2.2. `ftgs-if`'s `render_conventions` metadata records the renderer the file was made for: E6's constants, including the Jacobian clamp margin and the screen-extent rule that D-73 added, D-6's rasterize mode and D-34's background, the newer entries under the keys of Q-35. It is ignored on read, since the exchange model has no field for it (Q-24), so files written before revision 549a, without D-73's entries, read the same. Metadata follows the §2.2 introduction: integers in decimal, floats in shortest round-trip form, structured values as compact JSON; unknown keys are ignored and an unknown major version is refused.

`write_init_points` also takes extra metadata, string to string, that it writes beside its own keys; a reserved key (`format`, `version`, `time_unit`, `producer`), or a key or value that isn't a string, is refused before anything is written. `roma_init`'s provenance reaches the file this way: `ftgs init` (M14) passes `openftgs.init.roma_provenance(cfg)`, whose one key, `roma_init`, holds the settings, the resolved `use_custom_corr` and the commits of the vendored RoMa and, for Tiny RoMa, of XFeat, as compact JSON with sorted keys (Q-107, Q-T136.2). `read_init_points` doesn't return metadata; it ignores keys it doesn't know, as §2.2 asks.

## PLY snapshot

`write_ply` writes per-time arrays, as physical values, in the layout 3DGS viewers read, with logit opacities and log scales (Q-18). It is export-only: no reader exists, and no other part of the package consumes it.

## Render requests and renders

`read_render_request` reads a `ftgs-render-request` file into the `RenderRequest` of `openftgs.types`, and `write_renders` and `read_renders` write and read its `ftgs-renders` answer as `Renders` (§2.2, Q-56). The required and optional keys, and the rules for item ids and cameras, are Q-56's. The cameras are checked more strictly than `render()` checks its own: `render()` reads only the focal lengths, the principal point and the world-to-camera rotation and translation (Q-32), so a skewed K or a malformed last row would otherwise be dropped without a word. One private camera parser does the reading, shared with the scene-manifest reader.

A renders file records the producer and the SHA-256 of the request file's bytes as read, which the caller computes and the writer takes as 64 lowercase hex digits; reading with a hash rejects a file made for another request, as `read_init_points` rejects another time unit (Q-22). There is no request writer: the harness writes its own requests, and the tests write the JSON by hand.

## Scene manifests

`read_scene_manifest` reads a `ftgs-scene` file into the `SceneManifest` of `openftgs.types`: the image records in file order, the test images' mask paths aligned with them, the metadata, the time unit and the SHA-256 of the file's bytes as read, which the trainer stores in its checkpoints and compares on a resume (Q-17, Q-64). `camera_names` keeps the key order of the manifest's `cameras` object, which the records alone lose; it is the "manifest's camera order" that orients D-22's camera pairs (Q-108). Parsing lives here, beside the other formats, and `openftgs.data` builds the multi-view video from the result (owner answer, Q-64); M12's manifest writer will live here too.

Cameras go through the same parser as render requests, with the same checks (Q-56); a camera's `model`, when present, must be "pinhole", since the images are undistorted (§2.2). An image may override its camera's K or world-to-camera matrix; the override is checked the same way and gives that record a camera of its own, with the camera's name, while the other images share their camera's object. Paths must be relative and are resolved against the manifest's directory (§2.2); only test images may carry a mask; at least one image must be in the train split. Unknown keys are ignored. Whether the files exist is checked by the dataset, not here.

## Writing a scene manifest

`write_scene_manifest` writes the `ftgs-scene` file the converters produce, beside `read_scene_manifest` (Q-64, Q-84). It takes image records, their masks and the metadata; one camera entry per camera name comes from that name's first record, and a later record whose K or pose differs is written as a per-image override (§2.2). Camera values are written as the shortest decimals that read back to the same float32 values, so the reader rebuilds the converter's cameras exactly; timestamps keep their float64 value. Paths are written relative to the manifest. The file goes to a temporary name, is read back with `read_scene_manifest` and is renamed into place, and the writer returns what the reader returns, hash included.

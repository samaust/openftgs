# File formats in `openftgs.io`

How `openftgs.io` meets its §2.1 row: plain data in and out for the `ftgs-checkpoint`, `ftgs-if` and `ftgs-init` formats of §2.2, and a PLY snapshot for viewers. It imports neither `model` nor `train`, so it keeps its own copy of the nine tensor names, which a test pins to `FreeTimeGaussians.PARAM_NAMES`. Every rejection is a `ValueError` (Q-26).

## Checkpoint directory

A checkpoint holds exactly the files of the §2.2 table (Q-25):

- `manifest.json` carries the scalar fields of `TrainerState`, the configuration, the scene hash and time unit (Q-17), the library versions and the SHA-256 of every other file;
- `params.safetensors` and `stats.safetensors` hold tensors only;
- the optimizer and generator states are split by one generic rule: every tensor, at any depth of the nested dict, goes to the `.safetensors` file under its key path joined with `/`, and everything else stays in the `.json` file; loading merges them back;
- `sampler.json` is JSON only.

Tensors are looked for in nested dicts only: a tensor inside a list is refused, so per-device states go in a dict keyed by device. The safetensors files carry the format and version in their metadata; the JSON files are covered by the manifest's. A save also checks the stored-tensor dtype and the length of the statistics.

JSON is the round trip's limit: tuples come back as lists, and non-finite floats are refused. Loading checks the format, the major version, that every listed file is there and every hash before it builds the `TrainerState`, whose tensors are on the CPU; files the manifest doesn't list are ignored.

A save is atomic: the files go into a temporary sibling directory, which replaces the target by renames, the old directory being moved aside first and removed last. It never replaces a directory that isn't a checkpoint, unless that directory is empty. POSIX can't swap two directories in one rename, so a kill between the two renames leaves the old checkpoint set aside next to the target; the next save of that target restores it and clears the leftovers (Q-25).

## Exchange and init-point files

`ftgs-if` and `ftgs-init` files are validated on write and on read by the same functions that `openftgs.model` calls on in-memory data (Q-22): exact dtypes and shapes, finite values, and the ranges of §2.2. `ftgs-if`'s `render_conventions` metadata is written from E6's constants and ignored on read, since the exchange model has no field for it (Q-24). Metadata follows the §2.2 introduction: integers in decimal, floats in shortest round-trip form, structured values as compact JSON; unknown keys are ignored and an unknown major version is refused.

## PLY snapshot

`write_ply` writes per-time arrays, as physical values, in the layout 3DGS viewers read, with logit opacities and log scales (Q-18). It is export-only: no reader exists, and no other part of the package consumes it.

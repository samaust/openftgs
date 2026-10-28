# Command line

`ftgs` is openftgs's command line (§2.1, D-62): `convert`, `cache`, `init`, `train`, `render`,
`eval`, `export` and `bench`. `python -m openftgs.cli` runs the same program. Each command parses
its options and calls the library, so nothing is computed here that a Python caller couldn't do
with the same functions. `ftgs <command> --help` prints the text below (Q-242).

MODEL is either a checkpoint directory or an FTGS-IF exchange file (§2.2). ftgs train's `--seed`
replaces the configuration file's seed, and `--force` resumes despite a changed scene manifest.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success; for `ftgs bench`, the benchmark ran, whatever its budgets say |
| 1 | an unexpected exception, with its traceback |
| 2 | invalid input or environment: a usage error, a file that fails validation, a missing ffmpeg, ffprobe or COLMAP, a benchmark other than `ftgs bench convert` without a CUDA device, a configuration this build can't run, a missing `[init]` extra (for `init roma`, `ftgs bench init` and a DINO perceptual network), a trunk network's weights that can't be downloaded, whose cached file fails its check or whose conversion can't be written into torch.hub's cache, RoMa weights missing from torch.hub's cache for `ftgs bench init`, trunk network weights missing from torch.hub's cache for `ftgs bench losses`, which never downloads them and names each missing file, a recorded source video that can't be found for `ftgs bench convert`, a frame cache that doesn't fit in GPU memory, training or a benchmark that runs out of GPU memory, an output that can't be written |
| 3 | a numerical failure: training diverged, in `ftgs train` or a benchmark's training steps, or a render isn't finite (in `ftgs render`, `ftgs eval`, `ftgs train`'s final evaluation and `ftgs bench render`) |
| 130, 143 | `ftgs train` stopped by SIGINT or SIGTERM, after saving its checkpoint |

## `ftgs-adapter`

openftgs installs a second program, `ftgs-adapter`, for the differential harness of spec §3.3,
which compares two implementations of the method through files and commands only. It answers
the harness's adapter contract (`info`, `export`, `import`, `render`, `forward` and `train`, on the
`reference` or the `gsplat` backend) with openftgs's own functions, and keeps `ftgs`'s command list
as §2.1 gives it (Q-202). `ftgs-adapter --help` describes it. The harness itself is a separate
package, in the repository's `harness/` folder, and isn't released.

## Reference

```{argparse}
:module: openftgs.cli
:func: _parser
:prog: ftgs
:nodefaultconst:
```

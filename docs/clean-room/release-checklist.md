# Release checklist

M18's record of the release-time checks that spec §2.3's notes and D-79 call for, for openftgs 0.1.0 (PRD #355): the license scan and its sign-offs, the libraries bundled in dependency wheels, the vendored license files, the pretrained weights' terms, the absence of dataset material, and the source-log audit. Each section gives the commands run, the commit and environment they ran on, and their results; a sign-off is the owner's, given in the build session, and is recorded with its date.

## License scan

To be written by #366.

## Libraries bundled in dependency wheels

To be written by #366.

## The vendored RoMa

D-65; §2.3 notes ("checks the vendored directory's license files"); Q-113, Q-115 and Q-118, whose detecting test is this audit. Run on 2026-10-22 by ticket #367, on commit 7e1faf0 of `prd-355-release`, with the project's `.venv` (Python 3.14.8) and git 2.43.0.

**The upstream clone.** A blobless clone of RoMa (MIT), `git clone --filter=blob:none --no-checkout https://github.com/Parskatt/RoMa <clone>`, into a new, empty folder, `<clone>` = `.scratch/t367-roma/RoMa` inside the ticket's worktree (gitignored; the session's tooling refused git commands on the main checkout's `.scratch/`). Its HEAD is still 77f8d68803526dcddfd9b7a46bc76125bdc25f15 (2026-01-23). Its `LICENSE` was read before anything else: MIT, Copyright (c) 2023 Johan Edstedt, git blob ca95157, the blob of source log row 106. Source log row T367.1.

**The vendoring script's check mode.** Run from the worktree, not from the clone, with the clone's path as an argument:

```
.venv/bin/python -I scripts/vendor_roma.py --source <clone> --check
```

It rebuilds the copy in a temporary folder from the 22 inference files at 77f8d68 (each blob id checked against the commit's tree) and the eight license texts, fetched again at their pinned commits and checked against their SHA-256 (source log row T367.2), and compares every file with `src/openftgs/_vendor/romatch`. Result, exit 0:

```
22 files (3798 upstream lines), 10 changed; romatch imports rewritten: 12 at module level, 2 in RegressionMatcher.visualize_warp, TinyRoMa.visualize_warp; 8 license files
check: the copy and VENDOR.json match the rebuild
```

So the vendored copy, its eight license files and `VENDOR.json` are identical, byte for byte, to a fresh rebuild: Q-113's cut of the pose helpers and Q-115's modification notices included, and InLoc's text extracted from its README as Q-118 says (SHA-256 b35cbfd4… of the README, 95970e9a… of the text).

**NOTICE against VENDOR.json.** For each entry of `VENDOR.json`'s `licenses`, the file's SHA-256 was recomputed and NOTICE was searched for the project's name and for the file's path in parentheses:

```python
# .venv/bin/python -I, from the repository root
import hashlib, json, pathlib
root = pathlib.Path("src/openftgs/_vendor/romatch")
record = json.loads((root / "VENDOR.json").read_text())
notice = pathlib.Path("NOTICE").read_text()
for e in record["licenses"]:
    sha = hashlib.sha256((root / e["path"]).read_bytes()).hexdigest()
    named = f"({e['path']})" in notice and e["project"] in notice
    print(e["path"], e["project"], e["license"], named, sha == e["sha256"])
print(record["commit"] in notice)
```

| License file | Project | License | Named in NOTICE | SHA-256 as recorded |
| --- | --- | --- | --- | --- |
| `LICENSE` | RoMa | MIT | yes | yes |
| `LICENSES/DINOv2.txt` | DINOv2 | Apache-2.0 | yes | yes |
| `LICENSES/DINO.txt` | DINO | Apache-2.0 | yes | yes |
| `LICENSES/timm.txt` | timm | Apache-2.0 | yes | yes |
| `LICENSES/LoFTR.txt` | LoFTR | Apache-2.0 | yes | yes |
| `LICENSES/Patch2Pix.txt` | Patch2Pix | MIT | yes | yes |
| `LICENSES/Kornia.txt` | Kornia | Apache-2.0 | yes | yes |
| `LICENSES/InLoc.txt` | InLoc | MIT | yes | yes |

NOTICE also names RoMa's commit, 77f8d68803526dcddfd9b7a46bc76125bdc25f15, as `VENDOR.json` does. NOTICE names every license text that `VENDOR.json` records, and no other.

**The license files ship.** The wheel built for "No dataset material" below holds `openftgs/_vendor/romatch/LICENSE`, the seven files of `openftgs/_vendor/romatch/LICENSES/` and `openftgs/_vendor/romatch/VENDOR.json` (pyproject's package data), and `LICENSE` and `NOTICE` under `openftgs-0.1.0.dev0.dist-info/licenses/`.

**U-38**, CPU part, on the same commit: `tests/test_vendored_roma.py` and `tests/test_skeleton.py` with `-m "not gpu and not weights"`, 56 passed, 1 deselected (U-38's GPU comparison with upstream romatch, `test_u38_vendored_match_agrees_with_upstream_romatch`, which this ticket's CPU budget doesn't run).

## Pretrained weights' terms

To be written by #367.

## No dataset material

To be written by #367.

## Source-log audit

To be written by #368.

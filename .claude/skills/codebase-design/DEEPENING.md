# Deepening

How to deepen a cluster of shallow modules safely, given its dependencies. Assumes the vocabulary in [SKILL.md](SKILL.md): **module**, **interface**, **seam**, **adapter**.

## Dependency categories

When assessing a candidate for deepening, classify its dependencies. The category determines how the deepened module is tested across its seam.

### 1. In-process

Pure computation, in-memory state, no I/O: motion and temporal opacity, the SH basis, the losses, the relocation move. Always deepenable: merge the modules and test through the new interface directly. No adapter needed.

### 2. Local-substitutable

Dependencies that have a local stand-in the spec already provides or the tests can generate: gsplat (the pure-PyTorch reference backend), datasets (generated mini-fixtures), the filesystem (pytest's `tmp_path`). Deepenable if the stand-in exists. The deepened module is tested with the stand-in running in the test suite. The seam is internal; no port at the module's external interface.

### 3. Owned, across a process boundary (Ports & Adapters)

Our own code on the far side of a process or file boundary: the differential harness drives an implementation through the command line and the §2.2 file formats, never by import (§3.3). Define a **port** at the seam. The deep module owns the logic; the transport is injected as an **adapter**. Tests use an in-process adapter; production uses the command-line adapter.

Recommendation shape: *"Define a port at the seam, implement a command-line adapter for the harness runs and an in-process adapter for testing, so the logic sits in one deep module even though it runs across a process boundary."*

### 4. True external (stub or gate)

Software and data we don't control: the separate programs (ffmpeg, COLMAP, BackgroundMattingV2) and the pretrained weights downloaded at run time (RoMa, DINOv2, the LPIPS networks). The deepened module takes the external dependency as an injected port; tests provide a stub adapter (a stub executable that records its input, as U-37 does for COLMAP) or run behind the marker for tests that need downloads.

## Seam discipline

- **One adapter means a hypothetical seam. Two adapters means a real one.** Don't introduce a port unless at least two adapters are justified (typically production + test). A single-adapter seam is just indirection.
- **Internal seams vs external seams.** A deep module can have internal seams (private to its implementation, used by its own tests) as well as the external seam at its interface. Don't expose internal seams through the interface just because tests use them.

## Testing strategy: replace, don't layer

- Old unit tests on shallow modules become waste once tests at the deepened module's interface exist; delete them. The spec's §3 tests (U-n, C-n, S-n, B-n) are the exception and are never deleted: they pin the spec whatever shape the module takes. If deepening leaves one testing past the new interface, that is a sign the deepening crosses a §2.1 signature, not a reason to drop the test.
- Write new tests at the deepened module's interface. The **interface is the test surface**.
- Tests assert on observable outcomes through the interface, not internal state.
- Tests should survive internal refactors, since they describe behaviour, not implementation. If a test has to change when the implementation changes, it's testing past the interface.

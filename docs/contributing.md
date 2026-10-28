# Contributing

openftgs is a clean-room implementation: before contributing code, read the clean-room rules in
the repository's `CLAUDE.md` and §0 of {repo}`docs/spec/openftgs-spec.md`. In short, code is
written from the paper and the specification only; nothing from any other implementation of the
method is consulted; third-party source code is read only under a permissive license that allows
commercial use; and every source consulted gets a row in {repo}`docs/clean-room/source-log.md`.
Where the spec is silent, a provisional decision is recorded as a Q-n row in
{repo}`docs/clean-room/questions.md`.

The development install and the tests are in the repository's README: `pip install -e .[dev]`,
then `pytest`.

## Building the documentation

The site's sources are the pages under `docs/` (`index.md`, the guides, `api/index.rst`) and
`docs/conf.py`; everything else under `docs/` (the spec, the clean-room records, the module docs,
the glossary) stays outside the build and is linked on GitHub. Its tools are the dependency group
`docs` (Q-241), which never enters the package's metadata. The API pages import the package, so
PyTorch comes first; a CPU build is enough:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . --group docs
sphinx-build -W --keep-going -b html docs docs/_build/html
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

- The build needs network access: intersphinx fetches the Python, NumPy, SciPy and PyTorch
  inventories, so that `torch.Tensor` and `numpy.ndarray` link to their documentation.
- `-W` turns every warning into an error, and `nitpicky` is on, so every cross-reference must
  resolve. Read the Docs and the docs workflow build on Python 3.11; build there too before a
  pull request, since a NumPy `Returns` type line is cross-referenced on 3.11's tool versions.
- `docs/_build/` and the API stubs autosummary writes, `docs/api/generated/`, are ignored by git;
  remove both before a clean build.
- linkcheck runs locally only. GitHub's anchors and PyTorch's stable pages are exempt from its
  anchor checks.

## Docstrings

Every name in a public module's `__all__`, and every public method of a public class, has a NumPy
docstring (Q-243); the API pages render them.

- A one-line imperative summary that cites the governing spec item, then an extended summary.
- `Parameters`, and `Returns` or `Yields` for a function that returns or yields a value.
  Parameters carry no types: the annotations supply them. The type line under `Returns` is a
  resolvable name (`torch.Tensor`, `int`, `openftgs.render.RenderOutput`), never a description.
  A function that returns None has no `Returns` section, except where the Python 3.11 build
  needs one: there sphinx-autodoc-typehints 3.6.1 can put its "Return type" line under the
  `Notes` heading and fail the build, which a `Returns` section holding just `None` prevents
  (`FTGSOptimizer.step`, `write_metrics`).
- `Raises` where the code raises for user input; `Notes` for the other spec citations;
  `Examples` on the entry points users call directly. Examples are shown, not run.
- Tensors and arrays state their shape, dtype and range or units, as `[N, 3] float32`. Config
  fields name their provenance tag. A class's docstring documents its constructor's parameters
  under `Parameters`, and a dataclass its fields there too, so each shows with its type.
- Cross-references use full dotted paths across modules (``:func:`~openftgs.render.render` ``);
  code in text takes double backticks. A module constant gets an attribute docstring on the line
  after its assignment.
- Comments, including the `# Q-n` and `# D-n` markers, are never changed by docstring work.

`main` in `openftgs.cli` is a full example:

```python
def main(argv: Sequence[str] | None = None) -> int:
    """Run ``ftgs`` with ``argv`` and return its exit code (§2.1).

    Parse the arguments with the parser of §2.1's CLI block, set PyTorch's intra-op threads to
    the physical cores, unless ``OMP_NUM_THREADS`` or ``MKL_NUM_THREADS`` is set, then run the
    command and map its failures to §2.1's exit codes. ``--help`` and a usage error run no
    command and change nothing.

    Parameters
    ----------
    argv
        The arguments after the program's name, as ``["train", "scene.json", "--init",
        "init.safetensors", "--out", "run"]``; None, the default, reads the process's
        arguments, ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code: 0 success (``--help`` included), 2 invalid input or environment (a
        usage error included), 3 a numerical failure, and 128 plus the signal's number when
        SIGINT or SIGTERM stops ``ftgs train`` after it saved its checkpoint. Exit 2 covers a
        trunk network's weights that can't be downloaded, fail their check or can't be
        converted (Q-142, Q-258), a DINO network without the ``[init]`` extra (Q-248), and
        resuming from a checkpoint saved before revision 549o (Q-247).

    Notes
    -----
    Q-124: the module's one public name, run by the ``ftgs`` console script and by
    ``python -m openftgs.cli``. A failure the CLI reports is one line on stderr,
    ``ftgs <command>: error: <message>``, followed by a warning line for each failure that
    came after it (Q-159); any other exception is a bug and keeps its traceback (Q-26,
    Q-142). Signals: Q-133. Threads: §2.4, Q-176.

    Examples
    --------
    Train on a converted scene from Python, as ``ftgs train`` would:

    >>> from openftgs.cli import main
    >>> main(["train", "scene/scene.json", "--init", "init.safetensors", "--out", "run"])
    0
    """
```

### Ruff's `D` checks

`ruff check .` runs pydocstyle's rules, `D`, with the NumPy convention, plus D417, which requires
every parameter to be documented once a `Parameters` section exists (Q-243). Ruff follows each
module's `__all__`, so the checks apply to the public API. Exempt: the tests, the private modules
(the CLI's included), the differential harness, which is a separate distribution (Q-5), and the
maintainer scripts under `scripts/`. The documentation's `conf.py` is checked. `ruff format
--check .` and `mypy` pass as well.

### The no-runtime-change check

A change to docstrings must change nothing else. `scripts/no_runtime_change.py` compares each
file's syntax tree, with every docstring and other bare string statement removed, against a git
ref, `main` by default:

```bash
python scripts/no_runtime_change.py src/openftgs/sh.py src/openftgs/model.py
python scripts/no_runtime_change.py --ref v0.1.0 src/openftgs/render.py
```

It exits 0 when every file matches the ref, 1 when one differs (naming it) and 2 for invalid input.
Comments aren't in the syntax tree, so leaving them alone is up to you.

The CLI's help text is not a docstring: `help=` and `description=` values are runtime strings,
shown by `--help` (Q-242). Against a ref older than that help text, the check therefore names
`src/openftgs/cli/__init__.py` and `src/openftgs/cli/_bench.py`; for them, compare the diff by
hand and check that only those values and docstrings differ.

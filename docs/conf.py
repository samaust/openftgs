# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration of the openftgs documentation site (PRD #355, Q-241).

Build it from the repository root, with network access for the intersphinx inventories:

    sphinx-build -W --keep-going -b html docs docs/_build/html

Only the pages of the site are sources. Everything else already under ``docs/`` (the spec, the
clean-room records, the agent and module docs, the research notes, the glossary) stays where it is,
outside the build, and the site links it on GitHub.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx

_DOCS = Path(__file__).resolve().parent
# autodoc imports the source tree next to this file, so a build from any checkout or worktree
# documents that checkout's code, whatever copy of the package is installed.
sys.path.insert(0, str(_DOCS.parent / "src"))

# -- Project: read from the installed package's metadata (no __version__, which would be public
# API; D-62's names).
_META = metadata("openftgs")
project = _META["Name"]
author = _META["Author"]
release = _META["Version"]
version = ".".join(release.split(".")[:2])
copyright = f"2026, {author}"  # as NOTICE states it

# -- Repository links: the tag Read the Docs is building, and main everywhere else.
_REPOSITORY = "https://github.com/samaust/openftgs"
_REF = (
    os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag"
    else "main"
)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # after napoleon, so it types the fields napoleon writes
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinxarg.ext",
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_templates",
    "agents",
    "clean-room",
    "modules",
    "paper",
    "research",
    "spec",
    "GLOSSARY.md",
]

# -- MyST
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]

# -- autodoc and autosummary: the API pages follow each public module's __all__ (§2.1); no
# undoc-members, since dataclass fields are documented in their class's docstring.
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_member_order = "bysource"
# A method without a docstring of its own isn't documented with its base class's, so the pages
# never show PyTorch's text (and its cross-references) for an override such as nn.Module.forward.
autodoc_inherit_docstrings = False

# -- napoleon: NumPy docstrings only. Its type preprocessing turns each type line into a
# cross-reference, so the type line NumPy style requires under Returns resolves under nitpicky;
# the aliases map the short spellings docstrings use to their documented targets.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Tensor": "torch.Tensor",
    "ndarray": "numpy.ndarray",
    "np.ndarray": "numpy.ndarray",
    "Path": "pathlib.Path",
    "array_like": ":term:`array_like <numpy:array_like>`",
}

# -- sphinx-autodoc-typehints: the annotations' types go into the descriptions; autodoc's own
# autodoc_typehints is left at its default, so only this extension inserts types.
typehints_document_rtype = True
always_use_bars_union = True

# -- Cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}
intersphinx_timeout = 30

nitpicky = True
# Each entry with its reason; keep the list small.
nitpick_ignore: list[tuple[str, str]] = []

extlinks = {
    "repo": (f"{_REPOSITORY}/blob/{_REF}/%s", "%s"),
    "issue": (f"{_REPOSITORY}/issues/%s", "#%s"),
}

# -- linkcheck: GitHub builds its line and heading anchors with JavaScript, and PyTorch's stable
# pages are stubs that redirect, by script, to the current release's, so their anchors aren't on
# the page linkcheck reads; the pages themselves are still checked.
linkcheck_anchors_ignore_for_url = [
    r"https://github\.com/.*",
    r"https://docs\.pytorch\.org/docs/stable/.*",
]

# -- HTML: Furo, its view and edit buttons on GitHub.
html_theme = "furo"
html_title = f"{project} {release}"
html_theme_options = {
    "source_repository": f"{_REPOSITORY}/",
    "source_branch": _REF,
    "source_directory": "docs/",
}


def _no_source_buttons_on_stubs(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: object,
) -> None:
    """Drop the view and edit buttons from the generated API pages.

    Their sources, the stubs autosummary writes under api/generated/, aren't in the repository.
    """
    if pagename.startswith("api/generated/"):
        context["theme_top_of_page_buttons"] = []


def setup(app: Sphinx) -> None:
    """Connect the page-context handler."""
    app.connect("html-page-context", _no_source_buttons_on_stubs)

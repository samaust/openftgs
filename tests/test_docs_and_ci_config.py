# SPDX-License-Identifier: Apache-2.0
"""The documentation site's builds and CI's Python matrix, as configuration (PRD #355; #365).

Added tests, not §3 rows: the Read the Docs configuration and the docs workflow build the site
on Python 3.11, the floor (Q-1), with CPU torch from the PyTorch CPU index (Q-3), then the package
with the `docs` dependency group (Q-241), warnings as errors; CI's test job runs on 3.11 beside
3.14 (Q-1's and Q-29's detecting test); the project URLs name the site (Q-244). The workflows run
only on GitHub, so these tests read their files.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
SITE = "https://openftgs.readthedocs.io"
REPOSITORY = "https://github.com/samaust/openftgs"
# Without `?version=`, the badge reports the project's default version (source log 315).
BADGE = "https://app.readthedocs.org/projects/openftgs/badge/"


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _installs_cpu_torch_then_the_docs_group(commands: list[str]) -> None:
    """Pip is upgraded (`--group` needs pip 25.1), CPU torch comes from the CPU index, and only
    then the package with the docs group, so pip never fetches a CUDA torch. torchvision isn't
    installed: no public module imports it since revision 549o (#432), and the site builds
    without it."""
    torch = next(i for i, c in enumerate(commands) if CPU_INDEX in c)
    assert "torch" in commands[torch].split()
    assert not any("torchvision" in c for c in commands)
    docs = next(i for i, c in enumerate(commands) if "--group docs" in c)
    assert commands[docs].split()[-3:] == [".", "--group", "docs"]
    assert any("--upgrade pip" in c for c in commands[:torch])
    assert torch < docs


def test_read_the_docs_builds_on_3_11_with_cpu_torch_and_warnings_as_errors() -> None:
    """Version 2, Ubuntu 24.04, Python 3.11; the install step replaced by CPU torch, then the
    package with the docs group; Sphinx on docs/conf.py with fail_on_warning (-W --keep-going)."""
    config = _yaml(ROOT / ".readthedocs.yaml")
    assert config["version"] == 2
    assert config["build"]["os"] == "ubuntu-24.04"
    assert config["build"]["tools"] == {"python": "3.11"}
    _installs_cpu_torch_then_the_docs_group(config["build"]["jobs"]["install"])
    assert "python" not in config  # the install step above replaces python.install
    assert config["sphinx"] == {"configuration": "docs/conf.py", "fail_on_warning": True}


def _run(step: dict[str, Any]) -> list[str]:
    return [line.strip() for line in str(step.get("run", "")).splitlines() if line.strip()]


def test_docs_workflow_builds_the_site_as_read_the_docs_does_and_uploads_it() -> None:
    """On pull requests and pushes to main: Python 3.11 with a pip cache, the same install as Read
    the Docs, `sphinx-build -W --keep-going`, and the HTML uploaded; linkcheck stays out of CI."""
    workflow = _yaml(ROOT / ".github" / "workflows" / "docs.yml")
    assert workflow[True] == {"push": {"branches": ["main"]}, "pull_request": None}  # `on`
    assert workflow["permissions"] == {"contents": "read"}
    (job,) = workflow["jobs"].values()
    steps = job["steps"]
    python = next(s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python"))
    assert python["with"]["python-version"] == "3.11"
    assert python["with"]["cache"] == "pip"
    commands = [command for step in steps for command in _run(step)]
    rtd = _yaml(ROOT / ".readthedocs.yaml")["build"]["jobs"]["install"]
    assert [c for c in commands if "pip install" in c][: len(rtd)] == rtd
    _installs_cpu_torch_then_the_docs_group(commands)
    build = next(c for c in commands if "sphinx" in c and "-b html" in c)
    assert {"-W", "--keep-going"} <= set(build.split())
    assert build.endswith("docs docs/_build/html")
    assert not any("linkcheck" in c for c in commands)
    upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert upload["with"]["path"].rstrip("/") == "docs/_build/html"
    assert steps.index(upload) > next(i for i, s in enumerate(steps) if build in _run(s))
    # CI's lint job has no Sphinx, so its mypy can't check docs/conf.py; this job does.
    assert "mypy docs/conf.py" in commands


def _ci_jobs() -> list[dict[str, str]]:
    """The combinations of CI's `checks` matrix, as GitHub expands `include` (source log 316):
    an entry that would overwrite an original value becomes a combination of its own."""
    matrix = _yaml(CI)["jobs"]["checks"]["strategy"]["matrix"]
    original = [{"job": j, "python": p} for j in matrix["job"] for p in matrix["python"]]
    combinations = [dict(c) for c in original]
    for entry in matrix.get("include", []):
        extended = False
        for base, combination in zip(original, combinations, strict=True):
            if all(base.get(k, v) == v for k, v in entry.items()):
                combination.update(entry)
                extended = True
        if not extended:
            combinations.append(dict(entry))
    return combinations


def test_ci_runs_the_cpu_suite_on_3_11_beside_3_14() -> None:
    """Q-1 and Q-29's detecting test, the M18 CI matrix: the test job installs and runs on 3.11,
    the floor, as well as on 3.14; the other jobs stay on 3.14 alone."""
    combinations = _ci_jobs()
    assert sorted(c["python"] for c in combinations if c["job"] == "test") == ["3.11", "3.14"]
    for job in ("lint", "licenses", "wheel"):
        assert [c["python"] for c in combinations if c["job"] == job] == ["3.14"]
    floor = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert floor["requires-python"] == ">=" + min(c["python"] for c in combinations)
    checks = _yaml(CI)["jobs"]["checks"]
    python = next(s for s in checks["steps"] if "setup-python" in str(s.get("uses", "")))
    assert python["with"]["python-version"] == "${{ matrix.python }}"
    assert "matrix.python" in checks["name"]
    # upload-artifact v4's artifacts are immutable (source log 319): each test job keeps its own
    # coverage.
    upload = next(s for s in checks["steps"] if "upload-artifact" in str(s.get("uses", "")))
    assert "matrix.python" in upload["with"]["name"]


def test_project_urls_and_readme_point_to_the_site() -> None:
    """Q-244: the project URLs name the documentation site; the README carries its badge and a
    link to it."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["urls"]["Documentation"] == SITE
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    intro = readme.split("\n## ", 1)[0].splitlines()  # the title, the badges, the summary
    assert f"[![Documentation]({BADGE})]({SITE})" in intro[2]
    assert any(f"]({SITE})" in line for line in intro[3:])


# Q-244's classifiers, spelled as PyPI's classifier list spells them (source log T370.2).
RELEASE_CLASSIFIERS = {
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Environment :: GPU :: NVIDIA CUDA",
    "Typing :: Typed",
}


def test_release_metadata_is_version_0_1_0_with_q244_classifiers_and_urls() -> None:
    """Q-244 (#370): version 0.1.0; keywords; exactly Q-244's classifiers, with no license
    classifier, since PEP 639's `license` expression is set; and URLs for Homepage, Source,
    Documentation, Issues and Changelog, each on the project's site or repository."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.1.0"
    assert project["license"] == "Apache-2.0"
    assert project["keywords"] and all(isinstance(k, str) and k for k in project["keywords"])
    assert set(project["classifiers"]) == RELEASE_CLASSIFIERS
    assert len(project["classifiers"]) == len(RELEASE_CLASSIFIERS)
    assert not any(c.startswith("License ::") for c in project["classifiers"])
    urls = project["urls"]
    assert set(urls) == {"Homepage", "Source", "Documentation", "Issues", "Changelog"}
    assert urls["Homepage"] == urls["Source"] == REPOSITORY
    assert urls["Issues"] == f"{REPOSITORY}/issues"
    assert urls["Changelog"] == f"{REPOSITORY}/blob/main/docs/changelog.md"


def _markdown_targets(text: str) -> list[str]:
    """The targets of a Markdown document's inline links and images, `[text](target)`."""
    return re.findall(r"\]\(([^)\s]+)\)", text)


def test_readme_states_the_release_and_links_the_records_on_github() -> None:
    """#370 (D-62, §0, Q-244): the README is also PyPI's project description, so every link and
    image opens on github.com or another absolute URL, never a path relative to the repository;
    it installs with `pip install openftgs`, its status line is no longer M14's, and its
    clean-room statement links the spec, the source log, the questions (whose raw file is linked
    too, since GitHub doesn't render a Markdown file that large), the milestones and the release
    checklist, and says that the project page's renderer was read for a documentation-only
    survey (#352)."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# openftgs\n")
    targets = _markdown_targets(readme)
    assert targets
    relative = [t for t in targets if not t.startswith(("https://", "#"))]
    assert relative == []
    assert "pip install openftgs\n" in readme
    assert "M14" not in readme
    blob = f"{REPOSITORY}/blob/main/"
    for record in (
        "docs/spec/openftgs-spec.md",
        "docs/clean-room/source-log.md",
        "docs/clean-room/questions.md",
        "docs/clean-room/milestones.md",
        "docs/clean-room/release-checklist.md",
    ):
        assert blob + record in targets, record
    assert f"{REPOSITORY}/raw/main/docs/clean-room/questions.md" in targets
    clean_room = readme.split("\n## Clean-room process\n", 1)[1].split("\n## ", 1)[0]
    assert f"{REPOSITORY}/issues/352" in _markdown_targets(clean_room)
    assert "renderer" in clean_room

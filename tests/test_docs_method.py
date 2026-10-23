# SPDX-License-Identifier: Apache-2.0
"""The documentation site's method page against the spec's equations (PRD #355; #358).

An added test, not a §3 row: "each equation block equals the spec's LaTeX for its E-n" (source:
§1.2). The source of truth is the spec itself: §1.2's ``latex`` blocks, grouped under their
``**E<n> · <name>** <tag>`` lines, against the ``{math}`` blocks of ``docs/method.md``, grouped
under its ``### E<n> · <name>`` headings, each followed by a ``Provenance: <tag>`` line. The page
transcribes the equations verbatim, so every block, name and provenance tag is compared as text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "spec" / "openftgs-spec.md"
METHOD = ROOT / "docs" / "method.md"

_SPEC_ITEM = re.compile(r"^\*\*E(\d+) · (.+?)\*\* (.+)$")
_PAGE_ITEM = re.compile(r"^#{2,4} E(\d+) · (.+)$")
_PAGE_TAG = re.compile(r"^Provenance: (.+)$")


class _Item:
    def __init__(self, name: str, tag: str | None) -> None:
        self.name = name
        self.tag = tag
        self.blocks: list[str] = []


def _section(text: str, start: str, end: str) -> list[str]:
    lines = text.splitlines()
    first = lines.index(start)
    last = lines.index(end, first)
    return lines[first + 1 : last]


def _spec_items() -> dict[int, _Item]:
    lines = _section(
        SPEC.read_text(encoding="utf-8"),
        "### 1.2 Equations",
        "### 1.3 Initialization, relocation and optimization",
    )
    return _items(lines, _SPEC_ITEM, "```latex", tag_inline=True)


def _page_items() -> dict[int, _Item]:
    return _items(
        METHOD.read_text(encoding="utf-8").splitlines(), _PAGE_ITEM, "```{math}", tag_inline=False
    )


def _items(
    lines: list[str], header: re.Pattern[str], fence: str, *, tag_inline: bool
) -> dict[int, _Item]:
    items: dict[int, _Item] = {}
    current: _Item | None = None
    block: list[str] | None = None
    for line in lines:
        if block is not None:
            if line == "```":
                assert current is not None, "an equation block before the first E-n"
                current.blocks.append("\n".join(block))
                block = None
            else:
                block.append(line)
            continue
        if m := header.match(line):
            n = int(m[1])
            assert n not in items, f"E{n} appears twice"
            current = items[n] = _Item(m[2], m[3] if tag_inline else None)
        elif line == fence:
            block = []
        elif not tag_inline and current is not None and (t := _PAGE_TAG.match(line)):
            assert current.tag is None, f"E-n {current.name!r} has two provenance lines"
            current.tag = t[1]
    assert block is None, "an unclosed equation block"
    return items


def test_spec_parse_finds_e1_to_e17() -> None:
    """The spec side of the comparison: §1.2 holds E1 to E17, each with at least one block."""
    items = _spec_items()
    assert sorted(items) == list(range(1, 18))
    assert all(item.blocks for item in items.values())


def test_method_page_lists_e1_to_e17() -> None:
    """The method page has a heading for every E-n of §1.2, and no other."""
    assert sorted(_page_items()) == sorted(_spec_items())


def test_method_page_equations_equal_the_spec() -> None:
    """Each E-n's equation blocks on the method page equal §1.2's LaTeX blocks, in order (#358)."""
    spec, page = _spec_items(), _page_items()
    for n, item in spec.items():
        assert page[n].blocks == item.blocks, f"E{n}'s equation blocks differ from the spec's"


def test_method_page_names_and_tags_equal_the_spec() -> None:
    """Each E-n keeps the spec's name and its provenance tag, verbatim (#358)."""
    spec, page = _spec_items(), _page_items()
    for n, item in spec.items():
        assert page[n].name == item.name, f"E{n}'s name differs from the spec's"
        assert page[n].tag == item.tag, f"E{n}'s provenance tag differs from the spec's"

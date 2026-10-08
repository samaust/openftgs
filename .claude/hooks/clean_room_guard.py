#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Clean-room tripwire for Claude Code (PreToolUse hook).

Blocks tool calls that would reach another FreeTimeGS implementation or
EasyVolcap: web searches that name them, fetches of URLs that name them
(except the paper and the official project page), downloads and clones that
name them, and reads of local paths that name them.

It is a tripwire, not a guarantee: a forbidden source that never names
FreeTimeGS or EasyVolcap passes. CLAUDE.md's rules still apply to everything.

Protocol: Claude Code sends the tool call as JSON on stdin. Exit code 2 blocks
the call and shows stderr to Claude; exit code 0 lets it through. Any internal
error lets the call through, so a bug here never stops normal work.
"""

import datetime
import json
import os
import re
import sys
import urllib.parse

# "freetime" written as one word (FreeTimeGS, freetime-4d, FreeTimeGaussians),
# "free time" followed by gs / gaussian / splat, and EasyVolcap.
NAME = re.compile(
    r"freetime"
    r"|free[\s_.\-]+time[\s_.\-]*(?:gs|gaussian|splat)"
    r"|easy[\s_.\-]*volcap",
    re.IGNORECASE,
)

# The only URLs that may name FreeTimeGS: the project page itself (not its
# code links or demos) and the paper and supplementary on CVF open access.
ALLOWED_URL = re.compile(
    r"^https?://zju3dv\.github\.io/freetimegs/?(?:index\.html)?/?(?:[?#].*)?$"
    r"|^https?://openaccess\.thecvf\.com/content/CVPR2025/"
    r"(?:papers|html|supplemental)/Wang_FreeTimeGS_[^/?#]*$",
    re.IGNORECASE,
)

URL = re.compile(r"""https?://[^\s'"<>`)]+""", re.IGNORECASE)

# The package's own class name (spec §2.1), when it appears as a Python
# identifier rather than inside a path or URL.
OWN_CLASS = re.compile(r"(?<![/\\\w-])FreeTimeGaussians(?![/\\\w-])")

# Shell commands that download, clone or search for things.
FETCHING_COMMAND = re.compile(
    r"\b(?:git\s+(?:clone|fetch|pull|submodule|remote|ls-remote)"
    r"|curl|wget|aria2c|svn|hg\s+clone"
    r"|pip3?\s+(?:install|download)|python3?\s+-m\s+pip\s+(?:install|download)"
    r"|uv\s+(?:pip|add|tool)|uvx|pipx|poetry\s+add|pdm\s+add"
    r"|conda\s+(?:install|create)|mamba\s+(?:install|create)"
    r"|gh\s+(?:search|repo|api|gist|release|browse)"
    r"|hf\s+(?:download|repo|search)|huggingface-cli"
    r"|npx|npm\s+(?:install|i)\b)",
    re.IGNORECASE,
)

# The project's own repository: gh may read and write its issues, and nothing else's.
PROJECT_REPO = "samaust/openftgs"
GH_INVOCATION = re.compile(r"(?:^|[\s;&|(`])gh\s", re.IGNORECASE)
GH_REPO_FLAG = re.compile(r"""(?:^|\s)(?:-R|--repo)[=\s]+["']?([^\s"']+)""")
GH_REPO_ENV = re.compile(r"""(?:^|[\s;&|(])GH_REPO=["']?([^\s"']+)""")
GH_API_REPO = re.compile(r"""(?:^|[\s"'/])repos/([\w.-]+/[\w.-]+)""")
GH_REPO_POSITIONAL = re.compile(
    r"""\bgh\s+repo\s+(?:clone|fork|view|sync|edit|archive|delete|set-default)\s+["']?([^\s"'-][^\s"']*)""",
    re.IGNORECASE,
)
GH_GIST = re.compile(r"\bgh\s+gist\b", re.IGNORECASE)
GH_SEARCH = re.compile(r"\bgh\s+search\b", re.IGNORECASE)

READ_TOOLS = {"Read", "Grep", "Glob", "LS", "NotebookRead"}
PATH_FIELDS = ("file_path", "path", "notebook_path")

MESSAGE = (
    "Blocked by the openftgs clean-room guard: {why}\n"
    "Other FreeTimeGS implementations (including the authors' code) and EasyVolcap "
    "are forbidden sources (CLAUDE.md; spec section 0). Do not retry, rephrase or "
    "work around this. Use the paper (arXiv 2506.05348 v2), the spec in docs/spec/ "
    "and the other allowed sources. If you need something they don't give you, stop "
    "and ask the owner, and mention this blocked call in your next message."
)


def strip_allowed_urls(text):
    """Remove allowed URLs so a mention inside them doesn't count."""
    return URL.sub(lambda m: "" if ALLOWED_URL.match(m.group(0).rstrip(".,;:")) else m.group(0), text)


def names_forbidden(text):
    if not text:
        return False
    decoded = urllib.parse.unquote(str(text))
    return bool(NAME.search(decoded))


def normalize_repo(value):
    """'https://github.com/Owner/Repo.git' or 'Owner/Repo' -> 'owner/repo'."""
    value = value.strip().strip("\"'").lower()
    value = re.sub(r"^(?:https?://)?(?:www\.)?github\.com[/:]", "", value)
    value = re.sub(r"\.git$", "", value).strip("/")
    return value


def gh_repositories(command):
    """Every repository a gh invocation in `command` is pointed at."""
    repos = set()
    if "gh api" in command or re.search(r"\bgh\s+api\b", command):
        repos.update(GH_API_REPO.findall(command))
    repos.update(GH_REPO_FLAG.findall(command))
    repos.update(GH_REPO_ENV.findall(command))
    repos.update(GH_REPO_POSITIONAL.findall(command))
    return {normalize_repo(r) for r in repos}


def gh_scoped_to_project(command):
    """True when every gh invocation targets the project repository: explicitly,
    or by default through the clone's git remote. None when there is no gh."""
    if not GH_INVOCATION.search(command):
        return None
    if GH_GIST.search(command):
        return False  # gists belong to anyone
    repos = gh_repositories(command)
    if GH_SEARCH.search(command) and not repos:
        return False  # a GitHub-wide search
    return repos <= {PROJECT_REPO}


def check(tool, tool_input):
    """Return a reason string if the call must be blocked, else None."""
    if not isinstance(tool_input, dict):
        return None

    if tool == "WebSearch":
        if names_forbidden(tool_input.get("query", "")):
            return "web searches for FreeTimeGS or EasyVolcap are not allowed."
        return None

    if tool == "WebFetch":
        url = str(tool_input.get("url", ""))
        if names_forbidden(url) and not ALLOWED_URL.match(url):
            return "this URL names FreeTimeGS or EasyVolcap and is not the paper or the project page."
        return None

    if tool == "Bash":
        command = str(tool_input.get("command", ""))
        scoped = gh_scoped_to_project(command)
        if scoped is False:
            return f"gh may only target {PROJECT_REPO}; this command points it at another repository or at a GitHub-wide search."
        rest = OWN_CLASS.sub("", strip_allowed_urls(command))
        if not names_forbidden(rest):
            return None
        for m in FETCHING_COMMAND.finditer(rest):
            if scoped and m.group(0).lower().startswith("gh"):
                continue  # gh pointed at the project repository: its issues may name FreeTimeGS
            return "this command downloads, clones or searches for something that names FreeTimeGS or EasyVolcap."
        for token in re.split(r"""[\s'"=;&|()<>`]+""", rest):
            if ("/" in token or "\\" in token) and names_forbidden(token):
                return "this command touches a path or URL that names FreeTimeGS or EasyVolcap."
        return None

    if tool in READ_TOOLS:
        fields = list(PATH_FIELDS) + (["pattern"] if tool == "Glob" else [])
        for field in fields:
            if names_forbidden(tool_input.get(field, "")):
                return "this path names FreeTimeGS or EasyVolcap."
        return None

    if tool.startswith("mcp__"):
        text = strip_allowed_urls(json.dumps(tool_input, ensure_ascii=False))
        if names_forbidden(text):
            return "this MCP tool call names FreeTimeGS or EasyVolcap."
        return None

    return None


def log_block(event, why):
    project = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    record = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "session": event.get("session_id"),
        "tool": event.get("tool_name"),
        "input": event.get("tool_input"),
        "reason": why,
    }
    try:
        with open(os.path.join(project, ".claude", "clean-room-guard.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False)[:4000] + "\n")
    except OSError:
        pass


def main():
    try:
        event = json.load(sys.stdin)
        why = check(str(event.get("tool_name", "")), event.get("tool_input", {}))
    except Exception:  # never block normal work because of a bug here
        return 0
    if why is None:
        return 0
    log_block(event, why)
    print(MESSAGE.format(why=why), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

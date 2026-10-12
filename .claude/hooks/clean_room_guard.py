#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Clean-room tripwire for Claude Code (PreToolUse hook).

Blocks tool calls that would reach another FreeTimeGS implementation or
EasyVolcap: web searches that name them, fetches of URLs that name them
(except the official project page), downloads and clones that
name them, reads and writes of local paths that name them, and local
searches for them outside the repository.

Searching the repository's own files for these names is allowed (owner rule,
2026-10-12): grep, rg, ag, ack, find, fd and git grep run with every search
root inside the repository, and the Grep and Glob tools pointed inside it. A
search tool that does not skip .gitignore'd files may not walk into the
virtual environments (.venv, venv), which hold installed third-party code, not
the repository's.

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
import shlex
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

# The only URL that may name FreeTimeGS: the project page itself (not its
# code links or demos). The paper is read on arXiv (2506.05348 v2), whose
# URLs don't name it; the spec follows that version, not the CVF one.
ALLOWED_URL = re.compile(
    r"^https?://zju3dv\.github\.io/freetimegs/?(?:index\.html)?/?(?:[?#].*)?$",
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
# What follows gh in command position, after any global flags (-R x, --repo=x).
# gist and search count only there, so prose that mentions them doesn't; a
# search also needs a subcommand after it (`gh search repos ...`).
GH_LEADING_FLAGS = re.compile(r"""^(?:-{1,2}[\w-]+(?:[=\s]+[^-\s]\S*)?\s+)*""")
GH_GIST_TAIL = re.compile(r"gist\b", re.IGNORECASE)
GH_SEARCH_TAIL = re.compile(r"""search\s+[^\s`'")]""", re.IGNORECASE)
# Separators between simple shell commands; a -R/--repo flag belongs to the gh
# invocation only when it sits in the same simple command (not `ls -R`, `grep -R`).
SHELL_SEPARATORS = re.compile(r"[;&|\n]+")
# gh in command position: first in the simple command (after VAR=value
# assignments or a wrapper such as env, sudo or xargs), or right after $( ( or `.
# Prose that merely mentions gh, as in a commit message, doesn't match.
GH_AT_COMMAND = re.compile(
    r"(?:^|\$\(|[(`])\s*"
    r"(?:(?:[A-Za-z_]\w*=\S*|(?:env|command|exec|time|nohup|sudo|xargs)(?:\s+-\S+)*)\s+)*gh\s",
    re.IGNORECASE,
)

READ_TOOLS = {"Read", "Grep", "Glob", "LS", "NotebookRead"}
# A file written under a forbidden name would then be blocked from every read.
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
PATH_FIELDS = ("file_path", "path", "notebook_path")

# The repository, and the virtual environments inside it (.gitignore'd installed
# third-party code, not the repository's own files).
PROJECT_ROOT = os.path.realpath(
    os.environ.get("CLAUDE_PROJECT_DIR")
    or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ENVIRONMENTS = tuple(os.path.join(PROJECT_ROOT, name) for name in (".venv", "venv"))


class Searcher:
    """How a local search program reads its command line: whether it skips
    .gitignore'd files by default, the options that turn that off, the short and
    long options that take a value, the options whose value is a pattern or a file
    it reads, the options whose value is a search root, whether it recurses
    without being asked, the options that make it run other commands, and those
    after which it takes no pattern."""

    def __init__(self, *, skips_ignored, unignore=(), short_values="", long_values=(),
                 pattern_options=(), file_options=(), root_options=(), recursive=True,
                 recursive_options=(), exec_options=(), no_pattern_options=()):
        self.skips_ignored = skips_ignored
        self.unignore = set(unignore)
        self.short_values = short_values
        self.long_values = set(long_values)
        self.pattern_options = set(pattern_options)
        self.file_options = set(file_options)
        self.root_options = set(root_options)
        self.recursive = recursive
        self.recursive_options = set(recursive_options)
        self.exec_options = set(exec_options)
        self.no_pattern_options = set(no_pattern_options)


GREP = Searcher(
    skips_ignored=False,
    short_values="efmABCdD",
    long_values=("--regexp", "--file", "--max-count", "--after-context", "--before-context",
                 "--context", "--directories", "--devices", "--include", "--exclude",
                 "--exclude-dir", "--exclude-from", "--label", "--binary-files", "--group-separator"),
    pattern_options=("-e", "--regexp"),
    file_options=("-f", "--file", "--exclude-from"),
    recursive=False,
    recursive_options=("-r", "-R", "--recursive", "--dereference-recursive"),
)
FIND_FAMILY = {"fd": Searcher(
    skips_ignored=True,
    unignore=("-I", "-u", "--no-ignore", "--no-ignore-vcs", "--no-ignore-parent", "--unrestricted"),
    short_values="etEdjSoc",
    long_values=("--extension", "--type", "--exclude", "--max-depth", "--min-depth", "--exact-depth",
                 "--threads", "--size", "--changed-within", "--changed-before", "--owner", "--color",
                 "--search-path", "--base-directory", "--ignore-file", "--max-results"),
    file_options=("--ignore-file",),
    root_options=("--search-path", "--base-directory"),
    exec_options=("-x", "-X", "--exec", "--exec-batch"),
)}
FIND_FAMILY["fdfind"] = FIND_FAMILY["fd"]
SEARCHERS = {
    "grep": GREP,
    "egrep": GREP,
    "fgrep": GREP,
    "rg": Searcher(
        skips_ignored=True,
        unignore=("-u", "--no-ignore", "--no-ignore-vcs", "--no-ignore-dot", "--no-ignore-exclude",
                  "--no-ignore-files", "--no-ignore-global", "--no-ignore-parent", "--unrestricted"),
        short_values="efgtTmABCjdMrE",
        long_values=("--regexp", "--file", "--glob", "--iglob", "--type", "--type-not", "--type-add",
                     "--type-clear", "--max-count", "--after-context", "--before-context", "--context",
                     "--threads", "--max-depth", "--max-columns", "--replace", "--encoding", "--sort",
                     "--sortr", "--color", "--colors", "--pre", "--pre-glob", "--path-separator",
                     "--max-filesize", "--ignore-file", "--engine", "--context-separator",
                     "--field-match-separator", "--field-context-separator", "--dfa-size-limit",
                     "--regex-size-limit"),
        pattern_options=("-e", "--regexp"),
        file_options=("-f", "--file", "--ignore-file"),
        exec_options=("--pre",),
        no_pattern_options=("--files", "--type-list"),
    ),
    "ag": Searcher(
        skips_ignored=True,
        unignore=("-u", "-U", "-a", "--unrestricted", "--skip-vcs-ignores", "--all-types"),
        short_values="ABCGgmp",
        long_values=("--after", "--before", "--context", "--file-search-regex", "--max-count",
                     "--depth", "--ignore", "--ignore-dir", "--path-to-ignore", "--pager"),
        file_options=("-p", "--path-to-ignore"),
        exec_options=("--pager",),
        no_pattern_options=("-g", "--filename-pattern"),
    ),
    "ack": Searcher(
        skips_ignored=False,
        short_values="ABCm",
        long_values=("--after-context", "--before-context", "--context", "--max-count", "--type",
                     "--ignore-dir", "--ignore-file", "--match", "--files-from", "--pager"),
        pattern_options=("--match",),
        file_options=("--files-from",),
        exec_options=("--pager",),
    ),
    **FIND_FAMILY,
}
SYSTEM_SEARCHERS = {"locate", "plocate", "mlocate", "slocate"}
FIND_EXEC = {"-exec", "-execdir", "-ok", "-okdir"}
WRAPPERS = {"command", "nohup", "time", "nice", "env", "stdbuf", "timeout"}

MESSAGE = (
    "Blocked by the openftgs clean-room guard: {why}\n"
    "Other FreeTimeGS implementations (including the authors' code) and EasyVolcap "
    "are forbidden sources (CLAUDE.md; spec section 0). Do not retry, rephrase or "
    "work around this. Use the paper (arXiv 2506.05348 v2), the spec in docs/spec/ "
    "and the other allowed sources. If you need something they don't give you, stop "
    "and ask the owner, and mention this blocked call in your next message."
)
OUTSIDE = (
    "this searches for FreeTimeGS or EasyVolcap outside the repository's own files "
    "(only the repository may be searched for these names, and not its virtual environments)."
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


QUOTED = re.compile(r"""'[^']*'|"(?:[^"\\]|\\.)*\"""", re.DOTALL)
SUBSTITUTION = re.compile(r"""\$\((?:[^()]|\([^()]*\))*\)|`[^`]*`""")
SHELL_CODE_BEFORE = re.compile(r"""(?:^|[\s;&|(])(?:(?:ba|z|da|k)?sh\s+(?:-\S+\s+)*-c|eval)\s*$""")


def neutralize_prose(command):
    """Blank the quoted strings that contain whitespace (commit messages, issue
    bodies), so prose that mentions gh doesn't count as a gh call. Kept: quoted
    strings without whitespace (a repository name), the shell code given to
    `sh -c` or `eval`, and the $( ) and backtick substitutions inside double
    quotes, which the shell runs."""
    def replace(m):
        s = m.group(0)
        if not re.search(r"\s", s):
            return s
        if SHELL_CODE_BEFORE.search(command[: m.start()]):
            return " " + s[1:-1] + " "
        if s[0] == '"':
            subs = SUBSTITUTION.findall(s)
            if subs:
                return " " + " ; ".join(subs) + " "
        return '""'
    return QUOTED.sub(replace, command)


def gh_repositories(command):
    """Every repository a gh invocation in `command` is pointed at."""
    repos = set()
    if "gh api" in command or re.search(r"\bgh\s+api\b", command):
        repos.update(GH_API_REPO.findall(command))
    for simple_command in SHELL_SEPARATORS.split(command):
        if GH_AT_COMMAND.search(simple_command):
            repos.update(GH_REPO_FLAG.findall(simple_command))
    repos.update(GH_REPO_ENV.findall(command))
    repos.update(GH_REPO_POSITIONAL.findall(command))
    return {normalize_repo(r) for r in repos}


def gh_command_tails(command):
    """The text after each gh in command position, global flags skipped."""
    for simple_command in SHELL_SEPARATORS.split(command):
        for m in GH_AT_COMMAND.finditer(simple_command):
            tail = simple_command[m.end():]
            yield tail[GH_LEADING_FLAGS.match(tail).end():]


def gh_scoped_to_project(command):
    """True when every gh invocation targets the project repository: explicitly,
    or by default through the clone's git remote. None when there is no gh."""
    command = neutralize_prose(command)
    if not GH_INVOCATION.search(command):
        return None
    tails = list(gh_command_tails(command))
    if any(GH_GIST_TAIL.match(t) for t in tails):
        return False  # gists belong to anyone
    repos = gh_repositories(command)
    if any(GH_SEARCH_TAIL.match(t) for t in tails) and not repos:
        return False  # a GitHub-wide search
    return repos <= {PROJECT_REPO}


# Local searches: what the repository's own files are.


def resolve(path, cwd):
    """The real path a shell argument names, relative to cwd; for a glob, the
    directory before its first wildcard. None when it can't be known here (no
    working directory, or a variable or substitution in the path)."""
    if cwd is None or "$" in path or "`" in path:
        return None
    path = os.path.expanduser(path)
    wildcard = re.search(r"[*?\[{]", path)
    if wildcard:
        path = os.path.dirname(path[: wildcard.start()]) or "."
    return os.path.realpath(os.path.join(cwd, path))


def within(path, directory):
    return path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)


def in_repository(path):
    """Inside the repository and outside its virtual environments."""
    return (
        path is not None
        and within(path, PROJECT_ROOT)
        and not any(within(path, env) for env in ENVIRONMENTS)
    )


def holds_environment(path):
    """A directory a recursive search would walk into a virtual environment from."""
    return any(within(env, path) for env in ENVIRONMENTS)


def split_simple_commands(command):
    """[(text, reads_other_input)] for each simple command of a shell command,
    split at the unquoted operators ; & && | || ( ) and newlines; reads_other_input
    is true after a pipe or with an input redirection. None for what this guard
    doesn't split: command substitutions, here-documents, unbalanced quotes."""
    segments, current = [], []
    quote = None
    piped = redirected = False
    i, n = 0, len(command)

    def flush(next_piped):
        nonlocal current, piped, redirected
        text = "".join(current).strip()
        if text:
            segments.append((text, piped or redirected))
        current, piped, redirected = [], next_piped, False

    while i < n:
        c = command[i]
        if quote == "'":
            current.append(c)
            quote = None if c == "'" else quote
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            current.append(command[i : i + 2])
            i += 2
            continue
        if c == "`" or command.startswith("$(", i):
            return None
        if quote == '"':
            current.append(c)
            quote = None if c == '"' else quote
            i += 1
            continue
        if c in "'\"":
            quote = c
        elif command.startswith("<<", i):
            return None
        elif c == "<":
            redirected = True
        elif c == "|":
            double = command.startswith(("||", "|&"), i)
            flush(not command.startswith("||", i))
            i += 2 if double else 1
            continue
        elif c == "&" and (command.startswith("&>", i) or (i > 0 and command[i - 1] in "<>")):
            pass  # part of a redirection (&> file, 2>&1)
        elif c in ";&()\n":
            flush(False)
            i += 2 if command.startswith(("&&", ";;"), i) else 1
            continue
        current.append(c)
        i += 1
    if quote:
        return None
    flush(False)
    return segments


def command_words(words):
    """(words, via_xargs): the words of a simple command from its program on, with
    assignments and wrappers (env, nice, timeout, sudo …) skipped; via_xargs is
    true when xargs runs the program on arguments read from its input, which this
    guard can't see."""
    i, via_xargs = 0, False
    while i < len(words):
        word = words[i]
        if re.match(r"^[A-Za-z_]\w*=", word):
            i += 1
        elif word in WRAPPERS or word in ("xargs", "sudo", "doas"):
            via_xargs = via_xargs or word == "xargs"
            i += 1
            if word == "timeout":
                while i < len(words) and words[i].startswith("-"):
                    i += 1
                i += 1  # the duration
            while i < len(words) and (words[i].startswith("-") or "=" in words[i]):
                i += 1
        else:
            return words[i:], via_xargs
    return [], via_xargs


def parse_search(spec, args):
    """(patterns, roots, files, recursive, skips_ignored) of a search command's
    arguments, or None when it would run another command."""
    patterns, files, positional, option_roots = [], [], [], []
    pattern_given = False
    recursive, skips_ignored = spec.recursive, spec.skips_ignored
    takes_pattern = True
    i, options_done = 0, False

    def option(name, value):
        nonlocal pattern_given
        if name in spec.pattern_options:
            patterns.append(value)
            pattern_given = True
        elif name in spec.file_options:
            files.append(value)
        elif name in spec.root_options:
            option_roots.append(value)

    while i < len(args):
        arg = args[i]
        i += 1
        if options_done or arg == "-" or not arg.startswith("-"):
            positional.append(arg)
            continue
        if arg == "--":
            options_done = True
            continue
        if arg.startswith("--"):
            name, eq, value = arg.partition("=")
            if name in spec.exec_options:
                return None
            if name in spec.unignore:
                skips_ignored = False
            if name in spec.recursive_options:
                recursive = True
            if name in spec.no_pattern_options:
                takes_pattern = False
            if name in spec.long_values and not eq:
                if i >= len(args):
                    break
                value = args[i]
                i += 1
            option(name, value)
            continue
        j = 1
        while j < len(arg):
            flag = "-" + arg[j]
            j += 1
            if flag in spec.exec_options:
                return None
            if flag in spec.unignore:
                skips_ignored = False
            if flag in spec.recursive_options:
                recursive = True
            if flag in spec.no_pattern_options:
                takes_pattern = False
            if flag[1] in spec.short_values:
                value = arg[j:]
                if not value and i < len(args):
                    value = args[i]
                    i += 1
                option(flag, value)
                break
    if takes_pattern and not pattern_given and positional:
        patterns.append(positional.pop(0))
    return patterns, positional + option_roots, files, recursive, skips_ignored


def parse_find(args):
    """find's search roots, or None when an -exec-like action runs another command."""
    i = 0
    while i < len(args) and args[i] in ("-H", "-L", "-P"):
        i += 1
    roots = []
    while i < len(args) and not args[i].startswith(("-", "(", "!", ")")):
        roots.append(args[i])
        i += 1
    if any(arg in FIND_EXEC for arg in args[i:]):
        return None
    return [], roots, [], True, False


def git_grep_scope(words, cwd):
    """'inside' or 'outside' for git grep, None for any other git command. git grep
    searches the work tree of the directory it runs in, so it stays inside the
    repository unless it is pointed elsewhere or told to search ignored files."""
    i = 1
    while i < len(words) and words[i].startswith("-"):
        if words[i] == "-C" and i + 1 < len(words):
            cwd = resolve(words[i + 1], cwd)
            i += 2
        elif words[i] == "-c":
            i += 2
        elif words[i].startswith(("--git-dir", "--work-tree")):
            return "outside"
        else:
            i += 1
    if i >= len(words) or words[i] != "grep":
        return None
    if any(w in ("--no-index", "--no-exclude-standard") for w in words[i + 1 :]):
        return "outside"
    return "inside" if in_repository(cwd) else "outside"


def search_scope(words, cwd, reads_other_input):
    """'inside' when a simple command is a local search confined to the
    repository's own files, 'outside' for any other local search, and None when it
    is not a search."""
    program = os.path.basename(words[0])
    if program in SYSTEM_SEARCHERS:
        return "outside"
    if program == "git":
        return git_grep_scope(words, cwd)
    if program == "find":
        parsed = parse_find(words[1:])
    elif program in SEARCHERS:
        parsed = parse_search(SEARCHERS[program], words[1:])
    else:
        return None
    if parsed is None:
        return "outside"
    _, roots, files, recursive, skips_ignored = parsed
    if any(names_forbidden(path) for path in roots + files):
        return "outside"  # a path naming FreeTimeGS is never read
    if not roots:
        if reads_other_input:
            return "outside"  # it searches what the pipe or the redirection gives it
        if program in ("grep", "egrep", "fgrep") and not recursive:
            roots = []  # it reads its own empty input
        else:
            roots = ["."]
    for path in roots + files:
        if path == "-":
            return "outside"
        resolved = resolve(path, cwd)
        if not in_repository(resolved):
            return "outside"
        if path in roots and recursive and not skips_ignored and holds_environment(resolved):
            return "outside"
    return "inside"


def without_repository_searches(command, cwd):
    """The command without its simple commands that are searches confined to the
    repository, and whether another simple command searches elsewhere for a
    forbidden name."""
    segments = split_simple_commands(command)
    if segments is None:
        return command, False
    kept = []
    for text, reads_other_input in segments:
        try:
            words, via_xargs = command_words(shlex.split(text))
        except ValueError:
            kept.append(text)
            continue
        if not words:
            kept.append(text)
            continue
        if words[0] in ("cd", "pushd"):
            targets = [w for w in words[1:] if not w.startswith("-")]
            cwd = resolve(targets[0], cwd) if targets else resolve("~", cwd)
            kept.append(text)
            continue
        if words[0] == "popd" or not names_forbidden(text):
            if words[0] == "popd":
                cwd = None
            kept.append(text)
            continue
        scope = search_scope(words, cwd, reads_other_input)
        if scope is not None and via_xargs:
            scope = "outside"  # it searches files named by its input
        if scope == "outside":
            return command, True
        if scope is None:
            kept.append(text)
    return " ; ".join(kept), False


def check(tool, tool_input, cwd=None):
    """Return a reason string if the call must be blocked, else None. `cwd` is the
    session's working directory, which local searches are resolved against."""
    if not isinstance(tool_input, dict):
        return None
    if cwd is None:
        cwd = os.getcwd()

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
        rest, outside = without_repository_searches(rest, cwd)
        if outside:
            return OUTSIDE
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

    if tool in READ_TOOLS or tool in WRITE_TOOLS:
        for field in PATH_FIELDS:
            if names_forbidden(tool_input.get(field, "")):
                return "this path names FreeTimeGS or EasyVolcap."
        if tool == "Grep" and (
            names_forbidden(tool_input.get("pattern", "")) or names_forbidden(tool_input.get("glob", ""))
        ):
            # The Grep tool skips .gitignore'd files, so only its root must be inside.
            if not in_repository(resolve(str(tool_input.get("path") or "."), cwd)):
                return OUTSIDE
        if tool == "Glob" and names_forbidden(tool_input.get("pattern", "")):
            # Glob may walk into ignored directories: its pattern's fixed part must
            # stay inside the repository and away from the virtual environments.
            root = resolve(str(tool_input.get("path") or "."), cwd)
            start = resolve(str(tool_input.get("pattern", "")), root) if root else None
            if not in_repository(start) or holds_environment(start):
                return OUTSIDE
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
    tool_input = json.dumps(event.get("tool_input"), ensure_ascii=False)
    if len(tool_input) > 3000:
        tool_input = tool_input[:3000] + " [truncated]"  # truncate the field, so each line stays valid JSON
    record = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "session": event.get("session_id"),
        "tool": event.get("tool_name"),
        "input": tool_input,
        "reason": why,
    }
    try:
        with open(os.path.join(project, ".claude", "clean-room-guard.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def main():
    try:
        event = json.load(sys.stdin)
        cwd = event.get("cwd")
        why = check(
            str(event.get("tool_name", "")),
            event.get("tool_input", {}),
            cwd=str(cwd) if cwd else None,
        )
    except Exception:  # never block normal work because of a bug here
        return 0
    if why is None:
        return 0
    log_block(event, why)
    print(MESSAGE.format(why=why), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

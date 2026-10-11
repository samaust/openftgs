# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Repo constraints

- `gh` may target only this repository; the clean-room guard blocks anything else.
- Never delete or transfer issues (`gh issue delete` / `transfer` are denied in settings).
- `gh gist` is denied. Context pointers and any other shared text go in issue comments, never in gists.
- Links posted in issues are subject to the clean-room rules in CLAUDE.md.
- A Q-n row in `docs/clean-room/questions.md` that needs the owner's decision gets an issue, labelled and titled as "Q-n issues" below says; the answer goes back into that file through a short `q-<n>-answer` branch ("Where the clean-room files are written" in `docs/agents/clean-room.md`).

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,comments --jq '{number, title, body, labels: [.labels[].name], comments: [.comments[] | {author: .author.login, body}]}'`. The comment authors matter: only the owner's comments carry decisions ("Reading the tracker" in `docs/agents/clean-room.md`).
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v`; `gh` does this automatically when run inside a clone.

## Q-n issues

Two kinds of issue carry Q-n material. Both are labelled `q-n` and take no category role:

- **Q-n decision issue**: a numbered row of `docs/clean-room/questions.md` that needs the owner's decision. Title `Q-<n>: <short gap>`; the body gives the row's spec location, gap, provisional decision, alternatives and detecting test. It is opened only once the number is final, after the merge into `main` that brought the row. The owner answers in a comment; the answer goes back into the row through a short `q-<n>-answer` branch, and the issue closes after that merge.
- **Q-n candidates issue**: gaps found outside any PRD, without numbers yet. Title `Q-n candidates: <topic>`; the body lists each candidate with the columns of `questions.md` and the sources consulted. `triage` routes it into a PRD or an agent brief, whose implement run records the candidates.

Find them with `gh issue list --label q-n --state open`.

## Out-of-scope record

Rejected feature requests are recorded on the tracker, not in the repository: one canonical issue per rejected concept, labelled `wontfix` and `out-of-scope`, closed as not planned, with a decision comment (Decision, Reason citing the spec, Prior requests). Later requests for the same concept are closed with `Duplicate of #<canonical>`. List the record with `gh issue list --label out-of-scope --state closed`. Details: `.claude/skills/triage/OUT-OF-SCOPE.md`.

## Parent issues and blocking

GitHub's native relationships are the record: a ticket is a **sub-issue** of its PRD, and a blocking edge is an issue **dependency** ("blocked by" on one side, "blocking" on the other). Ticket bodies don't repeat them.

The write endpoints take an issue's database id, not its `#number`: `gh api repos/samaust/openftgs/issues/<n> --jq .id`. Put the path first in every `gh api` call, as below, so the settings allow rule matches it.

- **Add a sub-issue** (make #<child> a sub-issue of #<parent>): `gh api repos/samaust/openftgs/issues/<parent>/sub_issues --method POST -F sub_issue_id=<child-db-id>`
- **Remove a sub-issue**: `gh api repos/samaust/openftgs/issues/<parent>/sub_issue --method DELETE -F sub_issue_id=<child-db-id>`
- **List sub-issues**: `gh api repos/samaust/openftgs/issues/<parent>/sub_issues --jq '[.[] | {number, title, state}]'`
- **Get the parent**: `gh api repos/samaust/openftgs/issues/<n>/parent --jq '{number, title}'` (404 when the issue has no parent)
- **Add a blocked-by edge** (#<n> is blocked by #<blocker>): `gh api repos/samaust/openftgs/issues/<n>/dependencies/blocked_by --method POST -F issue_id=<blocker-db-id>`
- **Remove a blocked-by edge**: `gh api repos/samaust/openftgs/issues/<n>/dependencies/blocked_by/<blocker-db-id> --method DELETE`
- **List what blocks #<n>**: `gh api repos/samaust/openftgs/issues/<n>/dependencies/blocked_by --jq '[.[] | {number, title, state}]'`
- **List what #<n> is blocking**: `gh api repos/samaust/openftgs/issues/<n>/dependencies/blocking --jq '[.[] | {number, title, state}]'`
- **Count open blockers**: `gh api repos/samaust/openftgs/issues/<n> --jq .issue_dependencies_summary.blocked_by` (open blockers only; 0 means unblocked)

Outside an implement run, a ticket is on the **frontier** when it is open, is a sub-issue of its PRD, and has no open blockers. During a run, tickets stay open until the merge into `main`, so `implement-with-subagents` tracks the frontier from its own record of merged tickets (its step 6), not from this count.

## Pull requests as a triage surface

**PRs as a request surface: no.** `/triage` reads this flag; only issues are triaged in this repo.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

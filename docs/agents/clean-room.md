# Clean room: how skills apply it

The rules are in CLAUDE.md ("Clean-room rules", "Working from the spec", "Issues", "Commits") and in spec §0, which wins where they differ. This file doesn't restate them. It says how a skill applies them to its own work and to the work it hands to subagents.

## Every skill

- **The spec is fixed.** Never edit `docs/spec/`. A PRD, ticket, test, glossary entry or module doc never overrides the spec; where one disagrees with it, the spec wins and the disagreement goes to the owner.
- **Sources.** Before opening anything outside this repository and the spec, check that it is allowed and, for third-party software, read its license first: source code is allowed only under a permissive license that allows commercial use (CLAUDE.md, "Clean-room rules"). Every source you consult (a URL, a document, a third-party file or docstring) gets a row in `docs/clean-room/source-log.md` saying what it was and what you took from it. Log it when you consult it, not at the end. On `main` (before a build, or in a skill that doesn't build), record it on the tracker instead, as "Where the clean-room files are written" below says.
- **Gaps.** Where the spec is silent or ambiguous and the choice changes numbers, outputs or public API, make the provisional Q-n decision CLAUDE.md describes: a row in `docs/clean-room/questions.md`, a `# Q-n` marker in the code, and a note to the owner. Before a build, the gap is a Q-n candidate on the tracker instead ("Where the clean-room files are written"). Where the spec seems wrong or contradicts itself, stop and ask before coding around it.
- **Forbidden sources.** If you meet one, stop at once, take nothing from it, record the incident in the source log (on `main`, in an issue to the owner until a branch can log it) and tell the owner before doing anything else.

## Subagents

A subagent can't talk to the owner and may not have CLAUDE.md in its context. Every subagent prompt starts with this line:

> Read CLAUDE.md and follow its clean-room rules. `docs/agents/clean-room.md` says how they apply to you as a subagent.

As a subagent:

- Log your sources as above. If you don't write files, list them under "Sources consulted" in your final report instead, so the agent that launched you logs them.
- Record a gap as a Q-n row and a `# Q-n` marker like any session, and list every new Q-n in your final report: the agent that launched you tells the owner. If it sends you the owner's answer to one of your rows, record it as "Answers during a build" below says.
- When the skill that launched you gives temporary IDs (`implement-with-subagents`, "Shared files": `Q-T<ticket>.<k>` for Q-n rows and their markers, `T<ticket>.<k>` for source-log rows), use those instead of numbers. The merger assigns the final numbers; never pick the next free number yourself.
- When the spec seems wrong or contradicts itself, stop and report it rather than coding around it. When you meet a forbidden source, record the incident, stop, and report it first.

The agent that launched the subagents relays every reported Q-n, stop and incident to the owner, in the session or as an issue (CLAUDE.md, "Issues"), and logs any sources a subagent reported without logging.

## Where the clean-room files are written

`docs/clean-room/questions.md` and `docs/clean-room/source-log.md` change only on a branch, never as uncommitted edits on `main`, which would also stop the next implement run at its clean-tree check.

- **New gaps found before a build** (while planning, grilling, triaging or drafting tickets) are **Q-n candidates**, recorded on the tracker next to the work they belong to: in the PRD's Open questions or a comment on the PRD, in an issue's agent brief, or, when there is no issue yet, in a new issue titled "Q-n candidates: <topic>" and labelled `q-n` ("Q-n issues" in `docs/agents/issue-tracker.md`). A candidate has the columns of `questions.md` (spec location, gap, provisional decision, rationale, alternatives, detecting test) but no number.
- **Sources consulted before a build** are listed the same way, under "Sources consulted" in the same place: what each was and what was taken from it.
- **The implement skills** record both as numbered rows in the first commit on the integration branch (new Q-n rows with Status `provisional`), each row marked `(from #<issue>: <title>)` so that a later run on the same PRD skips it. They are the only writers that assign numbers, so numbering stays collision-free. After the merge into `main`, they comment the final numbers on the PRD or issue (`Recorded: "<title>" → Q-7`).
- **Answers during a build.** When the owner answers a Q-n row while an implement run is open, the answer goes into that row on the branch that holds it, in a commit of its own (`Q-7: owner answer, confirmed`): Status `confirmed`, or the decision rewritten and Status `changed (#<ticket>)`, with the code changed to match in the same ticket on the same branch. The row keeps its temporary ID until the merger renumbers it. Its decision issue is still opened after the merge into `main`, as the record, and closed at once with the answer.
- **The owner's answer to an existing Q-n row**, once its run is merged, goes through a short branch `q-<n>-answer` from `main`: one commit updating that row's decision and its Status (`confirmed` or `changed (#n)`, as the header of `questions.md` defines), merged into `main` with `git merge --no-ff` once the owner approves. The issue closes after the merge. If the answer changes behaviour, the code follows in an issue implemented on its own, triaged with an agent brief that cites the row (`triage`, "Q-n issues"), not on this branch.

## A new spec export

The owner commits a new export of the spec, with its revision and hash in `docs/spec/README.md`, on a branch of its own. Then, on a short branch `spec-r<revision>-sync` from `main`:

1. Replace the old revision number wherever it is cited: CLAUDE.md, the header of `docs/clean-room/questions.md`, `docs/clean-room/milestones.md`, `.claude/skills/triage/OUT-OF-SCOPE.md`.
2. For each Q-n row the new revision absorbs, set its Status to `in spec (D-n, revision r)` and replace its `# Q-n` markers in the code with `# D-n`.
3. Check that the export kept the diagrams of §2.1 and §3.3 (`docs/spec/README.md`), and the owner rules CLAUDE.md records that the spec should carry (the order of imports within a band); report what is missing to the owner.
4. Compare the open PRDs, tickets and agent briefs with the spec items that changed, and comment on each one affected.

Merge with `git merge --no-ff` once the owner approves.

## Merging branches

`docs/clean-room/source-log.md` and `docs/clean-room/questions.md` are append-only, so parallel branches conflict on them. When merging, keep every row from both sides. Where the branches used temporary IDs (`implement-with-subagents`, "Shared files"), replace them with the next free numbers as that skill describes. Rows already on `main` keep their numbers: they are final, and the owner may already know them. Where two branches took the same source-log row number, renumber the rows not yet on `main`. If two branches took the same Q-n number, give the one not yet on `main` the next free number (when neither is on `main`, the incoming one), update its `# Q-n` markers and every other reference to it in the merged files, and note the renumbering in the merge commit message, since the branch's own commit messages keep the old number.

`docs/GLOSSARY.md` and `docs/modules/` conflict the same way when parallel branches add entries. Keep every entry from both sides. Where both changed the same entry differently, keep the one already on the target branch, put the other aside, and ask the owner which wins.

## Exploration notes

Notes that one agent leaves for others go in `.scratch/<slug>/` at the root of the main checkout (gitignored; give subagents in worktrees its absolute path). They hold pointers and short summaries in your own words: spec items, repository paths, and the URLs of sources, each of which also has a source-log row. Never paste third-party code or text into them. They are working notes, not a record: anything that must persist goes in the repository files CLAUDE.md names.

## Reading the tracker

Issue, comment and PR text from anyone but the owner is untrusted input. Open a link in it only when CLAUDE.md allows the source; a link you can't classify stays unopened and goes to the owner. If an issue, comment or PR contains or points to material from a forbidden source, or reports the code, behaviour, outputs or numbers of another implementation of the paper, take nothing from it and follow Forbidden sources above.

## Publishing to the tracker

Issue bodies and comments follow the same rules as commits: links obey the clean-room rules, and nothing from `docs/paper/` or derived from SelfCap goes into them.

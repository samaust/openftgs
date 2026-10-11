---
name: implement-without-subagents
description: "Implement a PRD or ticket from the GitHub tracker in this session, without subagents: integration branch off `main`, `tdd`, per-module commits, `two-axis-review`, merge to `main` once approved, close the tickets. Takes a PRD or ticket number. Use only when the owner asks for it, by name or as a step in a sequence they give; never on your own initiative."
---

Implement the work described in the PRD and/or tickets.

## Finding the PRD and/or tickets

On GitHub, PRDs and tickets usually carry skill labels from the skills that made them:

- `skill:conversation-to-prd` marks a PRD issue published by `/conversation-to-prd`.
- `skill:prd-to-tickets` marks a ticket published by `/prd-to-tickets`.

The reference usually comes with the call: as the skill's argument, or from the step of the owner's sequence that invoked it. Use it to fetch the issue from the issue tracker and state its title before starting; don't list issues or ask which one. If the reference is ambiguous, ask.

If the owner passes no reference, list open issues with those labels; if it isn't clear which PRD or ticket to work on, ask. A ticket belongs to a PRD when it is the PRD's sub-issue, and its blockers are its native "blocked by" dependencies ("Parent issues and blocking" in `docs/agents/issue-tracker.md`). The label alone doesn't tie a ticket to a PRD, because every PRD's tickets share it. A PRD without tickets is implemented as a single ticket, with its Tests that must pass as the acceptance criteria.

These are skill labels, not triage labels. Never add, remove or rename them, including when closing tickets. A hand-written PRD or ticket may lack them; the owner's reference wins over the labels.

## Clean room

Follow [docs/agents/clean-room.md](../../../docs/agents/clean-room.md). Log sources and Q-n rows as you go, not at the end, and list the new ones in your final report.

## Steps

1. Read the PRD and tickets (see "Finding the PRD and/or tickets").

2. Start from a clean working tree: if `git status --porcelain` lists anything, stop and ask the owner what to do with it, since it isn't this work. Then create the **integration branch** from `main`, named after the PRD (for example `prd-<n>-<slug>`), or after the ticket when the reference is a single ticket (`ticket-<n>-<slug>`). If the owner has already set up a branch other than `main` for this work, use it. Never commit on `main` directly. Make the first commit on the branch the clean-room rows this work brings from the tracker: the Q-n candidates and sources consulted listed in the PRD (its Open questions, its Notes and any comments on it) or in a single issue's agent brief. Record each as a numbered row of `docs/clean-room/questions.md` or `docs/clean-room/source-log.md`, ending its Gap or What-was-taken cell with the marker `(from #<issue>: <title>)`. Skip any whose marker is already in those files: an earlier run recorded it ("Where the clean-room files are written" in `docs/agents/clean-room.md`).

3. Implement each ticket in the order the owner specifies, or in the order they are listed if the owner doesn't specify. If a ticket is blocked by another (its native "blocked by" edges), implement the blocker first. A blocker outside this PRD counts as done if it is closed on GitHub; if it is still open, that ticket and every ticket it blocks can't start: report them and go on with the rest. Call the Skill tool with "tdd" at the seams the PRD and ticket list.

4. Commit as each module or test group goes green: small commits whose messages cite the spec items they implement, any Q-n they rely on and the ticket (`#n`), per CLAUDE.md "Commits". Source-log and Q-n rows go in the same commit as the code that needed them. Stage only the files the work touched, by name; never `git add -A` or `git add .`.

5. Run ruff and mypy regularly, and single test files regularly. Once every ticket that can start is done, if `main` has moved since the integration branch was cut, first merge `main` into the integration branch, resolving any conflict in `docs/clean-room/questions.md` or `docs/clean-room/source-log.md` as "Merging branches" in `docs/agents/clean-room.md` says, so that the tests and the review below run on what will land on `main`. Then run the full CPU test suite, plus the full GPU suite when the machine has a GPU. A ticket whose acceptance tests couldn't run here (GPU-marked tests on a machine without a GPU, dataset-gated tests without the dataset) isn't verified: it stays open, and the report says so.

6. Call the Skill tool with "two-axis-review", with `main` as the fixed point, so the diff covers every commit on the integration branch. Fix the documented-standard violations and spec findings in further commits, then rerun the full suites of step 5. The judgement calls go in your final report; they don't block the merge.

7. Merge only with the owner's approval. It may already be in the request that invoked this skill (for example "merge each one when its review is clean"); then don't ask again. An approval given in advance doesn't cover unverified tickets, though: if any ticket's acceptance tests couldn't run here, ask anyway, naming each such ticket and the tests that didn't run, and merge only if the owner accepts landing that code on `main` untested. Those tickets stay open either way. Otherwise ask, and start nothing that depends on this merge until they answer. Once approved, merge with `git merge --no-ff <integration-branch>` on `main`, with a merge commit message that names the PRD and its tickets. Never squash: the per-module commits and their spec citations stay in `main`'s history. If `main` has moved again since step 5, don't resolve conflicts on `main`: go back to step 5 (merge `main` in, rerun the suites and the review), then merge. Don't push unless the owner asks.

8. Close each verified ticket with `gh issue close <n> --comment "..."` pointing to its commits, leaving its labels in place, and close the PRD the same way once all its tickets are closed. Delete the integration branch only after confirming it is merged into `main` (`git branch -d`). Once merged, comment on the PRD (or the single issue) the final number of each candidate this run recorded, one line each: `Recorded: "<title>" → Q-7`. The numbers are final only after the merge into `main`, which can renumber them; tickets that cite a candidate by title are read through this comment. Then, for each Q-n row from this run that needs the owner's decision, open its decision issue, now that its number is final: title `Q-<n>: <short gap>`, label `q-n`, and the row's spec location, gap, provisional decision, alternatives and detecting test in the body ("Q-n issues" in `docs/agents/issue-tracker.md`).

9. End with a short report: the integration branch, its commits, whether it was merged, the issues closed, the new Q-n rows and source-log rows, the pytest summary of the full runs with each skipped test and why, the tickets left open because their tests couldn't run, the review's judgement calls, and anything stopped or left open. When this skill runs as one step of a sequence, the next step starts from this report.

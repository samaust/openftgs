# Out-of-Scope Record

Rejected feature requests are recorded as closed GitHub issues labelled `out-of-scope`, not as files in the repository: triage runs on `main`, and the tracker needs no branch or merge. The record serves two purposes:

1. **Institutional memory**: why a request was rejected, so the reasoning isn't lost when the issue is closed.
2. **Deduplication**: when a new request matches a prior rejection, triage surfaces the earlier decision instead of re-litigating it.

In this repository, scope is the spec. Most rejections read "not in spec revision 549j" and cite the item that bounds it. A request that should change the spec is not rejected here: it is the owner's decision, made in the spec and arriving in a new export.

## The canonical issue

One issue per **concept**, not per request: the first request for a concept becomes its canonical issue. It carries the labels `wontfix` and `out-of-scope`, is closed as not planned (`gh issue close <n> --reason "not planned"`), and ends with a decision comment in this shape:

```markdown
## Out of scope: Interactive viewer

**Decision:** openftgs has no interactive viewer.

**Reason:** the command line is fixed by spec §2.1 (`openftgs.cli`): convert,
cache, init, train, render, eval, export and bench. Viewing renders is left to
external tools reading the files of §2.2.

**Prior requests:**
- #42: "Add a live viewer"
```

Name the concept in the heading so that someone reading the list of closed `out-of-scope` issues recognises it without opening it.

### Writing the reason

The reason is substantive and durable: not "we don't want this" but why, citing the spec item that bounds the scope (§n, D-n) or the technical constraint. A temporary circumstance ("not now") is a deferral, not a rejection: leave such a request open.

## When to check the record

During triage (step 1, Gather context), list the record with `gh issue list --label out-of-scope --state closed --json number,title,body,comments`, and match the new request against it by concept, not keyword: "live preview" matches "Interactive viewer". (`gh issue list --search "…"` narrows the list; the clean-room guard blocks GitHub-wide searches.) If there's a match, surface it to the owner: "This is similar to #42 (Interactive viewer), rejected because [reason]. Do you still feel the same way?"

The owner may:

- **Confirm**: close the new issue with `gh issue close <n> --duplicate-of <canonical> --comment "..."` (GitHub marks it a duplicate and links the two; the comment starts with the triage disclaimer), and add it to the canonical issue's Prior requests list by editing the decision comment ("Edit a comment" in `docs/agents/issue-tracker.md`).
- **Reconsider**: reopen the canonical issue, remove its `out-of-scope` label, and let the new request proceed through normal triage, or become a spec change.
- **Disagree**: the requests are related but distinct; proceed with normal triage.

## When to record a rejection

Only when an **enhancement** (not a bug) is rejected as `wontfix`. Don't label `out-of-scope` an issue closed as `wontfix` because it is **already implemented**: that is a built feature, not a rejected one, and it would poison the deduplication. Its closing comment points to where the feature lives instead.

The flow:

1. The owner decides a request is out of scope.
2. Check the record for a matching canonical issue.
3. If there is one: close the request as a duplicate of it and add it to its Prior requests, as under Confirm.
4. If there isn't: make this request the canonical issue. Post the decision comment, add the labels `wontfix` and `out-of-scope`, and close it as not planned.

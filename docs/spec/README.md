# Specification

`openftgs-spec.md` is an export of the owner's document "FreeTimeGS — Clean-Room Specification & Architecture", revision 549 (2026-10-07), amended in the repository as revisions 549a, 549b and 549c (2026-10-12) and 549d (2026-10-15).

- SHA-256: `c4d264865c2f93e65c7608f7e7bb007f2c6f33a0aa1294f201f9e50c449f7d53` (revision 549d)
- SHA-256 of revision 549c as committed: `315724dfeafb66d78f8004365b42343c5a6170e32e05ee81093bf01a1dab2b21`
- SHA-256 of revision 549b as committed: `a3b7d662a5dd7a146c830bf14b962b4aee1a94c02526a5eaaaf952af169ba87f`
- SHA-256 of revision 549a as committed: `dba7b3248516f828e951502cd38f3ad355b66079f1d786acbb0bcc7cd1f3492b`
- SHA-256 of revision 549 as committed: `46eb902931e83b9f0c71b1a512c015ed54f88bf525be30a1aa8073f6b4bed98c`
- SHA-256 of the export as received: `ac1d484d140a16f4947473dd7b8e6c21bafa0ff7a9f3141c6c7dfba384cc9368`

The Markdown export left the two diagrams as placeholders. On 2026-10-11, at the owner's request, they were redrawn in Mermaid from the owner's PDF export of the same document: the package layers in §2.1 and the harness data flow in §3.3. Two omissions were fixed while redrawing them: the `cli` box now lists all eight commands of §2.1 (`cache` and `bench` were missing), and the adapter boxes list `info` and the case generator lists the harness config, as the §3.3 adapter contract requires. Nothing else in the file changed.

Revision 549a is a one-time exception: on 2026-10-12 the owner asked the implementer to edit this copy in the owner's name, applying the wording the owner approved on issue #36 (D-70 to D-74, from Q-19, Q-30, Q-34, the M4 milestone review's finding 4 and Q-28) and keeping the Appendix B count and the revision notes consistent. The owner's document must receive the same changes, or its next export would drop them.

Revision 549b is a second such exception, on the same day: the owner asked the implementer to edit this copy in the owner's name once more, applying the wording the owner approved on issue #50 (D-75 and D-76, from Q-37 and Q-38: the gsplat backend calls gsplat's stage functions, returns `visible` in both layouts and inverts no view matrix; §2.1 `RenderOutput` and §2.4 rule 3 follow) and keeping the Appendix B count, D-49 and the revision notes consistent. The owner's document must receive these changes too, or its next export would drop them.

Revision 549c is a third such exception, on the same day: the owner asked the implementer to edit this copy in the owner's name once more, applying the wording the owner approved on issue #59 (D-77 and D-78, from Q-40 and Q-43: `total_loss` takes the perceptual network as a keyword and returns its terms as detached tensors, and U-11's gradcheck holds μt and ŝ at their base values in L\_reg under E8's stop-gradient; §2.1 `total_loss` and §3.1 U-11 follow) and keeping the Appendix B count and the revision notes consistent. The optional sentence #59 offered under E9 or §2.4 rule 3 was left out, because rule 3 already states that loss values are read every 100 iterations and D-77 records who builds and passes the network. The owner's document must receive these changes too, or its next export would drop them.

Revision 549d is a fourth such exception, on 2026-10-15: the owner asked the implementer, this one time, to make the changes of issue #108 in the owner's name. The three datasets the paper evaluates on are licensed for non-commercial research only and openftgs may later be used commercially, so the project validates on openftgs-synth, the owner's CC0 rendered dataset (D-79). §0 allows it as a source; §1.4 records the choice; §3.1 adds the synthetic-dataset checks G-1 to G-7 and marks C-1 to C-5 as not run by this project; §3.4 names its scenes for the benchmarks that named a dataset scene; Appendix A gains rows 41 to 44; and the Appendix B count and the revision notes are kept consistent. The owner's document must receive these changes too, or its next export would drop them.

Don't edit it here. Changes are made in the owner's document and arrive as a new export that replaces this file in a single commit, with the revision and hash above updated.

# Specification

`openftgs-spec.md` is an export of the owner's document "FreeTimeGS — Clean-Room Specification & Architecture", revision 549 (2026-10-07), amended in the repository as revision 549a (2026-10-12).

- SHA-256: `dba7b3248516f828e951502cd38f3ad355b66079f1d786acbb0bcc7cd1f3492b` (revision 549a)
- SHA-256 of revision 549 as committed: `46eb902931e83b9f0c71b1a512c015ed54f88bf525be30a1aa8073f6b4bed98c`
- SHA-256 of the export as received: `ac1d484d140a16f4947473dd7b8e6c21bafa0ff7a9f3141c6c7dfba384cc9368`

The Markdown export left the two diagrams as placeholders. On 2026-10-11, at the owner's request, they were redrawn in Mermaid from the owner's PDF export of the same document: the package layers in §2.1 and the harness data flow in §3.3. Two omissions were fixed while redrawing them: the `cli` box now lists all eight commands of §2.1 (`cache` and `bench` were missing), and the adapter boxes list `info` and the case generator lists the harness config, as the §3.3 adapter contract requires. Nothing else in the file changed.

Revision 549a is a one-time exception: on 2026-10-12 the owner asked the implementer to edit this copy in the owner's name, applying the wording the owner approved on issue #36 (D-70 to D-74, from Q-19, Q-30, Q-34, the M4 milestone review's finding 4 and Q-28) and keeping the Appendix B count and the revision notes consistent. The owner's document must receive the same changes, or its next export would drop them.

Don't edit it here. Changes are made in the owner's document and arrive as a new export that replaces this file in a single commit, with the revision and hash above updated.

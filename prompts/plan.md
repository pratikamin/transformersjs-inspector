# Plan prompt

Paste this into a **fresh** agent session — not the one that did research.
The point is a clean context window that reads the research doc as a document,
rather than one carrying all the exploration that produced it.

---

Read `docs/00-brief.md` and `docs/01-research.md`.

Write the implementation plan into `docs/02-plan.md`. Follow that file's section
headings exactly.

Rules:

1. **Specific enough to predict the diff.** Name functions, not areas. "Add a
   `retry` wrapper around `fetchUser` in `api/client.ts`" — not "improve error
   handling in the API layer."
2. **Every phase ends green.** Each phase must leave typecheck, tests, and lint
   passing on its own. If a phase cannot end green, it is two phases, or the
   ordering is wrong.
3. **Every phase has a real verification command**, taken from the brief's
   verification section or added deliberately. "Manually check the UI" is not a
   verification command.
4. **Name the alternative you rejected**, and why it loses. A plan with no
   rejected alternative was not a decision, it was the first idea.
5. **Respect the fence.** Nothing in the brief's "Explicitly not doing" or "Must
   not change" appears in the plan. If research showed the fence is untenable,
   say so at the top and stop — that is a conversation for the human, not a
   decision for you.
6. **Do not write code.** Do not modify any file other than `docs/02-plan.md`.

Then decompose the plan into `prd.json`. Each story is **one unit of work sized to
fit a single context window** — if a story would need more than roughly one focused
session, split it. Stories are ordered so the tree is green after every one.

Stop. A human reviews both files before the loop runs.

# Research prompt

Paste this into a fresh agent session. Attach nothing else — let it explore.

---

Read `docs/00-brief.md`.

Research this codebase and fill in `docs/01-research.md`. Follow that file's
section headings exactly; do not add sections or restructure it.

Rules:

1. **Use subagents for search.** Dispatch exploration to subagents and have them
   return condensed findings. Do not let raw grep, glob, or file-dump output into
   your own context — it is the single fastest way to poison this document.
2. **Cite `path:line`, never paste code blocks.** If a reader needs the code, they
   can open the file. Your job is the map, not the territory.
3. **Answer every open question in the brief.** If you cannot answer one from the
   code, write "Unknown — would require <specific thing>". Never guess and never
   soften a guess into a hedge that reads like knowledge.
4. **Challenge the premise.** If the brief describes a problem that does not exist,
   or misdiagnoses one that does, say so in the Verdict and stop. Do not write a
   research doc for the wrong problem out of politeness.
5. **Do not write a plan.** Do not suggest an approach. Do not modify any file
   other than `docs/01-research.md`. Describing what should change is planning;
   describing where change would land is research. Stay on the second side.

Keep your context under 60% utilization. If you are approaching it, write down
what you have and say which areas remain unexplored.

Stop when the document is complete. A human reviews it before anything else happens.

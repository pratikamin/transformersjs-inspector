# Research prompt — greenfield technology scan

> Use this instead of `research.md` when there is no existing system to map.
>
> The default research prompt answers "how does this work today." On greenfield there is
> no today. This one answers a different question: **what should we build this out of,
> and what will bite us.** Same gate, same discipline, different subject.
>
> Rewrite `docs/01-research.md`'s section headings to match your brief's open questions
> before running this — one section per question.

Paste this into a fresh agent session.

---

Read `docs/00-brief.md`.

Answer its open questions and fill in `docs/01-research.md`. Follow that file's section
headings exactly; do not add sections or restructure it.

Rules:

1. **Use subagents for search, and for the web.** Dispatch each open question to its own
   subagent and have it return condensed findings. Do not let raw search results, full
   documentation pages, or long code samples into your own context.
2. **Prototype the uncertain parts, do not speculate about them.** The questions worth
   asking on greenfield are usually feasibility questions, and feasibility is settled by
   building the smallest thing that would fail. Put throwaways in `/tmp`, not in the repo,
   and report what you actually ran and where it broke.
3. **Name specific libraries and versions**, with the reason each is or is not worth a
   dependency. "There are several options" is not a finding. Every dependency needs an
   argument, because on greenfield they are cheap to add and expensive to remove.
4. **Identify the load-bearing question and say which one it is.** One of the open
   questions is usually the constraint the whole architecture hangs off. If it does not
   hold, say so in the Verdict and stop — that is a conversation, not a decision for you
   to route around.
5. **Respect the fence.** If research shows something ruled out in the brief is genuinely
   unavoidable, say so at the top and stop. Do not quietly design it back in.
6. **Do not write a plan and do not write project code.** Modify no file other than
   `docs/01-research.md`.

Keep your context under 60% utilization. If you approach it, write down what you have and
say which questions remain open.

Stop when the document is complete. A human reviews it before planning starts.

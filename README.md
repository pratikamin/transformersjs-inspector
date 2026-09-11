# Agentic Build Kit

A fill-in-the-blank scaffold for building complex applications with coding agents.

It combines the two things that actually work, and keeps them in their lanes:

| Half | Method | What it buys you |
|---|---|---|
| **Think** | RPI (Research -> Plan -> Implement) with human gates | Correctness. A bad plan line costs hundreds of bad code lines; a bad research line costs thousands. |
| **Grind** | Ralph loop (stateless agent, state on disk) | Volume. Context resets every iteration, so quality does not decay over a 200-step build. |

Plus a third layer that makes both cheaper over time:

| Layer | Method | What it buys you |
|---|---|---|
| **Remember** | Agent Skills (`SKILL.md`, progressive disclosure) | Knowledge you paid for once gets reloaded for free, at ~100 tokens/skill until needed. |

## How to use it

```bash
cp -r ~/.claude/templates/agentic-build/. /path/to/your/project/
```

Then fill out, in this order. **Do not skip ahead** — each file is the input to the next.

1. `docs/00-brief.md` — you write this. By hand. This is the thinking you cannot outsource.
2. `docs/01-research.md` — agent writes, **you review**. Gate 1.
3. `docs/02-plan.md` — agent writes, **you review**. Gate 2. This is the highest-leverage review in the whole process.
4. `prd.json` — agent derives from the plan. One story = one context window.
5. `./scripts/loop.sh` — run it. Walk away. Come back to commits.

`progress.md` and `AGENTS.md` are written *by* the loop, not by you. Leave them alone except to correct a wrong lesson.

## The one rule

Keep context utilization between **40% and 60%**. Above that, output quality drops measurably. Every design decision in this kit — subagents for search, research docs instead of raw grep output, fresh context per loop iteration — exists to serve that number.

## Which half do I need?

- **Greenfield, well-understood domain** -> light brief, skip research, go straight to plan -> Ralph. Ralph shines here.
- **Greenfield with real unknowns** (unproven approach, unfamiliar platform) -> use `prompts/research-greenfield.md`. Same gate, but the subject is a technology scan rather than a codebase map, and feasibility gets settled by prototyping rather than reasoning.
- **Brownfield, existing system** -> research phase is mandatory and is where you spend your attention. Ralph is optional; often a supervised RPI pass is better.
- **One-off change** -> none of this. Just ask the agent.

Do not run a Ralph loop on a codebase whose failure modes you have not personally watched. Watch the first ten iterations by hand.

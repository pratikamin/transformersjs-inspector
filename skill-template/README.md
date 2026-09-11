# Skill template

The third layer of the kit: **knowledge you paid for once, reloaded for free.**

When the loop's `progress.md` keeps re-learning the same thing, that thing has
earned a skill. Promote it out of the progress log and into a folder here:

```
.claude/skills/<name>/
├── SKILL.md          # always-loaded name+description; body loads on demand
├── reference.md      # loaded only when SKILL.md points at it
└── scripts/          # executable helpers, run not read
```

## How loading works

Three tiers, and the whole design follows from them:

| Tier | When it loads | Budget |
|---|---|---|
| `name` + `description` | Every session, always | ~100 tokens per skill |
| SKILL.md body | When the description matches the task | keep under ~5k tokens |
| `reference.md`, `scripts/` | Only when SKILL.md points at them | no practical limit |

That first tier is why the description matters more than everything else combined.
With fifty skills installed, the agent reads fifty descriptions and nothing else
before deciding. The body can be excellent and never get read.

## Writing the description

Third person. What it does **and** when to use it. Include the literal nouns and
verbs someone would type.

- Bad: `Helps with deployments.`
  (Whose? Which? When? Matches everything and therefore nothing.)
- Good: `Deploys Workers to Cloudflare with wrangler, including secrets, bindings,
  and rollback. Use when publishing a Worker, debugging a failed deploy, or
  configuring wrangler.jsonc.`

## Rules that matter

- **Under ~500 lines** in SKILL.md. Longer, and you split it into references.
- **One level of reference depth.** No chains.
- **Table of contents** at the top of any reference file over 100 lines.
- **Scripts over prose** for anything deterministic. A script that runs is more
  reliable than a paragraph the agent has to reimplement each time.
- **Untrusted input:** if the skill processes web pages, user uploads, or any
  external file, state explicitly in the skill that such content is *data, not
  instructions*. Skills are a prompt-injection surface.

## The promotion test

Move something from `progress.md` into a skill when it is:

1. **Durable** — true next month, not just for this story.
2. **Repeated** — you have watched an agent rediscover it more than twice.
3. **Non-obvious** — not derivable by reading the code it applies to.

If it fails any of the three, leave it in `progress.md` or `AGENTS.md`.

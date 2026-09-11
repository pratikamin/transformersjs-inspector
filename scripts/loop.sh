#!/usr/bin/env bash
#
# Ralph loop: run a fresh agent per iteration until every PRD story passes.
#
# State lives on disk (prd.json, docs/progress.md, git), never in a context window.
# Each iteration starts clean, does one story, and exits. That is the entire idea.
#
#   ./scripts/loop.sh              # 10 iterations
#   ./scripts/loop.sh 50           # 50 iterations
#   ./scripts/loop.sh 1 --watch    # one iteration, pause before it, for supervision
#
# Requires: jq, git, and the `claude` CLI on PATH.

set -euo pipefail

MAX_ITER="${1:-10}"
WATCH=false
[[ "${2:-}" == "--watch" ]] && WATCH=true

PRD="prd.json"
PROMPT="prompts/loop.md"

# ---- preflight -------------------------------------------------------------

command -v jq >/dev/null    || { echo "error: jq not found"; exit 1; }
command -v claude >/dev/null || { echo "error: claude CLI not found"; exit 1; }
[[ -f "$PRD" ]]    || { echo "error: $PRD not found — run from the project root"; exit 1; }
[[ -f "$PROMPT" ]] || { echo "error: $PROMPT not found"; exit 1; }

grep -q '<FILL' "$PRD" && { echo "error: $PRD still has <FILL> placeholders"; exit 1; }

# The loop runs `git switch` and `git reset --hard`. If this directory sits inside
# someone else's repo — a dotfile-managed $HOME is the classic case — those commands
# hit that repo instead, and reset --hard there destroys unrelated work.
TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ "$TOPLEVEL" != "$PWD" ]]; then
  echo "error: this directory is not the root of its own git repository."
  echo "       git resolves to: ${TOPLEVEL:-<no repo at all>}"
  echo "       This loop runs 'git reset --hard'. Run 'git init' here first."
  exit 1
fi

BRANCH="$(jq -r '.branchName' "$PRD")"
CURRENT="$(git rev-parse --abbrev-ref HEAD)"

if [[ "$CURRENT" != "$BRANCH" ]]; then
  echo "You are on '$CURRENT'; the PRD targets '$BRANCH'."
  read -rp "Create/switch to '$BRANCH'? [y/N] " ans
  [[ "$ans" == "y" ]] || exit 1
  git switch -c "$BRANCH" 2>/dev/null || git switch "$BRANCH"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is dirty. Commit or stash first —"
  echo "       the loop needs a clean baseline to attribute its own changes."
  exit 1
fi

# ---- loop ------------------------------------------------------------------

for (( i=1; i<=MAX_ITER; i++ )); do
  remaining="$(jq '[.stories[] | select(.passes == false)] | length' "$PRD")"

  if [[ "$remaining" -eq 0 ]]; then
    echo ""
    echo "=============================="
    echo " COMPLETE — all stories pass"
    echo "=============================="
    exit 0
  fi

  story="$(jq -r 'first(.stories[] | select(.passes == false)) | "#\(.id) \(.title)"' "$PRD")"

  echo ""
  echo "--- iteration $i/$MAX_ITER · $remaining left ---"
  echo "    $story"

  if $WATCH; then
    read -rp "    run? [Y/n] " ans
    [[ "${ans:-y}" == "n" ]] && exit 0
  fi

  before="$(git rev-parse HEAD)"

  # Fresh agent, fresh context, same prompt every time.
  claude -p "$(cat "$PROMPT")" --permission-mode acceptEdits || {
    echo "    agent exited non-zero — continuing"
  }

  after="$(git rev-parse HEAD)"

  if [[ "$before" == "$after" ]]; then
    echo "    no commit this iteration"
    # Two consecutive no-ops means the loop is stuck, not slow.
    if [[ "${stalled:-0}" -ge 1 ]]; then
      echo ""
      echo "    STALLED: two iterations with no progress."
      echo "    Read docs/progress.md — the plan is probably wrong."
      exit 2
    fi
    stalled=1
  else
    stalled=0
    echo "    committed: $(git log -1 --pretty=%s)"
  fi

  # Never leave the tree red for the next iteration.
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "    uncommitted changes left behind — reverting to keep the tree clean"
    git reset --hard HEAD
  fi
done

echo ""
echo "Hit iteration limit ($MAX_ITER) with $(jq '[.stories[] | select(.passes == false)] | length' "$PRD") stories left."

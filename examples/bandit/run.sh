#!/usr/bin/env bash
# Replays a reward log through every bandit policy.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
# Binary location: $PROPAGON_BIN, else cargo's target dir (honors any
# target-dir override), else the conventional workspace path.
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4 || true)"
BIN="${PROPAGON_BIN:-${TARGET_DIR:-../../target}/release/propagon}"
mkdir -p out

for policy in greedy epsilon-greedy upper-confidence-bound kl-ucb thompson-beta thompson-gaussian exp3; do
  echo "== $policy" >&2
  "$BIN" bandit "$policy" rewards > "out/$policy.scores"
done

# Sliding-window UCB forgets old evidence — the policy for drifting arms.
echo "== sliding-window-ucb" >&2
"$BIN" bandit sliding-window-ucb --window 200 rewards > out/sliding-window-ucb.scores

# LinUCB shares strength across arms through per-round context features.
echo "== linucb" >&2
"$BIN" bandit linucb rewards.contextual > out/linucb.scores
echo "linucb pick for context (1, 0): $("$BIN" bandit linucb --select-for '1,0' rewards.contextual)" >&2

echo "== next arm to play, per policy" >&2
for policy in greedy thompson-beta kl-ucb exp3; do
  printf '%-22s -> %s\n' "$policy" "$("$BIN" bandit "$policy" --seed 42 --select 1 rewards)" >&2
done

# Selection streams survive restarts: save state mid-stream, resume, and
# the next pick matches an uninterrupted run.
"$BIN" bandit thompson-beta --seed 42 --select 1 --save-state out/ts.state.jsonl rewards > /dev/null
echo "resumed pick: $("$BIN" bandit thompson-beta --seed 42 --select 1 --load-state out/ts.state.jsonl rewards)" >&2

echo "done; scores in $(pwd)/out/" >&2

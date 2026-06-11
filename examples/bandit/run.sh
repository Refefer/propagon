#!/usr/bin/env bash
# Replays a reward log through every bandit policy.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
BIN="${PROPAGON_BIN:-../../target/release/propagon}"
mkdir -p out

for policy in greedy epsilon-greedy upper-confidence-bound kl-ucb thompson-beta thompson-gaussian exp3; do
  echo "== $policy" >&2
  "$BIN" bandit "$policy" rewards > "out/$policy.scores"
done

echo "== next arm to play, per policy" >&2
for policy in greedy thompson-beta kl-ucb exp3; do
  printf '%-22s -> %s\n' "$policy" "$("$BIN" bandit "$policy" --seed 42 --select 1 rewards)" >&2
done

# Selection streams survive restarts: save state mid-stream, resume, and
# the next pick matches an uninterrupted run.
"$BIN" bandit thompson-beta --seed 42 --select 1 --save-state out/ts.state.jsonl rewards > /dev/null
echo "resumed pick: $("$BIN" bandit thompson-beta --seed 42 --select 1 --load-state out/ts.state.jsonl rewards)" >&2

echo "done; scores in $(pwd)/out/" >&2

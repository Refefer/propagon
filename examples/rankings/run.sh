#!/usr/bin/env bash
# Aggregates a season of race finishing orders four ways.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
BIN="${PROPAGON_BIN:-../../target/release/propagon}"
DATA=f1-2024.rankings
mkdir -p out

run() { # run <name> <algo...>
  local name="$1"; shift
  echo "== $name" >&2
  "$BIN" rankings "$@" "$DATA" > "out/$name.scores"
}

run plackett-luce plackett-luce
run mc4           markov-chain
run borda-count   borda-count
run kemeny        kemeny --passes 5

echo "== plackett-luce podium" >&2
head -5 out/plackett-luce.scores >&2
echo "done; leaderboards in $(pwd)/out/" >&2

#!/usr/bin/env bash
# Aggregates a season of race finishing orders four ways.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
# Binary location: $PROPAGON_BIN, else cargo's target dir (honors any
# target-dir override), else the conventional workspace path.
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4 || true)"
BIN="${PROPAGON_BIN:-${TARGET_DIR:-../../target}/release/propagon}"
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
run footrule         footrule
run ilsr             i-luce-spectral-ranking

# Mallows needs every ballot to rank the identical item set; F1 seasons
# have DNFs, so it correctly refuses this data (shown here on purpose).
echo "== mallows (expected to refuse partial ballots)" >&2
"$BIN" rankings mallows "$DATA" > out/mallows.scores 2> out/mallows.err || true

echo "== plackett-luce podium" >&2
head -5 out/plackett-luce.scores >&2
echo "done; leaderboards in $(pwd)/out/" >&2

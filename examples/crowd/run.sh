#!/usr/bin/env bash
# Crowd-BT demo: joint ranking + annotator-reliability estimation.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
# Binary location: $PROPAGON_BIN, else cargo's target dir (honors any
# target-dir override), else the conventional workspace path.
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4 || true)"
BIN="${PROPAGON_BIN:-${TARGET_DIR:-../../target}/release/propagon}"
mkdir -p out

echo "== crowd bradley-terry (items, then annotator reliabilities)" >&2
"$BIN" crowd bradley-terry votes | tee out/crowd-bt.scores

# Baseline: what plain BT sees when the annotator column is ignored
# (rewritten into the tournament games format: winner TAB loser TAB 1).
awk -v OFS='\t' '{print $2, $3, 1}' votes > out/votes.pairs
"$BIN" tournament bradley-terry-model out/votes.pairs > out/naive.scores
echo "== naive bradley-terry written to out/naive.scores (compare orders)" >&2

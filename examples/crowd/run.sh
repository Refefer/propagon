#!/usr/bin/env bash
# Crowd-BT demo: joint ranking + annotator-reliability estimation.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
BIN="${PROPAGON_BIN:-../../target/release/propagon}"
mkdir -p out

echo "== crowd bradley-terry (items, then annotator reliabilities)" >&2
"$BIN" crowd bradley-terry votes | tee out/crowd-bt.scores

# Baseline: what plain BT sees when the annotator column is ignored.
awk '{print $2, $3}' votes > out/votes.pairs
"$BIN" tournament bradley-terry-model out/votes.pairs > out/naive.scores
echo "== naive bradley-terry written to out/naive.scores (compare orders)" >&2

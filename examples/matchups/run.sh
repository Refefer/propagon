#!/usr/bin/env bash
# Weng-Lin (OpenSkill) demos: free-for-all races and team matches.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
BIN="${PROPAGON_BIN:-../../target/release/propagon}"
mkdir -p out

echo "== f1 2024 as a 20-driver free-for-all per race" >&2
"$BIN" matchups weng-lin f1-2024.matchups > out/f1-bt.scores
"$BIN" matchups weng-lin --variant thurstone-mosteller f1-2024.matchups > out/f1-tm.scores
# In 20-way matches every race carries 19 pairwise updates, so sigma
# collapses fast; tau (openskill convention: 25/300) keeps ratings adaptive.
"$BIN" matchups weng-lin --tau 0.0833 f1-2024.matchups > out/f1-tau.scores
head -5 out/f1-bt.scores >&2

echo "== doubles league (teams, with ties)" >&2
"$BIN" matchups weng-lin doubles.matchups > out/doubles.scores
cat out/doubles.scores >&2

# Ratings are incremental: rate the first half of the season, save, then
# fold in the rest.
half=$(( $(wc -l < f1-2024.matchups) / 2 ))
head -n "$half" f1-2024.matchups > out/first-half.matchups
tail -n +"$((half + 1))" f1-2024.matchups > out/second-half.matchups
"$BIN" matchups weng-lin --save-state out/mid-season.jsonl out/first-half.matchups > /dev/null
"$BIN" matchups weng-lin --load-state out/mid-season.jsonl out/second-half.matchups > out/f1-resumed.scores
diff <(cat out/f1-bt.scores) <(cat out/f1-resumed.scores) > /dev/null \
  && echo "resumed season equals continuous season" >&2

echo "done; ratings in $(pwd)/out/" >&2

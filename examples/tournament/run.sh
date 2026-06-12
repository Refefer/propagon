#!/usr/bin/env bash
# Ranks baseball's 2018 season with every tournament algorithm.
#
# v2 reads team names directly (baseball.2018 holds quoted team names,
# tab-separated) — no remapping step. Outputs land in ./out/.
#
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
# Binary location: $PROPAGON_BIN, else cargo's target dir (honors any
# target-dir override), else the conventional workspace path.
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4 || true)"
BIN="${PROPAGON_BIN:-${TARGET_DIR:-../../target}/release/propagon}"
DATA=baseball.2018
mkdir -p out

run() { # run <name> <algo...>
  local name="$1"; shift
  echo "== $name" >&2
  "$BIN" tournament "$@" "$DATA" > "out/$name.scores"
}

run win-rate              win-rate --confidence-interval 0.95
run elo                   elo
run glicko2               glicko2
run bradley-terry-mm      bradley-terry-model --estimator mm
run bradley-terry-sgd     bradley-terry-model --estimator sgd
run luce-spectral-ranking luce-spectral-ranking --steps 20
run rank-centrality       rank-centrality
run random-utility-model  random-utility-model --passes 100
run kemeny                kemeny --passes 5
run borda-count           borda-count
run copeland              copeland
run colley                colley
run keener                keener
run bayes-bt              bayesian-bradley-terry --samples 1000
run thurstone-mosteller   thurstone-mosteller
run ilsr                  i-luce-spectral-ranking
run serial-rank           serial-rank
run random-walker         random-walker --bias 0.8
run whr                   whole-history-rating
run melo                  melo --k 1
run nash-averaging        nash-averaging --iterations 50000
run blade-chest           blade-chest --epochs 20
run gbt                   generalized-bradley-terry --tie-model none
run btm-bootstrap         bradley-terry-model --bootstrap 200

# Massey and offense-defense read the threshold as the margin of victory;
# the baseball file carries margin-1 wins (a valid if uninformative margin)
# just to demo the command shape.
echo "== massey" >&2
"$BIN" tournament massey "$DATA" > out/massey.scores
echo "== offense-defense" >&2
"$BIN" tournament offense-defense "$DATA" > out/offense-defense.scores

# HodgeRank also prints how *rankable* the season is (cyclic-flow share).
echo "== hodge-rank" >&2
"$BIN" tournament hodge-rank "$DATA" > out/hodge-rank.scores

# Resumable state demo: save glicko2 state, then continue from it.
"$BIN" tournament glicko2 --save-state out/glicko2.state.jsonl "$DATA" > /dev/null
"$BIN" tournament glicko2 --load-state out/glicko2.state.jsonl "$DATA" > out/glicko2-resumed.scores

echo "done; leaderboards in $(pwd)/out/" >&2

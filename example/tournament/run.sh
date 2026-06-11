#!/usr/bin/env bash
# Ranks baseball's 2018 season with every tournament algorithm.
#
# v2 reads team names directly (baseball.2018 holds quoted team names,
# tab-separated) — no remapping step. Outputs land in ./out/.
#
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
BIN="${PROPAGON_BIN:-../../target/release/propagon}"
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

# Resumable state demo: save glicko2 state, then continue from it.
"$BIN" tournament glicko2 --save-state out/glicko2.state.jsonl "$DATA" > /dev/null
"$BIN" tournament glicko2 --load-state out/glicko2.state.jsonl "$DATA" > out/glicko2-resumed.scores

echo "done; leaderboards in $(pwd)/out/" >&2

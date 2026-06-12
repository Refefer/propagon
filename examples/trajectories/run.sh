#!/usr/bin/env bash
# Exercises every trajectories algorithm against the sessions data.
set -euo pipefail
cd "$(dirname "$0")"

BIN="${PROPAGON_BIN:-$(cargo metadata --format-version 1 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/propagon}"

DATA=sessions
mkdir -p out

run() {
    local name="$1"; shift
    echo "== $name"
    "$BIN" trajectories "$@" "$DATA" > "out/$name.scores"
}

run monte-carlo monte-carlo --gamma 0.95
run monte-carlo-median monte-carlo --aggregate median --winsorize 0.05
run td td --alpha 0.1 --passes 25
run compare compare --replicates 2000 --pairwise 500
run behavior-cloning behavior-cloning

echo "done; leaderboards in $(pwd)/out/"

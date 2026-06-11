#!/usr/bin/env bash
# Node-importance demos on the Wikipedia article-link graph.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
# Binary location: $PROPAGON_BIN, else cargo's target dir (honors any
# target-dir override), else the conventional workspace path.
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4 || true)"
BIN="${PROPAGON_BIN:-${TARGET_DIR:-../../target}/release/propagon}"
DATA=articles
mkdir -p out

run() { # run <name> <algo...>
  local name="$1"; shift
  echo "== $name" >&2
  "$BIN" graph "$@" "$DATA" > "out/$name.scores"
}

run page-rank        page-rank --iterations 30
run page-rank-all    page-rank --iterations 30 --sink-dispersion all
run hits             hits
run katz             katz-centrality --alpha 0.05
run degree-in        degree
run degree-total     degree --direction total
run k-core           k-core
run birank           birank

# Split the graph into connected components (writes articles.0, ...).
"$BIN" graph components --min-graph-size 5 "$DATA"
mv -f "$DATA".[0-9]* out/ 2>/dev/null || true

echo "done; leaderboards in $(pwd)/out/" >&2

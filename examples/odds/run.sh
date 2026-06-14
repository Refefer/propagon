#!/usr/bin/env bash
# Betting & portfolio pipeline (§14): odds -> fair probabilities -> consolidate
# -> size, plus an LMSR market and calibration diagnostics.
# Build the binary first:  cargo build --release
set -euo pipefail

cd "$(dirname "$0")"
# Binary location: $PROPAGON_BIN, else cargo's target dir, else the workspace path.
TARGET_DIR="$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
  | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4 || true)"
BIN="${PROPAGON_BIN:-${TARGET_DIR:-../../target}/release/propagon}"
mkdir -p out

# §14.1 de-vig the book lines four ways.
for method in multiplicative power shin; do
  echo "== devig $method" >&2
  "$BIN" odds devig --method "$method" events.txt > "out/devig-$method.scores"
done

# §14.2 consolidate several books/models into one consensus.
echo "== opinion-pool (log)" >&2
"$BIN" odds opinion-pool --kind log forecasts.txt > out/opinion-pool.scores

# §14.3 LMSR market prices from a trade stream.
echo "== lmsr" >&2
"$BIN" odds lmsr --liquidity 100 trades.txt > out/lmsr.scores

# §14.4 size stakes with half-Kelly.
echo "== kelly (half)" >&2
"$BIN" odds kelly --fraction 0.5 picks.txt > out/kelly.stakes

# §14.5 calibration table + Brier/log-loss.
echo "== calibrate" >&2
"$BIN" odds calibrate --buckets 10 calibration.txt > out/calibration.table

echo "wrote out/*.scores, out/kelly.stakes, out/calibration.table" >&2

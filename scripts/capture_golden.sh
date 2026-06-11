#!/usr/bin/env bash
# Captures v1 propagon outputs on the example tournament data as golden files
# for the v2 parity test suite (crates/propagon-cli/tests/golden.rs).
#
# Usage: scripts/capture_golden.sh <path-to-v1-binary>
#
# Run ONCE against the last v1 build (commit 43dfe34 working tree); the golden
# files are committed and the v1 binary is not needed afterwards.
#
# Parity tiers consumed by the harness:
#   tier T (numeric tolerance + identical ranking): rate, glicko2, btm-mm,
#          btm-lr, page-rank, kemeny (insertion)
#   tier S (rank correlation only; RNG streams differ between v1/v2):
#          es-rum, lsr, birank
set -euo pipefail

V1_BIN="${1:?usage: capture_golden.sh <v1-binary>}"
EDGES="examples/tournament/baseball.2018.edges"
OUT="crates/propagon-cli/tests/golden"

run() { # run <outfile> <args...>
  local out="$1"; shift
  echo "capturing $out: $V1_BIN $EDGES $*" >&2
  "$V1_BIN" "$EDGES" "$@" > "$OUT/$out"
}

run rate-095.out    rate
run rate-090.out    rate --confidence-interval 0.9
run glicko2.out     glicko2
run glicko2-mu.out  glicko2 --use-mu
run btm-mm.out      btm-mm
run btm-lr.out      btm-lr
run es-rum.out      es-rum --passes 100
run kemeny.out      kemeny --passes 5
run page-rank.out   page-rank
run birank.out      birank
# lsr's v1 working tree prints debug lines; keep only "id: value" rows.
"$V1_BIN" "$EDGES" lsr --steps 20 | grep -E '^[0-9]+: ' > "$OUT/lsr.out"

echo "done; files in $OUT" >&2

# Betting & portfolio (§14)

The gambler's pipeline: a posted price is a crowd's revealed valuation. Strip
the bookmaker's margin to get a fair probability, consolidate several sources,
run a market, size a stake, and check calibration.

Run everything:

```bash
cargo build --release
./run.sh        # writes out/*
```

## Data files

- **`events.txt`** — three betting markets, one blank-line-separated event per
  block, each line `outcome  decimal-odds`. Outcome names are unique across the
  file.
- **`forecasts.txt`** — three sources' probabilities over the same election
  market, `source outcome probability` (each source sums to 1).
- **`trades.txt`** — an LMSR market's trade stream, `outcome shares`.
- **`picks.txt`** — model edges to size, `outcome win-probability decimal-odds`.
- **`calibration.txt`** — 100 past forecasts vs outcomes (`implied-probability 0|1`),
  constructed to show the favorite-longshot bias.

## What each step shows

- **`devig`** — `out/devig-*.scores` are fair probabilities per outcome (summing
  to 1 within each event). Compare `multiplicative` (margin spread
  proportionally) against `power`/`shin` (favorite-longshot bias removed); Shin
  also recovers an insider-share diagnostic.
- **`opinion-pool`** — the logarithmic pool (geometric mean of odds) is a sharper
  consensus than averaging probabilities.
- **`lmsr`** — the market prices *are* the crowd's consensus probability, summing
  to 1; deeper liquidity (`--liquidity`) moves the price less per trade.
- **`kelly`** — the growth-optimal stake fraction per pick (half-Kelly here);
  picks with no edge get 0.
- **`calibrate`** — a reliability table (predicted vs realized per bin, empty
  bins shown as `-`) plus the Brier score and log-loss in the header. The sample
  data shows the favorite-longshot bias: realized frequency falls *below* the
  implied probability for longshots and *above* it for favorites.

Things to try: raise `--liquidity` and watch LMSR prices move less; switch the
pool to `--kind linear` and see the consensus pulled toward the middle; add
`--extremize 1.5` to sharpen it.

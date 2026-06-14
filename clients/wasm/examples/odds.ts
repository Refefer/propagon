// Betting & portfolio (§14): de-vig odds, consolidate forecasts, run an LMSR
// market, and size a Kelly stake.
//   node examples/odds.ts
import { datasets, betting } from "../index.js";

// §14.1 — strip the bookmaker's margin out of posted odds.
const market = new datasets.OddsDataset();
market.pushEvent([
  ["home", 4.2],
  ["draw", 3.7],
  ["away", 1.95],
]);
const devig = betting.fitOddsDevig({ method: "shin" }, market);
console.log("Fair probabilities (Shin):", devig.sortedScores());
console.log("  insider share z =", devig.insiderShare(0));

// §14.2 — consolidate several books into one consensus.
const forecasts = new datasets.ForecastDataset();
forecasts.pushSource("pinnacle", 1, [["home", 0.52], ["away", 0.48]]);
forecasts.pushSource("betfair", 1, [["home", 0.49], ["away", 0.51]]);
const pool = betting.fitOpinionPool({ kind: "logarithmic" }, forecasts);
console.log("Consensus (log pool):", pool.sortedScores());

// §14.3 — an LMSR prediction market; prices are the consensus probability.
const trades = new datasets.MarketDataset();
trades.pushTrade("yes", 100);
trades.pushTrade("no", 20);
const lmsr = betting.fitLmsr({ b: 100 }, trades);
console.log("Market prices:", lmsr.sortedScores(), "cost", lmsr.cost());

// §14.4 — turn a probability + odds into a stake (b = decimal odds - 1).
console.log("Kelly stake p=0.58 @ 2.10:", betting.kellyFraction(0.58, 1.1));
console.log("  half-Kelly:", betting.fractionalKelly(0.58, 1.1, 0.5));

// §14.5 — score past forecasts.
console.log("Brier:", betting.brierScore([0.9, 0.2, 0.7], [true, false, true]));

"""Betting & portfolio (§14): de-vig odds into fair probabilities, consolidate
several forecasts, run an LMSR market, and size a stake with Kelly.

Run after `maturin develop`:  python examples/odds.py
"""

import propagon


def main() -> None:
    # §14.1 — strip the bookmaker's margin out of posted odds.
    market = propagon.OddsDataset()
    market.push_event([("home", 4.20), ("draw", 3.70), ("away", 1.95)])
    devig = propagon.OddsDevig(method="shin").fit(market)
    print("Fair probabilities (Shin):", devig.sorted_scores())
    print("  recovered insider share z =", devig.insider_share(0))

    # §14.2 — consolidate several books/models into one consensus.
    forecasts = propagon.ForecastDataset()
    forecasts.push_source("pinnacle", [("home", 0.52), ("away", 0.48)])
    forecasts.push_source("betfair", [("home", 0.49), ("away", 0.51)])
    forecasts.push_source("model", [("home", 0.55), ("away", 0.45)])
    pool = propagon.OpinionPool(kind="logarithmic").fit(forecasts)
    print("Consensus (log pool):", pool.sorted_scores())

    # §14.3 — an LMSR prediction market; prices are the crowd's consensus.
    trades = propagon.MarketDataset()
    for outcome, shares in [("yes", 100.0), ("no", 20.0), ("yes", 15.0)]:
        trades.push_trade(outcome, shares)
    lmsr = propagon.Lmsr(b=100.0).fit(trades)
    print("Market prices:", lmsr.sorted_scores(), "cost", round(lmsr.cost(), 4))

    # §14.4 — turn a probability + odds into a stake.
    p, decimal_odds = 0.58, 2.10
    b = decimal_odds - 1.0
    print(f"Kelly stake for p={p} @ {decimal_odds}:", propagon.kelly_fraction(p, b))
    print("  half-Kelly:", propagon.fractional_kelly(p, b, 0.5))

    # §14.5 — score how good past forecasts were.
    preds = [0.9, 0.2, 0.7, 0.4]
    outcomes = [True, False, True, False]
    print("Brier:", round(propagon.brier_score(preds, outcomes), 4))
    print("Log-loss:", round(propagon.log_loss(preds, outcomes), 4))


if __name__ == "__main__":
    main()

//! Prediction-market trades: a stream of `(outcome, shares)` orders
//! (`docs/algorithms.md` §14.3).
//!
//! The input to the [`Lmsr`](crate::algos::Lmsr) market-scoring-rule ranker:
//! each trade buys (positive) or sells (negative) shares of an outcome, moving
//! that outcome's price. Outcomes can be pre-declared so the price vector spans
//! the full universe even before every outcome has traded.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// A stream of `(outcome, shares)` trades over an outcome universe.
#[derive(Clone, Debug, Default)]
pub struct MarketDataset {
    interner: Interner,
    outcomes: Vec<u32>,
    shares: Vec<f64>,
}

impl MarketDataset {
    /// An empty market with an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a trade: `shares` of `outcome` (negative sells). `shares` must be
    /// finite; `0.0` is allowed and simply declares the outcome.
    pub fn push_trade(&mut self, outcome: &str, shares: f64) -> Result<()> {
        if !shares.is_finite() {
            return Err(Error::InvalidInput(format!(
                "trade shares must be finite, got {shares} for {outcome:?}"
            )));
        }
        let id = self.interner.intern(outcome);
        self.outcomes.push(id);
        self.shares.push(shares);
        Ok(())
    }

    /// Adds `outcome` to the universe without trading on it (price seeded at the
    /// uniform level). Idempotent.
    pub fn declare_outcome(&mut self, outcome: &str) {
        self.interner.intern(outcome);
    }

    /// Number of trades.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Whether the market holds no trades.
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Number of distinct outcomes in the universe.
    pub fn n_entities(&self) -> usize {
        self.interner.len()
    }

    /// The interner backing outcome ids.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// All `(outcome id, shares)` trades in insertion order.
    pub fn rows(&self) -> impl Iterator<Item = (u32, f64)> + '_ {
        self.outcomes
            .iter()
            .zip(&self.shares)
            .map(|(&o, &s)| (o, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_iterate() {
        let mut d = MarketDataset::new();
        d.push_trade("yes", 10.0).unwrap();
        d.push_trade("no", -2.0).unwrap();
        d.declare_outcome("maybe");
        assert_eq!(d.len(), 2);
        assert_eq!(d.n_entities(), 3);
        assert_eq!(d.rows().collect::<Vec<_>>(), vec![(0, 10.0), (1, -2.0)]);
    }

    #[test]
    fn rejects_non_finite() {
        let mut d = MarketDataset::new();
        assert!(d.push_trade("x", f64::NAN).is_err());
    }
}

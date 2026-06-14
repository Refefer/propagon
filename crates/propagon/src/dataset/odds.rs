//! Posted betting odds grouped by event (`docs/algorithms.md` §14.1).
//!
//! Each event is a set of mutually exclusive outcomes carrying **decimal** odds
//! (`o > 1`, where a unit stake returns `o` on a win). De-vigging
//! ([`OddsDevig`](crate::algos::OddsDevig)) strips the bookmaker's margin per
//! event to recover fair probabilities. CSR storage: `outcomes`/`odds` hold
//! every outcome concatenated; `event_offsets` cuts them into events.
//!
//! Outcome names are the ranked entities, so they must be unique across the
//! whole dataset — qualify them per event (e.g. `"race1:Alpha"`). A repeat,
//! within or across events, is rejected.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Decimal odds per outcome, grouped into events.
#[derive(Clone, Debug)]
pub struct OddsDataset {
    interner: Interner,
    outcomes: Vec<u32>,
    odds: Vec<f64>,
    event_offsets: Vec<usize>,
}

impl Default for OddsDataset {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            outcomes: Vec::new(),
            odds: Vec::new(),
            event_offsets: vec![0],
        }
    }
}

impl OddsDataset {
    /// An empty dataset with an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one event from `(outcome, decimal_odds)` pairs.
    ///
    /// Requires ≥ 2 outcomes, every `decimal_odds > 1.0` and finite, and every
    /// outcome name unseen elsewhere in the dataset. Nothing is mutated when the
    /// event is rejected.
    pub fn push_event(&mut self, outcomes: &[(&str, f64)]) -> Result<()> {
        if outcomes.len() < 2 {
            return Err(Error::InvalidInput(
                "an event needs at least two outcomes".into(),
            ));
        }
        // Validate fully before interning so a rejected event leaves no trace.
        let mut local = std::collections::HashSet::new();
        for &(name, o) in outcomes {
            if !o.is_finite() || o <= 1.0 {
                return Err(Error::InvalidInput(format!(
                    "decimal odds must be > 1.0, got {o} for {name:?}"
                )));
            }
            if self.interner.get(name).is_some() || !local.insert(name) {
                return Err(Error::InvalidInput(format!(
                    "duplicate outcome {name:?}; outcome names must be unique across the dataset"
                )));
            }
        }
        for &(name, o) in outcomes {
            self.outcomes.push(self.interner.intern(name));
            self.odds.push(o);
        }
        self.event_offsets.push(self.outcomes.len());
        Ok(())
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.event_offsets.len() - 1
    }

    /// Whether the dataset holds no events.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of distinct outcomes (the size of the entity space).
    pub fn n_entities(&self) -> usize {
        self.interner.len()
    }

    /// The interner backing this dataset's outcome ids.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Event `e` as `(outcome ids, decimal odds)`.
    pub fn event(&self, e: usize) -> (&[u32], &[f64]) {
        let r = self.event_offsets[e]..self.event_offsets[e + 1];
        (&self.outcomes[r.clone()], &self.odds[r])
    }

    /// All events in insertion order.
    pub fn events(&self) -> impl Iterator<Item = (&[u32], &[f64])> {
        (0..self.len()).map(|e| self.event(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_iterate() {
        let mut d = OddsDataset::new();
        d.push_event(&[("home", 2.0), ("draw", 3.5), ("away", 4.0)])
            .unwrap();
        d.push_event(&[("a", 1.5), ("b", 2.5)]).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.n_entities(), 5);

        let (ids, odds) = d.event(0);
        assert_eq!(ids.len(), 3);
        assert_eq!(odds, &[2.0, 3.5, 4.0]);
    }

    #[test]
    fn validation() {
        let mut d = OddsDataset::new();
        assert!(d.push_event(&[("solo", 2.0)]).is_err()); // < 2 outcomes
        assert!(d.push_event(&[("a", 1.0), ("b", 2.0)]).is_err()); // odds <= 1
        assert!(d.push_event(&[("a", 2.0), ("a", 3.0)]).is_err()); // dup in event
        d.push_event(&[("x", 2.0), ("y", 2.0)]).unwrap();
        assert!(d.push_event(&[("x", 2.0), ("z", 2.0)]).is_err()); // dup across events
        assert_eq!(d.len(), 1);
    }
}

//! Probability forecasts from several sources over a common outcome space
//! (`docs/algorithms.md` §14.2).
//!
//! Each source (a sportsbook's de-vigged line, a model, an expert) contributes
//! a probability vector over the shared outcomes; the
//! [`OpinionPool`](crate::algos::OpinionPool) consolidates them. CSR storage:
//! `out_ids`/`probs` hold every source's vector concatenated; `source_offsets`
//! cuts them into sources.
//!
//! Each source's probabilities must be non-negative and sum to 1; a source may
//! quote a subset of the outcomes (the pool's missing-coverage policy decides
//! what that means). Source names must be unique.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Per-source probability vectors over a common outcome space, with weights.
#[derive(Clone, Debug)]
pub struct ForecastDataset {
    interner: Interner,
    source_names: Interner,
    source_weights: Vec<f64>,
    out_ids: Vec<u32>,
    probs: Vec<f64>,
    source_offsets: Vec<usize>,
}

impl Default for ForecastDataset {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            source_names: Interner::new(),
            source_weights: Vec::new(),
            out_ids: Vec::new(),
            probs: Vec::new(),
            source_offsets: vec![0],
        }
    }
}

impl ForecastDataset {
    /// An empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a source's forecast (weight 1.0) from `(outcome, probability)`
    /// pairs.
    pub fn push_source(&mut self, name: &str, forecast: &[(&str, f64)]) -> Result<()> {
        self.push_source_weighted(name, 1.0, forecast)
    }

    /// Appends a weighted source's forecast.
    ///
    /// Requires `weight > 0`, ≥ 2 outcomes, each probability finite and in
    /// `[0, 1]`, the probabilities summing to 1 (±1e-9), outcomes unique within
    /// the source, and an unused source `name`. Nothing is mutated on rejection.
    pub fn push_source_weighted(
        &mut self,
        name: &str,
        weight: f64,
        forecast: &[(&str, f64)],
    ) -> Result<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "source weight must be positive, got {weight}"
            )));
        }
        if self.source_names.get(name).is_some() {
            return Err(Error::InvalidInput(format!("duplicate source {name:?}")));
        }
        if forecast.len() < 2 {
            return Err(Error::InvalidInput(
                "a forecast needs at least two outcomes".into(),
            ));
        }
        let mut local = std::collections::HashSet::new();
        let mut sum = 0.0;
        for &(outcome, p) in forecast {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(Error::InvalidInput(format!(
                    "forecast probability must be in [0, 1], got {p} for {outcome:?}"
                )));
            }
            if !local.insert(outcome) {
                return Err(Error::InvalidInput(format!(
                    "duplicate outcome {outcome:?} in source {name:?}"
                )));
            }
            sum += p;
        }
        if (sum - 1.0).abs() > 1e-9 {
            return Err(Error::InvalidInput(format!(
                "source {name:?} probabilities sum to {sum}, not 1"
            )));
        }

        self.source_names.intern(name);
        self.source_weights.push(weight);
        for &(outcome, p) in forecast {
            self.out_ids.push(self.interner.intern(outcome));
            self.probs.push(p);
        }
        self.source_offsets.push(self.out_ids.len());
        Ok(())
    }

    /// Number of sources.
    pub fn len(&self) -> usize {
        self.source_offsets.len() - 1
    }

    /// Whether the dataset holds no sources.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of distinct outcomes (the entity space).
    pub fn n_outcomes(&self) -> usize {
        self.interner.len()
    }

    /// The interner backing outcome ids.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Source `i` as `(weight, outcome ids, probabilities)`.
    pub fn source(&self, i: usize) -> (f64, &[u32], &[f64]) {
        let r = self.source_offsets[i]..self.source_offsets[i + 1];
        (
            self.source_weights[i],
            &self.out_ids[r.clone()],
            &self.probs[r],
        )
    }

    /// All sources in insertion order.
    pub fn sources(&self) -> impl Iterator<Item = (f64, &[u32], &[f64])> {
        (0..self.len()).map(|i| self.source(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_iterate() {
        let mut d = ForecastDataset::new();
        d.push_source("book1", &[("home", 0.6), ("away", 0.4)])
            .unwrap();
        d.push_source_weighted("book2", 2.0, &[("home", 0.5), ("away", 0.5)])
            .unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.n_outcomes(), 2);
        let (w, ids, probs) = d.source(1);
        assert_eq!(w, 2.0);
        assert_eq!(ids.len(), 2);
        assert_eq!(probs, &[0.5, 0.5]);
    }

    #[test]
    fn validation() {
        let mut d = ForecastDataset::new();
        assert!(d.push_source("s", &[("a", 1.0)]).is_err()); // < 2 outcomes
        assert!(d.push_source("s", &[("a", 0.7), ("b", 0.7)]).is_err()); // sum != 1
        assert!(d.push_source("s", &[("a", 0.5), ("a", 0.5)]).is_err()); // dup outcome
        d.push_source("s", &[("a", 0.5), ("b", 0.5)]).unwrap();
        assert!(d.push_source("s", &[("a", 0.5), ("b", 0.5)]).is_err()); // dup source
        assert_eq!(d.len(), 1);
    }
}

//! Contextual reward observations: `(arm, reward, context)` events for
//! contextual bandits such as LinUCB (`docs/algorithms.md` §8.1).
//!
//! Columnar like the other datasets; the feature column is stored flat with
//! stride `dim`. The dimensionality is fixed by the first pushed row — every
//! later row must match it, and feature values must be finite.
//!
//! `dim()` is `None` until the first row arrives: an empty dataset genuinely
//! has no dimensionality (it is missing, not defaulted).

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Columnar `(arm, reward, context)` event log.
#[derive(Clone, Debug, Default)]
pub struct ContextualRewardsDataset {
    interner: Interner,
    arms: Vec<u32>,
    rewards: Vec<f32>,
    /// Flat feature matrix, row-major with stride `dim`.
    features: Vec<f64>,
    /// Feature dimensionality; `None` until the first row fixes it.
    dim: Option<usize>,
}

impl ContextualRewardsDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one observed reward for `arm` in context `x`. The first push
    /// fixes the feature dimensionality; later pushes must match it, must be
    /// non-empty, and must be finite throughout.
    pub fn push(&mut self, arm: &str, reward: f32, x: &[f64]) -> Result<()> {
        self.validate_features(x)?;
        let a = self.interner.intern(arm);
        self.push_validated(a, reward, x);
        Ok(())
    }

    /// Records a reward for an already-interned arm id (bulk/load path).
    pub(crate) fn push_ids(&mut self, arm: u32, reward: f32, x: &[f64]) -> Result<()> {
        if arm >= self.interner.len() as u32 {
            return Err(Error::InvalidInput(format!(
                "arm id {arm} out of range ({} interned)",
                self.interner.len()
            )));
        }
        self.validate_features(x)?;
        self.push_validated(arm, reward, x);
        Ok(())
    }

    /// Shared validation: non-empty, finite, dimensionality-consistent.
    fn validate_features(&self, x: &[f64]) -> Result<()> {
        if x.is_empty() {
            return Err(Error::InvalidInput("context vector is empty".into()));
        }
        if let Some(bad) = x.iter().find(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "context vector contains a non-finite value: {bad}"
            )));
        }
        match self.dim {
            Some(d) if d != x.len() => Err(Error::InvalidInput(format!(
                "context dimension {} does not match the dataset's {d}",
                x.len()
            ))),
            _ => Ok(()),
        }
    }

    fn push_validated(&mut self, arm: u32, reward: f32, x: &[f64]) {
        self.dim = Some(x.len());
        self.arms.push(arm);
        self.rewards.push(reward);
        self.features.extend_from_slice(x);
    }

    /// Same interner (so the same arm universe) with no rows — the seed for
    /// resampled copies. `dim` starts unknown again and is re-fixed by the
    /// first resampled row, preserving "dim known ⟺ rows exist".
    pub(crate) fn empty_like(&self) -> Self {
        Self {
            interner: self.interner.clone(),
            ..Self::default()
        }
    }

    /// Appends one row copied verbatim from a dataset sharing this interner
    /// (the resample path); the source already validated the arm id,
    /// finiteness, and dimensionality, so re-validation is skipped.
    pub(crate) fn push_row_unchecked(&mut self, arm: u32, reward: f32, x: &[f64]) {
        self.push_validated(arm, reward, x);
    }

    /// Feature dimensionality, fixed by the first row (`None` while empty).
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.arms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }

    pub fn n_arms(&self) -> usize {
        self.interner.len()
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Interns an arm name ahead of bulk ingestion.
    pub fn intern(&mut self, name: &str) -> u32 {
        self.interner.intern(name)
    }

    /// All `(arm, reward, context)` rows in insertion order.
    pub fn rows(&self) -> impl Iterator<Item = (u32, f32, &[f64])> + '_ {
        // Stride 1 is never observed: `dim` is `None` only while no rows
        // exist, and then the zipped iterators are already empty.
        let stride = self.dim.unwrap_or(1);
        self.arms
            .iter()
            .zip(&self.rewards)
            .zip(self.features.chunks(stride))
            .map(|((&a, &r), x)| (a, r, x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_fixes_dim_and_iterates() {
        let mut d = ContextualRewardsDataset::new();
        assert_eq!(d.dim(), None);
        assert_eq!(d.rows().count(), 0);

        d.push("A", 1.0, &[0.5, 1.5]).unwrap();
        d.push("B", 0.0, &[1.0, -2.0]).unwrap();
        d.push("A", 0.5, &[0.0, 0.25]).unwrap();
        assert_eq!(d.dim(), Some(2));
        assert_eq!(d.n_arms(), 2);
        assert_eq!(d.len(), 3);

        let rows: Vec<(u32, f32, Vec<f64>)> =
            d.rows().map(|(a, r, x)| (a, r, x.to_vec())).collect();
        assert_eq!(
            rows,
            vec![
                (0, 1.0, vec![0.5, 1.5]),
                (1, 0.0, vec![1.0, -2.0]),
                (0, 0.5, vec![0.0, 0.25]),
            ]
        );
    }

    #[test]
    fn invalid_rows_are_rejected() {
        let mut d = ContextualRewardsDataset::new();
        assert!(matches!(d.push("A", 1.0, &[]), Err(Error::InvalidInput(_))));
        assert!(matches!(
            d.push("A", 1.0, &[f64::NAN]),
            Err(Error::InvalidInput(_))
        ));

        d.push("A", 1.0, &[1.0, 2.0]).unwrap();
        assert!(matches!(
            d.push("B", 1.0, &[1.0]),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            d.push("B", 1.0, &[1.0, f64::INFINITY]),
            Err(Error::InvalidInput(_))
        ));
        assert!(d.push_ids(7, 1.0, &[1.0, 2.0]).is_err());
        // Failed pushes leave no partial row behind.
        assert_eq!(d.len(), 1);
        assert_eq!(d.rows().count(), 1);
    }
}

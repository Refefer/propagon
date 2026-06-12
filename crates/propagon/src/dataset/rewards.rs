//! Scalar-reward observations: `(arm, reward)` events for bandits (FR-8).
//!
//! Also the natural landing format for mcrl-rs `(entity, return)` exports:
//! each per-state return sample is one row.

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Columnar `(arm, reward)` event log.
#[derive(Clone, Debug, Default)]
pub struct RewardsDataset {
    interner: Interner,
    arms: Vec<u32>,
    rewards: Vec<f32>,
}

impl RewardsDataset {
    /// An empty event log with an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one observed reward for `arm`.
    pub fn push(&mut self, arm: &str, reward: f32) {
        let a = self.interner.intern(arm);
        self.arms.push(a);
        self.rewards.push(reward);
    }

    /// Records a reward for an already-interned arm id.
    pub fn push_ids(&mut self, arm: u32, reward: f32) -> Result<()> {
        if arm >= self.interner.len() as u32 {
            return Err(Error::InvalidInput(format!(
                "arm id {arm} out of range ({} interned)",
                self.interner.len()
            )));
        }
        self.arms.push(arm);
        self.rewards.push(reward);
        Ok(())
    }

    /// Same interner (so the same arm universe) with no rows — the seed for
    /// resampled copies.
    pub(crate) fn empty_like(&self) -> Self {
        Self {
            interner: self.interner.clone(),
            ..Self::default()
        }
    }

    /// Appends one row without the [`RewardsDataset::push_ids`] range check;
    /// only resampling may use it, where the ids come from `rows()` of a
    /// dataset sharing this interner and so cannot be out of range.
    pub(crate) fn push_row_unchecked(&mut self, arm: u32, reward: f32) {
        self.arms.push(arm);
        self.rewards.push(reward);
    }

    /// Number of reward events.
    pub fn len(&self) -> usize {
        self.arms.len()
    }

    /// Whether the log holds no events.
    pub fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }

    /// Number of distinct arms seen by the interner.
    pub fn n_arms(&self) -> usize {
        self.interner.len()
    }

    /// The interner backing this log's arm ids.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Interns an arm name ahead of `push_ids` ingestion.
    pub fn intern(&mut self, name: &str) -> u32 {
        self.interner.intern(name)
    }

    /// All `(arm, reward)` rows in insertion order.
    pub fn rows(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.arms.iter().zip(&self.rewards).map(|(&a, &r)| (a, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_iterate() {
        let mut d = RewardsDataset::new();
        d.push("A", 1.0);
        d.push("B", 0.0);
        d.push("A", 0.5);
        assert_eq!(d.n_arms(), 2);
        assert_eq!(
            d.rows().collect::<Vec<_>>(),
            vec![(0, 1.0), (1, 0.0), (0, 0.5)]
        );
        assert!(d.push_ids(5, 1.0).is_err());
    }
}

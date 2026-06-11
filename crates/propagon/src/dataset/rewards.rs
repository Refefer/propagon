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

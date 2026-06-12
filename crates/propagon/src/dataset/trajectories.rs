//! Reward-bearing trajectories: CSR-packed episodes of `(state, reward)`
//! steps — the input for the §13 value-estimation family (Monte Carlo value,
//! TD(0), value comparison, behavior cloning).
//!
//! A step `(s, r)` means "being in `s` yielded reward `r` at that step"; an
//! episode is an ordered sequence of steps; the final step's reward is the
//! terminal payoff. Episode `i` is `states[offsets[i]..offsets[i+1]]` zipped
//! with the parallel `rewards` column (the [`RankingsDataset`] CSR layout).
//!
//! Rewards are validated finite at push time — NaN/inf would silently poison
//! every downstream return computation. Empty episodes collapse: calling
//! [`TrajectoriesDataset::end_episode`] twice in a row is a no-op, mirroring
//! `PairwiseDataset::new_period`. Steps pushed after the last `end_episode`
//! form a final open episode (also the pairwise convention).
//!
//! [`RankingsDataset`]: crate::dataset::RankingsDataset

use crate::error::{Error, Result};
use crate::interner::Interner;

/// Ragged list of episodes; each step pairs an interned state with the
/// reward observed at that step.
#[derive(Clone, Debug)]
pub struct TrajectoriesDataset {
    interner: Interner,
    states: Vec<u32>,
    rewards: Vec<f32>,
    /// Episode start offsets; always begins with 0, strictly increasing.
    episode_offsets: Vec<usize>,
}

impl Default for TrajectoriesDataset {
    fn default() -> Self {
        Self {
            interner: Interner::new(),
            states: Vec::new(),
            rewards: Vec::new(),
            episode_offsets: vec![0],
        }
    }
}

impl TrajectoriesDataset {
    /// An empty dataset with an empty interner and no episodes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one `(state, reward)` step to the current open episode,
    /// interning the state name. Non-finite rewards are rejected before any
    /// state mutates, so a failed push leaves no partial row behind.
    pub fn push_step(&mut self, state: &str, reward: f32) -> Result<()> {
        if !reward.is_finite() {
            return Err(Error::InvalidInput(format!(
                "step reward must be finite, got {reward}"
            )));
        }
        let id = self.interner.intern(state);
        self.states.push(id);
        self.rewards.push(reward);
        Ok(())
    }

    /// Appends a step for an already-interned state id (the io load path).
    pub(crate) fn push_step_ids(&mut self, state: u32, reward: f32) -> Result<()> {
        if state >= self.interner.len() as u32 {
            return Err(Error::InvalidInput(format!(
                "state id {state} out of range ({} interned)",
                self.interner.len()
            )));
        }
        if !reward.is_finite() {
            return Err(Error::InvalidInput(format!(
                "step reward must be finite, got {reward}"
            )));
        }
        self.states.push(state);
        self.rewards.push(reward);
        Ok(())
    }

    /// Same interner (so the same state universe) with no steps or episode
    /// boundaries — the seed for resampled copies.
    pub(crate) fn empty_like(&self) -> Self {
        Self {
            interner: self.interner.clone(),
            ..Self::default()
        }
    }

    /// Appends one whole episode copied verbatim from a dataset sharing
    /// this interner (the resample path) and closes it; the source already
    /// validated ids and reward finiteness, so re-validation is skipped.
    /// The parallel slices come from `episode()` and are equal-length by
    /// construction.
    pub(crate) fn push_episode_unchecked(&mut self, states: &[u32], rewards: &[f32]) {
        self.states.extend_from_slice(states);
        self.rewards.extend_from_slice(rewards);
        self.end_episode();
    }

    /// Closes the current episode at the current end of the data. Calling
    /// this with no steps since the previous boundary is a no-op (empty
    /// episodes collapse, mirroring `PairwiseDataset::new_period`).
    pub fn end_episode(&mut self) {
        let here = self.states.len();
        if self.episode_offsets.last() != Some(&here) {
            self.episode_offsets.push(here);
        }
    }

    /// Number of non-empty episodes. Steps pushed after the last
    /// [`TrajectoriesDataset::end_episode`] count as one final open episode.
    pub fn n_episodes(&self) -> usize {
        let closed = self.episode_offsets.len() - 1;
        let open_start = self.episode_offsets.last().copied().unwrap_or(0);
        closed + usize::from(self.states.len() > open_start)
    }

    /// Episode `i` as parallel `(states, rewards)` slices, in step order.
    /// An out-of-range `i` yields empty slices rather than panicking.
    pub fn episode(&self, i: usize) -> (&[u32], &[f32]) {
        let len = self.states.len();
        let start = self.episode_offsets.get(i).copied().unwrap_or(len);
        let end = self.episode_offsets.get(i + 1).copied().unwrap_or(len);
        (&self.states[start..end], &self.rewards[start..end])
    }

    /// All episodes in insertion order.
    pub fn episodes(&self) -> impl Iterator<Item = (&[u32], &[f32])> {
        (0..self.n_episodes()).map(|i| self.episode(i))
    }

    /// Total number of steps across all episodes.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the dataset holds no steps.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Number of distinct states seen by the interner.
    pub fn n_entities(&self) -> usize {
        self.interner.len()
    }

    /// The interner backing this dataset's state ids.
    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Interns a state name ahead of bulk ingestion (the io load path).
    pub(crate) fn intern(&mut self, name: &str) -> u32 {
        self.interner.intern(name)
    }

    /// All steps as `(state, reward)` in insertion order, ignoring episode
    /// boundaries (io serialization and frequency counting).
    pub(crate) fn steps(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.states.iter().zip(&self.rewards).map(|(&s, &r)| (s, r))
    }

    /// Interior episode boundaries for serialization: every offset after the
    /// leading 0, including a trailing boundary at `len()` when the last
    /// episode was explicitly ended.
    pub(crate) fn episode_starts_for_io(&self) -> Vec<usize> {
        self.episode_offsets[1..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds() -> TrajectoriesDataset {
        let mut d = TrajectoriesDataset::new();
        d.push_step("a", 1.0).unwrap();
        d.push_step("b", 0.5).unwrap();
        d.end_episode();
        d.push_step("b", 2.0).unwrap();
        d.end_episode();
        d
    }

    #[test]
    fn csr_packing_and_iteration() {
        let d = ds();
        assert_eq!(d.len(), 3);
        assert_eq!(d.n_episodes(), 2);
        assert_eq!(d.n_entities(), 2);
        assert_eq!(d.episode(0), (&[0u32, 1][..], &[1.0f32, 0.5][..]));
        assert_eq!(d.episode(1), (&[1u32][..], &[2.0f32][..]));

        let eps: Vec<_> = d.episodes().collect();
        assert_eq!(eps.len(), 2);
        assert_eq!(eps[1].1, &[2.0f32]);

        // Out-of-range access degrades to an empty episode, no panic.
        assert_eq!(d.episode(9), (&[][..], &[][..]));
    }

    #[test]
    fn empty_episodes_collapse() {
        let mut d = TrajectoriesDataset::new();
        d.end_episode();
        assert_eq!(d.n_episodes(), 0);
        assert!(d.is_empty());

        d.push_step("a", 1.0).unwrap();
        d.end_episode();
        d.end_episode();
        d.end_episode();
        assert_eq!(d.n_episodes(), 1);
    }

    #[test]
    fn trailing_steps_form_an_open_episode() {
        let mut d = ds();
        d.push_step("c", -1.0).unwrap();
        assert_eq!(d.n_episodes(), 3);
        assert_eq!(d.episode(2), (&[2u32][..], &[-1.0f32][..]));
    }

    #[test]
    fn non_finite_rewards_are_rejected() {
        let mut d = TrajectoriesDataset::new();
        assert!(matches!(
            d.push_step("a", f32::NAN),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            d.push_step("a", f32::INFINITY),
            Err(Error::InvalidInput(_))
        ));
        // Failed pushes leave no partial state behind.
        assert!(d.is_empty());
        assert_eq!(d.n_entities(), 0);

        d.push_step("a", 1.0).unwrap();
        assert!(d.push_step_ids(7, 1.0).is_err());
        assert!(d.push_step_ids(0, f32::NAN).is_err());
        assert_eq!(d.len(), 1);
    }
}

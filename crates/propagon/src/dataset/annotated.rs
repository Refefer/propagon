//! Annotator-tagged pairwise comparisons: `(annotator, winner, loser,
//! weight)` rows, columnar, with separate interners for entities and
//! annotators.
//!
//! The input shape for Crowd-BT (`docs/algorithms.md` §11.2): each vote
//! remembers who cast it, so annotator reliability can be estimated jointly
//! with the ranking.
//!
//! Invariant (relied on by `algos/`): every stored id was produced by the
//! owning interner — `push` interns both columns, so resolution is total.

use crate::interner::Interner;

/// Columnar `(annotator, winner, loser, weight)` votes.
#[derive(Clone, Debug, Default)]
pub struct AnnotatedPairsDataset {
    entities: Interner,
    annotators: Interner,
    annotator: Vec<u32>,
    winners: Vec<u32>,
    losers: Vec<u32>,
    weights: Vec<f32>,
}

impl AnnotatedPairsDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one vote: `annotator` judged `winner` better than `loser`.
    pub fn push(&mut self, annotator: &str, winner: &str, loser: &str, weight: f32) {
        self.annotator.push(self.annotators.intern(annotator));
        self.winners.push(self.entities.intern(winner));
        self.losers.push(self.entities.intern(loser));
        self.weights.push(weight);
    }

    /// Number of votes.
    pub fn len(&self) -> usize {
        self.winners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.winners.is_empty()
    }

    pub fn n_entities(&self) -> usize {
        self.entities.len()
    }

    pub fn n_annotators(&self) -> usize {
        self.annotators.len()
    }

    pub fn entities(&self) -> &Interner {
        &self.entities
    }

    pub fn annotators(&self) -> &Interner {
        &self.annotators
    }

    /// `(annotator, winner, loser, weight)` in insertion order.
    pub fn rows(&self) -> impl Iterator<Item = (u32, u32, u32, f32)> + '_ {
        (0..self.len()).map(|i| {
            (
                self.annotator[i],
                self.winners[i],
                self.losers[i],
                self.weights[i],
            )
        })
    }

    /// Used by dataset io to rebuild the columns; ids must come from the
    /// matching interners.
    pub(crate) fn push_ids(&mut self, annotator: u32, winner: u32, loser: u32, weight: f32) {
        self.annotator.push(annotator);
        self.winners.push(winner);
        self.losers.push(loser);
        self.weights.push(weight);
    }

    pub(crate) fn set_interners(&mut self, entities: Interner, annotators: Interner) {
        self.entities = entities;
        self.annotators = annotators;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_round_trip_is_byte_identical() {
        let mut d = AnnotatedPairsDataset::new();
        d.push("judge1", "a", "b", 1.0);
        d.push("judge2", "b", "a", 2.5);
        d.push("judge1", "a", "c", 1.0);

        let mut first = Vec::new();
        d.save_jsonl(&mut first).unwrap();
        let loaded = AnnotatedPairsDataset::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();

        assert_eq!(first, second);
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.n_annotators(), 2);
        assert_eq!(
            loaded.rows().collect::<Vec<_>>(),
            d.rows().collect::<Vec<_>>()
        );
    }

    #[test]
    fn push_and_iterate() {
        let mut d = AnnotatedPairsDataset::new();
        d.push("judge1", "a", "b", 1.0);
        d.push("judge2", "b", "a", 2.0);

        assert_eq!(d.len(), 2);
        assert_eq!(d.n_entities(), 2);
        assert_eq!(d.n_annotators(), 2);

        let rows: Vec<_> = d.rows().collect();
        assert_eq!(rows[0], (0, 0, 1, 1.0));
        assert_eq!(rows[1], (1, 1, 0, 2.0));
    }
}

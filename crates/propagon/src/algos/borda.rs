//! Borda count (`docs/algorithms.md` §6.1).
//!
//! From pairwise data: weighted win totals — the canonical counting
//! estimator, minimax-optimal for rank recovery under uniform schedules
//! (Shah & Wainwright 2018). From rankings: classical positional points
//! (`m - rank` per ballot).
//!
//! Assumes a roughly uniform comparison schedule for the pairwise mode's
//! optimality guarantee: with imbalanced schedules a win total just rewards
//! whoever played weak opponents most often, with no correction for strength
//! of schedule.
//!
//! Gotchas: the two entry points live on different score scales — pairwise
//! win-weight sums versus positional points — so their outputs are not
//! comparable, only their orders. Borda is not Condorcet-consistent (it can
//! rank a head-to-head winner second) and is manipulable by clones and
//! strategic burying.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{PairwiseDataset, RankingsDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// Borda-count ranker. No tunable parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Borda {}

/// Fitted Borda scores.
#[derive(Debug, Clone)]
pub struct BordaModel {
    params: Borda,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(BordaModel, "borda");

impl Ranker for Borda {
    type Data = PairwiseDataset;
    type Model = BordaModel;

    fn fit_opts(&self, data: &PairwiseDataset, _opts: &FitOptions<'_>) -> Result<BordaModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let tally = data.tally();
        let scores = tally.wins.iter().map(|&(_, wsum)| wsum).collect();
        Ok(BordaModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }
}

impl Borda {
    /// Positional Borda over full/partial rankings: an item at position `p`
    /// (0-based) in a ballot of `m` items earns `m - 1 - p` points.
    pub fn fit_rankings(&self, data: &RankingsDataset) -> Result<BordaModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let mut scores = vec![0.0; data.n_entities()];
        for ranking in data.rankings() {
            let m = ranking.len();
            for (p, &id) in ranking.iter().enumerate() {
                scores[id as usize] += (m - 1 - p) as f64;
            }
        }
        Ok(BordaModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    #[test]
    fn pairwise_win_totals() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("a", "c", 2.0);
        d.push("b", "c", 1.0);
        let m = Borda::default().fit(&d).unwrap();
        let order = m.sorted_scores();
        assert_eq!(order[0], ("a", 3.0));
        assert_eq!(order[1], ("b", 1.0));
        assert_eq!(order[2], ("c", 0.0));
    }

    #[test]
    fn positional_points() {
        let mut d = RankingsDataset::new();
        d.push_ranking(["a", "b", "c"]).unwrap(); // a:2 b:1 c:0
        d.push_ranking(["b", "a"]).unwrap(); // b:1 a:0
        let m = Borda::default().fit_rankings(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert_eq!(s["a"], 2.0);
        assert_eq!(s["b"], 2.0);
        assert_eq!(s["c"], 0.0);
    }
}

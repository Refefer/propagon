//! Colley matrix ratings (`docs/algorithms.md` §5.2).
//!
//! Laplace-smoothed, schedule-aware win rates (Colley 2002): solve
//! `C r = b` with `C = 2I + diag(games_i) − N` (N = games between each
//! pair) and `b_i = 1 + (wins_i − losses_i)/2`. `C` is symmetric positive
//! definite, so plain conjugate gradients converge; ratings center on 1/2
//! and always sum to n/2. "Bias-free" by construction: margins, venue, and
//! prior seasons never enter.
//!
//! Assumes row weights are game **counts** (a weight of 2 means the pair
//! played twice with the same outcome); fractional weights are accepted and
//! treated as fractional games.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::solver::SparseSymmetric;
use crate::traits::{FitOptions, Ranker};

/// Colley parameters (conjugate-gradient budget).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Colley {
    /// Maximum CG iterations.
    pub iterations: usize,
    /// Relative residual target.
    pub tolerance: f64,
}

impl Default for Colley {
    fn default() -> Self {
        Self {
            iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// Fitted Colley ratings (centered on 1/2).
#[derive(Debug, Clone)]
pub struct ColleyModel {
    params: Colley,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(ColleyModel, "colley");

impl Ranker for Colley {
    type Data = PairwiseDataset;
    type Model = ColleyModel;

    fn fit_opts(&self, data: &PairwiseDataset, _opts: &FitOptions<'_>) -> Result<ColleyModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = data.n_entities();
        let mut c = SparseSymmetric::new(n);
        let mut b = vec![1.0; n];

        for i in 0..n {
            c.add(i, i, 2.0);
        }

        for (w, l, x) in data.rows() {
            let games = f64::from(x);
            let (w, l) = (w as usize, l as usize);
            c.add(w, w, games);
            c.add(l, l, games);
            c.add(w, l, -games);
            b[w] += games / 2.0;
            b[l] -= games / 2.0;
        }

        c.compress();
        let scores = c.solve(&b, self.iterations, self.tolerance)?;

        Ok(ColleyModel {
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

    /// One game: C = [[3,-1],[-1,3]], b = (1.5, 0.5) → r = (5/8, 3/8).
    #[test]
    fn single_game_hand_solve() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        let m = Colley::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - 0.625).abs() < 1e-9);
        assert!((s["b"] - 0.375).abs() < 1e-9);
    }

    /// Ratings always average exactly 1/2 (Colley's conservation property).
    #[test]
    fn ratings_average_one_half() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.push("b", "c", 1.0);
        d.push("c", "d", 3.0);
        d.push("a", "d", 1.0);
        let m = Colley::default().fit(&d).unwrap();
        let total: f64 = m.scores().map(|(_, s)| s).sum();
        assert!(
            (total - 2.0).abs() < 1e-9,
            "4 teams sum to 2.0, got {total}"
        );
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        let m = Colley::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = ColleyModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

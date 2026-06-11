//! Massey ratings (`docs/algorithms.md` §5.1).
//!
//! Least-squares ratings from point margins (Massey 1997): each game asks
//! `r_winner − r_loser ≈ margin`; the normal equations are `M r = p` with
//! `M` the schedule Laplacian (diagonal = games played, off-diagonal =
//! −games between the pair) and `p` the net point differential. `M` is
//! singular with the constant vector as kernel; the canonical fix is the
//! mean-zero solution, solved here by projected conjugate gradients.
//!
//! Assumes the row weight **is the margin of victory** (margins ≤ 0 are
//! rejected). Requires a connected schedule — disconnected groups have no
//! common scale, surfaced as a solver error rather than silently bridged.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::solver::SparseSymmetric;
use crate::traits::{FitOptions, Ranker};

/// Massey parameters (conjugate-gradient budget).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Massey {
    /// Maximum CG iterations.
    pub iterations: usize,
    /// Relative residual target.
    pub tolerance: f64,
}

impl Default for Massey {
    fn default() -> Self {
        Self {
            iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// Fitted Massey ratings (mean zero; point-margin scale).
#[derive(Debug, Clone)]
pub struct MasseyModel {
    params: Massey,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(MasseyModel, "massey");

impl Ranker for Massey {
    type Data = PairwiseDataset;
    type Model = MasseyModel;

    fn fit_opts(&self, data: &PairwiseDataset, _opts: &FitOptions<'_>) -> Result<MasseyModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = data.n_entities();
        let mut m = SparseSymmetric::new(n);
        let mut p = vec![0.0; n];

        for (w, l, x) in data.rows() {
            let margin = f64::from(x);
            if margin <= 0.0 {
                return Err(Error::InvalidInput(format!(
                    "massey needs positive margins as row weights; got {margin} for {} vs {}",
                    data.interner().resolve(w),
                    data.interner().resolve(l)
                )));
            }

            let (w, l) = (w as usize, l as usize);
            m.add(w, w, 1.0);
            m.add(l, l, 1.0);
            m.add(w, l, -1.0);
            p[w] += margin;
            p[l] -= margin;
        }

        m.compress();
        let scores = m.solve_mean_zero(&p, self.iterations, self.tolerance)?;

        Ok(MasseyModel {
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

    /// Two teams, one game, margin 10 → ratings ±5.
    #[test]
    fn single_game_splits_margin() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 10.0);
        let m = Massey::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - 5.0).abs() < 1e-8);
        assert!((s["b"] + 5.0).abs() < 1e-8);
    }

    #[test]
    fn rejects_non_positive_margins() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 0.0);
        assert!(matches!(
            Massey::default().fit(&d),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "c", 7.0);
        d.push("a", "c", 2.0);
        let m = Massey::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = MasseyModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

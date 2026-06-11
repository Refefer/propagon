//! Copeland method (`docs/algorithms.md` §6.2).
//!
//! Score = pairwise majorities won, with ties worth half. Condorcet-consistent:
//! an entity that beats every opponent head-to-head always ranks first.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// Copeland ranker. No tunable parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Copeland {}

/// Fitted Copeland scores.
#[derive(Debug, Clone)]
pub struct CopelandModel {
    params: Copeland,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(CopelandModel, "copeland");

impl Ranker for Copeland {
    type Data = PairwiseDataset;
    type Model = CopelandModel;

    fn fit_opts(&self, data: &PairwiseDataset, _opts: &FitOptions<'_>) -> Result<CopelandModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        // Aggregate directed win weight per unordered pair.
        let mut margin: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, x) in data.rows() {
            let (key, sign) = if w < l { ((w, l), 1.0) } else { ((l, w), -1.0) };
            *margin.entry(key).or_default() += sign * f64::from(x);
        }
        let mut scores = vec![0.0; data.n_entities()];
        for ((a, b), m) in margin {
            if m > 0.0 {
                scores[a as usize] += 1.0;
            } else if m < 0.0 {
                scores[b as usize] += 1.0;
            } else {
                scores[a as usize] += 0.5;
                scores[b as usize] += 0.5;
            }
        }
        Ok(CopelandModel { params: *self, names: data.interner().clone(), scores })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    #[test]
    fn majorities_and_ties() {
        let mut d = PairwiseDataset::new();
        // a beats b twice, b beats a once -> a wins the pair
        d.push("a", "b", 1.0);
        d.push("a", "b", 1.0);
        d.push("b", "a", 1.0);
        // b vs c split evenly -> half point each
        d.push("b", "c", 1.0);
        d.push("c", "b", 1.0);
        // a beats c
        d.push("a", "c", 1.0);
        let m = Copeland::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert_eq!(s["a"], 2.0);
        assert_eq!(s["b"], 0.5);
        assert_eq!(s["c"], 0.5);
    }
}

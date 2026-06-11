//! Katz centrality (`docs/algorithms.md` §4.3; Katz 1953).
//!
//! Counts **all** incoming walks, geometrically discounted by length:
//! `x = Σ_{k≥1} α^k (Aᵀ)^k 1`, computed by iterating `x ← α Aᵀ x + 1`.
//! Interpolates between in-degree (α → 0) and eigenvector centrality
//! (α → 1/λ_max), and — unlike eigenvector centrality — gives sensible
//! scores on DAGs and other graphs without a strongly connected core.
//!
//! Edge weights multiply walk contributions. Assumes `α < 1/λ_max(A)`:
//! beyond that the series diverges, which surfaces as a typed numeric
//! error suggesting a smaller `alpha` (never NaN output). The default
//! α = 0.1 is safe for λ_max < 10 (e.g. any unweighted graph with max
//! in-degree below 10 has λ_max ≤ that bound).

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::GraphDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// Katz parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Katz {
    /// Walk discount; must stay below `1/λ_max` of the adjacency matrix.
    pub alpha: f64,
    /// Fixed-point iteration budget.
    pub iterations: usize,
    /// L1-change early-exit threshold.
    pub tolerance: f64,
}

impl Default for Katz {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            iterations: 100,
            tolerance: 1e-12,
        }
    }
}

/// Fitted Katz scores (discounted incoming-walk counts; ≥ 1 each).
#[derive(Debug, Clone)]
pub struct KatzModel {
    params: Katz,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(KatzModel, "katz");

impl Ranker for Katz {
    type Data = GraphDataset;
    type Model = KatzModel;

    fn fit_opts(&self, data: &GraphDataset, _opts: &FitOptions<'_>) -> Result<KatzModel> {
        let g = data.view();
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if self.alpha <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "katz alpha must be positive, got {}",
                self.alpha
            )));
        }

        let n = g.n_nodes();

        // Incoming adjacency: in_edges[d] lists (source, weight).
        let mut incoming: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for (s, d, w) in g.edges() {
            incoming[d as usize].push((s, f64::from(w)));
        }

        let mut x = vec![1.0f64; n];
        let mut next = vec![0.0f64; n];

        for _ in 0..self.iterations {
            for (d, inc) in incoming.iter().enumerate() {
                next[d] =
                    1.0 + self.alpha * inc.iter().map(|&(s, w)| w * x[s as usize]).sum::<f64>();
            }

            let magnitude: f64 = next.iter().map(|v| v.abs()).sum();
            if !magnitude.is_finite() || magnitude > 1e12 * n as f64 {
                return Err(Error::Numeric(format!(
                    "katz series diverged: alpha {} is at or above 1/λ_max for \
                     this graph — lower --alpha",
                    self.alpha
                )));
            }

            let change: f64 = x.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
            std::mem::swap(&mut x, &mut next);

            if change < self.tolerance {
                break;
            }
        }

        // Report Σ_{k≥1} (walks only): subtract the constant injection.
        let scores = x.iter().map(|v| v - 1.0).collect();

        Ok(KatzModel {
            params: *self,
            names: g.interner.clone(),
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// Dense oracle on a 3-path a→b→c: solving (I − αAᵀ)y = 1 by hand with
    /// α = 0.5 gives y = (1, 1.5, 1.75); Katz scores are y − 1.
    #[test]
    fn matches_dense_solve_on_path() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "c", 1.0);

        let m = Katz {
            alpha: 0.5,
            ..Default::default()
        }
        .fit(&g)
        .unwrap();
        let s: std::collections::HashMap<&str, f64> = m.scores().collect();
        assert!(s["a"].abs() < 1e-9);
        assert!((s["b"] - 0.5).abs() < 1e-9);
        assert!((s["c"] - 0.75).abs() < 1e-9);
    }

    /// α at/above 1/λ_max diverges with a typed error (2-cycle: λ_max = 1).
    #[test]
    fn divergence_is_a_typed_error() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "a", 1.0);

        let r = Katz {
            alpha: 1.5,
            iterations: 10_000,
            ..Default::default()
        }
        .fit(&g);
        assert!(matches!(r, Err(Error::Numeric(_))));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "c", 2.0);

        let m = Katz::default().fit(&g).unwrap();
        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = KatzModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

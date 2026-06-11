//! Rank Centrality (`docs/algorithms.md` §3.1, Negahban-Oh-Shah 2017).
//!
//! Random walk that drifts toward winners: from entity `i`, transition to
//! opponent `j` with probability proportional to the fraction of `i`-vs-`j`
//! comparisons that `j` won (scaled by `1/d_max`, remainder as self-loop).
//! The stationary distribution is a consistent, near-optimal Bradley-Terry
//! estimate — spectral speed, MLE-grade statistics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, Ranker};

/// Rank Centrality parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankCentrality {
    /// Maximum power-iteration sweeps.
    pub iterations: usize,
    /// Early exit when the L1 change per sweep drops below this.
    pub tolerance: f64,
}

impl Default for RankCentrality {
    fn default() -> Self {
        Self {
            iterations: 200,
            tolerance: 1e-10,
        }
    }
}

/// Fitted stationary distribution (sums to 1; higher is better).
#[derive(Debug, Clone)]
pub struct RankCentralityModel {
    params: RankCentrality,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(RankCentralityModel, "rank-centrality");

impl Ranker for RankCentrality {
    type Data = PairwiseDataset;
    type Model = RankCentralityModel;

    fn fit_opts(
        &self,
        data: &PairwiseDataset,
        opts: &FitOptions<'_>,
    ) -> Result<RankCentralityModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = data.n_entities();

        // Aggregate per ordered pair: weight of wins of `a` over `b`.
        let mut wins: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, x) in data.rows() {
            *wins.entry((w, l)).or_default() += f64::from(x);
        }

        // Outgoing transitions: i -> j with p = (wins of j over i) / (n_ij · d_max).
        let mut out: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        let mut degree = vec![0usize; n];
        let mut pairs: Vec<((u32, u32), f64)> = Vec::new();
        for (&(a, b), &w_ab) in &wins {
            if a < b {
                let w_ba = wins.get(&(b, a)).copied().unwrap_or(0.0);
                pairs.push(((a, b), w_ab));
                pairs.push(((b, a), w_ba));
                degree[a as usize] += 1;
                degree[b as usize] += 1;
            } else if !wins.contains_key(&(b, a)) {
                // one-directional pair seen only as (a, b) with a > b
                pairs.push(((a, b), w_ab));
                pairs.push(((b, a), 0.0));
                degree[a as usize] += 1;
                degree[b as usize] += 1;
            }
        }
        let d_max = degree.iter().copied().max().unwrap_or(1).max(1) as f64;

        let mut totals: HashMap<(u32, u32), f64> = HashMap::new();
        for &((a, b), w) in &pairs {
            let key = if a < b { (a, b) } else { (b, a) };
            *totals.entry(key).or_default() += w;
        }
        for ((a, b), w_ab) in pairs {
            let key = if a < b { (a, b) } else { (b, a) };
            let total = totals[&key];
            if total > 0.0 {
                // walk from the loser side toward b proportional to b's wins:
                // transition a -> b uses b's win fraction over a.
                let w_ba = total - w_ab;
                out[a as usize].push((b, w_ba / (total * d_max)));
            }
        }
        for adj in &mut out {
            adj.sort_unstable_by_key(|e| e.0);
        }

        // Power iteration with implicit self-loops (rows already sum ≤ 1).
        let scores = parallel::run_scoped(opts, || {
            let progress = opts.progress();
            progress.start("rank-centrality sweeps", Some(self.iterations as u64));
            let mut pi = vec![1.0 / n as f64; n];
            for it in 0..self.iterations {
                let frozen = &pi;
                let mut next = parallel::par_map_indexed(n, |i| {
                    let leaving: f64 = out[i].iter().map(|&(_, p)| p).sum();
                    frozen[i] * (1.0 - leaving)
                });
                for i in 0..n {
                    for &(j, p) in &out[i] {
                        next[j as usize] += pi[i] * p;
                    }
                }
                let delta: f64 = pi.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
                pi = next;
                progress.update(it as u64 + 1);
                if delta < self.tolerance {
                    break;
                }
            }
            progress.finish();
            pi
        });

        Ok(RankCentralityModel {
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
    fn recovers_bradley_terry_order() {
        // Strengths a=4, b=2, c=1; sample outcomes at the exact BT rates.
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 4.0);
        d.push("b", "a", 2.0);
        d.push("a", "c", 4.0);
        d.push("c", "a", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "b", 1.0);
        let m = RankCentrality::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        let total: f64 = m.scores().map(|(_, s)| s).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "stationary distribution sums to 1"
        );
    }

    #[test]
    fn stationary_distribution_matches_balance_condition() {
        // Two entities: a beats b 3:1 -> π ∝ wins toward each:
        // π_a · p(a->b) = π_b · p(b->a)  =>  π_a / π_b = 3.
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        let m = RankCentrality::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!(
            (s["a"] / s["b"] - 3.0).abs() < 1e-6,
            "ratio {}",
            s["a"] / s["b"]
        );
    }
}

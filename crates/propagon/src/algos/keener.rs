//! Keener ratings (`docs/algorithms.md` §3.3; Keener 1993).
//!
//! Margin-aware Perron-eigenvector rating: aggregate a nonnegative score
//! matrix `S_ij` ("amount i scored against j"), Laplace-smooth each pair to
//! `(S_ij + 1)/(S_ij + S_ji + 2)`, optionally skew with Keener's
//! `h(x) = 1/2 + sgn(x − 1/2)·√|2x−1|/2` (damps blowouts), optionally
//! divide team `i`'s row by its games played (for unequal schedules), then
//! rank by the Perron eigenvector of the result (power iteration,
//! normalized to sum 1). Strength of schedule is automatic: points against
//! strong opponents are worth more.
//!
//! Assumes rows mean **`(scorer, opponent, amount)`** — for score data push
//! both directions of every game (winner's points and loser's points); for
//! win/loss data push win counts and the smoothing supplies the rest. No
//! probabilistic semantics: ratings are eigenvector mass, not win odds.
//! Requires a connected schedule (otherwise the eigenvector concentrates on
//! one component).

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// Keener parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keener {
    /// Apply Keener's skew `h` to the smoothed proportions (damps margins;
    /// *Who's #1?* eq. 4.4 — "skewing is not always needed").
    pub skew: bool,
    /// Divide each team's row by its games played (*Who's #1?* eq. 4.5);
    /// matters only when schedules are unequal.
    pub normalize_games: bool,
    /// Power-iteration budget.
    pub iterations: usize,
    /// L1-change early-exit threshold.
    pub tolerance: f64,
}

impl Default for Keener {
    fn default() -> Self {
        Self {
            skew: true,
            normalize_games: true,
            iterations: 1000,
            tolerance: 1e-12,
        }
    }
}

/// Fitted Keener ratings (nonnegative, sum to 1).
#[derive(Debug, Clone)]
pub struct KeenerModel {
    params: Keener,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(KeenerModel, "keener");

/// Keener's skew: expands tiny advantages, compresses blowouts.
fn skew(x: f64) -> f64 {
    0.5 + 0.5 * (2.0 * x - 1.0).signum() * (2.0 * x - 1.0).abs().sqrt()
}

impl Ranker for Keener {
    type Data = PairwiseDataset;
    type Model = KeenerModel;

    fn fit_opts(&self, data: &PairwiseDataset, _opts: &FitOptions<'_>) -> Result<KeenerModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = data.n_entities();

        // Aggregate per ordered pair: scored[(i, j)] = amount i scored on j;
        // games counted once per row, per side.
        let mut scored: std::collections::HashMap<(u32, u32), f64> =
            std::collections::HashMap::new();
        let mut games = vec![0.0f64; n];

        for (w, l, x) in data.rows() {
            *scored.entry((w, l)).or_default() += f64::from(x);
            games[w as usize] += 1.0;
            games[l as usize] += 1.0;
        }

        // Smoothed (and optionally skewed/normalized) strength matrix as
        // row-major adjacency, deterministic order.
        let mut pairs: Vec<((u32, u32), f64)> = Vec::with_capacity(scored.len());

        for (&(i, j), &s_ij) in &scored {
            let s_ji = scored.get(&(j, i)).copied().unwrap_or(0.0);
            let mut a = (s_ij + 1.0) / (s_ij + s_ji + 2.0);

            if self.skew {
                a = skew(a);
            }
            if self.normalize_games {
                a /= games[i as usize].max(1.0);
            }
            pairs.push(((i, j), a));

            // The reverse direction exists even when j never scored
            // (smoothing gives it positive mass); add it unless j's own row
            // will produce it from its observed entry.
            if !scored.contains_key(&(j, i)) {
                let mut rev = (s_ji + 1.0) / (s_ij + s_ji + 2.0);

                if self.skew {
                    rev = skew(rev);
                }
                if self.normalize_games {
                    rev /= games[j as usize].max(1.0);
                }
                pairs.push(((j, i), rev));
            }
        }

        pairs.sort_unstable_by_key(|&(key, _)| key);

        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for ((i, j), a) in pairs {
            rows[i as usize].push((j, a));
        }

        // Power iteration: r' = A r, L1-normalized each step.
        let mut r = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];

        for _ in 0..self.iterations {
            for (i, row) in rows.iter().enumerate() {
                next[i] = row.iter().map(|&(j, a)| a * r[j as usize]).sum();
            }

            let total: f64 = next.iter().sum();
            if total <= 0.0 || total.is_nan() || total.is_infinite() {
                return Err(Error::Numeric(
                    "keener power iteration collapsed; is the schedule connected?".into(),
                ));
            }
            next.iter_mut().for_each(|v| *v /= total);

            let change: f64 = r.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
            std::mem::swap(&mut r, &mut next);

            if change < self.tolerance {
                break;
            }
        }

        Ok(KeenerModel {
            params: *self,
            names: data.interner().clone(),
            scores: r,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    #[test]
    fn skew_endpoints_fixed() {
        assert_eq!(skew(0.0), 0.0);
        assert_eq!(skew(0.5), 0.5);
        assert_eq!(skew(1.0), 1.0);
    }

    /// Symmetric scoring ties everyone.
    #[test]
    fn balanced_round_robin_ties() {
        let mut d = PairwiseDataset::new();
        for (a, b) in [("x", "y"), ("y", "z"), ("z", "x")] {
            d.push(a, b, 10.0);
            d.push(b, a, 10.0);
        }

        let m = Keener::default().fit(&d).unwrap();
        for (_, s) in m.scores() {
            assert!((s - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn dominant_scorer_ranks_first() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 30.0);
        d.push("b", "a", 10.0);
        d.push("b", "c", 30.0);
        d.push("c", "b", 10.0);
        d.push("a", "c", 40.0);
        d.push("c", "a", 5.0);

        let m = Keener::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        d.push("b", "c", 2.0);
        let m = Keener::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = KeenerModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

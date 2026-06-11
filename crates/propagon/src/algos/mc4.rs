//! MC4 Markov-chain rank aggregation (`docs/algorithms.md` §6.4; Dwork,
//! Kumar, Naor & Sivakumar, WWW 2001).
//!
//! From the current item, pick another uniformly at random; move there iff
//! a **strict majority** of the ballots ranking both items rank it better,
//! else stay. The stationary distribution aggregates the ballots
//! ("generalizes Copeland"). Handles partial, overlapping, different-length
//! ballots natively — a ballot only constrains the items it contains.
//!
//! The paper's chain can be reducible (a Condorcet winner absorbs); the
//! standard fix is PageRank-style teleportation `βS + (1−β)/n` (Langville &
//! Meyer, *Who's #1?* ch. 6 — web convention β = 0.85, sports data often
//! prefers 0.5–0.6). `damping = 1.0` is allowed but may concentrate all
//! mass on one item.
//!
//! Gotcha: the majority tally is a dense n×n matrix — fine for metasearch
//! and leaderboard sizes, wrong tool past ~10⁴ items.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::RankingsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, Ranker};

/// MC4 parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mc4 {
    /// Teleport mix: follow the majority chain with this probability.
    pub damping: f64,
    /// Power-iteration budget.
    pub iterations: usize,
    /// L1-change early-exit threshold.
    pub tolerance: f64,
}

impl Default for Mc4 {
    fn default() -> Self {
        Self {
            damping: 0.85,
            iterations: 200,
            tolerance: 1e-9,
        }
    }
}

/// Fitted MC4 stationary distribution (sums to 1).
#[derive(Debug, Clone)]
pub struct Mc4Model {
    params: Mc4,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(Mc4Model, "mc4");

impl Ranker for Mc4 {
    type Data = RankingsDataset;
    type Model = Mc4Model;

    fn fit_opts(&self, data: &RankingsDataset, opts: &FitOptions<'_>) -> Result<Mc4Model> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if !(0.0..=1.0).contains(&self.damping) {
            return Err(Error::InvalidInput(format!(
                "mc4 damping must lie in [0, 1], got {}",
                self.damping
            )));
        }

        let n = data.n_entities();

        // above[i*n + j] = ballots ranking both i and j with i better.
        let mut above = vec![0u32; n * n];

        for ballot in data.rankings() {
            for (p, &i) in ballot.iter().enumerate() {
                for &j in &ballot[p + 1..] {
                    above[i as usize * n + j as usize] += 1;
                }
            }
        }

        // incoming[j] = states that move to j; out_degree[i] = #moves out.
        let mut incoming: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut out_degree = vec![0usize; n];

        for i in 0..n {
            for j in 0..n {
                if i != j && above[j * n + i] > above[i * n + j] {
                    incoming[j].push(i as u32);
                    out_degree[i] += 1;
                }
            }
        }

        let progress = opts.progress;
        progress.start("power iterations", Some(self.iterations as u64));

        let teleport = (1.0 - self.damping) / n as f64;
        let mut pi = vec![1.0 / n as f64; n];

        for pass in 0..self.iterations {
            let frozen = &pi;
            let next: Vec<f64> = parallel::par_map_indexed(n, |j| {
                let moved: f64 = incoming[j]
                    .iter()
                    .map(|&i| frozen[i as usize] / n as f64)
                    .sum();
                let stay = frozen[j] * (1.0 - out_degree[j] as f64 / n as f64);
                self.damping * (moved + stay) + teleport
            });

            let change: f64 = pi.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
            pi = next;
            progress.update(pass as u64 + 1);

            if change < self.tolerance {
                break;
            }
        }

        progress.finish();

        Ok(Mc4Model {
            params: *self,
            names: data.interner().clone(),
            scores: pi,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn ballots(rows: &[&[&str]]) -> RankingsDataset {
        let mut d = RankingsDataset::new();
        for row in rows {
            d.push_ranking(row.iter().copied()).unwrap();
        }
        d
    }

    /// Brute-force oracle: solve the stationary equations of the exact
    /// teleported chain densely and compare to power iteration.
    #[test]
    fn power_iteration_matches_dense_stationary_solve() {
        let d = ballots(&[
            &["a", "b", "c", "d"],
            &["b", "a", "d", "c"],
            &["a", "c", "b"],
            &["d", "a", "b"],
            &["c", "d"],
        ]);
        let model = Mc4::default().fit(&d).unwrap();
        let n = 4;

        // Rebuild the full transition matrix densely.
        let mut above = vec![0u32; n * n];
        for ballot in d.rankings() {
            for (p, &i) in ballot.iter().enumerate() {
                for &j in &ballot[p + 1..] {
                    above[i as usize * n + j as usize] += 1;
                }
            }
        }

        let beta = 0.85;
        let mut t = vec![vec![(1.0 - beta) / n as f64; n]; n]; // row -> col
        for i in 0..n {
            let mut out = 0;
            for j in 0..n {
                if i != j && above[j * n + i] > above[i * n + j] {
                    t[i][j] += beta / n as f64;
                    out += 1;
                }
            }
            t[i][i] += beta * (1.0 - out as f64 / n as f64);
        }

        // Fixed-point iterate the dense chain to machine precision.
        let mut pi = vec![1.0 / n as f64; n];
        for _ in 0..10_000 {
            let mut next = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    next[j] += pi[i] * t[i][j];
                }
            }
            pi = next;
        }

        for (idx, (_, got)) in model.scores().enumerate() {
            assert!((got - pi[idx]).abs() < 1e-9, "{idx}: {got} vs {}", pi[idx]);
        }
    }

    /// A clear majority order is reproduced (Condorcet consistency).
    #[test]
    fn majority_order_wins() {
        let d = ballots(&[
            &["a", "b", "c"],
            &["a", "b", "c"],
            &["a", "c", "b"],
            &["b", "a", "c"],
        ]);
        let m = Mc4::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    /// Without teleport a Condorcet winner absorbs all mass; with it, every
    /// item keeps positive mass and the order is preserved.
    #[test]
    fn teleport_keeps_chain_ergodic() {
        let d = ballots(&[&["a", "b"], &["a", "c"], &["b", "c"]]);

        let absorbing = Mc4 {
            damping: 1.0,
            iterations: 5_000,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let s: std::collections::HashMap<&str, f64> = absorbing.scores().collect();
        assert!(s["a"] > 0.999, "condorcet winner absorbs: {}", s["a"]);

        let damped = Mc4::default().fit(&d).unwrap();
        for (_, score) in damped.scores() {
            assert!(score > 0.0);
        }
        let order: Vec<&str> = damped.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    /// Disjoint ballot universes stay finite and well-defined.
    #[test]
    fn disjoint_universes_do_not_panic() {
        let d = ballots(&[&["a", "b"], &["x", "y"]]);
        let m = Mc4::default().fit(&d).unwrap();
        let total: f64 = m.scores().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let d = ballots(&[&["a", "b", "c"], &["b", "a", "c"]]);
        let m = Mc4::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = Mc4Model::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

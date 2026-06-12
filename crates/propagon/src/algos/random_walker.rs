//! Random-walker rankings (`docs/algorithms.md` §3.4; Callaghan, Mucha &
//! Porter, Amer. Math. Monthly 114 (2007) 761–777).
//!
//! A population of independent "fan" walkers: from entity `i`, a walker
//! picks one of `i`'s games uniformly at random and sides with the game's
//! winner with probability `p > ½` (the loser with `1−p`). The stationary
//! share of fans ranks the entities. The bias `p` is the method's knob:
//! `p → 1` approaches win-percentage extremism, `p` near `½` smooths toward
//! schedule structure — sweeping `p` traces a family of rankings whose
//! stability is itself diagnostic.
//!
//! Relation to its random-walk siblings: Rank Centrality drifts on win
//! *fractions* with a fixed `1/d_max` normalization; this walk mixes wins
//! and losses per-game with the `p` bias and normalizes by each entity's
//! game count. Both are stationary distributions of comparison-graph walks.
//!
//! Gotcha: on a disconnected schedule the stationary mass splits across
//! components by their internal structure, so cross-component score ratios
//! are meaningless (the original application — full NCAA seasons — is
//! connected). Components are not bridged here; triage connectivity first.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, Ranker};

/// Random-walker ranking parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomWalker {
    /// Probability a walker sides with a game's winner. Must lie strictly
    /// in (½, 1): `½` carries no information, `1` makes undefeated entities
    /// absorbing.
    pub p: f64,
    /// Maximum power-iteration sweeps.
    pub iterations: usize,
    /// Early exit when the L1 change per sweep drops below this.
    pub tolerance: f64,
}

impl Default for RandomWalker {
    fn default() -> Self {
        Self {
            p: 0.75,
            iterations: 1000,
            tolerance: 1e-12,
        }
    }
}

/// Fitted stationary distribution (sums to 1; higher is better).
#[derive(Debug, Clone)]
pub struct RandomWalkerModel {
    params: RandomWalker,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(RandomWalkerModel, "random-walker");

impl Ranker for RandomWalker {
    type Data = PairwiseDataset;
    type Model = RandomWalkerModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<RandomWalkerModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if !(self.p > 0.5 && self.p < 1.0) {
            return Err(Error::InvalidInput(format!(
                "random-walker bias p must lie strictly in (0.5, 1), got {}",
                self.p
            )));
        }
        let n = data.n_entities();

        // Aggregate per ordered pair: weight of wins of `a` over `b`.
        let mut wins: HashMap<(u32, u32), f64> = HashMap::new();
        let mut games = vec![0.0f64; n];
        for (w, l, x) in data.rows() {
            let x = f64::from(x);
            *wins.entry((w, l)).or_default() += x;
            games[w as usize] += x;
            games[l as usize] += x;
        }

        // Outgoing transitions: i -> j with probability
        //   [p · (j's wins over i) + (1−p) · (i's wins over j)] / g_i.
        // The self-loop is the implicit remainder, so each row sums to 1.
        let mut out: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for (&(a, b), &w_ab) in &wins {
            // Process each unordered pair once, from its (a < b) key or from
            // the only direction present.
            if a > b && wins.contains_key(&(b, a)) {
                continue;
            }
            let w_ba = wins.get(&(b, a)).copied().unwrap_or(0.0);
            let (i, j, w_ij, w_ji) = (a, b, w_ab, w_ba);
            out[i as usize].push((
                j,
                (self.p * w_ji + (1.0 - self.p) * w_ij) / games[i as usize],
            ));
            out[j as usize].push((
                i,
                (self.p * w_ij + (1.0 - self.p) * w_ji) / games[j as usize],
            ));
        }
        for adj in &mut out {
            adj.sort_unstable_by_key(|e| e.0);
        }

        // Power iteration with implicit self-loops (rows sum to exactly 1).
        let scores = parallel::run_scoped(opts, || {
            let progress = opts.progress;
            progress.start("random-walker sweeps", Some(self.iterations as u64));
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

        Ok(RandomWalkerModel {
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

    /// One game between two teams: detailed balance gives
    /// `π_winner / π_loser = p / (1 − p)` exactly (the paper's 2-team case).
    #[test]
    fn two_team_ratio_is_p_over_one_minus_p() {
        for p in [0.6, 0.75, 0.9] {
            let mut d = PairwiseDataset::new();
            d.push("a", "b", 1.0);
            let algo = RandomWalker {
                p,
                ..RandomWalker::default()
            };
            let m = algo.fit(&d).unwrap();
            let s: std::collections::HashMap<_, _> = m.scores().collect();
            assert!(
                (s["a"] / s["b"] - p / (1.0 - p)).abs() < 1e-9,
                "p={p}: ratio {}",
                s["a"] / s["b"]
            );
        }
    }

    /// Chain a→b→c at p = 0.75: stationary distribution hand-solved from
    /// the 3×3 transition matrix.
    ///
    /// Transitions (g_a = 1, g_b = 2, g_c = 1):
    ///   a→b = ¼, b→a = ⅜, b→c = ⅛, c→b = ¾.
    /// Balance: π_a·¼ = π_b·⅜ and π_c·¾ = π_b·⅛
    ///   ⇒ π ∝ (3, 2, ⅓) ⇒ π = (9/16, 6/16, 1/16).
    #[test]
    fn three_team_chain_matches_hand_solution() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        let m = RandomWalker::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        for (name, want) in [("a", 9.0 / 16.0), ("b", 6.0 / 16.0), ("c", 1.0 / 16.0)] {
            assert!(
                (s[name] - want).abs() < 1e-9,
                "{name}: {} vs {want}",
                s[name]
            );
        }
    }

    /// Stronger bias concentrates more mass on the winner.
    #[test]
    fn mass_on_winner_is_monotone_in_p() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        let mut last = 0.0;
        for p in [0.55, 0.65, 0.75, 0.85, 0.95] {
            let algo = RandomWalker {
                p,
                ..RandomWalker::default()
            };
            let m = algo.fit(&d).unwrap();
            let s: std::collections::HashMap<_, _> = m.scores().collect();
            assert!(s["a"] > last, "p={p}");
            last = s["a"];
        }
    }

    #[test]
    fn rejects_uninformative_or_absorbing_bias() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        for p in [0.5, 1.0, 0.0, 1.5] {
            let algo = RandomWalker {
                p,
                ..RandomWalker::default()
            };
            assert!(matches!(algo.fit(&d), Err(Error::InvalidInput(_))), "p={p}");
        }
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let d = PairwiseDataset::new();
        assert!(matches!(
            RandomWalker::default().fit(&d),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.push("b", "c", 1.0);
        let m = RandomWalker::default().fit(&d).unwrap();
        let mut buf1 = Vec::new();
        m.save_jsonl(&mut buf1).unwrap();
        let m2 = RandomWalkerModel::load_jsonl(buf1.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf1, buf2);
    }
}

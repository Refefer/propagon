//! LeaderRank (`docs/algorithms.md` §4.9; Lü, Zhang, Yeung & Zhou,
//! PLoS ONE 6(6) 2011).
//!
//! PageRank with teleportation replaced by a **ground node** linked
//! bidirectionally to every real node: the augmented walk is irreducible
//! with *zero* model parameters — no damping factor to tune and no sink
//! policy to pick, because every node now has at least the out-edge to the
//! ground. After the walk converges, the ground node's stationary mass is
//! redistributed evenly over the real nodes (`S_i = π_i + π_g / n`), so the
//! reported scores sum to 1 (within floating-point error).
//!
//! Assumes an unweighted endorsement graph: parallel edges are deduplicated
//! and weights ignored (the same v1 semantics as `pagerank`); self-loops
//! are kept, also matching `pagerank`.
//!
//! Gotcha: `iterations` and `tolerance` only budget the power iteration —
//! the *model* itself is parameter-free; the defaults simply mean "run to
//! convergence".

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{GraphDataset, GraphView};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, Ranker};

/// LeaderRank parameters — an iteration budget only, since the model has no
/// knobs of its own.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeaderRank {
    /// Power-iteration budget.
    pub iterations: usize,
    /// L1-change early-exit threshold.
    pub tolerance: f64,
}

impl Default for LeaderRank {
    fn default() -> Self {
        Self {
            iterations: 1000,
            tolerance: 1e-12,
        }
    }
}

/// Fitted LeaderRank scores (sum to 1; higher is better).
#[derive(Debug, Clone)]
pub struct LeaderRankModel {
    params: LeaderRank,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(LeaderRankModel, "leader-rank");

impl LeaderRank {
    /// Fits from a borrowed graph view (e.g.
    /// [`PairwiseDataset::as_graph`](crate::PairwiseDataset::as_graph)).
    ///
    /// Builds the (n+1)-node augmented walk — deduplicated unweighted
    /// out-neighbors plus the bidirectional ground links — power-iterates
    /// from uniform until the L1 change drops below `tolerance` (or the
    /// budget runs out), then folds the ground mass back into the real
    /// nodes and drops the ground from the output.
    pub fn fit_view(&self, g: GraphView<'_>, opts: &FitOptions<'_>) -> Result<LeaderRankModel> {
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = g.n_nodes();
        let ground = n as u32;

        // Deduplicated unweighted out-adjacency over n + 1 nodes. Every real
        // node gains the edge to the ground and the ground points back at
        // all of them, so no node is a sink and the chain is irreducible.
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n + 1];
        for (s, d, _) in g.edges() {
            out[s as usize].push(d);
        }
        for adj in out.iter_mut().take(n) {
            adj.push(ground);
            adj.sort_unstable();
            adj.dedup();
        }
        out[n] = (0..n as u32).collect();

        // Incoming lists carrying each source's uniform out-share, so one
        // sweep is an independent gather per target node. Every row of `out`
        // is non-empty by construction, so the shares are finite.
        let mut incoming: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n + 1];
        for (src, adj) in out.iter().enumerate() {
            let share = 1.0 / adj.len() as f64;
            for &dst in adj {
                incoming[dst as usize].push((src as u32, share));
            }
        }

        let pi = parallel::run_scoped(opts, || {
            let progress = opts.progress;
            progress.start("leader-rank sweeps", Some(self.iterations as u64));

            let m = n + 1;
            let mut pi = vec![1.0 / m as f64; m];
            for it in 0..self.iterations {
                // Each target sums its fixed incoming list in order, so the
                // result is bit-stable at any thread count.
                let frozen = &pi;
                let next = parallel::par_map_indexed(m, |j| {
                    incoming[j]
                        .iter()
                        .map(|&(i, share)| frozen[i as usize] * share)
                        .sum::<f64>()
                });

                let change: f64 = pi.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
                pi = next;
                progress.update(it as u64 + 1);

                if change < self.tolerance {
                    break;
                }
            }
            progress.finish();
            pi
        });

        // Fold the ground mass back evenly; the walk conserves mass, so the
        // real-node scores sum to 1.
        let ground_share = pi[n] / n as f64;
        let scores = pi[..n].iter().map(|s| s + ground_share).collect();

        Ok(LeaderRankModel {
            params: *self,
            names: g.interner.clone(),
            scores,
        })
    }
}

impl Ranker for LeaderRank {
    type Data = GraphDataset;
    type Model = LeaderRankModel;

    fn fit_opts(&self, data: &GraphDataset, opts: &FitOptions<'_>) -> Result<LeaderRankModel> {
        self.fit_view(data.view(), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn scores_of(g: &GraphDataset) -> std::collections::HashMap<String, f64> {
        LeaderRank::default()
            .fit(g)
            .unwrap()
            .scores()
            .map(|(n, s)| (n.to_string(), s))
            .collect()
    }

    /// Single edge 1→2 over nodes {1, 2, 3}, solved by hand.
    ///
    /// Augmented out-adjacency (g = ground): 1:{2,g}, 2:{g}, 3:{g},
    /// g:{1,2,3}. Stationarity gives π_1 = π_g/3, π_2 = π_1/2 + π_g/3,
    /// π_3 = π_g/3, π_g = π_1/2 + π_2 + π_3; with Σπ = 1 that solves to
    /// π = (2/13, 3/13, 2/13, 6/13). Folding the ground mass back,
    /// S_i = π_i + π_g/3 = (4/13, 5/13, 4/13).
    #[test]
    fn three_node_single_edge_matches_hand_solution() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        g.intern("3");

        let s = scores_of(&g);
        for (name, want) in [("1", 4.0 / 13.0), ("2", 5.0 / 13.0), ("3", 4.0 / 13.0)] {
            assert!(
                (s[name] - want).abs() < 1e-9,
                "{name}: {} vs {want}",
                s[name]
            );
        }
    }

    #[test]
    fn scores_sum_to_one() {
        let mut g = GraphDataset::new();
        for (a, b) in [
            ("a", "b"),
            ("b", "c"),
            ("c", "a"),
            ("d", "a"),
            ("d", "b"),
            ("e", "d"),
            ("a", "e"),
        ] {
            g.push(a, b, 1.0);
        }

        let total: f64 = scores_of(&g).values().sum();
        assert!((total - 1.0).abs() < 1e-9, "sum {total}");
    }

    /// 1 ⇄ 2 is symmetric under relabeling, so the scores must be equal.
    #[test]
    fn symmetric_pair_scores_equally() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        g.push("2", "1", 1.0);

        let s = scores_of(&g);
        assert!((s["1"] - s["2"]).abs() < 1e-12);
        assert!((s["1"] - 0.5).abs() < 1e-9);
    }

    /// A lone node (self-loop is the only way to have an edge): all mass
    /// belongs to it once the ground is folded back, S_1 = 1.
    #[test]
    fn single_node_scores_one() {
        let mut g = GraphDataset::new();
        g.push("1", "1", 1.0);

        let s = scores_of(&g);
        assert!((s["1"] - 1.0).abs() < 1e-9, "{}", s["1"]);
    }

    /// Two disjoint components need no connectivity policy: the ground node
    /// bridges them, so every node gets a finite positive score.
    #[test]
    fn disconnected_components_work_without_policy() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("c", "d", 1.0);

        let s = scores_of(&g);
        for name in ["a", "b", "c", "d"] {
            assert!(s[name].is_finite() && s[name] > 0.0, "{name}: {}", s[name]);
        }

        let total: f64 = s.values().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let g = GraphDataset::new();
        assert!(matches!(
            LeaderRank::default().fit(&g),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "c", 1.0);
        g.push("c", "a", 1.0);
        g.push("a", "c", 1.0);

        let m = LeaderRank::default().fit(&g).unwrap();
        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = LeaderRankModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

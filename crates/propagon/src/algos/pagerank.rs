//! PageRank (`docs/algorithms.md` §4.4).
//!
//! Random-surfer importance over an endorsement graph (`src` endorses `dst`),
//! with v1's three sink policies for nodes with no outgoing endorsements.
//! Parallel edges between the same pair are deduplicated (v1 semantics);
//! edge weights are ignored.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{GraphDataset, GraphView};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// How to handle sink nodes (no outgoing edges).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sink {
    /// Sinks bounce their mass back along their incoming edges.
    #[default]
    Reverse,
    /// Sinks distribute their mass to every other node.
    All,
    /// Sinks absorb mass (the L1 norm decays; v1 behavior).
    None,
}

/// PageRank parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PageRank {
    pub damping: f64,
    pub iterations: usize,
    pub sink: Sink,
}

impl Default for PageRank {
    fn default() -> Self {
        Self { damping: 0.85, iterations: 10, sink: Sink::Reverse }
    }
}

/// Fitted PageRank scores.
#[derive(Debug, Clone)]
pub struct PageRankModel {
    params: PageRank,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(PageRankModel, "page-rank");

impl PageRank {
    /// Fits from a borrowed graph view (e.g.
    /// [`PairwiseDataset::as_graph`](crate::PairwiseDataset::as_graph)).
    pub fn fit_view(&self, g: GraphView<'_>, _opts: &FitOptions<'_>) -> Result<PageRankModel> {
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = g.n_nodes();

        // Deduplicated out-adjacency (v1 used HashSet values).
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (s, d, _) in g.edges() {
            out[s as usize].push(d);
        }
        for adj in &mut out {
            adj.sort_unstable();
            adj.dedup();
        }

        let is_sink: Vec<bool> = out.iter().map(|a| a.is_empty()).collect();
        let mut sink_pool: Vec<u32> = Vec::new();
        match self.sink {
            Sink::None => {}
            Sink::Reverse => {
                // Sinks send mass back to their endorsers.
                let mut reverse: Vec<Vec<u32>> = vec![Vec::new(); n];
                for src in 0..n {
                    for &dst in &out[src] {
                        if is_sink[dst as usize] {
                            reverse[dst as usize].push(src as u32);
                        }
                    }
                }
                for (node, mut backs) in reverse.into_iter().enumerate() {
                    if !backs.is_empty() {
                        backs.sort_unstable();
                        backs.dedup();
                        out[node] = backs;
                    }
                }
            }
            Sink::All => {
                sink_pool = (0..n as u32).filter(|&i| is_sink[i as usize]).collect();
            }
        }

        let mut policy = vec![1.0 / n as f64; n];
        let mut next = vec![0.0f64; n];
        for _ in 0..self.iterations {
            next.iter_mut().for_each(|v| *v = 0.0);

            for src in 0..n {
                if out[src].is_empty() {
                    continue;
                }
                let share = policy[src] / out[src].len() as f64;
                for &dst in &out[src] {
                    next[dst as usize] += share;
                }
            }

            if !sink_pool.is_empty() {
                // v1 Sink::All: every node receives the pooled sink mass,
                // minus its own contribution (no self-endorsement).
                let pooled: f64 =
                    sink_pool.iter().map(|&v| policy[v as usize]).sum::<f64>() / (n - 1) as f64;
                for (node, value) in next.iter_mut().enumerate() {
                    *value += pooled;
                    if is_sink[node] {
                        *value -= policy[node] / (n - 1) as f64;
                    }
                }
            }

            for value in next.iter_mut() {
                *value = *value * self.damping + (1.0 - self.damping) / n as f64;
            }
            std::mem::swap(&mut policy, &mut next);
        }

        Ok(PageRankModel { params: *self, names: g.interner.clone(), scores: policy })
    }
}

impl Ranker for PageRank {
    type Data = GraphDataset;
    type Model = PageRankModel;

    fn fit_opts(&self, data: &GraphDataset, opts: &FitOptions<'_>) -> Result<PageRankModel> {
        self.fit_view(data.view(), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// v1's test graph, translated to endorsement edges: v1 read each
    /// `(winner, loser)` match as `loser endorses winner`.
    fn graph() -> GraphDataset {
        let mut g = GraphDataset::new();
        for (winner, loser) in
            [("1", "2"), ("3", "2"), ("1", "3"), ("1", "4"), ("2", "4"), ("3", "4")]
        {
            g.push(loser, winner, 1.0);
        }
        g
    }

    fn scores(m: &PageRankModel) -> Vec<(String, f64)> {
        m.sorted_scores().into_iter().map(|(n, s)| (n.to_string(), s)).collect()
    }

    /// v1 `test_example`: 1 iteration, Sink::None.
    #[test]
    fn one_iteration_no_sink() {
        let pr = PageRank { damping: 0.85, iterations: 1, sink: Sink::None };
        let result = scores(&pr.fit(&graph()).unwrap());
        let expected = [("1", 0.427083), ("3", 0.214583), ("2", 0.108333), ("4", 0.0375)];
        for ((name, score), (want_name, want)) in result.iter().zip(expected) {
            assert_eq!(name, want_name);
            assert!((score - want).abs() < 1e-4, "{name}: {score} vs {want}");
        }
    }

    /// v1 `test_reverse` and `test_all_links`: same fixed point.
    #[test]
    fn reverse_and_all_sinks_converge_to_v1_values() {
        for sink in [Sink::Reverse, Sink::All] {
            let pr = PageRank { damping: 0.85, iterations: 10, sink };
            let result = scores(&pr.fit(&graph()).unwrap());
            let expected = [("1", 0.39064), ("3", 0.27099), ("2", 0.190172), ("4", 0.14818)];
            for ((name, score), (want_name, want)) in result.iter().zip(expected) {
                assert_eq!(name, want_name, "{sink:?}");
                assert!((score - want).abs() < 1e-4, "{sink:?} {name}: {score} vs {want}");
            }
            let total: f64 = result.iter().map(|e| e.1).sum();
            assert!((total - 1.0).abs() < 1e-5, "{sink:?} sums to 1");
        }
    }
}

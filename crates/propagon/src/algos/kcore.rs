//! k-core decomposition (`docs/algorithms.md` §4.11; Seidman 1983,
//! Batagelj & Zaversnik 2003).
//!
//! A node's **coreness** is the deepest k-core it survives: iteratively
//! strip nodes of degree < k and whatever remains is the k-core. Predicts
//! spreading power better than degree or betweenness (Kitsak et al. 2010)
//! and pairs naturally with `extract-components` as a graph triage tool.
//!
//! Assumes an **undirected, unweighted** reading of the graph: edge
//! direction and weights are ignored, parallel edges and self-loops
//! deduplicated/dropped. Coreness values are small integers reported as
//! scores.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::GraphDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// k-core has no tunable parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KCore {}

/// Fitted coreness per node.
#[derive(Debug, Clone)]
pub struct KCoreModel {
    params: KCore,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(KCoreModel, "k-core");

impl Ranker for KCore {
    type Data = GraphDataset;
    type Model = KCoreModel;

    fn fit_opts(&self, data: &GraphDataset, _opts: &FitOptions<'_>) -> Result<KCoreModel> {
        let g = data.view();
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = g.n_nodes();
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];

        for (s, d, _) in g.edges() {
            if s != d {
                adjacency[s as usize].push(d);
                adjacency[d as usize].push(s);
            }
        }
        for adj in &mut adjacency {
            adj.sort_unstable();
            adj.dedup();
        }

        // Peel: repeatedly remove everything of degree ≤ k (cascading)
        // before moving to k + 1; a node's coreness is the k it fell at.
        let mut degree: Vec<usize> = adjacency.iter().map(Vec::len).collect();
        let mut removed = vec![false; n];
        let mut coreness = vec![0u32; n];
        let mut remaining = n;
        let mut k = 0usize;

        while remaining > 0 {
            let mut queue: Vec<u32> = (0..n as u32)
                .filter(|&v| !removed[v as usize] && degree[v as usize] <= k)
                .collect();

            while let Some(v) = queue.pop() {
                let v = v as usize;
                if removed[v] {
                    continue;
                }
                removed[v] = true;
                remaining -= 1;
                coreness[v] = k as u32;

                for &u in &adjacency[v] {
                    let u = u as usize;
                    if !removed[u] {
                        degree[u] -= 1;
                        if degree[u] <= k {
                            queue.push(u as u32);
                        }
                    }
                }
            }
            k += 1;
        }

        Ok(KCoreModel {
            params: *self,
            names: g.interner.clone(),
            scores: coreness.into_iter().map(f64::from).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// Textbook fixture: a 4-clique (coreness 3) with a triangle hanging
    /// off one corner (coreness 2) and a pendant vertex (coreness 1).
    #[test]
    fn clique_triangle_pendant() {
        let mut g = GraphDataset::new();
        // 4-clique p q r s.
        for (a, b) in [
            ("p", "q"),
            ("p", "r"),
            ("p", "s"),
            ("q", "r"),
            ("q", "s"),
            ("r", "s"),
        ] {
            g.push(a, b, 1.0);
        }
        // Triangle s t u (s bridges).
        g.push("s", "t", 1.0);
        g.push("t", "u", 1.0);
        g.push("u", "s", 1.0);
        // Pendant v off u.
        g.push("u", "v", 1.0);

        let m = KCore::default().fit(&g).unwrap();
        let c: std::collections::HashMap<&str, f64> = m.scores().collect();

        for node in ["p", "q", "r", "s"] {
            assert_eq!(c[node], 3.0, "{node}");
        }
        assert_eq!(c["t"], 2.0);
        assert_eq!(c["u"], 2.0);
        assert_eq!(c["v"], 1.0);
    }

    /// Direction and parallel edges are ignored.
    #[test]
    fn undirected_deduplicated() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "a", 1.0);
        g.push("a", "b", 1.0);

        let m = KCore::default().fit(&g).unwrap();
        for (_, s) in m.scores() {
            assert_eq!(s, 1.0);
        }
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("b", "c", 1.0);
        g.push("c", "a", 1.0);

        let m = KCore::default().fit(&g).unwrap();
        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = KCoreModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

//! PageRank, personalized PageRank & random walk with restart
//! (`docs/algorithms.md` §4.4).
//!
//! Random-surfer importance over an endorsement graph (`src` endorses `dst`),
//! with v1's three sink policies for nodes with no outgoing endorsements.
//! Parallel edges between the same pair are deduplicated (v1 semantics);
//! edge weights are ignored.
//!
//! [`Teleport::Seeds`] concentrates the restart distribution on a seed set,
//! turning the same solver into personalized PageRank / random walk with
//! restart: scores become importance *as seen from the seeds*. TrustRank is
//! a recipe on top of this — teleport uniformly over a manually vetted
//! trusted set and spam unreachable from it scores near zero.
//!
//! Gotcha: under seeded teleport, [`Sink::All`] keeps its v1 semantics
//! (uniform over the other nodes) and therefore ignores personalization;
//! [`Sink::Uniform`] redistributes sink mass proportional to the teleport
//! vector (the textbook personalized dangling fix).

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
    /// Sinks distribute their mass to every other node (v1 semantics:
    /// the sink itself is excluded from its own redistribution; ignores
    /// personalization).
    All,
    /// The textbook treatment: a sink's row becomes the teleport
    /// distribution — uniform over **all** nodes, itself included, under
    /// [`Teleport::Uniform`] (Langville & Meyer's dangling-node fix),
    /// proportional to the seed weights under [`Teleport::Seeds`].
    Uniform,
    /// Sinks absorb mass (the L1 norm decays; v1 behavior).
    None,
}

/// Where the surfer restarts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Teleport {
    /// Restart anywhere: classic global PageRank.
    #[default]
    Uniform,
    /// Restart at the named seeds with the given positive weights
    /// (normalized internally): personalized PageRank / random walk with
    /// restart.
    Seeds(Vec<(String, f64)>),
}

/// PageRank parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PageRank {
    /// Probability the surfer follows an edge rather than teleporting.
    pub damping: f64,
    /// Number of power-iteration passes.
    pub iterations: usize,
    /// How mass at sink nodes (no outgoing edges) is handled.
    pub sink: Sink,
    /// Where the surfer restarts (uniform, or concentrated on seeds).
    pub teleport: Teleport,
}

impl Default for PageRank {
    fn default() -> Self {
        Self {
            damping: 0.85,
            iterations: 10,
            sink: Sink::Reverse,
            teleport: Teleport::Uniform,
        }
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

        // `None` = uniform restart, kept implicit so the classic path
        // reproduces v1's float operations exactly (golden stability).
        let seeds: Option<Vec<f64>> = match &self.teleport {
            Teleport::Uniform => None,
            Teleport::Seeds(list) => {
                if list.is_empty() {
                    return Err(Error::InvalidInput(
                        "teleport seed list is empty; use Teleport::Uniform for the classic walk"
                            .into(),
                    ));
                }
                let mut v = vec![0.0f64; n];
                for (name, weight) in list {
                    let id = g.interner.get(name).ok_or_else(|| {
                        Error::InvalidInput(format!("teleport seed '{name}' is not in the graph"))
                    })?;
                    if !(*weight > 0.0 && weight.is_finite()) {
                        return Err(Error::InvalidInput(format!(
                            "teleport seed '{name}' needs a positive finite weight, got {weight}"
                        )));
                    }
                    v[id as usize] += weight;
                }
                let total: f64 = v.iter().sum();
                v.iter_mut().for_each(|w| *w /= total);
                Some(v)
            }
        };

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
                for (src, adj) in out.iter().enumerate() {
                    for &dst in adj {
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
            Sink::All | Sink::Uniform => {
                sink_pool = (0..n as u32).filter(|&i| is_sink[i as usize]).collect();
            }
        }

        let mut policy = match &seeds {
            None => vec![1.0 / n as f64; n],
            Some(v) => v.clone(),
        };
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
                let sink_mass: f64 = sink_pool.iter().map(|&v| policy[v as usize]).sum();

                match self.sink {
                    // v1 Sink::All: every node receives the pooled sink mass,
                    // minus its own contribution (no self-endorsement).
                    Sink::All => {
                        let pooled = sink_mass / (n - 1) as f64;
                        for (node, value) in next.iter_mut().enumerate() {
                            *value += pooled;
                            if is_sink[node] {
                                *value -= policy[node] / (n - 1) as f64;
                            }
                        }
                    }
                    // Textbook: a sink's row becomes the teleport
                    // distribution (uniform over all nodes, self included,
                    // in the classic case).
                    Sink::Uniform => match &seeds {
                        None => {
                            let pooled = sink_mass / n as f64;
                            for value in next.iter_mut() {
                                *value += pooled;
                            }
                        }
                        Some(v) => {
                            for (value, &w) in next.iter_mut().zip(v) {
                                *value += sink_mass * w;
                            }
                        }
                    },
                    Sink::Reverse | Sink::None => {}
                }
            }

            match &seeds {
                None => {
                    for value in next.iter_mut() {
                        *value = *value * self.damping + (1.0 - self.damping) / n as f64;
                    }
                }
                Some(v) => {
                    for (value, &w) in next.iter_mut().zip(v) {
                        *value = *value * self.damping + (1.0 - self.damping) * w;
                    }
                }
            }
            std::mem::swap(&mut policy, &mut next);
        }

        Ok(PageRankModel {
            params: self.clone(),
            names: g.interner.clone(),
            scores: policy,
        })
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
        for (winner, loser) in [
            ("1", "2"),
            ("3", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
        ] {
            g.push(loser, winner, 1.0);
        }
        g
    }

    fn scores(m: &PageRankModel) -> Vec<(String, f64)> {
        m.sorted_scores()
            .into_iter()
            .map(|(n, s)| (n.to_string(), s))
            .collect()
    }

    /// v1 `test_example`: 1 iteration, Sink::None.
    #[test]
    fn one_iteration_no_sink() {
        let pr = PageRank {
            damping: 0.85,
            iterations: 1,
            sink: Sink::None,
            ..PageRank::default()
        };
        let result = scores(&pr.fit(&graph()).unwrap());
        let expected = [
            ("1", 0.427083),
            ("3", 0.214583),
            ("2", 0.108333),
            ("4", 0.0375),
        ];
        for ((name, score), (want_name, want)) in result.iter().zip(expected) {
            assert_eq!(name, want_name);
            assert!((score - want).abs() < 1e-4, "{name}: {score} vs {want}");
        }
    }

    /// v1 `test_reverse` and `test_all_links`: same fixed point.
    #[test]
    fn reverse_and_all_sinks_converge_to_v1_values() {
        for sink in [Sink::Reverse, Sink::All] {
            let pr = PageRank {
                damping: 0.85,
                iterations: 10,
                sink,
                ..PageRank::default()
            };
            let result = scores(&pr.fit(&graph()).unwrap());
            let expected = [
                ("1", 0.39064),
                ("3", 0.27099),
                ("2", 0.190172),
                ("4", 0.14818),
            ];
            for ((name, score), (want_name, want)) in result.iter().zip(expected) {
                assert_eq!(name, want_name, "{sink:?}");
                assert!(
                    (score - want).abs() < 1e-4,
                    "{sink:?} {name}: {score} vs {want}"
                );
            }
            let total: f64 = result.iter().map(|e| e.1).sum();
            assert!((total - 1.0).abs() < 1e-5, "{sink:?} sums to 1");
        }
    }

    /// Analytic personalized PageRank on the 3-cycle 1→2→3→1 with the
    /// restart concentrated on node 1:
    ///   p₂ = d·p₁, p₃ = d·p₂, p₁ = d·p₃ + (1−d)
    ///   ⇒ p₁ = (1−d)/(1−d³); at d = ½: p = (4/7, 2/7, 1/7).
    #[test]
    fn seeded_teleport_matches_analytic_cycle() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        g.push("2", "3", 1.0);
        g.push("3", "1", 1.0);
        let pr = PageRank {
            damping: 0.5,
            iterations: 100,
            sink: Sink::None,
            teleport: Teleport::Seeds(vec![("1".into(), 1.0)]),
        };
        let m = pr.fit(&g).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        for (node, want) in [("1", 4.0 / 7.0), ("2", 2.0 / 7.0), ("3", 1.0 / 7.0)] {
            assert!(
                (s[node] - want).abs() < 1e-9,
                "{node}: {} vs {want}",
                s[node]
            );
        }
    }

    /// Seeded teleport + Sink::Uniform: sink mass restarts at the seeds.
    /// Graph 1→2 (2 is a sink), seed = {1}:
    ///   p₁ = d·p₂ + (1−d), p₂ = d·p₁ ⇒ p₁ = 1/(1+d); at d = ½: (⅔, ⅓).
    #[test]
    fn seeded_teleport_with_uniform_sink_matches_analytic_chain() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        let pr = PageRank {
            damping: 0.5,
            iterations: 200,
            sink: Sink::Uniform,
            teleport: Teleport::Seeds(vec![("1".into(), 1.0)]),
        };
        let m = pr.fit(&g).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["1"] - 2.0 / 3.0).abs() < 1e-9, "p1 = {}", s["1"]);
        assert!((s["2"] - 1.0 / 3.0).abs() < 1e-9, "p2 = {}", s["2"]);
    }

    /// Seed weights normalize; multiple seeds split the restart mass.
    #[test]
    fn seed_weights_are_normalized() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        g.push("2", "1", 1.0);
        let symmetric = PageRank {
            teleport: Teleport::Seeds(vec![("1".into(), 5.0), ("2".into(), 5.0)]),
            ..PageRank::default()
        };
        let m = symmetric.fit(&g).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["1"] - 0.5).abs() < 1e-12);
        assert!((s["2"] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn seed_validation_errors() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        for seeds in [
            vec![],
            vec![("ghost".to_string(), 1.0)],
            vec![("1".to_string(), 0.0)],
            vec![("1".to_string(), f64::NAN)],
        ] {
            let pr = PageRank {
                teleport: Teleport::Seeds(seeds.clone()),
                ..PageRank::default()
            };
            assert!(
                matches!(pr.fit(&g), Err(Error::InvalidInput(_))),
                "{seeds:?}"
            );
        }
    }

    /// The seeded model round-trips byte-identically (Teleport rides in
    /// the params object).
    #[test]
    fn seeded_round_trip_is_byte_identical() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        g.push("2", "1", 1.0);
        let pr = PageRank {
            teleport: Teleport::Seeds(vec![("1".into(), 1.0)]),
            ..PageRank::default()
        };
        let m = pr.fit(&g).unwrap();
        let mut buf1 = Vec::new();
        m.save_jsonl(&mut buf1).unwrap();
        let m2 = PageRankModel::load_jsonl(buf1.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf1, buf2);
    }
}

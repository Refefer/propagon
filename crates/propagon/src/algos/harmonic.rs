//! Harmonic centrality (`docs/algorithms.md` §4.10; Marchiori & Latora 2000;
//! Boldi & Vigna 2014).
//!
//! `H(i) = Σ_{j≠i} 1/d(j→i)` under [`Direction::In`] (the endorsement
//! reading: "how easily does everyone reach you"), `Σ_{j≠i} 1/d(i→j)` under
//! `Out`, and undirected distances under `Total`. Unreachable pairs
//! contribute exactly 0 (`1/∞`), so disconnected graphs need no special
//! policy — the property that makes harmonic the only centrality satisfying
//! all of Boldi-Vigna's axioms.
//!
//! One SSSP pass per source j — BFS for [`EdgeCost::Unit`], binary-heap
//! Dijkstra for [`EdgeCost::Weight`] — scattering `1/d(j→·)` into the
//! accumulator. For `Out` the passes run over the *reversed* adjacency, so a
//! pass from j yields `d(i→j)` for every i and each node's score is still
//! covered when only a sample of passes runs. Time is O(V·E) unweighted
//! (O(V·(E + V log V)) weighted) with an O(V) distance scratch per in-flight
//! source; per-source contribution lists are merged sequentially in
//! source-index order, so results are bit-stable at any thread count.
//!
//! Gotchas: under `EdgeCost::Weight` an edge weight is a **length** (cost to
//! traverse), so a larger weight means *farther* and a smaller contribution
//! — the opposite of the endorsement-strength reading used by the weighted
//! eigenvector family; weights must be strictly positive. Under
//! [`SourceBudget::Sample`] scores are scaled by n/count (the Boldi-Vigna
//! unbiased estimator), and for `Direction::Out` the sampled passes are the
//! *targets* j of `d(i→j)` — every node still gets a score, estimated from a
//! subset of its destinations.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::algos::degree::Direction;
use crate::dataset::{GraphDataset, GraphView};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, Ranker};

/// How an edge enters the distance computation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EdgeCost {
    /// Every edge has length 1 (BFS distances); weights are ignored.
    #[default]
    Unit,
    /// The edge weight is its length (Dijkstra distances); must be > 0.
    Weight,
}

/// How many SSSP passes to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceBudget {
    /// Exact: one pass per node.
    #[default]
    All,
    /// Boldi-Vigna estimator: `count` distinct sources drawn with the seeded
    /// generator, accumulated scores scaled by n/count.
    Sample { count: usize, seed: u64 },
}

/// Harmonic-centrality parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Harmonic {
    /// Which way distance flows: `In` sums `1/d(j→i)` over reachers,
    /// `Out` sums `1/d(i→j)` over destinations, `Total` uses undirected
    /// distances.
    pub direction: Direction,
    /// Unit hops or weighted lengths.
    pub cost: EdgeCost,
    /// Exact computation or seeded source sampling.
    pub sources: SourceBudget,
}

/// Fitted harmonic centralities (≥ 0; higher is more central).
#[derive(Debug, Clone)]
pub struct HarmonicModel {
    params: Harmonic,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(HarmonicModel, "harmonic");

/// Pass adjacency with the orientation already baked in: forward edges for
/// `In`, reversed for `Out`, both for `Total`.
enum Paths {
    Unit(Vec<Vec<u32>>),
    Weighted(Vec<Vec<(u32, f64)>>),
}

impl Harmonic {
    /// Canonical seed for [`SourceBudget::Sample`] when the caller has no
    /// opinion (the CLI defers here; the library has no implicit default
    /// because `Sample` is not the default variant).
    pub const DEFAULT_SAMPLE_SEED: u64 = 2014;

    /// Fits from a borrowed graph view (e.g.
    /// [`PairwiseDataset::as_graph`](crate::PairwiseDataset::as_graph)).
    ///
    /// Resolves the source budget into a sorted pass list, runs one SSSP per
    /// pass in parallel, then merges the per-pass contribution lists
    /// sequentially in source order and applies the n/count sample scale.
    pub fn fit_view(&self, g: GraphView<'_>, opts: &FitOptions<'_>) -> Result<HarmonicModel> {
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = g.n_nodes();

        let (sources, scale) = self.choose_sources(n)?;
        let paths = self.build_paths(&g)?;

        let scores = parallel::run_scoped(opts, || {
            let progress = opts.progress;
            progress.start("harmonic sssp passes", Some(sources.len() as u64));

            let contribs = match &paths {
                Paths::Unit(adj) => {
                    parallel::par_map_indexed(sources.len(), |k| bfs_scatter(adj, sources[k]))
                }
                Paths::Weighted(adj) => {
                    parallel::par_map_indexed(sources.len(), |k| dijkstra_scatter(adj, sources[k]))
                }
            };

            // Sequential source-order merge: float accumulation order is
            // fixed, so results are bit-stable at any thread count.
            let mut h = vec![0.0f64; n];
            for (k, contrib) in contribs.into_iter().enumerate() {
                for (node, inv_d) in contrib {
                    h[node as usize] += inv_d;
                }
                progress.update(k as u64 + 1);
            }
            progress.finish();

            // `scale` is exactly 1.0 for All (and full-count samples), so
            // this multiplication leaves those bit-identical.
            for v in &mut h {
                *v *= scale;
            }
            h
        });

        Ok(HarmonicModel {
            params: *self,
            names: g.interner.clone(),
            scores,
        })
    }

    /// Resolves the source budget into a sorted pass list plus the
    /// Boldi-Vigna scale factor n/|passes|.
    ///
    /// Sampling draws a uniform distinct prefix via seeded Fisher-Yates and
    /// then sorts it, so the sequential merge always accumulates in node
    /// order — `Sample { count: n }` is bit-identical to `All`.
    fn choose_sources(&self, n: usize) -> Result<(Vec<u32>, f64)> {
        match self.sources {
            SourceBudget::All => Ok(((0..n as u32).collect(), 1.0)),
            SourceBudget::Sample { count, seed } => {
                if count == 0 {
                    return Err(Error::InvalidInput(
                        "harmonic sample count must be at least 1".into(),
                    ));
                }
                let count = count.min(n);

                let mut ids: Vec<u32> = (0..n as u32).collect();
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                for k in 0..count {
                    let j = rng.random_range(k..ids.len());
                    ids.swap(k, j);
                }
                ids.truncate(count);
                ids.sort_unstable();

                Ok((ids, n as f64 / count as f64))
            }
        }
    }

    /// Builds the pass adjacency for the configured direction and cost.
    ///
    /// A pass from j walks forward edges for `In` (computing `d(j→i)`),
    /// reversed edges for `Out` (the pass from j yields `d(i→j)` for every
    /// i), and both for `Total`. Under `EdgeCost::Weight` every edge weight
    /// must be strictly positive (Dijkstra's precondition); the offending
    /// edge is named otherwise.
    fn build_paths(&self, g: &GraphView<'_>) -> Result<Paths> {
        let n = g.n_nodes();

        match self.cost {
            EdgeCost::Unit => {
                let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
                for (s, d, _) in g.edges() {
                    match self.direction {
                        Direction::In => adj[s as usize].push(d),
                        Direction::Out => adj[d as usize].push(s),
                        Direction::Total => {
                            adj[s as usize].push(d);
                            adj[d as usize].push(s);
                        }
                    }
                }
                Ok(Paths::Unit(adj))
            }
            EdgeCost::Weight => {
                let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
                for (s, d, w) in g.edges() {
                    let w = f64::from(w);
                    if w <= 0.0 || w.is_nan() {
                        return Err(Error::InvalidInput(format!(
                            "harmonic with weighted edge cost requires strictly \
                             positive weights; edge {} → {} has weight {w}",
                            g.interner.resolve(s),
                            g.interner.resolve(d),
                        )));
                    }

                    match self.direction {
                        Direction::In => adj[s as usize].push((d, w)),
                        Direction::Out => adj[d as usize].push((s, w)),
                        Direction::Total => {
                            adj[s as usize].push((d, w));
                            adj[d as usize].push((s, w));
                        }
                    }
                }
                Ok(Paths::Weighted(adj))
            }
        }
    }
}

impl Ranker for Harmonic {
    type Data = GraphDataset;
    type Model = HarmonicModel;

    fn fit_opts(&self, data: &GraphDataset, opts: &FitOptions<'_>) -> Result<HarmonicModel> {
        self.fit_view(data.view(), opts)
    }
}

/// One BFS pass from `src`: `(node, 1/d(src→node))` for every node reached
/// at distance ≥ 1, in visit order.
fn bfs_scatter(adj: &[Vec<u32>], src: u32) -> Vec<(u32, f64)> {
    // u32::MAX is a safe "unvisited" sentinel: hop counts top out at n − 1,
    // and the interner caps n at u32 range.
    let mut dist = vec![u32::MAX; adj.len()];
    let mut queue = VecDeque::new();
    dist[src as usize] = 0;
    queue.push_back(src);

    let mut out = Vec::new();
    while let Some(u) = queue.pop_front() {
        let next = dist[u as usize] + 1;

        for &v in &adj[u as usize] {
            if dist[v as usize] == u32::MAX {
                dist[v as usize] = next;
                out.push((v, 1.0 / f64::from(next)));
                queue.push_back(v);
            }
        }
    }
    out
}

/// `f64` distance with a total order (`total_cmp`) so it can key a
/// [`BinaryHeap`].
struct HeapDist(f64);

impl PartialEq for HeapDist {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for HeapDist {}

impl Ord for HeapDist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for HeapDist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// One Dijkstra pass from `src`: `(node, 1/d(src→node))` for every node
/// settled at positive distance, in settle order.
///
/// Stale heap entries are skipped by the `du > dist` guard; each settled
/// node is emitted exactly once because re-pushes require a strict distance
/// improvement. Ties in distance settle in node-id order, keeping the
/// contribution list deterministic.
fn dijkstra_scatter(adj: &[Vec<(u32, f64)>], src: u32) -> Vec<(u32, f64)> {
    let mut dist = vec![f64::INFINITY; adj.len()];
    let mut heap: BinaryHeap<Reverse<(HeapDist, u32)>> = BinaryHeap::new();
    dist[src as usize] = 0.0;
    heap.push(Reverse((HeapDist(0.0), src)));

    let mut out = Vec::new();
    while let Some(Reverse((HeapDist(du), u))) = heap.pop() {
        if du > dist[u as usize] {
            continue;
        }

        if u != src {
            out.push((u, 1.0 / du));
        }

        for &(v, w) in &adj[u as usize] {
            let nd = du + w;

            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(Reverse((HeapDist(nd), v)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn fit(algo: Harmonic, g: &GraphDataset) -> std::collections::HashMap<String, f64> {
        algo.fit(g)
            .unwrap()
            .scores()
            .map(|(n, s)| (n.to_string(), s))
            .collect()
    }

    /// Directed path 1→2→3, unit cost.
    ///
    /// In: H(1) = 0 (no one reaches 1), H(2) = 1/d(1→2) = 1,
    /// H(3) = 1/d(2→3) + 1/d(1→3) = 1 + 1/2 = 1.5. Out mirrors it.
    #[test]
    fn directed_path_in_and_out() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 1.0);
        g.push("2", "3", 1.0);

        let h = fit(Harmonic::default(), &g);
        assert_eq!((h["1"], h["2"], h["3"]), (0.0, 1.0, 1.5));

        let h = fit(
            Harmonic {
                direction: Direction::Out,
                ..Harmonic::default()
            },
            &g,
        );
        assert_eq!((h["1"], h["2"], h["3"]), (1.5, 1.0, 0.0));
    }

    /// In-star a→c, b→c, d→c.
    ///
    /// In: H(c) = 3 (each leaf at distance 1); a leaf is reached by nobody
    /// (edges only point at c), so H(leaf) = 0. Total (undirected):
    /// H(c) = 3 and H(leaf) = 1/1 + 1/2 + 1/2 = 2 (c adjacent, the other
    /// leaves two hops away through c).
    #[test]
    fn star_in_and_total() {
        let mut g = GraphDataset::new();
        g.push("a", "c", 1.0);
        g.push("b", "c", 1.0);
        g.push("d", "c", 1.0);

        let h = fit(Harmonic::default(), &g);
        assert_eq!(h["c"], 3.0);
        for leaf in ["a", "b", "d"] {
            assert_eq!(h[leaf], 0.0, "{leaf}");
        }

        let h = fit(
            Harmonic {
                direction: Direction::Total,
                ..Harmonic::default()
            },
            &g,
        );
        assert_eq!(h["c"], 3.0);
        for leaf in ["a", "b", "d"] {
            assert_eq!(h[leaf], 2.0, "{leaf}");
        }
    }

    /// Weighted 1→2 (w=2), 2→3 (w=3), 1→3 (w=10): the two-hop route wins,
    /// d(1→3) = min(10, 2+3) = 5, so In-harmonic of 3 is 1/3 + 1/5; of 2 is
    /// 1/d(1→2) = 1/2; of 1 is 0.
    #[test]
    fn weighted_shortcut_loses_to_two_hop_route() {
        let mut g = GraphDataset::new();
        g.push("1", "2", 2.0);
        g.push("2", "3", 3.0);
        g.push("1", "3", 10.0);

        let h = fit(
            Harmonic {
                cost: EdgeCost::Weight,
                ..Harmonic::default()
            },
            &g,
        );
        assert!(
            (h["3"] - (1.0 / 3.0 + 1.0 / 5.0)).abs() < 1e-12,
            "{}",
            h["3"]
        );
        assert!((h["2"] - 0.5).abs() < 1e-12);
        assert_eq!(h["1"], 0.0);
    }

    /// A full-size sample is the same sorted source list with scale
    /// n/n = 1, so it must match All bit for bit.
    #[test]
    fn full_sample_equals_all_exactly() {
        let mut g = GraphDataset::new();
        for (a, b) in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("b", "e")] {
            g.push(a, b, 1.0);
        }
        let n = g.n_nodes();

        let exact = fit(Harmonic::default(), &g);
        let sampled = fit(
            Harmonic {
                sources: SourceBudget::Sample { count: n, seed: 42 },
                ..Harmonic::default()
            },
            &g,
        );
        assert_eq!(exact, sampled);
    }

    /// Two disjoint components: unreachable pairs contribute 0 and
    /// everything stays finite — no policy needed.
    #[test]
    fn disconnected_components_contribute_zero() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);
        g.push("c", "d", 1.0);

        let h = fit(Harmonic::default(), &g);
        assert_eq!((h["a"], h["b"], h["c"], h["d"]), (0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let g = GraphDataset::new();
        assert!(matches!(
            Harmonic::default().fit(&g),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn non_positive_weight_is_invalid_under_weighted_cost() {
        for bad in [0.0, -1.0] {
            let mut g = GraphDataset::new();
            g.push("a", "b", bad);

            let r = Harmonic {
                cost: EdgeCost::Weight,
                ..Harmonic::default()
            }
            .fit(&g);
            assert!(matches!(r, Err(Error::InvalidInput(_))), "weight {bad}");
        }
    }

    #[test]
    fn zero_sample_count_is_invalid() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 1.0);

        let r = Harmonic {
            sources: SourceBudget::Sample { count: 0, seed: 7 },
            ..Harmonic::default()
        }
        .fit(&g);
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 2.0);
        g.push("b", "c", 1.0);
        g.push("c", "a", 4.0);

        let m = Harmonic {
            direction: Direction::Total,
            cost: EdgeCost::Weight,
            sources: SourceBudget::Sample { count: 2, seed: 9 },
        }
        .fit(&g)
        .unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = HarmonicModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

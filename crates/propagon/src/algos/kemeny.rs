//! Kemeny-optimal consensus ranking (`docs/algorithms.md` §6.3).
//!
//! Finds the total order minimizing pairwise disagreement with the data —
//! the maximum-likelihood consensus under the Condorcet noise model, NP-hard
//! exactly. Two heuristics, as in v1: repeated best-position **insertion**
//! passes, or **differential evolution** over real-valued position scores.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::de::{DifferentialEvolution, Fitness};
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Search strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KemenyAlgo {
    #[default]
    Insertion,
    DiffEvo,
}

/// Search budget: how many insertion passes / DE fitness evaluations to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KemenyPasses {
    /// The canonical budget for the chosen algorithm: 1 insertion pass,
    /// or 50,000 DE evaluations.
    #[default]
    Auto,
    Fixed(usize),
}

/// Kemeny parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kemeny {
    pub passes: KemenyPasses,
    /// Entities with fewer comparisons than this are dropped from output.
    pub min_obs: usize,
    pub algo: KemenyAlgo,
    /// Seed for the DE search (v1 hardcoded 2020).
    pub seed: u64,
}

impl Default for Kemeny {
    fn default() -> Self {
        Self {
            passes: KemenyPasses::Auto,
            min_obs: 1,
            algo: KemenyAlgo::Insertion,
            seed: 2020,
        }
    }
}

impl Kemeny {
    fn effective_passes(&self) -> usize {
        match (self.passes, self.algo) {
            (KemenyPasses::Fixed(n), _) => n,
            (KemenyPasses::Auto, KemenyAlgo::Insertion) => 1,
            (KemenyPasses::Auto, KemenyAlgo::DiffEvo) => 50_000,
        }
    }

    /// v1 `insertion_kem`: repeated best-position insertion passes.
    fn insertion(&self, graph: &PrefGraph, progress: &dyn crate::Progress) -> Vec<usize> {
        let lookup: Vec<HashMap<usize, (usize, usize)>> = graph
            .iter()
            .map(|comps| comps.iter().map(|&(o, wn)| (o, wn)).collect())
            .collect();

        // Concordant pairs contributed by placing `idx` relative to `other`.
        let score = |idx: usize, other: usize, other_to_left: bool| -> usize {
            match lookup[idx].get(&other) {
                Some((wins, n)) => {
                    if other_to_left {
                        *wins
                    } else {
                        n - wins
                    }
                }
                None => 0,
            }
        };

        // Initial order: majority-matchups won, descending; ties broken by
        // descending first-appearance index (v1's stable sort + reverse).
        let mut old_policy: Vec<usize> = {
            let mut s: Vec<(usize, usize)> = graph
                .iter()
                .enumerate()
                .map(|(idx, comps)| {
                    let majorities = comps.iter().filter(|(_, (w, n))| *w * 2 > *n).count();
                    (idx, majorities)
                })
                .collect();
            s.sort_by_key(|&(idx, wins)| (std::cmp::Reverse(wins), std::cmp::Reverse(idx)));
            s.into_iter().map(|(idx, _)| idx).collect()
        };

        let passes = self.effective_passes();
        progress.start("insertion passes", Some(passes as u64));
        let mut policy: Vec<usize> = Vec::with_capacity(graph.len());
        let mut last_score = 0;
        for pass in 0..passes {
            policy.clear();
            let mut cur_score = 0;
            for idx in old_policy.iter().copied() {
                // Score of appending at the far left, then slide right.
                let mut total: usize = policy.iter().map(|&o| score(idx, o, false)).sum();
                let mut best_pos = 0;
                let mut best_score = total;
                for pos in 1..=policy.len() {
                    let other = policy[pos - 1];
                    total = total - score(idx, other, false) + score(idx, other, true);
                    if total > best_score {
                        best_pos = pos;
                        best_score = total;
                    }
                }
                cur_score += best_score;
                policy.insert(best_pos, idx);
            }
            progress.update(pass as u64 + 1);
            progress.message(&format!("concordant {cur_score}"));
            std::mem::swap(&mut old_policy, &mut policy);
            if last_score == cur_score {
                break;
            }
            last_score = cur_score;
        }
        progress.finish();

        // v1 returns reversed insertion order = best first.
        old_policy.reverse();
        old_policy
    }

    /// v1 `de_kem`: differential evolution over real-valued position scores.
    fn diff_evo(&self, graph: &PrefGraph, progress: &dyn crate::Progress) -> Result<Vec<usize>> {
        let n = graph.len();
        let de = DifferentialEvolution {
            dims: n,
            lambda: ((n as f32).powf(0.7) as usize).max(3),
            f: (0.1, 0.9),
            cr: 0.9,
            m: 0.1,
            exp: 3.0,
            polish_on_stale: 10_000,
            restart_on_stale: 0,
            range: 1.0,
        };

        let passes = self.effective_passes();
        progress.start("de evaluations", Some(passes as u64));
        let fit = KemenyFit { graph };
        let (_best, weights) = de.fit(&fit, passes, self.seed, None, |best, remaining| {
            progress.update((passes - remaining) as u64);
            progress.message(&format!("concordant {best:0.0}"));
        })?;
        progress.finish();

        let mut order: Vec<(usize, f32)> = weights.into_iter().enumerate().collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Ok(order.into_iter().map(|(idx, _)| idx).collect())
    }
}

impl Ranker for Kemeny {
    type Data = PairwiseDataset;
    type Model = KemenyModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<KemenyModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let graph = build_graph(data);
        let order_idx = match self.algo {
            KemenyAlgo::Insertion => self.insertion(&graph, opts.progress),
            KemenyAlgo::DiffEvo => {
                crate::parallel::run_scoped(opts, || self.diff_evo(&graph, opts.progress))?
            }
        };

        // min_obs filter (v1: total games per entity).
        let order: Vec<u32> = order_idx
            .into_iter()
            .filter(|&idx| graph[idx].iter().map(|(_, (_, n))| n).sum::<usize>() >= self.min_obs)
            .map(|idx| idx as u32)
            .collect();

        Ok(KemenyModel {
            params: *self,
            names: data.interner().clone(),
            order,
        })
    }
}

struct KemenyFit<'a> {
    graph: &'a PrefGraph,
}

impl Fitness for KemenyFit<'_> {
    /// v1 fitness: concordant comparisons under the candidate's order, with
    /// a soft penalty pulling misordered pairs together.
    fn score(&self, candidate: &[f32]) -> f32 {
        self.graph
            .iter()
            .enumerate()
            .map(|(winner_idx, losers)| {
                let w_score = candidate[winner_idx];
                losers
                    .iter()
                    .map(|&(loser_idx, (wins, n))| {
                        let l_score = candidate[loser_idx];
                        if loser_idx < winner_idx {
                            0.0
                        } else if w_score > l_score {
                            wins as f32
                                - if wins * 2 < n {
                                    (w_score - l_score).abs()
                                } else {
                                    0.0
                                }
                        } else {
                            (n - wins) as f32
                                - if wins * 2 > n {
                                    (w_score - l_score).abs()
                                } else {
                                    0.0
                                }
                        }
                    })
                    .sum::<f32>()
            })
            .sum::<f32>()
    }
}

/// `graph[i][j] = (wins of i over j, total games between i and j)`.
type PrefGraph = Vec<Vec<(usize, (usize, usize))>>;

fn build_graph(data: &PairwiseDataset) -> PrefGraph {
    let n = data.n_entities();
    let mut maps: Vec<HashMap<usize, (usize, usize)>> = vec![HashMap::new(); n];
    for (w, l, x) in data.rows() {
        let margin = x as usize;
        let e = maps[w as usize].entry(l as usize).or_insert((0, 0));
        e.0 += margin;
        e.1 += margin;
        let e = maps[l as usize].entry(w as usize).or_insert((0, 0));
        e.1 += margin;
    }
    maps.into_iter()
        .map(|m| {
            let mut v: Vec<_> = m.into_iter().collect();
            v.sort_unstable_by_key(|e| e.0);
            v
        })
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
struct RankLine {
    id: String,
    rank: usize, // n..1, higher is better (v1 output convention)
}

/// Consensus order (best first), exposed as descending rank scores.
#[derive(Debug, Clone)]
pub struct KemenyModel {
    params: Kemeny,
    names: Interner,
    /// Entity ids best-first.
    order: Vec<u32>,
}

impl KemenyModel {
    /// The consensus ranking, best first.
    pub fn order(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(|&id| self.names.resolve(id))
    }
}

impl RankModel for KemenyModel {
    fn algorithm(&self) -> &'static str {
        "kemeny"
    }

    /// Rank positions as scores: best entity gets `n`, worst gets `1`.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        let n = self.order.len();
        self.order
            .iter()
            .enumerate()
            .map(move |(pos, &id)| (self.names.resolve(id), (n - pos) as f64))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let n = self.order.len();
        let lines: Vec<RankLine> = self
            .order
            .iter()
            .enumerate()
            .map(|(pos, &id)| RankLine {
                id: self.names.resolve(id).to_string(),
                rank: n - pos,
            })
            .collect();
        state::save_model(w, "kemeny", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, mut lines): (Kemeny, Vec<RankLine>) = state::load_model(r, "kemeny")?;
        lines.sort_by_key(|l| std::cmp::Reverse(l.rank));
        let mut names = Interner::new();
        let order = lines.iter().map(|l| names.intern(&l.id)).collect();
        Ok(Self {
            params,
            names,
            order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v1 `test_simple`.
    #[test]
    fn linear_chain() {
        let mut d = PairwiseDataset::new();
        d.push("1", "0", 1.0);
        d.push("2", "1", 1.0);
        d.push("3", "2", 1.0);
        let m = Kemeny {
            passes: KemenyPasses::Fixed(1),
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let order: Vec<&str> = m.order().collect();
        assert_eq!(order, vec!["3", "2", "1", "0"]);
    }

    /// v1 `test_conflict`.
    #[test]
    fn conflicting_evidence() {
        let mut d = PairwiseDataset::new();
        d.push("1", "0", 1.0);
        d.push("0", "1", 2.0);
        d.push("2", "1", 1.0);
        d.push("3", "2", 1.0);
        let m = Kemeny {
            passes: KemenyPasses::Fixed(10),
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let order: Vec<&str> = m.order().collect();
        assert_eq!(order, vec!["0", "3", "2", "1"]);
    }

    #[test]
    fn diff_evo_agrees_on_clear_data() {
        let mut d = PairwiseDataset::new();
        for _ in 0..4 {
            d.push("best", "mid", 1.0);
            d.push("mid", "low", 1.0);
            d.push("best", "low", 1.0);
        }
        let m = Kemeny {
            algo: KemenyAlgo::DiffEvo,
            passes: KemenyPasses::Fixed(20_000),
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let order: Vec<&str> = m.order().collect();
        assert_eq!(order, vec!["best", "mid", "low"]);
    }

    #[test]
    fn round_trip_preserves_order() {
        let mut d = PairwiseDataset::new();
        d.push("1", "0", 1.0);
        d.push("2", "1", 1.0);
        let m = Kemeny::default().fit(&d).unwrap();
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = KemenyModel::load_jsonl(buf.as_slice()).unwrap();
        assert_eq!(
            m.order().collect::<Vec<_>>(),
            m2.order().collect::<Vec<_>>()
        );
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

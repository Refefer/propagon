//! Luce Spectral Ranking (`docs/algorithms.md` §3.2).
//!
//! Builds the LSR Markov chain from pairwise outcomes (mass flows loser →
//! winner) and estimates its stationary distribution, whose log/degree-
//! adjusted form is a consistent Plackett-Luce estimate (Maystre &
//! Grossglauser 2015). Two estimators:
//!
//! - [`Estimator::PowerMethod`] — deterministic power iteration. v2 rebuilds
//!   it on a *transposed* adjacency so every output entry accumulates
//!   independently: no atomics, bit-stable results at any thread count.
//! - [`Estimator::MonteCarlo`] — seeded random walks per start node; cheaper
//!   per step on huge graphs, noisier scores.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, RankModel as _, Ranker};

const EPS: f64 = 1e-8;

/// Stationary-distribution estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Estimator {
    #[default]
    PowerMethod,
    MonteCarlo,
}

/// LSR parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lsr {
    /// Power-method passes, or random-walk steps per node for Monte Carlo
    /// (v1 defaults: 10 and 1000 respectively).
    pub steps: usize,
    pub estimator: Estimator,
    /// Seed for the Monte Carlo walks.
    pub seed: u64,
}

impl Default for Lsr {
    fn default() -> Self {
        Self {
            steps: 10,
            estimator: Estimator::PowerMethod,
            seed: 2020,
        }
    }
}

/// Fitted LSR scores (log-scale, mean-centered; higher is better).
#[derive(Debug, Clone)]
pub struct LsrModel {
    params: Lsr,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(LsrModel, "lsr");

/// The chain in both orientations: `outgoing[i]` carries cumulative
/// transition weights for sampling; `incoming[j]` carries plain transition
/// probabilities for deterministic accumulation.
struct Chain {
    degree: Vec<f64>,
    outgoing_cum: Vec<Vec<(u32, f64)>>,
    incoming: Vec<Vec<(u32, f64)>>,
}

fn build_chain(data: &PairwiseDataset) -> Chain {
    let n = data.n_entities();
    let mut weights: Vec<std::collections::HashMap<u32, f64>> = vec![Default::default(); n];
    for (w, l, x) in data.rows() {
        // v1: mass flows loser -> winner with weight amt/2.
        *weights[l as usize].entry(w).or_default() += f64::from(x) / 2.0;
    }

    let mut degree = vec![0.0; n];
    let mut outgoing_cum: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut incoming: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut entries: Vec<(u32, f64)> = weights[i].iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable_by_key(|e| e.0); // deterministic order
        let d: f64 = entries.iter().map(|e| e.1).sum();
        degree[i] = d;
        let mut cum = 0.0;
        for (k, v) in entries {
            let p = v / d;
            incoming[k as usize].push((i as u32, p));
            cum += v;
            outgoing_cum[i].push((k, cum / d));
        }
    }
    Chain {
        degree,
        outgoing_cum,
        incoming,
    }
}

/// L2 residual ‖π − πM‖ (v1 `compute_error`).
fn residual(pi: &[f64], chain: &Chain) -> f64 {
    let mut est = vec![0.0; pi.len()];
    for (j, inc) in chain.incoming.iter().enumerate() {
        for &(i, p) in inc {
            est[j] += pi[i as usize] * p;
        }
    }
    pi.iter()
        .zip(&est)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Shared post-processing: discount degree, move to log scale, center.
fn finalize(mut pi: Vec<f64>, chain: &Chain) -> Vec<f64> {
    for (i, x) in pi.iter_mut().enumerate() {
        *x = (*x / chain.degree[i].max(EPS).sqrt()).ln();
    }
    let avg = pi.iter().sum::<f64>() / pi.len() as f64;
    pi.iter_mut().for_each(|v| *v -= avg);
    pi
}

impl Lsr {
    fn power_method(
        &self,
        chain: &Chain,
        init: Option<Vec<f64>>,
        progress: &dyn crate::Progress,
    ) -> (f64, Vec<f64>) {
        let n = chain.incoming.len();
        let mut pi = init.unwrap_or_else(|| vec![1.0 / n as f64; n]);

        progress.start("power iterations", Some(self.steps as u64));
        for pass in 0..self.steps {
            // π' = πM, one independent accumulation per target node.
            let frozen = &pi;
            let mut next = parallel::par_map_indexed(n, |j| {
                chain.incoming[j]
                    .iter()
                    .map(|&(i, p)| frozen[i as usize] * p)
                    .sum::<f64>()
            });
            // v1: add EPS per entry, then L1-normalize.
            let denom: f64 = next.iter().map(|v| v + EPS).sum();
            for v in &mut next {
                *v = (*v + EPS) / denom;
            }
            pi = next;
            progress.update(pass as u64 + 1);
            if pass % 5 == 0 {
                progress.message(&format!("residual {:0.3e}", residual(&pi, chain)));
            }
        }
        progress.finish();

        let err = residual(&pi, chain);
        (err, finalize(pi, chain))
    }

    fn monte_carlo(&self, chain: &Chain, progress: &dyn crate::Progress) -> (f64, Vec<f64>) {
        let n = chain.outgoing_cum.len();
        progress.start("random walks", Some(n as u64));

        // One independent seeded walk per start node (v1 semantics), counted
        // locally and summed — deterministic for a fixed seed.
        let counts = parallel::par_map_indexed(n, |start| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed + start as u64);
            let mut local = vec![0u64; n];
            let mut cur = start;
            for _ in 0..self.steps {
                local[cur] += 1;
                let adj = &chain.outgoing_cum[cur];
                if adj.is_empty() {
                    cur = rng.random_range(0..n);
                } else {
                    let p: f64 = rng.random();
                    let idx = adj.partition_point(|&(_, w)| w < p).min(adj.len() - 1);
                    cur = adj[idx].0 as usize;
                }
            }
            local
        });

        let mut pi = vec![0.0f64; n];
        for local in counts {
            for (i, c) in local.into_iter().enumerate() {
                pi[i] += c as f64;
            }
        }
        let total: f64 = pi.iter().sum();
        pi.iter_mut().for_each(|v| *v /= total);
        progress.finish();

        let err = residual(&pi, chain);
        (err, finalize(pi, chain))
    }
}

impl Ranker for Lsr {
    type Data = PairwiseDataset;
    type Model = LsrModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<LsrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let chain = build_chain(data);
        let (err, scores) = parallel::run_scoped(opts, || match self.estimator {
            Estimator::PowerMethod => self.power_method(&chain, None, opts.progress()),
            Estimator::MonteCarlo => self.monte_carlo(&chain, opts.progress()),
        });
        log::debug!("lsr residual: {err:0.3e}");
        Ok(LsrModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }

    fn fit_warm_opts(
        &self,
        data: &PairwiseDataset,
        init: &LsrModel,
        opts: &FitOptions<'_>,
    ) -> Result<LsrModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let chain = build_chain(data);
        // Invert the log/degree transform to recover a starting distribution.
        let n = data.n_entities();
        let mut pi = vec![1.0 / n as f64; n];
        for (name, s) in init.scores() {
            if let Some(id) = data.interner().get(name) {
                pi[id as usize] = s.exp() * chain.degree[id as usize].max(EPS).sqrt();
            }
        }
        let total: f64 = pi.iter().sum();
        pi.iter_mut().for_each(|v| *v /= total);

        let (err, scores) = parallel::run_scoped(opts, || {
            self.power_method(&chain, Some(pi), opts.progress())
        });
        log::debug!("lsr warm residual: {err:0.3e}");
        Ok(LsrModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v1 `test_creating_tournament_graph`: 3-cycle chain structure.
    #[test]
    fn chain_construction_matches_v1() {
        let mut d = PairwiseDataset::new();
        d.push("1", "2", 1.0);
        d.push("2", "3", 2.0);
        d.push("3", "1", 1.0);
        let chain = build_chain(&d);
        // ids: 1->0, 2->1, 3->2
        assert_eq!(chain.degree, vec![0.5, 0.5, 1.0]);
        assert_eq!(chain.outgoing_cum[0], vec![(2, 1.0)]); // 1 lost to 3
        assert_eq!(chain.outgoing_cum[1], vec![(0, 1.0)]); // 2 lost to 1
        assert_eq!(chain.outgoing_cum[2], vec![(1, 1.0)]); // 3 lost to 2
    }

    /// v1 `test_error` (completed: the original ended in `panic!()`).
    #[test]
    fn power_method_converges() {
        let mut d = PairwiseDataset::new();
        for (w, l, x) in [
            ("1", "2", 1.0),
            ("1", "3", 10.0),
            ("2", "3", 3.0),
            ("2", "1", 1.0),
            ("3", "1", 1.0),
            ("3", "2", 2.0),
            ("4", "1", 1.0),
            ("4", "3", 2.0),
            ("2", "4", 2.0),
        ] {
            d.push(w, l, x);
        }
        let chain = build_chain(&d);
        let lsr = Lsr {
            steps: 50,
            ..Default::default()
        };
        let (err, _) = lsr.power_method(&chain, None, &crate::NoProgress);
        assert!(err < 1e-3, "residual {err}");
    }

    #[test]
    fn dominant_player_ranks_first_with_both_estimators() {
        let mut d = PairwiseDataset::new();
        for _ in 0..10 {
            d.push("best", "mid", 1.0);
            d.push("mid", "worst", 1.0);
        }
        d.push("mid", "best", 1.0);
        d.push("worst", "mid", 1.0);
        d.push("best", "worst", 1.0);
        d.push("worst", "best", 1.0);

        for estimator in [Estimator::PowerMethod, Estimator::MonteCarlo] {
            let steps = if estimator == Estimator::PowerMethod {
                50
            } else {
                5000
            };
            let m = Lsr {
                estimator,
                steps,
                seed: 7,
            }
            .fit(&d)
            .unwrap();
            let order: Vec<&str> = m.sorted_scores().iter().map(|e| e.0).collect();
            assert_eq!(order, vec!["best", "mid", "worst"], "{estimator:?}");
        }
    }

    #[test]
    fn power_method_is_deterministic() {
        let mut d = PairwiseDataset::new();
        for i in 0..50u32 {
            for j in 0..50u32 {
                if i != j && (i + j) % 3 != 0 {
                    d.push(&i.to_string(), &j.to_string(), 1.0 + (i % 4) as f32);
                }
            }
        }
        let a = Lsr::default().fit(&d).unwrap();
        let b = Lsr::default().fit(&d).unwrap();
        let sa: Vec<f64> = a.scores().map(|(_, s)| s).collect();
        let sb: Vec<f64> = b.scores().map(|(_, s)| s).collect();
        assert_eq!(sa, sb, "bitwise deterministic across runs");
    }
}

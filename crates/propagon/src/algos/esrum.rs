//! ES-RUM: Gaussian random-utility model fit by evolution strategies
//! (`docs/algorithms.md` §1.5).
//!
//! Each entity gets a `(μ, σ)` pair — a Gaussian utility distribution — so an
//! entity can be "good but erratic". Fitting is gradient-free: populations of
//! perturbed parameter sets are scored against observed win rates and blended
//! back (a faithful port of v1's two-level ES: per-entity fine tuning plus
//! population-level steps).
//!
//! Identifiability caveat (v1-documented): the model is identified only up to
//! location/scale; outputs are normalized and meaningful **relatively**.

use std::collections::HashMap;

use rand::SeedableRng;
use rand_distr::{Distribution as _, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx;
use crate::parallel;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Utility-noise family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RumDistribution {
    /// Per-entity variance is learned (σ ≥ 1e-5).
    #[default]
    Gaussian,
    /// All variances pinned to 1 (Thurstone-style).
    FixedNormal,
}

impl RumDistribution {
    fn bound(self, sigma: &mut f64) {
        match self {
            RumDistribution::Gaussian => *sigma = sigma.max(1e-5),
            RumDistribution::FixedNormal => *sigma = 1.0,
        }
    }
}

/// ES-RUM parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EsRum {
    pub distribution: RumDistribution,
    /// ES iterations.
    pub passes: usize,
    /// Initial perturbation scale (decays on stagnation).
    pub alpha: f64,
    /// L2 regularization on (μ, σ).
    pub gamma: f64,
    /// Entities with fewer comparisons than this are dropped from output.
    pub min_obs: usize,
    /// Pseudo-count smoothing added to every pairwise record.
    pub prior: usize,
    pub seed: u64,
}

impl Default for EsRum {
    fn default() -> Self {
        Self {
            distribution: RumDistribution::default(),
            passes: 100,
            alpha: 1.0,
            gamma: 1e-3,
            min_obs: 1,
            prior: 0,
            seed: 2019,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RumLine {
    id: String,
    mu: f64,
    sigma: f64,
}

/// Fitted `(μ, σ)` per entity (normalized; relative scale only).
#[derive(Debug, Clone)]
pub struct EsRumModel {
    params: EsRum,
    names: Interner,
    entries: Vec<[f64; 2]>,
}

impl EsRumModel {
    /// `(name, μ, σ)` rows.
    pub fn distributions(&self) -> impl Iterator<Item = (&str, f64, f64)> {
        self.names
            .names()
            .zip(self.entries.iter())
            .map(|(n, e)| (n, e[0], e[1]))
    }
}

impl RankModel for EsRumModel {
    fn algorithm(&self) -> &'static str {
        "es-rum"
    }

    /// Primary score is μ.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.entries.iter().map(|e| e[0]))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<RumLine> = self
            .distributions()
            .map(|(id, mu, sigma)| RumLine {
                id: id.to_string(),
                mu,
                sigma,
            })
            .collect();
        state::save_model(w, "es-rum", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (EsRum, Vec<RumLine>) = state::load_model(r, "es-rum")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        let entries = lines.iter().map(|l| [l.mu, l.sigma]).collect();
        Ok(Self {
            params,
            names,
            entries,
        })
    }
}

/// Aggregated comparison graph: `comps[i]` lists `(opponent, wins_of_i, games)`.
type CompGraph = Vec<Vec<(usize, (usize, usize))>>;

impl EsRum {
    fn build_graph(&self, data: &PairwiseDataset) -> CompGraph {
        let n = data.n_entities();
        let mut maps: Vec<HashMap<usize, (usize, usize)>> = vec![HashMap::new(); n];
        for (w, l, x) in data.rows() {
            let margin = x as usize;
            let e = maps[w as usize]
                .entry(l as usize)
                .or_insert((self.prior, 2 * self.prior));
            e.0 += margin;
            e.1 += margin;
            let e = maps[l as usize]
                .entry(w as usize)
                .or_insert((self.prior, 2 * self.prior));
            e.1 += margin;
        }
        maps.into_iter()
            .map(|m| {
                let mut v: Vec<_> = m.into_iter().collect();
                v.sort_unstable_by_key(|e| e.0); // deterministic
                v
            })
            .collect()
    }

    /// v1 initial policy: rank by mean win rate, place μ at normal quantiles.
    fn initial_policy(&self, graph: &CompGraph) -> Vec<[f64; 2]> {
        let mut rates: Vec<(usize, f64)> = graph
            .iter()
            .enumerate()
            .map(|(idx, comps)| {
                let r = comps
                    .iter()
                    .map(|(_, (w, n))| *w as f64 / (*n).max(1) as f64)
                    .sum::<f64>()
                    / comps.len().max(1) as f64;
                (idx, r)
            })
            .collect();
        rates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut policy = vec![[0.0, 0.0]; graph.len()];
        for (rank, (idx, _)) in rates.into_iter().enumerate() {
            policy[idx] = [
                mathx::norm_ppf((rank + 1) as f64 / (policy.len() + 2) as f64),
                1e-5,
            ];
        }
        policy
    }

    /// Fitness of one entity's `(μ, σ)` against its comparison record:
    /// weighted absolute gap between predicted and observed win rates, plus
    /// L2 regularization (v1 `score`).
    fn score(&self, dist: &[f64; 2], comps: &[(usize, (usize, usize))], all: &[[f64; 2]]) -> f64 {
        let mut s_mu = dist[0];
        let mut s_sigma = dist[1];
        self.distribution.bound(&mut s_sigma);
        let _ = &mut s_mu;

        let fit = comps
            .iter()
            .map(|(o_idx, (wins, n))| {
                let [mut e_mu, mut e_sigma] = all[*o_idx];
                self.distribution.bound(&mut e_sigma);
                let _ = &mut e_mu;
                // P(this beats opponent) = CDF of N(e_mu - s_mu, s_sigma + e_sigma) at 0.
                let sd = s_sigma + e_sigma;
                let rate = mathx::norm_cdf((0.0 - (e_mu - s_mu)) / sd);
                *n as f64 * (rate - *wins as f64 / *n as f64).abs()
            })
            .sum::<f64>()
            / comps.len().max(1) as f64;

        fit + self.gamma * (s_mu * s_mu + s_sigma * s_sigma).sqrt()
    }

    fn score_all(&self, policy: &[[f64; 2]], graph: &CompGraph) -> f64 {
        parallel::par_map_indexed(policy.len(), |idx| {
            self.score(&policy[idx], &graph[idx], policy)
        })
        .into_iter()
        .sum()
    }

    /// Per-entity fine tuning (v1 `tune_indivs`): 20 local perturbations,
    /// blend the best 3, keep only if the entity's own fitness improves.
    fn tune_individuals(
        &self,
        policy: &[[f64; 2]],
        graph: &CompGraph,
        out: &mut [[f64; 2]],
        sigma: f64,
        it: usize,
    ) {
        let weights = decay_weights(3);
        parallel::par_for_each_mut(out, |c_idx, candidate| {
            let comps = &graph[c_idx];
            let seed = self.seed + (c_idx + it + 1) as u64;
            let normal = Normal::new(0.0, sigma).expect("sigma > 0");

            let mut grads: Vec<(f64, [f64; 2])> = (0..20)
                .map(|g_idx| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + g_idx as u64);
                    let g = [
                        candidate[0] + normal.sample(&mut rng),
                        candidate[1] + normal.sample(&mut rng),
                    ];
                    (self.score(&g, comps, policy), g)
                })
                .collect();
            grads.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut next = *candidate;
            for (w, (_, g)) in weights.iter().zip(grads.iter().take(3)) {
                next[0] += w * (g[0] - candidate[0]);
                next[1] += w * (g[1] - candidate[1]);
            }

            let old = self.score(&policy[c_idx], comps, policy);
            let new = self.score(&next, comps, policy);
            *candidate = if new > old { policy[c_idx] } else { next };
        });
    }
}

fn decay_weights(k: usize) -> Vec<f64> {
    let mut w: Vec<f64> = (0..k).map(|i| 1.0 / ((i + 2) as f64).ln()).collect();
    let s: f64 = w.iter().sum();
    w.iter_mut().for_each(|x| *x /= s);
    w
}

impl Ranker for EsRum {
    type Data = PairwiseDataset;
    type Model = EsRumModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<EsRumModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let progress = opts.progress();
        let graph = self.build_graph(data);
        let n_vocab = graph.len();

        let mut policy = self.initial_policy(&graph);
        let n_gradients = ((n_vocab as f64).powf(0.7).max(100.0)) as usize;
        let n_children = ((n_gradients as f64 / 10.0).max(3.0)) as usize;
        let weights = decay_weights(n_children);

        let decay_rate = ((self.alpha / 100.0).ln() / self.passes as f64).exp();
        let mut last_loss = self.score_all(&policy, &graph);
        let mut sigma_group = self.alpha;
        let mut sigma_indiv = self.alpha;

        let mut new_policy = policy.clone();
        progress.start("es-rum passes", Some(self.passes as u64));

        parallel::run_scoped(opts, || {
            for it in 0..self.passes {
                // Phase 1: per-entity tuning.
                self.tune_individuals(&policy, &graph, &mut new_policy, sigma_indiv, it);
                let mut new_score = self.score_all(&new_policy, &graph);
                if new_score < last_loss {
                    std::mem::swap(&mut policy, &mut new_policy);
                    last_loss = new_score;
                } else {
                    sigma_indiv *= decay_rate;
                }

                // Phase 2: population-level perturbations (v1 gating).
                if last_loss != new_score {
                    let normal = Normal::new(0.0, sigma_group).expect("sigma > 0");
                    let frozen = &policy;
                    let mut gradients: Vec<(f64, Vec<[f64; 2]>)> =
                        parallel::par_map_indexed(n_gradients, |idx| {
                            let seed = self.seed + (idx + it * n_gradients) as u64;
                            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                            let grad: Vec<[f64; 2]> = frozen
                                .iter()
                                .map(|p| {
                                    [
                                        p[0] + normal.sample(&mut rng),
                                        p[1] + normal.sample(&mut rng),
                                    ]
                                })
                                .collect();
                            (self.score_all(&grad, &graph), grad)
                        });
                    gradients.sort_by(|a, b| a.0.total_cmp(&b.0));

                    for (p_idx, arr) in new_policy.iter_mut().enumerate() {
                        let base = policy[p_idx];
                        let mut next = base;
                        for (w, (_, g)) in weights.iter().zip(gradients.iter().take(n_children)) {
                            next[0] += w * (g[p_idx][0] - base[0]);
                            next[1] += w * (g[p_idx][1] - base[1]);
                        }
                        *arr = next;
                    }

                    new_score = self.score_all(&new_policy, &graph);
                    if new_score < last_loss {
                        std::mem::swap(&mut policy, &mut new_policy);
                        last_loss = new_score;
                    } else {
                        sigma_group *= decay_rate;
                    }
                }

                progress.update(it as u64 + 1);
                progress.message(&format!(
                    "best {last_loss:0.5}, σ_ind {sigma_indiv:0.3}, σ_grp {sigma_group:0.3}"
                ));
            }
        });
        progress.finish();

        // Bound, filter by min_obs, and normalize (location/scale free).
        let keep: Vec<bool> = graph
            .iter()
            .map(|comps| comps.iter().map(|(_, (_, n))| n).sum::<usize>() >= self.min_obs)
            .collect();

        let mut names = Interner::new();
        let mut entries: Vec<[f64; 2]> = Vec::new();
        for (idx, kept) in keep.iter().enumerate() {
            if !kept {
                continue;
            }
            let mut mu = policy[idx][0];
            let mut sigma = policy[idx][1];
            self.distribution.bound(&mut sigma);
            let _ = &mut mu;
            names.intern(data.interner().name(idx as u32).expect("id resolves"));
            entries.push([mu, sigma]);
        }
        if entries.is_empty() {
            return Err(Error::InvalidInput(
                "min_obs filtered out every entity".into(),
            ));
        }

        let min_mu = entries.iter().map(|e| e[0]).fold(f64::INFINITY, f64::min);
        let max_sigma = entries
            .iter()
            .map(|e| e[1])
            .fold(f64::NEG_INFINITY, f64::max);
        for e in &mut entries {
            e[0] = (e[0] - min_mu) / max_sigma;
            e[1] /= max_sigma;
        }

        Ok(EsRumModel {
            params: *self,
            names,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn data() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        for _ in 0..8 {
            d.push("a", "b", 1.0);
            d.push("b", "c", 1.0);
            d.push("a", "c", 1.0);
        }
        d.push("b", "a", 1.0);
        d.push("c", "b", 1.0);
        d.push("c", "a", 1.0);
        d
    }

    #[test]
    fn recovers_order_and_is_seed_deterministic() {
        let algo = EsRum {
            passes: 30,
            ..Default::default()
        };
        let m1 = algo.fit(&data()).unwrap();
        let order: Vec<&str> = m1.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        let m2 = algo.fit(&data()).unwrap();
        let s1: Vec<f64> = m1.scores().map(|(_, s)| s).collect();
        let s2: Vec<f64> = m2.scores().map(|(_, s)| s).collect();
        assert_eq!(s1, s2, "same seed, same result");
    }

    #[test]
    fn normalization_pins_scale() {
        let m = EsRum {
            passes: 20,
            ..Default::default()
        }
        .fit(&data())
        .unwrap();
        let min_mu = m
            .distributions()
            .map(|(_, mu, _)| mu)
            .fold(f64::INFINITY, f64::min);
        let max_sigma = m
            .distributions()
            .map(|(_, _, s)| s)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min_mu.abs() < 1e-12, "min μ normalized to 0, got {min_mu}");
        assert!(
            (max_sigma - 1.0).abs() < 1e-12,
            "max σ normalized to 1, got {max_sigma}"
        );
    }

    #[test]
    fn round_trip() {
        let m = EsRum {
            passes: 5,
            ..Default::default()
        }
        .fit(&data())
        .unwrap();
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = EsRumModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

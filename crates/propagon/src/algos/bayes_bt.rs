//! Bayesian Bradley-Terry via Gibbs sampling (`docs/algorithms.md` §11.1;
//! Caron & Doucet, J. Comp. Graph. Stat. 21(1) 2012, arXiv:1011.1761).
//!
//! Gamma-prior BT with the paper's latent-variable scheme: for each pair
//! with `n_ij` total comparisons, `Z_ij | λ ~ Gamma(n_ij, λ_i + λ_j)`, and
//! each strength `λ_i | Z ~ Gamma(a + w_i, b + Σ_j Z_ij)` (`w_i` = i's
//! total wins). Both conditionals are exact, so the chain mixes well and
//! every draw is cheap.
//!
//! Output: posterior mean and a central credible interval per entity, on
//! the **normalized** scale `π = λ/Σλ` — the paper shows the overall scale
//! `Σλ` is not likelihood-identifiable (its rate `b` exists only to anchor
//! the prior), while π's posterior is well-defined. With `shape = 1` the
//! posterior mode coincides with the MLE that `bradley-terry-model` finds.
//!
//! No connectivity requirements: the prior keeps every conditional proper,
//! so undefeated/winless entities get honest wide posteriors instead of
//! divergence — the main practical reason to choose this over plain BT-MM
//! on sparse data. Sampling is inherently sequential; seeded and exactly
//! reproducible at any thread count.

use rand::SeedableRng;
use rand_distr::{Distribution as _, Gamma};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Bayesian Bradley-Terry parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BayesianBradleyTerry {
    /// Gamma prior shape `a` (per-entity). `1.0` makes the MAP coincide
    /// with the MLE; larger values shrink strengths together.
    pub shape: f64,
    /// Gamma prior rate `b`. Anchors the (non-identifiable) overall scale
    /// only; the reported π's are invariant to it.
    pub rate: f64,
    /// Posterior draws kept after burn-in.
    pub samples: usize,
    /// Discarded warm-up sweeps.
    pub burn_in: usize,
    /// Central credible-interval mass (e.g. 0.9 → 5%..95%).
    pub credible: f64,
    pub seed: u64,
}

impl Default for BayesianBradleyTerry {
    fn default() -> Self {
        Self {
            shape: 1.0,
            rate: 1.0,
            samples: 2000,
            burn_in: 500,
            credible: 0.9,
            seed: 2012,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PosteriorLine {
    id: String,
    mean: f64,
    lo: f64,
    hi: f64,
}

/// Posterior summaries of normalized BT strengths.
#[derive(Debug, Clone)]
pub struct BayesBtModel {
    params: BayesianBradleyTerry,
    names: Interner,
    mean: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl BayesBtModel {
    /// `(name, posterior mean, credible lo, credible hi)` per entity.
    pub fn posteriors(&self) -> impl Iterator<Item = (&str, f64, f64, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, n)| (n, self.mean[i], self.lo[i], self.hi[i]))
    }
}

impl RankModel for BayesBtModel {
    fn algorithm(&self) -> &'static str {
        "bayesian-bradley-terry"
    }

    /// Posterior means (normalized strengths, sum ≈ 1).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.mean.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<PosteriorLine> = self
            .posteriors()
            .map(|(id, mean, lo, hi)| PosteriorLine {
                id: id.to_string(),
                mean,
                lo,
                hi,
            })
            .collect();
        state::save_model(w, "bayesian-bradley-terry", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (BayesianBradleyTerry, Vec<PosteriorLine>) =
            state::load_model(r, "bayesian-bradley-terry")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            mean: lines.iter().map(|l| l.mean).collect(),
            lo: lines.iter().map(|l| l.lo).collect(),
            hi: lines.iter().map(|l| l.hi).collect(),
        })
    }
}

impl Ranker for BayesianBradleyTerry {
    type Data = PairwiseDataset;
    type Model = BayesBtModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<BayesBtModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if self.shape <= 0.0 || self.rate <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "gamma prior needs shape > 0 and rate > 0, got ({}, {})",
                self.shape, self.rate
            )));
        }
        if !(0.0..1.0).contains(&self.credible) {
            return Err(Error::InvalidInput(format!(
                "credible mass must lie in [0, 1), got {}",
                self.credible
            )));
        }

        let n = data.n_entities();

        // Per unordered pair: total comparison weight; per entity: win sum.
        let mut pair_n: std::collections::HashMap<(u32, u32), f64> =
            std::collections::HashMap::new();
        let mut wins = vec![0.0f64; n];

        for (w, l, x) in data.rows() {
            let x = f64::from(x);
            let key = (w.min(l), w.max(l));
            *pair_n.entry(key).or_default() += x;
            wins[w as usize] += x;
        }

        let mut pairs: Vec<((u32, u32), f64)> = pair_n.into_iter().collect();
        pairs.sort_unstable_by_key(|&(k, _)| k);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
        let mut lambda = vec![1.0f64; n];
        let mut z_sum = vec![0.0f64; n];
        let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(self.samples); n];

        let progress = opts.progress;
        let total_sweeps = self.burn_in + self.samples;
        progress.start("gibbs sweeps", Some(total_sweeps as u64));

        let bad = |e: rand_distr::GammaError| Error::Numeric(format!("gamma draw: {e}"));

        for sweep in 0..total_sweeps {
            z_sum.iter_mut().for_each(|z| *z = 0.0);

            for &((i, j), n_ij) in &pairs {
                let rate = lambda[i as usize] + lambda[j as usize];
                let z = Gamma::new(n_ij, 1.0 / rate).map_err(bad)?.sample(&mut rng);
                z_sum[i as usize] += z;
                z_sum[j as usize] += z;
            }

            for i in 0..n {
                let shape = self.shape + wins[i];
                let rate = self.rate + z_sum[i];
                lambda[i] = Gamma::new(shape, 1.0 / rate).map_err(bad)?.sample(&mut rng);
            }

            if sweep >= self.burn_in {
                let total: f64 = lambda.iter().sum();
                for (i, draw) in draws.iter_mut().enumerate() {
                    draw.push(lambda[i] / total);
                }
            }
            progress.update(sweep as u64 + 1);
        }

        progress.finish();

        let tail = (1.0 - self.credible) / 2.0;
        let mut mean = vec![0.0; n];
        let mut lo = vec![0.0; n];
        let mut hi = vec![0.0; n];

        for (i, draw) in draws.iter_mut().enumerate() {
            mean[i] = draw.iter().sum::<f64>() / draw.len() as f64;
            draw.sort_by(f64::total_cmp);
            lo[i] = quantile(draw, tail);
            hi[i] = quantile(draw, 1.0 - tail);
        }

        Ok(BayesBtModel {
            params: *self,
            names: data.interner().clone(),
            mean,
            lo,
            hi,
        })
    }
}

/// Empirical quantile of a sorted sample (linear interpolation).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let base = pos.floor() as usize;
    let frac = pos - base as f64;

    match sorted.get(base + 1) {
        Some(&next) => sorted[base] * (1.0 - frac) + next * frac,
        None => sorted[base],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn lopsided() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 8.0);
        d.push("b", "a", 2.0);
        d.push("b", "c", 8.0);
        d.push("c", "b", 2.0);
        d.push("a", "c", 9.0);
        d.push("c", "a", 1.0);
        d
    }

    #[test]
    fn recovers_order_with_sane_intervals() {
        let m = BayesianBradleyTerry::default().fit(&lopsided()).unwrap();
        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        for (_, mean, lo, hi) in m.posteriors() {
            assert!(lo <= mean && mean <= hi);
            assert!(lo > 0.0 && hi < 1.0);
        }
    }

    #[test]
    fn seeded_runs_are_identical() {
        let a = BayesianBradleyTerry::default().fit(&lopsided()).unwrap();
        let b = BayesianBradleyTerry::default().fit(&lopsided()).unwrap();
        let av: Vec<f64> = a.scores().map(|(_, s)| s).collect();
        let bv: Vec<f64> = b.scores().map(|(_, s)| s).collect();
        assert_eq!(av, bv);
    }

    /// More data → tighter intervals (same generating proportions).
    #[test]
    fn intervals_shrink_with_evidence() {
        let mut small = PairwiseDataset::new();
        small.push("a", "b", 3.0);
        small.push("b", "a", 1.0);

        let mut big = PairwiseDataset::new();
        big.push("a", "b", 300.0);
        big.push("b", "a", 100.0);

        let width = |d: &PairwiseDataset| {
            let m = BayesianBradleyTerry::default().fit(d).unwrap();
            let (_, _, lo, hi) = m.posteriors().next().unwrap();
            hi - lo
        };
        assert!(width(&big) < width(&small) / 2.0);
    }

    /// Winless entities keep proper (wide, low) posteriors — no divergence.
    #[test]
    fn winless_entities_stay_proper() {
        let mut d = PairwiseDataset::new();
        d.push("a", "dud", 5.0);
        d.push("b", "dud", 5.0);
        d.push("a", "b", 1.0);
        d.push("b", "a", 1.0);

        let m = BayesianBradleyTerry::default().fit(&d).unwrap();
        let s: std::collections::HashMap<&str, f64> = m.scores().collect();
        assert!(s["dud"] < s["a"] && s["dud"] < s["b"]);
        assert!(s["dud"] > 0.0);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let m = BayesianBradleyTerry {
            samples: 200,
            burn_in: 50,
            ..Default::default()
        }
        .fit(&lopsided())
        .unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = BayesBtModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

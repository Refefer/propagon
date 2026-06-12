//! Bootstrap comparison of Monte Carlo value estimates
//! (`docs/algorithms.md` §13.2; Efron 1979; Rubin 1981).
//!
//! The uncertainty layer over [`McValue`](crate::algos::McValue): per-state
//! confidence intervals from resampled episodes, and (optionally) pairwise
//! exceedance probabilities `P(V_b > V_a)` plus two-sample permutation
//! tests — the outputs that turn noisy `V̂(s)` point estimates into
//! decision-grade rankings and weighted preference edges.
//!
//! Resampling operates on whole **episodes**, never on individual return
//! samples, so within-episode correlation is preserved; the implied
//! assumption is that episodes are exchangeable (interference or
//! seasonality across episodes breaks it). Replicate `k` draws from its own
//! `seed + k` stream, and replicates are merged in index order, so results
//! are bit-stable at any thread count.
//!
//! Gotchas: under [`ResampleScheme::Bootstrap`] a rarely-observed state can
//! be absent from some replicates — its interval is computed over the
//! replicates that include it (`n_rep` records how many); the Bayesian
//! bootstrap weights every episode positively, so there `n_rep` always
//! equals `replicates`. [`PairwiseTests::On`] costs
//! `O(states² · permutations)` and is off by default.

use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution as _, Gamma};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::algos::mc_value::{Visit, episode_returns};
use crate::dataset::TrajectoriesDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx::quantile;
use crate::parallel;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// How episodes are reweighted in each replicate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResampleScheme {
    /// Efron's bootstrap: draw `n_episodes` episodes with replacement.
    #[default]
    Bootstrap,
    /// Rubin's Bayesian bootstrap: every episode gets a Gamma(1, 1) weight
    /// (normalization cancels in the weighted means), giving
    /// posterior-flavored replicate distributions with no empty cells.
    BayesianBootstrap,
}

/// Whether to compute pairwise exceedance + permutation tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum PairwiseTests {
    /// No pair statistics (the default — pairs cost
    /// `O(states² · permutations)`).
    #[default]
    Off,
    /// For every state pair: exceedance over shared replicates and a
    /// two-sided permutation test with this many label shuffles.
    On { permutations: usize },
}

/// Bootstrap value-comparison parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueCompare {
    /// Discount factor in `(0, 1]`.
    pub gamma: f64,
    pub visit: Visit,
    /// Resampling replicates (at least 2).
    pub replicates: usize,
    pub method: ResampleScheme,
    /// Central interval mass in `(0, 1)` (0.95 → 2.5%..97.5%).
    pub credible: f64,
    pub pairwise: PairwiseTests,
    /// States with fewer full-data return samples than this are excluded.
    pub min_observations: usize,
    pub seed: u64,
}

impl Default for ValueCompare {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            visit: Visit::default(),
            replicates: 2000,
            method: ResampleScheme::default(),
            credible: 0.95,
            pairwise: PairwiseTests::default(),
            min_observations: 2,
            seed: 2026,
        }
    }
}

impl ValueCompare {
    /// Rejects parameter values outside their documented domains (NaN
    /// included — the comparisons are written to fail on it).
    fn validate(&self) -> Result<()> {
        if !(self.gamma > 0.0 && self.gamma <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "gamma must lie in (0, 1], got {}",
                self.gamma
            )));
        }
        if self.replicates < 2 {
            return Err(Error::InvalidInput(format!(
                "need at least 2 replicates, got {}",
                self.replicates
            )));
        }
        if !(self.credible > 0.0 && self.credible < 1.0) {
            return Err(Error::InvalidInput(format!(
                "credible mass must lie in (0, 1), got {}",
                self.credible
            )));
        }
        Ok(())
    }

    /// Per-episode resampling weights for replicate `k`, drawn from the
    /// replicate's own `seed + k` stream: multinomial counts under the
    /// standard bootstrap, i.i.d. Gamma(1, 1) draws under the Bayesian one.
    fn replicate_weights(&self, k: usize, n_episodes: usize, exp1: &Gamma<f64>) -> Vec<f64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed.wrapping_add(k as u64));

        match self.method {
            ResampleScheme::Bootstrap => {
                let mut w = vec![0.0f64; n_episodes];
                for _ in 0..n_episodes {
                    w[rng.random_range(0..n_episodes)] += 1.0;
                }
                w
            }
            ResampleScheme::BayesianBootstrap => {
                (0..n_episodes).map(|_| exp1.sample(&mut rng)).collect()
            }
        }
    }
}

/// Domain separator for the permutation-test streams, keeping them disjoint
/// from the replicate streams seeded at `seed + k`.
const PERM_DOMAIN: u64 = 0x7065_726d_5f76_6331; // "perm_vc1"

/// Pairwise comparison between two emitted states (by emitted-interner id).
#[derive(Clone, Copy, Debug)]
struct PairStat {
    a: u32,
    b: u32,
    /// Fraction of shared replicates where b's mean exceeded a's.
    exceed: f64,
    /// Two-sided permutation p-value with the +1/(n+1) correction.
    p: f64,
}

/// Per-state intervals plus optional pair statistics.
#[derive(Debug, Clone)]
pub struct ValueCompareModel {
    params: ValueCompare,
    names: Interner,
    point: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Full-data return samples per state.
    n: Vec<u64>,
    /// Replicates whose resample included the state.
    n_rep: Vec<u64>,
    /// Empty unless fitted with [`PairwiseTests::On`].
    pairs: Vec<PairStat>,
}

impl ValueCompareModel {
    /// `(name, point estimate, interval lo, interval hi)` per state.
    pub fn intervals(&self) -> impl Iterator<Item = (&str, f64, f64, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, n)| (n, self.point[i], self.lo[i], self.hi[i]))
    }

    /// Pair statistics as `(a, b, exceedance, p)`: exceedance is the
    /// fraction of shared replicates where b's mean exceeded a's
    /// (`P(V_b > V_a)`), `p` the two-sided permutation p-value on the
    /// pooled raw returns. Empty unless fitted with [`PairwiseTests::On`].
    pub fn pairs(&self) -> impl Iterator<Item = (&str, &str, f64, f64)> {
        self.pairs.iter().map(|p| {
            (
                self.names.resolve(p.a),
                self.names.resolve(p.b),
                p.exceed,
                p.p,
            )
        })
    }
}

/// State and pair lines share one file, discriminated by `k`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "k", rename_all = "kebab-case")]
enum CompareLine {
    State {
        id: String,
        s: f64,
        lo: f64,
        hi: f64,
        n: u64,
        n_rep: u64,
    },
    Pair {
        a: String,
        b: String,
        exceed: f64,
        p: f64,
    },
}

impl RankModel for ValueCompareModel {
    fn algorithm(&self) -> &'static str {
        "value-compare"
    }

    /// Full-data point estimates (mean return per state).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.point.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let mut lines: Vec<CompareLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| CompareLine::State {
                id: id.to_string(),
                s: self.point[i],
                lo: self.lo[i],
                hi: self.hi[i],
                n: self.n[i],
                n_rep: self.n_rep[i],
            })
            .collect();
        lines.extend(self.pairs().map(|(a, b, exceed, p)| CompareLine::Pair {
            a: a.to_string(),
            b: b.to_string(),
            exceed,
            p,
        }));
        state::save_model(w, "value-compare", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (ValueCompare, Vec<CompareLine>) =
            state::load_model(r, "value-compare")?;

        let mut ids = Vec::new();
        let mut point = Vec::new();
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        let mut n = Vec::new();
        let mut n_rep = Vec::new();
        let mut raw_pairs = Vec::new();

        for line in &lines {
            match line {
                CompareLine::State {
                    id,
                    s,
                    lo: l,
                    hi: h,
                    n: cnt,
                    n_rep: nr,
                } => {
                    ids.push(id.as_str());
                    point.push(*s);
                    lo.push(*l);
                    hi.push(*h);
                    n.push(*cnt);
                    n_rep.push(*nr);
                }
                CompareLine::Pair { a, b, exceed, p } => raw_pairs.push((a, b, *exceed, *p)),
            }
        }

        let names = Interner::from_names(ids)?;
        let pairs = raw_pairs
            .into_iter()
            .map(|(a, b, exceed, p)| {
                let resolve = |name: &str| {
                    names.get(name).ok_or_else(|| {
                        Error::State(format!("pair line references unknown state {name:?}"))
                    })
                };
                Ok(PairStat {
                    a: resolve(a)?,
                    b: resolve(b)?,
                    exceed,
                    p,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            params,
            names,
            point,
            lo,
            hi,
            n,
            n_rep,
            pairs,
        })
    }
}

impl Ranker for ValueCompare {
    type Data = TrajectoriesDataset;
    type Model = ValueCompareModel;

    fn fit_opts(
        &self,
        data: &TrajectoriesDataset,
        opts: &FitOptions<'_>,
    ) -> Result<ValueCompareModel> {
        self.validate()?;
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n_ep = data.n_episodes();
        let per_episode: Vec<Vec<(u32, f64)>> = parallel::run_scoped(opts, || {
            parallel::par_map_indexed(n_ep, |e| {
                let (states, rewards) = data.episode(e);
                episode_returns(states, rewards, self.gamma, self.visit)
            })
        });

        let mut full: Vec<Vec<f64>> = vec![Vec::new(); data.n_entities()];
        for episode in &per_episode {
            for &(s, g) in episode {
                full[s as usize].push(g);
            }
        }

        // Emitted states: enough full-data samples, in dataset-id order.
        // A zero floor would emit unobserved states with NaN means.
        let floor = self.min_observations.max(1);
        let mut emit_of: Vec<Option<usize>> = vec![None; full.len()];
        let mut names = Interner::new();
        let mut point = Vec::new();
        let mut n = Vec::new();
        let mut emit_samples: Vec<Vec<f64>> = Vec::new();

        for (id, smp) in full.iter_mut().enumerate() {
            if smp.len() < floor {
                continue;
            }
            emit_of[id] = Some(names.len());
            names.intern(data.interner().resolve(id as u32));
            point.push(smp.iter().sum::<f64>() / smp.len() as f64);
            n.push(smp.len() as u64);
            emit_samples.push(std::mem::take(smp));
        }

        if names.is_empty() {
            return Err(Error::InvalidInput("no state met min_observations".into()));
        }
        let k = names.len();

        // Gamma(1, 1) = Exp(1); the parameters are fixed, so construction
        // cannot fail in practice — the error path keeps this panic-free.
        let exp1 = Gamma::new(1.0, 1.0).map_err(|e| Error::Numeric(format!("gamma draw: {e}")))?;

        let progress = opts.progress;
        progress.start("bootstrap replicates", Some(self.replicates as u64));
        let reps: Vec<Vec<Option<f64>>> = parallel::run_scoped(opts, || {
            parallel::par_map_indexed(self.replicates, |r| {
                let w = self.replicate_weights(r, n_ep, &exp1);
                let mut num = vec![0.0f64; k];
                let mut den = vec![0.0f64; k];

                for (e, episode) in per_episode.iter().enumerate() {
                    if w[e] == 0.0 {
                        continue;
                    }
                    for &(s, g) in episode {
                        if let Some(j) = emit_of[s as usize] {
                            num[j] += w[e] * g;
                            den[j] += w[e];
                        }
                    }
                }

                num.into_iter()
                    .zip(den)
                    .map(|(nm, d)| (d > 0.0).then(|| nm / d))
                    .collect()
            })
        });
        progress.finish();

        let tail = (1.0 - self.credible) / 2.0;
        let mut lo = vec![0.0f64; k];
        let mut hi = vec![0.0f64; k];
        let mut n_rep = vec![0u64; k];

        for j in 0..k {
            let mut col: Vec<f64> = reps.iter().filter_map(|r| r[j]).collect();
            if col.is_empty() {
                return Err(Error::Numeric(format!(
                    "state {:?} appeared in no bootstrap replicate; increase replicates",
                    names.resolve(j as u32)
                )));
            }
            n_rep[j] = col.len() as u64;
            col.sort_by(f64::total_cmp);
            lo[j] = quantile(&col, tail);
            hi[j] = quantile(&col, 1.0 - tail);
        }

        let pairs = match self.pairwise {
            PairwiseTests::Off => Vec::new(),
            PairwiseTests::On { permutations } => {
                self.pair_tests(permutations, &names, &point, &emit_samples, &reps, opts)?
            }
        };

        Ok(ValueCompareModel {
            params: *self,
            names,
            point,
            lo,
            hi,
            n,
            n_rep,
            pairs,
        })
    }
}

impl ValueCompare {
    /// Exceedance + permutation test for every unordered state pair
    /// `(i, j)`, `i < j`, in deterministic (i, j) order. Exceedance counts
    /// shared replicates with `mean_j > mean_i`; the permutation test
    /// shuffles pooled raw returns with a per-pair stream seeded from the
    /// master seed under [`PERM_DOMAIN`].
    fn pair_tests(
        &self,
        permutations: usize,
        names: &Interner,
        point: &[f64],
        emit_samples: &[Vec<f64>],
        reps: &[Vec<Option<f64>>],
        opts: &FitOptions<'_>,
    ) -> Result<Vec<PairStat>> {
        let k = names.len();
        let pair_idx: Vec<(usize, usize)> = (0..k)
            .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
            .collect();

        let progress = opts.progress;
        progress.start("pairwise tests", Some(pair_idx.len() as u64));
        let computed: Vec<Result<PairStat>> = parallel::run_scoped(opts, || {
            parallel::par_map_indexed(pair_idx.len(), |p| {
                let (i, j) = pair_idx[p];

                let mut shared = 0u64;
                let mut above = 0u64;
                for rep in reps {
                    if let (Some(a), Some(b)) = (rep[i], rep[j]) {
                        shared += 1;
                        if b > a {
                            above += 1;
                        }
                    }
                }
                if shared == 0 {
                    return Err(Error::Numeric(format!(
                        "states {:?} and {:?} share no bootstrap replicate; \
                         increase replicates",
                        names.resolve(i as u32),
                        names.resolve(j as u32)
                    )));
                }
                let exceed = above as f64 / shared as f64;

                let obs = (point[j] - point[i]).abs();
                let na = emit_samples[i].len();
                let mut pool: Vec<f64> = emit_samples[i]
                    .iter()
                    .chain(&emit_samples[j])
                    .copied()
                    .collect();
                let nb = pool.len() - na;

                let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                    (self.seed ^ PERM_DOMAIN).wrapping_add(p as u64),
                );
                let mut hits = 0usize;
                for _ in 0..permutations {
                    pool.shuffle(&mut rng);
                    let ma = pool[..na].iter().sum::<f64>() / na as f64;
                    let mb = pool[na..].iter().sum::<f64>() / nb as f64;
                    if (mb - ma).abs() >= obs {
                        hits += 1;
                    }
                }

                Ok(PairStat {
                    a: i as u32,
                    b: j as u32,
                    exceed,
                    // +1/(n+1): the observed labeling always counts.
                    p: (hits + 1) as f64 / (permutations + 1) as f64,
                })
            })
        });
        progress.finish();

        computed.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Normal;

    /// Appends `n` single-step episodes for `state` with N(mu, sigma)
    /// rewards from a seeded stream.
    fn gaussian_state(
        d: &mut TrajectoriesDataset,
        state: &str,
        mu: f64,
        sigma: f64,
        n: usize,
        seed: u64,
    ) {
        let normal = Normal::new(mu, sigma).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        for _ in 0..n {
            d.push_step(state, normal.sample(&mut rng) as f32).unwrap();
            d.end_episode();
        }
    }

    fn cfg(replicates: usize) -> ValueCompare {
        ValueCompare {
            replicates,
            ..Default::default()
        }
    }

    #[test]
    fn interval_covers_the_true_mean() {
        for method in [ResampleScheme::Bootstrap, ResampleScheme::BayesianBootstrap] {
            let mut d = TrajectoriesDataset::new();
            gaussian_state(&mut d, "S", 1.0, 0.5, 200, 7);

            let m = ValueCompare { method, ..cfg(500) }.fit(&d).unwrap();
            let (name, point, lo, hi) = m.intervals().next().unwrap();
            assert_eq!(name, "S");
            assert!(lo <= point && point <= hi);
            assert!(lo <= 1.0 && 1.0 <= hi, "[{lo}, {hi}] misses 1.0");
        }
    }

    /// Quadrupling the episode count roughly halves the interval width
    /// (√k scaling); 0.7 leaves slack for bootstrap noise.
    #[test]
    fn interval_width_shrinks_with_data() {
        let width = |n: usize| {
            let mut d = TrajectoriesDataset::new();
            gaussian_state(&mut d, "S", 1.0, 0.5, n, 7);
            let m = cfg(500).fit(&d).unwrap();
            let (_, _, lo, hi) = m.intervals().next().unwrap();
            hi - lo
        };
        assert!(width(400) < 0.7 * width(100));
    }

    /// B is one full unit better than A at σ = 0.25 — the exceedance must
    /// saturate. C and D get byte-identical reward draws split across
    /// distinct episodes, so their resampled means straddle each other
    /// symmetrically and the exceedance sits near ½.
    #[test]
    fn exceedance_separates_clear_winners_from_ties() {
        let mut d = TrajectoriesDataset::new();
        gaussian_state(&mut d, "A", 0.0, 0.25, 100, 11);
        gaussian_state(&mut d, "B", 1.0, 0.25, 100, 12);

        let vc = ValueCompare {
            pairwise: PairwiseTests::On { permutations: 99 },
            ..cfg(500)
        };
        let m = vc.fit(&d).unwrap();
        let (a, b, exceed, _) = m.pairs().next().unwrap();
        assert_eq!((a, b), ("A", "B"));
        assert!(exceed > 0.85, "exceedance {exceed} should be near 1");

        let mut tie = TrajectoriesDataset::new();
        gaussian_state(&mut tie, "C", 0.5, 0.25, 100, 21);
        gaussian_state(&mut tie, "D", 0.5, 0.25, 100, 21);
        let m = vc.fit(&tie).unwrap();
        let (_, _, exceed, _) = m.pairs().next().unwrap();
        assert!(
            (exceed - 0.5).abs() < 0.15,
            "exceedance {exceed} should be near 0.5"
        );
    }

    /// A planted unit shift at σ = 0.25 is unreachable by label shuffles
    /// (p bottoms out at 1/(n+1)); same-distribution draws stay
    /// non-significant.
    #[test]
    fn permutation_test_detects_real_shifts_only() {
        let vc = ValueCompare {
            pairwise: PairwiseTests::On { permutations: 999 },
            ..cfg(200)
        };

        let mut shifted = TrajectoriesDataset::new();
        gaussian_state(&mut shifted, "A", 0.0, 0.25, 50, 31);
        gaussian_state(&mut shifted, "B", 1.0, 0.25, 50, 32);
        let m = vc.fit(&shifted).unwrap();
        let (_, _, _, p) = m.pairs().next().unwrap();
        assert!(p < 0.01, "planted shift got p = {p}");

        let mut null = TrajectoriesDataset::new();
        gaussian_state(&mut null, "C", 0.5, 0.25, 50, 41);
        gaussian_state(&mut null, "D", 0.5, 0.25, 50, 42);
        let m = vc.fit(&null).unwrap();
        let (_, _, _, p) = m.pairs().next().unwrap();
        assert!(p > 0.2, "identical distributions got p = {p}");
    }

    #[test]
    fn seeded_runs_are_byte_identical() {
        let mut d = TrajectoriesDataset::new();
        gaussian_state(&mut d, "A", 0.0, 0.5, 40, 51);
        gaussian_state(&mut d, "B", 0.5, 0.5, 40, 52);

        let vc = ValueCompare {
            method: ResampleScheme::BayesianBootstrap,
            pairwise: PairwiseTests::On { permutations: 199 },
            ..cfg(300)
        };
        let save = |m: &ValueCompareModel| {
            let mut buf = Vec::new();
            m.save_jsonl(&mut buf).unwrap();
            buf
        };
        assert_eq!(save(&vc.fit(&d).unwrap()), save(&vc.fit(&d).unwrap()));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = TrajectoriesDataset::new();
        gaussian_state(&mut d, "A", 0.0, 0.5, 30, 61);
        gaussian_state(&mut d, "B", 0.5, 0.5, 30, 62);

        let vc = ValueCompare {
            pairwise: PairwiseTests::On { permutations: 99 },
            ..cfg(200)
        };
        let m = vc.fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = ValueCompareModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);

        // pair stats survive the trip
        assert_eq!(m.pairs().count(), loaded.pairs().count());
    }

    #[test]
    fn invalid_params_are_rejected() {
        let mut d = TrajectoriesDataset::new();
        gaussian_state(&mut d, "A", 0.0, 0.5, 10, 71);

        assert!(matches!(cfg(1).fit(&d), Err(Error::InvalidInput(_))));

        for credible in [0.0, 1.0, -0.5, f64::NAN] {
            let vc = ValueCompare {
                credible,
                ..Default::default()
            };
            assert!(matches!(vc.fit(&d), Err(Error::InvalidInput(_))));
        }

        for gamma in [0.0, 1.5, f64::NAN] {
            let vc = ValueCompare {
                gamma,
                ..Default::default()
            };
            assert!(matches!(vc.fit(&d), Err(Error::InvalidInput(_))));
        }

        assert!(matches!(
            ValueCompare::default().fit(&TrajectoriesDataset::new()),
            Err(Error::EmptyDataset)
        ));
    }
}

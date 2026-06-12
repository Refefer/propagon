//! Standard multi-armed bandits (`docs/algorithms.md` §8.1; PRD FR-8).
//!
//! One state shape — per-arm sufficient statistics `(n, Σr, Σr²)` — and five
//! exploration policies over it: greedy, ε-greedy, UCB1, and Thompson
//! Sampling with Beta or Gaussian posteriors.
//!
//! Two faces (FR-8):
//! - **Ranker**: [`BanditModel::scores`] ranks arms by the policy's estimate
//!   (posterior mean / UCB index) from logged `(arm, reward)` data.
//! - **Policy**: [`BanditModel::select`] / [`BanditModel::select_k`] pick the
//!   next arm(s) to play. Stochastic policies are deterministic given the
//!   `seed` parameter and the model state (a persisted draw counter advances
//!   the stream, so save → load → select is indistinguishable from an
//!   uninterrupted run).
//!
//! Because the statistics are sufficient, [`BanditModel::merge`] of two state
//! files equals processing the concatenated logs.

use rand::SeedableRng;
use rand_distr::{Beta, Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::dataset::RewardsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// Exploration policy. Serialized into state files; mixing states produced
/// under different policies is a parameter mismatch.
///
/// EXP3 caveat: replaying a reward log importance-weights each row by the
/// probability the *current* mix would have played that arm — exact when
/// the log was produced by this policy, an approximation otherwise; its
/// state is order-dependent, so `merge` approximates (rather than equals)
/// the concatenated log.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BanditPolicy {
    /// Always exploit the best empirical mean.
    Greedy,
    /// Exploit with probability `1 − epsilon`, explore uniformly otherwise.
    EpsilonGreedy {
        /// Probability of an exploratory (random-arm) round.
        epsilon: f64,
    },
    /// Optimism in the face of uncertainty:
    /// `mean + sqrt(exploration · ln(t) / n)`. `exploration = 2.0` is the
    /// classic UCB1 of Auer et al. (2002). Unpulled arms rank first.
    Ucb1 {
        /// Scales the confidence radius `sqrt(exploration · ln(t) / n)`.
        exploration: f64,
    },
    /// Thompson Sampling with a Beta posterior; rewards must lie in `[0, 1]`.
    ThompsonBeta {
        /// Beta prior's `α` (pseudo-successes before any data).
        prior_alpha: f64,
        /// Beta prior's `β` (pseudo-failures before any data).
        prior_beta: f64,
    },
    /// Thompson Sampling with a Gaussian posterior over the mean.
    /// `prior_weight` acts as pseudo-observations of `prior_mean`.
    ThompsonGaussian {
        /// The prior's mean reward.
        prior_mean: f64,
        /// How many pseudo-observations of `prior_mean` the prior carries.
        prior_weight: f64,
    },
    /// KL-UCB (Garivier & Cappé 2011): the UCB1 idea with the exact
    /// Bernoulli-KL confidence set — uniformly better constants. Rewards
    /// must lie in `[0, 1]`. `c` scales the `ln ln t` term: the theory
    /// wants `c ≥ 3`, the paper recommends `c = 0` in practice.
    KlUcb {
        /// Weight on the `ln ln t` term in the confidence bound.
        c: f64,
    },
    /// EXP3 (Auer et al. 2002): adversarial-setting exponential weights
    /// with exploration mix `gamma`. Rewards must lie in `[0, 1]`.
    Exp3 {
        /// Fraction of each round's probability mass spread uniformly for
        /// exploration.
        gamma: f64,
    },
}

impl BanditPolicy {
    /// Canonical per-policy parameter values (the single source the CLI and
    /// `Default` both draw from).
    pub const DEFAULT_EPSILON: f64 = 0.1;
    /// Classic UCB1 exploration constant (Auer et al. 2002).
    pub const DEFAULT_EXPLORATION: f64 = 2.0;
    /// Uniform Beta prior.
    pub const DEFAULT_PRIOR_ALPHA: f64 = 1.0;
    /// Uniform Beta prior.
    pub const DEFAULT_PRIOR_BETA: f64 = 1.0;
    /// Zero-mean Gaussian prior.
    pub const DEFAULT_PRIOR_MEAN: f64 = 0.0;
    /// One pseudo-observation of the prior mean.
    pub const DEFAULT_PRIOR_WEIGHT: f64 = 1.0;
    /// Garivier & Cappé's practical recommendation (theory wants ≥ 3).
    pub const DEFAULT_KL_C: f64 = 0.0;
    /// Conventional EXP3 exploration mix.
    pub const DEFAULT_EXP3_GAMMA: f64 = 0.1;
}

impl Default for BanditPolicy {
    fn default() -> Self {
        BanditPolicy::Ucb1 {
            exploration: Self::DEFAULT_EXPLORATION,
        }
    }
}

/// Bandit algorithm parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bandit {
    /// Exploration policy used to rank and select arms.
    pub policy: BanditPolicy,
    /// Seeds the policy's random stream (ε-greedy exploration, TS draws).
    pub seed: u64,
}

impl Default for Bandit {
    fn default() -> Self {
        Self {
            policy: BanditPolicy::default(),
            seed: 42,
        }
    }
}

/// What `save_jsonl` writes as the header `params`: the algorithm params plus
/// the draw counter (state, persisted so selection streams resume exactly).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PersistedParams {
    policy: BanditPolicy,
    seed: u64,
    draws: u64,
}

/// One arm's sufficient statistics in the state file. `g` is EXP3's
/// cumulative log-weight (0 under every other policy).
#[derive(Debug, Serialize, Deserialize)]
struct ArmLine {
    id: String,
    n: u64,
    sum: f64,
    sum_sq: f64,
    g: f64,
}

/// Accumulating bandit state: per-arm `(n, Σr, Σr²)` plus the draw counter.
#[derive(Debug, Clone)]
pub struct BanditModel {
    params: Bandit,
    names: Interner,
    n: Vec<u64>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    /// EXP3 log-weights (unused by other policies).
    g: Vec<f64>,
    draws: u64,
}

impl BanditModel {
    /// Number of arms the model has seen.
    pub fn n_arms(&self) -> usize {
        self.names.len()
    }

    /// Total observations across arms.
    pub fn total_n(&self) -> u64 {
        self.n.iter().sum()
    }

    fn mean(&self, i: usize) -> f64 {
        if self.n[i] == 0 {
            0.0
        } else {
            self.sum[i] / self.n[i] as f64
        }
    }

    fn sample_var(&self, i: usize) -> f64 {
        if self.n[i] < 2 {
            return 1.0; // weakly-informative fallback before two observations
        }
        let n = self.n[i] as f64;
        ((self.sum_sq[i] - self.sum[i] * self.sum[i] / n) / (n - 1.0)).max(1e-12)
    }

    /// The policy's deterministic per-arm estimate (what [`RankModel::scores`]
    /// reports): empirical/posterior mean, or the UCB index for UCB1.
    fn estimate(&self, i: usize) -> f64 {
        match self.params.policy {
            BanditPolicy::Greedy | BanditPolicy::EpsilonGreedy { .. } => self.mean(i),
            BanditPolicy::Ucb1 { exploration } => {
                if self.n[i] == 0 {
                    return f64::INFINITY;
                }
                let t = self.total_n().max(1) as f64;
                self.mean(i) + (exploration * t.ln() / self.n[i] as f64).sqrt()
            }
            BanditPolicy::ThompsonBeta {
                prior_alpha,
                prior_beta,
            } => {
                let a = prior_alpha + self.sum[i];
                let b = prior_beta + self.n[i] as f64 - self.sum[i];
                a / (a + b)
            }
            BanditPolicy::ThompsonGaussian {
                prior_mean,
                prior_weight,
            } => (prior_weight * prior_mean + self.sum[i]) / (prior_weight + self.n[i] as f64),
            BanditPolicy::KlUcb { c } => self.kl_ucb_index(i, c),
            BanditPolicy::Exp3 { gamma } => self.exp3_probs(gamma)[i],
        }
    }

    /// `max{q ∈ [μ̂, 1) : n·kl(μ̂, q) ≤ ln t + c·ln ln t}` by bisection.
    fn kl_ucb_index(&self, i: usize, c: f64) -> f64 {
        if self.n[i] == 0 {
            return f64::INFINITY;
        }

        let n = self.n[i] as f64;
        let p = self.mean(i).clamp(0.0, 1.0);
        let t = self.total_n().max(1) as f64;
        let mut bound = t.ln();
        if c > 0.0 && t > std::f64::consts::E {
            bound += c * t.ln().ln();
        }
        bound /= n;

        let mut lo = p;
        let mut hi = 1.0 - 1e-12;
        if bernoulli_kl(p, hi) <= bound {
            return hi;
        }

        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if bernoulli_kl(p, mid) <= bound {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// EXP3 arm probabilities: `(1−γ)·softmax(g) + γ/K`.
    fn exp3_probs(&self, gamma: f64) -> Vec<f64> {
        let k = self.n_arms() as f64;
        let peak = self.g.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let w: Vec<f64> = self.g.iter().map(|&e| (e - peak).exp()).collect();
        let total: f64 = w.iter().sum();
        w.iter()
            .map(|&wi| (1.0 - gamma) * wi / total + gamma / k)
            .collect()
    }

    /// Advances the persisted draw counter and returns this round's RNG.
    fn next_rng(&mut self) -> Xoshiro256PlusPlus {
        let stream = self.params.seed ^ self.draws.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.draws += 1;
        Xoshiro256PlusPlus::seed_from_u64(stream)
    }

    /// One round of per-arm draw values under the policy (higher = play it).
    fn draw_values(&mut self) -> Result<Vec<f64>> {
        let k = self.n_arms();
        if k == 0 {
            return Err(Error::InvalidInput("bandit has no arms".into()));
        }
        let mut rng = self.next_rng();
        let values = match self.params.policy {
            BanditPolicy::Greedy | BanditPolicy::Ucb1 { .. } | BanditPolicy::KlUcb { .. } => {
                (0..k).map(|i| self.estimate(i)).collect()
            }
            BanditPolicy::Exp3 { gamma } => {
                // Gumbel-max: sorting log p + Gumbel noise samples arms
                // without replacement proportional to the EXP3 mix.
                self.exp3_probs(gamma)
                    .into_iter()
                    .map(|p| {
                        let u: f64 = rand::Rng::random(&mut rng);
                        p.ln() - (-u.ln()).ln()
                    })
                    .collect()
            }
            BanditPolicy::EpsilonGreedy { epsilon } => {
                if !(0.0..=1.0).contains(&epsilon) {
                    return Err(Error::InvalidInput(format!(
                        "epsilon {epsilon} not in [0,1]"
                    )));
                }
                // With prob ε, this round is exploratory: random arm ordering.
                let explore: f64 = rand::Rng::random(&mut rng);
                if explore < epsilon {
                    (0..k).map(|_| rand::Rng::random::<f64>(&mut rng)).collect()
                } else {
                    (0..k).map(|i| self.estimate(i)).collect()
                }
            }
            BanditPolicy::ThompsonBeta {
                prior_alpha,
                prior_beta,
            } => {
                let mut v = Vec::with_capacity(k);
                for i in 0..k {
                    let a = prior_alpha + self.sum[i];
                    let b = prior_beta + self.n[i] as f64 - self.sum[i];
                    let dist = Beta::new(a, b)
                        .map_err(|e| Error::Numeric(format!("beta posterior for arm {i}: {e}")))?;
                    v.push(dist.sample(&mut rng));
                }
                v
            }
            BanditPolicy::ThompsonGaussian {
                prior_mean,
                prior_weight,
            } => {
                let mut v = Vec::with_capacity(k);
                for i in 0..k {
                    let w = prior_weight + self.n[i] as f64;
                    let mean = (prior_weight * prior_mean + self.sum[i]) / w;
                    let sd = (self.sample_var(i) / w).sqrt();
                    let dist = Normal::new(mean, sd).map_err(|e| {
                        Error::Numeric(format!("gaussian posterior for arm {i}: {e}"))
                    })?;
                    v.push(dist.sample(&mut rng));
                }
                v
            }
        };
        Ok(values)
    }

    /// Picks the next arm to play.
    pub fn select(&mut self) -> Result<&str> {
        let id = self.select_k(1)?[0];
        Ok(self.names.resolve(id))
    }

    /// Picks `k` distinct arms for this round, best first (e.g. a traffic
    /// split). Errors if `k` exceeds the number of arms.
    pub fn select_k(&mut self, k: usize) -> Result<Vec<u32>> {
        if k > self.n_arms() {
            return Err(Error::InvalidInput(format!(
                "requested {k} arms but only {} exist",
                self.n_arms()
            )));
        }
        let values = self.draw_values()?;
        let mut idx: Vec<u32> = (0..self.n_arms() as u32).collect();
        idx.sort_unstable_by(|&a, &b| {
            values[b as usize]
                .total_cmp(&values[a as usize])
                .then_with(|| a.cmp(&b))
        });
        idx.truncate(k);
        Ok(idx)
    }

    /// Resolves an arm id from [`BanditModel::select_k`] back to its name.
    pub fn arm_name(&self, id: u32) -> Option<&str> {
        self.names.name(id)
    }

    /// Folds another state file into this one. Equivalent to having processed
    /// the concatenation of both logs (sufficient statistics add exactly).
    pub fn merge(&mut self, other: &BanditModel) -> Result<()> {
        if self.params != other.params {
            return Err(Error::ParamMismatch(format!(
                "cannot merge bandit states with different params: {:?} vs {:?}",
                self.params, other.params
            )));
        }
        for (i, name) in other.names.names().enumerate() {
            let idx = self.intern_arm(name);
            self.n[idx] += other.n[i];
            self.sum[idx] += other.sum[i];
            self.sum_sq[idx] += other.sum_sq[i];
            // EXP3 log-weights add; note its replay is order-dependent, so
            // merged EXP3 state approximates (rather than equals) the
            // concatenated log — unlike every other policy.
            self.g[idx] += other.g[i];
        }
        self.draws += other.draws;
        Ok(())
    }

    fn intern_arm(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.n.len() {
            self.n.push(0);
            self.sum.push(0.0);
            self.sum_sq.push(0.0);
            self.g.push(0.0);
        }
        idx
    }
}

impl RankModel for BanditModel {
    fn algorithm(&self) -> &'static str {
        "bandit"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, n)| (n, self.estimate(i)))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            policy: self.params.policy,
            seed: self.params.seed,
            draws: self.draws,
        };
        let lines: Vec<ArmLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| ArmLine {
                id: id.to_string(),
                n: self.n[i],
                sum: self.sum[i],
                sum_sq: self.sum_sq[i],
                g: self.g[i],
            })
            .collect();
        state::save_model(w, "bandit", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<ArmLine>) = state::load_model(r, "bandit")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params: Bandit {
                policy: params.policy,
                seed: params.seed,
            },
            names,
            n: lines.iter().map(|l| l.n).collect(),
            sum: lines.iter().map(|l| l.sum).collect(),
            sum_sq: lines.iter().map(|l| l.sum_sq).collect(),
            g: lines.iter().map(|l| l.g).collect(),
            draws: params.draws,
        })
    }
}

/// Bernoulli KL divergence `kl(p ‖ q)` with the 0·ln 0 = 0 convention.
fn bernoulli_kl(p: f64, q: f64) -> f64 {
    let term = |a: f64, b: f64| if a <= 0.0 { 0.0 } else { a * (a / b).ln() };
    term(p, q) + term(1.0 - p, 1.0 - q)
}

impl OnlineRanker for Bandit {
    type Data = RewardsDataset;
    type Model = BanditModel;

    fn init(&self) -> BanditModel {
        BanditModel {
            params: *self,
            names: Interner::new(),
            n: Vec::new(),
            sum: Vec::new(),
            sum_sq: Vec::new(),
            g: Vec::new(),
            draws: 0,
        }
    }

    fn update_opts(
        &self,
        model: &mut BanditModel,
        data: &RewardsDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        let bounded = matches!(
            self.policy,
            BanditPolicy::ThompsonBeta { .. }
                | BanditPolicy::KlUcb { .. }
                | BanditPolicy::Exp3 { .. }
        );
        if let BanditPolicy::Exp3 { gamma } = self.policy
            && !(gamma > 0.0 && gamma <= 1.0)
        {
            return Err(Error::InvalidInput(format!(
                "exp3 gamma must lie in (0, 1], got {gamma}"
            )));
        }

        for (arm, reward) in data.rows() {
            let r = f64::from(reward);
            if bounded && !(0.0..=1.0).contains(&r) {
                return Err(Error::InvalidInput(format!(
                    "this policy requires rewards in [0,1], got {r}"
                )));
            }
            let name = data.interner().resolve(arm);
            let idx = model.intern_arm(name);

            // EXP3 replays the log through the policy: importance-weight by
            // the probability the *current* mix assigns the played arm.
            // (Offline approximation: assumes the log's arms were drawn from
            // this policy — see the module docs.)
            if let BanditPolicy::Exp3 { gamma } = self.policy {
                let k = model.n_arms() as f64;
                let p = model.exp3_probs(gamma)[idx];
                model.g[idx] += gamma * (r / p) / k;
            }

            model.n[idx] += 1;
            model.sum[idx] += r;
            model.sum_sq[idx] += r * r;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewards(rows: &[(&str, f32)]) -> RewardsDataset {
        let mut d = RewardsDataset::new();
        for (a, r) in rows {
            d.push(a, *r);
        }
        d
    }

    fn fit(policy: BanditPolicy, rows: &[(&str, f32)]) -> BanditModel {
        let b = Bandit { policy, seed: 7 };
        let mut m = b.init();
        b.update(&mut m, &rewards(rows)).unwrap();
        m
    }

    #[test]
    fn ucb1_matches_hand_computation() {
        // A: 2 pulls mean 0.5; B: 1 pull mean 1.0; t = 3
        let mut m = fit(
            BanditPolicy::Ucb1 { exploration: 2.0 },
            &[("A", 1.0), ("A", 0.0), ("B", 1.0)],
        );
        let t: f64 = 3.0;
        let ucb_a = 0.5 + (2.0 * t.ln() / 2.0).sqrt();
        let ucb_b = 1.0 + (2.0 * t.ln() / 1.0).sqrt();
        let s: std::collections::HashMap<_, _> =
            m.scores().map(|(n, v)| (n.to_string(), v)).collect();
        assert!((s["A"] - ucb_a).abs() < 1e-12);
        assert!((s["B"] - ucb_b).abs() < 1e-12);
        assert_eq!(m.select().unwrap(), "B");
    }

    #[test]
    fn unpulled_arms_rank_first_under_ucb() {
        let b = Bandit::default();
        let mut m = b.init();
        b.update(&mut m, &rewards(&[("A", 1.0)])).unwrap();
        m.intern_arm("FRESH"); // known to the model, never pulled
        assert_eq!(m.select().unwrap(), "FRESH");
    }

    #[test]
    fn thompson_beta_is_seed_deterministic_and_resumable() {
        let rows = &[("A", 1.0), ("A", 1.0), ("B", 0.0), ("B", 1.0)];
        let policy = BanditPolicy::ThompsonBeta {
            prior_alpha: 1.0,
            prior_beta: 1.0,
        };

        let mut m1 = fit(policy, rows);
        let mut m2 = fit(policy, rows);
        let seq1: Vec<String> = (0..20).map(|_| m1.select().unwrap().to_string()).collect();
        let seq2: Vec<String> = (0..20).map(|_| m2.select().unwrap().to_string()).collect();
        assert_eq!(seq1, seq2, "same seed + state => same selection stream");

        // save mid-stream, resume, and the stream continues identically
        let mut m3 = fit(policy, rows);
        let head: Vec<String> = (0..10).map(|_| m3.select().unwrap().to_string()).collect();
        let mut buf = Vec::new();
        m3.save_jsonl(&mut buf).unwrap();
        let mut m4 = BanditModel::load_jsonl(buf.as_slice()).unwrap();
        let tail3: Vec<String> = (0..10).map(|_| m3.select().unwrap().to_string()).collect();
        let tail4: Vec<String> = (0..10).map(|_| m4.select().unwrap().to_string()).collect();
        assert_eq!(head, seq1[..10].to_vec());
        assert_eq!(tail3, tail4, "save -> load -> select is indistinguishable");
    }

    #[test]
    fn thompson_beta_rejects_out_of_range_rewards() {
        let b = Bandit {
            policy: BanditPolicy::ThompsonBeta {
                prior_alpha: 1.0,
                prior_beta: 1.0,
            },
            seed: 1,
        };
        let mut m = b.init();
        assert!(b.update(&mut m, &rewards(&[("A", 2.0)])).is_err());
    }

    #[test]
    fn merge_equals_concatenated_logs() {
        let b = Bandit {
            policy: BanditPolicy::ThompsonGaussian {
                prior_mean: 0.0,
                prior_weight: 1.0,
            },
            seed: 3,
        };
        let log1 = &[("A", 1.0), ("B", 3.0)];
        let log2 = &[("B", 5.0), ("C", 2.0), ("A", 0.0)];

        let mut split_a = b.init();
        b.update(&mut split_a, &rewards(log1)).unwrap();
        let mut split_b = b.init();
        b.update(&mut split_b, &rewards(log2)).unwrap();
        split_a.merge(&split_b).unwrap();

        let mut joint = b.init();
        let mut all = log1.to_vec();
        all.extend_from_slice(log2);
        b.update(&mut joint, &rewards(&all)).unwrap();

        let mut buf1 = Vec::new();
        split_a.save_jsonl(&mut buf1).unwrap();
        let mut buf2 = Vec::new();
        joint.save_jsonl(&mut buf2).unwrap();
        assert_eq!(
            buf1, buf2,
            "merged state file == concatenated-log state file"
        );
    }

    #[test]
    fn epsilon_greedy_mostly_exploits() {
        let mut m = fit(
            BanditPolicy::EpsilonGreedy { epsilon: 0.1 },
            &[("GOOD", 1.0), ("GOOD", 1.0), ("BAD", 0.0), ("BAD", 0.0)],
        );
        let picks: Vec<String> = (0..200).map(|_| m.select().unwrap().to_string()).collect();
        let good = picks.iter().filter(|p| *p == "GOOD").count();
        assert!(good > 150, "exploited only {good}/200 times");
        assert!(good < 200, "never explored in 200 rounds");
    }
}

#[cfg(test)]
mod new_policy_tests {
    use super::*;

    fn rewards(rows: &[(&str, f32)]) -> RewardsDataset {
        let mut d = RewardsDataset::new();
        for (a, r) in rows {
            d.push(a, *r);
        }
        d
    }

    /// The bisected index satisfies its defining bound to solver precision:
    /// n·kl(μ̂, q*) ≈ ln t and no larger q qualifies.
    #[test]
    fn kl_ucb_index_solves_the_bound() {
        let b = Bandit {
            policy: BanditPolicy::KlUcb {
                c: BanditPolicy::DEFAULT_KL_C,
            },
            seed: 1,
        };
        let mut m = b.init();
        b.update(
            &mut m,
            &rewards(&[("a", 1.0), ("a", 0.0), ("a", 1.0), ("b", 1.0)]),
        )
        .unwrap();

        // arm a: n=3, mean=2/3, t=4 → bound = ln(4)/3.
        let q = m.kl_ucb_index(0, 0.0);
        let bound = 4f64.ln() / 3.0;
        assert!((bernoulli_kl(2.0 / 3.0, q) - bound).abs() < 1e-9, "{q}");
        assert!(q > 2.0 / 3.0 && q < 1.0);
        assert!(
            bernoulli_kl(2.0 / 3.0, (q + 1e-6).min(1.0 - 1e-12)) > bound,
            "q is maximal"
        );
    }

    /// KL-UCB dominates the empirical mean and shrinks with more data.
    #[test]
    fn kl_ucb_shrinks_with_evidence() {
        let b = Bandit {
            policy: BanditPolicy::KlUcb { c: 0.0 },
            seed: 1,
        };

        let mut small = b.init();
        b.update(&mut small, &rewards(&[("a", 1.0), ("a", 0.0), ("b", 0.0)]))
            .unwrap();
        let mut big = b.init();
        let many: Vec<(&str, f32)> = (0..200)
            .map(|i| ("a", if i % 2 == 0 { 1.0 } else { 0.0 }))
            .chain(std::iter::once(("b", 0.0)))
            .collect();
        b.update(&mut big, &rewards(&many)).unwrap();

        let gap_small = small.kl_ucb_index(0, 0.0) - 0.5;
        let gap_big = big.kl_ucb_index(0, 0.0) - 0.5;
        assert!(gap_big < gap_small, "{gap_big} vs {gap_small}");
    }

    /// Hand-computed EXP3 replay, two arms, two rows, γ = 0.5:
    /// row 1 ("a", 1): p_a = 0.5 → g_a += 0.5·(1/0.5)/2 = 0.5;
    /// row 2 ("a", 1): w = (e^0.5, 1), p_a = 0.5·e^.5/(e^.5+1) + 0.25
    ///   → g_a += 0.5·(1/p_a)/2.
    #[test]
    fn exp3_replay_arithmetic() {
        let b = Bandit {
            policy: BanditPolicy::Exp3 { gamma: 0.5 },
            seed: 1,
        };
        let mut m = b.init();
        // Seed both arms first (single-arm rounds would use k=1).
        b.update(&mut m, &rewards(&[("a", 0.0), ("b", 0.0)]))
            .unwrap();
        assert_eq!(m.g, vec![0.0, 0.0]);

        b.update(&mut m, &rewards(&[("a", 1.0)])).unwrap();
        assert!((m.g[0] - 0.5).abs() < 1e-12, "{}", m.g[0]);

        b.update(&mut m, &rewards(&[("a", 1.0)])).unwrap();
        let e = 0.5f64.exp();
        let p_a = 0.5 * e / (e + 1.0) + 0.25;
        assert!(
            (m.g[0] - (0.5 + 0.5 / p_a / 2.0)).abs() < 1e-12,
            "{}",
            m.g[0]
        );

        // Probabilities favor the rewarded arm but keep the γ/K floor.
        let probs = m.exp3_probs(0.5);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] >= 0.25 - 1e-12);
    }

    #[test]
    fn exp3_invalid_gamma_and_rewards_rejected() {
        let bad_gamma = Bandit {
            policy: BanditPolicy::Exp3 { gamma: 0.0 },
            seed: 1,
        };
        let mut m = bad_gamma.init();
        assert!(bad_gamma.update(&mut m, &rewards(&[("a", 1.0)])).is_err());

        let b = Bandit {
            policy: BanditPolicy::KlUcb { c: 0.0 },
            seed: 1,
        };
        let mut m = b.init();
        assert!(b.update(&mut m, &rewards(&[("a", 2.0)])).is_err());
    }

    /// Seeded EXP3 selection is reproducible and state round-trips with the
    /// log-weight column intact.
    #[test]
    fn exp3_round_trip_and_determinism() {
        let b = Bandit {
            policy: BanditPolicy::Exp3 {
                gamma: BanditPolicy::DEFAULT_EXP3_GAMMA,
            },
            seed: 9,
        };
        let mut m = b.init();
        b.update(
            &mut m,
            &rewards(&[("a", 1.0), ("b", 0.0), ("a", 1.0), ("c", 1.0)]),
        )
        .unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let mut loaded = BanditModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);

        // Same state, same draw counter → identical selection.
        assert_eq!(m.select().unwrap(), loaded.select().unwrap());
    }
}

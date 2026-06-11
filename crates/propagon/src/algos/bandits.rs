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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BanditPolicy {
    /// Always exploit the best empirical mean.
    Greedy,
    /// Exploit with probability `1 − epsilon`, explore uniformly otherwise.
    EpsilonGreedy { epsilon: f64 },
    /// Optimism in the face of uncertainty:
    /// `mean + sqrt(exploration · ln(t) / n)`. `exploration = 2.0` is the
    /// classic UCB1 of Auer et al. (2002). Unpulled arms rank first.
    Ucb1 { exploration: f64 },
    /// Thompson Sampling with a Beta posterior; rewards must lie in `[0, 1]`.
    ThompsonBeta { prior_alpha: f64, prior_beta: f64 },
    /// Thompson Sampling with a Gaussian posterior over the mean.
    /// `prior_weight` acts as pseudo-observations of `prior_mean`.
    ThompsonGaussian { prior_mean: f64, prior_weight: f64 },
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

/// One arm's sufficient statistics in the state file.
#[derive(Debug, Serialize, Deserialize)]
struct ArmLine {
    id: String,
    n: u64,
    sum: f64,
    sum_sq: f64,
}

/// Accumulating bandit state: per-arm `(n, Σr, Σr²)` plus the draw counter.
#[derive(Debug, Clone)]
pub struct BanditModel {
    params: Bandit,
    names: Interner,
    n: Vec<u64>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    draws: u64,
}

impl BanditModel {
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
        }
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
            BanditPolicy::Greedy | BanditPolicy::Ucb1 { .. } => {
                (0..k).map(|i| self.estimate(i)).collect()
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
            draws: params.draws,
        })
    }
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
            draws: 0,
        }
    }

    fn update_opts(
        &self,
        model: &mut BanditModel,
        data: &RewardsDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        let beta = matches!(self.policy, BanditPolicy::ThompsonBeta { .. });
        for (arm, reward) in data.rows() {
            let r = f64::from(reward);
            if beta && !(0.0..=1.0).contains(&r) {
                return Err(Error::InvalidInput(format!(
                    "thompson-beta requires rewards in [0,1], got {r}"
                )));
            }
            let name = data.interner().resolve(arm);
            let idx = model.intern_arm(name);
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

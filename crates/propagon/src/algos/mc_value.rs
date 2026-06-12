//! Monte Carlo state-value estimation (`docs/algorithms.md` §13.1;
//! Sutton & Barto 2018).
//!
//! Rolls discounted returns backward through each episode
//! (`G_t = r_t + γ·G_{t+1}`, terminal `G = 0`) and aggregates them per
//! state: first-visit keeps only each state's first occurrence per episode
//! (cleaner i.i.d. structure), every-visit keeps all of them (more data,
//! within-episode correlation). The aggregate is a mean or median,
//! optionally after winsorizing each state's samples at empirical
//! quantiles — the knobs for heavy-tailed returns (revenue).
//!
//! Estimates are on-policy: they rank states under whatever behavior
//! generated the logs. Episodes are processed in parallel but merged in
//! episode order, so results are bit-stable at any thread count.
//!
//! Gotcha: states with fewer than `min_observations` samples are excluded
//! from the model entirely — the model interner holds only emitted states,
//! so they are absent from scores and state files, not scored zero.

use serde::{Deserialize, Serialize};

use crate::algos::common::{self, ScoreCountLine};
use crate::dataset::TrajectoriesDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx::quantile;
use crate::parallel;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Which occurrences of a state within one episode contribute samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Visit {
    /// Only the first occurrence per episode (Sutton & Barto's first-visit
    /// MC: one sample per episode per state, cleaner i.i.d. structure).
    #[default]
    First,
    /// Every occurrence (more samples, within-episode correlation).
    Every,
}

/// How a state's return samples collapse into one estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Aggregate {
    /// Arithmetic mean of the return samples.
    #[default]
    Mean,
    /// Robust to outliers; the empirical 0.5-quantile with linear
    /// interpolation between the two central samples.
    Median,
}

/// Optional per-state outlier clamping applied before aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Winsorize {
    /// No clamping; samples are aggregated as-is.
    #[default]
    Off,
    /// Clamp each state's samples into its own `[q, 1−q]` empirical
    /// quantiles; `q` must lie in `(0, 0.5)`.
    Percentile(f64),
}

/// Monte Carlo value-estimation parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct McValue {
    /// Discount factor in `(0, 1]`; 1 weighs all future rewards equally.
    pub gamma: f64,
    /// Which within-episode occurrences of a state contribute samples.
    pub visit: Visit,
    /// How a state's return samples collapse into one estimate.
    pub aggregate: Aggregate,
    /// Optional per-state outlier clamping applied before aggregation.
    pub winsorize: Winsorize,
    /// States with fewer samples than this are excluded from the model.
    pub min_observations: usize,
}

impl Default for McValue {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            visit: Visit::default(),
            aggregate: Aggregate::default(),
            winsorize: Winsorize::default(),
            min_observations: 1,
        }
    }
}

impl McValue {
    /// Rejects parameter values outside their documented domains (NaN
    /// included — the comparisons are written to fail on it).
    fn validate(&self) -> Result<()> {
        if !(self.gamma > 0.0 && self.gamma <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "gamma must lie in (0, 1], got {}",
                self.gamma
            )));
        }

        if let Winsorize::Percentile(q) = self.winsorize
            && !(q > 0.0 && q < 0.5)
        {
            return Err(Error::InvalidInput(format!(
                "winsorize percentile must lie in (0, 0.5), got {q}"
            )));
        }
        Ok(())
    }
}

/// One episode's `(state, discounted return)` samples: a backward pass
/// computes `G_t = r_t + γ·G_{t+1}` (terminal `G = 0`), then occurrences
/// are emitted in step order — all of them under [`Visit::Every`], only
/// each state's first under [`Visit::First`]. Shared with `value_compare`.
pub(crate) fn episode_returns(
    states: &[u32],
    rewards: &[f32],
    gamma: f64,
    visit: Visit,
) -> Vec<(u32, f64)> {
    let mut returns = vec![0.0f64; states.len()];
    let mut g = 0.0f64;

    for t in (0..states.len()).rev() {
        g = f64::from(rewards[t]) + gamma * g;
        returns[t] = g;
    }

    match visit {
        Visit::Every => states.iter().copied().zip(returns).collect(),
        Visit::First => {
            let mut seen = std::collections::HashSet::new();
            states
                .iter()
                .copied()
                .zip(returns)
                .filter(|&(s, _)| seen.insert(s))
                .collect()
        }
    }
}

/// Estimated state values with their sample counts.
#[derive(Debug, Clone)]
pub struct McValueModel {
    params: McValue,
    names: Interner,
    estimates: Vec<f64>,
    counts: Vec<u64>,
}

impl McValueModel {
    /// `(name, number of return samples)` per emitted state.
    pub fn counts(&self) -> impl Iterator<Item = (&str, u64)> {
        self.names.names().zip(self.counts.iter().copied())
    }
}

impl RankModel for McValueModel {
    fn algorithm(&self) -> &'static str {
        "mc-value"
    }

    /// Estimated `V(s)` per emitted state (mean or median return).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.estimates.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines = common::score_count_lines(&self.names, &self.estimates, &self.counts);
        state::save_model(w, "mc-value", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (McValue, Vec<ScoreCountLine>) = state::load_model(r, "mc-value")?;
        let (names, estimates, counts) = common::from_score_count_lines(lines)?;
        Ok(Self {
            params,
            names,
            estimates,
            counts,
        })
    }
}

impl Ranker for McValue {
    type Data = TrajectoriesDataset;
    type Model = McValueModel;

    /// One backward pass per episode (parallel, merged sequentially in
    /// episode order), then per-state winsorize + aggregate.
    fn fit_opts(&self, data: &TrajectoriesDataset, opts: &FitOptions<'_>) -> Result<McValueModel> {
        self.validate()?;
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let per_episode: Vec<Vec<(u32, f64)>> = parallel::run_scoped(opts, || {
            parallel::par_map_indexed(data.n_episodes(), |e| {
                let (states, rewards) = data.episode(e);
                episode_returns(states, rewards, self.gamma, self.visit)
            })
        });

        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); data.n_entities()];
        for episode in &per_episode {
            for &(s, g) in episode {
                samples[s as usize].push(g);
            }
        }

        // A zero floor would emit unobserved states with NaN estimates;
        // every emitted state needs at least one sample.
        let floor = self.min_observations.max(1);
        let mut names = Interner::new();
        let mut estimates = Vec::new();
        let mut counts = Vec::new();

        for (id, smp) in samples.iter_mut().enumerate() {
            if smp.len() < floor {
                continue;
            }

            if let Winsorize::Percentile(q) = self.winsorize {
                let mut sorted = smp.clone();
                sorted.sort_by(f64::total_cmp);
                let lo = quantile(&sorted, q);
                let hi = quantile(&sorted, 1.0 - q);
                for v in smp.iter_mut() {
                    *v = v.max(lo).min(hi);
                }
            }

            let est = match self.aggregate {
                Aggregate::Mean => smp.iter().sum::<f64>() / smp.len() as f64,
                Aggregate::Median => {
                    let mut sorted = smp.clone();
                    sorted.sort_by(f64::total_cmp);
                    quantile(&sorted, 0.5)
                }
            };

            names.intern(data.interner().resolve(id as u32));
            estimates.push(est);
            counts.push(smp.len() as u64);
        }

        if estimates.is_empty() {
            return Err(Error::InvalidInput("no state met min_observations".into()));
        }
        Ok(McValueModel {
            params: *self,
            names,
            estimates,
            counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two episodes with a repeated state:
    /// ep1 = (A,1)(B,0)(A,2)(C,1); ep2 = (B,1)(A,0.5).
    fn fixture() -> TrajectoriesDataset {
        let mut d = TrajectoriesDataset::new();
        d.push_step("A", 1.0).unwrap();
        d.push_step("B", 0.0).unwrap();
        d.push_step("A", 2.0).unwrap();
        d.push_step("C", 1.0).unwrap();
        d.end_episode();
        d.push_step("B", 1.0).unwrap();
        d.push_step("A", 0.5).unwrap();
        d.end_episode();
        d
    }

    fn scores(m: &McValueModel) -> std::collections::HashMap<String, f64> {
        m.scores().map(|(n, s)| (n.to_string(), s)).collect()
    }

    /// γ = 0.9 backward pass, episode 1: G3 = 1; G2 = 2 + .9·1 = 2.9;
    /// G1 = 0 + .9·2.9 = 2.61; G0 = 1 + .9·2.61 = 3.349.
    /// Episode 2: G1 = 0.5; G0 = 1 + .9·0.5 = 1.45.
    ///
    /// First-visit samples: A {3.349, 0.5}, B {2.61, 1.45}, C {1.0}
    ///   → V(A) = 1.9245, V(B) = 2.03, V(C) = 1.0.
    /// Every-visit adds A's second occurrence (2.9)
    ///   → V(A) = (3.349 + 2.9 + 0.5)/3 = 6.749/3.
    #[test]
    fn first_and_every_visit_match_hand_solution() {
        let first = McValue {
            gamma: 0.9,
            ..Default::default()
        }
        .fit(&fixture())
        .unwrap();
        let s = scores(&first);
        assert!((s["A"] - 1.9245).abs() < 1e-12);
        assert!((s["B"] - 2.03).abs() < 1e-12);
        assert!((s["C"] - 1.0).abs() < 1e-12);
        let counts: std::collections::HashMap<_, _> = first.counts().collect();
        assert_eq!(counts["A"], 2);
        assert_eq!(counts["C"], 1);

        let every = McValue {
            gamma: 0.9,
            visit: Visit::Every,
            ..Default::default()
        }
        .fit(&fixture())
        .unwrap();
        let s = scores(&every);
        assert!((s["A"] - 6.749 / 3.0).abs() < 1e-12);
        assert!((s["B"] - 2.03).abs() < 1e-12);
        let counts: std::collections::HashMap<_, _> = every.counts().collect();
        assert_eq!(counts["A"], 3);
    }

    /// γ = 1 returns are plain reward sums: episode (A,1)(B,2)(A,3) gives
    /// first-visit A = 6, B = 5; every-visit A = (6 + 3)/2 = 4.5.
    #[test]
    fn undiscounted_returns_are_plain_sums() {
        let mut d = TrajectoriesDataset::new();
        d.push_step("A", 1.0).unwrap();
        d.push_step("B", 2.0).unwrap();
        d.push_step("A", 3.0).unwrap();
        d.end_episode();

        let s = scores(&McValue::default().fit(&d).unwrap());
        assert!((s["A"] - 6.0).abs() < 1e-12);
        assert!((s["B"] - 5.0).abs() < 1e-12);

        let every = McValue {
            visit: Visit::Every,
            ..Default::default()
        };
        let s = scores(&every.fit(&d).unwrap());
        assert!((s["A"] - 4.5).abs() < 1e-12);
    }

    /// One state with single-step episodes 1, 2, 3, 4, 100 (γ = 1, so the
    /// samples are the rewards). q = 0.25 on the sorted samples puts the
    /// clamp window at [quantile(.25), quantile(.75)] = [2, 4], so the
    /// clamped samples are {2, 2, 3, 4, 4} with mean exactly 3. Without
    /// winsorizing the outlier drags the mean to 110/5 = 22.
    #[test]
    fn winsorize_clamps_a_planted_outlier() {
        let mut d = TrajectoriesDataset::new();
        for r in [1.0, 2.0, 3.0, 4.0, 100.0] {
            d.push_step("S", r).unwrap();
            d.end_episode();
        }

        let raw = scores(&McValue::default().fit(&d).unwrap());
        assert!((raw["S"] - 22.0).abs() < 1e-12);

        let wins = McValue {
            winsorize: Winsorize::Percentile(0.25),
            ..Default::default()
        };
        let s = scores(&wins.fit(&d).unwrap());
        assert!((s["S"] - 3.0).abs() < 1e-12);
    }

    /// Median of {1, 2, 3, 4, 100} is 3 (outlier-immune); an even count
    /// {1, 2, 3, 10} interpolates the central pair to 2.5.
    #[test]
    fn median_aggregate() {
        let mut d = TrajectoriesDataset::new();
        for r in [1.0, 2.0, 3.0, 4.0, 100.0] {
            d.push_step("S", r).unwrap();
            d.end_episode();
        }
        for r in [1.0, 2.0, 3.0, 10.0] {
            d.push_step("T", r).unwrap();
            d.end_episode();
        }

        let median = McValue {
            aggregate: Aggregate::Median,
            ..Default::default()
        };
        let s = scores(&median.fit(&d).unwrap());
        assert!((s["S"] - 3.0).abs() < 1e-12);
        assert!((s["T"] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn min_observations_excludes_sparse_states() {
        let cfg = McValue {
            gamma: 0.9,
            min_observations: 2,
            ..Default::default()
        };
        let m = cfg.fit(&fixture()).unwrap();
        // C has one sample and is absent from the model entirely.
        let s = scores(&m);
        assert_eq!(s.len(), 2);
        assert!(!s.contains_key("C"));

        let all_excluded = McValue {
            min_observations: 10,
            ..Default::default()
        };
        assert!(matches!(
            all_excluded.fit(&fixture()),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let m = McValue {
            gamma: 0.9,
            winsorize: Winsorize::Percentile(0.1),
            ..Default::default()
        }
        .fit(&fixture())
        .unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = McValueModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(matches!(
            McValue::default().fit(&TrajectoriesDataset::new()),
            Err(Error::EmptyDataset)
        ));

        for gamma in [0.0, -0.5, 1.5, f64::NAN] {
            let cfg = McValue {
                gamma,
                ..Default::default()
            };
            assert!(matches!(cfg.fit(&fixture()), Err(Error::InvalidInput(_))));
        }

        for q in [0.0, 0.5, 0.9, f64::NAN] {
            let cfg = McValue {
                winsorize: Winsorize::Percentile(q),
                ..Default::default()
            };
            assert!(matches!(cfg.fit(&fixture()), Err(Error::InvalidInput(_))));
        }
    }
}

//! TD(0) state-value estimation (`docs/algorithms.md` §13.3;
//! Sutton & Barto 2018).
//!
//! Learns `V(s)` from individual transitions instead of complete episodes:
//! each step nudges the state's value toward the observed reward plus the
//! discounted value of the next state — `V(s_t) += α·(r_t + γ·V(s_{t+1}) −
//! V(s_t))` — and the final step of an episode bootstraps zero. Trades
//! Monte Carlo's variance for bias (the estimate leans on its own current
//! values).
//!
//! Order-dependent by definition: [`OnlineRanker::update_opts`] processes
//! episodes (and steps within them) in dataset insertion order, and state
//! persists across calls — replaying the same log in a different order
//! gives different values. `passes` repeats the sweep over the batch within
//! one update call; many passes over a fixed log converge toward the
//! certainty-equivalent values.

use serde::{Deserialize, Serialize};

use crate::algos::common::{self, ScoreCountLine};
use crate::dataset::TrajectoriesDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// TD(0) parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TdValue {
    /// Step size in `(0, 1]`; constant (no decay), so the estimate stays
    /// biased toward recent data — the online-tracking trade-off.
    pub alpha: f64,
    /// Discount factor in `(0, 1]`.
    pub gamma: f64,
    /// Sweeps over the batch within one update call.
    pub passes: usize,
    /// Value assigned to states on first sight.
    pub initial_value: f64,
}

impl Default for TdValue {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            gamma: 1.0,
            passes: 1,
            initial_value: 0.0,
        }
    }
}

impl TdValue {
    /// Rejects parameter values outside their documented domains (NaN
    /// included — the comparisons are written to fail on it).
    fn validate(&self) -> Result<()> {
        if !(self.alpha > 0.0 && self.alpha <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "alpha must lie in (0, 1], got {}",
                self.alpha
            )));
        }
        if !(self.gamma > 0.0 && self.gamma <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "gamma must lie in (0, 1], got {}",
                self.gamma
            )));
        }
        Ok(())
    }
}

/// TD value estimates keyed by state name, with per-state update counts.
#[derive(Debug, Clone)]
pub struct TdValueModel {
    params: TdValue,
    names: Interner,
    values: Vec<f64>,
    /// TD updates applied per state (each pass over a step counts once).
    counts: Vec<u64>,
}

impl TdValueModel {
    /// Index for `name`, interning unseen states at `initial_value` (the
    /// Elo idx pattern).
    fn idx(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.values.len() {
            self.values.push(self.params.initial_value);
            self.counts.push(0);
        }
        idx
    }
}

impl RankModel for TdValueModel {
    fn algorithm(&self) -> &'static str {
        "td-value"
    }

    /// Current `V(s)` per state.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.values.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines = common::score_count_lines(&self.names, &self.values, &self.counts);
        state::save_model(w, "td-value", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (TdValue, Vec<ScoreCountLine>) = state::load_model(r, "td-value")?;
        let (names, values, counts) = common::from_score_count_lines(lines)?;
        Ok(Self {
            params,
            names,
            values,
            counts,
        })
    }
}

impl OnlineRanker for TdValue {
    type Data = TrajectoriesDataset;
    type Model = TdValueModel;

    fn init(&self) -> TdValueModel {
        TdValueModel {
            params: *self,
            names: Interner::new(),
            values: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// `passes` sweeps over the batch; within each sweep, episodes and the
    /// steps inside them are processed in insertion order.
    fn update_opts(
        &self,
        model: &mut TdValueModel,
        data: &TrajectoriesDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        self.validate()?;

        for _ in 0..self.passes {
            for (states, rewards) in data.episodes() {
                for t in 0..states.len() {
                    let si = model.idx(data.interner().resolve(states[t]));

                    let target = match states.get(t + 1) {
                        Some(&next) => {
                            let ni = model.idx(data.interner().resolve(next));
                            f64::from(rewards[t]) + self.gamma * model.values[ni]
                        }
                        // Final step: the episode ends, bootstrap zero.
                        None => f64::from(rewards[t]),
                    };

                    model.values[si] += self.alpha * (target - model.values[si]);
                    model.counts[si] += 1;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(m: &TdValueModel) -> std::collections::HashMap<String, f64> {
        m.scores().map(|(n, s)| (n.to_string(), s)).collect()
    }

    /// Hand trace at α = 0.5, γ = 1, initial 0.
    /// Episode 1 = (A,1)(B,0):
    ///   t0: V(A) += .5·(1 + V(B) − V(A)) = .5·(1 + 0 − 0)   → V(A) = 0.5
    ///   t1: V(B) += .5·(0 − V(B)) = 0                       → V(B) = 0
    /// Episode 2 = (A,0)(B,2):
    ///   t0: V(A) += .5·(0 + V(B) − V(A)) = .5·(0 − 0.5)     → V(A) = 0.25
    ///   t1: V(B) += .5·(2 − V(B)) = 1                       → V(B) = 1
    /// All updates are exact in binary (powers of two).
    #[test]
    fn updates_match_hand_trace() {
        let td = TdValue {
            alpha: 0.5,
            ..Default::default()
        };
        let mut d = TrajectoriesDataset::new();
        d.push_step("A", 1.0).unwrap();
        d.push_step("B", 0.0).unwrap();
        d.end_episode();
        d.push_step("A", 0.0).unwrap();
        d.push_step("B", 2.0).unwrap();
        d.end_episode();

        let mut m = td.init();
        td.update(&mut m, &d).unwrap();
        let s = scores(&m);
        assert_eq!(s["A"], 0.25);
        assert_eq!(s["B"], 1.0);
        let n: std::collections::HashMap<_, _> =
            m.names.names().zip(m.counts.iter().copied()).collect();
        assert_eq!(n["A"], 2);
        assert_eq!(n["B"], 2);
    }

    /// Unseen states intern at `initial_value`; the bootstrap target for a
    /// non-terminal step reads it. With initial 5, α = 0.5, γ = 1, on the
    /// single episode (X,1)(Y,0):
    ///   t0: V(X) += .5·(1 + 5 − 5) = 0.5 → 5.5
    ///   t1: V(Y) += .5·(0 − 5) = −2.5    → 2.5
    #[test]
    fn initial_value_seeds_unseen_states() {
        let td = TdValue {
            alpha: 0.5,
            initial_value: 5.0,
            ..Default::default()
        };
        let mut d = TrajectoriesDataset::new();
        d.push_step("X", 1.0).unwrap();
        d.push_step("Y", 0.0).unwrap();
        d.end_episode();

        let mut m = td.init();
        td.update(&mut m, &d).unwrap();
        let s = scores(&m);
        assert_eq!(s["X"], 5.5);
        assert_eq!(s["Y"], 2.5);
    }

    /// Repeated sweeps over the deterministic 2-state MRP
    /// (S1,1) → (S2,2) → end, γ = 0.9, approach the certainty-equivalent
    /// values V(S2) = 2 and V(S1) = 1 + 0.9·2 = 2.8.
    #[test]
    fn many_passes_converge_to_analytic_values() {
        let td = TdValue {
            alpha: 0.1,
            gamma: 0.9,
            passes: 1000,
            ..Default::default()
        };
        let mut d = TrajectoriesDataset::new();
        d.push_step("S1", 1.0).unwrap();
        d.push_step("S2", 2.0).unwrap();
        d.end_episode();

        let mut m = td.init();
        td.update(&mut m, &d).unwrap();
        let s = scores(&m);
        assert!((s["S2"] - 2.0).abs() < 1e-9, "{}", s["S2"]);
        assert!((s["S1"] - 2.8).abs() < 1e-9, "{}", s["S1"]);
    }

    /// Two updates over split batches equal one update over the
    /// concatenated batch (state persists; history is never replayed).
    #[test]
    fn state_persists_across_update_calls() {
        let td = TdValue {
            alpha: 0.5,
            ..Default::default()
        };
        let mut d1 = TrajectoriesDataset::new();
        d1.push_step("A", 1.0).unwrap();
        d1.push_step("B", 0.0).unwrap();
        d1.end_episode();
        let mut d2 = TrajectoriesDataset::new();
        d2.push_step("A", 0.0).unwrap();
        d2.push_step("B", 2.0).unwrap();
        d2.end_episode();

        let mut split = td.init();
        td.update(&mut split, &d1).unwrap();
        td.update(&mut split, &d2).unwrap();

        // Matches the one-batch hand trace above.
        let s = scores(&split);
        assert_eq!(s["A"], 0.25);
        assert_eq!(s["B"], 1.0);
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let td = TdValue::default();
        let mut d = TrajectoriesDataset::new();
        d.push_step("A", 1.0).unwrap();
        d.push_step("B", 0.5).unwrap();
        d.end_episode();
        let mut m = td.init();
        td.update(&mut m, &d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = TdValueModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_params_are_rejected() {
        let mut d = TrajectoriesDataset::new();
        d.push_step("A", 1.0).unwrap();
        d.end_episode();

        for (alpha, gamma) in [
            (0.0, 1.0),
            (1.5, 1.0),
            (f64::NAN, 1.0),
            (0.1, 0.0),
            (0.1, 1.5),
            (0.1, f64::NAN),
        ] {
            let td = TdValue {
                alpha,
                gamma,
                ..Default::default()
            };
            let mut m = td.init();
            assert!(matches!(td.update(&mut m, &d), Err(Error::InvalidInput(_))));
        }
    }
}

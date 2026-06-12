//! Sliding-window UCB (`docs/algorithms.md` §8.1; Garivier & Moulines,
//! ALT 2011).
//!
//! UCB1 computed over only the last `window` reward events: a ring buffer
//! holds the recent `(arm, reward)` history and per-arm windowed statistics
//! are maintained incrementally (O(1) per push/evict), so arms whose payoff
//! drifts are re-estimated from recent data instead of their full history.
//!
//! Deliberately a sibling of [`Bandit`](crate::algos::Bandit) rather than a
//! [`BanditPolicy`](crate::algos::BanditPolicy) variant: the window makes the
//! state order-dependent — which events fall inside the window depends on
//! arrival order — so the exact-merge contract of
//! [`BanditModel::merge`](crate::algos::BanditModel::merge) cannot hold.
//! **This model therefore has no `merge`.**
//!
//! Assumes one logical event stream: `update` folds batches in arrival order
//! and `t` counts every event ever seen, evicted ones included.
//!
//! Gotcha: selection is deterministic (argmax of the index, ties by name) —
//! there is no RNG and hence no persisted draw counter; save → load → select
//! is trivially resumable.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::dataset::RewardsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// Sliding-window UCB parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlidingWindowUcb {
    /// Number of most-recent reward events the statistics cover. Zero is
    /// rejected at `update`/`select` time.
    pub window: usize,
    /// UCB exploration constant (2.0 is classic UCB1).
    pub exploration: f64,
}

impl SlidingWindowUcb {
    /// Default window size, in events (not rounds per arm).
    pub const DEFAULT_WINDOW: usize = 1000;
    /// Classic UCB1 exploration constant (Auer et al. 2002).
    pub const DEFAULT_EXPLORATION: f64 = 2.0;
}

impl Default for SlidingWindowUcb {
    fn default() -> Self {
        Self {
            window: Self::DEFAULT_WINDOW,
            exploration: Self::DEFAULT_EXPLORATION,
        }
    }
}

/// What `save_jsonl` writes as the header `params`: the algorithm params plus
/// the event ring (state). The ring is bounded by `window` by construction.
#[derive(Serialize, Deserialize)]
struct PersistedParams {
    window: usize,
    exploration: f64,
    t: u64,
    ring_a: Vec<u32>,
    ring_r: Vec<f32>,
}

/// One arm line: vocab only — the windowed statistics are derived state,
/// recomputed from the persisted ring on load.
#[derive(Serialize, Deserialize)]
struct ArmLine {
    id: String,
}

/// Sliding-window UCB state: the event ring plus incrementally maintained
/// windowed sufficient statistics.
#[derive(Debug, Clone)]
pub struct SwUcbModel {
    params: SlidingWindowUcb,
    names: Interner,
    /// The last `window` events as `(arm, reward)`, oldest first.
    ring: VecDeque<(u32, f32)>,
    /// All events ever seen (evicted ones included).
    t: u64,
    /// Windowed pull count per arm.
    n: Vec<u64>,
    /// Windowed reward sum per arm.
    sum: Vec<f64>,
}

impl SwUcbModel {
    /// Number of distinct arms the model has seen.
    pub fn n_arms(&self) -> usize {
        self.names.len()
    }

    /// Events currently inside the window.
    pub fn window_len(&self) -> usize {
        self.ring.len()
    }

    /// Total events ever observed (evicted ones included).
    pub fn total_n(&self) -> u64 {
        self.t
    }

    fn ensure_window(&self) -> Result<()> {
        if self.params.window == 0 {
            return Err(Error::InvalidInput(
                "sliding window must cover at least 1 event".into(),
            ));
        }
        Ok(())
    }

    /// Appends one event and evicts the oldest once the ring exceeds the
    /// window, keeping `n`/`sum` in sync in O(1).
    fn push_event(&mut self, arm: u32, reward: f32) {
        self.ring.push_back((arm, reward));
        self.n[arm as usize] += 1;
        self.sum[arm as usize] += f64::from(reward);
        self.t += 1;

        if self.ring.len() > self.params.window
            && let Some((old, r)) = self.ring.pop_front()
        {
            self.n[old as usize] -= 1;
            self.sum[old as usize] -= f64::from(r);
        }
    }

    /// The sliding-window UCB index: windowed mean plus
    /// `sqrt(exploration · ln(min(t, window)) / n_i)`. Arms with no events
    /// inside the window rank first at +∞ (the UCB1 convention). `ln(0)`
    /// cannot occur: any arm with windowed data implies `t ≥ 1`.
    fn index(&self, i: usize) -> f64 {
        if self.n[i] == 0 {
            return f64::INFINITY;
        }
        let horizon = self.t.min(self.params.window as u64) as f64;
        let mean = self.sum[i] / self.n[i] as f64;
        mean + (self.params.exploration * horizon.ln() / self.n[i] as f64).sqrt()
    }

    /// Picks the arm with the highest windowed index. Deterministic: no RNG,
    /// ties go to the lexicographically smaller name.
    pub fn select(&self) -> Result<&str> {
        let picks = self.select_k(1)?;
        picks
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidInput("bandit has no arms".into()))
    }

    /// Picks the `k` arms with the highest windowed indices, best first
    /// (ties by name). Errors if the model has no arms, `k` exceeds the
    /// number of arms, or the configured window is zero.
    pub fn select_k(&self, k: usize) -> Result<Vec<&str>> {
        self.ensure_window()?;
        if self.n_arms() == 0 {
            return Err(Error::InvalidInput("bandit has no arms".into()));
        }
        if k > self.n_arms() {
            return Err(Error::InvalidInput(format!(
                "requested {k} arms but only {} exist",
                self.n_arms()
            )));
        }

        let mut order: Vec<u32> = (0..self.n_arms() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            self.index(b as usize)
                .total_cmp(&self.index(a as usize))
                .then_with(|| self.names.resolve(a).cmp(self.names.resolve(b)))
        });
        order.truncate(k);
        Ok(order.into_iter().map(|i| self.names.resolve(i)).collect())
    }

    fn intern_arm(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.n.len() {
            self.n.push(0);
            self.sum.push(0.0);
        }
        idx
    }
}

impl RankModel for SwUcbModel {
    fn algorithm(&self) -> &'static str {
        "sliding-window-ucb"
    }

    /// Reports the windowed UCB index per arm — the same quantity selection
    /// maximizes, mirroring how the UCB1 bandit policy scores.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, n)| (n, self.index(i)))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            window: self.params.window,
            exploration: self.params.exploration,
            t: self.t,
            ring_a: self.ring.iter().map(|&(a, _)| a).collect(),
            ring_r: self.ring.iter().map(|&(_, r)| r).collect(),
        };
        let lines: Vec<ArmLine> = self
            .names
            .names()
            .map(|id| ArmLine { id: id.to_string() })
            .collect();
        state::save_model(w, "sliding-window-ucb", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<ArmLine>) =
            state::load_model(r, "sliding-window-ucb")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;

        if params.ring_a.len() != params.ring_r.len() {
            return Err(Error::State(format!(
                "ring columns disagree: {} arms vs {} rewards",
                params.ring_a.len(),
                params.ring_r.len()
            )));
        }
        if params.ring_a.len() > params.window {
            return Err(Error::State(format!(
                "ring holds {} events but the window is {}",
                params.ring_a.len(),
                params.window
            )));
        }
        if params.t < params.ring_a.len() as u64 {
            return Err(Error::State(format!(
                "event total {} is smaller than the ring length {}",
                params.t,
                params.ring_a.len()
            )));
        }

        let mut model = Self {
            params: SlidingWindowUcb {
                window: params.window,
                exploration: params.exploration,
            },
            names,
            ring: VecDeque::with_capacity(params.ring_a.len()),
            t: params.t,
            n: vec![0; lines.len()],
            sum: vec![0.0; lines.len()],
        };

        // Derived statistics are recomputed from the ring, not persisted.
        for (&a, &r) in params.ring_a.iter().zip(&params.ring_r) {
            if (a as usize) >= model.names.len() {
                return Err(Error::State(format!("ring arm id {a} out of vocab")));
            }
            model.ring.push_back((a, r));
            model.n[a as usize] += 1;
            model.sum[a as usize] += f64::from(r);
        }
        Ok(model)
    }
}

impl OnlineRanker for SlidingWindowUcb {
    type Data = RewardsDataset;
    type Model = SwUcbModel;

    fn init(&self) -> SwUcbModel {
        SwUcbModel {
            params: *self,
            names: Interner::new(),
            ring: VecDeque::new(),
            t: 0,
            n: Vec::new(),
            sum: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut SwUcbModel,
        data: &RewardsDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        model.ensure_window()?;
        for (arm, reward) in data.rows() {
            let name = data.interner().resolve(arm);
            let idx = model.intern_arm(name) as u32;
            model.push_event(idx, reward);
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

    /// window = 2; events (a,1.0), (b,0.0), (a,0.5). The third push evicts
    /// (a,1.0), leaving ring = [(b,0.0), (a,0.5)]:
    ///   n_a = 1, sum_a = 0.5;  n_b = 1, sum_b = 0.0;  t = 3.
    /// horizon = min(t, window) = 2, so both bonuses are sqrt(2·ln 2 / 1):
    ///   index_a = 0.5 + sqrt(2 ln 2),  index_b = 0.0 + sqrt(2 ln 2).
    #[test]
    fn windowed_index_matches_hand_computation() {
        let algo = SlidingWindowUcb {
            window: 2,
            exploration: 2.0,
        };
        let mut m = algo.init();
        algo.update(&mut m, &rewards(&[("a", 1.0), ("b", 0.0), ("a", 0.5)]))
            .unwrap();

        assert_eq!(m.window_len(), 2, "oldest event evicted");
        assert_eq!(m.total_n(), 3, "t counts evicted events too");

        let bonus = (2.0 * 2f64.ln()).sqrt();
        let s: std::collections::HashMap<_, _> =
            m.scores().map(|(n, v)| (n.to_string(), v)).collect();
        assert!((s["a"] - (0.5 + bonus)).abs() < 1e-12, "{}", s["a"]);
        assert!((s["b"] - bonus).abs() < 1e-12, "{}", s["b"]);
        assert_eq!(m.select().unwrap(), "a");
        assert_eq!(m.select_k(2).unwrap(), vec!["a", "b"]);
    }

    /// Phase 1: 60 events of a paying 1.0. Phase 2: 60 interleaved rounds of
    /// a paying 0.0 and b paying 0.3. With window 40 the model sees only the
    /// last 40 events (20 per arm): windowed means a = 0.0, b = 0.3 with
    /// equal bonuses, so the windowed policy switches to b — while the
    /// cumulative means (a: 60/120 = 0.5, b: 0.3) still rank a first.
    #[test]
    fn regime_change_recovers_within_window() {
        let algo = SlidingWindowUcb {
            window: 40,
            exploration: 2.0,
        };
        let mut all: Vec<(&str, f32)> = vec![("a", 1.0); 60];
        for _ in 0..60 {
            all.push(("a", 0.0));
            all.push(("b", 0.3));
        }
        let mut m = algo.init();
        algo.update(&mut m, &rewards(&all)).unwrap();

        assert_eq!(m.select().unwrap(), "b");

        // The cumulative-mean ordering over the full log differs.
        let mut sums = std::collections::HashMap::new();
        for &(arm, r) in &all {
            let e = sums.entry(arm).or_insert((0.0f64, 0u64));
            e.0 += f64::from(r);
            e.1 += 1;
        }
        let mean = |a: &str| sums[a].0 / sums[a].1 as f64;
        assert!(
            mean("a") > mean("b"),
            "cumulative means must still favor a: {} vs {}",
            mean("a"),
            mean("b")
        );
    }

    #[test]
    fn round_trip_is_byte_identical_and_indices_survive() {
        let algo = SlidingWindowUcb {
            window: 3,
            exploration: 2.0,
        };
        let mut m = algo.init();
        // Dyadic rewards keep the incremental and recomputed sums exact.
        algo.update(
            &mut m,
            &rewards(&[("a", 1.0), ("b", 0.5), ("a", 0.25), ("c", 0.75), ("b", 1.0)]),
        )
        .unwrap();
        assert_eq!(m.window_len(), 3);

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = SwUcbModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second, "save -> load -> save is byte-identical");

        let pre: Vec<(String, f64)> = m.scores().map(|(n, v)| (n.to_string(), v)).collect();
        let post: Vec<(String, f64)> = loaded.scores().map(|(n, v)| (n.to_string(), v)).collect();
        assert_eq!(pre, post, "indices recomputed from the ring match");
        assert_eq!(m.select().unwrap(), loaded.select().unwrap());
    }

    #[test]
    fn zero_window_and_empty_model_are_rejected() {
        let algo = SlidingWindowUcb {
            window: 0,
            exploration: 2.0,
        };
        let mut m = algo.init();
        assert!(matches!(
            algo.update(&mut m, &rewards(&[("a", 1.0)])),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(m.select(), Err(Error::InvalidInput(_))));

        let empty = SlidingWindowUcb::default().init();
        assert!(matches!(empty.select(), Err(Error::InvalidInput(_))));
        assert!(matches!(empty.select_k(1), Err(Error::InvalidInput(_))));
    }
}

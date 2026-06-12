//! Win-rate ranking with Wilson score intervals (`docs/algorithms.md` §7.1).
//!
//! Schedule-blind by design: an entity's score depends only on its own
//! win/loss record, never on opponent strength. The fast baseline every
//! schedule-aware method should beat.
//!
//! Incrementality: tallies are sufficient statistics, so [`WinRate`] is an
//! [`OnlineRanker`] — `update` merges counts and never replays history.
//!
//! Compatibility note: v1's `rate` ranks by the **upper** Wilson bound (the
//! optimistic end of the interval); that behavior is preserved under
//! [`Confidence::P90`]/[`Confidence::P95`].

use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::Result;
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// Which statistic to rank by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// Plain win rate (point estimate).
    P50,
    /// Wilson interval at 90% (z = 1.645).
    P90,
    /// Wilson interval at 95% (z = 1.96).
    #[default]
    P95,
}

/// The two-sided Wilson score interval for a binomial proportion at critical
/// value `z`, from weighted success/failure tallies. Returns
/// `(lower, upper)`; `(0, 0)` when there are no observations.
///
/// Reference values: Newcombe (1998), *Statistics in Medicine* 17:857-872,
/// Table I method 3 (tested in `tests/reference.rs`).
pub fn wilson_interval(successes: f64, failures: f64, z: f64) -> (f64, f64) {
    let n = successes + failures;
    if n == 0.0 {
        return (0.0, 0.0);
    }

    let z2 = z * z;
    let nz2 = n + z2;
    let center = (successes + z2 / 2.0) / nz2;
    let spread = (z / nz2) * (successes * failures / n + z2 / 4.0).sqrt();
    (center - spread, center + spread)
}

/// Win-rate ranker parameters. The struct is the algorithm; fields are params.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WinRate {
    /// Which statistic to rank by (point estimate or a Wilson upper bound).
    pub confidence: Confidence,
}

/// Per-entity win/loss tallies (weighted) — the model's entire state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TallyLine {
    id: String,
    wins: f64,
    losses: f64,
}

/// Fitted/accumulating win-rate state.
#[derive(Debug, Clone)]
pub struct WinRateModel {
    params: WinRate,
    names: Interner,
    wins: Vec<f64>,
    losses: Vec<f64>,
}

impl WinRateModel {
    /// v1-compatible Wilson statistic: the **upper** interval bound for
    /// P90/P95 (v1 ranked by the optimistic end), the point estimate for P50.
    fn statistic(&self, idx: usize) -> f64 {
        let (w, l) = (self.wins[idx], self.losses[idx]);
        let n = w + l;
        if n == 0.0 {
            return 0.0;
        }

        match self.params.confidence {
            Confidence::P50 => w / n,
            Confidence::P90 => wilson_interval(w, l, 1.645).1,
            Confidence::P95 => wilson_interval(w, l, 1.96).1,
        }
    }
}

impl RankModel for WinRateModel {
    fn algorithm(&self) -> &'static str {
        "rate"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names
            .names()
            .enumerate()
            .map(|(i, n)| (n, self.statistic(i)))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<TallyLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| TallyLine {
                id: id.to_string(),
                wins: self.wins[i],
                losses: self.losses[i],
            })
            .collect();
        state::save_model(w, "rate", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (WinRate, Vec<TallyLine>) = state::load_model(r, "rate")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            wins: lines.iter().map(|l| l.wins).collect(),
            losses: lines.iter().map(|l| l.losses).collect(),
        })
    }
}

impl OnlineRanker for WinRate {
    type Data = PairwiseDataset;
    type Model = WinRateModel;

    fn init(&self) -> WinRateModel {
        WinRateModel {
            params: *self,
            names: Interner::new(),
            wins: Vec::new(),
            losses: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut WinRateModel,
        data: &PairwiseDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        for (w, l, x) in data.rows() {
            // Resolve through names: id spaces differ across datasets.
            for (id, won) in [(w, true), (l, false)] {
                let name = data.interner().resolve(id);
                let idx = model.names.intern(name) as usize;
                if idx == model.wins.len() {
                    model.wins.push(0.0);
                    model.losses.push(0.0);
                }
                if won {
                    model.wins[idx] += f64::from(x);
                } else {
                    model.losses[idx] += f64::from(x);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> PairwiseDataset {
        // From v1 rate.rs test: team0 1-0, team1 1-2, team2 1-1.
        let mut d = PairwiseDataset::new();
        d.push("0", "1", 1.0);
        d.push("1", "2", 1.0);
        d.push("2", "1", 1.0);
        d.push("0", "1", 0.0); // zero weight: no effect
        d
    }

    fn fit(ci: Confidence) -> WinRateModel {
        let algo = WinRate { confidence: ci };
        let mut m = algo.init();
        algo.update(&mut m, &data()).unwrap();
        m
    }

    #[test]
    fn point_estimates_match_v1() {
        let m = fit(Confidence::P50);
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert_eq!(s["0"], 1.0);
        assert!((s["1"] - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(s["2"], 0.5);
    }

    #[test]
    fn wilson_p90_matches_v1_test_vector() {
        let m = fit(Confidence::P90);
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        // Expected values from the v1 rate.rs unit test.
        assert!((s["1"] - 0.74649).abs() < 1e-4, "got {}", s["1"]);
        assert!((s["2"] - 0.879148).abs() < 1e-4, "got {}", s["2"]);
    }

    #[test]
    fn update_merges_incrementally() {
        let algo = WinRate::default();
        let mut once = algo.init();
        algo.update(&mut once, &data()).unwrap();
        algo.update(&mut once, &data()).unwrap();

        let mut twice = algo.init();
        let mut both = data();
        for (w, l, x) in data().rows().collect::<Vec<_>>() {
            let wn = data().interner().name(w).unwrap().to_string();
            let ln = data().interner().name(l).unwrap().to_string();
            both.push(&wn, &ln, x);
        }
        algo.update(&mut twice, &both).unwrap();

        let a: Vec<_> = once
            .sorted_scores()
            .into_iter()
            .map(|(n, s)| (n.to_string(), s))
            .collect();
        let b: Vec<_> = twice
            .sorted_scores()
            .into_iter()
            .map(|(n, s)| (n.to_string(), s))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn round_trip() {
        let m = fit(Confidence::P95);
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = WinRateModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

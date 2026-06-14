//! Hanson's logarithmic market scoring rule (LMSR) as an online ranker
//! (`docs/algorithms.md` §14.3).
//!
//! A cost-function market maker: the state is the outstanding share vector `q`,
//! the cost function is `C(q) = b · ln Σᵢ exp(qᵢ/b)`, and the instantaneous
//! price of outcome `i` is the softmax `pᵢ = exp(qᵢ/b) / Σⱼ exp(qⱼ/b)` — the
//! crowd's stake-weighted consensus probability. A trade buying `Δ` shares of
//! outcome `k` sets `qₖ ← qₖ + Δ` and costs `C(q_after) − C(q_before)`. The
//! liquidity `b` trades price responsiveness against the market maker's bounded
//! worst-case subsidy `b · ln n`.
//!
//! Incremental: `q` is sufficient state, so [`Lmsr`] is an [`OnlineRanker`] —
//! `update` accumulates shares and never replays history.
//!
//! References: [Hanson 2003; 2007]; [Gneiting & Raftery 2007]. Worked values are
//! pinned in `tests/reference.rs`.

use serde::{Deserialize, Serialize};

use crate::dataset::MarketDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// LMSR parameters. The struct is the algorithm; fields are params.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lmsr {
    /// Liquidity `b > 0`: larger = deeper market (less price movement per trade)
    /// and a larger worst-case subsidy `b · ln n`.
    pub b: f64,
}

impl Default for Lmsr {
    fn default() -> Self {
        Self { b: 100.0 }
    }
}

/// One outcome's line in an LMSR state file.
#[derive(Debug, Serialize, Deserialize)]
struct QLine {
    id: String,
    q: f64,
}

/// Live market state: the outstanding share vector and its liquidity.
#[derive(Debug, Clone)]
pub struct LmsrModel {
    params: Lmsr,
    names: Interner,
    q: Vec<f64>,
}

impl LmsrModel {
    fn scaled(&self) -> Vec<f64> {
        self.q.iter().map(|&qi| qi / self.params.b).collect()
    }

    /// The instantaneous prices (softmax of `q/b`), aligned to outcome order.
    fn prices(&self) -> Vec<f64> {
        let scaled = self.scaled();
        let lse = mathx::logsumexp(&scaled);
        if !lse.is_finite() {
            return vec![0.0; self.q.len()];
        }
        scaled.iter().map(|&s| (s - lse).exp()).collect()
    }

    /// The price of `outcome`, or `None` if it is not in the market.
    pub fn price(&self, outcome: &str) -> Option<f64> {
        let idx = self.names.get(outcome)? as usize;
        self.prices().get(idx).copied()
    }

    /// The market maker's current cost `C(q) = b · ln Σ exp(qᵢ/b)`.
    pub fn cost(&self) -> f64 {
        self.params.b * mathx::logsumexp(&self.scaled())
    }

    /// The payment for a prospective trade of `shares` on `outcome`
    /// (`C(q + shares·eₖ) − C(q)`), or `None` if the outcome is unknown.
    pub fn trade_cost(&self, outcome: &str, shares: f64) -> Option<f64> {
        let idx = self.names.get(outcome)? as usize;
        let before = self.cost();
        let mut scaled = self.scaled();
        scaled[idx] += shares / self.params.b;
        let after = self.params.b * mathx::logsumexp(&scaled);
        Some(after - before)
    }
}

impl RankModel for LmsrModel {
    fn algorithm(&self) -> &'static str {
        "lmsr"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.prices())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<QLine> = self
            .names
            .names()
            .zip(&self.q)
            .map(|(id, &q)| QLine {
                id: id.to_string(),
                q,
            })
            .collect();
        state::save_model(w, "lmsr", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (Lmsr, Vec<QLine>) = state::load_model(r, "lmsr")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        let q = lines.iter().map(|l| l.q).collect();
        Ok(Self { params, names, q })
    }
}

impl OnlineRanker for Lmsr {
    type Data = MarketDataset;
    type Model = LmsrModel;

    fn init(&self) -> LmsrModel {
        LmsrModel {
            params: *self,
            names: Interner::new(),
            q: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut LmsrModel,
        data: &MarketDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        if !self.b.is_finite() || self.b <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "lmsr: liquidity b must be positive, got {}",
                self.b
            )));
        }

        let ensure = |model: &mut LmsrModel, name: &str| -> usize {
            let idx = model.names.intern(name) as usize;
            if idx == model.q.len() {
                model.q.push(0.0);
            }
            idx
        };

        // Seed the full declared universe (so prices span untraded outcomes).
        for name in data.interner().names() {
            ensure(model, name);
        }
        for (oid, shares) in data.rows() {
            let name = data.interner().resolve(oid);
            let idx = ensure(model, name);
            model.q[idx] += shares;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market(trades: &[(&str, f64)]) -> MarketDataset {
        let mut d = MarketDataset::new();
        for &(o, s) in trades {
            d.push_trade(o, s).unwrap();
        }
        d
    }

    fn fit(b: f64, d: &MarketDataset) -> LmsrModel {
        let algo = Lmsr { b };
        let mut m = algo.init();
        algo.update(&mut m, d).unwrap();
        m
    }

    #[test]
    fn uniform_when_no_net_position() {
        // Three declared outcomes, no net shares ⇒ equal prices and C(0)=b·ln n.
        let mut d = MarketDataset::new();
        for o in ["a", "b", "c"] {
            d.declare_outcome(o);
        }
        let m = fit(50.0, &d);
        for (_, p) in m.scores() {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
        assert!((m.cost() - 50.0 * 3.0_f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn exact_softmax() {
        // b=1, q=(ln1, ln2, ln3) ⇒ prices (1/6, 2/6, 3/6).
        let d = market(&[("a", 0.0), ("b", 2.0_f64.ln()), ("c", 3.0_f64.ln())]);
        let m = fit(1.0, &d);
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - 1.0 / 6.0).abs() < 1e-12);
        assert!((s["b"] - 2.0 / 6.0).abs() < 1e-12);
        assert!((s["c"] - 3.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn prices_sum_to_one() {
        let m = fit(10.0, &market(&[("a", 30.0), ("b", -5.0), ("c", 12.0)]));
        let s: f64 = m.scores().map(|(_, p)| p).sum();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trade_cost_matches_cost_difference() {
        let m = fit(20.0, &market(&[("a", 5.0), ("b", 1.0)]));
        let tc = m.trade_cost("a", 7.0).unwrap();
        // Independently recompute via a fresh fit of the post-trade book.
        let after = fit(20.0, &market(&[("a", 5.0), ("b", 1.0), ("a", 7.0)]));
        assert!((tc - (after.cost() - m.cost())).abs() < 1e-9);
    }

    #[test]
    fn split_updates_match_one_run() {
        let algo = Lmsr { b: 15.0 };
        let mut split = algo.init();
        algo.update(&mut split, &market(&[("a", 4.0), ("b", 1.0)]))
            .unwrap();
        algo.update(&mut split, &market(&[("a", 2.0), ("c", 3.0)]))
            .unwrap();
        let one = fit(15.0, &market(&[("a", 6.0), ("b", 1.0), ("c", 3.0)]));
        let a: Vec<_> = split
            .sorted_scores()
            .iter()
            .map(|(n, s)| (n.to_string(), *s))
            .collect();
        let b: Vec<_> = one
            .sorted_scores()
            .iter()
            .map(|(n, s)| (n.to_string(), *s))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn round_trip() {
        let m = fit(10.0, &market(&[("a", 30.0), ("b", -5.0)]));
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = LmsrModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

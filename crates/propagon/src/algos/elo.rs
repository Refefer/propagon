//! Elo rating system (`docs/algorithms.md` §2.1).
//!
//! Online SGD on the Bradley-Terry log-loss: after each game both ratings move
//! by `k × (outcome − expected)`. Order-dependent by definition — rows are
//! processed in dataset insertion order. For *static* skills prefer the
//! offline Bradley-Terry fits (§12.3 of the survey explains why).

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::Result;
use crate::interner::Interner;
use crate::traits::{FitOptions, OnlineRanker};

/// Elo parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Elo {
    /// Update step size.
    pub k: f64,
    /// Rating assigned to unseen entities.
    pub initial_rating: f64,
    /// Logistic scale: a `scale`-point gap means 10:1 expected odds.
    pub scale: f64,
}

impl Default for Elo {
    fn default() -> Self {
        Self {
            k: 32.0,
            initial_rating: 1500.0,
            scale: 400.0,
        }
    }
}

/// Elo ratings keyed by entity name.
#[derive(Debug, Clone)]
pub struct EloModel {
    params: Elo,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(EloModel, "elo");

impl EloModel {
    fn idx(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.scores.len() {
            self.scores.push(self.params.initial_rating);
        }
        idx
    }
}

impl OnlineRanker for Elo {
    type Data = PairwiseDataset;
    type Model = EloModel;

    fn init(&self) -> EloModel {
        EloModel {
            params: *self,
            names: Interner::new(),
            scores: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut EloModel,
        data: &PairwiseDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        let ln10_scale = std::f64::consts::LN_10 / self.scale;
        for (w, l, x) in data.rows() {
            let wname = data.interner().name(w).expect("dataset id resolves");
            let lname = data.interner().name(l).expect("dataset id resolves");
            let wi = model.idx(wname);
            let li = model.idx(lname);
            let expected = 1.0 / (1.0 + ((model.scores[li] - model.scores[wi]) * ln10_scale).exp());
            let delta = self.k * f64::from(x) * (1.0 - expected);
            model.scores[wi] += delta;
            model.scores[li] -= delta;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    #[test]
    fn equal_ratings_move_by_half_k() {
        let elo = Elo::default();
        let mut m = elo.init();
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        elo.update(&mut m, &d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        // expected = 0.5 at equal ratings -> winner gains k/2 = 16
        assert!((s["a"] - 1516.0).abs() < 1e-9);
        assert!((s["b"] - 1484.0).abs() < 1e-9);
    }

    #[test]
    fn order_matters_and_state_persists() {
        let elo = Elo::default();
        let mut m = elo.init();
        let mut d1 = PairwiseDataset::new();
        d1.push("a", "b", 1.0);
        let mut d2 = PairwiseDataset::new();
        d2.push("b", "a", 1.0);
        elo.update(&mut m, &d1).unwrap();
        elo.update(&mut m, &d2).unwrap();

        // After a win each, the second result was less expected for b only
        // because of the first; ratings are not back at the start.
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!(s["b"] > s["a"]);

        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = EloModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

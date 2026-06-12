//! Elo rating system (`docs/algorithms.md` §2.1).
//!
//! Online SGD on the Bradley-Terry log-loss: after each game both ratings move
//! by `k × (outcome − expected)`. Order-dependent by definition — games are
//! processed in dataset insertion order. Ties take Elo's classic half-score
//! convention (`S = ½`); margins are ignored (the margin-of-victory variant
//! is `elo-mov`). For *static* skills prefer the offline Bradley-Terry fits
//! (§12.3 of the survey explains why).

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{GameOutcome, GamesDataset};
use crate::error::{Error, Result};
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

impl Elo {
    /// Expected score of a player rated `ra` against one rated `rb`:
    /// `1 / (1 + 10^((rb − ra) / scale))`. With the standard 400-point
    /// scale: equal ratings → 0.5, +200 → ≈0.76, +400 → 10/11.
    pub fn expected_score(&self, ra: f64, rb: f64) -> f64 {
        1.0 / (1.0 + ((rb - ra) * std::f64::consts::LN_10 / self.scale).exp())
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
    type Data = GamesDataset;
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
        data: &GamesDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        for (g, view) in data.games().enumerate() {
            let (&[a], &[b]) = (view.side1, view.side2) else {
                return Err(Error::InvalidInput(format!(
                    "game {g} has a multi-player side; elo rates 1v1 games — \
                     use weng-lin on a matchups dataset for teams"
                )));
            };
            // Score of side 1: win 1, tie ½, loss 0; margins ignored.
            let s1 = match view.outcome {
                GameOutcome::Side1Win(_) => 1.0,
                GameOutcome::Tie => 0.5,
                GameOutcome::Side2Win(_) => 0.0,
            };
            let an = data.interner().resolve(a);
            let bn = data.interner().resolve(b);
            let ai = model.idx(an);
            let bi = model.idx(bn);
            let expected = self.expected_score(model.scores[ai], model.scores[bi]);
            let delta = self.k * f64::from(view.weight) * (s1 - expected);
            model.scores[ai] += delta;
            model.scores[bi] -= delta;
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
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        elo.update(&mut m, &d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        // expected = 0.5 at equal ratings -> winner gains k/2 = 16
        assert!((s["a"] - 1516.0).abs() < 1e-9);
        assert!((s["b"] - 1484.0).abs() < 1e-9);
    }

    #[test]
    fn tie_at_equal_ratings_moves_nothing() {
        let elo = Elo::default();
        let mut m = elo.init();
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 1.0).unwrap();
        elo.update(&mut m, &d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        // S = ½ equals the expectation at equal ratings: no movement.
        assert!((s["a"] - 1500.0).abs() < 1e-9);
        assert!((s["b"] - 1500.0).abs() < 1e-9);

        // A tie against a stronger opponent gains rating.
        let mut up = GamesDataset::new();
        up.push_pair("a", "b", 1.0).unwrap();
        up.push_game(&["b"], &["a"], GameOutcome::Tie, 1.0).unwrap();
        let mut m = elo.init();
        elo.update(&mut m, &up).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!(s["a"] > 1500.0 + 16.0 - 1e-9 - 16.0); // a still ahead overall
        assert!(s["b"] < 1500.0); // b lost, then only drew as the underdog...
        // the precise check: the tie moved b up from its post-loss rating.
        let after_loss = 1484.0;
        assert!(s["b"] > after_loss);
    }

    #[test]
    fn multiplayer_sides_are_rejected() {
        let elo = Elo::default();
        let mut m = elo.init();
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        assert!(matches!(
            elo.update(&mut m, &d),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn order_matters_and_state_persists() {
        let elo = Elo::default();
        let mut m = elo.init();
        let mut d1 = GamesDataset::new();
        d1.push_pair("a", "b", 1.0).unwrap();
        let mut d2 = GamesDataset::new();
        d2.push_pair("b", "a", 1.0).unwrap();
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

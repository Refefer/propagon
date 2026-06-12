//! Margin-of-victory Elo (`docs/algorithms.md` §2.1; Hvattum & Arntzen 2010).
//!
//! Plain Elo with the K-factor scaled by how decisive the win was: each game
//! moves both ratings by `k · λ · weight · (s₁ − expected)` where the
//! multiplier is `λ = (ln(1 + margin) / ln 2)^mov_exponent` for wins and
//! `λ = 1` for ties (a tie has no margin; `s₁ = ½` carries the signal).
//! Dividing Hvattum & Arntzen's `ln(1 + margin)` by `ln 2` normalizes the
//! multiplier so a margin-1 win has `λ = 1`: on win-only (margin 1) data
//! mov-elo reproduces plain Elo with the same `k`, making it a strict
//! generalization rather than an `ln 2 ≈ 0.69`-rescaled cousin.
//!
//! Like `elo`, this is online SGD on the Bradley-Terry log-loss:
//! order-dependent by definition, games processed in dataset insertion
//! order, 1v1 only.
//!
//! Gotcha: state files are tagged `elo-mov`, so elo and elo-mov state can
//! never cross-load even though the model shapes are identical.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::{GameOutcome, GamesDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, OnlineRanker};

/// Margin-of-victory Elo parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovElo {
    /// Update step size at margin 1 (the plain-Elo K).
    pub k: f64,
    /// Rating assigned to unseen entities.
    pub initial_rating: f64,
    /// Logistic scale: a `scale`-point gap means 10:1 expected odds.
    pub scale: f64,
    /// Exponent on the normalized margin multiplier
    /// `(ln(1 + margin) / ln 2)^mov_exponent`; 1.0 is Hvattum & Arntzen's
    /// logarithmic form, 0.0 disables margin scaling entirely.
    pub mov_exponent: f64,
}

impl Default for MovElo {
    fn default() -> Self {
        Self {
            k: 32.0,
            initial_rating: 1500.0,
            scale: 400.0,
            mov_exponent: 1.0,
        }
    }
}

impl MovElo {
    /// Expected score of a player rated `ra` against one rated `rb`:
    /// `1 / (1 + 10^((rb − ra) / scale))`. Duplicated from `Elo` on purpose —
    /// the two are separate rating systems whose state files never mix.
    pub fn expected_score(&self, ra: f64, rb: f64) -> f64 {
        1.0 / (1.0 + ((rb - ra) * std::f64::consts::LN_10 / self.scale).exp())
    }

    /// The margin multiplier `λ = (ln(1 + margin) / ln 2)^mov_exponent`,
    /// normalized so `margin = 1` gives `λ = 1` (plain Elo).
    fn multiplier(&self, margin: f32) -> f64 {
        (f64::from(margin).ln_1p() / std::f64::consts::LN_2).powf(self.mov_exponent)
    }
}

/// Margin-of-victory Elo ratings keyed by entity name.
#[derive(Debug, Clone)]
pub struct MovEloModel {
    params: MovElo,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(MovEloModel, "elo-mov");

impl MovEloModel {
    fn idx(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.scores.len() {
            self.scores.push(self.params.initial_rating);
        }
        idx
    }
}

impl OnlineRanker for MovElo {
    type Data = GamesDataset;
    type Model = MovEloModel;

    fn init(&self) -> MovEloModel {
        MovEloModel {
            params: *self,
            names: Interner::new(),
            scores: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut MovEloModel,
        data: &GamesDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        for (g, view) in data.games().enumerate() {
            let (&[a], &[b]) = (view.side1, view.side2) else {
                return Err(Error::InvalidInput(format!(
                    "game {g} has a multi-player side; elo-mov rates 1v1 games — \
                     use weng-lin on a matchups dataset for teams"
                )));
            };
            // Score of side 1 and margin multiplier: wins scale by λ(margin),
            // ties have no margin and keep λ = 1 (s₁ = ½ does the work).
            let (s1, lambda) = match view.outcome {
                GameOutcome::Side1Win(m) => (1.0, self.multiplier(m)),
                GameOutcome::Side2Win(m) => (0.0, self.multiplier(m)),
                GameOutcome::Tie => (0.5, 1.0),
            };
            let an = data.interner().resolve(a);
            let bn = data.interner().resolve(b);
            let ai = model.idx(an);
            let bi = model.idx(bn);
            let expected = self.expected_score(model.scores[ai], model.scores[bi]);
            let delta = self.k * lambda * f64::from(view.weight) * (s1 - expected);
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
    use crate::algos::Elo;

    fn ratings(m: &MovEloModel) -> std::collections::HashMap<String, f64> {
        m.scores().map(|(n, s)| (n.to_string(), s)).collect()
    }

    #[test]
    fn margin_one_matches_plain_elo() {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        d.push_game(&["b"], &["c"], GameOutcome::Side2Win(1.0), 2.0)
            .unwrap();
        d.push_game(&["a"], &["c"], GameOutcome::Tie, 1.0).unwrap();

        let mov = MovElo::default();
        let mut mm = mov.init();
        mov.update(&mut mm, &d).unwrap();

        let elo = Elo::default();
        let mut em = elo.init();
        elo.update(&mut em, &d).unwrap();

        let ms = ratings(&mm);
        let es: std::collections::HashMap<_, _> =
            em.scores().map(|(n, s)| (n.to_string(), s)).collect();
        for name in ["a", "b", "c"] {
            assert!(
                (ms[name] - es[name]).abs() < 1e-12,
                "{name}: mov {} vs elo {}",
                ms[name],
                es[name]
            );
        }
    }

    #[test]
    fn bigger_margins_move_ratings_more() {
        let mov = MovElo::default();
        let mut margins = Vec::new();
        for m in [1.0, 3.0, 10.0] {
            let mut d = GamesDataset::new();
            d.push_game(&["a"], &["b"], GameOutcome::Side1Win(m), 1.0)
                .unwrap();
            let mut model = mov.init();
            mov.update(&mut model, &d).unwrap();
            margins.push(ratings(&model)["a"]);
        }
        assert!(
            margins[0] < margins[1] && margins[1] < margins[2],
            "winner ratings by margin: {margins:?}"
        );
    }

    #[test]
    fn tie_at_equal_ratings_moves_nothing_and_underdog_gains() {
        let mov = MovElo::default();
        let mut m = mov.init();
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 1.0).unwrap();
        mov.update(&mut m, &d).unwrap();
        let s = ratings(&m);
        assert!((s["a"] - 1500.0).abs() < 1e-9);
        assert!((s["b"] - 1500.0).abs() < 1e-9);

        // b loses big, then ties as the underdog: the tie gains rating.
        let mut up = GamesDataset::new();
        up.push_game(&["a"], &["b"], GameOutcome::Side1Win(8.0), 1.0)
            .unwrap();
        let mut m = mov.init();
        mov.update(&mut m, &up).unwrap();
        let after_loss = ratings(&m)["b"];

        let mut tie = GamesDataset::new();
        tie.push_game(&["b"], &["a"], GameOutcome::Tie, 1.0)
            .unwrap();
        mov.update(&mut m, &tie).unwrap();
        assert!(
            ratings(&m)["b"] > after_loss,
            "underdog tie should gain: {} vs {after_loss}",
            ratings(&m)["b"]
        );
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mov = MovElo::default();
        let mut m = mov.init();
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(3.0), 1.0)
            .unwrap();
        d.push_game(&["b"], &["a"], GameOutcome::Side1Win(2.0), 1.0)
            .unwrap();
        mov.update(&mut m, &d).unwrap();

        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = MovEloModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }

    #[test]
    fn multiplayer_sides_are_rejected() {
        let mov = MovElo::default();
        let mut m = mov.init();
        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        assert!(matches!(
            mov.update(&mut m, &d),
            Err(Error::InvalidInput(_))
        ));
    }
}

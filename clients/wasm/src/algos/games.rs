//! Algorithms over `games-dataset`: Elo (online). Phase 1 surface.

use std::cell::RefCell;

use propagon::OnlineRanker;
use propagon::RankModel;
use propagon::algos::{Elo, EloModel as CoreEloModel};

use crate::Component;
use crate::algos::{bulk, save, score_of, sorted, top_k};
use crate::bindings::exports::propagon::core::datasets::GamesDatasetBorrow;
use crate::bindings::exports::propagon::core::games::{EloModel, EloParams, Guest, GuestEloModel};
use crate::bindings::exports::propagon::core::types::{Error, ScoresBulk};
use crate::datasets::GamesData;
use crate::errors::MapWit;

/// An Elo model plus the configured algorithm it was fit with (needed to fold
/// further batches in via `update`). A loaded model defaults its config.
pub struct EloMod {
    algo: Elo,
    model: RefCell<CoreEloModel>,
}

/// Merge optional WIT fields onto the core `Default`.
fn elo_params(p: EloParams) -> Elo {
    let mut e = Elo::default();
    if let Some(v) = p.k {
        e.k = v;
    }
    if let Some(v) = p.initial_rating {
        e.initial_rating = v;
    }
    if let Some(v) = p.scale {
        e.scale = v;
    }
    e
}

impl Guest for Component {
    type EloModel = EloMod;

    fn init_elo(params: EloParams) -> EloModel {
        let algo = elo_params(params);
        let model = algo.init();
        EloModel::new(EloMod {
            algo,
            model: RefCell::new(model),
        })
    }

    fn fit_elo(params: EloParams, data: GamesDatasetBorrow<'_>) -> Result<EloModel, Error> {
        let algo = elo_params(params);
        let ds = data.get::<GamesData>();
        let mut model = algo.init();
        algo.update(&mut model, &ds.0.borrow()).map_wit()?;
        Ok(EloModel::new(EloMod {
            algo,
            model: RefCell::new(model),
        }))
    }

    fn load_elo(state: String) -> Result<EloModel, Error> {
        let model = CoreEloModel::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(EloModel::new(EloMod {
            algo: Elo::default(),
            model: RefCell::new(model),
        }))
    }
}

impl GuestEloModel for EloMod {
    fn algorithm(&self) -> String {
        self.model.borrow().algorithm().to_string()
    }
    fn sorted_scores(&self) -> Vec<(String, f64)> {
        sorted(&*self.model.borrow())
    }
    fn score(&self, name: String) -> Option<f64> {
        score_of(&*self.model.borrow(), &name)
    }
    fn top(&self, k: u32) -> Vec<(String, f64)> {
        top_k(&*self.model.borrow(), k)
    }
    fn scores_bulk(&self) -> ScoresBulk {
        bulk(&*self.model.borrow())
    }
    fn save_state(&self) -> Result<String, Error> {
        save(&*self.model.borrow())
    }
    fn update(&self, data: GamesDatasetBorrow<'_>) -> Result<(), Error> {
        let ds = data.get::<GamesData>();
        let mut model = self.model.borrow_mut();
        self.algo.update(&mut model, &ds.0.borrow()).map_wit()
    }
}

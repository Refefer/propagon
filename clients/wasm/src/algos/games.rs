//! Algorithms over `games-dataset`: Elo and Glicko-2 (online).

use std::cell::RefCell;

use propagon::OnlineRanker;
use propagon::RankModel;
use propagon::algos::{Elo, EloModel as CoreElo, Glicko2, Glicko2Model as CoreGlicko2};

use crate::Component;
use crate::datasets::GamesData;
use crate::errors::MapWit;
use crate::wit::datasets::GamesDatasetBorrow;
use crate::wit::games::{
    EloModel, EloParams, Glicko2Model, Glicko2Params, Guest, GuestEloModel, GuestGlicko2Model,
    PlayerState,
};
use crate::wit::types::Error;

online_model!(
    EloMod,
    GuestEloModel,
    Elo,
    CoreElo,
    GamesData,
    GamesDatasetBorrow<'_>
);

online_model!(
    Glicko2Mod, GuestGlicko2Model, Glicko2, CoreGlicko2, GamesData, GamesDatasetBorrow<'_>,
    extras {
        fn players(&self) -> Vec<(String, PlayerState)> {
            self.model
                .borrow()
                .players()
                .map(|(n, p)| (n.to_string(), PlayerState { r: p.r, rd: p.rd, sigma: p.sigma }))
                .collect()
        }
    }
);

impl Guest for Component {
    type EloModel = EloMod;
    type Glicko2Model = Glicko2Mod;

    fn init_elo(params: EloParams) -> EloModel {
        let algo = merge_params!(
            params,
            Elo,
            scalar {
                k,
                initial_rating,
                scale
            }
        );
        let model = algo.init();
        EloModel::new(EloMod {
            algo,
            model: RefCell::new(model),
        })
    }
    fn fit_elo(params: EloParams, data: GamesDatasetBorrow<'_>) -> Result<EloModel, Error> {
        let algo = merge_params!(
            params,
            Elo,
            scalar {
                k,
                initial_rating,
                scale
            }
        );
        let ds = data.get::<GamesData>();
        let mut model = algo.init();
        algo.update(&mut model, &ds.0.borrow()).map_wit()?;
        Ok(EloModel::new(EloMod {
            algo,
            model: RefCell::new(model),
        }))
    }
    fn load_elo(state: String) -> Result<EloModel, Error> {
        let model = CoreElo::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(EloModel::new(EloMod {
            algo: Elo::default(),
            model: RefCell::new(model),
        }))
    }

    fn init_glicko2(params: Glicko2Params) -> Glicko2Model {
        let algo = merge_params!(
            params,
            Glicko2,
            scalar {
                tau,
                rating,
                rd,
                sigma
            }
        );
        let model = algo.init();
        Glicko2Model::new(Glicko2Mod {
            algo,
            model: RefCell::new(model),
        })
    }
    fn fit_glicko2(
        params: Glicko2Params,
        data: GamesDatasetBorrow<'_>,
    ) -> Result<Glicko2Model, Error> {
        let algo = merge_params!(
            params,
            Glicko2,
            scalar {
                tau,
                rating,
                rd,
                sigma
            }
        );
        let ds = data.get::<GamesData>();
        let mut model = algo.init();
        algo.update(&mut model, &ds.0.borrow()).map_wit()?;
        Ok(Glicko2Model::new(Glicko2Mod {
            algo,
            model: RefCell::new(model),
        }))
    }
    fn load_glicko2(state: String) -> Result<Glicko2Model, Error> {
        let model = CoreGlicko2::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(Glicko2Model::new(Glicko2Mod {
            algo: Glicko2::default(),
            model: RefCell::new(model),
        }))
    }
}

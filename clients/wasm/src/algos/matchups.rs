//! Algorithms over `matchups-dataset`: Weng-Lin / OpenSkill (online).

use std::cell::RefCell;

use propagon::OnlineRanker;
use propagon::RankModel;
use propagon::algos::{WengLin, WengLinModel as CoreWengLin};

use crate::Component;
use crate::datasets::MatchupsData;
use crate::errors::MapWit;
use crate::wit::datasets::MatchupsDatasetBorrow;
use crate::wit::matchups::{Guest, GuestWengLinModel, Rating, WengLinModel, WengLinParams};
use crate::wit::types::Error;

online_model!(
    WengLinMod, GuestWengLinModel, WengLin, CoreWengLin, MatchupsData, MatchupsDatasetBorrow<'_>,
    extras {
        fn ratings(&self) -> Vec<(String, Rating)> {
            self.model
                .borrow()
                .ratings()
                .map(|(n, r)| (n.to_string(), Rating { mu: r.mu, sigma: r.sigma }))
                .collect()
        }
    }
);

fn weng_lin(p: WengLinParams) -> WengLin {
    merge_params!(
        p,
        WengLin,
        scalar {
            mu,
            sigma,
            beta,
            kappa,
            epsilon,
            tau
        }
    )
}

impl Guest for Component {
    type WengLinModel = WengLinMod;

    fn init_weng_lin(params: WengLinParams) -> WengLinModel {
        let algo = weng_lin(params);
        let model = algo.init();
        WengLinModel::new(WengLinMod {
            algo,
            model: RefCell::new(model),
        })
    }
    fn fit_weng_lin(
        params: WengLinParams,
        data: MatchupsDatasetBorrow<'_>,
    ) -> Result<WengLinModel, Error> {
        let algo = weng_lin(params);
        let ds = data.get::<MatchupsData>();
        let mut model = algo.init();
        algo.update(&mut model, &ds.0.borrow()).map_wit()?;
        Ok(WengLinModel::new(WengLinMod {
            algo,
            model: RefCell::new(model),
        }))
    }
    fn load_weng_lin(state: String) -> Result<WengLinModel, Error> {
        let model = CoreWengLin::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(WengLinModel::new(WengLinMod {
            algo: WengLin::default(),
            model: RefCell::new(model),
        }))
    }
}

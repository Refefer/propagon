//! Algorithms over `rewards-dataset`: the multi-armed Bandit (online).

use std::cell::RefCell;

use propagon::OnlineRanker;
use propagon::RankModel;
use propagon::algos::{Bandit, BanditModel as CoreBandit};

use crate::Component;
use crate::datasets::RewardsData;
use crate::enums::bandit_policy;
use crate::errors::MapWit;
use crate::wit::datasets::RewardsDatasetBorrow;
use crate::wit::rewards::{BanditModel, BanditParams, Guest, GuestBanditModel};
use crate::wit::types::Error;

online_model!(
    BanditMod,
    GuestBanditModel,
    Bandit,
    CoreBandit,
    RewardsData,
    RewardsDatasetBorrow<'_>
);

fn bandit(p: BanditParams) -> Bandit {
    let mut b = Bandit::default();
    if let Some(policy) = p.policy {
        b.policy = bandit_policy(policy);
    }
    if let Some(seed) = p.seed {
        b.seed = seed;
    }
    b
}

impl Guest for Component {
    type BanditModel = BanditMod;

    fn init_bandit(params: BanditParams) -> BanditModel {
        let algo = bandit(params);
        let model = algo.init();
        BanditModel::new(BanditMod {
            algo,
            model: RefCell::new(model),
        })
    }
    fn fit_bandit(
        params: BanditParams,
        data: RewardsDatasetBorrow<'_>,
    ) -> Result<BanditModel, Error> {
        let algo = bandit(params);
        let ds = data.get::<RewardsData>();
        let mut model = algo.init();
        algo.update(&mut model, &ds.0.borrow()).map_wit()?;
        Ok(BanditModel::new(BanditMod {
            algo,
            model: RefCell::new(model),
        }))
    }
    fn load_bandit(state: String) -> Result<BanditModel, Error> {
        let model = CoreBandit::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(BanditModel::new(BanditMod {
            algo: Bandit::default(),
            model: RefCell::new(model),
        }))
    }
}

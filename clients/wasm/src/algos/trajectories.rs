//! Algorithms over `trajectories-dataset`: Monte-Carlo state value (batch).

use propagon::RankModel;
use propagon::Ranker;
use propagon::algos::{McValue, McValueModel as CoreMcValue};

use crate::Component;
use crate::datasets::TrajectoriesData;
use crate::errors::MapWit;
use crate::wit::datasets::TrajectoriesDatasetBorrow;
use crate::wit::trajectories::{
    Guest, GuestMcValueModel, McValueModel, McValueModelBorrow, McValueParams,
};
use crate::wit::types::Error;

batch_model!(McValueMod, GuestMcValueModel, CoreMcValue);

fn mc_value(p: McValueParams) -> McValue {
    merge_params!(p, McValue, scalar { gamma }, usize { min_observations })
}

impl Guest for Component {
    type McValueModel = McValueMod;

    fn fit_mc_value(
        params: McValueParams,
        data: TrajectoriesDatasetBorrow<'_>,
    ) -> Result<McValueModel, Error> {
        let algo = mc_value(params);
        let ds = data.get::<TrajectoriesData>();
        let model = algo.fit(&ds.0.borrow()).map_wit()?;
        Ok(McValueModel::new(McValueMod(model)))
    }
    fn fit_warm_mc_value(
        params: McValueParams,
        data: TrajectoriesDatasetBorrow<'_>,
        init: McValueModelBorrow<'_>,
    ) -> Result<McValueModel, Error> {
        let algo = mc_value(params);
        let ds = data.get::<TrajectoriesData>();
        let init = init.get::<McValueMod>();
        let model = algo.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(McValueModel::new(McValueMod(model)))
    }
    fn load_mc_value(state: String) -> Result<McValueModel, Error> {
        let model = CoreMcValue::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(McValueModel::new(McValueMod(model)))
    }
}

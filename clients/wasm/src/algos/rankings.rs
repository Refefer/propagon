//! Algorithms over `rankings-dataset`: Plackett-Luce (batch).

use propagon::RankModel;
use propagon::Ranker;
use propagon::algos::{PlackettLuce, PlackettLuceModel as CorePlackettLuce};

use crate::Component;
use crate::datasets::RankingsData;
use crate::errors::MapWit;
use crate::wit::datasets::RankingsDatasetBorrow;
use crate::wit::rankings::{
    Guest, GuestPlackettLuceModel, PlackettLuceModel, PlackettLuceModelBorrow, PlackettLuceParams,
};
use crate::wit::types::Error;

batch_model!(PlackettLuceMod, GuestPlackettLuceModel, CorePlackettLuce);

fn plackett_luce(p: PlackettLuceParams) -> PlackettLuce {
    merge_params!(p, PlackettLuce, scalar { tolerance }, usize { iterations })
}

impl Guest for Component {
    type PlackettLuceModel = PlackettLuceMod;

    fn fit_plackett_luce(
        params: PlackettLuceParams,
        data: RankingsDatasetBorrow<'_>,
    ) -> Result<PlackettLuceModel, Error> {
        let algo = plackett_luce(params);
        let ds = data.get::<RankingsData>();
        let model = algo.fit(&ds.0.borrow()).map_wit()?;
        Ok(PlackettLuceModel::new(PlackettLuceMod(model)))
    }
    fn fit_warm_plackett_luce(
        params: PlackettLuceParams,
        data: RankingsDatasetBorrow<'_>,
        init: PlackettLuceModelBorrow<'_>,
    ) -> Result<PlackettLuceModel, Error> {
        let algo = plackett_luce(params);
        let ds = data.get::<RankingsData>();
        let init = init.get::<PlackettLuceMod>();
        let model = algo.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(PlackettLuceModel::new(PlackettLuceMod(model)))
    }
    fn load_plackett_luce(state: String) -> Result<PlackettLuceModel, Error> {
        let model = CorePlackettLuce::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(PlackettLuceModel::new(PlackettLuceMod(model)))
    }
}

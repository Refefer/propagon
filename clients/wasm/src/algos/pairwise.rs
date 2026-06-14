//! Algorithms over `pairwise-dataset`: Bradley-Terry (MM) and Borda (batch).

use propagon::RankModel;
use propagon::Ranker;
use propagon::algos::{Borda, BordaModel as CoreBorda, BradleyTerryMM, BtmMmModel as CoreBtmMm};

use crate::Component;
use crate::datasets::PairwiseData;
use crate::errors::MapWit;
use crate::wit::datasets::PairwiseDatasetBorrow;
use crate::wit::pairwise::{
    BordaModel, BradleyTerryMmModel, BradleyTerryMmModelBorrow, BradleyTerryMmParams, Guest,
    GuestBordaModel, GuestBradleyTerryMmModel,
};
use crate::wit::types::Error;

batch_model!(BtmMmMod, GuestBradleyTerryMmModel, CoreBtmMm);
batch_model!(BordaMod, GuestBordaModel, CoreBorda);

fn btm_mm(p: BradleyTerryMmParams) -> BradleyTerryMM {
    merge_params!(
        p,
        BradleyTerryMM,
        scalar {
            tolerance,
            remove_total_losers,
            create_fake_games,
            random_subgraph_weight,
            seed
        },
        usize {
            iterations,
            min_graph_size,
            random_subgraph_links
        }
    )
}

impl Guest for Component {
    type BradleyTerryMmModel = BtmMmMod;
    type BordaModel = BordaMod;

    fn fit_bradley_terry_mm(
        params: BradleyTerryMmParams,
        data: PairwiseDatasetBorrow<'_>,
    ) -> Result<BradleyTerryMmModel, Error> {
        let algo = btm_mm(params);
        let ds = data.get::<PairwiseData>();
        let model = algo.fit(&ds.0.borrow()).map_wit()?;
        Ok(BradleyTerryMmModel::new(BtmMmMod(model)))
    }
    fn fit_warm_bradley_terry_mm(
        params: BradleyTerryMmParams,
        data: PairwiseDatasetBorrow<'_>,
        init: BradleyTerryMmModelBorrow<'_>,
    ) -> Result<BradleyTerryMmModel, Error> {
        let algo = btm_mm(params);
        let ds = data.get::<PairwiseData>();
        let init = init.get::<BtmMmMod>();
        let model = algo.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(BradleyTerryMmModel::new(BtmMmMod(model)))
    }
    fn load_bradley_terry_mm(state: String) -> Result<BradleyTerryMmModel, Error> {
        let model = CoreBtmMm::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(BradleyTerryMmModel::new(BtmMmMod(model)))
    }

    fn fit_borda(data: PairwiseDatasetBorrow<'_>) -> Result<BordaModel, Error> {
        let ds = data.get::<PairwiseData>();
        let model = Borda::default().fit(&ds.0.borrow()).map_wit()?;
        Ok(BordaModel::new(BordaMod(model)))
    }
    fn load_borda(state: String) -> Result<BordaModel, Error> {
        let model = CoreBorda::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(BordaModel::new(BordaMod(model)))
    }
}

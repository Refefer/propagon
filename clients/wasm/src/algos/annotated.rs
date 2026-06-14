//! Algorithms over `annotated-pairs-dataset`: crowd-aware Bradley-Terry (batch).

use propagon::RankModel;
use propagon::Ranker;
use propagon::algos::{CrowdBt, CrowdBtModel as CoreCrowdBt};

use crate::Component;
use crate::datasets::AnnotatedPairsData;
use crate::errors::MapWit;
use crate::wit::annotated::{
    CrowdBtModel, CrowdBtModelBorrow, CrowdBtParams, Guest, GuestCrowdBtModel,
};
use crate::wit::datasets::AnnotatedPairsDatasetBorrow;
use crate::wit::types::Error;

batch_model!(CrowdBtMod, GuestCrowdBtModel, CoreCrowdBt);

fn crowd_bt(p: CrowdBtParams) -> CrowdBt {
    merge_params!(
        p,
        CrowdBt,
        scalar {
            lambda,
            eta_prior_alpha,
            eta_prior_beta,
            tolerance
        },
        usize {
            iterations,
            inner_sweeps
        }
    )
}

impl Guest for Component {
    type CrowdBtModel = CrowdBtMod;

    fn fit_crowd_bt(
        params: CrowdBtParams,
        data: AnnotatedPairsDatasetBorrow<'_>,
    ) -> Result<CrowdBtModel, Error> {
        let algo = crowd_bt(params);
        let ds = data.get::<AnnotatedPairsData>();
        let model = algo.fit(&ds.0.borrow()).map_wit()?;
        Ok(CrowdBtModel::new(CrowdBtMod(model)))
    }
    fn fit_warm_crowd_bt(
        params: CrowdBtParams,
        data: AnnotatedPairsDatasetBorrow<'_>,
        init: CrowdBtModelBorrow<'_>,
    ) -> Result<CrowdBtModel, Error> {
        let algo = crowd_bt(params);
        let ds = data.get::<AnnotatedPairsData>();
        let init = init.get::<CrowdBtMod>();
        let model = algo.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(CrowdBtModel::new(CrowdBtMod(model)))
    }
    fn load_crowd_bt(state: String) -> Result<CrowdBtModel, Error> {
        let model = CoreCrowdBt::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(CrowdBtModel::new(CrowdBtMod(model)))
    }
}

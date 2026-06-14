//! Algorithms over `graph-dataset`: PageRank and HITS (batch).

use propagon::RankModel;
use propagon::Ranker;
use propagon::algos::{Hits, HitsModel as CoreHits, PageRank, PageRankModel as CorePageRank};

use crate::Component;
use crate::datasets::GraphData;
use crate::enums::{sink, teleport};
use crate::errors::MapWit;
use crate::wit::datasets::GraphDatasetBorrow;
use crate::wit::graph::{
    Guest, GuestHitsModel, GuestPageRankModel, HitsModel, HitsModelBorrow, HitsParams,
    PageRankModel, PageRankModelBorrow, PageRankParams,
};
use crate::wit::types::Error;

batch_model!(PageRankMod, GuestPageRankModel, CorePageRank);
batch_model!(HitsMod, GuestHitsModel, CoreHits);

fn pr_params(p: PageRankParams) -> PageRank {
    let mut pr = merge_params!(p, PageRank, scalar { damping }, usize { iterations });
    if let Some(s) = p.sink {
        pr.sink = sink(s);
    }
    if let Some(t) = p.teleport {
        pr.teleport = teleport(t);
    }
    pr
}

impl Guest for Component {
    type PageRankModel = PageRankMod;
    type HitsModel = HitsMod;

    fn fit_page_rank(
        params: PageRankParams,
        data: GraphDatasetBorrow<'_>,
    ) -> Result<PageRankModel, Error> {
        let pr = pr_params(params);
        let ds = data.get::<GraphData>();
        let model = pr.fit(&ds.0.borrow()).map_wit()?;
        Ok(PageRankModel::new(PageRankMod(model)))
    }
    fn fit_warm_page_rank(
        params: PageRankParams,
        data: GraphDatasetBorrow<'_>,
        init: PageRankModelBorrow<'_>,
    ) -> Result<PageRankModel, Error> {
        let pr = pr_params(params);
        let ds = data.get::<GraphData>();
        let init = init.get::<PageRankMod>();
        let model = pr.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(PageRankModel::new(PageRankMod(model)))
    }
    fn load_page_rank(state: String) -> Result<PageRankModel, Error> {
        let model = CorePageRank::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(PageRankModel::new(PageRankMod(model)))
    }

    fn fit_hits(params: HitsParams, data: GraphDatasetBorrow<'_>) -> Result<HitsModel, Error> {
        let h = merge_params!(params, Hits, scalar { tolerance }, usize { iterations });
        let ds = data.get::<GraphData>();
        let model = h.fit(&ds.0.borrow()).map_wit()?;
        Ok(HitsModel::new(HitsMod(model)))
    }
    fn fit_warm_hits(
        params: HitsParams,
        data: GraphDatasetBorrow<'_>,
        init: HitsModelBorrow<'_>,
    ) -> Result<HitsModel, Error> {
        let h = merge_params!(params, Hits, scalar { tolerance }, usize { iterations });
        let ds = data.get::<GraphData>();
        let init = init.get::<HitsMod>();
        let model = h.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(HitsModel::new(HitsMod(model)))
    }
    fn load_hits(state: String) -> Result<HitsModel, Error> {
        let model = CoreHits::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(HitsModel::new(HitsMod(model)))
    }
}

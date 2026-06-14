//! Algorithms over `graph-dataset`: PageRank (batch). Phase 1 surface.

use propagon::RankModel;
use propagon::Ranker;
use propagon::algos::{PageRank, PageRankModel as CorePageRankModel};

use crate::Component;
use crate::algos::{bulk, save, score_of, sorted, top_k};
use crate::bindings::exports::propagon::core::datasets::GraphDatasetBorrow;
use crate::bindings::exports::propagon::core::graph::{
    Guest, GuestPageRankModel, PageRankModel, PageRankParams,
};
use crate::bindings::exports::propagon::core::types::{Error, ScoresBulk};
use crate::datasets::GraphData;
use crate::enums::{sink, teleport};
use crate::errors::MapWit;

pub struct PageRankMod(CorePageRankModel);

fn pr_params(p: PageRankParams) -> PageRank {
    let mut pr = PageRank::default();
    if let Some(v) = p.damping {
        pr.damping = v;
    }
    if let Some(v) = p.iterations {
        pr.iterations = v as usize;
    }
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
        init: crate::bindings::exports::propagon::core::graph::PageRankModelBorrow<'_>,
    ) -> Result<PageRankModel, Error> {
        let pr = pr_params(params);
        let ds = data.get::<GraphData>();
        let init = init.get::<PageRankMod>();
        let model = pr.fit_warm(&ds.0.borrow(), &init.0).map_wit()?;
        Ok(PageRankModel::new(PageRankMod(model)))
    }

    fn load_page_rank(state: String) -> Result<PageRankModel, Error> {
        let model = CorePageRankModel::load_jsonl(state.as_bytes()).map_wit()?;
        Ok(PageRankModel::new(PageRankMod(model)))
    }
}

impl GuestPageRankModel for PageRankMod {
    fn algorithm(&self) -> String {
        self.0.algorithm().to_string()
    }
    fn sorted_scores(&self) -> Vec<(String, f64)> {
        sorted(&self.0)
    }
    fn score(&self, name: String) -> Option<f64> {
        score_of(&self.0, &name)
    }
    fn top(&self, k: u32) -> Vec<(String, f64)> {
        top_k(&self.0, k)
    }
    fn scores_bulk(&self) -> ScoresBulk {
        bulk(&self.0)
    }
    fn save_state(&self) -> Result<String, Error> {
        save(&self.0)
    }
}

//! Algorithms over `graph-dataset`: PageRank, HITS, BiRank, degree, harmonic,
//! Katz, k-core, and LeaderRank centralities (all batch).

use propagon::algos::{
    BiRank, BiRankModel as CoreBiRank, Degree, DegreeModel as CoreDegree, Harmonic,
    HarmonicModel as CoreHarmonic, Hits, HitsModel as CoreHits, KCore, KCoreModel as CoreKCore,
    Katz, KatzModel as CoreKatz, LeaderRank, LeaderRankModel as CoreLeaderRank, PageRank,
    PageRankModel as CorePageRank,
};

use crate::Component;
use crate::datasets::GraphData;
use crate::enums::{sink, source_budget, teleport, unit_enum};
use crate::wit::datasets::GraphDatasetBorrow;
use crate::wit::graph::{
    BiRankModel, BiRankModelBorrow, BiRankParams, DegreeModel, DegreeModelBorrow, DegreeParams,
    Guest, GuestBiRankModel, GuestDegreeModel, GuestHarmonicModel, GuestHitsModel, GuestKCoreModel,
    GuestKatzModel, GuestLeaderRankModel, GuestPageRankModel, HarmonicModel, HarmonicModelBorrow,
    HarmonicParams, HitsModel, HitsModelBorrow, HitsParams, KCoreModel, KatzModel, KatzModelBorrow,
    KatzParams, LeaderRankModel, LeaderRankModelBorrow, LeaderRankParams, PageRankModel,
    PageRankModelBorrow, PageRankParams,
};
use crate::wit::types::Error;

batch_model!(PageRankMod, GuestPageRankModel, CorePageRank);
batch_model!(HitsMod, GuestHitsModel, CoreHits);
batch_model!(BiRankMod, GuestBiRankModel, CoreBiRank);
batch_model!(DegreeMod, GuestDegreeModel, CoreDegree);
batch_model!(HarmonicMod, GuestHarmonicModel, CoreHarmonic);
batch_model!(KatzMod, GuestKatzModel, CoreKatz);
batch_model!(KCoreMod, GuestKCoreModel, CoreKCore);
batch_model!(LeaderRankMod, GuestLeaderRankModel, CoreLeaderRank);

fn pr_build(p: PageRankParams) -> Result<PageRank, Error> {
    let mut pr = merge_params!(p, PageRank, scalar { damping }, usize { iterations });
    if let Some(s) = p.sink {
        pr.sink = sink(s);
    }
    if let Some(t) = p.teleport {
        pr.teleport = teleport(t);
    }
    Ok(pr)
}
fn hits_build(p: HitsParams) -> Result<Hits, Error> {
    Ok(merge_params!(
        p,
        Hits,
        scalar { tolerance },
        usize { iterations }
    ))
}
fn bi_rank_build(p: BiRankParams) -> Result<BiRank, Error> {
    Ok(merge_params!(
        p,
        BiRank,
        scalar { alpha, beta, seed },
        usize { iterations }
    ))
}
fn degree_build(p: DegreeParams) -> Result<Degree, Error> {
    let mut d = Degree::default();
    if let Some(s) = p.direction {
        d.direction = unit_enum(&s, "direction")?;
    }
    Ok(d)
}
fn harmonic_build(p: HarmonicParams) -> Result<Harmonic, Error> {
    let mut h = Harmonic::default();
    if let Some(s) = p.direction {
        h.direction = unit_enum(&s, "direction")?;
    }
    if let Some(s) = p.cost {
        h.cost = unit_enum(&s, "cost")?;
    }
    if let Some(b) = p.sources {
        h.sources = source_budget(b);
    }
    Ok(h)
}
fn katz_build(p: KatzParams) -> Result<Katz, Error> {
    Ok(merge_params!(
        p,
        Katz,
        scalar { alpha, tolerance },
        usize { iterations }
    ))
}
fn leader_rank_build(p: LeaderRankParams) -> Result<LeaderRank, Error> {
    Ok(merge_params!(
        p,
        LeaderRank,
        scalar { tolerance },
        usize { iterations }
    ))
}

impl Guest for Component {
    batch_algo!(
        PageRankModel,
        PageRankMod,
        CorePageRank,
        PageRankParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        PageRankModel,
        PageRankModelBorrow<'_>,
        fit_page_rank,
        fit_warm_page_rank,
        load_page_rank,
        pr_build
    );
    batch_algo!(
        HitsModel,
        HitsMod,
        CoreHits,
        HitsParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        HitsModel,
        HitsModelBorrow<'_>,
        fit_hits,
        fit_warm_hits,
        load_hits,
        hits_build
    );
    batch_algo!(
        BiRankModel,
        BiRankMod,
        CoreBiRank,
        BiRankParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        BiRankModel,
        BiRankModelBorrow<'_>,
        fit_bi_rank,
        fit_warm_bi_rank,
        load_bi_rank,
        bi_rank_build
    );
    batch_algo!(
        DegreeModel,
        DegreeMod,
        CoreDegree,
        DegreeParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        DegreeModel,
        DegreeModelBorrow<'_>,
        fit_degree,
        fit_warm_degree,
        load_degree,
        degree_build
    );
    batch_algo!(
        HarmonicModel,
        HarmonicMod,
        CoreHarmonic,
        HarmonicParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        HarmonicModel,
        HarmonicModelBorrow<'_>,
        fit_harmonic,
        fit_warm_harmonic,
        load_harmonic,
        harmonic_build
    );
    batch_algo!(
        KatzModel,
        KatzMod,
        CoreKatz,
        KatzParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        KatzModel,
        KatzModelBorrow<'_>,
        fit_katz,
        fit_warm_katz,
        load_katz,
        katz_build
    );
    nofield_algo!(
        KCoreModel,
        KCoreMod,
        CoreKCore,
        KCore,
        GraphData,
        GraphDatasetBorrow<'_>,
        KCoreModel,
        fit_k_core,
        load_k_core
    );
    batch_algo!(
        LeaderRankModel,
        LeaderRankMod,
        CoreLeaderRank,
        LeaderRankParams,
        GraphData,
        GraphDatasetBorrow<'_>,
        LeaderRankModel,
        LeaderRankModelBorrow<'_>,
        fit_leader_rank,
        fit_warm_leader_rank,
        load_leader_rank,
        leader_rank_build
    );
}

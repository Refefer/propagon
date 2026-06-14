//! Algorithm + model classes, grouped by the dataset shape they consume.

pub mod annotated;
pub mod betting;
pub mod contextual;
pub mod games;
pub mod graph;
pub mod matchups;
pub mod pairwise;
pub mod rankings;
pub mod rewards;
pub mod trajectories;

use pyo3::prelude::*;

/// Registers every algorithm and model class on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- games ---
    m.add_class::<games::PlayerState>()?;
    m.add_class::<games::Elo>()?;
    m.add_class::<games::EloModel>()?;
    m.add_class::<games::Glicko2>()?;
    m.add_class::<games::Glicko2Model>()?;
    m.add_class::<games::MovElo>()?;
    m.add_class::<games::MovEloModel>()?;
    m.add_class::<games::MElo>()?;
    m.add_class::<games::MEloModel>()?;
    m.add_class::<games::GeneralizedBt>()?;
    m.add_class::<games::GeneralizedBtModel>()?;
    m.add_class::<games::TeamBradleyTerry>()?;
    m.add_class::<games::TeamBtModel>()?;
    // --- graph ---
    m.add_class::<graph::PageRank>()?;
    m.add_class::<graph::PageRankModel>()?;
    m.add_class::<graph::Hits>()?;
    m.add_class::<graph::HitsModel>()?;
    m.add_class::<graph::BiRank>()?;
    m.add_class::<graph::BiRankModel>()?;
    m.add_class::<graph::Degree>()?;
    m.add_class::<graph::DegreeModel>()?;
    m.add_class::<graph::Harmonic>()?;
    m.add_class::<graph::HarmonicModel>()?;
    m.add_class::<graph::Katz>()?;
    m.add_class::<graph::KatzModel>()?;
    m.add_class::<graph::KCore>()?;
    m.add_class::<graph::KCoreModel>()?;
    m.add_class::<graph::LeaderRank>()?;
    m.add_class::<graph::LeaderRankModel>()?;
    // --- rewards / contextual ---
    m.add_class::<rewards::Bandit>()?;
    m.add_class::<rewards::BanditModel>()?;
    m.add_class::<rewards::SlidingWindowUcb>()?;
    m.add_class::<rewards::SwUcbModel>()?;
    m.add_class::<contextual::LinUcb>()?;
    m.add_class::<contextual::LinUcbModel>()?;
    // --- annotated ---
    m.add_class::<annotated::CrowdBt>()?;
    m.add_class::<annotated::CrowdBtModel>()?;
    // --- matchups ---
    m.add_class::<matchups::WengLin>()?;
    m.add_class::<matchups::WengLinModel>()?;
    m.add_class::<matchups::Rating>()?;
    // --- rankings ---
    m.add_class::<rankings::PlackettLuce>()?;
    m.add_class::<rankings::PlackettLuceModel>()?;
    m.add_class::<rankings::Footrule>()?;
    m.add_class::<rankings::FootruleModel>()?;
    m.add_class::<rankings::Mallows>()?;
    m.add_class::<rankings::MallowsModel>()?;
    m.add_class::<rankings::Mc4>()?;
    m.add_class::<rankings::Mc4Model>()?;
    // --- trajectories ---
    m.add_class::<trajectories::McValue>()?;
    m.add_class::<trajectories::McValueModel>()?;
    m.add_class::<trajectories::BehaviorCloning>()?;
    m.add_class::<trajectories::BcModel>()?;
    m.add_class::<trajectories::ValueCompare>()?;
    m.add_class::<trajectories::ValueCompareModel>()?;
    m.add_class::<trajectories::TdValue>()?;
    m.add_class::<trajectories::TdValueModel>()?;
    // --- pairwise ---
    m.add_class::<pairwise::BradleyTerryMM>()?;
    m.add_class::<pairwise::BtmMmModel>()?;
    m.add_class::<pairwise::BradleyTerryLR>()?;
    m.add_class::<pairwise::BtmLrModel>()?;
    m.add_class::<pairwise::BayesianBradleyTerry>()?;
    m.add_class::<pairwise::BayesBtModel>()?;
    m.add_class::<pairwise::Colley>()?;
    m.add_class::<pairwise::ColleyModel>()?;
    m.add_class::<pairwise::Massey>()?;
    m.add_class::<pairwise::MasseyModel>()?;
    m.add_class::<pairwise::Keener>()?;
    m.add_class::<pairwise::KeenerModel>()?;
    m.add_class::<pairwise::ILsr>()?;
    m.add_class::<pairwise::ILsrModel>()?;
    m.add_class::<pairwise::NashAveraging>()?;
    m.add_class::<pairwise::NashAveragingModel>()?;
    m.add_class::<pairwise::OffenseDefense>()?;
    m.add_class::<pairwise::OffenseDefenseModel>()?;
    m.add_class::<pairwise::RandomWalker>()?;
    m.add_class::<pairwise::RandomWalkerModel>()?;
    m.add_class::<pairwise::RankCentrality>()?;
    m.add_class::<pairwise::RankCentralityModel>()?;
    m.add_class::<pairwise::SerialRank>()?;
    m.add_class::<pairwise::SerialRankModel>()?;
    m.add_class::<pairwise::ThurstoneMosteller>()?;
    m.add_class::<pairwise::ThurstoneModel>()?;
    m.add_class::<pairwise::Whr>()?;
    m.add_class::<pairwise::WhrModel>()?;
    m.add_class::<pairwise::Borda>()?;
    m.add_class::<pairwise::BordaModel>()?;
    m.add_class::<pairwise::Copeland>()?;
    m.add_class::<pairwise::CopelandModel>()?;
    m.add_class::<pairwise::BladeChest>()?;
    m.add_class::<pairwise::BladeChestModel>()?;
    m.add_class::<pairwise::EsRum>()?;
    m.add_class::<pairwise::EsRumModel>()?;
    m.add_class::<pairwise::HodgeRank>()?;
    m.add_class::<pairwise::HodgeModel>()?;
    m.add_class::<pairwise::Kemeny>()?;
    m.add_class::<pairwise::KemenyModel>()?;
    m.add_class::<pairwise::Lsr>()?;
    m.add_class::<pairwise::LsrModel>()?;
    m.add_class::<pairwise::CovariateBt>()?;
    m.add_class::<pairwise::CovariateBtModel>()?;
    m.add_class::<pairwise::WinRate>()?;
    m.add_class::<pairwise::WinRateModel>()?;
    m.add_class::<pairwise::DuelingBandit>()?;
    m.add_class::<pairwise::DuelingModel>()?;
    // --- betting / portfolio (§14) ---
    m.add_class::<betting::OddsDevig>()?;
    m.add_class::<betting::OddsDevigModel>()?;
    m.add_class::<betting::OpinionPool>()?;
    m.add_class::<betting::OpinionPoolModel>()?;
    m.add_class::<betting::Lmsr>()?;
    m.add_class::<betting::LmsrModel>()?;
    Ok(())
}

/// Names this module contributes to `__all__` (Python class names).
pub(crate) const EXPORTS: &[&str] = &[
    // games
    "PlayerState",
    "Elo",
    "EloModel",
    "Glicko2",
    "Glicko2Model",
    "MovElo",
    "MovEloModel",
    "MElo",
    "MEloModel",
    "GeneralizedBt",
    "GeneralizedBtModel",
    "TeamBradleyTerry",
    "TeamBradleyTerryModel",
    // graph
    "PageRank",
    "PageRankModel",
    "Hits",
    "HitsModel",
    "BiRank",
    "BiRankModel",
    "Degree",
    "DegreeModel",
    "Harmonic",
    "HarmonicModel",
    "Katz",
    "KatzModel",
    "KCore",
    "KCoreModel",
    "LeaderRank",
    "LeaderRankModel",
    // rewards / contextual
    "Bandit",
    "BanditModel",
    "SlidingWindowUcb",
    "SlidingWindowUcbModel",
    "LinUcb",
    "LinUcbModel",
    // annotated
    "CrowdBt",
    "CrowdBtModel",
    // matchups
    "WengLin",
    "WengLinModel",
    "Rating",
    // rankings
    "PlackettLuce",
    "PlackettLuceModel",
    "Footrule",
    "FootruleModel",
    "Mallows",
    "MallowsModel",
    "Mc4",
    "Mc4Model",
    // trajectories
    "McValue",
    "McValueModel",
    "BehaviorCloning",
    "BehaviorCloningModel",
    "ValueCompare",
    "ValueCompareModel",
    "TdValue",
    "TdValueModel",
    // pairwise
    "BradleyTerryMM",
    "BradleyTerryMMModel",
    "BradleyTerryLR",
    "BradleyTerryLRModel",
    "BayesianBradleyTerry",
    "BayesianBradleyTerryModel",
    "Colley",
    "ColleyModel",
    "Massey",
    "MasseyModel",
    "Keener",
    "KeenerModel",
    "ILsr",
    "ILsrModel",
    "NashAveraging",
    "NashAveragingModel",
    "OffenseDefense",
    "OffenseDefenseModel",
    "RandomWalker",
    "RandomWalkerModel",
    "RankCentrality",
    "RankCentralityModel",
    "SerialRank",
    "SerialRankModel",
    "ThurstoneMosteller",
    "ThurstoneMostellerModel",
    "Whr",
    "WhrModel",
    "Borda",
    "BordaModel",
    "Copeland",
    "CopelandModel",
    "BladeChest",
    "BladeChestModel",
    "EsRum",
    "EsRumModel",
    "HodgeRank",
    "HodgeRankModel",
    "Kemeny",
    "KemenyModel",
    "Lsr",
    "LsrModel",
    "CovariateBt",
    "CovariateBtModel",
    "WinRate",
    "WinRateModel",
    "DuelingBandit",
    "DuelingBanditModel",
    // betting / portfolio (§14)
    "OddsDevig",
    "OddsDevigModel",
    "OpinionPool",
    "OpinionPoolModel",
    "Lmsr",
    "LmsrModel",
];

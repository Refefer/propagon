//! WIT resource implementations for the nine columnar dataset shapes.
//!
//! Every resource method takes `&self` in the generated `Guest*` traits, so each
//! wrapper holds its `propagon` dataset behind a `RefCell` for the mutating push
//! methods. String ids are interned by the core. `weight`/`reward` are narrowed
//! to the `f32` the dataset layer stores (overflow is rejected).

use std::cell::RefCell;

use crate::Component;
use crate::bindings::exports::propagon::core::datasets::{
    Guest, GuestAnnotatedPairsDataset, GuestContextualRewardsDataset, GuestForecastDataset,
    GuestGamesDataset, GuestGraphDataset, GuestMarketDataset, GuestMatchupsDataset,
    GuestOddsDataset, GuestPairwiseDataset, GuestRankingsDataset, GuestRewardsDataset,
    GuestTrajectoriesDataset,
};
use crate::bindings::exports::propagon::core::types::{Error, GameOutcome};
use crate::convert::{as_str_slice, narrow_f32};
use crate::enums::game_outcome;
use crate::errors::MapWit;

pub struct PairwiseData(pub RefCell<propagon::PairwiseDataset>);
pub struct GamesData(pub RefCell<propagon::GamesDataset>);
pub struct GraphData(pub RefCell<propagon::GraphDataset>);
pub struct RewardsData(pub RefCell<propagon::RewardsDataset>);
pub struct ContextualRewardsData(pub RefCell<propagon::ContextualRewardsDataset>);
pub struct MatchupsData(pub RefCell<propagon::MatchupsDataset>);
pub struct AnnotatedPairsData(pub RefCell<propagon::AnnotatedPairsDataset>);
pub struct RankingsData(pub RefCell<propagon::RankingsDataset>);
pub struct TrajectoriesData(pub RefCell<propagon::TrajectoriesDataset>);
pub struct OddsData(pub RefCell<propagon::OddsDataset>);
pub struct ForecastData(pub RefCell<propagon::ForecastDataset>);
pub struct MarketData(pub RefCell<propagon::MarketDataset>);

impl Guest for Component {
    type PairwiseDataset = PairwiseData;
    type GamesDataset = GamesData;
    type GraphDataset = GraphData;
    type RewardsDataset = RewardsData;
    type ContextualRewardsDataset = ContextualRewardsData;
    type MatchupsDataset = MatchupsData;
    type AnnotatedPairsDataset = AnnotatedPairsData;
    type RankingsDataset = RankingsData;
    type TrajectoriesDataset = TrajectoriesData;
    type OddsDataset = OddsData;
    type ForecastDataset = ForecastData;
    type MarketDataset = MarketData;
}

impl GuestPairwiseDataset for PairwiseData {
    fn new() -> Self {
        Self(RefCell::new(propagon::PairwiseDataset::new()))
    }
    fn push(&self, winner: String, loser: String, weight: f64) -> Result<(), Error> {
        let w = narrow_f32(weight, "weight")?;
        self.0.borrow_mut().push(&winner, &loser, w);
        Ok(())
    }
    fn new_period(&self) {
        self.0.borrow_mut().new_period();
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn n_periods(&self) -> u32 {
        self.0.borrow().n_periods() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestGamesDataset for GamesData {
    fn new() -> Self {
        Self(RefCell::new(propagon::GamesDataset::new()))
    }
    fn push_pair(&self, winner: String, loser: String, weight: f64) -> Result<(), Error> {
        let w = narrow_f32(weight, "weight")?;
        self.0.borrow_mut().push_pair(&winner, &loser, w).map_wit()
    }
    fn push_game(
        &self,
        side1: Vec<String>,
        side2: Vec<String>,
        outcome: GameOutcome,
        weight: f64,
    ) -> Result<(), Error> {
        let w = narrow_f32(weight, "weight")?;
        let o = game_outcome(outcome)?;
        let s1 = as_str_slice(&side1);
        let s2 = as_str_slice(&side2);
        self.0.borrow_mut().push_game(&s1, &s2, o, w).map_wit()
    }
    fn new_period(&self) {
        self.0.borrow_mut().new_period();
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn n_periods(&self) -> u32 {
        self.0.borrow().n_periods() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestGraphDataset for GraphData {
    fn new() -> Self {
        Self(RefCell::new(propagon::GraphDataset::new()))
    }
    fn push(&self, src: String, dst: String, weight: f64) -> Result<(), Error> {
        let w = narrow_f32(weight, "weight")?;
        self.0.borrow_mut().push(&src, &dst, w);
        Ok(())
    }
    fn n_nodes(&self) -> u32 {
        self.0.borrow().n_nodes() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestRewardsDataset for RewardsData {
    fn new() -> Self {
        Self(RefCell::new(propagon::RewardsDataset::new()))
    }
    fn push(&self, arm: String, reward: f64) -> Result<(), Error> {
        let r = narrow_f32(reward, "reward")?;
        self.0.borrow_mut().push(&arm, r);
        Ok(())
    }
    fn n_arms(&self) -> u32 {
        self.0.borrow().n_arms() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestContextualRewardsDataset for ContextualRewardsData {
    fn new() -> Self {
        Self(RefCell::new(propagon::ContextualRewardsDataset::new()))
    }
    fn push(&self, arm: String, reward: f64, x: Vec<f64>) -> Result<(), Error> {
        let r = narrow_f32(reward, "reward")?;
        self.0.borrow_mut().push(&arm, r, &x).map_wit()
    }
    fn n_arms(&self) -> u32 {
        self.0.borrow().n_arms() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestMatchupsDataset for MatchupsData {
    fn new() -> Self {
        Self(RefCell::new(propagon::MatchupsDataset::new()))
    }
    fn push_match(&self, teams: Vec<Vec<String>>, ranks: Vec<u32>) -> Result<(), Error> {
        let owned: Vec<Vec<&str>> = teams
            .iter()
            .map(|t| t.iter().map(String::as_str).collect())
            .collect();
        let slices: Vec<&[&str]> = owned.iter().map(Vec::as_slice).collect();
        self.0.borrow_mut().push_match(&slices, &ranks).map_wit()
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestAnnotatedPairsDataset for AnnotatedPairsData {
    fn new() -> Self {
        Self(RefCell::new(propagon::AnnotatedPairsDataset::new()))
    }
    fn push(
        &self,
        annotator: String,
        winner: String,
        loser: String,
        weight: f64,
    ) -> Result<(), Error> {
        let w = narrow_f32(weight, "weight")?;
        self.0.borrow_mut().push(&annotator, &winner, &loser, w);
        Ok(())
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn n_annotators(&self) -> u32 {
        self.0.borrow().n_annotators() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestRankingsDataset for RankingsData {
    fn new() -> Self {
        Self(RefCell::new(propagon::RankingsDataset::new()))
    }
    fn push_ranking(&self, ranking: Vec<String>) -> Result<(), Error> {
        let items = as_str_slice(&ranking);
        self.0.borrow_mut().push_ranking(items).map_wit()
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestTrajectoriesDataset for TrajectoriesData {
    fn new() -> Self {
        Self(RefCell::new(propagon::TrajectoriesDataset::new()))
    }
    fn push_step(&self, state: String, reward: f64) -> Result<(), Error> {
        let r = narrow_f32(reward, "reward")?;
        self.0.borrow_mut().push_step(&state, r).map_wit()
    }
    fn end_episode(&self) {
        self.0.borrow_mut().end_episode();
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn n_episodes(&self) -> u32 {
        self.0.borrow().n_episodes() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestOddsDataset for OddsData {
    fn new() -> Self {
        Self(RefCell::new(propagon::OddsDataset::new()))
    }
    fn push_event(&self, outcomes: Vec<(String, f64)>) -> Result<(), Error> {
        let pairs: Vec<(&str, f64)> = outcomes.iter().map(|(n, o)| (n.as_str(), *o)).collect();
        self.0.borrow_mut().push_event(&pairs).map_wit()
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestForecastDataset for ForecastData {
    fn new() -> Self {
        Self(RefCell::new(propagon::ForecastDataset::new()))
    }
    fn push_source(
        &self,
        name: String,
        weight: f64,
        forecast: Vec<(String, f64)>,
    ) -> Result<(), Error> {
        let pairs: Vec<(&str, f64)> = forecast.iter().map(|(n, p)| (n.as_str(), *p)).collect();
        self.0
            .borrow_mut()
            .push_source_weighted(&name, weight, &pairs)
            .map_wit()
    }
    fn n_outcomes(&self) -> u32 {
        self.0.borrow().n_outcomes() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

impl GuestMarketDataset for MarketData {
    fn new() -> Self {
        Self(RefCell::new(propagon::MarketDataset::new()))
    }
    fn push_trade(&self, outcome: String, shares: f64) -> Result<(), Error> {
        self.0.borrow_mut().push_trade(&outcome, shares).map_wit()
    }
    fn declare_outcome(&self, outcome: String) {
        self.0.borrow_mut().declare_outcome(&outcome);
    }
    fn n_entities(&self) -> u32 {
        self.0.borrow().n_entities() as u32
    }
    fn len(&self) -> u32 {
        self.0.borrow().len() as u32
    }
    fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

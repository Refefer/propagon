//! Python wrappers for the nine columnar dataset shapes.
//!
//! Each class owns a `propagon` dataset (string ids are interned for the
//! caller). Push methods default `weight` to `1.0`; weights/rewards are stored
//! in single precision, so a `float` that overflows `f32` range is rejected as
//! `InvalidInputError`. Methods the core validates surface their errors through
//! the propagon exception hierarchy.
//!
//! The trivial `__len__`/`is_empty`/`__repr__` are written out per class rather
//! than factored into a macro: a function-like macro call inside `#[pymethods]`
//! expands too late for the proc-macro to register the methods on the class.

use pyo3::prelude::*;

use crate::convert::{as_str_slice, narrow_f32};
use crate::enums::GameOutcome;
use crate::errors::MapPy;

/// Win/loss pairs: `winner` beat `loser` (the lowered tournament shape).
#[pyclass(name = "PairwiseDataset", module = "propagon._propagon")]
pub struct PairwiseDataset {
    pub(crate) inner: propagon::PairwiseDataset,
}

#[pymethods]
impl PairwiseDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::PairwiseDataset::new(),
        }
    }

    /// Records that `winner` beat `loser`, observed `weight` times.
    #[pyo3(signature = (winner, loser, weight = 1.0))]
    fn push(&mut self, winner: &str, loser: &str, weight: f64) -> PyResult<()> {
        let w = narrow_f32(weight, "weight")?;
        self.inner.push(winner, loser, w);
        Ok(())
    }

    /// Starts a new period at the current end of the dataset.
    fn new_period(&mut self) {
        self.inner.new_period();
    }

    /// Number of distinct entities seen so far.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Number of non-empty periods.
    fn n_periods(&self) -> usize {
        self.inner.n_periods()
    }

    /// Number of rows.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dataset holds no rows.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("PairwiseDataset(rows={})", self.inner.len())
    }
}

/// Team-vs-team game results with margins (the tournament input shape).
#[pyclass(name = "GamesDataset", module = "propagon._propagon")]
pub struct GamesDataset {
    pub(crate) inner: propagon::GamesDataset,
}

#[pymethods]
impl GamesDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::GamesDataset::new(),
        }
    }

    /// Records a 1v1 result: `winner` beat `loser`, observed `weight` times.
    #[pyo3(signature = (winner, loser, weight = 1.0))]
    fn push_pair(&mut self, winner: &str, loser: &str, weight: f64) -> PyResult<()> {
        let w = narrow_f32(weight, "weight")?;
        self.inner.push_pair(winner, loser, w).map_py()
    }

    /// Records one game: roster `side1` faced roster `side2` with `outcome`
    /// (side 1's perspective), observed `weight` times.
    #[pyo3(signature = (side1, side2, outcome, weight = 1.0))]
    fn push_game(
        &mut self,
        side1: Vec<String>,
        side2: Vec<String>,
        outcome: &GameOutcome,
        weight: f64,
    ) -> PyResult<()> {
        let w = narrow_f32(weight, "weight")?;
        let s1 = as_str_slice(&side1);
        let s2 = as_str_slice(&side2);
        self.inner.push_game(&s1, &s2, outcome.inner, w).map_py()
    }

    /// Starts a new rating period at the current end of the dataset.
    fn new_period(&mut self) {
        self.inner.new_period();
    }

    /// Number of distinct entities seen so far.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Number of non-empty rating periods.
    fn n_periods(&self) -> usize {
        self.inner.n_periods()
    }

    /// Number of games.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dataset holds no games.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("GamesDataset(games={})", self.inner.len())
    }
}

/// A directed, weighted endorsement graph (`src` endorses `dst`).
#[pyclass(name = "GraphDataset", module = "propagon._propagon")]
pub struct GraphDataset {
    pub(crate) inner: propagon::GraphDataset,
}

#[pymethods]
impl GraphDataset {
    /// An empty graph.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::GraphDataset::new(),
        }
    }

    /// Adds an edge `src -> dst` with the given `weight`.
    #[pyo3(signature = (src, dst, weight = 1.0))]
    fn push(&mut self, src: &str, dst: &str, weight: f64) -> PyResult<()> {
        let w = narrow_f32(weight, "weight")?;
        self.inner.push(src, dst, w);
        Ok(())
    }

    /// Number of distinct nodes.
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    /// Number of edges.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the graph holds no edges.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("GraphDataset(edges={})", self.inner.len())
    }
}

/// A `(arm, reward)` event log for multi-armed bandits.
#[pyclass(name = "RewardsDataset", module = "propagon._propagon")]
pub struct RewardsDataset {
    pub(crate) inner: propagon::RewardsDataset,
}

#[pymethods]
impl RewardsDataset {
    /// An empty event log.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::RewardsDataset::new(),
        }
    }

    /// Records one observed `reward` for `arm`.
    fn push(&mut self, arm: &str, reward: f64) -> PyResult<()> {
        let r = narrow_f32(reward, "reward")?;
        self.inner.push(arm, r);
        Ok(())
    }

    /// Number of distinct arms.
    fn n_arms(&self) -> usize {
        self.inner.n_arms()
    }

    /// Number of reward events.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the log holds no events.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("RewardsDataset(events={})", self.inner.len())
    }
}

/// A `(arm, reward, context)` event log for contextual bandits.
#[pyclass(name = "ContextualRewardsDataset", module = "propagon._propagon")]
pub struct ContextualRewardsDataset {
    pub(crate) inner: propagon::ContextualRewardsDataset,
}

#[pymethods]
impl ContextualRewardsDataset {
    /// An empty event log.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::ContextualRewardsDataset::new(),
        }
    }

    /// Records a `reward` for `arm` observed in context `x` (a feature vector;
    /// its length must be consistent across rows).
    fn push(&mut self, arm: &str, reward: f64, x: Vec<f64>) -> PyResult<()> {
        let r = narrow_f32(reward, "reward")?;
        self.inner.push(arm, r, &x).map_py()
    }

    /// Number of distinct arms.
    fn n_arms(&self) -> usize {
        self.inner.n_arms()
    }

    /// Number of reward events.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the log holds no events.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("ContextualRewardsDataset(events={})", self.inner.len())
    }
}

/// Multi-team match results with ranks (the OpenSkill/Weng-Lin input shape).
#[pyclass(name = "MatchupsDataset", module = "propagon._propagon")]
pub struct MatchupsDataset {
    pub(crate) inner: propagon::MatchupsDataset,
}

#[pymethods]
impl MatchupsDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::MatchupsDataset::new(),
        }
    }

    /// Records one match: `teams` (each a roster) finishing at the given
    /// `ranks` (1 = best; equal ranks are ties).
    fn push_match(&mut self, teams: Vec<Vec<String>>, ranks: Vec<u32>) -> PyResult<()> {
        let owned: Vec<Vec<&str>> = teams
            .iter()
            .map(|t| t.iter().map(String::as_str).collect())
            .collect();
        let slices: Vec<&[&str]> = owned.iter().map(Vec::as_slice).collect();
        self.inner.push_match(&slices, &ranks).map_py()
    }

    /// Number of distinct entities seen so far.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Number of matches.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dataset holds no matches.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("MatchupsDataset(matches={})", self.inner.len())
    }
}

/// Annotator-tagged pairwise votes (the crowd input shape).
#[pyclass(name = "AnnotatedPairsDataset", module = "propagon._propagon")]
pub struct AnnotatedPairsDataset {
    pub(crate) inner: propagon::AnnotatedPairsDataset,
}

#[pymethods]
impl AnnotatedPairsDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::AnnotatedPairsDataset::new(),
        }
    }

    /// Records that, per `annotator`, `winner` beat `loser` (weight `weight`).
    #[pyo3(signature = (annotator, winner, loser, weight = 1.0))]
    fn push(&mut self, annotator: &str, winner: &str, loser: &str, weight: f64) -> PyResult<()> {
        let w = narrow_f32(weight, "weight")?;
        self.inner.push(annotator, winner, loser, w);
        Ok(())
    }

    /// Number of distinct entities seen so far.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Number of distinct annotators.
    fn n_annotators(&self) -> usize {
        self.inner.n_annotators()
    }

    /// Number of votes.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dataset holds no votes.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("AnnotatedPairsDataset(votes={})", self.inner.len())
    }
}

/// Best-first ballots (the rank-aggregation input shape).
#[pyclass(name = "RankingsDataset", module = "propagon._propagon")]
pub struct RankingsDataset {
    pub(crate) inner: propagon::RankingsDataset,
}

#[pymethods]
impl RankingsDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::RankingsDataset::new(),
        }
    }

    /// Records one best-first ballot (`ranking[0]` is most preferred).
    fn push_ranking(&mut self, ranking: Vec<String>) -> PyResult<()> {
        let items = as_str_slice(&ranking);
        self.inner.push_ranking(items).map_py()
    }

    /// Number of distinct entities seen so far.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Number of ballots.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dataset holds no ballots.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("RankingsDataset(ballots={})", self.inner.len())
    }
}

/// State/reward episodes (the trajectory / value-estimation input shape).
#[pyclass(name = "TrajectoriesDataset", module = "propagon._propagon")]
pub struct TrajectoriesDataset {
    pub(crate) inner: propagon::TrajectoriesDataset,
}

#[pymethods]
impl TrajectoriesDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::TrajectoriesDataset::new(),
        }
    }

    /// Appends one `(state, reward)` step to the current episode.
    fn push_step(&mut self, state: &str, reward: f64) -> PyResult<()> {
        let r = narrow_f32(reward, "reward")?;
        self.inner.push_step(state, r).map_py()
    }

    /// Closes the current episode and starts a new one.
    fn end_episode(&mut self) {
        self.inner.end_episode();
    }

    /// Number of distinct states seen so far.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Number of completed episodes.
    fn n_episodes(&self) -> usize {
        self.inner.n_episodes()
    }

    /// Number of steps.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Whether the dataset holds no steps.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("TrajectoriesDataset(steps={})", self.inner.len())
    }
}

/// Posted betting odds grouped by events (§14.1 de-vigging).
#[pyclass(name = "OddsDataset", module = "propagon._propagon")]
pub struct OddsDataset {
    pub(crate) inner: propagon::OddsDataset,
}

#[pymethods]
impl OddsDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::OddsDataset::new(),
        }
    }

    /// Appends one event from `(outcome, decimal_odds)` pairs. Outcome names
    /// must be unique across the dataset; every `decimal_odds` must exceed 1.0.
    fn push_event(&mut self, outcomes: Vec<(String, f64)>) -> PyResult<()> {
        let pairs: Vec<(&str, f64)> = outcomes.iter().map(|(n, o)| (n.as_str(), *o)).collect();
        self.inner.push_event(&pairs).map_py()
    }

    /// Number of events.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of distinct outcomes.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Whether the dataset holds no events.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("OddsDataset(events={})", self.inner.len())
    }
}

/// Several sources' probability forecasts over a common outcome space
/// (§14.2 opinion pools).
#[pyclass(name = "ForecastDataset", module = "propagon._propagon")]
pub struct ForecastDataset {
    pub(crate) inner: propagon::ForecastDataset,
}

#[pymethods]
impl ForecastDataset {
    /// An empty dataset.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::ForecastDataset::new(),
        }
    }

    /// Appends a source's forecast from `(outcome, probability)` pairs. The
    /// probabilities must be non-negative and sum to 1; `weight` defaults to 1.
    #[pyo3(signature = (name, forecast, weight = 1.0))]
    fn push_source(
        &mut self,
        name: &str,
        forecast: Vec<(String, f64)>,
        weight: f64,
    ) -> PyResult<()> {
        let pairs: Vec<(&str, f64)> = forecast.iter().map(|(n, p)| (n.as_str(), *p)).collect();
        self.inner
            .push_source_weighted(name, weight, &pairs)
            .map_py()
    }

    /// Number of sources.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of distinct outcomes.
    fn n_outcomes(&self) -> usize {
        self.inner.n_outcomes()
    }

    /// Whether the dataset holds no sources.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("ForecastDataset(sources={})", self.inner.len())
    }
}

/// A prediction-market trade stream (§14.3 LMSR).
#[pyclass(name = "MarketDataset", module = "propagon._propagon")]
pub struct MarketDataset {
    pub(crate) inner: propagon::MarketDataset,
}

#[pymethods]
impl MarketDataset {
    /// An empty market.
    #[new]
    fn new() -> Self {
        Self {
            inner: propagon::MarketDataset::new(),
        }
    }

    /// Records a trade: `shares` of `outcome` (negative sells; 0 just declares).
    fn push_trade(&mut self, outcome: &str, shares: f64) -> PyResult<()> {
        self.inner.push_trade(outcome, shares).map_py()
    }

    /// Adds `outcome` to the universe without trading on it.
    fn declare_outcome(&mut self, outcome: &str) {
        self.inner.declare_outcome(outcome);
    }

    /// Number of trades.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of distinct outcomes.
    fn n_entities(&self) -> usize {
        self.inner.n_entities()
    }

    /// Whether the market holds no trades.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("MarketDataset(trades={})", self.inner.len())
    }
}

/// Registers the dataset classes on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PairwiseDataset>()?;
    m.add_class::<GamesDataset>()?;
    m.add_class::<GraphDataset>()?;
    m.add_class::<RewardsDataset>()?;
    m.add_class::<ContextualRewardsDataset>()?;
    m.add_class::<MatchupsDataset>()?;
    m.add_class::<AnnotatedPairsDataset>()?;
    m.add_class::<RankingsDataset>()?;
    m.add_class::<TrajectoriesDataset>()?;
    m.add_class::<OddsDataset>()?;
    m.add_class::<ForecastDataset>()?;
    m.add_class::<MarketDataset>()?;
    Ok(())
}

/// Names this module contributes to `__all__`.
pub(crate) const EXPORTS: &[&str] = &[
    "PairwiseDataset",
    "GamesDataset",
    "GraphDataset",
    "RewardsDataset",
    "ContextualRewardsDataset",
    "MatchupsDataset",
    "AnnotatedPairsDataset",
    "RankingsDataset",
    "TrajectoriesDataset",
    "OddsDataset",
    "ForecastDataset",
    "MarketDataset",
];

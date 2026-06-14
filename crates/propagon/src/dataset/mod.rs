//! Shared, columnar datasets (FR-1).
//!
//! Algorithms never define private input formats: every consumer takes one of
//! the dataset types here (or a borrowed [`GraphView`]). Each dataset
//! owns an [`Interner`](crate::Interner) so callers push string ids directly;
//! pre-interned `u32` paths exist for bulk ingestion. Datasets are immutable
//! during fitting — one dataset feeds many algorithms unchanged.

mod annotated;
mod contextual;
mod forecast;
mod games;
mod graph;
mod io;
mod market;
mod matchups;
mod odds;
mod pairwise;
mod rankings;
mod resample;
mod rewards;
mod trajectories;

pub use annotated::AnnotatedPairsDataset;
pub use contextual::ContextualRewardsDataset;
pub use forecast::ForecastDataset;
pub use games::{GameOutcome, GameView, GamesDataset, MarginTies, TiePolicy};
pub use graph::{Adjacency, GraphDataset, GraphView};
pub use market::MarketDataset;
pub use matchups::MatchupsDataset;
pub use odds::OddsDataset;
pub use pairwise::{PairwiseDataset, Tally};
pub use rankings::RankingsDataset;
pub use resample::Resample;
pub use rewards::RewardsDataset;
pub use trajectories::TrajectoriesDataset;

//! Shared, columnar datasets (FR-1).
//!
//! Algorithms never define private input formats: every consumer takes one of
//! the four dataset types here (or a borrowed [`GraphView`]). Each dataset
//! owns an [`Interner`](crate::Interner) so callers push string ids directly;
//! pre-interned `u32` paths exist for bulk ingestion. Datasets are immutable
//! during fitting — one dataset feeds many algorithms unchanged.

mod graph;
mod io;
mod pairwise;
mod rankings;
mod rewards;

pub use graph::{Adjacency, GraphDataset, GraphView};
pub use pairwise::{PairwiseDataset, Tally};
pub use rankings::RankingsDataset;
pub use rewards::RewardsDataset;

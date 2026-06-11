//! # propagon — ranking from revealed preferences
//!
//! Algorithms that turn revealed preferences — match outcomes, pairwise
//! choices, rankings, interaction graphs, and reward events — into rankings:
//! Bradley-Terry, Elo, Glicko-2, Luce spectral ranking, rank aggregation,
//! centrality, and multi-armed bandits.
//!
//! The method catalog (what each algorithm assumes, when to use which) lives
//! in `docs/algorithms.md`; the §15 decision guide there maps situations to
//! algorithms. Product requirements live in `docs/PRD.md`.
//!
//! ## Shape of the API
//!
//! 1. Build one of the four [dataset] types (string ids are interned for you).
//! 2. Configure an algorithm struct (all params have defaults).
//! 3. [`fit`](Ranker::fit) (batch) or [`init`](OnlineRanker::init) +
//!    [`update`](OnlineRanker::update) (incremental).
//! 4. Read [`sorted_scores`](RankModel::sorted_scores), or persist with
//!    [`save_jsonl`](RankModel::save_jsonl) and resume later.
//!
//! ```
//! use propagon::PairwiseDataset;
//!
//! let mut ds = PairwiseDataset::new();
//! ds.push("ARI", "COL", 1.0);
//! ds.push("COL", "NYM", 1.0);
//! ds.push("ARI", "NYM", 1.0);
//!
//! assert_eq!(ds.n_entities(), 3);
//! assert_eq!(ds.len(), 3);
//! ```
//!
//! ## Features
//!
//! - `parallel` (default): multi-threaded fitting via rayon.
//! - `io` (default): `save_to_path`/`load_from_path` conveniences.

pub mod algos;
pub mod dataset;
mod error;
mod interner;
mod mathx;
pub mod parallel;
mod progress;
pub mod state;
mod traits;

pub use dataset::{
    Adjacency, GraphDataset, GraphView, PairwiseDataset, RankingsDataset, RewardsDataset, Tally,
};
pub use error::{Error, Result};
pub use interner::Interner;
pub use progress::{NoProgress, Progress};
pub use traits::{FitOptions, OnlineRanker, RankModel, Ranker};

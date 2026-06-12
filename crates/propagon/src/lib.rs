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
//! use propagon::algos::Glicko2;
//! use propagon::{GamesDataset, OnlineRanker, RankModel};
//!
//! // One dataset, string ids interned for you.
//! let mut week1 = GamesDataset::new();
//! week1.push_pair("ARI", "COL", 1.0).unwrap();
//! week1.push_pair("ARI", "NYM", 1.0).unwrap();
//! week1.push_pair("COL", "NYM", 1.0).unwrap();
//!
//! // Incremental fitting: state persists, history is never replayed.
//! let glicko = Glicko2::default();
//! let mut ratings = glicko.init();
//! glicko.update(&mut ratings, &week1).unwrap();
//!
//! let top = ratings.sorted_scores()[0].0.to_string();
//! assert_eq!(top, "ARI");
//!
//! // Human-readable, resumable state (FR-4/FR-5).
//! let mut state = Vec::new();
//! ratings.save_jsonl(&mut state).unwrap();
//! let restored = propagon::algos::Glicko2Model::load_jsonl(state.as_slice()).unwrap();
//! assert_eq!(restored.sorted_scores()[0].0, "ARI");
//! ```
//!
//! ## Features
//!
//! - `parallel` (default): multi-threaded fitting via rayon.
//! - `io` (default): `save_to_path`/`load_from_path` conveniences.

// AGENTS.md rule 7: unit tests fail loud by design; production code may not.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod algos;
pub mod dataset;
mod error;
mod interner;
mod mathx;
pub mod parallel;
mod progress;
mod solver;
pub mod state;
mod traits;

pub use dataset::{
    Adjacency, AnnotatedPairsDataset, ContextualRewardsDataset, GameOutcome, GameView,
    GamesDataset, GraphDataset, GraphView, MarginTies, MatchupsDataset, PairwiseDataset,
    RankingsDataset, Resample, RewardsDataset, Tally, TiePolicy, TrajectoriesDataset,
};
pub use error::{Error, Result};
pub use interner::Interner;
pub use progress::{NoProgress, Progress, SILENT};
#[cfg(feature = "parallel")]
pub use traits::Threading;
pub use traits::{FitOptions, OnlineRanker, RankModel, Ranker};

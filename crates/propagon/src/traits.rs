//! The two fitting contracts ([`Ranker`], [`OnlineRanker`]) and the common
//! model surface ([`RankModel`]).
//!
//! - **Batch** algorithms (`Ranker`) consume a whole dataset and produce a
//!   model; iterative ones override [`Ranker::fit_warm_opts`] to start from a
//!   previous model and converge faster on appended data.
//! - **Online** algorithms (`OnlineRanker`) own true incremental state: each
//!   [`OnlineRanker::update_opts`] call folds a new batch in without ever
//!   revisiting history (Glicko-2, Elo, bandits, Wilson tallies).
//!
//! Which algorithm supports which tier is documented in `docs/PRD.md` (FR-5).

use std::io::{BufRead, Write};

use crate::error::Result;
use crate::progress::Progress;

/// Where parallel fitting work executes.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy, Default)]
pub enum Threading<'a> {
    /// rayon's shared global pool (the conventional default).
    #[default]
    Shared,
    /// A caller-owned dedicated pool (FR-3: never reconfigure the global
    /// pool out from under the host application).
    Dedicated(&'a rayon::ThreadPool),
}

/// Execution options shared by all fitting entry points.
///
/// `Default` means: silent progress, the shared thread pool (or sequential
/// execution when the `parallel` feature is off).
#[derive(Clone, Copy)]
pub struct FitOptions<'a> {
    /// Progress sink; defaults to the silent [`crate::NoProgress`].
    pub progress: &'a dyn Progress,
    /// Thread-pool selection.
    #[cfg(feature = "parallel")]
    pub threading: Threading<'a>,
}

impl Default for FitOptions<'_> {
    fn default() -> Self {
        Self {
            progress: &crate::progress::SILENT,
            #[cfg(feature = "parallel")]
            threading: Threading::Shared,
        }
    }
}

/// A fitted model: named scores plus self-contained JSONL persistence.
pub trait RankModel: Sized {
    /// Stable algorithm tag written into (and validated against) state files.
    fn algorithm(&self) -> &'static str;

    /// Primary score per entity, unordered. Higher is better unless the
    /// implementing type documents otherwise (Kemeny exposes rank positions).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)>;

    /// Scores sorted descending, ties broken by name for determinism.
    fn sorted_scores(&self) -> Vec<(&str, f64)> {
        let mut v: Vec<_> = self.scores().collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        v
    }

    /// Serializes the model as header-line JSONL (see `docs/PRD.md` FR-4).
    fn save_jsonl<W: Write>(&self, w: W) -> Result<()>;

    /// Loads a model previously written by [`RankModel::save_jsonl`].
    /// Fails with [`Error::AlgorithmMismatch`](crate::Error::AlgorithmMismatch)
    /// on a state file from a different algorithm.
    fn load_jsonl<R: BufRead>(r: R) -> Result<Self>;

    /// Convenience: save to a filesystem path.
    #[cfg(feature = "io")]
    fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let f = std::fs::File::create(path)?;
        self.save_jsonl(std::io::BufWriter::new(f))
    }

    /// Convenience: load from a filesystem path.
    #[cfg(feature = "io")]
    fn load_from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        Self::load_jsonl(std::io::BufReader::new(f))
    }
}

/// Batch fitting: dataset in, model out.
pub trait Ranker {
    /// The dataset shape this algorithm consumes.
    type Data;
    /// The fitted model this algorithm produces.
    type Model: RankModel;

    /// Fits a model from `data`, honoring `opts` (progress sink, thread pool).
    fn fit_opts(&self, data: &Self::Data, opts: &FitOptions<'_>) -> Result<Self::Model>;

    /// Fits a model from `data` with default options.
    fn fit(&self, data: &Self::Data) -> Result<Self::Model> {
        self.fit_opts(data, &FitOptions::default())
    }

    /// Refits using `init` as the starting point. The default ignores the
    /// initialization (correct but cold); iterative algorithms override it.
    /// Contract: never converges to a worse objective than [`Ranker::fit`].
    fn fit_warm_opts(
        &self,
        data: &Self::Data,
        init: &Self::Model,
        opts: &FitOptions<'_>,
    ) -> Result<Self::Model> {
        let _ = init;
        self.fit_opts(data, opts)
    }

    /// Warm-starts from `init` with default options.
    fn fit_warm(&self, data: &Self::Data, init: &Self::Model) -> Result<Self::Model> {
        self.fit_warm_opts(data, init, &FitOptions::default())
    }
}

/// Incremental fitting: state evolves batch by batch, history is never replayed.
pub trait OnlineRanker {
    /// The dataset shape this algorithm consumes.
    type Data;
    /// The incremental model this algorithm maintains.
    type Model: RankModel;

    /// Fresh state with no observations.
    fn init(&self) -> Self::Model;

    /// Folds one batch into `model`, honoring `opts` (progress, threading).
    fn update_opts(
        &self,
        model: &mut Self::Model,
        data: &Self::Data,
        opts: &FitOptions<'_>,
    ) -> Result<()>;

    /// Folds one batch into `model` with default options.
    fn update(&self, model: &mut Self::Model, data: &Self::Data) -> Result<()> {
        self.update_opts(model, data, &FitOptions::default())
    }
}

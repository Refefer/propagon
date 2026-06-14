//! Algorithm bindings, grouped by dataset shape (mirrors clients/python/src/algos).
//!
//! Each fitted model is a WIT `resource` wrapping a `propagon` model. The small
//! generic helpers below build the common model surface (`sorted-scores`,
//! `score`, `top`, `scores-bulk`, `save-state`) from any [`RankModel`], so the
//! per-algorithm impls stay thin. A declarative macro replaces this boilerplate
//! once the surface is scaled (Phase 2+).

pub mod annotated;
pub mod games;
pub mod graph;
pub mod matchups;
pub mod pairwise;
pub mod rankings;
pub mod rewards;
pub mod trajectories;

use propagon::RankModel;

use crate::bindings::exports::propagon::core::types::{Error, ScoresBulk};
use crate::errors::MapWit;

/// Descending scores, names resolved to owned `String`s.
pub(crate) fn sorted<M: RankModel>(m: &M) -> Vec<(String, f64)> {
    m.sorted_scores()
        .into_iter()
        .map(|(n, s)| (n.to_string(), s))
        .collect()
}

/// The score for one entity, or `None` if it is unknown.
pub(crate) fn score_of<M: RankModel>(m: &M, name: &str) -> Option<f64> {
    m.scores().find(|(n, _)| *n == name).map(|(_, s)| s)
}

/// The top `k` entities by descending score.
pub(crate) fn top_k<M: RankModel>(m: &M, k: u32) -> Vec<(String, f64)> {
    m.sorted_scores()
        .into_iter()
        .take(k as usize)
        .map(|(n, s)| (n.to_string(), s))
        .collect()
}

/// Bulk export: positionally-aligned `ids`/`scores` (descending), so large
/// models cross the boundary as a Float64Array + id list, not an object graph.
pub(crate) fn bulk<M: RankModel>(m: &M) -> ScoresBulk {
    let scored = m.sorted_scores();
    let mut ids = Vec::with_capacity(scored.len());
    let mut scores = Vec::with_capacity(scored.len());
    for (n, s) in scored {
        ids.push(n.to_string());
        scores.push(s);
    }
    ScoresBulk { ids, scores }
}

/// The model serialized as header-line JSONL.
pub(crate) fn save<M: RankModel>(m: &M) -> Result<String, Error> {
    let mut buf = Vec::new();
    m.save_jsonl(&mut buf).map_wit()?;
    String::from_utf8(buf).map_err(|e| Error::Io(e.to_string()))
}

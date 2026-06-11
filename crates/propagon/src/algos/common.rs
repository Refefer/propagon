//! Shared plumbing for model types: the `{id, score}` line format and a
//! macro implementing [`RankModel`](crate::RankModel) for models whose entire
//! state is one `f64` per entity.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::interner::Interner;

/// One entity line in a simple score model file.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ScoreLine {
    pub id: String,
    pub s: f64,
}

pub(crate) fn score_lines(names: &Interner, scores: &[f64]) -> Vec<ScoreLine> {
    names
        .names()
        .zip(scores)
        .map(|(id, &s)| ScoreLine { id: id.to_string(), s })
        .collect()
}

pub(crate) fn from_score_lines(lines: Vec<ScoreLine>) -> Result<(Interner, Vec<f64>)> {
    let scores: Vec<f64> = lines.iter().map(|l| l.s).collect();
    let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
    Ok((names, scores))
}

/// Implements [`RankModel`](crate::RankModel) for a model shaped as
/// `{ params, names: Interner, scores: Vec<f64> }`.
macro_rules! impl_simple_score_model {
    ($model:ty, $tag:literal) => {
        impl crate::RankModel for $model {
            fn algorithm(&self) -> &'static str {
                $tag
            }

            fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
                self.names.names().zip(self.scores.iter().copied())
            }

            fn save_jsonl<W: std::io::Write>(&self, w: W) -> crate::Result<()> {
                let lines = crate::algos::common::score_lines(&self.names, &self.scores);
                crate::state::save_model(w, $tag, &self.params, &lines)
            }

            fn load_jsonl<R: std::io::BufRead>(r: R) -> crate::Result<Self> {
                let (params, lines) = crate::state::load_model(r, $tag)?;
                let (names, scores) = crate::algos::common::from_score_lines(lines)?;
                Ok(Self { params, names, scores })
            }
        }
    };
}

pub(crate) use impl_simple_score_model;

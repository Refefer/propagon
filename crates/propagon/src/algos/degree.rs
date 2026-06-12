//! Degree / strength centrality (`docs/algorithms.md` §4.1) — the
//! non-parametric counting baseline every fancier graph method should be
//! compared against.
//!
//! Weighted in-strength by default ("how much points at you"); `Direction`
//! switches to out-strength or total. Trivially interpretable, trivially
//! gameable (one node can inflate another's in-degree at will) — which is
//! exactly why PageRank exists; use this to sanity-check whether structure
//! is buying anything beyond raw popularity.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::GraphDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::traits::{FitOptions, Ranker};

/// Which incident edges count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Direction {
    /// Weighted in-degree (endorsements received).
    #[default]
    In,
    /// Weighted out-degree (endorsements given).
    Out,
    /// Both.
    Total,
}

/// Degree-centrality parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Degree {
    /// Which incident edges count: in-, out-, or total strength.
    pub direction: Direction,
}

/// Fitted strengths.
#[derive(Debug, Clone)]
pub struct DegreeModel {
    params: Degree,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(DegreeModel, "degree");

impl Ranker for Degree {
    type Data = GraphDataset;
    type Model = DegreeModel;

    fn fit_opts(&self, data: &GraphDataset, _opts: &FitOptions<'_>) -> Result<DegreeModel> {
        let g = data.view();
        if g.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let mut scores = vec![0.0f64; g.n_nodes()];
        for (s, d, w) in g.edges() {
            let w = f64::from(w);
            match self.direction {
                Direction::In => scores[d as usize] += w,
                Direction::Out => scores[s as usize] += w,
                Direction::Total => {
                    scores[s as usize] += w;
                    scores[d as usize] += w;
                }
            }
        }

        Ok(DegreeModel {
            params: *self,
            names: g.interner.clone(),
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    #[test]
    fn directions() {
        let mut g = GraphDataset::new();
        g.push("a", "b", 2.0);
        g.push("c", "b", 1.0);
        g.push("b", "a", 0.5);

        let by = |direction: Direction| -> std::collections::HashMap<String, f64> {
            Degree { direction }
                .fit(&g)
                .unwrap()
                .scores()
                .map(|(n, s)| (n.to_string(), s))
                .collect()
        };

        let i = by(Direction::In);
        assert_eq!((i["a"], i["b"], i["c"]), (0.5, 3.0, 0.0));
        let o = by(Direction::Out);
        assert_eq!((o["a"], o["b"], o["c"]), (2.0, 0.5, 1.0));
        let t = by(Direction::Total);
        assert_eq!((t["a"], t["b"], t["c"]), (2.5, 3.5, 1.0));
    }
}

//! HodgeRank — least-squares ranking on graphs with an inconsistency
//! diagnostic (`docs/algorithms.md` §3.6; Jiang, Lim, Yao & Ye 2011).
//!
//! Aggregates pairwise outcomes into a skew-symmetric edge flow `Y_ij`
//! ([`HodgeFlow`] picks the statistic), then finds node potentials `s`
//! minimizing `Σ w_ij (s_i − s_j − Y_ij)²` — the gradient component of the
//! Hodge decomposition, solved as a weighted graph Laplacian system.
//!
//! The leftover, `1 − ‖projection‖²/‖Y‖²`, is the **inconsistency**: the
//! share of the observed flow living in cycles (curl + harmonic) that *no*
//! score-based ranking can explain. Near 0 the data is essentially
//! rankable; near 1 a total order is fiction. This number is the unique
//! deliverable here — check it before trusting any ranking of the same
//! data.
//!
//! Assumes a connected comparison graph (disconnected groups share no
//! scale; the solver surfaces this as a convergence error). For
//! [`HodgeFlow::MeanMargin`] the row weight is read as a margin; for the
//! other flows it is a win weight.

use serde::{Deserialize, Serialize};

use crate::algos::common::{ScoreLine, from_score_lines, score_lines};
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::solver::SparseSymmetric;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Which skew-symmetric statistic each pair's comparisons aggregate into.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HodgeFlow {
    /// `ln((S_ij + 1) / (S_ji + 1))` — Laplace-smoothed log-odds; the
    /// scale-compatible choice when scores should resemble BT log-strengths.
    #[default]
    LogOdds,
    /// `(S_ij − S_ji) / (S_ij + S_ji)` — win-rate difference in [−1, 1].
    WinRateDelta,
    /// Mean signed margin per game (row weight read as the margin).
    MeanMargin,
}

/// HodgeRank parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HodgeRank {
    pub flow: HodgeFlow,
    /// Maximum CG iterations.
    pub iterations: usize,
    /// Relative residual target.
    pub tolerance: f64,
}

impl Default for HodgeRank {
    fn default() -> Self {
        Self {
            flow: HodgeFlow::LogOdds,
            iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// What the header `params` field persists: the inputs plus the fitted
/// inconsistency (state, so a reloaded model keeps its diagnostic).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PersistedParams {
    flow: HodgeFlow,
    iterations: usize,
    tolerance: f64,
    inconsistency: f64,
}

/// Fitted potentials (mean-zero) plus the cyclic-share diagnostic.
#[derive(Debug, Clone)]
pub struct HodgeModel {
    params: HodgeRank,
    names: Interner,
    scores: Vec<f64>,
    inconsistency: f64,
}

impl HodgeModel {
    /// Share of the observed flow that **no** ranking can explain:
    /// `Σ w (Y − ∇s)² / Σ w Y²`, in [0, 1]. Near 0 = consistent, rankable
    /// data; near 1 = the data is dominated by cycles.
    pub fn inconsistency(&self) -> f64 {
        self.inconsistency
    }
}

impl RankModel for HodgeModel {
    fn algorithm(&self) -> &'static str {
        "hodge-rank"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.scores.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            flow: self.params.flow,
            iterations: self.params.iterations,
            tolerance: self.params.tolerance,
            inconsistency: self.inconsistency,
        };
        let lines = score_lines(&self.names, &self.scores);
        state::save_model(w, "hodge-rank", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<ScoreLine>) =
            state::load_model(r, "hodge-rank")?;
        let (names, scores) = from_score_lines(lines)?;
        Ok(Self {
            params: HodgeRank {
                flow: params.flow,
                iterations: params.iterations,
                tolerance: params.tolerance,
            },
            names,
            scores,
            inconsistency: params.inconsistency,
        })
    }
}

impl Ranker for HodgeRank {
    type Data = PairwiseDataset;
    type Model = HodgeModel;

    fn fit_opts(&self, data: &PairwiseDataset, _opts: &FitOptions<'_>) -> Result<HodgeModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let n = data.n_entities();

        // Per ordered pair (lo, hi): (weight i→j of lo beating hi, reverse,
        // row counts in each direction).
        let mut pairs: std::collections::HashMap<(u32, u32), (f64, f64, f64, f64)> =
            std::collections::HashMap::new();

        for (w, l, x) in data.rows() {
            let x = f64::from(x);
            let (key, forward) = if w < l {
                ((w, l), true)
            } else {
                ((l, w), false)
            };
            let e = pairs.entry(key).or_insert((0.0, 0.0, 0.0, 0.0));

            if forward {
                e.0 += x;
                e.2 += 1.0;
            } else {
                e.1 += x;
                e.3 += 1.0;
            }
        }

        // Edge list: (i, j, weight, flow Y_ij), with Y antisymmetric.
        let mut edges: Vec<(u32, u32, f64, f64)> = Vec::with_capacity(pairs.len());

        for (&(i, j), &(s_ij, s_ji, c_ij, c_ji)) in &pairs {
            let (weight, flow) = match self.flow {
                HodgeFlow::LogOdds => (s_ij + s_ji, ((s_ij + 1.0) / (s_ji + 1.0)).ln()),
                HodgeFlow::WinRateDelta => (s_ij + s_ji, (s_ij - s_ji) / (s_ij + s_ji)),
                HodgeFlow::MeanMargin => (c_ij + c_ji, (s_ij - s_ji) / (c_ij + c_ji)),
            };
            edges.push((i, j, weight, flow));
        }

        // Deterministic assembly order (HashMap iteration varies).
        edges.sort_unstable_by_key(|&(i, j, _, _)| (i, j));

        let mut laplacian = SparseSymmetric::new(n);
        let mut divergence = vec![0.0; n];

        for &(i, j, w, y) in &edges {
            let (i, j) = (i as usize, j as usize);
            laplacian.add(i, i, w);
            laplacian.add(j, j, w);
            laplacian.add(i, j, -w);
            divergence[i] += w * y;
            divergence[j] -= w * y;
        }

        laplacian.compress();
        let scores = laplacian.solve_mean_zero(&divergence, self.iterations, self.tolerance)?;

        let mut residual = 0.0;
        let mut total = 0.0;

        for &(i, j, w, y) in &edges {
            let grad = scores[i as usize] - scores[j as usize];
            residual += w * (y - grad) * (y - grad);
            total += w * y * y;
        }

        let inconsistency = if total > 0.0 { residual / total } else { 0.0 };

        Ok(HodgeModel {
            params: *self,
            names: data.interner().clone(),
            scores,
            inconsistency,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flows generated exactly from potentials are fully explained: the
    /// recovered scores match and the inconsistency is ~0.
    #[test]
    fn gradient_flow_is_fully_explained() {
        // Potentials a=1, b=0, c=-1; MeanMargin flow = potential difference.
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        d.push("a", "c", 2.0);

        let m = HodgeRank {
            flow: HodgeFlow::MeanMargin,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();

        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!(m.inconsistency() < 1e-12, "{}", m.inconsistency());
        assert!((s["a"] - 1.0).abs() < 1e-8);
        assert!(s["b"].abs() < 1e-8);
        assert!((s["c"] + 1.0).abs() < 1e-8);
    }

    /// A perfect 3-cycle has no gradient component at all: scores tie and
    /// the inconsistency is 1 (pure curl).
    #[test]
    fn pure_cycle_is_fully_inconsistent() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        d.push("c", "a", 1.0);

        let m = HodgeRank {
            flow: HodgeFlow::MeanMargin,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();

        assert!(
            (m.inconsistency() - 1.0).abs() < 1e-9,
            "{}",
            m.inconsistency()
        );
        for (_, s) in m.scores() {
            assert!(s.abs() < 1e-9);
        }
    }

    /// Log-odds flow ranks a dominant round-robin correctly.
    #[test]
    fn log_odds_recovers_order() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 8.0);
        d.push("b", "a", 2.0);
        d.push("b", "c", 7.0);
        d.push("c", "b", 3.0);
        d.push("a", "c", 9.0);
        d.push("c", "a", 1.0);

        let m = HodgeRank::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
        assert!(m.inconsistency() < 0.5);
    }

    #[test]
    fn round_trip_preserves_inconsistency() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 2.0);
        d.push("b", "c", 2.0);
        d.push("c", "a", 1.0);

        let m = HodgeRank::default().fit(&d).unwrap();
        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = HodgeModel::load_jsonl(first.as_slice()).unwrap();
        assert_eq!(loaded.inconsistency(), m.inconsistency());

        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

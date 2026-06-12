//! Thurstone-Mosteller Case V ratings (`docs/algorithms.md` §1.3;
//! Thurstone 1927, Mosteller 1951).
//!
//! The original latent-score model: each entity's perceived quality is a
//! Gaussian draw around its scale value μ, and `P(i ≻ j) = Φ(μ_i − μ_j)`
//! (Case V: equal variances, zero correlation; the σ√2 is absorbed into the
//! μ scale). Fits the MLE by Jacobi-style diagonal Newton sweeps — every
//! entity steps `g_i / h_i` computed from a frozen copy of μ, with the
//! always-positive Fisher curvature as `h_i` — initialized from inverse-Φ of
//! smoothed empirical win rates and mean-centered every sweep (the
//! likelihood only identifies differences of μ).
//!
//! Assumes weights are win counts: a row `(a, b, x)` is `x` wins of `a`
//! over `b`.
//!
//! Gotcha: under separation (an undefeated or winless entity) the MLE does
//! not exist — μ runs away to ±∞, surfaced as a numeric error once any |μ|
//! passes 30. `pseudo_count` adds that many virtual wins in **both**
//! directions of every observed pair, which restores a finite optimum.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx::{norm_cdf, norm_pdf, norm_ppf};
use crate::traits::{FitOptions, Ranker};

/// Thurstone-Mosteller parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThurstoneMosteller {
    /// Maximum Newton sweeps.
    pub iterations: usize,
    /// Early exit when the mean |Δμ| per sweep drops below this.
    pub tolerance: f64,
    /// Virtual wins added in both directions of every observed pair
    /// (separation mitigation; 0 = pure MLE).
    pub pseudo_count: f64,
}

impl Default for ThurstoneMosteller {
    fn default() -> Self {
        Self {
            iterations: 1000,
            tolerance: 1e-10,
            pseudo_count: 0.0,
        }
    }
}

/// Fitted latent scale values μ (mean zero; higher is better).
#[derive(Debug, Clone)]
pub struct ThurstoneModel {
    params: ThurstoneMosteller,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(ThurstoneModel, "thurstone-mosteller");

/// Floor applied to Φ terms before dividing: Φ(−d) underflows past d ≈ 7,
/// and the divergence guard — not a 0/0 — should report that regime.
const PHI_FLOOR: f64 = 1e-12;

/// Divergence sentinel: Φ saturates within ~1e-12 by d ≈ 7, so no
/// non-separated dataset pushes any centered μ anywhere near this.
const MU_LIMIT: f64 = 30.0;

/// Jacobi damping: both endpoints of a pair step simultaneously, so a full
/// `g/h` step applies each pairwise correction twice — an oscillation that
/// *grows* even for two entities. Halving makes the two-entity sweep exactly
/// the 1-D Newton step on the gap.
const STEP_DAMPING: f64 = 0.5;

/// Shifts μ to mean zero (the likelihood is translation-invariant, so this
/// pins the one free degree of freedom).
fn center(mu: &mut [f64]) {
    let mean = mu.iter().sum::<f64>() / mu.len() as f64;
    mu.iter_mut().for_each(|m| *m -= mean);
}

impl Ranker for ThurstoneMosteller {
    type Data = PairwiseDataset;
    type Model = ThurstoneModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<ThurstoneModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if !(self.pseudo_count >= 0.0 && self.pseudo_count.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "pseudo_count must be finite and >= 0, got {}",
                self.pseudo_count
            )));
        }
        let n = data.n_entities();

        // Aggregate win weight per ordered pair, then regularize both
        // directions of every observed (unordered) pair.
        let mut wins: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, x) in data.rows() {
            *wins.entry((w, l)).or_default() += f64::from(x);
        }

        let mut unordered: Vec<(u32, u32)> =
            wins.keys().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        unordered.sort_unstable();
        unordered.dedup();

        if self.pseudo_count > 0.0 {
            for &(a, b) in &unordered {
                *wins.entry((a, b)).or_default() += self.pseudo_count;
                *wins.entry((b, a)).or_default() += self.pseudo_count;
            }
        }

        // Per-entity opponents as (j, w_ij, w_ji), in deterministic order.
        let mut neighbors: Vec<Vec<(u32, f64, f64)>> = vec![Vec::new(); n];
        for &(a, b) in &unordered {
            // A self-comparison carries no preference information.
            if a == b {
                continue;
            }
            let w_ab = wins.get(&(a, b)).copied().unwrap_or(0.0);
            let w_ba = wins.get(&(b, a)).copied().unwrap_or(0.0);
            neighbors[a as usize].push((b, w_ab, w_ba));
            neighbors[b as usize].push((a, w_ba, w_ab));
        }

        // Classical initialization: mean inverse-Φ of smoothed win rates
        // against each observed opponent. The clamp keeps norm_ppf strictly
        // inside its (0, 1) domain.
        let mut mu: Vec<f64> = neighbors
            .iter()
            .map(|adj| {
                if adj.is_empty() {
                    return 0.0;
                }
                let sum: f64 = adj
                    .iter()
                    .map(|&(_, w_ij, w_ji)| {
                        let p = ((w_ij + 0.5) / (w_ij + w_ji + 1.0)).clamp(1e-6, 1.0 - 1e-6);
                        norm_ppf(p)
                    })
                    .sum();
                sum / adj.len() as f64
            })
            .collect();
        center(&mut mu);

        let progress = opts.progress;
        progress.start("thurstone sweeps", Some(self.iterations as u64));

        for it in 0..self.iterations {
            let frozen = &mu;
            let mut next: Vec<f64> = (0..n)
                .map(|i| {
                    let mut g = 0.0;
                    let mut h = 0.0;

                    for &(j, w_ij, w_ji) in &neighbors[i] {
                        let d = frozen[i] - frozen[j as usize];
                        let p = norm_cdf(d).max(PHI_FLOOR);
                        let q = norm_cdf(-d).max(PHI_FLOOR);
                        let pdf = norm_pdf(d);
                        g += w_ij * pdf / p - w_ji * pdf / q;
                        h += (w_ij + w_ji) * pdf * pdf / (p * q);
                    }

                    // h = 0 only for entities with no comparisons (or pdf
                    // underflow far past the divergence limit): hold still.
                    if h > 0.0 {
                        frozen[i] + STEP_DAMPING * g / h
                    } else {
                        frozen[i]
                    }
                })
                .collect();
            center(&mut next);

            let runaway: Vec<&str> = next
                .iter()
                .enumerate()
                .filter(|&(_, &m)| !m.is_finite() || m.abs() > MU_LIMIT)
                .map(|(i, _)| data.interner().resolve(i as u32))
                .take(3)
                .collect();

            if !runaway.is_empty() {
                progress.finish();
                return Err(Error::Numeric(format!(
                    "thurstone MLE diverged at {:?}: the dataset is separable \
                     (some entity is undefeated or winless), so no finite \
                     optimum exists; set pseudo_count > 0 to regularize",
                    runaway.join(", ")
                )));
            }

            let delta: f64 = mu
                .iter()
                .zip(&next)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n as f64;
            mu = next;
            progress.update(it as u64 + 1);

            if delta < self.tolerance {
                break;
            }
        }

        progress.finish();
        center(&mut mu);

        Ok(ThurstoneModel {
            params: *self,
            names: data.interner().clone(),
            scores: mu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// Two entities, a beats b 3:1. The MLE's stationarity condition
    /// `3·φ(δ)/Φ(δ) = 1·φ(δ)/Φ(−δ)` reduces to `Φ(δ) = 3/4`, so the fitted
    /// gap must be the probit of 3/4.
    #[test]
    fn two_entity_gap_is_probit_of_win_rate() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        let m = ThurstoneMosteller::default().fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        let gap = s["a"] - s["b"];
        assert!(
            (gap - norm_ppf(0.75)).abs() < 1e-6,
            "gap {gap} vs {}",
            norm_ppf(0.75)
        );
    }

    #[test]
    fn transitive_data_recovers_order() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 4.0);
        d.push("b", "a", 2.0);
        d.push("a", "c", 4.0);
        d.push("c", "a", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "b", 1.0);
        let m = ThurstoneMosteller::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    /// An undefeated entity has no finite MLE; pseudo-counts restore one
    /// (and the undefeated entity still wins).
    #[test]
    fn separation_diverges_unless_regularized() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("a", "c", 2.0);
        d.push("b", "c", 2.0);
        d.push("c", "b", 1.0);
        assert!(matches!(
            ThurstoneMosteller::default().fit(&d),
            Err(Error::Numeric(_))
        ));

        let algo = ThurstoneMosteller {
            pseudo_count: 0.5,
            ..Default::default()
        };
        let m = algo.fit(&d).unwrap();
        assert_eq!(m.sorted_scores()[0].0, "a");
    }

    #[test]
    fn rejects_negative_pseudo_count() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        let algo = ThurstoneMosteller {
            pseudo_count: -0.5,
            ..Default::default()
        };
        assert!(matches!(algo.fit(&d), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn empty_dataset_is_an_error() {
        let d = PairwiseDataset::new();
        assert!(matches!(
            ThurstoneMosteller::default().fit(&d),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn scores_are_mean_centered() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 4.0);
        d.push("b", "a", 2.0);
        d.push("a", "c", 4.0);
        d.push("c", "a", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "b", 1.0);
        let m = ThurstoneMosteller::default().fit(&d).unwrap();
        let mean = m.scores().map(|(_, s)| s).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-12, "mean {mean}");
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);
        d.push("b", "c", 2.0);
        d.push("c", "b", 1.0);
        let m = ThurstoneMosteller::default().fit(&d).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = ThurstoneModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

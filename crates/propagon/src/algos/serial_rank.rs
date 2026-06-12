//! SerialRank — spectral ranking by seriation (`docs/algorithms.md` §3.5;
//! Fogel, d'Aspremont & Vojnović, NIPS 2014).
//!
//! Entities that beat similar sets of opponents are similar: build the
//! match-agreement similarity `S = ½(n·𝟙𝟙ᵀ + C Cᵀ)` from the net comparison
//! matrix `C`, then order entities by the Fiedler vector (second-smallest
//! eigenvector) of `S`'s Laplacian. When comparisons derive from any total
//! order, sorting by the Fiedler vector recovers it exactly, with robustness
//! to a bounded fraction of corrupted comparisons.
//!
//! **The output is an ordering, not strengths**: scores are Fiedler-vector
//! coordinates — their magnitudes carry spacing information on the seriation
//! axis but no win-probability semantics (the same carve-out Kemeny's
//! positions use). The sign is canonicalized so higher score = more wins.
//!
//! `S` is never materialized: every matvec uses
//! `Sx = ½(n·(𝟙ᵀx)·𝟙 + C(Cᵀx))` — two sparse passes over the comparison
//! pairs. The Fiedler vector comes from shifted power iteration on
//! `M = σI − L` (σ = a Gershgorin bound, so `M ⪰ 0`) with the all-ones
//! direction projected out each step.
//!
//! Gotchas: items sharing no comparisons still get `S_ij = n/2 > 0`, so the
//! Laplacian is connected and the solve always succeeds — but on genuinely
//! disconnected *data* the relative placement of the blocks is meaningless.
//! A near-zero eigengap (perfectly symmetric data) slows convergence; the
//! iteration cap then returns the best iterate with a warning, which still
//! orders sensibly.

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::parallel;
use crate::traits::{FitOptions, Ranker};

/// SerialRank parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SerialRank {
    /// Maximum power-iteration steps for the Fiedler vector.
    pub iterations: usize,
    /// Stop when successive iterates align to within this (1 − |⟨v, v'⟩|).
    pub tolerance: f64,
    /// Seed for the random start vector.
    pub seed: u64,
}

impl Default for SerialRank {
    fn default() -> Self {
        Self {
            iterations: 1000,
            tolerance: 1e-12,
            seed: 2014,
        }
    }
}

/// Fitted seriation coordinates (an ordering; magnitudes are spacings on
/// the seriation axis, not strengths).
#[derive(Debug, Clone)]
pub struct SerialRankModel {
    params: SerialRank,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(SerialRankModel, "serial-rank");

/// Sparse net-comparison matrix: `c_ij = (w_ij − w_ji)/(w_ij + w_ji)`,
/// stored once per ordered pair with a nonzero total.
struct NetComparisons {
    /// Row-major sparse rows of C (only nonzero entries).
    rows: Vec<Vec<(u32, f64)>>,
}

impl NetComparisons {
    fn build(data: &PairwiseDataset) -> Self {
        let n = data.n_entities();
        let mut wins: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, x) in data.rows() {
            *wins.entry((w, l)).or_default() += f64::from(x);
        }

        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for (&(a, b), &w_ab) in &wins {
            if a > b && wins.contains_key(&(b, a)) {
                continue;
            }
            let w_ba = wins.get(&(b, a)).copied().unwrap_or(0.0);
            let total = w_ab + w_ba;
            if total <= 0.0 {
                continue;
            }
            let c = (w_ab - w_ba) / total;
            if c != 0.0 {
                rows[a as usize].push((b, c));
                rows[b as usize].push((a, -c));
            }
        }
        for row in &mut rows {
            row.sort_unstable_by_key(|e| e.0);
        }
        NetComparisons { rows }
    }

    /// `y = Cᵀx` (C is antisymmetric, so `Cᵀx = −Cx`; computed directly
    /// from the stored rows for clarity).
    fn ct_mul(&self, x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
        for (i, row) in self.rows.iter().enumerate() {
            for &(j, c) in row {
                y[j as usize] += c * x[i];
            }
        }
    }

    /// `y = Cx`.
    fn c_mul(&self, x: &[f64], y: &mut [f64]) {
        for (i, row) in self.rows.iter().enumerate() {
            y[i] = row.iter().map(|&(j, c)| c * x[j as usize]).sum();
        }
    }

    /// `y = Sx` with `S = ½(n·(𝟙ᵀx)·𝟙 + C(Cᵀx))`, never materializing S.
    fn s_mul(&self, x: &[f64], scratch: &mut [f64], y: &mut [f64]) {
        let n = x.len() as f64;
        let ones_dot: f64 = x.iter().sum();
        self.ct_mul(x, scratch);
        self.c_mul(scratch, y);
        for v in y.iter_mut() {
            *v = 0.5 * (n * ones_dot + *v);
        }
    }
}

impl Ranker for SerialRank {
    type Data = PairwiseDataset;
    type Model = SerialRankModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<SerialRankModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n = data.n_entities();
        let net = NetComparisons::build(data);

        // Degrees d = S·𝟙 and the Gershgorin shift σ ≥ λ_max(L).
        let mut scratch = vec![0.0f64; n];
        let mut degree = vec![0.0f64; n];
        let ones = vec![1.0f64; n];
        net.s_mul(&ones, &mut scratch, &mut degree);
        let sigma = 2.0 * degree.iter().copied().fold(0.0f64, f64::max).max(1.0);

        let scores = parallel::run_scoped(opts, || {
            let progress = opts.progress;
            progress.start("fiedler iterations", Some(self.iterations as u64));

            // Random start, projected off 𝟙 and normalized.
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
            let mut v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
            project_and_normalize(&mut v);

            let mut sv = vec![0.0f64; n];
            let mut converged = false;
            for it in 0..self.iterations {
                // next = (σI − L)v = σv − d⊙v + Sv.
                net.s_mul(&v, &mut scratch, &mut sv);
                let mut next: Vec<f64> = (0..n)
                    .map(|i| sigma * v[i] - degree[i] * v[i] + sv[i])
                    .collect();
                project_and_normalize(&mut next);

                let align: f64 = v.iter().zip(&next).map(|(a, b)| a * b).sum();
                let done = 1.0 - align.abs() < self.tolerance;
                v = next;
                progress.update(it as u64 + 1);
                if done {
                    converged = true;
                    break;
                }
            }
            progress.finish();
            if !converged {
                log::warn!(
                    "serial-rank hit the iteration cap before the Fiedler \
                     vector settled (near-zero eigengap?); ordering may be soft"
                );
            }

            // Canonical sign: more total wins should mean a higher coordinate.
            let mut net_wins = vec![0.0f64; n];
            for (w, l, x) in data.rows() {
                net_wins[w as usize] += f64::from(x);
                net_wins[l as usize] -= f64::from(x);
            }
            let orient: f64 = v.iter().zip(&net_wins).map(|(a, b)| a * b).sum();
            if orient < 0.0 || (orient == 0.0 && v.first().copied().unwrap_or(0.0) < 0.0) {
                v.iter_mut().for_each(|x| *x = -*x);
            }
            v
        });

        Ok(SerialRankModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }
}

/// Removes the 𝟙/√n component and scales to unit norm (no-op safe on the
/// zero vector).
fn project_and_normalize(v: &mut [f64]) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    v.iter_mut().for_each(|x| *x -= mean);
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// Complete noiseless comparisons from a total order are recovered
    /// exactly (Fogel et al. 2014, exact-recovery regime).
    #[test]
    fn exact_recovery_on_noiseless_total_order() {
        let names = ["a", "b", "c", "d", "e", "f"];
        let mut d = PairwiseDataset::new();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                d.push(names[i], names[j], 1.0);
            }
        }
        let m = SerialRank::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, names);
    }

    /// Recovery survives a corrupted comparison (robustness claim).
    #[test]
    fn survives_one_flipped_comparison() {
        let names = ["a", "b", "c", "d", "e", "f"];
        let mut d = PairwiseDataset::new();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                // Flip exactly one far-apart pair: f beats a.
                if i == 0 && j == 5 {
                    d.push(names[j], names[i], 1.0);
                } else {
                    d.push(names[i], names[j], 1.0);
                }
            }
        }
        let m = SerialRank::default().fit(&d).unwrap();
        let order: Vec<&str> = m.sorted_scores().iter().map(|e| e.0).collect();
        assert_eq!(order, names);
    }

    /// 4-item hand check: complete noiseless order over {a,b,c,d}.
    ///
    /// C rows (a..d): c_ab = c_ac = c_ad = 1 etc. gives, for the chain of
    /// net comparisons, similarity S = ½(4·𝟙𝟙ᵀ + CCᵀ) with
    /// CCᵀ = [[3,1,-1,-3],[1,3,1,-1],[-1,1,3,1],[-3,-1,1,3]] · adjusted —
    /// rather than pin the full matrix, assert the structural consequence
    /// that is exact in this regime: Fiedler coordinates are strictly
    /// decreasing AND symmetric around 0 (v_a = −v_d, v_b = −v_c) because
    /// reversing the order is an automorphism of the fixture.
    #[test]
    fn fiedler_coordinates_are_symmetric_on_complete_order() {
        let names = ["a", "b", "c", "d"];
        let mut d = PairwiseDataset::new();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                d.push(names[i], names[j], 1.0);
            }
        }
        // tolerance 0 disables the alignment early-exit (it is quadratic in
        // the eigenvector residual, so it can stop with ~1e-4 coordinate
        // error); the full iteration budget pins the coordinates hard.
        let algo = SerialRank {
            tolerance: 0.0,
            ..SerialRank::default()
        };
        let m = algo.fit(&d).unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] + s["d"]).abs() < 1e-8, "outer pair symmetric");
        assert!((s["b"] + s["c"]).abs() < 1e-8, "inner pair symmetric");
        assert!(s["a"] > s["b"] && s["b"] > s["c"] && s["c"] > s["d"]);
    }

    #[test]
    fn deterministic_across_runs() {
        let mut d = PairwiseDataset::new();
        for i in 0..15u32 {
            for j in 0..15u32 {
                if i < j {
                    d.push(&i.to_string(), &j.to_string(), 1.0 + ((i + j) % 2) as f32);
                }
            }
        }
        let a = SerialRank::default().fit(&d).unwrap();
        let b = SerialRank::default().fit(&d).unwrap();
        let sa: Vec<f64> = a.scores().map(|(_, s)| s).collect();
        let sb: Vec<f64> = b.scores().map(|(_, s)| s).collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn empty_dataset_is_an_error() {
        assert!(matches!(
            SerialRank::default().fit(&PairwiseDataset::new()),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        let m = SerialRank::default().fit(&d).unwrap();
        let mut buf1 = Vec::new();
        m.save_jsonl(&mut buf1).unwrap();
        let m2 = SerialRankModel::load_jsonl(buf1.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf1, buf2);
    }
}

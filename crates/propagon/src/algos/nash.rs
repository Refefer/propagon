//! Nash averaging (`docs/algorithms.md` §9.2; Balduzzi, Tuyls, Pérolat &
//! Graepel, NeurIPS 2018; averaged-play convergence per Freund & Schapire
//! 1999).
//!
//! Builds the empirical antisymmetric payoff matrix
//! `â_ij = (w_ij − w_ji) / (w_ij + w_ji)` over observed pairs (unobserved
//! pairs contribute 0 — "assumed even"), finds the maximum-entropy Nash
//! equilibrium `p` of the symmetric zero-sum game `Â`, and scores every
//! entity against that adversarially-chosen mixture: `s_i = (Âp)_i`. The
//! ranking is invariant to duplicating entities — redundant near-clones
//! split equilibrium mass instead of inflating anyone's score.
//!
//! The equilibrium is computed without an LP: entropy-regularized
//! multiplicative weights on logits `z` with geometric temperature annealing
//! (`x = softmax(z)`, `z ← (1 − η)·z + (η/τ)·Âx`, whose τ-fixed-point is the
//! quantal-response equilibrium `x ∝ exp((Âx)/τ)`; τ halves every
//! `anneal_every` steps from 1, selecting the maxent equilibrium as τ → 0).
//! Suboptimality is certified by the duality gap `ε(x̄) = max_i (Âx̄)_i` of
//! the uniform iterate average x̄ — the game value is 0 by antisymmetry, so
//! ε bounds the distance from equilibrium, and Freund & Schapire give
//! `O(√(ln n / T))` for averaged play. The achieved gap is persisted and
//! exposed via [`NashAveragingModel::gap`].
//!
//! Assumes non-negative, finite comparison weights ([`Error::InvalidInput`]
//! otherwise); pairs whose aggregated weight is zero are treated as
//! unobserved. Fully deterministic — no RNG anywhere.
//!
//! Gotchas: `scores()` reports the Nash-averaged skill `s`, not the
//! distribution `p` ([`NashAveragingModel::distribution`]) — `p` is support
//! information (a fully cyclic rock-paper-scissors game legitimately yields
//! uniform `p` and all-zero `s`). The persisted `gap` is measured on the
//! iterate average, while `p` is the final annealed iterate.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Nash averaging parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NashAveraging {
    /// Maximum payoff evaluations (the initial uniform point counts as one).
    pub iterations: usize,
    /// Early exit once the averaged duality gap drops below this.
    pub tolerance: f64,
    /// Logit mixing rate η ∈ (0, 1].
    pub learning_rate: f64,
    /// Halve the temperature τ after this many updates.
    pub anneal_every: usize,
}

impl Default for NashAveraging {
    fn default() -> Self {
        Self {
            iterations: 200_000,
            tolerance: 1e-6,
            learning_rate: 0.5,
            anneal_every: 5_000,
        }
    }
}

/// What `save_jsonl` writes as the header `params`: the algorithm params plus
/// the duality gap the fit achieved (its certificate of equilibrium quality).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PersistedParams {
    iterations: usize,
    tolerance: f64,
    learning_rate: f64,
    anneal_every: usize,
    gap: f64,
}

/// One entity line: Nash-averaged skill `s` and equilibrium weight `p`.
#[derive(Debug, Serialize, Deserialize)]
struct EntityLine {
    id: String,
    s: f64,
    p: f64,
}

/// Fitted maxent Nash equilibrium and the skills measured against it.
#[derive(Debug, Clone)]
pub struct NashAveragingModel {
    params: NashAveraging,
    names: Interner,
    /// Nash-averaged skill `s_i = (Âp)_i`.
    s: Vec<f64>,
    /// Maxent Nash distribution.
    p: Vec<f64>,
    /// Duality gap achieved on the averaged iterate.
    gap: f64,
}

impl NashAveragingModel {
    /// The maxent Nash distribution over entities (sums to 1). Support
    /// width is the diagnostic: uniform means fully cyclic, a point mass
    /// means one entity dominates.
    pub fn distribution(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.p.iter().copied())
    }

    /// Duality gap `max_i (Âx̄)_i` achieved by the fit; the equilibrium is
    /// exact at 0, and any score is within `gap` of its value against a true
    /// equilibrium.
    pub fn gap(&self) -> f64 {
        self.gap
    }
}

impl RankModel for NashAveragingModel {
    fn algorithm(&self) -> &'static str {
        "nash-averaging"
    }

    /// Nash-averaged skill `s_i = (Âp)_i`; the distribution itself is in
    /// [`NashAveragingModel::distribution`].
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.s.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            iterations: self.params.iterations,
            tolerance: self.params.tolerance,
            learning_rate: self.params.learning_rate,
            anneal_every: self.params.anneal_every,
            gap: self.gap,
        };

        let lines: Vec<EntityLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| EntityLine {
                id: id.to_string(),
                s: self.s[i],
                p: self.p[i],
            })
            .collect();
        state::save_model(w, "nash-averaging", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<EntityLine>) =
            state::load_model(r, "nash-averaging")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params: NashAveraging {
                iterations: params.iterations,
                tolerance: params.tolerance,
                learning_rate: params.learning_rate,
                anneal_every: params.anneal_every,
            },
            names,
            s: lines.iter().map(|l| l.s).collect(),
            p: lines.iter().map(|l| l.p).collect(),
            gap: params.gap,
        })
    }
}

/// Sparse antisymmetric payoff: `(i, j, â_ij)` with `i < j`, sorted, so all
/// accumulation orders — and therefore the fit — are deterministic.
struct Payoff {
    entries: Vec<(u32, u32, f64)>,
}

impl Payoff {
    /// Aggregates the dataset into net win rates per unordered pair.
    /// Weights must be finite and non-negative; self-pairs and pairs with
    /// zero total weight contribute nothing (â = 0 is implicit).
    fn build(data: &PairwiseDataset) -> Result<Payoff> {
        let mut wins: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, x) in data.rows() {
            if !(x.is_finite() && x >= 0.0) {
                return Err(Error::InvalidInput(format!(
                    "nash averaging needs finite non-negative weights, got {x}"
                )));
            }
            if w != l {
                *wins.entry((w, l)).or_default() += f64::from(x);
            }
        }

        let mut keys: Vec<(u32, u32)> = wins
            .keys()
            .map(|&(a, b)| if a < b { (a, b) } else { (b, a) })
            .collect();
        keys.sort_unstable();
        keys.dedup();

        let mut entries = Vec::with_capacity(keys.len());
        for (i, j) in keys {
            let w_ij = wins.get(&(i, j)).copied().unwrap_or(0.0);
            let w_ji = wins.get(&(j, i)).copied().unwrap_or(0.0);
            let total = w_ij + w_ji;
            if total > 0.0 {
                entries.push((i, j, (w_ij - w_ji) / total));
            }
        }
        Ok(Payoff { entries })
    }

    /// `out = Â x`, exploiting antisymmetry: each stored `(i, j, v)` adds
    /// `v·x_j` to row `i` and `−v·x_i` to row `j`.
    fn matvec(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for &(i, j, v) in &self.entries {
            out[i as usize] += v * x[j as usize];
            out[j as usize] -= v * x[i as usize];
        }
    }
}

/// `x = softmax(z)`, max-subtracted for overflow safety.
fn softmax(z: &[f64], x: &mut [f64]) {
    let peak = z.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut total = 0.0;

    for (xi, &zi) in x.iter_mut().zip(z) {
        *xi = (zi - peak).exp();
        total += *xi;
    }

    for xi in x.iter_mut() {
        *xi /= total;
    }
}

impl Ranker for NashAveraging {
    type Data = PairwiseDataset;
    type Model = NashAveragingModel;

    fn fit_opts(
        &self,
        data: &PairwiseDataset,
        opts: &FitOptions<'_>,
    ) -> Result<NashAveragingModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if !(self.learning_rate > 0.0 && self.learning_rate <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "learning_rate must lie in (0, 1], got {}",
                self.learning_rate
            )));
        }
        if self.anneal_every == 0 {
            return Err(Error::InvalidInput(
                "anneal_every must be at least 1".into(),
            ));
        }

        let n = data.n_entities();
        // A 1-entity game has exactly one strategy: it is its own maxent
        // equilibrium with value 0, no iteration needed.
        if n == 1 {
            return Ok(NashAveragingModel {
                params: *self,
                names: data.interner().clone(),
                s: vec![0.0],
                p: vec![1.0],
                gap: 0.0,
            });
        }

        let payoff = Payoff::build(data)?;
        let progress = opts.progress;
        progress.start("nash-averaging steps", Some(self.iterations as u64));

        let mut z = vec![0.0; n];
        let mut x = vec![0.0; n];
        let mut ax = vec![0.0; n];
        let mut payoff_sum = vec![0.0; n];

        // The initial uniform point is iterate 1: with no observed payoff
        // asymmetry it is already exact (gap 0) and the loop never runs.
        softmax(&z, &mut x);
        payoff.matvec(&x, &mut ax);
        let mut evals = 1usize;
        let mut gap = averaged_gap(&mut payoff_sum, &ax, evals);
        progress.update(1);

        let eta = self.learning_rate;
        let mut tau = 1.0f64;
        while evals < self.iterations && gap >= self.tolerance {
            for (zi, &axi) in z.iter_mut().zip(&ax) {
                *zi = (1.0 - eta) * *zi + (eta / tau) * axi;
            }

            if evals.is_multiple_of(self.anneal_every) {
                // Floor τ above subnormal range so η/τ stays finite.
                tau = (tau * 0.5).max(f64::MIN_POSITIVE);
            }

            softmax(&z, &mut x);
            payoff.matvec(&x, &mut ax);
            evals += 1;
            gap = averaged_gap(&mut payoff_sum, &ax, evals);
            progress.update(evals as u64);
        }
        progress.finish();

        // p is the final annealed iterate (the maxent-selected equilibrium);
        // s scores everyone against it.
        let mut s = vec![0.0; n];
        payoff.matvec(&x, &mut s);
        Ok(NashAveragingModel {
            params: *self,
            names: data.interner().clone(),
            s,
            p: x,
            gap,
        })
    }
}

/// Folds one iterate's payoff vector into the running sum and returns the
/// duality gap of the uniform average: `max_i (Âx̄)_i = max_i Σ(Âx)_i / t`.
fn averaged_gap(payoff_sum: &mut [f64], ax: &[f64], evals: usize) -> f64 {
    let mut peak = f64::NEG_INFINITY;

    for (acc, &axi) in payoff_sum.iter_mut().zip(ax) {
        *acc += axi;
        peak = peak.max(*acc);
    }

    peak / evals as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rps() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 1.0);
        d.push("b", "c", 1.0);
        d.push("c", "a", 1.0);
        d
    }

    /// `a` beats everyone, `b` beats `c`: equilibrium mass concentrates on
    /// the dominant strategy.
    fn dominance() -> PairwiseDataset {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 4.0);
        d.push("a", "c", 4.0);
        d.push("b", "c", 3.0);
        d
    }

    #[test]
    fn exact_rps_is_uniform_with_zero_skill() {
        let algo = NashAveraging::default();
        let m = algo.fit(&rps()).unwrap();

        let p: std::collections::HashMap<_, _> = m.distribution().collect();
        for e in ["a", "b", "c"] {
            assert!((p[e] - 1.0 / 3.0).abs() < 1e-3, "p[{e}] = {}", p[e]);
        }
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        for e in ["a", "b", "c"] {
            assert!(s[e].abs() < 1e-3, "s[{e}] = {}", s[e]);
        }
        assert!(m.gap() < algo.tolerance, "gap {}", m.gap());
    }

    #[test]
    fn dominant_entity_takes_the_equilibrium() {
        let m = NashAveraging::default().fit(&dominance()).unwrap();

        let p: std::collections::HashMap<_, _> = m.distribution().collect();
        assert!(p["a"] > 0.99, "p[a] = {}", p["a"]);
        let total: f64 = m.distribution().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-9);

        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!(s["a"].abs() < 1e-2, "s[a] = {}", s["a"]);
        assert!(s["a"] > s["b"] && s["a"] > s["c"]);
        assert_eq!(m.sorted_scores()[0].0, "a");
    }

    /// Self-certification: recompute `max_i (Âp)_i` for the returned
    /// distribution directly from the data and check it against a loose
    /// bound — no reference numbers involved.
    #[test]
    fn returned_distribution_gap_recomputed() {
        let data = dominance();
        let m = NashAveraging::default().fit(&data).unwrap();

        let p: std::collections::HashMap<String, f64> =
            m.distribution().map(|(n, v)| (n.to_string(), v)).collect();
        // Â rows from the 1:0 aggregated rates: a beats b and c (+1), b
        // beats c (+1).
        let payoff_a = p["b"] + p["c"];
        let payoff_b = -p["a"] + p["c"];
        let payoff_c = -p["a"] - p["b"];
        let gap = payoff_a.max(payoff_b).max(payoff_c);
        assert!(gap <= 0.05, "recomputed gap {gap}");
        assert!(gap >= 0.0 - 1e-12, "a symmetric game's value is 0");
    }

    #[test]
    fn deterministic_bitwise_across_runs() {
        let algo = NashAveraging::default();
        let m1 = algo.fit(&dominance()).unwrap();
        let m2 = algo.fit(&dominance()).unwrap();

        let mut b1 = Vec::new();
        m1.save_jsonl(&mut b1).unwrap();
        let mut b2 = Vec::new();
        m2.save_jsonl(&mut b2).unwrap();
        assert_eq!(b1, b2, "no RNG anywhere: runs must be bit-identical");
    }

    #[test]
    fn round_trip_single_entity_and_empty() {
        let m = NashAveraging::default().fit(&dominance()).unwrap();
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = NashAveragingModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
        assert_eq!(m.gap(), m2.gap());

        // Single entity: degenerate equilibrium without iteration.
        let mut solo = PairwiseDataset::new();
        solo.push("only", "only", 1.0);
        let m = NashAveraging::default().fit(&solo).unwrap();
        assert_eq!(m.distribution().collect::<Vec<_>>(), vec![("only", 1.0)]);
        assert_eq!(m.scores().collect::<Vec<_>>(), vec![("only", 0.0)]);
        assert_eq!(m.gap(), 0.0);

        assert!(matches!(
            NashAveraging::default().fit(&PairwiseDataset::new()),
            Err(Error::EmptyDataset)
        ));
    }

    #[test]
    fn invalid_params_and_weights_are_rejected() {
        let mut bad = PairwiseDataset::new();
        bad.push("a", "b", -1.0);
        assert!(matches!(
            NashAveraging::default().fit(&bad),
            Err(Error::InvalidInput(_))
        ));

        let zero_anneal = NashAveraging {
            anneal_every: 0,
            ..NashAveraging::default()
        };
        assert!(zero_anneal.fit(&rps()).is_err());

        let bad_eta = NashAveraging {
            learning_rate: 1.5,
            ..NashAveraging::default()
        };
        assert!(bad_eta.fit(&rps()).is_err());
    }
}

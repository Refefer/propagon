//! Crowd-BT — annotator-aware Bradley-Terry (`docs/algorithms.md` §11.2;
//! Chen, Bennett, Collins-Thompson & Horvitz, WSDM 2013).
//!
//! Each annotator `k` has a reliability `η_k`: they report the BT-preferred
//! item with probability `η_k` and the reverse otherwise —
//! `P_k(i ≻ j) = η_k σ(s_i−s_j) + (1−η_k) σ(s_j−s_i)`. Fitting is EM over
//! the latent "was this vote truthful" indicator: the E-step computes each
//! vote's truthfulness responsibility, the M-step refits BT with
//! split/flipped weights (Hunter MM sweeps) and updates every `η_k` in
//! closed form under the paper's Beta(10, 1) prior. `η ≈ 1` reliable,
//! `≈ 0.5` spammer, `< 0.5` adversarial.
//!
//! Connectivity is unconditional: the paper's virtual-node regularizer
//! (every entity gets one virtual win and loss, weight λ, against an
//! anchored node) keeps the likelihood strictly concave — so no
//! undefeated/winless stripping, ever. Scores are anchored to the virtual
//! node at 0.
//!
//! Gotchas: initializing η = 1 exactly (the paper's choice for its
//! alternating optimizer) is a degenerate EM fixed point — initialization
//! here is the prior mean 10/11. The likelihood is invariant under the
//! global relabeling `(s, η) → (−s, 1−η)`; after convergence, if the
//! vote-weighted mean η is below ½ the fit is flipped to the convention
//! "annotators are on balance truthful".

use serde::{Deserialize, Serialize};

use crate::dataset::AnnotatedPairsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Crowd-BT parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrowdBt {
    /// Virtual-node regularization weight λ (must be positive; the paper
    /// used 0.5 in simulation and found λ ∈ [0.1, 10] robust).
    pub lambda: f64,
    /// Beta prior α on annotator reliability (paper: Beta(10, 1)).
    pub eta_prior_alpha: f64,
    /// Beta prior β.
    pub eta_prior_beta: f64,
    /// Outer EM iteration cap.
    pub iterations: usize,
    /// Relative change of the penalized log-likelihood that stops EM.
    pub tolerance: f64,
    /// Hunter MM sweeps per M-step.
    pub inner_sweeps: usize,
}

impl Default for CrowdBt {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            eta_prior_alpha: 10.0,
            eta_prior_beta: 1.0,
            iterations: 200,
            tolerance: 1e-8,
            inner_sweeps: 10,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CrowdLine {
    id: String,
    k: LineKind,
    v: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum LineKind {
    Entity,
    Annotator,
}

/// Fitted entity scores (`s = ln γ`, virtual node ≡ 0) plus per-annotator
/// reliabilities.
#[derive(Debug, Clone)]
pub struct CrowdBtModel {
    params: CrowdBt,
    entities: Interner,
    scores: Vec<f64>,
    annotators: Interner,
    etas: Vec<f64>,
}

impl CrowdBtModel {
    /// `(annotator, η)` reliabilities: ~1 truthful, ~0.5 random, <0.5
    /// adversarial.
    pub fn annotators(&self) -> impl Iterator<Item = (&str, f64)> {
        self.annotators.names().zip(self.etas.iter().copied())
    }
}

impl RankModel for CrowdBtModel {
    fn algorithm(&self) -> &'static str {
        "crowd-bt"
    }

    /// Entity scores only (log-strengths, virtual anchor at 0).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.entities.names().zip(self.scores.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<CrowdLine> = self
            .scores()
            .map(|(id, v)| CrowdLine {
                id: id.to_string(),
                k: LineKind::Entity,
                v,
            })
            .chain(self.annotators().map(|(id, v)| CrowdLine {
                id: id.to_string(),
                k: LineKind::Annotator,
                v,
            }))
            .collect();
        state::save_model(w, "crowd-bt", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (CrowdBt, Vec<CrowdLine>) = state::load_model(r, "crowd-bt")?;
        let mut entities = Interner::new();
        let mut annotators = Interner::new();
        let mut scores = Vec::new();
        let mut etas = Vec::new();

        for line in lines {
            match line.k {
                LineKind::Entity => {
                    entities.intern(&line.id);
                    scores.push(line.v);
                }
                LineKind::Annotator => {
                    annotators.intern(&line.id);
                    etas.push(line.v);
                }
            }
        }

        Ok(Self {
            params,
            entities,
            scores,
            annotators,
            etas,
        })
    }
}

impl Ranker for CrowdBt {
    type Data = AnnotatedPairsDataset;
    type Model = CrowdBtModel;

    fn fit_opts(
        &self,
        data: &AnnotatedPairsDataset,
        opts: &FitOptions<'_>,
    ) -> Result<CrowdBtModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        if self.lambda <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "crowd-bt needs lambda > 0 (the virtual node is what keeps \
                 the fit identifiable), got {}",
                self.lambda
            )));
        }
        if self.eta_prior_alpha < 1.0 || self.eta_prior_beta < 1.0 {
            return Err(Error::InvalidInput(format!(
                "eta prior must have alpha >= 1 and beta >= 1 for a proper \
                 closed-form MAP, got ({}, {})",
                self.eta_prior_alpha, self.eta_prior_beta
            )));
        }

        let n = data.n_entities();
        let m = data.n_annotators();
        let votes: Vec<(u32, u32, u32, f64)> = data
            .rows()
            .map(|(k, w, l, x)| (k, w, l, f64::from(x)))
            .collect();

        let mut gamma = vec![1.0f64; n];
        let mut eta = vec![self.eta_prior_alpha / (self.eta_prior_alpha + self.eta_prior_beta); m];
        let mut responsibility = vec![0.0f64; votes.len()];
        let mut last_ll = f64::NEG_INFINITY;

        let progress = opts.progress;
        progress.start("em iterations", Some(self.iterations as u64));

        for round in 0..self.iterations {
            // E-step: P(vote was truthful | data, current fit).
            for (v, &(k, w, l, x)) in votes.iter().enumerate() {
                let _ = x;
                let p = gamma[w as usize] / (gamma[w as usize] + gamma[l as usize]);
                let e = eta[k as usize];
                responsibility[v] = e * p / (e * p + (1.0 - e) * (1.0 - p));
            }

            // M-step (η): closed-form Beta MAP per annotator.
            let mut eta_num = vec![self.eta_prior_alpha - 1.0; m];
            let mut eta_den = vec![self.eta_prior_alpha + self.eta_prior_beta - 2.0; m];

            for (v, &(k, _, _, x)) in votes.iter().enumerate() {
                eta_num[k as usize] += x * responsibility[v];
                eta_den[k as usize] += x;
            }
            for (e, (num, den)) in eta.iter_mut().zip(eta_num.iter().zip(&eta_den)) {
                *e = num / den;
            }

            // M-step (s): Hunter MM on the split-weight BT problem.
            // wins[i] includes the virtual win; pair weights are symmetric
            // totals N_ij plus the virtual 2λ edge to the anchor.
            let mut wins = vec![self.lambda; n];
            let mut pair_n: std::collections::HashMap<(u32, u32), f64> =
                std::collections::HashMap::new();

            for (v, &(_, w, l, x)) in votes.iter().enumerate() {
                let r = responsibility[v];
                wins[w as usize] += x * r;
                wins[l as usize] += x * (1.0 - r);
                *pair_n.entry((w.min(l), w.max(l))).or_default() += x;
            }

            let mut adjacency: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
            let mut pairs: Vec<((u32, u32), f64)> = pair_n.into_iter().collect();
            pairs.sort_unstable_by_key(|&(key, _)| key);

            for ((i, j), total) in pairs {
                adjacency[i as usize].push((j, total));
                adjacency[j as usize].push((i, total));
            }

            for _ in 0..self.inner_sweeps {
                for i in 0..n {
                    let mut denom = 2.0 * self.lambda / (gamma[i] + 1.0); // virtual node, γ₀ = 1
                    for &(j, total) in &adjacency[i] {
                        denom += total / (gamma[i] + gamma[j as usize]);
                    }
                    gamma[i] = wins[i] / denom;
                }
            }

            // Penalized log-likelihood for the stopping rule.
            let mut ll = 0.0;
            for &(k, w, l, x) in &votes {
                let p = gamma[w as usize] / (gamma[w as usize] + gamma[l as usize]);
                let e = eta[k as usize];
                ll += x * (e * p + (1.0 - e) * (1.0 - p)).ln();
            }
            for &g in &gamma {
                ll += self.lambda * (g.ln() - 2.0 * (1.0 + g).ln());
            }
            // EM ascends the MAP objective, prior included.
            for &e in &eta {
                ll += (self.eta_prior_alpha - 1.0) * e.ln()
                    + (self.eta_prior_beta - 1.0) * (1.0 - e).max(f64::MIN_POSITIVE).ln();
            }

            progress.update(round as u64 + 1);
            progress.message(&format!("penalized loglik {ll:.6}"));

            if (ll - last_ll).abs() <= self.tolerance * ll.abs() {
                break;
            }
            last_ll = ll;
        }

        progress.finish();

        // Pin the global (s, η) ↔ (−s, 1−η) symmetry: annotators are on
        // balance truthful.
        let mut weight_sum = 0.0;
        let mut eta_mean = 0.0;
        for &(k, _, _, x) in &votes {
            eta_mean += x * eta[k as usize];
            weight_sum += x;
        }

        if eta_mean / weight_sum < 0.5 {
            gamma.iter_mut().for_each(|g| *g = 1.0 / *g);
            eta.iter_mut().for_each(|e| *e = 1.0 - *e);
        }

        Ok(CrowdBtModel {
            params: *self,
            entities: data.entities().clone(),
            scores: gamma.iter().map(|g| g.ln()).collect(),
            annotators: data.annotators().clone(),
            etas: eta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    /// Three annotators with known behavior: `good` votes the true order
    /// a > b > c, `spam` flips a coin, `troll` always inverts.
    fn crowd() -> AnnotatedPairsDataset {
        let mut d = AnnotatedPairsDataset::new();
        let truth = [("a", "b"), ("b", "c"), ("a", "c")];

        // Weights are large enough that the likelihood dominates the
        // Beta(10, 1) prior — with only a handful of votes the prior
        // (correctly) absolves a spammer.
        for (w, l) in truth {
            d.push("good", w, l, 120.0);
            d.push("good", l, w, 20.0); // genuine upsets
            d.push("spam", w, l, 50.0);
            d.push("spam", l, w, 50.0);
            d.push("troll", l, w, 70.0); // systematically inverted
        }
        d
    }

    #[test]
    fn recovers_order_and_reliabilities() {
        let m = CrowdBt::default().fit(&crowd()).unwrap();

        let order: Vec<&str> = m.sorted_scores().into_iter().map(|(n, _)| n).collect();
        assert_eq!(order, vec!["a", "b", "c"]);

        let etas: std::collections::HashMap<&str, f64> = m.annotators().collect();
        assert!(etas["good"] > 0.8, "good annotator: {}", etas["good"]);
        assert!(
            etas["spam"] > 0.35 && etas["spam"] < 0.75 && etas["spam"] < etas["good"],
            "spammer between troll and good: {}",
            etas["spam"]
        );
        assert!(etas["troll"] < 0.3, "troll detected: {}", etas["troll"]);
    }

    /// Likelihood ascent: more EM rounds never lose penalized likelihood.
    #[test]
    fn em_is_monotone() {
        let d = crowd();
        let ll = |iterations: usize| {
            let m = CrowdBt {
                iterations,
                tolerance: 0.0,
                ..Default::default()
            }
            .fit(&d)
            .unwrap();

            // Recompute the penalized objective from the fitted model.
            let gamma: std::collections::HashMap<&str, f64> =
                m.scores().map(|(n, s)| (n, s.exp())).collect();
            let etas: std::collections::HashMap<&str, f64> = m.annotators().collect();
            let mut ll = 0.0;
            for (k, w, l, x) in d.rows() {
                let gw = gamma[d.entities().name(w).unwrap()];
                let gl = gamma[d.entities().name(l).unwrap()];
                let p = gw / (gw + gl);
                let e = etas[d.annotators().name(k).unwrap()];
                ll += f64::from(x) * (e * p + (1.0 - e) * (1.0 - p)).ln();
            }
            for &g in gamma.values() {
                ll += 0.5 * (g.ln() - 2.0 * (1.0 + g).ln());
            }
            for &e in etas.values() {
                ll += 9.0 * e.ln(); // Beta(10, 1) prior, beta-1 = 0
            }
            ll
        };

        let (l1, l4, l16) = (ll(1), ll(4), ll(16));
        assert!(l4 >= l1 - 1e-9, "{l4} vs {l1}");
        assert!(l16 >= l4 - 1e-9, "{l16} vs {l4}");
    }

    #[test]
    fn lambda_zero_is_rejected() {
        let algo = CrowdBt {
            lambda: 0.0,
            ..Default::default()
        };
        assert!(matches!(algo.fit(&crowd()), Err(Error::InvalidInput(_))));
    }

    /// Larger λ shrinks scores toward the virtual anchor at 0.
    #[test]
    fn lambda_shrinks_scores() {
        let spread = |lambda: f64| {
            let m = CrowdBt {
                lambda,
                ..Default::default()
            }
            .fit(&crowd())
            .unwrap();
            m.scores().map(|(_, s)| s.abs()).fold(0.0, f64::max)
        };
        assert!(spread(10.0) < spread(0.1));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let m = CrowdBt::default().fit(&crowd()).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = CrowdBtModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

//! Bradley-Terry with covariates / conditional logit (`docs/algorithms.md`
//! §10.1; McFadden 1974; Agresti 2013).
//!
//! Entity strength is a linear function of known features, `s_i = β·x_i`,
//! optionally with per-entity intercepts `b_i` for partial pooling
//! (`s_i = β·x_i + b_i`); `P(i ≻ j) = σ(s_i − s_j)`. Fitting is Newton's
//! method on the ridge-penalized log-likelihood over aggregated ordered
//! pairs, with the dense symmetric Newton system solved by a self-contained
//! Cholesky factorization (the ridge keeps it positive definite). The payoff
//! over plain Bradley-Terry is cold-start scoring of unseen feature vectors
//! ([`CovariateBtModel::score`]) and coefficients that explain *why* things
//! win.
//!
//! Assumes every entity appearing in the dataset has a feature row in the
//! config, all rows share one dimensionality, and feature values are finite
//! — violations are typed [`Error::InvalidInput`]s naming the offenders.
//!
//! Gotchas: with `intercepts = true` the Newton system is `(d+n)×(d+n)`
//! dense, so each step costs `O((d+n)³)` — partial pooling over huge entity
//! sets is the wrong tool. Without the ridge, `β·x` and the intercepts are
//! confounded (any `β` can be absorbed into `b`), so `intercepts = true`
//! requires `l2 > 0`. `score()` for an unseen feature vector returns `β·x`
//! only — an unseen entity's intercept is its prior mean, 0. Periods are
//! ignored: this is a batch fitter over all rows.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dataset::PairwiseDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// Conditional-logit Bradley-Terry parameters. Owns the entity feature
/// table, so unlike most algorithm configs it is not `Copy`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CovariateBt {
    /// Entity feature vectors `(name, features)`; all rows must share one
    /// dimensionality. Entities in the data but absent here → typed error.
    pub features: Vec<(String, Vec<f64>)>,
    /// Ridge penalty on β (and intercepts when enabled). Must be > 0 when
    /// `intercepts` is on (identifiability).
    pub l2: f64,
    /// Per-entity intercepts `b_i` (partial pooling): `s_i = β·x_i + b_i`.
    /// Each Newton step then solves a dense `(d+n)×(d+n)` system.
    pub intercepts: bool,
    /// Maximum Newton steps.
    pub iterations: usize,
    /// Stop when the largest |Δθ| of a step drops below this.
    pub tolerance: f64,
}

impl CovariateBt {
    /// A config over `features` with the documented defaults:
    /// `l2 = 1e-4`, `intercepts = false`, `iterations = 500`,
    /// `tolerance = 1e-8`.
    pub fn new(features: Vec<(String, Vec<f64>)>) -> Self {
        Self {
            features,
            l2: 1e-4,
            intercepts: false,
            iterations: 500,
            tolerance: 1e-8,
        }
    }

    /// Rejects penalty settings the model is not identifiable (or finite)
    /// under.
    fn validate(&self) -> Result<()> {
        if !self.l2.is_finite() || self.l2 < 0.0 {
            return Err(Error::InvalidInput(format!(
                "l2 must be finite and >= 0, got {}",
                self.l2
            )));
        }
        if self.intercepts && self.l2 == 0.0 {
            return Err(Error::InvalidInput(
                "intercepts = true requires l2 > 0: without the ridge, β·x and the \
                 per-entity intercepts are confounded (any β is absorbable into b)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Validates the feature table against the dataset and indexes it by
    /// dense entity id: rejects duplicate rows, inconsistent or zero
    /// dimensionality, non-finite values, and entities without a row
    /// (naming up to 5). Returns the dimensionality and per-entity rows.
    fn entity_features(&self, data: &PairwiseDataset) -> Result<(usize, Vec<&[f64]>)> {
        let mut by_name: HashMap<&str, &[f64]> = HashMap::with_capacity(self.features.len());
        let mut dim: Option<(usize, &str)> = None;

        for (name, row) in &self.features {
            if by_name.insert(name.as_str(), row.as_slice()).is_some() {
                return Err(Error::InvalidInput(format!(
                    "duplicate feature row for {name:?}"
                )));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(Error::InvalidInput(format!(
                    "non-finite feature value for {name:?}"
                )));
            }
            match dim {
                None => dim = Some((row.len(), name)),
                Some((d, first)) if row.len() != d => {
                    return Err(Error::InvalidInput(format!(
                        "feature dimension mismatch: {name:?} has {}, {first:?} has {d}",
                        row.len()
                    )));
                }
                Some(_) => {}
            }
        }

        let mut rows = Vec::with_capacity(data.n_entities());
        let mut missing: Vec<&str> = Vec::new();

        for name in data.interner().names() {
            match by_name.get(name) {
                Some(&row) => rows.push(row),
                None => missing.push(name),
            }
        }

        if !missing.is_empty() {
            let shown = missing
                .iter()
                .take(5)
                .map(|n| format!("{n:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            let extra = missing.len().saturating_sub(5);
            let suffix = if extra > 0 {
                format!(" (+{extra} more)")
            } else {
                String::new()
            };
            return Err(Error::InvalidInput(format!(
                "entities missing feature rows: {shown}{suffix}"
            )));
        }

        match dim {
            Some((d, _)) if d > 0 => Ok((d, rows)),
            _ => Err(Error::InvalidInput(
                "feature vectors must have at least one dimension".into(),
            )),
        }
    }
}

impl Ranker for CovariateBt {
    type Data = PairwiseDataset;
    type Model = CovariateBtModel;

    fn fit_opts(&self, data: &PairwiseDataset, opts: &FitOptions<'_>) -> Result<CovariateBtModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        self.validate()?;
        let (d, x) = self.entity_features(data)?;

        let mut agg: HashMap<(u32, u32), f64> = HashMap::new();
        for (w, l, weight) in data.rows() {
            *agg.entry((w, l)).or_default() += f64::from(weight);
        }
        let mut pairs: Vec<(u32, u32, f64)> =
            agg.into_iter().map(|((w, l), n)| (w, l, n)).collect();
        pairs.sort_unstable_by_key(|&(w, l, _)| (w, l));

        let problem = Problem {
            pairs,
            x,
            d,
            l2: self.l2,
            intercepts: self.intercepts,
        };

        let p = problem.n_params();
        let mut theta = vec![0.0f64; p];
        let mut scores = problem.entity_scores(&theta);
        let mut obj = problem.objective(&theta, &scores);

        let progress = opts.progress;
        progress.start("newton steps", Some(self.iterations as u64));

        for it in 0..self.iterations {
            let (grad, mut a) = problem.newton_system(&theta, &scores);
            let mut delta = grad;
            cholesky_solve(&mut a, &mut delta, p)?;

            // Backtracking: halve the step until the penalized objective
            // stops decreasing; ascent is guaranteed because A is SPD.
            let mut accepted = None;
            let mut step = 1.0f64;
            for _ in 0..40 {
                let cand: Vec<f64> = theta
                    .iter()
                    .zip(&delta)
                    .map(|(t, dl)| t + step * dl)
                    .collect();
                let cand_scores = problem.entity_scores(&cand);
                let cand_obj = problem.objective(&cand, &cand_scores);
                if cand_obj >= obj {
                    accepted = Some((cand, cand_scores, cand_obj));
                    break;
                }
                step *= 0.5;
            }

            // No ascent step left: already at the optimum within fp noise.
            let Some((new_theta, new_scores, new_obj)) = accepted else {
                break;
            };
            let moved = theta
                .iter()
                .zip(&new_theta)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            (theta, scores, obj) = (new_theta, new_scores, new_obj);

            progress.update(it as u64 + 1);
            if it % 10 == 0 {
                progress.message(&format!("objective {obj:0.6e}"));
            }

            if moved < self.tolerance {
                break;
            }
        }
        progress.finish();

        let beta = theta[..d].to_vec();
        let b = if self.intercepts {
            InterceptState::Fitted(theta[d..].to_vec())
        } else {
            InterceptState::Disabled
        };

        Ok(CovariateBtModel {
            params: self.clone(),
            names: data.interner().clone(),
            beta,
            s: scores,
            b,
        })
    }
}

/// The penalized conditional-logit problem: aggregated ordered pairs plus
/// the per-entity feature rows, with θ laid out as `(β[0..d], b[0..n]?)`.
struct Problem<'a> {
    /// `(winner, loser, total weight)`, sorted for deterministic assembly.
    pairs: Vec<(u32, u32, f64)>,
    /// Feature row per dense entity id.
    x: Vec<&'a [f64]>,
    d: usize,
    l2: f64,
    intercepts: bool,
}

impl Problem<'_> {
    fn n_params(&self) -> usize {
        self.d + if self.intercepts { self.x.len() } else { 0 }
    }

    /// `s_i = β·x_i (+ b_i)` for every entity.
    fn entity_scores(&self, theta: &[f64]) -> Vec<f64> {
        self.x
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                let mut s = dot(&theta[..self.d], xi);
                if self.intercepts {
                    s += theta[self.d + i];
                }
                s
            })
            .collect()
    }

    /// Penalized log-likelihood at θ (`scores` = [`Problem::entity_scores`]):
    /// `Σ w_ij · ln σ(s_i − s_j) − (l2/2)‖θ‖²`.
    fn objective(&self, theta: &[f64], scores: &[f64]) -> f64 {
        let ll: f64 = self
            .pairs
            .iter()
            .map(|&(w, l, n)| n * ln_sigmoid(scores[w as usize] - scores[l as usize]))
            .sum();
        ll - 0.5 * self.l2 * dot(theta, theta)
    }

    /// Gradient of the penalized log-likelihood and the dense SPD matrix
    /// `A = −H` (row-major), assembled per aggregated pair from its sparse
    /// direction `z = ∂(s_i − s_j)/∂θ = (x_i − x_j, e_i − e_j)`.
    fn newton_system(&self, theta: &[f64], scores: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let p = self.n_params();
        let mut grad = vec![0.0f64; p];
        let mut a = vec![0.0f64; p * p];
        let mut z: Vec<(usize, f64)> = Vec::with_capacity(self.d + 2);

        for &(w, l, n) in &self.pairs {
            let (wi, li) = (w as usize, l as usize);
            let sig = sigmoid(scores[wi] - scores[li]);
            let resid = n * (1.0 - sig);
            let curv = n * sig * (1.0 - sig);

            z.clear();
            for (k, (xw, xl)) in self.x[wi].iter().zip(self.x[li]).enumerate() {
                let v = xw - xl;
                if v != 0.0 {
                    z.push((k, v));
                }
            }
            if self.intercepts {
                z.push((self.d + wi, 1.0));
                z.push((self.d + li, -1.0));
            }

            for &(row, v) in &z {
                grad[row] += resid * v;
                for &(col, u) in &z {
                    a[row * p + col] += curv * v * u;
                }
            }
        }

        for (k, t) in theta.iter().enumerate() {
            grad[k] -= self.l2 * t;
            a[k * p + k] += self.l2;
        }

        (grad, a)
    }
}

/// In-place Cholesky solve of `A·x = b` for a dense, row-major, symmetric
/// positive-definite `A` (only the lower triangle is read): `A` is replaced
/// by its factor `L`, `b` by the solution via the two triangular solves.
fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> Result<()> {
    for j in 0..n {
        let mut pivot = a[j * n + j];
        for k in 0..j {
            pivot -= a[j * n + k] * a[j * n + k];
        }
        if !pivot.is_finite() || pivot <= 0.0 {
            return Err(Error::Numeric(format!(
                "covariate-bt Newton system is not positive definite \
                 (pivot {pivot:e} at row {j}); increase l2"
            )));
        }
        let ljj = pivot.sqrt();
        a[j * n + j] = ljj;

        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / ljj;
        }
    }

    for i in 0..n {
        let mut v = b[i];
        for k in 0..i {
            v -= a[i * n + k] * b[k];
        }
        b[i] = v / a[i * n + i];
    }

    for i in (0..n).rev() {
        let mut v = b[i];
        for k in i + 1..n {
            v -= a[k * n + i] * b[k];
        }
        b[i] = v / a[i * n + i];
    }

    Ok(())
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln σ(x)`, computed without overflow for large |x|.
#[inline]
fn ln_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Everything the state file's `params` header carries: the config fields
/// plus the fitted coefficient vector.
#[derive(Serialize, Deserialize)]
struct PersistedParams {
    features: Vec<(String, Vec<f64>)>,
    l2: f64,
    intercepts: bool,
    iterations: usize,
    tolerance: f64,
    beta: Vec<f64>,
}

/// One entity line: fitted score, plus its intercept when enabled.
#[derive(Debug, Serialize, Deserialize)]
struct EntityLine {
    id: String,
    s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<f64>,
}

/// Whether per-entity intercepts were part of the fit.
#[derive(Debug, Clone)]
enum InterceptState {
    Disabled,
    /// One intercept per entity, in dense id order.
    Fitted(Vec<f64>),
}

/// Fitted conditional-logit model: coefficients β plus per-entity scores
/// (and intercepts when enabled).
#[derive(Debug, Clone)]
pub struct CovariateBtModel {
    params: CovariateBt,
    names: Interner,
    beta: Vec<f64>,
    /// `s_i = β·x_i (+ b_i)` per entity.
    s: Vec<f64>,
    b: InterceptState,
}

impl CovariateBtModel {
    /// The fitted feature coefficients β.
    pub fn coefficients(&self) -> &[f64] {
        &self.beta
    }

    /// Cold-start score `β·x` for an arbitrary (possibly unseen) feature
    /// vector. The vector must have the model's feature dimensionality;
    /// no intercept is added — an unseen entity's `b` is its prior mean, 0.
    pub fn score(&self, features: &[f64]) -> Result<f64> {
        if features.len() != self.beta.len() {
            return Err(Error::InvalidInput(format!(
                "feature vector has dimension {}, model expects {}",
                features.len(),
                self.beta.len()
            )));
        }
        Ok(dot(&self.beta, features))
    }
}

impl RankModel for CovariateBtModel {
    fn algorithm(&self) -> &'static str {
        "covariate-bt"
    }

    /// Fitted strengths `s_i` (log scale; differences are log odds).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.s.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let persisted = PersistedParams {
            features: self.params.features.clone(),
            l2: self.params.l2,
            intercepts: self.params.intercepts,
            iterations: self.params.iterations,
            tolerance: self.params.tolerance,
            beta: self.beta.clone(),
        };
        let lines: Vec<EntityLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| EntityLine {
                id: id.to_string(),
                s: self.s[i],
                b: match &self.b {
                    InterceptState::Disabled => None,
                    InterceptState::Fitted(v) => Some(v[i]),
                },
            })
            .collect();
        state::save_model(w, "covariate-bt", &persisted, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (pp, lines): (PersistedParams, Vec<EntityLine>) = state::load_model(r, "covariate-bt")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        let s: Vec<f64> = lines.iter().map(|l| l.s).collect();

        let b = if pp.intercepts {
            let mut v = Vec::with_capacity(lines.len());
            for line in &lines {
                match line.b {
                    Some(b) => v.push(b),
                    None => {
                        return Err(Error::State(format!(
                            "entity {:?} is missing its intercept",
                            line.id
                        )));
                    }
                }
            }
            InterceptState::Fitted(v)
        } else {
            InterceptState::Disabled
        };

        Ok(Self {
            params: CovariateBt {
                features: pp.features,
                l2: pp.l2,
                intercepts: pp.intercepts,
                iterations: pp.iterations,
                tolerance: pp.tolerance,
            },
            names,
            beta: pp.beta,
            s,
            b,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::BradleyTerryMM;

    fn feats(rows: &[(&str, &[f64])]) -> Vec<(String, Vec<f64>)> {
        rows.iter()
            .map(|&(n, x)| (n.to_string(), x.to_vec()))
            .collect()
    }

    /// Exact-rate data generated from s_i = 2·x_i recovers β = 2.
    #[test]
    fn recovers_known_coefficient_from_exact_rates() {
        let xs = [("a", 0.0), ("b", 0.5), ("c", 1.0)];
        let mut d = PairwiseDataset::new();
        for (i, &(ni, xi)) in xs.iter().enumerate() {
            for &(nj, xj) in &xs[i + 1..] {
                let p = sigmoid(2.0 * (xi - xj));
                d.push(ni, nj, (100.0 * p) as f32);
                d.push(nj, ni, (100.0 * (1.0 - p)) as f32);
            }
        }

        let algo = CovariateBt {
            l2: 1e-9,
            ..CovariateBt::new(xs.iter().map(|&(n, x)| (n.to_string(), vec![x])).collect())
        };
        let m = algo.fit(&d).unwrap();
        let beta = m.coefficients()[0];
        assert!((beta - 2.0).abs() < 1e-2, "beta = {beta}");
    }

    /// With all-zero features and tiny ridge, the intercepts are plain BT
    /// log-strengths: their gaps match Bradley-Terry MM's log ratios.
    #[test]
    fn intercepts_recover_plain_bt_log_strengths() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 8.0);
        d.push("b", "a", 2.0);
        d.push("b", "c", 8.0);
        d.push("c", "b", 2.0);
        d.push("a", "c", 9.0);
        d.push("c", "a", 1.0);

        let zero: &[f64] = &[0.0];
        let algo = CovariateBt {
            l2: 1e-8,
            intercepts: true,
            ..CovariateBt::new(feats(&[("a", zero), ("b", zero), ("c", zero)]))
        };
        let m = algo.fit(&d).unwrap();
        let s: HashMap<&str, f64> = m.scores().collect();

        let bt = BradleyTerryMM::default().fit(&d).unwrap();
        let pi: HashMap<&str, f64> = bt.scores().collect();

        for (i, j) in [("a", "b"), ("b", "c"), ("a", "c")] {
            let gap = s[i] - s[j];
            let bt_gap = (pi[i] / pi[j]).ln();
            assert!((gap - bt_gap).abs() < 1e-3, "{i} vs {j}: {gap} vs {bt_gap}");
        }
    }

    /// `score()` evaluates β·x for unseen feature vectors, and rejects
    /// dimension mismatches.
    #[test]
    fn cold_start_scores_unseen_feature_vectors() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 6.0);
        d.push("b", "a", 2.0);
        d.push("b", "c", 5.0);
        d.push("c", "b", 3.0);

        let algo = CovariateBt::new(feats(&[
            ("a", &[1.0, -0.5]),
            ("b", &[0.0, 0.5]),
            ("c", &[-1.0, 1.0]),
        ]));
        let m = algo.fit(&d).unwrap();

        let unseen = [0.25, -1.5];
        let want = m.coefficients()[0] * unseen[0] + m.coefficients()[1] * unseen[1];
        assert!((m.score(&unseen).unwrap() - want).abs() < 1e-12);
        assert!(matches!(m.score(&[1.0]), Err(Error::InvalidInput(_))));
    }

    /// Missing feature rows, dimension mismatches, and an unpenalized
    /// intercept fit are typed errors.
    #[test]
    fn invalid_feature_tables_are_typed_errors() {
        let mut d = PairwiseDataset::new();
        d.push("a", "ghost", 1.0);

        let one: &[f64] = &[1.0];
        let missing = CovariateBt::new(feats(&[("a", one)]));
        let err = missing.fit(&d).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidInput(m) if m.contains("ghost")),
            "{err}"
        );

        let mismatched = CovariateBt::new(feats(&[("a", one), ("ghost", &[1.0, 2.0])]));
        assert!(matches!(mismatched.fit(&d), Err(Error::InvalidInput(_))));

        let unpenalized = CovariateBt {
            intercepts: true,
            l2: 0.0,
            ..CovariateBt::new(feats(&[("a", one), ("ghost", one)]))
        };
        assert!(matches!(unpenalized.fit(&d), Err(Error::InvalidInput(_))));
    }

    /// The fitted θ zeroes the penalized gradient, recomputed analytically
    /// in the test from the model output alone.
    #[test]
    fn newton_zeroes_the_penalized_gradient() {
        let names = ["a", "b", "c"];
        let x: HashMap<&str, [f64; 2]> =
            HashMap::from([("a", [1.0, 0.2]), ("b", [-0.5, 1.0]), ("c", [0.3, -0.7])]);
        let rows: [(&str, &str, f64); 6] = [
            ("a", "b", 7.0),
            ("b", "a", 3.0),
            ("b", "c", 6.0),
            ("c", "b", 4.0),
            ("a", "c", 8.0),
            ("c", "a", 2.0),
        ];

        let mut d = PairwiseDataset::new();
        for &(w, l, n) in &rows {
            d.push(w, l, n as f32);
        }
        let algo = CovariateBt {
            l2: 0.1,
            intercepts: true,
            tolerance: 1e-12,
            ..CovariateBt::new(
                names
                    .iter()
                    .map(|&n| (n.to_string(), x[n].to_vec()))
                    .collect(),
            )
        };
        let m = algo.fit(&d).unwrap();

        let beta = m.coefficients();
        let s: HashMap<&str, f64> = m.scores().collect();
        let b: HashMap<&str, f64> = names
            .iter()
            .map(|&n| (n, s[n] - beta[0] * x[n][0] - beta[1] * x[n][1]))
            .collect();

        let mut g_beta = [-algo.l2 * beta[0], -algo.l2 * beta[1]];
        let mut g_b: HashMap<&str, f64> = names.iter().map(|&n| (n, -algo.l2 * b[n])).collect();
        for &(w, l, n) in &rows {
            let resid = n * (1.0 - sigmoid(s[w] - s[l]));
            g_beta[0] += resid * (x[w][0] - x[l][0]);
            g_beta[1] += resid * (x[w][1] - x[l][1]);
            *g_b.get_mut(w).unwrap() += resid;
            *g_b.get_mut(l).unwrap() -= resid;
        }

        for g in g_beta.iter().chain(g_b.values()) {
            assert!(g.abs() < 1e-6, "gradient component {g:e}");
        }
    }

    /// Round trips are byte-identical (with and without intercepts) and an
    /// empty dataset is a typed error.
    #[test]
    fn round_trip_and_empty_dataset() {
        let mut d = PairwiseDataset::new();
        d.push("a", "b", 3.0);
        d.push("b", "a", 1.0);

        let table = feats(&[("a", &[1.0]), ("b", &[-1.0])]);
        for intercepts in [false, true] {
            let algo = CovariateBt {
                intercepts,
                ..CovariateBt::new(table.clone())
            };
            let m = algo.fit(&d).unwrap();
            let mut first = Vec::new();
            m.save_jsonl(&mut first).unwrap();
            let loaded = CovariateBtModel::load_jsonl(first.as_slice()).unwrap();
            let mut second = Vec::new();
            loaded.save_jsonl(&mut second).unwrap();
            assert_eq!(first, second, "intercepts = {intercepts}");
        }

        let empty = PairwiseDataset::new();
        assert!(matches!(
            CovariateBt::new(table).fit(&empty),
            Err(Error::EmptyDataset)
        ));
    }
}

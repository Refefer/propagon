//! LinUCB — linear contextual bandit with disjoint per-arm models
//! (`docs/algorithms.md` §8.1; Li, Chu, Langford & Schapire, WWW 2010,
//! Algorithm 1; the hybrid shared-feature variant is out of scope).
//!
//! Each arm owns a ridge regression `θ_a = A_a⁻¹ b_a` with
//! `A_a = ridge·I + Σ x xᵀ` over the contexts it was played in;
//! [`LinUcbModel::select_for`] plays
//! `argmax_a θ_aᵀx + alpha·sqrt(xᵀ A_a⁻¹ x)`. The inverse is maintained
//! directly via Sherman-Morrison (O(d²) per update, no solves), and
//! re-symmetrized after every update to stop round-off drift.
//!
//! Assumes the feature dimension is constant: the first update fixes the
//! model's `dim`, and later datasets (or query contexts) of another length
//! are a [`ParamMismatch`](crate::Error::ParamMismatch).
//!
//! Gotchas:
//! - **Off-policy replay**: `update` folds a logged `(arm, reward, context)`
//!   row into the logged arm only — there is no propensity correction, so a
//!   log collected under a different policy yields the usual biased
//!   off-policy estimates.
//! - **`scores()` needs a context to be meaningful**: without one there is
//!   no canonical per-arm score, so it reports `θ_aᵀ x̄` at the mean observed
//!   context (the model tracks `Σx` / count). Use
//!   [`LinUcbModel::select_for`] for actual decisions.

use serde::{Deserialize, Serialize};

use crate::dataset::ContextualRewardsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// LinUCB parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinUcb {
    /// Width of the confidence bonus `alpha·sqrt(xᵀ A⁻¹ x)`.
    pub alpha: f64,
    /// Ridge regularizer: fresh arms start from `A = ridge·I`. Must be
    /// positive and finite.
    pub ridge: f64,
}

impl LinUcb {
    /// The paper's baseline confidence width.
    pub const DEFAULT_ALPHA: f64 = 1.0;
    /// Unit ridge (the paper's `A_a = I_d` initialization).
    pub const DEFAULT_RIDGE: f64 = 1.0;
}

impl Default for LinUcb {
    fn default() -> Self {
        Self {
            alpha: Self::DEFAULT_ALPHA,
            ridge: Self::DEFAULT_RIDGE,
        }
    }
}

/// One arm's ridge-regression state.
#[derive(Debug, Clone)]
struct ArmState {
    /// `A⁻¹`, d×d row-major; symmetric by re-symmetrization after updates.
    ainv: Vec<f64>,
    /// `b = Σ r·x`.
    b: Vec<f64>,
    /// Times this arm was played.
    n: u64,
}

impl ArmState {
    fn fresh(dim: usize, ridge: f64) -> Self {
        let mut ainv = vec![0.0; dim * dim];
        for i in 0..dim {
            ainv[i * dim + i] = 1.0 / ridge;
        }
        Self {
            ainv,
            b: vec![0.0; dim],
            n: 0,
        }
    }

    /// Folds one `(x, reward)` observation in: Sherman-Morrison for
    /// `A ← A + xxᵀ` on the inverse —
    /// `A⁻¹ ← A⁻¹ − (A⁻¹x)(A⁻¹x)ᵀ / (1 + xᵀA⁻¹x)` — followed by
    /// `A⁻¹ ← (A⁻¹ + A⁻ᵀ)/2` so floating-point drift cannot accumulate
    /// asymmetry, then `b += r·x`.
    fn observe(&mut self, x: &[f64], reward: f64) {
        let d = x.len();
        let v = mat_vec(&self.ainv, x, d);
        // A⁻¹ is positive definite, so the denominator is > 1.
        let denom = 1.0 + dot(x, &v);
        for i in 0..d {
            for j in 0..d {
                self.ainv[i * d + j] -= v[i] * v[j] / denom;
            }
        }

        for i in 0..d {
            for j in (i + 1)..d {
                let m = 0.5 * (self.ainv[i * d + j] + self.ainv[j * d + i]);
                self.ainv[i * d + j] = m;
                self.ainv[j * d + i] = m;
            }
        }

        for (bi, &xi) in self.b.iter_mut().zip(x) {
            *bi += reward * xi;
        }
        self.n += 1;
    }

    /// `θ = A⁻¹ b`, computed on demand.
    fn theta(&self, d: usize) -> Vec<f64> {
        mat_vec(&self.ainv, &self.b, d)
    }

    /// The LinUCB index `θᵀx + alpha·sqrt(xᵀ A⁻¹ x)`; the quadratic form is
    /// clamped at 0 against tiny negative round-off.
    fn index(&self, x: &[f64], alpha: f64) -> f64 {
        let d = x.len();
        let v = mat_vec(&self.ainv, x, d);
        dot(&self.theta(d), x) + alpha * dot(x, &v).max(0.0).sqrt()
    }
}

/// Row-major d×d matrix times vector.
fn mat_vec(m: &[f64], x: &[f64], d: usize) -> Vec<f64> {
    (0..d).map(|i| dot(&m[i * d..(i + 1) * d], x)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// What `save_jsonl` writes as the header `params`: algorithm params, the
/// fixed dimension, and the running context mean's sufficient statistics.
#[derive(Serialize, Deserialize)]
struct PersistedParams {
    alpha: f64,
    ridge: f64,
    dim: Option<usize>,
    ctx_sum: Vec<f64>,
    ctx_n: u64,
}

/// One arm's state line.
#[derive(Serialize, Deserialize)]
struct ArmLine {
    id: String,
    n: u64,
    ainv: Vec<f64>,
    b: Vec<f64>,
}

/// LinUCB state: per-arm ridge regressions plus the running context mean.
#[derive(Debug, Clone)]
pub struct LinUcbModel {
    params: LinUcb,
    names: Interner,
    /// Feature dimension; `None` until the first update fixes it.
    dim: Option<usize>,
    arms: Vec<ArmState>,
    /// Running context sum (for the `scores()` mean context).
    ctx_sum: Vec<f64>,
    ctx_n: u64,
}

impl LinUcbModel {
    /// Number of distinct arms the model has seen.
    pub fn n_arms(&self) -> usize {
        self.names.len()
    }

    /// Feature dimension, fixed by the first update (`None` before then).
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Picks the arm with the highest LinUCB index for context `x`:
    /// `argmax_a θ_aᵀx + alpha·sqrt(xᵀ A_a⁻¹ x)`. Deterministic — no RNG,
    /// ties go to the lexicographically smaller name.
    pub fn select_for(&self, x: &[f64]) -> Result<&str> {
        if self.names.is_empty() {
            return Err(Error::InvalidInput("model has no arms".into()));
        }
        let Some(d) = self.dim else {
            return Err(Error::InvalidInput("model has no observations".into()));
        };
        if x.len() != d {
            return Err(Error::ParamMismatch(format!(
                "context has dim {} but the model was fitted with dim {d}",
                x.len()
            )));
        }
        if let Some(bad) = x.iter().find(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(format!(
                "context contains a non-finite value: {bad}"
            )));
        }

        let scores: Vec<f64> = self
            .arms
            .iter()
            .map(|a| a.index(x, self.params.alpha))
            .collect();
        let mut best = 0u32;

        for i in 1..scores.len() as u32 {
            let cmp = scores[i as usize]
                .total_cmp(&scores[best as usize])
                .then_with(|| self.names.resolve(best).cmp(self.names.resolve(i)));
            if cmp == std::cmp::Ordering::Greater {
                best = i;
            }
        }
        Ok(self.names.resolve(best))
    }

    /// Mismatch-checks an incoming row dimension, fixing the model's `dim`
    /// (and sizing `ctx_sum`) on first contact.
    fn ensure_dim(&mut self, d: usize) -> Result<()> {
        match self.dim {
            None => {
                self.dim = Some(d);
                self.ctx_sum = vec![0.0; d];
                Ok(())
            }
            Some(md) if md == d => Ok(()),
            Some(md) => Err(Error::ParamMismatch(format!(
                "dataset rows have dim {d} but the model was fitted with dim {md}"
            ))),
        }
    }

    fn intern_arm(&mut self, name: &str, dim: usize, ridge: f64) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.arms.len() {
            self.arms.push(ArmState::fresh(dim, ridge));
        }
        idx
    }
}

impl RankModel for LinUcbModel {
    fn algorithm(&self) -> &'static str {
        "lin-ucb"
    }

    /// `θ_aᵀ x̄` at the mean observed context — without a query context
    /// there is no canonical per-arm score (see the module docs).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        let vals: Vec<f64> = match self.dim {
            Some(d) => {
                // `dim` is only ever set alongside a first observation, so
                // `ctx_n >= 1` here (load_jsonl re-validates this).
                let xbar: Vec<f64> = self.ctx_sum.iter().map(|s| s / self.ctx_n as f64).collect();
                self.arms.iter().map(|a| dot(&a.theta(d), &xbar)).collect()
            }
            None => Vec::new(),
        };
        self.names.names().zip(vals)
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            alpha: self.params.alpha,
            ridge: self.params.ridge,
            dim: self.dim,
            ctx_sum: self.ctx_sum.clone(),
            ctx_n: self.ctx_n,
        };
        let lines: Vec<ArmLine> = self
            .names
            .names()
            .zip(&self.arms)
            .map(|(id, a)| ArmLine {
                id: id.to_string(),
                n: a.n,
                ainv: a.ainv.clone(),
                b: a.b.clone(),
            })
            .collect();
        state::save_model(w, "lin-ucb", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<ArmLine>) = state::load_model(r, "lin-ucb")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;

        let d = match params.dim {
            Some(d) => d,
            None if lines.is_empty() => 0,
            None => {
                return Err(Error::State(
                    "arm lines present but params declare no dim".into(),
                ));
            }
        };
        if params.dim.is_some() && params.ctx_sum.len() != d {
            return Err(Error::State(format!(
                "ctx_sum holds {} values but dim is {d}",
                params.ctx_sum.len()
            )));
        }
        if params.dim.is_some() && params.ctx_n == 0 {
            return Err(Error::State(
                "params declare a dim but no observed contexts".into(),
            ));
        }

        let mut arms = Vec::with_capacity(lines.len());
        for line in &lines {
            if line.ainv.len() != d * d || line.b.len() != d {
                return Err(Error::State(format!(
                    "arm {:?} has {} ainv / {} b values but dim is {d}",
                    line.id,
                    line.ainv.len(),
                    line.b.len()
                )));
            }
            arms.push(ArmState {
                ainv: line.ainv.clone(),
                b: line.b.clone(),
                n: line.n,
            });
        }

        Ok(Self {
            params: LinUcb {
                alpha: params.alpha,
                ridge: params.ridge,
            },
            names,
            dim: params.dim,
            arms,
            ctx_sum: params.ctx_sum,
            ctx_n: params.ctx_n,
        })
    }
}

impl OnlineRanker for LinUcb {
    type Data = ContextualRewardsDataset;
    type Model = LinUcbModel;

    fn init(&self) -> LinUcbModel {
        LinUcbModel {
            params: *self,
            names: Interner::new(),
            dim: None,
            arms: Vec::new(),
            ctx_sum: Vec::new(),
            ctx_n: 0,
        }
    }

    fn update_opts(
        &self,
        model: &mut LinUcbModel,
        data: &ContextualRewardsDataset,
        _opts: &FitOptions<'_>,
    ) -> Result<()> {
        if !(self.ridge.is_finite() && self.ridge > 0.0) {
            return Err(Error::InvalidInput(format!(
                "ridge must be positive and finite, got {}",
                self.ridge
            )));
        }

        for (arm, reward, x) in data.rows() {
            model.ensure_dim(x.len())?;
            let name = data.interner().resolve(arm);
            let idx = model.intern_arm(name, x.len(), self.ridge);
            model.arms[idx].observe(x, f64::from(reward));

            for (s, &xi) in model.ctx_sum.iter_mut().zip(x) {
                *s += xi;
            }
            model.ctx_n += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds(rows: &[(&str, f32, &[f64])]) -> ContextualRewardsDataset {
        let mut d = ContextualRewardsDataset::new();
        for (a, r, x) in rows {
            d.push(a, *r, x).unwrap();
        }
        d
    }

    /// d = 1, ridge = 1, one arm, rows (x, r) = (1, 1), (2, 0.5), (1, 0):
    ///   A = 1 + 1² + 2² + 1² = 7 → A⁻¹ = 1/7;  b = Σxr = 1 + 1 + 0 = 2;
    ///   θ = b/A = 2/7.
    /// `scores()` evaluates θ at x̄ = (1+2+1)/3 = 4/3 → 2/7 · 4/3 = 8/21.
    #[test]
    fn d1_matches_closed_form_ridge_regression() {
        let algo = LinUcb {
            alpha: 0.5,
            ridge: 1.0,
        };
        let mut m = algo.init();
        algo.update(
            &mut m,
            &ds(&[("a", 1.0, &[1.0]), ("a", 0.5, &[2.0]), ("a", 0.0, &[1.0])]),
        )
        .unwrap();

        assert_eq!(m.dim(), Some(1));
        assert!((m.arms[0].ainv[0] - 1.0 / 7.0).abs() < 1e-12);
        assert!((m.arms[0].b[0] - 2.0).abs() < 1e-12);
        assert!((m.arms[0].theta(1)[0] - 2.0 / 7.0).abs() < 1e-12);

        let s: std::collections::HashMap<_, _> =
            m.scores().map(|(n, v)| (n.to_string(), v)).collect();
        assert!((s["a"] - 8.0 / 21.0).abs() < 1e-12, "{}", s["a"]);
    }

    /// d = 2, ridge = 1, one observation x = (1, 2), r = 1.
    /// A = I + xxᵀ = [[2, 2], [2, 5]], det = 10 − 4 = 6, so by the 2×2
    /// inverse formula A⁻¹ = [[5, −2], [−2, 2]]/6 = [[5/6, −1/3],
    /// [−1/3, 1/3]]. Sherman-Morrison reaches the same place:
    /// v = A₀⁻¹x = x, denom = 1 + xᵀx = 6, A⁻¹ = I − xxᵀ/6.
    /// θ = A⁻¹b with b = x = (1, 2): θ = (5/6 − 2/3, −1/3 + 2/3) = (1/6, 1/3).
    #[test]
    fn d2_single_sherman_morrison_step() {
        let algo = LinUcb::default();
        let mut m = algo.init();
        algo.update(&mut m, &ds(&[("a", 1.0, &[1.0, 2.0])]))
            .unwrap();

        let expected = [5.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0];
        for (got, want) in m.arms[0].ainv.iter().zip(expected) {
            assert!((got - want).abs() < 1e-15, "{got} vs {want}");
        }
        let theta = m.arms[0].theta(2);
        assert!((theta[0] - 1.0 / 6.0).abs() < 1e-15);
        assert!((theta[1] - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(m.arms[0].n, 1);
    }

    /// d = 1, ridge = 1, query x = 1.
    /// Arm a: one row (x=1, r=0.2) → A=2, θ=0.1, bonus = α·sqrt(1/2).
    /// Arm b: three rows (x=1, r=0.4) → A=4, θ=1.2/4=0.3, bonus = α·sqrt(1/4).
    /// Indices cross at α = 0.2/(sqrt(0.5) − 0.5) ≈ 0.966: a wins for α = 2
    /// (exploration dominates), b wins for α = 0.5 (exploitation dominates).
    #[test]
    fn select_for_arithmetic_pinned() {
        let data = ds(&[
            ("a", 0.2, &[1.0]),
            ("b", 0.4, &[1.0]),
            ("b", 0.4, &[1.0]),
            ("b", 0.4, &[1.0]),
        ]);

        let explore = LinUcb {
            alpha: 2.0,
            ridge: 1.0,
        };
        let mut m = explore.init();
        explore.update(&mut m, &data).unwrap();
        assert_eq!(m.select_for(&[1.0]).unwrap(), "a");

        let exploit = LinUcb {
            alpha: 0.5,
            ridge: 1.0,
        };
        let mut m = exploit.init();
        exploit.update(&mut m, &data).unwrap();
        assert_eq!(m.select_for(&[1.0]).unwrap(), "b");

        // Identical arm states tie; the lexicographically smaller name wins.
        let algo = LinUcb::default();
        let mut m = algo.init();
        algo.update(&mut m, &ds(&[("d", 0.5, &[1.0]), ("c", 0.5, &[1.0])]))
            .unwrap();
        assert_eq!(m.select_for(&[1.0]).unwrap(), "c");
    }

    #[test]
    fn ainv_stays_symmetric_after_many_updates() {
        let algo = LinUcb::default();
        let mut m = algo.init();

        let mut d = ContextualRewardsDataset::new();
        for k in 0..300u32 {
            // Deterministic, well-spread contexts (no RNG needed).
            let t = f64::from(k);
            let x = [(t * 0.7).sin(), (t * 1.3).cos(), 0.01 * t - 1.5];
            let arm = ["a", "b", "c"][(k % 3) as usize];
            d.push(arm, (t * 0.11).sin() as f32, &x).unwrap();
        }
        algo.update(&mut m, &d).unwrap();

        for arm in &m.arms {
            let mut worst = 0.0f64;
            for i in 0..3 {
                for j in 0..3 {
                    worst = worst.max((arm.ainv[i * 3 + j] - arm.ainv[j * 3 + i]).abs());
                }
            }
            assert!(worst < 1e-12, "asymmetry {worst}");
        }
    }

    #[test]
    fn round_trip_is_byte_identical_and_errors_are_typed() {
        let algo = LinUcb {
            alpha: 1.5,
            ridge: 2.0,
        };
        let mut m = algo.init();
        algo.update(
            &mut m,
            &ds(&[
                ("a", 1.0, &[0.5, -1.0]),
                ("b", 0.0, &[1.0, 0.25]),
                ("a", 0.5, &[-0.75, 2.0]),
            ]),
        )
        .unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = LinUcbModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second, "save -> load -> save is byte-identical");

        let q = [0.3, 0.7];
        assert_eq!(m.select_for(&q).unwrap(), loaded.select_for(&q).unwrap());
        assert_eq!(
            m.scores().collect::<Vec<_>>(),
            loaded.scores().collect::<Vec<_>>()
        );

        // dim mismatch between model and incoming data / query context
        assert!(matches!(
            algo.update(&mut m, &ds(&[("a", 1.0, &[1.0])])),
            Err(Error::ParamMismatch(_))
        ));
        assert!(matches!(m.select_for(&[1.0]), Err(Error::ParamMismatch(_))));
        assert!(matches!(
            m.select_for(&[f64::NAN, 0.0]),
            Err(Error::InvalidInput(_))
        ));

        // invalid configs and empty models
        let bad = LinUcb {
            alpha: 1.0,
            ridge: 0.0,
        };
        let mut fresh = bad.init();
        assert!(matches!(
            bad.update(&mut fresh, &ds(&[("a", 1.0, &[1.0])])),
            Err(Error::InvalidInput(_))
        ));
        let empty = LinUcb::default().init();
        assert!(matches!(
            empty.select_for(&[1.0]),
            Err(Error::InvalidInput(_))
        ));
    }
}

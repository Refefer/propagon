//! Logarithmic and linear opinion pools (`docs/algorithms.md` §14.2).
//!
//! Consolidate several sources' probability vectors over a common outcome space
//! into one consensus — the probability-space analogue of rank aggregation. The
//! **linear** pool is the weighted arithmetic mean `Σᵢ wᵢ pᵢⱼ`; the
//! **logarithmic** pool is the weighted geometric mean `∝ Πᵢ pᵢⱼ^{wᵢ}` (the
//! arithmetic mean of log-odds), which is sharper and is the unique *externally
//! Bayesian* pool — pooling then updating equals updating then pooling. Optional
//! **extremizing** raises the consensus to a power `a > 1`, pushing it away from
//! the timid middle to counter correlated sources' under-confidence.
//!
//! References: [Genest & Zidek 1986]; [Satopää et al. 2014]; [Dietrich & List
//! 2016]. Worked values are pinned in `tests/reference.rs`.

use serde::{Deserialize, Serialize};

use crate::algos::common::impl_simple_score_model;
use crate::dataset::ForecastDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx;
use crate::traits::{FitOptions, Ranker};

/// Which mean defines the consensus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PoolKind {
    /// Weighted arithmetic mean of probabilities.
    Linear,
    /// Weighted geometric mean of probabilities (mean of log-odds).
    #[default]
    Logarithmic,
}

/// What a source not quoting an outcome means.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Missing {
    /// Require every source to quote every outcome (else error). The default —
    /// pooling partial coverage is subtle, so it must be opted into.
    #[default]
    Error,
    /// A source simply does not contribute to outcomes it omits.
    Skip,
    /// Fill an omitted outcome with the uniform probability `1/n`.
    Uniform,
}

/// Opinion-pool parameters. The struct is the algorithm; fields are params.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpinionPool {
    /// Arithmetic (linear) or geometric (logarithmic) consensus.
    pub kind: PoolKind,
    /// Extremizing exponent `a ≥ 1` (1.0 = none; ~1.5–2.5 is the empirical
    /// sweet spot but is not defaulted because it changes the result).
    pub extremize: f64,
    /// Missing-coverage policy.
    pub missing: Missing,
    /// Optional clamp `pᵢⱼ ← max(pᵢⱼ, eps)` before logs (log pool only),
    /// trading purity for avoiding zero-probability vetoes. Off by default.
    pub eps_floor: Option<f64>,
}

impl Default for OpinionPool {
    fn default() -> Self {
        Self {
            kind: PoolKind::default(),
            extremize: 1.0,
            missing: Missing::default(),
            eps_floor: None,
        }
    }
}

/// Consensus probability per outcome.
#[derive(Debug, Clone)]
pub struct OpinionPoolModel {
    params: OpinionPool,
    names: Interner,
    scores: Vec<f64>,
}

impl_simple_score_model!(OpinionPoolModel, "opinion-pool");

impl OpinionPool {
    fn coverage_err(j: usize) -> Error {
        Error::InvalidInput(format!(
            "opinion pool: a source does not quote outcome #{j}; \
             set the missing-coverage policy to skip or uniform"
        ))
    }

    /// The consensus over `n_out` outcomes from `(weight, prob-or-missing)`
    /// rows. Weights are already normalized to sum to 1.
    fn pool(&self, mat: &[(f64, Vec<Option<f64>>)], n_out: usize) -> Result<Vec<f64>> {
        let mut out = match self.kind {
            PoolKind::Linear => self.pool_linear(mat, n_out)?,
            PoolKind::Logarithmic => self.pool_log(mat, n_out)?,
        };
        self.apply_extremize(&mut out)?;
        Ok(out)
    }

    fn pool_linear(&self, mat: &[(f64, Vec<Option<f64>>)], n_out: usize) -> Result<Vec<f64>> {
        let mut out = vec![0.0; n_out];
        for (j, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (w, row) in mat {
                match row[j] {
                    Some(p) => acc += w * p,
                    None => match self.missing {
                        Missing::Error => return Err(Self::coverage_err(j)),
                        Missing::Uniform => acc += w / n_out as f64,
                        Missing::Skip => {}
                    },
                }
            }
            *slot = acc;
        }
        let s: f64 = out.iter().sum();
        if s <= 0.0 {
            return Err(Error::Numeric(
                "linear opinion pool: zero total mass".into(),
            ));
        }
        for v in &mut out {
            *v /= s;
        }
        Ok(out)
    }

    fn pool_log(&self, mat: &[(f64, Vec<Option<f64>>)], n_out: usize) -> Result<Vec<f64>> {
        // ℓⱼ = Σᵢ wᵢ ln pᵢⱼ (weights sum to 1); a zero probability vetoes the
        // outcome (ℓⱼ = −∞) unless an eps-floor is set.
        let mut logp = vec![f64::NEG_INFINITY; n_out];
        for (j, slot) in logp.iter_mut().enumerate() {
            let mut acc = 0.0;
            let mut vetoed = false;
            for (w, row) in mat {
                let p = match row[j] {
                    Some(p) => p,
                    None => match self.missing {
                        Missing::Error => return Err(Self::coverage_err(j)),
                        Missing::Uniform => 1.0 / n_out as f64,
                        Missing::Skip => continue,
                    },
                };
                let p = self.eps_floor.map_or(p, |e| p.max(e));
                if p <= 0.0 {
                    vetoed = true;
                    break;
                }
                acc += w * p.ln();
            }
            if !vetoed {
                *slot = acc;
            }
        }
        let lse = mathx::logsumexp(&logp);
        if !lse.is_finite() {
            return Err(Error::Numeric(
                "log opinion pool annihilated all outcomes (every outcome vetoed)".into(),
            ));
        }
        Ok(logp.iter().map(|&l| (l - lse).exp()).collect())
    }

    fn apply_extremize(&self, p: &mut [f64]) -> Result<()> {
        let a = self.extremize;
        if !a.is_finite() || a <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "opinion pool: extremize exponent must be positive, got {a}"
            )));
        }
        if (a - 1.0).abs() < f64::EPSILON {
            return Ok(());
        }
        for v in p.iter_mut() {
            *v = v.powf(a);
        }
        let s: f64 = p.iter().sum();
        if s <= 0.0 {
            return Err(Error::Numeric(
                "opinion pool: extremize zeroed all mass".into(),
            ));
        }
        for v in p.iter_mut() {
            *v /= s;
        }
        Ok(())
    }
}

impl Ranker for OpinionPool {
    type Data = ForecastDataset;
    type Model = OpinionPoolModel;

    fn fit_opts(&self, data: &ForecastDataset, _opts: &FitOptions<'_>) -> Result<OpinionPoolModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n_out = data.n_outcomes();

        let wsum: f64 = data.sources().map(|(w, _, _)| w).sum();
        if wsum <= 0.0 {
            return Err(Error::Numeric(
                "opinion pool: source weights sum to zero".into(),
            ));
        }

        let mat: Vec<(f64, Vec<Option<f64>>)> = data
            .sources()
            .map(|(w, ids, probs)| {
                let mut row = vec![None; n_out];
                for (&id, &p) in ids.iter().zip(probs) {
                    row[id as usize] = Some(p);
                }
                (w / wsum, row)
            })
            .collect();

        let scores = self.pool(&mat, n_out)?;
        Ok(OpinionPoolModel {
            params: *self,
            names: data.interner().clone(),
            scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RankModel;

    fn two_sources(kind: PoolKind) -> OpinionPoolModel {
        let mut d = ForecastDataset::new();
        d.push_source("s1", &[("a", 0.8), ("b", 0.2)]).unwrap();
        d.push_source("s2", &[("a", 0.6), ("b", 0.4)]).unwrap();
        OpinionPool {
            kind,
            ..Default::default()
        }
        .fit(&d)
        .unwrap()
    }

    #[test]
    fn linear_is_arithmetic_mean() {
        let m = two_sources(PoolKind::Linear);
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - 0.7).abs() < 1e-12);
        assert!((s["b"] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn log_is_normalized_geometric_mean() {
        let m = two_sources(PoolKind::Logarithmic);
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        let ga = (0.8_f64 * 0.6).sqrt();
        let gb = (0.2_f64 * 0.4).sqrt();
        assert!((s["a"] - ga / (ga + gb)).abs() < 1e-12);
        assert!((s["b"] - gb / (ga + gb)).abs() < 1e-12);
    }

    #[test]
    fn extremize_sharpens_binary() {
        let mut d = ForecastDataset::new();
        // Both sources agree at 0.7/0.3 so the pool is exactly 0.7/0.3.
        d.push_source("s1", &[("a", 0.7), ("b", 0.3)]).unwrap();
        d.push_source("s2", &[("a", 0.7), ("b", 0.3)]).unwrap();
        let m = OpinionPool {
            kind: PoolKind::Logarithmic,
            extremize: 2.0,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        let want = 0.49 / (0.49 + 0.09);
        assert!((s["a"] - want).abs() < 1e-12, "got {}", s["a"]);
        assert!(s["a"] > 0.7, "extremize pushes away from 0.5");
    }

    #[test]
    fn log_pool_zero_vetoes() {
        let mut d = ForecastDataset::new();
        d.push_source("s1", &[("a", 0.5), ("b", 0.5)]).unwrap();
        d.push_source("s2", &[("a", 0.0), ("b", 1.0)]).unwrap();
        let m = OpinionPool {
            kind: PoolKind::Logarithmic,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - 0.0).abs() < 1e-12);
        assert!((s["b"] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn round_trip() {
        let m = two_sources(PoolKind::Logarithmic);
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = OpinionPoolModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

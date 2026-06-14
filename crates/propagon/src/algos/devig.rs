//! Odds de-vigging: posted prices → fair probabilities (`docs/algorithms.md`
//! §14.1).
//!
//! Each event's decimal odds `oᵢ` give raw implied probabilities `rᵢ = 1/oᵢ`
//! whose sum (the **booksum** `B`) exceeds 1 by the bookmaker's margin. The four
//! methods differ only in how they assume that margin is spread across outcomes;
//! all return fair probabilities `πᵢ` summing to 1 per event:
//!
//! - **Multiplicative** `πᵢ = rᵢ / B` — margin proportional to probability.
//! - **Additive** `πᵢ = rᵢ − (B−1)/n` — margin split equally (can go negative).
//! - **Power** `πᵢ = rᵢ^(1/τ)`, `τ` solved so `Σπᵢ = 1` — compresses the
//!   favorite-longshot bias. (For the usual over-round `B > 1`, `τ < 1`; the
//!   survey's "τ ≥ 1" is incorrect for that regime.)
//! - **Shin** — models the margin as defense against a fraction `z` of insider
//!   money (Jullien & Salanié 1994 closed form); recovers `z` as a diagnostic.
//!
//! The fair probability is a Bradley-Terry / Plackett-Luce strength on the
//! `log π` scale, so a de-vigged event drops straight onto a leaderboard.
//!
//! References: [Shin 1992; 1993]; [Jullien & Salanié 1994]; [Štrumbelj 2014];
//! [Clarke et al. 2017]. Worked vectors are pinned in `tests/reference.rs`.

use serde::{Deserialize, Serialize};

use crate::dataset::OddsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// How the bookmaker's margin is removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DevigMethod {
    /// `πᵢ = rᵢ / B`. Simple; leaves the favorite-longshot bias in place.
    Multiplicative,
    /// `πᵢ = rᵢ − (B−1)/n`. Margin split equally; can produce negatives.
    Additive,
    /// `πᵢ = rᵢ^(1/τ)`, `τ` solved so `Σπ = 1`. The practitioner default.
    #[default]
    Power,
    /// Shin's insider-trading model; also recovers the insider share `z`.
    Shin,
}

/// What to do when [`DevigMethod::Additive`] yields a negative probability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdditiveClamp {
    /// Fail loudly (the survey notes additive is "rarely used unmodified").
    #[default]
    Error,
    /// Clamp negatives to 0 and renormalize (a documented, lossy fallback).
    ClampRenormalize,
}

/// De-vigging parameters. The struct is the algorithm; fields are params.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OddsDevig {
    /// Which margin-removal map to apply.
    pub method: DevigMethod,
    /// Negative-probability policy for [`DevigMethod::Additive`].
    pub additive_clamp: AdditiveClamp,
    /// Root-finder tolerance for the power/Shin solves.
    pub tolerance: f64,
}

impl Default for OddsDevig {
    fn default() -> Self {
        Self {
            method: DevigMethod::default(),
            additive_clamp: AdditiveClamp::default(),
            tolerance: 1e-12,
        }
    }
}

impl OddsDevig {
    /// De-vigs one event's decimal `odds`, returning `(fair πᵢ, booksum B,
    /// insider share z)`. `z` is 0 for every method but Shin.
    fn devig_event(&self, odds: &[f64]) -> Result<(Vec<f64>, f64, f64)> {
        let r: Vec<f64> = odds.iter().map(|&o| 1.0 / o).collect();
        let b: f64 = r.iter().sum();
        let (pi, z) = match self.method {
            DevigMethod::Multiplicative => (r.iter().map(|&ri| ri / b).collect(), 0.0),
            DevigMethod::Additive => (self.additive(&r, b)?, 0.0),
            DevigMethod::Power => (self.power(&r, b)?, 0.0),
            DevigMethod::Shin => self.shin(&r, b),
        };
        Ok((pi, b, z))
    }

    fn additive(&self, r: &[f64], b: f64) -> Result<Vec<f64>> {
        let n = r.len() as f64;
        let pi: Vec<f64> = r.iter().map(|&ri| ri - (b - 1.0) / n).collect();
        if pi.iter().any(|&p| p < 0.0) {
            return match self.additive_clamp {
                AdditiveClamp::Error => Err(Error::Numeric(
                    "additive de-vig produced a negative probability; \
                     use power/shin or the clamp-renormalize policy"
                        .into(),
                )),
                AdditiveClamp::ClampRenormalize => {
                    let clamped: Vec<f64> = pi.iter().map(|&p| p.max(0.0)).collect();
                    let s: f64 = clamped.iter().sum();
                    if s <= 0.0 {
                        return Err(Error::Numeric(
                            "additive de-vig: all mass clamped away".into(),
                        ));
                    }
                    Ok(clamped.iter().map(|&p| p / s).collect())
                }
            };
        }
        Ok(pi)
    }

    fn power(&self, r: &[f64], b: f64) -> Result<Vec<f64>> {
        if (b - 1.0).abs() < 1e-15 {
            return Ok(r.to_vec()); // no margin: already fair
        }
        // Every rᵢ < 1 (decimal odds > 1), so g(τ) = Σ rᵢ^(1/τ) − 1 is strictly
        // increasing; g(1) = B − 1. The root is in (0, 1) when B > 1, in (1, ∞)
        // otherwise.
        let g = |tau: f64| r.iter().map(|&ri| ri.powf(1.0 / tau)).sum::<f64>() - 1.0;
        let (lo, hi) = if b > 1.0 { (1e-12, 1.0) } else { (1.0, 1e6) };
        let tau = mathx::bracketed_root(g, lo, hi, self.tolerance.max(1e-15), 200)
            .ok_or_else(|| Error::Numeric("power de-vig: no exponent solves Σπ=1".into()))?;
        Ok(r.iter().map(|&ri| ri.powf(1.0 / tau)).collect())
    }

    fn shin(&self, r: &[f64], b: f64) -> (Vec<f64>, f64) {
        let multiplicative = || r.iter().map(|&ri| ri / b).collect::<Vec<_>>();
        if b <= 1.0 + 1e-15 {
            return (multiplicative(), 0.0); // no over-round to attribute to insiders
        }
        let pi_of = |z: f64| -> Vec<f64> {
            r.iter()
                .map(|&ri| {
                    let num = (z * z + 4.0 * (1.0 - z) * ri * ri / b).sqrt() - z;
                    num / (2.0 * (1.0 - z))
                })
                .collect()
        };
        let h = |z: f64| pi_of(z).iter().sum::<f64>() - 1.0;
        match mathx::bracketed_root(h, 0.0, 1.0 - 1e-9, self.tolerance.max(1e-15), 200) {
            Some(z) if z > 0.0 => (pi_of(z), z),
            // No insider solution in range: fall back to multiplicative (z = 0).
            _ => (multiplicative(), 0.0),
        }
    }
}

/// One outcome's line in a de-vig state file.
#[derive(Debug, Serialize, Deserialize)]
struct DevigLine {
    id: String,
    ev: usize,
    pi: f64,
    b: f64,
    z: f64,
}

/// Per-outcome fair probabilities plus per-event diagnostics.
#[derive(Debug, Clone)]
pub struct OddsDevigModel {
    params: OddsDevig,
    names: Interner,
    event_of: Vec<usize>,
    pi: Vec<f64>,
    book_b: Vec<f64>,
    z: Vec<f64>,
}

impl OddsDevigModel {
    /// Number of events.
    pub fn n_events(&self) -> usize {
        self.book_b.len()
    }

    /// The booksum (overround + 1) of event `e`.
    pub fn booksum(&self, e: usize) -> Option<f64> {
        self.book_b.get(e).copied()
    }

    /// The estimated insider share `z` of event `e` (Shin only; 0 otherwise).
    pub fn insider_share(&self, e: usize) -> Option<f64> {
        self.z.get(e).copied()
    }
}

impl RankModel for OddsDevigModel {
    fn algorithm(&self) -> &'static str {
        "odds-devig"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.pi.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<DevigLine> = self
            .names
            .names()
            .enumerate()
            .map(|(i, id)| {
                let e = self.event_of[i];
                DevigLine {
                    id: id.to_string(),
                    ev: e,
                    pi: self.pi[i],
                    b: self.book_b[e],
                    z: self.z[e],
                }
            })
            .collect();
        state::save_model(w, "odds-devig", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (OddsDevig, Vec<DevigLine>) = state::load_model(r, "odds-devig")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        let n_events = lines.iter().map(|l| l.ev + 1).max().unwrap_or(0);
        let mut book_b = vec![0.0; n_events];
        let mut z = vec![0.0; n_events];
        let mut event_of = vec![0usize; lines.len()];
        let mut pi = vec![0.0; lines.len()];
        for (i, l) in lines.iter().enumerate() {
            event_of[i] = l.ev;
            pi[i] = l.pi;
            book_b[l.ev] = l.b;
            z[l.ev] = l.z;
        }
        Ok(Self {
            params,
            names,
            event_of,
            pi,
            book_b,
            z,
        })
    }
}

impl Ranker for OddsDevig {
    type Data = OddsDataset;
    type Model = OddsDevigModel;

    fn fit_opts(&self, data: &OddsDataset, _opts: &FitOptions<'_>) -> Result<OddsDevigModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let n_out = data.n_entities();
        let n_events = data.len();
        let mut pi = vec![0.0; n_out];
        let mut event_of = vec![0usize; n_out];
        let mut book_b = vec![0.0; n_events];
        let mut z = vec![0.0; n_events];

        for (e, (ids, odds)) in data.events().enumerate() {
            let (fair, b, zz) = self.devig_event(odds)?;
            book_b[e] = b;
            z[e] = zz;
            for (k, &id) in ids.iter().enumerate() {
                pi[id as usize] = fair[k];
                event_of[id as usize] = e;
            }
        }

        Ok(OddsDevigModel {
            params: *self,
            names: data.interner().clone(),
            event_of,
            pi,
            book_b,
            z,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_event(odds: &[(&str, f64)]) -> OddsDataset {
        let mut d = OddsDataset::new();
        d.push_event(odds).unwrap();
        d
    }

    fn fit(method: DevigMethod, d: &OddsDataset) -> OddsDevigModel {
        OddsDevig {
            method,
            ..Default::default()
        }
        .fit(d)
        .unwrap()
    }

    fn sums_to_one(m: &OddsDevigModel) {
        let s: f64 = m.scores().map(|(_, p)| p).sum();
        assert!((s - 1.0).abs() < 1e-9, "Σπ = {s}");
    }

    #[test]
    fn all_methods_sum_to_one() {
        let d = one_event(&[("a", 4.20), ("b", 3.70), ("c", 1.95)]);
        for m in [
            DevigMethod::Multiplicative,
            DevigMethod::Power,
            DevigMethod::Shin,
        ] {
            sums_to_one(&fit(m, &d));
        }
    }

    #[test]
    fn multiplicative_exact() {
        let d = one_event(&[("a", 2.0), ("b", 2.0)]);
        let m = fit(DevigMethod::Multiplicative, &d);
        let s: std::collections::HashMap<_, _> = m.scores().collect();
        assert!((s["a"] - 0.5).abs() < 1e-12);
        assert!((s["b"] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn additive_errors_on_negative_by_default() {
        // A lopsided three-way book drives the longshot negative under the
        // equal-split margin (impossible with only two outcomes).
        let d = one_event(&[("fav", 1.05), ("mid", 5.0), ("dog", 30.0)]);
        assert!(
            OddsDevig {
                method: DevigMethod::Additive,
                ..Default::default()
            }
            .fit(&d)
            .is_err()
        );
        // Clamp-renormalize recovers a valid distribution.
        let m = OddsDevig {
            method: DevigMethod::Additive,
            additive_clamp: AdditiveClamp::ClampRenormalize,
            ..Default::default()
        }
        .fit(&d)
        .unwrap();
        sums_to_one(&m);
    }

    #[test]
    fn shin_generative_round_trip() {
        // Build quoted odds from a known (π, z) via Shin's forward map — the
        // exact inverse of the de-vig solve — then recover both. With
        // gᵢ = √((1−z)πᵢ² + z πᵢ), the quotes are rᵢ = (Σⱼ gⱼ)·gᵢ (so the
        // booksum is B = (Σ gⱼ)²). The strongest devig test: no external numbers.
        let cases: [(Vec<f64>, f64); 3] = [
            (vec![0.6, 0.3, 0.1], 0.05),
            (vec![0.5, 0.5], 0.10),
            (vec![0.25, 0.25, 0.25, 0.25], 0.20),
        ];
        for (truth, z) in cases {
            let g: Vec<f64> = truth
                .iter()
                .map(|&p| ((1.0 - z) * p * p + z * p).sqrt())
                .collect();
            let sqrt_b: f64 = g.iter().sum();
            let r: Vec<f64> = g.iter().map(|&gi| sqrt_b * gi).collect();
            let odds: Vec<(String, f64)> = r
                .iter()
                .enumerate()
                .map(|(i, &ri)| (format!("o{i}"), 1.0 / ri))
                .collect();
            let pairs: Vec<(&str, f64)> = odds.iter().map(|(n, o)| (n.as_str(), *o)).collect();
            let d = one_event(&pairs);
            let m = fit(DevigMethod::Shin, &d);
            let s: std::collections::HashMap<_, _> = m.scores().collect();
            for (i, &p) in truth.iter().enumerate() {
                let key = format!("o{i}");
                assert!((s[key.as_str()] - p).abs() < 1e-9, "π recover");
            }
            assert!((m.insider_share(0).unwrap() - z).abs() < 1e-9, "z recover");
        }
    }

    #[test]
    fn power_and_shin_correct_favorite_longshot() {
        // A realistic over-round book (booksum > 1). Power and Shin shave the
        // longshot's implied probability MORE than the favorite's, so relative
        // to the proportional multiplicative split the favorite ends up higher
        // and the longshot lower — the favorite-longshot bias correction.
        let d = one_event(&[("fav", 1.25), ("mid", 4.50), ("dog", 15.0)]);
        let mm = fit(DevigMethod::Multiplicative, &d);
        let pm = fit(DevigMethod::Power, &d);
        let sm = fit(DevigMethod::Shin, &d);
        let mult: std::collections::HashMap<_, _> = mm.scores().collect();
        let pow: std::collections::HashMap<_, _> = pm.scores().collect();
        let shin: std::collections::HashMap<_, _> = sm.scores().collect();
        assert!(
            pow["fav"] > mult["fav"] && pow["dog"] < mult["dog"],
            "power"
        );
        assert!(
            shin["fav"] > mult["fav"] && shin["dog"] < mult["dog"],
            "shin"
        );
    }

    #[test]
    fn round_trip() {
        let d = one_event(&[("a", 4.20), ("b", 3.70), ("c", 1.95)]);
        let m = fit(DevigMethod::Shin, &d);
        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = OddsDevigModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);
    }
}

//! Weng-Lin Bayesian ratings for multi-team, multiplayer matches —
//! the method behind the OpenSkill libraries (`docs/algorithms.md` §2.5;
//! Weng & Lin, JMLR 12 (2011) 267–300).
//!
//! Closed-form (μ, σ) updates with TrueSkill-class accuracy and no factor
//! graph: team skill aggregates as the **sum** of member (μ, σ²); every
//! ordered team pair contributes a win/tie/loss term computed from the
//! **pre-match** aggregates; each team's total update is partitioned across
//! its players proportional to their variance. Two likelihood variants:
//!
//! - [`WengLinVariant::BradleyTerryFull`] (Algorithm 1) — logistic;
//! - [`WengLinVariant::ThurstoneMostellerFull`] (Algorithm 3) — probit with
//!   a draw margin `ε` and the paper's §6.1 numerical safeguards.
//!
//! Ranks are competition-style (smaller = better, ties share). `tau` is not
//! in the paper — it is the openskill convention (`σ ← √(σ² + τ²)` for
//! participants before each match) for keeping ratings adaptive; the
//! paper-faithful default is 0.
//!
//! Gotcha: the paper's Algorithm 3 display omits `γ_q` in the variance
//! update, but §6.1 states it applies to TM-full — implemented with γ, as
//! the openskill libraries do.

use serde::{Deserialize, Serialize};

use crate::dataset::MatchupsDataset;
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::mathx;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// Pairwise likelihood (paper Algorithms 1 and 3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WengLinVariant {
    #[default]
    BradleyTerryFull,
    ThurstoneMostellerFull,
}

/// The γ factor damping variance updates (paper §6.1 default vs Table 3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GammaPolicy {
    /// `γ_q = σ_i / c_iq` (the paper's recommended choice).
    #[default]
    SigmaOverC,
    /// `γ_q = 1/k` (k = teams in the match).
    OneOverK,
}

/// Weng-Lin parameters (paper §6.1 defaults).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WengLin {
    pub variant: WengLinVariant,
    /// Initial rating mean.
    pub mu: f64,
    /// Initial rating deviation (μ/3 ⇒ ~zero mass below 0).
    pub sigma: f64,
    /// Performance noise added to every pairwise comparison.
    pub beta: f64,
    /// Variance floor multiplier: σ² never shrinks below κ of itself.
    pub kappa: f64,
    /// Draw margin (Thurstone-Mosteller only).
    pub epsilon: f64,
    /// Pre-match additive dynamics: σ ← √(σ² + τ²) (0 = paper-faithful).
    pub tau: f64,
    pub gamma: GammaPolicy,
}

impl Default for WengLin {
    fn default() -> Self {
        Self {
            variant: WengLinVariant::BradleyTerryFull,
            mu: 25.0,
            sigma: 25.0 / 3.0,
            beta: 25.0 / 6.0,
            kappa: 0.0001,
            epsilon: 0.1,
            tau: 0.0,
            gamma: GammaPolicy::SigmaOverC,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RatingLine {
    id: String,
    mu: f64,
    sigma: f64,
}

/// One player's rating.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rating {
    pub mu: f64,
    pub sigma: f64,
}

impl Rating {
    /// Conservative single-number summary `μ − z·σ` (z = 3 is the
    /// openskill "ordinal" convention).
    pub fn ordinal(&self, z: f64) -> f64 {
        self.mu - z * self.sigma
    }
}

/// Accumulating Weng-Lin ratings keyed by player name.
#[derive(Debug, Clone)]
pub struct WengLinModel {
    params: WengLin,
    names: Interner,
    players: Vec<Rating>,
}

impl WengLinModel {
    /// `(name, rating)` for every seen player.
    pub fn ratings(&self) -> impl Iterator<Item = (&str, Rating)> {
        self.names.names().zip(self.players.iter().copied())
    }

    fn intern(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.players.len() {
            self.players.push(Rating {
                mu: self.params.mu,
                sigma: self.params.sigma,
            });
        }
        idx
    }
}

impl RankModel for WengLinModel {
    fn algorithm(&self) -> &'static str {
        "weng-lin"
    }

    /// Rating means (pair with [`WengLinModel::ratings`] for uncertainty).
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.players.iter().map(|p| p.mu))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<RatingLine> = self
            .ratings()
            .map(|(id, r)| RatingLine {
                id: id.to_string(),
                mu: r.mu,
                sigma: r.sigma,
            })
            .collect();
        state::save_model(w, "weng-lin", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (WengLin, Vec<RatingLine>) = state::load_model(r, "weng-lin")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        Ok(Self {
            params,
            names,
            players: lines
                .iter()
                .map(|l| Rating {
                    mu: l.mu,
                    sigma: l.sigma,
                })
                .collect(),
        })
    }
}

/// φ, the standard normal density.
fn pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Tiny-denominator guard from the paper's §6.1 (≈ smallest positive Φ they
/// observed before switching to the asymptote).
const PHI_FLOOR: f64 = 2.222_758_749e-162;

/// `V(x, t) = φ(x−t)/Φ(x−t)`, with the paper's safeguard `−x + t` as
/// `Φ → 0`.
fn v_win(x: f64, t: f64) -> f64 {
    let denom = mathx::norm_cdf(x - t);
    if denom <= PHI_FLOOR {
        return -x + t;
    }
    pdf(x - t) / denom
}

/// `W(x, t) = V(V + (x − t))`.
fn w_win(x: f64, t: f64) -> f64 {
    let v = v_win(x, t);
    v * (v + x - t)
}

/// Tie mean-shift `Ṽ(x, t)` (paper eq. 68), openskill-style fallback when
/// the probability mass between the margins underflows.
fn v_tie(x: f64, t: f64) -> f64 {
    let denom = mathx::norm_cdf(t - x) - mathx::norm_cdf(-t - x);
    if denom <= PHI_FLOOR {
        return if x < 0.0 { -x - t } else { -x + t };
    }
    (pdf(-t - x) - pdf(t - x)) / denom
}

/// Tie variance factor `W̃(x, t)` (paper eq. 69); fallback 1 keeps σ moving
/// toward the floor instead of producing NaN.
fn w_tie(x: f64, t: f64) -> f64 {
    let denom = mathx::norm_cdf(t - x) - mathx::norm_cdf(-t - x);
    if denom <= PHI_FLOOR {
        return 1.0;
    }
    let v = v_tie(x, t);
    ((t - x) * pdf(t - x) + (t + x) * pdf(-t - x)) / denom + v * v
}

impl OnlineRanker for WengLin {
    type Data = MatchupsDataset;
    type Model = WengLinModel;

    fn init(&self) -> WengLinModel {
        WengLinModel {
            params: *self,
            names: Interner::new(),
            players: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut WengLinModel,
        data: &MatchupsDataset,
        opts: &FitOptions<'_>,
    ) -> Result<()> {
        if model.params != *self {
            return Err(Error::ParamMismatch(format!(
                "model was built with {:?}, updater configured with {:?}",
                model.params, self
            )));
        }

        let progress = opts.progress;
        progress.start("matches", Some(data.len() as u64));

        for (m, match_teams) in data.matches().enumerate() {
            // Resolve rosters into model indices (interning new players).
            let teams: Vec<(u32, Vec<usize>)> = match_teams
                .map(|(rank, roster)| {
                    let roster = roster
                        .iter()
                        .map(|&id| model.intern(data.interner().resolve(id)))
                        .collect();
                    (rank, roster)
                })
                .collect();
            let k = teams.len();

            // Pre-match dynamics, then frozen team aggregates.
            if self.tau > 0.0 {
                for (_, roster) in &teams {
                    for &p in roster {
                        let s = &mut model.players[p];
                        s.sigma = (s.sigma * s.sigma + self.tau * self.tau).sqrt();
                    }
                }
            }

            let agg: Vec<(f64, f64)> = teams
                .iter()
                .map(|(_, roster)| {
                    let mu = roster.iter().map(|&p| model.players[p].mu).sum();
                    let var = roster
                        .iter()
                        .map(|&p| model.players[p].sigma * model.players[p].sigma)
                        .sum();
                    (mu, var)
                })
                .collect();

            // Per-team Ω and Δ from every ordered opponent pair.
            let mut omega = vec![0.0; k];
            let mut delta = vec![0.0; k];

            for i in 0..k {
                let (mu_i, var_i) = agg[i];

                for q in 0..k {
                    if q == i {
                        continue;
                    }
                    let (mu_q, var_q) = agg[q];
                    let c = (var_i + var_q + 2.0 * self.beta * self.beta).sqrt();
                    let gamma = match self.gamma {
                        GammaPolicy::SigmaOverC => var_i.sqrt() / c,
                        GammaPolicy::OneOverK => 1.0 / k as f64,
                    };

                    let (rank_i, rank_q) = (teams[i].0, teams[q].0);
                    let (d_q, e_q) = match self.variant {
                        WengLinVariant::BradleyTerryFull => {
                            let p_iq = 1.0 / (1.0 + ((mu_q - mu_i) / c).exp());
                            let s = match rank_q.cmp(&rank_i) {
                                std::cmp::Ordering::Greater => 1.0,
                                std::cmp::Ordering::Equal => 0.5,
                                std::cmp::Ordering::Less => 0.0,
                            };
                            (
                                var_i / c * (s - p_iq),
                                gamma * (var_i / (c * c)) * p_iq * (1.0 - p_iq),
                            )
                        }
                        WengLinVariant::ThurstoneMostellerFull => {
                            let x = (mu_i - mu_q) / c;
                            let t = self.epsilon / c;
                            let (v, w) = match rank_q.cmp(&rank_i) {
                                std::cmp::Ordering::Greater => (v_win(x, t), w_win(x, t)),
                                std::cmp::Ordering::Equal => (v_tie(x, t), w_tie(x, t)),
                                std::cmp::Ordering::Less => (-v_win(-x, t), w_win(-x, t)),
                            };
                            (var_i / c * v, gamma * (var_i / (c * c)) * w)
                        }
                    };

                    omega[i] += d_q;
                    delta[i] += e_q;
                }
            }

            // Partition each team's update across its players by variance.
            for (i, (_, roster)) in teams.iter().enumerate() {
                let (_, var_i) = agg[i];
                for &p in roster {
                    let s = &mut model.players[p];
                    let ratio = s.sigma * s.sigma / var_i;
                    s.mu += ratio * omega[i];
                    s.sigma *= (1.0 - ratio * delta[i]).max(self.kappa).sqrt();
                }
            }

            progress.update(m as u64 + 1);
        }

        progress.finish();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffa(names: &[&str]) -> MatchupsDataset {
        let mut d = MatchupsDataset::new();
        let teams: Vec<&[&str]> = names.iter().map(std::slice::from_ref).collect();
        d.push_ordered(&teams).unwrap();
        d
    }

    /// Symmetric tie: both players end identical, μ unchanged.
    #[test]
    fn tie_is_symmetric() {
        let mut d = MatchupsDataset::new();
        d.push_match(&[&["a"], &["b"]], &[1, 1]).unwrap();

        let algo = WengLin::default();
        let mut m = algo.init();
        algo.update(&mut m, &d).unwrap();

        let r: Vec<(&str, Rating)> = m.ratings().collect();
        assert!((r[0].1.mu - 25.0).abs() < 1e-12);
        assert!((r[1].1.mu - 25.0).abs() < 1e-12);
        assert_eq!(r[0].1.sigma, r[1].1.sigma);
        assert!(r[0].1.sigma < 25.0 / 3.0, "ties still reduce uncertainty");
    }

    /// Split updates equal one continuous run (FR-5).
    #[test]
    fn split_update_equals_continuous() {
        let algo = WengLin::default();

        let mut d1 = MatchupsDataset::new();
        d1.push_ordered(&[&["a"], &["b"], &["c"]]).unwrap();
        let mut d2 = MatchupsDataset::new();
        d2.push_ordered(&[&["c"], &["a"]]).unwrap();

        let mut split = algo.init();
        algo.update(&mut split, &d1).unwrap();
        algo.update(&mut split, &d2).unwrap();

        let mut both = MatchupsDataset::new();
        both.push_ordered(&[&["a"], &["b"], &["c"]]).unwrap();
        both.push_ordered(&[&["c"], &["a"]]).unwrap();
        let mut continuous = algo.init();
        algo.update(&mut continuous, &both).unwrap();

        let a: Vec<_> = split.ratings().map(|(n, r)| (n.to_string(), r)).collect();
        let b: Vec<_> = continuous
            .ratings()
            .map(|(n, r)| (n.to_string(), r))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn param_mismatch_is_rejected() {
        let algo = WengLin::default();
        let mut m = algo.init();
        let other = WengLin {
            beta: 1.0,
            ..Default::default()
        };
        assert!(matches!(
            other.update(&mut m, &ffa(&["a", "b"])),
            Err(Error::ParamMismatch(_))
        ));
    }

    /// Multi-team free-for-all: order of finish = order of ratings, and the
    /// middle finisher of a 3-way stays at μ₀ by symmetry.
    #[test]
    fn ffa_order_and_symmetry() {
        let algo = WengLin::default();
        let mut m = algo.init();
        algo.update(&mut m, &ffa(&["x", "y", "z"])).unwrap();

        let r: std::collections::HashMap<&str, Rating> = m.ratings().collect();
        assert!(r["x"].mu > r["y"].mu && r["y"].mu > r["z"].mu);
        assert!((r["y"].mu - 25.0).abs() < 1e-9, "middle is symmetric");
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let algo = WengLin::default();
        let mut m = algo.init();
        algo.update(&mut m, &ffa(&["a", "b", "c"])).unwrap();

        let mut first = Vec::new();
        m.save_jsonl(&mut first).unwrap();
        let loaded = WengLinModel::load_jsonl(first.as_slice()).unwrap();
        let mut second = Vec::new();
        loaded.save_jsonl(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

//! Glicko-2 rating system (`docs/algorithms.md` §2.3).
//!
//! Each entity carries a rating, a rating deviation (RD — how uncertain the
//! rating is), and a volatility σ (how erratic the entity's performance has
//! been). Updates happen per **rating period**: every period in the dataset
//! ([`GamesDataset::periods`](crate::GamesDataset::periods), blank-line
//! batches in the input file) is folded in as one Glicko-2 step, against the
//! ratings as they stood at the period's start.
//!
//! This is the flagship incremental algorithm (PRD FR-5): state lives in the
//! owned [`Glicko2Model`]; `update` never replays history. Ties take the
//! native `S = ½` path; margins are ignored; a game's repeat count scales
//! its period contribution (the game happened that many times).
//!
//! Algorithm reference: Glickman, "Example of the Glicko-2 system"
//! (<http://www.glicko.net/glicko/glicko2.pdf>); model from Glickman (2001).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dataset::{GameOutcome, GamesDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, OnlineRanker, RankModel};

/// Glicko-2 parameters. `rating`/`rd`/`sigma` are the values assigned to
/// unseen entities; `tau` constrains how fast volatility can change
/// (0.3–1.2; smaller = more stable).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glicko2 {
    /// System constant constraining how fast volatility changes (0.3–1.2;
    /// smaller = more stable).
    pub tau: f64,
    /// Rating assigned to unseen entities.
    pub rating: f64,
    /// Rating deviation assigned to unseen entities.
    pub rd: f64,
    /// Volatility assigned to unseen entities.
    pub sigma: f64,
}

impl Default for Glicko2 {
    fn default() -> Self {
        Self {
            tau: 0.5,
            rating: 1500.0,
            rd: 350.0,
            sigma: 0.06,
        }
    }
}

/// Glicko/Glicko-2 scale conversion constant (≈ 400 / ln 10).
const SCALE: f64 = 173.7178;

/// One entity's rating state.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlayerState {
    /// Rating on the display (Elo-like) scale.
    pub r: f64,
    /// Rating deviation on the display scale.
    pub rd: f64,
    /// Volatility.
    pub sigma: f64,
}

impl PlayerState {
    /// 95%-ish interval `(r − 2·rd, r + 2·rd)`.
    pub fn bounds(&self) -> (f64, f64) {
        (self.r - 2.0 * self.rd, self.r + 2.0 * self.rd)
    }
}

/// One entity line in the state file.
#[derive(Debug, Serialize, Deserialize)]
struct PlayerLine {
    id: String,
    r: f64,
    rd: f64,
    sigma: f64,
}

/// Owned, updateable Glicko-2 state.
#[derive(Debug, Clone)]
pub struct Glicko2Model {
    params: Glicko2,
    names: Interner,
    players: Vec<PlayerState>,
}

impl Glicko2Model {
    /// The configured parameters (e.g. for compatibility checks).
    pub fn params(&self) -> &Glicko2 {
        &self.params
    }

    /// Per-entity rating states in id order, paired with names.
    pub fn players(&self) -> impl Iterator<Item = (&str, &PlayerState)> {
        self.names.names().zip(self.players.iter())
    }

    /// Rating on the internal (μ) scale — what v1 printed with `--use-mu`
    /// as the first column of the default output.
    pub fn mu(&self, p: &PlayerState) -> f64 {
        (p.r - self.params.rating) / SCALE
    }

    /// Seeds or overwrites an entity's state (priors, migrations, tests).
    pub fn set_player(&mut self, name: &str, state: PlayerState) {
        let idx = self.intern(name);
        self.players[idx] = state;
    }

    fn intern(&mut self, name: &str) -> usize {
        let idx = self.names.intern(name) as usize;
        if idx == self.players.len() {
            self.players.push(PlayerState {
                r: self.params.rating,
                rd: self.params.rd,
                sigma: self.params.sigma,
            });
        }
        idx
    }

    fn mu_phi(&self, idx: usize) -> (f64, f64) {
        let p = &self.players[idx];
        ((p.r - self.params.rating) / SCALE, p.rd / SCALE)
    }

    /// One rating period: all games are scored against the ratings as they
    /// stood when the period began. `games` rows are `(side1, side2,
    /// side-1 score, multiplicity)` with score ∈ {1, ½, 0}.
    fn update_period(&mut self, games: &[(usize, usize, f64, f64)], tau: f64) -> Result<()> {
        // Accumulate v and Δ contributions per participant.
        let mut dv: HashMap<usize, (f64, f64)> = HashMap::new();
        for &(a, b, s1, weight) in games {
            for (me, them, score) in [(a, b, s1), (b, a, 1.0 - s1)] {
                let (mu, _) = self.mu_phi(me);
                let (mu_j, phi_j) = self.mu_phi(them);
                let g = g_of_phi(phi_j);
                let e = 1.0 / (1.0 + (-g * (mu - mu_j)).exp());
                let entry = dv.entry(me).or_insert((0.0, 0.0));
                entry.0 += weight * g * g * e * (1.0 - e);
                entry.1 += weight * g * (score - e);
            }
        }

        // Sort for deterministic floating-point evaluation order.
        let mut participants: Vec<_> = dv.into_iter().collect();
        participants.sort_unstable_by_key(|(idx, _)| *idx);

        for (idx, (v_i, d_i)) in participants {
            let (mu, phi) = self.mu_phi(idx);
            let sigma = self.players[idx].sigma;
            let v_t = 1.0 / v_i;
            let delta_t = v_t * d_i;
            let sigma_prime = compute_volatility(phi, sigma, delta_t, v_t, tau)?;

            let phi_star = (phi * phi + sigma_prime * sigma_prime).sqrt();
            let phi_prime = 1.0 / (1.0 / (phi_star * phi_star) + 1.0 / v_t).sqrt();
            let mu_prime = mu + phi_prime * phi_prime * d_i;

            self.players[idx] = PlayerState {
                r: mu_prime * SCALE + self.params.rating,
                rd: phi_prime * SCALE,
                sigma: sigma_prime,
            };
        }
        Ok(())
    }
}

impl RankModel for Glicko2Model {
    fn algorithm(&self) -> &'static str {
        "glicko2"
    }

    /// Primary score is the display-scale rating.
    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.players.iter().map(|p| p.r))
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let lines: Vec<PlayerLine> = self
            .players()
            .map(|(id, p)| PlayerLine {
                id: id.to_string(),
                r: p.r,
                rd: p.rd,
                sigma: p.sigma,
            })
            .collect();
        state::save_model(w, "glicko2", &self.params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (Glicko2, Vec<PlayerLine>) = state::load_model(r, "glicko2")?;
        let names = Interner::from_names(lines.iter().map(|l| l.id.as_str()))?;
        let players = lines
            .iter()
            .map(|l| PlayerState {
                r: l.r,
                rd: l.rd,
                sigma: l.sigma,
            })
            .collect();
        Ok(Self {
            params,
            names,
            players,
        })
    }
}

impl OnlineRanker for Glicko2 {
    type Data = GamesDataset;
    type Model = Glicko2Model;

    fn init(&self) -> Glicko2Model {
        Glicko2Model {
            params: *self,
            names: Interner::new(),
            players: Vec::new(),
        }
    }

    fn update_opts(
        &self,
        model: &mut Glicko2Model,
        data: &GamesDataset,
        opts: &FitOptions<'_>,
    ) -> Result<()> {
        if model.params != *self {
            return Err(Error::ParamMismatch(format!(
                "model was built with {:?}, updater configured with {:?}",
                model.params, self
            )));
        }
        let progress = opts.progress;
        progress.start("glicko2 periods", Some(data.n_periods() as u64));
        for (i, period) in data.periods().enumerate() {
            let start = period.start;
            let games: Vec<(usize, usize, f64, f64)> = data
                .period_games(period)
                .enumerate()
                .map(|(off, view)| {
                    let (&[a], &[b]) = (view.side1, view.side2) else {
                        return Err(Error::InvalidInput(format!(
                            "game {} has a multi-player side; glicko2 rates 1v1 \
                             games — use weng-lin on a matchups dataset for teams",
                            start + off
                        )));
                    };
                    let s1 = match view.outcome {
                        GameOutcome::Side1Win(_) => 1.0,
                        GameOutcome::Tie => 0.5,
                        GameOutcome::Side2Win(_) => 0.0,
                    };
                    let an = data.interner().resolve(a);
                    let bn = data.interner().resolve(b);
                    Ok((
                        model.intern(an),
                        model.intern(bn),
                        s1,
                        f64::from(view.weight),
                    ))
                })
                .collect::<Result<_>>()?;
            model.update_period(&games, self.tau)?;
            progress.update(i as u64 + 1);
        }
        progress.finish();
        Ok(())
    }
}

#[inline]
fn g_of_phi(phi: f64) -> f64 {
    1.0 / (1.0 + 3.0 * phi * phi / (std::f64::consts::PI * std::f64::consts::PI)).sqrt()
}

/// Solves for the new volatility σ′ by the Illinois-style root finder from
/// Glickman's reference note.
fn compute_volatility(phi: f64, sigma: f64, delta: f64, v: f64, tau: f64) -> Result<f64> {
    let eps = 1e-7;
    let a = (sigma * sigma).ln();
    let tau2 = tau * tau;

    let f = |x: f64| -> f64 {
        let ex = x.exp();
        let num = ex * (delta * delta - phi * phi - v - ex);
        let den = 2.0 * (phi * phi + v + ex);
        num / den - (x - a) / tau2
    };

    let mut lo = a;
    let mut hi = if delta * delta > phi * phi + v {
        (delta * delta - phi * phi - v).ln()
    } else {
        let mut k = 1.0;
        let mut found = false;
        for _ in 0..100 {
            if f(a - k * tau) >= 0.0 {
                found = true;
                break;
            }
            k += 1.0;
        }
        if !found {
            return Err(Error::Numeric(
                "glicko2 volatility bracket not found".into(),
            ));
        }
        a - k * tau
    };

    let mut f_lo = f(lo);
    let mut f_hi = f(hi);
    for _ in 0..100 {
        if (hi - lo).abs() <= eps || lo.is_nan() || hi.is_nan() {
            break;
        }
        let c = lo + (lo - hi) * f_lo / (f_hi - f_lo);
        let f_c = f(c);
        if f_c * f_hi < 0.0 {
            lo = hi;
            f_lo = f_hi;
        } else {
            f_lo /= 2.0;
        }
        hi = c;
        f_hi = f_c;
    }

    if lo.is_nan() || hi.is_nan() {
        return Err(Error::Numeric(
            "glicko2 volatility root finder diverged".into(),
        ));
    }
    Ok((lo / 2.0).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked example from Glickman's Glicko-2 note (ported v1 test).
    #[test]
    fn glickman_reference_example() {
        let algo = Glicko2::default();
        let mut model = algo.init();
        model.set_player(
            "1",
            PlayerState {
                r: 1500.0,
                rd: 200.0,
                sigma: 0.06,
            },
        );
        model.set_player(
            "2",
            PlayerState {
                r: 1400.0,
                rd: 30.0,
                sigma: 0.06,
            },
        );
        model.set_player(
            "3",
            PlayerState {
                r: 1550.0,
                rd: 100.0,
                sigma: 0.06,
            },
        );
        model.set_player(
            "4",
            PlayerState {
                r: 1700.0,
                rd: 300.0,
                sigma: 0.06,
            },
        );

        let mut d = GamesDataset::new();
        d.push_pair("1", "2", 1.0).unwrap();
        d.push_pair("3", "1", 1.0).unwrap();
        d.push_pair("4", "1", 1.0).unwrap();
        algo.update(&mut model, &d).unwrap();

        let p1 = model.players().find(|(n, _)| *n == "1").unwrap().1;
        assert!((p1.r - 1464.06).abs() < 0.01, "r = {}", p1.r);
        assert!((p1.rd - 151.52).abs() < 0.01, "rd = {}", p1.rd);
        assert!((p1.sigma - 0.05999).abs() < 0.01, "sigma = {}", p1.sigma);
    }

    /// Ported v1 `test_small_tau`.
    #[test]
    fn small_tau_keeps_volatility_bounded() {
        let algo = Glicko2::default();
        let mut model = algo.init();
        model.set_player(
            "low",
            PlayerState {
                r: 1078.224870320442,
                rd: 231.8396899251802,
                sigma: 0.0599557629529191,
            },
        );
        model.set_player(
            "high",
            PlayerState {
                r: 1922.738120392382,
                rd: 136.9997727497604,
                sigma: 0.06095741696613419,
            },
        );
        let mut d = GamesDataset::new();
        d.push_pair("low", "high", 1.0).unwrap();
        d.push_pair("low", "high", 1.0).unwrap();
        algo.update(&mut model, &d).unwrap();
        let p = model.players().find(|(n, _)| *n == "low").unwrap().1;
        assert!(p.sigma < 1.0);
    }

    /// FR-5 acceptance: saved state + new period == both periods in one go.
    #[test]
    fn resume_equals_continuous() {
        let algo = Glicko2 {
            tau: 0.5,
            ..Default::default()
        };

        // Two periods in one dataset.
        let mut both = GamesDataset::new();
        both.push_pair("a", "b", 1.0).unwrap();
        both.push_pair("c", "b", 1.0).unwrap();
        both.new_period();
        both.push_pair("b", "a", 1.0).unwrap();
        both.push_pair("a", "c", 1.0).unwrap();
        let mut continuous = algo.init();
        algo.update(&mut continuous, &both).unwrap();

        // Same data, split across an update + save/load + update.
        let mut p1 = GamesDataset::new();
        p1.push_pair("a", "b", 1.0).unwrap();
        p1.push_pair("c", "b", 1.0).unwrap();
        let mut p2 = GamesDataset::new();
        p2.push_pair("b", "a", 1.0).unwrap();
        p2.push_pair("a", "c", 1.0).unwrap();

        let mut resumed = algo.init();
        algo.update(&mut resumed, &p1).unwrap();
        let mut buf = Vec::new();
        resumed.save_jsonl(&mut buf).unwrap();
        let mut resumed = Glicko2Model::load_jsonl(buf.as_slice()).unwrap();
        algo.update(&mut resumed, &p2).unwrap();

        let mut a = Vec::new();
        continuous.save_jsonl(&mut a).unwrap();
        let mut b = Vec::new();
        resumed.save_jsonl(&mut b).unwrap();
        assert_eq!(
            String::from_utf8(a).unwrap(),
            String::from_utf8(b).unwrap(),
            "split update must equal continuous update"
        );
    }

    #[test]
    fn param_mismatch_is_rejected() {
        let a = Glicko2::default();
        let b = Glicko2 {
            tau: 1.0,
            ..Default::default()
        };
        let mut model = a.init();
        let mut d = GamesDataset::new();
        d.push_pair("x", "y", 1.0).unwrap();
        assert!(matches!(
            b.update(&mut model, &d),
            Err(Error::ParamMismatch(_))
        ));
    }
}

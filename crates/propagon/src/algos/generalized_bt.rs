//! Generalized Bradley-Terry: explicit ties and home advantage
//! (`docs/algorithms.md` §1.2; Rao & Kupper 1967; Davidson 1970;
//! Agresti 2013; Hunter 2004).
//!
//! One fixed-point chassis with two orthogonal extensions to the BT
//! likelihood over 1v1 games: a tie parameter — Davidson's `ν` with
//! `P(tie) ∝ ν·√(γπ_i·π_j)`, or Rao-Kupper's threshold `θ ≥ 1` — and a
//! multiplicative home advantage `γ` on side 1 (side 1 of a game is the
//! *home* side by crate convention). Games aggregate per ordered
//! (home, away) pair into home-win / away-win / tie weights, so venue
//! information survives even when the same two entities meet both ways.
//!
//! Estimation is cyclic sweeps over the score equations (the Hunter 2004 §5
//! style of generalization): all `π` move simultaneously via the
//! stationarity fixed point `π_i ← credit_i / Σ (∂D/∂π_i)/D` evaluated at
//! the current iterate (credit = wins + half-tie weight for Davidson, wins +
//! tie weight for Rao-Kupper), then the tie parameter (closed form for `ν`,
//! 64-step bisection of the profile gradient over `θ ∈ (1, 100]`), then `γ`
//! from its own score equation — each step using the freshest values. The
//! likelihood is concave in log-parameters, so the unique stationary point
//! is the MLE; the tests verify fits by zeroing numerical gradients of the
//! likelihood rather than trusting these update formulas.
//!
//! Assumes a Ford-style condition: under a tie model every entity needs at
//! least one win-or-tie and one loss-or-tie (ties keep the MLE finite);
//! under [`TieModel::None`] strict wins and losses both ways, like `ilsr`.
//! Offenders surface as a typed [`Error::Numeric`].
//!
//! Gotcha: the likelihood is scale-invariant in `π`, so strengths are
//! normalized to `Σπ = 1` every sweep; `γ` and the tie parameter are
//! scale-free and ride along in the persisted params.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::algos::common::{ScoreLine, from_score_lines, score_lines};
use crate::dataset::{GameOutcome, GamesDataset};
use crate::error::{Error, Result};
use crate::interner::Interner;
use crate::state;
use crate::traits::{FitOptions, RankModel, Ranker};

/// How tied games enter the likelihood.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TieModel {
    /// No tie parameter; tie games are a typed error.
    None,
    /// P(tie) ∝ ν·√(π_i π_j) (Davidson 1970).
    #[default]
    Davidson,
    /// Threshold θ ≥ 1 (Rao & Kupper 1967).
    RaoKupper,
}

/// Whether side 1 (the home side) gets a fitted multiplicative edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HomeAdvantage {
    /// No home edge; γ is pinned to 1.
    #[default]
    None,
    /// Estimate a multiplicative γ on side 1 (the home side).
    Estimate,
}

/// Generalized Bradley-Terry parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneralizedBt {
    /// Tie handling: explicit tie parameter or none.
    pub ties: TieModel,
    /// Home-advantage handling for side 1.
    pub home: HomeAdvantage,
    /// Maximum cyclic sweeps.
    pub iterations: usize,
    /// Early exit when `mean |Δπ| + |Δtie| + |Δγ|` drops below this.
    pub tolerance: f64,
}

impl Default for GeneralizedBt {
    fn default() -> Self {
        Self {
            ties: TieModel::default(),
            home: HomeAdvantage::default(),
            iterations: 10_000,
            tolerance: 1e-9,
        }
    }
}

/// Aggregated outcomes of one ordered (home, away) pair.
struct PairCounts {
    home: u32,
    away: u32,
    /// Weight of games the home side won.
    home_wins: f64,
    /// Weight of games the away side won at this venue.
    away_wins: f64,
    /// Weight of tied games.
    ties: f64,
}

impl GeneralizedBt {
    /// Aggregates 1v1 games per ordered (home, away) pair, validating the
    /// shape: multi-player sides and (under [`TieModel::None`]) tie games
    /// are typed errors. Pairs come out key-sorted so sweeps accumulate in
    /// a fixed order.
    fn aggregate(&self, data: &GamesDataset) -> Result<Vec<PairCounts>> {
        let mut agg: HashMap<(u32, u32), (f64, f64, f64)> = HashMap::new();

        for (g, view) in data.games().enumerate() {
            let (&[h], &[a]) = (view.side1, view.side2) else {
                return Err(Error::InvalidInput(format!(
                    "game {g} has a multi-player side; generalized-bradley-terry \
                     rates 1v1 games — use team-bradley-terry for teams"
                )));
            };
            let w = f64::from(view.weight);
            let entry = agg.entry((h, a)).or_default();

            match view.outcome {
                GameOutcome::Side1Win(_) => entry.0 += w,
                GameOutcome::Side2Win(_) => entry.1 += w,
                GameOutcome::Tie => {
                    if self.ties == TieModel::None {
                        return Err(Error::InvalidInput(format!(
                            "game {g} is a tie; TieModel::None cannot score ties — \
                             pick TieModel::Davidson or TieModel::RaoKupper"
                        )));
                    }
                    entry.2 += w;
                }
            }
        }

        let mut pairs: Vec<PairCounts> = agg
            .into_iter()
            .map(|((home, away), (home_wins, away_wins, ties))| PairCounts {
                home,
                away,
                home_wins,
                away_wins,
                ties,
            })
            .collect();
        pairs.sort_unstable_by_key(|p| (p.home, p.away));
        Ok(pairs)
    }

    /// Ford-style pre-check. With a tie parameter, an entity stays finite as
    /// long as it has some non-win and some non-loss credit (ties count for
    /// both); under [`TieModel::None`] it needs a strict win and a strict
    /// loss, exactly like the prior-free PL fitters.
    fn check_ford(&self, pairs: &[PairCounts], names: &Interner) -> Result<()> {
        let n = names.len();
        let mut won = vec![false; n];
        let mut lost = vec![false; n];
        let mut tied = vec![false; n];

        for p in pairs {
            let (h, a) = (p.home as usize, p.away as usize);
            if p.home_wins > 0.0 {
                won[h] = true;
                lost[a] = true;
            }
            if p.away_wins > 0.0 {
                won[a] = true;
                lost[h] = true;
            }
            if p.ties > 0.0 {
                tied[h] = true;
                tied[a] = true;
            }
        }

        let diverges = |i: usize| match self.ties {
            TieModel::None => !won[i] || !lost[i],
            // "win-or-tie and loss-or-tie" minimized: a tie alone supplies
            // finite credit in both directions.
            TieModel::Davidson | TieModel::RaoKupper => !(tied[i] || (won[i] && lost[i])),
        };

        let offenders: Vec<&str> = (0..n)
            .filter(|&i| diverges(i))
            .filter_map(|i| names.name(i as u32))
            .take(5)
            .collect();

        if offenders.is_empty() {
            Ok(())
        } else {
            Err(Error::Numeric(format!(
                "generalized BT MLE diverges: {} never win-or-tie or never \
                 lose-or-tie (use bayesian-bradley-terry, or bradley-terry-model's \
                 bridging options)",
                offenders.join(", ")
            )))
        }
    }

    /// One simultaneous π sweep: returns the un-normalized fixed-point
    /// denominators `Σ n·(∂D/∂π_i)/D` per entity at the current iterate.
    fn pi_denominators(&self, pairs: &[PairCounts], pi: &[f64], tie: f64, gamma: f64) -> Vec<f64> {
        let mut den = vec![0.0f64; pi.len()];

        for p in pairs {
            let (h, a) = (p.home as usize, p.away as usize);
            let (ph, pa) = (pi[h], pi[a]);

            match self.ties {
                TieModel::None => {
                    let n_g = p.home_wins + p.away_wins;
                    let d = gamma * ph + pa;
                    den[h] += n_g * gamma / d;
                    den[a] += n_g / d;
                }
                TieModel::Davidson => {
                    let n_g = p.home_wins + p.away_wins + p.ties;
                    let s = (gamma * ph * pa).sqrt();
                    let d = gamma * ph + pa + tie * s;
                    den[h] += n_g * (gamma + tie * s / (2.0 * ph)) / d;
                    den[a] += n_g * (1.0 + tie * s / (2.0 * pa)) / d;
                }
                TieModel::RaoKupper => {
                    // Home-win prob γπ_h/A, away-win π_a/B with
                    // A = γπ_h + θπ_a, B = π_a + θγπ_h; ties hit both.
                    let big_a = gamma * ph + tie * pa;
                    let big_b = pa + tie * gamma * ph;
                    let wt = p.home_wins + p.ties;
                    let vt = p.away_wins + p.ties;
                    den[h] += wt * gamma / big_a + vt * tie * gamma / big_b;
                    den[a] += wt * tie / big_a + vt / big_b;
                }
            }
        }
        den
    }

    /// Closed-form Davidson update `ν ← T / Σ n·√(γπ_hπ_a)/D` (the zero of
    /// `∂ℓ/∂ν = T/ν − Σ n·s/D`).
    fn davidson_nu(pairs: &[PairCounts], pi: &[f64], nu: f64, gamma: f64, t_total: f64) -> f64 {
        let mut denom = 0.0;

        for p in pairs {
            let (ph, pa) = (pi[p.home as usize], pi[p.away as usize]);
            let n_g = p.home_wins + p.away_wins + p.ties;
            let s = (gamma * ph * pa).sqrt();
            denom += n_g * s / (gamma * ph + pa + nu * s);
        }

        if denom > 0.0 { t_total / denom } else { 0.0 }
    }

    /// Rao-Kupper θ by 64 fixed halvings on the profile-likelihood gradient
    /// `g(θ) = 2θT/(θ²−1) − Σ [(w+t)·π_a/A + (v+t)·γπ_h/B]` over (1, 100].
    /// `g` decreases in θ and `g(1⁺) = +∞` whenever ties exist, so the sign
    /// change brackets the root; with no ties `g < 0` everywhere and the
    /// bisection collapses to θ = 1 (zero tie mass).
    fn rao_kupper_theta(pairs: &[PairCounts], pi: &[f64], gamma: f64, t_total: f64) -> f64 {
        if t_total <= 0.0 {
            return 1.0;
        }

        let gradient = |theta: f64| {
            let mut g = 2.0 * theta * t_total / (theta * theta - 1.0);
            for p in pairs {
                let (ph, pa) = (pi[p.home as usize], pi[p.away as usize]);
                let big_a = gamma * ph + theta * pa;
                let big_b = pa + theta * gamma * ph;
                g -= (p.home_wins + p.ties) * pa / big_a;
                g -= (p.away_wins + p.ties) * gamma * ph / big_b;
            }
            g
        };

        let (mut lo, mut hi) = (1.0f64, 100.0f64);
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if gradient(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// γ fixed point from `∂ℓ/∂γ = 0`. The numerator is the home-win weight
    /// plus the model's tie credit toward γ (ties carry a √γ factor under
    /// Davidson — half credit — and a full γπ_h factor under Rao-Kupper);
    /// the denominator is `Σ n·(∂denominator/∂γ)/denominator`.
    fn gamma_update(
        &self,
        pairs: &[PairCounts],
        pi: &[f64],
        tie: f64,
        gamma: f64,
        h_total: f64,
        t_total: f64,
    ) -> f64 {
        let mut den = 0.0;

        for p in pairs {
            let (ph, pa) = (pi[p.home as usize], pi[p.away as usize]);
            match self.ties {
                TieModel::None => {
                    let n_g = p.home_wins + p.away_wins;
                    den += n_g * ph / (gamma * ph + pa);
                }
                TieModel::Davidson => {
                    let n_g = p.home_wins + p.away_wins + p.ties;
                    let s = (gamma * ph * pa).sqrt();
                    let d = gamma * ph + pa + tie * s;
                    den += n_g * (ph + tie * s / (2.0 * gamma)) / d;
                }
                TieModel::RaoKupper => {
                    let big_a = gamma * ph + tie * pa;
                    let big_b = pa + tie * gamma * ph;
                    den += (p.home_wins + p.ties) * ph / big_a;
                    den += (p.away_wins + p.ties) * tie * ph / big_b;
                }
            }
        }

        let num = match self.ties {
            TieModel::None => h_total,
            TieModel::Davidson => h_total + t_total / 2.0,
            TieModel::RaoKupper => h_total + t_total,
        };
        num / den
    }
}

impl Ranker for GeneralizedBt {
    type Data = GamesDataset;
    type Model = GeneralizedBtModel;

    fn fit_opts(&self, data: &GamesDataset, opts: &FitOptions<'_>) -> Result<GeneralizedBtModel> {
        if data.is_empty() {
            return Err(Error::EmptyDataset);
        }
        let pairs = self.aggregate(data)?;
        self.check_ford(&pairs, data.interner())?;
        let n = data.n_entities();

        // Constant per-entity win/tie credit and the global totals the tie
        // and γ updates need.
        let mut credit = vec![0.0f64; n];
        let mut h_total = 0.0;
        let mut t_total = 0.0;
        for p in &pairs {
            let tie_share = match self.ties {
                TieModel::None => 0.0,
                TieModel::Davidson => p.ties / 2.0,
                TieModel::RaoKupper => p.ties,
            };
            credit[p.home as usize] += p.home_wins + tie_share;
            credit[p.away as usize] += p.away_wins + tie_share;
            h_total += p.home_wins;
            t_total += p.ties;
        }

        let mut pi = vec![1.0 / n as f64; n];
        let mut tie = match self.ties {
            TieModel::None => 0.0,
            TieModel::Davidson => 1.0,
            TieModel::RaoKupper => 2.0,
        };
        let mut gamma = 1.0;

        let progress = opts.progress;
        progress.start("generalized-bt sweeps", Some(self.iterations as u64));

        for iter in 0..self.iterations {
            let den = self.pi_denominators(&pairs, &pi, tie, gamma);
            let mut next: Vec<f64> = (0..n).map(|i| credit[i] / den[i]).collect();
            let total: f64 = next.iter().sum();
            next.iter_mut().for_each(|v| *v /= total);

            let d_pi = pi
                .iter()
                .zip(&next)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n as f64;
            pi = next;

            let new_tie = match self.ties {
                TieModel::None => 0.0,
                TieModel::Davidson => Self::davidson_nu(&pairs, &pi, tie, gamma, t_total),
                TieModel::RaoKupper => Self::rao_kupper_theta(&pairs, &pi, gamma, t_total),
            };
            let d_tie = (new_tie - tie).abs();
            tie = new_tie;

            let d_gamma = match self.home {
                HomeAdvantage::None => 0.0,
                HomeAdvantage::Estimate => {
                    let new_gamma = self.gamma_update(&pairs, &pi, tie, gamma, h_total, t_total);
                    let d = (new_gamma - gamma).abs();
                    gamma = new_gamma;
                    d
                }
            };

            let err = d_pi + d_tie + d_gamma;
            progress.update(iter as u64 + 1);

            if iter % 10 == 0 {
                progress.message(&format!("error {err:0.3e}"));
            }
            if err < self.tolerance {
                break;
            }
        }
        progress.finish();

        if pi.iter().any(|v| !v.is_finite()) || !tie.is_finite() || !gamma.is_finite() {
            return Err(Error::Numeric(
                "generalized BT produced non-finite parameters; the comparison \
                 graph is likely too sparse or one-sided"
                    .into(),
            ));
        }

        Ok(GeneralizedBtModel {
            params: *self,
            names: data.interner().clone(),
            scores: pi,
            tie_param: tie,
            gamma,
        })
    }
}

/// What `save_jsonl` writes as the header params: the algorithm params plus
/// the fitted tie parameter and home advantage (state that has no per-entity
/// line to live on).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PersistedParams {
    ties: TieModel,
    home: HomeAdvantage,
    iterations: usize,
    tolerance: f64,
    tie_param: f64,
    gamma: f64,
}

/// Fitted strengths (π normalized to sum to 1) plus the fitted tie parameter
/// and home advantage.
#[derive(Debug, Clone)]
pub struct GeneralizedBtModel {
    params: GeneralizedBt,
    names: Interner,
    scores: Vec<f64>,
    tie_param: f64,
    gamma: f64,
}

impl GeneralizedBtModel {
    /// The fitted tie parameter: Davidson's ν or Rao-Kupper's θ. Under
    /// [`TieModel::None`] there is no tie parameter and this reports `0.0`
    /// (the zero-tie-mass identity; note Rao-Kupper's identity would be 1.0).
    pub fn tie_parameter(&self) -> f64 {
        self.tie_param
    }

    /// The fitted multiplicative home advantage γ on side 1; `1.0` (no edge)
    /// under [`HomeAdvantage::None`].
    pub fn home_advantage(&self) -> f64 {
        self.gamma
    }
}

impl RankModel for GeneralizedBtModel {
    fn algorithm(&self) -> &'static str {
        "generalized-bt"
    }

    fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.names.names().zip(self.scores.iter().copied())
    }

    fn save_jsonl<W: std::io::Write>(&self, w: W) -> Result<()> {
        let params = PersistedParams {
            ties: self.params.ties,
            home: self.params.home,
            iterations: self.params.iterations,
            tolerance: self.params.tolerance,
            tie_param: self.tie_param,
            gamma: self.gamma,
        };
        let lines = score_lines(&self.names, &self.scores);
        state::save_model(w, "generalized-bt", &params, &lines)
    }

    fn load_jsonl<R: std::io::BufRead>(r: R) -> Result<Self> {
        let (params, lines): (PersistedParams, Vec<ScoreLine>) =
            state::load_model(r, "generalized-bt")?;
        let (names, scores) = from_score_lines(lines)?;
        Ok(Self {
            params: GeneralizedBt {
                ties: params.ties,
                home: params.home,
                iterations: params.iterations,
                tolerance: params.tolerance,
            },
            names,
            scores,
            tie_param: params.tie_param,
            gamma: params.gamma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::BradleyTerryMM;
    use crate::dataset::PairwiseDataset;

    fn scores_of(m: &GeneralizedBtModel) -> std::collections::HashMap<String, f64> {
        m.scores().map(|(n, s)| (n.to_string(), s)).collect()
    }

    /// Central-difference derivative of `f` at `x` (relative step).
    fn num_grad(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-6 * x.abs().max(1e-3);
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn tight() -> GeneralizedBt {
        GeneralizedBt {
            iterations: 200_000,
            tolerance: 1e-13,
            ..GeneralizedBt::default()
        }
    }

    /// Two entities, a beats b 5, b beats a 3, 4 ties. The fitted (π, ν)
    /// must zero the Davidson score equations — checked by numerically
    /// differentiating the log-likelihood written out independently here
    /// (the likelihood is scale-invariant in π, so at the constrained MLE
    /// every unconstrained partial vanishes).
    #[test]
    fn davidson_two_entity_stationarity() {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(1.0), 5.0)
            .unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Side2Win(1.0), 3.0)
            .unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 4.0).unwrap();

        let m = tight().fit(&d).unwrap();
        let s = scores_of(&m);
        let (pa, pb, nu) = (s["a"], s["b"], m.tie_parameter());
        assert!(nu > 0.0, "ties present, so ν must be positive: {nu}");

        let ll = |pa: f64, pb: f64, nu: f64| {
            let d = pa + pb + nu * (pa * pb).sqrt();
            5.0 * pa.ln() + 3.0 * pb.ln() + 4.0 * (nu.ln() + 0.5 * (pa * pb).ln()) - 12.0 * d.ln()
        };
        for (name, g) in [
            ("dpa", num_grad(|x| ll(x, pb, nu), pa)),
            ("dpb", num_grad(|x| ll(pa, x, nu), pb)),
            ("dnu", num_grad(|x| ll(pa, pb, x), nu)),
        ] {
            assert!(g.abs() < 1e-4, "{name} = {g} not stationary");
        }
    }

    /// Same counts under Rao-Kupper: the fitted (π, θ) zeroes that model's
    /// score equations.
    #[test]
    fn rao_kupper_two_entity_stationarity() {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(1.0), 5.0)
            .unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Side2Win(1.0), 3.0)
            .unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 4.0).unwrap();

        let m = GeneralizedBt {
            ties: TieModel::RaoKupper,
            ..tight()
        }
        .fit(&d)
        .unwrap();
        let s = scores_of(&m);
        let (pa, pb, th) = (s["a"], s["b"], m.tie_parameter());
        assert!(th > 1.0 && th < 100.0, "θ interior: {th}");

        let ll = |pa: f64, pb: f64, th: f64| {
            let big_a = pa + th * pb;
            let big_b = pb + th * pa;
            5.0 * (pa.ln() - big_a.ln())
                + 3.0 * (pb.ln() - big_b.ln())
                + 4.0 * (pa.ln() + pb.ln() + (th * th - 1.0).ln() - big_a.ln() - big_b.ln())
        };
        for (name, g) in [
            ("dpa", num_grad(|x| ll(x, pb, th), pa)),
            ("dpb", num_grad(|x| ll(pa, x, th), pb)),
            ("dth", num_grad(|x| ll(pa, pb, x), th)),
        ] {
            assert!(g.abs() < 1e-4, "{name} = {g} not stationary");
        }
    }

    /// With no ties and no home advantage the chassis is plain BT: strength
    /// ratios match Hunter's MM fit on the same comparisons.
    #[test]
    fn no_extensions_matches_plain_bt() {
        let rows: &[(&str, &str, f32)] = &[
            ("a", "b", 5.0),
            ("b", "a", 2.0),
            ("a", "c", 4.0),
            ("c", "a", 3.0),
            ("b", "c", 6.0),
            ("c", "b", 2.0),
            ("c", "d", 3.0),
            ("d", "c", 1.0),
            ("d", "a", 1.0),
            ("a", "d", 2.0),
            ("b", "d", 2.0),
            ("d", "b", 1.0),
        ];
        let mut games = GamesDataset::new();
        let mut pairs = PairwiseDataset::new();
        for &(w, l, x) in rows {
            games.push_pair(w, l, x).unwrap();
            pairs.push(w, l, x);
        }

        let reference = BradleyTerryMM {
            iterations: 50_000,
            tolerance: 1e-13,
            ..BradleyTerryMM::default()
        }
        .fit(&pairs)
        .unwrap();
        let ref_s: std::collections::HashMap<_, _> = reference.scores().collect();

        let m = GeneralizedBt {
            ties: TieModel::None,
            ..tight()
        }
        .fit(&games)
        .unwrap();
        let s = scores_of(&m);

        for (x, y) in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")] {
            let got = s[x] / s[y];
            let want = ref_s[x] / ref_s[y];
            assert!(
                (got - want).abs() < 1e-6,
                "{x}/{y}: {got} vs plain BT {want}"
            );
        }
        assert!((m.tie_parameter() - 0.0).abs() < 1e-12);
        assert!((m.home_advantage() - 1.0).abs() < 1e-12);
    }

    /// Perfectly venue-symmetric data: at every venue the home side wins
    /// with weight 2 and loses with weight 1, for every ordered pair. The
    /// MLE is uniform π and exactly γ = 2 (γ score equation:
    /// `H/γ = Σ n/(γ+1)` ⇒ `12(γ+1) = 18γ`). Swapping the weights flips the
    /// edge to the visitors: γ = 1/2.
    #[test]
    fn home_advantage_symmetric_fixture() {
        let fixture = |home_w: f32, away_w: f32| {
            let mut d = GamesDataset::new();
            let names = ["a", "b", "c"];
            for h in names {
                for a in names {
                    if h != a {
                        d.push_game(&[h], &[a], GameOutcome::Side1Win(1.0), home_w)
                            .unwrap();
                        d.push_game(&[h], &[a], GameOutcome::Side2Win(1.0), away_w)
                            .unwrap();
                    }
                }
            }
            d
        };
        let algo = GeneralizedBt {
            ties: TieModel::None,
            home: HomeAdvantage::Estimate,
            ..tight()
        };

        let m = algo.fit(&fixture(2.0, 1.0)).unwrap();
        for (_, s) in m.scores() {
            assert!((s - 1.0 / 3.0).abs() < 1e-9, "uniform strengths: {s}");
        }
        assert!(m.home_advantage() > 1.0);
        assert!(
            (m.home_advantage() - 2.0).abs() < 1e-9,
            "{}",
            m.home_advantage()
        );

        let m = algo.fit(&fixture(1.0, 2.0)).unwrap();
        assert!(m.home_advantage() < 1.0);
        assert!(
            (m.home_advantage() - 0.5).abs() < 1e-9,
            "{}",
            m.home_advantage()
        );
    }

    /// Asymmetric venue data: the fitted (π, γ) zeroes the γ score equation
    /// (and the π ones), again via numerical gradients of an independently
    /// written likelihood.
    #[test]
    fn home_model_stationarity() {
        // (home, away, home wins, away wins)
        let table: &[(&str, &str, f32, f32)] = &[
            ("a", "b", 3.0, 1.0),
            ("b", "a", 2.0, 1.0),
            ("a", "c", 4.0, 1.0),
            ("c", "a", 1.0, 2.0),
            ("b", "c", 3.0, 2.0),
            ("c", "b", 1.0, 3.0),
        ];
        let mut d = GamesDataset::new();
        for &(h, a, w, v) in table {
            d.push_game(&[h], &[a], GameOutcome::Side1Win(1.0), w)
                .unwrap();
            d.push_game(&[h], &[a], GameOutcome::Side2Win(1.0), v)
                .unwrap();
        }

        let m = GeneralizedBt {
            ties: TieModel::None,
            home: HomeAdvantage::Estimate,
            ..tight()
        }
        .fit(&d)
        .unwrap();
        let s = scores_of(&m);
        let gamma = m.home_advantage();

        let ll = |pi: &std::collections::HashMap<String, f64>, gamma: f64| {
            table
                .iter()
                .map(|&(h, a, w, v)| {
                    let (ph, pa) = (pi[h], pi[a]);
                    f64::from(w) * (gamma * ph).ln() + f64::from(v) * pa.ln()
                        - f64::from(w + v) * (gamma * ph + pa).ln()
                })
                .sum::<f64>()
        };

        let g_gamma = num_grad(|x| ll(&s, x), gamma);
        assert!(g_gamma.abs() < 1e-4, "dγ = {g_gamma} not stationary");

        for name in ["a", "b", "c"] {
            let g = num_grad(
                |x| {
                    let mut perturbed = s.clone();
                    perturbed.insert(name.to_string(), x);
                    ll(&perturbed, gamma)
                },
                s[name],
            );
            assert!(g.abs() < 1e-4, "d{name} = {g} not stationary");
        }
    }

    #[test]
    fn tie_under_tie_model_none_is_invalid_input() {
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        d.push_pair("b", "a", 1.0).unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 1.0).unwrap();
        let err = GeneralizedBt {
            ties: TieModel::None,
            ..GeneralizedBt::default()
        }
        .fit(&d)
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(m) if m.contains("tie")));
    }

    #[test]
    fn ford_violation_names_offenders() {
        let mut d = GamesDataset::new();
        d.push_pair("a", "b", 1.0).unwrap();
        d.push_pair("b", "a", 1.0).unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 1.0).unwrap();
        d.push_pair("champ", "a", 3.0).unwrap(); // champ never loses or ties
        let err = GeneralizedBt::default().fit(&d).unwrap_err();
        assert!(matches!(err, Error::Numeric(m) if m.contains("champ")));
    }

    #[test]
    fn empty_and_multiplayer_are_typed_errors() {
        assert!(matches!(
            GeneralizedBt::default().fit(&GamesDataset::new()),
            Err(Error::EmptyDataset)
        ));

        let mut d = GamesDataset::new();
        d.push_game(&["a", "b"], &["c"], GameOutcome::Side1Win(1.0), 1.0)
            .unwrap();
        assert!(matches!(
            GeneralizedBt::default().fit(&d),
            Err(Error::InvalidInput(m)) if m.contains("team-bradley-terry")
        ));
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut d = GamesDataset::new();
        d.push_game(&["a"], &["b"], GameOutcome::Side1Win(1.0), 3.0)
            .unwrap();
        d.push_game(&["b"], &["a"], GameOutcome::Side1Win(1.0), 2.0)
            .unwrap();
        d.push_game(&["a"], &["b"], GameOutcome::Tie, 2.0).unwrap();
        d.push_game(&["b"], &["a"], GameOutcome::Tie, 1.0).unwrap();

        let m = GeneralizedBt {
            home: HomeAdvantage::Estimate,
            ..GeneralizedBt::default()
        }
        .fit(&d)
        .unwrap();

        let mut buf = Vec::new();
        m.save_jsonl(&mut buf).unwrap();
        let m2 = GeneralizedBtModel::load_jsonl(buf.as_slice()).unwrap();
        let mut buf2 = Vec::new();
        m2.save_jsonl(&mut buf2).unwrap();
        assert_eq!(buf, buf2);

        // The fitted extension parameters ride along.
        assert_eq!(m.tie_parameter(), m2.tie_parameter());
        assert_eq!(m.home_advantage(), m2.home_advantage());
    }
}

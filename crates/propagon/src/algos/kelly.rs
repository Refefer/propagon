//! Kelly-criterion stake sizing (`docs/algorithms.md` §14.4).
//!
//! The *act-on-value* layer: given a calibrated win probability and the odds on
//! offer, return the fraction of bankroll that maximizes long-run growth. These
//! are free functions over plain numbers (a probability from any ranker, the
//! market's net decimal odds), not a [`RankModel`](crate::RankModel) — Kelly is
//! a decision rule, not a ranker.
//!
//! Conventions: `b` is the **net** decimal odds (profit per unit staked on a
//! win, i.e. `decimal_odds − 1`); a winning unit stake returns `1 + b`, a losing
//! one returns `0`. `p` is the win probability and `q = 1 − p`.
//!
//! References: [Kelly 1956]; [Breiman 1961]; Thorp, *The Kelly Criterion in
//! Blackjack, Sports Betting, and the Stock Market*. Worked values are pinned in
//! `tests/reference.rs`.

use crate::error::{Error, Result};

/// The growth-optimal stake fraction for a single bet.
///
/// `f* = (b·p − q) / b = (p(b + 1) − 1) / b`, where `b` is the net decimal odds
/// and `q = 1 − p`. A non-positive edge yields `0.0` (do not bet) rather than a
/// negative stake. Errors on `p ∉ [0, 1]` or `b ≤ 0` (or non-finite inputs).
pub fn kelly_fraction(p: f64, b: f64) -> Result<f64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(Error::InvalidInput(format!(
            "kelly: probability must be in [0, 1], got {p}"
        )));
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "kelly: net decimal odds must be positive, got {b}"
        )));
    }
    let f = (p * (b + 1.0) - 1.0) / b;
    Ok(f.max(0.0))
}

/// `λ · f*` — fractional Kelly, the standard hedge against an *estimated* `p`.
///
/// `λ ∈ {½, ¼}` are common; `λ` outside `[0, 1]` is allowed (the caller owns the
/// risk) but `λ < 0` is rejected as nonsensical. Same input validation as
/// [`kelly_fraction`].
pub fn fractional_kelly(p: f64, b: f64, lambda: f64) -> Result<f64> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::InvalidInput(format!(
            "kelly: fraction must be non-negative, got {lambda}"
        )));
    }
    Ok(lambda * kelly_fraction(p, b)?)
}

/// One simultaneous betting opportunity: win probability `p` at net decimal
/// odds `b` (win returns `1 + b` per unit, loss returns `0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opportunity {
    /// Win probability, in `[0, 1]`.
    pub p: f64,
    /// Net decimal odds (profit per unit on a win), `> 0`.
    pub b: f64,
}

/// Largest portfolio the joint-outcome enumeration will handle (`2^K` terms).
pub const MAX_PORTFOLIO_BETS: usize = 12;

/// Growth-optimal stakes for several **independent**, simultaneous bets sharing
/// one bankroll.
///
/// Maximizes `E[log(1 + Σₖ fₖ Xₖ)]` over `fₖ ≥ 0, Σ fₖ ≤ 1`, where `Xₖ` is bet
/// `k`'s net return (`+bₖ` on a win, `−1` on a loss). The expectation is taken
/// over the `2^K` joint win/lose outcomes (independence), and the concave
/// objective is maximized by projected-gradient ascent onto the capped simplex.
/// For independent events whose individual Kelly stakes already sum to `≤ 1` the
/// optimum decouples to the per-bet [`kelly_fraction`]; the bankroll cap binds
/// only when they would over-commit.
///
/// Errors on more than [`MAX_PORTFOLIO_BETS`] opportunities (the `2^K` blow-up;
/// decouple eventwise instead), an empty slice, or any invalid `(p, b)`.
pub fn portfolio_kelly(opps: &[Opportunity]) -> Result<Vec<f64>> {
    if opps.is_empty() {
        return Err(Error::InvalidInput("kelly: no opportunities".into()));
    }
    if opps.len() > MAX_PORTFOLIO_BETS {
        return Err(Error::InvalidInput(format!(
            "kelly: {} opportunities exceeds the {MAX_PORTFOLIO_BETS}-bet enumeration cap; \
             decouple independent events and size them separately",
            opps.len()
        )));
    }
    for o in opps {
        // Reuse the single-bet validation (ignore the value, keep the checks).
        kelly_fraction(o.p, o.b)?;
    }

    let k = opps.len();
    let mut f = vec![0.0; k];
    let mut step = 1.0;
    const MAX_ITER: usize = 4000;
    const TOL: f64 = 1e-12;

    for _ in 0..MAX_ITER {
        let (j, grad) = eval(opps, &f).ok_or_else(|| {
            Error::Numeric("kelly: portfolio objective left the feasible region".into())
        })?;

        // Backtracking line search: shrink the step until the projected point
        // is feasible and does not decrease the (concave) objective.
        let mut moved = 0.0;
        let mut accepted = false;
        for _ in 0..80 {
            let trial: Vec<f64> = f
                .iter()
                .zip(&grad)
                .map(|(&fi, &gi)| fi + step * gi)
                .collect();
            let proj = project_capped_simplex(&trial);
            if let Some((jt, _)) = eval(opps, &proj)
                && jt + 1e-15 >= j
            {
                moved = proj
                    .iter()
                    .zip(&f)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                f = proj;
                step *= 1.25;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted || moved < TOL {
            break;
        }
    }
    Ok(f)
}

/// The objective `J(f) = Σ_ω P(ω) ln W(ω)` and its gradient over the `2^K`
/// joint outcomes. Returns `None` if any wealth multiplier is non-positive
/// (`ln` undefined) — the caller treats that as infeasible and backs off.
fn eval(opps: &[Opportunity], f: &[f64]) -> Option<(f64, Vec<f64>)> {
    let k = opps.len();
    let mut j = 0.0;
    let mut grad = vec![0.0; k];
    for mask in 0u32..(1u32 << k) {
        let mut prob = 1.0;
        let mut w = 1.0;
        for (i, o) in opps.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                prob *= o.p;
                w += f[i] * o.b;
            } else {
                prob *= 1.0 - o.p;
                w -= f[i];
            }
        }
        if w <= 0.0 {
            return None;
        }
        j += prob * w.ln();
        for (i, o) in opps.iter().enumerate() {
            let dw = if (mask >> i) & 1 == 1 { o.b } else { -1.0 };
            grad[i] += prob * dw / w;
        }
    }
    Some((j, grad))
}

/// Euclidean projection onto `{ x ≥ 0, Σ x ≤ 1 }`.
fn project_capped_simplex(v: &[f64]) -> Vec<f64> {
    let clamped: Vec<f64> = v.iter().map(|&x| x.max(0.0)).collect();
    if clamped.iter().sum::<f64>() <= 1.0 {
        return clamped;
    }
    project_probability_simplex(v)
}

/// Euclidean projection onto `{ x ≥ 0, Σ x = 1 }` (Duchi et al. 2008).
fn project_probability_simplex(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    u.sort_unstable_by(|a, b| b.total_cmp(a));
    let mut css = 0.0;
    let mut theta = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        css += ui;
        let t = (css - 1.0) / (i as f64 + 1.0);
        if ui - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|&x| (x - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_bet_closed_forms() {
        // Even money, 60% edge: f* = 0.2 (the canonical Kelly example).
        assert!((kelly_fraction(0.6, 1.0).unwrap() - 0.2).abs() < 1e-12);
        // No edge or negative edge ⇒ no bet.
        assert_eq!(kelly_fraction(0.5, 1.0).unwrap(), 0.0);
        assert_eq!(kelly_fraction(0.4, 1.0).unwrap(), 0.0);
        // Longer odds.
        assert!((kelly_fraction(0.6, 2.0).unwrap() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn fractional_halves_the_stake() {
        assert!((fractional_kelly(0.6, 1.0, 0.5).unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(kelly_fraction(1.5, 1.0).is_err());
        assert!(kelly_fraction(0.6, 0.0).is_err());
        assert!(fractional_kelly(0.6, 1.0, -0.1).is_err());
        assert!(portfolio_kelly(&[]).is_err());
    }

    #[test]
    fn portfolio_reduces_to_single_bet() {
        for (p, b) in [(0.6, 1.0), (0.55, 2.0), (0.7, 0.5)] {
            let one = portfolio_kelly(&[Opportunity { p, b }]).unwrap();
            assert!(
                (one[0] - kelly_fraction(p, b).unwrap()).abs() < 1e-7,
                "{p},{b}"
            );
        }
    }

    #[test]
    fn portfolio_is_optimal_and_shrinks_stakes() {
        // Two independent small-edge bets. Settled simultaneously, the joint
        // log-objective couples them, so each optimal stake is no larger than
        // its standalone Kelly fraction (diversification shrinks stakes).
        let opps = [
            Opportunity { p: 0.55, b: 1.0 },
            Opportunity { p: 0.50, b: 3.0 },
        ];
        let f = portfolio_kelly(&opps).unwrap();
        for (fi, o) in f.iter().zip(&opps) {
            let solo = kelly_fraction(o.p, o.b).unwrap();
            assert!(*fi >= 0.0 && *fi <= solo + 1e-9, "fi={fi}, solo={solo}");
        }
        // It beats the naive eventwise allocation in expected log-growth, and
        // the interior optimum has a (near-)zero gradient.
        let eventwise: Vec<f64> = opps
            .iter()
            .map(|o| kelly_fraction(o.p, o.b).unwrap())
            .collect();
        let (j_opt, grad) = eval(&opps, &f).unwrap();
        let (j_event, _) = eval(&opps, &eventwise).unwrap();
        assert!(
            j_opt + 1e-12 >= j_event,
            "opt {j_opt} vs eventwise {j_event}"
        );
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(gnorm < 1e-4, "gradient norm {gnorm}");
    }
}

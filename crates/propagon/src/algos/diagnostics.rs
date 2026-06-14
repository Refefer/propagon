//! Market-efficiency & calibration diagnostics (`docs/algorithms.md` §14.5,
//! §15.5).
//!
//! Proper scoring rules ([`brier_score`], [`log_loss`]) and two betting-market
//! checks: [`closing_line_value`] scores a price against the sharp close, and
//! [`calibration_table`] buckets predicted vs realized frequency to expose the
//! favorite-longshot bias. All are free functions over plain forecast/outcome
//! arrays — evaluation, not ranking.
//!
//! References: [Brier 1950]; [Gneiting & Raftery 2007]; [Snowberg & Wolfers
//! 2010]. Worked values are pinned in `tests/reference.rs`.

use crate::error::{Error, Result};

fn check_lengths(forecasts: &[f64], outcomes: &[bool]) -> Result<()> {
    if forecasts.len() != outcomes.len() {
        return Err(Error::InvalidInput(format!(
            "diagnostics: {} forecasts but {} outcomes",
            forecasts.len(),
            outcomes.len()
        )));
    }
    if forecasts.is_empty() {
        return Err(Error::EmptyDataset);
    }
    for &p in forecasts {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidInput(format!(
                "diagnostics: forecast probability must be in [0, 1], got {p}"
            )));
        }
    }
    Ok(())
}

/// The binary Brier score: mean squared error between forecast probabilities
/// and `{0, 1}` outcomes, `BS = (1/N) Σ (pₜ − oₜ)²` ([Brier 1950]). Range
/// `[0, 1]`, lower is better. Errors on length mismatch, empty input, or a
/// forecast outside `[0, 1]`.
pub fn brier_score(forecasts: &[f64], outcomes: &[bool]) -> Result<f64> {
    check_lengths(forecasts, outcomes)?;
    let sum: f64 = forecasts
        .iter()
        .zip(outcomes)
        .map(|(&p, &o)| {
            let y = f64::from(u8::from(o));
            (p - y).powi(2)
        })
        .sum();
    Ok(sum / forecasts.len() as f64)
}

/// Mean binary log-loss (cross-entropy):
/// `LL = −(1/N) Σ [oₜ ln pₜ + (1 − oₜ) ln(1 − pₜ)]`. Forecasts are clamped to
/// `[eps, 1 − eps]` (default `eps = 1e-15`) so a confident miss is a large
/// finite penalty rather than `+∞`. Lower is better. Same input validation as
/// [`brier_score`].
pub fn log_loss(forecasts: &[f64], outcomes: &[bool]) -> Result<f64> {
    log_loss_eps(forecasts, outcomes, 1e-15)
}

/// [`log_loss`] with an explicit clamp `eps ∈ (0, 0.5)`.
pub fn log_loss_eps(forecasts: &[f64], outcomes: &[bool], eps: f64) -> Result<f64> {
    check_lengths(forecasts, outcomes)?;
    if !eps.is_finite() || !(0.0..0.5).contains(&eps) {
        return Err(Error::InvalidInput(format!(
            "diagnostics: log-loss eps must be in (0, 0.5), got {eps}"
        )));
    }
    let sum: f64 = forecasts
        .iter()
        .zip(outcomes)
        .map(|(&p, &o)| {
            let p = p.clamp(eps, 1.0 - eps);
            if o { p.ln() } else { (1.0 - p).ln() }
        })
        .sum();
    Ok(-sum / forecasts.len() as f64)
}

/// Closing-line value: the beat-the-close return of a price you took versus the
/// sharp closing price, both as **decimal** odds.
///
/// `CLV = taken / closing − 1`. Positive means you locked in a longer price than
/// the close (you beat it); zero means you matched it; negative means the market
/// moved past your price. (Note the opposite sign of the raw implied-probability
/// difference `1/closing − 1/taken`.) Errors on non-positive or non-finite odds.
pub fn closing_line_value(taken_decimal_odds: f64, closing_decimal_odds: f64) -> Result<f64> {
    if !taken_decimal_odds.is_finite() || taken_decimal_odds <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "clv: taken odds must be positive, got {taken_decimal_odds}"
        )));
    }
    if !closing_decimal_odds.is_finite() || closing_decimal_odds <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "clv: closing odds must be positive, got {closing_decimal_odds}"
        )));
    }
    Ok(taken_decimal_odds / closing_decimal_odds - 1.0)
}

/// One bin of a [`calibration_table`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationBin {
    /// Inclusive lower edge of the implied-probability bucket.
    pub lo: f64,
    /// Exclusive upper edge (inclusive for the final bucket).
    pub hi: f64,
    /// Mean predicted probability of the forecasts that fell in this bucket.
    pub mean_pred: f64,
    /// Realized frequency: the fraction whose outcome was true.
    pub realized_freq: f64,
    /// Number of forecasts in this bucket.
    pub count: usize,
}

/// Buckets forecasts into `n_buckets` equal-width bins on `[0, 1]` and reports,
/// per bin, the mean predicted probability against the realized frequency — the
/// calibration curve that exposes the favorite-longshot bias (longshots tend to
/// realize *below* their implied probability, favorites *above*). Empty bins are
/// returned with `count = 0` and `NaN` summaries. Errors on `n_buckets = 0` or
/// the usual length/range problems.
pub fn calibration_table(
    implied: &[f64],
    outcomes: &[bool],
    n_buckets: usize,
) -> Result<Vec<CalibrationBin>> {
    check_lengths(implied, outcomes)?;
    if n_buckets == 0 {
        return Err(Error::InvalidInput(
            "calibration: n_buckets must be > 0".into(),
        ));
    }

    let n = n_buckets as f64;
    let mut sum_pred = vec![0.0; n_buckets];
    let mut hits = vec![0usize; n_buckets];
    let mut count = vec![0usize; n_buckets];

    for (&p, &o) in implied.iter().zip(outcomes) {
        // Final bucket is closed on the right so p == 1.0 lands in it.
        let mut b = (p * n).floor() as usize;
        if b >= n_buckets {
            b = n_buckets - 1;
        }
        sum_pred[b] += p;
        count[b] += 1;
        if o {
            hits[b] += 1;
        }
    }

    Ok((0..n_buckets)
        .map(|b| {
            let c = count[b];
            let (mean_pred, realized_freq) = if c == 0 {
                (f64::NAN, f64::NAN)
            } else {
                (sum_pred[b] / c as f64, hits[b] as f64 / c as f64)
            };
            CalibrationBin {
                lo: b as f64 / n,
                hi: (b + 1) as f64 / n,
                mean_pred,
                realized_freq,
                count: c,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brier_exact() {
        // (0.1² + 0.2² + 0.3²) / 3 = 0.14 / 3.
        let bs = brier_score(&[0.9, 0.2, 0.7], &[true, false, true]).unwrap();
        assert!((bs - 0.14 / 3.0).abs() < 1e-12, "got {bs}");
    }

    #[test]
    fn log_loss_exact() {
        let ll = log_loss(&[0.9, 0.2, 0.7], &[true, false, true]).unwrap();
        let want = -(0.9_f64.ln() + 0.8_f64.ln() + 0.7_f64.ln()) / 3.0;
        assert!((ll - want).abs() < 1e-12, "got {ll}");
    }

    #[test]
    fn clv_sign_convention() {
        // Took 2.10, close 2.00: beat the close ⇒ positive.
        assert!((closing_line_value(2.10, 2.00).unwrap() - 0.05).abs() < 1e-12);
        // The reverse is negative.
        assert!((closing_line_value(2.00, 2.10).unwrap() + 0.047619047619047616).abs() < 1e-12);
    }

    #[test]
    fn calibration_buckets() {
        let table = calibration_table(&[0.05, 0.15, 0.95], &[false, false, true], 10).unwrap();
        assert_eq!(table.len(), 10);
        assert_eq!(table[0].count, 1);
        assert_eq!(table[1].count, 1);
        assert_eq!(table[9].count, 1);
        assert_eq!(table[5].count, 0);
        assert!(table[5].mean_pred.is_nan());
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(brier_score(&[0.5], &[true, false]).is_err());
        assert!(log_loss(&[], &[]).is_err());
    }
}

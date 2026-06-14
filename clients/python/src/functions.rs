//! Module-level free functions: Wilson intervals, connected-component
//! extraction, Kelly stake sizing, and calibration diagnostics (§14.4/§14.5).

use pyo3::prelude::*;

use crate::datasets::GraphDataset;
use crate::errors::MapPy;

/// The Wilson score interval for a binomial proportion.
///
/// Returns `(low, high)` for `successes` wins out of `successes + failures`
/// trials at the given `z` (default 1.96 ≈ 95%).
#[pyfunction]
#[pyo3(signature = (successes, failures, z = 1.96))]
fn wilson_interval(successes: f64, failures: f64, z: f64) -> (f64, f64) {
    propagon::algos::wilson_interval(successes, failures, z)
}

/// Splits a graph into its connected components, dropping any with fewer than
/// `min_size` nodes. Returns a list of independent `GraphDataset`s.
#[pyfunction]
#[pyo3(signature = (graph, min_size = 1))]
fn extract_components(graph: &GraphDataset, min_size: usize) -> Vec<GraphDataset> {
    propagon::algos::extract_components(graph.inner.view(), min_size)
        .into_iter()
        .map(|inner| GraphDataset { inner })
        .collect()
}

/// The growth-optimal Kelly stake fraction for a single bet at net decimal odds
/// `b` (= decimal odds − 1) with win probability `p`. A non-positive edge
/// returns 0.0. Raises `InvalidInputError` on `p ∉ [0, 1]` or `b ≤ 0`.
#[pyfunction]
fn kelly_fraction(p: f64, b: f64) -> PyResult<f64> {
    propagon::algos::kelly_fraction(p, b).map_py()
}

/// `lambda * kelly_fraction(p, b)` — fractional Kelly (e.g. lambda=0.5 for half).
#[pyfunction]
fn fractional_kelly(p: f64, b: f64, lambda: f64) -> PyResult<f64> {
    propagon::algos::fractional_kelly(p, b, lambda).map_py()
}

/// Growth-optimal stakes for several independent simultaneous bets, each given
/// as `(win_probability, net_decimal_odds)`. Maximizes expected log-growth over
/// the shared bankroll. Raises on more than 12 bets or any invalid `(p, b)`.
#[pyfunction]
fn portfolio_kelly(opportunities: Vec<(f64, f64)>) -> PyResult<Vec<f64>> {
    let opps: Vec<propagon::algos::Opportunity> = opportunities
        .into_iter()
        .map(|(p, b)| propagon::algos::Opportunity { p, b })
        .collect();
    propagon::algos::portfolio_kelly(&opps).map_py()
}

/// The binary Brier score (mean squared error of forecast vs `{0,1}` outcome).
#[pyfunction]
fn brier_score(forecasts: Vec<f64>, outcomes: Vec<bool>) -> PyResult<f64> {
    propagon::algos::brier_score(&forecasts, &outcomes).map_py()
}

/// Mean binary log-loss (cross-entropy), with forecasts clamped away from 0/1.
#[pyfunction]
fn log_loss(forecasts: Vec<f64>, outcomes: Vec<bool>) -> PyResult<f64> {
    propagon::algos::log_loss(&forecasts, &outcomes).map_py()
}

/// Closing-line value `taken / closing − 1` (positive = beat the close).
#[pyfunction]
fn closing_line_value(taken_decimal_odds: f64, closing_decimal_odds: f64) -> PyResult<f64> {
    propagon::algos::closing_line_value(taken_decimal_odds, closing_decimal_odds).map_py()
}

/// One calibration bin as `(lo, hi, mean_pred, realized_freq, count)`.
type CalibrationRow = (f64, f64, f64, f64, usize);

/// A calibration table: buckets implied probabilities into `n_buckets`
/// equal-width bins and reports, per bin, `(lo, hi, mean_pred, realized_freq,
/// count)`. Empty bins report `NaN` summaries.
#[pyfunction]
#[pyo3(signature = (implied, outcomes, n_buckets = 10))]
fn calibration_table(
    implied: Vec<f64>,
    outcomes: Vec<bool>,
    n_buckets: usize,
) -> PyResult<Vec<CalibrationRow>> {
    let table = propagon::algos::calibration_table(&implied, &outcomes, n_buckets).map_py()?;
    Ok(table
        .into_iter()
        .map(|b| (b.lo, b.hi, b.mean_pred, b.realized_freq, b.count))
        .collect())
}

/// Registers the free functions on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wilson_interval, m)?)?;
    m.add_function(wrap_pyfunction!(extract_components, m)?)?;
    m.add_function(wrap_pyfunction!(crate::load::load_state, m)?)?;
    m.add_function(wrap_pyfunction!(kelly_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(fractional_kelly, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_kelly, m)?)?;
    m.add_function(wrap_pyfunction!(brier_score, m)?)?;
    m.add_function(wrap_pyfunction!(log_loss, m)?)?;
    m.add_function(wrap_pyfunction!(closing_line_value, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_table, m)?)?;
    Ok(())
}

/// Names this module contributes to `__all__`.
pub(crate) const EXPORTS: &[&str] = &[
    "wilson_interval",
    "extract_components",
    "load_state",
    "kelly_fraction",
    "fractional_kelly",
    "portfolio_kelly",
    "brier_score",
    "log_loss",
    "closing_line_value",
    "calibration_table",
];

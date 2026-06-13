//! Module-level free functions: Wilson intervals and connected-component
//! extraction.

use pyo3::prelude::*;

use crate::datasets::GraphDataset;

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

/// Registers the free functions on the module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wilson_interval, m)?)?;
    m.add_function(wrap_pyfunction!(extract_components, m)?)?;
    m.add_function(wrap_pyfunction!(crate::load::load_state, m)?)?;
    Ok(())
}

/// Names this module contributes to `__all__`.
pub(crate) const EXPORTS: &[&str] = &["wilson_interval", "extract_components", "load_state"];

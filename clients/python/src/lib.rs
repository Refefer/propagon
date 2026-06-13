//! Python bindings for the `propagon` ranking library (PyO3 + maturin).
//!
//! The compiled extension is imported as `propagon._propagon`; the pure-Python
//! `propagon/__init__.py` shim re-exports its surface. See the crate README for
//! the public API.

// Several algorithm constructors mirror params structs with 8+ fields; the
// kwarg-per-field shape is the binding's whole point, so the arity lint does
// not apply. (A per-fn `#[allow]` is dropped by the `#[pymethods]` macro, so
// this is set crate-wide.)
#![allow(clippy::too_many_arguments)]

// `#[macro_use]` (not a path import) so the binding macros are in textual scope
// crate-wide, including the nested `scalar_params_body!` calls inside
// `scalar_online!`/`scalar_batch!`. Must precede the modules that use them.
#[macro_use]
mod macros;

mod algos;
mod convert;
mod datasets;
mod enums;
mod errors;
mod functions;
mod load;

use pyo3::prelude::*;

/// Assembles the package's `__all__` from each module's contribution.
fn all_exports() -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for group in [
        datasets::EXPORTS,
        enums::EXPORTS,
        algos::EXPORTS,
        functions::EXPORTS,
        errors::EXPORTS,
    ] {
        names.extend(group.iter().map(|s| (*s).to_string()));
    }
    names
}

/// The `_propagon` extension module (re-exported by the `propagon` package).
#[pymodule]
fn _propagon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    datasets::register(m)?;
    enums::register(m)?;
    algos::register(m)?;
    functions::register(m)?;
    errors::register(m)?;

    m.add("__all__", all_exports())?;
    Ok(())
}

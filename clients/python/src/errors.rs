//! The Python exception hierarchy and the `propagon::Error` → `PyErr` mapping.
//!
//! Rust's orphan rule forbids `impl From<propagon::Error> for PyErr` (both
//! types are foreign to this crate), so conversion goes through the free
//! [`to_pyerr`] function and the ergonomic [`MapPy`] extension trait
//! (`some_result.map_py()?`) instead of `?`-driven `From`.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    _propagon,
    PropagonError,
    PyException,
    "Base class for every error raised by propagon."
);
create_exception!(
    _propagon,
    InvalidInputError,
    PropagonError,
    "An argument was out of range, the wrong shape, or otherwise unacceptable."
);
create_exception!(
    _propagon,
    EmptyDatasetError,
    PropagonError,
    "An algorithm was asked to fit a dataset with no usable rows."
);
create_exception!(
    _propagon,
    NumericError,
    PropagonError,
    "The numerics failed: no convergence, a singular system, or a disconnected graph."
);
create_exception!(
    _propagon,
    StateError,
    PropagonError,
    "A saved state file was structurally invalid, mis-versioned, or unparseable."
);
create_exception!(
    _propagon,
    AlgorithmMismatchError,
    StateError,
    "A state file was produced by a different algorithm than the one loading it."
);
create_exception!(
    _propagon,
    ParamMismatchError,
    PropagonError,
    "Resumed state carries parameters incompatible with the current configuration."
);
create_exception!(
    _propagon,
    IoError,
    PropagonError,
    "An underlying I/O or JSON (de)serialization operation failed."
);

/// Maps a [`propagon::Error`] to the matching Python exception.
///
/// The `_` arm is load-bearing: `propagon::Error` is `#[non_exhaustive]`, so a
/// future variant must still compile here and degrade to the base class. Do not
/// remove it.
pub(crate) fn to_pyerr(err: propagon::Error) -> PyErr {
    use propagon::Error as E;
    let msg = err.to_string();
    match err {
        E::InvalidInput(_) => InvalidInputError::new_err(msg),
        E::EmptyDataset => EmptyDatasetError::new_err(msg),
        E::Numeric(_) => NumericError::new_err(msg),
        E::AlgorithmMismatch { .. } => AlgorithmMismatchError::new_err(msg),
        E::State(_) | E::Parse { .. } | E::Version { .. } => StateError::new_err(msg),
        E::ParamMismatch(_) => ParamMismatchError::new_err(msg),
        E::Io(_) | E::Json(_) => IoError::new_err(msg),
        // #[non_exhaustive]: any future variant maps to the base class.
        _ => PropagonError::new_err(msg),
    }
}

/// Ergonomic conversion of a `propagon::Result<T>` into a `PyResult<T>`.
///
/// Lets binding code write `expr.map_py()?` where the orphan rule blocks a
/// plain `?`.
pub(crate) trait MapPy<T> {
    /// Converts the error with [`to_pyerr`].
    fn map_py(self) -> PyResult<T>;
}

impl<T> MapPy<T> for propagon::Result<T> {
    fn map_py(self) -> PyResult<T> {
        self.map_err(to_pyerr)
    }
}

/// Registers every exception type on the module so they are importable as
/// `propagon.PropagonError`, `propagon.InvalidInputError`, …
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PropagonError", m.py().get_type::<PropagonError>())?;
    m.add("InvalidInputError", m.py().get_type::<InvalidInputError>())?;
    m.add("EmptyDatasetError", m.py().get_type::<EmptyDatasetError>())?;
    m.add("NumericError", m.py().get_type::<NumericError>())?;
    m.add("StateError", m.py().get_type::<StateError>())?;
    m.add(
        "AlgorithmMismatchError",
        m.py().get_type::<AlgorithmMismatchError>(),
    )?;
    m.add(
        "ParamMismatchError",
        m.py().get_type::<ParamMismatchError>(),
    )?;
    m.add("IoError", m.py().get_type::<IoError>())?;
    Ok(())
}

/// The names this module contributes to the package's `__all__`.
pub(crate) const EXPORTS: &[&str] = &[
    "PropagonError",
    "InvalidInputError",
    "EmptyDatasetError",
    "NumericError",
    "StateError",
    "AlgorithmMismatchError",
    "ParamMismatchError",
    "IoError",
];

//! `propagon::Error` -> WIT `error` mapping.
//!
//! Rust's orphan rule forbids `impl From<propagon::Error> for wit::Error` (both
//! foreign to the generated module from this file's view), so conversion goes
//! through the free [`to_wit_err`] function and the [`MapWit`] extension trait
//! (`some_result.map_wit()?`), mirroring the Python client's `MapPy`/`to_pyerr`.

use crate::bindings::exports::propagon::core::types::{
    Error, Mismatch, ParseError, VersionMismatch,
};

/// Maps a [`propagon::Error`] to the matching WIT `error` variant.
///
/// The `_` arm is load-bearing: `propagon::Error` is `#[non_exhaustive]`, so a
/// future variant must still compile and degrade to `invalid-input`. Do not
/// remove it.
pub(crate) fn to_wit_err(err: propagon::Error) -> Error {
    use propagon::Error as E;
    match err {
        E::InvalidInput(m) => Error::InvalidInput(m),
        E::EmptyDataset => Error::EmptyDataset,
        E::Numeric(m) => Error::Numeric(m),
        E::State(m) => Error::State(m),
        E::AlgorithmMismatch { expected, found } => {
            Error::AlgorithmMismatch(Mismatch { expected, found })
        }
        E::ParamMismatch(m) => Error::ParamMismatch(m),
        E::Version { found, supported } => Error::Version(VersionMismatch { found, supported }),
        E::Parse { line, msg } => Error::Parse(ParseError {
            line: line as u64,
            msg,
        }),
        E::Io(e) => Error::Io(e.to_string()),
        E::Json(e) => Error::Json(e.to_string()),
        // #[non_exhaustive]: any future variant degrades to invalid-input.
        other => Error::InvalidInput(other.to_string()),
    }
}

/// Ergonomic conversion of a `propagon::Result<T>` into a `Result<T, wit::Error>`.
pub(crate) trait MapWit<T> {
    /// Converts the error with [`to_wit_err`].
    fn map_wit(self) -> Result<T, Error>;
}

impl<T> MapWit<T> for propagon::Result<T> {
    fn map_wit(self) -> Result<T, Error> {
        self.map_err(to_wit_err)
    }
}

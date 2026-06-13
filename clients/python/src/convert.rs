//! Small marshalling helpers shared across the binding surface.

use pyo3::prelude::*;

use crate::errors::InvalidInputError;

/// Narrows a Python `float` (`f64`) to the `f32` the dataset layer stores.
///
/// propagon keeps weights and win margins in single precision, so some
/// precision loss on fractional values (e.g. `0.1`) is expected and matches the
/// library's own semantics — those are *not* rejected. What is rejected is a
/// finite `f64` that overflows `f32` range (e.g. `1e40`), which would silently
/// become `inf`; we raise a clear error at the boundary instead of letting an
/// opaque "not finite" surface from deep in the core.
pub(crate) fn narrow_f32(value: f64, what: &str) -> PyResult<f32> {
    let narrowed = value as f32;
    // A finite input that narrows to a non-finite f32 is an overflow; NaN in
    // stays NaN and is also rejected (the core would reject it anyway).
    if value.is_finite() && !narrowed.is_finite() {
        return Err(InvalidInputError::new_err(format!(
            "{what} {value} overflows the 32-bit float range propagon stores weights/margins in"
        )));
    }
    Ok(narrowed)
}

/// Borrows a `Vec<String>` (extracted from a Python `list[str]`) as the
/// `&[&str]` roster the dataset push methods take.
pub(crate) fn as_str_slice(owned: &[String]) -> Vec<&str> {
    owned.iter().map(String::as_str).collect()
}

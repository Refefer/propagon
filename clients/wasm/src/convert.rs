//! Small marshalling helpers shared across the binding surface.

use crate::bindings::exports::propagon::core::types::Error;

/// Narrows a WIT `f64` to the `f32` the dataset layer stores.
///
/// propagon keeps weights and margins in single precision; fractional precision
/// loss matches the library's own semantics and is *not* rejected. A finite
/// `f64` that overflows `f32` range (e.g. `1e40`) would silently become `inf`,
/// so it is rejected with a clear error at the boundary instead.
pub(crate) fn narrow_f32(value: f64, what: &str) -> Result<f32, Error> {
    let narrowed = value as f32;
    if value.is_finite() && !narrowed.is_finite() {
        return Err(Error::InvalidInput(format!(
            "{what} {value} overflows the 32-bit float range propagon stores weights/margins in"
        )));
    }
    Ok(narrowed)
}

/// Borrows a `&[String]` as the `&[&str]` roster the dataset push methods take.
pub(crate) fn as_str_slice(owned: &[String]) -> Vec<&str> {
    owned.iter().map(String::as_str).collect()
}

//! Conversions between WIT enums/variants and `propagon` enums.
//!
//! Unlike the Python client (which parses kebab-case strings via `unit_enum`),
//! the Component Model carries these as statically-typed `enum`/`variant`s, so
//! conversion is a total `match` with no parse step or "expected one of" error.

use crate::bindings::exports::propagon::core::graph::{Sink, Teleport};
use crate::bindings::exports::propagon::core::types::{Error, GameOutcome};
use crate::convert::narrow_f32;

/// WIT `game-outcome` -> `propagon::GameOutcome` (margins narrow f64 -> f32).
pub(crate) fn game_outcome(go: GameOutcome) -> Result<propagon::GameOutcome, Error> {
    Ok(match go {
        GameOutcome::Side1Win(m) => propagon::GameOutcome::Side1Win(narrow_f32(m, "margin")?),
        GameOutcome::Side2Win(m) => propagon::GameOutcome::Side2Win(narrow_f32(m, "margin")?),
        GameOutcome::Tie => propagon::GameOutcome::Tie,
    })
}

/// WIT `sink` -> `propagon::algos::Sink`.
pub(crate) fn sink(s: Sink) -> propagon::algos::Sink {
    use propagon::algos::Sink as P;
    match s {
        Sink::Reverse => P::Reverse,
        Sink::All => P::All,
        Sink::Uniform => P::Uniform,
        Sink::None => P::None,
    }
}

/// WIT `teleport` -> `propagon::algos::Teleport`.
pub(crate) fn teleport(t: Teleport) -> propagon::algos::Teleport {
    match t {
        Teleport::Uniform => propagon::algos::Teleport::Uniform,
        Teleport::Seeds(seeds) => propagon::algos::Teleport::Seeds(seeds),
    }
}

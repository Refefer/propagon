//! Algorithm implementations.
//!
//! Each module pairs an algorithm struct (its public fields are the params,
//! `Default` gives sensible values) with an owned, serializable model type.
//! See `docs/algorithms.md` for the survey entry behind each method and
//! `docs/PRD.md` FR-5 for which algorithms support incremental update vs
//! warm-start refitting.

mod bandits;
mod borda;
mod common;
mod copeland;
mod elo;
mod rate;

pub use bandits::{Bandit, BanditModel, BanditPolicy};
pub use borda::{Borda, BordaModel};
pub use copeland::{Copeland, CopelandModel};
pub use elo::{Elo, EloModel};
pub use rate::{Confidence, WinRate, WinRateModel};

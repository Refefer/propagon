//! Progress reporting without I/O in the core.
//!
//! Algorithms never print; they report through a [`Progress`] implementation
//! supplied via [`FitOptions`](crate::FitOptions). The CLI renders progress
//! bars; bindings can forward to callbacks; the default is silence.

/// Receiver for coarse progress events emitted during fitting.
///
/// All methods have no-op defaults so implementors override only what they
/// render. Implementations must be `Sync`: parallel fitters may report from
/// worker threads.
pub trait Progress: Sync {
    /// A new phase began (e.g. `"mm sweep"`), with an optional known total.
    fn start(&self, _phase: &str, _total: Option<u64>) {}
    /// Monotonic completion count within the current phase.
    fn update(&self, _done: u64) {}
    /// Free-form status detail (e.g. current convergence error).
    fn message(&self, _msg: &str) {}
    /// The current phase finished.
    fn finish(&self) {}
}

/// The default: report nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProgress;

impl Progress for NoProgress {}

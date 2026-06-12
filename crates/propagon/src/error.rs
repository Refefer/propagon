//! Error type shared across the crate.
//!
//! Every fallible public API returns [`Result`]. No public function panics on
//! malformed input — this is load-bearing for the FFI surfaces (Python, WASM)
//! that wrap this crate.

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// All failures the library can report.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An underlying I/O operation failed (reading or writing a file or stream).
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// A state or dataset JSONL value could not be (de)serialized.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// An input row was malformed; `line` is 1-based and `msg` says what failed.
    #[error("parse error at line {line}: {msg}")]
    Parse {
        /// 1-based line number of the offending row.
        line: usize,
        /// Human-readable reason the row could not be parsed.
        msg: String,
    },

    /// A saved state file was structurally invalid (bad header or missing field).
    #[error("state file error: {0}")]
    State(String),

    /// A state file was produced by a different algorithm than the one loading it.
    #[error("state file is for algorithm '{found}', expected '{expected}'")]
    AlgorithmMismatch {
        /// Algorithm tag the loader requires.
        expected: String,
        /// Algorithm tag found in the file.
        found: String,
    },

    /// A state file's schema version is not one this build understands.
    #[error("unsupported state schema version {found} (this build reads version {supported})")]
    Version {
        /// Schema version recorded in the file.
        found: u32,
        /// Schema version this build can read.
        supported: u32,
    },

    /// Resumed state carries parameters incompatible with the current config.
    #[error("parameter mismatch: {0}")]
    ParamMismatch(String),

    /// The dataset contains no usable rows.
    #[error("dataset is empty")]
    EmptyDataset,

    /// The caller passed an argument the algorithm cannot accept (out of range,
    /// wrong shape, an unrepresentable combination).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The numerics failed to produce a usable result (no convergence, a
    /// singular system, a disconnected comparison graph).
    #[error("numerical failure: {0}")]
    Numeric(String),
}

impl Error {
    /// Helper for parse failures with positional context.
    pub fn parse(line: usize, msg: impl Into<String>) -> Self {
        Error::Parse {
            line,
            msg: msg.into(),
        }
    }
}

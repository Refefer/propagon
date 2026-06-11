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
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("parse error at line {line}: {msg}")]
    Parse { line: usize, msg: String },

    #[error("state file error: {0}")]
    State(String),

    #[error("state file is for algorithm '{found}', expected '{expected}'")]
    AlgorithmMismatch { expected: String, found: String },

    #[error("unsupported state schema version {found} (this build reads version {supported})")]
    Version { found: u32, supported: u32 },

    #[error("parameter mismatch: {0}")]
    ParamMismatch(String),

    #[error("dataset is empty")]
    EmptyDataset,

    #[error("invalid input: {0}")]
    InvalidInput(String),

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

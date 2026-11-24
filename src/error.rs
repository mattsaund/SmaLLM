//! The error type shared by the whole engine.

use std::fmt;

/// Everything that can go wrong while loading or running a model.
#[derive(Debug)]
pub enum Error {
    /// The operating system refused a file or network operation.
    Io(std::io::Error),
    /// The file is not valid GGUF, or its contents contradict each other.
    Format(String),
    /// The file is valid but uses a feature this engine does not implement.
    Unsupported(String),
    /// The caller passed an argument that cannot be honored.
    Invalid(String),
    /// The context window has no room for the tokens that were submitted.
    ContextFull { n_ctx: usize, needed: usize },
}

/// Result alias used by every fallible function in the engine.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Shorthand for a malformed-file error.
    pub(crate) fn format(msg: impl Into<String>) -> Error {
        Error::Format(msg.into())
    }

    /// Shorthand for an unsupported-feature error.
    pub(crate) fn unsupported(msg: impl Into<String>) -> Error {
        Error::Unsupported(msg.into())
    }

    /// Shorthand for a bad-argument error.
    pub(crate) fn invalid(msg: impl Into<String>) -> Error {
        Error::Invalid(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Format(msg) => write!(f, "invalid model file: {msg}"),
            Error::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Error::Invalid(msg) => write!(f, "{msg}"),
            Error::ContextFull { n_ctx, needed } => {
                write!(
                    f,
                    "context full: {needed} tokens needed, the window holds {n_ctx}"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}

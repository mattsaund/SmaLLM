//! A loaded model and the sessions that run it.
//!
//! | File         | Role                                                    |
//! |--------------|---------------------------------------------------------|
//! | `config.rs`  | Hyperparameters, and the switches between architectures. |
//! | `weights.rs` | Finds each tensor in the file and checks its shape.      |
//! | `kv.rs`      | The key/value cache, grown in chunks.                    |
//! | `matmul.rs`  | The parallel matrix multiply.                            |
//! | `forward.rs` | The forward pass and the [`Session`] type.               |

mod config;
mod forward;
mod kv;
mod matmul;
mod weights;

#[cfg(test)]
mod tests;

use std::path::Path;

use crate::error::Result;
use crate::gguf::Gguf;
use crate::tokenizer::Tokenizer;

pub use config::{ARCHITECTURES, Activation, Config, Rope};
pub use forward::{Phase, Profile, Session, SessionConfig};
pub use weights::{Layer, Linear, Matrix, Weights};

/// A model file, opened and ready to run.
///
/// Loading reads only the metadata. The weights stay in the file mapping and
/// are paged in by the operating system as they are used, so a model can be
/// shared between sessions and processes at no extra memory cost.
pub struct Model {
    file: Gguf,
    config: Config,
    weights: Weights,
    tokenizer: Tokenizer,
}

impl Model {
    /// Opens the model file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Model> {
        Model::from_gguf(Gguf::open(path)?)
    }

    /// Opens a model file that is already in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Model> {
        Model::from_gguf(Gguf::from_bytes(bytes)?)
    }

    /// Builds a model from a parsed file.
    pub fn from_gguf(file: Gguf) -> Result<Model> {
        let tokenizer = Tokenizer::from_gguf(&file)?;
        let mut config = Config::from_gguf(&file, tokenizer.n_tokens())?;
        let weights = Weights::bind(&file, &mut config)?;
        Ok(Model {
            file,
            config,
            weights,
            tokenizer,
        })
    }

    /// The model's shape and switches.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The model's weights, bound to their roles.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The model's tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The underlying file: metadata and the tensor table.
    pub fn gguf(&self) -> &Gguf {
        &self.file
    }

    /// The chat template stored in the file, in Jinja syntax.
    pub fn chat_template(&self) -> Option<&str> {
        self.file.get_str("tokenizer.chat_template")
    }

    /// Total number of weights in the file.
    pub fn n_params(&self) -> u64 {
        self.file
            .tensors()
            .iter()
            .map(|tensor| tensor.n_elements() as u64)
            .sum()
    }

    /// Size of the model file in bytes.
    pub fn file_size(&self) -> usize {
        self.file.mapping().len()
    }
}

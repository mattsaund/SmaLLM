//! A loaded model and the sessions that run it.
//!
//! | File         | Role                                                    |
//! |--------------|---------------------------------------------------------|
//! | `config.rs`  | Hyperparameters, and the switches between architectures. |
//! | `weights.rs` | Finds each tensor in the file and checks its shape.      |
//! | `kv.rs`      | The key/value cache, grown in chunks.                    |
//! | `conv.rs`    | The state of short convolution blocks, with rollback.    |
//! | `encoder.rs` | Encoder models, for embeddings and reranking.            |
//! | `matmul.rs`  | The parallel matrix multiply.                            |
//! | `forward.rs` | The forward pass and the [`Session`] type.               |

mod config;
mod conv;
mod encoder;
mod forward;
mod kv;
mod matmul;
mod recurrent;
mod weights;

#[cfg(test)]
mod tests;

use std::path::Path;

use crate::error::Result;
use crate::gguf::Gguf;
use crate::tokenizer::Tokenizer;

pub use config::{
    ARCHITECTURES, Activation, Config, Gating, LayerSpec, Mixer, Rope, Routing, Yarn,
};
pub use encoder::{ENCODERS, Embedder, Pooling, normalize};
pub use forward::{Phase, Profile, Session, SessionConfig};
pub use weights::{
    AttentionWeights, ConvWeights, Dense, ExpertMatrix, Experts, FeedForward, Layer, Linear,
    Matrix, MixerWeights, PerLayerInputs, PerLayerWeights, Weights,
};

/// A model file, opened and ready to run.
///
/// Loading reads only the metadata. The weights stay in the file mapping and
/// are paged in by the operating system as they are used, so a model can be
/// shared between sessions and processes at no extra memory cost.
pub struct Model {
    file: Gguf,
    config: Config,
    weights: Weights,
    /// The weights of an encoder model, which [`Embedder`] runs. `None`
    /// for the decoders that [`Session`] runs.
    encoder: Option<encoder::Encoder>,
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
        let arch = file.get_str("general.architecture").unwrap_or("");
        if ENCODERS.contains(&arch) {
            let (config, encoder) = encoder::Encoder::bind(&file, arch, tokenizer.n_tokens())?;
            let weights = Weights::for_encoder(encoder.token_embd());
            return Ok(Model {
                file,
                config,
                weights,
                encoder: Some(encoder),
                tokenizer,
            });
        }
        let mut config = Config::from_gguf(&file, tokenizer.n_tokens())?;
        let weights = Weights::bind(&file, &mut config)?;
        Ok(Model {
            file,
            config,
            weights,
            encoder: None,
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

    /// The encoder weights of an embedding or reranking model.
    pub fn encoder(&self) -> Option<&encoder::Encoder> {
        self.encoder.as_ref()
    }

    /// True for embedding and reranking models, which [`Embedder`] runs
    /// in the place of a [`Session`].
    pub fn is_encoder(&self) -> bool {
        self.encoder.is_some()
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

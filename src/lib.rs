//! SmaLLM is a small, fast inference engine for GGUF language models.
//!
//! It targets low-power hardware: single-board computers, mini PCs and
//! laptops. The engine has no dependencies outside the Rust standard library.
//!
//! # Layers
//!
//! Each module depends only on the ones listed before it.
//!
//! | Module        | Role                                                     |
//! |---------------|----------------------------------------------------------|
//! | [`error`]     | The error type.                                          |
//! | [`gguf`]      | Reads the model file: metadata and the tensor table.     |
//! | [`tensor`]    | Tensor types and the bit layout of quantized blocks.     |
//! | [`kernels`]   | Dot products, scalar and SIMD, picked at startup.        |
//! | [`threads`]   | The thread pool that runs the kernels in parallel.       |
//! | [`tokenizer`] | Text to token ids and back.                              |
//! | [`model`]     | Weights, the forward pass, and the key/value cache.      |
//! | [`sampler`]   | Picks the next token from the logits.                    |
//! | [`chat`]      | Formats a conversation the way a model was trained on.   |
//! | [`generate`]  | The generation loop: prompt in, text out.                |
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use smallm::{generate, Model, Session, SessionConfig};
//!
//! let model = Arc::new(Model::load("model.gguf")?);
//! let mut session = Session::new(model, SessionConfig::default())?;
//! let options = generate::Options { max_tokens: 64, ..Default::default() };
//! generate::run(&mut session, "Once upon a time", &options, |piece| {
//!     print!("{piece}");
//!     true
//! })?;
//! # Ok::<(), smallm::Error>(())
//! ```

pub mod chat;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod kernels;
pub mod model;
pub mod sampler;
pub mod tensor;
pub mod threads;
pub mod tokenizer;

#[cfg(test)]
mod testing;

pub use error::{Error, Result};
pub use model::{Model, Session, SessionConfig};
pub use sampler::{Sampler, SamplerConfig};
pub use tokenizer::Tokenizer;

/// The version of the engine, from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

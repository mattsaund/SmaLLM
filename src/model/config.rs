//! A model's hyperparameters, read from GGUF metadata.
//!
//! Every supported architecture is the same decoder-only transformer with a
//! few switches. This file turns the architecture name into those switches,
//! so the forward pass has no per-architecture code.

use crate::error::{Error, Result};
use crate::gguf::Gguf;

/// Architectures this engine runs, as named in `general.architecture`.
pub const ARCHITECTURES: [&str; 4] = ["llama", "qwen2", "qwen3", "gemma3"];

/// Which two elements of an attention head a rotary embedding rotates
/// together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rope {
    /// Neighbors: elements `2i` and `2i + 1`.
    Pairs,
    /// Halves: elements `i` and `i + head_dim / 2`.
    Halves,
}

/// The activation inside the feed-forward gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    Gelu,
}

/// Everything the forward pass needs to know about a model's shape.
#[derive(Clone, Debug)]
pub struct Config {
    /// The architecture name from the file.
    pub arch: String,
    /// The model's display name, when the file has one.
    pub name: String,
    /// Tokens in the vocabulary.
    pub n_vocab: usize,
    /// The context length the model was trained with.
    pub n_ctx_train: usize,
    /// Width of the hidden state.
    pub n_embd: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Attention heads per block.
    pub n_head: usize,
    /// Key and value heads per block. Fewer than `n_head` when query heads
    /// share them (grouped-query attention).
    pub n_head_kv: usize,
    /// Width of one attention head.
    pub head_dim: usize,
    /// Width of the feed-forward layer. Zero until the weights are bound,
    /// when the file does not state it.
    pub n_ff: usize,
    /// Epsilon inside every RMS normalization.
    pub norm_eps: f32,
    /// How rotary embeddings pair up elements.
    pub rope: Rope,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Rotary frequency base in sliding-window layers.
    pub rope_base_local: f32,
    /// Multiplier on every rotary angle. One unless the model was trained
    /// with linear position scaling.
    pub rope_scale: f32,
    /// The feed-forward activation.
    pub activation: Activation,
    /// Multiplier on token embeddings.
    pub embedding_scale: f32,
    /// Multiplier on attention scores before the softmax.
    pub attention_scale: f32,
    /// Positions a sliding-window layer can look back, itself included.
    /// Zero when the model has no such layers.
    pub sliding_window: usize,
    /// Per layer: true when the layer uses the sliding window.
    pub local_layers: Vec<bool>,
    /// Soft cap applied to the output logits. Zero for none.
    pub logit_softcap: f32,
}

impl Config {
    /// Reads the configuration of a model file. `n_vocab` comes from the
    /// tokenizer, which is loaded first.
    pub fn from_gguf(file: &Gguf, n_vocab: usize) -> Result<Config> {
        let arch = file
            .get_str("general.architecture")
            .ok_or_else(|| Error::format("the file does not name its architecture"))?
            .to_string();
        if !ARCHITECTURES.contains(&arch.as_str()) {
            return Err(Error::unsupported(format!(
                "model architecture '{arch}' (supported: {})",
                ARCHITECTURES.join(", ")
            )));
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let required = |suffix: &str| {
            file.get_usize(&key(suffix))
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    Error::format(format!("missing or invalid metadata: {}", key(suffix)))
                })
        };

        let n_ctx_train = required("context_length")?;
        let n_embd = required("embedding_length")?;
        let n_layer = required("block_count")?;
        let n_head = required("attention.head_count")?;
        let n_head_kv = file
            .get_usize(&key("attention.head_count_kv"))
            .filter(|&v| v > 0)
            .unwrap_or(n_head);
        let head_dim = file
            .get_usize(&key("attention.key_length"))
            .filter(|&v| v > 0)
            .unwrap_or(n_embd / n_head);
        let value_dim = file
            .get_usize(&key("attention.value_length"))
            .unwrap_or(head_dim);
        let rope_dims = file
            .get_usize(&key("rope.dimension_count"))
            .unwrap_or(head_dim);
        let n_ff = file.get_usize(&key("feed_forward_length")).unwrap_or(0);

        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::format(format!(
                "attention head width {head_dim} must be even and non-zero"
            )));
        }
        if n_head % n_head_kv != 0 {
            return Err(Error::format(format!(
                "{n_head} heads cannot share {n_head_kv} key/value heads"
            )));
        }
        if value_dim != head_dim {
            return Err(Error::unsupported(
                "attention with different key and value widths",
            ));
        }
        if rope_dims != head_dim {
            return Err(Error::unsupported("rotary embeddings on part of a head"));
        }
        if file.get_usize(&key("expert_count")).unwrap_or(0) > 0 {
            return Err(Error::unsupported("mixture-of-experts models"));
        }
        // Guard the sizes that later multiply into buffer lengths.
        for (what, value) in [
            ("layers", n_layer),
            ("heads", n_head),
            ("embedding width", n_embd),
        ] {
            if value > 1 << 20 {
                return Err(Error::format(format!(
                    "implausible number of {what}: {value}"
                )));
            }
        }

        // Position scaling other than linear changes the rotary formula.
        let mut rope_scale = 1.0;
        match file.get_str(&key("rope.scaling.type")).unwrap_or("linear") {
            "none" | "linear" => {
                let factor = file
                    .get_f32(&key("rope.scaling.factor"))
                    .or_else(|| file.get_f32(&key("rope.scale_linear")))
                    .unwrap_or(0.0);
                if factor != 0.0 {
                    rope_scale = 1.0 / factor;
                }
            }
            other => {
                return Err(Error::unsupported(format!(
                    "rotary position scaling '{other}'"
                )));
            }
        }

        let mut config = Config {
            name: file.get_str("general.name").unwrap_or("").to_string(),
            n_vocab,
            n_ctx_train,
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            norm_eps: file
                .get_f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope: Rope::Halves,
            rope_base: file.get_f32(&key("rope.freq_base")).unwrap_or(10000.0),
            rope_base_local: 10000.0,
            rope_scale,
            activation: Activation::Silu,
            embedding_scale: 1.0,
            attention_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: 0,
            local_layers: vec![false; n_layer],
            logit_softcap: 0.0,
            arch,
        };

        match config.arch.as_str() {
            // Llama weights are stored so that neighbors rotate together.
            "llama" => config.rope = Rope::Pairs,
            "qwen2" | "qwen3" => {}
            "gemma3" => {
                config.activation = Activation::Gelu;
                config.embedding_scale = (n_embd as f32).sqrt();
                config.logit_softcap = file
                    .get_f32("gemma3.final_logit_softcapping")
                    .unwrap_or(0.0);
                // The largest Gemma 3 scales attention by its nominal head
                // width, which differs from the stored key width.
                if n_layer == 62 {
                    config.attention_scale = 1.0 / ((n_embd / n_head) as f32).sqrt();
                }
                let window = file
                    .get_usize("gemma3.attention.sliding_window")
                    .unwrap_or(0);
                if window > 0 {
                    config.sliding_window = window;
                    config.rope_base_local =
                        file.get_f32("gemma3.rope.freq_base_swa").unwrap_or(10000.0);
                    config.local_layers = sliding_window_layers(file, n_layer, 6);
                }
            }
            _ => unreachable!("architecture was checked above"),
        }
        Ok(config)
    }

    /// Width of the query projection: all heads side by side.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Width of the key and value projections.
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
}

/// Which layers use the sliding window.
///
/// The file holds either one flag per layer or a period: in every run of
/// `period` layers, all but the last are windowed.
fn sliding_window_layers(file: &Gguf, n_layer: usize, default_period: usize) -> Vec<bool> {
    let key = "gemma3.attention.sliding_window_pattern";
    if let Some(flags) = file.get_array(key) {
        return (0..n_layer)
            .map(|layer| flags.number(layer).unwrap_or(0.0) != 0.0)
            .collect();
    }
    let period = file.get_usize(key).unwrap_or(default_period);
    (0..n_layer)
        .map(|layer| period == 0 || layer % period < period - 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::testing::Builder;

    fn base(arch: &str) -> Builder {
        with_kv_heads(arch, 4)
    }

    fn with_kv_heads(arch: &str, kv_heads: u32) -> Builder {
        let mut builder = Builder::new();
        builder
            .string("general.architecture", arch)
            .u32(&format!("{arch}.context_length"), 2048)
            .u32(&format!("{arch}.embedding_length"), 64)
            .u32(&format!("{arch}.block_count"), 12)
            .u32(&format!("{arch}.attention.head_count"), 8)
            .u32(&format!("{arch}.attention.head_count_kv"), kv_heads);
        builder
    }

    fn config(builder: &Builder) -> Result<Config> {
        Config::from_gguf(&Gguf::from_bytes(builder.build()).unwrap(), 100)
    }

    #[test]
    fn llama_defaults() {
        let c = config(&base("llama")).unwrap();
        assert_eq!(
            (c.n_embd, c.n_layer, c.n_head, c.n_head_kv, c.head_dim),
            (64, 12, 8, 4, 8)
        );
        assert_eq!((c.q_dim(), c.kv_dim()), (64, 32));
        assert_eq!(c.rope, Rope::Pairs);
        assert_eq!(c.activation, Activation::Silu);
        assert_eq!(c.rope_base, 10000.0);
        assert!((c.attention_scale - 1.0 / 8f32.sqrt()).abs() < 1e-7);
        assert!(c.local_layers.iter().all(|local| !local));
    }

    #[test]
    fn qwen_rotates_halves() {
        assert_eq!(config(&base("qwen2")).unwrap().rope, Rope::Halves);
        assert_eq!(config(&base("qwen3")).unwrap().rope, Rope::Halves);
    }

    #[test]
    fn gemma3_switches() {
        let mut builder = base("gemma3");
        builder
            .u32("gemma3.attention.sliding_window", 512)
            .f32("gemma3.rope.freq_base", 1e6);
        let c = config(&builder).unwrap();
        assert_eq!(c.activation, Activation::Gelu);
        assert_eq!(c.embedding_scale, 8.0);
        assert_eq!(c.sliding_window, 512);
        assert_eq!((c.rope_base, c.rope_base_local), (1e6, 10000.0));
        // Every sixth layer sees the whole context.
        let expected: Vec<bool> = (0..12).map(|layer| layer % 6 != 5).collect();
        assert_eq!(c.local_layers, expected);
    }

    #[test]
    fn unsupported_models_are_named() {
        let error = config(&base("mamba")).unwrap_err().to_string();
        assert!(
            error.contains("mamba") && error.contains("qwen2"),
            "{error}"
        );

        let mut builder = base("llama");
        builder.u32("llama.expert_count", 8);
        assert!(config(&builder).is_err());

        let mut builder = base("llama");
        builder.string("llama.rope.scaling.type", "yarn");
        assert!(config(&builder).is_err());

        let mut builder = base("llama");
        builder.u32("llama.rope.dimension_count", 4);
        assert!(config(&builder).is_err());
    }

    #[test]
    fn inconsistent_shapes_are_rejected() {
        let mut builder = Builder::new();
        builder
            .string("general.architecture", "llama")
            .u32("llama.context_length", 8);
        assert!(config(&builder).is_err());

        assert!(config(&with_kv_heads("llama", 3)).is_err());

        let mut builder = base("llama");
        builder.u32("llama.attention.key_length", 7);
        assert!(config(&builder).is_err());
    }

    #[test]
    fn linear_position_scaling_is_applied() {
        let mut builder = base("llama");
        builder.f32("llama.rope.scaling.factor", 4.0);
        assert_eq!(config(&builder).unwrap().rope_scale, 0.25);
    }
}

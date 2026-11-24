//! A model's hyperparameters, read from GGUF metadata.
//!
//! Every supported architecture is a decoder-only stack of blocks. Each
//! block first mixes information across positions, with attention or with a
//! short convolution, and then runs a feed-forward network. This file turns
//! the architecture name and its metadata into a few model-wide switches and
//! one [`LayerSpec`] per block, so the forward pass has no per-architecture
//! code.

use crate::error::{Error, Result};
use crate::gguf::Gguf;

/// Architectures this engine runs, as named in `general.architecture`.
pub const ARCHITECTURES: [&str; 20] = [
    "llama",
    "qwen2",
    "qwen3",
    "qwen2moe",
    "qwen3moe",
    "gemma3",
    "gemma4",
    "lfm2",
    "lfm2moe",
    "lfm3",
    "granite",
    "granitemoe",
    "olmoe",
    "smollm3",
    "gemma2",
    "phi3",
    "gpt-oss",
    "phi2",
    "deepseek",
    "deepseek2",
];

/// Which two elements of an attention head a rotary embedding rotates
/// together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rope {
    /// Neighbors: elements `2i` and `2i + 1`.
    Pairs,
    /// Halves: elements `i` and `i + rope_dims / 2`.
    Halves,
}

/// The activation inside the feed-forward gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Silu,
    Gelu,
    /// OpenAI's SwiGLU (gpt-oss): the gate is capped at 7 and goes through
    /// `x * sigmoid(1.702 * x)`, and the up projection is clamped to
    /// `[-7, 7]` and has one added.
    SwigluOai,
}

/// How a mixture of experts turns the router's scores into weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gating {
    /// A softmax over every expert's score.
    Softmax,
    /// A sigmoid of each score on its own.
    Sigmoid,
    /// A softmax over the scores of the chosen experts only.
    SoftmaxChosen,
}

/// How the mixture-of-experts layers pick and weigh their experts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Routing {
    /// Experts in each mixture layer.
    pub count: usize,
    /// Experts each token goes through.
    pub used: usize,
    pub gating: Gating,
    /// Divide the chosen experts' weights by their sum.
    pub normalize: bool,
    /// Multiplier on the weights. One for none.
    pub scale: f32,
    /// Groups the experts are split into, for routing limited to the best
    /// groups (DeepSeek V3). Zero or one for no groups.
    pub groups: usize,
    /// Groups a token's experts may come from.
    pub groups_used: usize,
}

/// YaRN, a way to stretch rotary positions over a longer context than the
/// model was trained with. Low frequencies are scaled down to fit the
/// longer context, high ones are kept as they are, and a ramp blends the
/// two in between. Rotated values are also scaled up a little.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    /// The context length the model was trained with before stretching.
    pub original_context: usize,
    /// Rotations over the original context below which a pair is kept as
    /// it is. Pairs that turn fewer times than `beta_slow` are scaled.
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// Multiplier on rotated values, before the part each layer's scaling
    /// adds. One for most models: see `RopeTables::new`.
    pub attn_factor: f32,
}

/// The shape of multi-head latent attention (DeepSeek V2 and V3).
///
/// Each position stores one small latent vector and one rotary key that
/// every head shares. A head's own key and value are projections of that
/// latent. The engine never computes them: it moves each query into the
/// latent space instead (see `forward/mla.rs`). So the cache holds
/// `kv_rank + rope_dims` floats per position and layer, which is that
/// layer's `head_dim`, and the layer has one key/value head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mla {
    /// Width of the latent vector.
    pub kv_rank: usize,
    /// Elements at the start of each query and key head that do not
    /// rotate. The rotary part follows, `rope_dims` wide.
    pub nope_dims: usize,
    /// Width of each value head.
    pub value_dims: usize,
}

/// How a block mixes information across positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    /// Attention over earlier positions, with a key/value cache.
    Attention,
    /// A short causal convolution over the last few positions, gated, as in
    /// Liquid AI's LFM models. It keeps a small state instead of a cache.
    ShortConv,
}

/// The shape of one block.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSpec {
    pub mixer: Mixer,
    /// Attention heads. Zero in convolution blocks.
    pub n_head: usize,
    /// Key and value heads. Fewer than `n_head` when query heads share them
    /// (grouped-query attention).
    pub n_head_kv: usize,
    /// Width of one attention head.
    pub head_dim: usize,
    /// Elements at the start of each head that rotate by position. The rest
    /// pass through.
    pub rope_dims: usize,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Multiplier on every rotary angle. One unless the model was trained
    /// with linear position scaling.
    pub rope_scale: f32,
    /// Divide the rotary frequencies by the model's `rope_freqs.weight`.
    pub rope_factors: bool,
    /// Positions the layer looks back, itself included. Zero for all.
    pub window: usize,
    /// The layer whose keys and values this layer reads, when it computes
    /// none of its own.
    pub kv_source: Option<usize>,
    /// Multiplier on attention scores before the softmax.
    pub attention_scale: f32,
    /// Width of the feed-forward layer. Zero until the weights are bound,
    /// when the file does not state it. In a mixture-of-experts layer it is
    /// the width of the shared expert, or zero when there is none.
    pub n_ff: usize,
    /// Width of each expert in a mixture-of-experts layer, set when the
    /// weights are bound. Zero in other layers.
    pub n_ff_expert: usize,
}

impl LayerSpec {
    /// Width of the query projection: all heads side by side.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Width of the key and value projections.
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// True when the layer computes and stores keys and values.
    pub fn owns_kv(&self) -> bool {
        self.mixer == Mixer::Attention && self.kv_source.is_none()
    }
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
    /// Number of blocks.
    pub n_layer: usize,
    /// The shape of each block.
    pub layers: Vec<LayerSpec>,
    /// Epsilon inside every RMS normalization.
    pub norm_eps: f32,
    /// How rotary embeddings pair up elements.
    pub rope: Rope,
    /// The feed-forward activation.
    pub activation: Activation,
    /// Multiplier on token embeddings.
    pub embedding_scale: f32,
    /// Soft cap applied to the output logits. Zero for none.
    pub logit_softcap: f32,
    /// Value vectors are RMS-normalized, without a learned gain, before
    /// they are stored. Gemma 4 does this.
    pub value_norm: bool,
    /// Width of the short convolution, in positions. Zero when the model has
    /// no convolution blocks.
    pub conv_kernel: usize,
    /// Width of each layer's slice of the per-layer token embeddings. Zero
    /// for models without them. Gemma 4 has them.
    pub per_layer_embd: usize,
    /// Multiplier on each block's output before it joins the hidden state
    /// (Granite). One for none.
    pub residual_scale: f32,
    /// The logits are divided by this (Granite). One for none.
    pub logit_scale: f32,
    /// Soft cap on attention scores, `cap * tanh(score / cap)` (Gemma 2).
    /// Zero for none.
    pub attention_softcap: f32,
    /// Multiplier on rotated queries and keys, which LongRoPE models such as
    /// Phi-3 use to keep attention sharp at long context. One for none.
    pub rope_attn_factor: f32,
    /// The context length a LongRoPE model was trained with before its
    /// context was extended. Longer sessions use its long rotary divisors.
    /// Zero for models without them.
    pub long_context_after: usize,
    /// YaRN's settings, for models that stretch their positions that way.
    pub yarn: Option<Yarn>,
    /// LayerNorm with a bias, where most models use RMS norm (Phi-2).
    pub layer_norm: bool,
    /// Attention and the feed-forward network read the same normalized
    /// input, and both add to the hidden state (Phi-2).
    pub parallel: bool,
    /// The feed-forward network is `down(act(up(x)))`, without a gate
    /// (Phi-2).
    pub ungated: bool,
    /// The expert settings, for models with mixture-of-experts layers.
    pub routing: Option<Routing>,
    /// The latent attention shape, for models whose attention layers use
    /// it (DeepSeek V2 and V3).
    pub mla: Option<Mla>,
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
        let meta = Meta { file, arch: &arch };

        let n_ctx_train = meta.required("context_length")?;
        let n_embd = meta.required("embedding_length")?;
        // A file with multi-token prediction counts its extra blocks as
        // layers. They come last, and this engine does not run them.
        let n_layer = meta
            .required("block_count")?
            .checked_sub(meta.usize("nextn_predict_layers").unwrap_or(0))
            .filter(|&layers| layers > 0)
            .ok_or_else(|| Error::format("more prediction blocks than blocks"))?;
        // Guard the sizes that later multiply into buffer lengths.
        for (what, value) in [("layers", n_layer), ("embedding width", n_embd)] {
            if value > 1 << 20 {
                return Err(Error::format(format!(
                    "implausible number of {what}: {value}"
                )));
            }
        }
        let n_head = meta.per_layer("attention.head_count", n_layer)?;
        let n_head_kv = meta
            .per_layer("attention.head_count_kv", n_layer)
            .unwrap_or_else(|_| n_head.clone());
        let n_ff = meta
            .per_layer("feed_forward_length", n_layer)
            .unwrap_or_else(|_| vec![0; n_layer]);
        if n_head.iter().any(|&heads| heads == 0 || heads > 1 << 16) {
            return Err(Error::format(format!(
                "missing or invalid metadata: {arch}.attention.head_count"
            )));
        }
        let routing = meta.routing()?;

        // Head widths: one for full attention, and maybe another for the
        // sliding-window layers.
        let head_dim = meta
            .usize("attention.key_length")
            .filter(|&width| width > 0)
            .unwrap_or(n_embd / n_head[0]);
        let head_dim_local = meta.usize("attention.key_length_swa").unwrap_or(head_dim);
        for (key, width) in [
            ("attention.value_length", head_dim),
            ("attention.value_length_swa", head_dim_local),
        ] {
            // Latent attention has values of their own width, see `Mla`.
            if arch != "deepseek2" && meta.usize(key).is_some_and(|value| value != width) {
                return Err(Error::unsupported(
                    "attention with different key and value widths",
                ));
            }
        }
        let rope_dims = meta.usize("rope.dimension_count").unwrap_or(head_dim);
        let rope_dims_local = meta
            .usize("rope.dimension_count_swa")
            .unwrap_or(head_dim_local.min(rope_dims));
        let rope_base = meta.f32("rope.freq_base").unwrap_or(10000.0);
        // Gemma 2's windowed layers rotate like its others, unless the file
        // says otherwise. Elsewhere they default to a base of 10000.
        let default_local_base = if matches!(arch.as_str(), "gemma2" | "gpt-oss") {
            rope_base
        } else {
            10000.0
        };
        let rope_base_local = meta.f32("rope.freq_base_swa").unwrap_or(default_local_base);
        let rope_scale = meta.rope_scale()?;
        let window = meta.usize("attention.sliding_window").unwrap_or(0);

        let mut config = Config {
            name: file.get_str("general.name").unwrap_or("").to_string(),
            n_vocab,
            n_ctx_train,
            n_embd,
            n_layer,
            layers: (0..n_layer)
                .map(|layer| LayerSpec {
                    mixer: Mixer::Attention,
                    n_head: n_head[layer],
                    n_head_kv: n_head_kv[layer],
                    head_dim,
                    rope_dims,
                    rope_base,
                    rope_scale,
                    rope_factors: true,
                    window: 0,
                    kv_source: None,
                    attention_scale: 1.0 / (head_dim as f32).sqrt(),
                    n_ff: n_ff[layer],
                    n_ff_expert: 0,
                })
                .collect(),
            norm_eps: meta.f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            rope: Rope::Halves,
            activation: Activation::Silu,
            embedding_scale: 1.0,
            logit_softcap: 0.0,
            value_norm: false,
            conv_kernel: 0,
            per_layer_embd: 0,
            residual_scale: 1.0,
            logit_scale: 1.0,
            attention_softcap: 0.0,
            rope_attn_factor: meta.f32("rope.scaling.attn_factor").unwrap_or(1.0),
            long_context_after: 0,
            yarn: meta.yarn(n_ctx_train),
            layer_norm: false,
            parallel: false,
            ungated: false,
            routing,
            mla: None,
            arch: arch.clone(),
        };

        // Turns layer `index` into a sliding-window layer with the widths
        // and rotary settings those layers use.
        let make_local = |layer: &mut LayerSpec| {
            layer.window = window;
            layer.head_dim = head_dim_local;
            layer.rope_dims = rope_dims_local;
            layer.rope_base = rope_base_local;
            layer.rope_scale = 1.0;
            layer.rope_factors = false;
            layer.attention_scale = 1.0 / (head_dim_local as f32).sqrt();
        };

        match arch.as_str() {
            // Llama weights are stored so that neighbors rotate together.
            // Mixtral is a Llama with experts.
            "llama" => config.rope = Rope::Pairs,
            "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" | "olmoe" => {}
            // SmolLM3 leaves every fourth layer unrotated.
            "smollm3" => {
                config.rope = Rope::Pairs;
                for layer in config.layers.iter_mut().skip(3).step_by(4) {
                    layer.rope_dims = 0;
                }
            }
            "granite" | "granitemoe" => {
                config.rope = Rope::Pairs;
                config.embedding_scale = meta.f32("embedding_scale").unwrap_or(1.0);
                config.residual_scale = meta.f32("residual_scale").unwrap_or(1.0);
                config.logit_scale = meta.f32("logit_scale").unwrap_or(1.0);
                if let Some(scale) = meta.f32("attention.scale").filter(|&scale| scale != 0.0) {
                    for layer in &mut config.layers {
                        layer.attention_scale = scale;
                    }
                }
            }
            "gemma2" => {
                config.activation = Activation::Gelu;
                config.embedding_scale = (n_embd as f32).sqrt();
                config.logit_softcap = meta.f32("final_logit_softcapping").unwrap_or(30.0);
                config.attention_softcap = meta.f32("attn_logit_softcapping").unwrap_or(50.0);
                // Every other layer looks back over a window, by default
                // 4096 positions.
                let window = if window > 0 { window } else { 4096 };
                for (layer, local) in config
                    .layers
                    .iter_mut()
                    .zip(meta.window_pattern(n_layer, 2))
                {
                    if local {
                        make_local(layer);
                        layer.window = window;
                        layer.rope_scale = rope_scale;
                    }
                }
                // The largest Gemma 2 scales attention by its nominal head
                // width, which differs from the stored key width, in every
                // layer. So this comes after the windowed layers are set.
                if n_layer == 46 {
                    let scale = 1.0 / ((n_embd / n_head[0]) as f32).sqrt();
                    config
                        .layers
                        .iter_mut()
                        .for_each(|layer| layer.attention_scale = scale);
                }
            }
            // gpt-oss: every other layer looks back over a short window, and
            // rotates like the others, YaRN included. Its experts are
            // weighed by a softmax over the chosen ones, and their gate is
            // OpenAI's clamped SwiGLU.
            "gpt-oss" => {
                config.activation = Activation::SwigluOai;
                for (layer, local) in config
                    .layers
                    .iter_mut()
                    .zip(meta.window_pattern(n_layer, 2))
                {
                    if local {
                        make_local(layer);
                        layer.rope_scale = rope_scale;
                    }
                }
                if let Some(routing) = &mut config.routing {
                    routing.gating = Gating::SoftmaxChosen;
                }
            }
            // Phi-2 (and Phi-1.5) is built like GPT-J: LayerNorm, attention
            // and an ungated GELU network side by side, part of each head
            // rotated.
            "phi2" => {
                config.layer_norm = true;
                config.parallel = true;
                config.ungated = true;
                config.activation = Activation::Gelu;
                config.norm_eps = meta.f32("attention.layer_norm_epsilon").unwrap_or(1e-5);
            }
            // Phi-3 rotates halves. Its long-context versions bring two sets
            // of rotary divisors, chosen by the length of the session.
            "phi3" => {
                config.long_context_after = meta
                    .usize("rope.scaling.original_context_length")
                    .unwrap_or(n_ctx_train);
            }
            "gemma3" => {
                config.activation = Activation::Gelu;
                config.embedding_scale = (n_embd as f32).sqrt();
                config.logit_softcap = meta.f32("final_logit_softcapping").unwrap_or(0.0);
                if window > 0 {
                    for (layer, local) in config
                        .layers
                        .iter_mut()
                        .zip(meta.window_pattern(n_layer, 6))
                    {
                        if local {
                            make_local(layer);
                        }
                    }
                }
                // The largest Gemma 3 scales attention by its nominal head
                // width, which differs from the stored key width, in every
                // layer. So this comes after the windowed layers are set.
                if n_layer == 62 {
                    let scale = 1.0 / ((n_embd / n_head[0]) as f32).sqrt();
                    config
                        .layers
                        .iter_mut()
                        .for_each(|layer| layer.attention_scale = scale);
                }
            }
            "gemma4" => {
                config.activation = Activation::Gelu;
                config.embedding_scale = (n_embd as f32).sqrt();
                config.logit_softcap = meta.f32("final_logit_softcapping").unwrap_or(0.0);
                config.value_norm = true;
                config.per_layer_embd = meta.usize("embedding_length_per_layer_input").unwrap_or(0);
                for (layer, local) in config
                    .layers
                    .iter_mut()
                    .zip(meta.window_pattern(n_layer, 0))
                {
                    if local {
                        make_local(layer);
                    }
                }
                // Queries and keys are normalized, so scores need no scale.
                config
                    .layers
                    .iter_mut()
                    .for_each(|layer| layer.attention_scale = 1.0);
                // The last layers read the keys and values of the last
                // layer of their kind that has its own.
                let shared = meta.usize("attention.shared_kv_layers").unwrap_or(0);
                let first_shared = n_layer.saturating_sub(shared);
                if shared > 0 && first_shared < 2 {
                    return Err(Error::format("too many layers share their keys and values"));
                }
                for layer in &mut config.layers[first_shared..] {
                    let local = layer.window > 0;
                    layer.kv_source = Some(first_shared - if local { 2 } else { 1 });
                }
            }
            "lfm2" | "lfm2moe" | "lfm3" => {
                config.conv_kernel = meta.required("shortconv.l_cache")?;
                for layer in &mut config.layers {
                    if layer.n_head_kv == 0 {
                        layer.mixer = Mixer::ShortConv;
                        layer.n_head = 0;
                        layer.rope_dims = 0;
                        layer.rope_factors = false;
                    }
                }
                if arch == "lfm3" {
                    for (layer, local) in config
                        .layers
                        .iter_mut()
                        .zip(meta.window_pattern(n_layer, 0))
                    {
                        if local && layer.mixer == Mixer::Attention {
                            make_local(layer);
                        }
                    }
                } else if window > 0 {
                    // LFM2 can window every attention layer.
                    for layer in &mut config.layers {
                        if layer.mixer == Mixer::Attention {
                            layer.window = window;
                        }
                    }
                }
            }
            // DeepSeek-MoE: a Llama with experts. llama.cpp weighs them by a
            // softmax and never normalizes, whatever the file says.
            "deepseek" => {
                config.rope = Rope::Pairs;
                if let Some(routing) = &mut config.routing {
                    routing.gating = Gating::Softmax;
                    routing.normalize = false;
                }
            }
            // DeepSeek V2 and V3, and the models converted to their shape:
            // latent attention, rotated in pairs, and experts.
            "deepseek2" => {
                config.rope = Rope::Pairs;
                let mla = meta.mla(rope_dims)?;
                let (rotary, score_scale) = meta.deepseek2_scales(
                    1.0 / rope_scale,
                    mla.nope_dims + rope_dims,
                    config.rope_attn_factor,
                    config.yarn.is_some(),
                );
                if let Some(yarn) = &mut config.yarn {
                    yarn.attn_factor = rotary;
                }
                for layer in &mut config.layers {
                    layer.n_head_kv = 1;
                    layer.head_dim = rope_dims + mla.kv_rank;
                    layer.attention_scale = score_scale;
                }
                config.mla = Some(mla);
                // Files without a gating function weigh experts by a
                // softmax, except GLM 4.7 Lite, which llama.cpp knows by
                // its shape.
                let unnamed = matches!(meta.usize("expert_gating_func"), None | Some(0));
                if unnamed && matches!(n_layer, 47 | 48) && n_vocab == 154880 {
                    if let Some(routing) = &mut config.routing {
                        routing.gating = Gating::Sigmoid;
                    }
                }
            }
            _ => unreachable!("architecture was checked above"),
        }
        config.check()?;
        Ok(config)
    }

    /// Checks the shapes that the forward pass relies on.
    fn check(&self) -> Result<()> {
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.mixer != Mixer::Attention {
                continue;
            }
            if layer.head_dim == 0 || layer.head_dim % 2 != 0 || layer.head_dim > 1 << 14 {
                return Err(Error::format(format!(
                    "layer {index}: attention head width {} must be even and non-zero",
                    layer.head_dim
                )));
            }
            if layer.n_head_kv == 0 || layer.n_head % layer.n_head_kv != 0 {
                return Err(Error::format(format!(
                    "layer {index}: {} heads cannot share {} key/value heads",
                    layer.n_head, layer.n_head_kv
                )));
            }
            if layer.rope_dims > layer.head_dim || layer.rope_dims % 2 != 0 {
                return Err(Error::format(format!(
                    "layer {index}: {} rotary elements do not fit a head of {}",
                    layer.rope_dims, layer.head_dim
                )));
            }
            if let Some(source) = layer.kv_source {
                let owner = &self.layers[source];
                let same = owner.owns_kv()
                    && owner.n_head_kv == layer.n_head_kv
                    && owner.head_dim == layer.head_dim;
                if source >= index || !same {
                    return Err(Error::format(format!(
                        "layer {index} cannot read the keys and values of layer {source}"
                    )));
                }
            }
        }
        if let Some(routing) = &self.routing {
            if routing.used == 0 || routing.used > routing.count || routing.count > 1 << 16 {
                return Err(Error::format(format!(
                    "{} of {} experts per token",
                    routing.used, routing.count
                )));
            }
            // The chosen groups must hold enough experts to choose from.
            let groups = routing.groups;
            if groups > 1
                && (routing.count % groups != 0
                    || routing.groups_used == 0
                    || routing.groups_used > groups
                    || routing.used > routing.groups_used * (routing.count / groups))
            {
                return Err(Error::format(format!(
                    "{} experts in {groups} groups, {} of them used, cannot give {} experts",
                    routing.count, routing.groups_used, routing.used
                )));
            }
        }
        if self.conv_kernel == 1 || self.conv_kernel > 64 {
            return Err(Error::format(format!(
                "a short convolution of {} positions",
                self.conv_kernel
            )));
        }
        Ok(())
    }

    /// The widest of each per-layer size, for sizing buffers.
    pub fn max_q_dim(&self) -> usize {
        self.layers.iter().map(LayerSpec::q_dim).max().unwrap_or(0)
    }

    pub fn max_kv_dim(&self) -> usize {
        self.layers.iter().map(LayerSpec::kv_dim).max().unwrap_or(0)
    }

    pub fn max_n_ff(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.n_ff)
            .max()
            .unwrap_or(0)
    }

    pub fn max_head_dim(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.head_dim)
            .max()
            .unwrap_or(0)
    }

    /// The first attention block, which stands for the model in summaries.
    pub fn attention(&self) -> Option<&LayerSpec> {
        self.layers
            .iter()
            .find(|layer| layer.mixer == Mixer::Attention)
    }

    /// Number of blocks of each mixer: attention, then convolution.
    pub fn mixers(&self) -> (usize, usize) {
        let attention = self
            .layers
            .iter()
            .filter(|layer| layer.mixer == Mixer::Attention)
            .count();
        (attention, self.n_layer - attention)
    }
}

/// Reads metadata keys under the architecture's prefix.
struct Meta<'a> {
    file: &'a Gguf,
    arch: &'a str,
}

impl Meta<'_> {
    fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.arch)
    }

    fn usize(&self, suffix: &str) -> Option<usize> {
        self.file.get_usize(&self.key(suffix))
    }

    fn f32(&self, suffix: &str) -> Option<f32> {
        self.file.get_f32(&self.key(suffix))
    }

    fn required(&self, suffix: &str) -> Result<usize> {
        self.usize(suffix)
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                Error::format(format!("missing or invalid metadata: {}", self.key(suffix)))
            })
    }

    /// A number that is either one value for every layer, or an array with
    /// one value per layer.
    fn per_layer(&self, suffix: &str, n_layer: usize) -> Result<Vec<usize>> {
        let key = self.key(suffix);
        if let Some(array) = self.file.get_array(&key) {
            if array.len() < n_layer {
                return Err(Error::format(format!("{key} has fewer values than layers")));
            }
            return (0..n_layer)
                .map(|layer| {
                    array
                        .number(layer)
                        .filter(|value| *value >= 0.0 && *value <= u32::MAX as f64)
                        .map(|value| value as usize)
                        .ok_or_else(|| Error::format(format!("{key} holds a bad value")))
                })
                .collect();
        }
        self.file
            .get_usize(&key)
            .map(|value| vec![value; n_layer])
            .ok_or_else(|| Error::format(format!("missing or invalid metadata: {key}")))
    }

    /// Which layers use the sliding window. The file holds one flag per
    /// layer, or a period: in every run of `period` layers, all but the last
    /// are windowed. `default_period` applies when it holds neither, and
    /// zero then means no layer is windowed.
    fn window_pattern(&self, n_layer: usize, default_period: usize) -> Vec<bool> {
        let key = self.key("attention.sliding_window_pattern");
        if let Some(flags) = self.file.get_array(&key) {
            return (0..n_layer)
                .map(|layer| flags.number(layer).unwrap_or(0.0) != 0.0)
                .collect();
        }
        match self.file.get_usize(&key).unwrap_or(default_period) {
            0 => vec![false; n_layer],
            period => (0..n_layer)
                .map(|layer| layer % period < period - 1)
                .collect(),
        }
    }

    /// The expert settings, when the model has experts. The defaults are
    /// those of each architecture in llama.cpp.
    fn routing(&self) -> Result<Option<Routing>> {
        let count = self.usize("expert_count").unwrap_or(0);
        if count == 0 {
            return Ok(None);
        }
        let used = self.required("expert_used_count")?;
        let gating = match self.usize("expert_gating_func") {
            None | Some(0) | Some(1) => Gating::Softmax,
            Some(2) => Gating::Sigmoid,
            Some(3) => Gating::SoftmaxChosen,
            Some(other) => {
                return Err(Error::unsupported(format!(
                    "expert gating function {other}"
                )));
            }
        };
        let normalize = self
            .file
            .get_bool(&self.key("expert_weights_norm"))
            .unwrap_or(!matches!(
                self.arch,
                "qwen2moe" | "olmoe" | "gpt-oss" | "deepseek" | "deepseek2"
            ));
        Ok(Some(Routing {
            count,
            used,
            gating,
            normalize,
            scale: self
                .f32("expert_weights_scale")
                .filter(|&scale| scale != 0.0)
                .unwrap_or(1.0),
            groups: self.usize("expert_group_count").unwrap_or(0),
            groups_used: self.usize("expert_group_used_count").unwrap_or(0),
        }))
    }

    /// The latent attention shape of a DeepSeek V2 model. Older files give
    /// the widths of a key and a value head as `key_length` and
    /// `value_length`. Newer ones, which store the up-projection in two
    /// halves, give the cache's widths there and the heads' widths as
    /// `key_length_mla` and `value_length_mla`.
    fn mla(&self, rope_dims: usize) -> Result<Mla> {
        let kv_rank = self.required("attention.kv_lora_rank")?;
        let (key, value) = match self.usize("attention.key_length_mla") {
            Some(key) => (key, self.required("attention.value_length_mla")?),
            None => (
                self.required("attention.key_length")?,
                self.required("attention.value_length")?,
            ),
        };
        let mla = Mla {
            kv_rank,
            nope_dims: key.saturating_sub(rope_dims),
            value_dims: value,
        };
        let sizes = [mla.kv_rank, mla.nope_dims, mla.value_dims];
        if sizes.iter().any(|&size| size == 0 || size > 1 << 16) {
            return Err(Error::format(format!(
                "latent attention of rank {kv_rank} with keys of {key} and values of {value}"
            )));
        }
        Ok(mla)
    }

    /// DeepSeek V2's YaRN corrections, made as llama.cpp makes them.
    /// Returns YaRN's factor on rotated values and the scale on attention
    /// scores.
    ///
    /// The file's `yarn_log_multiplier` is a tenth of the model's
    /// `mscale_all_dim`. `llama-context.cpp` picks a rotary factor that
    /// cancels the magnitude YaRN adds to rotated values, for DeepSeek V2
    /// and V3 entirely. `deepseek2.cpp` takes that factor back out and puts
    /// `mscale` on both sides of the query-key product instead, as the
    /// square in the score scale. `stretch` is the YaRN or linear factor,
    /// one for none, and `key_width` the width of a key head.
    fn deepseek2_scales(
        &self,
        stretch: f32,
        key_width: usize,
        rope_attn_factor: f32,
        yarn: bool,
    ) -> (f32, f32) {
        let log_mul = self
            .f32("rope.scaling.yarn_log_multiplier")
            .map_or(0.0, |tenth| tenth / 0.1);
        // The magnitude YaRN gives to a stretch of `scale`.
        let magnitude = |scale: f32, mscale: f32| {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * mscale * scale.ln() + 1.0
            }
        };
        let rotary = if yarn {
            let ratio = if log_mul != 0.0 {
                // llama.cpp reads DeepSeek V2's `mscale` as equal to its
                // `mscale_all_dim`, unless that is one.
                let mscale = if log_mul != 1.0 { log_mul } else { 1.0 };
                magnitude(stretch, mscale) / magnitude(stretch, log_mul)
            } else {
                magnitude(stretch, 1.0)
            };
            ratio * (1.0 / (1.0 + 0.1 * stretch.ln()))
        } else {
            1.0
        };
        let original = rotary * rope_attn_factor * (1.0 + 0.1 * stretch.ln());
        let mscale = original * (1.0 + 0.1 * log_mul * stretch.ln());
        (rotary, mscale * mscale / (key_width as f32).sqrt())
    }

    /// The rotary scaling type the file names, linear by default.
    fn rope_scaling(&self) -> &str {
        self.file
            .get_str(&self.key("rope.scaling.type"))
            .unwrap_or("linear")
    }

    /// YaRN's settings, when the file asks for it. `n_ctx_train` is the
    /// original context when the file does not give one.
    fn yarn(&self, n_ctx_train: usize) -> Option<Yarn> {
        (self.rope_scaling() == "yarn").then(|| Yarn {
            original_context: self
                .usize("rope.scaling.original_context_length")
                .unwrap_or(n_ctx_train),
            beta_fast: self.f32("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            beta_slow: self.f32("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            attn_factor: 1.0,
        })
    }

    /// The position scaling, as a multiplier on rotary angles. YaRN scales
    /// angles too, and adds a ramp that `yarn` describes.
    fn rope_scale(&self) -> Result<f32> {
        match self.rope_scaling() {
            "none" | "linear" | "yarn" => {
                let factor = self
                    .f32("rope.scaling.factor")
                    .or_else(|| self.f32("rope.scale_linear"))
                    .unwrap_or(0.0);
                Ok(if factor != 0.0 { 1.0 / factor } else { 1.0 })
            }
            other => Err(Error::unsupported(format!(
                "rotary position scaling '{other}'"
            ))),
        }
    }
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
        let layer = &c.layers[0];
        assert_eq!(
            (
                c.n_embd,
                c.n_layer,
                layer.n_head,
                layer.n_head_kv,
                layer.head_dim
            ),
            (64, 12, 8, 4, 8)
        );
        assert_eq!((layer.q_dim(), layer.kv_dim()), (64, 32));
        assert_eq!(c.rope, Rope::Pairs);
        assert_eq!(c.activation, Activation::Silu);
        assert_eq!(layer.rope_base, 10000.0);
        assert!((layer.attention_scale - 1.0 / 8f32.sqrt()).abs() < 1e-7);
        assert!(
            c.layers
                .iter()
                .all(|layer| layer.window == 0 && layer.owns_kv())
        );
        assert_eq!(c.mixers(), (12, 0));
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
        // Every sixth layer sees the whole context, with the global base.
        for (index, layer) in c.layers.iter().enumerate() {
            let local = index % 6 != 5;
            assert_eq!(layer.window, if local { 512 } else { 0 });
            assert_eq!(layer.rope_base, if local { 10000.0 } else { 1e6 });
        }
    }

    #[test]
    fn the_largest_gemmas_scale_every_layer_alike() {
        for (arch, n_layer) in [("gemma3", 62), ("gemma2", 46)] {
            let mut builder = Builder::new();
            builder
                .string("general.architecture", arch)
                .u32(&format!("{arch}.context_length"), 2048)
                .u32(&format!("{arch}.embedding_length"), 96)
                .u32(&format!("{arch}.block_count"), n_layer)
                .u32(&format!("{arch}.attention.head_count"), 4)
                .u32(&format!("{arch}.attention.head_count_kv"), 2)
                .u32(&format!("{arch}.attention.key_length"), 32)
                .u32(&format!("{arch}.attention.sliding_window"), 64);
            let c = config(&builder).unwrap();
            // The nominal head width is 96 / 4 = 24, not the stored 32.
            let scale = 1.0 / 24f32.sqrt();
            assert!(
                c.layers.iter().all(|layer| layer.attention_scale == scale),
                "{arch}"
            );
            assert!(c.layers.iter().any(|layer| layer.window > 0), "{arch}");
        }
    }

    #[test]
    fn lfm_models_mix_convolution_and_attention() {
        let mut builder = Builder::new();
        builder
            .string("general.architecture", "lfm2")
            .u32("lfm2.context_length", 2048)
            .u32("lfm2.embedding_length", 64)
            .u32("lfm2.block_count", 4)
            .u32("lfm2.attention.head_count", 8)
            .i32s("lfm2.attention.head_count_kv", &[0, 0, 4, 0])
            .u32("lfm2.shortconv.l_cache", 3);
        let c = config(&builder).unwrap();
        let mixers: Vec<Mixer> = c.layers.iter().map(|layer| layer.mixer).collect();
        use Mixer::*;
        assert_eq!(mixers, [ShortConv, ShortConv, Attention, ShortConv]);
        assert_eq!((c.conv_kernel, c.mixers()), (3, (1, 3)));
    }

    #[test]
    fn gemma4_layers_differ_by_kind_and_share_caches() {
        let mut builder = Builder::new();
        builder
            .string("general.architecture", "gemma4")
            .u32("gemma4.context_length", 2048)
            .u32("gemma4.embedding_length", 64)
            .u32("gemma4.block_count", 6)
            .u32("gemma4.attention.head_count", 4)
            .u32("gemma4.attention.head_count_kv", 1)
            .u32("gemma4.attention.key_length", 32)
            .u32("gemma4.attention.key_length_swa", 16)
            .u32("gemma4.attention.sliding_window", 8)
            .u32("gemma4.attention.shared_kv_layers", 2)
            .bools(
                "gemma4.attention.sliding_window_pattern",
                &[true, false, true, false, true, false],
            )
            .i32s("gemma4.feed_forward_length", &[32, 32, 32, 64, 64, 64]);
        let c = config(&builder).unwrap();
        let heads: Vec<usize> = c.layers.iter().map(|layer| layer.head_dim).collect();
        assert_eq!(heads, [16, 32, 16, 32, 16, 32]);
        let sources: Vec<Option<usize>> = c.layers.iter().map(|layer| layer.kv_source).collect();
        // Layers 4 and 5 read layers 2 and 3, the last of their kinds.
        assert_eq!(sources, [None, None, None, None, Some(2), Some(3)]);
        assert_eq!(c.layers[5].n_ff, 64);
        assert!(c.value_norm && c.layers.iter().all(|layer| layer.attention_scale == 1.0));
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
        assert!(config(&builder).is_err(), "experts need a used count");
        builder.u32("llama.expert_used_count", 9);
        assert!(config(&builder).is_err(), "more used than there are");

        let mut builder = base("llama");
        builder.string("llama.rope.scaling.type", "dynamic");
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

        let mut builder = base("llama");
        builder.u32("llama.rope.dimension_count", 12);
        assert!(config(&builder).is_err());
    }

    #[test]
    fn linear_position_scaling_is_applied() {
        let mut builder = base("llama");
        builder.f32("llama.rope.scaling.factor", 4.0);
        assert_eq!(config(&builder).unwrap().layers[0].rope_scale, 0.25);
        assert_eq!(config(&builder).unwrap().yarn, None);
    }

    /// DeepSeek V2 Lite's metadata. `newer` states the widths as files with
    /// a split up-projection do, and `yarn` adds the model's YaRN.
    fn deepseek2_lite(newer: bool, yarn: bool) -> Builder {
        let mut builder = Builder::new();
        builder
            .string("general.architecture", "deepseek2")
            .u32("deepseek2.context_length", 163840)
            .u32("deepseek2.embedding_length", 2048)
            .u32("deepseek2.block_count", 27)
            .u32("deepseek2.feed_forward_length", 10944)
            .u32("deepseek2.attention.head_count", 16)
            .u32("deepseek2.attention.kv_lora_rank", 512)
            .u32("deepseek2.rope.dimension_count", 64)
            .u32("deepseek2.expert_count", 64)
            .u32("deepseek2.expert_used_count", 6);
        if newer {
            builder
                .u32("deepseek2.attention.head_count_kv", 1)
                .u32("deepseek2.attention.key_length", 576)
                .u32("deepseek2.attention.value_length", 512)
                .u32("deepseek2.attention.key_length_mla", 192)
                .u32("deepseek2.attention.value_length_mla", 192);
        } else {
            builder
                .u32("deepseek2.attention.head_count_kv", 16)
                .u32("deepseek2.attention.key_length", 192)
                .u32("deepseek2.attention.value_length", 128);
        }
        if yarn {
            builder
                .string("deepseek2.rope.scaling.type", "yarn")
                .f32("deepseek2.rope.scaling.factor", 40.0)
                .u32("deepseek2.rope.scaling.original_context_length", 4096)
                .f32("deepseek2.rope.scaling.yarn_log_multiplier", 0.0707);
        }
        builder
    }

    #[test]
    fn deepseek2_attends_through_a_latent() {
        let c = config(&deepseek2_lite(false, true)).unwrap();
        let mla = c.mla.unwrap();
        assert_eq!(
            (mla.kv_rank, mla.nope_dims, mla.value_dims),
            (512, 128, 128)
        );
        let layer = &c.layers[0];
        assert_eq!(
            (layer.n_head, layer.n_head_kv, layer.head_dim),
            (16, 1, 576)
        );
        assert_eq!((layer.rope_dims, c.rope), (64, Rope::Pairs));
        // llama.cpp's numbers for this model: YaRN's magnitude on rotated
        // values is canceled, and the scores take mscale twice.
        let stretch = 40f64.ln();
        let mscale = 1.0 + 0.1 * 0.707 * stretch;
        let expected = mscale * mscale / 192f64.sqrt();
        assert!((layer.attention_scale as f64 - expected).abs() < 1e-6);
        let rotary = c.yarn.unwrap().attn_factor as f64;
        assert!((rotary * (1.0 + 0.1 * stretch) - 1.0).abs() < 1e-6);
        // Experts are weighed by a softmax and not normalized by default.
        let routing = c.routing.unwrap();
        assert_eq!(
            (routing.gating, routing.normalize, routing.groups),
            (Gating::Softmax, false, 0)
        );
    }

    #[test]
    fn newer_deepseek2_files_give_the_cache_widths() {
        // The split up-projection's files, V3's routing and no YaRN.
        let groups = |count: u32, used: u32| {
            let mut builder = deepseek2_lite(true, false);
            builder
                .u32("deepseek2.expert_gating_func", 2)
                .bool("deepseek2.expert_weights_norm", true)
                .u32("deepseek2.expert_group_count", count)
                .u32("deepseek2.expert_group_used_count", used);
            config(&builder)
        };
        let c = groups(8, 3).unwrap();
        let mla = c.mla.unwrap();
        assert_eq!(
            (mla.kv_rank, mla.nope_dims, mla.value_dims),
            (512, 128, 192)
        );
        assert_eq!(c.layers[0].head_dim, 576);
        assert!((c.layers[0].attention_scale - 1.0 / 192f32.sqrt()).abs() < 1e-7);
        let routing = c.routing.unwrap();
        assert_eq!((routing.gating, routing.normalize), (Gating::Sigmoid, true));
        assert_eq!((routing.groups, routing.groups_used), (8, 3));

        // Groups must divide the experts and hold enough of them.
        assert!(groups(5, 3).is_err());
        assert!(groups(8, 0).is_err());
        assert!(groups(32, 1).is_err(), "two experts cannot give six");
        assert!(groups(32, 3).is_ok());
    }

    #[test]
    fn prediction_blocks_are_not_layers() {
        let with = |blocks: u32| {
            let mut builder = deepseek2_lite(false, true);
            builder.u32("deepseek2.nextn_predict_layers", blocks);
            config(&builder)
        };
        assert_eq!(with(1).unwrap().n_layer, 26);
        assert!(with(27).is_err());
    }

    #[test]
    fn deepseek_moe_neither_normalizes_nor_uses_a_sigmoid() {
        let mut builder = base("deepseek");
        builder
            .u32("deepseek.expert_count", 8)
            .u32("deepseek.expert_used_count", 2)
            .u32("deepseek.expert_gating_func", 2)
            .bool("deepseek.expert_weights_norm", true);
        let c = config(&builder).unwrap();
        let routing = c.routing.unwrap();
        assert_eq!(
            (routing.gating, routing.normalize),
            (Gating::Softmax, false)
        );
        assert_eq!((c.rope, c.mla), (Rope::Pairs, None));
    }

    #[test]
    fn yarn_settings_are_read() {
        let mut builder = base("llama");
        builder
            .string("llama.rope.scaling.type", "yarn")
            .f32("llama.rope.scaling.factor", 4.0)
            .f32("llama.rope.scaling.yarn_beta_fast", 16.0);
        let c = config(&builder).unwrap();
        assert_eq!(c.layers[0].rope_scale, 0.25);
        let yarn = c.yarn.unwrap();
        assert_eq!(yarn.original_context, c.n_ctx_train);
        assert_eq!((yarn.beta_fast, yarn.beta_slow), (16.0, 1.0));
        builder.u32("llama.rope.scaling.original_context_length", 64);
        assert_eq!(config(&builder).unwrap().yarn.unwrap().original_context, 64);
    }
}

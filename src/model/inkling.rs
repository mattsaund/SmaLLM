//! The weights of Inkling 2 and Inkling 2 Mini (Thinking Machines).
//!
//! Inkling's blocks differ from every other architecture here in what they
//! carry, so they are bound by this module rather than by `weights.rs`. The
//! settings are `Inkling` in `config.rs`, and the forward pass is in
//! `forward/inkling.rs`.
//!
//! Tensor names and shapes follow llama.cpp, which reads the same files.

use crate::error::{Error, Result};
use crate::gguf::Gguf;

use super::config::Config;
use super::weights::{Experts, Linear, experts, vector, whole};

/// How a layer normalizes the inputs of its attention.
pub enum InputNorms {
    /// A windowed layer: one norm. Its query, key and value read the same
    /// input, except in a later pass of a loop, where gates make them
    /// differ and each is normalized with this weight.
    Local(Vec<f32>),
    /// A full-attention layer. See [`GlobalNorms`].
    Global(Box<GlobalNorms>),
}

/// The input norms of a full-attention layer: one each for the query, key
/// and value inputs, which always differ. Such a layer also turns its key
/// and value inputs into extra convolution taps, one set per key/value
/// head.
pub struct GlobalNorms {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub k_taps: Linear,
    pub v_taps: Linear,
}

/// One Inkling block: attention, then a mixture of experts.
pub struct InklingLayer {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    /// The output projection of the attention.
    pub out: Linear,
    /// A sigmoid gate on each channel of the attention output.
    pub gate: Linear,
    /// Each head's relative-position query, `rel_len` wide.
    pub rel: Linear,
    /// Each head's sink score. It joins the softmax as a score of its own.
    pub sink: Linear,
    /// Per-element scales of the query heads and of the relative queries,
    /// applied before their length is capped. One head after another.
    pub q_clip: Vec<f32>,
    pub rel_clip: Vec<f32>,
    /// The bias table: for each distance in `0..rel_extent`, `rel_len`
    /// weights that a relative query dots with.
    pub rel_bias: Vec<f32>,
    pub rel_extent: usize,
    /// Short convolutions, `conv_taps` per channel with the oldest first:
    /// over the keys, the values, the attention output and the
    /// feed-forward output.
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub attn_conv: Vec<f32>,
    pub ffn_conv: Vec<f32>,
    pub norms: InputNorms,
    /// The queries that mix the earlier groups' outputs into the attention
    /// input: one for a windowed layer, one each for the query, key and
    /// value of a full-attention layer. `None` in the first layer.
    pub attn_res_query: Option<Vec<f32>>,
    /// The query that mixes them into the feed-forward input.
    pub ffn_res_query: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub experts: Experts,
}

/// How a message from the previous pass joins one input of a repeated
/// layer: `input - forget * input + gain * message`.
pub struct LoopGate {
    /// Maps the input and the message, each normalized, side by side, to a
    /// narrow bottleneck.
    pub down: Linear,
    /// Maps the bottleneck to the gain and the forgetting, through tanh.
    pub gain: Linear,
    pub forget: Linear,
}

/// What a repeated layer needs in its later passes.
pub struct LoopLayer {
    /// Four queries that mix the previous pass's outputs into messages: for
    /// the query, key and value inputs, and for the feed-forward input.
    pub message_query: Vec<f32>,
    pub gates: [LoopGate; 4],
}

/// Every weight of an Inkling model besides the token embeddings and the
/// output, which `Weights` holds as it does for other models.
pub struct InklingWeights {
    /// The token embeddings are RMS-normalized with this gain.
    pub embd_norm: Vec<f32>,
    /// The query that mixes the groups' outputs into the final hidden
    /// state.
    pub output_res_query: Vec<f32>,
    pub layers: Vec<InklingLayer>,
    /// The loop weights of the repeated layers, from `loop.start` on. Empty
    /// for a model without a loop.
    pub loop_layers: Vec<LoopLayer>,
}

/// Binds an Inkling model's weights. Sets each layer's expert width, which
/// the file gives through the tensors.
pub fn bind(file: &Gguf, config: &mut Config) -> Result<InklingWeights> {
    let inkling = config
        .inkling
        .ok_or_else(|| Error::invalid("not an Inkling model"))?;
    let routing = config
        .routing
        .ok_or_else(|| Error::format("Inkling without experts"))?;
    let d = config.n_embd;
    let (rel, taps) = (inkling.rel_len, inkling.conv_taps);
    let mut layers = Vec::with_capacity(config.n_layer);
    for (index, spec) in config.layers.iter_mut().enumerate() {
        let name = |part: &str| format!("blk.{index}.{part}");
        let (q_dim, kv_dim, n_head) = (spec.q_dim(), spec.kv_dim(), spec.n_head);
        let local = spec.window > 0;
        let rel_extent = if local {
            spec.window
        } else {
            inkling.rel_extent
        };
        let norms = if local {
            InputNorms::Local(vector(file, &name("attn_norm.weight"), d)?)
        } else {
            InputNorms::Global(Box::new(GlobalNorms {
                q: vector(file, &name("attn_q_inp_norm.weight"), d)?,
                k: vector(file, &name("attn_k_inp_norm.weight"), d)?,
                v: vector(file, &name("attn_v_inp_norm.weight"), d)?,
                k_taps: whole(file, &name("attn_k_conv_proj"), taps * spec.n_head_kv, d)?,
                v_taps: whole(file, &name("attn_v_conv_proj"), taps * spec.n_head_kv, d)?,
            }))
        };
        let attn_res_query = if index > 0 {
            let queries = if local { 1 } else { 3 };
            Some(vector(file, &name("attn_res_query.weight"), queries * d)?)
        } else {
            None
        };
        let experts = experts(file, index, d, routing.count)?;
        if experts.shared.is_some() || experts.biases.is_some() {
            return Err(Error::format("Inkling experts with unexpected tensors"));
        }
        spec.n_ff_expert = experts.gate.rows;
        layers.push(InklingLayer {
            q: whole(file, &name("attn_q"), q_dim, d)?,
            k: whole(file, &name("attn_k"), kv_dim, d)?,
            v: whole(file, &name("attn_v"), kv_dim, d)?,
            out: whole(file, &name("attn_output"), d, q_dim)?,
            gate: whole(file, &name("attn_gate"), q_dim, d)?,
            rel: whole(file, &name("attn_rel"), rel * n_head, d)?,
            sink: whole(file, &name("attn_sink_proj"), n_head, d)?,
            q_clip: vector(file, &name("attn_q_clip.weight"), q_dim)?,
            rel_clip: vector(file, &name("attn_rel_clip.weight"), rel * n_head)?,
            rel_bias: vector(file, &name("attn_rel_bias.weight"), rel * rel_extent)?,
            rel_extent,
            k_conv: vector(file, &name("attn_k_conv.weight"), taps * kv_dim)?,
            v_conv: vector(file, &name("attn_v_conv.weight"), taps * kv_dim)?,
            attn_conv: vector(file, &name("attn_post_conv.weight"), taps * d)?,
            ffn_conv: vector(file, &name("ffn_post_conv.weight"), taps * d)?,
            norms,
            attn_res_query,
            ffn_res_query: vector(file, &name("ffn_res_query.weight"), d)?,
            ffn_norm: vector(file, &name("ffn_norm.weight"), d)?,
            experts,
        });
    }

    let mut loop_layers = Vec::new();
    if let Some(lp) = inkling.looped {
        for index in lp.start..lp.end {
            let name = |part: &str| format!("blk.{index}.{part}");
            let gate = |which: &str| -> Result<LoopGate> {
                Ok(LoopGate {
                    down: whole(
                        file,
                        &name(&format!("loop_{which}_ctrl_down")),
                        lp.rank,
                        2 * d,
                    )?,
                    gain: whole(file, &name(&format!("loop_{which}_ctrl_gain")), d, lp.rank)?,
                    forget: whole(
                        file,
                        &name(&format!("loop_{which}_ctrl_forget")),
                        d,
                        lp.rank,
                    )?,
                })
            };
            loop_layers.push(LoopLayer {
                message_query: vector(file, &name("loop_msg_query.weight"), 4 * d)?,
                gates: [gate("q")?, gate("k")?, gate("v")?, gate("ffn")?],
            });
        }
    }

    Ok(InklingWeights {
        embd_norm: vector(file, "token_embd_norm.weight", d)?,
        output_res_query: vector(file, "output_res_query.weight", d)?,
        layers,
        loop_layers,
    })
}

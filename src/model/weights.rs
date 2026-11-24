//! Binds the tensors of a model file to the roles the forward pass uses.
//!
//! Weight matrices are never copied. A [`Matrix`] records where a tensor
//! sits in the file mapping, and the kernels read it from there. Only the
//! small vectors, norm weights, biases and convolution kernels, are expanded
//! to f32.
//!
//! Binding checks every shape against the configuration, so the forward
//! pass can index without further checks.

use crate::error::{Error, Result};
use crate::gguf::{Gguf, TensorInfo};
use crate::tensor::TensorType;
use crate::tensor::quant::dequantize_row;

use super::config::{Config, LayerSpec, Mixer, Mla};

/// A weight matrix stored in the model file, one row after another.
#[derive(Clone, Copy, Debug)]
pub struct Matrix {
    /// Element type of the stored rows.
    pub ty: TensorType,
    /// Number of rows: the output width.
    pub rows: usize,
    /// Elements per row: the input width.
    pub cols: usize,
    /// Byte offset of the first row from the start of the file.
    pub offset: usize,
    /// Bytes per row.
    pub row_bytes: usize,
}

/// A linear map: a run of rows of a matrix, plus an optional bias.
///
/// Most maps use a whole matrix. A model that stores its query, key and
/// value projections as one tensor yields three maps over that matrix.
#[derive(Clone, Debug)]
pub struct Linear {
    pub matrix: Matrix,
    /// First row of the matrix this map uses.
    pub first_row: usize,
    /// Number of rows: the output width of the map.
    pub rows: usize,
    pub bias: Option<Vec<f32>>,
}

/// The weights of an attention block.
pub struct AttentionWeights {
    pub q: Linear,
    /// Keys and values. `None` in a layer that reads another layer's cache.
    /// The value map is `None` when the keys, before their norm and
    /// rotation, serve as values (Gemma 4).
    pub kv: Option<(Linear, Option<Linear>)>,
    /// Normalization of queries and keys, when the model has it: per head,
    /// or over all heads at once when the weight is that wide.
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// One gate per head on the attention output, when the model has it:
    /// each head's output is scaled by the sigmoid of its gate.
    pub gate: Option<Linear>,
    /// One learned score per head that joins every softmax as if it were
    /// another position with no value (gpt-oss). It lets a head attend to
    /// nothing.
    pub sinks: Option<Vec<f32>>,
    pub out: Linear,
}

/// The weights of a short convolution block.
///
/// The file stores one input projection whose output is three vectors side
/// by side. They are bound as three maps: the input gate `b`, the output
/// gate `c` and the signal `x`.
pub struct ConvWeights {
    pub b: Linear,
    pub c: Linear,
    pub x: Linear,
    /// The kernel, tap by tap: `kernel[tap * n_embd + channel]`. The last
    /// tap applies to the current position.
    pub kernel: Vec<f32>,
    pub out_proj: Linear,
}

/// One matrix per attention head, stored in one tensor, one head's rows
/// after another's.
#[derive(Clone, Copy, Debug)]
pub struct HeadMatrix {
    pub matrix: Matrix,
    /// Rows of one head's matrix.
    pub rows: usize,
    /// Rows from one head's matrix to the next.
    pub stride: usize,
    /// Row where head zero's matrix starts.
    pub offset: usize,
}

impl HeadMatrix {
    /// Head `index`'s matrix as a linear map.
    pub fn head(&self, index: usize) -> Linear {
        Linear {
            matrix: self.matrix,
            first_row: index * self.stride + self.offset,
            rows: self.rows,
            bias: None,
        }
    }
}

/// How a latent attention block moves the unrotated part of each query
/// head into the latent space: through the transpose of the key half of
/// the head's up-projection.
pub enum Absorb {
    /// `attn_k_b`, which newer files store transposed already: per head,
    /// one row per latent element. An ordinary multiply applies it.
    Rows(HeadMatrix),
    /// The key half of an older file's `attn_kv_b`: per head, one row per
    /// element of the query's unrotated part. Each row is added to the
    /// result, scaled by its element of the query.
    Columns(HeadMatrix),
}

/// The weights of a latent attention block (DeepSeek V2 and V3). See
/// [`Mla`](super::config::Mla) for the shape.
pub struct MlaWeights {
    /// The compression of the hidden state that queries are made from, and
    /// its norm. `None` when `q` reads the hidden state itself.
    pub q_a: Option<(Linear, Vec<f32>)>,
    /// Every head's query: the unrotated part, then the rotary part.
    pub q: Linear,
    /// The latent, then the rotary key that all heads share.
    pub kv_a: Linear,
    /// Normalization of the latent.
    pub kv_a_norm: Vec<f32>,
    pub absorb: Absorb,
    /// The value half of each head's up-projection: from the latent the
    /// head attended to, to its value.
    pub values: HeadMatrix,
    pub out: Linear,
}

/// What a block mixes positions with.
pub enum MixerWeights {
    Attention(AttentionWeights),
    ShortConv(ConvWeights),
    Mla(MlaWeights),
}

/// A layer's part of the per-layer token embeddings (Gemma 4).
pub struct PerLayerWeights {
    /// Projects the hidden state down to the per-layer width.
    pub gate: Linear,
    /// Projects it back up.
    pub proj: Linear,
    pub post_norm: Vec<f32>,
}

/// A gated feed-forward network: `down(act(gate(x)) * up(x))`.
pub struct Dense {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

/// The same matrix of every expert, stored back to back in one tensor.
#[derive(Clone, Copy, Debug)]
pub struct ExpertMatrix {
    pub matrix: Matrix,
    /// Rows of one expert's matrix.
    pub rows: usize,
    /// Rows from one expert's matrix to the next.
    pub stride: usize,
    /// Row where expert zero's matrix starts.
    pub offset: usize,
}

impl ExpertMatrix {
    /// Expert `index`'s matrix as a linear map.
    pub fn expert(&self, index: usize) -> Linear {
        Linear {
            matrix: self.matrix,
            first_row: index * self.stride + self.offset,
            rows: self.rows,
            bias: None,
        }
    }
}

/// A mixture of experts: each token goes through a few of many small
/// feed-forward networks, chosen by a router.
pub struct Experts {
    /// Scores every expert for a token.
    pub router: Linear,
    /// Added to the scores when experts are chosen, but not when their
    /// outputs are weighed (LFM2 MoE).
    pub choice_bias: Option<Vec<f32>>,
    pub gate: ExpertMatrix,
    pub up: ExpertMatrix,
    pub down: ExpertMatrix,
    /// Each expert's biases on its gate, up and down projections (gpt-oss):
    /// one row per expert. They stay in the file until a row is used.
    pub biases: Option<[Matrix; 3]>,
    /// A network that every token goes through as well (Qwen2 MoE), with
    /// an optional gate: one score per token, through a sigmoid.
    pub shared: Option<(Dense, Option<Linear>)>,
}

/// An ungated feed-forward network: `down(act(up(x)))`.
pub struct Plain {
    pub up: Linear,
    pub down: Linear,
}

/// The feed-forward part of a block.
pub enum FeedForward {
    Dense(Box<Dense>),
    Experts(Box<Experts>),
    Plain(Box<Plain>),
}

/// The weights of one block.
pub struct Layer {
    pub attn_norm: Vec<f32>,
    /// The bias of a LayerNorm (Phi-2).
    pub attn_norm_bias: Option<Vec<f32>>,
    pub mixer: MixerWeights,
    /// Normalization of the mixer's output, when the model has it.
    pub attn_post_norm: Option<Vec<f32>>,
    /// Empty when attention and the feed-forward network share one norm.
    pub ffn_norm: Vec<f32>,
    pub ffn: FeedForward,
    /// Normalization of the feed-forward output, when the model has it.
    pub ffn_post_norm: Option<Vec<f32>>,
    pub per_layer: Option<PerLayerWeights>,
    /// A multiplier on the block's output, when the model has one.
    pub output_scale: Option<f32>,
}

/// Token embeddings that give every layer its own extra input (Gemma 4).
pub struct PerLayerInputs {
    /// One row per token: every layer's slice side by side.
    pub token_embd: Matrix,
    /// Projects the main embedding to the same shape.
    pub model_proj: Linear,
    /// Normalizes each layer's slice of that projection.
    pub proj_norm: Vec<f32>,
}

/// All weights of a model.
pub struct Weights {
    /// Token embeddings, one row per token.
    pub token_embd: Matrix,
    /// The output projection. The same matrix as `token_embd` when the
    /// model ties them.
    pub output: Linear,
    pub output_norm: Vec<f32>,
    /// The bias of a final LayerNorm (Phi-2).
    pub output_norm_bias: Option<Vec<f32>>,
    /// Per-frequency divisors for rotary angles, when the model has them.
    pub rope_factors: Option<Vec<f32>>,
    /// The divisors a LongRoPE model uses past its original context. Its
    /// short ones are in `rope_factors`.
    pub rope_factors_long: Option<Vec<f32>>,
    pub layers: Vec<Layer>,
    pub per_layer_inputs: Option<PerLayerInputs>,
}

impl Weights {
    /// Binds every tensor the configuration calls for. Fills in each
    /// layer's `n_ff` when the metadata left it out.
    pub fn bind(file: &Gguf, config: &mut Config) -> Result<Weights> {
        let d = config.n_embd;

        let token_embd = matrix(file, "token_embd.weight", config.n_vocab, d)?;
        let output_matrix = match file.tensor("output.weight") {
            Some(_) => matrix(file, "output.weight", config.n_vocab, d)?,
            None => token_embd,
        };
        let output = Linear {
            matrix: output_matrix,
            first_row: 0,
            rows: config.n_vocab,
            bias: optional_vector(file, "output.bias", config.n_vocab)?,
        };
        // LFM2 files name the final norm after the embeddings.
        let output_norm = match file.tensor("output_norm.weight") {
            Some(_) => vector(file, "output_norm.weight", d)?,
            None => vector(file, "token_embd_norm.weight", d)?,
        };
        let output_norm_bias = if config.layer_norm {
            Some(vector(file, "output_norm.bias", d)?)
        } else {
            None
        };
        // One table of rotary divisors serves every layer that uses one, so
        // those layers must rotate the same number of elements.
        let mut rotating = config.layers.iter().filter(|layer| {
            layer.mixer == Mixer::Attention && layer.rope_factors && layer.rope_dims > 0
        });
        let rope_factors = match rotating.next() {
            Some(first) if file.tensor("rope_freqs.weight").is_some() => {
                if rotating.any(|layer| layer.rope_dims != first.rope_dims) {
                    return Err(Error::unsupported(
                        "rotary divisors for layers that rotate different widths",
                    ));
                }
                optional_vector(file, "rope_freqs.weight", first.rope_dims / 2)?
            }
            _ => None,
        };
        // LongRoPE (Phi-3) stores short and long divisors under other names.
        let rotated = config
            .layers
            .iter()
            .find(|layer| layer.mixer == Mixer::Attention)
            .map_or(0, |layer| layer.rope_dims / 2);
        let rope_factors = match rope_factors {
            Some(factors) => Some(factors),
            None => optional_vector(file, "rope_factors_short.weight", rotated)?,
        };
        let rope_factors_long = optional_vector(file, "rope_factors_long.weight", rotated)?;

        let per_layer = config.per_layer_embd;
        let per_layer_inputs = if per_layer > 0 {
            let width = per_layer * config.n_layer;
            Some(PerLayerInputs {
                token_embd: matrix(file, "per_layer_token_embd.weight", config.n_vocab, width)?,
                model_proj: whole(file, "per_layer_model_proj", width, d)?,
                proj_norm: vector(file, "per_layer_proj_norm.weight", per_layer)?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.n_layer);
        for (index, spec) in config.layers.iter_mut().enumerate() {
            let name = |part: &str| format!("blk.{index}.{part}");
            let layer_whole =
                |part: &str, rows: usize, cols: usize| whole(file, &name(part), rows, cols);

            let mixer = match spec.mixer {
                Mixer::Attention if config.mla.is_some() => {
                    let shape = config
                        .mla
                        .expect("the arm requires a latent attention shape");
                    MixerWeights::Mla(mla(file, index, d, spec, &shape)?)
                }
                Mixer::Attention => {
                    let (q_dim, kv_dim) = (spec.q_dim(), spec.kv_dim());
                    let owns_kv = spec.owns_kv();
                    // Query, key and value are separate tensors or one fused
                    // tensor. A layer that reads another layer's cache has
                    // only a query.
                    let (q, kv) = if file.tensor(&name("attn_qkv.weight")).is_some() {
                        let fused = matrix(file, &name("attn_qkv.weight"), q_dim + 2 * kv_dim, d)?;
                        let bias =
                            optional_vector(file, &name("attn_qkv.bias"), q_dim + 2 * kv_dim)?;
                        let part = |first_row: usize, rows: usize| Linear {
                            matrix: fused,
                            first_row,
                            rows,
                            bias: bias
                                .as_ref()
                                .map(|b| b[first_row..first_row + rows].to_vec()),
                        };
                        (
                            part(0, q_dim),
                            Some((part(q_dim, kv_dim), Some(part(q_dim + kv_dim, kv_dim)))),
                        )
                    } else {
                        let q = layer_whole("attn_q", q_dim, d)?;
                        let kv = if owns_kv {
                            let k = layer_whole("attn_k", kv_dim, d)?;
                            // Gemma 4 may leave out the values, and then the
                            // keys serve as values.
                            let v = match file.tensor(&name("attn_v.weight")) {
                                None if config.value_norm => None,
                                _ => Some(layer_whole("attn_v", kv_dim, d)?),
                            };
                            Some((k, v))
                        } else {
                            None
                        };
                        (q, kv)
                    };
                    MixerWeights::Attention(AttentionWeights {
                        q,
                        kv: if owns_kv { kv } else { None },
                        q_norm: head_norm(file, &name("attn_q_norm.weight"), spec.head_dim, q_dim)?,
                        k_norm: head_norm(
                            file,
                            &name("attn_k_norm.weight"),
                            spec.head_dim,
                            kv_dim,
                        )?,
                        gate: match file.tensor(&name("attn_gate.weight")) {
                            Some(_) => Some(layer_whole("attn_gate", spec.n_head, d)?),
                            None => None,
                        },
                        sinks: optional_vector(file, &name("attn_sinks.weight"), spec.n_head)?,
                        out: layer_whole("attn_output", d, q_dim)?,
                    })
                }
                Mixer::ShortConv => {
                    let taps = config.conv_kernel;
                    let stored = vector(file, &name("shortconv.conv.weight"), taps * d)?;
                    // Stored channel by channel. The convolution reads it
                    // tap by tap.
                    let mut kernel = vec![0.0; taps * d];
                    for channel in 0..d {
                        for tap in 0..taps {
                            kernel[tap * d + channel] = stored[channel * taps + tap];
                        }
                    }
                    let in_proj = layer_whole("shortconv.in_proj", 3 * d, d)?;
                    let part = |index: usize| Linear {
                        matrix: in_proj.matrix,
                        first_row: index * d,
                        rows: d,
                        bias: in_proj
                            .bias
                            .as_ref()
                            .map(|bias| bias[index * d..(index + 1) * d].to_vec()),
                    };
                    MixerWeights::ShortConv(ConvWeights {
                        b: part(0),
                        c: part(1),
                        x: part(2),
                        kernel,
                        out_proj: layer_whole("shortconv.out_proj", d, d)?,
                    })
                }
            };

            let ffn = match &config.routing {
                Some(routing) if file.tensor(&name("ffn_gate_inp.weight")).is_some() => {
                    let experts = experts(file, index, d, routing.count)?;
                    spec.n_ff_expert = experts.gate.rows;
                    spec.n_ff = experts
                        .shared
                        .as_ref()
                        .map_or(0, |(shared, _)| shared.down.matrix.cols);
                    FeedForward::Experts(Box::new(experts))
                }
                _ => {
                    if spec.n_ff == 0 {
                        spec.n_ff = file
                            .tensor(&name("ffn_down.weight"))
                            .map(TensorInfo::cols)
                            .ok_or_else(|| {
                                Error::format(format!(
                                    "missing tensor: {}",
                                    name("ffn_down.weight")
                                ))
                            })?;
                    }
                    if config.ungated {
                        FeedForward::Plain(Box::new(Plain {
                            up: layer_whole("ffn_up", spec.n_ff, d)?,
                            down: layer_whole("ffn_down", d, spec.n_ff)?,
                        }))
                    } else {
                        FeedForward::Dense(Box::new(dense(file, &name("ffn"), "", d, spec.n_ff)?))
                    }
                }
            };

            let per_layer_weights = if per_layer > 0 {
                Some(PerLayerWeights {
                    gate: layer_whole("inp_gate", per_layer, d)?,
                    proj: layer_whole("proj", d, per_layer)?,
                    post_norm: vector(file, &name("post_norm.weight"), d)?,
                })
            } else {
                None
            };

            // gpt-oss names its feed-forward norm "post_attention_norm",
            // which elsewhere normalizes the attention output.
            let (attn_post_norm, ffn_norm) = if config.arch == "gpt-oss" {
                (None, vector(file, &name("post_attention_norm.weight"), d)?)
            } else if config.parallel {
                (None, Vec::new())
            } else {
                (
                    optional_vector(file, &name("post_attention_norm.weight"), d)?,
                    vector(file, &name("ffn_norm.weight"), d)?,
                )
            };
            layers.push(Layer {
                attn_norm: vector(file, &name("attn_norm.weight"), d)?,
                attn_norm_bias: if config.layer_norm {
                    Some(vector(file, &name("attn_norm.bias"), d)?)
                } else {
                    None
                },
                mixer,
                attn_post_norm,
                ffn_norm,
                ffn,
                ffn_post_norm: optional_vector(file, &name("post_ffw_norm.weight"), d)?,
                per_layer: per_layer_weights,
                output_scale: optional_vector(file, &name("layer_output_scale.weight"), 1)?
                    .map(|scale| scale[0]),
            });
        }
        Ok(Weights {
            token_embd,
            output,
            output_norm,
            output_norm_bias,
            rope_factors,
            rope_factors_long,
            layers,
            per_layer_inputs,
        })
    }
}

impl Weights {
    /// The decoder weights of an encoder model, which has none: only its
    /// token embeddings, shared with the encoder.
    pub(super) fn for_encoder(token_embd: Matrix) -> Weights {
        Weights {
            token_embd,
            output: Linear {
                matrix: token_embd,
                first_row: 0,
                rows: token_embd.rows,
                bias: None,
            },
            output_norm: Vec::new(),
            output_norm_bias: None,
            rope_factors: None,
            rope_factors_long: None,
            layers: Vec::new(),
            per_layer_inputs: None,
        }
    }
}

/// Binds a gated feed-forward network: `{prefix}_gate{suffix}`,
/// `{prefix}_up{suffix}` and `{prefix}_down{suffix}`. The gate is its own
/// tensor, or the first half of a double-height up matrix.
fn dense(file: &Gguf, prefix: &str, suffix: &str, d: usize, n_ff: usize) -> Result<Dense> {
    let part = |role: &str| format!("{prefix}_{role}{suffix}");
    let (gate, up) = if file.tensor(&format!("{}.weight", part("gate"))).is_some() {
        (
            whole(file, &part("gate"), n_ff, d)?,
            whole(file, &part("up"), n_ff, d)?,
        )
    } else {
        let fused = matrix(file, &format!("{}.weight", part("up")), 2 * n_ff, d)?;
        let half = |first_row: usize| Linear {
            matrix: fused,
            first_row,
            rows: n_ff,
            bias: None,
        };
        (half(0), half(n_ff))
    };
    Ok(Dense {
        gate,
        up,
        down: whole(file, &part("down"), d, n_ff)?,
    })
}

/// Binds the experts of layer `index`, and its shared expert if it has one.
fn experts(file: &Gguf, index: usize, d: usize, count: usize) -> Result<Experts> {
    let name = |part: &str| format!("blk.{index}.{part}");
    let down = file.tensor(&name("ffn_down_exps.weight")).ok_or_else(|| {
        Error::format(format!("missing tensor: {}", name("ffn_down_exps.weight")))
    })?;
    let n_ff = down.cols();
    // Gate and up are separate tensors, or one with each expert's gate
    // rows followed by its up rows.
    let (gate, up) = if file.tensor(&name("ffn_gate_up_exps.weight")).is_some() {
        let fused = stacked(file, &name("ffn_gate_up_exps.weight"), count, 2 * n_ff, d)?;
        let half = |offset: usize| ExpertMatrix {
            matrix: fused,
            rows: n_ff,
            stride: 2 * n_ff,
            offset,
        };
        (half(0), half(n_ff))
    } else {
        let each = |part: &str| -> Result<ExpertMatrix> {
            Ok(ExpertMatrix {
                matrix: stacked(file, &name(part), count, n_ff, d)?,
                rows: n_ff,
                stride: n_ff,
                offset: 0,
            })
        };
        (each("ffn_gate_exps.weight")?, each("ffn_up_exps.weight")?)
    };
    let shared = match file.tensor(&name("ffn_down_shexp.weight")) {
        Some(tensor) => {
            let n_shared = tensor.cols();
            let gate = match file.tensor(&name("ffn_gate_inp_shexp.weight")) {
                Some(_) => Some(whole(file, &name("ffn_gate_inp_shexp"), 1, d)?),
                None => None,
            };
            Some((dense(file, &name("ffn"), "_shexp", d, n_shared)?, gate))
        }
        None => None,
    };
    let biases = if file.tensor(&name("ffn_down_exps.bias")).is_some() {
        Some([
            matrix(file, &name("ffn_gate_exps.bias"), count, n_ff)?,
            matrix(file, &name("ffn_up_exps.bias"), count, n_ff)?,
            matrix(file, &name("ffn_down_exps.bias"), count, d)?,
        ])
    } else {
        None
    };
    Ok(Experts {
        // The router's bias, when it has one, is added to every score.
        router: whole(file, &name("ffn_gate_inp"), count, d)?,
        choice_bias: optional_vector(file, &name("exp_probs_b.bias"), count)?,
        gate,
        up,
        biases,
        down: ExpertMatrix {
            matrix: stacked(file, &name("ffn_down_exps.weight"), count, d, n_ff)?,
            rows: d,
            stride: d,
            offset: 0,
        },
        shared,
    })
}

/// Binds the latent attention block of layer `index`.
///
/// Queries come from the hidden state directly (`attn_q`), or through a
/// compression (`attn_q_a`, its norm, `attn_q_b`). The up-projection from
/// the latent to every head's key and value is one tensor in older files
/// (`attn_kv_b`, each head's key rows then its value rows). Newer files
/// store its two halves apart, the key half transposed (`attn_k_b`,
/// `attn_v_b`). Both are bound where they lie, without copying.
fn mla(file: &Gguf, index: usize, d: usize, spec: &LayerSpec, shape: &Mla) -> Result<MlaWeights> {
    let name = |part: &str| format!("blk.{index}.{part}");
    let n_head = spec.n_head;
    let (rank, nope, value) = (shape.kv_rank, shape.nope_dims, shape.value_dims);
    let q_width = n_head * (nope + spec.rope_dims);
    let (q_a, q) = match file.tensor(&name("attn_q_a.weight")) {
        Some(tensor) => {
            let q_rank = tensor.rows();
            let q_a = whole(file, &name("attn_q_a"), q_rank, d)?;
            let norm = vector(file, &name("attn_q_a_norm.weight"), q_rank)?;
            let q = whole(file, &name("attn_q_b"), q_width, q_rank)?;
            (Some((q_a, norm)), q)
        }
        None => (None, whole(file, &name("attn_q"), q_width, d)?),
    };
    let (absorb, values) = if file.tensor(&name("attn_k_b.weight")).is_some() {
        let per_head = |part: &str, rows: usize, cols: usize| -> Result<HeadMatrix> {
            Ok(HeadMatrix {
                matrix: stacked(file, &name(part), n_head, rows, cols)?,
                rows,
                stride: rows,
                offset: 0,
            })
        };
        (
            Absorb::Rows(per_head("attn_k_b.weight", rank, nope)?),
            per_head("attn_v_b.weight", value, rank)?,
        )
    } else {
        let stride = nope + value;
        let both = matrix(file, &name("attn_kv_b.weight"), n_head * stride, rank)?;
        let half = |rows: usize, offset: usize| HeadMatrix {
            matrix: both,
            rows,
            stride,
            offset,
        };
        (Absorb::Columns(half(nope, 0)), half(value, nope))
    };
    Ok(MlaWeights {
        q_a,
        q,
        kv_a: whole(file, &name("attn_kv_a_mqa"), rank + spec.rope_dims, d)?,
        kv_a_norm: vector(file, &name("attn_kv_a_norm.weight"), rank)?,
        absorb,
        values,
        out: whole(file, &name("attn_output"), d, n_head * value)?,
    })
}

/// Binds a tensor that holds one `rows` by `cols` matrix per expert, as one
/// tall matrix.
fn stacked(file: &Gguf, name: &str, count: usize, rows: usize, cols: usize) -> Result<Matrix> {
    let (tensor, ty) = supported(file, name)?;
    if tensor.dims != [cols, rows, count] {
        return Err(Error::format(format!(
            "tensor {name} has shape {:?}, expected [{cols}, {rows}, {count}]",
            tensor.dims
        )));
    }
    let row_bytes = ty
        .row_bytes(cols)
        .ok_or_else(|| Error::format(format!("tensor {name} has a bad row size")))?;
    Ok(Matrix {
        ty,
        rows: rows * count,
        cols,
        offset: tensor.offset,
        row_bytes,
    })
}

/// Binds the whole matrix `{name}.weight`, with `{name}.bias` when there is
/// one.
pub(super) fn whole(file: &Gguf, name: &str, rows: usize, cols: usize) -> Result<Linear> {
    Ok(Linear {
        matrix: matrix(file, &format!("{name}.weight"), rows, cols)?,
        first_row: 0,
        rows,
        bias: optional_vector(file, &format!("{name}.bias"), rows)?,
    })
}

/// Looks up a tensor and requires a type this engine computes with.
fn supported<'a>(file: &'a Gguf, name: &str) -> Result<(&'a TensorInfo, TensorType)> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| Error::format(format!("missing tensor: {name}")))?;
    let ty = tensor.ty.ok_or_else(|| {
        Error::unsupported(format!(
            "tensor {name} uses the {} format",
            tensor.type_name()
        ))
    })?;
    Ok((tensor, ty))
}

/// Binds the matrix `name`, which must have exactly `rows` by `cols`
/// elements.
pub(super) fn matrix(file: &Gguf, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
    let (tensor, ty) = supported(file, name)?;
    // A single row may be stored as a vector.
    let shape_ok = match tensor.dims.len() {
        1 => rows == 1 && tensor.dims[0] == cols,
        2 => tensor.cols() == cols && tensor.rows() == rows,
        _ => false,
    };
    if !shape_ok {
        return Err(Error::format(format!(
            "tensor {name} has shape {:?}, expected [{cols}, {rows}]",
            tensor.dims
        )));
    }
    // The file reader already checked that `cols` divides into whole blocks.
    let row_bytes = ty
        .row_bytes(cols)
        .ok_or_else(|| Error::format(format!("tensor {name} has a bad row size")))?;
    Ok(Matrix {
        ty,
        rows,
        cols,
        offset: tensor.offset,
        row_bytes,
    })
}

/// Reads the vector `name` of `len` elements as f32. A tensor of any shape
/// with that many elements will do.
pub(super) fn vector(file: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
    let (tensor, ty) = supported(file, name)?;
    if tensor.n_elements() != len {
        return Err(Error::format(format!(
            "tensor {name} has {} elements, expected {len}",
            tensor.n_elements()
        )));
    }
    let mut values = vec![0.0; len];
    dequantize_row(ty, file.tensor_data(tensor), &mut values);
    Ok(values)
}

/// A norm of queries or keys: one head wide, applied to each head, or as
/// wide as all heads together (OLMoE), applied to the whole vector.
fn head_norm(
    file: &Gguf,
    name: &str,
    head_dim: usize,
    all_heads: usize,
) -> Result<Option<Vec<f32>>> {
    match file.tensor(name) {
        Some(tensor) if tensor.n_elements() == all_heads => vector(file, name, all_heads).map(Some),
        _ => optional_vector(file, name, head_dim),
    }
}

/// Like [`vector`], for tensors a model may leave out.
pub(super) fn optional_vector(file: &Gguf, name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    match file.tensor(name) {
        Some(_) => vector(file, name, len).map(Some),
        None => Ok(None),
    }
}

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

use super::config::{Config, Mixer};

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

/// What a block mixes positions with.
pub enum MixerWeights {
    Attention(AttentionWeights),
    ShortConv(ConvWeights),
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
    /// A network that every token goes through as well (Qwen2 MoE), with
    /// an optional gate: one score per token, through a sigmoid.
    pub shared: Option<(Dense, Option<Linear>)>,
}

/// The feed-forward part of a block.
pub enum FeedForward {
    Dense(Box<Dense>),
    Experts(Box<Experts>),
}

/// The weights of one block.
pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub mixer: MixerWeights,
    /// Normalization of the mixer's output, when the model has it.
    pub attn_post_norm: Option<Vec<f32>>,
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
    /// Per-frequency divisors for rotary angles, when the model has them.
    pub rope_factors: Option<Vec<f32>>,
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
                    FeedForward::Dense(Box::new(dense(file, &name("ffn"), "", d, spec.n_ff)?))
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

            layers.push(Layer {
                attn_norm: vector(file, &name("attn_norm.weight"), d)?,
                mixer,
                attn_post_norm: optional_vector(file, &name("post_attention_norm.weight"), d)?,
                ffn_norm: vector(file, &name("ffn_norm.weight"), d)?,
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
            rope_factors,
            layers,
            per_layer_inputs,
        })
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
    Ok(Experts {
        router: whole(file, &name("ffn_gate_inp"), count, d)?,
        choice_bias: optional_vector(file, &name("exp_probs_b.bias"), count)?,
        gate,
        up,
        down: ExpertMatrix {
            matrix: stacked(file, &name("ffn_down_exps.weight"), count, d, n_ff)?,
            rows: d,
            stride: d,
            offset: 0,
        },
        shared,
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
fn whole(file: &Gguf, name: &str, rows: usize, cols: usize) -> Result<Linear> {
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
fn matrix(file: &Gguf, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
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
fn vector(file: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
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
fn optional_vector(file: &Gguf, name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    match file.tensor(name) {
        Some(_) => vector(file, name, len).map(Some),
        None => Ok(None),
    }
}

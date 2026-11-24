//! Binds the tensors of a model file to the roles the forward pass uses.
//!
//! Weight matrices are never copied. A [`Matrix`] records where a tensor
//! sits in the file mapping, and the kernels read it from there. Only the
//! small vectors, norm weights and biases, are expanded to f32.
//!
//! Binding checks every shape against the configuration, so the forward
//! pass can index without further checks.

use crate::error::{Error, Result};
use crate::gguf::{Gguf, TensorInfo};
use crate::tensor::TensorType;
use crate::tensor::quant::dequantize_row;

use super::config::Config;

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

/// The weights of one transformer block.
pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    /// Per-head normalization of queries and keys, when the model has it.
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub attn_out: Linear,
    /// Normalization of the attention output, when the model has it.
    pub attn_post_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
    /// Normalization of the feed-forward output, when the model has it.
    pub ffn_post_norm: Option<Vec<f32>>,
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
}

impl Weights {
    /// Binds every tensor the configuration calls for. Fills in
    /// `config.n_ff` when the metadata left it out.
    pub fn bind(file: &Gguf, config: &mut Config) -> Result<Weights> {
        let d = config.n_embd;
        let (q_dim, kv_dim) = (config.q_dim(), config.kv_dim());

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
        let output_norm = vector(file, "output_norm.weight", d)?;
        let rope_factors = optional_vector(file, "rope_freqs.weight", config.head_dim / 2)?;

        if config.n_ff == 0 {
            config.n_ff = file
                .tensor("blk.0.ffn_down.weight")
                .map(TensorInfo::cols)
                .ok_or_else(|| Error::format("missing tensor: blk.0.ffn_down.weight"))?;
        }
        let n_ff = config.n_ff;

        let mut layers = Vec::with_capacity(config.n_layer);
        for index in 0..config.n_layer {
            let name = |part: &str| format!("blk.{index}.{part}");
            let whole = |part: &str, rows: usize, cols: usize| -> Result<Linear> {
                Ok(Linear {
                    matrix: matrix(file, &name(&format!("{part}.weight")), rows, cols)?,
                    first_row: 0,
                    rows,
                    bias: optional_vector(file, &name(&format!("{part}.bias")), rows)?,
                })
            };

            // Query, key and value are three tensors or one fused tensor.
            let (q, k, v) = if file.tensor(&name("attn_qkv.weight")).is_some() {
                let fused = matrix(file, &name("attn_qkv.weight"), q_dim + 2 * kv_dim, d)?;
                let bias = optional_vector(file, &name("attn_qkv.bias"), q_dim + 2 * kv_dim)?;
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
                    part(q_dim, kv_dim),
                    part(q_dim + kv_dim, kv_dim),
                )
            } else {
                (
                    whole("attn_q", q_dim, d)?,
                    whole("attn_k", kv_dim, d)?,
                    whole("attn_v", kv_dim, d)?,
                )
            };

            // The gate is its own tensor, or the first half of a double
            // height `ffn_up`.
            let (ffn_gate, ffn_up) = if file.tensor(&name("ffn_gate.weight")).is_some() {
                (whole("ffn_gate", n_ff, d)?, whole("ffn_up", n_ff, d)?)
            } else {
                let fused = matrix(file, &name("ffn_up.weight"), 2 * n_ff, d)?;
                let part = |first_row: usize| Linear {
                    matrix: fused,
                    first_row,
                    rows: n_ff,
                    bias: None,
                };
                (part(0), part(n_ff))
            };

            layers.push(Layer {
                attn_norm: vector(file, &name("attn_norm.weight"), d)?,
                q,
                k,
                v,
                q_norm: optional_vector(file, &name("attn_q_norm.weight"), config.head_dim)?,
                k_norm: optional_vector(file, &name("attn_k_norm.weight"), config.head_dim)?,
                attn_out: whole("attn_output", d, q_dim)?,
                attn_post_norm: optional_vector(file, &name("post_attention_norm.weight"), d)?,
                ffn_norm: vector(file, &name("ffn_norm.weight"), d)?,
                ffn_gate,
                ffn_up,
                ffn_down: whole("ffn_down", d, n_ff)?,
                ffn_post_norm: optional_vector(file, &name("post_ffw_norm.weight"), d)?,
            });
        }
        Ok(Weights {
            token_embd,
            output,
            output_norm,
            rope_factors,
            layers,
        })
    }
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
    if tensor.dims.len() != 2 || tensor.cols() != cols || tensor.rows() != rows {
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

/// Reads the vector `name` of `len` elements as f32.
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

/// Like [`vector`], for tensors a model may leave out.
fn optional_vector(file: &Gguf, name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    match file.tensor(name) {
        Some(_) => vector(file, name, len).map(Some),
        None => Ok(None),
    }
}

//! Encoder models: BERT and its relatives, for embeddings and reranking.
//!
//! An encoder reads its whole input at once, every token seeing every other
//! one, and turns it into one vector. For an embedding model the vector
//! places texts of similar meaning near each other. For a reranker it is a
//! score of how well a document answers a query. There is no cache and no
//! generation, so this is a pass of its own, built from the same matrix
//! engine and kernels as the decoder's.
//!
//! A BERT block normalizes after each step, where decoders normalize
//! before:
//!
//! ```text
//! x = norm(x + attention(x))           every token sees every other one
//! x = norm(x + down(act(up(x))))       GELU, or a SiLU gate (nomic-bert)
//! ```
//!
//! BERT adds learned position embeddings to its input. nomic-bert rotates
//! queries and keys instead.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::kernels::{self, Kernels};
use crate::tensor::quant::dequantize_row;
use crate::threads::{ThreadPool, default_batch_threads};

use super::Model;
use super::config::{Activation, Config, LayerSpec, Mixer, Rope};
use super::matmul::{ActScratch, Engine, Target};
use super::weights::{Linear, Matrix, matrix, vector, whole};

/// Encoder architectures, as named in `general.architecture`.
pub const ENCODERS: [&str; 2] = ["bert", "nomic-bert"];

/// How the token outputs become one result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// The average of every token's output.
    Mean,
    /// The output of the first token, `[CLS]`.
    Cls,
    /// The output of the last token.
    Last,
    /// The first token's output through a classifier: a reranker's score.
    Rank,
}

/// The activation of a reranker's classifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClassifierActivation {
    Tanh,
    Gelu,
    Silu,
}

/// Layer normalization: `(x - mean) / deviation * weight + bias`.
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn bind(file: &Gguf, name: &str, len: usize) -> Result<LayerNorm> {
        Ok(LayerNorm {
            weight: vector(file, &format!("{name}.weight"), len)?,
            bias: vector(file, &format!("{name}.bias"), len)?,
        })
    }

    /// Normalizes one row in place. The statistics are taken in double
    /// precision, as ggml takes its sums.
    fn apply(&self, x: &mut [f32], eps: f32) {
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let scale = 1.0 / (variance + eps as f64).sqrt();
        for ((value, weight), bias) in x.iter_mut().zip(&self.weight).zip(&self.bias) {
            *value = ((*value as f64 - mean) * scale) as f32 * weight + bias;
        }
    }
}

/// The weights of one encoder block.
struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    attn_norm: LayerNorm,
    up: Linear,
    /// The SiLU gate of nomic-bert. BERT applies GELU to `up` alone.
    gate: Option<Linear>,
    down: Linear,
    out_norm: LayerNorm,
}

/// An encoder model's weights and shape.
pub struct Encoder {
    n_head: usize,
    head_dim: usize,
    n_ff: usize,
    eps: f32,
    /// The rotary base, for models that rotate rather than add positions.
    rope_base: Option<f32>,
    token_embd: Matrix,
    /// Learned position embeddings, one row per position.
    positions: Option<Matrix>,
    /// The embedding of token type zero, added to every token.
    token_type: Option<Vec<f32>>,
    embd_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    /// A reranker's classifier: a dense layer, then the output layer.
    classifier: Option<Linear>,
    classifier_out: Option<Linear>,
    classifier_activation: ClassifierActivation,
    pooling: Pooling,
}

impl Encoder {
    /// Binds an encoder model, and describes it as a [`Config`] for the
    /// parts of the program that show a model's shape.
    pub fn bind(file: &Gguf, arch: &str, n_vocab: usize) -> Result<(Config, Encoder)> {
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let required = |suffix: &str| {
            file.get_usize(&key(suffix))
                .filter(|&value| value > 0 && value < 1 << 20)
                .ok_or_else(|| {
                    Error::format(format!("missing or invalid metadata: {}", key(suffix)))
                })
        };
        let (d, n_layer, n_ff, n_head) = (
            required("embedding_length")?,
            required("block_count")?,
            required("feed_forward_length")?,
            required("attention.head_count")?,
        );
        let n_ctx_train = required("context_length")?;
        if d % n_head != 0 {
            return Err(Error::format("the heads do not divide the embedding"));
        }
        let head_dim = d / n_head;
        let eps = file
            .get_f32(&key("attention.layer_norm_epsilon"))
            .unwrap_or(1e-12);
        if file.get_bool(&key("attention.causal")) == Some(true) {
            return Err(Error::unsupported("encoder models with causal attention"));
        }
        // A file that does not say how to pool, and carries a classifier,
        // is a reranker. llama.cpp needs `--reranking` to know that.
        let has_classifier = file.tensor("cls.output.weight").is_some();
        let pooling = match file.get_usize(&key("pooling_type")) {
            Some(2) => Pooling::Cls,
            Some(3) => Pooling::Last,
            Some(4) => Pooling::Rank,
            None if has_classifier => Pooling::Rank,
            // None and mean both average here: one vector per text.
            _ => Pooling::Mean,
        };
        let classifier_activation = match file.get_str(&key("classifier.activation")) {
            None | Some("tanh") => ClassifierActivation::Tanh,
            Some("gelu") => ClassifierActivation::Gelu,
            Some("silu") => ClassifierActivation::Silu,
            Some(other) => {
                return Err(Error::unsupported(format!(
                    "classifier activation '{other}'"
                )));
            }
        };
        let rope_base =
            (arch == "nomic-bert").then(|| file.get_f32(&key("rope.freq_base")).unwrap_or(10000.0));

        let mut layers = Vec::with_capacity(n_layer);
        for index in 0..n_layer {
            let name = |part: &str| format!("blk.{index}.{part}");
            let (q, k, v) = if file.tensor(&name("attn_qkv.weight")).is_some() {
                let fused = whole(file, &name("attn_qkv"), 3 * d, d)?;
                let part = |at: usize| Linear {
                    matrix: fused.matrix,
                    first_row: at * d,
                    rows: d,
                    bias: fused
                        .bias
                        .as_ref()
                        .map(|bias| bias[at * d..(at + 1) * d].to_vec()),
                };
                (part(0), part(1), part(2))
            } else {
                (
                    whole(file, &name("attn_q"), d, d)?,
                    whole(file, &name("attn_k"), d, d)?,
                    whole(file, &name("attn_v"), d, d)?,
                )
            };
            layers.push(EncoderLayer {
                q,
                k,
                v,
                out: whole(file, &name("attn_output"), d, d)?,
                attn_norm: LayerNorm::bind(file, &name("attn_output_norm"), d)?,
                up: whole(file, &name("ffn_up"), n_ff, d)?,
                gate: match file.tensor(&name("ffn_gate.weight")) {
                    Some(_) => Some(whole(file, &name("ffn_gate"), n_ff, d)?),
                    None => None,
                },
                down: whole(file, &name("ffn_down"), d, n_ff)?,
                out_norm: LayerNorm::bind(file, &name("layer_output_norm"), d)?,
            });
        }

        let positions = match (rope_base, file.tensor("position_embd.weight")) {
            (None, Some(tensor)) => Some(matrix(file, "position_embd.weight", tensor.rows(), d)?),
            (None, None) => return Err(Error::format("missing tensor: position_embd.weight")),
            (Some(_), _) => None,
        };
        let token_type = match file.tensor("token_types.weight") {
            Some(tensor) => {
                let types = matrix(file, "token_types.weight", tensor.rows(), d)?;
                let mut row = vec![0.0; d];
                read_row(file, &types, 0, &mut row);
                Some(row)
            }
            None => None,
        };
        let classifier_out = match file.tensor("cls.output.weight") {
            Some(tensor) => Some(whole(file, "cls.output", tensor.rows(), d)?),
            None => None,
        };
        if pooling == Pooling::Rank
            && classifier_out.is_none()
            && file.tensor("cls.weight").is_none()
        {
            return Err(Error::format("a reranker without a classifier"));
        }
        let encoder = Encoder {
            n_head,
            head_dim,
            n_ff,
            eps,
            rope_base,
            token_embd: matrix(file, "token_embd.weight", n_vocab, d)?,
            positions,
            token_type,
            embd_norm: LayerNorm::bind(file, "token_embd_norm", d)?,
            layers,
            classifier: match file.tensor("cls.weight") {
                Some(_) => Some(whole(file, "cls", d, d)?),
                None => None,
            },
            classifier_out,
            classifier_activation,
            pooling,
        };
        // Only the fields the program shows matter for an encoder: the
        // decoder's forward pass never runs it.
        let layer = LayerSpec {
            mixer: Mixer::Attention,
            n_head,
            n_head_kv: n_head,
            head_dim,
            rope_dims: if rope_base.is_some() { head_dim } else { 0 },
            rope_base: rope_base.unwrap_or(0.0),
            rope_scale: 1.0,
            rope_factors: false,
            window: 0,
            kv_source: None,
            attention_scale: 1.0 / (head_dim as f32).sqrt(),
            n_ff,
            n_ff_expert: 0,
        };
        let config = Config {
            arch: arch.to_string(),
            name: file.get_str("general.name").unwrap_or("").to_string(),
            n_vocab,
            n_ctx_train,
            n_embd: d,
            n_layer,
            layers: vec![layer; n_layer],
            norm_eps: eps,
            rope: Rope::Halves,
            activation: Activation::Gelu,
            embedding_scale: 1.0,
            logit_softcap: 0.0,
            value_norm: false,
            conv_kernel: 0,
            per_layer_embd: 0,
            residual_scale: 1.0,
            logit_scale: 1.0,
            attention_softcap: 0.0,
            rope_attn_factor: 1.0,
            long_context_after: 0,
            yarn: None,
            layer_norm: true,
            parallel: false,
            ungated: false,
            routing: None,
            rwkv: None,
        };
        Ok((config, encoder))
    }

    /// How the model pools its output.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// The token embeddings.
    pub(super) fn token_embd(&self) -> Matrix {
        self.token_embd
    }

    /// The most tokens one input may have.
    pub fn max_tokens(&self) -> Option<usize> {
        self.positions.map(|positions| positions.rows)
    }
}

/// Reads row `row` of `matrix` as f32.
fn read_row(file: &Gguf, matrix: &Matrix, row: usize, out: &mut [f32]) {
    let start = matrix.offset + row * matrix.row_bytes;
    dequantize_row(
        matrix.ty,
        &file.mapping().as_slice()[start..start + matrix.row_bytes],
        out,
    );
}

/// GELU, in the tanh form ggml uses.
fn gelu(x: f32) -> f32 {
    const ROOT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (ROOT_2_OVER_PI * x * (1.0 + 0.044_715 * x * x)).tanh())
}

/// Runs an encoder model.
///
/// Holds a thread pool and nothing else between calls, so it costs a few
/// threads when idle. One input is read at a time.
pub struct Embedder {
    model: Arc<Model>,
    kernels: &'static Kernels,
    pool: ThreadPool,
    acts: ActScratch,
}

impl Embedder {
    /// Prepares to run `model`, which must be an encoder, on `threads`
    /// threads. Zero picks one per physical core: reading a whole input at
    /// once is the arithmetic work every core helps with.
    pub fn new(model: Arc<Model>, threads: usize) -> Result<Embedder> {
        if model.encoder().is_none() {
            return Err(Error::invalid(
                "this is not an embedding or reranking model",
            ));
        }
        let threads = if threads == 0 {
            default_batch_threads()
        } else {
            threads
        };
        if threads > 4096 {
            return Err(Error::invalid("thread count out of range"));
        }
        Ok(Embedder {
            model,
            kernels: kernels::best(),
            pool: ThreadPool::new(threads),
            acts: ActScratch::default(),
        })
    }

    /// The model this runs.
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// How the model pools its output.
    pub fn pooling(&self) -> Pooling {
        self.encoder().pooling
    }

    fn encoder(&self) -> &Encoder {
        self.model.encoder().expect("checked when made")
    }

    /// The pooled output for `tokens`: an embedding, or a reranker's
    /// scores. Embeddings are not normalized. See [`normalize`].
    pub fn embed(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let model = Arc::clone(&self.model);
        let (config, encoder) = (model.config(), model.encoder().expect("checked when made"));
        let (d, n) = (config.n_embd, tokens.len());
        if n == 0 {
            return Err(Error::invalid("the input is empty"));
        }
        if let Some(limit) = encoder.max_tokens() {
            if n > limit {
                return Err(Error::invalid(format!(
                    "the input is {n} tokens, and this model reads at most {limit}"
                )));
            }
        }
        if let Some(&bad) = tokens
            .iter()
            .find(|&&token| token as usize >= config.n_vocab)
        {
            return Err(Error::invalid(format!(
                "token id {bad} is outside the vocabulary"
            )));
        }
        let file = model.gguf();
        let engine = Engine {
            kernels: self.kernels,
            pool: &self.pool,
            map: file.mapping(),
        };
        let kernels = self.kernels;
        let eps = encoder.eps;

        // The input: token, type and position embeddings, normalized.
        let mut x = vec![0.0f32; n * d];
        let mut row = vec![0.0f32; d];
        for (position, (out, &token)) in x.chunks_exact_mut(d).zip(tokens).enumerate() {
            read_row(file, &encoder.token_embd, token as usize, out);
            if let Some(token_type) = &encoder.token_type {
                out.iter_mut().zip(token_type).for_each(|(v, t)| *v += t);
            }
            if let Some(positions) = &encoder.positions {
                read_row(file, positions, position, &mut row);
                out.iter_mut().zip(&row).for_each(|(v, p)| *v += p);
            }
            encoder.embd_norm.apply(out, eps);
        }

        let (heads, head_dim) = (encoder.n_head, encoder.head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (mut q, mut k, mut v) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
        let mut attended = vec![0.0; n * d];
        let mut h = vec![0.0; n * d];
        let mut up = vec![0.0; n * encoder.n_ff];
        let mut gate = vec![
            0.0;
            if encoder.layers.iter().any(|l| l.gate.is_some()) {
                n * encoder.n_ff
            } else {
                0
            }
        ];
        // One head's queries, keys and values side by side, and its scores.
        let (mut qh, mut kh, mut vh) = (
            vec![0.0; n * head_dim],
            vec![0.0; n * head_dim],
            vec![0.0; n * head_dim],
        );
        let mut scores = vec![0.0f32; n * n];
        let mut out_h = vec![0.0f32; n * head_dim];
        let angles = encoder
            .rope_base
            .map(|base| rotary_angles(base, head_dim, n));

        for layer in &encoder.layers {
            let mut targets = [
                Target::new(&layer.q, &mut q),
                Target::new(&layer.k, &mut k),
                Target::new(&layer.v, &mut v),
            ];
            engine.multiply(&mut self.acts, &x, d, &mut targets);
            if let Some(angles) = &angles {
                for (token, (q_row, k_row)) in
                    q.chunks_exact_mut(d).zip(k.chunks_exact_mut(d)).enumerate()
                {
                    let table = &angles[token * head_dim..(token + 1) * head_dim];
                    for head in q_row
                        .chunks_exact_mut(head_dim)
                        .chain(k_row.chunks_exact_mut(head_dim))
                    {
                        rotate_halves(head, table);
                    }
                }
            }
            // Attention, head by head, over every position.
            for head in 0..heads {
                let columns = head * head_dim..(head + 1) * head_dim;
                for (token, ((qr, kr), vr)) in qh
                    .chunks_exact_mut(head_dim)
                    .zip(kh.chunks_exact_mut(head_dim))
                    .zip(vh.chunks_exact_mut(head_dim))
                    .enumerate()
                {
                    let at = token * d;
                    qr.iter_mut()
                        .zip(&q[at + columns.start..at + columns.end])
                        .for_each(|(o, i)| *o = i * scale);
                    kr.copy_from_slice(&k[at + columns.start..at + columns.end]);
                    vr.copy_from_slice(&v[at + columns.start..at + columns.end]);
                }
                kernels.dot_rows(&qh, &kh, head_dim, &mut scores, n);
                for row in scores.chunks_exact_mut(n) {
                    kernels.softmax(row);
                }
                out_h.fill(0.0);
                kernels.mix_rows(&scores, &vh, head_dim, &mut out_h, n);
                for (token, o) in out_h.chunks_exact(head_dim).enumerate() {
                    let at = token * d;
                    attended[at + columns.start..at + columns.end].copy_from_slice(o);
                }
            }
            engine.multiply(
                &mut self.acts,
                &attended,
                d,
                &mut [Target::new(&layer.out, &mut h)],
            );
            for (x_row, h_row) in x.chunks_exact_mut(d).zip(h.chunks_exact(d)) {
                x_row.iter_mut().zip(h_row).for_each(|(a, b)| *a += b);
                layer.attn_norm.apply(x_row, eps);
            }

            // The feed-forward network.
            match &layer.gate {
                Some(gate_map) => {
                    let mut targets = [
                        Target::new(gate_map, &mut gate),
                        Target::new(&layer.up, &mut up),
                    ];
                    engine.multiply(&mut self.acts, &x, d, &mut targets);
                    kernels.silu_mul(&mut gate, &up);
                    engine.multiply(
                        &mut self.acts,
                        &gate,
                        encoder.n_ff,
                        &mut [Target::new(&layer.down, &mut h)],
                    );
                }
                None => {
                    engine.multiply(
                        &mut self.acts,
                        &x,
                        d,
                        &mut [Target::new(&layer.up, &mut up)],
                    );
                    up.iter_mut().for_each(|value| *value = gelu(*value));
                    engine.multiply(
                        &mut self.acts,
                        &up,
                        encoder.n_ff,
                        &mut [Target::new(&layer.down, &mut h)],
                    );
                }
            }
            for (x_row, h_row) in x.chunks_exact_mut(d).zip(h.chunks_exact(d)) {
                x_row.iter_mut().zip(h_row).for_each(|(a, b)| *a += b);
                layer.out_norm.apply(x_row, eps);
            }
        }

        // Pooling.
        let pooled = match encoder.pooling {
            Pooling::Mean => {
                let mut sum = vec![0.0f32; d];
                for row in x.chunks_exact(d) {
                    sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
                }
                sum.iter_mut().for_each(|s| *s /= n as f32);
                sum
            }
            Pooling::Cls | Pooling::Rank => x[..d].to_vec(),
            Pooling::Last => x[(n - 1) * d..].to_vec(),
        };
        if encoder.pooling != Pooling::Rank {
            return Ok(pooled);
        }
        let mut hidden = pooled;
        if let Some(classifier) = &encoder.classifier {
            let mut dense = vec![0.0f32; d];
            engine.multiply(
                &mut self.acts,
                &hidden,
                d,
                &mut [Target::new(classifier, &mut dense)],
            );
            for value in &mut dense {
                *value = match encoder.classifier_activation {
                    ClassifierActivation::Tanh => value.tanh(),
                    ClassifierActivation::Gelu => gelu_erf(*value),
                    ClassifierActivation::Silu => *value / (1.0 + (-*value).exp()),
                };
            }
            hidden = dense;
        }
        match &encoder.classifier_out {
            Some(out) => {
                let mut scores = vec![0.0f32; out.rows];
                engine.multiply(
                    &mut self.acts,
                    &hidden,
                    d,
                    &mut [Target::new(out, &mut scores)],
                );
                Ok(scores)
            }
            None => Ok(hidden),
        }
    }
}

/// GELU in its exact form, with the error function, as a classifier uses
/// it. The error function is evaluated by the Abramowitz and Stegun
/// approximation, good to 1.5e-7.
fn gelu_erf(x: f32) -> f32 {
    let z = (x as f64) / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = (1.0 - poly * (-z * z).exp()).copysign(z);
    (0.5 * x as f64 * (1.0 + erf)) as f32
}

/// `(cos, sin)` of each rotated pair, for positions `0..n`, rotating whole
/// heads in halves (NEOX order).
fn rotary_angles(base: f32, head_dim: usize, n: usize) -> Vec<f32> {
    let mut table = Vec::with_capacity(n * head_dim);
    for position in 0..n {
        for i in 0..head_dim / 2 {
            let frequency = (base as f64).powf(-2.0 * i as f64 / head_dim as f64);
            let (sin, cos) = (position as f64 * frequency).sin_cos();
            table.push(cos as f32);
            table.push(sin as f32);
        }
    }
    table
}

/// Rotates a head's halves by the angles of one position.
fn rotate_halves(head: &mut [f32], table: &[f32]) {
    let (low, high) = head.split_at_mut(head.len() / 2);
    for ((a, b), angle) in low.iter_mut().zip(high).zip(table.chunks_exact(2)) {
        let (x0, x1) = (*a, *b);
        *a = x0 * angle[0] - x1 * angle[1];
        *b = x0 * angle[1] + x1 * angle[0];
    }
}

/// Scales `vector` to length one, as embeddings are compared.
pub fn normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let scale = (1.0 / norm) as f32;
        vector.iter_mut().for_each(|v| *v *= scale);
    }
}

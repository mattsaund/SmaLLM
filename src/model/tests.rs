//! Forward pass tests on small synthetic models.
//!
//! The optimized pass is checked against `oracle`, a slow transcription of
//! the same math with no cache, no batching and no SIMD. Agreement with the
//! reference implementation on real models is checked by the golden tests
//! in `tests/`.

use std::sync::Arc;

use super::*;
use crate::error::Error;
use crate::kernels::{self, Kernels};
use crate::tensor::TensorType;
use crate::testing::TinyModel;
use kv::CHUNK_POSITIONS;

const ARCHS: [&str; 22] = [
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
    "rwkv6",
    "rwkv7",
];

const QUANT_TYPES: [TensorType; 25] = [
    TensorType::Q4_0,
    TensorType::Q4_1,
    TensorType::Q5_0,
    TensorType::Q5_1,
    TensorType::Q8_0,
    TensorType::Iq4Nl,
    TensorType::Q2K,
    TensorType::Q3K,
    TensorType::Q4K,
    TensorType::Q5K,
    TensorType::Q6K,
    TensorType::Iq4Xs,
    TensorType::Iq3S,
    TensorType::Iq3Xxs,
    TensorType::Iq2S,
    TensorType::Iq2Xs,
    TensorType::Iq2Xxs,
    TensorType::Iq1M,
    TensorType::Iq1S,
    TensorType::Tq2_0,
    TensorType::Tq1_0,
    TensorType::Mxfp4,
    TensorType::Nvfp4,
    TensorType::Q1_0,
    TensorType::Q2_0,
];

const PROMPT: [u32; 9] = [1, 5, 9, 12, 7, 30, 8, 8, 21];

/// A prompt long enough to fill a batch that decodes weight rows up front.
const LONG_PROMPT: [u32; 40] = [
    1, 5, 9, 12, 7, 30, 8, 8, 21, 3, 17, 4, 26, 11, 2, 19, 6, 23, 14, 29, 10, 1, 13, 27, 5, 18, 22,
    9, 31, 16, 7, 25, 20, 2, 28, 15, 12, 24, 6, 3,
];

fn load(spec: &TinyModel) -> Arc<Model> {
    Arc::new(Model::from_bytes(spec.build()).expect("the synthetic model loads"))
}

/// A session whose cache keeps keys and values exactly, so that it can be
/// held to the f64 oracle.
fn session(
    model: &Arc<Model>,
    threads: usize,
    batch: usize,
    kernels: Option<&'static Kernels>,
) -> Session {
    let config = SessionConfig {
        n_ctx: 2048,
        threads,
        batch,
        kernels,
        cache: CacheType::F32,
        ..Default::default()
    };
    Session::new(Arc::clone(model), config).unwrap()
}

/// The logits after evaluating `tokens` in a fresh session.
fn logits(model: &Arc<Model>, tokens: &[u32], threads: usize, batch: usize) -> Vec<f32> {
    let mut session = session(model, threads, batch, None);
    session.eval(tokens).unwrap();
    session.logits().to_vec()
}

/// Largest difference between two logit vectors, relative to their scale.
fn relative_error(got: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(got.len(), expected.len());
    let scale = expected
        .iter()
        .fold(1.0f64, |max, value| max.max(value.abs()));
    let worst = got
        .iter()
        .zip(expected)
        .fold(0.0f64, |max, (g, e)| max.max((*g as f64 - e).abs()));
    worst / scale
}

/// A direct implementation of the model in f64.
mod oracle {
    use super::super::*;
    use crate::tensor::quant::dequantize_row;

    /// Applies a linear map to one vector.
    fn apply(model: &Model, linear: &Linear, x: &[f64]) -> Vec<f64> {
        let matrix = &linear.matrix;
        let data = model.gguf().mapping().as_slice();
        let mut row = vec![0.0f32; matrix.cols];
        (0..linear.rows)
            .map(|r| {
                let start = matrix.offset + (linear.first_row + r) * matrix.row_bytes;
                dequantize_row(matrix.ty, &data[start..start + matrix.row_bytes], &mut row);
                let dot: f64 = row.iter().zip(x).map(|(w, v)| *w as f64 * v).sum();
                dot + linear.bias.as_ref().map_or(0.0, |bias| bias[r] as f64)
            })
            .collect()
    }

    /// Row `row` of a matrix.
    fn row_of(model: &Model, matrix: &Matrix, row: usize) -> Vec<f64> {
        let data = model.gguf().mapping().as_slice();
        let mut out = vec![0.0f32; matrix.cols];
        let start = matrix.offset + row * matrix.row_bytes;
        dequantize_row(matrix.ty, &data[start..start + matrix.row_bytes], &mut out);
        out.iter().map(|v| *v as f64).collect()
    }

    /// A whole tensor of the file, as stored.
    fn tensor(model: &Model, name: &str) -> Vec<f64> {
        let file = model.gguf();
        let info = file.tensor(name).expect("tensor exists");
        let mut out = vec![0.0f32; info.n_elements()];
        dequantize_row(info.ty.unwrap(), file.tensor_data(info), &mut out);
        out.iter().map(|v| *v as f64).collect()
    }

    fn rms_norm(x: &[f64], weight: Option<&[f32]>, eps: f32) -> Vec<f64> {
        let mean: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let scale = 1.0 / (mean + eps as f64).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| v * scale * weight.map_or(1.0, |w| w[i] as f64))
            .collect()
    }

    /// LayerNorm when there is a bias, RMS norm otherwise.
    fn norm(x: &[f64], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f64> {
        let Some(bias) = bias else {
            return rms_norm(x, Some(weight), eps);
        };
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        x.iter()
            .enumerate()
            .map(|(i, v)| {
                (v - mean) / (variance + eps as f64).sqrt() * weight[i] as f64 + bias[i] as f64
            })
            .collect()
    }

    fn rotate(
        head: &mut [f64],
        position: usize,
        spec: &LayerSpec,
        config: &Config,
        factors: Option<&[f32]>,
    ) {
        let n = spec.rope_dims;
        let mut scale = config.rope_attn_factor as f64;
        // YaRN as Hugging Face's transformers writes it: the correction
        // range, then a linear ramp between interpolated and extrapolated
        // frequencies.
        let ramp = config.yarn.map(|yarn| {
            let correction_dim = |rotations: f32| {
                let context = yarn.original_context as f64;
                n as f64 * (context / (rotations as f64 * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * (spec.rope_base as f64).ln())
            };
            let factor = 1.0 / spec.rope_scale as f64;
            scale *= yarn.attn_factor as f64 * (0.1 * factor.ln() + 1.0);
            let low = correction_dim(yarn.beta_fast).floor().max(0.0);
            let high = correction_dim(yarn.beta_slow).ceil().min(n as f64 - 1.0);
            (low, if low == high { high + 0.001 } else { high })
        });
        for i in 0..n / 2 {
            let frequency = (spec.rope_base as f64).powf(-2.0 * i as f64 / n as f64);
            let divisor = factors.map_or(1.0, |f| f[i] as f64);
            let extrapolation = frequency / divisor;
            let interpolation = extrapolation * spec.rope_scale as f64;
            let inv_freq = match ramp {
                Some((low, high)) => {
                    let linear = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let extrapolation_factor = 1.0 - linear;
                    interpolation * (1.0 - extrapolation_factor)
                        + extrapolation * extrapolation_factor
                }
                None => interpolation,
            };
            let angle = position as f64 * inv_freq;
            let (a, b) = match config.rope {
                Rope::Pairs => (2 * i, 2 * i + 1),
                Rope::Halves => (i, i + n / 2),
            };
            let (x0, x1) = (head[a], head[b]);
            head[a] = (x0 * angle.cos() - x1 * angle.sin()) * scale;
            head[b] = (x0 * angle.sin() + x1 * angle.cos()) * scale;
        }
    }

    /// The rotary divisors of a layer. The tests' sessions hold 2048
    /// tokens, past every synthetic model's original context, so a
    /// LongRoPE model uses its long ones.
    fn factors_of<'a>(model: &'a Model, spec: &LayerSpec) -> Option<&'a [f32]> {
        let w = model.weights();
        let factors = match &w.rope_factors_long {
            Some(long) if 2048 > model.config().long_context_after => Some(long.as_slice()),
            _ => w.rope_factors.as_deref(),
        };
        factors.filter(|_| spec.rope_factors)
    }

    /// The gate's activation times the up projection.
    fn activate(gate: f64, up: f64, activation: Activation) -> f64 {
        match activation {
            Activation::Silu => gate / (1.0 + (-gate).exp()) * up,
            Activation::Gelu => {
                let z = (2.0 / std::f64::consts::PI).sqrt() * (gate + 0.044715 * gate.powi(3));
                0.5 * gate * (1.0 + z.tanh()) * up
            }
            Activation::SwigluOai => {
                let (gate, up) = (gate.min(7.0), up.clamp(-7.0, 7.0));
                gate / (1.0 + (-1.702 * gate).exp()) * (up + 1.0)
            }
        }
    }

    /// A gated feed-forward network, with each projection's bias when it
    /// has one (gpt-oss experts).
    fn feed_forward(model: &Model, w: &Dense, h: &[f64]) -> Vec<f64> {
        feed_forward_biased(model, w, h, None)
    }

    fn feed_forward_biased(
        model: &Model,
        w: &Dense,
        h: &[f64],
        biases: Option<[Vec<f64>; 3]>,
    ) -> Vec<f64> {
        let activation = model.config().activation;
        let add = |mut values: Vec<f64>, index: usize| {
            if let Some(biases) = &biases {
                values
                    .iter_mut()
                    .zip(&biases[index])
                    .for_each(|(v, b)| *v += b);
            }
            values
        };
        let gate = add(apply(model, &w.gate, h), 0);
        let up = add(apply(model, &w.up, h), 1);
        let mixed: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| activate(*g, *u, activation))
            .collect();
        add(apply(model, &w.down, &mixed), 2)
    }

    /// A mixture of experts, following llama.cpp's routing step by step.
    fn mixture(model: &Model, w: &Experts, h: &[f64]) -> Vec<f64> {
        let c = model.config();
        let routing = c.routing.expect("routing");
        let scores = apply(model, &w.router, h);
        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        let softmax = |x: &[f64]| {
            let top = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = x.iter().map(|v| (v - top).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect::<Vec<f64>>()
        };
        let probs: Vec<f64> = match routing.gating {
            Gating::Softmax => softmax(&scores),
            Gating::Sigmoid => scores.iter().map(|&s| sigmoid(s)).collect(),
            Gating::SoftmaxChosen => scores.clone(),
        };
        let key = |e: usize| probs[e] + w.choice_bias.as_ref().map_or(0.0, |b| b[e] as f64);
        // Group-limited routing: groups are ranked by the sum of their two
        // best keys, and experts come from the best groups only.
        let mut allowed = vec![true; routing.count];
        if routing.groups > 1 {
            let size = routing.count / routing.groups;
            let mut ratings: Vec<(usize, f64)> = (0..routing.groups)
                .map(|group| {
                    let mut keys: Vec<f64> = (group * size..(group + 1) * size).map(key).collect();
                    keys.sort_by(|a, b| b.total_cmp(a));
                    (group, keys.iter().take(2).sum())
                })
                .collect();
            ratings.sort_by(|a, b| b.1.total_cmp(&a.1));
            allowed.fill(false);
            for &(group, _) in &ratings[..routing.groups_used] {
                allowed[group * size..(group + 1) * size].fill(true);
            }
        }
        let mut order: Vec<usize> = (0..routing.count).filter(|&e| allowed[e]).collect();
        order.sort_by(|&a, &b| key(b).total_cmp(&key(a)));
        let chosen = &order[..routing.used];
        let mut weights: Vec<f64> = chosen.iter().map(|&e| probs[e]).collect();
        if routing.gating == Gating::SoftmaxChosen {
            weights = softmax(&weights);
        }
        if routing.normalize {
            let sum = weights.iter().sum::<f64>().max(6.103515625e-5);
            weights.iter_mut().for_each(|w| *w /= sum);
        }
        weights.iter_mut().for_each(|w| *w *= routing.scale as f64);

        let mut out = vec![0.0; c.n_embd];
        for (&expert, weight) in chosen.iter().zip(weights) {
            let dense = Dense {
                gate: w.gate.expert(expert),
                up: w.up.expert(expert),
                down: w.down.expert(expert),
            };
            let biases = w
                .biases
                .as_ref()
                .map(|matrices| matrices.each_ref().map(|m| row_of(model, m, expert)));
            let result = feed_forward_biased(model, &dense, h, biases);
            out.iter_mut()
                .zip(result)
                .for_each(|(o, r)| *o += weight * r);
        }
        if let Some((shared, gate)) = &w.shared {
            let g = gate
                .as_ref()
                .map_or(1.0, |gate| sigmoid(apply(model, gate, h)[0]));
            let result = feed_forward(model, shared, h);
            out.iter_mut().zip(result).for_each(|(o, r)| *o += g * r);
        }
        out
    }

    fn add(x: &mut [f64], update: &[f64]) {
        x.iter_mut()
            .zip(update)
            .for_each(|(value, delta)| *value += delta);
    }

    /// Each layer's keys and values for every position, once computed.
    type Caches = [Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)>];

    /// Attention over every position of a layer, from scratch.
    fn attention(
        model: &Model,
        index: usize,
        a: &AttentionWeights,
        inputs: &[Vec<f64>],
        caches: &mut Caches,
    ) -> Vec<Vec<f64>> {
        let c = model.config();
        let spec = &c.layers[index];
        let hd = spec.head_dim;
        let factors = factors_of(model, spec);
        if let Some((wk, wv)) = &a.kv {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (position, h) in inputs.iter().enumerate() {
                let mut k = apply(model, wk, h);
                let mut v = match wv {
                    Some(wv) => apply(model, wv, h),
                    None => k.clone(),
                };
                // A norm as wide as all heads applies to the whole vector.
                if let Some(weight) = a.k_norm.as_deref().filter(|w| w.len() != hd) {
                    k = rms_norm(&k, Some(weight), c.norm_eps);
                }
                for head in k.chunks_exact_mut(hd) {
                    if let Some(weight) = a.k_norm.as_deref().filter(|w| w.len() == hd) {
                        head.copy_from_slice(&rms_norm(head, Some(weight), c.norm_eps));
                    }
                    rotate(head, position, spec, c, factors);
                }
                if c.value_norm {
                    for head in v.chunks_exact_mut(hd) {
                        head.copy_from_slice(&rms_norm(head, None, c.norm_eps));
                    }
                }
                keys.push(k);
                values.push(v);
            }
            caches[index] = Some((keys, values));
        }
        let (keys, values) = caches[spec.kv_source.unwrap_or(index)]
            .as_ref()
            .expect("the source layer ran first");
        let group = spec.n_head / spec.n_head_kv;
        inputs
            .iter()
            .enumerate()
            .map(|(position, h)| {
                let mut q = apply(model, &a.q, h);
                if let Some(weight) = a.q_norm.as_deref().filter(|w| w.len() != hd) {
                    q = rms_norm(&q, Some(weight), c.norm_eps);
                }
                for head in q.chunks_exact_mut(hd) {
                    if let Some(weight) = a.q_norm.as_deref().filter(|w| w.len() == hd) {
                        head.copy_from_slice(&rms_norm(head, Some(weight), c.norm_eps));
                    }
                    rotate(head, position, spec, c, factors);
                }
                let first = match spec.window {
                    0 => 0,
                    window => (position + 1).saturating_sub(window),
                };
                let gates = a.gate.as_ref().map(|gate| apply(model, gate, h));
                let mut attended = vec![0.0f64; spec.q_dim()];
                for head in 0..spec.n_head {
                    let kv_head = head / group * hd;
                    let q = &q[head * hd..(head + 1) * hd];
                    let scores: Vec<f64> = (first..=position)
                        .map(|j| {
                            let k = &keys[j][kv_head..kv_head + hd];
                            let score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>()
                                * spec.attention_scale as f64;
                            match c.attention_softcap as f64 {
                                0.0 => score,
                                cap => cap * (score / cap).tanh(),
                            }
                        })
                        .collect();
                    // A sink counts in the softmax's total as one more
                    // score, with no value.
                    let sink = a.sinks.as_ref().map(|sinks| sinks[head] as f64);
                    let max = scores
                        .iter()
                        .chain(&sink)
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = scores.iter().chain(&sink).map(|s| (s - max).exp()).sum();
                    let gate = gates
                        .as_ref()
                        .map_or(1.0, |g| 1.0 / (1.0 + (-g[head]).exp()));
                    for (offset, score) in scores.iter().enumerate() {
                        let weight = (score - max).exp() / total * gate;
                        let v = &values[first + offset][kv_head..kv_head + hd];
                        for (out, value) in attended[head * hd..(head + 1) * hd].iter_mut().zip(v) {
                            *out += weight * value;
                        }
                    }
                }
                apply(model, &a.out, &attended)
            })
            .collect()
    }

    /// A matrix of the file as `rows` vectors of `cols`, applied to `x`.
    fn product(matrix: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        assert_eq!(matrix.len(), rows * cols);
        matrix
            .chunks_exact(cols)
            .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
            .collect()
    }

    /// Latent attention over every position of a layer, from scratch and
    /// the way the paper writes it: each head rebuilds its keys and values
    /// from the latent, and the queries stay as projected. The engine does
    /// the opposite. Tensors are read by name, with the file's own layout.
    fn latent_attention(model: &Model, index: usize, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (c, file) = (model.config(), model.gguf());
        let spec = &c.layers[index];
        let (d, eps, n_head) = (c.n_embd, c.norm_eps, spec.n_head);
        let meta = |key: &str| file.get_usize(&format!("{}.{key}", c.arch)).unwrap();
        let name = |part: &str| format!("blk.{index}.{part}");
        let has = |part: &str| file.tensor(&name(part)).is_some();
        let weights = |part: &str| tensor(model, &name(part));
        let gains = |part: &str| -> Vec<f32> { weights(part).iter().map(|v| *v as f32).collect() };

        let (rank, rope) = (meta("attention.kv_lora_rank"), meta("rope.dimension_count"));
        let (key, value) = if has("attn_k_b.weight") {
            (
                meta("attention.key_length_mla"),
                meta("attention.value_length_mla"),
            )
        } else {
            (meta("attention.key_length"), meta("attention.value_length"))
        };
        let nope = key - rope;
        // Each head's key half `uk[h]` (nope rows of rank) and value half
        // `uv[h]` (value rows of rank). Newer files store the key half
        // transposed and the halves apart. Older ones stack each head's
        // key rows and value rows in one matrix.
        let mut uk = vec![vec![0.0; nope * rank]; n_head];
        let mut uv = vec![vec![0.0; value * rank]; n_head];
        if has("attn_k_b.weight") {
            let (k_b, v_b) = (weights("attn_k_b.weight"), weights("attn_v_b.weight"));
            for h in 0..n_head {
                for r in 0..rank {
                    for j in 0..nope {
                        uk[h][j * rank + r] = k_b[(h * rank + r) * nope + j];
                    }
                }
                uv[h].copy_from_slice(&v_b[h * value * rank..(h + 1) * value * rank]);
            }
        } else {
            let kv_b = weights("attn_kv_b.weight");
            for h in 0..n_head {
                let first = h * (nope + value) * rank;
                uk[h].copy_from_slice(&kv_b[first..first + nope * rank]);
                uv[h].copy_from_slice(&kv_b[first + nope * rank..first + (nope + value) * rank]);
            }
        }

        // Every position's keys and values, head by head.
        let kv_a = weights("attn_kv_a_mqa.weight");
        let latent_gain = gains("attn_kv_a_norm.weight");
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (position, h) in inputs.iter().enumerate() {
            let projected = product(&kv_a, rank + rope, d, h);
            let latent = rms_norm(&projected[..rank], Some(&latent_gain), eps);
            let mut shared = projected[rank..].to_vec();
            rotate(&mut shared, position, spec, c, None);
            let mut position_keys = Vec::new();
            let mut position_values = Vec::new();
            for h in 0..n_head {
                let mut head_key = product(&uk[h], nope, rank, &latent);
                head_key.extend(&shared);
                position_keys.push(head_key);
                position_values.push(product(&uv[h], value, rank, &latent));
            }
            keys.push(position_keys);
            values.push(position_values);
        }

        let out = weights("attn_output.weight");
        inputs
            .iter()
            .enumerate()
            .map(|(position, h)| {
                let q = if has("attn_q_a.weight") {
                    let q_a = weights("attn_q_a.weight");
                    let q_rank = q_a.len() / d;
                    let compressed = product(&q_a, q_rank, d, h);
                    let compressed =
                        rms_norm(&compressed, Some(&gains("attn_q_a_norm.weight")), eps);
                    product(
                        &weights("attn_q_b.weight"),
                        n_head * key,
                        q_rank,
                        &compressed,
                    )
                } else {
                    product(&weights("attn_q.weight"), n_head * key, d, h)
                };
                let mut attended = Vec::new();
                for (head, query) in q.chunks_exact(key).enumerate() {
                    let mut query = query.to_vec();
                    rotate(&mut query[nope..], position, spec, c, None);
                    let scores: Vec<f64> = (0..=position)
                        .map(|j| {
                            let dot: f64 =
                                query.iter().zip(&keys[j][head]).map(|(a, b)| a * b).sum();
                            dot * spec.attention_scale as f64
                        })
                        .collect();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                    let mut mixed = vec![0.0; value];
                    for (j, score) in scores.iter().enumerate() {
                        let weight = (score - max).exp() / total;
                        for (out, v) in mixed.iter_mut().zip(&values[j][head]) {
                            *out += weight * v;
                        }
                    }
                    attended.extend(mixed);
                }
                product(&out, d, n_head * value, &attended)
            })
            .collect()
    }

    /// A short convolution over every position of a layer, from scratch.
    /// The kernel is read as the file stores it, channel by channel.
    fn short_conv(
        model: &Model,
        index: usize,
        cw: &ConvWeights,
        inputs: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let c = model.config();
        let (d, taps) = (c.n_embd, c.conv_kernel);
        let kernel = tensor(model, &format!("blk.{index}.shortconv.conv.weight"));
        let gated: Vec<Vec<f64>> = inputs
            .iter()
            .map(|h| {
                let b = apply(model, &cw.b, h);
                let signal = apply(model, &cw.x, h);
                b.iter().zip(&signal).map(|(b, s)| b * s).collect()
            })
            .collect();
        inputs
            .iter()
            .enumerate()
            .map(|(position, h)| {
                let gate = apply(model, &cw.c, h);
                let mixed: Vec<f64> = (0..d)
                    .map(|channel| {
                        // Tap `taps - 1` is the current position.
                        let sum: f64 = (0..taps)
                            .filter_map(|tap| {
                                let earlier = (position + tap).checked_sub(taps - 1)?;
                                Some(kernel[channel * taps + tap] * gated[earlier][channel])
                            })
                            .sum();
                        gate[channel] * sum
                    })
                    .collect();
                apply(model, &cw.out_proj, &mixed)
            })
            .collect()
    }

    /// A Mamba block over every position, from scratch. It reads the small
    /// tensors as the file stores them, applies whole projection matrices
    /// and splits their outputs where llama.cpp does, and carries the
    /// state in f64 with the formulas of the Mamba papers.
    fn mamba(model: &Model, index: usize, w: &MambaWeights, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let c = model.config();
        let Mixer::Mamba(ssm) = c.layers[index].mixer else {
            panic!("layer {index} is not a Mamba block");
        };
        let stored = |part: &str| tensor(model, &format!("blk.{index}.{part}"));
        let whole = |matrix: &Matrix| Linear {
            matrix: *matrix,
            first_row: 0,
            rows: matrix.rows,
            bias: None,
        };
        let silu = |x: f64| x / (1.0 + (-x).exp());
        let softplus = |x: f64| (1.0 + x.exp()).ln();
        let (inner, n, taps, groups) = (ssm.inner, ssm.state, ssm.conv_kernel, ssm.groups);
        let heads = if ssm.mamba2 { ssm.dt_rank } else { inner };
        let head_dim = inner / heads;
        let channels = if ssm.mamba2 {
            inner + 2 * groups * n
        } else {
            inner
        };
        // The kernel is stored channel by channel.
        let (kernel, conv_bias) = (stored("ssm_conv1d.weight"), stored("ssm_conv1d.bias"));
        let (a, d, dt_bias) = (stored("ssm_a"), stored("ssm_d"), stored("ssm_dt.bias"));

        // Mamba 1's input projection holds x, then z. Mamba 2's holds z,
        // then x with B and C, then the time steps.
        let mut signal = Vec::new();
        let mut gates = Vec::new();
        let mut steps = Vec::new();
        for h in inputs {
            let all = apply(model, &whole(&w.x.matrix), h);
            if ssm.mamba2 {
                gates.push(all[..inner].to_vec());
                signal.push(all[inner..inner + channels].to_vec());
                let raw = &all[inner + channels..];
                steps.push(raw.iter().zip(&dt_bias).map(|(s, b)| s + b).collect());
            } else {
                signal.push(all[..inner].to_vec());
                gates.push(all[inner..].to_vec());
                steps.push(Vec::new());
            }
        }
        // The causal convolution sees zeros before the first position.
        let convolved: Vec<Vec<f64>> = (0..inputs.len())
            .map(|position| {
                (0..channels)
                    .map(|channel| {
                        let sum: f64 = (0..taps)
                            .filter_map(|tap| {
                                let earlier = (position + tap).checked_sub(taps - 1)?;
                                Some(kernel[channel * taps + tap] * signal[earlier][channel])
                            })
                            .sum();
                        silu(sum + conv_bias[channel])
                    })
                    .collect()
            })
            .collect();

        let gain: Vec<f32> = match ssm.mamba2 {
            true => stored("ssm_norm.weight")
                .iter()
                .map(|v| *v as f32)
                .collect(),
            false => Vec::new(),
        };
        let mut state = vec![0.0f64; inner * n];
        let mut outputs = Vec::new();
        for (position, x) in convolved.iter().enumerate() {
            let mut y = vec![0.0f64; inner];
            match &w.low_rank {
                // Mamba 1: every channel is a head, and every element of
                // its state decays at its own rate.
                Some(low_rank) => {
                    let rank = ssm.dt_rank;
                    let all = apply(model, &whole(&low_rank.dt.matrix), x);
                    let mut low = all[..rank].to_vec();
                    let mut b = all[rank..rank + n].to_vec();
                    let mut cc = all[rank + n..rank + 2 * n].to_vec();
                    if ssm.dt_b_c_norm {
                        low = rms_norm(&low, None, c.norm_eps);
                        b = rms_norm(&b, None, c.norm_eps);
                        cc = rms_norm(&cc, None, c.norm_eps);
                    }
                    let dt = apply(model, &w.dt, &low);
                    for channel in 0..inner {
                        let delta = softplus(dt[channel]);
                        for s in 0..n {
                            let at = channel * n + s;
                            let decay = (delta * a[at]).exp();
                            state[at] = state[at] * decay + delta * b[s] * x[channel];
                            y[channel] += state[at] * cc[s];
                        }
                        y[channel] += d[channel] * x[channel];
                    }
                }
                // Mamba 2: a head decays as one, and reads the B and C of
                // its group, which follow the signal.
                None => {
                    for head in 0..heads {
                        let delta = softplus(steps[position][head]);
                        let decay = (delta * a[head]).exp();
                        let group = head / (heads / groups);
                        let b = &x[inner + group * n..][..n];
                        let cc = &x[inner + (groups + group) * n..][..n];
                        for channel in head * head_dim..(head + 1) * head_dim {
                            for s in 0..n {
                                let at = channel * n + s;
                                state[at] = state[at] * decay + delta * b[s] * x[channel];
                                y[channel] += state[at] * cc[s];
                            }
                            y[channel] += d[head] * x[channel];
                        }
                    }
                }
            }
            let mut gated: Vec<f64> = y
                .iter()
                .zip(&gates[position])
                .map(|(y, z)| y * silu(*z))
                .collect();
            if ssm.mamba2 {
                let width = inner / groups;
                gated = gated
                    .chunks(width)
                    .zip(gain.chunks(width))
                    .flat_map(|(part, gain)| rms_norm(part, Some(gain), c.norm_eps))
                    .collect();
            }
            outputs.push(apply(model, &w.out, &gated));
        }
        outputs
    }

    /// A matrix of the file applied to `x`, read by name rather than from
    /// the bound weights.
    fn by_name(model: &Model, name: &str, x: &[f64]) -> Vec<f64> {
        let matrix = tensor(model, name);
        matrix
            .chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
            .collect()
    }

    /// Layer normalization in f64, with the file's weight and bias.
    fn layer_norm(model: &Model, x: &[f64], name: &str) -> Vec<f64> {
        let weight = tensor(model, &format!("{name}.weight"));
        let bias = tensor(model, &format!("{name}.bias"));
        let mut out = standardize(x, model.config().norm_eps as f64);
        for ((value, w), b) in out.iter_mut().zip(weight).zip(bias) {
            *value = *value * w + b;
        }
        out
    }

    /// `x` shifted to zero mean and scaled to unit variance.
    fn standardize(x: &[f64], eps: f64) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        x.iter()
            .map(|v| (v - mean) / (variance + eps).sqrt())
            .collect()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// What one RWKV block carries from token to token: the previous
    /// token's normalized inputs to its two parts, and each head's matrix.
    struct RwkvState {
        time_shift: Vec<f64>,
        channel_shift: Vec<f64>,
        heads: Vec<Vec<Vec<f64>>>,
    }

    /// The logits of an RWKV model after `tokens`, one token at a time as
    /// llama.cpp's graphs for RWKV-6 and RWKV-7 write it. Row `i` of a
    /// head's matrix is value channel `i` in RWKV-7 and key channel `i` in
    /// RWKV-6, as in llama.cpp's state.
    fn rwkv_logits(model: &Model, tokens: &[u32]) -> Vec<f64> {
        let c = model.config();
        let v7 = c.arch == "rwkv7";
        let d = c.n_embd;
        let n = c.rwkv.expect("an RWKV model").head_size;
        let heads = d / n;
        let rescale = c.rwkv.map_or(0, |rwkv| rwkv.rescale_every);
        let mut states: Vec<RwkvState> = (0..c.n_layer)
            .map(|_| RwkvState {
                time_shift: vec![0.0; d],
                channel_shift: vec![0.0; d],
                heads: vec![vec![vec![0.0; n]; n]; heads],
            })
            .collect();
        let mut x = Vec::new();
        for &token in tokens {
            let embedding = row_of(model, &model.weights().token_embd, token as usize);
            x = layer_norm(model, &embedding, "token_embd_norm");
            let mut first_value: Vec<f64> = Vec::new();
            for (layer, state) in states.iter_mut().enumerate() {
                let t = |part: &str| tensor(model, &format!("blk.{layer}.{part}.weight"));
                let mat = |part: &str, input: &[f64]| {
                    by_name(model, &format!("blk.{layer}.{part}.weight"), input)
                };

                // Time mixing.
                let xa = layer_norm(model, &x, &format!("blk.{layer}.attn_norm"));
                let sx: Vec<f64> = state
                    .time_shift
                    .iter()
                    .zip(&xa)
                    .map(|(p, c)| p - c)
                    .collect();
                state.time_shift = xa.clone();
                let moved = |amounts: &[f64]| -> Vec<f64> {
                    xa.iter()
                        .zip(&sx)
                        .zip(amounts)
                        .map(|((x, s), a)| x + s * a)
                        .collect()
                };
                let mut y = vec![0.0; d];
                let (r, k, v, gate);
                if v7 {
                    let lerp = t("time_mix_lerp_fused");
                    let input = |part: usize| moved(&lerp[part * d..(part + 1) * d]);
                    let (xr, xw, xk, xv, xa_, xg) =
                        (input(0), input(1), input(2), input(3), input(4), input(5));
                    let rr = mat("time_mix_receptance", &xr);
                    let kk0 = mat("time_mix_key", &xk);
                    let mut vv = mat("time_mix_value", &xv);
                    let low = |part: &str, input: &[f64], f: fn(f64) -> f64| {
                        let inner: Vec<f64> = mat(&format!("time_mix_{part}1"), input)
                            .into_iter()
                            .map(f)
                            .collect();
                        mat(&format!("time_mix_{part}2"), &inner)
                    };
                    let w0 = t("time_mix_w0");
                    let decay: Vec<f64> = low("w", &xw, f64::tanh)
                        .iter()
                        .zip(&w0)
                        .map(|(l, b)| (-0.606531 * sigmoid(b + l)).exp())
                        .collect();
                    if layer == 0 {
                        first_value = vv.clone();
                    } else {
                        let v0 = t("time_mix_v0");
                        let mix = low("v", &xv, |x| x);
                        for i in 0..d {
                            vv[i] += (first_value[i] - vv[i]) * sigmoid(v0[i] + mix[i]);
                        }
                    }
                    let g = low("g", &xg, sigmoid);
                    let a0 = t("time_mix_a0");
                    let a: Vec<f64> = low("a", &xa_, |x| x)
                        .iter()
                        .zip(&a0)
                        .map(|(l, b)| sigmoid(b + l))
                        .collect();
                    let (k_k, k_a) = (t("time_mix_k_k"), t("time_mix_k_a"));
                    let mut kk: Vec<f64> = kk0.iter().zip(&k_k).map(|(k, s)| k * s).collect();
                    for head in kk.chunks_exact_mut(n) {
                        let norm = head.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                        head.iter_mut().for_each(|v| *v /= norm);
                    }
                    let key: Vec<f64> = (0..d)
                        .map(|i| kk0[i] + (a[i] * kk0[i] * k_a[i] - kk0[i] * k_a[i]))
                        .collect();
                    for (head, matrix) in state.heads.iter_mut().enumerate() {
                        let span = head * n..(head + 1) * n;
                        for (i, row) in matrix.iter_mut().enumerate() {
                            let sa: f64 = span.clone().map(|j| -kk[j] * row[j - span.start]).sum();
                            let vi = vv[span.start + i];
                            let mut out = 0.0;
                            for j in span.clone() {
                                let s = &mut row[j - span.start];
                                *s = *s * decay[j] + vi * key[j] + sa * kk[j] * a[j];
                                out += *s * rr[j];
                            }
                            y[span.start + i] = out;
                        }
                    }
                    (r, k, v, gate) = (rr, key, vv, g);
                } else {
                    let xxx = moved(&t("time_mix_lerp_x"));
                    let inner: Vec<f64> = mat("time_mix_w1", &xxx)
                        .into_iter()
                        .map(f64::tanh)
                        .collect();
                    let rank = inner.len() / 5;
                    let up = t("time_mix_w2");
                    let lerp = t("time_mix_lerp_fused");
                    // Part p's correction: matrix p of the stacked tensor
                    // applied to chunk p of the inner vector.
                    let input = |part: usize| -> Vec<f64> {
                        let chunk = &inner[part * rank..(part + 1) * rank];
                        (0..d)
                            .map(|i| {
                                let at = (part * d + i) * rank;
                                let correction: f64 = up[at..at + rank]
                                    .iter()
                                    .zip(chunk)
                                    .map(|(w, x)| w * x)
                                    .sum();
                                xa[i] + sx[i] * (lerp[part * d + i] + correction)
                            })
                            .collect()
                    };
                    let (xw, xk, xv, xr, xg) = (input(0), input(1), input(2), input(3), input(4));
                    let rr = mat("time_mix_receptance", &xr);
                    let key = mat("time_mix_key", &xk);
                    let vv = mat("time_mix_value", &xv);
                    let g: Vec<f64> = mat("time_mix_gate", &xg)
                        .into_iter()
                        .map(|x| x * sigmoid(x))
                        .collect();
                    let inner: Vec<f64> = mat("time_mix_decay_w1", &xw)
                        .into_iter()
                        .map(f64::tanh)
                        .collect();
                    let base = t("time_mix_decay");
                    let decay: Vec<f64> = mat("time_mix_decay_w2", &inner)
                        .iter()
                        .zip(&base)
                        .map(|(l, b)| (-(b + l).exp()).exp())
                        .collect();
                    let u = t("time_mix_first");
                    for (head, matrix) in state.heads.iter_mut().enumerate() {
                        let start = head * n;
                        for (i, row) in matrix.iter_mut().enumerate() {
                            let key_i = start + i;
                            for (j, s) in row.iter_mut().enumerate() {
                                let kv = key[key_i] * vv[start + j];
                                y[start + j] += rr[key_i] * (u[key_i] * kv + *s);
                                *s = *s * decay[key_i] + kv;
                            }
                        }
                    }
                    (r, k, v, gate) = (rr, key, vv, g);
                }
                // Each head's output: normalized, a bonus for the current
                // token (RWKV-7), gated, projected.
                let (ln_w, ln_b) = (
                    t("time_mix_ln"),
                    tensor(model, &format!("blk.{layer}.time_mix_ln.bias")),
                );
                let r_k = if v7 { t("time_mix_r_k") } else { Vec::new() };
                for head in 0..heads {
                    let span = head * n..(head + 1) * n;
                    let normed = standardize(&y[span.clone()], 64e-5);
                    let bonus: f64 = if v7 {
                        span.clone().map(|j| r[j] * k[j] * r_k[j]).sum()
                    } else {
                        0.0
                    };
                    for (offset, j) in span.enumerate() {
                        y[j] = (normed[offset] * ln_w[j] + ln_b[j] + bonus * v[j]) * gate[j];
                    }
                }
                add(&mut x, &mat("time_mix_output", &y));

                // Channel mixing.
                let xf = layer_norm(model, &x, &format!("blk.{layer}.attn_norm_2"));
                let sx: Vec<f64> = state
                    .channel_shift
                    .iter()
                    .zip(&xf)
                    .map(|(p, c)| p - c)
                    .collect();
                state.channel_shift = xf.clone();
                let moved = |amounts: &[f64]| -> Vec<f64> {
                    xf.iter()
                        .zip(&sx)
                        .zip(amounts)
                        .map(|((x, s), a)| x + s * a)
                        .collect()
                };
                let keys: Vec<f64> = mat("channel_mix_key", &moved(&t("channel_mix_lerp_k")))
                    .into_iter()
                    .map(|k| k.max(0.0).powi(2))
                    .collect();
                let mut out = mat("channel_mix_value", &keys);
                if !v7 {
                    let receptance =
                        mat("channel_mix_receptance", &moved(&t("channel_mix_lerp_r")));
                    for (o, r) in out.iter_mut().zip(receptance) {
                        *o *= sigmoid(r);
                    }
                }
                add(&mut x, &out);
                if rescale > 0 && (layer + 1) % rescale == 0 {
                    x.iter_mut().for_each(|v| *v *= 0.5);
                }
            }
        }
        let last = layer_norm(model, &x, "output_norm");
        by_name(model, "output.weight", &last)
    }

    /// The logits after `tokens`, computed from scratch.
    pub fn logits(model: &Model, tokens: &[u32]) -> Vec<f64> {
        if model.config().rwkv.is_some() {
            return rwkv_logits(model, tokens);
        }
        let (c, w) = (model.config(), model.weights());
        let eps = c.norm_eps;
        let mut x: Vec<Vec<f64>> = tokens
            .iter()
            .map(|&token| {
                row_of(model, &w.token_embd, token as usize)
                    .iter()
                    .map(|v| v * c.embedding_scale as f64)
                    .collect()
            })
            .collect();

        // Gemma 4: each token's input to every layer.
        let per = c.per_layer_embd;
        let per_layer: Vec<Vec<f64>> = match &w.per_layer_inputs {
            None => Vec::new(),
            Some(inputs) => x
                .iter()
                .zip(tokens)
                .map(|(row, &token)| {
                    let projected = apply(model, &inputs.model_proj, row);
                    let own = row_of(model, &inputs.token_embd, token as usize);
                    projected
                        .chunks(per)
                        .zip(own.chunks(per))
                        .flat_map(|(projected, own)| {
                            let scaled: Vec<f64> = projected
                                .iter()
                                .map(|v| v / (c.n_embd as f64).sqrt())
                                .collect();
                            let normed = rms_norm(&scaled, Some(&inputs.proj_norm), eps);
                            normed
                                .into_iter()
                                .zip(own)
                                .map(|(n, e)| (n + e * (per as f64).sqrt()) / 2f64.sqrt())
                                .collect::<Vec<f64>>()
                        })
                        .collect()
                })
                .collect(),
        };

        let mut caches = vec![None; c.n_layer];
        for (index, layer) in w.layers.iter().enumerate() {
            let inputs: Vec<Vec<f64>> = x
                .iter()
                .map(|row| norm(row, &layer.attn_norm, layer.attn_norm_bias.as_deref(), eps))
                .collect();
            let updates = match &layer.mixer {
                MixerWeights::Attention(a) => attention(model, index, a, &inputs, &mut caches),
                MixerWeights::ShortConv(cw) => short_conv(model, index, cw, &inputs),
                MixerWeights::Rwkv(_) => unreachable!("RWKV models have an oracle of their own"),
                MixerWeights::Mamba(mw) => mamba(model, index, mw, &inputs),
                MixerWeights::Mla(_) => latent_attention(model, index, &inputs),
            };
            for (position, mut update) in updates.into_iter().enumerate() {
                if let Some(weight) = &layer.attn_post_norm {
                    update = rms_norm(&update, Some(weight), eps);
                }
                update
                    .iter_mut()
                    .for_each(|v| *v *= c.residual_scale as f64);
                add(&mut x[position], &update);

                // Mamba blocks are their mixer alone.
                if let FeedForward::None = layer.ffn {
                    continue;
                }
                // A parallel block's network reads the block's input.
                let h = if c.parallel {
                    inputs[position].clone()
                } else {
                    rms_norm(&x[position], Some(&layer.ffn_norm), eps)
                };
                let mut update = match &layer.ffn {
                    FeedForward::Dense(dense) => feed_forward(model, dense, &h),
                    FeedForward::Experts(experts) => mixture(model, experts, &h),
                    FeedForward::Plain(plain) => {
                        let up: Vec<f64> = apply(model, &plain.up, &h)
                            .iter()
                            .map(|&u| activate(u, 1.0, c.activation))
                            .collect();
                        apply(model, &plain.down, &up)
                    }
                    FeedForward::None => unreachable!("skipped above"),
                    FeedForward::ChannelMix(_) => unreachable!("handled with RWKV above"),
                };
                if let Some(weight) = &layer.ffn_post_norm {
                    update = rms_norm(&update, Some(weight), eps);
                }
                update
                    .iter_mut()
                    .for_each(|v| *v *= c.residual_scale as f64);
                add(&mut x[position], &update);

                if let Some(plw) = &layer.per_layer {
                    let gate = apply(model, &plw.gate, &x[position]);
                    let input = &per_layer[position][index * per..(index + 1) * per];
                    let mixed: Vec<f64> = gate
                        .iter()
                        .zip(input)
                        .map(|(g, i)| activate(*g, *i, Activation::Gelu))
                        .collect();
                    let update = apply(model, &plw.proj, &mixed);
                    add(
                        &mut x[position],
                        &rms_norm(&update, Some(&plw.post_norm), eps),
                    );
                }
                if let Some(scale) = layer.output_scale {
                    x[position].iter_mut().for_each(|v| *v *= scale as f64);
                }
            }
        }

        let last = norm(
            x.last().expect("at least one token"),
            &w.output_norm,
            w.output_norm_bias.as_deref(),
            eps,
        );
        let mut logits = apply(model, &w.output, &last);
        logits
            .iter_mut()
            .for_each(|logit| *logit /= c.logit_scale as f64);
        if c.logit_softcap > 0.0 {
            let cap = c.logit_softcap as f64;
            logits
                .iter_mut()
                .for_each(|logit| *logit = cap * (*logit / cap).tanh());
        }
        logits
    }
}

/// DeepSeek V2 models beyond the default one, which stores its latent's
/// up-projection as one tensor, as older files do:
///
/// - the up-projection in two tensors, as newer files store it, with values
///   wider than the unrotated part of the keys;
/// - DeepSeek V3: compressed queries, and experts chosen by sigmoid scores
///   with a bias, from the best groups only, with normalized and scaled
///   weights;
/// - no YaRN, with the older up-projection and compressed queries.
fn deepseek2_variants(ty: TensorType) -> Vec<TinyModel> {
    let base = TinyModel::new("deepseek2", ty);
    let mut split = base.clone();
    split.split_kv_b = true;
    split.value_dims = 2 * base.nope_dims;
    let mut v3 = base.clone();
    v3.split_kv_b = true;
    v3.q_rank = base.n_embd;
    v3.gating = 2;
    v3.choice_bias = true;
    v3.expert_groups = 4;
    v3.groups_used = 2;
    v3.expert_norm = Some(true);
    v3.expert_scale = 2.5;
    let mut plain = base.clone();
    plain.yarn = false;
    plain.q_rank = base.n_embd;
    vec![split, v3, plain]
}

#[test]
fn float_models_match_the_oracle() {
    for arch in ARCHS {
        let mut variants = vec![TinyModel::new(arch, TensorType::F32)];
        if matches!(arch, "llama" | "qwen2" | "qwen3" | "gemma3" | "lfm2") {
            let mut fused = TinyModel::new(arch, TensorType::F32);
            fused.fused = true;
            fused.bias = true;
            variants.push(fused);
            let mut with_factors = TinyModel::new(arch, TensorType::F32);
            with_factors.rope_factors = true;
            with_factors.n_head_kv = 4;
            variants.push(with_factors);
        }
        if arch == "llama" || arch == "qwen3" {
            // Only part of each head rotates.
            let mut partial = TinyModel::new(arch, TensorType::F32);
            partial.rope_dims = partial.head_dim() / 2;
            variants.push(partial);
        }
        if arch == "llama" {
            // Mixtral: a Llama with experts, gate and up stored together,
            // weighed by a softmax over the chosen experts only.
            let mut mixtral = TinyModel::new(arch, TensorType::F32);
            mixtral.experts = 4;
            mixtral.experts_used = 2;
            mixtral.fused_experts = true;
            mixtral.gating = 3;
            variants.push(mixtral);
        }
        if arch == "gemma4" {
            // Values of their own, and no layers that share caches.
            let mut plain = TinyModel::new(arch, TensorType::F32);
            plain.values_from_keys = false;
            plain.shared_kv = 0;
            variants.push(plain);
        }
        if arch == "gemma3" {
            let mut windowed = TinyModel::new(arch, TensorType::F32);
            windowed.window = 3;
            windowed.n_layer = 7;
            variants.push(windowed);
        }
        if arch == "deepseek2" {
            variants.extend(deepseek2_variants(TensorType::F32));
        }
        for (index, spec) in variants.iter().enumerate() {
            let model = load(spec);
            let expected = oracle::logits(&model, &PROMPT);
            for (threads, batch) in [(1, 1), (1, 4), (3, 32)] {
                let got = logits(&model, &PROMPT, threads, batch);
                let error = relative_error(&got, &expected);
                assert!(
                    error < 2e-4,
                    "{arch} variant {index} threads {threads} batch {batch}: error {error}"
                );
            }
        }
    }
}

#[test]
fn half_float_models_match_the_oracle() {
    for ty in [TensorType::F16, TensorType::BF16] {
        let model = load(&TinyModel::new("llama", ty));
        let expected = oracle::logits(&model, &PROMPT);
        let error = relative_error(&logits(&model, &PROMPT, 2, 4), &expected);
        assert!(error < 2e-4, "{}: error {error}", ty.name());
    }
}

#[test]
fn quantized_models_stay_close_to_the_oracle() {
    // The engine rounds activations to 8 bits before each multiply, so it
    // cannot match exactly. A wrong bit layout would be off by far more.
    for ty in QUANT_TYPES {
        for arch in ["llama", "gemma3"] {
            let model = load(&TinyModel::new(arch, ty));
            let expected = oracle::logits(&model, &PROMPT);
            let error = relative_error(&logits(&model, &PROMPT, 2, 4), &expected);
            assert!(error < 0.08, "{arch} {}: error {error}", ty.name());
        }
    }
}

#[test]
fn exact_kernels_match_the_oracle_on_quantized_models() {
    // Without activation rounding, only float precision separates the
    // engine from the oracle, whatever the weight format.
    for ty in QUANT_TYPES {
        let model = load(&TinyModel::new("qwen3", ty));
        let expected = oracle::logits(&model, &PROMPT);
        let mut session = session(&model, 2, 4, Some(kernels::exact()));
        assert_eq!(session.kernels(), "exact");
        session.eval(&PROMPT).unwrap();
        let error = relative_error(session.logits(), &expected);
        assert!(error < 2e-4, "{}: error {error}", ty.name());
    }
}

#[test]
fn every_backend_matches_the_scalar_one() {
    let mut types = QUANT_TYPES.to_vec();
    types.extend([TensorType::F32, TensorType::F16, TensorType::BF16]);
    for ty in types {
        let model = load(&TinyModel::new("qwen3", ty));
        let mut reference = session(&model, 1, 4, Some(kernels::scalar()));
        reference.eval(&PROMPT).unwrap();
        let expected: Vec<f64> = reference.logits().iter().map(|v| *v as f64).collect();
        for backend in kernels::available() {
            let mut session = session(&model, 2, 4, Some(backend));
            assert_eq!(session.kernels(), backend.name);
            session.eval(&PROMPT).unwrap();
            let error = relative_error(session.logits(), &expected);
            assert!(
                error < 2e-4,
                "{} {}: error {error}",
                backend.name,
                ty.name()
            );
        }
    }
}

#[test]
fn blocks_longer_than_an_activation_group_take_both_routes() {
    // A block of NVFP4 or Q2_0 spans two groups of quantized activations,
    // and a block of Q1_0 four, each group with a scale of its own. Small
    // batches take the row kernels, and a batch of the long prompt the tile
    // route where the backend has one. Every vector of a batch must find
    // the scales of its own groups. The scales of the wrong groups are off
    // by about half.
    //
    // The routes are not compared with each other. They add in other
    // orders, and in these models a last-bit difference can move an
    // activation across a rounding step of the 8-bit quantization, which
    // then shows in the logits.
    for ty in [TensorType::Nvfp4, TensorType::Q1_0, TensorType::Q2_0] {
        let model = load(&TinyModel::new("llama", ty));
        let expected = oracle::logits(&model, &LONG_PROMPT);
        for batch in [1, 4, 32] {
            let error = relative_error(&logits(&model, &LONG_PROMPT, 2, batch), &expected);
            assert!(error < 0.08, "{} batch {batch}: error {error}", ty.name());
        }
    }
}

#[test]
fn batching_and_threads_do_not_change_the_result() {
    for ty in [TensorType::F32, TensorType::Q8_0, TensorType::Q4K] {
        let model = load(&TinyModel::new("llama", ty));
        let expected = logits(&model, &PROMPT, 1, 1);
        // Batches this small take the same route as single tokens, so the
        // result is the same to the last bit.
        for (threads, batch) in [(1, 2), (1, 7), (4, 1), (4, 3), (8, 7)] {
            assert_eq!(
                logits(&model, &PROMPT, threads, batch),
                expected,
                "threads {threads} batch {batch}"
            );
        }
        // Larger batches decode each weight row ahead of the multiply. That
        // adds the same numbers in another order, so only rounding differs.
        // The thread count still changes nothing.
        let wide = logits(&model, &LONG_PROMPT, 1, 32);
        let narrow: Vec<f64> = logits(&model, &LONG_PROMPT, 1, 1)
            .iter()
            .map(|v| *v as f64)
            .collect();
        assert!(
            relative_error(&wide, &narrow) < 1e-4,
            "{}",
            relative_error(&wide, &narrow)
        );
        for (threads, batch) in [(4, 32), (8, 32), (3, 16)] {
            let again = logits(&model, &LONG_PROMPT, threads, batch);
            if batch == 32 {
                assert_eq!(again, wide, "threads {threads} batch {batch}");
            } else {
                assert!(relative_error(&again, &narrow) < 1e-4);
            }
        }
        // Reading prompts on more threads than generating changes nothing.
        let config = SessionConfig {
            n_ctx: 2048,
            threads: 1,
            batch_threads: 5,
            batch: 32,
            cache: CacheType::F32,
            ..Default::default()
        };
        let mut split = Session::new(Arc::clone(&model), config).unwrap();
        assert_eq!((split.threads(), split.batch_threads()), (1, 5));
        split.eval(&LONG_PROMPT).unwrap();
        assert_eq!(split.logits(), logits(&model, &LONG_PROMPT, 1, 32));
        split.eval(&[3]).unwrap();
        let mut whole = LONG_PROMPT.to_vec();
        whole.push(3);
        assert!(
            relative_error(
                split.logits(),
                &logits(&model, &whole, 1, 1)
                    .iter()
                    .map(|v| *v as f64)
                    .collect::<Vec<_>>()
            ) < 1e-4
        );

        // Feeding the prompt in pieces is the same as feeding it at once.
        let mut session = session(&model, 2, 4, None);
        session.eval(&PROMPT[..4]).unwrap();
        session.eval(&PROMPT[4..5]).unwrap();
        session.eval(&[]).unwrap();
        session.eval(&PROMPT[5..]).unwrap();
        assert_eq!(session.logits(), expected);
        assert_eq!(session.tokens(), PROMPT);
    }
}

#[test]
fn rolling_back_restores_an_earlier_state() {
    let model = load(&TinyModel::new("qwen2", TensorType::Q8_0));
    let mut session = session(&model, 2, 4, None);
    session.eval(&PROMPT).unwrap();
    session.truncate(4);
    assert_eq!(session.tokens(), &PROMPT[..4]);
    session.eval(&[33, 34]).unwrap();

    let mut replay = PROMPT[..4].to_vec();
    replay.extend([33, 34]);
    assert_eq!(session.logits(), logits(&model, &replay, 2, 4));

    session.clear();
    assert!(session.tokens().is_empty());
    assert_eq!(session.cache_bytes(), 0);
    session.eval(&PROMPT).unwrap();
    assert_eq!(session.logits(), logits(&model, &PROMPT, 2, 4));
}

#[test]
fn new_layer_kinds_batch_and_thread_exactly() {
    // Batches below the tile route compute each token the same way, so the
    // convolution state, shared caches and skipped rows must not change a
    // bit of the result.
    for arch in [
        "gemma4",
        "lfm2",
        "lfm3",
        "qwen2moe",
        "lfm2moe",
        "olmoe",
        "granitemoe",
        "deepseek",
        "deepseek2",
        "rwkv6",
        "rwkv7",
    ] {
        for ty in [TensorType::F32, TensorType::Q8_0] {
            let model = load(&TinyModel::new(arch, ty));
            let expected = logits(&model, &PROMPT, 1, 1);
            for (threads, batch) in [(1, 2), (1, 7), (4, 3), (3, 5)] {
                assert_eq!(
                    logits(&model, &PROMPT, threads, batch),
                    expected,
                    "{arch} {} threads {threads} batch {batch}",
                    ty.name()
                );
            }
            // Large batches add in another order, so only rounding differs.
            let wide = logits(&model, &LONG_PROMPT, 3, 32);
            let narrow: Vec<f64> = logits(&model, &LONG_PROMPT, 1, 1)
                .iter()
                .map(|v| *v as f64)
                .collect();
            let error = relative_error(&wide, &narrow);
            assert!(error < 1e-3, "{arch} {}: error {error}", ty.name());
        }
    }
}

#[test]
fn new_layer_kinds_stay_close_to_the_oracle_when_quantized() {
    for arch in ["gemma4", "lfm2", "lfm3", "qwen3moe", "lfm2moe", "deepseek"] {
        for ty in [TensorType::Q8_0, TensorType::Q4_0, TensorType::Q4K] {
            let model = load(&TinyModel::new(arch, ty));
            let expected = oracle::logits(&model, &PROMPT);
            let error = relative_error(&logits(&model, &PROMPT, 2, 4), &expected);
            assert!(error < 0.08, "{arch} {}: error {error}", ty.name());
        }
    }
}

#[test]
fn latent_attention_matches_the_oracle_when_quantized() {
    // Each file layout, in formats with small and large blocks. The large
    // ones make the heads wide enough to spread over threads.
    for ty in [
        TensorType::Q8_0,
        TensorType::Q4_0,
        TensorType::Q4K,
        TensorType::Q6K,
    ] {
        let mut specs = vec![TinyModel::new("deepseek2", ty)];
        specs.extend(deepseek2_variants(ty));
        for (index, spec) in specs.iter_mut().enumerate() {
            // Without rounding activations, only float precision separates
            // the engine from the oracle, whatever the weight format.
            let model = load(spec);
            let expected = oracle::logits(&model, &PROMPT);
            let mut session = session(&model, 3, 4, Some(kernels::exact()));
            session.eval(&PROMPT).unwrap();
            let error = relative_error(session.logits(), &expected);
            assert!(
                error < 2e-4,
                "{} variant {index} exact: error {error}",
                ty.name()
            );

            // Rounded activations add a percent or so, and more in these
            // models, whose random weights make them sensitive: rounding one
            // activation the other way can move the logits by a few percent,
            // or change a choice of experts. So this part runs without
            // experts, on the formats whose random weights stay small, with
            // a margin. A misread layout or a misplaced head is off by far
            // more.
            if ty == TensorType::Q6K {
                continue;
            }
            spec.experts = 0;
            let model = load(spec);
            let expected = oracle::logits(&model, &PROMPT);
            for (threads, batch) in [(1, 1), (4, 4), (3, 32)] {
                let error = relative_error(&logits(&model, &PROMPT, threads, batch), &expected);
                assert!(
                    error < 0.12,
                    "{} variant {index} threads {threads} batch {batch}: error {error}",
                    ty.name()
                );
            }
        }
    }
}

#[test]
fn latent_attention_batches_and_threads_exactly() {
    // Below the tile route every token is computed the same way, so how
    // the prompt is split and spread must not change a bit.
    for ty in [TensorType::F32, TensorType::Q8_0, TensorType::Q4K] {
        for spec in deepseek2_variants(ty) {
            let model = load(&spec);
            let expected = logits(&model, &PROMPT, 1, 1);
            for (threads, batch) in [(1, 2), (1, 7), (4, 3), (3, 5)] {
                assert_eq!(
                    logits(&model, &PROMPT, threads, batch),
                    expected,
                    "{} threads {threads} batch {batch}",
                    ty.name()
                );
            }
            // Large batches add in another order. With rounded activations
            // that can round one of them the other way, which in a model this
            // small moves the logits by a percent or two. The exact kernels
            // show what is left of the reordering.
            let exact = |threads, batch| {
                let mut session = session(&model, threads, batch, Some(kernels::exact()));
                session.eval(&LONG_PROMPT).unwrap();
                session.logits().to_vec()
            };
            let narrow: Vec<f64> = exact(1, 1).iter().map(|v| *v as f64).collect();
            let error = relative_error(&exact(3, 32), &narrow);
            assert!(error < 1e-5, "{}: error {error}", ty.name());
        }
    }
}

#[test]
fn group_limited_routing_changes_the_choice() {
    // The V3 variant keeps two of four groups. Letting every group in must
    // change the result, or the test above would not cover the groups.
    let v3 = deepseek2_variants(TensorType::F32).remove(1);
    assert_eq!(v3.expert_groups, 4);
    let mut open = v3.clone();
    open.expert_groups = 0;
    let (grouped, ungrouped) = (load(&v3), load(&open));
    let routing = grouped.config().routing.unwrap();
    assert_eq!((routing.groups, routing.groups_used), (4, 2));
    let expected = oracle::logits(&grouped, &PROMPT);
    let unexpected = oracle::logits(&ungrouped, &PROMPT);
    assert!(relative_error(&logits(&grouped, &PROMPT, 2, 4), &expected) < 2e-4);
    assert!(relative_error(&logits(&ungrouped, &PROMPT, 2, 4), &unexpected) < 2e-4);
    assert!(relative_error(&logits(&grouped, &PROMPT, 2, 4), &unexpected) > 1e-3);
}

#[test]
fn latent_attention_caches_one_row_per_position() {
    // A layer's cache holds the rotary key and the latent, and no values.
    let spec = TinyModel::new("deepseek2", TensorType::F32);
    let model = load(&spec);
    let layer = &model.config().layers[0];
    assert_eq!(
        (layer.n_head_kv, layer.head_dim),
        (1, spec.rope_dims + spec.kv_rank)
    );
    let mut session = session(&model, 1, 4, None);
    session.eval(&PROMPT).unwrap();
    let row = (spec.rope_dims + spec.kv_rank) * std::mem::size_of::<f32>();
    assert_eq!(session.cache_bytes(), spec.n_layer * CHUNK_POSITIONS * row);
}

#[test]
fn convolution_state_survives_rolling_back() {
    // Long enough to pass two checkpoints of the convolution state.
    let tokens: Vec<u32> = (0..600u32).map(|i| 3 + (i * 7) % 37).collect();
    let next = [30, 31, 32];
    for arch in ["lfm2", "lfm3"] {
        let model = load(&TinyModel::new(arch, TensorType::F32));
        let mut session = session(&model, 2, 5, None);
        session.eval(&tokens).unwrap();
        let full_size = session.cache_bytes();
        // Short cuts stay within the recent inputs. Long ones restore a
        // checkpoint and read the tokens after it again.
        for len in [599, 596, 520, 300, 256, 255, 3, 0] {
            session.truncate(len);
            assert_eq!(session.tokens(), &tokens[..len]);
            session.eval(&next).unwrap();
            let mut expected = tokens[..len].to_vec();
            expected.extend(next);
            assert_eq!(
                session.logits(),
                logits(&model, &expected, 2, 5),
                "{arch} cut to {len}"
            );
            // Back to the whole context for the next cut.
            session.truncate(len);
            session.eval(&tokens[len..]).unwrap();
            assert_eq!(session.logits(), logits(&model, &tokens, 2, 5));
        }
        assert_eq!(session.cache_bytes(), full_size);
        session.clear();
        assert!(session.cache_bytes() < full_size / 4);
    }
}

/// The models with Mamba blocks, in weights of type `ty`: Mamba, Falcon
/// Mamba, Mamba 2 and Granite 4.0 H.
fn mamba_models(ty: TensorType) -> Vec<TinyModel> {
    let mut falcon = TinyModel::new("mamba", ty);
    falcon.ssm_norms = true;
    vec![
        TinyModel::new("mamba", ty),
        falcon,
        TinyModel::new("mamba2", ty),
        TinyModel::new("granitehybrid", ty),
    ]
}

#[test]
fn mamba_models_match_the_oracle() {
    for spec in mamba_models(TensorType::F32) {
        let model = load(&spec);
        // The long prompt fills batches that take the tile route.
        for tokens in [&PROMPT[..], &LONG_PROMPT[..]] {
            let expected = oracle::logits(&model, tokens);
            for (threads, batch) in [(1, 1), (1, 4), (3, 32)] {
                let got = logits(&model, tokens, threads, batch);
                let error = relative_error(&got, &expected);
                assert!(
                    error < 2e-4,
                    "{} norms {} {} tokens, threads {threads} batch {batch}: error {error}",
                    spec.arch,
                    spec.ssm_norms,
                    tokens.len()
                );
            }
        }
    }
}

#[test]
fn quantized_mamba_models_stay_close_to_the_oracle() {
    for ty in [TensorType::Q8_0, TensorType::Q4_0, TensorType::Q4K] {
        for spec in mamba_models(ty) {
            let model = load(&spec);
            let expected = oracle::logits(&model, &PROMPT);
            let error = relative_error(&logits(&model, &PROMPT, 2, 4), &expected);
            assert!(error < 0.08, "{} {}: error {error}", spec.arch, ty.name());
        }
    }
}

#[test]
fn mamba_models_batch_and_thread_exactly() {
    // The scan runs every token in order whatever the batch, and threads
    // split it by heads, so neither changes a bit of the result below the
    // batches that take the tile route.
    for ty in [TensorType::F32, TensorType::Q8_0] {
        for spec in mamba_models(ty) {
            let model = load(&spec);
            let expected = logits(&model, &PROMPT, 1, 1);
            for (threads, batch) in [(1, 2), (1, 7), (4, 3), (3, 5), (8, 1)] {
                assert_eq!(
                    logits(&model, &PROMPT, threads, batch),
                    expected,
                    "{} {} threads {threads} batch {batch}",
                    spec.arch,
                    ty.name()
                );
            }
            let wide = logits(&model, &LONG_PROMPT, 3, 32);
            let narrow: Vec<f64> = logits(&model, &LONG_PROMPT, 1, 1)
                .iter()
                .map(|v| *v as f64)
                .collect();
            let error = relative_error(&wide, &narrow);
            assert!(error < 1e-3, "{} {}: error {error}", spec.arch, ty.name());
        }
    }
}

#[test]
fn mamba_state_survives_rolling_back() {
    // A chat: a prompt, a reply generated one token at a time, then cuts
    // back into the reply, to the end of the prompt, and before it. The
    // session restores the state saved at the end of a prompt, or starts
    // over, and reads the tokens after it again.
    let prompt: Vec<u32> = (0..50u32).map(|i| 3 + (i * 7) % 37).collect();
    let reply = [30, 31, 32, 33];
    let next = [34, 35];
    for spec in mamba_models(TensorType::F32) {
        let model = load(&spec);
        let mut session = session(&model, 2, 5, None);
        session.eval(&prompt).unwrap();
        for token in reply {
            session.eval(&[token]).unwrap();
        }
        let mut context = prompt.clone();
        context.extend(reply);
        for len in [53, 50, 49, 20, 0] {
            session.truncate(len);
            assert_eq!(session.tokens(), &context[..len]);
            session.eval(&next).unwrap();
            let mut expected = context[..len].to_vec();
            expected.extend(next);
            assert_eq!(
                session.logits(),
                logits(&model, &expected, 2, 5),
                "{} cut to {len}",
                spec.arch
            );
            // Back to the whole conversation for the next cut.
            session.truncate(len);
            session.eval(&context[len..]).unwrap();
            assert_eq!(session.logits(), logits(&model, &context, 2, 5));
        }
        // The state and the copies saved at prompt ends, which are a few.
        let state = session.cache_bytes();
        session.clear();
        assert!(session.cache_bytes() < state);
    }
}

#[test]
fn rwkv_models_stay_close_to_the_oracle_when_quantized() {
    for arch in ["rwkv6", "rwkv7"] {
        for ty in QUANT_TYPES {
            let model = load(&TinyModel::new(arch, ty));
            let expected = oracle::logits(&model, &PROMPT);
            // Without rounding activations, only float precision separates
            // the engine from the oracle, whatever the weight format.
            let mut exact = session(&model, 2, 4, Some(kernels::exact()));
            exact.eval(&PROMPT).unwrap();
            let error = relative_error(exact.logits(), &expected);
            assert!(error < 2e-4, "{arch} {} exact: error {error}", ty.name());
            // Rounding activations to 8 bits costs more here than in other
            // blocks. Each head's output is normalized, which magnifies the
            // small errors of a quiet head. A wrong bit layout would still
            // be off by far more.
            let error = relative_error(&logits(&model, &PROMPT, 2, 4), &expected);
            assert!(error < 0.3, "{arch} {}: error {error}", ty.name());
        }
    }
}

#[test]
fn rwkv_state_survives_rolling_back() {
    // Prompts of several lengths, read one after another as a chat would,
    // leave saved states behind. Each cut restores one, or starts over,
    // and reads the tokens after it again.
    let tokens: Vec<u32> = (0..90u32).map(|i| 1 + (i * 7) % 39).collect();
    let next = [30, 31, 32];
    for arch in ["rwkv6", "rwkv7"] {
        let model = load(&TinyModel::new(arch, TensorType::F32));
        let mut session = session(&model, 2, 5, None);
        for end in [20, 50, 51, 90] {
            let start = session.tokens().len();
            session.eval(&tokens[start..end]).unwrap();
        }
        let full_size = session.cache_bytes();
        for len in [89, 60, 50, 49, 20, 7, 0] {
            session.truncate(len);
            assert_eq!(session.tokens(), &tokens[..len]);
            session.eval(&next).unwrap();
            let mut expected = tokens[..len].to_vec();
            expected.extend(next);
            assert_eq!(
                session.logits(),
                logits(&model, &expected, 2, 5),
                "{arch} cut to {len}"
            );
            session.truncate(len);
            session.eval(&tokens[len..]).unwrap();
            assert_eq!(session.logits(), logits(&model, &tokens, 2, 5));
        }
        assert!(session.cache_bytes() <= full_size + 3 * model_state_bytes(&model));
        session.clear();
        assert_eq!(session.cache_bytes(), model_state_bytes(&model));
    }
}

/// Bytes of one copy of every block's recurrent state.
fn model_state_bytes(model: &Model) -> usize {
    let c = model.config();
    let per_block = 2 * c.n_embd + c.n_embd * c.rwkv.map_or(0, |rwkv| rwkv.head_size);
    c.n_layer * per_block * 4
}

#[test]
fn shifting_the_context_matches_a_fresh_start_in_one_layer() {
    // With one layer, every key and value depends only on its own token and
    // position, so after a shift the session holds exactly what a fresh one
    // reading the kept tokens would. Rotating keys back must get there.
    let tokens: Vec<u32> = (0..300u32).map(|i| 3 + (i * 7) % 37).collect();
    for arch in ["llama", "qwen3", "gemma3", "lfm2", "smollm3", "deepseek2"] {
        let mut spec = TinyModel::new(arch, TensorType::F32);
        spec.n_layer = 1;
        spec.rope_factors = arch == "llama";
        let model = load(&spec);
        let mut session = session(&model, 2, 8, None);
        session.eval(&tokens).unwrap();
        session.shift(3, 260);
        let mut kept = tokens[..3].to_vec();
        kept.extend(&tokens[263..]);
        assert_eq!(session.tokens(), kept);
        session.eval(&[30, 31]).unwrap();
        kept.extend([30, 31]);
        let fresh: Vec<f64> = logits(&model, &kept, 2, 8)
            .iter()
            .map(|v| *v as f64)
            .collect();
        let error = relative_error(session.logits(), &fresh);
        assert!(error < 1e-4, "{arch}: error {error}");
    }
}

#[test]
fn a_long_context_crosses_cache_chunks() {
    for arch in ["llama", "deepseek2"] {
        let mut spec = TinyModel::new(arch, TensorType::F32);
        spec.n_layer = if arch == "llama" { 1 } else { 2 };
        let model = load(&spec);
        let tokens: Vec<u32> = (0..600u32).map(|i| 3 + (i * 7) % 37).collect();
        let expected = oracle::logits(&model, &tokens);
        let got = logits(&model, &tokens, 4, 32);
        let error = relative_error(&got, &expected);
        assert!(error < 5e-4, "{arch}: error {error}");
        assert_eq!(logits(&model, &tokens, 2, 7), got, "{arch}");
    }
}

#[test]
fn a_sliding_window_hides_older_positions() {
    let mut unlimited = TinyModel::new("gemma3", TensorType::F32);
    unlimited.n_layer = 6;
    let mut wide = unlimited.clone();
    wide.window = 64;
    let mut narrow = unlimited.clone();
    narrow.window = 2;

    // A window wider than the context changes only the rotary base of the
    // windowed layers, which the synthetic model sets equal to the global.
    let reference = logits(&load(&unlimited), &PROMPT, 1, 4);
    assert_eq!(logits(&load(&wide), &PROMPT, 1, 4), reference);
    assert_ne!(logits(&load(&narrow), &PROMPT, 1, 4), reference);

    let model = load(&narrow);
    let windows: Vec<usize> = model
        .config()
        .layers
        .iter()
        .map(|layer| layer.window)
        .collect();
    assert_eq!(windows, [2, 2, 2, 2, 2, 0]);
}

#[test]
fn limits_are_enforced() {
    let model = load(&TinyModel::new("llama", TensorType::F32));
    let config = SessionConfig {
        n_ctx: 4,
        threads: 1,
        ..Default::default()
    };
    let mut session = Session::new(Arc::clone(&model), config).unwrap();
    assert_eq!(session.n_ctx(), 4);
    session.eval(&[1, 2, 3]).unwrap();
    assert!(matches!(
        session.eval(&[4, 5]),
        Err(Error::ContextFull {
            n_ctx: 4,
            needed: 5
        })
    ));
    // A rejected call leaves the context as it was.
    assert_eq!(session.tokens(), [1, 2, 3]);
    session.eval(&[4]).unwrap();
    assert!(matches!(
        session.eval(&[40]),
        Err(Error::ContextFull { .. })
    ));

    let mut session = Session::new(Arc::clone(&model), SessionConfig::default()).unwrap();
    assert_eq!(session.n_ctx(), 512);
    assert!(matches!(session.eval(&[40]), Err(Error::Invalid(_))));
    assert!(session.tokens().is_empty());
}

#[test]
fn model_facts_are_reported() {
    let spec = TinyModel::new("qwen3", TensorType::Q4K);
    let model = load(&spec);
    let c = model.config();
    assert_eq!(
        (c.arch.as_str(), c.name.as_str()),
        ("qwen3", "tiny test model")
    );
    assert_eq!(
        (c.n_vocab, c.n_embd, c.layers[0].n_ff, c.layers[0].head_dim),
        (40, 256, 256, 64)
    );
    assert_eq!(model.tokenizer().n_tokens(), 40);
    assert!(model.n_params() > 256 * 256 * 10);
    assert_eq!(model.file_size(), spec.build().len());
    assert!(model.chat_template().is_none());
}

#[test]
fn broken_models_fail_to_load() {
    // A matrix of the wrong shape.
    let mut spec = TinyModel::new("llama", TensorType::F32);
    spec.n_ff = 64;
    let mut bytes = spec.build();
    assert!(Model::from_bytes(bytes.clone()).is_ok());
    // Claim a different embedding width than the tensors have.
    let key = b"llama.embedding_length";
    let at = bytes
        .windows(key.len())
        .position(|window| window == key)
        .unwrap()
        + key.len()
        + 4;
    bytes[at..at + 4].copy_from_slice(&64u32.to_le_bytes());
    let error = Model::from_bytes(bytes)
        .err()
        .expect("shape mismatch")
        .to_string();
    assert!(error.contains("token_embd.weight"), "{error}");

    // An architecture this engine does not run.
    let mut spec = TinyModel::new("llama", TensorType::F32);
    spec.arch = "bloom";
    assert!(matches!(
        Model::from_bytes(spec.build()),
        Err(Error::Unsupported(_))
    ));
}

/// A chat goes back into the last reply, and a regenerated answer goes
/// back to just before the last prompt token. Both find the state saved
/// there and read nothing before it again.
#[test]
fn recurrent_states_are_saved_where_chats_go_back_to() {
    let model =
        Arc::new(Model::from_bytes(TinyModel::new("mamba2", TensorType::F32).build()).unwrap());
    let config = SessionConfig {
        threads: 1,
        ..Default::default()
    };
    let mut session = Session::new(model, config).unwrap();
    let prompt: Vec<u32> = (3..13).collect();
    session.eval(&prompt).unwrap();
    assert_eq!(session.saved_positions(), [9]);
    for token in [20, 21, 22] {
        session.eval(&[token]).unwrap();
    }
    assert_eq!(session.saved_positions(), [9]);
    // Regenerating: the prompt again, its last token read afresh. Had the
    // session read the prompt again from the start, it would have saved a
    // new state at 8.
    session.truncate(9);
    assert_eq!(session.tokens(), &prompt[..9]);
    assert_eq!(session.saved_positions(), [9]);
    session.eval(&prompt[9..]).unwrap();
    // The next turn: back into the reply, which is read again from the
    // saved state, then a new prompt saves a new one.
    session.eval(&[20, 21]).unwrap();
    let next: Vec<u32> = [&prompt[..], &[20, 21, 30, 31, 32]].concat();
    session.truncate(11);
    session.eval(&next[11..]).unwrap();
    assert_eq!(session.tokens(), &next[..]);
    assert_eq!(session.saved_positions(), [next.len() - 1]);
}

/// The default cache rounds keys and values to 16-bit floats. That moves
/// the logits by about a thousandth, as in llama.cpp, and no more.
#[test]
fn half_caches_stay_close_to_the_oracle() {
    for arch in [
        "llama",
        "qwen3",
        "gemma3",
        "gpt-oss",
        "deepseek2",
        "granitehybrid",
    ] {
        let model = load(&TinyModel::new(arch, TensorType::F32));
        let expected = oracle::logits(&model, &LONG_PROMPT);
        let config = SessionConfig {
            threads: 2,
            ..Default::default()
        };
        assert_eq!(config.cache, CacheType::F16);
        let mut session = Session::new(Arc::clone(&model), config).unwrap();
        session.eval(&LONG_PROMPT).unwrap();
        let error = relative_error(session.logits(), &expected);
        assert!(error < 1e-2, "{arch}: error {error}");
        // Half the memory of the exact cache. Granite 4.0 H also keeps a
        // recurrent state, which stays in 32-bit floats.
        let mut exact = self::session(&model, 2, 32, None);
        exact.eval(&LONG_PROMPT).unwrap();
        if arch == "granitehybrid" {
            assert!(session.cache_bytes() < exact.cache_bytes(), "{arch}");
        } else {
            assert_eq!(session.cache_bytes() * 2, exact.cache_bytes(), "{arch}");
        }
    }
}

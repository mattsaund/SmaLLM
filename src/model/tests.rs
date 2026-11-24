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

const ARCHS: [&str; 18] = [
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
    "deepseek",
    "deepseek2",
];

const QUANT_TYPES: [TensorType; 11] = [
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

    fn activate(x: f64, activation: Activation) -> f64 {
        match activation {
            Activation::Silu => x / (1.0 + (-x).exp()),
            Activation::Gelu => {
                let z = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + z.tanh())
            }
        }
    }

    /// A gated feed-forward network.
    fn feed_forward(model: &Model, w: &Dense, h: &[f64]) -> Vec<f64> {
        let activation = model.config().activation;
        let gate = apply(model, &w.gate, h);
        let up = apply(model, &w.up, h);
        let mixed: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| activate(*g, activation) * u)
            .collect();
        apply(model, &w.down, &mixed)
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
            let result = feed_forward(model, &dense, h);
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
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
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

    /// The logits after `tokens`, computed from scratch.
    pub fn logits(model: &Model, tokens: &[u32]) -> Vec<f64> {
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
                .map(|row| rms_norm(row, Some(&layer.attn_norm), eps))
                .collect();
            let updates = match &layer.mixer {
                MixerWeights::Attention(a) => attention(model, index, a, &inputs, &mut caches),
                MixerWeights::ShortConv(cw) => short_conv(model, index, cw, &inputs),
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

                let h = rms_norm(&x[position], Some(&layer.ffn_norm), eps);
                let mut update = match &layer.ffn {
                    FeedForward::Dense(dense) => feed_forward(model, dense, &h),
                    FeedForward::Experts(experts) => mixture(model, experts, &h),
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
                        .map(|(g, i)| activate(*g, Activation::Gelu) * i)
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

        let last = rms_norm(
            x.last().expect("at least one token"),
            Some(&w.output_norm),
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
    spec.arch = "rwkv7";
    assert!(matches!(
        Model::from_bytes(spec.build()),
        Err(Error::Unsupported(_))
    ));
}

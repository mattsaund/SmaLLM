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

const ARCHS: [&str; 16] = [
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
        let mut order: Vec<usize> = (0..routing.count).collect();
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
                MixerWeights::Mamba(mw) => mamba(model, index, mw, &inputs),
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
                let h = rms_norm(&x[position], Some(&layer.ffn_norm), eps);
                let mut update = match &layer.ffn {
                    FeedForward::Dense(dense) => feed_forward(model, dense, &h),
                    FeedForward::Experts(experts) => mixture(model, experts, &h),
                    FeedForward::None => unreachable!("skipped above"),
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
    for arch in ["gemma4", "lfm2", "lfm3", "qwen3moe", "lfm2moe"] {
        for ty in [TensorType::Q8_0, TensorType::Q4_0, TensorType::Q4K] {
            let model = load(&TinyModel::new(arch, ty));
            let expected = oracle::logits(&model, &PROMPT);
            let error = relative_error(&logits(&model, &PROMPT, 2, 4), &expected);
            assert!(error < 0.08, "{arch} {}: error {error}", ty.name());
        }
    }
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
fn recurrent_state_survives_rolling_back() {
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
fn shifting_the_context_matches_a_fresh_start_in_one_layer() {
    // With one layer, every key and value depends only on its own token and
    // position, so after a shift the session holds exactly what a fresh one
    // reading the kept tokens would. Rotating keys back must get there.
    let tokens: Vec<u32> = (0..300u32).map(|i| 3 + (i * 7) % 37).collect();
    for arch in ["llama", "qwen3", "gemma3", "lfm2", "smollm3"] {
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
    let mut spec = TinyModel::new("llama", TensorType::F32);
    spec.n_layer = 1;
    let model = load(&spec);
    let tokens: Vec<u32> = (0..600u32).map(|i| 3 + (i * 7) % 37).collect();
    let expected = oracle::logits(&model, &tokens);
    let got = logits(&model, &tokens, 4, 32);
    let error = relative_error(&got, &expected);
    assert!(error < 5e-4, "error {error}");
    assert_eq!(logits(&model, &tokens, 2, 7), got);
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

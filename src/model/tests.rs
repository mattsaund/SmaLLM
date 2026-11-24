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

const ARCHS: [&str; 4] = ["llama", "qwen2", "qwen3", "gemma3"];

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

    fn rms_norm(x: &[f64], weight: &[f32], eps: f32) -> Vec<f64> {
        let mean: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let scale = 1.0 / (mean + eps as f64).sqrt();
        x.iter()
            .zip(weight)
            .map(|(v, w)| v * scale * *w as f64)
            .collect()
    }

    fn rotate(
        head: &mut [f64],
        position: usize,
        base: f32,
        config: &Config,
        factors: Option<&[f32]>,
    ) {
        let n = head.len();
        for i in 0..n / 2 {
            let frequency = (base as f64).powf(-2.0 * i as f64 / n as f64);
            let divisor = factors.map_or(1.0, |f| f[i] as f64);
            let angle = position as f64 * frequency * config.rope_scale as f64 / divisor;
            let (a, b) = match config.rope {
                Rope::Pairs => (2 * i, 2 * i + 1),
                Rope::Halves => (i, i + n / 2),
            };
            let (x0, x1) = (head[a], head[b]);
            head[a] = x0 * angle.cos() - x1 * angle.sin();
            head[b] = x0 * angle.sin() + x1 * angle.cos();
        }
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

    /// The logits after `tokens`, computed from scratch.
    pub fn logits(model: &Model, tokens: &[u32]) -> Vec<f64> {
        let (c, w) = (model.config(), model.weights());
        let data = model.gguf().mapping().as_slice();
        let (hd, kv_dim) = (c.head_dim, c.kv_dim());
        let factors = w.rope_factors.as_deref();

        let mut x: Vec<Vec<f64>> = tokens
            .iter()
            .map(|&token| {
                let mut row = vec![0.0f32; c.n_embd];
                let start = w.token_embd.offset + token as usize * w.token_embd.row_bytes;
                dequantize_row(
                    w.token_embd.ty,
                    &data[start..start + w.token_embd.row_bytes],
                    &mut row,
                );
                row.iter()
                    .map(|v| *v as f64 * c.embedding_scale as f64)
                    .collect()
            })
            .collect();

        for (index, layer) in w.layers.iter().enumerate() {
            let local = c.local_layers[index];
            let base = if local {
                c.rope_base_local
            } else {
                c.rope_base
            };
            let mut queries = Vec::new();
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (position, row) in x.iter().enumerate() {
                let h = rms_norm(row, &layer.attn_norm, c.norm_eps);
                let mut q = apply(model, &layer.q, &h);
                let mut k = apply(model, &layer.k, &h);
                for head in q.chunks_exact_mut(hd) {
                    if let Some(weight) = &layer.q_norm {
                        head.copy_from_slice(&rms_norm(head, weight, c.norm_eps));
                    }
                    rotate(head, position, base, c, factors);
                }
                for head in k.chunks_exact_mut(hd) {
                    if let Some(weight) = &layer.k_norm {
                        head.copy_from_slice(&rms_norm(head, weight, c.norm_eps));
                    }
                    rotate(head, position, base, c, factors);
                }
                queries.push(q);
                keys.push(k);
                values.push(apply(model, &layer.v, &h));
            }

            for position in 0..x.len() {
                let first = if local && c.sliding_window > 0 {
                    (position + 1).saturating_sub(c.sliding_window)
                } else {
                    0
                };
                let mut attended = vec![0.0f64; c.q_dim()];
                for head in 0..c.n_head {
                    let kv_head = head / (c.n_head / c.n_head_kv) * hd;
                    debug_assert!(kv_head + hd <= kv_dim);
                    let q = &queries[position][head * hd..(head + 1) * hd];
                    let scores: Vec<f64> = (first..=position)
                        .map(|j| {
                            let k = &keys[j][kv_head..kv_head + hd];
                            q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>()
                                * c.attention_scale as f64
                        })
                        .collect();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                    for (offset, score) in scores.iter().enumerate() {
                        let weight = (score - max).exp() / total;
                        let v = &values[first + offset][kv_head..kv_head + hd];
                        for (out, value) in attended[head * hd..(head + 1) * hd].iter_mut().zip(v) {
                            *out += weight * value;
                        }
                    }
                }
                let mut update = apply(model, &layer.attn_out, &attended);
                if let Some(weight) = &layer.attn_post_norm {
                    update = rms_norm(&update, weight, c.norm_eps);
                }
                x[position]
                    .iter_mut()
                    .zip(&update)
                    .for_each(|(value, delta)| *value += delta);

                let h = rms_norm(&x[position], &layer.ffn_norm, c.norm_eps);
                let gate = apply(model, &layer.ffn_gate, &h);
                let up = apply(model, &layer.ffn_up, &h);
                let mixed: Vec<f64> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| activate(*g, c.activation) * u)
                    .collect();
                let mut update = apply(model, &layer.ffn_down, &mixed);
                if let Some(weight) = &layer.ffn_post_norm {
                    update = rms_norm(&update, weight, c.norm_eps);
                }
                x[position]
                    .iter_mut()
                    .zip(&update)
                    .for_each(|(value, delta)| *value += delta);
            }
        }

        let last = rms_norm(
            x.last().expect("at least one token"),
            &w.output_norm,
            c.norm_eps,
        );
        let mut logits = apply(model, &w.output, &last);
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
        let mut fused = TinyModel::new(arch, TensorType::F32);
        fused.fused = true;
        fused.bias = true;
        variants.push(fused);
        let mut with_factors = TinyModel::new(arch, TensorType::F32);
        with_factors.rope_factors = true;
        with_factors.n_head_kv = 4;
        variants.push(with_factors);
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
    assert_eq!(model.config().sliding_window, 2);
    assert_eq!(
        model.config().local_layers,
        [true, true, true, true, true, false]
    );
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
        (c.n_vocab, c.n_embd, c.n_ff, c.head_dim),
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

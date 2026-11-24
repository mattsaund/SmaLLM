//! Shared helpers for unit tests: random data and small synthetic models.

use crate::gguf::testing::Builder;
use crate::tensor::TensorType;
use crate::tensor::f16::f32_to_f16;

/// A small deterministic generator, so a failure reproduces exactly.
pub(crate) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A float in `[low, high)`.
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        low + unit * (high - low)
    }

    pub fn floats(&mut self, n: usize, low: f32, high: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(low, high)).collect()
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 32) as u8).collect()
    }
}

/// Byte offsets of the half-float fields inside one block of `ty`.
fn half_offsets(ty: TensorType) -> &'static [usize] {
    match ty {
        TensorType::Q4_0 | TensorType::Q5_0 | TensorType::Q8_0 | TensorType::Iq4Nl => &[0],
        TensorType::Q4_1 | TensorType::Q5_1 | TensorType::Q4K | TensorType::Q5K => &[0, 2],
        TensorType::Q2K => &[80, 82],
        TensorType::Q3K => &[108],
        TensorType::Q6K => &[208],
        _ => &[],
    }
}

/// Overwrites the scale fields of every block with finite values of at most
/// `magnitude`. Random bytes are valid weights in every format, but not
/// valid scales.
pub(crate) fn set_scales(rng: &mut Rng, ty: TensorType, row: &mut [u8], magnitude: f32) {
    for block in row.chunks_exact_mut(ty.block_bytes()) {
        for &at in half_offsets(ty) {
            let scale = rng.range(-magnitude, magnitude);
            block[at..at + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        }
    }
}

/// The bytes of `n` random weights of type `ty`.
pub(crate) fn random_weights(rng: &mut Rng, ty: TensorType, n: usize, magnitude: f32) -> Vec<u8> {
    match ty {
        TensorType::F32 => rng
            .floats(n, -magnitude, magnitude)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorType::F16 => rng
            .floats(n, -magnitude, magnitude)
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect(),
        TensorType::BF16 => rng
            .floats(n, -magnitude, magnitude)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => {
            let mut bytes = rng.bytes(ty.row_bytes(n).expect("whole blocks"));
            // Quantized values reach about 127, so the scale carries the
            // magnitude.
            set_scales(rng, ty, &mut bytes, magnitude / 64.0);
            bytes
        }
    }
}

/// The GGUF type id of a tensor type.
pub(crate) fn type_id(ty: TensorType) -> u32 {
    (0..64)
        .find(|&id| TensorType::from_id(id) == Some(ty))
        .expect("every type has an id")
}

/// The shape and options of a synthetic model.
#[derive(Clone)]
pub(crate) struct TinyModel {
    pub arch: &'static str,
    /// Element type of every weight matrix.
    pub ty: TensorType,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    /// Adds per-head query and key norms.
    pub qk_norm: bool,
    /// Adds norms after attention and after the feed-forward.
    pub post_norm: bool,
    /// Adds biases to the query, key and value projections.
    pub bias: bool,
    /// Stores query, key and value as one tensor, and the gate with `up`.
    pub fused: bool,
    /// Sliding window for architectures that support one. Zero for none.
    pub window: usize,
    /// Stores per-frequency rotary divisors.
    pub rope_factors: bool,
    pub seed: u64,
}

impl TinyModel {
    /// A two-layer model small enough to evaluate in microseconds.
    pub fn new(arch: &'static str, ty: TensorType) -> TinyModel {
        let width = ty.block_len().max(32);
        TinyModel {
            arch,
            ty,
            n_embd: width,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            n_ff: width,
            n_vocab: 40,
            qk_norm: arch == "qwen3" || arch == "gemma3",
            post_norm: arch == "gemma3",
            bias: arch == "qwen2",
            fused: false,
            window: 0,
            rope_factors: false,
            seed: 1,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Serializes the model as the bytes of a GGUF file.
    pub fn build(&self) -> Vec<u8> {
        let mut rng = Rng(self.seed);
        let arch = self.arch;
        let (d, n_ff) = (self.n_embd, self.n_ff);
        let (q_dim, kv_dim) = (d, self.n_head_kv * self.head_dim());
        let mut builder = Builder::new();
        builder
            .string("general.architecture", arch)
            .string("general.name", "tiny test model")
            .u32(&format!("{arch}.context_length"), 512)
            .u32(&format!("{arch}.embedding_length"), d as u32)
            .u32(&format!("{arch}.block_count"), self.n_layer as u32)
            .u32(&format!("{arch}.feed_forward_length"), n_ff as u32)
            .u32(&format!("{arch}.attention.head_count"), self.n_head as u32)
            .u32(
                &format!("{arch}.attention.head_count_kv"),
                self.n_head_kv as u32,
            )
            .f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5)
            .f32(&format!("{arch}.rope.freq_base"), 10000.0);
        if self.window > 0 {
            builder.u32(
                &format!("{arch}.attention.sliding_window"),
                self.window as u32,
            );
        }

        // A minimal vocabulary: three special tokens, then one per id.
        let texts: Vec<String> = (0..self.n_vocab)
            .map(|id| match id {
                0 => "<unk>".to_string(),
                1 => "<s>".to_string(),
                2 => "</s>".to_string(),
                _ => format!("w{id}"),
            })
            .collect();
        let refs: Vec<&[u8]> = texts.iter().map(|text| text.as_bytes()).collect();
        let types: Vec<i32> = (0..self.n_vocab)
            .map(|id| if id < 3 { 3 } else { 1 })
            .collect();
        builder
            .string("tokenizer.ggml.model", "llama")
            .strings("tokenizer.ggml.tokens", &refs)
            .f32s("tokenizer.ggml.scores", &vec![0.0; self.n_vocab])
            .i32s("tokenizer.ggml.token_type", &types);

        let id = type_id(self.ty);
        let mut matrix =
            |builder: &mut Builder, name: &str, rows: usize, cols: usize, magnitude: f32| {
                let data = random_weights(&mut rng, self.ty, rows * cols, magnitude);
                builder.tensor(name, &[cols as u64, rows as u64], id, data);
            };
        let weight_scale = 1.5 / (d as f32).sqrt();
        matrix(&mut builder, "token_embd.weight", self.n_vocab, d, 1.0);
        matrix(&mut builder, "output.weight", self.n_vocab, d, weight_scale);
        for layer in 0..self.n_layer {
            let name = |part: &str| format!("blk.{layer}.{part}");
            if self.fused {
                matrix(
                    &mut builder,
                    &name("attn_qkv.weight"),
                    q_dim + 2 * kv_dim,
                    d,
                    weight_scale,
                );
                matrix(
                    &mut builder,
                    &name("ffn_up.weight"),
                    2 * n_ff,
                    d,
                    weight_scale,
                );
            } else {
                matrix(&mut builder, &name("attn_q.weight"), q_dim, d, weight_scale);
                matrix(
                    &mut builder,
                    &name("attn_k.weight"),
                    kv_dim,
                    d,
                    weight_scale,
                );
                matrix(
                    &mut builder,
                    &name("attn_v.weight"),
                    kv_dim,
                    d,
                    weight_scale,
                );
                matrix(
                    &mut builder,
                    &name("ffn_gate.weight"),
                    n_ff,
                    d,
                    weight_scale,
                );
                matrix(&mut builder, &name("ffn_up.weight"), n_ff, d, weight_scale);
            }
            matrix(
                &mut builder,
                &name("attn_output.weight"),
                d,
                q_dim,
                weight_scale,
            );
            matrix(
                &mut builder,
                &name("ffn_down.weight"),
                d,
                n_ff,
                weight_scale,
            );
        }

        // Vectors are always stored as f32.
        let mut rng = Rng(self.seed ^ 0x5eed);
        let mut vector = |builder: &mut Builder, name: &str, len: usize, low: f32, high: f32| {
            builder.tensor_f32(name, &[len as u64], &rng.floats(len, low, high));
        };
        vector(&mut builder, "output_norm.weight", d, 0.5, 1.5);
        if self.rope_factors {
            vector(
                &mut builder,
                "rope_freqs.weight",
                self.head_dim() / 2,
                1.0,
                8.0,
            );
        }
        for layer in 0..self.n_layer {
            let name = |part: &str| format!("blk.{layer}.{part}");
            vector(&mut builder, &name("attn_norm.weight"), d, 0.5, 1.5);
            vector(&mut builder, &name("ffn_norm.weight"), d, 0.5, 1.5);
            if self.qk_norm {
                vector(
                    &mut builder,
                    &name("attn_q_norm.weight"),
                    self.head_dim(),
                    0.5,
                    1.5,
                );
                vector(
                    &mut builder,
                    &name("attn_k_norm.weight"),
                    self.head_dim(),
                    0.5,
                    1.5,
                );
            }
            if self.post_norm {
                vector(
                    &mut builder,
                    &name("post_attention_norm.weight"),
                    d,
                    0.5,
                    1.5,
                );
                vector(&mut builder, &name("post_ffw_norm.weight"), d, 0.5, 1.5);
            }
            if self.bias {
                if self.fused {
                    vector(
                        &mut builder,
                        &name("attn_qkv.bias"),
                        q_dim + 2 * kv_dim,
                        -0.2,
                        0.2,
                    );
                } else {
                    vector(&mut builder, &name("attn_q.bias"), q_dim, -0.2, 0.2);
                    vector(&mut builder, &name("attn_k.bias"), kv_dim, -0.2, 0.2);
                    vector(&mut builder, &name("attn_v.bias"), kv_dim, -0.2, 0.2);
                }
            }
        }
        builder.build()
    }
}

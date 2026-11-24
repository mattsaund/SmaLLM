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
    /// Elements of each head that rotate. Zero for all of them.
    pub rope_dims: usize,
    /// Head width, when it is not `n_embd / n_head`. Sliding-window layers
    /// of LFM3 and Gemma 4 use the default width, the others this one.
    pub key_length: usize,
    /// Taps of the short convolution, which then replaces attention in the
    /// even layers (LFM models). Zero for none.
    pub conv_kernel: usize,
    /// Gates each attention head's output (LFM3).
    pub head_gate: bool,
    /// The last layers that read earlier layers' caches (Gemma 4).
    pub shared_kv: usize,
    /// Width of each layer's per-layer input (Gemma 4). Zero for none.
    pub per_layer: usize,
    /// Leaves out the value projections, so keys serve as values (Gemma 4).
    pub values_from_keys: bool,
    /// Scales the output of every block (Gemma 4).
    pub output_scale: bool,
    /// Doubles the feed-forward width of the second half of the layers.
    pub wide_late_ff: bool,
    /// Experts per mixture-of-experts layer. Zero for dense models.
    pub experts: usize,
    /// Experts each token goes through.
    pub experts_used: usize,
    /// Leading layers that stay dense in a model with experts.
    pub dense_lead: usize,
    /// GGUF's code for how router scores become weights: 1 for a softmax,
    /// 2 for a sigmoid, 3 for a softmax over the chosen experts.
    pub gating: u32,
    /// A bias on the router scores, used only to choose (LFM2 MoE).
    pub choice_bias: bool,
    /// A gated expert that every token goes through (Qwen2 MoE).
    pub shared_expert: bool,
    /// Stores each expert's gate and up projections as one tensor.
    pub fused_experts: bool,
    /// Query and key norms as wide as all heads together (OLMoE).
    pub wide_qk_norm: bool,
    /// Granite's multipliers on embeddings, residuals, scores and logits.
    pub granite_scales: bool,
    /// Phi-3's LongRoPE: short and long rotary divisors, a short original
    /// context, and a rotary attention factor.
    pub longrope: bool,
    /// YaRN position scaling, with a short original context so that its
    /// ramp falls inside the small heads.
    pub yarn: bool,
    /// gpt-oss: attention sinks, an output bias, biases on the router and
    /// on every expert, and the feed-forward norm under another name.
    pub gpt_oss: bool,
    /// Phi-2: LayerNorm with biases, attention and an ungated network side
    /// by side, biases on every projection and on the output.
    pub phi2: bool,
    pub seed: u64,
}

impl TinyModel {
    /// A model small enough to evaluate in microseconds: two layers, or
    /// more for the architectures whose layers differ.
    pub fn new(arch: &'static str, ty: TensorType) -> TinyModel {
        let width = ty.block_len().max(32);
        let mut model = TinyModel {
            arch,
            ty,
            n_embd: width,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            n_ff: width,
            n_vocab: 40,
            qk_norm: matches!(
                arch,
                "qwen3" | "qwen3moe" | "gemma3" | "gemma4" | "lfm2" | "lfm2moe" | "lfm3" | "olmoe"
            ),
            post_norm: matches!(arch, "gemma2" | "gemma3" | "gemma4" | "lfm3"),
            bias: matches!(arch, "qwen2" | "qwen2moe" | "gpt-oss" | "phi2"),
            fused: false,
            window: 0,
            rope_factors: false,
            rope_dims: 0,
            key_length: 0,
            conv_kernel: 0,
            head_gate: false,
            shared_kv: 0,
            per_layer: 0,
            values_from_keys: false,
            output_scale: false,
            wide_late_ff: false,
            experts: 0,
            experts_used: 0,
            dense_lead: 0,
            gating: 1,
            choice_bias: false,
            shared_expert: false,
            fused_experts: false,
            wide_qk_norm: arch == "olmoe",
            granite_scales: matches!(arch, "granite" | "granitemoe"),
            longrope: arch == "phi3",
            yarn: matches!(arch, "qwen3" | "gpt-oss"),
            gpt_oss: arch == "gpt-oss",
            phi2: arch == "phi2",
            seed: 1,
        };
        if matches!(
            arch,
            "qwen2moe" | "qwen3moe" | "olmoe" | "granitemoe" | "lfm2moe" | "gpt-oss"
        ) {
            model.experts = 4;
            model.experts_used = 2;
        }
        match arch {
            "lfm2" => {
                model.n_layer = 4;
                model.conv_kernel = 3;
            }
            "lfm2moe" => {
                model.n_layer = 4;
                model.conv_kernel = 3;
                model.dense_lead = 1;
                model.gating = 2;
                model.choice_bias = true;
            }
            "qwen2moe" => model.shared_expert = true,
            // Phi-3 stores query, key and value together, and the gate
            // with `up`.
            "phi3" => model.fused = true,
            // Phi-2 rotates part of each head.
            "phi2" => model.rope_dims = model.head_dim() / 2,
            // Every other layer of Gemma 2 and gpt-oss is windowed.
            "gemma2" | "gpt-oss" => {
                model.n_layer = 4;
                model.window = 3;
            }
            // Every fourth layer is left unrotated.
            "smollm3" => model.n_layer = 4,
            "lfm3" => {
                model.n_layer = 6;
                model.conv_kernel = 4;
                model.head_gate = true;
                model.key_length = 2 * model.head_dim();
                model.window = 3;
                model.rope_factors = true;
            }
            "gemma4" => {
                model.n_layer = 6;
                model.n_head_kv = 1;
                model.key_length = 2 * model.head_dim();
                model.window = 3;
                model.rope_factors = true;
                model.shared_kv = 3;
                model.per_layer = width;
                model.values_from_keys = true;
                model.output_scale = true;
                model.wide_late_ff = true;
            }
            _ => {}
        }
        model
    }

    /// The default head width: the embedding split between the heads.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// True when layer `index` is a short convolution.
    pub fn is_conv(&self, index: usize) -> bool {
        self.conv_kernel > 0 && index % 2 == 0
    }

    /// True when layer `index` uses the sliding window, for the
    /// architectures that store a flag per layer.
    pub fn is_local(&self, index: usize) -> bool {
        self.window > 0 && index % 3 != 2
    }

    /// True when the architecture stores one sliding-window flag per layer.
    fn has_window_flags(&self) -> bool {
        matches!(self.arch, "lfm3" | "gemma4") && self.window > 0
    }

    /// Head width of layer `index`.
    pub fn layer_head_dim(&self, index: usize) -> usize {
        let local = self.has_window_flags() && self.is_local(index);
        match self.key_length {
            0 => self.head_dim(),
            _ if local => self.head_dim(),
            width => width,
        }
    }

    /// Feed-forward width of layer `index`.
    pub fn layer_n_ff(&self, index: usize) -> usize {
        if self.wide_late_ff && index >= self.n_layer / 2 {
            2 * self.n_ff
        } else {
            self.n_ff
        }
    }

    /// True when layer `index` is a mixture of experts.
    pub fn has_experts(&self, index: usize) -> bool {
        self.experts > 0 && index >= self.dense_lead
    }

    /// True when layer `index` computes keys and values of its own.
    fn owns_kv(&self, index: usize) -> bool {
        !self.is_conv(index) && index + self.shared_kv < self.n_layer
    }

    /// Serializes the model as the bytes of a GGUF file.
    pub fn build(&self) -> Vec<u8> {
        let mut rng = Rng(self.seed);
        let arch = self.arch;
        let d = self.n_embd;
        let mut builder = Builder::new();
        builder
            .string("general.architecture", arch)
            .string("general.name", "tiny test model")
            .u32(&format!("{arch}.context_length"), 512)
            .u32(&format!("{arch}.embedding_length"), d as u32)
            .u32(&format!("{arch}.block_count"), self.n_layer as u32)
            .u32(&format!("{arch}.attention.head_count"), self.n_head as u32)
            .f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5)
            .f32(&format!("{arch}.attention.layer_norm_epsilon"), 1e-5)
            .f32(&format!("{arch}.rope.freq_base"), 10000.0);
        let layers = 0..self.n_layer;
        if self.wide_late_ff {
            let widths: Vec<i32> = layers.clone().map(|i| self.layer_n_ff(i) as i32).collect();
            builder.i32s(&format!("{arch}.feed_forward_length"), &widths);
        } else {
            builder.u32(&format!("{arch}.feed_forward_length"), self.n_ff as u32);
        }
        if self.conv_kernel > 0 {
            // A convolution layer has no key/value heads.
            let heads: Vec<i32> = layers
                .clone()
                .map(|i| {
                    if self.is_conv(i) {
                        0
                    } else {
                        self.n_head_kv as i32
                    }
                })
                .collect();
            builder
                .i32s(&format!("{arch}.attention.head_count_kv"), &heads)
                .u32(
                    &format!("{arch}.shortconv.l_cache"),
                    self.conv_kernel as u32,
                );
        } else {
            builder.u32(
                &format!("{arch}.attention.head_count_kv"),
                self.n_head_kv as u32,
            );
        }
        if self.key_length > 0 {
            builder.u32(
                &format!("{arch}.attention.key_length"),
                self.key_length as u32,
            );
            if self.has_window_flags() {
                builder.u32(
                    &format!("{arch}.attention.key_length_swa"),
                    self.head_dim() as u32,
                );
            }
        }
        if self.rope_dims > 0 {
            builder.u32(
                &format!("{arch}.rope.dimension_count"),
                self.rope_dims as u32,
            );
        }
        if self.window > 0 {
            builder.u32(
                &format!("{arch}.attention.sliding_window"),
                self.window as u32,
            );
        }
        if self.has_window_flags() {
            let flags: Vec<bool> = layers.clone().map(|i| self.is_local(i)).collect();
            builder
                .bools(&format!("{arch}.attention.sliding_window_pattern"), &flags)
                .f32(&format!("{arch}.rope.freq_base_swa"), 1000.0);
        }
        if self.shared_kv > 0 {
            builder.u32(
                &format!("{arch}.attention.shared_kv_layers"),
                self.shared_kv as u32,
            );
        }
        if self.per_layer > 0 {
            builder.u32(
                &format!("{arch}.embedding_length_per_layer_input"),
                self.per_layer as u32,
            );
        }
        if self.experts > 0 {
            builder
                .u32(&format!("{arch}.expert_count"), self.experts as u32)
                .u32(
                    &format!("{arch}.expert_used_count"),
                    self.experts_used as u32,
                )
                .u32(&format!("{arch}.expert_gating_func"), self.gating)
                .u32(
                    &format!("{arch}.expert_feed_forward_length"),
                    self.n_ff as u32,
                );
        }
        if arch == "gemma2" {
            // Small caps, so that they matter for random weights.
            builder
                .f32(&format!("{arch}.attn_logit_softcapping"), 5.0)
                .f32(&format!("{arch}.final_logit_softcapping"), 10.0);
        }
        if self.longrope {
            builder
                .u32(&format!("{arch}.rope.scaling.original_context_length"), 256)
                .f32(&format!("{arch}.rope.scaling.attn_factor"), 1.1);
        }
        if self.yarn {
            builder
                .string(&format!("{arch}.rope.scaling.type"), "yarn")
                .f32(&format!("{arch}.rope.scaling.factor"), 4.0)
                .u32(&format!("{arch}.rope.scaling.original_context_length"), 64);
        }
        if self.granite_scales {
            builder
                .f32(&format!("{arch}.embedding_scale"), 2.0)
                .f32(&format!("{arch}.residual_scale"), 0.5)
                .f32(&format!("{arch}.logit_scale"), 3.0)
                .f32(&format!("{arch}.attention.scale"), 0.3);
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
        if self.per_layer > 0 {
            let width = self.per_layer * self.n_layer;
            matrix(
                &mut builder,
                "per_layer_token_embd.weight",
                self.n_vocab,
                width,
                1.0,
            );
            matrix(
                &mut builder,
                "per_layer_model_proj.weight",
                width,
                d,
                weight_scale,
            );
        }
        for layer in layers.clone() {
            let name = |part: &str| format!("blk.{layer}.{part}");
            let n_ff = self.layer_n_ff(layer);
            let head_dim = self.layer_head_dim(layer);
            let (q_dim, kv_dim) = (self.n_head * head_dim, self.n_head_kv * head_dim);
            if self.is_conv(layer) {
                matrix(
                    &mut builder,
                    &name("shortconv.in_proj.weight"),
                    3 * d,
                    d,
                    weight_scale,
                );
                matrix(
                    &mut builder,
                    &name("shortconv.out_proj.weight"),
                    d,
                    d,
                    weight_scale,
                );
            } else if self.fused {
                matrix(
                    &mut builder,
                    &name("attn_qkv.weight"),
                    q_dim + 2 * kv_dim,
                    d,
                    weight_scale,
                );
            } else {
                matrix(&mut builder, &name("attn_q.weight"), q_dim, d, weight_scale);
                if self.owns_kv(layer) {
                    matrix(
                        &mut builder,
                        &name("attn_k.weight"),
                        kv_dim,
                        d,
                        weight_scale,
                    );
                    if !self.values_from_keys {
                        matrix(
                            &mut builder,
                            &name("attn_v.weight"),
                            kv_dim,
                            d,
                            weight_scale,
                        );
                    }
                }
            }
            if !self.is_conv(layer) {
                matrix(
                    &mut builder,
                    &name("attn_output.weight"),
                    d,
                    q_dim,
                    weight_scale,
                );
                if self.head_gate {
                    matrix(
                        &mut builder,
                        &name("attn_gate.weight"),
                        self.n_head,
                        d,
                        weight_scale,
                    );
                }
            }
            if self.has_experts(layer) {
                // One matrix per expert, stacked in a three-dimensional
                // tensor, innermost dimension first.
                let count = self.experts;
                let mut expert_rng = Rng(self.seed ^ (0xe0 + layer as u64));
                let mut stacked = |builder: &mut Builder, part: &str, rows: usize, cols: usize| {
                    let n = count * rows * cols;
                    let data = random_weights(&mut expert_rng, self.ty, n, weight_scale);
                    let dims = [cols as u64, rows as u64, count as u64];
                    builder.tensor(&name(part), &dims, id, data);
                };
                if self.fused_experts {
                    stacked(&mut builder, "ffn_gate_up_exps.weight", 2 * n_ff, d);
                } else {
                    stacked(&mut builder, "ffn_gate_exps.weight", n_ff, d);
                    stacked(&mut builder, "ffn_up_exps.weight", n_ff, d);
                }
                stacked(&mut builder, "ffn_down_exps.weight", d, n_ff);
                matrix(
                    &mut builder,
                    &name("ffn_gate_inp.weight"),
                    count,
                    d,
                    weight_scale,
                );
                if self.shared_expert {
                    matrix(
                        &mut builder,
                        &name("ffn_gate_shexp.weight"),
                        n_ff,
                        d,
                        weight_scale,
                    );
                    matrix(
                        &mut builder,
                        &name("ffn_up_shexp.weight"),
                        n_ff,
                        d,
                        weight_scale,
                    );
                    matrix(
                        &mut builder,
                        &name("ffn_down_shexp.weight"),
                        d,
                        n_ff,
                        weight_scale,
                    );
                }
            } else if self.phi2 {
                matrix(&mut builder, &name("ffn_up.weight"), n_ff, d, weight_scale);
                matrix(
                    &mut builder,
                    &name("ffn_down.weight"),
                    d,
                    n_ff,
                    weight_scale,
                );
            } else if self.fused {
                matrix(
                    &mut builder,
                    &name("ffn_up.weight"),
                    2 * n_ff,
                    d,
                    weight_scale,
                );
                matrix(
                    &mut builder,
                    &name("ffn_down.weight"),
                    d,
                    n_ff,
                    weight_scale,
                );
            } else {
                matrix(
                    &mut builder,
                    &name("ffn_gate.weight"),
                    n_ff,
                    d,
                    weight_scale,
                );
                matrix(&mut builder, &name("ffn_up.weight"), n_ff, d, weight_scale);
                matrix(
                    &mut builder,
                    &name("ffn_down.weight"),
                    d,
                    n_ff,
                    weight_scale,
                );
            }
            if self.per_layer > 0 {
                let per = self.per_layer;
                matrix(&mut builder, &name("inp_gate.weight"), per, d, weight_scale);
                matrix(&mut builder, &name("proj.weight"), d, per, weight_scale);
            }
        }

        // Vectors are always stored as f32.
        let mut rng = Rng(self.seed ^ 0x5eed);
        let mut vector = |builder: &mut Builder, name: &str, len: usize, low: f32, high: f32| {
            builder.tensor_f32(name, &[len as u64], &rng.floats(len, low, high));
        };
        // LFM2 files name the final norm after the embeddings.
        let output_norm = if arch == "lfm2" {
            "token_embd_norm.weight"
        } else {
            "output_norm.weight"
        };
        vector(&mut builder, output_norm, d, 0.5, 1.5);
        if self.phi2 {
            vector(&mut builder, "output_norm.bias", d, -0.2, 0.2);
            vector(&mut builder, "output.bias", self.n_vocab, -0.2, 0.2);
        }
        if self.rope_factors {
            // Divisors for the full-attention layers' rotation.
            let full = (0..self.n_layer)
                .find(|&i| !self.is_conv(i) && !(self.has_window_flags() && self.is_local(i)))
                .map_or(self.head_dim(), |i| self.layer_head_dim(i));
            let dims = if self.rope_dims > 0 {
                self.rope_dims
            } else {
                full
            };
            vector(&mut builder, "rope_freqs.weight", dims / 2, 1.0, 8.0);
        }
        if self.per_layer > 0 {
            vector(
                &mut builder,
                "per_layer_proj_norm.weight",
                self.per_layer,
                0.5,
                1.5,
            );
        }
        if self.longrope {
            let pairs = if self.rope_dims > 0 {
                self.rope_dims
            } else {
                self.head_dim()
            } / 2;
            vector(&mut builder, "rope_factors_short.weight", pairs, 1.0, 2.0);
            vector(&mut builder, "rope_factors_long.weight", pairs, 2.0, 16.0);
        }
        for layer in layers {
            let name = |part: &str| format!("blk.{layer}.{part}");
            let head_dim = self.layer_head_dim(layer);
            let (q_dim, kv_dim) = (self.n_head * head_dim, self.n_head_kv * head_dim);
            vector(&mut builder, &name("attn_norm.weight"), d, 0.5, 1.5);
            if self.phi2 {
                // One LayerNorm, before attention and the network both.
                let n_ff = self.layer_n_ff(layer);
                vector(&mut builder, &name("attn_norm.bias"), d, -0.2, 0.2);
                vector(&mut builder, &name("attn_output.bias"), d, -0.2, 0.2);
                vector(&mut builder, &name("ffn_up.bias"), n_ff, -0.2, 0.2);
                vector(&mut builder, &name("ffn_down.bias"), d, -0.2, 0.2);
            } else {
                // gpt-oss calls its feed-forward norm the post-attention
                // norm.
                let ffn_norm = if self.gpt_oss {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                };
                vector(&mut builder, &name(ffn_norm), d, 0.5, 1.5);
            }
            if self.gpt_oss {
                let n_ff = self.layer_n_ff(layer);
                vector(
                    &mut builder,
                    &name("attn_sinks.weight"),
                    self.n_head,
                    -1.0,
                    2.0,
                );
                vector(&mut builder, &name("attn_output.bias"), d, -0.2, 0.2);
                vector(
                    &mut builder,
                    &name("ffn_gate_inp.bias"),
                    self.experts,
                    -0.5,
                    0.5,
                );
                // One bias row per expert.
                let count = self.experts as u64;
                let mut bias_rng = Rng(self.seed ^ (0xb0 + layer as u64));
                for (part, width) in [
                    ("ffn_gate_exps.bias", n_ff),
                    ("ffn_up_exps.bias", n_ff),
                    ("ffn_down_exps.bias", d),
                ] {
                    let values = bias_rng.floats(width * self.experts, -0.3, 0.3);
                    builder.tensor_f32(&name(part), &[width as u64, count], &values);
                }
            }
            if self.is_conv(layer) {
                let taps = self.conv_kernel as u64;
                let mut taps_rng = Rng(self.seed ^ (0xc0 + layer as u64));
                let kernel = taps_rng.floats(self.conv_kernel * d, -1.0, 1.0);
                builder.tensor_f32(&name("shortconv.conv.weight"), &[taps, d as u64], &kernel);
            } else if self.qk_norm {
                let (q_width, k_width) = if self.wide_qk_norm {
                    (q_dim, kv_dim)
                } else {
                    (head_dim, head_dim)
                };
                vector(&mut builder, &name("attn_q_norm.weight"), q_width, 0.5, 1.5);
                if self.owns_kv(layer) {
                    vector(&mut builder, &name("attn_k_norm.weight"), k_width, 0.5, 1.5);
                }
            }
            if self.has_experts(layer) {
                if self.choice_bias {
                    vector(
                        &mut builder,
                        &name("exp_probs_b.bias"),
                        self.experts,
                        -0.5,
                        0.5,
                    );
                }
                if self.shared_expert {
                    vector(
                        &mut builder,
                        &name("ffn_gate_inp_shexp.weight"),
                        d,
                        -0.2,
                        0.2,
                    );
                }
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
            if self.per_layer > 0 {
                vector(&mut builder, &name("post_norm.weight"), d, 0.5, 1.5);
            }
            if self.output_scale {
                vector(
                    &mut builder,
                    &name("layer_output_scale.weight"),
                    1,
                    0.5,
                    1.5,
                );
            }
            if self.bias && !self.is_conv(layer) {
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

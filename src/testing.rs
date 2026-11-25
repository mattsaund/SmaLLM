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
        TensorType::Q4_0
        | TensorType::Q5_0
        | TensorType::Q8_0
        | TensorType::Iq4Nl
        | TensorType::Q1_0
        | TensorType::Q2_0 => &[0],
        TensorType::Q4_1 | TensorType::Q5_1 | TensorType::Q4K | TensorType::Q5K => &[0, 2],
        TensorType::Q2K => &[80, 82],
        TensorType::Q3K => &[108],
        TensorType::Q6K => &[208],
        TensorType::Iq4Xs
        | TensorType::Iq3S
        | TensorType::Iq3Xxs
        | TensorType::Iq2S
        | TensorType::Iq2Xs
        | TensorType::Iq2Xxs
        | TensorType::Iq1S => &[0],
        TensorType::Tq2_0 => &[64],
        TensorType::Tq1_0 => &[52],
        _ => &[],
    }
}

/// How far the weights of `ty` reach for a scale of one, compared with the
/// 127 of Q8_0: the largest integer weight times the largest sub-block
/// scale the format can encode, over 127.
///
/// Random scales of the IQ, TQ, FP4 and the small Q formats are divided by
/// it, so that their random weights come out about as large as Q8_0's. The
/// older formats count as 1, as they did when their tests were tuned.
fn reach(ty: TensorType) -> f32 {
    let largest = match ty {
        // IQ4_NL levels up to 127, times 6-bit scales from -32.
        TensorType::Iq4Xs => 127.0 * 32.0,
        // Grid values up to 15, times 1 + 2 * 15.
        TensorType::Iq3S => 15.0 * 31.0,
        // Grid values up to 62, times (0.5 + 15) / 2.
        TensorType::Iq3Xxs => 62.0 * 7.75,
        // Grid values up to 43, times (0.5 + 15) / 4.
        TensorType::Iq2S | TensorType::Iq2Xs | TensorType::Iq2Xxs => 43.0 * 3.875,
        // A grid value of 1 plus a delta of 1/8, times 2 * 7 + 1.
        TensorType::Iq1M | TensorType::Iq1S => 1.125 * 15.0,
        TensorType::Tq2_0 | TensorType::Tq1_0 | TensorType::Q1_0 => 1.0,
        TensorType::Q2_0 => 2.0,
        // The largest FP4 level, doubled. The scale is set directly.
        TensorType::Mxfp4 | TensorType::Nvfp4 => 12.0,
        _ => 127.0,
    };
    largest / 127.0
}

/// The largest UE4M3 scale byte of NVFP4 that stands for at most `scale`,
/// in ggml's reading, which halves it.
fn ue4m3_below(scale: f32) -> u8 {
    // Twice the scale, as the 4-bit exponent with a bias of 7 and the top
    // 3 bits of the mantissa. Tiny scales keep a mantissa without the
    // leading one, and the smallest of all become zero.
    let doubled = 2.0 * scale.clamp(0.0, 224.0);
    let exponent = ((doubled.to_bits() >> 23) & 0xff) as i32 - 127 + 7;
    if exponent >= 1 {
        ((exponent as u8) << 3) | ((doubled.to_bits() >> 20) & 7) as u8
    } else {
        (doubled * 512.0) as u8
    }
}

/// Overwrites the scale fields of every block with finite values of at most
/// `magnitude`. Random bytes are valid weights in every format, but not
/// valid scales.
pub(crate) fn set_scales(rng: &mut Rng, ty: TensorType, row: &mut [u8], magnitude: f32) {
    let magnitude = magnitude / reach(ty);
    for block in row.chunks_exact_mut(ty.block_bytes()) {
        for &at in half_offsets(ty) {
            let scale = rng.range(-magnitude, magnitude);
            block[at..at + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        }
        match ty {
            // IQ1_M keeps its half scale in the top nibbles of its four
            // scale words, the first word's nibble lowest.
            TensorType::Iq1M => {
                let bits = f32_to_f16(rng.range(-magnitude, magnitude));
                for k in 0..4 {
                    let nibble = ((bits >> (4 * k)) & 0x0f) as u8;
                    block[49 + 2 * k] = (block[49 + 2 * k] & 0x0f) | (nibble << 4);
                }
            }
            // An MXFP4 scale is a power of two, stored as its exponent.
            // Exponent `e` stands for 2 to the `e - 128`, so the float
            // exponent of `scale`, plus one, picks the power just below it.
            TensorType::Mxfp4 => {
                let scale = rng.range(0.0, magnitude);
                block[0] = ((scale.to_bits() >> 23) & 0xff) as u8 + 1;
            }
            // NVFP4 has four unsigned 8-bit float scales, one per 16.
            TensorType::Nvfp4 => {
                for scale in &mut block[..4] {
                    *scale = ue4m3_below(rng.range(0.0, magnitude));
                }
            }
            _ => {}
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

/// The type a matrix of `ty` is stored in when its rows are `cols` long:
/// `ty` when they divide into its blocks, otherwise Q8_0, or f32 when they
/// do not divide into Q8_0's blocks either. Quantizers do the same.
pub(crate) fn fitting(ty: TensorType, cols: usize) -> TensorType {
    if cols % ty.block_len() == 0 {
        ty
    } else if cols % TensorType::Q8_0.block_len() == 0 {
        TensorType::Q8_0
    } else {
        TensorType::F32
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
    /// An expert that every token goes through (Qwen2 MoE, DeepSeek).
    pub shared_expert: bool,
    /// Gates the shared expert's output (Qwen2 MoE).
    pub shared_gate: bool,
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
    /// Groups of experts for group-limited routing (DeepSeek V3). Zero for
    /// none.
    pub expert_groups: usize,
    /// Groups a token's experts may come from.
    pub groups_used: usize,
    /// The router's multiplier on expert weights. One for none.
    pub expert_scale: f32,
    /// Overrides the architecture's default for normalizing expert
    /// weights.
    pub expert_norm: Option<bool>,
    /// Width of the latent of latent attention (DeepSeek V2), which then
    /// replaces attention. Zero for ordinary attention.
    pub kv_rank: usize,
    /// Width of the compressed query of latent attention. Zero for queries
    /// projected from the hidden state directly.
    pub q_rank: usize,
    /// Widths of the unrotated part of a query or key head, and of a value
    /// head, in latent attention. The rotary part is `rope_dims` wide.
    pub nope_dims: usize,
    pub value_dims: usize,
    /// Stores the latent's up-projection as two tensors, the key half
    /// transposed, as newer files do. Older files store one.
    pub split_kv_b: bool,
    /// gpt-oss: attention sinks, an output bias, biases on the router and
    /// on every expert, and the feed-forward norm under another name.
    pub gpt_oss: bool,
    /// Phi-2: LayerNorm with biases, attention and an ungated network side
    /// by side, biases on every projection and on the output.
    pub phi2: bool,
    /// Floats of state per channel of the Mamba blocks. Zero for models
    /// without them, which [`build`](Self::build) writes in another way.
    pub ssm_state: usize,
    /// Groups of Mamba 2 heads that share B and C.
    pub ssm_groups: usize,
    /// Normalizes Mamba 1's time step, B and C (Falcon Mamba).
    pub ssm_norms: bool,
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
            shared_gate: true,
            fused_experts: false,
            wide_qk_norm: arch == "olmoe",
            granite_scales: matches!(arch, "granite" | "granitemoe"),
            longrope: arch == "phi3",
            yarn: matches!(arch, "qwen3" | "gpt-oss" | "deepseek2"),
            expert_groups: 0,
            groups_used: 0,
            expert_scale: 1.0,
            expert_norm: None,
            kv_rank: 0,
            q_rank: 0,
            nope_dims: 0,
            value_dims: 0,
            split_kv_b: false,
            gpt_oss: arch == "gpt-oss",
            phi2: arch == "phi2",
            ssm_state: 0,
            ssm_groups: 1,
            ssm_norms: false,
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
            // DeepSeek-MoE: a dense first layer, then experts with an
            // ungated shared expert.
            "deepseek" => {
                model.n_layer = 3;
                model.experts = 8;
                model.experts_used = 3;
                model.dense_lead = 1;
                model.shared_expert = true;
                model.shared_gate = false;
                model.expert_scale = 1.5;
            }
            // DeepSeek V2: the same around latent attention, with keys and
            // queries half rotary, and values as wide as keys. Variants add
            // the compressed query, the split up-projection and V3's
            // routing.
            "deepseek2" => {
                let head_dim = model.head_dim();
                model.n_layer = 3;
                model.experts = 8;
                model.experts_used = 3;
                model.dense_lead = 1;
                model.shared_expert = true;
                model.shared_gate = false;
                model.kv_rank = width;
                model.nope_dims = head_dim;
                model.rope_dims = head_dim / 2;
                model.value_dims = head_dim;
            }
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
            // Mamba models have no attention and no feed-forward network.
            // Their state rows do not fill the lanes of the scan's sums.
            "mamba" | "mamba2" => {
                model.n_head = 0;
                model.n_head_kv = 0;
                model.n_ff = 0;
                model.ssm_state = if arch == "mamba" { 12 } else { 20 };
                model.ssm_groups = 2;
            }
            // Two Mamba 2 blocks to every attention block.
            "granitehybrid" => {
                model.n_layer = 4;
                model.ssm_state = 20;
                model.ssm_groups = 2;
                model.granite_scales = true;
            }
            // Three blocks: RWKV-7's later blocks mix in the first one's
            // values, and RWKV-6 halves the state after the second.
            "rwkv6" | "rwkv7" => model.n_layer = 3,
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
        if self.ssm_state > 0 {
            return self.build_ssm();
        }
        if self.arch.starts_with("rwkv") {
            return self.build_rwkv();
        }
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
                )
                .u32(
                    &format!("{arch}.leading_dense_block_count"),
                    self.dense_lead as u32,
                );
            if self.expert_groups > 0 {
                builder
                    .u32(
                        &format!("{arch}.expert_group_count"),
                        self.expert_groups as u32,
                    )
                    .u32(
                        &format!("{arch}.expert_group_used_count"),
                        self.groups_used as u32,
                    );
            }
            if self.expert_scale != 1.0 {
                builder.f32(&format!("{arch}.expert_weights_scale"), self.expert_scale);
            }
            if let Some(normalize) = self.expert_norm {
                builder.bool(&format!("{arch}.expert_weights_norm"), normalize);
            }
            if self.shared_expert {
                builder.u32(&format!("{arch}.expert_shared_count"), 1);
            }
        }
        if self.kv_rank > 0 {
            // Newer files give the cache's widths as the key and value
            // lengths, and the heads' widths under their own names.
            let (rank, key) = (
                self.kv_rank as u32,
                (self.nope_dims + self.rope_dims) as u32,
            );
            builder
                .u32(&format!("{arch}.attention.kv_lora_rank"), rank)
                .u32(&format!("{arch}.attention.q_lora_rank"), self.q_rank as u32);
            if self.split_kv_b {
                builder
                    .u32(
                        &format!("{arch}.attention.key_length"),
                        rank + self.rope_dims as u32,
                    )
                    .u32(&format!("{arch}.attention.value_length"), rank)
                    .u32(&format!("{arch}.attention.key_length_mla"), key)
                    .u32(
                        &format!("{arch}.attention.value_length_mla"),
                        self.value_dims as u32,
                    );
            } else {
                builder
                    .u32(&format!("{arch}.attention.key_length"), key)
                    .u32(
                        &format!("{arch}.attention.value_length"),
                        self.value_dims as u32,
                    );
            }
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
            if arch == "deepseek2" {
                // As DeepSeek V2 files store it: a tenth of `mscale_all_dim`.
                builder.f32(&format!("{arch}.rope.scaling.yarn_log_multiplier"), 0.0707);
            }
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
                let ty = fitting(self.ty, cols);
                let data = random_weights(&mut rng, ty, rows * cols, magnitude);
                builder.tensor(name, &[cols as u64, rows as u64], type_id(ty), data);
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
            } else if self.kv_rank > 0 {
                let (rank, rope) = (self.kv_rank, self.rope_dims);
                let (nope, value) = (self.nope_dims, self.value_dims);
                let q_width = self.n_head * (nope + rope);
                if self.q_rank > 0 {
                    let q_scale = 1.5 / (self.q_rank as f32).sqrt();
                    let q_rank = self.q_rank;
                    matrix(
                        &mut builder,
                        &name("attn_q_a.weight"),
                        q_rank,
                        d,
                        weight_scale,
                    );
                    matrix(
                        &mut builder,
                        &name("attn_q_b.weight"),
                        q_width,
                        q_rank,
                        q_scale,
                    );
                } else {
                    matrix(
                        &mut builder,
                        &name("attn_q.weight"),
                        q_width,
                        d,
                        weight_scale,
                    );
                }
                matrix(
                    &mut builder,
                    &name("attn_kv_a_mqa.weight"),
                    rank + rope,
                    d,
                    weight_scale,
                );
                let latent_scale = 1.5 / (rank as f32).sqrt();
                if self.split_kv_b {
                    // One matrix per head, stacked like the experts'.
                    let mut head_rng = Rng(self.seed ^ (0xa0 + layer as u64));
                    let n_head = self.n_head;
                    let mut per_head = |builder: &mut Builder, part: &str, rows, cols| {
                        let ty = fitting(self.ty, cols);
                        let n = n_head * rows * cols;
                        let data = random_weights(&mut head_rng, ty, n, latent_scale);
                        let dims = [cols as u64, rows as u64, n_head as u64];
                        builder.tensor(&name(part), &dims, type_id(ty), data);
                    };
                    per_head(&mut builder, "attn_k_b.weight", rank, nope);
                    per_head(&mut builder, "attn_v_b.weight", value, rank);
                } else {
                    // Each head's key rows, then its value rows.
                    let rows = self.n_head * (nope + value);
                    matrix(
                        &mut builder,
                        &name("attn_kv_b.weight"),
                        rows,
                        rank,
                        latent_scale,
                    );
                }
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
                // Latent attention's output reads its value heads.
                let out_cols = if self.kv_rank > 0 {
                    self.n_head * self.value_dims
                } else {
                    q_dim
                };
                matrix(
                    &mut builder,
                    &name("attn_output.weight"),
                    d,
                    out_cols,
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
                if self.shared_expert && self.shared_gate {
                    vector(
                        &mut builder,
                        &name("ffn_gate_inp_shexp.weight"),
                        d,
                        -0.2,
                        0.2,
                    );
                }
            }
            if self.kv_rank > 0 {
                vector(
                    &mut builder,
                    &name("attn_kv_a_norm.weight"),
                    self.kv_rank,
                    0.5,
                    1.5,
                );
                if self.q_rank > 0 {
                    vector(
                        &mut builder,
                        &name("attn_q_a_norm.weight"),
                        self.q_rank,
                        0.5,
                        1.5,
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

/// Synthetic models with Mamba blocks: Mamba and Falcon Mamba, Mamba 2,
/// and Granite 4.0 H, which mixes Mamba 2 blocks with attention blocks that
/// do not rotate, and gives every block a feed-forward network.
impl TinyModel {
    /// Channels of each Mamba 2 head.
    const SSM_HEAD_DIM: usize = 16;

    /// True when layer `index` is a Mamba block. Granite 4.0 H has two of
    /// them to every attention block.
    pub fn is_mamba(&self, index: usize) -> bool {
        self.ssm_state > 0 && (self.arch != "granitehybrid" || index % 3 != 1)
    }

    /// Serializes a model with Mamba blocks as the bytes of a GGUF file,
    /// with the tensor names and shapes of llama.cpp's converter.
    fn build_ssm(&self) -> Vec<u8> {
        let (arch, d) = (self.arch, self.n_embd);
        let (mamba2, hybrid) = (arch != "mamba", arch == "granitehybrid");
        let (inner, state, groups) = (2 * d, self.ssm_state, self.ssm_groups);
        let heads = if mamba2 {
            inner / Self::SSM_HEAD_DIM
        } else {
            inner
        };
        // Mamba 1's time step rank, as Hugging Face sets it by default.
        let rank = if mamba2 { heads } else { d.div_ceil(16) };
        let channels = if mamba2 {
            inner + 2 * groups * state
        } else {
            inner
        };
        let taps = 4;
        let layers = 0..self.n_layer;

        let mut builder = Builder::new();
        let key = |suffix: &str| format!("{arch}.{suffix}");
        builder
            .string("general.architecture", arch)
            .string("general.name", "tiny test model")
            .u32(&key("context_length"), 512)
            .u32(&key("embedding_length"), d as u32)
            .u32(&key("block_count"), self.n_layer as u32)
            .u32(&key("attention.head_count"), self.n_head as u32)
            .u32(&key("feed_forward_length"), self.n_ff as u32)
            .f32(&key("attention.layer_norm_rms_epsilon"), 1e-5)
            .u32(&key("ssm.conv_kernel"), taps as u32)
            .u32(&key("ssm.inner_size"), inner as u32)
            .u32(&key("ssm.state_size"), state as u32)
            .u32(&key("ssm.time_step_rank"), rank as u32);
        if mamba2 {
            builder.u32(&key("ssm.group_count"), groups as u32);
        } else {
            builder.bool(&key("ssm.dt_b_c_rms"), self.ssm_norms);
        }
        if hybrid {
            // Attention layers have key/value heads, Mamba layers none.
            let kv_heads: Vec<i32> = layers
                .clone()
                .map(|i| {
                    if self.is_mamba(i) {
                        0
                    } else {
                        self.n_head_kv as i32
                    }
                })
                .collect();
            builder
                .i32s(&key("attention.head_count_kv"), &kv_heads)
                .f32(&key("rope.freq_base"), 10000.0)
                .bool(&key("rope.scaling.finetuned"), false)
                .f32(&key("embedding_scale"), 2.0)
                .f32(&key("residual_scale"), 0.5)
                .f32(&key("logit_scale"), 3.0)
                .f32(&key("attention.scale"), 0.3);
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

        let mut rng = Rng(self.seed);
        let mut matrix = |builder: &mut Builder, name: &str, rows: usize, cols: usize| {
            // Mamba 1's time step projection is too narrow for blocks, so
            // files keep it in half floats, as llama.cpp's quantizer does.
            let ty = if cols % self.ty.block_len() == 0 {
                self.ty
            } else {
                TensorType::F16
            };
            let magnitude = 1.5 / (cols as f32).sqrt();
            let data = random_weights(&mut rng, ty, rows * cols, magnitude);
            builder.tensor(name, &[cols as u64, rows as u64], type_id(ty), data);
        };
        matrix(&mut builder, "token_embd.weight", self.n_vocab, d);
        // Mamba models tie the output to the embeddings.
        if arch != "mamba" {
            matrix(&mut builder, "output.weight", self.n_vocab, d);
        }
        let head_dim = if hybrid { self.head_dim() } else { 0 };
        for layer in layers.clone() {
            let name = |part: &str| format!("blk.{layer}.{part}");
            if self.is_mamba(layer) {
                let in_rows = if mamba2 {
                    inner + channels + heads
                } else {
                    2 * inner
                };
                matrix(&mut builder, &name("ssm_in.weight"), in_rows, d);
                if !mamba2 {
                    matrix(&mut builder, &name("ssm_x.weight"), rank + 2 * state, inner);
                    matrix(&mut builder, &name("ssm_dt.weight"), inner, rank);
                }
                matrix(&mut builder, &name("ssm_out.weight"), d, inner);
            } else {
                let (q_dim, kv_dim) = (self.n_head * head_dim, self.n_head_kv * head_dim);
                matrix(&mut builder, &name("attn_q.weight"), q_dim, d);
                matrix(&mut builder, &name("attn_k.weight"), kv_dim, d);
                matrix(&mut builder, &name("attn_v.weight"), kv_dim, d);
                matrix(&mut builder, &name("attn_output.weight"), d, q_dim);
            }
            if hybrid {
                matrix(&mut builder, &name("ffn_gate.weight"), self.n_ff, d);
                matrix(&mut builder, &name("ffn_up.weight"), self.n_ff, d);
                matrix(&mut builder, &name("ffn_down.weight"), d, self.n_ff);
            }
        }

        // Vectors are stored as f32. The decay rates are negative, and with
        // the time step's bias they keep a few positions in the state.
        let mut rng = Rng(self.seed ^ 0x55a);
        let mut vector =
            |builder: &mut Builder, name: &str, dims: &[usize], low: f32, high: f32| {
                let len = dims.iter().product();
                let dims: Vec<u64> = dims.iter().map(|&dim| dim as u64).collect();
                builder.tensor_f32(name, &dims, &rng.floats(len, low, high));
            };
        vector(&mut builder, "output_norm.weight", &[d], 0.5, 1.5);
        for layer in layers {
            let name = |part: &str| format!("blk.{layer}.{part}");
            vector(&mut builder, &name("attn_norm.weight"), &[d], 0.5, 1.5);
            if hybrid {
                vector(&mut builder, &name("ffn_norm.weight"), &[d], 0.5, 1.5);
            }
            if !self.is_mamba(layer) {
                continue;
            }
            let per_head = if mamba2 { 1 } else { state };
            let mut rates = Rng(self.seed ^ (0xa0 + layer as u64));
            let a: Vec<f32> = rates
                .floats(per_head * heads, -2.0, 0.5)
                .iter()
                .map(|log| -log.exp())
                .collect();
            builder.tensor_f32(&name("ssm_a"), &[per_head as u64, heads as u64], &a);
            vector(
                &mut builder,
                &name("ssm_conv1d.weight"),
                &[taps, channels],
                -1.0,
                1.0,
            );
            vector(
                &mut builder,
                &name("ssm_conv1d.bias"),
                &[channels],
                -0.2,
                0.2,
            );
            vector(&mut builder, &name("ssm_dt.bias"), &[heads], -2.0, 0.0);
            if mamba2 {
                vector(&mut builder, &name("ssm_d"), &[1, heads], 0.5, 1.5);
                vector(
                    &mut builder,
                    &name("ssm_norm.weight"),
                    &[inner / groups, groups],
                    0.5,
                    1.5,
                );
            } else {
                vector(&mut builder, &name("ssm_d"), &[inner], 0.5, 1.5);
            }
        }
        builder.build()
    }
}

impl TinyModel {
    /// Serializes an RWKV model, in the shapes llama.cpp's converter
    /// writes. The low-rank maps are f32 whatever the type of the large
    /// matrices, as converted files keep them in floats.
    fn build_rwkv(&self) -> Vec<u8> {
        let arch = self.arch;
        let v7 = arch == "rwkv7";
        let (d, n_ff) = (self.n_embd, self.n_ff);
        let head_size = self.head_dim();
        let mut builder = Builder::new();
        builder
            .string("general.architecture", arch)
            .string("general.name", "tiny test model")
            .u32(&format!("{arch}.context_length"), 1 << 20)
            .u32(&format!("{arch}.embedding_length"), d as u32)
            .u32(&format!("{arch}.block_count"), self.n_layer as u32)
            .u32(&format!("{arch}.attention.head_count"), 0)
            .u32(&format!("{arch}.wkv.head_size"), head_size as u32)
            .u32(&format!("{arch}.feed_forward_length"), n_ff as u32)
            .f32(&format!("{arch}.attention.layer_norm_epsilon"), 1e-5);
        if !v7 {
            builder.u32(&format!("{arch}.rescale_every_n_layers"), 2);
        }

        // An RWKV World vocabulary: a control token, then plain words.
        let texts: Vec<String> = (0..self.n_vocab)
            .map(|id| match id {
                0 => "<s>".to_string(),
                _ => format!("w{id} "),
            })
            .collect();
        let refs: Vec<&[u8]> = texts.iter().map(|text| text.as_bytes()).collect();
        let types: Vec<i32> = (0..self.n_vocab)
            .map(|id| if id == 0 { 3 } else { 1 })
            .collect();
        builder
            .string("tokenizer.ggml.model", "rwkv")
            .strings("tokenizer.ggml.tokens", &refs)
            .i32s("tokenizer.ggml.token_type", &types)
            .u32("tokenizer.ggml.bos_token_id", 0)
            .u32("tokenizer.ggml.eos_token_id", 0);

        // The large matrices, in the model's type.
        let id = type_id(self.ty);
        let mut rng = Rng(self.seed);
        let mut matrix =
            |builder: &mut Builder, name: &str, rows: usize, cols: usize, magnitude: f32| {
                let data = random_weights(&mut rng, self.ty, rows * cols, magnitude);
                builder.tensor(name, &[cols as u64, rows as u64], id, data);
            };
        let scale = |width: usize| 1.5 / (width as f32).sqrt();
        matrix(&mut builder, "token_embd.weight", self.n_vocab, d, 1.0);
        matrix(&mut builder, "output.weight", self.n_vocab, d, scale(d));
        for layer in 0..self.n_layer {
            let name = |part: &str| format!("blk.{layer}.{part}.weight");
            let mut squares = vec![
                "time_mix_key",
                "time_mix_value",
                "time_mix_receptance",
                "time_mix_output",
            ];
            if !v7 {
                squares.extend(["time_mix_gate", "channel_mix_receptance"]);
            }
            for part in squares {
                matrix(&mut builder, &name(part), d, d, scale(d));
            }
            matrix(&mut builder, &name("channel_mix_key"), n_ff, d, scale(d));
            matrix(
                &mut builder,
                &name("channel_mix_value"),
                d,
                n_ff,
                scale(n_ff),
            );
        }

        // Everything else in f32. Amounts that move toward the previous
        // token lie between 0 and 1.
        let mut rng = Rng(self.seed ^ 0x5eed);
        let mut floats = |builder: &mut Builder, name: &str, dims: &[u64], low: f32, high: f32| {
            let n = dims.iter().product::<u64>() as usize;
            builder.tensor_f32(name, dims, &rng.floats(n, low, high));
        };
        let d64 = d as u64;
        for norm in ["token_embd_norm", "output_norm"] {
            floats(&mut builder, &format!("{norm}.weight"), &[d64], 0.5, 1.5);
            floats(&mut builder, &format!("{norm}.bias"), &[d64], -0.2, 0.2);
        }
        // Ranks of the low-rank maps: decay, learning rate, value mix and
        // gate in RWKV-7, then the token shift's five parts and the decay
        // in RWKV-6.
        let (w_rank, a_rank, v_rank, g_rank) = (8u64, 8u64, 4u64, 16u64);
        let (mix_rank, decay_rank) = (4u64, 8u64);
        for layer in 0..self.n_layer {
            let name = |part: &str| format!("blk.{layer}.{part}");
            for norm in ["attn_norm", "attn_norm_2", "time_mix_ln"] {
                let weight = name(&format!("{norm}.weight"));
                floats(&mut builder, &weight, &[d64], 0.5, 1.5);
                floats(
                    &mut builder,
                    &name(&format!("{norm}.bias")),
                    &[d64],
                    -0.2,
                    0.2,
                );
            }
            floats(
                &mut builder,
                &name("channel_mix_lerp_k.weight"),
                &[d64],
                0.0,
                1.0,
            );
            if v7 {
                let lerp = name("time_mix_lerp_fused.weight");
                floats(&mut builder, &lerp, &[d64, 1, 1, 6], 0.0, 1.0);
                // Each low-rank map: a base (none for the gate), a map down
                // to its rank and one back up, each scaled for its input.
                // The converter gives the first block stand-ins for the
                // value mix, of the learning rate's rank.
                let value_rank = if layer == 0 { a_rank } else { v_rank };
                let maps = [
                    ("w", w_rank),
                    ("a", a_rank),
                    ("v", value_rank),
                    ("g", g_rank),
                ];
                for (part, rank) in maps {
                    if part != "g" {
                        let base = name(&format!("time_mix_{part}0.weight"));
                        floats(&mut builder, &base, &[d64], -1.0, 1.0);
                    }
                    let (down, up) = (scale(d), scale(rank as usize));
                    let first = name(&format!("time_mix_{part}1.weight"));
                    floats(&mut builder, &first, &[d64, rank], -down, down);
                    let second = name(&format!("time_mix_{part}2.weight"));
                    floats(&mut builder, &second, &[rank, d64], -up, up);
                }
                floats(&mut builder, &name("time_mix_k_k.weight"), &[d64], 0.5, 1.5);
                floats(&mut builder, &name("time_mix_k_a.weight"), &[d64], 0.0, 1.0);
                floats(
                    &mut builder,
                    &name("time_mix_r_k.weight"),
                    &[d64],
                    -0.5,
                    0.5,
                );
            } else {
                let (down, up) = (scale(d), scale(mix_rank as usize));
                let wide = [d64, 5 * mix_rank];
                floats(
                    &mut builder,
                    &name("time_mix_w1.weight"),
                    &wide,
                    -down,
                    down,
                );
                let stacked = [mix_rank, d64, 5];
                floats(&mut builder, &name("time_mix_w2.weight"), &stacked, -up, up);
                floats(
                    &mut builder,
                    &name("time_mix_lerp_x.weight"),
                    &[d64],
                    0.0,
                    1.0,
                );
                let lerp = name("time_mix_lerp_fused.weight");
                floats(&mut builder, &lerp, &[d64, 1, 1, 5], 0.0, 1.0);
                let heads = (d / head_size) as u64;
                let first = [head_size as u64, heads];
                floats(
                    &mut builder,
                    &name("time_mix_first.weight"),
                    &first,
                    -1.0,
                    1.0,
                );
                floats(
                    &mut builder,
                    &name("time_mix_decay.weight"),
                    &[d64],
                    -2.0,
                    1.0,
                );
                let (down, up) = (scale(d), scale(decay_rank as usize));
                let first = name("time_mix_decay_w1.weight");
                floats(&mut builder, &first, &[d64, decay_rank], -down, down);
                let second = name("time_mix_decay_w2.weight");
                floats(&mut builder, &second, &[decay_rank, d64], -up, up);
                floats(
                    &mut builder,
                    &name("channel_mix_lerp_r.weight"),
                    &[d64],
                    0.0,
                    1.0,
                );
            }
        }
        builder.build()
    }
}

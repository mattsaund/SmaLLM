//! The forward pass: token ids in, next-token logits out.
//!
//! A [`Session`] is one conversation with a model. It owns the key/value
//! cache, the convolution state, the scratch buffers and the thread pool.
//! Evaluating tokens appends them to the context and leaves the logits for
//! the token that follows.
//!
//! Every block does the same two things:
//!
//! ```text
//! h      = norm(x)
//! x     += [norm] mix(h)                 attention or a short convolution
//! h      = norm(x)
//! x     += [norm] ((act(h * Wgate) * (h * Wup)) * Wdown)
//! ```
//!
//! Attention mixes in every visible earlier position:
//!
//! ```text
//! q k v  = h * Wq, h * Wk, h * Wv        (optional per-head norms)
//! q k    = rotate by position (RoPE)
//! a      = softmax(q . K) . V            over the visible positions
//! mix    = [gate] a * Wo
//! ```
//!
//! A short convolution (the LFM models) mixes in the last few positions:
//!
//! ```text
//! b c s  = h * Wb, h * Wc, h * Ws        two gates and a signal
//! y      = sum over taps i of kernel[i] * (b * s)[position - taps + 1 + i]
//! mix    = (c * y) * Wout
//! ```
//!
//! Gemma 4 also adds a small input of its own to every block, looked up per
//! token. DeepSeek's latent attention is in `forward/mla.rs`. The
//! differences between architectures are all in [`Config`] and in which
//! optional tensors a layer has.
//!
//! A Mamba block mixes positions through a recurrent state instead, and
//! most Mamba models have no feed-forward network. `forward/mamba.rs` has
//! its math.

mod mamba;
mod mla;

use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::gguf::Mapping;
use crate::kernels::{self, Kernels};
use crate::tensor::quant::dequantize_row;
use crate::threads::{ThreadPool, default_batch_threads, default_threads};

use super::Model;
use super::config::{Activation, Config, Gating, LayerSpec, Mixer, Rope, Routing, Yarn};
use super::conv::ConvState;
use super::kv::{CacheType, KvCache};
use super::matmul::{ActScratch, Engine, Target};
use super::recurrent::{self, RecurrentState};
use super::rwkv::{self, RwkvScratch};
use super::weights::{
    AttentionWeights, ConvWeights, Dense, Experts, FeedForward, Linear, Matrix, MixerWeights,
    PerLayerInputs, PerLayerWeights, Plain,
};
use mamba::MambaScratch;
use mla::MlaScratch;

/// Context length used when the caller does not choose one.
const DEFAULT_CONTEXT: usize = 4096;

/// Tokens per forward pass while reading a prompt.
const DEFAULT_BATCH: usize = 32;

/// Attention over fewer multiply-adds than this runs on the calling thread.
const INLINE_ATTENTION_WORK: usize = 64 * 1024;

/// The stages of a forward pass that [`Profile`] times separately.
///
/// Convolution blocks are timed with the attention stages: their input
/// projection as `Project`, the convolution as `Attention` and their output
/// projection as `AttentionOut`. Mamba blocks are timed the same way, with
/// their scan as `Attention`. RWKV blocks too, with the state update as
/// `Attention` and the work between projections and update as `Rotate`.
/// Their channel mixing is timed as the feed-forward stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Token embedding lookup and rotary tables.
    Embed,
    /// Normalizations and residual updates.
    Norm,
    /// The query, key and value projections.
    Project,
    /// Per-head norms, rotation and the cache write.
    Rotate,
    /// Scores, softmax and the weighted sum of values.
    Attention,
    /// The attention output projection.
    AttentionOut,
    /// The feed-forward gate and up projections.
    FeedForwardUp,
    /// The gate activation.
    Activation,
    /// The feed-forward down projection.
    FeedForwardDown,
    /// The final norm and the vocabulary projection.
    Logits,
}

impl Phase {
    /// Every phase, in the order a pass runs them.
    pub const ALL: [Phase; 10] = [
        Phase::Embed,
        Phase::Norm,
        Phase::Project,
        Phase::Rotate,
        Phase::Attention,
        Phase::AttentionOut,
        Phase::FeedForwardUp,
        Phase::Activation,
        Phase::FeedForwardDown,
        Phase::Logits,
    ];

    /// A short label for reports.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Embed => "embed",
            Phase::Norm => "norm",
            Phase::Project => "q/k/v projection",
            Phase::Rotate => "rotate + cache",
            Phase::Attention => "attention",
            Phase::AttentionOut => "attention output",
            Phase::FeedForwardUp => "feed-forward up",
            Phase::Activation => "activation",
            Phase::FeedForwardDown => "feed-forward down",
            Phase::Logits => "logits",
        }
    }
}

/// Where the time of a session's forward passes went.
///
/// Off by default. Turn it on with [`Session::set_profiling`]. The cost is
/// one clock read per stage.
#[derive(Clone, Debug)]
pub struct Profile {
    /// Total time per phase, indexed like [`Phase::ALL`].
    pub phases: [Duration; 10],
    /// Time spent quantizing activations, which is part of the projection
    /// phases above.
    pub quantize: Duration,
    last: Instant,
}

impl Profile {
    fn new() -> Profile {
        Profile {
            phases: [Duration::ZERO; 10],
            quantize: Duration::ZERO,
            last: Instant::now(),
        }
    }

    /// Time across all phases.
    pub fn total(&self) -> Duration {
        self.phases.iter().sum()
    }
}

/// Charges the time since the previous call to `phase`.
fn lap(profile: &mut Option<Profile>, phase: Phase) {
    if let Some(profile) = profile {
        let now = Instant::now();
        profile.phases[phase as usize] += now - profile.last;
        profile.last = now;
    }
}

/// Options for a [`Session`].
#[derive(Clone, Default)]
pub struct SessionConfig {
    /// Context window in tokens. Zero picks the model's training context,
    /// capped at 4096. Any size is accepted: the cache grows as it fills.
    pub n_ctx: usize,
    /// Threads to generate with. Zero picks one per performance core.
    pub threads: usize,
    /// Threads to read prompts with. Zero picks one per physical core,
    /// efficiency cores included, unless `threads` is set, which it then
    /// follows. Reading a prompt is arithmetic that every core speeds up,
    /// where generating waits on memory.
    pub batch_threads: usize,
    /// Tokens evaluated together while reading a prompt. Zero picks 32.
    /// Larger batches read prompts faster and use more scratch memory.
    pub batch: usize,
    /// The kernel backend. `None` picks the fastest one the CPU supports.
    pub kernels: Option<&'static Kernels>,
    /// What the key/value cache stores: 16-bit floats by default, half the
    /// memory of 32-bit ones.
    pub cache: CacheType,
}

/// The rotary tables of one set of rotary settings.
struct RopeKind {
    /// Elements of a head that rotate.
    dims: usize,
    /// Angle per position of each rotated pair.
    freqs: Vec<f64>,
    /// What every rotated value is multiplied by: the model's rotary
    /// attention factor, and YaRN's.
    scale: f32,
    /// `(cos, sin)` of every pair, for each token of the batch.
    table: Vec<f32>,
}

/// The rotary position tables for the batch being evaluated. Layers with
/// the same rotary settings share one table.
struct RopeTables {
    kinds: Vec<RopeKind>,
    /// The table each layer reads. Unused for layers without attention.
    of_layer: Vec<usize>,
}

impl RopeTables {
    fn new(config: &Config, factors: Option<&[f32]>) -> RopeTables {
        let mut kinds = Vec::new();
        let mut settings = Vec::new();
        let mut of_layer = Vec::with_capacity(config.n_layer);
        for layer in &config.layers {
            if layer.mixer != Mixer::Attention {
                of_layer.push(0);
                continue;
            }
            let factors = factors.filter(|_| layer.rope_factors);
            let key = (
                layer.rope_dims,
                layer.rope_base.to_bits(),
                layer.rope_scale.to_bits(),
                factors.is_some(),
            );
            let index = match settings.iter().position(|seen| *seen == key) {
                Some(index) => index,
                None => {
                    settings.push(key);
                    // YaRN scales rotated values up by a tenth of the log of
                    // how far the layer stretches positions, as llama.cpp
                    // does.
                    let scale = match &config.yarn {
                        Some(yarn) => {
                            let stretch = 1.0 / layer.rope_scale;
                            yarn.attn_factor * (1.0 + 0.1 * stretch.ln())
                        }
                        None => 1.0,
                    };
                    kinds.push(RopeKind {
                        dims: layer.rope_dims,
                        freqs: frequencies(layer, factors, config.yarn.as_ref()),
                        scale: scale * config.rope_attn_factor,
                        table: Vec::new(),
                    });
                    kinds.len() - 1
                }
            };
            of_layer.push(index);
        }
        RopeTables { kinds, of_layer }
    }

    /// Computes the tables for `count` tokens starting at `first_position`.
    ///
    /// Angles are computed in double precision. A position far into a long
    /// context times a high frequency loses most of its fraction in f32.
    fn fill(&mut self, first_position: usize, count: usize) {
        for kind in &mut self.kinds {
            kind.table.clear();
            for index in 0..count {
                let position = (first_position + index) as f64;
                for frequency in &kind.freqs {
                    let (sin, cos) = (position * frequency).sin_cos();
                    kind.table.push(cos as f32 * kind.scale);
                    kind.table.push(sin as f32 * kind.scale);
                }
            }
        }
    }

    /// `(cos, sin)` of every pair for a rotation back by `delta` positions,
    /// one table for each set of rotary settings.
    fn back(&self, delta: usize) -> Vec<Vec<f32>> {
        self.kinds
            .iter()
            .map(|kind| {
                kind.freqs
                    .iter()
                    .flat_map(|frequency| {
                        let (sin, cos) = (-(delta as f64) * frequency).sin_cos();
                        [cos as f32, sin as f32]
                    })
                    .collect()
            })
            .collect()
    }

    /// The `(cos, sin)` pairs of `layer` for token `index` of the batch.
    fn row(&self, layer: usize, index: usize) -> &[f32] {
        let kind = &self.kinds[self.of_layer[layer]];
        &kind.table[index * kind.dims..(index + 1) * kind.dims]
    }
}

/// The rotary divisors a session of `n_ctx` tokens uses: a LongRoPE model's
/// long ones when the session reaches past the model's original context,
/// as llama.cpp chooses them.
fn rope_factors(model: &Model, n_ctx: usize) -> Option<&[f32]> {
    let weights = model.weights();
    match &weights.rope_factors_long {
        Some(long) if n_ctx > model.config().long_context_after => Some(long),
        _ => weights.rope_factors.as_deref(),
    }
}

/// The angle per position of each rotated pair of a layer's heads.
///
/// Under YaRN, the pairs that turn many times over the original context
/// keep their frequency, the ones that turn few times are scaled like the
/// rest, and a linear ramp blends the two in between. The ramp's ends are
/// computed as in ggml, from the pairs that make `beta_fast` and
/// `beta_slow` turns.
fn frequencies(layer: &LayerSpec, factors: Option<&[f32]>, yarn: Option<&Yarn>) -> Vec<f64> {
    let dims = layer.rope_dims;
    let base = layer.rope_base as f64;
    let scale = layer.rope_scale as f64;
    // The pairs at which the ramp starts and ends.
    let ramp = yarn.map(|yarn| {
        let pair = |turns: f32| {
            let context = yarn.original_context as f64;
            dims as f64 * (context / (turns as f64 * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = pair(yarn.beta_fast).floor().max(0.0);
        let high = pair(yarn.beta_slow).ceil().min(dims as f64 - 1.0);
        (low, high)
    });
    (0..dims / 2)
        .map(|i| {
            let exponent = -2.0 * i as f64 / dims as f64;
            let divisor = factors.map_or(1.0, |f| f[i] as f64);
            let kept = base.powf(exponent) / divisor;
            let scaled = kept * scale;
            match ramp {
                // How much of the kept frequency to use: all of it below
                // the ramp, none above.
                Some((low, high)) => {
                    let along = (i as f64 - low) / (high - low).max(0.001);
                    let keep = 1.0 - along.clamp(0.0, 1.0);
                    scaled * (1.0 - keep) + kept * keep
                }
                None => scaled,
            }
        })
        .collect()
}

/// Rotates the rotary part of one head by its position.
fn rotate(vector: &mut [f32], table: &[f32], rope: Rope) {
    let half = vector.len() / 2;
    match rope {
        Rope::Pairs => {
            for (pair, angle) in vector.chunks_exact_mut(2).zip(table.chunks_exact(2)) {
                let (x0, x1) = (pair[0], pair[1]);
                pair[0] = x0 * angle[0] - x1 * angle[1];
                pair[1] = x0 * angle[1] + x1 * angle[0];
            }
        }
        Rope::Halves => {
            let (low, high) = vector.split_at_mut(half);
            for ((x0, x1), angle) in low.iter_mut().zip(high).zip(table.chunks_exact(2)) {
                let (a, b) = (*x0, *x1);
                *x0 = a * angle[0] - b * angle[1];
                *x1 = a * angle[1] + b * angle[0];
            }
        }
    }
}

/// Normalizes `x` into `out`: LayerNorm when there is a bias, RMS norm
/// otherwise.
fn norm(
    kernels: &Kernels,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    match bias {
        Some(bias) => layer_norm(x, weight, bias, eps, out),
        None => rms_norm(kernels, x, weight, eps, out),
    }
}

/// LayerNorm: `(x - mean) / sqrt(variance + eps) * weight + bias`. The sums
/// are taken in double precision, as ggml takes them.
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let scale = 1.0 / (variance + eps as f64).sqrt();
    for (((out, &v), w), b) in out.iter_mut().zip(x).zip(weight).zip(bias) {
        *out = ((v as f64 - mean) * scale) as f32 * w + b;
    }
}

/// RMS normalization with a learned gain: `out = x / rms(x) * weight`.
fn rms_norm(kernels: &Kernels, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let scale = 1.0 / (kernels.sum_squares(x) / x.len() as f32 + eps).sqrt();
    for ((out, value), gain) in out.iter_mut().zip(x).zip(weight) {
        *out = value * scale * gain;
    }
}

/// [`rms_norm`] in place.
fn rms_norm_in_place(kernels: &Kernels, x: &mut [f32], weight: &[f32], eps: f32) {
    let scale = 1.0 / (kernels.sum_squares(x) / x.len() as f32 + eps).sqrt();
    for (value, gain) in x.iter_mut().zip(weight) {
        *value *= scale * gain;
    }
}

/// RMS normalization without a gain, in place.
fn rms_norm_unit(kernels: &Kernels, x: &mut [f32], eps: f32) {
    let scale = 1.0 / (kernels.sum_squares(x) / x.len() as f32 + eps).sqrt();
    x.iter_mut().for_each(|value| *value *= scale);
}

/// A query or key norm that applies to each head on its own.
fn head_norm(norm: &Option<Vec<f32>>, head_dim: usize) -> Option<&[f32]> {
    norm.as_deref().filter(|weight| weight.len() == head_dim)
}

/// A query or key norm that applies to all heads together (OLMoE).
fn whole_norm(norm: &Option<Vec<f32>>, head_dim: usize) -> Option<&[f32]> {
    norm.as_deref().filter(|weight| weight.len() != head_dim)
}

/// Reads row `row` of `matrix` as f32.
fn read_row(map: &Mapping, matrix: &Matrix, row: usize, out: &mut [f32]) {
    let start = matrix.offset + row * matrix.row_bytes;
    dequantize_row(
        matrix.ty,
        &map.as_slice()[start..start + matrix.row_bytes],
        out,
    );
}

/// Buffers of the attention blocks, sized for a full batch.
struct AttentionScratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// The attention output before its projection.
    attended: Vec<f32>,
    /// Output gates, one per head, for models that have them.
    gates: Vec<f32>,
    /// Attention scores and a scaled query, one pair per thread slot.
    slots: Slots,
}

/// Buffers of the convolution blocks. Empty for models without them.
#[derive(Default)]
struct ConvScratch {
    /// The input gate, output gate and signal of each token.
    b: Vec<f32>,
    c: Vec<f32>,
    signal: Vec<f32>,
    /// The convolution's inputs: the saved state, then one row per token.
    history: Vec<f32>,
    /// One token's weighted sum over the kernel taps.
    sum: Vec<f32>,
}

/// Buffers of the mixture-of-experts layers. Empty for other models.
#[derive(Default)]
struct ExpertScratch {
    /// The router's scores, one row per token.
    scores: Vec<f32>,
    /// Experts already picked for the row being routed.
    taken: Vec<bool>,
    /// Each row's experts and weights, row after row.
    chosen: Vec<(usize, f32)>,
    /// Where each expert's rows start in the grouped order, and the next
    /// free place while they are sorted there.
    starts: Vec<usize>,
    next: Vec<usize>,
    /// The grouped order: for each expert, the rows routed to it, with
    /// their weights.
    pairs: Vec<(usize, f32)>,
    /// The rows of `pairs` that each expert in use reads.
    groups: Vec<Range<usize>>,
    /// The gate, up and down maps of each expert in use.
    linears: Vec<Linear>,
    /// The expert each group belongs to.
    group_experts: Vec<usize>,
    /// One expert's bias row, read from the file.
    bias: Vec<f32>,
    /// The input of every pair, then its gate and up projections, then
    /// its output.
    inputs: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    out: Vec<f32>,
    /// The shared expert's gate, one per row.
    shared_gates: Vec<f32>,
}

/// Buffers of Gemma 4's per-layer inputs. Empty for other models.
#[derive(Default)]
struct PerLayerScratch {
    /// Each token's input to every layer, side by side.
    inputs: Vec<f32>,
    /// One token's row of the per-layer embeddings.
    row: Vec<f32>,
    /// The per-layer gate of each active token.
    gate: Vec<f32>,
}

/// Buffers reused by every forward pass. Sizes are for a full batch.
struct Scratch {
    /// The hidden state, one row per token.
    x: Vec<f32>,
    /// A normalized copy of the hidden state, then block outputs.
    h: Vec<f32>,
    attention: AttentionScratch,
    conv: ConvScratch,
    mamba: MambaScratch,
    /// The feed-forward gate and up projections.
    gate: Vec<f32>,
    up: Vec<f32>,
    /// The normalized input of a parallel block, kept for its feed-forward
    /// network while attention runs. Empty for other models.
    parallel: Vec<f32>,
    experts: ExpertScratch,
    per_layer: PerLayerScratch,
    rwkv: RwkvScratch,
    mla: MlaScratch,
    acts: ActScratch,
    rope: RopeTables,
}

/// Per-thread scratch for attention.
struct Slots(Vec<UnsafeCell<(Vec<f32>, Vec<f32>)>>);

// SAFETY: the thread pool hands each slot index to one thread at a time, and
// a slot's buffers are only reached through that index.
unsafe impl Sync for Slots {}

/// One conversation with a model.
pub struct Session {
    model: Arc<Model>,
    kernels: &'static Kernels,
    /// Threads for generating, one token at a time.
    pool: ThreadPool,
    /// Threads for reading prompts, when their number differs.
    batch_pool: Option<ThreadPool>,
    kv: KvCache,
    conv: ConvState,
    /// The states of recurrent blocks (Mamba, RWKV).
    recurrent: RecurrentState,
    n_ctx: usize,
    batch: usize,
    /// One past the last block that keeps something for later tokens: a
    /// key/value cache, a convolution state or a recurrent state. From that
    /// block on, only the rows whose logits are wanted are computed in
    /// full.
    state_end: usize,
    scratch: Scratch,
    /// Logits for the token after the last evaluated one.
    logits: Vec<f32>,
    /// The tokens in the context, in order.
    tokens: Vec<u32>,
    /// Stage timings, when profiling is on.
    profile: Option<Profile>,
}

impl Session {
    /// Starts an empty conversation with `model`.
    pub fn new(model: Arc<Model>, config: SessionConfig) -> Result<Session> {
        if model.is_encoder() {
            return Err(Error::invalid(
                "this is an embedding or reranking model, which an Embedder runs",
            ));
        }
        let c = model.config();
        let n_ctx = if config.n_ctx == 0 {
            c.n_ctx_train.min(DEFAULT_CONTEXT)
        } else {
            config.n_ctx
        };
        let batch = if config.batch == 0 {
            DEFAULT_BATCH
        } else {
            config.batch
        };
        let threads = if config.threads == 0 {
            default_threads()
        } else {
            config.threads
        };
        let batch_threads = match (config.batch_threads, config.threads) {
            (0, 0) => default_batch_threads(),
            (0, _) => threads,
            (chosen, _) => chosen,
        };
        if threads > 4096 || batch_threads > 4096 || batch > 1 << 16 {
            return Err(Error::invalid("thread count or batch size out of range"));
        }
        let kernels = config.kernels.unwrap_or_else(kernels::best);
        let pool = ThreadPool::new(threads);
        let batch_pool = (batch_threads != threads).then(|| ThreadPool::new(batch_threads));

        let d = c.n_embd;
        let is_conv: Vec<bool> = c
            .layers
            .iter()
            .map(|layer| layer.mixer == Mixer::ShortConv)
            .collect();
        let conv = ConvState::new(&is_conv, c.conv_kernel, d);
        let conv_scratch = if is_conv.contains(&true) {
            ConvScratch {
                b: vec![0.0; batch * d],
                c: vec![0.0; batch * d],
                signal: vec![0.0; batch * d],
                history: vec![0.0; (conv.rows() + batch) * d],
                sum: vec![0.0; d],
            }
        } else {
            ConvScratch::default()
        };
        let mamba = c
            .layers
            .iter()
            .find_map(|layer| match layer.mixer {
                Mixer::Mamba(ssm) => Some(MambaScratch::new(&ssm, batch)),
                _ => None,
            })
            .unwrap_or_default();
        let per = c.per_layer_embd;
        let per_layer = if per > 0 {
            PerLayerScratch {
                inputs: vec![0.0; batch * per * c.n_layer],
                row: vec![0.0; per * c.n_layer],
                gate: vec![0.0; batch * per],
            }
        } else {
            PerLayerScratch::default()
        };
        let gated_heads = model
            .weights()
            .layers
            .iter()
            .zip(&c.layers)
            .filter(|(layer, _)| {
                matches!(&layer.mixer, MixerWeights::Attention(attention) if attention.gate.is_some())
            })
            .map(|(_, spec)| spec.n_head)
            .max()
            .unwrap_or(0);
        let (q_dim, kv_dim, n_ff) = (c.max_q_dim(), c.max_kv_dim(), c.max_n_ff());
        let experts = match &c.routing {
            Some(routing) => {
                let n_ff_expert = c
                    .layers
                    .iter()
                    .map(|layer| layer.n_ff_expert)
                    .max()
                    .unwrap_or(0);
                let pairs = batch * routing.used;
                ExpertScratch {
                    scores: vec![0.0; batch * routing.count],
                    inputs: vec![0.0; pairs * d],
                    gate: vec![0.0; pairs * n_ff_expert],
                    up: vec![0.0; pairs * n_ff_expert],
                    out: vec![0.0; pairs * d],
                    shared_gates: vec![0.0; batch],
                    ..ExpertScratch::default()
                }
            }
            None => ExpertScratch::default(),
        };
        let scratch = Scratch {
            x: vec![0.0; batch * d],
            h: vec![0.0; batch * d],
            attention: AttentionScratch {
                q: vec![0.0; batch * q_dim],
                k: vec![0.0; batch * kv_dim],
                v: vec![0.0; batch * kv_dim],
                attended: vec![0.0; batch * q_dim],
                gates: vec![0.0; batch * gated_heads],
                slots: Slots(
                    (0..threads.max(batch_threads))
                        .map(|_| UnsafeCell::new((Vec::new(), Vec::new())))
                        .collect(),
                ),
            },
            conv: conv_scratch,
            mamba,
            gate: vec![0.0; batch * n_ff],
            // RWKV's channel mixing has no up projection.
            up: vec![0.0; if c.rwkv.is_some() { 0 } else { batch * n_ff }],
            parallel: if c.parallel {
                vec![0.0; batch * d]
            } else {
                Vec::new()
            },
            experts,
            per_layer,
            rwkv: RwkvScratch::new(c, model.weights(), batch),
            mla: MlaScratch::new(&model, batch),
            acts: ActScratch::default(),
            rope: RopeTables::new(c, rope_factors(&model, n_ctx)),
        };
        // Layers that read another layer's cache have none of their own.
        // Latent attention keeps keys only, and reads values out of them.
        let shapes: Vec<(usize, usize, usize)> = c
            .layers
            .iter()
            .map(|layer| {
                if layer.owns_kv() {
                    let values = if c.mla.is_some() { 0 } else { layer.head_dim };
                    (layer.n_head_kv, layer.head_dim, values)
                } else {
                    (0, 0, 0)
                }
            })
            .collect();
        let recurrent = RecurrentState::new(&recurrent::state_sizes(c));
        let state_end = (0..c.n_layer)
            .rev()
            .find(|&index| {
                let layer = &c.layers[index];
                layer.owns_kv() || layer.mixer == Mixer::ShortConv || recurrent.has(index)
            })
            .map_or(0, |index| index + 1);
        let logits = vec![0.0; c.n_vocab];
        Ok(Session {
            model,
            kernels,
            pool,
            batch_pool,
            kv: KvCache::new(&shapes, config.cache, kernels),
            conv,
            recurrent,
            n_ctx,
            batch,
            state_end,
            scratch,
            logits,
            tokens: Vec::new(),
            profile: None,
        })
    }

    /// The model this session runs.
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// The context window in tokens.
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// The tokens in the context, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of threads the session generates with.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Number of threads the session reads prompts with.
    pub fn batch_threads(&self) -> usize {
        self.batch_pool.as_ref().unwrap_or(&self.pool).threads()
    }

    /// Name of the kernel backend in use.
    pub fn kernels(&self) -> &'static str {
        self.kernels.name
    }

    /// Bytes of memory held for the context: the key/value cache and the
    /// convolution state.
    pub fn cache_bytes(&self) -> usize {
        self.kv.allocated_bytes() + self.conv.bytes() + self.recurrent.bytes()
    }

    /// Starts or stops timing the stages of every forward pass. Starting
    /// resets the timings.
    pub fn set_profiling(&mut self, on: bool) {
        self.profile = on.then(Profile::new);
        self.scratch.acts.set_timing(on);
    }

    /// The stage timings gathered since profiling was turned on.
    pub fn profile(&self) -> Option<Profile> {
        let mut profile = self.profile.clone()?;
        profile.quantize = self.scratch.acts.quantize_time();
        Some(profile)
    }

    /// Scores for the token that follows the context, one per vocabulary
    /// entry. Meaningful after [`eval`](Self::eval) has run on at least one
    /// token.
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Removes every token from position `len` on. The logits are stale
    /// until the next [`eval`](Self::eval).
    ///
    /// A model with convolution or recurrent blocks may have to read tokens
    /// before `len` again to rebuild their state (see `conv.rs` and
    /// `recurrent.rs`).
    pub fn truncate(&mut self, len: usize) {
        if len >= self.tokens.len() {
            return;
        }
        // Both kinds of state must stand at the same position. Each one
        // stays where it is when asked for its own position, so this ends.
        let mut resume = len;
        loop {
            let next = self.recurrent.rewind(self.conv.rewind(resume));
            if next == resume {
                break;
            }
            resume = next;
        }
        self.kv.truncate(resume);
        let replay = self.tokens[resume..len].to_vec();
        self.tokens.truncate(resume);
        self.run(&replay, false);
    }

    /// Removes tokens `keep..keep + discard` from the context and moves the
    /// later ones down to close the gap. This is how a full context makes
    /// room for more: see [`Overflow`](crate::generate::Overflow).
    ///
    /// The moved tokens are not read again. Their cached keys are rotated
    /// back by `discard` positions, which rotary embeddings allow: a key
    /// rotated for position `p` and then back by `d` is the key for `p - d`.
    /// What the kept tokens took from the removed ones stays in their
    /// cache, as in llama.cpp's context shift. The last few tokens always
    /// stay, since a convolution's state is made of them. The logits are
    /// stale until the next [`eval`](Self::eval).
    pub fn shift(&mut self, keep: usize, discard: usize) {
        let len = self.tokens.len();
        let keep = keep.min(len);
        let discard = discard.min(len.saturating_sub(keep + self.conv.rows()));
        if discard == 0 {
            return;
        }
        let model = Arc::clone(&self.model);
        let config = model.config();
        let rope = &self.scratch.rope;
        let back = rope.back(discard);
        self.kv.shift(keep, discard, |layer, key| {
            let dims = config.layers[layer].rope_dims;
            rotate(&mut key[..dims], &back[rope.of_layer[layer]], config.rope);
        });
        self.conv.shift(keep, discard);
        self.recurrent.shift(keep, discard);
        self.tokens.drain(keep..keep + discard);
    }

    /// Empties the context.
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Appends `tokens` to the context and computes the logits for the
    /// token after them.
    pub fn eval(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let needed = self.tokens.len() + tokens.len();
        if needed > self.n_ctx {
            return Err(Error::ContextFull {
                n_ctx: self.n_ctx,
                needed,
            });
        }
        let n_vocab = self.model.config().n_vocab;
        if let Some(&bad) = tokens.iter().find(|&&token| token as usize >= n_vocab) {
            return Err(Error::invalid(format!(
                "token id {bad} is outside the vocabulary of {n_vocab}"
            )));
        }
        self.run(tokens, true);
        Ok(())
    }

    /// Evaluates `tokens` in batches. Only the last batch computes logits,
    /// and only when `want_logits` is set.
    fn run(&mut self, tokens: &[u32], want_logits: bool) {
        // A recurrent state is saved just before the last token of a
        // prompt, where a later prompt is likely to go back to (see
        // `recurrent.rs`). Single generated tokens are not worth a copy.
        let held = usize::from(tokens.len() > 1 && !self.recurrent.is_empty());
        let (head, last) = tokens.split_at(tokens.len() - held);
        let mut batches = head.chunks(self.batch).peekable();
        while let Some(batch) = batches.next() {
            let final_batch = held == 0 && batches.peek().is_none();
            self.forward(batch, want_logits && final_batch);
        }
        if held == 1 {
            self.recurrent.save();
            self.forward(last, want_logits);
        }
    }

    /// The positions of the saved recurrent states, for tests.
    #[cfg(test)]
    pub(crate) fn saved_positions(&self) -> Vec<usize> {
        self.recurrent.saved()
    }

    /// Runs one batch through the model.
    fn forward(&mut self, tokens: &[u32], want_logits: bool) {
        let Session {
            model,
            kernels,
            pool,
            batch_pool,
            kv,
            conv,
            recurrent,
            state_end,
            scratch,
            logits,
            tokens: history,
            profile,
            ..
        } = self;
        if let Some(profile) = profile {
            profile.last = Instant::now();
        }
        let kernels: &'static Kernels = kernels;
        let model: &Model = model;
        // A batch of tokens is arithmetic and gets every core. One token
        // waits on memory and gets the fast ones.
        let pool: &ThreadPool = match batch_pool {
            Some(batch_pool) if tokens.len() > 1 => batch_pool,
            _ => pool,
        };
        let (config, weights) = (model.config(), model.weights());
        let map = model.gguf().mapping();
        let Scratch {
            x,
            h,
            attention,
            conv: conv_scratch,
            mamba,
            gate,
            up,
            parallel,
            experts,
            per_layer,
            rwkv: rwkv_scratch,
            mla: mla_scratch,
            acts,
            rope,
        } = scratch;

        let count = tokens.len();
        let first_position = kv.len();
        let (d, eps) = (config.n_embd, config.norm_eps);

        // Token embeddings start the hidden state.
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            read_row(map, &weights.token_embd, token as usize, row);
            if config.embedding_scale != 1.0 {
                row.iter_mut()
                    .for_each(|value| *value *= config.embedding_scale);
            }
            if let Some((weight, bias)) = &weights.embd_norm {
                rwkv::layer_norm(row, weight, bias, eps);
            }
        }
        rope.fill(first_position, count);
        kv.reserve(first_position + count);
        let mut pass = Pass {
            engine: Engine { kernels, pool, map },
            config,
            rope,
            kv,
            conv,
            recurrent,
            acts,
            profile,
            count,
            first_position,
        };
        if let Some(inputs) = &weights.per_layer_inputs {
            pass.per_layer_inputs(inputs, tokens, &x[..count * d], per_layer);
        }
        pass.lap(Phase::Embed);

        let state_end = *state_end;
        for (index, (layer, spec)) in weights.layers.iter().zip(&config.layers).enumerate() {
            // Before the last block that keeps state, every row matters.
            // From it on, only the rows whose logits are wanted do, though
            // that block still stores its state for every token.
            let active = if index + 1 < state_end {
                0..count
            } else if want_logits {
                count - 1..count
            } else {
                count..count
            };
            let keeps_state =
                spec.owns_kv() || spec.mixer == Mixer::ShortConv || pass.recurrent.has(index);
            let inputs = if keeps_state {
                0..count
            } else {
                active.clone()
            };
            if inputs.is_empty() {
                continue;
            }
            let run = LayerRun {
                index,
                spec,
                inputs,
                active,
            };
            // RWKV blocks take a path of their own.
            if let MixerWeights::Rwkv(_) = &layer.mixer {
                pass.rwkv_block(&run, layer, x, h, rwkv_scratch, gate);
                continue;
            }

            // The mixer: attention or a convolution over positions.
            let attn_norm_bias = layer.attn_norm_bias.as_deref();
            for token in run.inputs.clone() {
                let row = token * d..(token + 1) * d;
                norm(
                    kernels,
                    &x[row.clone()],
                    &layer.attn_norm,
                    attn_norm_bias,
                    eps,
                    &mut h[row],
                );
            }
            pass.lap(Phase::Norm);
            // A parallel block's feed-forward network reads the same input.
            if config.parallel {
                let rows = run.active.start * d..run.active.end * d;
                parallel[..rows.len()].copy_from_slice(&h[rows]);
            }
            match &layer.mixer {
                MixerWeights::Attention(weights) => pass.attention(&run, weights, h, attention),
                MixerWeights::ShortConv(weights) => pass.short_conv(&run, weights, h, conv_scratch),
                MixerWeights::Mamba(weights) => pass.mamba(&run, weights, h, mamba),
                MixerWeights::Rwkv(_) => unreachable!("RWKV blocks took their own path"),
                MixerWeights::Mla(weights) => {
                    pass.latent_attention(&run, weights, h, attention, mla_scratch)
                }
            }
            let n_active = run.active.len();
            if n_active == 0 {
                continue;
            }
            let x_active = &mut x[run.active.start * d..run.active.end * d];
            let h_active = &mut h[..n_active * d];
            add_residual(
                kernels,
                x_active,
                h_active,
                layer.attn_post_norm.as_deref(),
                d,
                eps,
                config.residual_scale,
            );
            // Mamba blocks are their mixer alone.
            if let FeedForward::None = layer.ffn {
                pass.lap(Phase::Norm);
                continue;
            }

            // Feed-forward: a gated network, or a mixture of them.
            if config.parallel {
                h_active.copy_from_slice(&parallel[..n_active * d]);
            } else {
                for (out, row) in h_active.chunks_exact_mut(d).zip(x_active.chunks_exact(d)) {
                    rms_norm(kernels, row, &layer.ffn_norm, eps, out);
                }
            }
            pass.lap(Phase::Norm);
            match &layer.ffn {
                FeedForward::Dense(dense) => pass.dense(dense, spec.n_ff, h_active, gate, up),
                FeedForward::Plain(plain) => pass.plain(plain, spec.n_ff, h_active, gate, up),
                FeedForward::Experts(weights) => {
                    pass.experts(weights, spec, h_active, (gate, up), experts)
                }
                FeedForward::None => unreachable!("Mamba blocks skip the feed-forward network"),
                FeedForward::ChannelMix(_) => unreachable!("RWKV blocks took their own path"),
            }
            add_residual(
                kernels,
                x_active,
                h_active,
                layer.ffn_post_norm.as_deref(),
                d,
                eps,
                config.residual_scale,
            );

            if let Some(weights) = &layer.per_layer {
                pass.per_layer_update(&run, weights, x_active, h_active, per_layer);
            }
            if let Some(scale) = layer.output_scale {
                x_active.iter_mut().for_each(|value| *value *= scale);
            }
            pass.lap(Phase::Norm);
        }
        pass.kv.set_len(first_position + count);
        pass.conv.advance(first_position + count);
        pass.recurrent.advance(first_position + count);
        history.extend_from_slice(tokens);

        if want_logits {
            let last = &x[(count - 1) * d..count * d];
            let normalized = &mut h[..d];
            let bias = weights.output_norm_bias.as_deref();
            norm(kernels, last, &weights.output_norm, bias, eps, normalized);
            pass.multiply(
                normalized,
                d,
                &mut [Target::new(&weights.output, &mut logits[..])],
            );
            if config.logit_scale != 1.0 {
                let scale = 1.0 / config.logit_scale;
                logits.iter_mut().for_each(|logit| *logit *= scale);
            }
            if config.logit_softcap > 0.0 {
                let cap = config.logit_softcap;
                logits
                    .iter_mut()
                    .for_each(|logit| *logit = cap * (*logit / cap).tanh());
            }
            pass.lap(Phase::Logits);
        }
    }
}

/// What the steps of one forward pass share.
pub(super) struct Pass<'a> {
    pub(super) engine: Engine<'a>,
    pub(super) config: &'a Config,
    rope: &'a RopeTables,
    kv: &'a mut KvCache,
    conv: &'a mut ConvState,
    pub(super) recurrent: &'a mut RecurrentState,
    acts: &'a mut ActScratch,
    profile: &'a mut Option<Profile>,
    /// Tokens in the batch.
    pub(super) count: usize,
    /// Context position of the batch's first token.
    first_position: usize,
}

/// One block's share of a batch.
pub(super) struct LayerRun<'a> {
    pub(super) index: usize,
    spec: &'a LayerSpec,
    /// Rows the block reads: every row when it keeps state for later
    /// tokens, otherwise the active rows.
    inputs: Range<usize>,
    /// Rows whose output is needed.
    pub(super) active: Range<usize>,
}

impl Pass<'_> {
    pub(super) fn lap(&mut self, phase: Phase) {
        lap(self.profile, phase);
    }

    pub(super) fn multiply(&mut self, x: &[f32], cols: usize, targets: &mut [Target<'_>]) {
        self.engine.multiply(self.acts, x, cols, targets);
    }

    /// Multiplies the input rows of `h` by the `kept` targets, whose results
    /// the block keeps for every token, and the active rows by the
    /// `queried` targets. It is one multiply when the rows are the same, so
    /// a generated token wakes the threads once.
    fn project<'t>(
        &mut self,
        h: &[f32],
        run: &LayerRun,
        mut kept: Vec<Target<'t>>,
        mut queried: Vec<Target<'t>>,
    ) {
        let d = self.config.n_embd;
        let rows = |range: &Range<usize>| &h[range.start * d..range.end * d];
        if run.inputs == run.active {
            kept.append(&mut queried);
            self.multiply(rows(&run.inputs), d, &mut kept);
        } else {
            self.multiply(rows(&run.inputs), d, &mut kept);
            self.multiply(rows(&run.active), d, &mut queried);
        }
    }

    /// An attention block. Reads the normalized inputs in `h` and leaves
    /// the block's output for the active rows at the start of `h`.
    fn attention(
        &mut self,
        run: &LayerRun,
        w: &AttentionWeights,
        h: &mut [f32],
        s: &mut AttentionScratch,
    ) {
        let (config, rope, spec) = (self.config, self.rope, run.spec);
        let kernels = self.engine.kernels;
        let (d, eps) = (config.n_embd, config.norm_eps);
        let (q_dim, kv_dim, head_dim) = (spec.q_dim(), spec.kv_dim(), spec.head_dim);
        let (count, n_active) = (self.count, run.active.len());
        let n_gates = if w.gate.is_some() { spec.n_head } else { 0 };
        let q = &mut s.q[..n_active * q_dim];
        let k = &mut s.k[..count * kv_dim];
        let v = &mut s.v[..count * kv_dim];
        let gates = &mut s.gates[..n_active * n_gates];

        // Keys and values for every row the cache needs. Queries and gates
        // for the active rows only.
        let mut kept = Vec::with_capacity(4);
        if let Some((wk, wv)) = &w.kv {
            kept.push(Target::new(wk, &mut *k));
            if let Some(wv) = wv {
                kept.push(Target::new(wv, &mut *v));
            }
        }
        let mut queried = Vec::with_capacity(4);
        queried.push(Target::new(&w.q, &mut *q));
        if let Some(wg) = &w.gate {
            queried.push(Target::new(wg, &mut *gates));
        }
        self.project(h, run, kept, queried);
        self.lap(Phase::Project);

        // Per-head normalization, rotation by position, and the cache.
        if let Some((_, wv)) = &w.kv {
            if wv.is_none() {
                v.copy_from_slice(k);
            }
            let rows = k.chunks_exact_mut(kv_dim).zip(v.chunks_exact_mut(kv_dim));
            for (token, (key, value)) in rows.enumerate() {
                let angles = rope.row(run.index, token);
                if let Some(weight) = whole_norm(&w.k_norm, head_dim) {
                    rms_norm_in_place(kernels, key, weight, eps);
                }
                for head in key.chunks_exact_mut(head_dim) {
                    if let Some(weight) = head_norm(&w.k_norm, head_dim) {
                        rms_norm_in_place(kernels, head, weight, eps);
                    }
                    rotate(&mut head[..spec.rope_dims], angles, config.rope);
                }
                if config.value_norm {
                    for head in value.chunks_exact_mut(head_dim) {
                        rms_norm_unit(kernels, head, eps);
                    }
                }
                self.kv
                    .write(run.index, self.first_position + token, key, value);
            }
        }
        for (row, token) in q.chunks_exact_mut(q_dim).zip(run.active.clone()) {
            let angles = rope.row(run.index, token);
            if let Some(weight) = whole_norm(&w.q_norm, head_dim) {
                rms_norm_in_place(kernels, row, weight, eps);
            }
            for head in row.chunks_exact_mut(head_dim) {
                if let Some(weight) = head_norm(&w.q_norm, head_dim) {
                    rms_norm_in_place(kernels, head, weight, eps);
                }
                rotate(&mut head[..spec.rope_dims], angles, config.rope);
            }
        }
        self.lap(Phase::Rotate);
        if n_active == 0 {
            return;
        }

        // A layer without a cache of its own reads an earlier layer's.
        let out = &mut s.attended[..n_active * q_dim];
        let query = Query {
            q,
            first_position: self.first_position + run.active.start,
            window: spec.window,
            softcap: config.attention_softcap,
            sinks: w.sinks.as_deref(),
        };
        let source = spec.kv_source.unwrap_or(run.index);
        attend(&self.engine, self.kv, source, spec, &query, &s.slots, out);
        self.lap(Phase::Attention);

        // Each head's output is scaled by the sigmoid of its gate.
        if n_gates > 0 {
            for (row, gates) in out.chunks_exact_mut(q_dim).zip(gates.chunks_exact(n_gates)) {
                for (head, &gate) in row.chunks_exact_mut(head_dim).zip(gates) {
                    let scale = 1.0 / (1.0 + (-gate).exp());
                    head.iter_mut().for_each(|value| *value *= scale);
                }
            }
        }
        self.multiply(
            out,
            q_dim,
            &mut [Target::new(&w.out, &mut h[..n_active * d])],
        );
        self.lap(Phase::AttentionOut);
    }

    /// A short convolution block. Reads the normalized inputs in `h` and
    /// leaves the block's output for the active rows at the start of `h`.
    fn short_conv(&mut self, run: &LayerRun, w: &ConvWeights, h: &mut [f32], s: &mut ConvScratch) {
        let d = self.config.n_embd;
        let (count, n_active, rows) = (self.count, run.active.len(), self.conv.rows());
        let b = &mut s.b[..count * d];
        let signal = &mut s.signal[..count * d];
        let c = &mut s.c[..n_active * d];
        let kept = vec![Target::new(&w.b, &mut *b), Target::new(&w.x, &mut *signal)];
        let queried = vec![Target::new(&w.c, &mut *c)];
        self.project(h, run, kept, queried);
        self.lap(Phase::Project);

        // The convolution reads the inputs b * signal: the saved state of
        // the positions before the batch, then one row per token.
        let history = &mut s.history[..(rows + count) * d];
        let (state, fresh) = history.split_at_mut(rows * d);
        self.conv.read(run.index, self.first_position, state);
        for ((input, b), signal) in fresh.iter_mut().zip(b.iter()).zip(signal.iter()) {
            *input = b * signal;
        }
        // Each active token sums the inputs of the `taps` positions that
        // end at its own, weighted per channel, then applies its gate c.
        let sum = &mut s.sum[..d];
        for (gate, token) in c.chunks_exact_mut(d).zip(run.active.clone()) {
            sum.fill(0.0);
            for (tap, weights) in w.kernel.chunks_exact(d).enumerate() {
                let input = &history[(token + tap) * d..(token + tap + 1) * d];
                for ((total, weight), value) in sum.iter_mut().zip(weights).zip(input) {
                    *total += weight * value;
                }
            }
            for (gate, total) in gate.iter_mut().zip(sum.iter()) {
                *gate *= total;
            }
        }
        self.conv
            .write(run.index, self.first_position, count, history);
        self.lap(Phase::Attention);
        self.multiply(
            c,
            d,
            &mut [Target::new(&w.out_proj, &mut h[..n_active * d])],
        );
        self.lap(Phase::AttentionOut);
    }

    /// A gated feed-forward network on the rows of `h`, which it replaces
    /// with its output.
    /// An ungated feed-forward network: `down(act(up(x)))`. The activation
    /// runs on the gate kernel, against ones.
    fn plain(&mut self, w: &Plain, n_ff: usize, h: &mut [f32], act: &mut [f32], ones: &mut [f32]) {
        let d = self.config.n_embd;
        let rows = h.len() / d;
        let (act, ones) = (&mut act[..rows * n_ff], &mut ones[..rows * n_ff]);
        self.multiply(h, d, &mut [Target::new(&w.up, &mut *act)]);
        self.lap(Phase::FeedForwardUp);
        ones.fill(1.0);
        let (kernels, pool) = (self.engine.kernels, self.engine.pool);
        apply_gate(kernels, pool, self.config.activation, act, ones);
        self.lap(Phase::Activation);
        self.multiply(act, n_ff, &mut [Target::new(&w.down, h)]);
        self.lap(Phase::FeedForwardDown);
    }

    fn dense(&mut self, w: &Dense, n_ff: usize, h: &mut [f32], gate: &mut [f32], up: &mut [f32]) {
        let d = self.config.n_embd;
        let rows = h.len() / d;
        let (gate, up) = (&mut gate[..rows * n_ff], &mut up[..rows * n_ff]);
        let mut targets = [
            Target::new(&w.gate, &mut *gate),
            Target::new(&w.up, &mut *up),
        ];
        self.multiply(h, d, &mut targets);
        self.lap(Phase::FeedForwardUp);
        apply_gate(
            self.engine.kernels,
            self.engine.pool,
            self.config.activation,
            gate,
            up,
        );
        self.lap(Phase::Activation);
        self.multiply(gate, n_ff, &mut [Target::new(&w.down, h)]);
        self.lap(Phase::FeedForwardDown);
    }

    /// A mixture of experts on the rows of `h`, which it replaces with its
    /// output. The router picks a few experts for each row, and the row's
    /// output is their outputs summed with the router's weights.
    ///
    /// The rows are sorted by expert, so each expert multiplies all of its
    /// rows at once, and every expert in use runs in the same multiply. A
    /// generated token wakes the threads twice per layer, not twice per
    /// expert. `dense` holds the buffers of a shared expert.
    fn experts(
        &mut self,
        w: &Experts,
        spec: &LayerSpec,
        h: &mut [f32],
        dense: (&mut [f32], &mut [f32]),
        s: &mut ExpertScratch,
    ) {
        let config = self.config;
        let routing = config.routing.expect("a model with experts has routing");
        let kernels = self.engine.kernels;
        let (d, n_ff, used) = (config.n_embd, spec.n_ff_expert, routing.used);
        let rows = h.len() / d;
        let total = rows * used;

        // The router's scores, and each row's choice of experts.
        let scores = &mut s.scores[..rows * routing.count];
        self.multiply(h, d, &mut [Target::new(&w.router, &mut *scores)]);
        s.chosen.clear();
        for row in scores.chunks_exact_mut(routing.count) {
            choose(
                kernels,
                &routing,
                w.choice_bias.as_deref(),
                row,
                &mut s.taken,
                &mut s.chosen,
            );
        }

        // Sort the pairs of row and expert by expert.
        s.starts.clear();
        s.starts.resize(routing.count + 1, 0);
        for &(expert, _) in &s.chosen {
            s.starts[expert + 1] += 1;
        }
        for expert in 0..routing.count {
            s.starts[expert + 1] += s.starts[expert];
        }
        s.next.clear();
        s.next.extend_from_slice(&s.starts[..routing.count]);
        s.pairs.clear();
        s.pairs.resize(total, (0, 0.0));
        for (index, &(expert, weight)) in s.chosen.iter().enumerate() {
            s.pairs[s.next[expert]] = (index / used, weight);
            s.next[expert] += 1;
        }
        s.groups.clear();
        s.linears.clear();
        s.group_experts.clear();
        for expert in 0..routing.count {
            let group = s.starts[expert]..s.starts[expert + 1];
            if !group.is_empty() {
                s.groups.push(group);
                s.group_experts.push(expert);
                s.linears.extend([
                    w.gate.expert(expert),
                    w.up.expert(expert),
                    w.down.expert(expert),
                ]);
            }
        }
        let inputs = &mut s.inputs[..total * d];
        for (input, &(row, _)) in inputs.chunks_exact_mut(d).zip(&s.pairs) {
            input.copy_from_slice(&h[row * d..(row + 1) * d]);
        }

        // Every expert's gate and up projections in one multiply.
        let (gate, up) = (&mut s.gate[..total * n_ff], &mut s.up[..total * n_ff]);
        let mut targets = Vec::with_capacity(2 * s.groups.len());
        let (mut gate_rest, mut up_rest) = (&mut *gate, &mut *up);
        for (group, maps) in s.groups.iter().zip(s.linears.chunks_exact(3)) {
            let size = group.len() * n_ff;
            let (gate_part, gate_tail) = std::mem::take(&mut gate_rest).split_at_mut(size);
            let (up_part, up_tail) = std::mem::take(&mut up_rest).split_at_mut(size);
            (gate_rest, up_rest) = (gate_tail, up_tail);
            for (linear, out) in [(&maps[0], gate_part), (&maps[1], up_part)] {
                targets.push(Target {
                    linear,
                    out,
                    inputs: Some(group.clone()),
                });
            }
        }
        self.multiply(inputs, d, &mut targets);
        drop(targets);
        if let Some([gate_bias, up_bias, _]) = &w.biases {
            let map = self.engine.map;
            for (values, bias) in [(&mut *gate, gate_bias), (&mut *up, up_bias)] {
                add_expert_biases(
                    map,
                    bias,
                    &s.groups,
                    &s.group_experts,
                    n_ff,
                    values,
                    &mut s.bias,
                );
            }
        }
        self.lap(Phase::FeedForwardUp);
        apply_gate(kernels, self.engine.pool, config.activation, gate, up);
        self.lap(Phase::Activation);

        // Every expert's down projection, each on its own rows.
        let out = &mut s.out[..total * d];
        let mut targets = Vec::with_capacity(s.groups.len());
        let mut rest = &mut *out;
        for (group, maps) in s.groups.iter().zip(s.linears.chunks_exact(3)) {
            let (part, tail) = std::mem::take(&mut rest).split_at_mut(group.len() * d);
            rest = tail;
            targets.push(Target {
                linear: &maps[2],
                out: part,
                inputs: Some(group.clone()),
            });
        }
        self.multiply(gate, n_ff, &mut targets);
        drop(targets);
        if let Some([_, _, down_bias]) = &w.biases {
            let map = self.engine.map;
            add_expert_biases(
                map,
                down_bias,
                &s.groups,
                &s.group_experts,
                d,
                out,
                &mut s.bias,
            );
        }
        self.lap(Phase::FeedForwardDown);

        // A shared expert sees every row, scaled by its own gate. Then the
        // chosen experts' outputs are added with their weights.
        match &w.shared {
            Some((shared, shared_gate)) => {
                let gates = &mut s.shared_gates[..rows];
                match shared_gate {
                    Some(linear) => {
                        self.multiply(h, d, &mut [Target::new(linear, &mut *gates)]);
                        gates
                            .iter_mut()
                            .for_each(|g| *g = 1.0 / (1.0 + (-*g).exp()));
                    }
                    None => gates.fill(1.0),
                }
                self.dense(shared, spec.n_ff, h, dense.0, dense.1);
                for (row, &g) in h.chunks_exact_mut(d).zip(gates.iter()) {
                    row.iter_mut().for_each(|value| *value *= g);
                }
            }
            None => h.fill(0.0),
        }
        for (output, &(row, weight)) in out.chunks_exact(d).zip(&s.pairs) {
            for (value, x) in h[row * d..(row + 1) * d].iter_mut().zip(output) {
                *value += weight * x;
            }
        }
    }

    /// Gemma 4's per-layer inputs for each token: its own row of the
    /// per-layer embeddings, plus a projection of its main embedding that
    /// is normalized layer by layer.
    fn per_layer_inputs(
        &mut self,
        w: &PerLayerInputs,
        tokens: &[u32],
        x: &[f32],
        s: &mut PerLayerScratch,
    ) {
        let config = self.config;
        let (d, per) = (config.n_embd, config.per_layer_embd);
        let width = per * config.n_layer;
        let inputs = &mut s.inputs[..tokens.len() * width];
        self.multiply(x, d, &mut [Target::new(&w.model_proj, &mut *inputs)]);
        let projection_scale = 1.0 / (d as f32).sqrt();
        let embedding_scale = (per as f32).sqrt();
        for (row, &token) in inputs.chunks_exact_mut(width).zip(tokens) {
            read_row(self.engine.map, &w.token_embd, token as usize, &mut s.row);
            for (input, embedded) in row.chunks_exact_mut(per).zip(s.row.chunks_exact(per)) {
                input
                    .iter_mut()
                    .for_each(|value| *value *= projection_scale);
                rms_norm_in_place(self.engine.kernels, input, &w.proj_norm, config.norm_eps);
                for (value, embedded) in input.iter_mut().zip(embedded) {
                    *value =
                        (*value + embedded * embedding_scale) * std::f32::consts::FRAC_1_SQRT_2;
                }
            }
        }
    }

    /// Adds a block's per-layer input to the active rows `x`: the hidden
    /// state picks what it needs from the input through a gate, and the
    /// result is projected back up. `h` is scratch for the update.
    fn per_layer_update(
        &mut self,
        run: &LayerRun,
        w: &PerLayerWeights,
        x: &mut [f32],
        h: &mut [f32],
        s: &mut PerLayerScratch,
    ) {
        let config = self.config;
        let (d, per, n_active) = (config.n_embd, config.per_layer_embd, run.active.len());
        let width = per * config.n_layer;
        let gate = &mut s.gate[..n_active * per];
        self.multiply(x, d, &mut [Target::new(&w.gate, &mut *gate)]);
        self.lap(Phase::FeedForwardUp);
        for (gate, token) in gate.chunks_exact_mut(per).zip(run.active.clone()) {
            let start = token * width + run.index * per;
            self.engine
                .kernels
                .gelu_mul(gate, &s.inputs[start..start + per]);
        }
        self.lap(Phase::Activation);
        self.multiply(gate, per, &mut [Target::new(&w.proj, &mut *h)]);
        self.lap(Phase::FeedForwardDown);
        add_residual(
            self.engine.kernels,
            x,
            h,
            Some(&w.post_norm),
            d,
            config.norm_eps,
            1.0,
        );
    }
}

/// Picks the experts for one row of router scores, as llama.cpp does. The
/// scores become probabilities, the best `routing.used` of them (after the
/// choice bias) are chosen, and their probabilities become the weights.
fn choose(
    kernels: &Kernels,
    routing: &Routing,
    bias: Option<&[f32]>,
    scores: &mut [f32],
    taken: &mut Vec<bool>,
    chosen: &mut Vec<(usize, f32)>,
) {
    match routing.gating {
        Gating::Softmax => kernels.softmax(scores),
        Gating::Sigmoid => scores
            .iter_mut()
            .for_each(|score| *score = 1.0 / (1.0 + (-*score).exp())),
        Gating::SoftmaxChosen => {}
    }
    taken.clear();
    taken.resize(scores.len(), false);
    if routing.groups > 1 {
        keep_best_groups(routing, bias, scores, taken);
    }
    let first = chosen.len();
    for _ in 0..routing.used {
        let mut best: Option<(usize, f32)> = None;
        for (expert, &score) in scores.iter().enumerate() {
            let key = score + bias.map_or(0.0, |bias| bias[expert]);
            if !taken[expert] && best.is_none_or(|(_, top)| key > top) {
                best = Some((expert, key));
            }
        }
        let Some((expert, _)) = best else { break };
        taken[expert] = true;
        chosen.push((expert, scores[expert]));
    }
    let weights = &mut chosen[first..];
    if routing.gating == Gating::SoftmaxChosen {
        let top = weights
            .iter()
            .fold(f32::NEG_INFINITY, |top, w| top.max(w.1));
        weights.iter_mut().for_each(|w| w.1 = (w.1 - top).exp());
        let sum: f32 = weights.iter().map(|w| w.1).sum();
        weights.iter_mut().for_each(|w| w.1 /= sum);
    }
    if routing.normalize {
        // The floor is llama.cpp's, so a row of tiny weights stays finite.
        let sum = weights.iter().map(|w| w.1).sum::<f32>().max(6.103_515_6e-5);
        weights.iter_mut().for_each(|w| w.1 /= sum);
    }
    if routing.scale != 1.0 {
        weights.iter_mut().for_each(|w| w.1 *= routing.scale);
    }
}

/// Routing limited to groups of experts (DeepSeek V3), as llama.cpp does
/// it: each group is rated by the sum of its two best scores after the
/// choice bias, and the experts outside the `routing.groups_used` best
/// groups are marked as taken, so that none of them is chosen.
fn keep_best_groups(routing: &Routing, bias: Option<&[f32]>, scores: &[f32], taken: &mut [bool]) {
    let size = routing.count / routing.groups;
    let key = |expert: usize| scores[expert] + bias.map_or(0.0, |bias| bias[expert]);
    let rating = |group: usize| {
        let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for expert in group * size..(group + 1) * size {
            let key = key(expert);
            if key > best {
                (best, second) = (key, best);
            } else if key > second {
                second = key;
            }
        }
        if size > 1 { best + second } else { best }
    };
    // Every group starts out of reach, and the best ones are let back in
    // one at a time. A group's first expert tells whether it is in.
    taken.fill(true);
    for _ in 0..routing.groups_used {
        let mut best: Option<(usize, f32)> = None;
        for group in (0..routing.groups).filter(|&group| taken[group * size]) {
            let rating = rating(group);
            if best.is_none_or(|(_, top)| rating > top) {
                best = Some((group, rating));
            }
        }
        let Some((group, _)) = best else { break };
        taken[group * size..(group + 1) * size].fill(false);
    }
}

/// Adds each expert's bias row to the rows routed to it. `values` holds
/// `width` floats per row, in the grouped order.
fn add_expert_biases(
    map: &Mapping,
    biases: &Matrix,
    groups: &[Range<usize>],
    experts: &[usize],
    width: usize,
    values: &mut [f32],
    row: &mut Vec<f32>,
) {
    row.resize(width, 0.0);
    for (group, &expert) in groups.iter().zip(experts) {
        read_row(map, biases, expert, row);
        for values in values[group.start * width..group.end * width].chunks_exact_mut(width) {
            values
                .iter_mut()
                .zip(row.iter())
                .for_each(|(value, bias)| *value += bias);
        }
    }
}

/// OpenAI's SwiGLU: with `x = min(g, 7)` and `y = clamp(u, -7, 7)`, the
/// result is `x * sigmoid(1.702 * x) * (y + 1)`. It is computed as the SiLU
/// kernel's `silu(a) * b`, with `a = 1.702 * x` and `b = (y + 1) / 1.702`,
/// so that it runs on the fast kernel.
fn swiglu_oai(kernels: &Kernels, gate: &mut [f32], up: &mut [f32]) {
    const ALPHA: f32 = 1.702;
    const LIMIT: f32 = 7.0;
    for (g, u) in gate.iter_mut().zip(up.iter_mut()) {
        *g = ALPHA * g.min(LIMIT);
        *u = (u.clamp(-LIMIT, LIMIT) + 1.0) / ALPHA;
    }
    kernels.silu_mul(gate, up);
}

/// Applies the gate of the feed-forward network: `gate = act(gate) * up`.
/// Large batches are split between the threads.
fn apply_gate(
    kernels: &Kernels,
    pool: &ThreadPool,
    activation: Activation,
    gate: &mut [f32],
    up: &mut [f32],
) {
    /// Elements per parallel piece. Fewer are not worth waking a thread for.
    const GRAIN: usize = 16 * 1024;
    let apply = |gate: &mut [f32], up: &mut [f32]| match activation {
        Activation::Silu => kernels.silu_mul(gate, up),
        Activation::Gelu => kernels.gelu_mul(gate, up),
        Activation::SwigluOai => swiglu_oai(kernels, gate, up),
    };
    let n = gate.len().min(up.len());
    if n <= GRAIN || pool.threads() == 1 {
        return apply(&mut gate[..n], &mut up[..n]);
    }
    let (gates, ups) = (
        SharedOutput(gate.as_mut_ptr()),
        SharedOutput(up.as_mut_ptr()),
    );
    pool.ranges(n, GRAIN, &|range| {
        // SAFETY: the ranges are disjoint and lie inside `gate` and `up`.
        let gate = unsafe { gates.slice(range.start, range.len()) };
        let up = unsafe { ups.slice(range.start, range.len()) };
        apply(gate, up);
    });
}

/// Adds a block's output to the hidden state, normalizing it first when the
/// model has a post-norm, and scaling it when the model scales residuals.
fn add_residual(
    kernels: &Kernels,
    x: &mut [f32],
    update: &mut [f32],
    post_norm: Option<&[f32]>,
    d: usize,
    eps: f32,
    scale: f32,
) {
    for (row, update) in x.chunks_exact_mut(d).zip(update.chunks_exact_mut(d)) {
        if let Some(weight) = post_norm {
            rms_norm_in_place(kernels, update, weight, eps);
        }
        if scale != 1.0 {
            update.iter_mut().for_each(|delta| *delta *= scale);
        }
        for (value, delta) in row.iter_mut().zip(update.iter()) {
            *value += delta;
        }
    }
}

/// The queries of one attention call.
struct Query<'a> {
    /// One row per token, all heads side by side.
    q: &'a [f32],
    /// Context position of the first row.
    first_position: usize,
    /// How many positions a token may look back, itself included. Zero for
    /// no limit.
    window: usize,
    /// Soft cap on the scores, `cap * tanh(score / cap)`. Zero for none.
    softcap: f32,
    /// Each query head's sink score, when the model has them.
    sinks: Option<&'a [f32]>,
}

/// Where a parallel loop writes, shared between its threads.
struct SharedOutput(*mut f32);

// SAFETY: every unit of work writes a slice of its own.
unsafe impl Sync for SharedOutput {}

impl SharedOutput {
    /// The `len` floats starting at `start`.
    ///
    /// # Safety
    ///
    /// The range must lie inside the output, and no other thread may use it
    /// at the same time.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self, start: usize, len: usize) -> &mut [f32] {
        // SAFETY: guaranteed by the caller.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// Computes attention for every token and head of `query`, reading the
/// keys and values of cache layer `layer`.
///
/// For one head: score every visible position by the dot product of the
/// query with that position's key, turn the scores into weights with a
/// softmax, and sum the positions' values with those weights.
///
/// The unit of work is one token and one key/value head. Query heads that
/// share that head are handled together, so its keys and values are read
/// from memory once instead of once per query head.
fn attend(
    engine: &Engine<'_>,
    kv: &KvCache,
    layer: usize,
    spec: &LayerSpec,
    query: &Query<'_>,
    slots: &Slots,
    out: &mut [f32],
) {
    let kernels = engine.kernels;
    let (head_dim, q_dim, n_kv) = (spec.head_dim, spec.q_dim(), spec.n_head_kv);
    let shared = spec.n_head / n_kv;
    let group_dim = shared * head_dim;
    let units = query.q.len() / q_dim * n_kv;
    let output = SharedOutput(out.as_mut_ptr());

    let unit = |index: usize, slot: usize| {
        let (token, kv_head) = (index / n_kv, index % n_kv);
        let position = query.first_position + token;
        let first = if query.window > 0 {
            (position + 1).saturating_sub(query.window)
        } else {
            0
        };
        let end = position + 1;
        let visible = end - first;
        // Each head's scores, with room for its sink at the end.
        let stride = visible + usize::from(query.sinks.is_some());
        // The query heads that read this key/value head sit side by side.
        let group = token * q_dim + kv_head * group_dim;

        // SAFETY: this thread is the only user of its slot, and this unit
        // the only writer of its slice of the output.
        let (scores, scaled) = unsafe { &mut *slots.0[slot].get() };
        let out = unsafe { output.slice(group, group_dim) };

        // Scaling the queries once is cheaper than scaling every score.
        scaled.clear();
        scaled.extend(
            query.q[group..group + group_dim]
                .iter()
                .map(|value| value * spec.attention_scale),
        );
        scores.clear();
        scores.resize(shared * stride, 0.0);

        kv.for_each_run(layer, kv_head, first, end, |run_first, keys, _| {
            kernels.dot_rows_any(
                scaled,
                keys,
                head_dim,
                &mut scores[run_first - first..],
                stride,
            );
        });
        for (head, head_scores) in scores.chunks_exact_mut(stride).enumerate() {
            if query.softcap > 0.0 {
                let cap = query.softcap;
                head_scores[..visible]
                    .iter_mut()
                    .for_each(|score| *score = cap * (*score / cap).tanh());
            }
            // A sink joins the softmax as one more score. Its weight is
            // never used, so the others add up to less than one.
            if let Some(sinks) = query.sinks {
                head_scores[visible] = sinks[kv_head * shared + head];
            }
            kernels.softmax(head_scores);
        }
        out.fill(0.0);
        kv.for_each_run(layer, kv_head, first, end, |run_first, _, values| {
            kernels.mix_rows_any(&scores[run_first - first..], values, head_dim, out, stride);
        });
    };

    // A unit reads about two floats per visible position and head element.
    let visible = query.first_position + query.q.len() / q_dim;
    let visible = if query.window > 0 {
        visible.min(query.window)
    } else {
        visible
    };
    let work_per_unit = 2 * visible * group_dim;
    if units * work_per_unit <= INLINE_ATTENTION_WORK {
        (0..units).for_each(|index| unit(index, 0));
    } else {
        engine.pool.run_slotted(units, &unit);
    }
}

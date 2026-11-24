//! The forward pass: token ids in, next-token logits out.
//!
//! A [`Session`] is one conversation with a model. It owns the key/value
//! cache, the scratch buffers and the thread pool. Evaluating tokens appends
//! them to the context and leaves the logits for the token that follows.
//!
//! Every block of the transformer does the same thing:
//!
//! ```text
//! h      = norm(x)
//! q k v  = h * Wq, h * Wk, h * Wv          (optional per-head norm)
//! q k    = rotate by position (RoPE)
//! a      = softmax(q . K) . V               over every earlier position
//! x     += [norm] (a * Wo)
//! h      = norm(x)
//! x     += [norm] ((act(h * Wgate) * (h * Wup)) * Wdown)
//! ```
//!
//! The architecture differences are all in [`Config`] and in which optional
//! tensors a layer has.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::kernels::{self, Kernels};
use crate::tensor::quant::dequantize_row;
use crate::threads::{ThreadPool, default_threads};

use super::Model;
use super::config::{Activation, Config, Rope};
use super::kv::KvCache;
use super::matmul::{ActScratch, Engine, Target};

/// Context length used when the caller does not choose one.
const DEFAULT_CONTEXT: usize = 4096;

/// Tokens per forward pass while reading a prompt.
const DEFAULT_BATCH: usize = 32;

/// Attention over fewer multiply-adds than this runs on the calling thread.
const INLINE_ATTENTION_WORK: usize = 64 * 1024;

/// The stages of a forward pass that [`Profile`] times separately.
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
    /// Threads to compute with. Zero picks one per performance core.
    pub threads: usize,
    /// Tokens evaluated together while reading a prompt. Zero picks 32.
    /// Larger batches read prompts faster and use more scratch memory.
    pub batch: usize,
    /// The kernel backend. `None` picks the fastest one the CPU supports.
    pub kernels: Option<&'static Kernels>,
}

/// The rotary position tables for the batch being evaluated.
struct RopeTables {
    /// Frequency of each element pair, for the global and the local base.
    freqs: [Vec<f64>; 2],
    /// `(cos, sin)` of every pair, for each token of the batch.
    tables: [Vec<f32>; 2],
    /// True when some layers use the local base.
    has_local: bool,
}

impl RopeTables {
    fn new(config: &Config, factors: Option<&[f32]>) -> RopeTables {
        let pairs = config.head_dim / 2;
        let freqs = |base: f32| -> Vec<f64> {
            (0..pairs)
                .map(|i| {
                    let frequency = (base as f64).powf(-2.0 * i as f64 / config.head_dim as f64);
                    let divisor = factors.map_or(1.0, |f| f[i] as f64);
                    frequency * config.rope_scale as f64 / divisor
                })
                .collect()
        };
        RopeTables {
            freqs: [freqs(config.rope_base), freqs(config.rope_base_local)],
            tables: [Vec::new(), Vec::new()],
            has_local: config.local_layers.iter().any(|&local| local),
        }
    }

    /// Computes the tables for `count` tokens starting at `first_position`.
    ///
    /// Angles are computed in double precision. A position far into a long
    /// context times a high frequency loses most of its fraction in f32.
    fn fill(&mut self, first_position: usize, count: usize) {
        let bases = if self.has_local { 2 } else { 1 };
        for base in 0..bases {
            let freqs = &self.freqs[base];
            let table = &mut self.tables[base];
            table.clear();
            for index in 0..count {
                let position = (first_position + index) as f64;
                for frequency in freqs {
                    let (sin, cos) = (position * frequency).sin_cos();
                    table.push(cos as f32);
                    table.push(sin as f32);
                }
            }
        }
    }

    /// The `(cos, sin)` pairs for token `index` of the batch.
    fn row(&self, local: bool, index: usize, head_dim: usize) -> &[f32] {
        &self.tables[usize::from(local)][index * head_dim..(index + 1) * head_dim]
    }
}

/// Rotates one head's vector by its position.
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

/// Buffers reused by every forward pass. Sizes are for a full batch.
struct Scratch {
    /// The hidden state, one row per token.
    x: Vec<f32>,
    /// A normalized copy of the hidden state, then projection outputs.
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// The attention output before its projection.
    attended: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    acts: ActScratch,
    rope: RopeTables,
    /// Attention scores and a scaled query, one pair per thread slot.
    slots: Slots,
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
    pool: ThreadPool,
    kv: KvCache,
    n_ctx: usize,
    batch: usize,
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
        if threads > 4096 || batch > 1 << 16 {
            return Err(Error::invalid("thread count or batch size out of range"));
        }
        let kernels = config.kernels.unwrap_or_else(kernels::best);
        let pool = ThreadPool::new(threads);

        let (d, q_dim, kv_dim, n_ff) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
        let scratch = Scratch {
            x: vec![0.0; batch * d],
            h: vec![0.0; batch * d],
            q: vec![0.0; batch * q_dim],
            k: vec![0.0; batch * kv_dim],
            v: vec![0.0; batch * kv_dim],
            attended: vec![0.0; batch * q_dim],
            gate: vec![0.0; batch * n_ff],
            up: vec![0.0; batch * n_ff],
            acts: ActScratch::default(),
            rope: RopeTables::new(c, model.weights().rope_factors.as_deref()),
            slots: Slots(
                (0..pool.threads())
                    .map(|_| UnsafeCell::new((Vec::new(), Vec::new())))
                    .collect(),
            ),
        };
        let kv = KvCache::new(c.n_layer, c.n_head_kv, c.head_dim);
        let logits = vec![0.0; c.n_vocab];
        Ok(Session {
            model,
            kernels,
            pool,
            kv,
            n_ctx,
            batch,
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

    /// Number of threads the session computes with.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Name of the kernel backend in use.
    pub fn kernels(&self) -> &'static str {
        self.kernels.name
    }

    /// Bytes of memory held by the key/value cache.
    pub fn cache_bytes(&self) -> usize {
        self.kv.allocated_bytes()
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
    pub fn truncate(&mut self, len: usize) {
        self.kv.truncate(len);
        self.tokens.truncate(len);
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
        let mut batches = tokens.chunks(self.batch).peekable();
        while let Some(batch) = batches.next() {
            // Only the last batch needs logits.
            self.forward(batch, batches.peek().is_none());
        }
        Ok(())
    }

    /// Runs one batch through the model.
    fn forward(&mut self, tokens: &[u32], want_logits: bool) {
        let Session {
            model,
            kernels,
            pool,
            kv,
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
        let (config, weights) = (model.config(), model.weights());
        let map = model.gguf().mapping();
        let engine = Engine { kernels, pool, map };
        let Scratch {
            x,
            h,
            q,
            k,
            v,
            attended,
            gate,
            up,
            acts,
            rope,
            slots,
        } = scratch;

        let count = tokens.len();
        let first_position = kv.len();
        let (d, q_dim, kv_dim, n_ff) =
            (config.n_embd, config.q_dim(), config.kv_dim(), config.n_ff);
        let (head_dim, eps) = (config.head_dim, config.norm_eps);

        // Token embeddings start the hidden state.
        let embd = &weights.token_embd;
        for (row, &token) in x.chunks_exact_mut(d).zip(tokens) {
            let start = embd.offset + token as usize * embd.row_bytes;
            dequantize_row(embd.ty, &map.as_slice()[start..start + embd.row_bytes], row);
            if config.embedding_scale != 1.0 {
                row.iter_mut()
                    .for_each(|value| *value *= config.embedding_scale);
            }
        }

        rope.fill(first_position, count);
        kv.reserve(first_position + count);
        lap(profile, Phase::Embed);

        for (index, layer) in weights.layers.iter().enumerate() {
            // In the last layer only the final token still matters: nothing
            // reads the other rows again. Their keys and values are still
            // stored for later tokens to attend to.
            let active = match (index + 1 == config.n_layer, want_logits) {
                (false, _) => 0..count,
                (true, true) => count - 1..count,
                (true, false) => count..count,
            };
            let n_active = active.len();
            let local = config.local_layers[index];

            // Attention input: the normalized hidden state of every token.
            for (out, row) in h.chunks_exact_mut(d).zip(x.chunks_exact(d)).take(count) {
                rms_norm(kernels, row, &layer.attn_norm, eps, out);
            }
            lap(profile, Phase::Norm);
            let k_out = &mut k[..count * kv_dim];
            let v_out = &mut v[..count * kv_dim];
            let q_out = &mut q[..n_active * q_dim];
            if n_active == count {
                let mut targets = [
                    Target {
                        linear: &layer.q,
                        out: &mut *q_out,
                    },
                    Target {
                        linear: &layer.k,
                        out: &mut *k_out,
                    },
                    Target {
                        linear: &layer.v,
                        out: &mut *v_out,
                    },
                ];
                engine.multiply(acts, &h[..count * d], d, &mut targets);
            } else {
                let mut targets = [
                    Target {
                        linear: &layer.k,
                        out: &mut *k_out,
                    },
                    Target {
                        linear: &layer.v,
                        out: &mut *v_out,
                    },
                ];
                engine.multiply(acts, &h[..count * d], d, &mut targets);
                let mut targets = [Target {
                    linear: &layer.q,
                    out: &mut *q_out,
                }];
                engine.multiply(acts, &h[active.start * d..active.end * d], d, &mut targets);
            }
            lap(profile, Phase::Project);

            // Per-head normalization, then rotation by position.
            for (row, token) in k_out.chunks_exact_mut(kv_dim).zip(0..count) {
                for head in row.chunks_exact_mut(head_dim) {
                    if let Some(weight) = &layer.k_norm {
                        rms_norm_in_place(kernels, head, weight, eps);
                    }
                    rotate(head, rope.row(local, token, head_dim), config.rope);
                }
            }
            for (row, token) in q_out.chunks_exact_mut(q_dim).zip(active.clone()) {
                for head in row.chunks_exact_mut(head_dim) {
                    if let Some(weight) = &layer.q_norm {
                        rms_norm_in_place(kernels, head, weight, eps);
                    }
                    rotate(head, rope.row(local, token, head_dim), config.rope);
                }
            }
            for (token, (key, value)) in k_out
                .chunks_exact(kv_dim)
                .zip(v_out.chunks_exact(kv_dim))
                .enumerate()
            {
                kv.write(index, first_position + token, key, value);
            }
            lap(profile, Phase::Rotate);
            if n_active == 0 {
                continue;
            }

            // Attention, its output projection, and the residual update.
            let window = if local { config.sliding_window } else { 0 };
            let attended = &mut attended[..n_active * q_dim];
            let query = Query {
                q: q_out,
                first_position: first_position + active.start,
                window,
            };
            attend(&engine, kv, index, config, &query, slots, attended);
            lap(profile, Phase::Attention);

            let x_active = &mut x[active.start * d..active.end * d];
            let h_active = &mut h[..n_active * d];
            engine.multiply(
                acts,
                attended,
                q_dim,
                &mut [Target {
                    linear: &layer.attn_out,
                    out: &mut *h_active,
                }],
            );
            lap(profile, Phase::AttentionOut);
            add_residual(
                kernels,
                x_active,
                h_active,
                layer.attn_post_norm.as_deref(),
                d,
                eps,
            );

            // Feed-forward: a gated projection up and a projection down.
            for (out, row) in h_active.chunks_exact_mut(d).zip(x_active.chunks_exact(d)) {
                rms_norm(kernels, row, &layer.ffn_norm, eps, out);
            }
            lap(profile, Phase::Norm);
            let gate = &mut gate[..n_active * n_ff];
            let up = &mut up[..n_active * n_ff];
            let mut targets = [
                Target {
                    linear: &layer.ffn_gate,
                    out: &mut *gate,
                },
                Target {
                    linear: &layer.ffn_up,
                    out: &mut *up,
                },
            ];
            engine.multiply(acts, h_active, d, &mut targets);
            lap(profile, Phase::FeedForwardUp);
            match config.activation {
                Activation::Silu => kernels.silu_mul(gate, up),
                Activation::Gelu => kernels.gelu_mul(gate, up),
            }
            lap(profile, Phase::Activation);
            engine.multiply(
                acts,
                gate,
                n_ff,
                &mut [Target {
                    linear: &layer.ffn_down,
                    out: &mut *h_active,
                }],
            );
            lap(profile, Phase::FeedForwardDown);
            add_residual(
                kernels,
                x_active,
                h_active,
                layer.ffn_post_norm.as_deref(),
                d,
                eps,
            );
            lap(profile, Phase::Norm);
        }
        kv.set_len(first_position + count);
        history.extend_from_slice(tokens);

        if want_logits {
            let last = &x[(count - 1) * d..count * d];
            let normalized = &mut h[..d];
            rms_norm(kernels, last, &weights.output_norm, eps, normalized);
            engine.multiply(
                acts,
                normalized,
                d,
                &mut [Target {
                    linear: &weights.output,
                    out: &mut logits[..],
                }],
            );
            if config.logit_softcap > 0.0 {
                let cap = config.logit_softcap;
                logits
                    .iter_mut()
                    .for_each(|logit| *logit = cap * (*logit / cap).tanh());
            }
            lap(profile, Phase::Logits);
        }
    }
}

/// Adds a block's output to the hidden state, normalizing it first when the
/// model has a post-norm.
fn add_residual(
    kernels: &Kernels,
    x: &mut [f32],
    update: &mut [f32],
    post_norm: Option<&[f32]>,
    d: usize,
    eps: f32,
) {
    for (row, update) in x.chunks_exact_mut(d).zip(update.chunks_exact_mut(d)) {
        if let Some(weight) = post_norm {
            rms_norm_in_place(kernels, update, weight, eps);
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
}

/// Where attention writes, shared between threads.
struct AttendOutput(*mut f32);

// SAFETY: every (token, head) unit writes its own disjoint slice.
unsafe impl Sync for AttendOutput {}

impl AttendOutput {
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

/// Computes attention for every token and head of `query`.
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
    config: &Config,
    query: &Query<'_>,
    slots: &Slots,
    out: &mut [f32],
) {
    let kernels = engine.kernels;
    let (head_dim, q_dim, n_kv) = (config.head_dim, config.q_dim(), config.n_head_kv);
    let shared = config.n_head / n_kv;
    let group_dim = shared * head_dim;
    let units = query.q.len() / q_dim * n_kv;
    let output = AttendOutput(out.as_mut_ptr());

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
                .map(|value| value * config.attention_scale),
        );
        scores.clear();
        scores.resize(shared * visible, 0.0);

        kv.for_each_run(layer, kv_head, first, end, |run_first, keys, _| {
            kernels.dot_rows(
                scaled,
                keys,
                head_dim,
                &mut scores[run_first - first..],
                visible,
            );
        });
        for head_scores in scores.chunks_exact_mut(visible) {
            kernels.softmax(head_scores);
        }
        out.fill(0.0);
        kv.for_each_run(layer, kv_head, first, end, |run_first, _, values| {
            kernels.mix_rows(&scores[run_first - first..], values, head_dim, out, visible);
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

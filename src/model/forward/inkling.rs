//! The forward pass of Inkling 2 and Inkling 2 Mini (Thinking Machines), as
//! llama.cpp computes it. `Inkling` in `config.rs` describes the shape, and
//! `inkling.rs` binds the weights.
//!
//! A block reads more than the residual stream before it. The layers come
//! in groups, and the first layer of each group keeps its input aside and
//! starts a new stream. The inputs of every block's attention and
//! feed-forward network then attend over depth: the kept streams and the
//! current one are normalized, scored against learned queries, and summed
//! with the softmax of those scores (`mix`).
//!
//! Attention has no rotation. Each query head also makes a short relative
//! query, which dots with a learned row of a table for every distance
//! between positions, and that bias joins the scores. Queries and keys are
//! capped in length and values are normalized. Short convolutions run over
//! the keys, the values and both block outputs, each with a residual.
//! Full-attention layers give their key and value convolutions extra taps
//! that the input decides, and grow their queries with the log of the
//! position.
//!
//! Inkling 2 Mini runs a range of layers several times. Each pass after the
//! first starts again from the input of the first, and messages mixed from
//! the previous pass's outputs join its layers' inputs through gates
//! (`loop_gate`). Every pass has cache slots of its own, numbered as
//! llama.cpp numbers them.

use std::ops::Range;

use super::super::config::Config;
use super::super::inkling::{InklingWeights, InputNorms, LoopGate, LoopLayer};
use super::{
    AttentionScratch, ExpertScratch, Kernels, Pass, Phase, Query, RelBias, Target, attend,
    rms_norm_in_place, rms_norm_unit,
};

/// The epsilon of the RMS norms inside the depth attention and the loop
/// gates. llama.cpp fixes it rather than reading it from the file.
const MIX_EPS: f32 = 1e-6;

/// Saved residual streams, with their buffers reused from one batch to the
/// next.
#[derive(Default)]
struct Streams {
    kept: Vec<Vec<f32>>,
    spare: Vec<Vec<f32>>,
}

impl Streams {
    /// Keeps a copy of `rows`.
    fn push(&mut self, rows: &[f32]) {
        let mut stream = self.spare.pop().unwrap_or_default();
        stream.clear();
        stream.extend_from_slice(rows);
        self.kept.push(stream);
    }

    /// Lets go of every stream after the first `len`.
    fn truncate(&mut self, len: usize) {
        while self.kept.len() > len {
            if let Some(stream) = self.kept.pop() {
                self.spare.push(stream);
            }
        }
    }
}

/// Buffers of the loop gates.
#[derive(Default)]
struct GateScratch {
    /// An input and a message, each normalized, side by side.
    both: Vec<f32>,
    bottleneck: Vec<f32>,
    gain: Vec<f32>,
    forget: Vec<f32>,
}

/// Buffers of one depth attention.
#[derive(Default)]
struct MixScratch {
    /// One token's streams, normalized.
    states: Vec<f32>,
    scores: Vec<f32>,
}

/// Buffers of an Inkling forward pass, sized for a full batch. Empty for
/// other models.
#[derive(Default)]
pub(super) struct InklingScratch {
    /// Each token's growth factor for full-attention queries.
    tau: Vec<f32>,
    /// The streams the inputs attend over.
    streams: Streams,
    /// What the messages of a pass mix: the previous pass's streams after
    /// its first, then its output.
    bank: Streams,
    bank_out: Vec<f32>,
    /// The input of the repeated layers, where every pass starts.
    prelude: Vec<f32>,
    /// A block's input, kept for its residual, then its feed-forward input.
    input: Vec<f32>,
    /// The attention's query, key and value inputs, one block of rows each,
    /// normalized in place. The first block then holds the feed-forward
    /// input.
    inputs: Vec<f32>,
    /// The four messages of a later pass: for the query, key and value
    /// inputs, and for the feed-forward input.
    messages: Vec<f32>,
    rel: Vec<f32>,
    sinks: Vec<f32>,
    gate: Vec<f32>,
    k_taps: Vec<f32>,
    v_taps: Vec<f32>,
    /// A slot's convolution inputs: the saved rows, then one per token.
    history: Vec<f32>,
    /// A block output, before and after its convolution.
    out: Vec<f32>,
    /// One sink score of zero per head, which joins every softmax.
    zero_sinks: Vec<f32>,
    loop_gates: GateScratch,
    mix: MixScratch,
}

impl InklingScratch {
    /// Buffers for batches of up to `batch` tokens of `config`'s model.
    pub(super) fn new(config: &Config, batch: usize) -> InklingScratch {
        let Some(inkling) = config.inkling else {
            return InklingScratch::default();
        };
        let d = config.n_embd;
        let rows = batch * d;
        let (q_dim, kv_dim) = (config.max_q_dim(), config.max_kv_dim());
        let most = |f: fn(&super::LayerSpec) -> usize| config.layers.iter().map(f).max();
        let n_head = most(|layer| layer.n_head).unwrap_or(0);
        let n_head_kv = most(|layer| layer.n_head_kv).unwrap_or(0);
        let taps = inkling.conv_taps;
        let channels = inkling.conv_channels(d, kv_dim);
        let looped = |len: usize| {
            if inkling.looped.is_some() {
                vec![0.0; len]
            } else {
                Vec::new()
            }
        };
        let rank = inkling.looped.map_or(0, |lp| lp.rank);
        InklingScratch {
            tau: Vec::with_capacity(batch),
            streams: Streams::default(),
            bank: Streams::default(),
            bank_out: looped(rows),
            prelude: looped(rows),
            input: vec![0.0; rows],
            inputs: vec![0.0; 3 * rows],
            messages: looped(4 * rows),
            rel: vec![0.0; batch * n_head * inkling.rel_len],
            sinks: vec![0.0; batch * n_head],
            gate: vec![0.0; batch * q_dim],
            k_taps: vec![0.0; batch * taps * n_head_kv],
            v_taps: vec![0.0; batch * taps * n_head_kv],
            history: vec![0.0; (taps - 1 + batch) * channels],
            out: vec![0.0; rows],
            zero_sinks: vec![0.0; n_head],
            loop_gates: GateScratch {
                both: looped(2 * rows),
                bottleneck: looped(batch * rank),
                gain: looped(rows),
                forget: looped(rows),
            },
            mix: MixScratch::default(),
        }
    }
}

/// The buffers one Inkling block works in.
pub(super) struct Buffers<'b> {
    pub(super) attention: &'b mut AttentionScratch,
    pub(super) experts: &'b mut ExpertScratch,
    pub(super) inkling: &'b mut InklingScratch,
}

impl Pass<'_> {
    /// Runs every block of an Inkling model over the batch. `x` holds the
    /// token embeddings and ends with the last block's output. When
    /// `want_logits` is set, the last token's row ends as the final mix of
    /// the streams, ready for the output norm.
    pub(super) fn inkling(
        &mut self,
        w: &InklingWeights,
        x: &mut [f32],
        b: &mut Buffers<'_>,
        want_logits: bool,
    ) {
        let config = self.config;
        let inkling = config.inkling.expect("an Inkling model has its settings");
        let kernels = self.engine.kernels;
        let (d, count, eps) = (config.n_embd, self.count, config.norm_eps);
        let rows = count * d;
        let x = &mut x[..rows];

        for row in x.chunks_exact_mut(d) {
            rms_norm_in_place(kernels, row, &w.embd_norm, eps);
        }
        // Full-attention queries grow with the log of the position:
        // `1 + alpha * ln(max(1, (position + 1) / floor))`.
        let s = &mut *b.inkling;
        s.tau.clear();
        for token in 0..count {
            let position = (self.first_position + token + 1) as f32;
            let ratio = (position / inkling.log_floor).max(1.0);
            s.tau.push(1.0 + inkling.log_alpha * ratio.ln());
        }
        s.streams.truncate(0);
        self.lap(Phase::Embed);

        let n_layer = config.n_layer;
        let (start, end, passes) = match inkling.looped {
            Some(lp) => (lp.start, lp.end, lp.passes),
            None => (n_layer, n_layer, 1),
        };
        for index in 0..start {
            self.inkling_block(w, index, index, x, None, b);
        }
        if start < end {
            let s = &mut *b.inkling;
            s.prelude[..rows].copy_from_slice(x);
            let before = s.streams.kept.len();
            for pass in 0..passes {
                let s = &mut *b.inkling;
                if pass > 0 {
                    // The previous pass's streams after its first, which is
                    // the input every pass starts from, and its output.
                    s.bank.truncate(0);
                    for stream in &s.streams.kept[before + 1..] {
                        s.bank.push(stream);
                    }
                    s.bank_out[..rows].copy_from_slice(x);
                    x.copy_from_slice(&s.prelude[..rows]);
                    s.streams.truncate(before);
                }
                for index in start..end {
                    let slot = if pass == 0 {
                        index
                    } else {
                        n_layer + (pass - 1) * (end - start) + (index - start)
                    };
                    let looped = if pass == 0 {
                        None
                    } else {
                        let lp = &w.loop_layers[index - start];
                        let s = &mut *b.inkling;
                        mix(
                            kernels,
                            (&s.bank.kept, &s.bank_out[..rows]),
                            &lp.message_query,
                            d,
                            0..count,
                            &mut s.mix,
                            &mut s.messages[..4 * rows],
                        );
                        Some(lp)
                    };
                    self.inkling_block(w, slot, index, x, looped, b);
                }
            }
        }
        for index in end..n_layer {
            self.inkling_block(w, index, index, x, None, b);
        }

        if want_logits {
            let s = &mut *b.inkling;
            let last = count - 1;
            mix(
                kernels,
                (&s.streams.kept, &*x),
                &w.output_res_query,
                d,
                last..count,
                &mut s.mix,
                &mut s.out[..d],
            );
            x[last * d..].copy_from_slice(&s.out[..d]);
        }
    }

    /// Block `index`, running in cache slot `slot`. `looped` is the layer's
    /// loop weights in a pass after the first, whose messages are in the
    /// scratch.
    fn inkling_block(
        &mut self,
        w: &InklingWeights,
        slot: usize,
        index: usize,
        x: &mut [f32],
        looped: Option<&LoopLayer>,
        b: &mut Buffers<'_>,
    ) {
        let config = self.config;
        let inkling = config.inkling.expect("an Inkling model has its settings");
        let kernels = self.engine.kernels;
        let (layer, spec) = (&w.layers[index], &config.layers[index]);
        let (d, count, eps) = (config.n_embd, self.count, config.norm_eps);
        let rows = count * d;
        let (q_dim, kv_dim, head_dim) = (spec.q_dim(), spec.kv_dim(), spec.head_dim);
        let (n_head, n_kv, rel_len) = (spec.n_head, spec.n_head_kv, inkling.rel_len);
        let local = spec.window > 0;
        let taps = inkling.conv_taps;
        let channels = inkling.conv_channels(d, config.max_kv_dim());
        let Buffers {
            attention: a,
            experts,
            inkling: s,
        } = b;

        s.input[..rows].copy_from_slice(x);

        // The attention inputs. A windowed layer's query, key and value
        // read one input, unless gates make them differ.
        let distinct = !local || looped.is_some();
        match &layer.attn_res_query {
            Some(queries) => {
                let n_queries = queries.len() / d;
                mix(
                    kernels,
                    (&s.streams.kept, &*x),
                    queries,
                    d,
                    0..count,
                    &mut s.mix,
                    &mut s.inputs[..n_queries * rows],
                );
                if distinct && n_queries == 1 {
                    let (first, rest) = s.inputs.split_at_mut(rows);
                    rest[..rows].copy_from_slice(first);
                    rest[rows..2 * rows].copy_from_slice(first);
                }
            }
            None => {
                let copies = if distinct { 3 } else { 1 };
                for input in s.inputs[..copies * rows].chunks_exact_mut(rows) {
                    input.copy_from_slice(x);
                }
            }
        }
        if let Some(lp) = looped {
            for (which, gate) in lp.gates[..3].iter().enumerate() {
                let part = which * rows..(which + 1) * rows;
                self.loop_gate(
                    gate,
                    &mut s.inputs[part.clone()],
                    &s.messages[part],
                    &mut s.loop_gates,
                );
            }
        }
        match &layer.norms {
            InputNorms::Local(gain) => {
                let copies = if distinct { 3 } else { 1 };
                for row in s.inputs[..copies * rows].chunks_exact_mut(d) {
                    rms_norm_in_place(kernels, row, gain, eps);
                }
            }
            InputNorms::Global(norms) => {
                let gains = [&norms.q, &norms.k, &norms.v];
                for (input, gain) in s.inputs[..3 * rows].chunks_exact_mut(rows).zip(gains) {
                    for row in input.chunks_exact_mut(d) {
                        rms_norm_in_place(kernels, row, gain, eps);
                    }
                }
            }
        }
        self.lap(Phase::Norm);

        // Projections. One multiply when the inputs are the same.
        let tap_width = taps * n_kv;
        {
            let q = &mut a.q[..count * q_dim];
            let k = &mut a.k[..count * kv_dim];
            let v = &mut a.v[..count * kv_dim];
            let mut q_targets = vec![
                Target::new(&layer.q, q),
                Target::new(&layer.gate, &mut s.gate[..count * q_dim]),
                Target::new(&layer.rel, &mut s.rel[..count * n_head * rel_len]),
                Target::new(&layer.sink, &mut s.sinks[..count * n_head]),
            ];
            let mut k_targets = vec![Target::new(&layer.k, k)];
            let mut v_targets = vec![Target::new(&layer.v, v)];
            if let InputNorms::Global(norms) = &layer.norms {
                k_targets.push(Target::new(
                    &norms.k_taps,
                    &mut s.k_taps[..count * tap_width],
                ));
                v_targets.push(Target::new(
                    &norms.v_taps,
                    &mut s.v_taps[..count * tap_width],
                ));
            }
            if distinct {
                self.multiply(&s.inputs[..rows], d, &mut q_targets);
                self.multiply(&s.inputs[rows..2 * rows], d, &mut k_targets);
                self.multiply(&s.inputs[2 * rows..3 * rows], d, &mut v_targets);
            } else {
                q_targets.append(&mut k_targets);
                q_targets.append(&mut v_targets);
                self.multiply(&s.inputs[..rows], d, &mut q_targets);
            }
        }
        self.lap(Phase::Project);

        // Keys and values go through their convolutions, with the rows the
        // slot kept from earlier positions.
        let history = &mut s.history[..(taps - 1 + count) * channels];
        self.conv.read(
            slot,
            self.first_position,
            &mut history[..(taps - 1) * channels],
        );
        let at = |token: usize, column: usize| (taps - 1 + token) * channels + column;
        let (k, v) = (&mut a.k[..count * kv_dim], &mut a.v[..count * kv_dim]);
        for (token, (key, value)) in k
            .chunks_exact(kv_dim)
            .zip(v.chunks_exact(kv_dim))
            .enumerate()
        {
            history[at(token, 0)..at(token, kv_dim)].copy_from_slice(key);
            history[at(token, kv_dim)..at(token, 2 * kv_dim)].copy_from_slice(value);
        }
        let (k_taps, v_taps) = match &layer.norms {
            InputNorms::Global(_) => (
                Some((&s.k_taps[..count * tap_width], n_kv)),
                Some((&s.v_taps[..count * tap_width], n_kv)),
            ),
            InputNorms::Local(_) => (None, None),
        };
        let shape = (channels, taps);
        convolve(history, shape, 0, &layer.k_conv, k_taps, k);
        convolve(history, shape, kv_dim, &layer.v_conv, v_taps, v);

        // Lengths are capped, values normalized, and full-attention
        // queries scaled by position. Then the cache.
        let q = &mut a.q[..count * q_dim];
        let rel = &mut s.rel[..count * n_head * rel_len];
        for token in 0..count {
            let tau = s.tau[token];
            let heads = q[token * q_dim..(token + 1) * q_dim].chunks_exact_mut(head_dim);
            for (head, scale) in heads.zip(layer.q_clip.chunks_exact(head_dim)) {
                clip(head, Some(scale), inkling.qk_clip);
                if !local {
                    head.iter_mut().for_each(|value| *value *= tau);
                }
            }
            let width = n_head * rel_len;
            let heads = rel[token * width..(token + 1) * width].chunks_exact_mut(rel_len);
            for (head, scale) in heads.zip(layer.rel_clip.chunks_exact(rel_len)) {
                clip(head, Some(scale), inkling.rel_clip);
                if !local {
                    head.iter_mut().for_each(|value| *value *= tau);
                }
            }
            let key = &mut k[token * kv_dim..(token + 1) * kv_dim];
            for head in key.chunks_exact_mut(head_dim) {
                clip(head, None, inkling.qk_clip);
            }
            let value = &mut v[token * kv_dim..(token + 1) * kv_dim];
            for head in value.chunks_exact_mut(head_dim) {
                rms_norm_unit(kernels, head, eps);
            }
            self.kv.write(slot, self.first_position + token, key, value);
        }
        self.lap(Phase::Rotate);

        let out = &mut a.attended[..count * q_dim];
        let query = Query {
            q: &a.q[..count * q_dim],
            first_position: self.first_position,
            window: spec.window,
            softcap: 0.0,
            sinks: Some(&s.zero_sinks[..n_head]),
            bias: Some(RelBias {
                queries: &s.rel[..count * n_head * rel_len],
                sinks: &s.sinks[..count * n_head],
                table: &layer.rel_bias,
                len: rel_len,
            }),
        };
        attend(&self.engine, self.kv, slot, spec, &query, &a.slots, out);
        self.lap(Phase::Attention);

        // A sigmoid gate on every channel, the output projection, and its
        // convolution.
        for (value, gate) in out.iter_mut().zip(&s.gate[..count * q_dim]) {
            *value *= 1.0 / (1.0 + (-gate).exp());
        }
        self.multiply(
            out,
            q_dim,
            &mut [Target::new(&layer.out, &mut s.out[..rows])],
        );
        let history = &mut s.history[..(taps - 1 + count) * channels];
        let attn_column = 2 * kv_dim;
        for (token, row) in s.out[..rows].chunks_exact(d).enumerate() {
            history[at(token, attn_column)..at(token, attn_column + d)].copy_from_slice(row);
        }
        convolve(
            history,
            shape,
            attn_column,
            &layer.attn_conv,
            None,
            &mut s.out[..rows],
        );
        self.lap(Phase::AttentionOut);

        // The first block of a group keeps its input as a stream and starts
        // a new one. The others add to the stream they read.
        if index % inkling.block == 0 {
            s.streams.push(&s.input[..rows]);
            x.copy_from_slice(&s.out[..rows]);
        } else {
            for ((x, input), out) in x.iter_mut().zip(&s.input[..rows]).zip(&s.out[..rows]) {
                *x = input + out;
            }
        }

        // The feed-forward input attends over depth too, then the experts,
        // then a convolution of their output.
        s.input[..rows].copy_from_slice(x);
        mix(
            kernels,
            (&s.streams.kept, &*x),
            &layer.ffn_res_query,
            d,
            0..count,
            &mut s.mix,
            &mut s.inputs[..rows],
        );
        if let Some(lp) = looped {
            self.loop_gate(
                &lp.gates[3],
                &mut s.inputs[..rows],
                &s.messages[3 * rows..4 * rows],
                &mut s.loop_gates,
            );
        }
        for row in s.inputs[..rows].chunks_exact_mut(d) {
            rms_norm_in_place(kernels, row, &layer.ffn_norm, eps);
        }
        self.lap(Phase::Norm);
        self.experts(
            &layer.experts,
            spec,
            &mut s.inputs[..rows],
            (&mut [], &mut []),
            experts,
        );
        let history = &mut s.history[..(taps - 1 + count) * channels];
        let ffn_column = 2 * kv_dim + d;
        for (token, row) in s.inputs[..rows].chunks_exact(d).enumerate() {
            history[at(token, ffn_column)..at(token, ffn_column + d)].copy_from_slice(row);
        }
        convolve(
            history,
            shape,
            ffn_column,
            &layer.ffn_conv,
            None,
            &mut s.out[..rows],
        );
        for ((x, input), out) in x.iter_mut().zip(&s.input[..rows]).zip(&s.out[..rows]) {
            *x = input + out;
        }
        self.conv.write(slot, self.first_position, count, history);
        self.lap(Phase::Norm);
    }

    /// Joins a message from the previous pass to an input of a repeated
    /// layer: `input - forget * input + gain * message`. The gain and the
    /// forgetting come from both, each normalized, through a narrow
    /// bottleneck and a tanh.
    fn loop_gate(
        &mut self,
        gate: &LoopGate,
        input: &mut [f32],
        message: &[f32],
        s: &mut GateScratch,
    ) {
        let kernels = self.engine.kernels;
        let d = self.config.n_embd;
        let count = input.len() / d;
        let both = &mut s.both[..count * 2 * d];
        for ((pair, x), m) in both
            .chunks_exact_mut(2 * d)
            .zip(input.chunks_exact(d))
            .zip(message.chunks_exact(d))
        {
            let (left, right) = pair.split_at_mut(d);
            left.copy_from_slice(x);
            rms_norm_unit(kernels, left, MIX_EPS);
            right.copy_from_slice(m);
            rms_norm_unit(kernels, right, MIX_EPS);
        }
        let rank = gate.down.rows;
        let bottleneck = &mut s.bottleneck[..count * rank];
        self.multiply(
            both,
            2 * d,
            &mut [Target::new(&gate.down, &mut *bottleneck)],
        );
        let (gain, forget) = (&mut s.gain[..count * d], &mut s.forget[..count * d]);
        self.multiply(
            bottleneck,
            rank,
            &mut [
                Target::new(&gate.gain, &mut *gain),
                Target::new(&gate.forget, &mut *forget),
            ],
        );
        for (((x, m), g), f) in input.iter_mut().zip(message).zip(&*gain).zip(&*forget) {
            let (g, f) = (g.tanh(), f.tanh());
            *x = (*x - f * *x) + g * m;
        }
    }
}

/// Attention over depth, llama.cpp's `mix`. For each token in `rows`, the
/// kept streams and the current one in `sources` are RMS-normalized, each
/// is scored against every query of width `d`, and the softmax of the
/// scores weighs their sum. `out` receives one block of `rows.len()` rows
/// per query.
fn mix(
    kernels: &Kernels,
    sources: (&[Vec<f32>], &[f32]),
    queries: &[f32],
    d: usize,
    rows: Range<usize>,
    s: &mut MixScratch,
    out: &mut [f32],
) {
    let (streams, current) = sources;
    let n_states = streams.len() + 1;
    let n_rows = rows.len();
    s.states.resize(n_states * d, 0.0);
    s.scores.resize(n_states, 0.0);
    let scale = 1.0 / (d as f32).sqrt();
    for (index, token) in rows.enumerate() {
        // The newest stream first, as llama.cpp stacks them, and the
        // current one last.
        let row = token * d..(token + 1) * d;
        let sources = streams
            .iter()
            .rev()
            .map(|stream| &stream[row.clone()])
            .chain(std::iter::once(&current[row.clone()]));
        for (state, source) in s.states.chunks_exact_mut(d).zip(sources) {
            state.copy_from_slice(source);
            rms_norm_unit(kernels, state, MIX_EPS);
        }
        for (q, query) in queries.chunks_exact(d).enumerate() {
            for (score, state) in s.scores.iter_mut().zip(s.states.chunks_exact(d)) {
                *score = kernels.dot(state, query) * scale;
            }
            kernels.softmax(&mut s.scores);
            let at = (q * n_rows + index) * d;
            let target = &mut out[at..at + d];
            target.fill(0.0);
            for (&weight, state) in s.scores.iter().zip(s.states.chunks_exact(d)) {
                for (value, &x) in target.iter_mut().zip(state) {
                    *value += weight * x;
                }
            }
        }
    }
}

/// Scales a head element by element when `scale` is given, then shortens
/// it to length `limit` if it is longer. llama.cpp does this with an L2
/// norm whose epsilon is the limit: `x * limit / max(|x|, limit)`.
fn clip(head: &mut [f32], scale: Option<&[f32]>, limit: f32) {
    if let Some(scale) = scale {
        head.iter_mut()
            .zip(scale)
            .for_each(|(value, s)| *value *= s);
    }
    let sum: f64 = head.iter().map(|&value| (value * value) as f64).sum();
    let inverse = 1.0 / (sum as f32).sqrt().max(limit);
    head.iter_mut()
        .for_each(|value| *value = (*value * inverse) * limit);
}

/// A causal short convolution with a residual, as llama.cpp's `convolve`
/// computes it, over the `width` channels from `column` of `history`.
///
/// `history` holds `taps - 1` saved rows, then one row per token, `channels`
/// floats each (`shape` is `(channels, taps)`). `weights` has `taps` floats
/// per channel, the oldest position first. `extra` adds taps that each token
/// sets for itself, one set per head of channels. The output for a token is
/// its input plus the sums, scaled down by `sqrt(taps + 1)`, or by
/// `sqrt(2 * taps + 1)` with the extra taps.
fn convolve(
    history: &[f32],
    shape: (usize, usize),
    column: usize,
    weights: &[f32],
    extra: Option<(&[f32], usize)>,
    out: &mut [f32],
) {
    let (channels, taps) = shape;
    let width = weights.len() / taps;
    let count = out.len() / width;
    let multiplier = if extra.is_some() { 2 } else { 1 };
    let scale = 1.0 / ((multiplier * taps + 1) as f32).sqrt();
    let input = |row: usize, channel: usize| history[row * channels + column + channel];
    for token in 0..count {
        let target = &mut out[token * width..(token + 1) * width];
        for (channel, (value, taps_of)) in target
            .iter_mut()
            .zip(weights.chunks_exact(taps))
            .enumerate()
        {
            let mut sum = 0.0;
            for (tap, &weight) in taps_of.iter().enumerate() {
                sum += input(token + tap, channel) * weight;
            }
            *value = sum;
        }
        if let Some((extra, heads)) = extra {
            let per_head = width / heads;
            let own = &extra[token * taps * heads..(token + 1) * taps * heads];
            for lag in 0..taps {
                let row = token + taps - 1 - lag;
                for (head, part) in target.chunks_exact_mut(per_head).enumerate() {
                    let tap = own[head * taps + lag];
                    for (offset, value) in part.iter_mut().enumerate() {
                        *value += tap * input(row, head * per_head + offset);
                    }
                }
            }
        }
        for (channel, value) in target.iter_mut().enumerate() {
            *value = (input(token + taps - 1, channel) + *value) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_heads_are_shortened_to_the_limit() {
        let mut head = [3.0, 4.0];
        clip(&mut head, None, 2.5);
        assert!((head[0] - 1.5).abs() < 1e-6 && (head[1] - 2.0).abs() < 1e-6);
        // A short head keeps its length, after its scale.
        let mut head = [0.3, 0.4];
        clip(&mut head, Some(&[2.0, 1.0]), 10.0);
        assert!((head[0] - 0.6).abs() < 1e-6 && (head[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn convolutions_sum_the_last_positions_with_a_residual() {
        // Two channels, three taps: two saved rows, then two tokens.
        let history = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let weights = [0.5, 0.25, 1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0; 4];
        convolve(&history, (2, 3), 0, &weights, None, &mut out);
        let scale = 1.0 / 2.0;
        // Token 0 reads rows 0, 1, 2. Channel 0: 0.5 + 0.5 + 3 = 4, plus 3.
        assert_eq!(out[0], (3.0 + 4.0) * scale);
        assert_eq!(out[1], (30.0 + 30.0) * scale);
        assert_eq!(out[2], (4.0 + (1.0 + 0.75 + 4.0)) * scale);
        // Extra taps per head: one head, lag 0 reads the token itself.
        let extra = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut with_extra = [0.0; 4];
        convolve(
            &history,
            (2, 3),
            0,
            &weights,
            Some((&extra, 1)),
            &mut with_extra,
        );
        let scale = 1.0 / 7f32.sqrt();
        assert!((with_extra[0] - (3.0 + 4.0 + 3.0) * scale).abs() < 1e-5);
    }

    #[test]
    fn depth_attention_weighs_normalized_streams() {
        let kernels = crate::kernels::best();
        // One kept stream and the current one, width 2, one token.
        let streams = vec![vec![2.0, 0.0]];
        let current = [0.0, 3.0];
        // A query that prefers the first element.
        let query = [10.0, 0.0];
        let mut out = [0.0; 2];
        mix(
            kernels,
            (&streams, &current),
            &query,
            2,
            0..1,
            &mut MixScratch::default(),
            &mut out,
        );
        // Normalized, the streams are (sqrt 2, 0) and (0, sqrt 2).
        let root = 2f32.sqrt();
        let (a, b) = ((10.0 * root / root).exp(), 1.0);
        let first = a / (a + b);
        assert!((out[0] - first * root).abs() < 1e-4, "{out:?}");
        assert!((out[1] - (1.0 - first) * root).abs() < 1e-4, "{out:?}");
    }
}

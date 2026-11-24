//! Multi-head latent attention, the attention of DeepSeek V2 and V3.
//!
//! Each position contributes one latent vector `c` and one rotary key
//! `k_r` that every head shares. In the model as trained, head `i` rebuilds
//! its key and value from the latent through its own up-projection:
//!
//! ```text
//! c k_r  = norm(h * Wkv_a), h * Wkv_a    the latent, normalized, and the key
//! key_i  = [c * Wk_i, rotate(k_r)]       unrotated part, then rotary part
//! val_i  = c * Wv_i
//! q_i    = [q_n, rotate(q_r)]            from h * Wq, or a compressed query
//! ```
//!
//! Rebuilding keys and values for every position and head would undo the
//! point of the latent. So the engine moves each query into the latent
//! space instead, which gives the same scores:
//!
//! ```text
//! q_n . (c * Wk_i) = (q_n * Wk_i^T) . c
//! score  = [rotate(q_r), q_n * Wk_i^T] . [rotate(k_r), c]
//! a_i    = softmax(score) . [rotate(k_r), c]    the attended latent
//! out_i  = a_i[latent part] * Wv_i
//! ```
//!
//! The cache then holds `rope_dims + kv_rank` floats per position and
//! layer, the rotary key first and the latent after it, and the latent
//! serves as the value too. Attention over it is attention with a single
//! key/value head, `head_dim` wide, that all query heads share. The rotary
//! key comes first so that shifting the context rotates it where every
//! other layer's rotary part sits.
//!
//! This is a part of the forward pass in its own file. As a child module of
//! `forward.rs` it extends [`Pass`] with what that file's attention blocks
//! use.

use std::ops::Range;

use super::{
    AttentionScratch, INLINE_ATTENTION_WORK, LayerRun, Pass, Phase, SharedOutput, Slots, read_row,
    rms_norm_in_place, rotate,
};
use crate::model::Model;
use crate::model::config::LayerSpec;
use crate::model::kv::KvCache;
use crate::model::matmul::{Engine, Target};
use crate::model::weights::{Absorb, HeadMatrix, Linear, MixerWeights, MlaWeights};

/// Buffers of the latent attention blocks, beyond those every attention
/// block has. Empty for other models.
///
/// The attention buffers serve here too, each sized for a full batch: the
/// key buffer holds each token's latent and rotary key, the query buffer
/// the queries in the latent space, and the output buffer the attended
/// latents. A latent attention layer's `head_dim` makes them wide enough.
#[derive(Default)]
pub(super) struct MlaScratch {
    /// The compressed queries, for models that compress them.
    compressed: Vec<f32>,
    /// Each active token's query heads as projected: the unrotated part,
    /// then the rotary part.
    q: Vec<f32>,
    heads: HeadBuffers,
    /// Each active token's values, all heads side by side.
    values: Vec<f32>,
}

/// Buffers of a multiply that applies each head's own matrix to that head's
/// inputs.
#[derive(Default)]
struct HeadBuffers {
    /// The inputs, all of one head's tokens before the next head's, and
    /// the outputs in the same order.
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    /// One linear map per head.
    linears: Vec<Linear>,
}

impl MlaScratch {
    /// Buffers for batches of up to `batch` tokens of `model`.
    pub(super) fn new(model: &Model, batch: usize) -> MlaScratch {
        let config = model.config();
        let Some(shape) = config.mla else {
            return MlaScratch::default();
        };
        let n_head = config
            .layers
            .iter()
            .map(|layer| layer.n_head)
            .max()
            .unwrap_or(0);
        let rotary = config
            .layers
            .iter()
            .map(|layer| layer.rope_dims)
            .max()
            .unwrap_or(0);
        let q_rank = model
            .weights()
            .layers
            .iter()
            .filter_map(|layer| match &layer.mixer {
                MixerWeights::Mla(weights) => weights.q_a.as_ref().map(|(q_a, _)| q_a.rows),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let widest = shape.kv_rank.max(shape.nope_dims).max(shape.value_dims);
        MlaScratch {
            compressed: vec![0.0; batch * q_rank],
            q: vec![0.0; batch * n_head * (shape.nope_dims + rotary)],
            heads: HeadBuffers {
                inputs: vec![0.0; batch * n_head * widest],
                outputs: vec![0.0; batch * n_head * widest],
                linears: Vec::with_capacity(n_head),
            },
            values: vec![0.0; batch * n_head * shape.value_dims],
        }
    }
}

/// The widths of one latent attention layer.
#[derive(Clone, Copy)]
struct Widths {
    n_head: usize,
    /// The unrotated part of a query or key head, and the rotary part.
    nope: usize,
    rotary: usize,
    /// The latent.
    rank: usize,
    /// A value head.
    value: usize,
    /// A cache row, and a query head in the latent space: `rotary + rank`.
    row: usize,
}

impl Widths {
    /// A projected query head.
    fn q_head(&self) -> usize {
        self.nope + self.rotary
    }
}

impl Pass<'_> {
    /// A latent attention block. Reads the normalized inputs in `h` and
    /// leaves the block's output for the active rows at the start of `h`.
    pub(super) fn latent_attention(
        &mut self,
        run: &LayerRun,
        w: &MlaWeights,
        h: &mut [f32],
        a: &mut AttentionScratch,
        s: &mut MlaScratch,
    ) {
        let (config, rope, spec) = (self.config, self.rope, run.spec);
        let shape = config.mla.expect("a latent attention block has its shape");
        let kernels = self.engine.kernels;
        let (d, eps) = (config.n_embd, config.norm_eps);
        let (count, n_active) = (self.count, run.active.len());
        let at = Widths {
            n_head: spec.n_head,
            nope: shape.nope_dims,
            rotary: spec.rope_dims,
            rank: shape.kv_rank,
            value: shape.value_dims,
            row: spec.head_dim,
        };
        let q_width = at.n_head * at.q_head();

        // The latent and rotary key of every row the cache needs, the
        // queries of the active rows only.
        let projected = &mut a.k[..count * at.row];
        let q = &mut s.q[..n_active * q_width];
        let kept = vec![Target::new(&w.kv_a, &mut *projected)];
        match &w.q_a {
            None => self.project(h, run, kept, vec![Target::new(&w.q, &mut *q)]),
            Some((q_a, norm)) => {
                let compressed = &mut s.compressed[..n_active * q_a.rows];
                self.project(h, run, kept, vec![Target::new(q_a, &mut *compressed)]);
                for row in compressed.chunks_exact_mut(q_a.rows) {
                    rms_norm_in_place(kernels, row, norm, eps);
                }
                self.multiply(compressed, q_a.rows, &mut [Target::new(&w.q, &mut *q)]);
            }
        }
        self.lap(Phase::Project);

        // Cache rows: the rotated key first, then the normalized latent.
        for (token, row) in projected.chunks_exact_mut(at.row).enumerate() {
            row.rotate_right(at.rotary);
            let (key, latent) = row.split_at_mut(at.rotary);
            rotate(key, rope.row(run.index, token), config.rope);
            rms_norm_in_place(kernels, latent, &w.kv_a_norm, eps);
            self.kv
                .write(run.index, self.first_position + token, row, &[]);
        }
        if n_active == 0 {
            self.lap(Phase::Rotate);
            return;
        }
        // Each query head starts with its rotated part.
        let queries = &mut a.q[..n_active * at.n_head * at.row];
        let rows = queries
            .chunks_exact_mut(at.n_head * at.row)
            .zip(q.chunks_exact_mut(q_width));
        for ((query_row, q_row), token) in rows.zip(run.active.clone()) {
            let angles = rope.row(run.index, token);
            let heads = query_row
                .chunks_exact_mut(at.row)
                .zip(q_row.chunks_exact_mut(at.q_head()));
            for (query, head) in heads {
                let rotated = &mut head[at.nope..];
                rotate(rotated, angles, config.rope);
                query[..at.rotary].copy_from_slice(rotated);
            }
        }
        self.lap(Phase::Rotate);

        // Then comes its unrotated part, moved into the latent space.
        match &w.absorb {
            Absorb::Rows(matrices) => self.absorb_rows(matrices, at, q, queries, &mut s.heads),
            Absorb::Columns(matrices) => self.absorb_columns(matrices, at, q, queries, &a.slots),
        }
        self.lap(Phase::Project);

        let attended = &mut a.attended[..n_active * at.n_head * at.row];
        let first_position = self.first_position + run.active.start;
        attend_latent(
            &self.engine,
            self.kv,
            run.index,
            spec,
            queries,
            first_position,
            &a.slots,
            attended,
        );
        self.lap(Phase::Attention);

        // Each head's value is its attended latent through the value half of
        // its up-projection. The latent follows the rotary part.
        let heads = &mut s.heads;
        for (token, row) in attended.chunks_exact(at.n_head * at.row).enumerate() {
            for (head, attended) in row.chunks_exact(at.row).enumerate() {
                let place = (head * n_active + token) * at.rank;
                heads.inputs[place..place + at.rank].copy_from_slice(&attended[at.rotary..]);
            }
        }
        let outputs = per_head_multiply(self, &w.values, at.n_head, n_active, at.rank, heads);
        let values = &mut s.values[..n_active * at.n_head * at.value];
        gather_by_token(outputs, values, at.n_head, n_active, at.value, at.value, 0);
        self.multiply(
            values,
            at.n_head * at.value,
            &mut [Target::new(&w.out, &mut h[..n_active * d])],
        );
        self.lap(Phase::AttentionOut);
    }

    /// Moves the unrotated part of every query head into the latent space
    /// with `attn_k_b`, one multiply for all heads. Writes the results after
    /// the rotated parts in `queries`.
    fn absorb_rows(
        &mut self,
        matrices: &HeadMatrix,
        at: Widths,
        q: &[f32],
        queries: &mut [f32],
        heads: &mut HeadBuffers,
    ) {
        let n_active = q.len() / (at.n_head * at.q_head());
        for (token, q_row) in q.chunks_exact(at.n_head * at.q_head()).enumerate() {
            for (head, part) in q_row.chunks_exact(at.q_head()).enumerate() {
                let place = (head * n_active + token) * at.nope;
                heads.inputs[place..place + at.nope].copy_from_slice(&part[..at.nope]);
            }
        }
        let outputs = per_head_multiply(self, matrices, at.n_head, n_active, at.nope, heads);
        gather_by_token(
            outputs, queries, at.n_head, n_active, at.rank, at.row, at.rotary,
        );
    }

    /// Moves the unrotated part of every query head into the latent space
    /// with the key half of an older file's `attn_kv_b`. Its rows are the
    /// columns of the map, so each row is decoded and added to the result,
    /// scaled by the query element it stands for. A row is decoded once for
    /// all the batch's tokens. Heads are spread over the threads.
    fn absorb_columns(
        &self,
        matrices: &HeadMatrix,
        at: Widths,
        q: &[f32],
        queries: &mut [f32],
        slots: &Slots,
    ) {
        let engine = &self.engine;
        let kernels = engine.kernels;
        let n_active = q.len() / (at.n_head * at.q_head());
        let output = SharedOutput(queries.as_mut_ptr());
        let head_work = |head: usize, slot: usize| {
            // SAFETY: this thread is the only user of its slot.
            let (_, decoded) = unsafe { &mut *slots.0[slot].get() };
            decoded.resize(at.rank, 0.0);
            // SAFETY: each head writes only its own latent parts, which no
            // other head's slices overlap.
            let latent = |token: usize| unsafe {
                output.slice((token * at.n_head + head) * at.row + at.rotary, at.rank)
            };
            (0..n_active).for_each(|token| latent(token).fill(0.0));
            let linear = matrices.head(head);
            for element in 0..at.nope {
                read_row(
                    engine.map,
                    &linear.matrix,
                    linear.first_row + element,
                    decoded,
                );
                for token in 0..n_active {
                    let weight = q[(token * at.n_head + head) * at.q_head() + element];
                    kernels.axpy(weight, decoded, latent(token));
                }
            }
        };
        // Decoding dominates: one row of `rank` per element of each head.
        if at.n_head * at.nope * at.rank * (n_active + 4) <= INLINE_ATTENTION_WORK {
            (0..at.n_head).for_each(|head| head_work(head, 0));
        } else {
            engine.pool.run_slotted(at.n_head, &head_work);
        }
    }
}

/// Applies each head's matrix of `matrices` to that head's inputs, in one
/// multiply for all heads. The inputs are `n_active` vectors of `cols` per
/// head, head after head, at the start of `heads.inputs`. Returns the
/// outputs, in the same order.
fn per_head_multiply<'h>(
    pass: &mut Pass<'_>,
    matrices: &HeadMatrix,
    n_head: usize,
    n_active: usize,
    cols: usize,
    heads: &'h mut HeadBuffers,
) -> &'h [f32] {
    let HeadBuffers {
        inputs,
        outputs,
        linears,
    } = heads;
    linears.clear();
    linears.extend((0..n_head).map(|head| matrices.head(head)));
    let outputs = &mut outputs[..n_head * n_active * matrices.rows];
    let mut targets = Vec::with_capacity(n_head);
    let mut rest = &mut *outputs;
    for (head, linear) in linears.iter().enumerate() {
        let (part, tail) = std::mem::take(&mut rest).split_at_mut(n_active * linear.rows);
        rest = tail;
        targets.push(Target {
            linear,
            out: part,
            inputs: Some(head * n_active..(head + 1) * n_active),
        });
    }
    pass.multiply(&inputs[..n_head * n_active * cols], cols, &mut targets);
    drop(targets);
    outputs
}

/// Copies head-major results, `width` floats per head and token, into
/// token-major rows of `n_head` blocks of `stride` floats, `skip` floats
/// into each block.
fn gather_by_token(
    head_major: &[f32],
    rows: &mut [f32],
    n_head: usize,
    n_active: usize,
    width: usize,
    stride: usize,
    skip: usize,
) {
    for head in 0..n_head {
        for token in 0..n_active {
            let from = (head * n_active + token) * width;
            let to = (token * n_head + head) * stride + skip;
            rows[to..to + width].copy_from_slice(&head_major[from..from + width]);
        }
    }
}

/// Attention over the latent cache of `layer` for every token and head of
/// `queries`, which hold one query per head and token in the latent space.
/// Writes the attended rows to `out` in the same layout.
///
/// Every head reads the same cache rows, so a unit of work takes a token
/// and a few heads: the score kernel takes up to four queries in one pass
/// over the rows. When there are few tokens, as in generation, units get
/// fewer heads so that every thread has work. The models with latent
/// attention have no window and no cap on scores.
#[allow(clippy::too_many_arguments)]
fn attend_latent(
    engine: &Engine<'_>,
    kv: &KvCache,
    layer: usize,
    spec: &LayerSpec,
    queries: &[f32],
    first_position: usize,
    slots: &Slots,
    out: &mut [f32],
) {
    let kernels = engine.kernels;
    let (n_head, width) = (spec.n_head, spec.head_dim);
    let tokens = queries.len() / (n_head * width);
    let per_unit = (tokens * n_head / engine.pool.threads()).clamp(1, 4);
    let chunks = n_head.div_ceil(per_unit);
    let output = SharedOutput(out.as_mut_ptr());

    let unit = |index: usize, slot: usize| {
        let (token, chunk) = (index / chunks, index % chunks);
        let heads: Range<usize> = chunk * per_unit..((chunk + 1) * per_unit).min(n_head);
        let visible = first_position + token + 1;
        let start = (token * n_head + heads.start) * width;
        let len = heads.len() * width;

        // SAFETY: this thread is the only user of its slot, and this unit
        // the only writer of its slice of the output.
        let (scores, scaled) = unsafe { &mut *slots.0[slot].get() };
        let out = unsafe { output.slice(start, len) };

        scaled.clear();
        scaled.extend(
            queries[start..start + len]
                .iter()
                .map(|value| value * spec.attention_scale),
        );
        scores.clear();
        scores.resize(heads.len() * visible, 0.0);
        kv.for_each_run(layer, 0, 0, visible, |run_first, rows, _| {
            kernels.dot_rows(scaled, rows, width, &mut scores[run_first..], visible);
        });
        for head_scores in scores.chunks_exact_mut(visible) {
            kernels.softmax(head_scores);
        }
        // The rows serve as values. Their rotary part is mixed as well and
        // left unused.
        out.fill(0.0);
        kv.for_each_run(layer, 0, 0, visible, |run_first, rows, _| {
            kernels.mix_rows(&scores[run_first..], rows, width, out, visible);
        });
    };

    let units = tokens * chunks;
    let work_per_unit = 2 * (first_position + tokens) * per_unit * width;
    if units * work_per_unit <= INLINE_ATTENTION_WORK {
        (0..units).for_each(|index| unit(index, 0));
    } else {
        engine.pool.run_slotted(units, &unit);
    }
}

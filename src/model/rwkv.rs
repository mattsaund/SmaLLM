//! RWKV blocks: a recurrent state per head in the place of attention.
//!
//! An RWKV block has the shape of a transformer block. A part that mixes
//! positions (time mixing) and a feed-forward part (channel mixing) each
//! read the hidden state through a layer norm and add to it. Both read the
//! previous token as well: every input moves part of the way from the
//! current token's normalized input toward the previous token's, by a
//! learned amount per channel. This is called token shift.
//!
//! The time mixing keeps a square matrix `S` per head, `head_size` wide,
//! that every token updates. For one head and one token, with receptance
//! `r`, key `k`, value `v` and a decay `w` between 0 and 1 per channel:
//!
//! ```text
//! RWKV-6   y = S^T r + (r . (u * k)) v      u: a learned bonus for the current token
//!          S = diag(w) S + k v^T
//!
//! RWKV-7   S = S diag(w) - (S kk) (kk * a)^T + v k^T
//!          y = S r
//! ```
//!
//! RWKV-7 removes from the state what it held for the normalized key `kk`,
//! at a learned rate `a`, before it writes the new value. Its values also
//! take part of the first block's value. Each head's output `y` is then
//! normalized, gated and projected back to the hidden state.
//!
//! A block keeps nothing else: the previous token's two normalized inputs
//! and the heads' matrices. So memory does not grow with the context. The
//! state lives in `recurrent.rs`, which saves and restores it.
//!
//! The formulas follow llama.cpp's graphs and its `rwkv_wkv6` and
//! `rwkv_wkv7` operations. The projections go through the parallel matrix
//! engine, batched over a prompt's tokens. The state update runs head by
//! head on the thread pool, one token after another. The steps in between
//! treat each token on its own, so a prompt's tokens share them out over
//! the pool as well.

use std::ops::Range;
use std::slice;

use crate::kernels::exp_approx;
use crate::threads::ThreadPool;

use super::config::Config;
use super::forward::{LayerRun, Pass, Phase};
use super::matmul::Target;
use super::weights::{
    FeedForward, Layer, Linear, MixerWeights, TimeMix, TimeMix6, TimeMix7, TimeMixVersion, Weights,
};

/// RWKV-7 decays the state by `exp(-DECAY_SCALE * sigmoid(w))`, with this
/// value of `exp(-0.5)` as llama.cpp writes it.
const DECAY_SCALE: f32 = 0.606531;

/// Epsilon of the norm of each head's output, as llama.cpp sets it.
const HEAD_NORM_EPS: f32 = 64e-5;

/// Epsilon of the normalization of RWKV-7's removal key.
const KEY_NORM_EPS: f32 = 1e-12;

/// A state update with fewer multiply-adds than this runs on the calling
/// thread.
const INLINE_STATE_WORK: usize = 64 * 1024;

/// Floats of state an RWKV block keeps: the previous token's normalized
/// inputs to the time mixing and to the channel mixing, then the matrix of
/// every head.
pub fn state_size(config: &Config) -> usize {
    let head_size = config.rwkv.map_or(0, |rwkv| rwkv.head_size);
    2 * config.n_embd + config.n_embd * head_size
}

/// Buffers of the RWKV blocks, sized for a full batch. Empty for other
/// models.
#[derive(Default)]
pub(super) struct RwkvScratch {
    /// A block's normalized input, then the heads' output, one row per
    /// token.
    normed: Vec<f32>,
    /// The previous token's normalized input minus the current one, one row
    /// per token.
    shift: Vec<f32>,
    /// The inputs of the projections, one part per input, each a row per
    /// token. Later the outputs of the low-rank maps.
    parts: Vec<f32>,
    /// Receptance, key, value and one more vector, a part each: the gate
    /// (RWKV-6) or the normalized key (RWKV-7).
    rkv: Vec<f32>,
    /// The narrow side of the low-rank maps.
    low: Vec<f32>,
    /// RWKV-7: the first block's values, which later blocks mix in.
    first_value: Vec<f32>,
}

impl RwkvScratch {
    /// Buffers for batches of up to `batch` tokens.
    pub fn new(config: &Config, weights: &Weights, batch: usize) -> RwkvScratch {
        if config.rwkv.is_none() {
            return RwkvScratch::default();
        }
        let rows = batch * config.n_embd;
        let mut low = 0;
        let mut first_value = 0;
        for layer in &weights.layers {
            let MixerWeights::Rwkv(time_mix) = &layer.mixer else {
                continue;
            };
            let width = match &time_mix.version {
                // The map's output in token order, then sorted by part,
                // then the decay's narrow side.
                TimeMixVersion::V6(parts) => 2 * parts.mix_down.rows + parts.decay_down.rows,
                TimeMixVersion::V7(parts) => {
                    first_value = rows;
                    let value = parts.value_mix.as_ref().map_or(0, |(_, v1, _)| v1.rows);
                    parts.w1.rows + parts.a1.rows + parts.g1.rows + value
                }
            };
            low = low.max(width);
        }
        RwkvScratch {
            normed: vec![0.0; rows],
            shift: vec![0.0; rows],
            parts: vec![0.0; 6 * rows],
            rkv: vec![0.0; 4 * rows],
            low: vec![0.0; batch * low],
            first_value: vec![0.0; first_value],
        }
    }
}

impl Pass<'_> {
    /// One RWKV block. Every row of `x` goes through the time mixing, which
    /// the state needs, and the active rows through the channel mixing.
    /// `h` and `ffn` are scratch, `ffn` as wide as the channel mixing.
    pub(super) fn rwkv_block(
        &mut self,
        run: &LayerRun,
        layer: &Layer,
        x: &mut [f32],
        h: &mut [f32],
        s: &mut RwkvScratch,
        ffn: &mut [f32],
    ) {
        let MixerWeights::Rwkv(time_mix) = &layer.mixer else {
            unreachable!("only RWKV blocks take this path");
        };
        let config = self.config;
        let (d, eps, count) = (config.n_embd, config.norm_eps, self.count);
        let rows = count * d;
        let bias = layer
            .attn_norm_bias
            .as_deref()
            .expect("RWKV blocks are bound with layer norms");
        let state = self.recurrent.block(run.index);
        norm_and_shift(
            self.engine.pool,
            &x[..rows],
            (&layer.attn_norm, bias),
            eps,
            &mut s.normed[..rows],
            &mut state[..d],
            &mut s.shift[..rows],
        );
        self.lap(Phase::Norm);

        match &time_mix.version {
            TimeMixVersion::V6(parts) => self.time_mix6(run, time_mix, parts, h, s),
            TimeMixVersion::V7(parts) => self.time_mix7(run, time_mix, parts, h, s),
        }
        let update = &h[..rows];
        per_token(
            self.engine.pool,
            count,
            [(&mut x[..rows], d)],
            &|token, [x]| {
                add(x, &update[token * d..(token + 1) * d]);
            },
        );
        self.channel_mix(run, layer, x, h, s, ffn);

        // RWKV-6 halves the hidden state every few blocks.
        let every = config.rwkv.map_or(0, |rwkv| rwkv.rescale_every);
        if every > 0 && (run.index + 1) % every == 0 {
            let active = &mut x[run.active.start * d..run.active.end * d];
            active.iter_mut().for_each(|value| *value *= 0.5);
        }
        self.lap(Phase::Norm);
    }

    /// RWKV-7's time mixing of every row. Reads the normalized rows and
    /// their token shift from the scratch, and leaves the output in `h`.
    fn time_mix7(
        &mut self,
        run: &LayerRun,
        time_mix: &TimeMix,
        w: &TimeMix7,
        h: &mut [f32],
        s: &mut RwkvScratch,
    ) {
        let config = self.config;
        let (d, count) = (config.n_embd, self.count);
        let rows = count * d;
        let RwkvScratch {
            normed,
            shift,
            parts,
            rkv,
            low,
            first_value,
        } = s;
        let (normed, shift) = (&mut normed[..rows], &shift[..rows]);

        // The six inputs: receptance, decay, key, value, learning rate and
        // gate, each moved toward the previous token by its own amounts.
        let pool = self.engine.pool;
        let parts = &mut parts[..6 * rows];
        let mut each_part = parts.chunks_exact_mut(rows);
        let part_rows: [_; 6] = std::array::from_fn(|_| (each_part.next().unwrap(), d));
        per_token(pool, count, part_rows, &|token, rows| {
            let span = token * d..(token + 1) * d;
            for (row, lerp) in rows.into_iter().zip(w.lerp.chunks_exact(d)) {
                lerp_row(row, &normed[span.clone()], &shift[span.clone()], lerp);
            }
        });

        // Every map in one multiply, each reading its own input.
        let value_rank = w.value_mix.as_ref().map_or(0, |(_, v1, _)| v1.rows);
        let (low_w, rest) = low.split_at_mut(count * w.w1.rows);
        let (low_a, rest) = rest.split_at_mut(count * w.a1.rows);
        let (low_g, rest) = rest.split_at_mut(count * w.g1.rows);
        let low_v = &mut rest[..count * value_rank];
        let (r, rest) = rkv.split_at_mut(rows);
        let (k, rest) = rest.split_at_mut(rows);
        let (v, rest) = rest.split_at_mut(rows);
        let kk = &mut rest[..rows];
        let part = |index: usize| Some(index * count..(index + 1) * count);
        let mut targets = vec![
            target(&time_mix.receptance, &mut *r, part(0)),
            target(&w.w1, &mut *low_w, part(1)),
            target(&time_mix.key, &mut *k, part(2)),
            target(&time_mix.value, &mut *v, part(3)),
            target(&w.a1, &mut *low_a, part(4)),
            target(&w.g1, &mut *low_g, part(5)),
        ];
        if let Some((_, v1, _)) = &w.value_mix {
            targets.push(target(v1, &mut *low_v, part(3)));
        }
        self.multiply(parts, d, &mut targets);
        drop(targets);

        // Back up from the low ranks, into the parts the inputs used.
        low_w.iter_mut().for_each(|value| *value = value.tanh());
        low_g.iter_mut().for_each(|value| *value = sigmoid(*value));
        let (decay, rest) = parts.split_at_mut(rows);
        let (rate, rest) = rest.split_at_mut(rows);
        let (gate, rest) = rest.split_at_mut(rows);
        let value_mix = &mut rest[..rows];
        self.multiply(low_w, w.w1.rows, &mut [Target::new(&w.w2, &mut *decay)]);
        self.multiply(low_a, w.a1.rows, &mut [Target::new(&w.a2, &mut *rate)]);
        self.multiply(low_g, w.g1.rows, &mut [Target::new(&w.g2, &mut *gate)]);
        if let Some((_, _, v2)) = &w.value_mix {
            self.multiply(low_v, value_rank, &mut [Target::new(v2, &mut *value_mix)]);
        }
        self.lap(Phase::Project);

        // Per channel: the decay, the learning rate, the first block's share
        // of the value, the normalized key to remove, and the key to write.
        // Row by row, so that each loop runs over slices of one length and
        // vectorizes.
        let head_size = config.rwkv.map_or(d, |rwkv| rwkv.head_size);
        let value_mix = &*value_mix;
        let buffers = [
            (&mut *decay, d),
            (&mut *rate, d),
            (&mut *v, d),
            (&mut first_value[..rows], d),
            (&mut *k, d),
            (&mut *kk, d),
        ];
        per_token(pool, count, buffers, &|token, rows| {
            let [decay, rate, value, first, key, kk] = rows;
            let mix = &value_mix[token * d..(token + 1) * d];
            for (decay, base) in decay.iter_mut().zip(&w.w0) {
                *decay = exp_approx(-DECAY_SCALE * sigmoid(base + *decay));
            }
            for (rate, base) in rate.iter_mut().zip(&w.a0) {
                *rate = sigmoid(base + *rate);
            }
            match &w.value_mix {
                None => first.copy_from_slice(value),
                Some((v0, _, _)) => {
                    let mixes = v0.iter().zip(mix);
                    for ((value, first), (base, mix)) in value.iter_mut().zip(&*first).zip(mixes) {
                        *value += (first - *value) * sigmoid(base + mix);
                    }
                }
            }
            for ((kk, key), scale) in kk.iter_mut().zip(&*key).zip(&w.k_k) {
                *kk = key * scale;
            }
            for head in kk.chunks_exact_mut(head_size) {
                let norm = sum_of_squares(head).sqrt().max(KEY_NORM_EPS);
                head.iter_mut().for_each(|value| *value /= norm);
            }
            // The key written moves toward zero where the rate is low, and
            // the removal is scaled by the rate: `b = kk * a`, kept in
            // `rate`.
            let scales = kk.iter().zip(&w.k_a);
            for ((key, rate), (kk, scale)) in key.iter_mut().zip(rate.iter_mut()).zip(scales) {
                *key *= 1.0 + (*rate - 1.0) * scale;
                *rate *= kk;
            }
        });
        self.lap(Phase::Rotate);

        let state = &mut self.recurrent.block(run.index)[2 * d..];
        let inputs = Wkv7 {
            r,
            w: decay,
            k,
            v,
            kk,
            b: rate,
        };
        wkv7(self.engine.pool, state, head_size, d, &inputs, normed);
        self.lap(Phase::Attention);

        // Each head's output: normalized, plus a bonus for the current
        // token's value, then gated.
        let (norm_weight, norm_bias) = &time_mix.head_norm;
        let (r, k, v, gate) = (&*r, &*k, &*v, &*gate);
        per_token(pool, count, [(&mut *normed, d)], &|token, [y]| {
            let span = token * d..(token + 1) * d;
            head_norm(y, head_size, norm_weight, norm_bias);
            let heads = y
                .chunks_exact_mut(head_size)
                .zip(r[span.clone()].chunks_exact(head_size))
                .zip(k[span.clone()].chunks_exact(head_size))
                .zip(
                    v[span.clone()]
                        .chunks_exact(head_size)
                        .zip(w.r_k.chunks_exact(head_size)),
                );
            for (((y, r), k), (v, r_k)) in heads {
                let bonus: f32 = r.iter().zip(k).zip(r_k).map(|((r, k), s)| r * k * s).sum();
                for (y, v) in y.iter_mut().zip(v) {
                    *y += bonus * v;
                }
            }
            for (y, gate) in y.iter_mut().zip(&gate[span]) {
                *y *= gate;
            }
        });
        self.multiply(
            normed,
            d,
            &mut [Target::new(&time_mix.output, &mut h[..rows])],
        );
        self.lap(Phase::AttentionOut);
    }

    /// RWKV-6's time mixing of every row. Reads the normalized rows and
    /// their token shift from the scratch, and leaves the output in `h`.
    fn time_mix6(
        &mut self,
        run: &LayerRun,
        time_mix: &TimeMix,
        w: &TimeMix6,
        h: &mut [f32],
        s: &mut RwkvScratch,
    ) {
        let config = self.config;
        let (d, count) = (config.n_embd, self.count);
        let rows = count * d;
        let RwkvScratch {
            normed,
            shift,
            parts,
            rkv,
            low,
            ..
        } = s;
        let (normed, shift) = (&mut normed[..rows], &shift[..rows]);
        let parts = &mut parts[..5 * rows];

        // How far each input moves depends on the token: a low-rank map of
        // one shifted input gives five corrections to the learned amounts.
        let pool = self.engine.pool;
        let wide = w.mix_down.rows;
        let rank = wide / 5;
        per_token(pool, count, [(&mut parts[..rows], d)], &|token, [row]| {
            let span = token * d..(token + 1) * d;
            lerp_row(row, &normed[span.clone()], &shift[span], &w.lerp_x);
        });
        let (by_token, rest) = low.split_at_mut(count * wide);
        let (by_part, rest) = rest.split_at_mut(count * wide);
        let decay_low = &mut rest[..count * w.decay_down.rows];
        self.multiply(
            &parts[..rows],
            d,
            &mut [Target::new(&w.mix_down, &mut *by_token)],
        );
        // Sort the corrections by part, so each map back up reads a run of
        // whole vectors.
        for (token, corrections) in by_token.chunks_exact(wide).enumerate() {
            for (index, correction) in corrections.chunks_exact(rank).enumerate() {
                let at = (index * count + token) * rank;
                for (out, value) in by_part[at..at + rank].iter_mut().zip(correction) {
                    *out = value.tanh();
                }
            }
        }
        let mut targets: Vec<Target> = parts
            .chunks_exact_mut(rows)
            .zip(&w.mix_up)
            .enumerate()
            .map(|(index, (out, linear))| {
                target(linear, out, Some(index * count..(index + 1) * count))
            })
            .collect();
        self.multiply(by_part, rank, &mut targets);
        drop(targets);
        // The five inputs: decay, key, value, receptance and gate.
        let mut each_part = parts.chunks_exact_mut(rows);
        let part_rows: [_; 5] = std::array::from_fn(|_| (each_part.next().unwrap(), d));
        per_token(pool, count, part_rows, &|token, rows| {
            let span = token * d..(token + 1) * d;
            let (normed, shift) = (&normed[span.clone()], &shift[span]);
            for (out, lerp) in rows.into_iter().zip(w.lerp.chunks_exact(d)) {
                let channels = out.iter_mut().zip(normed.iter().zip(shift)).zip(lerp);
                for ((out, (normed, shift)), lerp) in channels {
                    *out = normed + shift * (lerp + *out);
                }
            }
        });

        let (r, rest) = rkv.split_at_mut(rows);
        let (k, rest) = rest.split_at_mut(rows);
        let (v, rest) = rest.split_at_mut(rows);
        let gate = &mut rest[..rows];
        let part = |index: usize| Some(index * count..(index + 1) * count);
        let mut targets = [
            target(&w.decay_down, &mut *decay_low, part(0)),
            target(&time_mix.key, &mut *k, part(1)),
            target(&time_mix.value, &mut *v, part(2)),
            target(&time_mix.receptance, &mut *r, part(3)),
            target(&w.gate, &mut *gate, part(4)),
        ];
        self.multiply(parts, d, &mut targets);
        decay_low.iter_mut().for_each(|value| *value = value.tanh());
        let decay = &mut parts[..rows];
        self.multiply(
            decay_low,
            w.decay_down.rows,
            &mut [Target::new(&w.decay_up, &mut *decay)],
        );
        self.lap(Phase::Project);

        let buffers = [(&mut *decay, d), (&mut *gate, d)];
        per_token(pool, count, buffers, &|_, [decay, gate]| {
            for (decay, base) in decay.iter_mut().zip(&w.decay) {
                *decay = exp_approx(-exp_approx(base + *decay));
            }
            gate.iter_mut().for_each(|value| *value *= sigmoid(*value));
        });
        self.lap(Phase::Rotate);

        let head_size = config.rwkv.map_or(d, |rwkv| rwkv.head_size);
        let state = &mut self.recurrent.block(run.index)[2 * d..];
        let inputs = Wkv6 {
            r,
            w: decay,
            k,
            v,
            u: &w.first,
        };
        wkv6(self.engine.pool, state, head_size, d, &inputs, normed);
        self.lap(Phase::Attention);

        let (norm_weight, norm_bias) = &time_mix.head_norm;
        let gate = &*gate;
        per_token(pool, count, [(&mut *normed, d)], &|token, [y]| {
            head_norm(y, head_size, norm_weight, norm_bias);
            for (y, gate) in y.iter_mut().zip(&gate[token * d..(token + 1) * d]) {
                *y *= gate;
            }
        });
        self.multiply(
            normed,
            d,
            &mut [Target::new(&time_mix.output, &mut h[..rows])],
        );
        self.lap(Phase::AttentionOut);
    }

    /// The channel mixing: `value(relu(key(x))^2)`, gated by a receptance
    /// in RWKV-6. Every row's normalized input goes into the state, but
    /// only the active rows are computed and added to `x`.
    fn channel_mix(
        &mut self,
        run: &LayerRun,
        layer: &Layer,
        x: &mut [f32],
        h: &mut [f32],
        s: &mut RwkvScratch,
        ffn: &mut [f32],
    ) {
        let FeedForward::ChannelMix(mix) = &layer.ffn else {
            unreachable!("RWKV blocks are bound with a channel mix");
        };
        let config = self.config;
        let (d, eps, count) = (config.n_embd, config.norm_eps, self.count);
        let rows = count * d;
        let normed = &mut s.normed[..rows];
        let shift = &mut s.shift[..rows];
        let state = self.recurrent.block(run.index);
        norm_and_shift(
            self.engine.pool,
            &x[..rows],
            (&layer.ffn_norm, &mix.norm_bias),
            eps,
            normed,
            &mut state[d..2 * d],
            shift,
        );
        self.lap(Phase::Norm);

        let active = run.active.start * d..run.active.end * d;
        let n_active = run.active.len();
        if n_active == 0 {
            return;
        }
        // The inputs of the key and, in RWKV-6, of the receptance, for the
        // active rows only.
        let pool = self.engine.pool;
        let n_ff = mix.key.rows;
        let inputs = &mut s.parts[..2 * n_active * d];
        let (key_input, receptance_input) = inputs.split_at_mut(n_active * d);
        let (normed, shift) = (&normed[active.clone()], &shift[active.clone()]);
        let buffers = [(key_input, d), (receptance_input, d)];
        per_token(pool, n_active, buffers, &|token, [key, receptance]| {
            let span = token * d..(token + 1) * d;
            let (normed, shift) = (&normed[span.clone()], &shift[span]);
            lerp_row(key, normed, shift, &mix.lerp_k);
            if let Some((lerp_r, _)) = &mix.receptance {
                lerp_row(receptance, normed, shift, lerp_r);
            }
        });
        let keys = &mut ffn[..n_active * n_ff];
        let receptance = &mut s.rkv[..n_active * d];
        match &mix.receptance {
            None => self.multiply(
                &inputs[..n_active * d],
                d,
                &mut [Target::new(&mix.key, &mut *keys)],
            ),
            Some((_, linear)) => {
                let mut targets = [
                    target(&mix.key, &mut *keys, Some(0..n_active)),
                    target(linear, &mut *receptance, Some(n_active..2 * n_active)),
                ];
                self.multiply(inputs, d, &mut targets);
            }
        }
        self.lap(Phase::FeedForwardUp);
        per_token(pool, n_active, [(&mut *keys, n_ff)], &|_, [keys]| {
            for value in keys {
                let positive = value.max(0.0);
                *value = positive * positive;
            }
        });
        self.lap(Phase::Activation);
        let out = &mut h[..n_active * d];
        self.multiply(keys, n_ff, &mut [Target::new(&mix.value, &mut *out)]);
        let receptance = &*receptance;
        let buffers = [(&mut x[active], d), (out, d)];
        per_token(pool, n_active, buffers, &|token, [x, out]| {
            if mix.receptance.is_some() {
                let gates = &receptance[token * d..(token + 1) * d];
                for (out, gate) in out.iter_mut().zip(gates) {
                    *out *= sigmoid(*gate);
                }
            }
            add(x, out);
        });
        self.lap(Phase::FeedForwardDown);
    }
}

/// Batches of fewer tokens than this run their per-token steps on the
/// calling thread.
const PARALLEL_TOKENS: usize = 4;

/// A buffer with one row per token, shared by the threads of a loop over
/// tokens.
#[derive(Clone, Copy)]
struct TokenRows {
    data: *mut f32,
    width: usize,
}

// SAFETY: a loop over tokens hands each token to one thread, and a token
// reaches only its own row, which no other token's row overlaps.
unsafe impl Sync for TokenRows {}
unsafe impl Send for TokenRows {}

/// Calls `body` with each token's row of every buffer, on the thread pool
/// when the batch has several tokens. The buffers hold one row per token,
/// each as wide as given with it. The steps that use this treat every
/// token on its own, so any thread may take any token, and the work that is
/// not a matrix multiply does not hold back a prompt that every core reads.
fn per_token<const N: usize>(
    pool: &ThreadPool,
    count: usize,
    buffers: [(&mut [f32], usize); N],
    body: &(dyn Fn(usize, [&mut [f32]; N]) + Sync),
) {
    let rows = buffers.map(|(buffer, width)| {
        assert!(buffer.len() >= count * width, "a buffer holds every row");
        TokenRows {
            data: buffer.as_mut_ptr(),
            width,
        }
    });
    let token = |index: usize| {
        // SAFETY: the rows lie inside their buffers, as checked above, the
        // rows of different tokens do not overlap, and the pool runs each
        // token on one thread.
        let parts = rows.map(|rows| unsafe {
            slice::from_raw_parts_mut(rows.data.add(index * rows.width), rows.width)
        });
        body(index, parts);
    };
    if count < PARALLEL_TOKENS || pool.threads() == 1 {
        (0..count).for_each(token);
    } else {
        pool.run(count, &token);
    }
}

/// Layer-normalizes every row of `x` into `normed`, then writes each row's
/// token shift into `shift`: the previous row minus the current one. The
/// row before the first is `last`, which holds the previous batch's last
/// row and then takes this batch's.
fn norm_and_shift(
    pool: &ThreadPool,
    x: &[f32],
    (weight, bias): (&[f32], &[f32]),
    eps: f32,
    normed: &mut [f32],
    last: &mut [f32],
    shift: &mut [f32],
) {
    let d = weight.len();
    let count = x.len() / d;
    per_token(pool, count, [(&mut *normed, d)], &|token, [row]| {
        row.copy_from_slice(&x[token * d..(token + 1) * d]);
        layer_norm(row, weight, bias, eps);
    });
    let normed = &*normed;
    let previous_batch = &*last;
    per_token(pool, count, [(shift, d)], &|token, [shift]| {
        let previous = match token {
            0 => previous_batch,
            _ => &normed[(token - 1) * d..token * d],
        };
        let current = &normed[token * d..(token + 1) * d];
        for ((shift, previous), current) in shift.iter_mut().zip(previous).zip(current) {
            *shift = previous - current;
        }
    });
    last.copy_from_slice(&normed[(count - 1) * d..]);
}

/// A target that reads the input vectors `inputs` of a multiply.
fn target<'a>(linear: &'a Linear, out: &'a mut [f32], inputs: Option<Range<usize>>) -> Target<'a> {
    Target {
        linear,
        out,
        inputs,
    }
}

/// The logistic function, with an `exp` that loops vectorize.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_approx(-x))
}

/// `x += update`, element by element.
fn add(x: &mut [f32], update: &[f32]) {
    for (value, delta) in x.iter_mut().zip(update) {
        *value += delta;
    }
}

/// Moves one token's row of `normed` toward the previous token's by the
/// amounts in `lerp`, one per channel: `normed + shift * lerp`, where
/// `shift` holds the previous token's row minus this one.
fn lerp_row(out: &mut [f32], normed: &[f32], shift: &[f32], lerp: &[f32]) {
    for ((out, (normed, shift)), lerp) in out.iter_mut().zip(normed.iter().zip(shift)).zip(lerp) {
        *out = normed + shift * lerp;
    }
}

/// Sums `f` of every element of `x` in eight lanes, which the compiler
/// turns into vector instructions.
#[inline(always)]
fn lane_sum(x: &[f32], f: impl Fn(f32) -> f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for chunk in &mut chunks {
        for lane in 0..8 {
            lanes[lane] += f(chunk[lane]);
        }
    }
    let tail: f32 = chunks.remainder().iter().map(|&value| f(value)).sum();
    lanes.iter().sum::<f32>() + tail
}

/// The sum of the squares of `x`.
fn sum_of_squares(x: &[f32]) -> f32 {
    lane_sum(x, |value| value * value)
}

/// Normalizes `x` to zero mean and unit variance, in place.
fn standardize(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let mean = lane_sum(x, |value| value) / n;
    x.iter_mut().for_each(|value| *value -= mean);
    let scale = 1.0 / (sum_of_squares(x) / n + eps).sqrt();
    x.iter_mut().for_each(|value| *value *= scale);
}

/// Layer normalization of one row: `(x - mean) / deviation * weight +
/// bias`.
pub(super) fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    standardize(x, eps);
    for ((value, weight), bias) in x.iter_mut().zip(weight).zip(bias) {
        *value = *value * weight + bias;
    }
}

/// Normalizes each head of `row` on its own, then scales and shifts every
/// channel: a group norm with one group per head.
fn head_norm(row: &mut [f32], head_size: usize, weight: &[f32], bias: &[f32]) {
    for head in row.chunks_exact_mut(head_size) {
        standardize(head, HEAD_NORM_EPS);
    }
    for ((value, weight), bias) in row.iter_mut().zip(weight).zip(bias) {
        *value = *value * weight + bias;
    }
}

/// Dot product in eight lanes, which the compiler turns into vector
/// instructions.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a, b) in (&mut a_chunks).zip(&mut b_chunks) {
        for lane in 0..8 {
            lanes[lane] += a[lane] * b[lane];
        }
    }
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(a, b)| a * b)
        .sum();
    lanes.iter().sum::<f32>() + tail
}

/// The state and output that the heads of an update share. Each head
/// writes only its own matrix and its own channels of each output row.
struct Heads {
    state: *mut f32,
    out: *mut f32,
}

// SAFETY: see the type's description. The heads' parts are disjoint.
unsafe impl Sync for Heads {}

impl Heads {
    /// The matrix of `head`, `size` floats.
    ///
    /// # Safety
    ///
    /// The matrix must lie inside the state, and only one thread may use it
    /// at a time.
    #[allow(clippy::mut_from_ref)]
    unsafe fn matrix(&self, head: usize, size: usize) -> &mut [f32] {
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts_mut(self.state.add(head * size), size) }
    }

    /// `len` floats of the output from `start`.
    ///
    /// # Safety
    ///
    /// As for [`matrix`](Self::matrix), for the output.
    #[allow(clippy::mut_from_ref)]
    unsafe fn out(&self, start: usize, len: usize) -> &mut [f32] {
        // SAFETY: guaranteed by the caller.
        unsafe { slice::from_raw_parts_mut(self.out.add(start), len) }
    }
}

/// Runs `head` for every head of a state update, on the thread pool when
/// the update is large enough. `work` is the update's multiply-adds.
fn each_head(pool: &ThreadPool, heads: usize, work: usize, head: &(dyn Fn(usize) + Sync)) {
    if work <= INLINE_STATE_WORK {
        (0..heads).for_each(head);
    } else {
        pool.run(heads, head);
    }
}

/// The per-token inputs of RWKV-7's state update, one row of `d` per token.
struct Wkv7<'a> {
    r: &'a [f32],
    /// The decay.
    w: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    /// The normalized key whose content is removed.
    kk: &'a [f32],
    /// How much of it each channel removes: `kk * a`.
    b: &'a [f32],
}

/// RWKV-7's state update and output, for every token, head by head. Row
/// `i` of a head's matrix belongs to value channel `i`, as in llama.cpp:
///
/// ```text
/// S[i] = S[i] * w + v[i] * k - (S[i] . kk) * b
/// y[i] = S[i] . r
/// ```
fn wkv7(pool: &ThreadPool, state: &mut [f32], n: usize, d: usize, x: &Wkv7, out: &mut [f32]) {
    let heads = d / n;
    let count = out.len() / d;
    let shared = Heads {
        state: state.as_mut_ptr(),
        out: out.as_mut_ptr(),
    };
    let update = |head: usize| {
        // SAFETY: each head takes its own matrix and its own channels.
        let matrix = unsafe { shared.matrix(head, n * n) };
        for token in 0..count {
            let at = token * d + head * n;
            let span = at..at + n;
            let (r, w, k) = (&x.r[span.clone()], &x.w[span.clone()], &x.k[span.clone()]);
            let (v, kk, b) = (&x.v[span.clone()], &x.kk[span.clone()], &x.b[span]);
            // SAFETY: as above.
            let y = unsafe { shared.out(at, n) };
            for ((row, &value), y) in matrix.chunks_exact_mut(n).zip(v).zip(y.iter_mut()) {
                let removed = dot(row, kk);
                for (((s, w), k), b) in row.iter_mut().zip(w).zip(k).zip(b) {
                    *s = *s * w + value * k - removed * b;
                }
                *y = dot(row, r);
            }
        }
    };
    each_head(pool, heads, count * heads * n * n * 3, &update);
}

/// The per-token inputs of RWKV-6's state update, one row of `d` per token,
/// and the bonus `u` for the current token, one per channel.
struct Wkv6<'a> {
    r: &'a [f32],
    w: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    u: &'a [f32],
}

/// RWKV-6's output and state update, for every token, head by head. Row `i`
/// of a head's matrix belongs to key channel `i`, as in llama.cpp:
///
/// ```text
/// y    = sum over i of r[i] * (S[i] + u[i] * k[i] * v)
/// S[i] = S[i] * w[i] + k[i] * v
/// ```
fn wkv6(pool: &ThreadPool, state: &mut [f32], n: usize, d: usize, x: &Wkv6, out: &mut [f32]) {
    let heads = d / n;
    let count = out.len() / d;
    let shared = Heads {
        state: state.as_mut_ptr(),
        out: out.as_mut_ptr(),
    };
    let update = |head: usize| {
        // SAFETY: each head takes its own matrix and its own channels.
        let matrix = unsafe { shared.matrix(head, n * n) };
        let u = &x.u[head * n..(head + 1) * n];
        for token in 0..count {
            let at = token * d + head * n;
            let span = at..at + n;
            let (r, w, k, v) = (
                &x.r[span.clone()],
                &x.w[span.clone()],
                &x.k[span.clone()],
                &x.v[span],
            );
            // SAFETY: as above.
            let y = unsafe { shared.out(at, n) };
            y.fill(0.0);
            let mut bonus = 0.0;
            for (((row, &r), &w), (&k, &u)) in matrix
                .chunks_exact_mut(n)
                .zip(r)
                .zip(w)
                .zip(k.iter().zip(u))
            {
                bonus += r * u * k;
                for ((y, s), v) in y.iter_mut().zip(row.iter_mut()).zip(v) {
                    *y += r * *s;
                    *s = *s * w + k * v;
                }
            }
            for (y, v) in y.iter_mut().zip(v) {
                *y += bonus * v;
            }
        }
    };
    each_head(pool, heads, count * heads * n * n * 2, &update);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_shift_carries_the_last_row_over() {
        // Four tokens of two channels, enough to run on the pool. Each row
        // normalizes to plus or minus one.
        let pool = ThreadPool::new(3);
        let x = [1.0, 2.0, 2.0, 1.0, 5.0, 9.0, 0.0, 4.0];
        let mut normed = [0.0; 8];
        let mut last = [10.0, 20.0];
        let mut shift = [0.0; 8];
        let norm = ([1.0, 1.0].as_slice(), [0.0, 0.0].as_slice());
        norm_and_shift(&pool, &x, norm, 0.0, &mut normed, &mut last, &mut shift);
        assert_eq!(normed, [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]);
        assert_eq!(shift, [11.0, 19.0, -2.0, 2.0, 2.0, -2.0, 0.0, 0.0]);
        assert_eq!(last, [-1.0, 1.0]);
    }

    #[test]
    fn layer_norm_centers_and_scales() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        layer_norm(&mut x, &[1.0, 1.0, 2.0, 2.0], &[0.0, 1.0, 0.0, 0.0], 0.0);
        let unit = 1.0 / 1.25f32.sqrt();
        let expected = [-1.5 * unit, -0.5 * unit + 1.0, 1.0 * unit, 3.0 * unit];
        for (got, expected) in x.iter().zip(expected) {
            assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
        }
    }

    /// Times the state updates of a 1.5B model's block, one token at a time
    /// and in batches, on one thread. A measurement, not a check. Run it
    /// with `cargo test --release --lib state_update_speed -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore = "a benchmark, run on request"]
    fn state_update_speed() {
        use crate::testing::Rng;
        let (d, n) = (2048, 64);
        let pool = ThreadPool::new(1);
        let mut rng = Rng(7);
        for count in [1, 32] {
            let inputs: Vec<Vec<f32>> = (0..6).map(|_| rng.floats(count * d, 0.0, 1.0)).collect();
            let mut state = vec![0.0; d * n];
            let mut out = vec![0.0; count * d];
            let rounds = 200 / count;
            let wkv = Wkv7 {
                r: &inputs[0],
                w: &inputs[1],
                k: &inputs[2],
                v: &inputs[3],
                kk: &inputs[4],
                b: &inputs[5],
            };
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                wkv7(&pool, &mut state, n, d, &wkv, &mut out);
            }
            let v7 = start.elapsed().as_secs_f64() / (rounds * count) as f64;
            let wkv = Wkv6 {
                r: &inputs[0],
                w: &inputs[1],
                k: &inputs[2],
                v: &inputs[3],
                u: &inputs[4][..d],
            };
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                wkv6(&pool, &mut state, n, d, &wkv, &mut out);
            }
            let v6 = start.elapsed().as_secs_f64() / (rounds * count) as f64;
            println!(
                "batch {count:>2}: RWKV-7 {:.1} us, RWKV-6 {:.1} us per token and block",
                v7 * 1e6,
                v6 * 1e6
            );
        }
    }

    #[test]
    fn dot_products_take_every_element() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..19).map(|i| 1.0 + (i % 3) as f32).collect();
        let expected: f32 = a.iter().zip(&b).map(|(a, b)| a * b).sum();
        assert_eq!(dot(&a, &b), expected);
        assert_eq!(lane_sum(&a, |value| value), 171.0);
    }
}

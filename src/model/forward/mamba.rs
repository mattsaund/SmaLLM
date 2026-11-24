//! Mamba blocks: a short convolution, then a selective scan.
//!
//! A Mamba block widens the normalized hidden state `h` into a signal of
//! `inner` channels and an output gate `z`. Each channel of the signal is
//! convolved with its last few positions, and a scan then carries a state
//! of `state` floats per channel from one position to the next:
//!
//! ```text
//! x z    = h * Wx, h * Wz
//! x      = silu(conv(x) + bias)            over the last few positions
//! dt     = softplus(time step)             one per head, for each token
//! s      = s * exp(dt * A) + dt * x * B    one row of the state per channel
//! y      = s . C + D * x
//! mix    = [norm] (y * silu(z)) * Wout
//! ```
//!
//! Mamba 1 makes every channel a head, gives each element of the state its
//! own decay rate in `A`, and computes the time step, `B` and `C` from the
//! convolved signal (Falcon Mamba normalizes them). Mamba 2 groups the
//! channels into heads with one decay rate and one time step each, which
//! the input projection yields directly. Its `B` and `C`, one pair per
//! group of heads, go through the convolution with the signal. It also
//! normalizes the gated output in groups before the output projection.
//!
//! The scan is sequential over positions but independent between heads, so
//! the threads split it by heads, and each runs every token of the batch
//! through its share while that share of the state stays in its cache. The
//! projections around the scan go through the matrix engine with the whole
//! batch at once.
//!
//! The state lives in the session's [`RecurrentState`]: per block, the
//! convolution's inputs at the last `conv_kernel - 1` positions, then the
//! scan's state. Mamba 2 keeps one row of `state` floats per channel.
//! Mamba 1 keeps one row of `inner` floats per element of the state, which
//! suits its loops (see [`Scan::mamba1`]).
//!
//! [`RecurrentState`]: super::RecurrentState

use std::ops::Range;

use super::{LayerRun, Pass, Phase, SharedOutput, rms_norm_in_place, rms_norm_unit};
use crate::kernels::exp_approx;
use crate::model::config::{Mixer, Ssm};
use crate::model::matmul::Target;
use crate::model::weights::MambaWeights;
use crate::threads::ThreadPool;

/// Floats of state that one parallel piece of a loop handles at least.
/// Smaller pieces cost more to hand to a thread than to compute.
const MIN_PIECE: usize = 16 * 1024;

/// Lanes of the running sums in Mamba 2's state loop. The sums are kept
/// apart per lane so that the loop vectorizes without reordering additions.
const LANES: usize = 8;

/// Channels a piece of Mamba 1's scan takes through the tokens together.
/// Their state, a few kilobytes, stays in the core's own cache.
const BLOCK: usize = 64;

/// Buffers of the Mamba blocks, sized for a full batch. Empty for models
/// without them.
#[derive(Default)]
pub(super) struct MambaScratch {
    /// The convolution's inputs: the saved ones, then one row per token.
    history: Vec<f32>,
    /// The convolution's output, one row per token: the signal, and in
    /// Mamba 2 also each group's B and C.
    signal: Vec<f32>,
    /// Mamba 1's time step in low-rank form, B and C, one row per token.
    low: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    /// Each head's time step before the softplus, one row per token.
    dt: Vec<f32>,
    /// The output gate of each active token, then the gated output.
    z: Vec<f32>,
    /// The scan's output for each active token.
    y: Vec<f32>,
}

impl MambaScratch {
    /// Buffers for batches of `batch` tokens through blocks of shape `ssm`.
    pub fn new(ssm: &Ssm, batch: usize) -> MambaScratch {
        let (rows, channels) = (ssm.conv_kernel - 1, ssm.conv_channels());
        let (low, b_c) = if ssm.mamba2 {
            (0, 0)
        } else {
            (ssm.dt_rank, ssm.state)
        };
        MambaScratch {
            history: vec![0.0; (rows + batch) * channels],
            signal: vec![0.0; batch * channels],
            low: vec![0.0; batch * low],
            b: vec![0.0; batch * b_c],
            c: vec![0.0; batch * b_c],
            dt: vec![0.0; batch * ssm.heads()],
            z: vec![0.0; batch * ssm.inner],
            y: vec![0.0; batch * ssm.inner],
        }
    }
}

impl Pass<'_> {
    /// A Mamba block. Reads the normalized inputs in `h`, folds every token
    /// into the block's state, and leaves the block's output for the active
    /// rows at the start of `h`.
    pub(super) fn mamba(
        &mut self,
        run: &LayerRun,
        w: &MambaWeights,
        h: &mut [f32],
        s: &mut MambaScratch,
    ) {
        let Mixer::Mamba(ssm) = run.spec.mixer else {
            unreachable!("Mamba weights belong to a Mamba block");
        };
        let (kernels, pool) = (self.engine.kernels, self.engine.pool);
        let (d, eps) = (self.config.n_embd, self.config.norm_eps);
        let (count, n_active) = (self.count, run.active.len());
        let (inner, rows, channels) = (ssm.inner, ssm.conv_kernel - 1, ssm.conv_channels());
        let heads = ssm.heads();

        // The signal of every token goes into the convolution's history,
        // after the inputs saved from earlier positions. Mamba 2's time
        // steps are needed for every token too, the gate for active ones.
        let history = &mut s.history[..(rows + count) * channels];
        let dt = &mut s.dt[..count * heads];
        let z = &mut s.z[..n_active * inner];
        let mut kept = vec![Target::new(&w.x, &mut history[rows * channels..])];
        if ssm.mamba2 {
            kept.push(Target::new(&w.dt, &mut *dt));
        }
        self.project(h, run, kept, vec![Target::new(&w.z, &mut *z)]);
        self.lap(Phase::Project);

        let state = self.recurrent.block(run.index);
        let conv_state = &mut state[..rows * channels];
        history[..rows * channels].copy_from_slice(conv_state);
        let signal = &mut s.signal[..count * channels];
        convolve(pool, history, &w.conv, &w.conv_bias, signal);
        conv_state.copy_from_slice(&history[count * channels..]);
        self.lap(Phase::Attention);

        // Mamba 1 projects the convolved signal to the time step in
        // low-rank form, and to B and C, then the time step back up.
        let (b, c) = match &w.low_rank {
            Some(low_rank) => {
                let rank = ssm.dt_rank;
                let low = &mut s.low[..count * rank];
                let b = &mut s.b[..count * ssm.state];
                let c = &mut s.c[..count * ssm.state];
                self.multiply(
                    signal,
                    inner,
                    &mut [
                        Target::new(&low_rank.dt, &mut *low),
                        Target::new(&low_rank.b, &mut *b),
                        Target::new(&low_rank.c, &mut *c),
                    ],
                );
                if ssm.dt_b_c_norm {
                    let rows = low
                        .chunks_exact_mut(rank)
                        .chain(b.chunks_exact_mut(ssm.state))
                        .chain(c.chunks_exact_mut(ssm.state));
                    for row in rows {
                        rms_norm_unit(kernels, row, eps);
                    }
                }
                self.multiply(low, rank, &mut [Target::new(&w.dt, &mut *dt)]);
                self.lap(Phase::Project);
                (&s.b[..count * ssm.state], &s.c[..count * ssm.state])
            }
            None => (&[][..], &[][..]),
        };

        let state = &mut self.recurrent.block(run.index)[rows * channels..];
        let scan = Scan {
            ssm,
            signal,
            b,
            c,
            dt,
            a: &w.a,
            d: &w.d,
            count,
            active: run.active.clone(),
        };
        let y = &mut s.y[..n_active * inner];
        if ssm.mamba2 {
            scan.mamba2(pool, state, y);
        } else {
            scan.mamba1(pool, state, y);
        }

        // The output, gated by the SiLU of z. Mamba 2 normalizes each
        // group's share of it.
        kernels.silu_mul(z, y);
        if let Some(gain) = &w.norm {
            let width = inner / ssm.groups;
            for row in z.chunks_exact_mut(inner) {
                for (group, gain) in row.chunks_exact_mut(width).zip(gain.chunks_exact(width)) {
                    rms_norm_in_place(kernels, group, gain, eps);
                }
            }
        }
        self.lap(Phase::Attention);
        self.multiply(z, inner, &mut [Target::new(&w.out, &mut h[..n_active * d])]);
        self.lap(Phase::AttentionOut);
    }
}

/// Splits `units` independent units of `work` floats each into pieces for
/// the threads: several per thread, so that a slow core does not hold up
/// the rest, but none below [`MIN_PIECE`] floats. A small job stays on the
/// calling thread.
fn split(pool: &ThreadPool, units: usize, work: usize, body: &(dyn Fn(Range<usize>) + Sync)) {
    let smallest = MIN_PIECE.div_ceil(work.max(1));
    let balanced = units.div_ceil(pool.threads() * 4);
    pool.ranges(units, smallest.max(balanced), body);
}

/// The convolution of every token, through a SiLU. Row `t` of `out` is
/// `silu(bias + sum over taps k of kernel[k] * history[t + k])`, per
/// channel. `history` holds the saved inputs, then one row per token.
///
/// The channels are independent, so threads split them.
fn convolve(pool: &ThreadPool, history: &[f32], kernel: &[f32], bias: &[f32], out: &mut [f32]) {
    let channels = bias.len();
    let (count, taps) = (out.len() / channels, kernel.len() / channels);
    let output = SharedOutput(out.as_mut_ptr());
    split(pool, channels, count * taps, &|range| {
        for token in 0..count {
            // SAFETY: the pieces are disjoint ranges of channels, so each
            // writes its own part of every row, and the rows lie in `out`.
            let out = unsafe { output.slice(token * channels + range.start, range.len()) };
            out.fill(0.0);
            for (tap, weights) in kernel.chunks_exact(channels).enumerate() {
                let input = &history[(token + tap) * channels..][range.clone()];
                for ((sum, weight), value) in out.iter_mut().zip(&weights[range.clone()]).zip(input)
                {
                    *sum += weight * value;
                }
            }
            // The bias comes after the sum, as in llama.cpp.
            for (value, bias) in out.iter_mut().zip(&bias[range.clone()]) {
                let x = *value + bias;
                *value = x / (1.0 + exp_approx(-x));
            }
        }
    });
}

/// `ln(1 + e^x)`, which turns a raw time step into a positive one.
///
/// It is computed as `max(x, 0) + ln(1 + e^-|x|)`, without branches, so
/// that a loop of them vectorizes. Large inputs come out unchanged, as in
/// llama.cpp.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + ln_1p_unit(exp_approx(-x.abs()))
}

/// `ln(1 + t)` for `t` between 0 and 1, with a relative error near 1e-7.
///
/// It sums the series of `2 atanh(u)` with `u = t / (2 + t)`, which is at
/// most a third, so seven terms reach float precision.
#[inline(always)]
fn ln_1p_unit(t: f32) -> f32 {
    let u = t / (2.0 + t);
    let u2 = u * u;
    let mut p = 1.0 / 13.0;
    p = p * u2 + 1.0 / 11.0;
    p = p * u2 + 1.0 / 9.0;
    p = p * u2 + 1.0 / 7.0;
    p = p * u2 + 1.0 / 5.0;
    p = p * u2 + 1.0 / 3.0;
    p = p * u2 + 1.0;
    2.0 * u * p
}

/// What a scan over one batch reads. Every thread shares it.
struct Scan<'a> {
    ssm: Ssm,
    /// The convolved signal of every token, `conv_channels` floats each.
    signal: &'a [f32],
    /// Mamba 1's B and C of every token. Mamba 2 reads them from `signal`.
    b: &'a [f32],
    c: &'a [f32],
    /// Each head's time step for every token, before the softplus.
    dt: &'a [f32],
    /// The decay rates and the skip weights, as [`MambaWeights`] has them.
    a: &'a [f32],
    d: &'a [f32],
    /// Tokens in the batch.
    count: usize,
    /// Tokens whose output is wanted.
    active: Range<usize>,
}

impl Scan<'_> {
    /// Mamba 1's scan. Every channel is a head of its own, and each element
    /// of its state decays at its own rate: `s = s * exp(dt * a) + dt * x
    /// * B`. Writes `y = s . C + D * x` of the active tokens to `y`.
    ///
    /// The state and the rates are stored element by element, so the loops
    /// run along channels: the same arithmetic on neighboring floats, which
    /// vectorizes, with no sums across lanes. Each channel still adds up
    /// its elements in order, as llama.cpp does. Threads split the
    /// channels, and take them in blocks through every token, so that a
    /// block's state stays in the core's cache.
    fn mamba1(&self, pool: &ThreadPool, state: &mut [f32], y: &mut [f32]) {
        let (inner, width) = (self.ssm.inner, self.ssm.state);
        let states = SharedOutput(state.as_mut_ptr());
        let outputs = SharedOutput(y.as_mut_ptr());
        // An element costs an exponential, about four of Mamba 2's.
        split(pool, inner, 4 * self.count * width, &|range| {
            let mut deltas = [0.0f32; BLOCK];
            let mut inputs = [0.0f32; BLOCK];
            let mut sums = [0.0f32; BLOCK];
            for first in range.clone().step_by(BLOCK) {
                let block = first..(first + BLOCK).min(range.end);
                let n = block.len();
                let (deltas, inputs, sums) = (&mut deltas[..n], &mut inputs[..n], &mut sums[..n]);
                for token in 0..self.count {
                    let x = &self.signal[token * inner..][block.clone()];
                    let dt = &self.dt[token * inner..][block.clone()];
                    for (((delta, input), &dt), &x) in
                        deltas.iter_mut().zip(&mut *inputs).zip(dt).zip(x)
                    {
                        *delta = softplus(dt);
                        *input = x * *delta;
                    }
                    sums.fill(0.0);
                    let b = &self.b[token * width..(token + 1) * width];
                    let c = &self.c[token * width..(token + 1) * width];
                    for (element, (&b, &c)) in b.iter().zip(c).enumerate() {
                        let at = element * inner + first;
                        // SAFETY: each piece updates the state of its own
                        // channels only, and the block lies inside a row.
                        let state = unsafe { states.slice(at, n) };
                        let rates = &self.a[at..at + n];
                        let lanes = state.iter_mut().zip(rates).zip(&*deltas).zip(&*inputs);
                        for ((((state, &rate), &delta), &input), sum) in lanes.zip(&mut *sums) {
                            *state = *state * exp_approx(delta * rate) + b * input;
                            *sum += *state * c;
                        }
                    }
                    if self.active.contains(&token) {
                        let at = (token - self.active.start) * inner + first;
                        // SAFETY: as above, for the outputs of the block.
                        let out = unsafe { outputs.slice(at, n) };
                        let skips = self.d[block.clone()].iter().zip(x);
                        for ((out, &sum), (&d, &x)) in out.iter_mut().zip(&*sums).zip(skips) {
                            *out = sum + d * x;
                        }
                    }
                }
            }
        });
    }

    /// Mamba 2's scan. The state of a head decays by one rate per token,
    /// `s = s * exp(dt * a) + dt * x * B`, with the B and C of the head's
    /// group. Writes `y = s . C + D * x` of the active tokens to `y`.
    ///
    /// Threads split the heads. Each takes its heads one at a time through
    /// every token, so that the head's state stays in the cache.
    fn mamba2(&self, pool: &ThreadPool, state: &mut [f32], y: &mut [f32]) {
        let ssm = self.ssm;
        let (inner, width, heads) = (ssm.inner, ssm.state, ssm.heads());
        let (channels, head_dim) = (ssm.conv_channels(), inner / heads);
        let per_group = heads / ssm.groups;
        let states = SharedOutput(state.as_mut_ptr());
        let outputs = SharedOutput(y.as_mut_ptr());
        split(pool, heads, self.count * head_dim * width, &|range| {
            for head in range {
                let first = head * head_dim;
                // SAFETY: each piece updates the state rows of its own
                // heads, and writes their outputs only.
                let rows = unsafe { states.slice(first * width, head_dim * width) };
                let group = head / per_group;
                for token in 0..self.count {
                    let delta = softplus(self.dt[token * heads + head]);
                    let decay = (delta * self.a[head]).exp();
                    let signal = &self.signal[token * channels..(token + 1) * channels];
                    let b = &signal[inner + group * width..][..width];
                    let c = &signal[inner + (ssm.groups + group) * width..][..width];
                    let x = &signal[first..first + head_dim];
                    let active = self.active.contains(&token);
                    for (index, (row, &x)) in rows.chunks_exact_mut(width).zip(x).enumerate() {
                        let sum = decay_all(row, b, c, decay, x * delta);
                        if active {
                            let at = (token - self.active.start) * inner + first + index;
                            // SAFETY: as above.
                            unsafe { outputs.slice(at, 1)[0] = sum + self.d[head] * x };
                        }
                    }
                }
            }
        });
    }
}

/// One step of a Mamba 2 state row: `row = row * decay + b * x_dt`, and
/// returns the new `row . c`.
///
/// The update and the sum are two loops. As one, the compiler left the
/// update unvectorized. The row is a few hundred bytes, so the second pass
/// reads it from the core's own cache.
#[inline(always)]
fn decay_all(row: &mut [f32], b: &[f32], c: &[f32], decay: f32, x_dt: f32) -> f32 {
    for (value, &b) in row.iter_mut().zip(b) {
        *value = *value * decay + b * x_dt;
    }
    let mut sums = [0.0f32; LANES];
    let mut rows = row.chunks_exact(LANES);
    let mut cs = c.chunks_exact(LANES);
    for (row, c) in (&mut rows).zip(&mut cs) {
        for lane in 0..LANES {
            sums[lane] += row[lane] * c[lane];
        }
    }
    let mut sum: f32 = sums.iter().sum();
    for (value, c) in rows.remainder().iter().zip(cs.remainder()) {
        sum += value * c;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_is_accurate_everywhere() {
        let mut worst = 0.0f64;
        for step in 0..200_000 {
            let x = -40.0 + step as f32 * 0.0004;
            let exact = (x as f64).exp().ln_1p();
            worst = worst.max(((softplus(x) as f64 - exact) / exact).abs());
        }
        assert!(worst < 4e-7, "relative error {worst}");
        // Large time steps pass through, as llama.cpp's do.
        for x in [20.5f32, 50.0, 1e4] {
            assert_eq!(softplus(x), x);
        }
    }
}

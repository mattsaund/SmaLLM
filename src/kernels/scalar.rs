//! Portable kernels in plain Rust.
//!
//! These run on every target and define the expected result of every other
//! backend. The loops are written so the compiler can vectorize them: fixed
//! lane counts, no early exits, and no order-dependent float reductions.
//!
//! The functions ending in `_impl` are also compiled a second time by the
//! SIMD backends, inside functions that enable wider instructions.

use std::slice;

use super::{ActRef, DotQuant, Kernels};
use crate::tensor::TensorType;
use crate::tensor::f16::{bf16_to_f32, f16_to_f32};
use crate::tensor::quant::{self, Group};

pub static KERNELS: Kernels = Kernels {
    name: "scalar",
    dot,
    axpy,
    sum_squares,
    dot_rows,
    mix_rows,
    silu_mul,
    gelu_mul,
    softmax,
    quantize,
    row_f32,
    row_f16,
    row_bf16,
    row_quant,
    dot_decoded,
};

/// Kernels that never round activations.
///
/// Quantized weight rows are expanded to floats and multiplied with the
/// activations as they are. This is many times slower than the integer
/// kernels and exists to measure how much their rounding costs.
pub static EXACT: Kernels = Kernels {
    name: "exact",
    dot,
    axpy,
    sum_squares,
    dot_rows,
    mix_rows,
    silu_mul,
    gelu_mul,
    softmax,
    quantize,
    row_f32,
    row_f16,
    row_bf16,
    row_quant: exact::row_quant,
    dot_decoded,
};

/// Number of independent accumulators in float reductions.
const LANES: usize = 8;

/// Adds the lanes of an accumulator in a fixed pairwise order.
#[inline(always)]
fn reduce(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

/// See [`Kernels::dot`].
#[inline(always)]
pub(super) unsafe fn dot_impl(a: *const f32, b: *const f32, n: usize) -> f32 {
    // SAFETY: the caller guarantees both pointers are valid for `n` reads.
    let (a, b) = unsafe { (slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) };
    let mut lanes = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in (&mut a_chunks).zip(&mut b_chunks) {
        for l in 0..LANES {
            lanes[l] += x[l] * y[l];
        }
    }
    let mut sum = reduce(lanes);
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += x * y;
    }
    sum
}

/// See [`Kernels::axpy`].
#[inline(always)]
pub(super) unsafe fn axpy_impl(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
    // SAFETY: the caller guarantees `x` is valid for `n` reads and `y` for
    // `n` writes, and that they do not overlap.
    let (x, y) = unsafe { (slice::from_raw_parts(x, n), slice::from_raw_parts_mut(y, n)) };
    for (out, value) in y.iter_mut().zip(x) {
        *out += alpha * value;
    }
}

/// See [`Kernels::sum_squares`].
#[inline(always)]
pub(super) unsafe fn sum_squares_impl(x: *const f32, n: usize) -> f32 {
    // SAFETY: the caller guarantees the pointer is valid for `n` reads.
    let x = unsafe { slice::from_raw_parts(x, n) };
    let mut lanes = [0.0f32; LANES];
    let mut chunks = x.chunks_exact(LANES);
    for chunk in &mut chunks {
        for l in 0..LANES {
            lanes[l] += chunk[l] * chunk[l];
        }
    }
    let mut sum = reduce(lanes);
    for value in chunks.remainder() {
        sum += value * value;
    }
    sum
}

/// See [`Kernels::dot_rows`].
#[inline(always)]
pub(super) unsafe fn dot_rows_impl(
    queries: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    scores: *mut f32,
    stride: usize,
) {
    for j in 0..n {
        for h in 0..n_queries {
            // SAFETY: the caller guarantees `n_queries` queries and `n` rows
            // of `len` floats, and room for every score.
            unsafe {
                *scores.add(h * stride + j) = dot_impl(queries.add(h * len), rows.add(j * len), len)
            };
        }
    }
}

/// See [`Kernels::mix_rows`].
#[inline(always)]
pub(super) unsafe fn mix_rows_impl(
    weights: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    outs: *mut f32,
    stride: usize,
) {
    for h in 0..n_queries {
        for j in 0..n {
            // SAFETY: as in `dot_rows_impl`.
            unsafe {
                axpy_impl(
                    *weights.add(h * stride + j),
                    rows.add(j * len),
                    outs.add(h * len),
                    len,
                )
            };
        }
    }
}

/// A fast `exp` with a relative error near 1e-7.
///
/// Splits `x = k * ln(2) + r` with `|r| <= ln(2) / 2`, evaluates a
/// polynomial for `exp(r)`, and scales by `2^k` through the float's exponent
/// bits. Inputs are clamped to the range where the result is a normal float.
#[inline(always)]
pub(super) fn exp_approx(x: f32) -> f32 {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    // ln(2) split in two so that `k * LN2_HIGH` is exact.
    const LN2_HIGH: f32 = 0.693_145_75;
    const LN2_LOW: f32 = 1.428_606_8e-6;
    // Adding and subtracting 1.5 * 2^23 rounds to the nearest integer
    // without a rounding instruction, so the loop vectorizes everywhere.
    const ROUND: f32 = 12_582_912.0;

    let x = x.clamp(-87.0, 88.0);
    let k = (x * LOG2_E + ROUND) - ROUND;
    let r = (x - k * LN2_HIGH) - k * LN2_LOW;
    // Taylor series of exp(r) to the sixth power, in Horner form.
    let mut p = 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    let two_k = f32::from_bits(((k as i32 + 127) as u32) << 23);
    p * two_k
}

/// See [`Kernels::silu_mul`]. `silu(x) = x / (1 + exp(-x))`.
#[inline(always)]
pub(super) unsafe fn silu_mul_impl(gate: *mut f32, up: *const f32, n: usize) {
    // SAFETY: the caller guarantees `gate` is valid for `n` writes and `up`
    // for `n` reads, and that they do not overlap.
    let (gate, up) = unsafe {
        (
            slice::from_raw_parts_mut(gate, n),
            slice::from_raw_parts(up, n),
        )
    };
    for (g, u) in gate.iter_mut().zip(up) {
        *g = *g / (1.0 + exp_approx(-*g)) * u;
    }
}

/// See [`Kernels::gelu_mul`]. Uses the tanh form of GELU, which simplifies
/// to `x * sigmoid(2z)` with `z = sqrt(2 / pi) * x * (1 + 0.044715 * x^2)`.
#[inline(always)]
pub(super) unsafe fn gelu_mul_impl(gate: *mut f32, up: *const f32, n: usize) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const CUBIC: f32 = 0.044_715;
    // SAFETY: as in `silu_mul_impl`.
    let (gate, up) = unsafe {
        (
            slice::from_raw_parts_mut(gate, n),
            slice::from_raw_parts(up, n),
        )
    };
    for (g, u) in gate.iter_mut().zip(up) {
        let x = *g;
        let z = SQRT_2_OVER_PI * x * (1.0 + CUBIC * x * x);
        *g = x / (1.0 + exp_approx(-2.0 * z)) * u;
    }
}

/// See [`Kernels::softmax`].
#[inline(always)]
pub(super) unsafe fn softmax_impl(x: *mut f32, n: usize) {
    // SAFETY: the caller guarantees the pointer is valid for `n` reads and
    // writes.
    let x = unsafe { slice::from_raw_parts_mut(x, n) };
    let mut max_lanes = [f32::NEG_INFINITY; LANES];
    let mut chunks = x.chunks_exact(LANES);
    for chunk in &mut chunks {
        for l in 0..LANES {
            if chunk[l] > max_lanes[l] {
                max_lanes[l] = chunk[l];
            }
        }
    }
    let mut max = f32::NEG_INFINITY;
    for value in max_lanes.iter().chain(chunks.remainder()) {
        if *value > max {
            max = *value;
        }
    }
    // Exponentiate first and sum second: a sum inside the first loop would
    // have to run in order and would stop it from vectorizing.
    for value in x.iter_mut() {
        *value = exp_approx(*value - max);
    }
    let mut lanes = [0.0f32; LANES];
    let mut chunks = x.chunks_exact(LANES);
    for chunk in &mut chunks {
        for l in 0..LANES {
            lanes[l] += chunk[l];
        }
    }
    let mut sum = reduce(lanes);
    for value in chunks.remainder() {
        sum += value;
    }
    let inverse = 1.0 / sum;
    for value in x.iter_mut() {
        *value *= inverse;
    }
}

/// See [`Kernels::quantize`].
///
/// Each group is scaled so its largest magnitude maps to 127, then rounded
/// to the nearest integer, ties to even.
#[inline(always)]
pub(super) unsafe fn quantize_impl(
    x: *const f32,
    n: usize,
    group: usize,
    q: *mut i8,
    scale: *mut f32,
    sums: *mut i16,
    wsum: *mut f32,
) {
    const ROUND: f32 = 12_582_912.0;
    // SAFETY: the caller guarantees the buffer sizes documented on
    // `Kernels::quantize`.
    let (x, q, scale, sums, wsum) = unsafe {
        (
            slice::from_raw_parts(x, n),
            slice::from_raw_parts_mut(q, n),
            slice::from_raw_parts_mut(scale, n / group),
            slice::from_raw_parts_mut(sums, n / 16),
            slice::from_raw_parts_mut(wsum, n / group),
        )
    };
    let groups = x.chunks_exact(group).zip(q.chunks_exact_mut(group));
    for (g, (values, out)) in groups.enumerate() {
        let mut max_lanes = [0.0f32; LANES];
        for chunk in values.chunks_exact(LANES) {
            for l in 0..LANES {
                let magnitude = chunk[l].abs();
                if magnitude > max_lanes[l] {
                    max_lanes[l] = magnitude;
                }
            }
        }
        let mut max = 0.0f32;
        for lane in max_lanes {
            if lane > max {
                max = lane;
            }
        }
        let group_sums = &mut sums[g * group / 16..(g + 1) * group / 16];
        if max == 0.0 {
            out.fill(0);
            group_sums.fill(0);
            scale[g] = 0.0;
            wsum[g] = 0.0;
            continue;
        }
        let inverse = 127.0 / max;
        for (o, value) in out.iter_mut().zip(values) {
            *o = ((value * inverse + ROUND) - ROUND) as i32 as i8;
        }
        let mut total = 0i32;
        for (sum, chunk) in group_sums.iter_mut().zip(out.chunks_exact(16)) {
            let mut partial = 0i32;
            for value in chunk {
                partial += *value as i32;
            }
            *sum = partial as i16;
            total += partial;
        }
        scale[g] = max / 127.0;
        wsum[g] = scale[g] * total as f32;
    }
}

pub(super) unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe { dot_impl(a, b, n) }
}

pub(super) unsafe fn axpy(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { axpy_impl(alpha, x, y, n) }
}

pub(super) unsafe fn sum_squares(x: *const f32, n: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe { sum_squares_impl(x, n) }
}

pub(super) unsafe fn dot_rows(
    queries: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    scores: *mut f32,
    stride: usize,
) {
    // SAFETY: forwarded contract.
    unsafe { dot_rows_impl(queries, n_queries, rows, n, len, scores, stride) }
}

pub(super) unsafe fn mix_rows(
    weights: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    outs: *mut f32,
    stride: usize,
) {
    // SAFETY: forwarded contract.
    unsafe { mix_rows_impl(weights, n_queries, rows, n, len, outs, stride) }
}

pub(super) unsafe fn silu_mul(gate: *mut f32, up: *const f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { silu_mul_impl(gate, up, n) }
}

pub(super) unsafe fn gelu_mul(gate: *mut f32, up: *const f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { gelu_mul_impl(gate, up, n) }
}

pub(super) unsafe fn softmax(x: *mut f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { softmax_impl(x, n) }
}

pub(super) unsafe fn quantize(
    x: *const f32,
    n: usize,
    group: usize,
    q: *mut i8,
    scale: *mut f32,
    sums: *mut i16,
    wsum: *mut f32,
) {
    // SAFETY: forwarded contract.
    unsafe { quantize_impl(x, n, group, q, scale, sums, wsum) }
}

/// Dot product of a row of float bytes and float activations.
///
/// `WIDTH` is the size of one stored element and `convert` decodes it. The
/// row is read bytewise, so it does not need to be aligned.
#[inline(always)]
unsafe fn row_float<const WIDTH: usize>(
    row: *const u8,
    x: *const f32,
    n: usize,
    convert: impl Fn(&[u8]) -> f32,
) -> f32 {
    // SAFETY: the caller guarantees `row` holds `n` elements of `WIDTH`
    // bytes and `x` is valid for `n` reads.
    let (row, x) = unsafe {
        (
            slice::from_raw_parts(row, n * WIDTH),
            slice::from_raw_parts(x, n),
        )
    };
    let mut lanes = [0.0f32; LANES];
    let mut raw_chunks = row.chunks_exact(WIDTH * LANES);
    let mut x_chunks = x.chunks_exact(LANES);
    for (raw, values) in (&mut raw_chunks).zip(&mut x_chunks) {
        for l in 0..LANES {
            lanes[l] += convert(&raw[l * WIDTH..(l + 1) * WIDTH]) * values[l];
        }
    }
    let mut sum = reduce(lanes);
    for (raw, value) in raw_chunks
        .remainder()
        .chunks_exact(WIDTH)
        .zip(x_chunks.remainder())
    {
        sum += convert(raw) * value;
    }
    sum
}

/// Row of little-endian f32 values.
pub(super) unsafe fn row_f32(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe {
        row_float::<4>(row, x, n, |raw| {
            f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
        })
    }
}

/// Row of little-endian half floats.
pub(super) unsafe fn row_f16(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe {
        row_float::<2>(row, x, n, |raw| {
            f16_to_f32(u16::from_le_bytes([raw[0], raw[1]]))
        })
    }
}

/// Row of little-endian brain floats.
pub(super) unsafe fn row_bf16(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe {
        row_float::<2>(row, x, n, |raw| {
            bf16_to_f32(u16::from_le_bytes([raw[0], raw[1]]))
        })
    }
}

/// Dot product of a quantized row and quantized activations, for any format.
///
/// Each block is decoded to integers, multiplied with the activation
/// integers sub-block by sub-block, and scaled back to floats:
///
/// ```text
/// sum over sub-blocks of (scale * sum(q * a) + offset * sum(a)) * act_scale
/// ```
#[inline(always)]
unsafe fn dot_blocks(
    decode: quant::Decoder,
    block_bytes: usize,
    group: usize,
    row: *const u8,
    n: usize,
    act: ActRef,
) -> f32 {
    let mut decoded = Group::new();
    let mut total = 0.0f32;
    for g in 0..n / group {
        // SAFETY: the caller guarantees `row` holds `n / group` blocks and
        // `act` describes `n` quantized activations.
        let (block, q, sums, act_scale) = unsafe {
            (
                slice::from_raw_parts(row.add(g * block_bytes), block_bytes),
                slice::from_raw_parts(act.q.add(g * group), group),
                slice::from_raw_parts(act.sums.add(g * group / 16), group / 16),
                *act.scale.add(g),
            )
        };
        decode(block, &mut decoded);
        let mut partial = 0.0f32;
        for sub in 0..group / 16 {
            let mut integer = 0i32;
            for l in 0..16 {
                integer += decoded.q[sub * 16 + l] as i32 * q[sub * 16 + l] as i32;
            }
            partial += decoded.scale[sub] * integer as f32 + decoded.offset[sub] * sums[sub] as f32;
        }
        total += partial * act_scale;
    }
    total
}

/// See [`DotDecoded`](super::DotDecoded). This is the definition the other
/// backends are tested against.
pub(super) unsafe fn dot_decoded(
    q: *const i8,
    scale: *const f32,
    n: usize,
    act: ActRef,
    group: usize,
) -> f32 {
    // SAFETY: the caller guarantees `n` weights, `n / 4` scales, and
    // activations quantized for `n` elements in groups of `group`.
    let (q, scale, a, act_scale) = unsafe {
        (
            slice::from_raw_parts(q, n),
            slice::from_raw_parts(scale, n / 4),
            slice::from_raw_parts(act.q, n),
            slice::from_raw_parts(act.scale, n / group),
        )
    };
    let mut total = 0.0f32;
    for (g, act_scale) in act_scale.iter().enumerate() {
        let mut partial = 0.0f32;
        for run in g * group / 4..(g + 1) * group / 4 {
            let mut integer = 0i32;
            for l in run * 4..run * 4 + 4 {
                integer += q[l] as i32 * a[l] as i32;
            }
            partial += scale[run] * integer as f32;
        }
        total += partial * act_scale;
    }
    total
}

/// Defines the row kernel of one quantized format on top of `dot_blocks`.
macro_rules! scalar_row {
    ($name:ident, $decode:path, $block_bytes:expr, $group:expr) => {
        unsafe fn $name(row: *const u8, n: usize, act: ActRef) -> f32 {
            // SAFETY: forwarded contract.
            unsafe { dot_blocks($decode, $block_bytes, $group, row, n, act) }
        }
    };
}

scalar_row!(row_q4_0, quant::decode_q4_0, 18, 32);
scalar_row!(row_q4_1, quant::decode_q4_1, 20, 32);
scalar_row!(row_q5_0, quant::decode_q5_0, 22, 32);
scalar_row!(row_q5_1, quant::decode_q5_1, 24, 32);
scalar_row!(row_q8_0, quant::decode_q8_0, 34, 32);
scalar_row!(row_iq4_nl, quant::decode_iq4_nl, 18, 32);
scalar_row!(row_q2_k, quant::decode_q2_k, 84, 256);
scalar_row!(row_q3_k, quant::decode_q3_k, 110, 256);
scalar_row!(row_q4_k, quant::decode_q4_k, 144, 256);
scalar_row!(row_q5_k, quant::decode_q5_k, 176, 256);
scalar_row!(row_q6_k, quant::decode_q6_k, 210, 256);

/// The scalar row kernel for a quantized type.
pub(super) fn row_quant(ty: TensorType) -> DotQuant {
    match ty {
        TensorType::Q4_0 => row_q4_0,
        TensorType::Q4_1 => row_q4_1,
        TensorType::Q5_0 => row_q5_0,
        TensorType::Q5_1 => row_q5_1,
        TensorType::Q8_0 => row_q8_0,
        TensorType::Iq4Nl => row_iq4_nl,
        TensorType::Q2K => row_q2_k,
        TensorType::Q3K => row_q3_k,
        TensorType::Q4K => row_q4_k,
        TensorType::Q5K => row_q5_k,
        TensorType::Q6K => row_q6_k,
        TensorType::F32 | TensorType::F16 | TensorType::BF16 => {
            unreachable!("float types use the float row kernels")
        }
    }
}

/// The row kernels of [`EXACT`].
mod exact {
    use std::cell::RefCell;
    use std::slice;

    use super::dot_impl;
    use crate::kernels::{ActRef, DotQuant};
    use crate::tensor::TensorType;
    use crate::tensor::quant::dequantize_row;

    thread_local! {
        /// The expanded weight row, reused between calls.
        static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }

    /// Expands a row of type `ty` and multiplies it with float activations.
    ///
    /// The activations arrive in `act.wsum`, which the forward pass points
    /// at the unrounded input when it runs with these kernels.
    unsafe fn dot_expanded(ty: TensorType, row: *const u8, n: usize, act: ActRef) -> f32 {
        ROW.with_borrow_mut(|weights| {
            weights.resize(n, 0.0);
            // SAFETY: the caller guarantees `row` holds `n` weights of type
            // `ty` and `act.wsum` holds `n` floats.
            unsafe {
                let bytes = slice::from_raw_parts(row, ty.row_bytes(n).unwrap_or(0));
                dequantize_row(ty, bytes, weights);
                dot_impl(weights.as_ptr(), act.wsum, n)
            }
        })
    }

    /// Defines the exact row kernel of one quantized format.
    macro_rules! exact_row {
        ($name:ident, $ty:expr) => {
            unsafe fn $name(row: *const u8, n: usize, act: ActRef) -> f32 {
                // SAFETY: forwarded contract.
                unsafe { dot_expanded($ty, row, n, act) }
            }
        };
    }

    exact_row!(row_q4_0, TensorType::Q4_0);
    exact_row!(row_q4_1, TensorType::Q4_1);
    exact_row!(row_q5_0, TensorType::Q5_0);
    exact_row!(row_q5_1, TensorType::Q5_1);
    exact_row!(row_q8_0, TensorType::Q8_0);
    exact_row!(row_iq4_nl, TensorType::Iq4Nl);
    exact_row!(row_q2_k, TensorType::Q2K);
    exact_row!(row_q3_k, TensorType::Q3K);
    exact_row!(row_q4_k, TensorType::Q4K);
    exact_row!(row_q5_k, TensorType::Q5K);
    exact_row!(row_q6_k, TensorType::Q6K);

    pub fn row_quant(ty: TensorType) -> DotQuant {
        match ty {
            TensorType::Q4_0 => row_q4_0,
            TensorType::Q4_1 => row_q4_1,
            TensorType::Q5_0 => row_q5_0,
            TensorType::Q5_1 => row_q5_1,
            TensorType::Q8_0 => row_q8_0,
            TensorType::Iq4Nl => row_iq4_nl,
            TensorType::Q2K => row_q2_k,
            TensorType::Q3K => row_q3_k,
            TensorType::Q4K => row_q4_k,
            TensorType::Q5K => row_q5_k,
            TensorType::Q6K => row_q6_k,
            TensorType::F32 | TensorType::F16 | TensorType::BF16 => {
                unreachable!("float types use the float row kernels")
            }
        }
    }
}

//! AVX2 kernels for x86-64.
//!
//! Every kernel follows the same plan. Weights are unpacked to one byte each
//! in a 256-bit register, 32 at a time. `maddubs` multiplies them with 32
//! activation bytes and adds neighbors into 16-bit sums, and `madd` widens
//! those to 32 bits while applying the sub-block scales. Only the last step
//! of a block touches floating point.
//!
//! All functions here require AVX2, FMA and F16C. [`supported`] checks for
//! them once, and nothing else may call into this module.

// Intrinsics that only compute on registers are called outside `unsafe`
// blocks here. Compilers before Rust 1.87 declare those intrinsics unsafe
// too and would warn about every such call. Pointer reads and writes still
// sit in `unsafe` blocks with a SAFETY comment.
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use super::scalar;
use super::{ActRef, DotQuant, Kernels};
use crate::tensor::TensorType;
use crate::tensor::f16::f16_to_f32;

pub static KERNELS: Kernels = Kernels {
    name: "avx2",
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
};

/// True when the running CPU has every feature these kernels use.
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

/// The row kernel for a quantized type. Formats without an AVX2 kernel fall
/// back to the scalar one.
fn row_quant(ty: TensorType) -> DotQuant {
    match ty {
        TensorType::Q4_0 => row_q4_0,
        TensorType::Q4_1 => row_q4_1,
        TensorType::Q5_0 => row_q5_0,
        TensorType::Q5_1 => row_q5_1,
        TensorType::Q8_0 => row_q8_0,
        TensorType::Q4K => row_q4_k,
        TensorType::Q5K => row_q5_k,
        TensorType::Q6K => row_q6_k,
        _ => scalar::row_quant(ty),
    }
}

// ---------------------------------------------------------------------------
// Float vectors
// ---------------------------------------------------------------------------

/// Adds the eight lanes of a float register.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn hsum(v: __m256) -> f32 {
    let quad = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    let single = _mm_add_ss(pair, _mm_shuffle_ps::<1>(pair, pair));
    _mm_cvtss_f32(single)
}

/// Dot product of two float vectors. Neither pointer needs to be aligned.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    // SAFETY: the caller guarantees both pointers are valid for `n` reads,
    // and every access below stays under `n`.
    unsafe {
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 8)),
                _mm256_loadu_ps(b.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 16)),
                _mm256_loadu_ps(b.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.add(i + 24)),
                _mm256_loadu_ps(b.add(i + 24)),
                acc3,
            );
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), acc0);
            i += 8;
        }
        let mut sum = hsum(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ));
        while i < n {
            sum += a.add(i).read_unaligned() * b.add(i).read_unaligned();
            i += 1;
        }
        sum
    }
}

/// `y += alpha * x`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn axpy(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
    let factor = _mm256_set1_ps(alpha);
    let mut i = 0;
    // SAFETY: the caller guarantees `x` is valid for `n` reads and `y` for
    // `n` writes.
    unsafe {
        while i + 8 <= n {
            let updated =
                _mm256_fmadd_ps(factor, _mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i)));
            _mm256_storeu_ps(y.add(i), updated);
            i += 8;
        }
        while i < n {
            *y.add(i) += alpha * *x.add(i);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Scores `n` rows against `G` queries. Each row is loaded once and shared
/// by the queries, which is what makes grouped-query attention cheap.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_rows_group<const G: usize>(
    queries: *const f32,
    rows: *const f32,
    n: usize,
    len: usize,
    scores: *mut f32,
    stride: usize,
) {
    // SAFETY: the caller guarantees `G` queries and `n` rows of `len`
    // floats, and `G` score runs `stride` apart with room for `n` scores.
    unsafe {
        for j in 0..n {
            let row = rows.add(j * len);
            let mut acc = [_mm256_setzero_ps(); G];
            let mut i = 0;
            while i + 8 <= len {
                let values = _mm256_loadu_ps(row.add(i));
                for (g, acc) in acc.iter_mut().enumerate() {
                    *acc = _mm256_fmadd_ps(values, _mm256_loadu_ps(queries.add(g * len + i)), *acc);
                }
                i += 8;
            }
            for (g, acc) in acc.iter().enumerate() {
                let mut sum = hsum(*acc);
                for t in i..len {
                    sum += *row.add(t) * *queries.add(g * len + t);
                }
                *scores.add(g * stride + j) = sum;
            }
        }
    }
}

/// See [`Kernels::dot_rows`]. Queries are taken four at a time.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_rows(
    queries: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    scores: *mut f32,
    stride: usize,
) {
    let mut h = 0;
    // SAFETY: forwarded contract, one group of queries at a time.
    unsafe {
        while h < n_queries {
            let (q, s) = (queries.add(h * len), scores.add(h * stride));
            match n_queries - h {
                1 => dot_rows_group::<1>(q, rows, n, len, s, stride),
                2 => dot_rows_group::<2>(q, rows, n, len, s, stride),
                3 => dot_rows_group::<3>(q, rows, n, len, s, stride),
                _ => dot_rows_group::<4>(q, rows, n, len, s, stride),
            }
            h += 4;
        }
    }
}

/// Adds the weighted rows to `G` outputs, `8 * K` floats at a time. The
/// partial outputs live in registers across all rows, so no row waits on a
/// store from the row before it.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn mix_rows_block<const G: usize, const K: usize>(
    weights: *const f32,
    rows: *const f32,
    n: usize,
    len: usize,
    outs: *mut f32,
    stride: usize,
) {
    // SAFETY: the caller guarantees `G` outputs with `8 * K` writable floats
    // at `outs + g * len`, `n` rows with `8 * K` readable floats at
    // `rows + j * len`, and `G` weight runs `stride` apart.
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); K]; G];
        for (g, out) in acc.iter_mut().enumerate() {
            for (k, part) in out.iter_mut().enumerate() {
                *part = _mm256_loadu_ps(outs.add(g * len + k * 8));
            }
        }
        for j in 0..n {
            let row = rows.add(j * len);
            let mut values = [_mm256_setzero_ps(); K];
            for (k, value) in values.iter_mut().enumerate() {
                *value = _mm256_loadu_ps(row.add(k * 8));
            }
            for (g, out) in acc.iter_mut().enumerate() {
                let weight = broadcast(weights.add(g * stride + j));
                for (part, value) in out.iter_mut().zip(&values) {
                    *part = _mm256_fmadd_ps(weight, *value, *part);
                }
            }
        }
        for (g, out) in acc.iter().enumerate() {
            for (k, part) in out.iter().enumerate() {
                _mm256_storeu_ps(outs.add(g * len + k * 8), *part);
            }
        }
    }
}

/// See [`Kernels::mix_rows`]. Outputs are taken three at a time and 32
/// floats at a time, which fills the sixteen registers.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mix_rows(
    weights: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    outs: *mut f32,
    stride: usize,
) {
    let mut h = 0;
    // SAFETY: forwarded contract. Each call covers a block of columns of a
    // group of outputs, so the pointers are offset by the block's start.
    unsafe {
        while h < n_queries {
            let group = (n_queries - h).min(3);
            let (w, o) = (weights.add(h * stride), outs.add(h * len));
            let mut i = 0;
            while i + 32 <= len {
                match group {
                    1 => mix_rows_block::<1, 4>(w, rows.add(i), n, len, o.add(i), stride),
                    2 => mix_rows_block::<2, 4>(w, rows.add(i), n, len, o.add(i), stride),
                    _ => mix_rows_block::<3, 4>(w, rows.add(i), n, len, o.add(i), stride),
                }
                i += 32;
            }
            while i + 8 <= len {
                match group {
                    1 => mix_rows_block::<1, 1>(w, rows.add(i), n, len, o.add(i), stride),
                    2 => mix_rows_block::<2, 1>(w, rows.add(i), n, len, o.add(i), stride),
                    _ => mix_rows_block::<3, 1>(w, rows.add(i), n, len, o.add(i), stride),
                }
                i += 8;
            }
            // The last few columns, one float at a time.
            for g in 0..group {
                for j in 0..n {
                    let weight = *w.add(g * stride + j);
                    for t in i..len {
                        *o.add(g * len + t) += weight * *rows.add(j * len + t);
                    }
                }
            }
            h += group;
        }
    }
}

// The element-wise kernels reuse the portable loops. Compiled here with AVX2
// enabled, the compiler vectorizes them eight floats at a time.

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn sum_squares(x: *const f32, n: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe { scalar::sum_squares_impl(x, n) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn silu_mul(gate: *mut f32, up: *const f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { scalar::silu_mul_impl(gate, up, n) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gelu_mul(gate: *mut f32, up: *const f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { scalar::gelu_mul_impl(gate, up, n) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn softmax(x: *mut f32, n: usize) {
    // SAFETY: forwarded contract.
    unsafe { scalar::softmax_impl(x, n) }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn quantize(
    x: *const f32,
    n: usize,
    group: usize,
    q: *mut i8,
    scale: *mut f32,
    sums: *mut i16,
    wsum: *mut f32,
) {
    // SAFETY: forwarded contract.
    unsafe { scalar::quantize_impl(x, n, group, q, scale, sums, wsum) }
}

// ---------------------------------------------------------------------------
// Float weight rows
// ---------------------------------------------------------------------------

/// Row of f32 bytes times float activations.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_f32(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: `dot` reads both sides unaligned.
    unsafe { dot(row as *const f32, x, n) }
}

/// Row of half floats times float activations, widened with F16C.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_f16(row: *const u8, x: *const f32, n: usize) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    // SAFETY: the caller guarantees `row` holds `n` halves and `x` is valid
    // for `n` reads.
    unsafe {
        while i + 16 <= n {
            let w0 = _mm256_cvtph_ps(_mm_loadu_si128(row.add(i * 2) as *const __m128i));
            let w1 = _mm256_cvtph_ps(_mm_loadu_si128(row.add(i * 2 + 16) as *const __m128i));
            acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.add(i + 8)), acc1);
            i += 16;
        }
        let mut sum = hsum(_mm256_add_ps(acc0, acc1));
        while i < n {
            let half = (row.add(i * 2) as *const u16).read_unaligned();
            sum += f16_to_f32(half) * *x.add(i);
            i += 1;
        }
        sum
    }
}

/// Row of brain floats times float activations.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_bf16(row: *const u8, x: *const f32, n: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    // SAFETY: as in `row_f16`.
    unsafe {
        while i + 8 <= n {
            let halves = _mm_loadu_si128(row.add(i * 2) as *const __m128i);
            // A brain float is the top 16 bits of an f32.
            let w = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(halves)));
            acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x.add(i)), acc);
            i += 8;
        }
        let mut sum = hsum(acc);
        while i < n {
            let half = (row.add(i * 2) as *const u16).read_unaligned();
            sum += f32::from_bits((half as u32) << 16) * *x.add(i);
            i += 1;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// Shared pieces of the quantized kernels
// ---------------------------------------------------------------------------

/// Reads the half float scale at `p` into all eight lanes.
///
/// The value is broadcast before it is converted. Converting a single lane
/// compiles to an insert into whatever register is at hand, which makes
/// every block wait for an accumulator it has nothing to do with.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn half(p: *const u8) -> __m256 {
    // SAFETY: the caller guarantees two readable bytes at `p`.
    let bits = unsafe { (p as *const i16).read_unaligned() };
    _mm256_cvtph_ps(_mm_set1_epi16(bits))
}

/// Reads the float at `p` into all eight lanes.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn broadcast(p: *const f32) -> __m256 {
    // SAFETY: the caller guarantees a readable float at `p`.
    unsafe { _mm256_broadcast_ss(&*p) }
}

/// Reads an unaligned little-endian u32 at `p`.
#[inline]
unsafe fn word(p: *const u8) -> u32 {
    // SAFETY: the caller guarantees four readable bytes at `p`.
    unsafe { (p as *const u32).read_unaligned() }
}

/// Unpacks 16 bytes of nibbles into 32 bytes: the 16 low nibbles in the low
/// half, the 16 high nibbles in the high half.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn nibbles(p: *const u8) -> __m256i {
    // SAFETY: the caller guarantees 16 readable bytes at `p`.
    let bytes = unsafe { _mm_loadu_si128(p as *const __m128i) };
    let both = _mm256_set_m128i(_mm_srli_epi16::<4>(bytes), bytes);
    _mm256_and_si256(both, _mm256_set1_epi8(0x0f))
}

/// Expands the 32 bits at `p` into 32 bytes: 0x10 where the bit is set.
/// These are the fifth bits of the 5-bit formats.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn fifth_bits(p: *const u8) -> __m256i {
    // SAFETY: the caller guarantees four readable bytes at `p`.
    let bits = unsafe { word(p) };
    // Byte `i` of the result first receives the source byte holding bit `i`.
    let byte_of_bit = _mm256_set_epi64x(
        0x0303_0303_0303_0303,
        0x0202_0202_0202_0202,
        0x0101_0101_0101_0101,
        0,
    );
    let spread = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), byte_of_bit);
    // Then each byte keeps only its own bit.
    let bit_in_byte = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let is_set = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit_in_byte), bit_in_byte);
    _mm256_and_si256(is_set, _mm256_set1_epi8(0x10))
}

/// Like [`fifth_bits`], for values that are stored with a bias of 16: the
/// result is 0xF0 where the bit is clear. Combined with a nibble by OR, that
/// reads as `nibble - 16` in a signed byte, and a set bit leaves `nibble`,
/// which is `nibble + 16 - 16`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn fifth_bits_biased(p: *const u8) -> __m256i {
    // SAFETY: the caller guarantees four readable bytes at `p`.
    let bits = unsafe { word(p) };
    let byte_of_bit = _mm256_set_epi64x(
        0x0303_0303_0303_0303,
        0x0202_0202_0202_0202,
        0x0101_0101_0101_0101,
        0,
    );
    let spread = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), byte_of_bit);
    let bit_in_byte = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
    let is_set = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit_in_byte), bit_in_byte);
    _mm256_andnot_si256(is_set, _mm256_set1_epi8(0xf0u8 as i8))
}

/// Sum of 32 products of unsigned weights and signed activations, spread
/// over eight float lanes. Weights must be at most 127.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_unsigned(weights: __m256i, acts: __m256i) -> __m256 {
    let pairs = _mm256_maddubs_epi16(weights, acts);
    _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

/// Sum of 32 products of signed weights and signed activations. `maddubs`
/// needs an unsigned first operand, so the weight signs move to the
/// activations.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_signed(weights: __m256i, acts: __m256i) -> __m256 {
    let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(weights), _mm256_sign_epi8(acts, weights));
    _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
}

// ---------------------------------------------------------------------------
// Formats with 32-weight blocks
// ---------------------------------------------------------------------------

/// One unpacked block of a format without an offset:
/// `weight[i] = scale * weights[i]`, with signed weights.
struct SignedBlock {
    weights: __m256i,
    /// The scale, in every lane.
    scale: __m256,
}

/// One unpacked block of a format with an offset:
/// `weight[i] = scale * weights[i] + offset`, with unsigned weights.
struct OffsetBlock {
    weights: __m256i,
    /// The scale, in every lane.
    scale: __m256,
    offset: f32,
}

/// Q4_0: `weight = scale * (nibble - 8)`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_q4_0(p: *const u8) -> SignedBlock {
    // SAFETY: the caller guarantees an 18-byte block at `p`.
    unsafe {
        SignedBlock {
            weights: _mm256_sub_epi8(nibbles(p.add(2)), _mm256_set1_epi8(8)),
            scale: half(p),
        }
    }
}

/// Q5_0: `weight = scale * (value - 16)`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_q5_0(p: *const u8) -> SignedBlock {
    // SAFETY: the caller guarantees a 22-byte block at `p`.
    unsafe {
        let weights = _mm256_or_si256(nibbles(p.add(6)), fifth_bits_biased(p.add(2)));
        SignedBlock {
            weights,
            scale: half(p),
        }
    }
}

/// Q8_0: `weight = scale * byte`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_q8_0(p: *const u8) -> SignedBlock {
    // SAFETY: the caller guarantees a 34-byte block at `p`.
    unsafe {
        SignedBlock {
            weights: _mm256_loadu_si256(p.add(2) as *const __m256i),
            scale: half(p),
        }
    }
}

/// Q4_1: `weight = scale * nibble + minimum`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_q4_1(p: *const u8) -> OffsetBlock {
    // SAFETY: the caller guarantees a 20-byte block at `p`.
    unsafe {
        OffsetBlock {
            weights: nibbles(p.add(4)),
            scale: half(p),
            offset: _mm256_cvtss_f32(half(p.add(2))),
        }
    }
}

/// Q5_1: `weight = scale * value + minimum`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_q5_1(p: *const u8) -> OffsetBlock {
    // SAFETY: the caller guarantees a 24-byte block at `p`.
    unsafe {
        let weights = _mm256_or_si256(nibbles(p.add(8)), fifth_bits(p.add(4)));
        OffsetBlock {
            weights,
            scale: half(p),
            offset: _mm256_cvtss_f32(half(p.add(2))),
        }
    }
}

/// Defines the row kernel of a format whose blocks unpack to signed
/// weights and a scale.
///
/// The dot product of one block is `scale * act_scale * sum(w * a)`. Blocks
/// alternate between two accumulators so consecutive blocks do not wait on
/// each other.
macro_rules! signed_block_row {
    ($name:ident, $load:ident, $block_bytes:expr) => {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn $name(row: *const u8, n: usize, act: ActRef) -> f32 {
            let blocks = n / 32;
            let mut acc = [_mm256_setzero_ps(), _mm256_setzero_ps()];
            let mut b = 0;
            // SAFETY: the caller guarantees `row` holds `blocks` blocks and
            // `act` describes `n` quantized activations.
            unsafe {
                while b < blocks {
                    for lane in 0..2 {
                        if b + lane >= blocks {
                            break;
                        }
                        let index = b + lane;
                        let block = $load(row.add(index * $block_bytes));
                        let acts = _mm256_loadu_si256(act.q.add(index * 32) as *const __m256i);
                        let scale = _mm256_mul_ps(block.scale, broadcast(act.scale.add(index)));
                        acc[lane] =
                            _mm256_fmadd_ps(scale, dot_signed(block.weights, acts), acc[lane]);
                    }
                    b += 2;
                }
                hsum(_mm256_add_ps(acc[0], acc[1]))
            }
        }
    };
}

/// Defines the row kernel of a format whose blocks unpack to unsigned
/// weights with a scale and an offset.
///
/// The dot product of one block is
/// `scale * act_scale * sum(w * a) + offset * act_scale * sum(a)`.
/// The last factor is precomputed per block as `act.wsum`.
macro_rules! offset_block_row {
    ($name:ident, $load:ident, $block_bytes:expr) => {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn $name(row: *const u8, n: usize, act: ActRef) -> f32 {
            let blocks = n / 32;
            let mut acc = [_mm256_setzero_ps(), _mm256_setzero_ps()];
            let mut correction = [0.0f32, 0.0];
            let mut b = 0;
            // SAFETY: the caller guarantees `row` holds `blocks` blocks and
            // `act` describes `n` quantized activations.
            unsafe {
                while b < blocks {
                    for lane in 0..2 {
                        if b + lane >= blocks {
                            break;
                        }
                        let index = b + lane;
                        let block = $load(row.add(index * $block_bytes));
                        let acts = _mm256_loadu_si256(act.q.add(index * 32) as *const __m256i);
                        let scale = _mm256_mul_ps(block.scale, broadcast(act.scale.add(index)));
                        acc[lane] =
                            _mm256_fmadd_ps(scale, dot_unsigned(block.weights, acts), acc[lane]);
                        correction[lane] += block.offset * *act.wsum.add(index);
                    }
                    b += 2;
                }
                hsum(_mm256_add_ps(acc[0], acc[1])) + (correction[0] + correction[1])
            }
        }
    };
}

signed_block_row!(row_q4_0, load_q4_0, 18);
signed_block_row!(row_q5_0, load_q5_0, 22);
signed_block_row!(row_q8_0, load_q8_0, 34);
offset_block_row!(row_q4_1, load_q4_1, 20);
offset_block_row!(row_q5_1, load_q5_1, 24);

// ---------------------------------------------------------------------------
// Formats with 256-weight super-blocks
// ---------------------------------------------------------------------------

/// Shuffle masks that broadcast one 16-bit scale to a whole register.
/// Mask `j` copies scale `j` of eight. The source register must hold the
/// same eight scales in both halves.
static SPREAD_SCALE_32: [[u8; 32]; 8] = {
    let mut masks = [[0u8; 32]; 8];
    let mut j = 0;
    while j < 8 {
        let mut i = 0;
        while i < 32 {
            masks[j][i] = (2 * j + (i & 1)) as u8;
            i += 1;
        }
        j += 1;
    }
    masks
};

/// Shuffle masks for sub-blocks of 16 weights. Mask `k` copies scale `2k`
/// to the low half of the register and scale `2k + 1` to the high half.
static SPREAD_SCALE_16: [[u8; 32]; 4] = {
    let mut masks = [[0u8; 32]; 4];
    let mut k = 0;
    while k < 4 {
        let mut i = 0;
        while i < 32 {
            let scale = 2 * k + (i / 16);
            masks[k][i] = (2 * scale + (i & 1)) as u8;
            i += 1;
        }
        k += 1;
    }
    masks
};

/// Loads one of the shuffle masks above.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn mask(table: &[u8; 32]) -> __m256i {
    // SAFETY: the table is 32 readable bytes.
    unsafe { _mm256_loadu_si256(table.as_ptr() as *const __m256i) }
}

/// The scales and minimums of a Q4_K or Q5_K super-block.
struct K4Scales {
    /// Eight 16-bit scales, repeated in both halves of the register.
    scales: __m256i,
    /// Eight 16-bit minimums.
    minimums: __m128i,
}

/// Unpacks the eight 6-bit scales and eight 6-bit minimums stored in the 12
/// bytes at `p`. The first four of each are the low 6 bits of bytes 0..8.
/// The last four take 4 bits from bytes 8..12 and 2 bits from the tops of
/// bytes 0..8.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn k4_scales(p: *const u8) -> K4Scales {
    // SAFETY: the caller guarantees 12 readable bytes at `p`.
    let (w0, w1, w2) = unsafe { (word(p), word(p.add(4)), word(p.add(8))) };
    let scales_low = w0 & 0x3f3f_3f3f;
    let minimums_low = w1 & 0x3f3f_3f3f;
    let scales_high = (w2 & 0x0f0f_0f0f) | (((w0 >> 6) & 0x0303_0303) << 4);
    let minimums_high = ((w2 >> 4) & 0x0f0f_0f0f) | (((w1 >> 6) & 0x0303_0303) << 4);
    let packed = _mm_set_epi32(
        minimums_high as i32,
        minimums_low as i32,
        scales_high as i32,
        scales_low as i32,
    );
    let widened = _mm256_cvtepu8_epi16(packed);
    let scales = _mm256_castsi256_si128(widened);
    K4Scales {
        scales: _mm256_broadcastsi128_si256(scales),
        minimums: _mm256_extracti128_si256::<1>(widened),
    }
}

/// The minimum term of a Q4_K or Q5_K super-block: the sum over its eight
/// sub-blocks of `minimum * sum(activations)`, as four integer lanes.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn k4_minimum_term(minimums: __m128i, sums: *const i16) -> __m128i {
    // SAFETY: the caller guarantees 16 readable sums.
    let sums16 = unsafe { _mm256_loadu_si256(sums as *const __m256i) };
    // Sub-blocks are 32 wide, so add the sums of 16 in pairs.
    let sums32 = _mm_hadd_epi16(
        _mm256_castsi256_si128(sums16),
        _mm256_extracti128_si256::<1>(sums16),
    );
    _mm_madd_epi16(minimums, sums32)
}

/// Q4_K: 144-byte super-blocks of eight 32-weight sub-blocks.
/// `weight = d * scale * nibble - dmin * minimum`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_q4_k(row: *const u8, n: usize, act: ActRef) -> f32 {
    const BLOCK_BYTES: usize = 144;
    let low_nibble = _mm256_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    let mut acc_min = _mm_setzero_ps();
    // SAFETY: the caller guarantees `row` holds `n / 256` super-blocks and
    // `act` describes `n` quantized activations.
    unsafe {
        for sb in 0..n / 256 {
            let p = row.add(sb * BLOCK_BYTES);
            let acts = act.q.add(sb * 256);
            let act_scale = broadcast(act.scale.add(sb));
            let d = _mm256_mul_ps(half(p), act_scale);
            let dmin = _mm256_castps256_ps128(_mm256_mul_ps(half(p.add(2)), act_scale));

            let k4 = k4_scales(p.add(4));
            let minimum_term = k4_minimum_term(k4.minimums, act.sums.add(sb * 16));
            acc_min = _mm_fnmadd_ps(dmin, _mm_cvtepi32_ps(minimum_term), acc_min);

            let mut sum = _mm256_setzero_si256();
            for pair in 0..4 {
                // 32 bytes hold two sub-blocks: low nibbles, then high nibbles.
                let bits = _mm256_loadu_si256(p.add(16 + pair * 32) as *const __m256i);
                let low = _mm256_and_si256(bits, low_nibble);
                let high = _mm256_and_si256(_mm256_srli_epi16::<4>(bits), low_nibble);
                let acts_low = _mm256_loadu_si256(acts.add(pair * 64) as *const __m256i);
                let acts_high = _mm256_loadu_si256(acts.add(pair * 64 + 32) as *const __m256i);
                let scale_low = _mm256_shuffle_epi8(k4.scales, mask(&SPREAD_SCALE_32[pair * 2]));
                let scale_high =
                    _mm256_shuffle_epi8(k4.scales, mask(&SPREAD_SCALE_32[pair * 2 + 1]));
                let term_low = _mm256_madd_epi16(_mm256_maddubs_epi16(low, acts_low), scale_low);
                let term_high =
                    _mm256_madd_epi16(_mm256_maddubs_epi16(high, acts_high), scale_high);
                sum = _mm256_add_epi32(sum, _mm256_add_epi32(term_low, term_high));
            }
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum), acc);
        }
        hsum(acc) + hsum(_mm256_zextps128_ps256(acc_min))
    }
}

/// Q5_K: 176-byte super-blocks. Like Q4_K with a fifth bit per weight,
/// stored as 32 bytes of bit pairs after the scales.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_q5_k(row: *const u8, n: usize, act: ActRef) -> f32 {
    const BLOCK_BYTES: usize = 176;
    let low_nibble = _mm256_set1_epi8(0x0f);
    let bit0 = _mm256_set1_epi8(1);
    let bit1 = _mm256_set1_epi8(2);
    let mut acc = _mm256_setzero_ps();
    let mut acc_min = _mm_setzero_ps();
    // SAFETY: as in `row_q4_k`.
    unsafe {
        for sb in 0..n / 256 {
            let p = row.add(sb * BLOCK_BYTES);
            let acts = act.q.add(sb * 256);
            let act_scale = broadcast(act.scale.add(sb));
            let d = _mm256_mul_ps(half(p), act_scale);
            let dmin = _mm256_castps256_ps128(_mm256_mul_ps(half(p.add(2)), act_scale));

            let k4 = k4_scales(p.add(4));
            let minimum_term = k4_minimum_term(k4.minimums, act.sums.add(sb * 16));
            acc_min = _mm_fnmadd_ps(dmin, _mm_cvtepi32_ps(minimum_term), acc_min);

            // Byte `l` of the high bits holds weight `l` of all eight
            // sub-blocks, two bits per pair. Shifting right by two moves the
            // next pair's bits into place.
            let mut high_bits = _mm256_loadu_si256(p.add(16) as *const __m256i);
            let mut sum = _mm256_setzero_si256();
            for pair in 0..4 {
                let bits = _mm256_loadu_si256(p.add(48 + pair * 32) as *const __m256i);
                let fifth_low = _mm256_slli_epi16::<4>(_mm256_and_si256(high_bits, bit0));
                let fifth_high = _mm256_slli_epi16::<3>(_mm256_and_si256(high_bits, bit1));
                high_bits = _mm256_srli_epi16::<2>(high_bits);
                let low = _mm256_or_si256(_mm256_and_si256(bits, low_nibble), fifth_low);
                let high = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(bits), low_nibble),
                    fifth_high,
                );
                let acts_low = _mm256_loadu_si256(acts.add(pair * 64) as *const __m256i);
                let acts_high = _mm256_loadu_si256(acts.add(pair * 64 + 32) as *const __m256i);
                let scale_low = _mm256_shuffle_epi8(k4.scales, mask(&SPREAD_SCALE_32[pair * 2]));
                let scale_high =
                    _mm256_shuffle_epi8(k4.scales, mask(&SPREAD_SCALE_32[pair * 2 + 1]));
                let term_low = _mm256_madd_epi16(_mm256_maddubs_epi16(low, acts_low), scale_low);
                let term_high =
                    _mm256_madd_epi16(_mm256_maddubs_epi16(high, acts_high), scale_high);
                sum = _mm256_add_epi32(sum, _mm256_add_epi32(term_low, term_high));
            }
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum), acc);
        }
        hsum(acc) + hsum(_mm256_zextps128_ps256(acc_min))
    }
}

/// Q6_K: 210-byte super-blocks of sixteen 16-weight sub-blocks.
/// `weight = d * scale * (value - 32)`.
///
/// The values are kept unsigned, 0 to 63, so `maddubs` can take them
/// directly. The `- 32` is applied afterwards as `32 * scale * sum(acts)`.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_q6_k(row: *const u8, n: usize, act: ActRef) -> f32 {
    const BLOCK_BYTES: usize = 210;
    let low_nibble = _mm256_set1_epi8(0x0f);
    let two_bits = _mm256_set1_epi8(3);
    let mut acc = _mm256_setzero_ps();
    // SAFETY: as in `row_q4_k`.
    unsafe {
        for sb in 0..n / 256 {
            let p = row.add(sb * BLOCK_BYTES);
            let acts = act.q.add(sb * 256);
            let d = _mm256_mul_ps(half(p.add(208)), broadcast(act.scale.add(sb)));

            let scales = _mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(192) as *const __m128i));
            let sums = _mm256_loadu_si256(act.sums.add(sb * 16) as *const __m256i);
            let bias = _mm256_slli_epi32::<5>(_mm256_madd_epi16(scales, sums));
            let mut sum = _mm256_sub_epi32(_mm256_setzero_si256(), bias);

            for half_index in 0..2 {
                // The eight scales of this half, repeated in both halves of
                // the register so the shuffles can reach them.
                let half_scales = if half_index == 0 {
                    _mm256_permute2x128_si256::<0x00>(scales, scales)
                } else {
                    _mm256_permute2x128_si256::<0x11>(scales, scales)
                };
                let first = _mm256_loadu_si256(p.add(half_index * 64) as *const __m256i);
                let second = _mm256_loadu_si256(p.add(half_index * 64 + 32) as *const __m256i);
                let high = _mm256_loadu_si256(p.add(128 + half_index * 32) as *const __m256i);

                // Four runs of 32 weights. Each takes a nibble from `first`
                // or `second` and two more bits from `high`.
                let top0 = _mm256_slli_epi16::<4>(_mm256_and_si256(high, two_bits));
                let top1 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                    _mm256_srli_epi16::<2>(high),
                    two_bits,
                ));
                let top2 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                    _mm256_srli_epi16::<4>(high),
                    two_bits,
                ));
                let top3 = _mm256_slli_epi16::<4>(_mm256_and_si256(
                    _mm256_srli_epi16::<6>(high),
                    two_bits,
                ));
                let values = [
                    _mm256_or_si256(_mm256_and_si256(first, low_nibble), top0),
                    _mm256_or_si256(_mm256_and_si256(second, low_nibble), top1),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(first), low_nibble),
                        top2,
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(second), low_nibble),
                        top3,
                    ),
                ];
                for (run, value) in values.into_iter().enumerate() {
                    let run_acts =
                        _mm256_loadu_si256(acts.add(half_index * 128 + run * 32) as *const __m256i);
                    let run_scales = _mm256_shuffle_epi8(half_scales, mask(&SPREAD_SCALE_16[run]));
                    let term = _mm256_madd_epi16(_mm256_maddubs_epi16(value, run_acts), run_scales);
                    sum = _mm256_add_epi32(sum, term);
                }
            }
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sum), acc);
        }
        hsum(acc)
    }
}

//! NEON kernels for AArch64.
//!
//! The plan matches the AVX2 kernels with 128-bit registers: unpack 16
//! weights to one byte each, multiply them with 16 activation bytes, and
//! accumulate 32-bit sums. Only the last step of a block is floating point.
//!
//! The byte multiply comes in two forms, chosen when the process starts:
//!
//! | Form       | Instruction                 | CPUs                         |
//! |------------|-----------------------------|------------------------------|
//! | `Widening` | `smull` and pairwise adds   | Every AArch64 CPU.           |
//! | `Sdot`     | `sdot`, 16 products at once | ARMv8.2 with the dot product |
//! |            |                             | extension, such as the       |
//! |            |                             | Cortex-A76 in a Raspberry Pi |
//! |            |                             | 5.                           |
//!
//! Each kernel is written once, generic over the form, and instantiated
//! twice. NEON itself is part of the AArch64 baseline.

use std::arch::aarch64::*;
use std::arch::asm;

use super::scalar;
use super::{ActRef, DotDecoded, DotQuant, Kernels};
use crate::tensor::TensorType;

/// Kernels for every AArch64 CPU.
pub static KERNELS: Kernels = kernels("neon", row_quant_widening, dot_decoded);

/// Kernels for CPUs with the dot product extension.
pub static KERNELS_DOTPROD: Kernels = kernels("neon+dotprod", row_quant_sdot, dot_decoded_sdot);

/// Builds a kernel table. Only the kernels that multiply bytes differ
/// between the two tables.
const fn kernels(
    name: &'static str,
    row_quant: fn(TensorType) -> DotQuant,
    dot_decoded: DotDecoded,
) -> Kernels {
    Kernels {
        name,
        dot,
        axpy,
        // NEON is always on, so the portable loops already vectorize.
        sum_squares: scalar::sum_squares,
        dot_rows: scalar::dot_rows,
        mix_rows: scalar::mix_rows,
        silu_mul: scalar::silu_mul,
        gelu_mul: scalar::gelu_mul,
        softmax: scalar::softmax,
        quantize: scalar::quantize,
        row_f32,
        row_f16,
        row_bf16,
        row_quant,
        dot_decoded,
    }
}

/// Row kernels built on widening multiplies. Formats without a NEON kernel
/// fall back to the scalar one.
fn row_quant_widening(ty: TensorType) -> DotQuant {
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

/// Row kernels built on the `sdot` instruction.
fn row_quant_sdot(ty: TensorType) -> DotQuant {
    match ty {
        TensorType::Q4_0 => row_q4_0_sdot,
        TensorType::Q4_1 => row_q4_1_sdot,
        TensorType::Q5_0 => row_q5_0_sdot,
        TensorType::Q5_1 => row_q5_1_sdot,
        TensorType::Q8_0 => row_q8_0_sdot,
        TensorType::Q4K => row_q4_k_sdot,
        TensorType::Q5K => row_q5_k_sdot,
        TensorType::Q6K => row_q6_k_sdot,
        _ => scalar::row_quant(ty),
    }
}

// ---------------------------------------------------------------------------
// Float vectors
// ---------------------------------------------------------------------------

/// Dot product of two float vectors. Neither pointer needs to be aligned.
unsafe fn dot(a: *const f32, b: *const f32, n: usize) -> f32 {
    // SAFETY: the caller guarantees both pointers are valid for `n` reads,
    // and every access below stays under `n`.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(a.add(i + 4)), vld1q_f32(b.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(a.add(i + 8)), vld1q_f32(b.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(a.add(i + 12)), vld1q_f32(b.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            acc0 = vfmaq_f32(acc0, vld1q_f32(a.add(i)), vld1q_f32(b.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
        while i < n {
            sum += a.add(i).read_unaligned() * b.add(i).read_unaligned();
            i += 1;
        }
        sum
    }
}

/// `y += alpha * x`.
unsafe fn axpy(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
    // SAFETY: the caller guarantees `x` is valid for `n` reads and `y` for
    // `n` writes.
    unsafe {
        let mut i = 0;
        while i + 4 <= n {
            vst1q_f32(
                y.add(i),
                vfmaq_n_f32(vld1q_f32(y.add(i)), vld1q_f32(x.add(i)), alpha),
            );
            i += 4;
        }
        while i < n {
            *y.add(i) += alpha * *x.add(i);
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Half floats
// ---------------------------------------------------------------------------

/// Reads the half float at `p` as an f32.
///
/// Converting between half and single precision is a base ARMv8
/// instruction, but Rust reaches it only through inline assembly on the
/// oldest compiler this crate supports.
#[inline(always)]
unsafe fn half(p: *const u8) -> f32 {
    // SAFETY: the caller guarantees two readable bytes at `p`. The assembly
    // only touches the two registers it is given.
    unsafe {
        let bits = u16::from_le((p as *const u16).read_unaligned()) as u32;
        let value: f32;
        asm!(
            "fmov {v:s}, {bits:w}",
            "fcvt {v:s}, {v:h}",
            bits = in(reg) bits,
            v = out(vreg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
        value
    }
}

/// Reads the four half floats at `p` as four f32 values.
#[inline(always)]
unsafe fn halves(p: *const u8) -> float32x4_t {
    // SAFETY: the caller guarantees eight readable bytes at `p`. The
    // assembly only touches the two registers it is given.
    unsafe {
        let bits = u64::from_le((p as *const u64).read_unaligned());
        let value: float32x4_t;
        asm!(
            "fmov {v:d}, {bits}",
            "fcvtl {v:v}.4s, {v:v}.4h",
            bits = in(reg) bits,
            v = out(vreg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
        value
    }
}

// ---------------------------------------------------------------------------
// Float weight rows
// ---------------------------------------------------------------------------

/// Row of f32 bytes times float activations.
unsafe fn row_f32(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: `dot` reads both sides unaligned.
    unsafe { dot(row as *const f32, x, n) }
}

/// Row of half floats times float activations.
unsafe fn row_f16(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: the caller guarantees `row` holds `n` halves and `x` is valid
    // for `n` reads.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 8 <= n {
            acc0 = vfmaq_f32(acc0, halves(row.add(i * 2)), vld1q_f32(x.add(i)));
            acc1 = vfmaq_f32(acc1, halves(row.add(i * 2 + 8)), vld1q_f32(x.add(i + 4)));
            i += 8;
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            sum += half(row.add(i * 2)) * *x.add(i);
            i += 1;
        }
        sum
    }
}

/// Row of brain floats times float activations.
unsafe fn row_bf16(row: *const u8, x: *const f32, n: usize) -> f32 {
    // SAFETY: as in `row_f16`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= n {
            // A brain float is the top 16 bits of an f32.
            let halves = vld1_u16(row.add(i * 2) as *const u16);
            let w = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(halves)));
            acc = vfmaq_f32(acc, w, vld1q_f32(x.add(i)));
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while i < n {
            let half = u16::from_le((row.add(i * 2) as *const u16).read_unaligned());
            sum += f32::from_bits((half as u32) << 16) * *x.add(i);
            i += 1;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// The byte multiply
// ---------------------------------------------------------------------------

/// Adds the 16 products of two byte registers into four 32-bit lanes.
trait ByteDot {
    unsafe fn add_products(sum: int32x4_t, weights: int8x16_t, acts: int8x16_t) -> int32x4_t;
}

/// Widening multiplies to 16 bits, then pairwise adds to 32 bits.
struct Widening;

/// The `sdot` instruction.
struct Sdot;

impl ByteDot for Widening {
    #[inline(always)]
    unsafe fn add_products(sum: int32x4_t, weights: int8x16_t, acts: int8x16_t) -> int32x4_t {
        // SAFETY: NEON is part of the AArch64 baseline.
        unsafe {
            let low = vmull_s8(vget_low_s8(weights), vget_low_s8(acts));
            let high = vmull_high_s8(weights, acts);
            vpadalq_s16(vpadalq_s16(sum, low), high)
        }
    }
}

impl ByteDot for Sdot {
    #[inline(always)]
    unsafe fn add_products(sum: int32x4_t, weights: int8x16_t, acts: int8x16_t) -> int32x4_t {
        // SAFETY: only instantiated inside functions that enable `dotprod`,
        // which are only selected when the CPU reports the extension.
        unsafe { sdot(sum, weights, acts) }
    }
}

/// The `sdot` instruction: multiplies 16 pairs of signed bytes and adds each
/// group of four products to one 32-bit lane of `sum`.
///
/// Written in assembly because the intrinsic for it is newer than the
/// oldest compiler this crate supports.
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn sdot(mut sum: int32x4_t, weights: int8x16_t, acts: int8x16_t) -> int32x4_t {
    // SAFETY: the instruction only touches the three registers it is given,
    // and this function enables the extension it belongs to.
    unsafe {
        asm!(
            "sdot {sum:v}.4s, {weights:v}.16b, {acts:v}.16b",
            sum = inout(vreg) sum,
            weights = in(vreg) weights,
            acts = in(vreg) acts,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sum
}

/// See [`DotDecoded`]. One step handles 16 weights: their products land in
/// four 32-bit lanes, which all share one stored scale. Two accumulators
/// take turns so that consecutive steps do not wait on each other.
#[inline(always)]
unsafe fn decoded_row<D: ByteDot>(
    q: *const i8,
    scale: *const f32,
    n: usize,
    act: ActRef,
    group: usize,
) -> f32 {
    let runs = n / 16;
    // Runs per activation group, as a shift: 1 for groups of 32, 4 for 256.
    let shift = (group / 16).trailing_zeros();
    // SAFETY: the caller guarantees `n` weights and activations, `n / 4`
    // scales and `n / group` activation scales. Every index below stays
    // under `runs * 16 = n`.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
        for run in 0..runs {
            let weights = vld1q_s8(q.add(run * 16));
            let acts = vld1q_s8(act.q.add(run * 16));
            let products = D::add_products(vdupq_n_s32(0), weights, acts);
            let scales = vmulq_n_f32(vld1q_f32(scale.add(run * 4)), *act.scale.add(run >> shift));
            acc[run & 1] = vfmaq_f32(acc[run & 1], vcvtq_f32_s32(products), scales);
        }
        vaddvq_f32(vaddq_f32(acc[0], acc[1]))
    }
}

/// [`decoded_row`] for every CPU.
unsafe fn dot_decoded(q: *const i8, scale: *const f32, n: usize, act: ActRef, group: usize) -> f32 {
    // SAFETY: forwarded contract.
    unsafe { decoded_row::<Widening>(q, scale, n, act, group) }
}

/// [`decoded_row`] for CPUs with the dot product extension.
#[target_feature(enable = "dotprod")]
unsafe fn dot_decoded_sdot(
    q: *const i8,
    scale: *const f32,
    n: usize,
    act: ActRef,
    group: usize,
) -> f32 {
    // SAFETY: forwarded contract.
    unsafe { decoded_row::<Sdot>(q, scale, n, act, group) }
}

/// Reads an unaligned little-endian u32 at `p`.
#[inline(always)]
unsafe fn word(p: *const u8) -> u32 {
    // SAFETY: the caller guarantees four readable bytes at `p`.
    unsafe { u32::from_le((p as *const u32).read_unaligned()) }
}

// ---------------------------------------------------------------------------
// Formats with 32-weight blocks
// ---------------------------------------------------------------------------

/// One unpacked block of 32 weights: `weight[i] = scale * w[i] + offset`.
struct Block {
    /// Weights 0..16.
    low: int8x16_t,
    /// Weights 16..32.
    high: int8x16_t,
    scale: f32,
    offset: f32,
}

/// A format with 32-weight blocks.
trait Block32 {
    /// Bytes in one block.
    const BYTES: usize;
    /// True when blocks carry an offset. The others skip that term.
    const HAS_OFFSET: bool;
    /// Unpacks the block at `p`.
    unsafe fn load(p: *const u8) -> Block;
}

/// Splits 16 bytes at `p` into their low and high nibbles.
#[inline(always)]
unsafe fn nibbles(p: *const u8) -> (uint8x16_t, uint8x16_t) {
    // SAFETY: the caller guarantees 16 readable bytes at `p`.
    unsafe {
        let bytes = vld1q_u8(p);
        (vandq_u8(bytes, vdupq_n_u8(0x0f)), vshrq_n_u8::<4>(bytes))
    }
}

/// Selects one bit per byte: bit `i % 8` in byte `i`.
static BIT_IN_BYTE: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Expands the 16 bits at `p` into 16 bytes: 0x10 where the bit is set.
/// These are the fifth bits of the 5-bit formats.
#[inline(always)]
unsafe fn fifth_bits(p: *const u8) -> uint8x16_t {
    // SAFETY: the caller guarantees two readable bytes at `p`.
    unsafe {
        let spread = vcombine_u8(vdup_n_u8(*p), vdup_n_u8(*p.add(1)));
        let is_set = vtstq_u8(spread, vld1q_u8(BIT_IN_BYTE.as_ptr()));
        vandq_u8(is_set, vdupq_n_u8(0x10))
    }
}

struct Q4_0;
struct Q4_1;
struct Q5_0;
struct Q5_1;
struct Q8_0;

impl Block32 for Q4_0 {
    const BYTES: usize = 18;
    const HAS_OFFSET: bool = false;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Block {
        // SAFETY: the caller guarantees an 18-byte block at `p`.
        unsafe {
            // Stored with a bias of 8, removed here so no offset is needed.
            let (low, high) = nibbles(p.add(2));
            let bias = vdupq_n_s8(8);
            Block {
                low: vsubq_s8(vreinterpretq_s8_u8(low), bias),
                high: vsubq_s8(vreinterpretq_s8_u8(high), bias),
                scale: half(p),
                offset: 0.0,
            }
        }
    }
}

impl Block32 for Q4_1 {
    const BYTES: usize = 20;
    const HAS_OFFSET: bool = true;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Block {
        // SAFETY: the caller guarantees a 20-byte block at `p`.
        unsafe {
            let (low, high) = nibbles(p.add(4));
            Block {
                low: vreinterpretq_s8_u8(low),
                high: vreinterpretq_s8_u8(high),
                scale: half(p),
                offset: half(p.add(2)),
            }
        }
    }
}

impl Block32 for Q5_0 {
    const BYTES: usize = 22;
    const HAS_OFFSET: bool = false;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Block {
        // SAFETY: the caller guarantees a 22-byte block at `p`.
        unsafe {
            // Stored with a bias of 16, removed here so no offset is needed.
            let (low, high) = nibbles(p.add(6));
            let low = vorrq_u8(low, fifth_bits(p.add(2)));
            let high = vorrq_u8(high, fifth_bits(p.add(4)));
            let bias = vdupq_n_s8(16);
            Block {
                low: vsubq_s8(vreinterpretq_s8_u8(low), bias),
                high: vsubq_s8(vreinterpretq_s8_u8(high), bias),
                scale: half(p),
                offset: 0.0,
            }
        }
    }
}

impl Block32 for Q5_1 {
    const BYTES: usize = 24;
    const HAS_OFFSET: bool = true;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Block {
        // SAFETY: the caller guarantees a 24-byte block at `p`.
        unsafe {
            let (low, high) = nibbles(p.add(8));
            let low = vorrq_u8(low, fifth_bits(p.add(4)));
            let high = vorrq_u8(high, fifth_bits(p.add(6)));
            Block {
                low: vreinterpretq_s8_u8(low),
                high: vreinterpretq_s8_u8(high),
                scale: half(p),
                offset: half(p.add(2)),
            }
        }
    }
}

impl Block32 for Q8_0 {
    const BYTES: usize = 34;
    const HAS_OFFSET: bool = false;
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Block {
        // SAFETY: the caller guarantees a 34-byte block at `p`.
        unsafe {
            Block {
                low: vld1q_s8(p.add(2) as *const i8),
                high: vld1q_s8(p.add(18) as *const i8),
                scale: half(p),
                offset: 0.0,
            }
        }
    }
}

/// Row kernel for any 32-weight block format.
///
/// The dot product of one block is
/// `scale * act_scale * sum(w * a) + offset * act_scale * sum(a)`.
/// The last factor is precomputed per block as `act.wsum`. Blocks alternate
/// between two accumulators so consecutive blocks do not wait on each other.
#[inline(always)]
unsafe fn block_row<F: Block32, D: ByteDot>(row: *const u8, n: usize, act: ActRef) -> f32 {
    let blocks = n / 32;
    // SAFETY: the caller guarantees `row` holds `blocks` blocks and `act`
    // describes `n` quantized activations.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0), vdupq_n_f32(0.0)];
        let mut correction = [0.0f32, 0.0];
        let mut b = 0;
        while b < blocks {
            for lane in 0..2 {
                if b + lane >= blocks {
                    break;
                }
                let index = b + lane;
                let block = F::load(row.add(index * F::BYTES));
                let acts_low = vld1q_s8(act.q.add(index * 32));
                let acts_high = vld1q_s8(act.q.add(index * 32 + 16));
                let products = D::add_products(vdupq_n_s32(0), block.low, acts_low);
                let products = D::add_products(products, block.high, acts_high);
                let scale = block.scale * *act.scale.add(index);
                acc[lane] = vfmaq_n_f32(acc[lane], vcvtq_f32_s32(products), scale);
                if F::HAS_OFFSET {
                    correction[lane] += block.offset * *act.wsum.add(index);
                }
            }
            b += 2;
        }
        vaddvq_f32(vaddq_f32(acc[0], acc[1])) + (correction[0] + correction[1])
    }
}

// ---------------------------------------------------------------------------
// Formats with 256-weight super-blocks
// ---------------------------------------------------------------------------

/// Unpacks the eight 6-bit scales and eight 6-bit minimums stored in the 12
/// bytes at `p`. The first four of each are the low 6 bits of bytes 0..8.
/// The last four take 4 bits from bytes 8..12 and 2 bits from the tops of
/// bytes 0..8.
#[inline(always)]
unsafe fn k4_scales(p: *const u8) -> ([i32; 8], [i32; 8]) {
    // SAFETY: the caller guarantees 12 readable bytes at `p`.
    let (w0, w1, w2) = unsafe { (word(p), word(p.add(4)), word(p.add(8))) };
    let scales_low = w0 & 0x3f3f_3f3f;
    let minimums_low = w1 & 0x3f3f_3f3f;
    let scales_high = (w2 & 0x0f0f_0f0f) | (((w0 >> 6) & 0x0303_0303) << 4);
    let minimums_high = ((w2 >> 4) & 0x0f0f_0f0f) | (((w1 >> 6) & 0x0303_0303) << 4);
    let scale_bytes = (scales_low as u64 | (scales_high as u64) << 32).to_le_bytes();
    let minimum_bytes = (minimums_low as u64 | (minimums_high as u64) << 32).to_le_bytes();
    (
        scale_bytes.map(|v| v as i32),
        minimum_bytes.map(|v| v as i32),
    )
}

/// The minimum term of a Q4_K or Q5_K super-block: the sum over its eight
/// sub-blocks of `minimum * sum(activations)`.
#[inline(always)]
unsafe fn k4_minimum_term(minimums: &[i32; 8], sums: *const i16) -> i32 {
    let mut total = 0i32;
    for (j, minimum) in minimums.iter().enumerate() {
        // SAFETY: the caller guarantees 16 readable sums. Sub-blocks are 32
        // wide, so each uses two sums of 16.
        let sum = unsafe { *sums.add(j * 2) as i32 + *sums.add(j * 2 + 1) as i32 };
        total += minimum * sum;
    }
    total
}

/// Q4_K: 144-byte super-blocks of eight 32-weight sub-blocks.
/// `weight = d * scale * nibble - dmin * minimum`.
#[inline(always)]
unsafe fn q4_k_row<D: ByteDot>(row: *const u8, n: usize, act: ActRef) -> f32 {
    const BLOCK_BYTES: usize = 144;
    let mut total = 0.0f32;
    // SAFETY: the caller guarantees `row` holds `n / 256` super-blocks and
    // `act` describes `n` quantized activations.
    unsafe {
        let low_nibble = vdupq_n_u8(0x0f);
        for sb in 0..n / 256 {
            let p = row.add(sb * BLOCK_BYTES);
            let acts = act.q.add(sb * 256);
            let act_scale = *act.scale.add(sb);
            let d = half(p) * act_scale;
            let dmin = half(p.add(2)) * act_scale;
            let (scales, minimums) = k4_scales(p.add(4));
            let minimum_term = k4_minimum_term(&minimums, act.sums.add(sb * 16));

            let mut sum = vdupq_n_s32(0);
            for pair in 0..4 {
                // 32 bytes hold two sub-blocks: low nibbles, then high nibbles.
                let bits0 = vld1q_u8(p.add(16 + pair * 32));
                let bits1 = vld1q_u8(p.add(16 + pair * 32 + 16));
                let a = acts.add(pair * 64);
                let low = D::add_products(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vandq_u8(bits0, low_nibble)),
                    vld1q_s8(a),
                );
                let low = D::add_products(
                    low,
                    vreinterpretq_s8_u8(vandq_u8(bits1, low_nibble)),
                    vld1q_s8(a.add(16)),
                );
                let high = D::add_products(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(vshrq_n_u8::<4>(bits0)),
                    vld1q_s8(a.add(32)),
                );
                let high = D::add_products(
                    high,
                    vreinterpretq_s8_u8(vshrq_n_u8::<4>(bits1)),
                    vld1q_s8(a.add(48)),
                );
                sum = vmlaq_n_s32(sum, low, scales[pair * 2]);
                sum = vmlaq_n_s32(sum, high, scales[pair * 2 + 1]);
            }
            total += d * vaddvq_s32(sum) as f32 - dmin * minimum_term as f32;
        }
    }
    total
}

/// Q5_K: 176-byte super-blocks. Like Q4_K with a fifth bit per weight,
/// stored as 32 bytes of bit pairs after the scales.
#[inline(always)]
unsafe fn q5_k_row<D: ByteDot>(row: *const u8, n: usize, act: ActRef) -> f32 {
    const BLOCK_BYTES: usize = 176;
    let mut total = 0.0f32;
    // SAFETY: as in `q4_k_row`.
    unsafe {
        let low_nibble = vdupq_n_u8(0x0f);
        let bit0 = vdupq_n_u8(1);
        let bit1 = vdupq_n_u8(2);
        for sb in 0..n / 256 {
            let p = row.add(sb * BLOCK_BYTES);
            let acts = act.q.add(sb * 256);
            let act_scale = *act.scale.add(sb);
            let d = half(p) * act_scale;
            let dmin = half(p.add(2)) * act_scale;
            let (scales, minimums) = k4_scales(p.add(4));
            let minimum_term = k4_minimum_term(&minimums, act.sums.add(sb * 16));

            // Byte `l` of the high bits holds weight `l` of all eight
            // sub-blocks, two bits per pair. Shifting right by two moves the
            // next pair's bits into place.
            let mut high_bits0 = vld1q_u8(p.add(16));
            let mut high_bits1 = vld1q_u8(p.add(32));
            let mut sum = vdupq_n_s32(0);
            for pair in 0..4 {
                let bits0 = vld1q_u8(p.add(48 + pair * 32));
                let bits1 = vld1q_u8(p.add(48 + pair * 32 + 16));
                let low0 = vorrq_u8(
                    vandq_u8(bits0, low_nibble),
                    vshlq_n_u8::<4>(vandq_u8(high_bits0, bit0)),
                );
                let low1 = vorrq_u8(
                    vandq_u8(bits1, low_nibble),
                    vshlq_n_u8::<4>(vandq_u8(high_bits1, bit0)),
                );
                let high0 = vorrq_u8(
                    vshrq_n_u8::<4>(bits0),
                    vshlq_n_u8::<3>(vandq_u8(high_bits0, bit1)),
                );
                let high1 = vorrq_u8(
                    vshrq_n_u8::<4>(bits1),
                    vshlq_n_u8::<3>(vandq_u8(high_bits1, bit1)),
                );
                high_bits0 = vshrq_n_u8::<2>(high_bits0);
                high_bits1 = vshrq_n_u8::<2>(high_bits1);
                let a = acts.add(pair * 64);
                let low = D::add_products(vdupq_n_s32(0), vreinterpretq_s8_u8(low0), vld1q_s8(a));
                let low = D::add_products(low, vreinterpretq_s8_u8(low1), vld1q_s8(a.add(16)));
                let high = D::add_products(
                    vdupq_n_s32(0),
                    vreinterpretq_s8_u8(high0),
                    vld1q_s8(a.add(32)),
                );
                let high = D::add_products(high, vreinterpretq_s8_u8(high1), vld1q_s8(a.add(48)));
                sum = vmlaq_n_s32(sum, low, scales[pair * 2]);
                sum = vmlaq_n_s32(sum, high, scales[pair * 2 + 1]);
            }
            total += d * vaddvq_s32(sum) as f32 - dmin * minimum_term as f32;
        }
    }
    total
}

/// Q6_K: 210-byte super-blocks of sixteen 16-weight sub-blocks.
/// `weight = d * scale * (value - 32)`.
///
/// The values are kept unsigned, 0 to 63. The `- 32` is applied afterwards
/// as `32 * scale * sum(acts)`, which costs less than adjusting every byte.
#[inline(always)]
unsafe fn q6_k_row<D: ByteDot>(row: *const u8, n: usize, act: ActRef) -> f32 {
    const BLOCK_BYTES: usize = 210;
    let mut total = 0.0f32;
    // SAFETY: as in `q4_k_row`.
    unsafe {
        let low_nibble = vdupq_n_u8(0x0f);
        let two_bits = vdupq_n_u8(3);
        for sb in 0..n / 256 {
            let p = row.add(sb * BLOCK_BYTES);
            let acts = act.q.add(sb * 256);
            let d = half(p.add(208)) * *act.scale.add(sb);

            // The sum over sub-blocks of scale * sum(acts), for the bias.
            let scale_bytes = vld1q_s8(p.add(192) as *const i8);
            let scales_low = vmovl_s8(vget_low_s8(scale_bytes));
            let scales_high = vmovl_high_s8(scale_bytes);
            let sums_low = vld1q_s16(act.sums.add(sb * 16));
            let sums_high = vld1q_s16(act.sums.add(sb * 16 + 8));
            let mut bias = vmull_s16(vget_low_s16(scales_low), vget_low_s16(sums_low));
            bias = vmlal_high_s16(bias, scales_low, sums_low);
            bias = vmlal_s16(bias, vget_low_s16(scales_high), vget_low_s16(sums_high));
            bias = vmlal_high_s16(bias, scales_high, sums_high);
            let mut sum = vnegq_s32(vshlq_n_s32::<5>(bias));

            for half_index in 0..2 {
                for side in 0..2 {
                    // Each register is one sub-block of 16 weights. A value
                    // takes a nibble from `first` or `second` and two more
                    // bits from `high`.
                    let first = vld1q_u8(p.add(half_index * 64 + side * 16));
                    let second = vld1q_u8(p.add(half_index * 64 + 32 + side * 16));
                    let high = vld1q_u8(p.add(128 + half_index * 32 + side * 16));
                    let values = [
                        vorrq_u8(
                            vandq_u8(first, low_nibble),
                            vshlq_n_u8::<4>(vandq_u8(high, two_bits)),
                        ),
                        vorrq_u8(
                            vandq_u8(second, low_nibble),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(high), two_bits)),
                        ),
                        vorrq_u8(
                            vshrq_n_u8::<4>(first),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(high), two_bits)),
                        ),
                        vorrq_u8(
                            vshrq_n_u8::<4>(second),
                            vshlq_n_u8::<4>(vshrq_n_u8::<6>(high)),
                        ),
                    ];
                    for (run, value) in values.into_iter().enumerate() {
                        let sub = half_index * 8 + run * 2 + side;
                        let products = D::add_products(
                            vdupq_n_s32(0),
                            vreinterpretq_s8_u8(value),
                            vld1q_s8(acts.add(sub * 16)),
                        );
                        sum = vmlaq_n_s32(sum, products, *p.add(192 + sub) as i8 as i32);
                    }
                }
            }
            total += d * vaddvq_s32(sum) as f32;
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Instantiations
// ---------------------------------------------------------------------------

/// Defines the two instances of a generic row kernel: one for every CPU and
/// one for CPUs with the dot product extension.
macro_rules! variants {
    ($widening:ident, $sdot:ident, $kernel:ident $(, $format:ty)?) => {
        unsafe fn $widening(row: *const u8, n: usize, act: ActRef) -> f32 {
            // SAFETY: forwarded contract.
            unsafe { $kernel::<$($format,)? Widening>(row, n, act) }
        }

        #[target_feature(enable = "dotprod")]
        unsafe fn $sdot(row: *const u8, n: usize, act: ActRef) -> f32 {
            // SAFETY: forwarded contract.
            unsafe { $kernel::<$($format,)? Sdot>(row, n, act) }
        }
    };
}

variants!(row_q4_0, row_q4_0_sdot, block_row, Q4_0);
variants!(row_q4_1, row_q4_1_sdot, block_row, Q4_1);
variants!(row_q5_0, row_q5_0_sdot, block_row, Q5_0);
variants!(row_q5_1, row_q5_1_sdot, block_row, Q5_1);
variants!(row_q8_0, row_q8_0_sdot, block_row, Q8_0);
variants!(row_q4_k, row_q4_k_sdot, q4_k_row);
variants!(row_q5_k, row_q5_k_sdot, q5_k_row);
variants!(row_q6_k, row_q6_k_sdot, q6_k_row);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::f16::f16_to_f32;

    #[test]
    fn the_conversion_instruction_matches_the_software_conversion() {
        for bits in 0..=u16::MAX {
            let bytes = bits.to_le_bytes();
            // SAFETY: two readable bytes.
            let converted = unsafe { half(bytes.as_ptr()) };
            let expected = f16_to_f32(bits);
            if expected.is_nan() {
                assert!(converted.is_nan());
            } else {
                assert_eq!(converted.to_bits(), expected.to_bits(), "half {bits:#06x}");
            }
        }
    }

    #[test]
    fn four_halves_convert_in_order() {
        let values = [0x3c00u16, 0xc000, 0x0001, 0x7bff];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // SAFETY: eight readable bytes.
        let converted = unsafe { halves(bytes.as_ptr()) };
        let mut out = [0.0f32; 4];
        // SAFETY: `out` has room for four floats.
        unsafe { vst1q_f32(out.as_mut_ptr(), converted) };
        for (value, half) in out.iter().zip(values) {
            assert_eq!(*value, f16_to_f32(half));
        }
    }
}

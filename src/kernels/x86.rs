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

use std::arch::asm;
use std::arch::x86_64::*;
use std::slice;

use super::scalar;
use super::{ActRef, DotQuant, Kernels, TileDot};
use crate::tensor::TensorType;
use crate::tensor::f16::{f16_to_f32, f32_to_f16};
use crate::tensor::quant::{self, Decoder, Group, Tile};

/// Kernels for every CPU with AVX2.
pub static KERNELS: Kernels = kernels("avx2", tile_dot);

/// Kernels for CPUs that also have AVX-VNNI.
pub static KERNELS_VNNI: Kernels = kernels("avx2+vnni", tile_dot_vnni);

/// Builds a kernel table. Only the tile kernel differs between the two.
const fn kernels(name: &'static str, tile_dot: TileDot) -> Kernels {
    Kernels {
        name,
        dot,
        axpy,
        sum_squares,
        dot_rows,
        mix_rows,
        dot_rows_half,
        mix_rows_half,
        to_half,
        silu_mul,
        gelu_mul,
        softmax,
        quantize,
        row_f32,
        row_f16,
        row_bf16,
        row_quant,
        tile_rows: 8,
        tile_dot,
        tile_fill,
    }
}

/// Asks the CPU about itself. A function of its own because older
/// compilers declare the instruction unsafe and newer ones do not.
unsafe fn cpuid(leaf: u32, sub_leaf: u32) -> CpuidResult {
    __cpuid_count(leaf, sub_leaf)
}

/// True when the running CPU has AVX-VNNI, which multiplies bytes and adds
/// the products in fours with one instruction. Intel has it since the 12th
/// generation Core and the N100, AMD since Zen 5.
///
/// Read from CPUID directly, because the standard library's check for it is
/// newer than the oldest compiler this crate supports. The operating system
/// already saves the registers involved when [`supported`] is true.
pub fn has_vnni() -> bool {
    // SAFETY: every x86-64 CPU has the CPUID instruction.
    unsafe { cpuid(0, 0).eax >= 7 && cpuid(7, 0).eax >= 1 && cpuid(7, 1).eax & (1 << 4) != 0 }
}

/// True when the running CPU has every feature these kernels use.
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

/// The row kernel for a quantized type. Formats without an AVX2 decoder
/// are decoded by the portable decoders and multiplied with AVX2. A new
/// format falls back to the scalar kernel until it is listed here.
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
        TensorType::Iq4Nl => row_iq4_nl,
        TensorType::Mxfp4 => row_mxfp4,
        TensorType::Q2K => row_q2_k,
        TensorType::Q3K => row_q3_k,
        TensorType::Iq4Xs => row_iq4_xs,
        TensorType::Iq3S => row_iq3_s,
        TensorType::Iq3Xxs => row_iq3_xxs,
        TensorType::Iq2S => row_iq2_s,
        TensorType::Iq2Xs => row_iq2_xs,
        TensorType::Iq2Xxs => row_iq2_xxs,
        TensorType::Iq1M => row_iq1_m,
        TensorType::Iq1S => row_iq1_s,
        TensorType::Tq2_0 => row_tq2_0,
        TensorType::Tq1_0 => row_tq1_0,
        TensorType::Nvfp4 => row_nvfp4,
        TensorType::Q2_0 => row_q2_0,
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

/// See [`TileDot`], for tiles of 8 rows.
///
/// A step loads 4 columns of all 8 rows, 32 bytes, and multiplies them with
/// the same 4 activation bytes repeated in every lane. `maddubs` adds the
/// products in pairs, which leaves two 16-bit sums per row. Four steps make
/// a sub-block of 16 columns. Its sums are widened to one 32-bit sum per
/// row, converted to floats once, and scaled with one scale per row.
///
/// How early the 16-bit sums must be widened depends on the tile:
///
/// - Weights up to 31: the pair sums of all four steps fit in 16 bits, so
///   they are added there and widened once.
/// - Weights up to 63: two steps fit.
/// - Signed weights: `maddubs` needs an unsigned operand, so it multiplies
///   the weights' magnitudes with activations that carry the weights'
///   signs. One step fills 16 bits.
///
/// With `VNNI`, one instruction per step multiplies and adds straight into
/// 32-bit sums, so nothing has to fit in 16 bits. It also wants unsigned
/// weights. Signed ones are moved up by 128, and the surplus, 128 times the
/// sum of the activations, is taken off again. That sum is already known
/// from quantizing them.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn tile_dot_impl<const VNNI: bool>(
    tile: &Tile,
    act: ActRef,
    scaled_sums: *const f32,
    group: usize,
    out: *mut f32,
) {
    debug_assert_eq!(tile.height, 8);
    debug_assert!(tile.signed || tile.max <= 63);
    debug_assert!(group == 32 || group == 256);
    let (q, scale, offset) = (tile.q.as_ptr(), tile.scale.as_ptr(), tile.offset.as_ptr());
    let ones = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_ps();

    // The weights and activations of step `$step`.
    macro_rules! operands {
        ($step:expr) => {{
            let weights = _mm256_loadu_si256(q.add($step * 32) as *const __m256i);
            let four = (act.q.add($step * 4) as *const i32).read_unaligned();
            (weights, _mm256_set1_epi32(four))
        }};
    }
    // The pair sums of step `$step`, for unsigned weights.
    macro_rules! pairs {
        ($step:expr) => {{
            let (weights, acts) = operands!($step);
            _mm256_maddubs_epi16(weights, acts)
        }};
    }
    // The row sums of the sub-block that starts at step `$step`, by the
    // VNNI instruction. `$flip` is XORed into the weights first. Two chains
    // of two steps keep each addition from waiting on three others.
    macro_rules! vnni_rows {
        ($step:expr, $flip:expr) => {{
            let (w0, a0) = operands!($step);
            let (w1, a1) = operands!($step + 1);
            let (w2, a2) = operands!($step + 2);
            let (w3, a3) = operands!($step + 3);
            let none = _mm256_setzero_si256();
            let even = dpbusd(none, _mm256_xor_si256(w0, $flip), a0);
            let even = dpbusd(even, _mm256_xor_si256(w2, $flip), a2);
            let odd = dpbusd(none, _mm256_xor_si256(w1, $flip), a1);
            let odd = dpbusd(odd, _mm256_xor_si256(w3, $flip), a3);
            _mm256_add_epi32(even, odd)
        }};
    }
    // The row sums of step `$step`, for signed weights.
    macro_rules! signed_rows {
        ($step:expr) => {{
            let (weights, acts) = operands!($step);
            let magnitudes = _mm256_abs_epi8(weights);
            let pairs = _mm256_maddubs_epi16(magnitudes, _mm256_sign_epi8(acts, weights));
            _mm256_madd_epi16(pairs, ones)
        }};
    }
    // The scaled row sums of sub-block `$sub`, added to `$acc`. `$sums`
    // yields the row sums of the sub-block whose first step is `$first`.
    macro_rules! scaled {
        ($acc:expr, $sub:expr, |$first:ident| $sums:expr) => {{
            let $first = $sub * 4;
            let sums = _mm256_cvtepi32_ps($sums);
            _mm256_fmadd_ps(sums, _mm256_loadu_ps(scale.add($sub * 8)), $acc)
        }};
    }
    // The offset term of sub-block `$sub`, added to `$acc`.
    macro_rules! shifted {
        ($acc:expr, $sub:expr) => {{
            let offsets = _mm256_loadu_ps(offset.add($sub * 8));
            _mm256_fmadd_ps(offsets, broadcast(scaled_sums.add($sub)), $acc)
        }};
    }
    // Walks a tile whose activation groups hold `$pairs` pairs of
    // sub-blocks. The two sub-blocks of a pair have accumulators of their
    // own, so neither waits for the addition of the other. With
    // `$offsets`, the offset term is gathered on the way, again in two
    // accumulators.
    macro_rules! walk {
        ($pairs:expr, $offsets:expr, |$first:ident| $sums:expr) => {{
            let mut total = zero;
            let (mut shift_even, mut shift_odd) = (zero, zero);
            let mut sub = 0;
            for g in 0..tile.cols / ($pairs * 32) {
                let (mut even, mut odd) = (zero, zero);
                for _ in 0..$pairs {
                    even = scaled!(even, sub, |$first| $sums);
                    odd = scaled!(odd, sub + 1, |$first| $sums);
                    if $offsets {
                        shift_even = shifted!(shift_even, sub);
                        shift_odd = shifted!(shift_odd, sub + 1);
                    }
                    sub += 2;
                }
                let partial = _mm256_add_ps(even, odd);
                total = _mm256_fmadd_ps(partial, broadcast(act.scale.add(g)), total);
            }
            _mm256_add_ps(total, _mm256_add_ps(shift_even, shift_odd))
        }};
    }
    // The walk for either group size, each with a loop of its own.
    macro_rules! reduce {
        ($offsets:expr, |$first:ident| $sums:expr) => {
            if group == 32 {
                walk!(1, $offsets, |$first| $sums)
            } else {
                walk!(8, $offsets, |$first| $sums)
            }
        };
    }

    // SAFETY: the tile holds `cols * 8` weights and `cols / 16 * 8` scales
    // and offsets. The caller guarantees activations for `cols` elements in
    // groups of `group`, their scaled sums when the tile has offsets, and
    // room for 8 outputs. Every index below stays inside those sizes.
    unsafe {
        let total = if VNNI && tile.signed {
            let flip = _mm256_set1_epi8(-128);
            reduce!(tile.has_offset, |s| {
                let surplus = 128 * *act.sums.add(s / 4) as i32;
                _mm256_sub_epi32(vnni_rows!(s, flip), _mm256_set1_epi32(surplus))
            })
        } else if VNNI {
            let flip = _mm256_setzero_si256();
            reduce!(tile.has_offset, |s| vnni_rows!(s, flip))
        } else if tile.signed {
            // Signed tiles come from formats without offsets.
            reduce!(tile.has_offset, |s| {
                let low = _mm256_add_epi32(signed_rows!(s), signed_rows!(s + 1));
                let high = _mm256_add_epi32(signed_rows!(s + 2), signed_rows!(s + 3));
                _mm256_add_epi32(low, high)
            })
        } else if tile.max <= 31 {
            reduce!(tile.has_offset, |s| {
                let low = _mm256_add_epi16(pairs!(s), pairs!(s + 1));
                let high = _mm256_add_epi16(pairs!(s + 2), pairs!(s + 3));
                _mm256_madd_epi16(_mm256_add_epi16(low, high), ones)
            })
        } else {
            reduce!(tile.has_offset, |s| {
                let low = _mm256_add_epi16(pairs!(s), pairs!(s + 1));
                let high = _mm256_add_epi16(pairs!(s + 2), pairs!(s + 3));
                _mm256_add_epi32(_mm256_madd_epi16(low, ones), _mm256_madd_epi16(high, ones))
            })
        };
        _mm256_storeu_ps(out, total);
    }
}

/// The tile kernel for every CPU with AVX2.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_dot(tile: &Tile, act: ActRef, scaled_sums: *const f32, group: usize, out: *mut f32) {
    // SAFETY: forwarded contract.
    unsafe { tile_dot_impl::<false>(tile, act, scaled_sums, group, out) }
}

/// The tile kernel for CPUs with AVX-VNNI.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_dot_vnni(
    tile: &Tile,
    act: ActRef,
    scaled_sums: *const f32,
    group: usize,
    out: *mut f32,
) {
    // SAFETY: forwarded contract. This table is only chosen when the CPU
    // reports the extension.
    unsafe { tile_dot_impl::<true>(tile, act, scaled_sums, group, out) }
}

/// The AVX-VNNI instruction `vpdpbusd`: multiplies 32 unsigned bytes of
/// `weights` with 32 signed bytes of `acts` and adds each four neighboring
/// products to one 32-bit lane of `sums`.
///
/// Written in assembly because the intrinsic for it is newer than the
/// oldest compiler this crate supports. The `{vex}` prefix selects the
/// AVX-VNNI encoding, not the AVX-512 one.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dpbusd(mut sums: __m256i, weights: __m256i, acts: __m256i) -> __m256i {
    // SAFETY: the instruction only touches the three registers it is
    // given. The caller guarantees the CPU has it.
    unsafe {
        asm!(
            "{{vex}} vpdpbusd {sums}, {weights}, {acts}",
            sums = inout(ymm_reg) sums,
            weights = in(ymm_reg) weights,
            acts = in(ymm_reg) acts,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    sums
}

// ---------------------------------------------------------------------------
// Filling tiles
// ---------------------------------------------------------------------------

/// See [`TileFill`](super::TileFill).
///
/// The common formats are unpacked 32 weights at a time for all eight rows,
/// and the eight registers are transposed straight into the tile. The other
/// formats take the portable fill.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn tile_fill(tile: &mut Tile, ty: TensorType, height: usize, data: &[u8], row_bytes: usize) {
    let fill: unsafe fn(&mut Tile, [*const u8; 8]) -> bool = match ty {
        TensorType::Q4_0 => fill_q4_0,
        TensorType::Q5_0 => fill_q5_0,
        TensorType::Q8_0 => fill_q8_0,
        TensorType::Iq4Nl => fill_iq4_nl,
        TensorType::Mxfp4 => fill_mxfp4,
        TensorType::Nvfp4 => fill_nvfp4,
        TensorType::Q2_0 => fill_q2_0,
        TensorType::Q4K => fill_q4_k,
        TensorType::Q5K => fill_q5_k,
        TensorType::Q6K => fill_q6_k,
        TensorType::Iq4Xs => fill_iq4_xs,
        _ => return tile.fill(ty, height, data, row_bytes),
    };
    if height != 8 {
        return tile.fill(ty, height, data, row_bytes);
    }
    tile.prepare(ty, 8, row_bytes);
    // Rows the tile lacks are read from a row of zeros, which decodes to
    // weights of zero in each of these formats, and then cleared: in some
    // formats zeros stand for other integers times a scale of zero.
    let blank = tile.blank_row(row_bytes).as_ptr();
    let mut rows = [blank; 8];
    let mut given = 0;
    for (row, bytes) in rows.iter_mut().zip(data.chunks_exact(row_bytes)) {
        *row = bytes.as_ptr();
        given += 1;
    }
    // SAFETY: every pointer in `rows` is valid for `row_bytes` bytes, which
    // is a whole number of blocks, and `prepare` sized the tile for them.
    tile.has_offset = unsafe { fill(tile, rows) };
    tile.clear_rows(given);
}

/// Turns eight registers of eight 32-bit lanes into their transpose: lane
/// `r` of result `k` is lane `k` of input `r`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn transpose(rows: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(rows[0], rows[1]);
    let t1 = _mm256_unpackhi_epi32(rows[0], rows[1]);
    let t2 = _mm256_unpacklo_epi32(rows[2], rows[3]);
    let t3 = _mm256_unpackhi_epi32(rows[2], rows[3]);
    let t4 = _mm256_unpacklo_epi32(rows[4], rows[5]);
    let t5 = _mm256_unpackhi_epi32(rows[4], rows[5]);
    let t6 = _mm256_unpacklo_epi32(rows[6], rows[7]);
    let t7 = _mm256_unpackhi_epi32(rows[6], rows[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

/// Writes 32 columns of a tile: `weights[r]` holds those columns of row `r`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn store_columns(q: *mut i8, weights: [__m256i; 8]) {
    for (step, lanes) in transpose(weights).into_iter().enumerate() {
        // SAFETY: the caller guarantees room for 8 steps of 32 bytes at `q`.
        unsafe { _mm256_storeu_si256(q.add(step * 32) as *mut __m256i, lanes) };
    }
}

/// Writes one value per row for each of 8 sub-blocks: `values[r]` holds the
/// eight values of row `r`. Each is written `repeat` times, for formats
/// whose scales span more than one sub-block of 16 columns.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn store_affine(out: *mut f32, values: [__m256; 8], repeat: usize) {
    let lanes = transpose(values.map(|row| _mm256_castps_si256(row)));
    for (index, lanes) in lanes.into_iter().enumerate() {
        for copy in 0..repeat {
            // SAFETY: the caller guarantees room for `8 * repeat` runs of
            // eight floats at `out`.
            unsafe {
                _mm256_storeu_ps(
                    out.add((index * repeat + copy) * 8),
                    _mm256_castsi256_ps(lanes),
                )
            };
        }
    }
}

/// Lanes that hold something other than zero, as a mask to gather with OR.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn nonzero(values: __m256) -> __m256 {
    _mm256_cmp_ps::<_CMP_NEQ_UQ>(values, _mm256_setzero_ps())
}

/// The half-float scales at byte `at` of each of eight rows.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn half_scales(rows: &[*const u8; 8], at: usize) -> __m256 {
    let mut halves = [0i16; 8];
    for (half, row) in halves.iter_mut().zip(rows) {
        // SAFETY: the caller guarantees two readable bytes at `at` in every
        // row.
        *half = unsafe { (row.add(at) as *const i16).read_unaligned() };
    }
    // SAFETY: eight halves are 16 readable bytes.
    _mm256_cvtph_ps(unsafe { _mm_loadu_si128(halves.as_ptr() as *const __m128i) })
}

/// The MXFP4 scales at byte `at` of each of eight rows: E8M0 exponents,
/// read the way [`quant::e8m0_half`] reads them.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn e8m0_scales(rows: &[*const u8; 8], at: usize) -> __m256 {
    let mut scales = [0.0f32; 8];
    for (scale, row) in scales.iter_mut().zip(rows) {
        // SAFETY: the caller guarantees a readable byte at `at` in every
        // row.
        *scale = quant::e8m0_half(unsafe { *row.add(at) });
    }
    // SAFETY: eight floats are 32 readable bytes.
    unsafe { _mm256_loadu_ps(scales.as_ptr()) }
}

/// Defines the fill of a format with blocks of 32 weights. `$weights`
/// unpacks the block at `$p` into 32 bytes, shifted by `$bias` where the
/// format is signed. `$scales` reads the scale of each row's block, one of
/// [`half_scales`] and [`e8m0_scales`]. Returns whether any offset is not
/// zero.
macro_rules! block32_fill {
    ($name:ident, $bytes:expr, $bias:expr, $scales:ident, |$p:ident| $weights:expr) => {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn $name(tile: &mut Tile, rows: [*const u8; 8]) -> bool {
            let (q, scale, offset) = (
                tile.q.as_mut_ptr(),
                tile.scale.as_mut_ptr(),
                tile.offset.as_mut_ptr(),
            );
            let zero = _mm256_setzero_ps();
            let mut any = zero;
            // SAFETY: the caller guarantees `tile.cols / 32` blocks behind
            // every row pointer and a tile sized for them.
            unsafe {
                for block in 0..tile.cols / 32 {
                    let mut weights = [_mm256_setzero_si256(); 8];
                    for r in 0..8 {
                        let $p = rows[r].add(block * $bytes);
                        weights[r] = $weights;
                    }
                    store_columns(q.add(block * 256), weights);
                    // One scale per row, shared by both sub-blocks.
                    let scales = $scales(&rows, block * $bytes);
                    let offsets = _mm256_sub_ps(zero, _mm256_mul_ps(_mm256_set1_ps($bias), scales));
                    for sub in block * 2..block * 2 + 2 {
                        _mm256_storeu_ps(scale.add(sub * 8), scales);
                        _mm256_storeu_ps(offset.add(sub * 8), offsets);
                    }
                    any = _mm256_or_ps(any, nonzero(offsets));
                }
            }
            _mm256_movemask_ps(any) != 0
        }
    };
}

block32_fill!(fill_q4_0, 18, 8.0, half_scales, |p| nibbles(p.add(2)));
block32_fill!(fill_q5_0, 22, 16.0, half_scales, |p| _mm256_or_si256(
    nibbles(p.add(6)),
    fifth_bits(p.add(2))
));
block32_fill!(fill_q8_0, 34, 0.0, half_scales, |p| _mm256_loadu_si256(
    p.add(2) as *const __m256i
));
// The non-linear formats look their nibbles up, and keep their signs.
block32_fill!(fill_iq4_nl, 18, 0.0, half_scales, |p| _mm256_shuffle_epi8(
    levels(&quant::IQ4_NL_LEVELS),
    nibbles(p.add(2))
));
block32_fill!(fill_mxfp4, 17, 0.0, e8m0_scales, |p| _mm256_shuffle_epi8(
    levels(&quant::MXFP4_LEVELS),
    nibbles(p.add(1))
));

/// The fill of NVFP4: blocks of 64 signed FP4 levels, which the tile keeps
/// as they are, and a scale for every sub-block of 16. The format has no
/// offsets, and its scales are finite, so the portable fill's offsets are
/// all zero.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn fill_nvfp4(tile: &mut Tile, rows: [*const u8; 8]) -> bool {
    const BYTES: usize = 36;
    let (q, scale, offset) = (
        tile.q.as_mut_ptr(),
        tile.scale.as_mut_ptr(),
        tile.offset.as_mut_ptr(),
    );
    // SAFETY: the caller guarantees `tile.cols / 64` blocks behind every
    // row pointer and a tile sized for them.
    unsafe {
        for block in 0..tile.cols / 64 {
            let mut first = [_mm256_setzero_si256(); 8];
            let mut second = [_mm256_setzero_si256(); 8];
            for r in 0..8 {
                let p = rows[r].add(block * BYTES);
                // Sub-blocks 0 and 2, and 1 and 3, put back in order.
                let (even, odd) = nvfp4_weights(p);
                first[r] = _mm256_permute2x128_si256::<0x20>(even, odd);
                second[r] = _mm256_permute2x128_si256::<0x31>(even, odd);
                for s in 0..4 {
                    *scale.add((block * 4 + s) * 8 + r) = quant::ue4m3_half(*p.add(s));
                }
            }
            store_columns(q.add(block * 512), first);
            store_columns(q.add(block * 512 + 256), second);
            for sub in block * 4..block * 4 + 4 {
                _mm256_storeu_ps(offset.add(sub * 8), _mm256_setzero_ps());
            }
        }
    }
    false
}

/// The fill of Q2_0: blocks of 64 values of 0 to 3, stored with a bias of
/// one, which is how the tile wants them, and one half scale per block.
/// Returns whether any offset is not zero.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn fill_q2_0(tile: &mut Tile, rows: [*const u8; 8]) -> bool {
    const BYTES: usize = 18;
    let (q, scale, offset) = (
        tile.q.as_mut_ptr(),
        tile.scale.as_mut_ptr(),
        tile.offset.as_mut_ptr(),
    );
    let zero = _mm256_setzero_ps();
    let mut any = zero;
    // SAFETY: the caller guarantees `tile.cols / 64` blocks behind every
    // row pointer and a tile sized for them.
    unsafe {
        for block in 0..tile.cols / 64 {
            for half in 0..2 {
                let mut values = [_mm256_setzero_si256(); 8];
                for r in 0..8 {
                    let p = rows[r].add(block * BYTES + 2);
                    let bytes = _mm256_castsi128_si256(_mm_loadu_si128(p as *const __m128i));
                    values[r] = q2_0_values(bytes, half);
                }
                store_columns(q.add((block * 2 + half) * 256), values);
            }
            // One scale per row for all four sub-blocks. The offsets take
            // the bias off, as `0 - 1 * scale`, the portable fill's sum.
            let scales = half_scales(&rows, block * BYTES);
            let offsets = _mm256_sub_ps(zero, _mm256_mul_ps(_mm256_set1_ps(1.0), scales));
            for sub in block * 4..block * 4 + 4 {
                _mm256_storeu_ps(scale.add(sub * 8), scales);
                _mm256_storeu_ps(offset.add(sub * 8), offsets);
            }
            any = _mm256_or_ps(any, nonzero(offsets));
        }
    }
    _mm256_movemask_ps(any) != 0
}

/// The scales and offsets of the Q4_K or Q5_K super-block at `p`, as
/// floats: eight of each, one per sub-block of 32 weights.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn k4_affine(p: *const u8) -> (__m256, __m256) {
    // SAFETY: the caller guarantees 16 readable bytes at `p`.
    let (w0, w1, w2) = unsafe { (word(p.add(4)), word(p.add(8)), word(p.add(12))) };
    // The same unpacking as `k4_scales`, one byte per value.
    let scales_low = w0 & 0x3f3f_3f3f;
    let minimums_low = w1 & 0x3f3f_3f3f;
    let scales_high = (w2 & 0x0f0f_0f0f) | (((w0 >> 6) & 0x0303_0303) << 4);
    let minimums_high = ((w2 >> 4) & 0x0f0f_0f0f) | (((w1 >> 6) & 0x0303_0303) << 4);
    let floats = |low: u32, high: u32| {
        let bytes = _mm_set_epi64x(0, (low as u64 | (high as u64) << 32) as i64);
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes))
    };
    // SAFETY: as above.
    let (d, dmin) = unsafe { (half(p), half(p.add(2))) };
    let scale = _mm256_mul_ps(d, floats(scales_low, scales_high));
    let minimum = _mm256_mul_ps(dmin, floats(minimums_low, minimums_high));
    (scale, _mm256_xor_ps(minimum, _mm256_set1_ps(-0.0)))
}

/// The scales and offsets of the IQ4_XS super-block at `p`, as floats: one
/// of each per run of 32 weights.
///
/// A run's scale has 4 low bits in a nibble of the bytes at `p + 4` and 2
/// high bits in the 16-bit word at `p + 2`. The format has no offsets, but
/// the portable fill writes `0 - 0 * scale`, which is not zero when the
/// scale is not a number, so that is what is written here too.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn iq4_xs_affine(p: *const u8) -> (__m256, __m256) {
    // SAFETY: the caller guarantees 8 readable bytes at `p`.
    let (high, low) = unsafe { ((p.add(2) as *const u16).read_unaligned(), word(p.add(4))) };
    let low_bits = _mm256_srlv_epi32(
        _mm256_set1_epi32(low as i32),
        _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28),
    );
    let high_bits = _mm256_srlv_epi32(
        _mm256_set1_epi32(high as i32),
        _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14),
    );
    let bits = _mm256_or_si256(
        _mm256_and_si256(low_bits, _mm256_set1_epi32(0x0f)),
        _mm256_slli_epi32::<4>(_mm256_and_si256(high_bits, _mm256_set1_epi32(3))),
    );
    let integers = _mm256_sub_epi32(bits, _mm256_set1_epi32(32));
    // SAFETY: as above.
    let scale = _mm256_mul_ps(unsafe { half(p) }, _mm256_cvtepi32_ps(integers));
    let zero = _mm256_setzero_ps();
    (scale, _mm256_sub_ps(zero, _mm256_mul_ps(zero, scale)))
}

/// Defines the fill of a format with super-blocks of eight runs of 32
/// weights: Q4_K, Q5_K and IQ4_XS. `$affine` yields the scales and offsets
/// of the super-block at its argument, one per run, and `$weights` the low
/// and the high run of pair `$pair` of the super-block at `$p`.
macro_rules! k4_fill {
    ($name:ident, $bytes:expr, $affine:ident, |$p:ident, $pair:ident| $weights:expr) => {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn $name(tile: &mut Tile, rows: [*const u8; 8]) -> bool {
            let (q, scale, offset) = (
                tile.q.as_mut_ptr(),
                tile.scale.as_mut_ptr(),
                tile.offset.as_mut_ptr(),
            );
            let mut any = _mm256_setzero_ps();
            // SAFETY: the caller guarantees `tile.cols / 256` super-blocks
            // behind every row pointer and a tile sized for them.
            unsafe {
                for block in 0..tile.cols / 256 {
                    let mut scales = [_mm256_setzero_ps(); 8];
                    let mut offsets = [_mm256_setzero_ps(); 8];
                    for r in 0..8 {
                        (scales[r], offsets[r]) = $affine(rows[r].add(block * $bytes));
                        any = _mm256_or_ps(any, nonzero(offsets[r]));
                    }
                    // Each scale covers 32 weights, two sub-blocks of 16.
                    store_affine(scale.add(block * 128), scales, 2);
                    store_affine(offset.add(block * 128), offsets, 2);
                    for $pair in 0..4 {
                        let mut low = [_mm256_setzero_si256(); 8];
                        let mut high = [_mm256_setzero_si256(); 8];
                        for r in 0..8 {
                            let $p = rows[r].add(block * $bytes);
                            (low[r], high[r]) = $weights;
                        }
                        store_columns(q.add((block * 8 + $pair * 2) * 256), low);
                        store_columns(q.add((block * 8 + $pair * 2 + 1) * 256), high);
                    }
                }
            }
            _mm256_movemask_ps(any) != 0
        }
    };
}

k4_fill!(fill_q4_k, 144, k4_affine, |p, pair| {
    let bytes = _mm256_loadu_si256(p.add(16 + pair * 32) as *const __m256i);
    let mask = _mm256_set1_epi8(0x0f);
    (
        _mm256_and_si256(bytes, mask),
        _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), mask),
    )
});

k4_fill!(fill_q5_k, 176, k4_affine, |p, pair| {
    let bytes = _mm256_loadu_si256(p.add(48 + pair * 32) as *const __m256i);
    let fifth = _mm256_loadu_si256(p.add(16) as *const __m256i);
    let mask = _mm256_set1_epi8(0x0f);
    // Bits `2 * pair` and `2 * pair + 1` of each byte are the fifth bits of
    // the low and the high weight.
    let bit = |shift: usize| {
        let wanted = _mm256_set1_epi8(1 << shift);
        let is_set = _mm256_cmpeq_epi8(_mm256_and_si256(fifth, wanted), wanted);
        _mm256_and_si256(is_set, _mm256_set1_epi8(0x10))
    };
    (
        _mm256_or_si256(_mm256_and_si256(bytes, mask), bit(2 * pair)),
        _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), mask),
            bit(2 * pair + 1),
        ),
    )
});

// Each run of IQ4_XS is laid out like an IQ4_NL block: 16 bytes, low
// nibbles first. A pair of runs is 32 bytes.
k4_fill!(fill_iq4_xs, 136, iq4_xs_affine, |p, pair| {
    let table = levels(&quant::IQ4_NL_LEVELS);
    (
        _mm256_shuffle_epi8(table, nibbles(p.add(8 + pair * 32))),
        _mm256_shuffle_epi8(table, nibbles(p.add(24 + pair * 32))),
    )
});

/// The fill of Q6_K. Weights are 4 low bits and 2 high bits, stored with a
/// bias of 32, which is how the tile wants them.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn fill_q6_k(tile: &mut Tile, rows: [*const u8; 8]) -> bool {
    const BYTES: usize = 210;
    let (q, scale, offset) = (
        tile.q.as_mut_ptr(),
        tile.scale.as_mut_ptr(),
        tile.offset.as_mut_ptr(),
    );
    let zero = _mm256_setzero_ps();
    let low_mask = _mm256_set1_epi8(0x0f);
    let two_bits = _mm256_set1_epi8(3);
    let mut any = zero;
    // SAFETY: the caller guarantees `tile.cols / 256` super-blocks behind
    // every row pointer and a tile sized for them.
    unsafe {
        for block in 0..tile.cols / 256 {
            // Sixteen signed 8-bit scales per row, times the block's scale.
            for part in 0..2 {
                let mut scales = [zero; 8];
                let mut offsets = [zero; 8];
                for r in 0..8 {
                    let p = rows[r].add(block * BYTES);
                    let raw = _mm_loadl_epi64(p.add(192 + part * 8) as *const __m128i);
                    let integers = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw));
                    scales[r] = _mm256_mul_ps(half(p.add(208)), integers);
                    offsets[r] =
                        _mm256_sub_ps(zero, _mm256_mul_ps(_mm256_set1_ps(32.0), scales[r]));
                    any = _mm256_or_ps(any, nonzero(offsets[r]));
                }
                store_affine(scale.add((block * 16 + part * 8) * 8), scales, 1);
                store_affine(offset.add((block * 16 + part * 8) * 8), offsets, 1);
            }
            // Each half of the block is 64 bytes of nibbles and 32 bytes of
            // high bit pairs, and unpacks to four runs of 32 weights.
            for part in 0..2 {
                let mut runs = [[_mm256_setzero_si256(); 8]; 4];
                for r in 0..8 {
                    let p = rows[r].add(block * BYTES);
                    let first = _mm256_loadu_si256(p.add(part * 64) as *const __m256i);
                    let second = _mm256_loadu_si256(p.add(part * 64 + 32) as *const __m256i);
                    let high = _mm256_loadu_si256(p.add(128 + part * 32) as *const __m256i);
                    let top =
                        |shift: __m256i| _mm256_slli_epi16::<4>(_mm256_and_si256(shift, two_bits));
                    runs[0][r] = _mm256_or_si256(_mm256_and_si256(first, low_mask), top(high));
                    runs[1][r] = _mm256_or_si256(
                        _mm256_and_si256(second, low_mask),
                        top(_mm256_srli_epi16::<2>(high)),
                    );
                    runs[2][r] = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(first), low_mask),
                        top(_mm256_srli_epi16::<4>(high)),
                    );
                    runs[3][r] = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16::<4>(second), low_mask),
                        top(_mm256_srli_epi16::<6>(high)),
                    );
                }
                for (run, weights) in runs.into_iter().enumerate() {
                    store_columns(q.add((block * 8 + part * 4 + run) * 256), weights);
                }
            }
        }
    }
    _mm256_movemask_ps(any) != 0
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

/// Eight 16-bit floats at `p`, widened to 32 bits by F16C.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_halves(p: *const u16) -> __m256 {
    // SAFETY: the caller guarantees eight readable halves at `p`.
    unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p.cast())) }
}

/// [`dot_rows_group`] with rows of 16-bit floats.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_rows_half_group<const G: usize>(
    queries: *const f32,
    rows: *const u16,
    n: usize,
    len: usize,
    scores: *mut f32,
    stride: usize,
) {
    // SAFETY: as in `dot_rows_group`, with `n` rows of `len` halves.
    unsafe {
        for j in 0..n {
            let row = rows.add(j * len);
            let mut acc = [_mm256_setzero_ps(); G];
            let mut i = 0;
            while i + 8 <= len {
                let values = load_halves(row.add(i));
                for (g, acc) in acc.iter_mut().enumerate() {
                    *acc = _mm256_fmadd_ps(values, _mm256_loadu_ps(queries.add(g * len + i)), *acc);
                }
                i += 8;
            }
            for (g, acc) in acc.iter().enumerate() {
                let mut sum = hsum(*acc);
                for t in i..len {
                    sum += f16_to_f32(*row.add(t)) * *queries.add(g * len + t);
                }
                *scores.add(g * stride + j) = sum;
            }
        }
    }
}

/// See [`Kernels::dot_rows_any`]. Queries are taken four at a time.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_rows_half(
    queries: *const f32,
    n_queries: usize,
    rows: *const u16,
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
                1 => dot_rows_half_group::<1>(q, rows, n, len, s, stride),
                2 => dot_rows_half_group::<2>(q, rows, n, len, s, stride),
                3 => dot_rows_half_group::<3>(q, rows, n, len, s, stride),
                _ => dot_rows_half_group::<4>(q, rows, n, len, s, stride),
            }
            h += 4;
        }
    }
}

/// [`mix_rows_block`] with rows of 16-bit floats.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn mix_rows_half_block<const G: usize, const K: usize>(
    weights: *const f32,
    rows: *const u16,
    n: usize,
    len: usize,
    outs: *mut f32,
    stride: usize,
) {
    // SAFETY: as in `mix_rows_block`, with rows of halves.
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
                *value = load_halves(row.add(k * 8));
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

/// See [`Kernels::mix_rows_any`]. Laid out like [`mix_rows`].
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn mix_rows_half(
    weights: *const f32,
    n_queries: usize,
    rows: *const u16,
    n: usize,
    len: usize,
    outs: *mut f32,
    stride: usize,
) {
    let mut h = 0;
    // SAFETY: forwarded contract, as in `mix_rows`.
    unsafe {
        while h < n_queries {
            let group = (n_queries - h).min(3);
            let (w, o) = (weights.add(h * stride), outs.add(h * len));
            let mut i = 0;
            while i + 32 <= len {
                match group {
                    1 => mix_rows_half_block::<1, 4>(w, rows.add(i), n, len, o.add(i), stride),
                    2 => mix_rows_half_block::<2, 4>(w, rows.add(i), n, len, o.add(i), stride),
                    _ => mix_rows_half_block::<3, 4>(w, rows.add(i), n, len, o.add(i), stride),
                }
                i += 32;
            }
            while i + 8 <= len {
                match group {
                    1 => mix_rows_half_block::<1, 1>(w, rows.add(i), n, len, o.add(i), stride),
                    2 => mix_rows_half_block::<2, 1>(w, rows.add(i), n, len, o.add(i), stride),
                    _ => mix_rows_half_block::<3, 1>(w, rows.add(i), n, len, o.add(i), stride),
                }
                i += 8;
            }
            for g in 0..group {
                for j in 0..n {
                    let weight = *w.add(g * stride + j);
                    for t in i..len {
                        *o.add(g * len + t) += weight * f16_to_f32(*rows.add(j * len + t));
                    }
                }
            }
            h += group;
        }
    }
}

/// See [`Kernels::to_half`]. F16C rounds to nearest, ties to even, as the
/// portable conversion does.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn to_half(x: *const f32, out: *mut u16, n: usize) {
    let mut i = 0;
    // SAFETY: the caller guarantees `n` reads and `n` writes.
    unsafe {
        while i + 8 <= n {
            let halves = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(x.add(i)));
            _mm_storeu_si128(out.add(i).cast(), halves);
            i += 8;
        }
        for t in i..n {
            *out.add(t) = f32_to_f16(*x.add(t));
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

/// See [`Kernels::quantize`]. The results equal the portable version's to
/// the last bit.
///
/// A group is scanned once for its largest magnitude. Then 32 floats at a
/// time are scaled, rounded and narrowed to bytes. Converting a float to an
/// integer rounds to the nearest, ties to even, which is the rounding the
/// portable version spells out.
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
    let sign = _mm256_set1_ps(-0.0);
    // Narrowing packs each half of a register on its own. This puts the
    // bytes back in order.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let to_unsigned = _mm256_set1_epi8(-128);

    // The eight floats at `$p`, scaled by `$inverse` and rounded. A value
    // that is not a number becomes zero, as it does in the portable version.
    macro_rules! rounded {
        ($p:expr, $inverse:expr) => {{
            let scaled = _mm256_mul_ps(_mm256_loadu_ps($p), $inverse);
            let is_number = _mm256_cmp_ps::<_CMP_ORD_Q>(scaled, scaled);
            _mm256_cvtps_epi32(_mm256_and_ps(scaled, is_number))
        }};
    }

    // SAFETY: the caller guarantees the buffer sizes documented on
    // `Kernels::quantize`, and `group` is a multiple of 32.
    unsafe {
        for g in 0..n / group {
            let values = x.add(g * group);
            let out = q.add(g * group);
            let group_sums = sums.add(g * group / 16);

            // `max` returns its second operand when the first is not a
            // number, so such values never become the largest.
            let mut largest = _mm256_setzero_ps();
            for i in (0..group).step_by(8) {
                let magnitudes = _mm256_andnot_ps(sign, _mm256_loadu_ps(values.add(i)));
                largest = _mm256_max_ps(magnitudes, largest);
            }
            let quad = _mm_max_ps(
                _mm256_castps256_ps128(largest),
                _mm256_extractf128_ps::<1>(largest),
            );
            let pair = _mm_max_ps(quad, _mm_movehl_ps(quad, quad));
            let max = _mm_cvtss_f32(_mm_max_ss(pair, _mm_shuffle_ps::<1>(pair, pair)));
            if max == 0.0 {
                std::ptr::write_bytes(out, 0, group);
                std::ptr::write_bytes(group_sums, 0, group / 16);
                *scale.add(g) = 0.0;
                *wsum.add(g) = 0.0;
                continue;
            }

            let inverse = _mm256_set1_ps(127.0 / max);
            let mut total = 0i64;
            for block in 0..group / 32 {
                let p = values.add(block * 32);
                let low = _mm256_packs_epi32(rounded!(p, inverse), rounded!(p.add(8), inverse));
                let high =
                    _mm256_packs_epi32(rounded!(p.add(16), inverse), rounded!(p.add(24), inverse));
                let bytes = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(low, high), order);
                _mm256_storeu_si256(out.add(block * 32) as *mut __m256i, bytes);
                // Sums of unsigned bytes come in one instruction, eight
                // bytes at a time. Each byte is shifted up by 128 for it.
                let eights =
                    _mm256_sad_epu8(_mm256_xor_si256(bytes, to_unsigned), _mm256_setzero_si256());
                let first = _mm256_extract_epi64::<0>(eights) + _mm256_extract_epi64::<1>(eights)
                    - 16 * 128;
                let second = _mm256_extract_epi64::<2>(eights) + _mm256_extract_epi64::<3>(eights)
                    - 16 * 128;
                *group_sums.add(block * 2) = first as i16;
                *group_sums.add(block * 2 + 1) = second as i16;
                total += first + second;
            }
            *scale.add(g) = max / 127.0;
            *wsum.add(g) = *scale.add(g) * total as i32 as f32;
        }
    }
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

/// A 16-entry table of signed bytes in both halves of a register, for
/// looking 4-bit indices up with `pshufb`.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn levels(table: &[i8; 16]) -> __m256i {
    // SAFETY: the table is 16 readable bytes.
    let half = unsafe { _mm_loadu_si128(table.as_ptr() as *const __m128i) };
    _mm256_broadcastsi128_si256(half)
}

/// IQ4_NL: `weight = scale * level[nibble]`, with the non-linear levels.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_iq4_nl(p: *const u8) -> SignedBlock {
    // SAFETY: the caller guarantees an 18-byte block at `p`.
    unsafe {
        SignedBlock {
            weights: _mm256_shuffle_epi8(levels(&quant::IQ4_NL_LEVELS), nibbles(p.add(2))),
            scale: half(p),
        }
    }
}

/// MXFP4: `weight = 2^(e - 128) * level[nibble]`, with the doubled FP4
/// levels.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn load_mxfp4(p: *const u8) -> SignedBlock {
    // SAFETY: the caller guarantees a 17-byte block at `p`.
    unsafe {
        SignedBlock {
            weights: _mm256_shuffle_epi8(levels(&quant::MXFP4_LEVELS), nibbles(p.add(1))),
            scale: _mm256_set1_ps(quant::e8m0_half(*p)),
        }
    }
}

signed_block_row!(row_q4_0, load_q4_0, 18);
signed_block_row!(row_q5_0, load_q5_0, 22);
signed_block_row!(row_q8_0, load_q8_0, 34);
signed_block_row!(row_iq4_nl, load_iq4_nl, 18);
signed_block_row!(row_mxfp4, load_mxfp4, 17);
offset_block_row!(row_q4_1, load_q4_1, 20);
offset_block_row!(row_q5_1, load_q5_1, 24);

// ---------------------------------------------------------------------------
// Formats with 64-weight blocks
// ---------------------------------------------------------------------------

/// The weights of the NVFP4 block at `p`, looked up in the doubled FP4
/// levels: sub-blocks 0 and 2 in the first register, 1 and 3 in the
/// second, each sub-block in order in its half.
///
/// A sub-block's 8 bytes hold its first 8 weights in their low nibbles
/// and its last 8 in their high nibbles. Interleaving the 64-bit halves of
/// the low and the high nibbles puts each sub-block in one 128-bit lane,
/// without moving anything across lanes.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn nvfp4_weights(p: *const u8) -> (__m256i, __m256i) {
    // SAFETY: the caller guarantees a 36-byte block at `p`.
    let bytes = unsafe { _mm256_loadu_si256(p.add(4) as *const __m256i) };
    let mask = _mm256_set1_epi8(0x0f);
    let low = _mm256_and_si256(bytes, mask);
    let high = _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), mask);
    let table = levels(&quant::MXFP4_LEVELS);
    (
        _mm256_shuffle_epi8(table, _mm256_unpacklo_epi64(low, high)),
        _mm256_shuffle_epi8(table, _mm256_unpackhi_epi64(low, high)),
    )
}

/// The four scales of the NVFP4 block at `p`, from the table of every
/// UE4M3 byte.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn nvfp4_scales(p: *const u8) -> __m128 {
    let scale = |s: usize| {
        // SAFETY: the caller guarantees a 36-byte block at `p`, which
        // starts with its four scales.
        quant::ue4m3_half(unsafe { *p.add(s) })
    };
    _mm_setr_ps(scale(0), scale(1), scale(2), scale(3))
}

/// Row kernel for NVFP4: 36-byte blocks of four sub-blocks of 16 weights,
/// each with an 8-bit float scale.
/// `weight = ue4m3(scale) / 2 * level[nibble]`.
///
/// Each register holds two sub-blocks, one in each 128-bit half. A half is
/// scaled by its sub-block's own scale times the scale of its group of
/// activations. The four products of a block are made once and spread to
/// the halves that need them. The two registers of a block have
/// accumulators of their own.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_nvfp4(row: *const u8, n: usize, act: ActRef) -> f32 {
    let first = _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2);
    let second = _mm256_setr_epi32(1, 1, 1, 1, 3, 3, 3, 3);
    let mut acc = [_mm256_setzero_ps(), _mm256_setzero_ps()];
    // SAFETY: the caller guarantees `row` holds `n / 64` blocks of 36 bytes
    // and `act` describes `n` activations in groups of 32.
    unsafe {
        for b in 0..n / 64 {
            let p = row.add(b * 36);
            let a = act.q.add(b * 64);
            let (even, odd) = nvfp4_weights(p);
            // Sub-blocks 0 and 2, then 1 and 3, as the weights hold them.
            let acts_even = _mm256_loadu2_m128i(a.add(32) as *const __m128i, a as *const __m128i);
            let acts_odd =
                _mm256_loadu2_m128i(a.add(48) as *const __m128i, a.add(16) as *const __m128i);
            // The activation scales of the block's two groups, each twice:
            // the first group holds sub-blocks 0 and 1.
            let pair = _mm_castsi128_ps(_mm_loadl_epi64(act.scale.add(2 * b) as *const __m128i));
            let scales = _mm_mul_ps(nvfp4_scales(p), _mm_unpacklo_ps(pair, pair));
            let scales = _mm256_castps128_ps256(scales);
            let scale_even = _mm256_permutevar8x32_ps(scales, first);
            let scale_odd = _mm256_permutevar8x32_ps(scales, second);
            acc[0] = _mm256_fmadd_ps(dot_signed(even, acts_even), scale_even, acc[0]);
            acc[1] = _mm256_fmadd_ps(dot_signed(odd, acts_odd), scale_odd, acc[1]);
        }
        hsum(_mm256_add_ps(acc[0], acc[1]))
    }
}

/// The 32 values of half `half` of a Q2_0 block, 0 to 3 each, in order.
/// `bytes` holds the block's 16 bytes of values in its low 128 bits.
///
/// A byte holds four neighboring values, lowest bits first. A 128-bit half
/// of the result holds 16 values, which come from 4 bytes. Each of its four
/// 32-bit words takes those 4 bytes and shifts them right by twice its
/// place, so word `j` holds value `4k + j` in its byte `k`. One byte
/// shuffle puts them in order.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn q2_0_values(bytes: __m256i, half: usize) -> __m256i {
    let words = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    let words = _mm256_add_epi32(words, _mm256_set1_epi32(2 * half as i32));
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let spread = _mm256_srlv_epi32(_mm256_permutevar8x32_epi32(bytes, words), shifts);
    let values = _mm256_and_si256(spread, _mm256_set1_epi8(3));
    // Byte `4j + k` of each half holds value `4k + j`.
    let order = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10,
        14, 3, 7, 11, 15,
    );
    _mm256_shuffle_epi8(values, order)
}

/// Row kernel for Q2_0: 18-byte blocks of 64 weights, each a 2-bit value
/// of 0 to 3 that stands for one less, and a half scale per block.
/// `weight = d * (value - 1)`.
///
/// The values are multiplied as they are, unsigned, which needs no sign
/// handling. The sum of the activations, which their quantizer already
/// computed as `act.wsum`, then takes the 1 back off every weight. The two
/// groups of activations in a block have accumulators of their own.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_q2_0(row: *const u8, n: usize, act: ActRef) -> f32 {
    let mut acc = [_mm256_setzero_ps(), _mm256_setzero_ps()];
    let mut correction = 0.0f32;
    // SAFETY: the caller guarantees `row` holds `n / 64` blocks of 18 bytes
    // and `act` describes `n` activations in groups of 32.
    unsafe {
        for b in 0..n / 64 {
            let p = row.add(b * 18);
            let d = half(p);
            let bytes = _mm256_castsi128_si256(_mm_loadu_si128(p.add(2) as *const __m128i));
            for (g, acc) in acc.iter_mut().enumerate() {
                let values = q2_0_values(bytes, g);
                let acts = _mm256_loadu_si256(act.q.add(b * 64 + g * 32) as *const __m256i);
                let scale = _mm256_mul_ps(d, broadcast(act.scale.add(2 * b + g)));
                *acc = _mm256_fmadd_ps(dot_unsigned(values, acts), scale, *acc);
            }
            let sums = *act.wsum.add(2 * b) + *act.wsum.add(2 * b + 1);
            correction += _mm256_cvtss_f32(d) * sums;
        }
        hsum(_mm256_add_ps(acc[0], acc[1])) - correction
    }
}

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

/// Row kernel for IQ4_XS: super-blocks of eight runs of 32, each laid out
/// like an IQ4_NL block, with a 6-bit scale per run.
/// `weight = d * (scale - 32) * level[nibble]`.
///
/// The levels are looked up with `pshufb`. Runs alternate between two
/// accumulators so consecutive runs do not wait on each other.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_iq4_xs(row: *const u8, n: usize, act: ActRef) -> f32 {
    let table = levels(&quant::IQ4_NL_LEVELS);
    let mut acc = [_mm256_setzero_ps(), _mm256_setzero_ps()];
    // SAFETY: the caller guarantees `row` holds `n / 256` super-blocks of
    // 136 bytes and `act` describes `n` quantized activations.
    unsafe {
        for sb in 0..n / 256 {
            let p = row.add(sb * 136);
            let d = _mm256_mul_ps(half(p), broadcast(act.scale.add(sb)));
            // Two high bits of each scale in a 16-bit word, four low bits
            // in a nibble of the next four bytes.
            let high = (p.add(2) as *const u16).read_unaligned() as u32;
            let low = word(p.add(4));
            for ib in 0..8 {
                let bits = ((low >> (4 * ib)) & 0x0f) | (((high >> (2 * ib)) & 3) << 4);
                let scale = _mm256_mul_ps(d, _mm256_set1_ps(bits as f32 - 32.0));
                let weights = _mm256_shuffle_epi8(table, nibbles(p.add(8 + 16 * ib)));
                let acts = _mm256_loadu_si256(act.q.add(sb * 256 + 32 * ib) as *const __m256i);
                acc[ib % 2] = _mm256_fmadd_ps(scale, dot_signed(weights, acts), acc[ib % 2]);
            }
        }
        hsum(_mm256_add_ps(acc[0], acc[1]))
    }
}

/// Row kernel for TQ2_0: 2-bit values of 0, 1 or 2 that stand for -1, 0
/// and 1, with one scale per 256.
///
/// The values are multiplied as they are, unsigned, which needs no sign
/// handling. The sum of the activations, which their quantizer already
/// computed as `act.wsum`, then takes the 1 back off every weight.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_tq2_0(row: *const u8, n: usize, act: ActRef) -> f32 {
    let two_bits = _mm256_set1_epi8(3);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut correction = 0.0f32;
    // SAFETY: the caller guarantees `row` holds `n / 256` blocks of 66
    // bytes and `act` describes `n` quantized activations.
    unsafe {
        for sb in 0..n / 256 {
            let p = row.add(sb * 66);
            let acts = act.q.add(sb * 256);
            let mut sum = _mm256_setzero_si256();
            for half_index in 0..2 {
                // Byte `m` of each half holds weights `m`, `m + 32`, `m + 64`
                // and `m + 96` of it, lowest bits first.
                let bytes = _mm256_loadu_si256(p.add(32 * half_index) as *const __m256i);
                let parts = [
                    bytes,
                    _mm256_srli_epi16::<2>(bytes),
                    _mm256_srli_epi16::<4>(bytes),
                    _mm256_srli_epi16::<6>(bytes),
                ];
                for (l, part) in parts.into_iter().enumerate() {
                    let values = _mm256_and_si256(part, two_bits);
                    let run =
                        _mm256_loadu_si256(acts.add(128 * half_index + 32 * l) as *const __m256i);
                    let pairs = _mm256_maddubs_epi16(values, run);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pairs, ones));
                }
            }
            let d = f16_to_f32((p.add(64) as *const u16).read_unaligned());
            let scale = _mm256_set1_ps(d * *act.scale.add(sb));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale, acc);
            correction += d * *act.wsum.add(sb);
        }
        hsum(acc) - correction
    }
}

// ---------------------------------------------------------------------------
// Formats without a kernel of their own
// ---------------------------------------------------------------------------

/// The exact integer sums of 32 products of signed bytes: one for the
/// first 16 and one for the last 16.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn sub_block_sums(weights: __m256i, acts: __m256i) -> [i32; 2] {
    let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(weights), _mm256_sign_epi8(acts, weights));
    // Four 32-bit lanes for each sub-block, the first sub-block lowest.
    let fours = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
    let low = _mm256_castsi256_si128(fours);
    let high = _mm256_extracti128_si256::<1>(fours);
    let twos = _mm_hadd_epi32(low, high);
    let ones = _mm_hadd_epi32(twos, twos);
    [_mm_cvtsi128_si32(ones), _mm_extract_epi32::<1>(ones)]
}

/// The row kernel of a format that has no AVX2 decoder.
///
/// Each block is decoded by its portable decoder in `tensor::quant` into
/// signed bytes and sub-block scales. The bytes are multiplied with AVX2,
/// 32 at a time, into one exact integer per sub-block. The float steps
/// that follow are the scalar kernel's, in its order, so the result is the
/// scalar kernel's to the bit. They are a few operations per 16 weights,
/// next to a decoder that takes far longer. The decoders are inlined
/// here, so they are compiled with AVX2 as well.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn dot_decoded(
    decode: Decoder,
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
        // `act` describes `n` quantized activations. A group holds 256
        // weights and 16 scales, so every index below is in bounds.
        unsafe {
            let block = slice::from_raw_parts(row.add(g * block_bytes), block_bytes);
            decode(block, &mut decoded);
            let acts = act.q.add(g * group);
            let sums = act.sums.add(g * group / 16);
            let mut partial = 0.0f32;
            for pair in 0..group / 32 {
                let at = pair * 32;
                let weights = _mm256_loadu_si256(decoded.q.as_ptr().add(at) as *const __m256i);
                let run = _mm256_loadu_si256(acts.add(at) as *const __m256i);
                let integers = sub_block_sums(weights, run);
                for (sub, integer) in (2 * pair..).zip(integers) {
                    partial += decoded.scale[sub] * integer as f32
                        + decoded.offset[sub] * *sums.add(sub) as f32;
                }
            }
            total += partial * *act.scale.add(g);
        }
    }
    total
}

/// Defines the row kernel of one format on top of [`dot_decoded`].
macro_rules! decoded_row {
    ($name:ident, $decode:path, $block_bytes:expr, $group:expr) => {
        #[target_feature(enable = "avx2,fma,f16c")]
        unsafe fn $name(row: *const u8, n: usize, act: ActRef) -> f32 {
            // SAFETY: forwarded contract.
            unsafe { dot_decoded($decode, $block_bytes, $group, row, n, act) }
        }
    };
}

decoded_row!(row_q2_k, quant::decode_q2_k, 84, 256);
decoded_row!(row_q3_k, quant::decode_q3_k, 110, 256);
decoded_row!(row_iq3_s, quant::decode_iq3_s, 110, 256);
decoded_row!(row_iq3_xxs, quant::decode_iq3_xxs, 98, 256);
decoded_row!(row_iq2_s, quant::decode_iq2_s, 82, 256);
decoded_row!(row_iq2_xs, quant::decode_iq2_xs, 74, 256);
decoded_row!(row_iq2_xxs, quant::decode_iq2_xxs, 66, 256);
decoded_row!(row_iq1_m, quant::decode_iq1_m, 56, 256);
decoded_row!(row_iq1_s, quant::decode_iq1_s, 50, 256);
decoded_row!(row_tq1_0, quant::decode_tq1_0, 54, 256);

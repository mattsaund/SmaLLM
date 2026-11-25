//! Scalar decoders for every quantized block format.
//!
//! This file is the single description of each format's bit layout. The SIMD
//! kernels in `kernels` reimplement the hot formats for speed, and are tested
//! against these decoders.
//!
//! Every format reduces to the same shape. A group of weights is a run of
//! small integers `q`, split into sub-blocks of 16. Each sub-block has a float
//! scale and offset:
//!
//! ```text
//! weight[i] = scale[i / 16] * q[i] + offset[i / 16]
//! ```
//!
//! A group is one block: 32 weights for the simple formats and MXFP4, 64
//! for Q2_0 and NVFP4, 128 for Q1_0, and 256 for the K, IQ and TQ formats.
//!
//! The IQ formats look weights up in grids of allowed patterns, which live
//! in `iq_tables.rs`. Each decoder here follows ggml's dequantization in
//! `ggml-quants.c` step for step, so that `dequantize_row` gives the same
//! floats as ggml, bit for bit. `tests/quant.rs` checks that on blocks that
//! ggml quantized.

use super::TensorType;
use super::f16::{bf16_to_f32, f16_to_f32};
use super::iq_tables::{
    IQ1S_GRID, IQ1S_LEVELS, IQ2_LEVELS, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID,
};
// The SIMD kernels look MXFP4 levels up themselves.
pub(crate) use super::iq_tables::MXFP4_LEVELS;

/// One decoded group of weights. A format fills as many values of `q` as
/// its block holds, and one scale and offset for each 16 of them. Formats
/// with blocks of 256 fill everything.
pub struct Group {
    /// The integer weights.
    pub q: [i8; 256],
    /// One scale per sub-block of 16 weights.
    pub scale: [f32; 16],
    /// One offset per sub-block of 16 weights.
    pub offset: [f32; 16],
}

impl Group {
    /// A zeroed group, ready to be decoded into.
    pub fn new() -> Group {
        Group {
            q: [0; 256],
            scale: [0.0; 16],
            offset: [0.0; 16],
        }
    }
}

impl Default for Group {
    fn default() -> Group {
        Group::new()
    }
}

/// Decodes one block of a quantized type into a [`Group`].
pub type Decoder = fn(block: &[u8], out: &mut Group);

/// One weight row decoded to the common shape:
///
/// ```text
/// weight[i] = scale[i / 16] * q[i] + offset[i / 16]
/// ```
#[derive(Default)]
pub struct DecodedRow {
    /// The integer weights.
    pub q: Vec<i8>,
    /// One scale per sub-block of 16 weights.
    pub scale: Vec<f32>,
    /// One offset per sub-block of 16 weights.
    pub offset: Vec<f32>,
}

/// Decodes one row of a quantized tensor into `row`.
///
/// `bytes` must hold a whole number of blocks of type `ty`. A float type has
/// no blocks and leaves the row empty.
pub fn decode_row(ty: TensorType, bytes: &[u8], row: &mut DecodedRow) {
    let n = match decoder(ty) {
        Some(_) => bytes.len() / ty.block_bytes() * ty.block_len(),
        None => 0,
    };
    row.q.resize(n, 0);
    row.scale.resize(n / 16, 0.0);
    row.offset.resize(n / 16, 0.0);
    // One loop per format, so that each has its decoder built in.
    match ty {
        TensorType::F32 | TensorType::F16 | TensorType::BF16 => {}
        TensorType::Q4_0 => decode_blocks(ty, bytes, row, decode_q4_0),
        TensorType::Q4_1 => decode_blocks(ty, bytes, row, decode_q4_1),
        TensorType::Q5_0 => decode_blocks(ty, bytes, row, decode_q5_0),
        TensorType::Q5_1 => decode_blocks(ty, bytes, row, decode_q5_1),
        TensorType::Q8_0 => decode_blocks(ty, bytes, row, decode_q8_0),
        TensorType::Iq4Nl => decode_blocks(ty, bytes, row, decode_iq4_nl),
        TensorType::Q2K => decode_blocks(ty, bytes, row, decode_q2_k),
        TensorType::Q3K => decode_blocks(ty, bytes, row, decode_q3_k),
        TensorType::Q4K => decode_blocks(ty, bytes, row, decode_q4_k),
        TensorType::Q5K => decode_blocks(ty, bytes, row, decode_q5_k),
        TensorType::Q6K => decode_blocks(ty, bytes, row, decode_q6_k),
        TensorType::Iq4Xs => decode_blocks(ty, bytes, row, decode_iq4_xs),
        TensorType::Iq3S => decode_blocks(ty, bytes, row, decode_iq3_s),
        TensorType::Iq3Xxs => decode_blocks(ty, bytes, row, decode_iq3_xxs),
        TensorType::Iq2S => decode_blocks(ty, bytes, row, decode_iq2_s),
        TensorType::Iq2Xs => decode_blocks(ty, bytes, row, decode_iq2_xs),
        TensorType::Iq2Xxs => decode_blocks(ty, bytes, row, decode_iq2_xxs),
        TensorType::Iq1M => decode_blocks(ty, bytes, row, decode_iq1_m),
        TensorType::Iq1S => decode_blocks(ty, bytes, row, decode_iq1_s),
        TensorType::Tq2_0 => decode_blocks(ty, bytes, row, decode_tq2_0),
        TensorType::Tq1_0 => decode_blocks(ty, bytes, row, decode_tq1_0),
        TensorType::Mxfp4 => decode_blocks(ty, bytes, row, decode_mxfp4),
        TensorType::Nvfp4 => decode_blocks(ty, bytes, row, decode_nvfp4),
        TensorType::Q1_0 => decode_blocks(ty, bytes, row, decode_q1_0),
        TensorType::Q2_0 => decode_blocks(ty, bytes, row, decode_q2_0),
    }
}

/// Decodes the blocks of a row one after another with `decode`.
#[inline(always)]
fn decode_blocks(
    ty: TensorType,
    bytes: &[u8],
    row: &mut DecodedRow,
    decode: impl Fn(&[u8], &mut Group),
) {
    let group = ty.block_len();
    let subs = group / 16;
    let mut decoded = Group::new();
    let blocks = bytes.chunks_exact(ty.block_bytes());
    let q = row.q.chunks_exact_mut(group);
    let scale = row.scale.chunks_exact_mut(subs);
    let offset = row.offset.chunks_exact_mut(subs);
    for (((block, q), scale), offset) in blocks.zip(q).zip(scale).zip(offset) {
        decode(block, &mut decoded);
        q.copy_from_slice(&decoded.q[..group]);
        scale.copy_from_slice(&decoded.scale[..subs]);
        offset.copy_from_slice(&decoded.offset[..subs]);
    }
}

/// How the integer weights of a format become non-negative: what to add to
/// each of them, and the largest value that gives. `None` for the formats
/// that use the whole signed byte.
///
/// A row shifted this way stands for the same weights once `bias * scale`
/// is taken off its offsets. Kernels that multiply an unsigned byte by a
/// signed one read such rows without any sign handling.
pub fn unsigned_range(ty: TensorType) -> Option<(i8, u8)> {
    match ty {
        TensorType::Q2K => Some((0, 3)),
        TensorType::Q3K => Some((4, 7)),
        TensorType::Q4_0 => Some((8, 15)),
        TensorType::Q4_1 | TensorType::Q4K => Some((0, 15)),
        TensorType::Q5_0 => Some((16, 31)),
        TensorType::Q5_1 | TensorType::Q5K => Some((0, 31)),
        TensorType::Q6K => Some((32, 63)),
        // The two small formats are stored with a bias of one. Kept that
        // way, the shift costs a tile an offset term and saves the sign
        // handling of every multiply.
        TensorType::Q1_0 => Some((1, 2)),
        TensorType::Q2_0 => Some((1, 3)),
        TensorType::Q8_0 | TensorType::Iq4Nl | TensorType::Iq4Xs => None,
        // Grid, ternary and FP4 weights are signed, and they stay that way.
        TensorType::Iq3S
        | TensorType::Iq3Xxs
        | TensorType::Iq2S
        | TensorType::Iq2Xs
        | TensorType::Iq2Xxs
        | TensorType::Iq1M
        | TensorType::Iq1S
        | TensorType::Tq2_0
        | TensorType::Tq1_0
        | TensorType::Mxfp4
        | TensorType::Nvfp4 => None,
        TensorType::F32 | TensorType::F16 | TensorType::BF16 => None,
    }
}

/// A buffer whose first element sits at the start of a cache line.
///
/// SIMD registers are loaded from a tile at steps of their own size. From
/// an aligned start none of those loads straddles two cache lines, which
/// costs a load twice over where it happens.
#[derive(Default)]
pub struct Aligned<T> {
    /// The allocation, a cache line longer than what is in use.
    data: Vec<T>,
    /// Index of the first element in use.
    start: usize,
    /// Number of elements in use.
    len: usize,
}

impl<T: Copy + Default> Aligned<T> {
    /// Bytes in a cache line on every CPU this engine has kernels for.
    const LINE: usize = 64;

    /// Sets the number of elements. What they hold afterwards is not
    /// defined: the caller writes all of them.
    pub fn resize(&mut self, len: usize) {
        let spare = Self::LINE / size_of::<T>();
        if self.data.len() < len + spare {
            self.data.resize(len + spare, T::default());
        }
        // The allocation is aligned for `T`, so the distance to the next
        // line is a whole number of elements, and fewer than `spare`.
        let address = self.data.as_ptr() as usize;
        self.start = (address.next_multiple_of(Self::LINE) - address) / size_of::<T>();
        self.len = len;
    }
}

impl<T> std::ops::Deref for Aligned<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.data[self.start..self.start + self.len]
    }
}

impl<T> std::ops::DerefMut for Aligned<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data[self.start..self.start + self.len]
    }
}

impl<T: Copy + Default> From<&[T]> for Aligned<T> {
    fn from(values: &[T]) -> Aligned<T> {
        let mut aligned = Aligned::default();
        aligned.resize(values.len());
        aligned.copy_from_slice(values);
        aligned
    }
}

/// Several weight rows, decoded and interleaved for the batched multiply.
///
/// Columns are taken four at a time. For each run of four columns the tile
/// holds those four weights of its first row, then of its second, and so
/// on. A SIMD register loaded from that spot has one 32-bit lane per row,
/// so a single multiply serves every row of the tile, and each lane's sum
/// is one row's result. Nothing has to be added across lanes at the end.
///
/// Scales and offsets are laid out the same way: for each sub-block of 16
/// columns, one value per row.
#[derive(Default)]
pub struct Tile {
    /// Rows in the tile. Rows past the ones filled are zero.
    pub height: usize,
    /// Columns in every row.
    pub cols: usize,
    /// `cols * height` integer weights.
    pub q: Aligned<i8>,
    /// `cols / 16 * height` scales.
    pub scale: Aligned<f32>,
    /// `cols / 16 * height` offsets.
    pub offset: Aligned<f32>,
    /// True when some offset is not zero. Other tiles skip the offset term.
    pub has_offset: bool,
    /// True when weights use the whole signed byte. Otherwise every weight
    /// lies between zero and `max`.
    pub signed: bool,
    /// The largest weight an unsigned tile can hold.
    pub max: u8,
    /// The row being decoded.
    pub(crate) row: DecodedRow,
    /// A row of zero bytes, which stands in for the rows a tile lacks.
    pub(crate) blank: Vec<u8>,
}

impl Tile {
    /// Sizes the tile for `height` rows of type `ty`, `row_bytes` long, and
    /// sets what follows from the type. The caller writes the contents.
    pub fn prepare(&mut self, ty: TensorType, height: usize, row_bytes: usize) {
        let cols = row_bytes / ty.block_bytes() * ty.block_len();
        let range = unsigned_range(ty);
        self.height = height;
        self.cols = cols;
        self.signed = range.is_none();
        self.max = range.map_or(0, |(_, max)| max);
        self.has_offset = false;
        self.q.resize(cols * height);
        self.scale.resize(cols / 16 * height);
        self.offset.resize(cols / 16 * height);
    }

    /// A row of `row_bytes` zero bytes. In every format it decodes to
    /// weights of zero, though not always to integer weights of zero: a
    /// zero nibble of IQ4_NL is the level -127, times a scale of zero.
    pub fn blank_row(&mut self, row_bytes: usize) -> &[u8] {
        if self.blank.len() < row_bytes {
            self.blank.resize(row_bytes, 0);
        }
        &self.blank[..row_bytes]
    }

    /// Zeroes the weights, scales and offsets of rows `first..height`. That
    /// is how [`fill`](Self::fill) leaves the rows it is not given, so a
    /// faster fill that reads them from a [`blank_row`](Self::blank_row)
    /// calls this to end up with the same tile.
    pub fn clear_rows(&mut self, first: usize) {
        let height = self.height;
        if first >= height {
            return;
        }
        for run in self.q.chunks_exact_mut(4 * height) {
            run[4 * first..].fill(0);
        }
        for scales in self.scale.chunks_exact_mut(height) {
            scales[first..].fill(0.0);
        }
        for offsets in self.offset.chunks_exact_mut(height) {
            offsets[first..].fill(0.0);
        }
    }

    /// Fills the tile with the rows stored in `data`, `row_bytes` apart.
    ///
    /// `data` holds at most `height` rows of type `ty`. This is the portable
    /// way to fill a tile, and the definition for the faster ones.
    pub fn fill(&mut self, ty: TensorType, height: usize, data: &[u8], row_bytes: usize) {
        let count = data.len() / row_bytes;
        assert!(count <= height, "a tile holds at most {height} rows");
        self.prepare(ty, height, row_bytes);
        let bias = unsigned_range(ty).map_or(0, |(bias, _)| bias);
        if count < height {
            self.q.fill(0);
            self.scale.fill(0.0);
            self.offset.fill(0.0);
        }
        for (r, bytes) in data.chunks_exact(row_bytes).enumerate() {
            decode_row(ty, bytes, &mut self.row);
            if bias != 0 {
                for q in self.row.q.iter_mut() {
                    *q += bias;
                }
            }
            // Four weights at a time move to this row's place in each run.
            let runs = self.q.chunks_exact_mut(4 * height);
            for (run, four) in runs.zip(self.row.q.chunks_exact(4)) {
                run[r * 4..r * 4 + 4].copy_from_slice(four);
            }
            let scales = self.scale.chunks_exact_mut(height);
            let offsets = self.offset.chunks_exact_mut(height);
            let decoded = self.row.scale.iter().zip(&self.row.offset);
            for ((scales, offsets), (scale, offset)) in scales.zip(offsets).zip(decoded) {
                let offset = offset - bias as f32 * scale;
                scales[r] = *scale;
                offsets[r] = offset;
                self.has_offset |= offset != 0.0;
            }
        }
    }
}

/// The decoder for a quantized type, or `None` for float types.
pub fn decoder(ty: TensorType) -> Option<Decoder> {
    Some(match ty {
        TensorType::F32 | TensorType::F16 | TensorType::BF16 => return None,
        TensorType::Q4_0 => decode_q4_0,
        TensorType::Q4_1 => decode_q4_1,
        TensorType::Q5_0 => decode_q5_0,
        TensorType::Q5_1 => decode_q5_1,
        TensorType::Q8_0 => decode_q8_0,
        TensorType::Iq4Nl => decode_iq4_nl,
        TensorType::Q2K => decode_q2_k,
        TensorType::Q3K => decode_q3_k,
        TensorType::Q4K => decode_q4_k,
        TensorType::Q5K => decode_q5_k,
        TensorType::Q6K => decode_q6_k,
        TensorType::Iq4Xs => decode_iq4_xs,
        TensorType::Iq3S => decode_iq3_s,
        TensorType::Iq3Xxs => decode_iq3_xxs,
        TensorType::Iq2S => decode_iq2_s,
        TensorType::Iq2Xs => decode_iq2_xs,
        TensorType::Iq2Xxs => decode_iq2_xxs,
        TensorType::Iq1M => decode_iq1_m,
        TensorType::Iq1S => decode_iq1_s,
        TensorType::Tq2_0 => decode_tq2_0,
        TensorType::Tq1_0 => decode_tq1_0,
        TensorType::Mxfp4 => decode_mxfp4,
        TensorType::Nvfp4 => decode_nvfp4,
        TensorType::Q1_0 => decode_q1_0,
        TensorType::Q2_0 => decode_q2_0,
    })
}

/// Expands one row of any tensor type to f32.
///
/// `bytes` must hold exactly `out.len()` elements of type `ty`.
pub fn dequantize_row(ty: TensorType, bytes: &[u8], out: &mut [f32]) {
    match ty {
        TensorType::F32 => {
            for (value, raw) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            }
        }
        TensorType::F16 => {
            for (value, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *value = f16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
            }
        }
        TensorType::BF16 => {
            for (value, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *value = bf16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
            }
        }
        _ => {
            let decode = decoder(ty).expect("quantized type");
            let block_len = ty.block_len();
            let mut group = Group::new();
            let blocks = bytes.chunks_exact(ty.block_bytes());
            for (block, values) in blocks.zip(out.chunks_exact_mut(block_len)) {
                decode(block, &mut group);
                for (i, value) in values.iter_mut().enumerate() {
                    *value = group.scale[i / 16] * group.q[i] as f32 + group.offset[i / 16];
                }
            }
        }
    }
}

/// Reads a little-endian half float at byte `at`.
#[inline]
fn half(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Sets the scale and offset of sub-blocks `first..first + count`.
#[inline]
fn set_affine(out: &mut Group, first: usize, count: usize, scale: f32, offset: f32) {
    for sub in first..first + count {
        out.scale[sub] = scale;
        out.offset[sub] = offset;
    }
}

/// Q4_0, 18 bytes: half scale, then 16 bytes of nibbles.
/// Low nibbles are weights 0..16, high nibbles are weights 16..32.
/// weight = scale * (nibble - 8)
pub(crate) fn decode_q4_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 18] = block.try_into().expect("Q4_0 block is 18 bytes");
    for j in 0..16 {
        out.q[j] = (b[2 + j] & 0x0f) as i8 - 8;
        out.q[j + 16] = (b[2 + j] >> 4) as i8 - 8;
    }
    set_affine(out, 0, 2, half(b, 0), 0.0);
}

/// Q4_1, 20 bytes: half scale, half minimum, then 16 bytes of nibbles.
/// weight = scale * nibble + minimum
pub(crate) fn decode_q4_1(block: &[u8], out: &mut Group) {
    let b: &[u8; 20] = block.try_into().expect("Q4_1 block is 20 bytes");
    for j in 0..16 {
        out.q[j] = (b[4 + j] & 0x0f) as i8;
        out.q[j + 16] = (b[4 + j] >> 4) as i8;
    }
    set_affine(out, 0, 2, half(b, 0), half(b, 2));
}

/// Q5_0, 22 bytes: half scale, 32 high bits, then 16 bytes of nibbles.
/// Bit `i` of the high bits is the fifth bit of weight `i`.
/// weight = scale * (value - 16)
pub(crate) fn decode_q5_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 22] = block.try_into().expect("Q5_0 block is 22 bytes");
    let high = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for j in 0..16 {
        let low_bit = ((high >> j) & 1) as u8;
        let high_bit = ((high >> (j + 16)) & 1) as u8;
        out.q[j] = ((b[6 + j] & 0x0f) | (low_bit << 4)) as i8 - 16;
        out.q[j + 16] = ((b[6 + j] >> 4) | (high_bit << 4)) as i8 - 16;
    }
    set_affine(out, 0, 2, half(b, 0), 0.0);
}

/// Q5_1, 24 bytes: half scale, half minimum, 32 high bits, 16 bytes of
/// nibbles. weight = scale * value + minimum
pub(crate) fn decode_q5_1(block: &[u8], out: &mut Group) {
    let b: &[u8; 24] = block.try_into().expect("Q5_1 block is 24 bytes");
    let high = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for j in 0..16 {
        let low_bit = ((high >> j) & 1) as u8;
        let high_bit = ((high >> (j + 16)) & 1) as u8;
        out.q[j] = ((b[8 + j] & 0x0f) | (low_bit << 4)) as i8;
        out.q[j + 16] = ((b[8 + j] >> 4) | (high_bit << 4)) as i8;
    }
    set_affine(out, 0, 2, half(b, 0), half(b, 2));
}

/// Q8_0, 34 bytes: half scale, then 32 signed bytes.
/// weight = scale * byte
pub(crate) fn decode_q8_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 34] = block.try_into().expect("Q8_0 block is 34 bytes");
    for j in 0..32 {
        out.q[j] = b[2 + j] as i8;
    }
    set_affine(out, 0, 2, half(b, 0), 0.0);
}

/// The non-linear levels a 4-bit IQ4_NL index selects.
pub const IQ4_NL_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_NL, 18 bytes: half scale, then 16 bytes of nibbles that index
/// [`IQ4_NL_LEVELS`]. weight = scale * level
pub(crate) fn decode_iq4_nl(block: &[u8], out: &mut Group) {
    let b: &[u8; 18] = block.try_into().expect("IQ4_NL block is 18 bytes");
    for j in 0..16 {
        out.q[j] = IQ4_NL_LEVELS[(b[2 + j] & 0x0f) as usize];
        out.q[j + 16] = IQ4_NL_LEVELS[(b[2 + j] >> 4) as usize];
    }
    set_affine(out, 0, 2, half(b, 0), 0.0);
}

/// Q2_K, 84 bytes: 16 scale bytes, 64 bytes of 2-bit values, half scale,
/// half minimum scale. Each scale byte holds a 4-bit scale (low) and a 4-bit
/// minimum (high) for one sub-block of 16.
/// weight = d * scale * value - dmin * minimum
pub(crate) fn decode_q2_k(block: &[u8], out: &mut Group) {
    let b: &[u8; 84] = block.try_into().expect("Q2_K block is 84 bytes");
    let d = half(b, 80);
    let dmin = half(b, 82);
    for half_index in 0..2 {
        for pair in 0..4 {
            let shift = 2 * pair;
            for side in 0..2 {
                let sub = half_index * 8 + pair * 2 + side;
                let packed = b[sub];
                out.scale[sub] = d * (packed & 0x0f) as f32;
                out.offset[sub] = -(dmin * (packed >> 4) as f32);
                let source = 16 + half_index * 32 + side * 16;
                for l in 0..16 {
                    out.q[sub * 16 + l] = ((b[source + l] >> shift) & 3) as i8;
                }
            }
        }
    }
}

/// Q3_K, 110 bytes: 32 bytes of high bits, 64 bytes of 2-bit low values,
/// 12 bytes holding sixteen 6-bit scales, half scale.
/// weight = d * (scale - 32) * (value - 4 if the high bit is clear)
pub(crate) fn decode_q3_k(block: &[u8], out: &mut Group) {
    let b: &[u8; 110] = block.try_into().expect("Q3_K block is 110 bytes");
    let d = half(b, 108);
    let packed = &b[96..108];
    for half_index in 0..2 {
        for pair in 0..4 {
            let shift = 2 * pair;
            let high_mask = 1u8 << (half_index * 4 + pair);
            for side in 0..2 {
                let sub = half_index * 8 + pair * 2 + side;
                // Scale `sub`: four bits from bytes 0..8, two from bytes 8..12.
                let (column, group) = (sub % 4, sub / 4);
                let low = if group < 2 {
                    packed[group * 4 + column] & 0x0f
                } else {
                    packed[(group - 2) * 4 + column] >> 4
                };
                let high = (packed[8 + column] >> (2 * group)) & 3;
                let scale = (low | (high << 4)) as i32 - 32;
                out.scale[sub] = d * scale as f32;
                out.offset[sub] = 0.0;
                let source = 32 + half_index * 32 + side * 16;
                for l in 0..16 {
                    let low_bits = ((b[source + l] >> shift) & 3) as i8;
                    let has_high = b[side * 16 + l] & high_mask != 0;
                    out.q[sub * 16 + l] = if has_high { low_bits } else { low_bits - 4 };
                }
            }
        }
    }
}

/// Unpacks the eight 6-bit scales and eight 6-bit minimums that Q4_K and
/// Q5_K store in 12 bytes. Returns `(scale, minimum)` for sub-block `j`.
#[inline]
fn k4_scale_min(packed: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (packed[j] & 63, packed[j + 4] & 63)
    } else {
        (
            (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4),
            (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4),
        )
    }
}

/// Q4_K, 144 bytes: half scale, half minimum scale, 12 bytes of packed
/// scales, 128 bytes of nibbles. Eight sub-blocks of 32.
/// weight = d * scale * nibble - dmin * minimum
pub(crate) fn decode_q4_k(block: &[u8], out: &mut Group) {
    let b: &[u8; 144] = block.try_into().expect("Q4_K block is 144 bytes");
    let d = half(b, 0);
    let dmin = half(b, 2);
    for j in 0..8 {
        let (scale, minimum) = k4_scale_min(&b[4..16], j);
        set_affine(out, j * 2, 2, d * scale as f32, -(dmin * minimum as f32));
    }
    for pair in 0..4 {
        for l in 0..32 {
            let byte = b[16 + pair * 32 + l];
            out.q[pair * 64 + l] = (byte & 0x0f) as i8;
            out.q[pair * 64 + 32 + l] = (byte >> 4) as i8;
        }
    }
}

/// Q5_K, 176 bytes: half scale, half minimum scale, 12 bytes of packed
/// scales, 32 bytes of high bits, 128 bytes of nibbles.
/// weight = d * scale * value - dmin * minimum
pub(crate) fn decode_q5_k(block: &[u8], out: &mut Group) {
    let b: &[u8; 176] = block.try_into().expect("Q5_K block is 176 bytes");
    let d = half(b, 0);
    let dmin = half(b, 2);
    for j in 0..8 {
        let (scale, minimum) = k4_scale_min(&b[4..16], j);
        set_affine(out, j * 2, 2, d * scale as f32, -(dmin * minimum as f32));
    }
    for pair in 0..4 {
        for l in 0..32 {
            let byte = b[48 + pair * 32 + l];
            let high = b[16 + l];
            let low_bit = (high >> (2 * pair)) & 1;
            let high_bit = (high >> (2 * pair + 1)) & 1;
            out.q[pair * 64 + l] = ((byte & 0x0f) | (low_bit << 4)) as i8;
            out.q[pair * 64 + 32 + l] = ((byte >> 4) | (high_bit << 4)) as i8;
        }
    }
}

/// Q6_K, 210 bytes: 128 bytes of low nibbles, 64 bytes of high bit pairs,
/// 16 signed scale bytes, half scale. Sixteen sub-blocks of 16.
/// weight = d * scale * (value - 32)
pub(crate) fn decode_q6_k(block: &[u8], out: &mut Group) {
    let b: &[u8; 210] = block.try_into().expect("Q6_K block is 210 bytes");
    let d = half(b, 208);
    for sub in 0..16 {
        out.scale[sub] = d * (b[192 + sub] as i8) as f32;
        out.offset[sub] = 0.0;
    }
    for half_index in 0..2 {
        let base = half_index * 128;
        for l in 0..32 {
            let first = b[half_index * 64 + l];
            let second = b[half_index * 64 + 32 + l];
            let high = b[128 + half_index * 32 + l];
            out.q[base + l] = ((first & 0x0f) | ((high & 3) << 4)) as i8 - 32;
            out.q[base + 32 + l] = ((second & 0x0f) | (((high >> 2) & 3) << 4)) as i8 - 32;
            out.q[base + 64 + l] = ((first >> 4) | (((high >> 4) & 3) << 4)) as i8 - 32;
            out.q[base + 96 + l] = ((second >> 4) | (((high >> 6) & 3) << 4)) as i8 - 32;
        }
    }
}

/// IQ4_XS, 136 bytes: half scale, a 16-bit word of high scale bits, 4
/// bytes of low scale bits, then 128 bytes of nibbles that index
/// [`IQ4_NL_LEVELS`]. Eight sub-blocks of 32, each laid out like an IQ4_NL
/// block without its scale. Each has a 6-bit scale: 4 low bits from a
/// nibble, low nibble first, and 2 high bits from the word.
/// weight = d * (scale - 32) * level
pub(crate) fn decode_iq4_xs(block: &[u8], out: &mut Group) {
    let b: &[u8; 136] = block.try_into().expect("IQ4_XS block is 136 bytes");
    let d = half(b, 0);
    let high_bits = u16::from_le_bytes([b[2], b[3]]);
    for ib in 0..8 {
        let low = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0x0f;
        let high = ((high_bits >> (2 * ib)) & 3) as u8;
        let scale = (low | (high << 4)) as i32 - 32;
        set_affine(out, 2 * ib, 2, d * scale as f32, 0.0);
        for j in 0..16 {
            let byte = b[8 + 16 * ib + j];
            out.q[32 * ib + j] = IQ4_NL_LEVELS[(byte & 0x0f) as usize];
            out.q[32 * ib + 16 + j] = IQ4_NL_LEVELS[(byte >> 4) as usize];
        }
    }
}

/// The eight signs of a group in IQ2_XXS, IQ2_XS and IQ3_XXS. Seven bits
/// are stored. The eighth makes the number of minus signs even, which is
/// what ggml's `ksigns_iq2xs` table holds. Bit `j` set means weight `j` is
/// negative.
#[inline(always)]
fn even_signs(stored: u32) -> u8 {
    let seven = (stored & 127) as u8;
    seven | (((seven.count_ones() & 1) as u8) << 7)
}

// The grid decoders handle a group of eight weights as the eight bytes of
// a 64-bit word, first weight in the lowest byte. Expanding a grid code,
// applying signs and storing then take a few word operations for all
// eight weights together.

/// Builds the table that expands one byte of a packed grid code, four
/// 2-bit indices into `levels`, into a word of those four levels, first
/// level in the lowest byte. Runs when compiling.
const fn quads(levels: [i8; 4]) -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut j = 0;
        while j < 4 {
            let level = levels[(byte >> (2 * j)) & 3] as u8;
            table[byte] |= (level as u32) << (8 * j);
            j += 1;
        }
        byte += 1;
    }
    table
}

/// Eight times each of `levels`. See [`put_iq1`].
const fn eight_times(levels: [i8; 4]) -> [i8; 4] {
    let mut j = 0;
    let mut times = [0i8; 4];
    while j < 4 {
        times[j] = 8 * levels[j];
        j += 1;
    }
    times
}

/// The four magnitudes of each byte of an IQ2 grid code.
static IQ2_QUADS: [u32; 256] = quads(IQ2_LEVELS);

/// Eight times the four values of each byte of an IQ1 grid code.
static IQ1_QUADS: [u32; 256] = quads(eight_times(IQ1S_LEVELS));

/// Expands a packed grid code to its eight values, through a table of
/// [`quads`].
#[inline(always)]
fn expand(quads: &[u32; 256], code: u16) -> u64 {
    quads[(code & 0xff) as usize] as u64 | (quads[(code >> 8) as usize] as u64) << 32
}

/// Negates the weights of a group whose bit is set in `signs`, all eight at
/// once. Every byte must lie between 1 and 127, so that negating one never
/// borrows from its neighbor. The grid magnitudes all do.
#[inline(always)]
fn apply_signs(values: u64, signs: u8) -> u64 {
    // Byte `j` keeps bit `j` of the signs.
    let spread = (signs as u64).wrapping_mul(0x0101_0101_0101_0101) & 0x8040_2010_0804_0201;
    // Adding 0x7f sets the top bit of the bytes whose bit is set, and no
    // byte overflows into the next.
    let top = (spread + 0x7f7f_7f7f_7f7f_7f7f) & 0x8080_8080_8080_8080;
    let mask = (top >> 7) * 0xff;
    // Negating is inverting and adding one.
    (values ^ mask) + (mask & 0x0101_0101_0101_0101)
}

/// Adds two groups of eight weights byte by byte, without carries from one
/// byte into the next: the low 7 bits are added, and the top bit of each
/// byte is worked out on its own.
#[inline(always)]
fn add_bytes(a: u64, b: u64) -> u64 {
    const TOP: u64 = 0x8080_8080_8080_8080;
    ((a & !TOP) + (b & !TOP)) ^ ((a ^ b) & TOP)
}

/// Writes a group of eight weights to the start of `out`.
#[inline(always)]
fn put8(out: &mut [i8], group: u64) {
    for (out, byte) in out[..8].iter_mut().zip(group.to_le_bytes()) {
        *out = byte as i8;
    }
}

/// The eight magnitudes of an IQ3 group: two grid entries of four.
#[inline(always)]
fn iq3_pair(grid: &[u32], first: usize, second: usize) -> u64 {
    grid[first] as u64 | (grid[second] as u64) << 32
}

/// The scale of a group in IQ2 and IQ3_XXS, `d * (0.5 + nibble) * factor`.
/// The operations run in the order ggml runs them, so the result rounds
/// the same way.
#[inline(always)]
fn half_step_scale(d: f32, nibble: u32, factor: f32) -> f32 {
    d * (0.5 + nibble as f32) * factor
}

/// IQ2_XXS, 66 bytes: half scale, then 8 bytes for each group of 32. The
/// first 4 index [`IQ2XXS_GRID`], one per 8 weights. The other 4 are a
/// 32-bit word: four 7-bit sign fields, then a 4-bit scale on top.
/// weight = d * (0.5 + scale) / 4 * sign * grid
pub(crate) fn decode_iq2_xxs(block: &[u8], out: &mut Group) {
    let b: &[u8; 66] = block.try_into().expect("IQ2_XXS block is 66 bytes");
    let d = half(b, 0);
    for ib in 0..8 {
        let at = 2 + 8 * ib;
        let word = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]);
        set_affine(out, 2 * ib, 2, half_step_scale(d, word >> 28, 0.25), 0.0);
        for l in 0..4 {
            let values = expand(&IQ2_QUADS, IQ2XXS_GRID[b[at + l] as usize]);
            let signs = even_signs(word >> (7 * l));
            put8(&mut out.q[32 * ib + 8 * l..], apply_signs(values, signs));
        }
    }
}

/// Sets the scales of group `ib` in IQ2_XS and IQ2_S from its scale byte:
/// two 4-bit scales, the low nibble for the first 16 weights.
#[inline(always)]
fn set_iq2_scales(out: &mut Group, ib: usize, d: f32, byte: u8) {
    let byte = byte as u32;
    set_affine(out, 2 * ib, 1, half_step_scale(d, byte & 0x0f, 0.25), 0.0);
    set_affine(out, 2 * ib + 1, 1, half_step_scale(d, byte >> 4, 0.25), 0.0);
}

/// IQ2_XS, 74 bytes: half scale, 32 16-bit words, 8 scale bytes. Each word
/// is a group of 8 weights: a 9-bit index into [`IQ2XS_GRID`] with 7 sign
/// bits on top. Each scale byte holds two 4-bit scales, one for each 16
/// weights of a group of 32, low nibble first.
/// weight = d * (0.5 + scale) / 4 * sign * grid
pub(crate) fn decode_iq2_xs(block: &[u8], out: &mut Group) {
    let b: &[u8; 74] = block.try_into().expect("IQ2_XS block is 74 bytes");
    let d = half(b, 0);
    for ib in 0..8 {
        set_iq2_scales(out, ib, d, b[66 + ib]);
        for l in 0..4 {
            let at = 2 + 2 * (4 * ib + l);
            let word = u16::from_le_bytes([b[at], b[at + 1]]);
            let values = expand(&IQ2_QUADS, IQ2XS_GRID[(word & 511) as usize]);
            let signs = even_signs((word >> 9) as u32);
            put8(&mut out.q[32 * ib + 8 * l..], apply_signs(values, signs));
        }
    }
}

/// IQ2_S, 82 bytes: half scale, 32 bytes of low index bits, 32 sign bytes,
/// 8 bytes of high index bits, 8 scale bytes. Each group of 8 weights has
/// a 10-bit index into [`IQ2S_GRID`], 8 bits from its own byte and 2 from
/// the high bits of its group of 32, and a whole byte of signs. The scales
/// are as in IQ2_XS.
/// weight = d * (0.5 + scale) / 4 * sign * grid
pub(crate) fn decode_iq2_s(block: &[u8], out: &mut Group) {
    let b: &[u8; 82] = block.try_into().expect("IQ2_S block is 82 bytes");
    let d = half(b, 0);
    for ib in 0..8 {
        set_iq2_scales(out, ib, d, b[74 + ib]);
        let high = b[66 + ib] as usize;
        for l in 0..4 {
            let index = b[2 + 4 * ib + l] as usize | ((high << (8 - 2 * l)) & 0x300);
            let values = expand(&IQ2_QUADS, IQ2S_GRID[index]);
            let signs = b[34 + 4 * ib + l];
            put8(&mut out.q[32 * ib + 8 * l..], apply_signs(values, signs));
        }
    }
}

/// IQ3_XXS, 98 bytes: half scale, 64 bytes that index [`IQ3XXS_GRID`], one
/// per 4 weights, then a 32-bit word for each group of 32: four 7-bit sign
/// fields and a 4-bit scale on top.
/// weight = d * (0.5 + scale) / 2 * sign * grid
pub(crate) fn decode_iq3_xxs(block: &[u8], out: &mut Group) {
    let b: &[u8; 98] = block.try_into().expect("IQ3_XXS block is 98 bytes");
    let d = half(b, 0);
    for ib in 0..8 {
        let at = 66 + 4 * ib;
        let word = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
        set_affine(out, 2 * ib, 2, half_step_scale(d, word >> 28, 0.5), 0.0);
        for l in 0..4 {
            let first = b[2 + 8 * ib + 2 * l] as usize;
            let second = b[3 + 8 * ib + 2 * l] as usize;
            let values = iq3_pair(&IQ3XXS_GRID, first, second);
            let signs = even_signs(word >> (7 * l));
            put8(&mut out.q[32 * ib + 8 * l..], apply_signs(values, signs));
        }
    }
}

/// IQ3_S, 110 bytes: half scale, 64 bytes of low index bits, 8 bytes of
/// high index bits, 32 sign bytes, 4 scale bytes. Each 4 weights have a
/// 9-bit index into [`IQ3S_GRID`], 8 bits from their own byte and 1 from
/// the high bits of their group of 32. Each group of 32 has a 4-bit scale,
/// two to a byte, low nibble first.
/// weight = d * (1 + 2 * scale) * sign * grid
pub(crate) fn decode_iq3_s(block: &[u8], out: &mut Group) {
    let b: &[u8; 110] = block.try_into().expect("IQ3_S block is 110 bytes");
    let d = half(b, 0);
    for ib in 0..8 {
        let nibble = (b[106 + ib / 2] >> (4 * (ib % 2))) & 0x0f;
        set_affine(out, 2 * ib, 2, d * (1 + 2 * nibble as i32) as f32, 0.0);
        let high = b[66 + ib] as usize;
        for l in 0..4 {
            let first = b[2 + 8 * ib + 2 * l] as usize | ((high << (8 - 2 * l)) & 256);
            let second = b[3 + 8 * ib + 2 * l] as usize | ((high << (7 - 2 * l)) & 256);
            let values = iq3_pair(&IQ3S_GRID, first, second);
            let signs = b[74 + 4 * ib + l];
            put8(&mut out.q[32 * ib + 8 * l..], apply_signs(values, signs));
        }
    }
}

/// Writes the eight weights of an IQ1 grid code to the start of `out`, as
/// `8 * grid + delta`. `delta` is [`iq1_delta`].
///
/// IQ1 weights are `dl * (grid + delta / 8)`, with grid values of -1, 0 or
/// 1 and a delta of -1 or 1. Eight times `grid + delta / 8` is a small
/// integer, and the scale becomes `dl / 8`. Both steps are exact, so the
/// weight comes out as the same float ggml computes.
#[inline(always)]
fn put_iq1(out: &mut [i8], code: u16, delta: u64) {
    put8(out, add_bytes(expand(&IQ1_QUADS, code), delta));
}

/// The delta of an IQ1 group of 8, in eighths, in each of eight bytes: -1
/// when its sign bit is set, else 1. See [`put_iq1`].
#[inline(always)]
fn iq1_delta(negative: bool) -> u64 {
    if negative {
        u64::MAX
    } else {
        0x0101_0101_0101_0101
    }
}

/// IQ1_S, 50 bytes: half scale, 32 bytes of low index bits, then a 16-bit
/// word for each group of 32. The word holds the high 3 bits of the four
/// 11-bit indices into [`IQ1S_GRID`], then a 3-bit scale, then the sign of
/// the group's delta in the top bit.
/// weight = d * (2 * scale + 1) * (grid + delta / 8)
pub(crate) fn decode_iq1_s(block: &[u8], out: &mut Group) {
    let b: &[u8; 50] = block.try_into().expect("IQ1_S block is 50 bytes");
    let d = half(b, 0);
    for ib in 0..8 {
        let word = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as usize;
        let dl = d * (2 * ((word >> 12) & 7) + 1) as f32;
        set_affine(out, 2 * ib, 2, dl / 8.0, 0.0);
        let delta = iq1_delta(word & 0x8000 != 0);
        for l in 0..4 {
            let index = b[2 + 4 * ib + l] as usize | (((word >> (3 * l)) & 7) << 8);
            put_iq1(&mut out.q[32 * ib + 8 * l..], IQ1S_GRID[index], delta);
        }
    }
}

/// IQ1_M, 56 bytes: 32 bytes of low index bits, 16 bytes of high index
/// bits and delta signs, then four 16-bit scale words. The block's half
/// scale is spread over the top 4 bits of those words, first word lowest.
/// The other 12 bits of each word are four 3-bit scales, one for each 16
/// weights. A byte of the 16 serves two groups of 8: in each nibble, 3
/// high index bits and then the sign of the delta.
/// weight = d * (2 * scale + 1) * (grid + delta / 8)
pub(crate) fn decode_iq1_m(block: &[u8], out: &mut Group) {
    let b: &[u8; 56] = block.try_into().expect("IQ1_M block is 56 bytes");
    let words: [u16; 4] =
        std::array::from_fn(|k| u16::from_le_bytes([b[48 + 2 * k], b[49 + 2 * k]]));
    let d = f16_to_f32(
        (words[0] >> 12)
            | ((words[1] >> 8) & 0x00f0)
            | ((words[2] >> 4) & 0x0f00)
            | (words[3] & 0xf000),
    );
    for ib in 0..8 {
        let scales = (words[ib / 2] >> (6 * (ib % 2))) as usize;
        let first = d * (2 * (scales & 7) + 1) as f32;
        let second = d * (2 * ((scales >> 3) & 7) + 1) as f32;
        set_affine(out, 2 * ib, 1, first / 8.0, 0.0);
        set_affine(out, 2 * ib + 1, 1, second / 8.0, 0.0);
        let low = b[32 + 2 * ib] as usize;
        let high = b[33 + 2 * ib] as usize;
        let groups = [
            (b[4 * ib] as usize | ((low << 8) & 0x700), low & 0x08),
            (b[4 * ib + 1] as usize | ((low << 4) & 0x700), low & 0x80),
            (b[4 * ib + 2] as usize | ((high << 8) & 0x700), high & 0x08),
            (b[4 * ib + 3] as usize | ((high << 4) & 0x700), high & 0x80),
        ];
        for (l, (index, sign)) in groups.into_iter().enumerate() {
            let delta = iq1_delta(sign != 0);
            put_iq1(&mut out.q[32 * ib + 8 * l..], IQ1S_GRID[index], delta);
        }
    }
}

/// TQ2_0, 66 bytes: 64 bytes of 2-bit values, then a half scale. In each
/// half of the block, byte `m` holds weights `m`, `m + 32`, `m + 64` and
/// `m + 96` of that half's 128, lowest bits first.
/// weight = d * (value - 1)
pub(crate) fn decode_tq2_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 66] = block.try_into().expect("TQ2_0 block is 66 bytes");
    for half_index in 0..2 {
        for l in 0..4 {
            for m in 0..32 {
                let value = (b[32 * half_index + m] >> (2 * l)) & 3;
                out.q[128 * half_index + 32 * l + m] = value as i8 - 1;
            }
        }
    }
    set_affine(out, 0, 16, half(b, 64), 0.0);
}

/// TQ1_0, 54 bytes: 48 bytes of five weights each, 4 bytes of four weights
/// each, then a half scale. The bytes come in runs of 32, 16 and 4. Digit
/// `n` of byte `m` in a run is weight `n * run + m` of that run's weights.
/// weight = d * digit
pub(crate) fn decode_tq1_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 54] = block.try_into().expect("TQ1_0 block is 54 bytes");
    let (first, rest) = out.q.split_at_mut(160);
    let (second, third) = rest.split_at_mut(80);
    put_trits::<32>(b[..32].try_into().expect("a run of 32"), 5, first);
    put_trits::<16>(b[32..48].try_into().expect("a run of 16"), 5, second);
    put_trits::<4>(b[48..52].try_into().expect("a run of 4"), 4, third);
    set_affine(out, 0, 16, half(b, 52), 0.0);
}

/// Writes `digits` digits of each byte of a TQ1_0 run as -1, 0 or 1: digit
/// `n` of byte `m` goes to `out[n * LEN + m]`.
///
/// A byte holds its digits as a fraction of 256, the first digit most
/// significant. Tripling the byte carries the next digit into the high
/// byte of the product, and leaves the digits still to come in the low
/// byte. This is ggml's `byte * 3^n` modulo 256, one factor at a time.
/// Each digit is one loop over the run, which the compiler vectorizes.
#[inline(always)]
fn put_trits<const LEN: usize>(bytes: &[u8; LEN], digits: usize, out: &mut [i8]) {
    let mut rest = *bytes;
    for weights in out.chunks_exact_mut(LEN).take(digits) {
        for (weight, rest) in weights.iter_mut().zip(rest.iter_mut()) {
            let tripled = *rest as u16 * 3;
            *weight = (tripled >> 8) as i8 - 1;
            *rest = tripled as u8;
        }
    }
}

/// Reads an E8M0 exponent `e` as 2 to the `e - 128`, half the power of two
/// it stands for. The MXFP4 levels are doubled, and this halves them back.
/// The two smallest exponents give subnormal floats, as in ggml.
#[inline]
pub(crate) fn e8m0_half(e: u8) -> f32 {
    let bits = if e < 2 {
        0x0020_0000u32 << e
    } else {
        (e as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

/// MXFP4, 17 bytes: an E8M0 exponent, then 16 bytes of nibbles that index
/// [`MXFP4_LEVELS`]. Low nibbles are weights 0..16, high nibbles 16..32.
/// weight = 2^(e - 128) * level
pub(crate) fn decode_mxfp4(block: &[u8], out: &mut Group) {
    let b: &[u8; 17] = block.try_into().expect("MXFP4 block is 17 bytes");
    for j in 0..16 {
        out.q[j] = MXFP4_LEVELS[(b[1 + j] & 0x0f) as usize];
        out.q[j + 16] = MXFP4_LEVELS[(b[1 + j] >> 4) as usize];
    }
    set_affine(out, 0, 2, e8m0_half(b[0]), 0.0);
}

/// Reads an unsigned E4M3 scale of NVFP4 as half the value it stands for,
/// as ggml's `ggml_ue4m3_to_fp32` does. The FP4 levels are doubled, and
/// this halves them back.
#[inline(always)]
pub(crate) fn ue4m3_half(x: u8) -> f32 {
    UE4M3_HALF[x as usize]
}

/// [`ue4m3_half`] of every byte. A table, because a lookup is cheaper than
/// the arithmetic, and the SIMD kernels read it too. Every entry is a
/// normal float, so reading one is exact even where the CPU is set to
/// flush tiny floats to zero.
pub(crate) static UE4M3_HALF: [f32; 256] = ue4m3_table();

/// Builds [`UE4M3_HALF`] while compiling, from the bits of each float.
///
/// A byte has 4 exponent bits with a bias of 7 and 3 mantissa bits. ggml
/// ignores the top bit, and reads both zero and 0x7F, which E4M3 uses for
/// NaN, as zero. 0xFF has the same low bits as 0x7F but is not caught by
/// that rule, so it reads as 240. These cases follow ggml exactly.
const fn ue4m3_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut x = 0;
    while x < 256 {
        let exponent = ((x >> 3) & 0x0f) as u32;
        let mantissa = (x & 7) as u32;
        let bits = if x == 0x7f || (exponent == 0 && mantissa == 0) {
            0
        } else if exponent == 0 {
            // A subnormal: mantissa * 2^-9, halved. As an f32 it is normal,
            // with the mantissa's top bit as the leading one.
            let top = 31 - mantissa.leading_zeros();
            ((top + 117) << 23) | ((mantissa - (1 << top)) << (23 - top))
        } else {
            // (1 + mantissa / 8) * 2^(exponent - 7), halved.
            ((exponent + 119) << 23) | (mantissa << 20)
        };
        table[x] = f32::from_bits(bits);
        x += 1;
    }
    table
}

/// NVFP4, 36 bytes: four UE4M3 scales, one for each 16 weights, then 32
/// bytes of nibbles that index [`MXFP4_LEVELS`]. Each sub-block of 16 has
/// 8 bytes of its own. Byte `j` of them holds weight `j` in its low nibble
/// and weight `j + 8` in its high nibble.
/// weight = ue4m3(scale) / 2 * level
pub(crate) fn decode_nvfp4(block: &[u8], out: &mut Group) {
    let b: &[u8; 36] = block.try_into().expect("NVFP4 block is 36 bytes");
    for sub in 0..4 {
        for j in 0..8 {
            let byte = b[4 + 8 * sub + j];
            out.q[16 * sub + j] = MXFP4_LEVELS[(byte & 0x0f) as usize];
            out.q[16 * sub + j + 8] = MXFP4_LEVELS[(byte >> 4) as usize];
        }
        set_affine(out, sub, 1, ue4m3_half(b[sub]), 0.0);
    }
}

/// Q1_0, 18 bytes: half scale, then 16 bytes of sign bits for 128 weights.
/// Bit `j % 8` of byte `j / 8` belongs to weight `j`. A set bit is plus.
/// weight = d or -d
pub(crate) fn decode_q1_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 18] = block.try_into().expect("Q1_0 block is 18 bytes");
    for (j, q) in out.q[..128].iter_mut().enumerate() {
        let bit = (b[2 + j / 8] >> (j % 8)) & 1;
        *q = 2 * bit as i8 - 1;
    }
    set_affine(out, 0, 8, half(b, 0), 0.0);
}

/// Q2_0, 18 bytes: half scale, then 16 bytes of 2-bit values for 64
/// weights. Byte `j / 4` holds weight `j`, starting at bit `2 * (j % 4)`.
/// weight = d * (value - 1)
pub(crate) fn decode_q2_0(block: &[u8], out: &mut Group) {
    let b: &[u8; 18] = block.try_into().expect("Q2_0 block is 18 bytes");
    for (j, q) in out.q[..64].iter_mut().enumerate() {
        *q = ((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i8 - 1;
    }
    set_affine(out, 0, 4, half(b, 0), 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::f16::f32_to_f16;

    /// Writes a half float into `block` at byte `at`.
    fn put_half(block: &mut [u8], at: usize, value: f32) {
        block[at..at + 2].copy_from_slice(&f32_to_f16(value).to_le_bytes());
    }

    fn row(ty: TensorType, block: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0; ty.block_len()];
        dequantize_row(ty, block, &mut out);
        out
    }

    #[test]
    fn float_rows_are_read_little_endian() {
        let bytes: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = [0.0; 2];
        dequantize_row(TensorType::F32, &bytes, &mut out);
        assert_eq!(out, [1.5, -2.0]);

        let bytes: Vec<u8> = [0x3c00u16, 0xc000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        dequantize_row(TensorType::F16, &bytes, &mut out);
        assert_eq!(out, [1.0, -2.0]);
        let bytes: Vec<u8> = [0x3f80u16, 0xc000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        dequantize_row(TensorType::BF16, &bytes, &mut out);
        assert_eq!(out, [1.0, -2.0]);
    }

    #[test]
    fn q8_0_scales_signed_bytes() {
        let mut block = [0u8; 34];
        put_half(&mut block, 0, 0.5);
        block[2] = 4;
        block[3] = (-6i8) as u8;
        block[33] = 127;
        let out = row(TensorType::Q8_0, &block);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -3.0);
        assert_eq!(out[31], 63.5);
    }

    #[test]
    fn q4_0_puts_low_nibbles_first() {
        let mut block = [0u8; 18];
        put_half(&mut block, 0, 1.0);
        block[2] = 0xf0; // weight 0 has nibble 0, weight 16 has nibble 15
        block[17] = 0x09; // weight 15 has nibble 9, weight 31 has nibble 0
        let out = row(TensorType::Q4_0, &block);
        assert_eq!(out[0], -8.0);
        assert_eq!(out[16], 7.0);
        assert_eq!(out[15], 1.0);
        assert_eq!(out[31], -8.0);
    }

    #[test]
    fn q4_1_adds_the_minimum() {
        let mut block = [0u8; 20];
        put_half(&mut block, 0, 0.25);
        put_half(&mut block, 2, -1.0);
        block[4] = 0x84;
        let out = row(TensorType::Q4_1, &block);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], 1.0);
        assert_eq!(out[1], -1.0);
    }

    #[test]
    fn q5_formats_take_the_fifth_bit_from_the_mask() {
        let mut block = [0u8; 22];
        put_half(&mut block, 0, 1.0);
        block[2..6].copy_from_slice(&((1u32 << 3) | (1u32 << 20)).to_le_bytes());
        block[6 + 3] = 0x05;
        block[6 + 4] = 0x70;
        let out = row(TensorType::Q5_0, &block);
        assert_eq!(out[3], 5.0); // 16 + 5 - 16
        assert_eq!(out[20], 7.0); // 16 + 7 - 16
        assert_eq!(out[0], -16.0);

        let mut block = [0u8; 24];
        put_half(&mut block, 0, 2.0);
        put_half(&mut block, 2, 0.5);
        block[4..8].copy_from_slice(&(1u32 << 31).to_le_bytes());
        block[8 + 15] = 0xf0;
        let out = row(TensorType::Q5_1, &block);
        assert_eq!(out[31], 62.5); // (16 + 15) * 2 + 0.5
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn iq4_nl_uses_the_level_table() {
        let mut block = [0u8; 18];
        put_half(&mut block, 0, 1.0);
        block[2] = 0xf0;
        let out = row(TensorType::Iq4Nl, &block);
        assert_eq!(out[0], -127.0);
        assert_eq!(out[16], 113.0);
    }

    #[test]
    fn q4_k_unpacks_scales_and_minimums() {
        let mut block = [0u8; 144];
        put_half(&mut block, 0, 1.0);
        put_half(&mut block, 2, 1.0);
        // Sub-block 0: scale 3, minimum 2. Sub-block 5: scale 33, minimum 50.
        block[4] = 3;
        block[4 + 4] = 2;
        block[4 + 9] = (33 & 0x0f) | ((50 & 0x0f) << 4);
        block[4 + 1] = (33 >> 4) << 6;
        block[4 + 5] = (50 >> 4) << 6;
        block[16] = 0x05; // weight 0 nibble 5, weight 32 nibble 0
        block[16 + 64] = 0x70; // weight 128 nibble 0, weight 160 nibble 7
        let out = row(TensorType::Q4K, &block);
        assert_eq!(out[0], 3.0 * 5.0 - 2.0);
        assert_eq!(out[1], -2.0);
        assert_eq!(out[160], 33.0 * 7.0 - 50.0);
        assert_eq!(out[161], -50.0);
    }

    #[test]
    fn q5_k_adds_the_high_bit() {
        let mut block = [0u8; 176];
        put_half(&mut block, 0, 1.0);
        block[4] = 1; // sub-block 0 scale 1
        block[4 + 1] = 1; // sub-block 1 scale 1
        block[16 + 7] = 0b11; // weight 7 and weight 39 get +16
        block[48 + 7] = 0x21;
        let out = row(TensorType::Q5K, &block);
        assert_eq!(out[7], 17.0);
        assert_eq!(out[39], 18.0);
    }

    #[test]
    fn q6_k_combines_nibbles_and_bit_pairs() {
        let mut block = [0u8; 210];
        put_half(&mut block, 208, 1.0);
        for sub in 0..16 {
            block[192 + sub] = (sub as i8 - 8) as u8;
        }
        block[5] = 0xa3; // weight 5 low nibble 3, weight 69 low nibble 10
        block[128 + 5] = 0b0001_0010; // weight 5 high 2, weight 69 high 1
        let out = row(TensorType::Q6K, &block);
        assert_eq!(out[5], -8.0 * ((3 | (2 << 4)) - 32) as f32);
        assert_eq!(out[69], -4.0 * ((10 | (1 << 4)) - 32) as f32);
        assert_eq!(out[255], 7.0 * -32.0);
    }

    #[test]
    fn q2_k_and_q3_k_place_sub_blocks_in_order() {
        let mut block = [0u8; 84];
        put_half(&mut block, 80, 1.0);
        put_half(&mut block, 82, 1.0);
        block[3] = 0x25; // sub-block 3: scale 5, minimum 2
        block[16 + 16] = 0b0000_1100; // sub-block 3 weight 0 has value 3
        let out = row(TensorType::Q2K, &block);
        assert_eq!(out[48], 5.0 * 3.0 - 2.0);
        assert_eq!(out[49], -2.0);

        let mut block = [0u8; 110];
        put_half(&mut block, 108, 1.0);
        // All sixteen scales set to 33, which decodes to +1.
        for i in 0..8 {
            block[96 + i] = 0x11;
        }
        for i in 8..12 {
            block[96 + i] = 0xaa;
        }
        block[32] = 0b01; // sub-block 0 weight 0 low bits 1
        block[0] = 0b1; // with the high bit set, so no -4
        let out = row(TensorType::Q3K, &block);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -4.0);
    }

    /// The 64-bit FNV-1a hash, which `tools/gen_iq_tables.py` records for
    /// each of ggml's grids.
    fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3)
        })
    }

    #[test]
    fn grid_codes_unpack_to_the_bytes_of_ggml_grids() {
        use crate::tensor::iq_tables::GRID_HASHES;
        // Each grid expanded with the helpers its decoders use.
        let iq2 = |grid: &[u16]| -> Vec<u8> {
            let groups = grid.iter().map(|&code| expand(&IQ2_QUADS, code));
            groups.flat_map(u64::to_le_bytes).collect()
        };
        let iq3 = |grid: &[u32]| -> Vec<u8> { grid.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let iq1 = |grid: &[u16]| -> Vec<u8> {
            let mut weights = [0i8; 8];
            let mut all = Vec::new();
            for &code in grid {
                // A delta of zero leaves eight times the grid values.
                put_iq1(&mut weights, code, 0);
                all.extend(weights.iter().map(|weight| (weight / 8) as u8));
            }
            all
        };
        for (name, hash) in GRID_HASHES {
            let unpacked = match name {
                "IQ2XXS" => iq2(&IQ2XXS_GRID),
                "IQ2XS" => iq2(&IQ2XS_GRID),
                "IQ2S" => iq2(&IQ2S_GRID),
                "IQ3XXS" => iq3(&IQ3XXS_GRID),
                "IQ3S" => iq3(&IQ3S_GRID),
                "IQ1S" => iq1(&IQ1S_GRID),
                _ => panic!("no grid named {name}"),
            };
            assert_eq!(fnv1a(unpacked), hash, "{name}");
        }
    }

    #[test]
    fn word_operations_match_byte_by_byte_ones() {
        // Every sign byte on magnitudes from 1 to 127, and byte sums that
        // cross zero in both directions.
        let magnitudes = [1u8, 8, 25, 43, 62, 127, 4, 15];
        let values = u64::from_le_bytes(magnitudes);
        for signs in 0..=255u8 {
            let got = apply_signs(values, signs).to_le_bytes();
            for j in 0..8 {
                let expected = if signs & (1 << j) != 0 {
                    -(magnitudes[j] as i8)
                } else {
                    magnitudes[j] as i8
                };
                assert_eq!(got[j] as i8, expected, "signs {signs:#04x} byte {j}");
            }
        }
        let a = [-8i8, 0, 8, -8, 0, 8, 127, -128];
        for b in [[1i8; 8], [-1i8; 8], [5, -5, 7, -7, 1, -1, 0, 0]] {
            let word = |bytes: [i8; 8]| u64::from_le_bytes(bytes.map(|v| v as u8));
            let got = add_bytes(word(a), word(b)).to_le_bytes();
            for j in 0..8 {
                assert_eq!(got[j] as i8, a[j].wrapping_add(b[j]), "{a:?} + {b:?}");
            }
        }
    }

    #[test]
    fn the_eighth_sign_keeps_the_count_of_minus_signs_even() {
        // Entries of ggml's ksigns_iq2xs table.
        for (stored, signs) in [(0, 0), (1, 129), (3, 3), (7, 135), (100, 228), (127, 255)] {
            assert_eq!(even_signs(stored), signs);
            assert_eq!(
                even_signs(stored | 128),
                signs,
                "bits above the seventh are ignored"
            );
        }
    }

    #[test]
    fn e8m0_exponents_include_the_subnormal_ones() {
        assert_eq!(e8m0_half(0), 2f32.powi(-128));
        assert_eq!(e8m0_half(1), 2f32.powi(-127));
        assert_eq!(e8m0_half(2), 2f32.powi(-126));
        assert_eq!(e8m0_half(128), 1.0);
        assert_eq!(e8m0_half(255), 2f32.powi(127));
    }

    #[test]
    fn tq1_0_reads_the_digits_ggml_packs() {
        // Packs base-3 digits the way ggml's quantize_row_tq1_0_ref does:
        // the first digit most significant, scaled up to a fraction of 256
        // and rounded up.
        let pack = |digits: &[u8]| -> u8 {
            let value = digits
                .iter()
                .fold(0u16, |value, digit| value * 3 + *digit as u16);
            let value = value * 3u16.pow(5 - digits.len() as u32);
            (value * 256).div_ceil(243) as u8
        };
        let weights: Vec<u8> = (0..256).map(|i| ((i * 7 + i / 5) % 3) as u8).collect();
        let mut block = [0u8; 54];
        for m in 0..32 {
            let digits: Vec<u8> = (0..5).map(|n| weights[n * 32 + m]).collect();
            block[m] = pack(&digits);
        }
        for m in 0..16 {
            let digits: Vec<u8> = (0..5).map(|n| weights[160 + n * 16 + m]).collect();
            block[32 + m] = pack(&digits);
        }
        for m in 0..4 {
            let digits: Vec<u8> = (0..4).map(|n| weights[240 + n * 4 + m]).collect();
            block[48 + m] = pack(&digits);
        }
        put_half(&mut block, 52, 0.5);
        let out = row(TensorType::Tq1_0, &block);
        for (i, weight) in weights.iter().enumerate() {
            assert_eq!(out[i], 0.5 * (*weight as f32 - 1.0), "weight {i}");
        }
    }

    #[test]
    fn iq1_m_gathers_its_scale_from_four_nibbles() {
        let mut block = [0u8; 56];
        let bits = f32_to_f16(0.25);
        for k in 0..4 {
            block[49 + 2 * k] = (((bits >> (4 * k)) & 0x0f) as u8) << 4;
        }
        // Group 0 takes grid entry 0, eight values of -1. The first
        // 16 weights have scale 3 (times 2, plus 1) and a positive delta.
        block[48] = 3;
        let out = row(TensorType::Iq1M, &block);
        assert_eq!(out[0], 0.25 * 7.0 * (-1.0 + 0.125));
        assert_eq!(out[16], 0.25 * (-1.0 + 0.125));
    }

    /// ggml's `ggml_ue4m3_to_fp32`, written out the way ggml computes it.
    fn ggml_ue4m3_to_fp32(x: u8) -> f32 {
        if x == 0 || x == 0x7f {
            return 0.0;
        }
        let pow2 = |e: i32| f32::from_bits(((e + 127) as u32) << 23);
        let exponent = ((x >> 3) & 0x0f) as i32;
        let mantissa = (x & 7) as f32;
        let raw = if exponent == 0 {
            mantissa * pow2(-9)
        } else {
            (1.0 + mantissa / 8.0) * pow2(exponent - 7)
        };
        raw * 0.5
    }

    #[test]
    fn ue4m3_scales_read_like_ggml() {
        for x in 0..=255u8 {
            let (got, want) = (ue4m3_half(x), ggml_ue4m3_to_fp32(x));
            assert_eq!(got.to_bits(), want.to_bits(), "{x:#04x}: {got:e} {want:e}");
        }
        // One, the largest finite scale, the smallest one, and the edges.
        assert_eq!(ue4m3_half(0x38), 0.5);
        assert_eq!(ue4m3_half(0x7e), 224.0);
        assert_eq!(ue4m3_half(0x01), 2f32.powi(-10));
        assert_eq!(ue4m3_half(0x7f), 0.0);
        assert_eq!(ue4m3_half(0xff), 240.0);
        assert_eq!(ue4m3_half(0xb8), 0.5, "the top bit is ignored");
    }

    #[test]
    fn nvfp4_sub_blocks_have_scales_and_bytes_of_their_own() {
        let mut block = [0u8; 36];
        block[..4].copy_from_slice(&[0x38, 0x40, 0x30, 0x01]);
        block[4] = 0xf1; // sub-block 0: weight 0 has level 1, weight 8 level -12
        block[4 + 8 + 7] = 0x27; // sub-block 1: weight 7 has level 12, weight 15 level 2
        block[4 + 24] = 0x09; // sub-block 3: weight 0 has level -1
        let out = row(TensorType::Nvfp4, &block);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[8], -6.0);
        assert_eq!(out[16 + 7], 12.0);
        assert_eq!(out[16 + 15], 2.0);
        assert_eq!(out[48], -(2f32.powi(-10)));
        assert!(out[32..48].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q2_0_packs_four_weights_into_each_byte() {
        let mut block = [0u8; 18];
        put_half(&mut block, 0, 0.5);
        block[2] = 0b11_10_01_00; // weights 0 to 3: values 0, 1, 2 and 3
        block[17] = 0b00_11_01_10; // weights 60 to 63: values 2, 1, 3 and 0
        let out = row(TensorType::Q2_0, &block);
        assert_eq!(out[..4], [-0.5, 0.0, 0.5, 1.0]);
        assert_eq!(out[60..], [0.5, 0.0, 1.0, -0.5]);
        assert_eq!(out[4], -0.5);
    }

    #[test]
    fn q1_0_keeps_one_sign_per_bit() {
        let mut block = [0u8; 18];
        put_half(&mut block, 0, 0.25);
        block[2] = 0b1000_0001; // weights 0 and 7 are positive
        block[17] = 0b0100_0000; // weight 126 is positive
        let out = row(TensorType::Q1_0, &block);
        assert_eq!(
            out[..8],
            [0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.25]
        );
        assert_eq!(out[126], 0.25);
        assert_eq!(out[127], -0.25);
        assert_eq!(out.iter().filter(|v| **v > 0.0).count(), 3);
    }
}

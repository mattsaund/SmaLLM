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
//! A group is 32 weights for the simple formats and 256 for the K formats.

use super::TensorType;
use super::f16::{bf16_to_f32, f16_to_f32};

/// One decoded group of weights. Simple formats fill the first 32 values of
/// `q` and the first 2 scales and offsets. K formats fill everything.
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
        TensorType::Q8_0 | TensorType::Iq4Nl => None,
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
    /// weights of zero.
    pub fn blank_row(&mut self, row_bytes: usize) -> &[u8] {
        if self.blank.len() < row_bytes {
            self.blank.resize(row_bytes, 0);
        }
        &self.blank[..row_bytes]
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
}

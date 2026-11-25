//! The engine's hot loops: dot products and element-wise math.
//!
//! Nearly all inference time is spent multiplying weight matrices by
//! activation vectors. Weights stay in their quantized file format. Before a
//! multiply, the activation vector is quantized to 8-bit integers once, and
//! each weight row is then reduced with integer dot products.
//!
//! Three implementations exist, with identical results up to float rounding:
//!
//! | Backend  | Used on                                   |
//! |----------|-------------------------------------------|
//! | `scalar` | Every target. The reference for the tests. |
//! | `x86`    | x86-64 with AVX2, FMA and F16C.           |
//! | `arm`    | AArch64 with NEON, and SDOT when present. |
//!
//! [`best`] picks one when the process starts.

mod scalar;

#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "x86_64")]
mod x86;

use std::sync::OnceLock;

use crate::tensor::TensorType;
use crate::tensor::quant::Tile;

/// A fast `exp` that loops vectorize, for model code outside the kernels:
/// the Mamba scan and RWKV's decays.
pub(crate) use scalar::exp_approx;

/// One activation vector, quantized for integer dot products.
///
/// The vector is split into groups of 32 or 256 values to match the weight
/// format. Each group has its own scale: `value[i] = scale[group] * q[i]`.
#[derive(Clone, Copy)]
pub struct ActRef {
    /// The quantized values, one per element.
    pub q: *const i8,
    /// One scale per group.
    pub scale: *const f32,
    /// The sum of `q` over each run of 16 elements.
    pub sums: *const i16,
    /// `scale * sum(q)` for each group.
    pub wsum: *const f32,
}

/// Dot product of a quantized weight row and a quantized activation vector.
///
/// `n` is the element count, a multiple of the weight type's block length.
pub type DotQuant = unsafe fn(row: *const u8, n: usize, act: ActRef) -> f32;

/// Dot product of a float weight row, stored as bytes, and float activations.
pub type DotFloat = unsafe fn(row: *const u8, x: *const f32, n: usize) -> f32;

/// Reduces every row of a [`Tile`] against one quantized activation vector:
///
/// ```text
/// out[r] = sum over i of (scale[i / 16][r] * q[i][r] * act.q[i] * act.scale[i / group])
///        + sum over s of (offset[s][r] * scaled_sums[s])
/// ```
///
/// `scaled_sums` holds `act.scale * sum(act.q)` for each run of 16 elements
/// and is only read when the tile has offsets. `group` is the group size
/// the activations were quantized with, 32 or 256. `out` receives one float
/// per row of the tile.
pub type TileDot =
    unsafe fn(tile: &Tile, act: ActRef, scaled_sums: *const f32, group: usize, out: *mut f32);

/// [`TileDot`] for four activation vectors at once. `out` receives the
/// outputs of each vector in turn, one float per row of the tile.
///
/// It must give exactly what four calls to the backend's [`TileDot`] give,
/// so that a token's results never depend on how its batch is grouped. A
/// backend has one when reading a weight once for several tokens pays.
pub type TileDot4 = unsafe fn(
    tile: &Tile,
    acts: &[ActRef; 4],
    scaled_sums: &[*const f32; 4],
    group: usize,
    out: *mut f32,
);

/// Fills a tile with the rows in `data`, exactly as [`Tile::fill`] does, or
/// as [`Tile::fill_signed`] does for a backend with signed tiles.
///
/// A backend may decode the formats it cares about with its own
/// instructions. The result must not differ from the portable fill.
pub type TileFill =
    unsafe fn(tile: &mut Tile, ty: TensorType, height: usize, data: &[u8], row_bytes: usize);

/// How to reduce one row of a weight matrix.
#[derive(Clone, Copy)]
pub enum RowDot {
    /// The row holds floats. Activations are used as they are.
    Float(DotFloat),
    /// The row holds quantized blocks. Activations are quantized first.
    Quant(DotQuant),
}

/// A set of kernel implementations for one instruction set.
pub struct Kernels {
    /// Short name of the backend, shown by `smallm info` and the benchmark.
    pub name: &'static str,
    dot: unsafe fn(a: *const f32, b: *const f32, n: usize) -> f32,
    axpy: unsafe fn(alpha: f32, x: *const f32, y: *mut f32, n: usize),
    sum_squares: unsafe fn(x: *const f32, n: usize) -> f32,
    dot_rows: RowsFn,
    mix_rows: RowsFn,
    dot_rows_half: RowsHalfFn,
    mix_rows_half: RowsHalfFn,
    to_half: unsafe fn(x: *const f32, out: *mut u16, n: usize),
    silu_mul: unsafe fn(gate: *mut f32, up: *const f32, n: usize),
    gelu_mul: unsafe fn(gate: *mut f32, up: *const f32, n: usize),
    softmax: unsafe fn(x: *mut f32, n: usize),
    quantize: QuantizeFn,
    row_f32: DotFloat,
    row_f16: DotFloat,
    row_bf16: DotFloat,
    row_quant: fn(TensorType) -> DotQuant,
    /// Rows this backend multiplies at once in the batched route. Zero when
    /// it has no kernel worth using for that.
    tile_rows: usize,
    /// True when the tile kernels want signed tiles, from
    /// [`Tile::fill_signed`].
    signed_tiles: bool,
    tile_dot: TileDot,
    /// The kernel for four vectors at once, when the backend has one.
    tile_dot4: Option<TileDot4>,
    tile_fill: TileFill,
}

/// The shape shared by the two attention kernels: `n_queries` vectors on one
/// side, `n` rows on the other, all `len` floats long. `stride` is the
/// distance between the per-query runs of the scalar side.
type RowsFn = unsafe fn(
    per_query: *const f32,
    n_queries: usize,
    rows: *const f32,
    n: usize,
    len: usize,
    out: *mut f32,
    stride: usize,
);

/// [`RowsFn`] with rows of 16-bit floats, as a half-precision key/value
/// cache holds them.
type RowsHalfFn = unsafe fn(
    per_query: *const f32,
    n_queries: usize,
    rows: *const u16,
    n: usize,
    len: usize,
    out: *mut f32,
    stride: usize,
);

/// Rows for the attention kernels, in 32-bit floats or in IEEE 16-bit
/// floats stored as their bits.
#[derive(Clone, Copy, Debug)]
pub enum Rows<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
}

impl Rows<'_> {
    /// Elements in all the rows together.
    pub fn len(&self) -> usize {
        match self {
            Rows::F32(rows) => rows.len(),
            Rows::F16(rows) => rows.len(),
        }
    }

    /// True when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Quantizes `n` floats to 8-bit integers in groups of `group` elements.
///
/// Writes `n` values to `q`, `n / group` scales to `scale` and `wsum`, and
/// `n / 16` partial sums to `sums`.
type QuantizeFn = unsafe fn(
    x: *const f32,
    n: usize,
    group: usize,
    q: *mut i8,
    scale: *mut f32,
    sums: *mut i16,
    wsum: *mut f32,
);

impl Kernels {
    /// True for the [`exact`] kernels, which read unrounded activations.
    pub fn is_exact(&self) -> bool {
        std::ptr::eq(self, exact())
    }

    /// Dot product of two float vectors of the same length.
    #[inline]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        // SAFETY: both pointers are valid for `n` reads.
        unsafe { (self.dot)(a.as_ptr(), b.as_ptr(), n) }
    }

    /// `y += alpha * x`.
    #[inline]
    pub fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        let n = x.len().min(y.len());
        // SAFETY: `x` is valid for `n` reads and `y` for `n` writes.
        unsafe { (self.axpy)(alpha, x.as_ptr(), y.as_mut_ptr(), n) }
    }

    /// Sum of the squares of `x`.
    #[inline]
    pub fn sum_squares(&self, x: &[f32]) -> f32 {
        // SAFETY: the pointer is valid for `x.len()` reads.
        unsafe { (self.sum_squares)(x.as_ptr(), x.len()) }
    }

    /// Scores every row against every query:
    /// `scores[h * stride + j] = dot(queries[h], rows[j])`.
    ///
    /// `queries` holds `n_queries` vectors of `len` floats, `rows` holds
    /// whole rows of `len` floats, and `scores` holds `n_queries` runs
    /// `stride` apart with room for one score per row.
    #[inline]
    pub fn dot_rows(
        &self,
        queries: &[f32],
        rows: &[f32],
        len: usize,
        scores: &mut [f32],
        stride: usize,
    ) {
        let (n_queries, n) = (queries.len() / len, rows.len() / len);
        if n_queries == 0 || n == 0 {
            return;
        }
        assert!(stride >= n && scores.len() >= (n_queries - 1) * stride + n);
        // SAFETY: the assert above bounds every score index, and the other
        // two slices hold whole vectors.
        unsafe {
            (self.dot_rows)(
                queries.as_ptr(),
                n_queries,
                rows.as_ptr(),
                n,
                len,
                scores.as_mut_ptr(),
                stride,
            )
        }
    }

    /// Adds a weighted sum of the rows to each output:
    /// `outs[h] += sum over j of weights[h * stride + j] * rows[j]`.
    ///
    /// The shapes mirror [`dot_rows`](Self::dot_rows), with `outs` in the
    /// place of `queries` and `weights` in the place of `scores`.
    #[inline]
    pub fn mix_rows(
        &self,
        weights: &[f32],
        rows: &[f32],
        len: usize,
        outs: &mut [f32],
        stride: usize,
    ) {
        let (n_queries, n) = (outs.len() / len, rows.len() / len);
        if n_queries == 0 || n == 0 {
            return;
        }
        assert!(stride >= n && weights.len() >= (n_queries - 1) * stride + n);
        // SAFETY: as in `dot_rows`.
        unsafe {
            (self.mix_rows)(
                weights.as_ptr(),
                n_queries,
                rows.as_ptr(),
                n,
                len,
                outs.as_mut_ptr(),
                stride,
            )
        }
    }

    /// [`dot_rows`](Self::dot_rows) on rows of either precision.
    #[inline]
    pub fn dot_rows_any(
        &self,
        queries: &[f32],
        rows: Rows<'_>,
        len: usize,
        scores: &mut [f32],
        stride: usize,
    ) {
        let rows = match rows {
            Rows::F32(rows) => return self.dot_rows(queries, rows, len, scores, stride),
            Rows::F16(rows) => rows,
        };
        let (n_queries, n) = (queries.len() / len, rows.len() / len);
        if n_queries == 0 || n == 0 {
            return;
        }
        assert!(stride >= n && scores.len() >= (n_queries - 1) * stride + n);
        // SAFETY: as in `dot_rows`.
        unsafe {
            (self.dot_rows_half)(
                queries.as_ptr(),
                n_queries,
                rows.as_ptr(),
                n,
                len,
                scores.as_mut_ptr(),
                stride,
            )
        }
    }

    /// [`mix_rows`](Self::mix_rows) on rows of either precision.
    #[inline]
    pub fn mix_rows_any(
        &self,
        weights: &[f32],
        rows: Rows<'_>,
        len: usize,
        outs: &mut [f32],
        stride: usize,
    ) {
        let rows = match rows {
            Rows::F32(rows) => return self.mix_rows(weights, rows, len, outs, stride),
            Rows::F16(rows) => rows,
        };
        let (n_queries, n) = (outs.len() / len, rows.len() / len);
        if n_queries == 0 || n == 0 {
            return;
        }
        assert!(stride >= n && weights.len() >= (n_queries - 1) * stride + n);
        // SAFETY: as in `dot_rows`.
        unsafe {
            (self.mix_rows_half)(
                weights.as_ptr(),
                n_queries,
                rows.as_ptr(),
                n,
                len,
                outs.as_mut_ptr(),
                stride,
            )
        }
    }

    /// Rounds `x` to 16-bit floats in `out`, to nearest with ties to even.
    #[inline]
    pub fn to_half(&self, x: &[f32], out: &mut [u16]) {
        let n = x.len().min(out.len());
        // SAFETY: `x` is valid for `n` reads and `out` for `n` writes.
        unsafe { (self.to_half)(x.as_ptr(), out.as_mut_ptr(), n) }
    }

    /// `gate[i] = silu(gate[i]) * up[i]`, the SwiGLU gate.
    #[inline]
    pub fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        let n = gate.len().min(up.len());
        // SAFETY: `gate` is valid for `n` writes and `up` for `n` reads.
        unsafe { (self.silu_mul)(gate.as_mut_ptr(), up.as_ptr(), n) }
    }

    /// `gate[i] = gelu(gate[i]) * up[i]`, the GeGLU gate.
    #[inline]
    pub fn gelu_mul(&self, gate: &mut [f32], up: &[f32]) {
        let n = gate.len().min(up.len());
        // SAFETY: `gate` is valid for `n` writes and `up` for `n` reads.
        unsafe { (self.gelu_mul)(gate.as_mut_ptr(), up.as_ptr(), n) }
    }

    /// Replaces `x` with its softmax.
    #[inline]
    pub fn softmax(&self, x: &mut [f32]) {
        // SAFETY: the pointer is valid for `x.len()` reads and writes.
        unsafe { (self.softmax)(x.as_mut_ptr(), x.len()) }
    }

    /// Quantizes `x` in groups of `group` elements. See [`ActRef`].
    ///
    /// `x.len()` must be a multiple of `group`, and `group` of 16.
    #[inline]
    pub fn quantize(
        &self,
        x: &[f32],
        group: usize,
        q: &mut [i8],
        scale: &mut [f32],
        sums: &mut [i16],
        wsum: &mut [f32],
    ) {
        let n = x.len();
        assert!(
            group % 16 == 0 && n % group == 0,
            "activation length must divide into groups"
        );
        assert!(
            q.len() >= n
                && scale.len() >= n / group
                && sums.len() >= n / 16
                && wsum.len() >= n / group
        );
        // SAFETY: the asserts above guarantee every buffer is large enough.
        unsafe {
            (self.quantize)(
                x.as_ptr(),
                n,
                group,
                q.as_mut_ptr(),
                scale.as_mut_ptr(),
                sums.as_mut_ptr(),
                wsum.as_mut_ptr(),
            )
        }
    }

    /// How many rows a tile for this backend holds. Zero when the backend
    /// should not be given tiles.
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }

    /// True when this backend's tiles are signed, filled as
    /// [`Tile::fill_signed`] fills them. Otherwise as [`Tile::fill`] does.
    pub fn signed_tiles(&self) -> bool {
        self.signed_tiles
    }

    /// The kernel for tiles of decoded rows.
    pub fn tile_dot(&self) -> TileDot {
        self.tile_dot
    }

    /// The kernel for a tile and four vectors at once, when this backend
    /// has one.
    pub fn tile_dot4(&self) -> Option<TileDot4> {
        self.tile_dot4
    }

    /// Fills `tile` with the rows stored in `data`, `row_bytes` apart, for
    /// this backend's tile kernel. `data` holds at most
    /// [`tile_rows`](Self::tile_rows) whole rows of type `ty`.
    pub fn fill_tile(&self, tile: &mut Tile, ty: TensorType, data: &[u8], row_bytes: usize) {
        assert!(self.tile_rows > 0 && ty.row_bytes(1).is_none());
        assert!(data.len() % row_bytes == 0 && data.len() / row_bytes <= self.tile_rows);
        assert!(row_bytes % ty.block_bytes() == 0);
        // SAFETY: the asserts above are the contract of the fill: whole
        // rows of whole blocks, and no more of them than a tile holds.
        unsafe { (self.tile_fill)(tile, ty, self.tile_rows, data, row_bytes) }
    }

    /// The row kernel for a tensor type.
    pub fn row_dot(&self, ty: TensorType) -> RowDot {
        match ty {
            TensorType::F32 => RowDot::Float(self.row_f32),
            TensorType::F16 => RowDot::Float(self.row_f16),
            TensorType::BF16 => RowDot::Float(self.row_bf16),
            _ => RowDot::Quant((self.row_quant)(ty)),
        }
    }
}

/// Asks the memory system to start loading `bytes` into the second-level
/// cache, one request per cache line, so that a later read finds them
/// there. A hint: it changes no result, and does nothing on a machine
/// without kernels here.
///
/// The batched multiply reads the rows of each tile once, four rows at a
/// time. Asking for the next tile while the current one is in use hides the
/// wait for memory, which the hardware prefetcher, following one stream
/// best, does not.
pub fn prefetch(bytes: &[u8]) {
    for line in bytes.chunks(64) {
        let address = line.as_ptr();
        // SAFETY: a prefetch reads nothing the program can see and cannot
        // fault. The address is inside `bytes` anyway.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!(
                "prfm pldl2keep, [{address}]",
                address = in(reg) address,
                options(nostack, preserves_flags, readonly)
            );
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T1, _mm_prefetch};
            _mm_prefetch::<_MM_HINT_T1>(address as *const i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = address;
    }
}

/// The portable kernels. Always available.
pub fn scalar() -> &'static Kernels {
    &scalar::KERNELS
}

/// Reference kernels that multiply without rounding the activations.
///
/// They are many times slower than the others. Use them to measure what the
/// 8-bit activation rounding of the fast kernels costs in accuracy.
pub fn exact() -> &'static Kernels {
    &scalar::EXACT
}

/// The fastest kernels this CPU supports.
///
/// The environment variable `SMALLM_KERNELS` overrides the choice with
/// `scalar` or `exact`, which is useful when comparing results.
pub fn best() -> &'static Kernels {
    static BEST: OnceLock<&'static Kernels> = OnceLock::new();
    BEST.get_or_init(|| match std::env::var("SMALLM_KERNELS").as_deref() {
        Ok("scalar") => scalar(),
        Ok("exact") => exact(),
        _ => detect(),
    })
}

/// Every backend this CPU can run, the fastest last. Used by the tests to
/// check each one against the scalar reference.
pub fn available() -> Vec<&'static Kernels> {
    // Targets without a SIMD backend add nothing to the list.
    #[cfg_attr(
        not(any(target_arch = "aarch64", target_arch = "x86_64")),
        allow(unused_mut)
    )]
    let mut list = vec![scalar()];
    #[cfg(target_arch = "aarch64")]
    {
        list.push(&arm::KERNELS);
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            list.push(&arm::KERNELS_DOTPROD);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if x86::supported() {
            list.push(&x86::KERNELS);
            if x86::has_vnni() {
                list.push(&x86::KERNELS_VNNI);
            }
        }
    }
    list
}

/// Picks the best backend for the running CPU.
fn detect() -> &'static Kernels {
    available().last().copied().unwrap_or_else(scalar)
}

#[cfg(test)]
mod tests;

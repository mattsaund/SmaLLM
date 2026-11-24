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
    silu_mul: unsafe fn(gate: *mut f32, up: *const f32, n: usize),
    gelu_mul: unsafe fn(gate: *mut f32, up: *const f32, n: usize),
    softmax: unsafe fn(x: *mut f32, n: usize),
    quantize: QuantizeFn,
    row_f32: DotFloat,
    row_f16: DotFloat,
    row_bf16: DotFloat,
    row_quant: fn(TensorType) -> DotQuant,
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

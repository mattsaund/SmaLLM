//! Multiplies weight matrices by batches of activation vectors.
//!
//! This is where the engine spends its time. One call handles every matrix
//! that reads the same input, for example the query, key and value
//! projections of a layer. Doing them together has two benefits: the input
//! is quantized once, and the thread pool is woken once.
//!
//! Rows are the unit of parallel work. For each row, the dot product with
//! every vector of the batch is taken before moving on, so a weight row is
//! read from memory once per batch.

use std::time::{Duration, Instant};

use crate::gguf::Mapping;
use crate::kernels::{ActRef, Kernels, RowDot};
use crate::tensor::Family;
use crate::threads::ThreadPool;

use super::weights::Linear;

/// The most matrices one call can multiply.
pub(super) const MAX_TARGETS: usize = 3;

/// A chunk of rows should cost at least this many multiply-adds, or handing
/// it to a thread costs more than computing it.
const MIN_CHUNK_WORK: usize = 16 * 1024;

/// A whole call below this many multiply-adds runs on the calling thread.
const INLINE_WORK: usize = 128 * 1024;

/// What a multiplication needs besides its operands.
pub(super) struct Engine<'a> {
    pub kernels: &'static Kernels,
    pub pool: &'a ThreadPool,
    pub map: &'a Mapping,
}

/// One output of a multiplication: a linear map and where its result goes.
pub(super) struct Target<'a> {
    pub linear: &'a Linear,
    /// One output row per input vector, each `linear.rows` wide.
    pub out: &'a mut [f32],
}

/// Activation vectors quantized for one block family.
#[derive(Default)]
struct QuantSet {
    q: Vec<i8>,
    scale: Vec<f32>,
    sums: Vec<i16>,
    wsum: Vec<f32>,
}

impl QuantSet {
    /// Quantizes every vector of `x` in groups of `group` elements.
    fn fill(&mut self, kernels: &Kernels, x: &[f32], cols: usize, group: usize) {
        let count = x.len() / cols;
        self.q.resize(count * cols, 0);
        self.scale.resize(count * cols / group, 0.0);
        self.sums.resize(count * cols / 16, 0);
        self.wsum.resize(count * cols / group, 0.0);
        let vectors = x.chunks_exact(cols);
        let q = self.q.chunks_exact_mut(cols);
        let scale = self.scale.chunks_exact_mut(cols / group);
        let sums = self.sums.chunks_exact_mut(cols / 16);
        let wsum = self.wsum.chunks_exact_mut(cols / group);
        for ((((vector, q), scale), sums), wsum) in vectors.zip(q).zip(scale).zip(sums).zip(wsum) {
            kernels.quantize(vector, group, q, scale, sums, wsum);
        }
    }

    /// The quantized form of vector `index`.
    fn act(&self, index: usize, cols: usize, group: usize) -> ActRef {
        // SAFETY: `fill` sized every buffer for `index < count`.
        unsafe {
            ActRef {
                q: self.q.as_ptr().add(index * cols),
                scale: self.scale.as_ptr().add(index * cols / group),
                sums: self.sums.as_ptr().add(index * cols / 16),
                wsum: self.wsum.as_ptr().add(index * cols / group),
            }
        }
    }
}

/// Reusable buffers for quantized activations: one set per block family.
#[derive(Default)]
pub(super) struct ActScratch {
    block32: QuantSet,
    block256: QuantSet,
    /// Time spent quantizing, gathered while `timing` is on.
    timing: bool,
    quantize_time: Duration,
}

impl ActScratch {
    /// Starts or stops timing quantization. Starting resets the total.
    pub fn set_timing(&mut self, on: bool) {
        self.timing = on;
        self.quantize_time = Duration::ZERO;
    }

    /// Time spent quantizing since timing was turned on.
    pub fn quantize_time(&self) -> Duration {
        self.quantize_time
    }
}

/// One target, resolved to what the row loop needs.
#[derive(Clone, Copy)]
struct Plan {
    dot: RowDot,
    family: Family,
    /// First row of this target in the combined row numbering.
    first: usize,
    rows: usize,
    /// The first stored row, and the distance between rows.
    data: *const u8,
    row_bytes: usize,
    out: *mut f32,
}

/// Everything the row loop shares between threads.
struct Job<'a> {
    plans: &'a [Option<Plan>],
    x: *const f32,
    cols: usize,
    count: usize,
    scratch: &'a ActScratch,
    /// True when the kernels read the float input directly.
    exact: bool,
}

// SAFETY: the job is read-only except for `Plan::out`, and threads write
// disjoint elements of it: each handles its own range of rows.
unsafe impl Sync for Job<'_> {}

impl Job<'_> {
    /// Computes the outputs of rows `range` in the combined numbering.
    fn run(&self, range: std::ops::Range<usize>) {
        for plan in self.plans.iter().flatten() {
            let start = range.start.max(plan.first);
            let end = range.end.min(plan.first + plan.rows);
            for global in start..end {
                let row = global - plan.first;
                // SAFETY: `row < plan.rows`, so the row pointer stays inside
                // the tensor and each output index inside `out`. The kernels
                // receive buffers of the sizes their contracts require.
                unsafe {
                    let weights = plan.data.add(row * plan.row_bytes);
                    for index in 0..self.count {
                        let value = match plan.dot {
                            RowDot::Float(dot) => {
                                dot(weights, self.x.add(index * self.cols), self.cols)
                            }
                            RowDot::Quant(dot) => {
                                let act = if self.exact {
                                    // The exact kernels read the float input
                                    // through the one pointer they use.
                                    let input = self.x.add(index * self.cols);
                                    ActRef {
                                        q: [].as_ptr(),
                                        scale: input,
                                        sums: [].as_ptr(),
                                        wsum: input,
                                    }
                                } else {
                                    match plan.family {
                                        Family::Block256 => {
                                            self.scratch.block256.act(index, self.cols, 256)
                                        }
                                        _ => self.scratch.block32.act(index, self.cols, 32),
                                    }
                                };
                                dot(weights, self.cols, act)
                            }
                        };
                        *plan.out.add(index * plan.rows + row) = value;
                    }
                }
            }
        }
    }
}

impl Engine<'_> {
    /// Applies every target's linear map to every vector of `x`.
    ///
    /// `x` holds vectors of `cols` elements back to back. Each target's
    /// `out` receives one row per vector.
    pub fn multiply(
        &self,
        scratch: &mut ActScratch,
        x: &[f32],
        cols: usize,
        targets: &mut [Target<'_>],
    ) {
        let count = x.len() / cols;
        if count == 0 || targets.is_empty() {
            return;
        }
        assert!(targets.len() <= MAX_TARGETS && x.len() == count * cols);

        // Quantize the input once for each block family in use.
        let exact = self.kernels.is_exact();
        let uses = |family| {
            !exact
                && targets
                    .iter()
                    .any(|target| target.linear.matrix.ty.family() == family)
        };
        let start = scratch.timing.then(Instant::now);
        if uses(Family::Block32) {
            scratch.block32.fill(self.kernels, x, cols, 32);
        }
        if uses(Family::Block256) {
            scratch.block256.fill(self.kernels, x, cols, 256);
        }
        if let Some(start) = start {
            scratch.quantize_time += start.elapsed();
        }

        let mut plans = [None; MAX_TARGETS];
        let mut total_rows = 0;
        for (plan, target) in plans.iter_mut().zip(targets.iter_mut()) {
            let matrix = &target.linear.matrix;
            assert_eq!(matrix.cols, cols, "matrix width must match the input");
            assert_eq!(
                target.out.len(),
                count * target.linear.rows,
                "output size must match"
            );
            assert!(target.linear.first_row + target.linear.rows <= matrix.rows);
            let offset = matrix.offset + target.linear.first_row * matrix.row_bytes;
            assert!(offset + target.linear.rows * matrix.row_bytes <= self.map.len());
            *plan = Some(Plan {
                dot: self.kernels.row_dot(matrix.ty),
                family: matrix.ty.family(),
                first: total_rows,
                rows: target.linear.rows,
                data: self.map.as_slice()[offset..].as_ptr(),
                row_bytes: matrix.row_bytes,
                out: target.out.as_mut_ptr(),
            });
            total_rows += target.linear.rows;
        }
        let job = Job {
            plans: &plans,
            x: x.as_ptr(),
            cols,
            count,
            scratch,
            exact,
        };
        let grain = rows_per_chunk(total_rows, cols * count, self.pool.threads());
        self.pool.ranges(total_rows, grain, &|range| job.run(range));

        for target in targets.iter_mut() {
            if let Some(bias) = &target.linear.bias {
                for row in target.out.chunks_exact_mut(target.linear.rows) {
                    for (value, bias) in row.iter_mut().zip(bias) {
                        *value += bias;
                    }
                }
            }
        }
    }
}

/// How many rows each parallel chunk gets. Returns all of them, which keeps
/// the work on the calling thread, when the job is too small to share.
fn rows_per_chunk(rows: usize, work_per_row: usize, threads: usize) -> usize {
    let work_per_row = work_per_row.max(1);
    if threads <= 1 || rows * work_per_row <= INLINE_WORK {
        return rows.max(1);
    }
    // Several chunks per thread, so a slow core does not hold up the rest.
    let balanced = rows.div_ceil(threads * 8);
    balanced.max(MIN_CHUNK_WORK.div_ceil(work_per_row)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_jobs_stay_on_one_thread() {
        assert_eq!(rows_per_chunk(64, 64, 8), 64);
        assert_eq!(rows_per_chunk(4096, 4096, 1), 4096);
        assert_eq!(rows_per_chunk(0, 10, 8), 1);
    }

    #[test]
    fn large_jobs_split_into_several_chunks_per_thread() {
        let grain = rows_per_chunk(4096, 4096, 8);
        assert_eq!(grain, 64);
        // Chunks never fall below the minimum amount of work.
        let grain = rows_per_chunk(2048, 512, 16);
        assert!(grain * 512 >= MIN_CHUNK_WORK);
    }
}

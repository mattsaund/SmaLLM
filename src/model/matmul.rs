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
//!
//! A quantized row can be reduced in two ways:
//!
//! - **Directly.** The kernel unpacks the row's blocks as it multiplies.
//!   This is the route for one vector, which is every generated token.
//! - **In tiles.** A few rows are unpacked once into small integers and
//!   interleaved, so that one SIMD multiply serves all of them, and each
//!   vector of the batch is reduced against the whole tile. With a batch of
//!   vectors, which is how a prompt is read, unpacking is paid once per row
//!   instead of once per vector, and several rows cost one multiply.

use std::cell::RefCell;
use std::ops::Range;
use std::slice;
use std::time::{Duration, Instant};

use crate::gguf::Mapping;
use crate::kernels::{ActRef, Kernels, RowDot};
use crate::tensor::quant::Tile;
use crate::tensor::{Family, TensorType};
use crate::threads::ThreadPool;

use super::weights::Linear;

/// The most matrices one call can multiply.
pub(super) const MAX_TARGETS: usize = 4;

/// A chunk of rows should cost at least this many multiply-adds, or handing
/// it to a thread costs more than computing it.
const MIN_CHUNK_WORK: usize = 16 * 1024;

/// A whole call below this many multiply-adds runs on the calling thread.
const INLINE_WORK: usize = 128 * 1024;

/// A batch with fewer elements than this is quantized on the calling
/// thread.
const INLINE_QUANTIZE: usize = 16 * 1024;

/// A batch of at least this many vectors takes the tile route. Below it,
/// unpacking a row inside each dot product costs less than decoding it up
/// front.
const TILE_BATCH: usize = 8;

/// The most rows any backend puts in a tile.
const MAX_TILE_ROWS: usize = 8;

thread_local! {
    /// The weight rows this thread is working on, decoded.
    static TILE: RefCell<Tile> = RefCell::new(Tile::default());
}

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
    /// `scale * sum(q)` for each run of 16 elements: what the offsets of a
    /// tile are multiplied by. Only filled for the tile route.
    scaled_sums: Vec<f32>,
}

/// The buffers of a [`QuantSet`] while a batch is quantized. Threads share
/// them, and each writes only the parts that belong to its own vectors.
struct QuantOut {
    q: *mut i8,
    scale: *mut f32,
    sums: *mut i16,
    wsum: *mut f32,
    scaled_sums: *mut f32,
}

// SAFETY: see the type's description.
unsafe impl Sync for QuantOut {}

/// The slices of a [`QuantOut`] that belong to one vector.
struct QuantSlots<'a> {
    q: &'a mut [i8],
    scale: &'a mut [f32],
    sums: &'a mut [i16],
    wsum: &'a mut [f32],
    scaled_sums: &'a mut [f32],
}

impl QuantOut {
    /// The parts of every buffer that belong to vector `index`. The scaled
    /// sums are empty unless `tiled` is set.
    ///
    /// # Safety
    ///
    /// The buffers must be sized for more than `index` vectors of `cols`
    /// elements in groups of `group`, and no other thread may be working on
    /// the same vector.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slots(&self, index: usize, cols: usize, group: usize, tiled: bool) -> QuantSlots<'_> {
        let (groups, runs) = (cols / group, cols / 16);
        let scaled = if tiled { runs } else { 0 };
        // SAFETY: guaranteed by the caller.
        unsafe {
            QuantSlots {
                q: slice::from_raw_parts_mut(self.q.add(index * cols), cols),
                scale: slice::from_raw_parts_mut(self.scale.add(index * groups), groups),
                sums: slice::from_raw_parts_mut(self.sums.add(index * runs), runs),
                wsum: slice::from_raw_parts_mut(self.wsum.add(index * groups), groups),
                scaled_sums: slice::from_raw_parts_mut(
                    self.scaled_sums.add(index * scaled),
                    scaled,
                ),
            }
        }
    }
}

impl QuantSet {
    /// Quantizes every vector of `x` in groups of `group` elements. A batch
    /// is split between the threads of `pool`.
    fn fill(
        &mut self,
        kernels: &Kernels,
        pool: &ThreadPool,
        x: &[f32],
        cols: usize,
        group: usize,
        tiled: bool,
    ) {
        let count = x.len() / cols;
        self.q.resize(count * cols, 0);
        self.scale.resize(count * cols / group, 0.0);
        self.sums.resize(count * cols / 16, 0);
        self.wsum.resize(count * cols / group, 0.0);
        if tiled {
            self.scaled_sums.resize(count * cols / 16, 0.0);
        }
        let out = QuantOut {
            q: self.q.as_mut_ptr(),
            scale: self.scale.as_mut_ptr(),
            sums: self.sums.as_mut_ptr(),
            wsum: self.wsum.as_mut_ptr(),
            scaled_sums: self.scaled_sums.as_mut_ptr(),
        };
        let quantize = |index: usize| {
            // SAFETY: every buffer was sized above for `count` vectors, and
            // each index is handed to one thread.
            let slots = unsafe { out.slots(index, cols, group, tiled) };
            let vector = &x[index * cols..(index + 1) * cols];
            kernels.quantize(vector, group, slots.q, slots.scale, slots.sums, slots.wsum);
            let scaled = slots.scaled_sums.chunks_exact_mut(group / 16);
            let sums = slots.sums.chunks_exact(group / 16);
            for ((scaled, sums), scale) in scaled.zip(sums).zip(slots.scale.iter()) {
                for (scaled, sum) in scaled.iter_mut().zip(sums) {
                    *scaled = scale * *sum as f32;
                }
            }
        };
        // A vector is cheap to quantize, so a thread takes several at once.
        let grain = (INLINE_QUANTIZE / 2 / cols.max(1)).max(1);
        if count * cols <= INLINE_QUANTIZE {
            (0..count).for_each(quantize);
        } else {
            pool.ranges(count, grain, &|range| range.for_each(&quantize));
        }
    }

    /// The scaled sums of vector `index`. See the field.
    fn scaled_sums(&self, index: usize, cols: usize) -> &[f32] {
        &self.scaled_sums[index * cols / 16..(index + 1) * cols / 16]
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
    ty: TensorType,
    family: Family,
    /// True when the rows take the tile route.
    tiled: bool,
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
    kernels: &'static Kernels,
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
    /// The quantized activations a target of `family` reads.
    fn acts(&self, family: Family) -> &QuantSet {
        match family {
            Family::Block256 => &self.scratch.block256,
            _ => &self.scratch.block32,
        }
    }

    /// Computes rows `rows` of one target by the tile route.
    fn run_tiled(&self, plan: &Plan, rows: Range<usize>) {
        let group = plan.ty.block_len();
        let height = self.kernels.tile_rows();
        let acts = self.acts(plan.family);
        let dot = self.kernels.tile_dot();
        TILE.with_borrow_mut(|tile| {
            let mut row = rows.start;
            while row < rows.end {
                let count = height.min(rows.end - row);
                // SAFETY: rows `row..row + count` are below `plan.rows`, so
                // they lie inside the tensor, which `multiply` checked
                // against the mapping.
                let bytes = unsafe {
                    slice::from_raw_parts(
                        plan.data.add(row * plan.row_bytes),
                        count * plan.row_bytes,
                    )
                };
                self.kernels.fill_tile(tile, plan.ty, bytes, plan.row_bytes);
                debug_assert_eq!(tile.cols, self.cols);
                let mut results = [0.0f32; MAX_TILE_ROWS];
                for index in 0..self.count {
                    let act = acts.act(index, self.cols, group);
                    let sums = acts.scaled_sums(index, self.cols).as_ptr();
                    // SAFETY: the tile and the activations both hold `cols`
                    // columns, `results` has room for a whole tile, and
                    // `index` and `row + r` bound the output index.
                    unsafe {
                        dot(tile, act, sums, group, results.as_mut_ptr());
                        for (r, value) in results[..count].iter().enumerate() {
                            *plan.out.add(index * plan.rows + row + r) = *value;
                        }
                    }
                }
                row += count;
            }
        });
    }

    /// Computes the outputs of rows `range` in the combined numbering.
    fn run(&self, range: Range<usize>) {
        for plan in self.plans.iter().flatten() {
            let start = range.start.max(plan.first);
            let end = range.end.min(plan.first + plan.rows);
            if start >= end {
                continue;
            }
            if plan.tiled {
                self.run_tiled(plan, start - plan.first..end - plan.first);
                continue;
            }
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
                                    let group = plan.ty.block_len();
                                    self.acts(plan.family).act(index, self.cols, group)
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
        let height = self.kernels.tile_rows();
        assert!(height <= MAX_TILE_ROWS);
        let tiled = !exact && height > 0 && count >= TILE_BATCH;
        let start = scratch.timing.then(Instant::now);
        if uses(Family::Block32) {
            scratch
                .block32
                .fill(self.kernels, self.pool, x, cols, 32, tiled);
        }
        if uses(Family::Block256) {
            scratch
                .block256
                .fill(self.kernels, self.pool, x, cols, 256, tiled);
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
                ty: matrix.ty,
                family: matrix.ty.family(),
                tiled: tiled && matrix.ty.family() != Family::Float,
                first: total_rows,
                rows: target.linear.rows,
                data: self.map.as_slice()[offset..].as_ptr(),
                row_bytes: matrix.row_bytes,
                out: target.out.as_mut_ptr(),
            });
            total_rows += target.linear.rows;
        }
        let job = Job {
            kernels: self.kernels,
            plans: &plans,
            x: x.as_ptr(),
            cols,
            count,
            scratch,
            exact,
        };
        let mut grain = rows_per_chunk(total_rows, cols * count, self.pool.threads());
        if tiled {
            // Whole tiles per chunk, so no thread pads a tile with empty rows.
            grain = grain.next_multiple_of(height);
        }
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

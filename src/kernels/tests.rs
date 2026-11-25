//! Checks the scalar kernels against the plain definition of each operation,
//! and every other backend against the scalar kernels.

use super::*;
use crate::tensor::f16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use crate::tensor::quant::{Tile, dequantize_row};
use crate::testing::{Rng, random_weights, set_scales};

const QUANT_TYPES: [TensorType; 25] = [
    TensorType::Q4_0,
    TensorType::Q4_1,
    TensorType::Q5_0,
    TensorType::Q5_1,
    TensorType::Q8_0,
    TensorType::Iq4Nl,
    TensorType::Q2K,
    TensorType::Q3K,
    TensorType::Q4K,
    TensorType::Q5K,
    TensorType::Q6K,
    TensorType::Iq4Xs,
    TensorType::Iq3S,
    TensorType::Iq3Xxs,
    TensorType::Iq2S,
    TensorType::Iq2Xs,
    TensorType::Iq2Xxs,
    TensorType::Iq1M,
    TensorType::Iq1S,
    TensorType::Tq2_0,
    TensorType::Tq1_0,
    TensorType::Mxfp4,
    TensorType::Nvfp4,
    TensorType::Q1_0,
    TensorType::Q2_0,
];

/// An activation vector quantized by one backend.
struct Acts {
    q: Vec<i8>,
    scale: Vec<f32>,
    sums: Vec<i16>,
    wsum: Vec<f32>,
    group: usize,
}

impl Acts {
    fn new(kernels: &Kernels, x: &[f32], group: usize) -> Acts {
        let n = x.len();
        let mut acts = Acts {
            q: vec![0; n],
            scale: vec![0.0; n / group],
            sums: vec![0; n / 16],
            wsum: vec![0.0; n / group],
            group,
        };
        kernels.quantize(
            x,
            group,
            &mut acts.q,
            &mut acts.scale,
            &mut acts.sums,
            &mut acts.wsum,
        );
        acts
    }

    fn as_ref(&self) -> ActRef {
        ActRef {
            q: self.q.as_ptr(),
            scale: self.scale.as_ptr(),
            sums: self.sums.as_ptr(),
            wsum: self.wsum.as_ptr(),
        }
    }

    /// The floats the quantized values stand for.
    fn reconstruct(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &q)| self.scale[i / self.group] * q as f32)
            .collect()
    }
}

/// The group size activations are quantized in for `ty`. It is the block
/// length of most formats, but not of the ones with blocks of 64 or 128.
fn group_of(ty: TensorType) -> usize {
    ty.family().group()
}

/// Runs a backend's row kernel for a quantized type.
fn row_dot(kernels: &Kernels, ty: TensorType, row: &[u8], n: usize, acts: &Acts) -> f32 {
    match kernels.row_dot(ty) {
        // SAFETY: `row` holds `n` weights and `acts` holds `n` activations.
        RowDot::Quant(dot) => unsafe { dot(row.as_ptr(), n, acts.as_ref()) },
        RowDot::Float(_) => panic!("{} is not a quantized type", ty.name()),
    }
}

/// The scaled sums of an activation vector: what tile offsets multiply.
fn scaled_sums(acts: &Acts) -> Vec<f32> {
    acts.sums
        .iter()
        .enumerate()
        .map(|(sub, sum)| acts.scale[sub * 16 / acts.group] * *sum as f32)
        .collect()
}

/// The tile heights to test a backend with. The portable kernel is the
/// definition for every height, so it is tested with both heights in use.
fn tile_heights(kernels: &Kernels) -> Vec<usize> {
    match kernels.tile_rows() {
        0 => vec![4, 8],
        height => vec![height],
    }
}

/// Reduces the rows stored in `matrix` against `acts` with a backend's tile
/// kernel, the way the batched multiply does. Returns one result per row.
fn tile_dots(
    kernels: &Kernels,
    height: usize,
    ty: TensorType,
    matrix: &[u8],
    n: usize,
    acts: &Acts,
) -> Vec<f32> {
    let row_bytes = ty.row_bytes(n).unwrap();
    let mut tile = Tile::default();
    let mut results = Vec::new();
    for rows in matrix.chunks(height * row_bytes) {
        // The portable kernel is tested on tiles from the portable fill.
        match kernels.tile_rows() {
            0 => tile.fill(ty, height, rows, row_bytes),
            _ => kernels.fill_tile(&mut tile, ty, rows, row_bytes),
        }
        assert_eq!((tile.cols, tile.height), (n, height));
        let mut out = vec![f32::NAN; height];
        // SAFETY: the tile and `acts` both describe `n` columns, the scaled
        // sums cover them, and `out` has one slot per tile row.
        unsafe {
            (kernels.tile_dot())(
                &tile,
                acts.as_ref(),
                scaled_sums(acts).as_ptr(),
                acts.group,
                out.as_mut_ptr(),
            )
        };
        // Rows the tile was padded with come out as zero.
        assert!(
            out[rows.len() / row_bytes..]
                .iter()
                .all(|value| *value == 0.0)
        );
        results.extend_from_slice(&out[..rows.len() / row_bytes]);
    }
    results
}

/// The dot product by definition: expand both sides to floats and multiply.
/// Returns the result and the sum of the magnitudes of its terms.
fn reference_dot(ty: TensorType, row: &[u8], x: &[f32]) -> (f64, f64) {
    let mut weights = vec![0.0f32; x.len()];
    dequantize_row(ty, row, &mut weights);
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for (w, v) in weights.iter().zip(x) {
        sum += *w as f64 * *v as f64;
        magnitude += (*w as f64 * *v as f64).abs();
    }
    (sum, magnitude)
}

fn assert_close(got: f32, expected: f64, magnitude: f64, what: &str) {
    let tolerance = 2e-5 * magnitude + 1e-6;
    assert!(
        (got as f64 - expected).abs() <= tolerance,
        "{what}: got {got}, expected {expected}, tolerance {tolerance}"
    );
}

#[test]
fn a_backend_is_selected() {
    let all = available();
    assert_eq!(all[0].name, "scalar");
    assert!(all.iter().any(|k| k.name == best().name));
}

#[test]
fn exp_is_accurate_across_its_range() {
    let mut worst = 0.0f64;
    let mut x = -87.0f32;
    while x < 88.0 {
        let got = scalar::exp_approx(x) as f64;
        let expected = (x as f64).exp();
        worst = worst.max(((got - expected) / expected).abs());
        x += 0.0137;
    }
    assert!(worst < 4e-7, "worst relative error {worst}");
    // Out of range inputs clamp instead of overflowing.
    assert!(scalar::exp_approx(1000.0).is_finite());
    assert!(scalar::exp_approx(-1000.0) > 0.0);
}

#[test]
fn float_kernels_match_the_definition() {
    let mut rng = Rng(11);
    for kernels in available() {
        for n in 0..70 {
            let a = rng.floats(n, -2.0, 2.0);
            let b = rng.floats(n, -2.0, 2.0);
            let expected: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            assert_close(kernels.dot(&a, &b), expected, n as f64, kernels.name);

            let squares: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum();
            assert_close(kernels.sum_squares(&a), squares, n as f64, kernels.name);

            let mut y = b.clone();
            kernels.axpy(0.75, &a, &mut y);
            for i in 0..n {
                assert_close(y[i], b[i] as f64 + 0.75 * a[i] as f64, 1.0, kernels.name);
            }
        }
    }
}

#[test]
fn attention_kernels_match_the_definition() {
    let mut rng = Rng(21);
    for kernels in available() {
        // Lengths with and without a tail, and every query group size.
        for len in [1, 7, 8, 9, 24, 32, 40, 64, 71] {
            for n_queries in 1..=6 {
                for n in [0, 1, 2, 5, 33] {
                    let queries = rng.floats(n_queries * len, -2.0, 2.0);
                    let rows = rng.floats(n * len, -2.0, 2.0);
                    let stride = n + 3;
                    let what = format!("{} len {len} queries {n_queries} rows {n}", kernels.name);

                    // Scores land at their slots and nowhere else.
                    let mut scores = vec![-7.0f32; n_queries * stride];
                    kernels.dot_rows(&queries, &rows, len, &mut scores, stride);
                    for h in 0..n_queries {
                        for j in 0..stride {
                            let got = scores[h * stride + j];
                            if j >= n {
                                assert_eq!(got, -7.0, "{what}: wrote past the rows");
                                continue;
                            }
                            let expected: f64 = (0..len)
                                .map(|i| queries[h * len + i] as f64 * rows[j * len + i] as f64)
                                .sum();
                            assert_close(got, expected, len as f64, &what);
                        }
                    }

                    let weights = rng.floats(n_queries * stride, -1.0, 1.0);
                    let start = rng.floats(n_queries * len, -1.0, 1.0);
                    let mut outs = start.clone();
                    kernels.mix_rows(&weights, &rows, len, &mut outs, stride);
                    for h in 0..n_queries {
                        for i in 0..len {
                            let mixed: f64 = (0..n)
                                .map(|j| weights[h * stride + j] as f64 * rows[j * len + i] as f64)
                                .sum();
                            assert_close(
                                outs[h * len + i],
                                start[h * len + i] as f64 + mixed,
                                n as f64 + 1.0,
                                &what,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn gates_match_the_definition() {
    let mut rng = Rng(12);
    for kernels in available() {
        for n in [0, 1, 7, 8, 9, 33, 100] {
            let gate = rng.floats(n, -9.0, 9.0);
            let up = rng.floats(n, -2.0, 2.0);

            let mut silu = gate.clone();
            kernels.silu_mul(&mut silu, &up);
            let mut gelu = gate.clone();
            kernels.gelu_mul(&mut gelu, &up);
            for i in 0..n {
                let x = gate[i] as f64;
                let expected_silu = x / (1.0 + (-x).exp()) * up[i] as f64;
                assert_close(silu[i], expected_silu, 1.0, kernels.name);
                let z = (2.0 / std::f64::consts::PI).sqrt() * x * (1.0 + 0.044715 * x * x);
                let expected_gelu = 0.5 * x * (1.0 + z.tanh()) * up[i] as f64;
                assert_close(gelu[i], expected_gelu, 1.0, kernels.name);
            }
        }
    }
}

#[test]
fn softmax_matches_the_definition() {
    let mut rng = Rng(13);
    for kernels in available() {
        for n in [1, 2, 7, 8, 9, 31, 200] {
            let logits = rng.floats(n, -12.0, 12.0);
            let mut out = logits.clone();
            kernels.softmax(&mut out);
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let total: f64 = logits.iter().map(|v| (*v as f64 - max).exp()).sum();
            let mut sum = 0.0f64;
            for i in 0..n {
                let expected = (logits[i] as f64 - max).exp() / total;
                assert_close(out[i], expected, 1.0, kernels.name);
                sum += out[i] as f64;
            }
            assert!((sum - 1.0).abs() < 1e-5, "{}: sums to {sum}", kernels.name);
        }
        kernels.softmax(&mut []);
    }
}

#[test]
fn quantized_activations_reconstruct_their_input() {
    let mut rng = Rng(14);
    for group in [32, 256] {
        let mut x = rng.floats(group * 4, -3.0, 3.0);
        // One group of zeros, and one dominated by a single large value.
        x[group..group * 2].fill(0.0);
        x[group * 2 + 5] = 400.0;
        // Exact halves, which must round to the even neighbor.
        x[group * 3] = 3.0;
        x[group * 3 + 1] = 3.0 * 0.5 / 127.0;
        x[group * 3 + 2] = -3.0 * 2.5 / 127.0;
        let reference = Acts::new(scalar(), &x, group);

        for g in 0..4 {
            let q = &reference.q[g * group..(g + 1) * group];
            let largest = q.iter().map(|v| v.unsigned_abs()).max().unwrap();
            assert_eq!(largest, if g == 1 { 0 } else { 127 });
            let total: i32 = q.iter().map(|&v| v as i32).sum();
            assert!((reference.wsum[g] - reference.scale[g] * total as f32).abs() < 1e-4);
        }
        for (i, restored) in reference.reconstruct().iter().enumerate() {
            let step = reference.scale[i / group];
            assert!((restored - x[i]).abs() <= step * 0.5001, "element {i}");
        }
        for (i, sum) in reference.sums.iter().enumerate() {
            let expected: i32 = reference.q[i * 16..(i + 1) * 16]
                .iter()
                .map(|&v| v as i32)
                .sum();
            assert_eq!(*sum as i32, expected);
        }

        for kernels in available() {
            let acts = Acts::new(kernels, &x, group);
            assert_eq!(acts.q, reference.q, "{}", kernels.name);
            assert_eq!(acts.sums, reference.sums, "{}", kernels.name);
            assert_eq!(acts.scale, reference.scale, "{}", kernels.name);
            assert_eq!(acts.wsum, reference.wsum, "{}", kernels.name);
        }

        // Values that are not numbers, and infinities, quantize the same
        // way on every backend.
        let mut odd = rng.floats(group * 3, -3.0, 3.0);
        odd[3] = f32::NAN;
        odd[group + 7] = f32::INFINITY;
        odd[group + 9] = -1.0;
        odd[group * 2 + 1] = f32::NEG_INFINITY;
        odd[group * 2 + 2] = f32::NAN;
        let reference = Acts::new(scalar(), &odd, group);
        for kernels in available() {
            let acts = Acts::new(kernels, &odd, group);
            assert_eq!(acts.q, reference.q, "{}", kernels.name);
            assert_eq!(acts.sums, reference.sums, "{}", kernels.name);
            assert_eq!(acts.scale, reference.scale, "{}", kernels.name);
        }
    }
}

#[test]
fn scalar_rows_match_the_definition() {
    let mut rng = Rng(15);
    for ty in QUANT_TYPES {
        let n = 512;
        let row = random_weights(&mut rng, ty, n, 3.2);
        let x = rng.floats(n, -3.0, 3.0);
        let acts = Acts::new(scalar(), &x, group_of(ty));
        // The kernel sees the quantized activations, so compare against the
        // floats those stand for.
        let (expected, magnitude) = reference_dot(ty, &row, &acts.reconstruct());
        assert_close(
            row_dot(scalar(), ty, &row, n, &acts),
            expected,
            magnitude,
            ty.name(),
        );
    }
}

#[test]
fn every_backend_matches_scalar_rows() {
    let mut rng = Rng(16);
    for kernels in available() {
        for ty in QUANT_TYPES {
            // One, two, three and five blocks: odd counts exercise the tail
            // of kernels that work on two blocks at a time.
            for blocks in [1, 2, 3, 5] {
                let n = blocks * ty.block_len();
                let row = random_weights(&mut rng, ty, n, 3.2);
                let x = rng.floats(n, -3.0, 3.0);
                let acts = Acts::new(scalar(), &x, group_of(ty));
                let (expected, magnitude) = reference_dot(ty, &row, &acts.reconstruct());
                let got = row_dot(kernels, ty, &row, n, &acts);
                assert_close(
                    got,
                    expected,
                    magnitude,
                    &format!("{} {} n={n}", kernels.name, ty.name()),
                );
            }
        }
    }
}

#[test]
fn extreme_weights_and_activations_do_not_saturate() {
    let mut rng = Rng(17);
    for kernels in available() {
        for ty in QUANT_TYPES {
            let n = 2 * ty.block_len().max(32);
            for fill in [0x00u8, 0x7f, 0x80, 0xff] {
                let mut row = vec![fill; ty.row_bytes(n).unwrap()];
                set_scales(&mut rng, ty, &mut row, 0.05);
                // Constant activations quantize to +127 or -127 everywhere.
                for value in [2.5f32, -2.5] {
                    let x = vec![value; n];
                    let acts = Acts::new(scalar(), &x, group_of(ty));
                    assert!(acts.q.iter().all(|q| q.unsigned_abs() == 127));
                    let (expected, magnitude) = reference_dot(ty, &row, &acts.reconstruct());
                    let what = format!("{} {} fill {fill:#04x} x {value}", kernels.name, ty.name());
                    assert_close(
                        row_dot(kernels, ty, &row, n, &acts),
                        expected,
                        magnitude,
                        &what,
                    );
                    for height in tile_heights(kernels) {
                        let got = tile_dots(kernels, height, ty, &row, n, &acts);
                        assert_close(got[0], expected, magnitude, &format!("tile {what}"));
                    }
                }
            }
        }
    }
}

#[test]
fn every_nvfp4_scale_reads_the_same_on_every_backend() {
    // The random scales of the other tests are ordinary ones. Here 64 rows
    // of one block each hold all 256 scale bytes between them, among them
    // the subnormal ones and those that ggml reads in its own way. Each
    // row is checked on its own, so that a tiny scale read wrongly is not
    // lost next to a large one.
    let ty = TensorType::Nvfp4;
    let n = ty.block_len();
    let mut rng = Rng(24);
    let mut matrix = rng.bytes(64 * ty.block_bytes());
    for (b, block) in matrix.chunks_exact_mut(ty.block_bytes()).enumerate() {
        for (s, scale) in block[..4].iter_mut().enumerate() {
            *scale = (4 * b + s) as u8;
        }
    }
    let x = rng.floats(n, -3.0, 3.0);
    let acts = Acts::new(scalar(), &x, group_of(ty));
    for kernels in available() {
        let tiles: Vec<Vec<f32>> = tile_heights(kernels)
            .into_iter()
            .map(|height| tile_dots(kernels, height, ty, &matrix, n, &acts))
            .collect();
        for (b, row) in matrix.chunks_exact(ty.block_bytes()).enumerate() {
            let (expected, magnitude) = reference_dot(ty, row, &acts.reconstruct());
            let what = format!(
                "{} scales {:#04x} to {:#04x}",
                kernels.name,
                4 * b,
                4 * b + 3
            );
            assert_close(
                row_dot(kernels, ty, row, n, &acts),
                expected,
                magnitude,
                &what,
            );
            for got in &tiles {
                assert_close(got[b], expected, magnitude, &format!("tile {what}"));
            }
        }
    }
}

#[test]
fn tiles_match_the_definition() {
    let mut rng = Rng(21);
    for kernels in available() {
        for height in tile_heights(kernels) {
            for ty in QUANT_TYPES {
                // Full tiles, a tile with one row missing, and a lone row.
                for (blocks, rows) in [(1, height), (2, height - 1), (3, 1), (5, 2 * height + 3)] {
                    let n = blocks * ty.block_len();
                    let matrix = random_weights(&mut rng, ty, rows * n, 3.2);
                    let x = rng.floats(n, -3.0, 3.0);
                    let acts = Acts::new(scalar(), &x, group_of(ty));
                    let got = tile_dots(kernels, height, ty, &matrix, n, &acts);
                    assert_eq!(got.len(), rows);
                    let row_bytes = ty.row_bytes(n).unwrap();
                    for (r, row) in matrix.chunks_exact(row_bytes).enumerate() {
                        let (expected, magnitude) = reference_dot(ty, row, &acts.reconstruct());
                        let what = format!(
                            "tile {} {} n={n} row {r} of {rows}",
                            kernels.name,
                            ty.name()
                        );
                        assert_close(got[r], expected, magnitude, &what);
                    }
                }
            }
        }
    }
}

/// A kernel for four vectors must give exactly what four single calls give,
/// so that results never depend on how a batch is grouped.
#[test]
fn four_vector_tiles_match_one_vector_tiles_exactly() {
    let mut rng = Rng(29);
    for kernels in available() {
        let Some(dot4) = kernels.tile_dot4() else {
            continue;
        };
        let height = kernels.tile_rows();
        let mut tile = Tile::default();
        for ty in QUANT_TYPES {
            let group = group_of(ty);
            // Two groups or blocks, whichever is longer, so that sums cross
            // from one group to the next.
            let n = 2 * ty.block_len().max(group);
            let row_bytes = ty.row_bytes(n).unwrap();
            let matrix = random_weights(&mut rng, ty, height * n, 3.2);
            kernels.fill_tile(&mut tile, ty, &matrix, row_bytes);
            let acts: Vec<Acts> = (0..4)
                .map(|_| Acts::new(kernels, &rng.floats(n, -2.0, 2.0), group))
                .collect();
            let sums: Vec<Vec<f32>> = acts.iter().map(scaled_sums).collect();
            let mut expected = vec![f32::NAN; 4 * height];
            for (k, (act, sum)) in acts.iter().zip(&sums).enumerate() {
                // SAFETY: the tile and the activations both hold `n` columns,
                // and each output has room for a whole tile.
                unsafe {
                    (kernels.tile_dot())(
                        &tile,
                        act.as_ref(),
                        sum.as_ptr(),
                        group,
                        expected[k * height..].as_mut_ptr(),
                    )
                };
            }
            let mut got = vec![f32::NAN; 4 * height];
            let refs = [0, 1, 2, 3].map(|k| acts[k].as_ref());
            let pointers = [0, 1, 2, 3].map(|k| sums[k].as_ptr());
            // SAFETY: as above, for four vectors at once.
            unsafe { dot4(&tile, &refs, &pointers, group, got.as_mut_ptr()) };
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&got),
                bits(&expected),
                "{} {}",
                kernels.name,
                ty.name()
            );
        }
    }
}

#[test]
fn every_backend_fills_tiles_like_the_portable_fill() {
    let mut rng = Rng(23);
    for kernels in available() {
        let height = kernels.tile_rows();
        if height == 0 {
            continue;
        }
        let (mut expected, mut got) = (Tile::default(), Tile::default());
        for ty in QUANT_TYPES {
            // Every number of rows a tile can hold, on rows of two lengths,
            // with random bytes and with the extreme ones.
            for rows in 1..=height {
                for blocks in [1, 3] {
                    let n = blocks * ty.block_len();
                    let row_bytes = ty.row_bytes(n).unwrap();
                    let mut matrices = vec![random_weights(&mut rng, ty, rows * n, 3.2)];
                    for fill in [0x00u8, 0x7f, 0x80, 0xff] {
                        matrices.push(vec![fill; rows * row_bytes]);
                    }
                    for matrix in matrices {
                        match kernels.signed_tiles() {
                            true => expected.fill_signed(ty, height, &matrix, row_bytes),
                            false => expected.fill(ty, height, &matrix, row_bytes),
                        }
                        kernels.fill_tile(&mut got, ty, &matrix, row_bytes);
                        let what = format!("{} {} {rows} rows of {n}", kernels.name, ty.name());
                        assert_eq!(
                            (got.height, got.cols, got.signed, got.max, got.has_offset),
                            (
                                expected.height,
                                expected.cols,
                                expected.signed,
                                expected.max,
                                expected.has_offset
                            ),
                            "{what}"
                        );
                        assert_eq!(&got.q[..], &expected.q[..], "{what}: weights");
                        // Compared as bits, except that NaN equals NaN and
                        // the two zeros equal each other.
                        let same = |a: &[f32], b: &[f32]| {
                            a.iter()
                                .zip(b)
                                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
                        };
                        assert!(same(&got.scale, &expected.scale), "{what}: scales");
                        assert!(same(&got.offset, &expected.offset), "{what}: offsets");
                    }
                }
            }
        }
    }
}

#[test]
fn tile_kernels_handle_the_whole_byte_range() {
    // No weight format produces -128, but a signed tile may hold any byte.
    let mut rng = Rng(22);
    for kernels in available() {
        for height in tile_heights(kernels) {
            for (n, group) in [(32, 32), (160, 32), (256, 256), (768, 256)] {
                let q: Vec<i8> = (0..n * height)
                    .map(|i| [-128i8, 127, -1, 0, 1, -127][(i * 7 + i / 5) % 6])
                    .collect();
                let tile = Tile {
                    height,
                    cols: n,
                    q: q.as_slice().into(),
                    scale: rng.floats(n / 16 * height, -0.5, 0.5).as_slice().into(),
                    offset: rng.floats(n / 16 * height, -0.5, 0.5).as_slice().into(),
                    has_offset: true,
                    signed: true,
                    ..Tile::default()
                };
                let x: Vec<f32> = (0..n)
                    .map(|i| if i % 3 == 0 { -2.5 } else { 2.5 })
                    .collect();
                let acts = Acts::new(scalar(), &x, group);
                let sums = scaled_sums(&acts);
                let mut out = vec![0.0f32; height];
                // SAFETY: the tile, `acts` and `sums` describe `n` columns,
                // and `out` has one slot per row.
                unsafe {
                    (kernels.tile_dot())(
                        &tile,
                        acts.as_ref(),
                        sums.as_ptr(),
                        group,
                        out.as_mut_ptr(),
                    )
                };
                for (r, got) in out.iter().enumerate() {
                    let mut expected = 0.0f64;
                    let mut magnitude = 0.0f64;
                    for col in 0..n {
                        let weight = tile.q[(col / 4) * 4 * height + r * 4 + col % 4] as f64;
                        let term = tile.scale[col / 16 * height + r] as f64
                            * weight
                            * acts.q[col] as f64
                            * acts.scale[col / group] as f64;
                        expected += term;
                        magnitude += term.abs();
                    }
                    for (sub, sum) in sums.iter().enumerate() {
                        let term = tile.offset[sub * height + r] as f64 * *sum as f64;
                        expected += term;
                        magnitude += term.abs();
                    }
                    let what = format!("{} n={n} group={group} row {r}", kernels.name);
                    assert_close(*got, expected, magnitude, &what);
                }
            }
        }
    }
}

#[test]
fn float_rows_match_the_definition() {
    let mut rng = Rng(18);
    for kernels in available() {
        for n in 0..45 {
            let weights = rng.floats(n, -2.0, 2.0);
            let x = rng.floats(n, -2.0, 2.0);

            let f32_row: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
            let halves: Vec<u16> = weights.iter().map(|w| f32_to_f16(*w)).collect();
            let f16_row: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
            let brains: Vec<u16> = weights.iter().map(|w| (w.to_bits() >> 16) as u16).collect();
            let bf16_row: Vec<u8> = brains.iter().flat_map(|h| h.to_le_bytes()).collect();

            let cases: [(TensorType, &[u8], Vec<f32>); 3] = [
                (TensorType::F32, &f32_row, weights.clone()),
                (
                    TensorType::F16,
                    &f16_row,
                    halves.iter().map(|h| f16_to_f32(*h)).collect(),
                ),
                (
                    TensorType::BF16,
                    &bf16_row,
                    brains.iter().map(|h| bf16_to_f32(*h)).collect(),
                ),
            ];
            for (ty, row, stored) in cases {
                let expected: f64 = stored
                    .iter()
                    .zip(&x)
                    .map(|(w, v)| *w as f64 * *v as f64)
                    .sum();
                let got = match kernels.row_dot(ty) {
                    // SAFETY: `row` holds `n` weights and `x` holds `n` floats.
                    RowDot::Float(dot) => unsafe { dot(row.as_ptr(), x.as_ptr(), n) },
                    RowDot::Quant(_) => panic!("{} is a float type", ty.name()),
                };
                assert_close(
                    got,
                    expected,
                    n as f64,
                    &format!("{} {} n={n}", kernels.name, ty.name()),
                );
            }
        }
    }
}

/// A microbenchmark, not a check. Run it with:
/// `cargo test --release --lib kernel_speed -- --ignored --nocapture`
///
/// Prints the time per weight of every row kernel on a matrix small enough
/// to stay in the CPU cache, so the number reflects the kernel alone.
#[test]
#[ignore = "a benchmark, run on request"]
fn kernel_speed() {
    let mut rng = Rng(19);
    let (rows, cols) = (32, 1024);
    let x = rng.floats(cols, -3.0, 3.0);
    println!("picoseconds per weight, {rows} rows of {cols}:");
    for ty in QUANT_TYPES {
        let matrix = random_weights(&mut rng, ty, rows * cols, 3.2);
        let row_bytes = ty.row_bytes(cols).unwrap();
        let acts = Acts::new(scalar(), &x, group_of(ty));
        let mut line = format!("  {:<7}", ty.name());
        for kernels in available() {
            let mut sink = 0.0f32;
            let rounds = 2000;
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                for row in matrix.chunks_exact(row_bytes) {
                    sink += row_dot(kernels, ty, row, cols, &acts);
                }
            }
            let picos = start.elapsed().as_secs_f64() * 1e12 / (rounds * rows * cols) as f64;
            std::hint::black_box(sink);
            line.push_str(&format!("  {} {picos:>6.1}", kernels.name));
        }

        // The tile route: filling a tile, then reducing it once. Both are
        // counted per weight of the tile.
        for kernels in available() {
            let height = kernels.tile_rows();
            if height == 0 {
                continue;
            }
            let mut tile = Tile::default();
            let rounds = 400;
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                for rows in matrix.chunks_exact(height * row_bytes) {
                    kernels.fill_tile(&mut tile, ty, std::hint::black_box(rows), row_bytes);
                }
            }
            let picos = start.elapsed().as_secs_f64() * 1e12 / (rounds * rows * cols) as f64;
            line.push_str(&format!("  fill {picos:>6.1}"));
            let sums = scaled_sums(&acts);
            let mut out = [0.0f32; 8];
            let rounds = 40000;
            let start = std::time::Instant::now();
            for _ in 0..rounds {
                // SAFETY: the tile, `acts` and `sums` describe `cols`
                // columns, and `out` has room for a tile's rows.
                unsafe {
                    (kernels.tile_dot())(
                        &tile,
                        acts.as_ref(),
                        sums.as_ptr(),
                        acts.group,
                        out.as_mut_ptr(),
                    )
                };
                std::hint::black_box(&mut out);
            }
            let picos = start.elapsed().as_secs_f64() * 1e12 / (rounds * height * cols) as f64;
            line.push_str(&format!("  tile-{} {picos:>5.1}", kernels.name));
            // Four vectors at once, counted per weight and vector.
            if let Some(dot4) = kernels.tile_dot4() {
                let refs = [0, 1, 2, 3].map(|_| acts.as_ref());
                let pointers = [0, 1, 2, 3].map(|_| sums.as_ptr());
                let mut out = [0.0f32; 32];
                let rounds = 10000;
                let start = std::time::Instant::now();
                for _ in 0..rounds {
                    // SAFETY: as above, and `out` has room for four tiles.
                    unsafe { dot4(&tile, &refs, &pointers, acts.group, out.as_mut_ptr()) };
                    std::hint::black_box(&mut out);
                }
                let picos =
                    start.elapsed().as_secs_f64() * 1e12 / (rounds * 4 * height * cols) as f64;
                line.push_str(&format!("  tile4 {picos:>5.1}"));
            }
        }
        println!("{line}");
    }

    // Reading a prompt passes a whole batch of activations by each tile. For
    // wide matrices they no longer fit the nearest caches. Picoseconds per
    // weight and vector.
    for kernels in available() {
        let Some(dot4) = kernels.tile_dot4() else {
            continue;
        };
        let ty = TensorType::Q4_0;
        let vectors = 64;
        // Four distinct vectors stay in the nearest cache, 64 do not. Two
        // tiles per group of vectors, as the multiply takes them at this
        // width, read each group's activations from further out once.
        for (cols, distinct, count) in [(2048, 4, 1), (2048, 64, 1), (2048, 64, 2), (7680, 64, 1)] {
            let height = kernels.tile_rows();
            let row_bytes = ty.row_bytes(cols).unwrap();
            let tiles: Vec<Tile> = (0..count)
                .map(|_| {
                    let rows = random_weights(&mut rng, ty, height * cols, 3.2);
                    let mut tile = Tile::default();
                    kernels.fill_tile(&mut tile, ty, &rows, row_bytes);
                    tile
                })
                .collect();
            // `distinct` different vectors, repeated to make up the batch.
            let batch: Vec<Acts> = (0..distinct)
                .map(|_| Acts::new(scalar(), &rng.floats(cols, -3.0, 3.0), 32))
                .collect();
            let batch: Vec<&Acts> = (0..vectors).map(|v| &batch[v % distinct]).collect();
            let sums: Vec<Vec<f32>> = batch.iter().map(|acts| scaled_sums(acts)).collect();
            let mut out = [0.0f32; 32];
            let mut best = f64::MAX;
            for _ in 0..20 {
                let start = std::time::Instant::now();
                for group in 0..vectors / 4 {
                    let refs = [0, 1, 2, 3].map(|k| batch[group * 4 + k].as_ref());
                    let pointers = [0, 1, 2, 3].map(|k| sums[group * 4 + k].as_ptr());
                    for tile in &tiles {
                        // SAFETY: the tile and every vector hold `cols`
                        // columns, and `out` has room for four tiles.
                        unsafe { dot4(tile, &refs, &pointers, 32, out.as_mut_ptr()) };
                        std::hint::black_box(&mut out);
                    }
                }
                best = best.min(start.elapsed().as_secs_f64());
            }
            let picos = best * 1e12 / (vectors * count * height * cols) as f64;
            println!(
                "  {} Q4_0 tile4, a batch of {vectors} vectors of {cols}, {distinct} distinct, {count} tiles: {picos:.1}",
                kernels.name
            );
        }
    }

    // Generating reads every weight from memory once per token. A matrix
    // far larger than the caches shows what that costs.
    let (rows, cols) = (16384, 2048);
    let x = rng.floats(cols, -3.0, 3.0);
    println!("picoseconds per weight, {rows} rows of {cols}, read from memory:");
    for ty in [
        TensorType::Q4_0,
        TensorType::Q8_0,
        TensorType::Q4K,
        TensorType::Q6K,
    ] {
        let matrix = random_weights(&mut rng, ty, rows * cols, 3.2);
        let row_bytes = ty.row_bytes(cols).unwrap();
        let acts = Acts::new(scalar(), &x, group_of(ty));
        let mut line = format!("  {:<7}", ty.name());
        for kernels in available() {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut sink = 0.0f32;
                let start = std::time::Instant::now();
                for row in matrix.chunks_exact(row_bytes) {
                    sink += row_dot(kernels, ty, row, cols, &acts);
                }
                best = best.min(start.elapsed().as_secs_f64());
                std::hint::black_box(sink);
            }
            line.push_str(&format!(
                "  {} {:>6.1}",
                kernels.name,
                best * 1e12 / (rows * cols) as f64
            ));
        }
        println!("{line}");
    }
}

#[test]
fn half_attention_kernels_match_the_definition() {
    let mut rng = Rng(23);
    for kernels in available() {
        for len in [1, 7, 8, 9, 24, 32, 40, 64, 71, 576] {
            for n_queries in [1, 2, 3, 4, 5, 8] {
                for n in [0, 1, 2, 5, 33] {
                    let queries = rng.floats(n_queries * len, -2.0, 2.0);
                    let halves: Vec<u16> = rng
                        .floats(n * len, -2.0, 2.0)
                        .iter()
                        .map(|&value| f32_to_f16(value))
                        .collect();
                    let rows: Vec<f64> = halves.iter().map(|&h| f16_to_f32(h) as f64).collect();
                    let stride = n + 1;
                    let what = format!("{} len {len} queries {n_queries} rows {n}", kernels.name);

                    let mut scores = vec![-7.0f32; n_queries * stride];
                    kernels.dot_rows_any(&queries, Rows::F16(&halves), len, &mut scores, stride);
                    for h in 0..n_queries {
                        for j in 0..n {
                            let expected: f64 = (0..len)
                                .map(|i| queries[h * len + i] as f64 * rows[j * len + i])
                                .sum();
                            assert_close(scores[h * stride + j], expected, len as f64, &what);
                        }
                        assert_eq!(scores[h * stride + n], -7.0, "{what}: wrote past the rows");
                    }

                    let weights = rng.floats(n_queries * stride, -1.0, 1.0);
                    let start = rng.floats(n_queries * len, -1.0, 1.0);
                    let mut outs = start.clone();
                    kernels.mix_rows_any(&weights, Rows::F16(&halves), len, &mut outs, stride);
                    for h in 0..n_queries {
                        for i in 0..len {
                            let mixed: f64 = (0..n)
                                .map(|j| weights[h * stride + j] as f64 * rows[j * len + i])
                                .sum();
                            let expected = start[h * len + i] as f64 + mixed;
                            assert_close(outs[h * len + i], expected, n as f64 + 1.0, &what);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn half_conversions_are_exact() {
    // Every half widens exactly, through the branch-free conversion too.
    for half in 0..=u16::MAX {
        let (slow, fast) = (f16_to_f32(half), scalar::half_to_f32(half));
        if slow.is_nan() {
            assert!(fast.is_nan(), "{half:#06x}");
        } else {
            assert_eq!(slow.to_bits(), fast.to_bits(), "{half:#06x}");
        }
    }
    // Rounding to halves agrees with the portable conversion on every
    // backend, ties and the edges of the range included.
    let mut values = vec![
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        65504.0,
        65520.0,
        1e9,
        -1e9,
        6.1e-5,
        5.96e-8,
        2.98e-8,
        1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0 + 1.0 / 2048.0,
        1.0 + 3.0 / 2048.0,
    ];
    let mut rng = Rng(29);
    values.extend(rng.floats(4093, -70000.0, 70000.0));
    values.extend(rng.floats(1000, -1e-4, 1e-4));
    for kernels in available() {
        let mut out = vec![0u16; values.len()];
        kernels.to_half(&values, &mut out);
        for (value, half) in values.iter().zip(&out) {
            assert_eq!(*half, f32_to_f16(*value), "{} {value}", kernels.name);
        }
    }
}

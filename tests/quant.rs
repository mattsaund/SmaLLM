//! Checks that SmaLLM expands weight blocks to exactly the floats ggml
//! does.
//!
//! The files in `tests/golden/quant` come from `tools/reference/quant.c`.
//! It quantizes pseudo-random rows with ggml and expands the blocks again
//! with ggml's own code. Each file holds, little-endian: the number of
//! elements as a 32-bit integer, the blocks, and ggml's float for each
//! element. `tools/reference/README.md` says how to write them again.

use smallm::tensor::TensorType;
use smallm::tensor::quant::{DecodedRow, decode_row, dequantize_row};

/// The formats that have vectors, all checked against ggml.
const FORMATS: [TensorType; 11] = [
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
];

/// One format's vectors: the blocks and the floats ggml made of them.
struct Vectors {
    blocks: Vec<u8>,
    floats: Vec<f32>,
}

/// Reads the vectors of `ty`, named like ggml names the type.
fn vectors(ty: TensorType) -> Vectors {
    let name = ty.name().to_lowercase();
    let path = format!(
        "{}/tests/golden/quant/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let count = u32::from_le_bytes(data[..4].try_into().expect("a count")) as usize;
    let block_bytes = ty.row_bytes(count).expect("whole blocks");
    assert_eq!(data.len(), 4 + block_bytes + 4 * count, "{path}: size");
    let (blocks, floats) = data[4..].split_at(block_bytes);
    let floats = floats
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().expect("four bytes")))
        .collect();
    Vectors {
        blocks: blocks.to_vec(),
        floats,
    }
}

/// True when two floats have the same bits, or are both zero. A zero's
/// sign may differ: ggml writes -0.0 for a negative weight times a zero
/// scale, and the `+ offset` of SmaLLM's weight shape turns that into 0.0.
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0)
}

#[test]
fn every_format_expands_to_the_floats_of_ggml() {
    for ty in FORMATS {
        let Vectors { blocks, floats } = vectors(ty);
        let mut got = vec![f32::NAN; floats.len()];
        dequantize_row(ty, &blocks, &mut got);
        for (i, (got, want)) in got.iter().zip(&floats).enumerate() {
            assert!(
                same(*got, *want),
                "{} weight {i}: got {got:e}, ggml has {want:e}",
                ty.name()
            );
        }
    }
}

#[test]
fn the_tile_route_sees_the_same_weights() {
    // Tiles are filled from `decode_row`, which keeps the integer weights
    // and their scales apart. Put back together, they must be ggml's floats
    // too.
    let mut row = DecodedRow::default();
    for ty in FORMATS {
        let Vectors { blocks, floats } = vectors(ty);
        decode_row(ty, &blocks, &mut row);
        assert_eq!(row.q.len(), floats.len(), "{}", ty.name());
        for (i, want) in floats.iter().enumerate() {
            let got = row.scale[i / 16] * row.q[i] as f32 + row.offset[i / 16];
            assert!(
                same(got, *want),
                "{} weight {i}: got {got:e}, ggml has {want:e}",
                ty.name()
            );
        }
    }
}

#[test]
fn the_vectors_are_worth_checking() {
    // Guards the vectors themselves: a file of zeros would prove nothing.
    for ty in FORMATS {
        let Vectors { floats, .. } = vectors(ty);
        let nonzero = floats.iter().filter(|v| **v != 0.0).count();
        assert!(2 * nonzero > floats.len(), "{}: mostly zeros", ty.name());
        assert!(floats.iter().any(|v| *v > 0.0), "{}", ty.name());
        assert!(floats.iter().any(|v| *v < 0.0), "{}", ty.name());
        assert!(floats.iter().all(|v| v.is_finite()), "{}", ty.name());
    }
}

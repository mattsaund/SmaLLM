//! Tensor element types and how their bytes are laid out.
//!
//! A GGUF tensor is a matrix stored row by row. Float types store one value
//! per element. Quantized types store blocks: a fixed number of elements
//! packed into a fixed number of bytes with one or more shared scales.

pub mod f16;
mod iq_tables;
pub mod quant;

/// The element types this engine can compute with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorType {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// 16-bit brain float: the top half of an f32.
    BF16,
    /// 4 bits per weight, one scale per 32.
    Q4_0,
    /// 4 bits per weight, one scale and one minimum per 32.
    Q4_1,
    /// 5 bits per weight, one scale per 32.
    Q5_0,
    /// 5 bits per weight, one scale and one minimum per 32.
    Q5_1,
    /// 8 bits per weight, one scale per 32.
    Q8_0,
    /// 2 bits per weight in 256-weight super-blocks.
    Q2K,
    /// 3 bits per weight in 256-weight super-blocks.
    Q3K,
    /// 4 bits per weight in 256-weight super-blocks.
    Q4K,
    /// 5 bits per weight in 256-weight super-blocks.
    Q5K,
    /// 6 bits per weight in 256-weight super-blocks.
    Q6K,
    /// 4 bits per weight through a non-linear table, one scale per 32.
    Iq4Nl,
    /// The IQ4_NL table in 256-weight super-blocks with 6-bit scales, 4.25
    /// bits per weight.
    Iq4Xs,
    /// Four weights at a time from a grid of 512 patterns, 3.44 bits per
    /// weight.
    Iq3S,
    /// Four weights at a time from a grid of 256 patterns, 3.06 bits per
    /// weight.
    Iq3Xxs,
    /// Eight weights at a time from a grid of 1024 patterns, 2.56 bits per
    /// weight.
    Iq2S,
    /// Eight weights at a time from a grid of 512 patterns, 2.31 bits per
    /// weight.
    Iq2Xs,
    /// Eight weights at a time from a grid of 256 patterns, 2.06 bits per
    /// weight.
    Iq2Xxs,
    /// Eight weights at a time from a grid of 2048 patterns, with a scale
    /// per 16 weights, 1.75 bits per weight.
    Iq1M,
    /// Eight weights at a time from a grid of 2048 patterns, with a scale
    /// per 32 weights, 1.56 bits per weight.
    Iq1S,
    /// Weights of -1, 0 or 1, two bits each, one scale per 256.
    Tq2_0,
    /// Weights of -1, 0 or 1, five to a byte, one scale per 256.
    Tq1_0,
    /// 4-bit floats with a power of two shared by 32 of them.
    Mxfp4,
    /// 4-bit floats in blocks of 64, with an 8-bit float scale for each 16.
    Nvfp4,
    /// Only the sign of each weight, one bit, with one scale per 128.
    Q1_0,
    /// Weights of -1, 0, 1 or 2, two bits each, with one scale per 64.
    Q2_0,
}

/// How a type's blocks group, which decides how activations are quantized
/// before an integer dot product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// One float per element. Activations stay in floating point.
    Float,
    /// Activations in groups of 32, each with its own float scale, the way
    /// ggml quantizes them to Q8_0. Most of these formats have blocks of 32
    /// too. NVFP4 and Q2_0 have blocks of 64 and Q1_0 blocks of 128, which
    /// span two or four groups.
    Block32,
    /// Super-blocks of 256 elements with integer sub-block scales.
    /// Activations are quantized in groups of 256.
    Block256,
}

impl Family {
    /// Elements per scale when activations are quantized for this family.
    /// Float types are not quantized, and count one element per group.
    pub fn group(self) -> usize {
        match self {
            Family::Float => 1,
            Family::Block32 => 32,
            Family::Block256 => 256,
        }
    }
}

impl TensorType {
    /// Maps a GGUF type id to a supported type.
    pub fn from_id(id: u32) -> Option<TensorType> {
        Some(match id {
            0 => TensorType::F32,
            1 => TensorType::F16,
            2 => TensorType::Q4_0,
            3 => TensorType::Q4_1,
            6 => TensorType::Q5_0,
            7 => TensorType::Q5_1,
            8 => TensorType::Q8_0,
            10 => TensorType::Q2K,
            11 => TensorType::Q3K,
            12 => TensorType::Q4K,
            13 => TensorType::Q5K,
            14 => TensorType::Q6K,
            16 => TensorType::Iq2Xxs,
            17 => TensorType::Iq2Xs,
            18 => TensorType::Iq3Xxs,
            19 => TensorType::Iq1S,
            20 => TensorType::Iq4Nl,
            21 => TensorType::Iq3S,
            22 => TensorType::Iq2S,
            23 => TensorType::Iq4Xs,
            29 => TensorType::Iq1M,
            30 => TensorType::BF16,
            34 => TensorType::Tq1_0,
            35 => TensorType::Tq2_0,
            39 => TensorType::Mxfp4,
            40 => TensorType::Nvfp4,
            41 => TensorType::Q1_0,
            42 => TensorType::Q2_0,
            _ => return None,
        })
    }

    /// The name used by GGUF tools.
    pub fn name(self) -> &'static str {
        match self {
            TensorType::F32 => "F32",
            TensorType::F16 => "F16",
            TensorType::BF16 => "BF16",
            TensorType::Q4_0 => "Q4_0",
            TensorType::Q4_1 => "Q4_1",
            TensorType::Q5_0 => "Q5_0",
            TensorType::Q5_1 => "Q5_1",
            TensorType::Q8_0 => "Q8_0",
            TensorType::Q2K => "Q2_K",
            TensorType::Q3K => "Q3_K",
            TensorType::Q4K => "Q4_K",
            TensorType::Q5K => "Q5_K",
            TensorType::Q6K => "Q6_K",
            TensorType::Iq4Nl => "IQ4_NL",
            TensorType::Iq4Xs => "IQ4_XS",
            TensorType::Iq3S => "IQ3_S",
            TensorType::Iq3Xxs => "IQ3_XXS",
            TensorType::Iq2S => "IQ2_S",
            TensorType::Iq2Xs => "IQ2_XS",
            TensorType::Iq2Xxs => "IQ2_XXS",
            TensorType::Iq1M => "IQ1_M",
            TensorType::Iq1S => "IQ1_S",
            TensorType::Tq2_0 => "TQ2_0",
            TensorType::Tq1_0 => "TQ1_0",
            TensorType::Mxfp4 => "MXFP4",
            TensorType::Nvfp4 => "NVFP4",
            TensorType::Q1_0 => "Q1_0",
            TensorType::Q2_0 => "Q2_0",
        }
    }

    /// Elements in one block. For most types that is the group size of
    /// their family, but a few have longer blocks.
    pub fn block_len(self) -> usize {
        match self {
            TensorType::Nvfp4 | TensorType::Q2_0 => 64,
            TensorType::Q1_0 => 128,
            _ => self.family().group(),
        }
    }

    /// Bytes in one block.
    pub fn block_bytes(self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::F16 | TensorType::BF16 => 2,
            TensorType::Q4_0 | TensorType::Iq4Nl => 18,
            TensorType::Q4_1 => 20,
            TensorType::Q5_0 => 22,
            TensorType::Q5_1 => 24,
            TensorType::Q8_0 => 34,
            TensorType::Q2K => 84,
            TensorType::Q3K => 110,
            TensorType::Q4K => 144,
            TensorType::Q5K => 176,
            TensorType::Q6K => 210,
            TensorType::Iq4Xs => 136,
            TensorType::Iq3S => 110,
            TensorType::Iq3Xxs => 98,
            TensorType::Iq2S => 82,
            TensorType::Iq2Xs => 74,
            TensorType::Iq2Xxs | TensorType::Tq2_0 => 66,
            TensorType::Iq1M => 56,
            TensorType::Iq1S => 50,
            TensorType::Tq1_0 => 54,
            TensorType::Mxfp4 => 17,
            TensorType::Nvfp4 => 36,
            TensorType::Q1_0 | TensorType::Q2_0 => 18,
        }
    }

    /// The block grouping of this type.
    pub fn family(self) -> Family {
        match self {
            TensorType::F32 | TensorType::F16 | TensorType::BF16 => Family::Float,
            TensorType::Q4_0
            | TensorType::Q4_1
            | TensorType::Q5_0
            | TensorType::Q5_1
            | TensorType::Q8_0
            | TensorType::Iq4Nl
            | TensorType::Mxfp4
            | TensorType::Nvfp4
            | TensorType::Q1_0
            | TensorType::Q2_0 => Family::Block32,
            TensorType::Q2K
            | TensorType::Q3K
            | TensorType::Q4K
            | TensorType::Q5K
            | TensorType::Q6K
            | TensorType::Iq4Xs
            | TensorType::Iq3S
            | TensorType::Iq3Xxs
            | TensorType::Iq2S
            | TensorType::Iq2Xs
            | TensorType::Iq2Xxs
            | TensorType::Iq1M
            | TensorType::Iq1S
            | TensorType::Tq2_0
            | TensorType::Tq1_0 => Family::Block256,
        }
    }

    /// Bytes in a row of `cols` elements, or `None` when `cols` does not
    /// divide into whole blocks or the size overflows.
    pub fn row_bytes(self, cols: usize) -> Option<usize> {
        let block = self.block_len();
        if cols % block != 0 {
            return None;
        }
        (cols / block).checked_mul(self.block_bytes())
    }
}

/// The name of any GGUF tensor type id, including the ones this engine does
/// not compute with. Used for listings and error messages.
pub fn type_name(id: u32) -> &'static str {
    if let Some(ty) = TensorType::from_id(id) {
        return ty.name();
    }
    match id {
        9 => "Q8_1",
        15 => "Q8_K",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        43 => "F8_E4M3",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_through_names() {
        for id in 0..64 {
            if let Some(ty) = TensorType::from_id(id) {
                assert_eq!(type_name(id), ty.name());
                assert!(ty.block_bytes() > 0);
            }
        }
    }

    #[test]
    fn row_bytes_requires_whole_blocks() {
        assert_eq!(TensorType::F32.row_bytes(10), Some(40));
        assert_eq!(TensorType::Q8_0.row_bytes(64), Some(68));
        assert_eq!(TensorType::Q4K.row_bytes(512), Some(288));
        assert_eq!(TensorType::Q4K.row_bytes(576), None);
        assert_eq!(TensorType::Q8_0.row_bytes(usize::MAX - 31), None);
    }

    #[test]
    fn long_blocks_still_quantize_activations_in_groups_of_32() {
        // ggml multiplies these formats with Q8_0 activations, whose
        // blocks are 32 long, so a block of weights spans several groups.
        for (ty, len, bytes) in [
            (TensorType::Nvfp4, 64, 36),
            (TensorType::Q2_0, 64, 18),
            (TensorType::Q1_0, 128, 18),
        ] {
            assert_eq!(
                (ty.block_len(), ty.block_bytes()),
                (len, bytes),
                "{}",
                ty.name()
            );
            assert_eq!(ty.family().group(), 32, "{}", ty.name());
            assert_eq!(ty.row_bytes(256), Some(256 / len * bytes), "{}", ty.name());
            assert_eq!(ty.row_bytes(32), None, "{}", ty.name());
        }
    }
}

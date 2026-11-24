//! Tensor element types and how their bytes are laid out.
//!
//! A GGUF tensor is a matrix stored row by row. Float types store one value
//! per element. Quantized types store blocks: a fixed number of elements
//! packed into a fixed number of bytes with one or more shared scales.

pub mod f16;
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
}

/// How a type's blocks group, which decides how activations are quantized
/// before an integer dot product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// One float per element. Activations stay in floating point.
    Float,
    /// Blocks of 32 elements, each with its own float scale.
    Block32,
    /// Super-blocks of 256 elements with integer sub-block scales.
    Block256,
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
            20 => TensorType::Iq4Nl,
            30 => TensorType::BF16,
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
        }
    }

    /// Elements in one block.
    pub fn block_len(self) -> usize {
        match self.family() {
            Family::Float => 1,
            Family::Block32 => 32,
            Family::Block256 => 256,
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
            | TensorType::Iq4Nl => Family::Block32,
            TensorType::Q2K
            | TensorType::Q3K
            | TensorType::Q4K
            | TensorType::Q5K
            | TensorType::Q6K => Family::Block256,
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
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
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
}

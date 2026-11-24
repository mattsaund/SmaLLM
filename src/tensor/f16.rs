//! Half-precision float conversion.
//!
//! Quantized blocks store their scales as IEEE 754 binary16. Rust has no
//! stable 16-bit float type, so the conversion is done on the bit pattern.

/// Converts IEEE binary16 bits to an f32. Exact for every input.
#[inline]
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let fraction = (half & 0x03ff) as u32;
    let bits = match exponent {
        // Zero or subnormal: the value is fraction * 2^-24.
        0 => {
            let magnitude = fraction as f32 * (1.0 / 16_777_216.0);
            return f32::from_bits(magnitude.to_bits() | sign);
        }
        // Infinity or NaN.
        31 => sign | 0x7f80_0000 | (fraction << 13),
        // Normal: rebias the exponent from 15 to 127.
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(bits)
}

/// Converts brain float bits to an f32. BF16 is the upper half of an f32.
#[inline]
pub fn bf16_to_f32(half: u16) -> f32 {
    f32::from_bits((half as u32) << 16)
}

/// Converts an f32 to IEEE binary16 bits, rounding to nearest, ties to even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 255 {
        // Infinity keeps a zero fraction. NaN keeps a non-zero one.
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        // Subnormal result, or zero when the value is too small.
        if half_exponent < -10 {
            return sign;
        }
        let full = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let kept_lsb = (full >> shift) & 1;
        let rounded = (full + (1 << (shift - 1)) - 1 + kept_lsb) >> shift;
        return sign | rounded as u16;
    }
    // Normal result. A rounding carry moves into the exponent on its own.
    let kept_lsb = (mantissa >> 13) & 1;
    let rounded = mantissa + 0x0fff + kept_lsb;
    sign | (((half_exponent as u32) << 10) + (rounded >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_convert() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn every_half_survives_a_round_trip() {
        for half in 0..=u16::MAX {
            let value = f16_to_f32(half);
            if value.is_nan() {
                assert!(f16_to_f32(f32_to_f16(value)).is_nan());
            } else {
                assert_eq!(f32_to_f16(value), half, "half {half:#06x}");
            }
        }
    }

    #[test]
    fn rounding_is_to_nearest_even() {
        // Halfway between 1.0 and the next half rounds down to the even one.
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        // Halfway between the next two rounds up to the even one.
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_f16(1.0e9), 0x7c00);
        assert_eq!(f32_to_f16(1.0e-9), 0x0000);
    }

    #[test]
    fn brain_float_is_the_top_half() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
    }
}

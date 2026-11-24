//! Character classes, case and accent mappings, and the byte-level
//! alphabet used by BPE tokenizers.

use super::unicode_maps::{LOWERCASE, NFD};
use super::unicode_tables::{BY_NUMBER, RUNS};

pub(super) use super::unicode_tables::Category;

/// The general category of `c`.
pub(super) fn category(c: char) -> Category {
    let code_point = c as u32;
    // The last run that starts at or before the code point. The first run
    // starts at zero, so there always is one.
    let index = RUNS.partition_point(|&run| run >> 5 <= code_point) - 1;
    BY_NUMBER[(RUNS[index] & 31) as usize]
}

impl Category {
    /// The major class: `'L'` for letters, `'N'` for numbers, and so on.
    pub(super) fn major(self) -> char {
        use Category::*;
        match self {
            Cc | Cf | Cn | Co | Cs => 'C',
            Ll | Lm | Lo | Lt | Lu => 'L',
            Mc | Me | Mn => 'M',
            Nd | Nl | No => 'N',
            Pc | Pd | Pe | Pf | Pi | Po | Ps => 'P',
            Sc | Sk | Sm | So => 'S',
            Zl | Zp | Zs => 'Z',
        }
    }

    /// The two-letter name, such as `"Lu"`.
    pub(super) fn name(self) -> &'static str {
        use Category::*;
        match self {
            Cc => "Cc",
            Cf => "Cf",
            Cn => "Cn",
            Co => "Co",
            Cs => "Cs",
            Ll => "Ll",
            Lm => "Lm",
            Lo => "Lo",
            Lt => "Lt",
            Lu => "Lu",
            Mc => "Mc",
            Me => "Me",
            Mn => "Mn",
            Nd => "Nd",
            Nl => "Nl",
            No => "No",
            Pc => "Pc",
            Pd => "Pd",
            Pe => "Pe",
            Pf => "Pf",
            Pi => "Pi",
            Po => "Po",
            Ps => "Ps",
            Sc => "Sc",
            Sk => "Sk",
            Sm => "Sm",
            So => "So",
            Zl => "Zl",
            Zp => "Zp",
            Zs => "Zs",
        }
    }
}

/// Unicode general category L.
pub(super) fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphabetic()
    } else {
        category(c).major() == 'L'
    }
}

/// Unicode general category N.
pub(super) fn is_number(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        category(c).major() == 'N'
    }
}

/// Unicode general category P.
pub(super) fn is_punctuation(c: char) -> bool {
    category(c).major() == 'P'
}

/// The Unicode White_Space property.
pub(super) fn is_whitespace(c: char) -> bool {
    c.is_whitespace()
}

/// A control character in llama.cpp's sense: general category C, except
/// code points that are not assigned.
pub(super) fn is_control(c: char) -> bool {
    let category = category(c);
    category.major() == 'C' && category != Category::Cn
}

/// Unicode general category M: accents and other marks.
pub(super) fn is_mark(c: char) -> bool {
    category(c).major() == 'M'
}

/// Unicode general category S.
pub(super) fn is_symbol(c: char) -> bool {
    category(c).major() == 'S'
}

/// The simple lowercase form of `c`, by llama.cpp's table. One character
/// for one, unlike Rust's full mapping, which can give two.
pub(super) fn to_lower(c: char) -> char {
    let code_point = c as u32;
    let index = LOWERCASE.partition_point(|&(first, ..)| first <= code_point);
    let Some(&(first, count, step, offset)) = index.checked_sub(1).map(|i| &LOWERCASE[i]) else {
        return c;
    };
    let distance = code_point - first;
    if distance % step as u32 == 0 && distance / (step as u32) < count as u32 {
        char::from_u32(code_point.wrapping_add_signed(offset)).unwrap_or(c)
    } else {
        c
    }
}

/// The first code point of the canonical decomposition of `c`, which for
/// an accented letter is the letter without its accent. llama.cpp
/// decomposes only this far.
pub(super) fn decompose_first(c: char) -> char {
    let code_point = c as u32;
    let index = NFD.partition_point(|&(first, _)| first <= code_point);
    let Some(&(first, packed)) = index.checked_sub(1).map(|i| &NFD[i]) else {
        return c;
    };
    if code_point - first <= packed >> 21 {
        char::from_u32(packed & ((1 << 21) - 1)).unwrap_or(c)
    } else {
        c
    }
}

/// Neither whitespace, a letter nor a number.
pub(super) fn is_other(c: char) -> bool {
    !is_whitespace(c) && !is_letter(c) && !is_number(c)
}

/// A Han character, by the ranges llama.cpp uses: the CJK unified
/// ideographs, their extensions, and the compatibility ideographs.
pub(super) fn is_han(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2CEB0..=0x2EBEF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// Maps a byte to its character in the GPT-2 byte-level alphabet.
///
/// Byte-level BPE vocabularies store every token as printable text. Bytes
/// that are printable Latin-1 characters stand for themselves. The other 68
/// are moved, in order, to the code points from U+0100 up. A space becomes
/// U+0120 and a line feed U+010A.
pub(super) fn byte_to_char(byte: u8) -> char {
    let code_point = match byte {
        0x00..=0x20 => 0x100 + byte as u32,
        0x7f..=0xa0 => 0x121 + (byte - 0x7f) as u32,
        0xad => 0x143,
        _ => byte as u32,
    };
    // Every value above is a valid scalar below U+0144.
    char::from_u32(code_point).unwrap_or('\u{fffd}')
}

/// The inverse of [`byte_to_char`]. `None` for characters outside the
/// alphabet.
pub(super) fn char_to_byte(c: char) -> Option<u8> {
    let code_point = c as u32;
    Some(match code_point {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => code_point as u8,
        0x100..=0x120 => (code_point - 0x100) as u8,
        0x121..=0x142 => (code_point - 0x121 + 0x7f) as u8,
        0x143 => 0xad,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_cover_common_scripts() {
        for c in ['a', 'Z', 'é', 'ß', 'я', '中', 'あ', 'ا', 'ह'] {
            assert!(is_letter(c), "{c}");
            assert!(!is_number(c) && !is_other(c), "{c}");
        }
        for c in ['0', '9', '٣', '५', '²', 'Ⅷ'] {
            assert!(is_number(c), "{c}");
            assert!(!is_letter(c), "{c}");
        }
        for c in [' ', '\t', '\n', '\r', '\u{a0}', '\u{3000}', '\u{2028}'] {
            assert!(is_whitespace(c));
            assert!(!is_other(c));
        }
        for c in ['!', '.', ',', '?', '(', '_', '¿', '。', '「'] {
            assert!(is_punctuation(c), "{c}");
            assert!(is_other(c), "{c}");
        }
        // Symbols and marks are "other" but not punctuation.
        for c in ['$', '+', '<', '^', '~', '€', '😀', '\u{301}'] {
            assert!(is_other(c), "{c}");
            assert!(!is_punctuation(c), "{c}");
        }
    }

    #[test]
    fn categories_are_exact() {
        let cases = [
            ('A', "Lu"),
            ('a', "Ll"),
            ('ǅ', "Lt"),
            ('ʰ', "Lm"),
            ('中', "Lo"),
            ('\u{301}', "Mn"),
            ('\u{903}', "Mc"),
            ('7', "Nd"),
            ('Ⅷ', "Nl"),
            ('½', "No"),
            ('_', "Pc"),
            ('-', "Pd"),
            ('(', "Ps"),
            (')', "Pe"),
            ('«', "Pi"),
            ('»', "Pf"),
            ('!', "Po"),
            ('$', "Sc"),
            ('^', "Sk"),
            ('+', "Sm"),
            ('©', "So"),
            (' ', "Zs"),
            ('\u{2028}', "Zl"),
            ('\u{2029}', "Zp"),
            ('\n', "Cc"),
            ('\u{200d}', "Cf"),
            ('\u{e000}', "Co"),
            ('\u{378}', "Cn"),
            ('\u{10ffff}', "Cn"),
            ('\0', "Cc"),
        ];
        for (c, name) in cases {
            assert_eq!(category(c).name(), name, "{c:?}");
        }
    }

    #[test]
    fn han_is_the_cjk_ideographs() {
        for c in ['中', '国', '\u{3400}', '\u{20000}', '\u{f900}'] {
            assert!(is_han(c), "{c}");
        }
        for c in ['a', 'あ', 'ア', '한', '。'] {
            assert!(!is_han(c), "{c}");
        }
    }

    #[test]
    fn the_byte_alphabet_is_a_bijection() {
        let mut seen = std::collections::HashSet::new();
        for byte in 0..=255u8 {
            let c = byte_to_char(byte);
            assert!(seen.insert(c), "byte {byte:#04x} collides");
            assert_eq!(char_to_byte(c), Some(byte));
            assert!(
                !c.is_whitespace() && !c.is_control(),
                "byte {byte:#04x} maps to {c:?}"
            );
        }
        assert_eq!(byte_to_char(b' '), '\u{120}');
        assert_eq!(byte_to_char(b'\n'), '\u{10a}');
        assert_eq!(byte_to_char(b'A'), 'A');
        assert_eq!(byte_to_char(0xad), '\u{143}');
        assert_eq!(char_to_byte('\u{144}'), None);
        assert_eq!(char_to_byte(' '), None);
    }
}

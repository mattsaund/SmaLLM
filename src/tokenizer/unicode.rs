//! Character classes and the byte-level alphabet used by BPE tokenizers.

use std::cmp::Ordering;

use super::unicode_tables::{LETTER, PUNCTUATION};

/// True when `c` falls inside one of the sorted ranges of `table`.
fn in_table(table: &[(u32, u32)], c: char) -> bool {
    let code_point = c as u32;
    table
        .binary_search_by(|&(low, high)| {
            if high < code_point {
                Ordering::Less
            } else if low > code_point {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// Unicode general category L.
pub(super) fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphabetic()
    } else {
        in_table(&LETTER, c)
    }
}

/// Unicode general category N.
pub(super) fn is_number(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        c.is_numeric()
    }
}

/// Unicode general category P.
pub(super) fn is_punctuation(c: char) -> bool {
    in_table(&PUNCTUATION, c)
}

/// The Unicode White_Space property.
pub(super) fn is_whitespace(c: char) -> bool {
    c.is_whitespace()
}

/// Neither whitespace, a letter nor a number.
pub(super) fn is_other(c: char) -> bool {
    !is_whitespace(c) && !is_letter(c) && !is_number(c)
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

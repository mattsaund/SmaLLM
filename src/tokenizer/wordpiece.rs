//! WordPiece, the tokenizer of BERT models.
//!
//! The text is first cleaned and cut into words. Whitespace separates
//! words, control characters are dropped, and every punctuation mark, ASCII
//! symbol and Chinese character is a word on its own. When the model asks,
//! letters are lowercased and accents stripped. Each word is then covered
//! from the left by the longest tokens that match, the first one marked
//! with `▁` as the start of a word. A word that cannot be covered becomes
//! the unknown token. This follows llama.cpp's WordPiece rules exactly.

use super::Tokenizer;
use super::unicode::{
    decompose_first, is_control, is_mark, is_punctuation, is_symbol, is_whitespace, to_lower,
};

/// How a WordPiece model cleans text.
#[derive(Clone, Copy, Debug)]
pub(super) struct Normalizer {
    pub lowercase: bool,
    pub strip_accents: bool,
}

impl Tokenizer {
    /// Appends the tokens of `text` to `out`.
    pub(super) fn encode_wordpiece(&self, text: &str, normalizer: Normalizer, out: &mut Vec<u32>) {
        for word in words(text, normalizer) {
            let word = format!("\u{2581}{word}");
            let bytes = word.as_bytes();
            let start = out.len();
            let mut at = 0;
            while at < bytes.len() {
                // The longest token that starts here.
                let longest = (at + 1..=bytes.len().min(at + self.max_token_len + 1))
                    .rev()
                    .find_map(|end| self.find(&bytes[at..end]).map(|id| (id, end)));
                match longest {
                    Some((id, end)) => {
                        out.push(id);
                        at = end;
                    }
                    None => {
                        // A word is all tokens or none.
                        out.truncate(start);
                        break;
                    }
                }
            }
            if out.len() == start {
                out.extend(self.unk);
            }
        }
    }
}

/// Cleans `text` and cuts it into words.
fn words(text: &str, normalizer: Normalizer) -> Vec<String> {
    let mut words = vec![String::new()];
    for c in text.chars() {
        let c = if normalizer.strip_accents {
            decompose_first(c)
        } else {
            c
        };
        if is_whitespace(c) {
            if !words.last().is_some_and(String::is_empty) {
                words.push(String::new());
            }
            continue;
        }
        let code_point = c as u32;
        if code_point == 0 || code_point == 0xFFFD || is_control(c) {
            continue;
        }
        if normalizer.strip_accents && is_mark(c) {
            continue;
        }
        let shown = if normalizer.lowercase { to_lower(c) } else { c };
        let alone =
            is_punctuation(c) || (code_point < 0x7F && is_symbol(c)) || is_chinese(code_point);
        let last = words.last_mut().expect("there is always a word");
        if alone {
            if !last.is_empty() {
                words.push(String::new());
            }
            let last = words.last_mut().expect("just pushed");
            last.push(shown);
            words.push(String::new());
        } else {
            last.push(shown);
        }
    }
    if words.last().is_some_and(String::is_empty) {
        words.pop();
    }
    words
}

/// The ranges of Chinese characters that BERT splits apart, as in
/// llama.cpp, which follows Hugging Face's code in starting the fifth range
/// at 0x2B920.
fn is_chinese(code_point: u32) -> bool {
    matches!(
        code_point,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_cleaned_and_split() {
        let normal = Normalizer {
            lowercase: true,
            strip_accents: true,
        };
        assert_eq!(
            words("Héllo, WORLD!  a\u{0}b\u{200D}c 中文", normal),
            ["hello", ",", "world", "!", "abc", "中", "文"]
        );
        let keep = Normalizer {
            lowercase: false,
            strip_accents: false,
        };
        assert_eq!(words("Héllo$x", keep), ["Héllo", "$", "x"]);
        assert!(words("   ", normal).is_empty());
    }
}

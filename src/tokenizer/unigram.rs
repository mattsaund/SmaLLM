//! Unigram, the SentencePiece tokenizer of T5 and XLM-RoBERTa models.
//!
//! The text is normalized first, with the map of replacements the model
//! carries: a compact trie that maps prefixes of the input to replacement
//! strings. Spaces then become `▁`. The tokens are chosen by the Viterbi
//! algorithm: of all the ways to cover the text with tokens, the one whose
//! token scores add up highest. A character no token covers becomes the
//! unknown token, at a penalty. This follows llama.cpp's Unigram tokenizer
//! exactly, down to adding scores in double precision.

use std::collections::HashSet;
use std::sync::Arc;

use crate::gguf::Mapping;

use super::{Tokenizer, attr};

/// The SentencePiece space mark.
const SPACE_MARK: &[u8] = "\u{2581}".as_bytes();

/// How far below the worst token score the unknown token scores.
const UNKNOWN_PENALTY: f64 = 10.0;

/// The model's normalization map, left in the model file.
///
/// It starts with the byte length of a XOR-compressed double array (XCDA),
/// then holds the array, one 32-bit entry per node, then the replacement
/// strings, each ending in a zero byte.
pub(super) struct Charsmap {
    map: Arc<Mapping>,
    /// File offset and length of the whole map.
    start: usize,
    len: usize,
}

impl Charsmap {
    pub fn new(map: &Arc<Mapping>, start: usize, len: usize) -> Charsmap {
        Charsmap {
            map: Arc::clone(map),
            start,
            len,
        }
    }

    fn bytes(&self) -> &[u8] {
        self.map
            .as_slice()
            .get(self.start..self.start + self.len)
            .unwrap_or(&[])
    }

    /// The array entries and the replacement strings.
    fn parts(&self) -> (&[u8], &[u8]) {
        let bytes = self.bytes();
        let Some(size) = bytes.get(..4) else {
            return (&[], &[]);
        };
        let size = u32::from_le_bytes(size.try_into().unwrap()) as usize;
        let size = size.min(bytes.len() - 4) / 4 * 4;
        (&bytes[4..4 + size], &bytes[4 + size..])
    }

    /// The longest prefix of `input` the map replaces: its length, and
    /// where its replacement starts. Walks the trie byte by byte, as
    /// llama.cpp does, and stops on anything malformed.
    fn longest_match(&self, input: &[u8]) -> Option<(usize, usize)> {
        let (nodes, _) = self.parts();
        let node = |index: u32| -> Option<u32> {
            let at = index as usize * 4;
            nodes
                .get(at..at + 4)
                .map(|entry| u32::from_le_bytes(entry.try_into().unwrap()))
        };
        // The fields packed into a node, as SentencePiece lays them out.
        let base = |packed: u32| (packed >> 10) << ((packed & (1 << 9)) >> 6);
        let check = |packed: u32| packed & ((1 << 31) | 0xFF);
        let is_leaf = |packed: u32| (packed >> 8) & 1 == 1;
        let value = |packed: u32| packed & ((1 << 31) - 1);

        let mut index = base(node(0)?);
        let mut longest = None;
        for (offset, &byte) in input.iter().enumerate() {
            if byte == 0 {
                break;
            }
            index ^= byte as u32;
            let packed = node(index)?;
            if check(packed) != byte as u32 {
                break;
            }
            index ^= base(packed);
            if is_leaf(packed) {
                longest = Some((offset + 1, value(node(index)?) as usize));
            }
        }
        longest
    }

    /// The replacement string that starts at `offset`.
    fn replacement(&self, offset: usize) -> Option<&[u8]> {
        let (_, strings) = self.parts();
        let rest = strings.get(offset..)?;
        let end = rest.iter().position(|&byte| byte == 0)?;
        Some(&rest[..end])
    }
}

/// What a Unigram tokenizer needs besides the vocabulary.
pub(super) struct Unigram {
    pub charsmap: Option<Charsmap>,
    /// Every prefix of every user-defined token. llama.cpp keeps such a
    /// prefix of the input as it is, even when it is not a whole token.
    pub user_prefixes: HashSet<Vec<u8>>,
    /// The score of the unknown token: the worst score, less a penalty.
    pub unknown_score: f64,
    pub escape_whitespaces: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,
}

impl Unigram {
    /// Gathers what the tokenizer needs from the vocabulary.
    pub fn new(
        texts: impl Iterator<Item = Vec<u8>>,
        attrs: &[u8],
        scores: &[f32],
        charsmap: Option<Charsmap>,
    ) -> Unigram {
        let mut user_prefixes = HashSet::new();
        let mut worst = f32::MAX;
        for (id, text) in texts.enumerate() {
            let flags = attrs.get(id).copied().unwrap_or(0);
            if flags & attr::NORMAL != 0 {
                worst = worst.min(scores.get(id).copied().unwrap_or(0.0));
            }
            if flags & attr::USER_DEFINED != 0 {
                for end in 1..=text.len() {
                    user_prefixes.insert(text[..end].to_vec());
                }
            }
        }
        Unigram {
            charsmap,
            user_prefixes,
            unknown_score: worst as f64 - UNKNOWN_PENALTY,
            escape_whitespaces: true,
            add_space_prefix: false,
            remove_extra_whitespaces: false,
            treat_whitespace_as_suffix: false,
        }
    }

    /// The normalized form of the start of `input`, and how many bytes of
    /// the input it stands for.
    fn normalize_prefix<'a>(&'a self, input: &'a [u8]) -> (&'a [u8], usize) {
        // A run that starts a user-defined token stays as it is.
        let kept = (1..=input.len())
            .take_while(|&end| self.user_prefixes.contains(&input[..end]))
            .last()
            .unwrap_or(0);
        if kept > 0 {
            return (&input[..kept], kept);
        }
        if let Some(charsmap) = &self.charsmap {
            if let Some((length, offset)) = charsmap.longest_match(input) {
                if let Some(replacement) = charsmap.replacement(offset) {
                    return (replacement, length);
                }
            }
        }
        // Otherwise one character, as it is.
        let length = utf8_length(input[0]).min(input.len());
        match std::str::from_utf8(&input[..length]) {
            Ok(_) => (&input[..length], length),
            Err(_) => ("\u{FFFD}".as_bytes(), 1),
        }
    }

    /// Normalizes `input` and writes spaces as the model expects them.
    fn normalize(&self, input: &[u8]) -> Vec<u8> {
        let space: &[u8] = if self.escape_whitespaces {
            SPACE_MARK
        } else {
            b" "
        };
        let prepend = !self.treat_whitespace_as_suffix && self.add_space_prefix;
        let append = self.treat_whitespace_as_suffix && self.add_space_prefix;
        let merge = self.remove_extra_whitespaces;
        let mut out = Vec::with_capacity(input.len() * 3);
        let mut space_prepended = false;
        let mut in_word = false;
        let mut offset = 0;
        while offset < input.len() {
            let (normalized, consumed) = self.normalize_prefix(&input[offset..]);
            for &byte in normalized {
                if byte != b' ' {
                    if !in_word {
                        in_word = true;
                        if (prepend && !space_prepended) || merge {
                            out.extend_from_slice(space);
                            space_prepended = true;
                        }
                    }
                    out.push(byte);
                } else {
                    in_word = false;
                    if !merge {
                        out.extend_from_slice(space);
                    }
                }
            }
            offset += consumed.max(1);
        }
        if append {
            out.extend_from_slice(space);
        }
        out
    }
}

/// The length of a UTF-8 sequence from its first byte, as llama.cpp
/// reckons it: a stray continuation byte counts as one.
fn utf8_length(first: u8) -> usize {
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4][(first >> 4) as usize]
}

/// The best way found so far to tokenize the text up to a position.
#[derive(Clone, Copy)]
struct Best {
    token: u32,
    /// Where the last token starts.
    start: usize,
    score: f64,
}

impl Tokenizer {
    /// Appends the tokens of `text` to `out`.
    pub(super) fn encode_unigram(&self, unigram: &Unigram, text: &[u8], out: &mut Vec<u32>) {
        let normalized = unigram.normalize(text);
        let len = normalized.len();
        if len == 0 {
            return;
        }
        let unk = self.unk.unwrap_or(0);
        let mut best = vec![
            Best {
                token: unk,
                start: 0,
                score: f64::MIN,
            };
            len + 1
        ];
        best[0].score = 0.0;
        let usable = attr::NORMAL | attr::USER_DEFINED | attr::UNUSED;

        let mut offset = 0;
        while offset < len {
            let char_len = utf8_length(normalized[offset]).min(len - offset);
            let here = best[offset];
            let mut covers_char = false;
            let longest = (len - offset).min(self.max_token_len);
            for end in offset + 1..=offset + longest {
                let Some(id) = self.find(&normalized[offset..end]) else {
                    continue;
                };
                let flags = self.attrs[id as usize];
                if flags & usable == 0 {
                    continue;
                }
                if end - offset == char_len {
                    covers_char = true;
                }
                // User-defined tokens score zero, above every learned
                // score, so they are always chosen.
                let score = if flags & attr::USER_DEFINED != 0 {
                    0.0
                } else {
                    self.scores.get(id as usize).copied().unwrap_or(0.0) as f64
                };
                let challenger = here.score + score;
                if challenger > best[end].score {
                    best[end] = Best {
                        token: id,
                        start: offset,
                        score: challenger,
                    };
                }
            }
            if !covers_char {
                let end = offset + char_len;
                let challenger = here.score + unigram.unknown_score;
                if challenger > best[end].score {
                    best[end] = Best {
                        token: unk,
                        start: offset,
                        score: challenger,
                    };
                }
            }
            offset += char_len;
        }

        // Back from the end, joining runs of unknown tokens into one.
        let first = out.len();
        let mut at = len;
        let mut previous_unknown = false;
        loop {
            let step = best[at];
            let unknown = step.token == unk;
            if !(previous_unknown && unknown) {
                out.push(step.token);
            }
            if step.start == 0 {
                break;
            }
            previous_unknown = unknown;
            at = step.start;
        }
        out[first..].reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_lengths_follow_the_first_byte() {
        assert_eq!(utf8_length(b'a'), 1);
        assert_eq!(utf8_length(0xC3), 2);
        assert_eq!(utf8_length(0xE2), 3);
        assert_eq!(utf8_length(0xF0), 4);
        assert_eq!(utf8_length(0x80), 1);
    }
}

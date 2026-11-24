//! The RWKV World tokenizer: at each position, the longest token that the
//! text starts with.
//!
//! An RWKV World vocabulary holds arbitrary bytes, not text. The file spells
//! each token the way Python writes a byte string: a newline is `\n`, a
//! byte outside printable ASCII is `\xNN`, and a backslash is `\\`. So every
//! spelling with a backslash stands for other bytes, and every spelling
//! without one stands for itself.
//!
//! Tokenizing takes the longest token whose bytes start the rest of the
//! text, from left to right, as llama.cpp's trie does. Every token takes
//! part, control tokens included, so `<s>` in a text is token 0 whether or
//! not special tokens are parsed.
//!
//! The tokens that spell themselves are found through the vocabulary's own
//! index. Only the others, about a quarter of the vocabulary, get a table
//! of their own here.

use super::Tokenizer;
use super::vocab::{HASH_SEED, hash_more};

/// The tokens whose spelling has escapes, by the bytes they stand for.
pub(super) struct Escaped {
    /// The bytes of every escaped token, one token after another.
    bytes: Vec<u8>,
    /// For each escaped token: its id, and where its bytes end in `bytes`.
    entries: Vec<(u32, u32)>,
    /// Open-addressing hash table over `entries`: an entry index plus one,
    /// zero for an empty slot.
    slots: Vec<u32>,
    /// For each byte, the length of the longest token that starts with it.
    /// This bounds the search for a match.
    longest: [u16; 256],
}

impl Escaped {
    /// Reads the spelling of every token. `spellings` yields them in id
    /// order.
    pub fn new<'a>(spellings: impl Iterator<Item = &'a [u8]>) -> Escaped {
        let mut bytes = Vec::new();
        let mut entries = Vec::new();
        let mut longest = [0u16; 256];
        let mut unescaped = Vec::new();
        for (id, spelling) in spellings.enumerate() {
            let token: &[u8] = if spelling.contains(&b'\\') {
                unescaped.clear();
                unescape(spelling, &mut unescaped);
                bytes.extend_from_slice(&unescaped);
                entries.push((id as u32, bytes.len() as u32));
                &unescaped
            } else {
                spelling
            };
            if let Some(&first) = token.first() {
                let len = token.len().min(u16::MAX as usize) as u16;
                longest[first as usize] = longest[first as usize].max(len);
            }
        }
        // A load factor of at most two thirds keeps probe runs short.
        let capacity = (entries.len() * 3 / 2).next_power_of_two().max(16);
        let mut table = Escaped {
            bytes,
            entries,
            slots: vec![0; capacity],
            longest,
        };
        for index in 0..table.entries.len() {
            let key = table.entry(index);
            if key.is_empty() {
                continue;
            }
            let mask = table.slots.len() - 1;
            let mut slot = hash_more(HASH_SEED, key) as usize & mask;
            loop {
                match table.slots[slot] {
                    0 => {
                        table.slots[slot] = index as u32 + 1;
                        break;
                    }
                    // The same bytes twice: the later token wins, as in
                    // llama.cpp's trie.
                    existing if table.entry(existing as usize - 1) == key => {
                        table.slots[slot] = index as u32 + 1;
                        break;
                    }
                    _ => slot = (slot + 1) & mask,
                }
            }
        }
        table
    }

    /// The bytes of entry `index`.
    fn entry(&self, index: usize) -> &[u8] {
        let start = match index {
            0 => 0,
            _ => self.entries[index - 1].1 as usize,
        };
        &self.bytes[start..self.entries[index].1 as usize]
    }

    /// The escaped token that stands for `bytes`, whose hash is `hash`.
    fn find(&self, bytes: &[u8], hash: u64) -> Option<u32> {
        let mask = self.slots.len() - 1;
        let mut slot = hash as usize & mask;
        loop {
            match self.slots[slot] {
                0 => return None,
                value if self.entry(value as usize - 1) == bytes => {
                    return Some(self.entries[value as usize - 1].0);
                }
                _ => slot = (slot + 1) & mask,
            }
        }
    }
}

/// Appends the bytes a token spelling stands for to `out`, exactly as
/// llama.cpp's `llama_unescape_rwkv_token` reads it. A `\x` takes the next
/// two characters as hex digits, lowercase as Python writes them. Any other
/// escaped character stands for itself.
pub(super) fn unescape(spelling: &[u8], out: &mut Vec<u8>) {
    let mut escaping = false;
    let mut hex_left = 0;
    let mut hex = 0u8;
    for &c in spelling {
        if hex_left > 0 {
            // The digit arithmetic wraps like llama.cpp's on bytes.
            let digit = if c >= b'a' {
                c.wrapping_sub(b'a' - 10)
            } else {
                c.wrapping_sub(b'0')
            };
            hex = (hex << 4).wrapping_add(digit);
            hex_left -= 1;
            if hex_left == 0 {
                out.push(hex);
                hex = 0;
            }
        } else if escaping {
            match c {
                b't' => out.push(b'\t'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b'x' => hex_left = 2,
                other => out.push(other),
            }
            escaping = false;
        } else if c == b'\\' {
            escaping = true;
        } else {
            out.push(c);
        }
    }
}

impl Tokenizer {
    /// The token that stands for `bytes`. `hash` is their hash and
    /// `has_backslash` says whether they hold a backslash. When escaped and
    /// plain spellings stand for the same bytes, the later token wins, as
    /// in llama.cpp's trie.
    fn find_rwkv(
        &self,
        escaped: &Escaped,
        bytes: &[u8],
        hash: u64,
        has_backslash: bool,
    ) -> Option<u32> {
        let from_escaped = escaped.find(bytes, hash);
        // A plain spelling has no backslash, since a backslash is itself
        // escaped. So bytes with one can only be an escaped token.
        let plain = if has_backslash {
            None
        } else {
            self.token_index
                .find(&self.tokens, hash, |candidate| candidate == bytes)
        };
        from_escaped.max(plain)
    }

    /// Appends the tokens of `text` to `out`: at each position, the longest
    /// token that the text starts with. A byte that no token starts with
    /// becomes the unknown token, when the model has one.
    pub(super) fn encode_rwkv(&self, escaped: &Escaped, text: &[u8], out: &mut Vec<u32>) {
        let mut at = 0;
        while at < text.len() {
            let limit = escaped.longest[text[at] as usize] as usize;
            let end_limit = text.len().min(at + limit);
            // The hash of each longer prefix follows from the one before.
            let mut hash = HASH_SEED;
            let mut has_backslash = false;
            let mut found = None;
            for end in at + 1..=end_limit {
                let byte = text[end - 1];
                hash = hash_more(hash, &[byte]);
                has_backslash |= byte == b'\\';
                if let Some(id) = self.find_rwkv(escaped, &text[at..end], hash, has_backslash) {
                    found = Some((id, end));
                }
            }
            match found {
                Some((id, end)) => {
                    out.push(id);
                    at = end;
                }
                None => {
                    out.extend(self.unk);
                    at += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unescaped(spelling: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        unescape(spelling, &mut out);
        out
    }

    #[test]
    fn spellings_unescape_as_python_writes_them() {
        assert_eq!(unescaped(b"plain"), b"plain");
        assert_eq!(unescaped(b"\\n\\n"), b"\n\n");
        assert_eq!(unescaped(b"a\\tb\\r"), b"a\tb\r");
        assert_eq!(unescaped(b"\\x00\\xff\\xc3\\xa9"), b"\x00\xff\xc3\xa9");
        assert_eq!(unescaped(b"\\\\"), b"\\");
        assert_eq!(unescaped(b"it\\'s"), b"it's");
        // An escape cut short at the end is dropped.
        assert_eq!(unescaped(b"a\\"), b"a");
        assert_eq!(unescaped(b"a\\x4"), b"a");
    }

    #[test]
    fn escaped_tokens_are_found_by_their_bytes() {
        let spellings: [&[u8]; 5] = [b"<s>", b"\\n", b"a", b"\\x00", b"\\n"];
        let table = Escaped::new(spellings.into_iter());
        let find = |bytes: &[u8]| table.find(bytes, hash_more(HASH_SEED, bytes));
        assert_eq!(find(b"\x00"), Some(3));
        // Two spellings of a newline: the later one wins.
        assert_eq!(find(b"\n"), Some(4));
        // Plain spellings are not in this table.
        assert_eq!(find(b"a"), None);
        assert_eq!(table.longest[b'<' as usize], 3);
        assert_eq!(table.longest[b'\n' as usize], 1);
        assert_eq!(table.longest[b'b' as usize], 0);
    }
}

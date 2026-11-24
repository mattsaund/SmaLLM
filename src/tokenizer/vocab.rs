//! String tables that stay in the model file.
//!
//! A vocabulary can hold a quarter of a million tokens, and a BPE model as
//! many merge rules. Copying them into owned strings would cost megabytes
//! and most of the load time. A [`StringTable`] instead records where each
//! string sits in the file mapping, and an [`Index`] finds a string by
//! content with an open-addressing hash table of 4-byte slots.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gguf::{Array, Mapping, ValueType};

/// The strings of one GGUF string array, addressed by index.
pub(super) struct StringTable {
    map: Arc<Mapping>,
    /// File offset of the text of each string. Its 64-bit length sits in the
    /// eight bytes before it.
    offsets: Vec<u32>,
    /// File offset the entries of `offsets` are relative to.
    base: usize,
}

impl StringTable {
    /// Records the position of every string in `array`.
    pub fn new(map: &Arc<Mapping>, array: &Array<'_>, what: &str) -> Result<StringTable> {
        if array.elem_type() != ValueType::String {
            return Err(Error::format(format!("{what} is not an array of strings")));
        }
        if array.raw().len() > u32::MAX as usize {
            return Err(Error::unsupported(format!("{what} larger than 4 GiB")));
        }
        let mut offsets = Vec::with_capacity(array.len());
        let mut strings = array.strings();
        loop {
            let offset = strings.offset();
            if strings.next().is_none() {
                break;
            }
            offsets.push(offset as u32);
        }
        if offsets.len() != array.len() {
            return Err(Error::format(format!("{what} is truncated")));
        }
        Ok(StringTable {
            map: Arc::clone(map),
            offsets,
            base: array.file_offset(),
        })
    }

    /// Number of strings.
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// The bytes of string `index`. Empty when the index is out of range.
    pub fn get(&self, index: usize) -> &[u8] {
        let Some(&offset) = self.offsets.get(index) else {
            return &[];
        };
        let data = self.map.as_slice();
        let start = self.base + offset as usize;
        // The array was walked when the file was parsed, so these bounds
        // hold. A mismatch yields an empty string instead of a panic.
        let Some(len_bytes) = data.get(start - 8..start) else {
            return &[];
        };
        let len = u64::from_le_bytes(len_bytes.try_into().expect("eight bytes")) as usize;
        data.get(start..start.saturating_add(len)).unwrap_or(&[])
    }
}

/// Starting state of the string hash.
pub(super) const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// Feeds `bytes` into an FNV-1a hash state.
pub(super) fn hash_more(mut state: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        state ^= byte as u64;
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}

/// A hash index over a [`StringTable`]: finds the index of a string by
/// content.
pub(super) struct Index {
    /// Slot values are string index plus one. Zero marks an empty slot.
    slots: Vec<u32>,
    mask: usize,
}

impl Index {
    /// Indexes every non-empty string of `table`. When the same string
    /// appears twice, `keep_first` decides which index a lookup returns.
    pub fn build(table: &StringTable, keep_first: bool) -> Index {
        let capacity = (table.len() * 2).next_power_of_two().max(16);
        let mut index = Index {
            slots: vec![0; capacity],
            mask: capacity - 1,
        };
        for id in 0..table.len() {
            let text = table.get(id);
            if text.is_empty() {
                continue;
            }
            let mut slot = hash_more(HASH_SEED, text) as usize & index.mask;
            loop {
                match index.slots[slot] {
                    0 => {
                        index.slots[slot] = id as u32 + 1;
                        break;
                    }
                    existing if table.get(existing as usize - 1) == text => {
                        if !keep_first {
                            index.slots[slot] = id as u32 + 1;
                        }
                        break;
                    }
                    _ => slot = (slot + 1) & index.mask,
                }
            }
        }
        index
    }

    /// Finds the string with hash `hash` for which `matches` returns true.
    pub fn find(
        &self,
        table: &StringTable,
        hash: u64,
        matches: impl Fn(&[u8]) -> bool,
    ) -> Option<u32> {
        let mut slot = hash as usize & self.mask;
        loop {
            match self.slots[slot] {
                0 => return None,
                value if matches(table.get(value as usize - 1)) => return Some(value - 1),
                _ => slot = (slot + 1) & self.mask,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use crate::gguf::testing::Builder;

    fn table(strings: &[&[u8]]) -> StringTable {
        let mut builder = Builder::new();
        builder.strings("list", strings);
        let file = Gguf::from_bytes(builder.build()).unwrap();
        let array = file.get_array("list").unwrap();
        StringTable::new(file.mapping(), &array, "list").unwrap()
    }

    #[test]
    fn strings_are_read_from_the_file() {
        let strings = table(&[b"alpha", b"", b"\xe2\x96\x81b", b"z"]);
        assert_eq!(strings.len(), 4);
        assert_eq!(strings.get(0), b"alpha");
        assert_eq!(strings.get(1), b"");
        assert_eq!(strings.get(2), "\u{2581}b".as_bytes());
        assert_eq!(strings.get(3), b"z");
        assert_eq!(strings.get(4), b"");
    }

    #[test]
    fn lookups_find_every_string() {
        let words: Vec<String> = (0..5000).map(|i| format!("token{i}")).collect();
        let refs: Vec<&[u8]> = words.iter().map(|w| w.as_bytes()).collect();
        let strings = table(&refs);
        let index = Index::build(&strings, false);
        for (id, word) in words.iter().enumerate() {
            let found = index.find(&strings, hash_more(HASH_SEED, word.as_bytes()), |s| {
                s == word.as_bytes()
            });
            assert_eq!(found, Some(id as u32));
        }
        assert_eq!(
            index.find(&strings, hash_more(HASH_SEED, b"missing"), |s| s
                == b"missing"),
            None
        );
    }

    #[test]
    fn duplicates_resolve_as_requested() {
        let strings = table(&[b"a", b"b", b"a"]);
        let last = Index::build(&strings, false);
        let first = Index::build(&strings, true);
        let hash = hash_more(HASH_SEED, b"a");
        assert_eq!(last.find(&strings, hash, |s| s == b"a"), Some(2));
        assert_eq!(first.find(&strings, hash, |s| s == b"a"), Some(0));
    }

    #[test]
    fn hashing_in_parts_equals_hashing_at_once() {
        let whole = hash_more(HASH_SEED, b"left right");
        let parts = hash_more(hash_more(hash_more(HASH_SEED, b"left"), b" "), b"right");
        assert_eq!(whole, parts);
    }

    #[test]
    fn other_array_types_are_rejected() {
        let mut builder = Builder::new();
        builder.f32s("list", &[1.0]);
        let file = Gguf::from_bytes(builder.build()).unwrap();
        let array = file.get_array("list").unwrap();
        assert!(StringTable::new(file.mapping(), &array, "list").is_err());
    }
}

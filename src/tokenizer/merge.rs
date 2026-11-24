//! The pair merging loop shared by both tokenizer kinds.
//!
//! Tokenizing a word starts from its characters and repeatedly joins the
//! best adjacent pair until no pair can be joined. SentencePiece ranks a
//! pair by the score of the token it would form, and BPE by the position of
//! the pair in the model's merge list. The loop is the same either way.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Marks the end of the symbol chain.
const NONE: usize = usize::MAX;

/// A piece of the text: a byte range linked to its neighbors. A symbol that
/// was merged into its left neighbor has length zero.
#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    len: usize,
    prev: usize,
    next: usize,
}

/// A candidate merge of two adjacent symbols. Ordered so the heap yields the
/// highest priority first and, among equals, the leftmost pair.
type Pair = (u32, Reverse<usize>, usize, usize);

/// Merges adjacent symbols of `text` until no pair can merge.
///
/// `lengths` yields the byte length of each starting symbol and must cover
/// `text` exactly. `priority` receives the bytes of an adjacent pair and the
/// length of its left part, and returns `None` when the pair may not merge.
/// The final symbols are written to `out` as `(start, length)` byte ranges.
pub(super) fn merge(
    text: &[u8],
    lengths: impl Iterator<Item = usize>,
    mut priority: impl FnMut(&[u8], usize) -> Option<u32>,
    out: &mut Vec<(usize, usize)>,
) {
    out.clear();
    let mut symbols: Vec<Symbol> = Vec::new();
    let mut start = 0;
    for len in lengths {
        let index = symbols.len();
        symbols.push(Symbol {
            start,
            len,
            prev: index.wrapping_sub(1),
            next: index + 1,
        });
        start += len;
    }
    debug_assert_eq!(start, text.len());
    let Some(last) = symbols.last_mut() else {
        return;
    };
    last.next = NONE;

    // Queues the pair `(left, right)` when the model allows it to merge.
    let mut queue: BinaryHeap<Pair> = BinaryHeap::new();
    let mut consider =
        |symbols: &[Symbol], queue: &mut BinaryHeap<Pair>, left: usize, right: usize| {
            if left == NONE || right == NONE {
                return;
            }
            let (l, r) = (symbols[left], symbols[right]);
            let bytes = &text[l.start..r.start + r.len];
            if let Some(key) = priority(bytes, l.len) {
                queue.push((key, Reverse(left), right, bytes.len()));
            }
        };
    for right in 1..symbols.len() {
        consider(&symbols, &mut queue, right - 1, right);
    }

    while let Some((_, Reverse(left), right, size)) = queue.pop() {
        let (l, r) = (symbols[left], symbols[right]);
        // A queued pair is stale once either side has merged elsewhere:
        // symbols only ever grow or vanish, so the combined size differs.
        if l.len == 0 || r.len == 0 || l.len + r.len != size {
            continue;
        }
        symbols[left].len += r.len;
        symbols[left].next = r.next;
        symbols[right].len = 0;
        if r.next != NONE {
            symbols[r.next].prev = left;
        }
        consider(&symbols, &mut queue, symbols[left].prev, left);
        consider(&symbols, &mut queue, left, symbols[left].next);
    }

    // The first symbol is never merged away, so the chain starts there.
    let mut index = 0;
    while index != NONE {
        let symbol = symbols[index];
        out.push((symbol.start, symbol.len));
        index = symbol.next;
    }
}

/// Byte length of the UTF-8 sequence that starts with `lead`.
pub(super) fn utf8_len(lead: u8) -> usize {
    match lead >> 4 {
        0xc | 0xd => 2,
        0xe => 3,
        0xf => 4,
        _ => 1,
    }
}

/// Yields the byte length of each character of `text`. A truncated final
/// sequence is clipped to the bytes that exist.
pub(super) fn char_lengths(text: &[u8]) -> impl Iterator<Item = usize> + '_ {
    let mut pos = 0;
    std::iter::from_fn(move || {
        if pos >= text.len() {
            return None;
        }
        let len = utf8_len(text[pos]).min(text.len() - pos);
        pos += len;
        Some(len)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Merges with a rule table: pairs listed earlier merge first.
    fn run(text: &str, rules: &[(&str, &str)]) -> Vec<String> {
        let mut out = Vec::new();
        let priority = |bytes: &[u8], split: usize| {
            rules
                .iter()
                .position(|(l, r)| {
                    l.as_bytes() == &bytes[..split] && r.as_bytes() == &bytes[split..]
                })
                .map(|rank| u32::MAX - rank as u32)
        };
        merge(
            text.as_bytes(),
            char_lengths(text.as_bytes()),
            priority,
            &mut out,
        );
        out.iter()
            .map(|&(start, len)| text[start..start + len].to_string())
            .collect()
    }

    #[test]
    fn merges_follow_rank_order() {
        let rules = [("l", "l"), ("h", "e"), ("he", "ll"), ("hell", "o")];
        assert_eq!(run("hello", &rules), ["hello"]);
        assert_eq!(run("hell", &rules), ["hell"]);
        assert_eq!(run("help", &rules), ["he", "l", "p"]);
    }

    #[test]
    fn equal_ranks_merge_left_to_right() {
        assert_eq!(run("aaa", &[("a", "a")]), ["aa", "a"]);
        assert_eq!(run("aaaa", &[("a", "a")]), ["aa", "aa"]);
        assert_eq!(run("aaaa", &[("a", "a"), ("aa", "aa")]), ["aaaa"]);
    }

    #[test]
    fn text_without_rules_stays_as_characters() {
        assert_eq!(run("héy", &[]), ["h", "é", "y"]);
        assert!(run("", &[("a", "b")]).is_empty());
        assert_eq!(run("x", &[("x", "x")]), ["x"]);
    }

    #[test]
    fn character_lengths_follow_utf8() {
        let lengths: Vec<usize> = char_lengths("a\u{e9}\u{4e2d}\u{1f600}".as_bytes()).collect();
        assert_eq!(lengths, [1, 2, 3, 4]);
        // A sequence cut short is clipped, not read past the end.
        let lengths: Vec<usize> = char_lengths(&[b'a', 0xe4, 0xb8]).collect();
        assert_eq!(lengths, [1, 2]);
    }

    #[test]
    fn long_runs_finish_quickly() {
        let text = "a".repeat(200_000);
        let pieces = run(&text, &[("a", "a")]);
        assert_eq!(pieces.len(), 100_000);
    }
}

//! Splits text into words before byte pair merging.
//!
//! BPE models never merge across word boundaries, and each model family
//! defines "word" with its own regular expression. A regex engine would be
//! the largest piece of code in the tokenizer, so the handful of expressions
//! in use are written out here as small scanners.
//!
//! A pre-tokenizer is a list of stages. Each stage splits every piece left
//! by the stage before it.

use super::unicode::{is_letter, is_number, is_other, is_punctuation, is_whitespace};

/// One splitting pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    /// `[\p{P}$+<=>^~|]+` Runs of punctuation and a few symbols.
    PunctuationRuns,
    /// `\p{N}` Every number character on its own.
    EachNumber,
    /// `\p{N}+` Runs of numbers.
    NumberRuns,
    /// `[0-9][0-9][0-9]` Groups of exactly three ASCII digits.
    ThreeDigits,
    /// The GPT-2 expression:
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    Gpt2,
    /// The Llama 3 expression, with numbers in groups of up to `max_digits`:
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|`
    /// ` ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    Llama3 { max_digits: usize },
}

/// How a named pre-tokenizer splits and merges.
#[derive(Clone, Copy, Debug)]
pub(super) struct PreTokenizer {
    /// The splitting passes, in order.
    pub stages: &'static [Stage],
    /// A word that is itself a token is emitted whole, without merging.
    pub ignore_merges: bool,
    /// The family always starts a prompt with the BOS token.
    pub add_bos: bool,
    /// False when the name was not recognized and the default is used.
    pub known: bool,
}

const GPT2: &[Stage] = &[Stage::Gpt2];
const SMOLLM: &[Stage] = &[Stage::EachNumber, Stage::Gpt2];
const LLAMA3: &[Stage] = &[Stage::Llama3 { max_digits: 3 }];
const QWEN2: &[Stage] = &[Stage::Llama3 { max_digits: 1 }];
const DEFAULT: &[Stage] = &[
    Stage::PunctuationRuns,
    Stage::Gpt2,
    Stage::NumberRuns,
    Stage::ThreeDigits,
];

impl PreTokenizer {
    /// Looks up a pre-tokenizer by its `tokenizer.ggml.pre` name.
    pub fn for_name(name: &str) -> PreTokenizer {
        let plain = |stages| PreTokenizer {
            stages,
            ignore_merges: false,
            add_bos: false,
            known: true,
        };
        match name {
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0"
            | "jina-v5-nano" | "lfm2" => PreTokenizer {
                stages: LLAMA3,
                ignore_merges: true,
                add_bos: true,
                known: true,
            },
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2"
            | "hunyuan" => plain(QWEN2),
            "smollm" | "starcoder" | "refact" | "command-r" | "codeshell" | "exaone"
            | "minerva-7b" => plain(SMOLLM),
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
            | "jina-v2-de" | "a.x-4.0" | "mellum" | "mpt" | "olmo" | "jais" | "trillion"
            | "exaone4" => plain(GPT2),
            "default" => plain(DEFAULT),
            _ => PreTokenizer {
                stages: DEFAULT,
                ignore_merges: false,
                add_bos: false,
                known: false,
            },
        }
    }
}

/// Splits `text` into words. Returns the byte offset where each word ends,
/// in order. The words cover the text exactly.
pub(super) fn split(stages: &[Stage], text: &str) -> Vec<usize> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    // Word boundaries as character indices. Each stage refines them.
    let mut ends = vec![chars.len()];
    for stage in stages {
        let mut refined = Vec::with_capacity(ends.len() * 2);
        let mut start = 0;
        for &end in &ends {
            let mut words = Words {
                ends: &mut refined,
                base: start,
            };
            let piece = &chars[start..end];
            match *stage {
                Stage::PunctuationRuns => split_runs(piece, &mut words, is_punctuation_or_symbol),
                Stage::EachNumber => split_each(piece, &mut words, is_number),
                Stage::NumberRuns => split_runs(piece, &mut words, is_number),
                Stage::ThreeDigits => split_three_digits(piece, &mut words),
                Stage::Gpt2 => split_gpt2(piece, &mut words),
                Stage::Llama3 { max_digits } => split_llama3(piece, &mut words, max_digits),
            }
            start = end;
        }
        ends = refined;
    }
    // Convert character indices to byte offsets.
    let mut byte_ends = Vec::with_capacity(ends.len());
    let mut next = ends.iter().copied().peekable();
    let mut offset = 0;
    for (index, c) in chars.iter().enumerate() {
        offset += c.len_utf8();
        if next.peek() == Some(&(index + 1)) {
            byte_ends.push(offset);
            next.next();
        }
    }
    byte_ends
}

/// Collects the word boundaries produced while scanning one piece.
struct Words<'a> {
    ends: &'a mut Vec<usize>,
    /// Character index of the piece within the whole text.
    base: usize,
}

impl Words<'_> {
    /// Ends a word at character `pos` of the piece. Empty words are dropped.
    fn end_at(&mut self, pos: usize) {
        let end = self.base + pos;
        if end > self.base && self.ends.last() != Some(&end) {
            self.ends.push(end);
        }
    }
}

/// The characters of the default pre-tokenizer's first expression.
fn is_punctuation_or_symbol(c: char) -> bool {
    matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '~' | '|') || is_punctuation(c)
}

/// Makes each run of matching characters a word, and each run between them.
fn split_runs(piece: &[char], words: &mut Words<'_>, matches: fn(char) -> bool) {
    for pos in 1..piece.len() {
        if matches(piece[pos]) != matches(piece[pos - 1]) {
            words.end_at(pos);
        }
    }
    words.end_at(piece.len());
}

/// Makes each matching character a word of its own.
fn split_each(piece: &[char], words: &mut Words<'_>, matches: fn(char) -> bool) {
    for (pos, &c) in piece.iter().enumerate() {
        if matches(c) {
            words.end_at(pos);
            words.end_at(pos + 1);
        }
    }
    words.end_at(piece.len());
}

/// Makes each group of three ASCII digits a word, scanning left to right.
fn split_three_digits(piece: &[char], words: &mut Words<'_>) {
    let mut pos = 0;
    while pos + 3 <= piece.len() {
        if piece[pos..pos + 3].iter().all(char::is_ascii_digit) {
            words.end_at(pos);
            words.end_at(pos + 3);
            pos += 3;
        } else {
            pos += 1;
        }
    }
    words.end_at(piece.len());
}

/// Length of an English contraction suffix at `pos`: `'s`, `'t`, `'m`, `'d`,
/// `'re`, `'ve` or `'ll`. Zero when there is none.
fn contraction(piece: &[char], pos: usize, any_case: bool) -> usize {
    let fold = |c: char| if any_case { c.to_ascii_lowercase() } else { c };
    if piece[pos] != '\'' || pos + 1 >= piece.len() {
        return 0;
    }
    let first = fold(piece[pos + 1]);
    if matches!(first, 's' | 't' | 'm' | 'd') {
        return 2;
    }
    if pos + 2 < piece.len()
        && matches!(
            (first, fold(piece[pos + 2])),
            ('r', 'e') | ('v', 'e') | ('l', 'l')
        )
    {
        return 3;
    }
    0
}

/// Index just past the run of characters matching `class` that starts at
/// `pos`.
fn run_end(piece: &[char], mut pos: usize, class: fn(char) -> bool) -> usize {
    while pos < piece.len() && class(piece[pos]) {
        pos += 1;
    }
    pos
}

/// Handles the whitespace alternatives shared by both expressions:
/// `\s+(?!\S)` leaves the last space of a run to prefix the next word, and
/// `\s+` takes a run that ends the text. Returns the new position, or `pos`
/// when there is no whitespace here.
fn whitespace_end(piece: &[char], pos: usize) -> usize {
    let end = run_end(piece, pos, is_whitespace);
    if end - pos > 1 && end < piece.len() {
        end - 1
    } else {
        end
    }
}

/// Splits by the GPT-2 expression.
fn split_gpt2(piece: &[char], words: &mut Words<'_>) {
    let mut pos = 0;
    while pos < piece.len() {
        let suffix = contraction(piece, pos, false);
        if suffix > 0 {
            pos += suffix;
            words.end_at(pos);
            continue;
        }
        // An optional single space, then a run of letters, of numbers, or of
        // anything else that is not whitespace.
        let lead = usize::from(piece[pos] == ' ');
        if let Some(&first) = piece.get(pos + lead) {
            let class: Option<fn(char) -> bool> = if is_letter(first) {
                Some(is_letter)
            } else if is_number(first) {
                Some(is_number)
            } else if !is_whitespace(first) {
                Some(is_other)
            } else {
                None
            };
            if let Some(class) = class {
                pos = run_end(piece, pos + lead, class);
                words.end_at(pos);
                continue;
            }
        }
        let after_whitespace = whitespace_end(piece, pos);
        pos = if after_whitespace > pos {
            after_whitespace
        } else {
            pos + 1
        };
        words.end_at(pos);
    }
}

/// Splits by the Llama 3 expression. Qwen uses the same expression with
/// `max_digits` of 1.
fn split_llama3(piece: &[char], words: &mut Words<'_>, max_digits: usize) {
    let is_newline = |c: char| c == '\r' || c == '\n';
    let mut pos = 0;
    while pos < piece.len() {
        let c = piece[pos];
        let suffix = contraction(piece, pos, true);
        if suffix > 0 {
            pos += suffix;
            words.end_at(pos);
            continue;
        }
        // `[^\r\n\p{L}\p{N}]?\p{L}+` Letters, with at most one leading
        // character of another kind.
        if !is_newline(c) && !is_number(c) {
            let next_is_letter = piece.get(pos + 1).is_some_and(|&next| is_letter(next));
            if is_letter(c) || next_is_letter {
                pos = run_end(piece, pos + 1, is_letter);
                words.end_at(pos);
                continue;
            }
        }
        // `\p{N}{1,3}` Numbers, in groups.
        if is_number(c) {
            let end = run_end(piece, pos, is_number);
            while pos < end {
                pos = (pos + max_digits).min(end);
                words.end_at(pos);
            }
            continue;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*` Punctuation and symbols, with an
        // optional leading space and any trailing line breaks.
        let lead = usize::from(c == ' ');
        if piece.get(pos + lead).is_some_and(|&first| is_other(first)) {
            pos = run_end(piece, pos + lead, is_other);
            pos = run_end(piece, pos, is_newline);
            words.end_at(pos);
            continue;
        }
        // `\s*[\r\n]+` Whitespace up to and including its last line break.
        let whitespace = run_end(piece, pos, is_whitespace);
        if let Some(last_newline) = (pos..whitespace).rev().find(|&i| is_newline(piece[i])) {
            pos = last_newline + 1;
            words.end_at(pos);
            continue;
        }
        let after_whitespace = whitespace_end(piece, pos);
        pos = if after_whitespace > pos {
            after_whitespace
        } else {
            pos + 1
        };
        words.end_at(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splits `text` and returns the words as strings.
    fn words(stages: &[Stage], text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        for end in split(stages, text) {
            out.push(text[start..end].to_string());
            start = end;
        }
        assert_eq!(start, text.len(), "words must cover the text");
        out
    }

    #[test]
    fn gpt2_keeps_one_leading_space_with_each_word() {
        assert_eq!(words(GPT2, "Hello world"), ["Hello", " world"]);
        assert_eq!(words(GPT2, "a  b"), ["a", " ", " b"]);
        assert_eq!(words(GPT2, "end   "), ["end", "   "]);
        assert_eq!(words(GPT2, " x"), [" x"]);
        assert_eq!(words(GPT2, "it's 42!"), ["it", "'s", " 42", "!"]);
        assert_eq!(
            words(GPT2, "we'll they've I'm"),
            ["we", "'ll", " they", "'ve", " I", "'m"]
        );
        assert_eq!(words(GPT2, "It'S"), ["It", "'", "S"]);
        assert_eq!(words(GPT2, "a\nb"), ["a", "\n", "b"]);
        assert_eq!(words(GPT2, "x \n y"), ["x", " \n", " y"]);
        assert_eq!(words(GPT2, "héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(words(GPT2, "1+1=2"), ["1", "+", "1", "=", "2"]);
        assert_eq!(words(GPT2, "'"), ["'"]);
        assert!(words(GPT2, "").is_empty());
    }

    #[test]
    fn llama3_groups_digits_and_attaches_newlines() {
        assert_eq!(words(LLAMA3, "Hello world"), ["Hello", " world"]);
        assert_eq!(words(LLAMA3, "12345"), ["123", "45"]);
        assert_eq!(words(LLAMA3, "a 1234567"), ["a", " ", "123", "456", "7"]);
        assert_eq!(words(LLAMA3, "It'S we'LL"), ["It", "'S", " we", "'LL"]);
        assert_eq!(words(LLAMA3, "x!!\n\ny"), ["x", "!!\n\n", "y"]);
        assert_eq!(words(LLAMA3, "a \n\n b"), ["a", " \n\n", " b"]);
        assert_eq!(words(LLAMA3, "a   b"), ["a", "  ", " b"]);
        assert_eq!(words(LLAMA3, "\tfoo"), ["\tfoo"]);
        assert_eq!(words(LLAMA3, "(foo)"), ["(foo", ")"]);
        assert_eq!(words(LLAMA3, "a ."), ["a", " ."]);
        assert_eq!(words(LLAMA3, "tail "), ["tail", " "]);
        assert_eq!(words(LLAMA3, "\n\nx"), ["\n\n", "x"]);
        assert_eq!(words(LLAMA3, "\r\n"), ["\r\n"]);
    }

    #[test]
    fn qwen2_splits_every_digit() {
        assert_eq!(words(QWEN2, "x 2024"), ["x", " ", "2", "0", "2", "4"]);
        assert_eq!(words(QWEN2, "Hello, world!"), ["Hello", ",", " world", "!"]);
    }

    #[test]
    fn smollm_isolates_digits_before_the_gpt2_pass() {
        assert_eq!(words(SMOLLM, "abc 123"), ["abc", " ", "1", "2", "3"]);
        assert_eq!(words(SMOLLM, "v2 ok"), ["v", "2", " ok"]);
    }

    #[test]
    fn default_separates_punctuation_runs_and_digit_triples() {
        assert_eq!(words(DEFAULT, "Hi, there..."), ["Hi", ",", " there", "..."]);
        assert_eq!(words(DEFAULT, "a 12345"), ["a", " ", "123", "45"]);
        assert_eq!(words(DEFAULT, "x<=y"), ["x", "<=", "y"]);
    }

    #[test]
    fn names_map_to_families() {
        assert_eq!(PreTokenizer::for_name("llama-bpe").stages, LLAMA3);
        assert!(PreTokenizer::for_name("llama-bpe").ignore_merges);
        assert_eq!(PreTokenizer::for_name("qwen2").stages, QWEN2);
        assert_eq!(PreTokenizer::for_name("smollm").stages, SMOLLM);
        assert_eq!(PreTokenizer::for_name("gpt-2").stages, GPT2);
        assert!(PreTokenizer::for_name("default").known);
        assert!(!PreTokenizer::for_name("something-new").known);
    }

    #[test]
    fn every_split_covers_arbitrary_text() {
        let samples = [
            "The quick brown fox, 3.14159 times!\n\n\tTabs and  spaces   \r\n",
            "日本語のテキスト、123と456。 ¿Qué? «да» 😀👍 e\u{301}",
            "'''' 's't're  'x ''' ' ",
            "    \n    \n",
            "a\u{a0}b\u{3000}c\u{2028}d",
        ];
        for stages in [GPT2, SMOLLM, LLAMA3, QWEN2, DEFAULT] {
            for text in samples {
                let ends = split(stages, text);
                assert_eq!(ends.last().copied(), Some(text.len()));
                assert!(ends.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(ends.iter().all(|&end| text.is_char_boundary(end)));
            }
        }
    }
}

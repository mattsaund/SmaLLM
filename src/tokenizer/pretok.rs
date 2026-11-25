//! Splits text into words before byte pair merging.
//!
//! BPE models never merge across word boundaries, and each model family
//! defines "word" with its own regular expressions. The families and their
//! expressions are in `pretok_table.rs`, generated from llama.cpp's table, and
//! a model names its family in `tokenizer.ggml.pre`.
//!
//! A pre-tokenizer is a list of stages. Each stage splits every piece left
//! by the stage before it: the matches of its expression become pieces, and
//! so does the text between them.
//!
//! `regex.rs` runs any expression. The most common ones also have scanners
//! written out here, which do the same job several times faster. A test
//! holds each scanner to the engine.

use super::pretok_table::{DEFAULT_PATTERNS, FAMILIES};
use super::regex::Regex;
use super::unicode::{is_letter, is_number, is_other, is_punctuation, is_whitespace};
use crate::error::{Error, Result};

/// One family of the table in `pretok_table.rs`.
pub(super) struct Family {
    /// The `tokenizer.ggml.pre` names that mean this family.
    pub names: &'static [&'static str],
    /// The expressions of the stages, in order.
    pub patterns: &'static [&'static str],
    /// A word that is itself a token is emitted whole, without merging.
    pub ignore_merges: bool,
    /// The family always starts a prompt with the BOS token.
    pub add_bos: bool,
    /// Spaces become `\u{2581}` before splitting, as in SentencePiece.
    pub escape_spaces: bool,
    /// The family has no BOS token, whatever the file says.
    pub no_bos: bool,
    /// The vocabulary spells text as raw UTF-8 rather than in the GPT-2
    /// byte-level alphabet.
    pub raw_bytes: bool,
}

/// A splitting pass written out by hand, for an expression that is used
/// often. Each does exactly what its expression does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scan {
    /// `[\p{P}$+<=>^~|]+` Runs of punctuation and a few symbols.
    PunctuationRuns,
    /// `\p{N}` Every number character on its own.
    EachNumber,
    /// `\p{N}+` Runs of numbers.
    NumberRuns,
    /// `[0-9][0-9][0-9]` Groups of exactly three ASCII digits.
    ThreeDigits,
    /// `[^\n]+|[\n]+` Lines, and runs of line feeds.
    Newlines,
    /// The GPT-2 expression:
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
    Gpt2,
    /// The Llama 3 expression, with numbers in groups of up to `max_digits`:
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|`
    /// ` ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    Llama3 { max_digits: usize },
}

/// The expressions the scanners stand for, as the table spells them.
const SCANNED: [(&str, Scan); 8] = [
    (r"[\p{P}\$\+<=>\^~\|]+", Scan::PunctuationRuns),
    (r"\p{N}", Scan::EachNumber),
    (r"\p{N}+", Scan::NumberRuns),
    (r"[0-9][0-9][0-9]", Scan::ThreeDigits),
    (r"[^\n]+|[\n]+", Scan::Newlines),
    (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
        Scan::Gpt2,
    ),
    (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        Scan::Llama3 { max_digits: 3 },
    ),
    (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        Scan::Llama3 { max_digits: 1 },
    ),
];

/// One stage of a pre-tokenizer.
enum Stage {
    Scan(Scan),
    Regex(Box<Regex>),
}

impl Stage {
    /// The stage for an expression: its scanner, or else the expression
    /// compiled.
    fn new(pattern: &str) -> Result<Stage> {
        if let Some((_, scan)) = SCANNED.iter().find(|(scanned, _)| *scanned == pattern) {
            return Ok(Stage::Scan(*scan));
        }
        Regex::new(pattern)
            .map(|regex| Stage::Regex(Box::new(regex)))
            .map_err(|error| {
                Error::unsupported(format!("pre-tokenizer expression {pattern:?}: {error}"))
            })
    }
}

/// How a family splits text, and its switches.
pub(super) struct PreTokenizer {
    stages: Vec<Stage>,
    pub ignore_merges: bool,
    pub add_bos: bool,
    pub escape_spaces: bool,
    pub no_bos: bool,
    pub raw_bytes: bool,
    /// False when the name was not recognized and the default is used.
    pub known: bool,
}

impl PreTokenizer {
    /// Looks up a family by its `tokenizer.ggml.pre` name. An empty or
    /// unknown name gets the default expressions, as in llama.cpp.
    pub fn for_name(name: &str) -> Result<PreTokenizer> {
        let family = FAMILIES.iter().find(|family| family.names.contains(&name));
        let patterns = family.map_or(DEFAULT_PATTERNS, |family| family.patterns);
        Ok(PreTokenizer {
            stages: patterns
                .iter()
                .map(|pattern| Stage::new(pattern))
                .collect::<Result<_>>()?,
            ignore_merges: family.is_some_and(|family| family.ignore_merges),
            add_bos: family.is_some_and(|family| family.add_bos),
            escape_spaces: family.is_some_and(|family| family.escape_spaces),
            no_bos: family.is_some_and(|family| family.no_bos),
            raw_bytes: family.is_some_and(|family| family.raw_bytes),
            known: family.is_some() || name == "default",
        })
    }

    /// Splits `text` into words. Returns the byte offset where each word
    /// ends, in order. The words cover the text exactly.
    pub fn split(&self, text: &str) -> Vec<usize> {
        split_with(&self.stages, text)
    }
}

/// Splits `text` with the given stages. See [`PreTokenizer::split`].
fn split_with(stages: &[Stage], text: &str) -> Vec<usize> {
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
            let piece = &chars[start..end];
            match stage {
                Stage::Regex(regex) => regex.split(piece, start, &mut refined),
                Stage::Scan(scan) => {
                    let mut words = Words {
                        ends: &mut refined,
                        base: start,
                    };
                    match *scan {
                        Scan::PunctuationRuns => {
                            split_runs(piece, &mut words, is_punctuation_or_symbol)
                        }
                        Scan::EachNumber => split_each(piece, &mut words, is_number),
                        Scan::NumberRuns => split_runs(piece, &mut words, is_number),
                        Scan::ThreeDigits => split_three_digits(piece, &mut words),
                        Scan::Newlines => split_runs(piece, &mut words, |c| c == '\n'),
                        Scan::Gpt2 => split_gpt2(piece, &mut words),
                        Scan::Llama3 { max_digits } => split_llama3(piece, &mut words, max_digits),
                    }
                }
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

    /// The stages for a list of expressions.
    fn stages(patterns: &[&str]) -> Vec<Stage> {
        patterns
            .iter()
            .map(|pattern| Stage::new(pattern).unwrap())
            .collect()
    }

    /// The same, always through the regex engine.
    fn compiled(patterns: &[&str]) -> Vec<Stage> {
        patterns
            .iter()
            .map(|pattern| Stage::Regex(Box::new(Regex::new(pattern).unwrap())))
            .collect()
    }

    fn family(name: &str) -> &'static [&'static str] {
        FAMILIES
            .iter()
            .find(|family| family.names.contains(&name))
            .unwrap()
            .patterns
    }

    /// Splits `text` and returns the words as strings.
    fn words(stages: &[Stage], text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        for end in split_with(stages, text) {
            out.push(text[start..end].to_string());
            start = end;
        }
        assert_eq!(start, text.len(), "words must cover the text");
        out
    }

    #[test]
    fn gpt2_keeps_one_leading_space_with_each_word() {
        let gpt2 = stages(family("gpt-2"));
        assert_eq!(words(&gpt2, "Hello world"), ["Hello", " world"]);
        assert_eq!(words(&gpt2, "a  b"), ["a", " ", " b"]);
        assert_eq!(words(&gpt2, "end   "), ["end", "   "]);
        assert_eq!(words(&gpt2, " x"), [" x"]);
        assert_eq!(words(&gpt2, "it's 42!"), ["it", "'s", " 42", "!"]);
        assert_eq!(
            words(&gpt2, "we'll they've I'm"),
            ["we", "'ll", " they", "'ve", " I", "'m"]
        );
        assert_eq!(words(&gpt2, "It'S"), ["It", "'", "S"]);
        assert_eq!(words(&gpt2, "a\nb"), ["a", "\n", "b"]);
        assert_eq!(words(&gpt2, "x \n y"), ["x", " \n", " y"]);
        assert_eq!(words(&gpt2, "héllo wörld"), ["héllo", " wörld"]);
        assert_eq!(words(&gpt2, "1+1=2"), ["1", "+", "1", "=", "2"]);
        assert_eq!(words(&gpt2, "'"), ["'"]);
        assert!(words(&gpt2, "").is_empty());
    }

    #[test]
    fn llama3_groups_digits_and_attaches_newlines() {
        let llama3 = stages(family("llama-bpe"));
        assert_eq!(words(&llama3, "Hello world"), ["Hello", " world"]);
        assert_eq!(words(&llama3, "12345"), ["123", "45"]);
        assert_eq!(words(&llama3, "a 1234567"), ["a", " ", "123", "456", "7"]);
        assert_eq!(words(&llama3, "It'S we'LL"), ["It", "'S", " we", "'LL"]);
        assert_eq!(words(&llama3, "x!!\n\ny"), ["x", "!!\n\n", "y"]);
        assert_eq!(words(&llama3, "a \n\n b"), ["a", " \n\n", " b"]);
        assert_eq!(words(&llama3, "a   b"), ["a", "  ", " b"]);
        assert_eq!(words(&llama3, "\tfoo"), ["\tfoo"]);
        assert_eq!(words(&llama3, "(foo)"), ["(foo", ")"]);
        assert_eq!(words(&llama3, "a ."), ["a", " ."]);
        assert_eq!(words(&llama3, "tail "), ["tail", " "]);
        assert_eq!(words(&llama3, "\n\nx"), ["\n\n", "x"]);
        assert_eq!(words(&llama3, "\r\n"), ["\r\n"]);
    }

    #[test]
    fn qwen2_splits_every_digit() {
        let qwen2 = stages(family("qwen2"));
        assert_eq!(words(&qwen2, "x 2024"), ["x", " ", "2", "0", "2", "4"]);
        assert_eq!(
            words(&qwen2, "Hello, world!"),
            ["Hello", ",", " world", "!"]
        );
    }

    #[test]
    fn smollm_isolates_digits_before_the_gpt2_pass() {
        let smollm = stages(family("smollm"));
        assert_eq!(words(&smollm, "abc 123"), ["abc", " ", "1", "2", "3"]);
        assert_eq!(words(&smollm, "v2 ok"), ["v", "2", " ok"]);
    }

    #[test]
    fn default_separates_punctuation_runs_and_digit_triples() {
        let default = stages(DEFAULT_PATTERNS);
        assert_eq!(
            words(&default, "Hi, there..."),
            ["Hi", ",", " there", "..."]
        );
        assert_eq!(words(&default, "a 12345"), ["a", " ", "123", "45"]);
        assert_eq!(words(&default, "x<=y"), ["x", "<=", "y"]);
    }

    #[test]
    fn other_families_split_by_their_own_rules() {
        // DeepSeek V3 isolates groups of three digits first.
        let deepseek = stages(family("deepseek-v3"));
        assert_eq!(words(&deepseek, "abc1234"), ["abc", "123", "4"]);
        // Gemma 4 only splits lines.
        let gemma = stages(family("gemma4"));
        assert_eq!(words(&gemma, "a b\n\nc"), ["a b", "\n\n", "c"]);
        // SuperBPE splits digit runs into threes from the right.
        let superbpe = stages(family("superbpe"));
        assert_eq!(words(&superbpe, "x1234567"), ["x", "1", "234", "567"]);
        // Kimi K2 keeps Han characters apart from other letters.
        let kimi = stages(family("kimi-k2"));
        assert_eq!(words(&kimi, "abc中文def"), ["abc", "中文", "def"]);
    }

    #[test]
    fn names_map_to_families() {
        let llama = PreTokenizer::for_name("llama-bpe").unwrap();
        assert!(llama.ignore_merges && llama.add_bos && llama.known);
        let gemma = PreTokenizer::for_name("gemma4").unwrap();
        assert!(gemma.escape_spaces && gemma.raw_bytes);
        assert!(PreTokenizer::for_name("chatglm-bpe").unwrap().no_bos);
        // Inkling 2 splits words as GPT-4o does, takes a word that is a
        // token whole, and has no BOS token.
        let inkling = PreTokenizer::for_name("inkling2").unwrap();
        assert!(inkling.ignore_merges && inkling.no_bos && !inkling.add_bos && inkling.known);
        assert_eq!(family("inkling2"), family("gpt-4o"));
        assert!(PreTokenizer::for_name("default").unwrap().known);
        assert!(!PreTokenizer::for_name("something-new").unwrap().known);
    }

    #[test]
    fn every_family_compiles() {
        for family in &FAMILIES {
            for pattern in family.patterns {
                if let Err(error) = Stage::new(pattern) {
                    panic!("{:?}: {error}", family.names);
                }
            }
        }
    }

    /// Short texts built from the characters the expressions treat
    /// specially, in a fixed pseudo-random order so failures repeat.
    fn tricky_texts() -> Vec<String> {
        let alphabet = [
            "a", "Z", "é", "ß", "中", "あ", "ア", "한", "я", "ह", "\u{301}", "1", "٣", "²", "Ⅷ",
            " ", "  ", "\n", "\r", "\t", "\u{a0}", "\u{3000}", "'", "'s", "'S", "'ll", "'ve", ".",
            ",", "!", "?", "$", "+", "<", "=", "^", "|", "`", "-", "_", "(", "）", "。", "…", "«",
            "😀", "\u{200c}",
        ];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..3000)
            .map(|_| {
                let length = 1 + (next() % 12) as usize;
                (0..length)
                    .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn every_scanner_splits_like_its_expression() {
        let texts = tricky_texts();
        for (pattern, scan) in SCANNED {
            let fast = vec![Stage::Scan(scan)];
            let slow = compiled(&[pattern]);
            for text in &texts {
                assert_eq!(
                    words(&fast, text),
                    words(&slow, text),
                    "{scan:?} on {text:?}"
                );
            }
        }
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
        let families = FAMILIES
            .iter()
            .map(|family| family.patterns)
            .chain([DEFAULT_PATTERNS]);
        for patterns in families {
            let stages = stages(patterns);
            for text in samples {
                let ends = split_with(&stages, text);
                assert_eq!(ends.last().copied(), Some(text.len()));
                assert!(ends.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(ends.iter().all(|&end| text.is_char_boundary(end)));
            }
        }
    }
}

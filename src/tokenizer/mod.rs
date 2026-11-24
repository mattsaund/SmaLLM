//! Text to token ids and back.
//!
//! Two tokenizer kinds cover the models this engine runs:
//!
//! | Kind           | GGUF name | Used by                                     |
//! |----------------|-----------|---------------------------------------------|
//! | SentencePiece  | `llama`   | Llama 2, TinyLlama, Mistral, Gemma 1 to 3   |
//! | BPE            | `gpt2`    | Llama 3, Qwen, SmolLM, LFM and most others  |
//! | BPE, raw UTF-8 | `gemma4`  | Gemma 4                                     |
//!
//! Both start from characters and merge adjacent pairs (see `merge.rs`). They
//! differ in how text is prepared and how a pair is ranked. BPE models also
//! split text into words first, each family in its own way (see
//! `pretok.rs`).
//!
//! The vocabulary is never copied out of the model file. See `vocab.rs`.

mod merge;
mod pretok;
mod pretok_table;
mod regex;
mod unicode;
mod unicode_tables;
mod vocab;

use std::ops::Range;

use crate::error::{Error, Result};
use crate::gguf::Gguf;
use pretok::PreTokenizer;
use vocab::{HASH_SEED, Index, StringTable, hash_more};

/// The algorithm a tokenizer uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Pairs merge by the score of the token they form.
    SentencePiece,
    /// Pairs merge in the order of the model's merge list.
    Bpe,
}

/// What a token is for, as bit flags.
mod attr {
    pub const NORMAL: u8 = 1;
    pub const UNKNOWN: u8 = 2;
    pub const CONTROL: u8 = 4;
    pub const USER_DEFINED: u8 = 8;
    pub const BYTE: u8 = 32;
    /// Tokens matched as whole strings before any other tokenization.
    pub const SPECIAL: u8 = CONTROL | USER_DEFINED | UNKNOWN;
}

/// Token texts that end a turn or a document in the models in circulation,
/// as llama.cpp lists them. A model's own end tokens are added to these.
const END_TOKENS: [&str; 24] = [
    "<|eot_id|>",
    "<|eot|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "[e~[",
    "<|ifm|im_end|>",
];

/// SentencePiece writes a space as this character.
const SPACE_MARK: &str = "\u{2581}";

/// A stretch of input: plain text, or a special token found in it.
enum Fragment {
    Text(Range<usize>),
    Token(u32),
}

/// A model's tokenizer.
pub struct Tokenizer {
    kind: Kind,
    tokens: StringTable,
    token_index: Index,
    /// Attribute flags of each token. See [`attr`].
    attrs: Vec<u8>,
    /// SentencePiece scores. Empty for BPE.
    scores: Vec<f32>,
    /// BPE merge rules and their index. The position of a rule is its rank.
    merges: Option<(StringTable, Index)>,
    pre: PreTokenizer,
    pre_name: String,
    /// Special tokens, longest text first: the order they are searched in.
    special: Vec<u32>,
    /// SentencePiece fallback token for each byte value.
    byte_tokens: Vec<Option<u32>>,
    bos: Option<u32>,
    eos: Option<u32>,
    eot: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    /// Tokens that end generation.
    end_tokens: Vec<u32>,
}

impl Tokenizer {
    /// Builds the tokenizer described by a model file's metadata.
    pub fn from_gguf(file: &Gguf) -> Result<Tokenizer> {
        let model = file.get_str("tokenizer.ggml.model").unwrap_or("llama");
        let kind = match model {
            "llama" => Kind::SentencePiece,
            "gpt2" | "gemma4" => Kind::Bpe,
            "bert" | "t5" | "rwkv" | "plamo2" | "plamo3" => {
                return Err(Error::unsupported(format!(
                    "tokenizer type '{model}', which only models this engine cannot run yet use"
                )));
            }
            other => return Err(Error::unsupported(format!("tokenizer type '{other}'"))),
        };
        let token_array = file
            .get_array("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::format("the model has no vocabulary (tokenizer.ggml.tokens)"))?;
        let tokens = StringTable::new(file.mapping(), &token_array, "tokenizer.ggml.tokens")?;
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            return Err(Error::format("the vocabulary is empty"));
        }
        let token_index = Index::build(&tokens, false);

        // Token types become attribute flags. A missing table means every
        // token is ordinary text.
        let mut attrs = vec![attr::NORMAL; n_tokens];
        if let Some(types) = file.get_array("tokenizer.ggml.token_type") {
            if types.len() < n_tokens {
                return Err(Error::format(
                    "tokenizer.ggml.token_type is shorter than the vocabulary",
                ));
            }
            for (id, flags) in attrs.iter_mut().enumerate() {
                *flags = match types.number(id).unwrap_or(0.0) as i64 {
                    1 => attr::NORMAL,
                    2 => attr::UNKNOWN,
                    3 => attr::CONTROL,
                    4 => attr::USER_DEFINED,
                    6 => attr::BYTE,
                    _ => 0,
                };
            }
        }

        let mut scores = Vec::new();
        let mut merges = None;
        // A Gemma 4 vocabulary names its family by its type.
        let pre_name = match model {
            "gemma4" => "gemma4".to_string(),
            _ => file.get_str("tokenizer.ggml.pre").unwrap_or("").to_string(),
        };
        let mut pre = PreTokenizer::for_name("default")?;
        match kind {
            Kind::SentencePiece => {
                if let Some(array) = file.get_array("tokenizer.ggml.scores") {
                    if array.len() < n_tokens {
                        return Err(Error::format(
                            "tokenizer.ggml.scores is shorter than the vocabulary",
                        ));
                    }
                    scores = (0..n_tokens)
                        .map(|id| array.number(id).unwrap_or(0.0) as f32)
                        .collect();
                } else {
                    scores = vec![0.0; n_tokens];
                }
            }
            Kind::Bpe => {
                let array = file.get_array("tokenizer.ggml.merges").ok_or_else(|| {
                    Error::format("the BPE tokenizer has no merge list (tokenizer.ggml.merges)")
                })?;
                let table = StringTable::new(file.mapping(), &array, "tokenizer.ggml.merges")?;
                let index = Index::build(&table, true);
                merges = Some((table, index));
                pre = PreTokenizer::for_name(&pre_name)?;
            }
        }

        // Special token ids: defaults by kind, then the file's own values.
        let in_range = |id: Option<u32>| id.filter(|&id| (id as usize) < n_tokens);
        let read_id = |key: &str, default: Option<u32>| {
            let from_file = file.get_u64(key).and_then(|id| u32::try_from(id).ok());
            in_range(from_file).or(in_range(default))
        };
        // SentencePiece models reserve ids 1 and 2 by convention. BPE
        // models have no convention, so a missing key means no such token.
        let (default_bos, default_eos) = match kind {
            Kind::SentencePiece => (Some(1), Some(2)),
            Kind::Bpe => (None, None),
        };
        let bos = read_id("tokenizer.ggml.bos_token_id", default_bos).filter(|_| !pre.no_bos);
        let eos = read_id("tokenizer.ggml.eos_token_id", default_eos);
        let mut eot = read_id("tokenizer.ggml.eot_token_id", None);
        let eom = read_id("tokenizer.ggml.eom_token_id", None);

        let default_add_bos = kind == Kind::SentencePiece || pre.add_bos;
        // Gemma 4 files can say not to add BOS, but the model needs it, so
        // llama.cpp adds it anyway.
        let add_bos = (file
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(default_add_bos)
            || model == "gemma4")
            && bos.is_some();
        let add_eos = file
            .get_bool("tokenizer.ggml.add_eos_token")
            .unwrap_or(false)
            && eos.is_some();
        let add_space_prefix = file
            .get_bool("tokenizer.ggml.add_space_prefix")
            .unwrap_or(kind == Kind::SentencePiece);

        let mut tokenizer = Tokenizer {
            kind,
            tokens,
            token_index,
            attrs,
            scores,
            merges,
            pre,
            pre_name,
            special: Vec::new(),
            byte_tokens: Vec::new(),
            bos,
            eos,
            eot,
            add_bos,
            add_eos,
            add_space_prefix,
            end_tokens: Vec::new(),
        };

        // Generation stops at the model's own end tokens and at any token
        // whose text is a known end marker. Those are control tokens even
        // when the file does not say so.
        let mut end_tokens: Vec<u32> = [eos, eot, eom].into_iter().flatten().collect();
        let marked: Vec<u32> = END_TOKENS
            .iter()
            .filter_map(|text| tokenizer.find(text.as_bytes()))
            .collect();
        end_tokens.extend(&marked);
        let has_end = |text: &str, ends: &[u32]| {
            tokenizer
                .find(text.as_bytes())
                .is_some_and(|id| ends.contains(&id))
        };
        // Two exceptions, as in llama.cpp. Formats that end a reply with
        // `<|return|>` or `<|flush|>` use `<|end|>` inside a reply.
        let mut ordinary = Vec::new();
        if (has_end("<|return|>", &end_tokens) && has_end("<|call|>", &end_tokens))
            || (has_end("<|calls|>", &end_tokens) && has_end("<|flush|>", &end_tokens))
        {
            ordinary.extend(
                tokenizer
                    .find(b"<|end|>")
                    .map(|id| (id, attr::USER_DEFINED)),
            );
        }
        // And in Gemma 4 and a few other vocabularies, `</s>` is plain text.
        if has_end("<|tool_response>", &end_tokens)
            || has_end("<|plamo:eos|>", &end_tokens)
            || tokenizer.find(b"<rfl:parameter name=\"").is_some()
        {
            ordinary.extend(tokenizer.find(b"</s>").map(|id| (id, attr::NORMAL)));
        }
        end_tokens.retain(|id| !ordinary.iter().any(|(other, _)| other == id));
        for &id in &end_tokens {
            tokenizer.attrs[id as usize] |= attr::CONTROL;
        }
        for &(id, flags) in &ordinary {
            tokenizer.attrs[id as usize] = flags;
        }
        eot = eot.or_else(|| marked.into_iter().find(|id| end_tokens.contains(id)));
        end_tokens.sort_unstable();
        end_tokens.dedup();
        tokenizer.end_tokens = end_tokens;
        tokenizer.eot = eot;

        let mut special: Vec<u32> = (0..n_tokens as u32)
            .filter(|&id| tokenizer.attrs[id as usize] & attr::SPECIAL != 0)
            .collect();
        special.retain(|&id| !tokenizer.tokens.get(id as usize).is_empty());
        special.sort_by_key(|&id| std::cmp::Reverse(tokenizer.tokens.get(id as usize).len()));
        tokenizer.special = special;

        if kind == Kind::SentencePiece || tokenizer.pre.raw_bytes {
            tokenizer.byte_tokens = (0..=255u8)
                .map(|byte| {
                    let escaped = format!("<0x{byte:02X}>");
                    tokenizer
                        .find(escaped.as_bytes())
                        .or_else(|| tokenizer.find(&[byte]))
                })
                .collect();
        }
        Ok(tokenizer)
    }

    /// The algorithm this tokenizer uses.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Number of tokens in the vocabulary.
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// The beginning-of-text token.
    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// The end-of-text token.
    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    /// The end-of-turn token.
    pub fn eot(&self) -> Option<u32> {
        self.eot
    }

    /// True when prompts start with the beginning-of-text token.
    pub fn adds_bos(&self) -> bool {
        self.add_bos
    }

    /// True when `token` ends generation.
    pub fn is_end(&self, token: u32) -> bool {
        self.end_tokens.contains(&token)
    }

    /// True when `token` is a control token, which is not shown as text.
    pub fn is_control(&self, token: u32) -> bool {
        self.attrs
            .get(token as usize)
            .is_some_and(|flags| flags & attr::CONTROL != 0)
    }

    /// The pre-tokenizer name from the file, and whether it was recognized.
    /// An unrecognized name falls back to a default that may split text
    /// differently from the model's training.
    pub fn pre_tokenizer(&self) -> (&str, bool) {
        (
            &self.pre_name,
            self.kind == Kind::SentencePiece || self.pre.known,
        )
    }

    /// The vocabulary text of `token`, exactly as stored in the file.
    pub fn token_text(&self, token: u32) -> &[u8] {
        self.tokens.get(token as usize)
    }

    /// Looks up a token by its vocabulary text.
    pub fn find(&self, text: &[u8]) -> Option<u32> {
        self.token_index
            .find(&self.tokens, hash_more(HASH_SEED, text), |candidate| {
                candidate == text
            })
    }

    /// Converts text to token ids.
    ///
    /// `add_special` adds the beginning and end tokens the model expects.
    /// `parse_special` turns control token text such as `<|im_end|>` into
    /// the control token. Leave it off for text that comes from a user.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        self.encode_parts(&[(text, parse_special)], add_special)
    }

    /// Converts a text made of several parts to token ids.
    ///
    /// The parts are tokenized as one string, exactly as if they had been
    /// joined. The flag of each part says whether control token text inside
    /// it becomes a control token. A chat prompt uses this to trust its own
    /// markup and nothing a user typed.
    pub fn encode_parts(&self, parts: &[(&str, bool)], add_special: bool) -> Vec<u32> {
        let mut joined = String::with_capacity(parts.iter().map(|(text, _)| text.len()).sum());
        // Byte ranges in which control tokens may be matched.
        let mut trusted = Vec::new();
        for &(text, parse_special) in parts {
            if parse_special && !text.is_empty() {
                trusted.push(joined.len()..joined.len() + text.len());
            }
            joined.push_str(text);
        }
        let bytes = joined.as_bytes();
        let mut out = Vec::with_capacity(bytes.len() / 3 + 2);
        if add_special && self.add_bos {
            out.extend(self.bos);
        }
        // A space is put before text that starts the prompt or follows a
        // special token, when the model wants one.
        let mut after_special = true;
        for fragment in self.partition(bytes, &trusted) {
            match fragment {
                Fragment::Token(id) => {
                    out.push(id);
                    after_special = true;
                }
                Fragment::Text(range) => {
                    match self.kind {
                        Kind::SentencePiece => {
                            self.encode_sentencepiece(&bytes[range], after_special, &mut out);
                        }
                        Kind::Bpe => {
                            self.encode_bpe(&String::from_utf8_lossy(&bytes[range]), &mut out)
                        }
                    }
                    after_special = false;
                }
            }
        }
        if add_special && self.add_eos {
            out.extend(self.eos);
        }
        out
    }

    /// Appends the bytes `token` stands for to `out`.
    ///
    /// Control tokens produce nothing unless `special` is set. The bytes of
    /// one token are not always whole UTF-8 characters.
    pub fn append_piece(&self, token: u32, special: bool, out: &mut Vec<u8>) {
        let Some(&flags) = self.attrs.get(token as usize) else {
            return;
        };
        if !special && flags & (attr::CONTROL | attr::UNKNOWN) != 0 {
            return;
        }
        let text = self.tokens.get(token as usize);
        if flags & attr::SPECIAL != 0 {
            out.extend_from_slice(text);
        } else if flags & attr::NORMAL != 0 {
            if self.kind == Kind::SentencePiece || self.pre.escape_spaces {
                append_unmarked(text, out);
            } else if self.pre.raw_bytes {
                out.extend_from_slice(text);
            } else {
                append_byte_level(text, out);
            }
        } else if flags & attr::BYTE != 0 {
            out.extend(parse_byte_token(text));
        }
    }

    /// Converts token ids back to text. Bytes that do not form valid UTF-8
    /// become replacement characters.
    pub fn decode(&self, tokens: &[u32], special: bool) -> String {
        let mut bytes = Vec::new();
        for &token in tokens {
            self.append_piece(token, special, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Splits `text` at special tokens.
    ///
    /// Tokens are tried longest first, and each one splits every remaining
    /// stretch of text at each place it occurs. Control tokens are matched
    /// only where they lie entirely inside one of the `trusted` ranges.
    fn partition(&self, text: &[u8], trusted: &[Range<usize>]) -> Vec<Fragment> {
        if text.is_empty() {
            return Vec::new();
        }
        let is_trusted = |start: usize, end: usize| {
            trusted
                .iter()
                .any(|range| range.start <= start && end <= range.end)
        };
        let mut fragments = vec![Fragment::Text(0..text.len())];
        // Most special tokens start with a byte the text does not contain.
        let mut present = [false; 256];
        for &byte in text {
            present[byte as usize] = true;
        }
        for &id in &self.special {
            let needs_trust = self.attrs[id as usize] & (attr::CONTROL | attr::UNKNOWN) != 0;
            if needs_trust && trusted.is_empty() {
                continue;
            }
            let needle = self.tokens.get(id as usize);
            if !present[needle[0] as usize] || find_bytes(text, needle).is_none() {
                continue;
            }
            let mut split = Vec::with_capacity(fragments.len() + 2);
            for fragment in fragments {
                let Fragment::Text(range) = fragment else {
                    split.push(fragment);
                    continue;
                };
                // `pos` is where the pending text starts, and `search` where
                // the next match may start. They differ after a match that
                // was skipped for lying in untrusted text.
                let (mut pos, mut search) = (range.start, range.start);
                while let Some(found) = find_bytes(&text[search..range.end], needle) {
                    let at = search + found;
                    if needs_trust && !is_trusted(at, at + needle.len()) {
                        search = at + 1;
                        continue;
                    }
                    if at > pos {
                        split.push(Fragment::Text(pos..at));
                    }
                    split.push(Fragment::Token(id));
                    pos = at + needle.len();
                    search = pos;
                }
                if pos < range.end {
                    split.push(Fragment::Text(pos..range.end));
                }
            }
            fragments = split;
        }
        fragments
    }

    /// Tokenizes plain text the SentencePiece way.
    fn encode_sentencepiece(&self, text: &[u8], after_special: bool, out: &mut Vec<u32>) {
        // Spaces become the visible space mark, which is how the vocabulary
        // spells them.
        let mut marked = Vec::with_capacity(text.len() + text.len() / 2 + 4);
        if self.add_space_prefix && after_special {
            marked.extend_from_slice(SPACE_MARK.as_bytes());
        }
        for &byte in text {
            if byte == b' ' {
                marked.extend_from_slice(SPACE_MARK.as_bytes());
            } else {
                marked.push(byte);
            }
        }

        // Merge the pair that forms the highest scoring token first.
        let score_of = |pair: &[u8], _split: usize| {
            self.find(pair)
                .map(|id| score_key(self.scores[id as usize]))
        };
        let mut pieces = Vec::new();
        merge::merge(&marked, merge::char_lengths(&marked), score_of, &mut pieces);

        for (start, len) in pieces {
            let piece = &marked[start..start + len];
            match self.find(piece) {
                Some(id) => out.push(id),
                // A character outside the vocabulary is spelled in bytes.
                None => out.extend(
                    piece
                        .iter()
                        .filter_map(|&byte| self.byte_tokens[byte as usize]),
                ),
            }
        }
    }

    /// Tokenizes plain text the BPE way.
    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        let Some((merges, merge_index)) = &self.merges else {
            return;
        };
        // Some families spell spaces the SentencePiece way.
        let escaped;
        let text = if self.pre.escape_spaces {
            escaped = text.replace(' ', SPACE_MARK);
            escaped.as_str()
        } else {
            text
        };
        // The rank of a merge rule is its position in the list. Lower ranks
        // merge first.
        let rank_of = |pair: &[u8], split: usize| {
            let (left, right) = pair.split_at(split);
            let hash = hash_more(hash_more(hash_more(HASH_SEED, left), b" "), right);
            let rank = merge_index.find(merges, hash, |rule| {
                rule.len() == pair.len() + 1
                    && rule[..split] == *left
                    && rule[split] == b' '
                    && rule[split + 1..] == *right
            })?;
            Some(u32::MAX - rank)
        };

        let mut word = Vec::new();
        let mut pieces = Vec::new();
        let mut start = 0;
        for end in self.pre.split(text) {
            word.clear();
            let raw = &text.as_bytes()[start..end];
            start = end;
            if self.pre.raw_bytes {
                word.extend_from_slice(raw);
            } else {
                // The vocabulary spells every byte as a printable character.
                let mut utf8 = [0u8; 4];
                for &byte in raw {
                    word.extend_from_slice(
                        unicode::byte_to_char(byte)
                            .encode_utf8(&mut utf8)
                            .as_bytes(),
                    );
                }
            }

            // A word that is itself a token skips merging, in families that
            // say so. Runs of line feeds always do in raw families, where
            // the merges never join them.
            let whole = self.pre.ignore_merges
                || (self.pre.raw_bytes && raw.iter().all(|&byte| byte == b'\n'));
            if whole {
                if let Some(id) = self.find(&word) {
                    out.push(id);
                    continue;
                }
            }
            merge::merge(&word, merge::char_lengths(&word), rank_of, &mut pieces);
            for &(piece_start, len) in &pieces {
                let piece = &word[piece_start..piece_start + len];
                match self.find(piece) {
                    Some(id) => out.push(id),
                    // Raw families spell unknown bytes as `<0xNN>` tokens,
                    // byte-level ones as their one-character tokens.
                    None if self.pre.raw_bytes => out.extend(
                        piece
                            .iter()
                            .filter_map(|&byte| self.byte_tokens[byte as usize]),
                    ),
                    None => out.extend(piece.iter().filter_map(|&byte| self.find(&[byte]))),
                }
            }
        }
    }
}

/// Maps a score to an integer that sorts the same way, so the merge queue
/// can compare priorities exactly.
fn score_key(score: f32) -> u32 {
    let bits = score.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Position of the first occurrence of `needle` in `haystack`.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut pos = 0;
    while pos <= last_start {
        pos += haystack[pos..=last_start]
            .iter()
            .position(|&byte| byte == first)?;
        if &haystack[pos..pos + needle.len()] == needle {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

/// Appends SentencePiece token text with space marks turned back to spaces.
fn append_unmarked(text: &[u8], out: &mut Vec<u8>) {
    let mark = SPACE_MARK.as_bytes();
    let mut rest = text;
    while let Some(at) = find_bytes(rest, mark) {
        out.extend_from_slice(&rest[..at]);
        out.push(b' ');
        rest = &rest[at + mark.len()..];
    }
    out.extend_from_slice(rest);
}

/// Appends byte-level BPE token text as the bytes its characters stand for.
fn append_byte_level(text: &[u8], out: &mut Vec<u8>) {
    match std::str::from_utf8(text) {
        Ok(text) => {
            let mut utf8 = [0u8; 4];
            for c in text.chars() {
                match unicode::char_to_byte(c) {
                    Some(byte) => out.push(byte),
                    None => out.extend_from_slice(c.encode_utf8(&mut utf8).as_bytes()),
                }
            }
        }
        Err(_) => out.extend_from_slice(text),
    }
}

/// The byte a token spelled `<0xNN>` stands for.
fn parse_byte_token(text: &[u8]) -> Option<u8> {
    let hex = text.strip_prefix(b"<0x")?.strip_suffix(b">")?;
    u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()
}

#[cfg(test)]
mod tests;

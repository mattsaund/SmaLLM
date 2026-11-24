//! Tokenizer tests on small hand-made vocabularies.
//!
//! Agreement with the reference tokenizer on real models is checked by the
//! golden tests in `tests/`.

use super::*;
use crate::gguf::testing::Builder;

/// Builds a SentencePiece tokenizer from `(text, score, type)` entries.
fn sentencepiece(entries: &[(&str, f32, i32)], configure: impl Fn(&mut Builder)) -> Tokenizer {
    let texts: Vec<&[u8]> = entries.iter().map(|entry| entry.0.as_bytes()).collect();
    let scores: Vec<f32> = entries.iter().map(|entry| entry.1).collect();
    let types: Vec<i32> = entries.iter().map(|entry| entry.2).collect();
    let mut builder = Builder::new();
    builder
        .string("tokenizer.ggml.model", "llama")
        .strings("tokenizer.ggml.tokens", &texts)
        .f32s("tokenizer.ggml.scores", &scores)
        .i32s("tokenizer.ggml.token_type", &types);
    configure(&mut builder);
    Tokenizer::from_gguf(&Gguf::from_bytes(builder.build()).unwrap()).unwrap()
}

/// A small SentencePiece vocabulary. Ids 0 to 2 are the usual specials.
fn small_sentencepiece() -> Tokenizer {
    sentencepiece(
        &[
            ("<unk>", 0.0, 2),
            ("<s>", 0.0, 3),
            ("</s>", 0.0, 3),
            ("\u{2581}", -2.0, 1),
            ("a", -3.0, 1),
            ("b", -3.0, 1),
            ("c", -3.0, 1),
            ("ab", -1.0, 1),
            ("\u{2581}ab", -0.5, 1),
            ("bc", -1.5, 1),
            ("<0xC3>", 0.0, 6),
            ("<0xA9>", 0.0, 6),
            ("<|tool|>", 0.0, 4),
        ],
        |_| {},
    )
}

/// Builds a byte-level BPE tokenizer.
fn bpe(tokens: &[&str], merges: &[&str], pre: &str, configure: impl Fn(&mut Builder)) -> Tokenizer {
    let texts: Vec<&[u8]> = tokens.iter().map(|t| t.as_bytes()).collect();
    let rules: Vec<&[u8]> = merges.iter().map(|m| m.as_bytes()).collect();
    let mut builder = Builder::new();
    builder
        .string("tokenizer.ggml.model", "gpt2")
        .string("tokenizer.ggml.pre", pre)
        .strings("tokenizer.ggml.tokens", &texts)
        .strings("tokenizer.ggml.merges", &rules);
    configure(&mut builder);
    Tokenizer::from_gguf(&Gguf::from_bytes(builder.build()).unwrap()).unwrap()
}

/// A small BPE vocabulary. `\u{120}` is how the vocabulary spells a space.
fn small_bpe() -> Tokenizer {
    bpe(
        &[
            "h", "e", "l", "o", "\u{120}", "w", "r", "d", "he", "ll", "hell", "hello", "\u{120}w",
            "!", "<|end|>", "\u{10a}",
        ],
        &["h e", "l l", "he ll", "hell o", "\u{120} w"],
        "gpt-2",
        |builder| {
            builder.i32s(
                "tokenizer.ggml.token_type",
                &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1],
            );
        },
    )
}

#[test]
fn sentencepiece_merges_by_score() {
    let tokenizer = small_sentencepiece();
    assert_eq!(tokenizer.kind(), Kind::SentencePiece);
    // " ab" forms one token because that merge has the best score.
    assert_eq!(tokenizer.encode("ab", false, false), [8]);
    // "abc": "ab" (-1.0) outranks "bc" (-1.5), leaving "c" on its own.
    assert_eq!(tokenizer.encode("abc", false, false), [8, 6]);
    assert_eq!(tokenizer.encode("c ab", false, false), [3, 6, 8]);
}

#[test]
fn sentencepiece_adds_bos_and_a_leading_space() {
    let tokenizer = small_sentencepiece();
    assert_eq!(tokenizer.encode("a", true, false), [1, 3, 4]);
    assert_eq!(tokenizer.encode("", true, false), [1]);
    assert!(tokenizer.encode("", false, false).is_empty());

    let no_prefix = sentencepiece(
        &[
            ("<unk>", 0.0, 2),
            ("<s>", 0.0, 3),
            ("</s>", 0.0, 3),
            ("a", 0.0, 1),
        ],
        |builder| {
            builder
                .bool("tokenizer.ggml.add_space_prefix", false)
                .bool("tokenizer.ggml.add_bos_token", false);
        },
    );
    assert_eq!(no_prefix.encode("a", true, false), [3]);
}

#[test]
fn sentencepiece_spells_unknown_characters_in_bytes() {
    let tokenizer = small_sentencepiece();
    // U+00E9 is the bytes C3 A9, both of which have byte tokens.
    assert_eq!(tokenizer.encode("\u{e9}", false, false), [3, 10, 11]);
    assert_eq!(tokenizer.decode(&[3, 10, 11], false), " \u{e9}");
    // A character with no byte tokens is dropped, not turned into garbage.
    assert_eq!(tokenizer.encode("z", false, false), [3]);
}

#[test]
fn sentencepiece_decodes_marks_back_to_spaces() {
    let tokenizer = small_sentencepiece();
    assert_eq!(tokenizer.decode(&[8, 3, 6], false), " ab c");
    // Control tokens are hidden unless asked for.
    assert_eq!(tokenizer.decode(&[1, 8, 2], false), " ab");
    assert_eq!(tokenizer.decode(&[1, 8, 2], true), "<s> ab</s>");
    assert_eq!(tokenizer.decode(&[9999], true), "");
}

#[test]
fn special_tokens_are_found_in_text() {
    let tokenizer = small_sentencepiece();
    // A user-defined token is always matched.
    assert_eq!(tokenizer.encode("ab<|tool|>ab", false, false), [8, 12, 8]);
    // A control token is matched only when special tokens are parsed.
    assert_eq!(tokenizer.encode("ab</s>", false, true), [8, 2]);
    assert_ne!(tokenizer.encode("ab</s>", false, false), [8, 2]);
    // Text right after a special token gets the leading space again.
    assert_eq!(tokenizer.encode("<s>a", false, true), [1, 3, 4]);
}

#[test]
fn control_tokens_are_parsed_only_in_trusted_parts() {
    let tokenizer = small_sentencepiece();
    // The same bytes: a control token in the markup, plain text in the
    // user's part.
    let ids = tokenizer.encode_parts(&[("</s>", true), ("ab</s>", false), ("</s>", true)], false);
    assert_eq!(ids.first(), Some(&2));
    assert_eq!(ids.last(), Some(&2));
    assert_eq!(ids.iter().filter(|&&id| id == 2).count(), 2);
    // A control token that straddles a part boundary is not trusted.
    let ids = tokenizer.encode_parts(&[("</", true), ("s>", false)], false);
    assert!(!ids.contains(&2));
    // Parts are tokenized as one string, so words may span them.
    assert_eq!(
        tokenizer.encode_parts(&[("a", false), ("b", false)], false),
        tokenizer.encode("ab", false, false)
    );
    // User-defined tokens are matched everywhere.
    assert_eq!(tokenizer.encode_parts(&[("<|tool|>", false)], false), [12]);
}

#[test]
fn bpe_merges_in_rank_order() {
    let tokenizer = small_bpe();
    assert_eq!(tokenizer.kind(), Kind::Bpe);
    assert_eq!(tokenizer.encode("hello", false, false), [11]);
    assert_eq!(tokenizer.encode("hell", false, false), [10]);
    assert_eq!(
        tokenizer.encode("hello world!", false, false),
        [11, 12, 3, 6, 2, 7, 13]
    );
    assert_eq!(
        tokenizer.decode(&[11, 12, 3, 6, 2, 7, 13], false),
        "hello world!"
    );
}

#[test]
fn bpe_has_no_implicit_bos_or_space() {
    let tokenizer = small_bpe();
    assert!(!tokenizer.adds_bos());
    assert_eq!(tokenizer.encode("he", true, false), [8]);
    assert_eq!(tokenizer.encode("\n", false, false), [15]);
    assert_eq!(tokenizer.decode(&[15], false), "\n");
}

#[test]
fn bpe_end_tokens_are_detected_by_text() {
    let tokenizer = small_bpe();
    assert!(tokenizer.is_end(14));
    assert!(tokenizer.is_control(14));
    assert!(!tokenizer.is_end(11));
    assert_eq!(tokenizer.eot(), Some(14));
    assert_eq!(tokenizer.encode("he<|end|>", false, true), [8, 14]);
    // Unparsed, the marker is ordinary text, and its unknown bytes vanish.
    assert_eq!(tokenizer.encode("he<|end|>", false, false), [8, 1, 7]);
    assert_eq!(tokenizer.decode(&[8, 14], false), "he");
    assert_eq!(tokenizer.decode(&[8, 14], true), "he<|end|>");
}

#[test]
fn bpe_whole_word_tokens_skip_merging_when_the_family_says_so() {
    let tokens = ["a", "b", "ab", "abab"];
    // No rule builds "abab" from two "ab", so merging stops at ["ab", "ab"].
    let merging = bpe(&tokens, &["a b"], "gpt-2", |_| {});
    assert_eq!(merging.encode("abab", false, false), [2, 2]);
    // Llama 3 style vocabularies take the whole word when it is a token.
    let whole = bpe(&tokens, &["a b"], "llama-bpe", |_| {});
    assert_eq!(whole.encode("abab", false, false), [3]);
}

#[test]
fn unknown_pre_tokenizers_are_reported() {
    assert_eq!(small_bpe().pre_tokenizer(), ("gpt-2", true));
    let odd = bpe(&["a"], &[], "brand-new", |_| {});
    assert_eq!(odd.pre_tokenizer(), ("brand-new", false));
    assert!(small_sentencepiece().pre_tokenizer().1);
}

#[test]
fn special_ids_come_from_the_file_when_valid() {
    let tokenizer = bpe(&["a", "b", "c"], &[], "gpt-2", |builder| {
        builder
            .u32("tokenizer.ggml.bos_token_id", 1)
            .u32("tokenizer.ggml.eos_token_id", 2)
            .bool("tokenizer.ggml.add_bos_token", true)
            .bool("tokenizer.ggml.add_eos_token", true);
    });
    assert_eq!((tokenizer.bos(), tokenizer.eos()), (Some(1), Some(2)));
    assert_eq!(tokenizer.encode("a", true, false), [1, 0, 2]);
    assert!(tokenizer.is_end(2));

    // Ids outside the vocabulary are ignored instead of trusted.
    let broken = bpe(&["a", "b", "c"], &[], "gpt-2", |builder| {
        builder
            .u32("tokenizer.ggml.bos_token_id", 99)
            .bool("tokenizer.ggml.add_bos_token", true);
    });
    assert_eq!(broken.bos(), None);
    assert_eq!(broken.encode("a", true, false), [0]);
}

#[test]
fn malformed_tokenizers_are_rejected() {
    let mut builder = Builder::new();
    builder.string("tokenizer.ggml.model", "llama");
    assert!(Tokenizer::from_gguf(&Gguf::from_bytes(builder.build()).unwrap()).is_err());

    let mut builder = Builder::new();
    builder
        .string("tokenizer.ggml.model", "gpt2")
        .strings("tokenizer.ggml.tokens", &[b"a"]);
    assert!(Tokenizer::from_gguf(&Gguf::from_bytes(builder.build()).unwrap()).is_err());

    let mut builder = Builder::new();
    builder
        .string("tokenizer.ggml.model", "bert")
        .strings("tokenizer.ggml.tokens", &[b"a"]);
    assert!(Tokenizer::from_gguf(&Gguf::from_bytes(builder.build()).unwrap()).is_err());

    let mut builder = Builder::new();
    builder
        .string("tokenizer.ggml.model", "llama")
        .strings("tokenizer.ggml.tokens", &[b"a", b"b"])
        .f32s("tokenizer.ggml.scores", &[0.0]);
    assert!(Tokenizer::from_gguf(&Gguf::from_bytes(builder.build()).unwrap()).is_err());
}

#[test]
fn byte_search_finds_first_matches() {
    assert_eq!(find_bytes(b"hello", b"ll"), Some(2));
    assert_eq!(find_bytes(b"hello", b"lo"), Some(3));
    assert_eq!(find_bytes(b"aab", b"ab"), Some(1));
    assert_eq!(find_bytes(b"hello", b"x"), None);
    assert_eq!(find_bytes(b"hi", b"hello"), None);
    assert_eq!(find_bytes(b"hi", b""), None);
}

#[test]
fn score_keys_sort_like_scores() {
    let scores = [-1000.0f32, -2.5, -0.0, 0.0, 1e-9, 3.0, 1e9];
    for pair in scores.windows(2) {
        assert!(
            score_key(pair[0]) <= score_key(pair[1]),
            "{} vs {}",
            pair[0],
            pair[1]
        );
    }
    assert!(score_key(-1.0) < score_key(-0.5));
}

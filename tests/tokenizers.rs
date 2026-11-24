//! Tokenization against the reference results that llama.cpp keeps.
//!
//! llama.cpp's repository has a folder of tokenizer test files:
//! `models/ggml-vocab-<name>.gguf` holds a vocabulary, `.gguf.inp` a list of
//! tricky texts, and `.gguf.out` the tokens each text becomes. The tokens
//! were made with the tokenizers the models were trained with, so they are
//! the ground truth, not llama.cpp's own output.
//!
//! The files are not in this repository. Point `SMALLM_TEST_VOCABS` at that
//! folder and every vocabulary with test files is checked. Without it the
//! test is skipped. `SMALLM_TEST_REQUIRE_VOCABS=1` turns a missing folder
//! into a failure, which is how CI makes sure the test ran.

use std::path::{Path, PathBuf};

use smallm::Tokenizer;
use smallm::gguf::Gguf;

/// What separates the texts in a `.inp` file.
const SEPARATOR: &str = "\n__ggml_vocab_test__\n";

/// Tokenizer types that only encoder models use: WordPiece for BERT, and
/// Unigram for T5. The engine runs decoder models, so these are skipped.
const ENCODER_TOKENIZERS: [&str; 2] = ["bert", "t5"];

/// One vocabulary and its test cases.
struct Suite {
    name: String,
    vocab: PathBuf,
    cases: Vec<(String, Vec<u32>)>,
}

/// Reads every vocabulary in `dir` that has both test files.
fn suites(dir: &Path) -> Vec<Suite> {
    let mut found = Vec::new();
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
    for entry in entries {
        let path = entry.unwrap().path();
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(name) = file_name
            .strip_prefix("ggml-vocab-")
            .and_then(|rest| rest.strip_suffix(".gguf"))
        else {
            continue;
        };
        let input = path.with_extension("gguf.inp");
        let output = path.with_extension("gguf.out");
        let (Ok(input), Ok(output)) = (
            std::fs::read_to_string(&input),
            std::fs::read_to_string(&output),
        ) else {
            continue;
        };
        // The file ends with a separator, and nothing after it is a text.
        let texts: Vec<String> = input
            .strip_suffix(SEPARATOR)
            .unwrap_or(&input)
            .split(SEPARATOR)
            .map(str::to_string)
            .collect();
        let expected: Vec<Vec<u32>> = output
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|id| id.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(
            texts.len(),
            expected.len(),
            "{name}: the input and output files disagree"
        );
        found.push(Suite {
            name: name.to_string(),
            vocab: path,
            cases: texts.into_iter().zip(expected).collect(),
        });
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

#[test]
fn every_reference_vocabulary_tokenizes_like_its_original() {
    let Some(dir) = std::env::var_os("SMALLM_TEST_VOCABS").map(PathBuf::from) else {
        assert!(
            std::env::var_os("SMALLM_TEST_REQUIRE_VOCABS").is_none(),
            "SMALLM_TEST_VOCABS is required but not set"
        );
        eprintln!("skipped: SMALLM_TEST_VOCABS is not set");
        return;
    };
    let suites = suites(&dir);
    assert!(
        !suites.is_empty(),
        "no vocabularies with test files in {}",
        dir.display()
    );

    let mut failures = Vec::new();
    for suite in &suites {
        let file = Gguf::open(&suite.vocab).unwrap();
        // The tokenizers of encoder models come with those models.
        let kind = file.get_str("tokenizer.ggml.model").unwrap_or_default();
        if ENCODER_TOKENIZERS.contains(&kind) {
            eprintln!(
                "{:<16} skipped: '{kind}' tokenizers are for encoder models",
                suite.name
            );
            continue;
        }
        let tokenizer = match Tokenizer::from_gguf(&file) {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                failures.push(format!("{}: does not load: {error}", suite.name));
                continue;
            }
        };
        let mut wrong = 0;
        for (text, expected) in &suite.cases {
            let got = tokenizer.encode(text, false, false);
            if &got != expected {
                if wrong < 3 {
                    failures.push(format!(
                        "{}: {:?}\n    got      {got:?}\n    expected {expected:?}",
                        suite.name, text
                    ));
                }
                wrong += 1;
            }
        }
        eprintln!(
            "{:<16} {}/{} texts match",
            suite.name,
            suite.cases.len() - wrong,
            suite.cases.len()
        );
        if wrong > 3 {
            failures.push(format!("{}: {} more texts differ", suite.name, wrong - 3));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Times tokenization of a large text with each reference vocabulary. A
/// measurement, not a check. Run it with:
/// `SMALLM_TEST_VOCABS=... SMALLM_BENCH_TEXT=file cargo test --release --test tokenizers -- --ignored --nocapture`
#[test]
#[ignore = "a benchmark, run on request"]
fn tokenizer_speed() {
    let (Some(dir), Some(text)) = (
        std::env::var_os("SMALLM_TEST_VOCABS"),
        std::env::var_os("SMALLM_BENCH_TEXT"),
    ) else {
        return;
    };
    let text = std::fs::read_to_string(text).unwrap();
    for suite in suites(Path::new(&dir)) {
        let file = Gguf::open(&suite.vocab).unwrap();
        let Ok(tokenizer) = Tokenizer::from_gguf(&file) else {
            continue;
        };
        let start = std::time::Instant::now();
        let tokens = tokenizer.encode(&text, false, false);
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "{:<16} {:>8} tokens  {:>6.1} MB/s",
            suite.name,
            tokens.len(),
            text.len() as f64 / seconds / 1e6
        );
    }
}

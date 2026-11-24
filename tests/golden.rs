//! Agreement with the reference implementation on real models.
//!
//! Each file in `tests/golden` holds what llama.cpp produced for one model:
//! the tokens of a test text, and a greedy generation from a prompt. These
//! tests replay the same inputs. Tokenization must match exactly.
//!
//! Generation is checked one step at a time along the reference's own
//! output. Both engines round activations to 8 bits, in slightly different
//! ways, so their logits differ by about one percent. When two tokens are
//! nearly tied, that is enough to swap them. A step therefore passes when
//! the reference token is the engine's choice or within a small margin of
//! it, and only a few steps may need the margin.
//!
//! Model files are not in the repository. Point `SMALLM_TEST_MODELS` at a
//! folder that holds them. A model that is not there is skipped, unless its
//! name is listed in `SMALLM_TEST_REQUIRE` (comma separated), which is how
//! CI makes sure it never passes by skipping everything.
//!
//! The golden files are written by `tools/reference/golden.sh`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use smallm::sampler::argmax;
use smallm::{Model, Session, SessionConfig};

/// The reference results for one model.
struct Golden {
    /// File name of the model.
    model: String,
    /// Tokens of `texts/mixed.txt`, with control tokens left as text.
    tokens_plain: Vec<u32>,
    /// Tokens of `texts/mixed.txt`, with control tokens parsed.
    tokens_special: Vec<u32>,
    /// File name of the prompt, in `prompts/`.
    prompt: String,
    /// Tokens of the prompt.
    context: Vec<u32>,
    /// Tokens generated greedily after the prompt.
    greedy: Vec<u32>,
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn ids(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .map(|id| id.parse().expect("token ids are numbers"))
        .collect()
}

fn read_golden(name: &str) -> Golden {
    let path = golden_dir().join(format!("{name}.golden"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let field = |prefix: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("{} has no '{prefix}' line", path.display()))
            .trim()
            .to_string()
    };
    Golden {
        model: field("model "),
        tokens_plain: ids(&field("tokens plain ")),
        tokens_special: ids(&field("tokens special ")),
        prompt: field("prompt "),
        context: ids(&field("context ")),
        greedy: ids(&field("greedy ")),
    }
}

/// Finds the model file, or decides the test may be skipped.
fn find_model(name: &str, file: &str) -> Option<PathBuf> {
    let required = std::env::var("SMALLM_TEST_REQUIRE").unwrap_or_default();
    let is_required = required.split(',').any(|required| required.trim() == name);
    let path = std::env::var_os("SMALLM_TEST_MODELS").map(|dir| PathBuf::from(dir).join(file));
    match path {
        Some(path) if path.is_file() => Some(path),
        _ if is_required => panic!("{file} is required by SMALLM_TEST_REQUIRE but was not found"),
        _ => {
            eprintln!("skipped: {file} is not in SMALLM_TEST_MODELS");
            None
        }
    }
}

/// Shows where two token sequences first differ.
fn assert_same_tokens(got: &[u32], expected: &[u32], what: &str) {
    if got == expected {
        return;
    }
    let at = got
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(expected.len()));
    let window = |tokens: &[u32]| tokens[at.saturating_sub(4)..(at + 4).min(tokens.len())].to_vec();
    panic!(
        "{what}: {} tokens, expected {}. First difference at {at}: got {:?}, expected {:?}",
        got.len(),
        expected.len(),
        window(got),
        window(expected)
    );
}

/// How far below the engine's best logit the reference token may score.
/// The two engines' logits differ by up to about one unit on a span of 25,
/// and a wrong computation misses by several.
const TIE_MARGIN: f32 = 1.0;

/// How many steps of one generation may differ within the margin. Most
/// models need none. A model that reasons in free text before it answers
/// has many near-equal continuations and needs a few.
const MAX_TIES: usize = 5;

/// Replays one golden file against the engine.
fn check(name: &str) {
    let golden = read_golden(name);
    let Some(path) = find_model(name, &golden.model) else {
        return;
    };
    let model =
        Arc::new(Model::load(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display())));
    let tokenizer = model.tokenizer();

    let text = std::fs::read_to_string(golden_dir().join("texts").join("mixed.txt")).unwrap();
    assert_same_tokens(
        &tokenizer.encode(&text, true, false),
        &golden.tokens_plain,
        "plain tokens",
    );
    assert_same_tokens(
        &tokenizer.encode(&text, true, true),
        &golden.tokens_special,
        "special tokens",
    );

    let prompt =
        std::fs::read_to_string(golden_dir().join("prompts").join(&golden.prompt)).unwrap();
    let context = tokenizer.encode(&prompt, true, true);
    assert_same_tokens(&context, &golden.context, "prompt tokens");

    let config = SessionConfig {
        n_ctx: context.len() + golden.greedy.len() + 8,
        threads: 4,
        ..Default::default()
    };
    let mut session = Session::new(Arc::clone(&model), config).unwrap();
    session.eval(&context).unwrap();
    let mut ties = 0;
    for (step, &expected) in golden.greedy.iter().enumerate() {
        let logits = session.logits();
        let chosen = argmax(logits);
        if chosen != expected {
            let margin = logits[chosen as usize] - logits[expected as usize];
            assert!(
                margin <= TIE_MARGIN,
                "step {step}: chose token {chosen}, the reference chose {expected}, {margin:.3} logits lower"
            );
            ties += 1;
        }
        if tokenizer.is_end(expected) {
            break;
        }
        // Continue along the reference's path, so one tie does not turn
        // every later step into a comparison of different texts.
        session.eval(&[expected]).unwrap();
    }
    assert!(
        ties <= MAX_TIES,
        "{ties} steps differed from the reference within the tie margin"
    );
}

#[test]
fn stories_260k() {
    check("stories260K");
}

#[test]
fn stories_15m_q4_0() {
    check("stories15M-q4_0");
}

#[test]
fn smollm2_135m_q4_k_m() {
    check("SmolLM2-135M-Instruct-Q4_K_M");
}

#[test]
fn qwen2_5_0_5b_q4_k_m() {
    check("qwen2.5-0.5b-instruct-q4_k_m");
}

#[test]
fn qwen3_0_6b_q4_k_m() {
    check("Qwen3-0.6B-Q4_K_M");
}

#[test]
fn gemma_3_270m_q4_k_m() {
    check("gemma-3-270m-it-Q4_K_M");
}

#[test]
fn llama_3_2_1b_q4_k_m() {
    check("Llama-3.2-1B-Instruct-Q4_K_M");
}

#[test]
fn tinyllama_1_1b_q4_k_m() {
    check("tinyllama-1.1b-chat-v1.0.Q4_K_M");
}

#[test]
fn lfm2_5_350m_q4_k_m() {
    check("LFM2.5-350M-Q4_K_M");
}

#[test]
fn lfm2_5_1_2b_q4_k_m() {
    check("LFM2.5-1.2B-Instruct-Q4_K_M");
}

#[test]
fn gemma_4_e2b_q4_0() {
    check("gemma-4-E2B-it-Q4_0");
}

#[test]
fn lfm2_8b_a1b_q4_k_m() {
    check("LFM2-8B-A1B-Q4_K_M");
}

#[test]
fn granite_3_1_1b_a400m_q4_k_m() {
    check("granite-3.1-1b-a400m-instruct-Q4_K_M");
}

#[test]
fn gemma_2_2b_q4_k_m() {
    check("gemma-2-2b-it-Q4_K_M");
}

#[test]
fn phi_4_mini_q4_k_m() {
    check("microsoft_Phi-4-mini-instruct-Q4_K_M");
}

#[test]
fn every_golden_file_has_a_test() {
    let tested = [
        "gemma-2-2b-it-Q4_K_M",
        "microsoft_Phi-4-mini-instruct-Q4_K_M",
        "LFM2-8B-A1B-Q4_K_M",
        "granite-3.1-1b-a400m-instruct-Q4_K_M",
        "LFM2.5-350M-Q4_K_M",
        "LFM2.5-1.2B-Instruct-Q4_K_M",
        "gemma-4-E2B-it-Q4_0",
        "stories260K",
        "stories15M-q4_0",
        "SmolLM2-135M-Instruct-Q4_K_M",
        "qwen2.5-0.5b-instruct-q4_k_m",
        "Qwen3-0.6B-Q4_K_M",
        "gemma-3-270m-it-Q4_K_M",
        "Llama-3.2-1B-Instruct-Q4_K_M",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M",
    ];
    for entry in std::fs::read_dir(golden_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|extension| extension == "golden")
        {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            assert!(
                tested.contains(&name.as_str()),
                "{name}.golden has no test in tests/golden.rs"
            );
            // Every golden file must also be well formed.
            let golden = read_golden(&name);
            assert!(
                !golden.context.is_empty()
                    && !golden.greedy.is_empty()
                    && !golden.tokens_plain.is_empty()
            );
        }
    }
}

//! Compares this engine's logits with a file of reference logits, to check
//! a new architecture against llama.cpp by hand.
//!
//! The reference harness in `tools/reference` writes the file:
//!
//! ```sh
//! ref ids model.gguf ids.txt reference.bin
//! SMALLM_COMPARE="model.gguf ids.txt reference.bin" \
//!     cargo test --release --test compare -- --ignored --nocapture
//! ```
//!
//! `ids.txt` holds token ids separated by spaces. The test evaluates them in
//! one batch and one token at a time, and prints how far the logits after
//! the last token are from the reference. It fails when the reference's top
//! token is not near the top here, or when the logits are far apart.

use std::sync::Arc;

use smallm::{Model, Session, SessionConfig};

/// How the logits compare.
struct Agreement {
    /// Largest difference, relative to the largest reference logit.
    max_error: f64,
    /// Cosine similarity of the two logit vectors.
    cosine: f64,
    /// Whether both pick the same top token.
    same_top: bool,
    /// How many of the reference's ten best tokens are in our ten best.
    top10: usize,
}

fn agreement(got: &[f32], reference: &[f32]) -> Agreement {
    let scale = reference
        .iter()
        .fold(1e-9f64, |max, v| max.max(v.abs() as f64));
    let max_error = got
        .iter()
        .zip(reference)
        .fold(0.0f64, |max, (a, b)| max.max((*a as f64 - *b as f64).abs()))
        / scale;
    let dot: f64 = got
        .iter()
        .zip(reference)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let best = |v: &[f32]| {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        order.truncate(10);
        order
    };
    let (ours, theirs) = (best(got), best(reference));
    Agreement {
        max_error,
        cosine: dot / (norm(got) * norm(reference)).max(1e-30),
        same_top: ours[0] == theirs[0],
        top10: theirs.iter().filter(|token| ours.contains(token)).count(),
    }
}

fn logits(model: &Arc<Model>, ids: &[u32], batch: usize, exact: bool) -> Vec<f32> {
    let config = SessionConfig {
        n_ctx: ids.len() + 8,
        batch,
        kernels: exact.then(smallm::kernels::exact),
        ..Default::default()
    };
    let mut session = Session::new(Arc::clone(model), config).expect("session");
    session.eval(ids).expect("evaluation");
    session.logits().to_vec()
}

#[test]
#[ignore = "needs a model and a reference file, see the top of this file"]
fn logits_match_the_reference() {
    let Ok(spec) = std::env::var("SMALLM_COMPARE") else {
        eprintln!("set SMALLM_COMPARE to \"model ids reference\"");
        return;
    };
    let parts: Vec<&str> = spec.split_whitespace().collect();
    let [model_path, ids_path, reference_path] = parts[..] else {
        panic!("SMALLM_COMPARE needs three paths: model, ids, reference logits");
    };
    let model = Arc::new(Model::load(model_path).expect("model loads"));
    let ids: Vec<u32> = std::fs::read_to_string(ids_path)
        .expect("ids file")
        .split_whitespace()
        .map(|id| id.parse().expect("token ids are numbers"))
        .collect();
    let reference: Vec<f32> = std::fs::read(reference_path)
        .expect("reference file")
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(
        reference.len(),
        model.config().n_vocab,
        "one logit per token"
    );

    let show = |what: &str, found: &Agreement| {
        println!(
            "{} tokens, {what}: max error {:.4}, cosine {:.6}, same top token {}, top 10 shared {}",
            ids.len(),
            found.max_error,
            found.cosine,
            found.same_top,
            found.top10
        );
    };
    for batch in [32, 1] {
        let found = agreement(&logits(&model, &ids, batch, false), &reference);
        show(&format!("batch {batch} against the reference"), &found);
        assert!(found.cosine > 0.99, "the logits point elsewhere");
        assert!(found.top10 >= 7, "the best tokens differ");
    }
    // Both engines round activations before multiplying, each in its own
    // way. Float math without that rounding shows how much of the
    // difference is rounding: the reference should sit about as far from
    // it as this engine does.
    let exact = logits(&model, &ids, 1, true);
    show(
        "reference against float math",
        &agreement(&reference, &exact),
    );
    for batch in [32, 1] {
        show(
            &format!("batch {batch} against float math"),
            &agreement(&logits(&model, &ids, batch, false), &exact),
        );
    }
    // Float math in batches only adds in another order.
    let batched = agreement(&logits(&model, &ids, 32, true), &exact);
    show("float math in batches against one by one", &batched);
    assert!(batched.cosine > 0.99999, "batching changes the result");
}

/// Compares an encoder's pooled output with a reference file from
/// `ref embed`:
///
/// ```sh
/// ref embed model.gguf text.txt reference.bin
/// SMALLM_COMPARE_EMBED="model.gguf text.txt reference.bin" \
///     cargo test --release --test compare -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a model and a reference file, see above"]
fn embeddings_match_the_reference() {
    let Ok(spec) = std::env::var("SMALLM_COMPARE_EMBED") else {
        eprintln!("set SMALLM_COMPARE_EMBED to \"model text reference\"");
        return;
    };
    let parts: Vec<&str> = spec.split_whitespace().collect();
    let [model_path, text_path, reference_path] = parts[..] else {
        panic!("SMALLM_COMPARE_EMBED needs three paths: model, text, reference");
    };
    let model = Arc::new(Model::load(model_path).expect("model loads"));
    let text = std::fs::read_to_string(text_path).expect("text file");
    let tokens = model.tokenizer().encode(&text, true, true);
    let reference: Vec<f32> = std::fs::read(reference_path)
        .expect("reference file")
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut embedder = smallm::model::Embedder::new(Arc::clone(&model), 0).expect("an encoder");
    let got = embedder.embed(&tokens).expect("embedding");
    assert_eq!(got.len(), reference.len(), "one value per reference value");
    let found = agreement(&got, &reference);
    println!(
        "{} tokens, {:?} pooling, {} values: max error {:.5}, cosine {:.7}",
        tokens.len(),
        embedder.pooling(),
        got.len(),
        found.max_error,
        found.cosine
    );
    if got.len() <= 4 {
        println!("scores: ours {got:?}, reference {reference:?}");
    }
    assert!(
        found.cosine > 0.999 || got.len() <= 4,
        "the embedding points elsewhere"
    );
}

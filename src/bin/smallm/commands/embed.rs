//! `smallm embed` and `smallm rerank`: run an encoder model.
//!
//! An embedding model turns a text into a vector, and texts of similar
//! meaning get vectors that point the same way. A reranker scores how well
//! each document answers a query. Both read a whole text at once, so these
//! commands have no chat and no generation.

use std::io::BufRead;
use std::sync::Arc;

use smallm::model::{Embedder, Pooling, normalize};

use super::load_model;
use crate::args::{self, Args, Spec, value};

const SPECS: [Spec; 1] = [value(
    "threads",
    Some('t'),
    "Threads to compute with (default: every physical core)",
)];

pub fn embed_help() -> String {
    format!(
        "Usage: smallm embed <model> [text...]\n\n\
         Prints the embedding of each text, one JSON array per line, scaled\n\
         to length one so that a dot product compares two texts. Without\n\
         texts, each line of standard input is one.\n\n\
         Options:\n{}",
        args::describe(&SPECS)
    )
}

pub fn rerank_help() -> String {
    format!(
        "Usage: smallm rerank <model> <query> <document>...\n\n\
         Scores how well each document answers the query, with a reranking\n\
         model, and prints the documents from best to worst, each after its\n\
         score.\n\n\
         Options:\n{}",
        args::describe(&SPECS)
    )
}

/// Loads an encoder model and makes its embedder.
fn embedder(args: &Args, usage: &str) -> Result<Embedder, String> {
    let name = args
        .positional(0)
        .ok_or(format!("which model? Usage: {usage}"))?;
    let loaded = load_model(name)?;
    if !loaded.model.is_encoder() {
        return Err(format!(
            "{name} generates text: use 'smallm run'. This command needs an embedding or reranking model"
        ));
    }
    let threads = args.number("threads")?.unwrap_or(0);
    Embedder::new(Arc::clone(&loaded.model), threads).map_err(|error| error.to_string())
}

pub fn embed(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &SPECS)?;
    let mut embedder = embedder(&args, "smallm embed <model> [text...]")?;
    if embedder.pooling() == Pooling::Rank {
        return Err("this is a reranking model: use 'smallm rerank'".into());
    }
    let mut texts: Vec<String> = args.rest(1).to_vec();
    if texts.is_empty() {
        for line in std::io::stdin().lock().lines() {
            texts.push(line.map_err(|error| format!("cannot read standard input: {error}"))?);
        }
    }
    let model = Arc::clone(embedder.model());
    for text in texts {
        let tokens = model.tokenizer().encode(&text, true, true);
        let mut vector = embedder.embed(&tokens).map_err(|error| error.to_string())?;
        normalize(&mut vector);
        let numbers: Vec<String> = vector.iter().map(|v| format!("{v:.6}")).collect();
        println!("[{}]", numbers.join(","));
    }
    Ok(())
}

pub fn rerank(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &SPECS)?;
    let usage = "smallm rerank <model> <query> <document>...";
    let mut embedder = embedder(&args, usage)?;
    if embedder.pooling() != Pooling::Rank {
        return Err("this model makes embeddings, not scores: use 'smallm embed'".into());
    }
    let query = args
        .positional(1)
        .ok_or(format!("what query? Usage: {usage}"))?;
    let documents = args.rest(2);
    if documents.is_empty() {
        return Err(format!("which documents? Usage: {usage}"));
    }
    let model = Arc::clone(embedder.model());
    let mut scored = Vec::with_capacity(documents.len());
    for document in documents {
        let tokens = model.tokenizer().encode_pair(query, document);
        let scores = embedder.embed(&tokens).map_err(|error| error.to_string())?;
        scored.push((scores.first().copied().unwrap_or(0.0), document));
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (score, document) in scored {
        println!("{score:>9.4}  {document}");
    }
    Ok(())
}

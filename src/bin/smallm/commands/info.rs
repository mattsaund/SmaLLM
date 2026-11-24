//! `smallm info` and `smallm tokenize`: look inside a model.

use smallm::chat::Template;
use smallm::gguf::{Value, ValueType};
use smallm::model::{LayerSpec, Mixer};
use smallm::tokenizer::Kind;

use super::{describe, human_count, load_model, main_weight_type};
use crate::args::{self, Args, Spec, switch};
use crate::models::human_bytes;

const INFO_SPECS: [Spec; 2] = [
    switch(
        "metadata",
        Some('m'),
        "List every metadata entry in the file",
    ),
    switch(
        "tensors",
        None,
        "List every tensor with its format and shape",
    ),
];

pub fn help() -> String {
    format!(
        "Usage: smallm info <model> [options]\n\n\
         Shows what a model is: its size, shape, format and tokenizer.\n\n\
         Options:\n{}",
        args::describe(&INFO_SPECS)
    )
}

pub fn main(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &INFO_SPECS)?;
    let name = args
        .positional(0)
        .ok_or("which model? Usage: smallm info <model>")?;
    let loaded = load_model(name)?;
    let model = &loaded.model;
    let (config, file, tokenizer) = (model.config(), model.gguf(), model.tokenizer());

    let row = |label: &str, value: String| println!("{label:<14}{value}");
    row("model", describe(model));
    row("architecture", config.arch.clone());
    row("parameters", human_count(model.n_params()));
    row("file size", human_bytes(model.file_size() as u64));
    row("weights", main_weight_type(model).to_string());
    row("layers", config.n_layer.to_string());
    let dense = config
        .layers
        .iter()
        .filter(|layer| layer.n_ff_expert == 0)
        .map(|layer| layer.n_ff);
    row(
        "width",
        format!("{} (feed-forward {})", config.n_embd, spread(dense)),
    );
    if let Some(routing) = &config.routing {
        let layers = config.layers.iter().filter(|layer| layer.n_ff_expert > 0);
        row(
            "experts",
            format!(
                "{} of {} per token, each {} wide, in {} layers",
                routing.used,
                routing.count,
                spread(layers.clone().map(|layer| layer.n_ff_expert)),
                layers.count()
            ),
        );
    }
    let attention: Vec<&LayerSpec> = config
        .layers
        .iter()
        .filter(|layer| layer.mixer == Mixer::Attention)
        .collect();
    if !attention.is_empty() {
        let heads = spread(attention.iter().map(|layer| layer.n_head));
        let widths = spread(attention.iter().map(|layer| layer.head_dim));
        let kv_heads = spread(attention.iter().map(|layer| layer.n_head_kv));
        row(
            "attention",
            format!("{heads} heads of {widths}, {kv_heads} key/value heads"),
        );
    }
    let (n_attention, n_conv) = config.mixers();
    if n_conv > 0 {
        row(
            "blocks",
            format!("{n_attention} attention, {n_conv} short convolution"),
        );
    }
    let windowed: Vec<usize> = attention
        .iter()
        .filter(|layer| layer.window > 0)
        .map(|layer| layer.window)
        .collect();
    if let Some(window) = windowed.first() {
        row(
            "window",
            format!(
                "{window} positions in {} of {} attention layers",
                windowed.len(),
                attention.len()
            ),
        );
    }
    let sharing = attention
        .iter()
        .filter(|layer| layer.kv_source.is_some())
        .count();
    if sharing > 0 {
        row(
            "cache",
            format!("{sharing} layers read the keys and values of earlier ones"),
        );
    }
    row(
        "context",
        format!("{} tokens in training", config.n_ctx_train),
    );
    let kind = match tokenizer.kind() {
        Kind::SentencePiece => "SentencePiece".to_string(),
        Kind::Bpe => format!("BPE ({})", tokenizer.pre_tokenizer().0),
    };
    row(
        "tokenizer",
        format!("{kind}, {} tokens", tokenizer.n_tokens()),
    );
    let template = match model.chat_template() {
        None => "none (plain completion)".to_string(),
        Some(jinja) => match Template::detect(jinja) {
            Some(template) => template.name().to_string(),
            None => "not recognized (use --template or --raw)".to_string(),
        },
    };
    row("chat format", template);
    row(
        "loaded in",
        format!("{:.1} ms", loaded.load_time.as_secs_f64() * 1e3),
    );

    if args.has("metadata") {
        println!("\nmetadata (GGUF version {}):", file.version());
        for (key, value) in file.entries() {
            println!("  {key} = {}", show_value(&value));
        }
    }
    if args.has("tensors") {
        println!("\ntensors:");
        for tensor in file.tensors() {
            let shape: Vec<String> = tensor.dims.iter().map(usize::to_string).collect();
            println!(
                "  {:<44} {:<8} [{}]",
                tensor.name,
                tensor.type_name(),
                shape.join(", ")
            );
        }
    }
    Ok(())
}

/// Shows a per-layer size: one number when every layer agrees, otherwise
/// the smallest and the largest.
fn spread(values: impl Iterator<Item = usize>) -> String {
    let (low, high) = values.fold((usize::MAX, 0), |(low, high), value| {
        (low.min(value), high.max(value))
    });
    match (low, high) {
        (usize::MAX, _) => "none".to_string(),
        (low, high) if low == high => low.to_string(),
        (low, high) => format!("{low} to {high}"),
    }
}

/// Formats a metadata value on one line. Long strings and arrays are cut.
fn show_value(value: &Value<'_>) -> String {
    const LIMIT: usize = 80;
    match value {
        Value::Str(bytes) => {
            let text = String::from_utf8_lossy(bytes).replace('\n', "\\n");
            if text.chars().count() > LIMIT {
                format!(
                    "\"{}...\" ({} bytes)",
                    text.chars().take(LIMIT).collect::<String>(),
                    bytes.len()
                )
            } else {
                format!("\"{text}\"")
            }
        }
        Value::Array(array) => {
            let shown = array.len().min(6);
            let items: Vec<String> = if array.elem_type() == ValueType::String {
                array
                    .strings()
                    .take(shown)
                    .map(|s| format!("{:?}", String::from_utf8_lossy(s)))
                    .collect()
            } else {
                (0..shown)
                    .filter_map(|index| array.number(index))
                    .map(|number| number.to_string())
                    .collect()
            };
            let more = if array.len() > shown { ", ..." } else { "" };
            format!(
                "[{}{more}] ({} {} values)",
                items.join(", "),
                array.len(),
                array.elem_type().name()
            )
        }
        Value::Bool(v) => v.to_string(),
        Value::F32(v) => v.to_string(),
        Value::F64(v) => v.to_string(),
        Value::U8(v) => v.to_string(),
        Value::I8(v) => v.to_string(),
        Value::U16(v) => v.to_string(),
        Value::I16(v) => v.to_string(),
        Value::U32(v) => v.to_string(),
        Value::I32(v) => v.to_string(),
        Value::U64(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
    }
}

const TOKENIZE_SPECS: [Spec; 2] = [
    switch(
        "no-special",
        None,
        "Do not add the beginning token or parse control tokens",
    ),
    switch("ids", None, "Print only the token ids, on one line"),
];

pub fn tokenize_help() -> String {
    format!(
        "Usage: smallm tokenize <model> <text> [options]\n\n\
         Shows the tokens a model reads a text as.\n\n\
         Options:\n{}",
        args::describe(&TOKENIZE_SPECS)
    )
}

pub fn tokenize(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &TOKENIZE_SPECS)?;
    let name = args
        .positional(0)
        .ok_or("which model? Usage: smallm tokenize <model> <text>")?;
    let text = args.rest(1).join(" ");
    let loaded = load_model(name)?;
    let tokenizer = loaded.model.tokenizer();
    let special = !args.has("no-special");
    let tokens = tokenizer.encode(&text, special, special);
    if args.has("ids") {
        let ids: Vec<String> = tokens.iter().map(u32::to_string).collect();
        println!("{}", ids.join(" "));
        return Ok(());
    }
    for token in &tokens {
        let mut piece = Vec::new();
        tokenizer.append_piece(*token, true, &mut piece);
        println!("{token:>8}  {:?}", String::from_utf8_lossy(&piece));
    }
    println!("{} tokens", tokens.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_strings_are_cut() {
        let short = show_value(&Value::Str(b"llama"));
        assert_eq!(short, "\"llama\"");
        let long = "x".repeat(200);
        let shown = show_value(&Value::Str(long.as_bytes()));
        assert!(
            shown.len() < 120 && shown.ends_with("(200 bytes)"),
            "{shown}"
        );
        assert_eq!(show_value(&Value::Str(b"a\nb")), "\"a\\nb\"");
        assert_eq!(show_value(&Value::U32(7)), "7");
        assert_eq!(show_value(&Value::Bool(true)), "true");
    }
}

//! The commands of the `smallm` program, and what they share.

pub mod bench;
pub mod config;
pub mod info;
pub mod pull;
pub mod run;
pub mod serve;
pub mod uninstall;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use smallm::chat::Template;
use smallm::generate::{Finish, Overflow, Stats};
use smallm::tensor::type_name;
use smallm::{Model, SamplerConfig, SessionConfig};

use crate::args::{Args, Spec, switch, value};
use crate::models;

/// Options of every command that runs a model.
pub const SESSION_SPECS: [Spec; 3] = [
    value(
        "threads",
        Some('t'),
        "Threads to compute with (default: one per performance core)",
    ),
    value(
        "ctx",
        Some('c'),
        "Context window in tokens (default: the model's, up to 4096)",
    ),
    value(
        "batch",
        Some('b'),
        "Prompt tokens evaluated together (default: 32)",
    ),
];

/// Options for what happens when the context window fills.
pub const CONTEXT_SPECS: [Spec; 2] = [
    value(
        "overflow",
        None,
        "When the context is full: 'middle' keeps the start and drops the oldest after it, \
         'slide' drops the oldest, 'stop' ends the answer (default: middle)",
    ),
    value(
        "keep",
        None,
        "Tokens at the start that 'middle' keeps (default: the system prompt and first question)",
    ),
];

/// Reads `--overflow` and `--keep`.
pub fn overflow_options(args: &Args) -> Result<(Overflow, Option<usize>), String> {
    let overflow = match args.get("overflow") {
        None => Overflow::default(),
        Some(name) => Overflow::from_name(name).ok_or_else(|| {
            let names: Vec<&str> = Overflow::ALL.iter().map(|choice| choice.name()).collect();
            format!("--overflow is one of {}, not '{name}'", names.join(", "))
        })?,
    };
    Ok((overflow, args.number("keep")?))
}

/// Options of every command that generates text.
pub const SAMPLER_SPECS: [Spec; 9] = [
    value("temp", None, "Randomness, 0 for none (default: 0.8)"),
    value(
        "top-k",
        None,
        "Keep the k most likely tokens, 0 for all (default: 40)",
    ),
    value(
        "top-p",
        None,
        "Keep tokens within this probability mass (default: 0.95)",
    ),
    value(
        "min-p",
        None,
        "Drop tokens below this share of the best (default: 0.05)",
    ),
    value(
        "repeat-penalty",
        None,
        "Penalty on recent tokens, 1 for none (default: 1.0)",
    ),
    value(
        "repeat-last-n",
        None,
        "Tokens the penalties look back (default: 64)",
    ),
    value(
        "frequency-penalty",
        None,
        "Lower a token's score for each recent use (default: 0)",
    ),
    value(
        "presence-penalty",
        None,
        "Lower the score of every recently used token (default: 0)",
    ),
    value("seed", None, "Seed of the random draw (default: random)"),
];

/// Options that choose how a prompt is formatted.
pub const TEMPLATE_SPECS: [Spec; 2] = [
    value(
        "template",
        None,
        "Chat format: chatml, llama3, llama2, mistral, gemma, phi3, zephyr, deepseek",
    ),
    switch(
        "raw",
        None,
        "Complete the prompt as plain text, with no chat format",
    ),
];

/// A model, and how long it took to open.
pub struct Loaded {
    pub model: Arc<Model>,
    pub load_time: Duration,
}

/// Finds and opens the model the user named.
pub fn load_model(name: &str) -> Result<Loaded, String> {
    let path = models::resolve(name)?;
    let start = Instant::now();
    let model = Model::load(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let (pre, known) = model.tokenizer().pre_tokenizer();
    if !known {
        eprintln!(
            "smallm: warning: unknown pre-tokenizer '{pre}'. Text may be split differently from the model's training."
        );
    }
    Ok(Loaded {
        model: Arc::new(model),
        load_time: start.elapsed(),
    })
}

/// Reads the session options.
pub fn session_config(args: &Args) -> Result<SessionConfig, String> {
    Ok(SessionConfig {
        n_ctx: args.number("ctx")?.unwrap_or(0),
        threads: args.number("threads")?.unwrap_or(0),
        batch: args.number("batch")?.unwrap_or(0),
        kernels: None,
    })
}

/// Reads the sampling options. Without a seed, each run draws its own.
pub fn sampler_config(args: &Args) -> Result<SamplerConfig, String> {
    let defaults = SamplerConfig::default();
    let clock_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |time| time.as_nanos() as u64);
    Ok(SamplerConfig {
        temperature: args.number("temp")?.unwrap_or(defaults.temperature),
        top_k: args.number("top-k")?.unwrap_or(defaults.top_k),
        top_p: args.number("top-p")?.unwrap_or(defaults.top_p),
        min_p: args.number("min-p")?.unwrap_or(defaults.min_p),
        repeat_penalty: args
            .number("repeat-penalty")?
            .unwrap_or(defaults.repeat_penalty),
        repeat_last_n: args
            .number("repeat-last-n")?
            .unwrap_or(defaults.repeat_last_n),
        frequency_penalty: args
            .number("frequency-penalty")?
            .unwrap_or(defaults.frequency_penalty),
        presence_penalty: args
            .number("presence-penalty")?
            .unwrap_or(defaults.presence_penalty),
        seed: args.number("seed")?.unwrap_or(clock_seed),
    })
}

/// Picks the chat format: the one asked for, else the model's own. `None`
/// means plain completion.
pub fn chat_template(args: &Args, model: &Model) -> Result<Option<Template>, String> {
    if args.has("raw") {
        return Ok(None);
    }
    if let Some(name) = args.get("template") {
        return Template::from_name(name)
            .map(Some)
            .ok_or_else(|| format!("unknown chat template '{name}'"));
    }
    Ok(model.chat_template().and_then(Template::detect))
}

/// The weight format that holds most of a model's parameters.
pub fn main_weight_type(model: &Model) -> &'static str {
    let mut elements: HashMap<u32, u64> = HashMap::new();
    for tensor in model.gguf().tensors() {
        *elements.entry(tensor.type_id).or_default() += tensor.n_elements() as u64;
    }
    elements
        .into_iter()
        .max_by_key(|&(id, count)| (count, id))
        .map_or("unknown", |(id, _)| type_name(id))
}

/// One line that names a model: name, size and format.
pub fn describe(model: &Model) -> String {
    let config = model.config();
    let name = if config.name.is_empty() {
        config.arch.as_str()
    } else {
        config.name.as_str()
    };
    format!(
        "{name} ({} parameters, {}, {})",
        human_count(model.n_params()),
        main_weight_type(model),
        models::human_bytes(model.file_size() as u64)
    )
}

/// One line on how an answer went: its speed, the prompt's speed, and how
/// much of the context window is in use.
///
/// It reads like
/// `42 tokens at 38.1 tok/s | prompt 23 tokens at 812 tok/s | context 612/4096 (15%), 9.4 MB`.
pub fn stats_line(stats: &Stats) -> String {
    let evaluated = stats.prompt_tokens - stats.reused_tokens;
    let mut prompt = format!(
        "prompt {evaluated} tokens at {:.0} tok/s",
        stats.prompt_speed()
    );
    if stats.reused_tokens > 0 {
        prompt.push_str(&format!(" (+{} reused)", stats.reused_tokens));
    }
    let mut line = format!(
        "{} tokens at {:.1} tok/s | {prompt} | context {}/{} ({:.0}%), {}",
        stats.generated_tokens,
        stats.generate_speed(),
        stats.context_tokens,
        stats.context_size,
        stats.context_used() * 100.0,
        models::human_bytes(stats.context_bytes as u64)
    );
    if stats.dropped_tokens > 0 {
        line.push_str(&format!(", {} oldest tokens dropped", stats.dropped_tokens));
    }
    let stopped = match stats.finish {
        Finish::End | Finish::Stop => None,
        Finish::Length => Some("token limit"),
        Finish::Context => Some("context full"),
        Finish::Canceled => Some("interrupted"),
    };
    if let Some(reason) = stopped {
        line.push_str(&format!(" | stopped: {reason}"));
    }
    line
}

/// Formats a parameter count: `134.5M`.
pub fn human_count(count: u64) -> String {
    match count {
        0..1_000 => count.to_string(),
        1_000..1_000_000 => format!("{:.1}K", count as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1}M", count as f64 / 1e6),
        _ => format!("{:.2}B", count as f64 / 1e9),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stats_line_says_speed_and_context() {
        let stats = Stats {
            prompt_tokens: 512,
            reused_tokens: 489,
            prompt_time: Duration::from_millis(50),
            generated_tokens: 42,
            generate_time: Duration::from_secs(2),
            finish: Finish::Context,
            context_tokens: 554,
            context_size: 4096,
            dropped_tokens: 0,
            context_bytes: 9_400_000,
        };
        assert_eq!(
            stats_line(&stats),
            "42 tokens at 21.0 tok/s | prompt 23 tokens at 460 tok/s (+489 reused) | \
             context 554/4096 (14%), 9.4 MB | stopped: context full"
        );
        let slid = Stats {
            dropped_tokens: 2048,
            finish: Finish::End,
            ..stats
        };
        assert!(stats_line(&slid).ends_with("9.4 MB, 2048 oldest tokens dropped"));
    }

    #[test]
    fn counts_are_readable() {
        assert_eq!(human_count(512), "512");
        assert_eq!(human_count(260_000), "260.0K");
        assert_eq!(human_count(134_515_008), "134.5M");
        assert_eq!(human_count(1_235_814_400), "1.24B");
    }
}

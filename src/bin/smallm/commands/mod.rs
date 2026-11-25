//! The commands of the `smallm` program, and what they share.

pub mod bench;
pub mod config;
pub mod embed;
pub mod info;
pub mod pull;
pub mod run;
pub mod serve;
pub mod uninstall;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use smallm::chat::{Format, Template, Thinking};
use smallm::generate::{Finish, Overflow, Stats};
use smallm::model::CacheType;
use smallm::tensor::type_name;
use smallm::{Model, SamplerConfig, SessionConfig};

use crate::args::{Args, Spec, switch, value};
use crate::models;

/// Options of every command that runs a model.
pub const SESSION_SPECS: [Spec; 7] = [
    value(
        "threads",
        Some('t'),
        "Threads to generate with (default: one per performance core)",
    ),
    value(
        "threads-batch",
        None,
        "Threads to read prompts with (default: every core, or --threads when set)",
    ),
    value(
        "ctx",
        Some('c'),
        "Context window in tokens (default: the model's, up to 4096)",
    ),
    value(
        "batch",
        Some('b'),
        "Prompt tokens evaluated together (default: 64 on x86 and 32 elsewhere, or 128 for models \
         with experts)",
    ),
    value(
        "cache-type",
        None,
        "What the key/value cache keeps: f16 (default, half the memory) or f32",
    ),
    value(
        "cpus",
        None,
        "Run on these CPUs only, such as 6-7 or 0,2,4-5. Thread counts follow them. \
         Linux and Android (default: all)",
    ),
    switch(
        "mlock",
        None,
        "Read the whole model into memory at load and keep it there, for slow storage. \
         Needs a locked-memory limit (ulimit -l) as large as the file",
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
/// Sampling options. Where the model file recommends a setting, that is the
/// default, else the built-in one in parentheses.
pub const SAMPLER_SPECS: [Spec; 19] = [
    value(
        "temp",
        None,
        "Randomness, 0 for none (default: the model's, else 0.8)",
    ),
    value(
        "top-k",
        None,
        "Keep the k most likely tokens, 0 for all (default: the model's, else 40)",
    ),
    value(
        "top-p",
        None,
        "Keep tokens within this probability mass (default: the model's, else 0.95)",
    ),
    value(
        "min-p",
        None,
        "Drop tokens below this share of the best (default: the model's, else 0.05)",
    ),
    value(
        "repeat-penalty",
        None,
        "Penalty on recent tokens, 1 for none (default: the model's, else 1.0)",
    ),
    value(
        "repeat-last-n",
        None,
        "Tokens the penalties look back (default: the model's, else 64)",
    ),
    value(
        "typical",
        None,
        "Keep tokens of typical surprise up to this mass, 1 for all (default: 1)",
    ),
    value(
        "top-nsigma",
        None,
        "Keep tokens within n standard deviations of the best logit, 0 for all (default: 0)",
    ),
    value(
        "xtc-probability",
        None,
        "How often XTC drops the most likely tokens, 0 for never (default: the model's, else 0)",
    ),
    value(
        "xtc-threshold",
        None,
        "Probability a token needs for XTC to drop it (default: the model's, else 0.1)",
    ),
    value(
        "dynatemp-range",
        None,
        "Let the temperature move this far with the entropy, 0 for fixed (default: 0)",
    ),
    value(
        "dynatemp-exp",
        None,
        "How the dynamic temperature follows the entropy (default: 1)",
    ),
    value(
        "mirostat",
        None,
        "Mirostat version, 0 for off (default: the model's, else 0)",
    ),
    value(
        "mirostat-ent",
        None,
        "Mirostat's target surprise, tau (default: the model's, else 5)",
    ),
    value(
        "mirostat-lr",
        None,
        "Mirostat's learning rate, eta (default: the model's, else 0.1)",
    ),
    value(
        "logit-bias",
        None,
        "Add to a token's logit, as TOKEN+BIAS or TOKEN-BIAS (-inf bans it). Repeatable",
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
pub const TEMPLATE_SPECS: [Spec; 3] = [
    value(
        "template",
        None,
        "Chat format: chatml, llama3, llama2, mistral, gemma, gemma4, granite, phi3, phi4-mini, \
         zephyr, deepseek, deepseek2, kimi, harmony, rwkv-world",
    ),
    switch(
        "raw",
        None,
        "Complete the prompt as plain text, with no chat format",
    ),
    value(
        "think",
        None,
        "Whether a reasoning model thinks first: on or off, or low, medium or high for gpt-oss, \
         which cannot turn it off (default: as the model does)",
    ),
];

/// A model, and how long it took to open.
pub struct Loaded {
    pub model: Arc<Model>,
    pub load_time: Duration,
}

/// Finds and opens the model the user named. `--cpus` applies before the
/// model opens, so every thread started after it stays on those CPUs, and
/// `--mlock` reads the weights in once the model is open.
pub fn load_model(name: &str, args: &Args) -> Result<Loaded, String> {
    if let Some(list) = args.get("cpus") {
        smallm::threads::run_on_cpus(list).map_err(|error| format!("--cpus: {error}"))?;
    }
    let path = models::resolve(name)?;
    let start = Instant::now();
    let model = Model::load(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if args.has("mlock") {
        let size = models::human_bytes(model.file_size() as u64);
        model.lock_in_memory().map_err(|error| {
            format!(
                "--mlock: could not keep the model's {size} in memory: {error}. The limit is \
                 'ulimit -l'. Raise it, or set LimitMEMLOCK= for a service."
            )
        })?;
        eprintln!("locked the model's {size} in memory");
    }
    let (pre, known) = model.tokenizer().pre_tokenizer();
    if !known {
        // Older files name none, and then llama.cpp's default is used too.
        let which = if pre.is_empty() {
            "the file names no pre-tokenizer, so the default one is used".to_string()
        } else {
            format!("unknown pre-tokenizer '{pre}'")
        };
        eprintln!(
            "smallm: warning: {which}. Text may be split differently from the model's training."
        );
    }
    Ok(Loaded {
        model: Arc::new(model),
        load_time: start.elapsed(),
    })
}

/// The thread counts of a session, for status lines: "8 threads" or
/// "8 threads, 20 for prompts".
pub fn thread_counts(session: &smallm::Session) -> String {
    let threads = match session.threads() {
        1 => "1 thread".to_string(),
        n => format!("{n} threads"),
    };
    if session.batch_threads() == session.threads() {
        threads
    } else {
        format!("{threads}, {} for prompts", session.batch_threads())
    }
}

/// Reads the session options.
pub fn session_config(args: &Args) -> Result<SessionConfig, String> {
    Ok(SessionConfig {
        n_ctx: args.number("ctx")?.unwrap_or(0),
        threads: args.number("threads")?.unwrap_or(0),
        batch_threads: args.number("threads-batch")?.unwrap_or(0),
        batch: args.number("batch")?.unwrap_or(0),
        kernels: None,
        cache: match args.get("cache-type") {
            None => CacheType::default(),
            Some(name) => CacheType::from_name(name)
                .ok_or_else(|| format!("unknown cache type '{name}'. Known: f16, f32"))?,
        },
    })
}

/// Reads the sampling options. Without a seed, each run draws its own.
/// Reads the sampling options. A setting not given on the command line or
/// in the settings file takes the value the model file recommends, if any,
/// else the built-in default. Without a seed, each run draws its own.
pub fn sampler_config(args: &Args, model: &Model) -> Result<SamplerConfig, String> {
    let defaults = SamplerConfig::from_model_file(model.gguf());
    let clock_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |time| time.as_nanos() as u64);
    let mut logit_bias = Vec::new();
    for entry in args.all("logit-bias") {
        logit_bias.push(parse_logit_bias(&entry)?);
    }
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
        typical_p: args.number("typical")?.unwrap_or(defaults.typical_p),
        top_n_sigma: args.number("top-nsigma")?.unwrap_or(defaults.top_n_sigma),
        xtc_probability: args
            .number("xtc-probability")?
            .unwrap_or(defaults.xtc_probability),
        xtc_threshold: args
            .number("xtc-threshold")?
            .unwrap_or(defaults.xtc_threshold),
        dynatemp_range: args
            .number("dynatemp-range")?
            .unwrap_or(defaults.dynatemp_range),
        dynatemp_exponent: args
            .number("dynatemp-exp")?
            .unwrap_or(defaults.dynatemp_exponent),
        mirostat: match args.number::<u8>("mirostat")? {
            Some(version) if version > 2 => {
                return Err(format!("--mirostat is 0, 1 or 2, not {version}"));
            }
            Some(version) => version,
            None => defaults.mirostat,
        },
        mirostat_tau: args
            .number("mirostat-ent")?
            .unwrap_or(defaults.mirostat_tau),
        mirostat_eta: args.number("mirostat-lr")?.unwrap_or(defaults.mirostat_eta),
        logit_bias,
        seed: args.number("seed")?.unwrap_or(clock_seed),
    })
}

/// Reads a `--logit-bias` entry, `TOKEN+BIAS` or `TOKEN-BIAS`, as llama.cpp
/// writes it. `-inf` bans the token.
pub fn parse_logit_bias(entry: &str) -> Result<(u32, f32), String> {
    let split = entry
        .find(['+', '-'])
        .ok_or_else(|| format!("--logit-bias wants TOKEN+BIAS or TOKEN-BIAS, not '{entry}'"))?;
    let token = entry[..split]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("--logit-bias: '{}' is not a token id", &entry[..split]))?;
    let bias = entry[split..]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("--logit-bias: '{}' is not a number", &entry[split..]))?;
    Ok((token, bias))
}

/// Picks the chat format: the one asked for, else the model's own. `None`
/// means plain completion.
/// The chat format of a run: `--template`, or the one the model's own
/// template is recognized as, with the choice of `--think`. `None` for plain
/// text.
pub fn chat_template(args: &Args, model: &Model) -> Result<Option<Format>, String> {
    if args.has("raw") {
        return Ok(None);
    }
    let thinking = match args.get("think") {
        None => Thinking::Default,
        Some(name) => Thinking::from_name(name).ok_or_else(|| {
            format!("--think is on, off, low, medium, high or auto, not '{name}'")
        })?,
    };
    let template = match args.get("template") {
        Some(name) => Template::from_name(name).ok_or_else(|| {
            let known: Vec<&str> = Template::ALL.iter().map(|t| t.name()).collect();
            format!(
                "unknown chat template '{name}'. Known: {}",
                known.join(", ")
            )
        })?,
        None => match model.chat_template().and_then(Template::detect) {
            Some(template) => template,
            None => return Ok(None),
        },
    };
    let format = Format::new(template, model.chat_template(), thinking);
    if args.given("think") && thinking != Thinking::Default && !format.can_switch_thinking() {
        eprintln!(
            "smallm: this model's chat format has no switch for thinking, so --think does nothing"
        );
    }
    Ok(Some(format))
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
    fn logit_bias_entries_are_read_as_llama_cpp_writes_them() {
        assert_eq!(parse_logit_bias("15043+1"), Ok((15043, 1.0)));
        assert_eq!(parse_logit_bias("7-0.5"), Ok((7, -0.5)));
        assert_eq!(parse_logit_bias("3-inf"), Ok((3, f32::NEG_INFINITY)));
        assert!(parse_logit_bias("15043").is_err());
        assert!(parse_logit_bias("x+1").is_err());
        assert!(parse_logit_bias("4+y").is_err());
    }

    #[test]
    fn the_template_help_names_every_chat_format() {
        let help = TEMPLATE_SPECS[0].help;
        for template in Template::ALL {
            assert!(help.contains(template.name()), "{}", template.name());
        }
    }

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

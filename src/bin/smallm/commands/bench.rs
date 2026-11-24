//! `smallm bench`: measure how fast a model runs on this machine.
//!
//! The benchmark feeds a fixed synthetic prompt and then generates with
//! greedy sampling, so two runs do the same work and can be compared.

use std::time::Instant;

use smallm::Session;
use smallm::model::Phase;
use smallm::sampler::argmax;

use super::{SESSION_SPECS, describe, load_model, session_config};
use crate::args::{self, Args, Spec, switch, value};
use crate::models::human_bytes;

const SPECS: [Spec; 4] = [
    value(
        "prompt-tokens",
        Some('p'),
        "Tokens in the prompt (default: 256)",
    ),
    value("gen-tokens", Some('n'), "Tokens to generate (default: 128)"),
    value("repeat", Some('r'), "Runs to take the best of (default: 3)"),
    switch(
        "profile",
        None,
        "Show where generation time goes, stage by stage",
    ),
];

fn specs() -> Vec<Spec> {
    SPECS.into_iter().chain(SESSION_SPECS).collect()
}

pub fn help() -> String {
    format!(
        "Usage: smallm bench <model> [options]\n\n\
         Measures prompt reading and generation speed.\n\n\
         Options:\n{}",
        args::describe(&specs())
    )
}

/// The speeds of one benchmark run, in tokens per second.
struct Run {
    prompt: f64,
    generate: f64,
}

pub fn main(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &specs())?;
    let name = args
        .positional(0)
        .ok_or("which model? Usage: smallm bench <model>")?;
    let n_prompt: usize = args.number("prompt-tokens")?.unwrap_or(256).max(1);
    let n_gen: usize = args.number("gen-tokens")?.unwrap_or(128);
    let repeat: usize = args.number("repeat")?.unwrap_or(3).max(1);

    let loaded = load_model(name)?;
    let model = &loaded.model;
    let memory_loaded = memory_in_use();
    let mut config = session_config(&args)?;
    if config.n_ctx == 0 {
        config.n_ctx = n_prompt + n_gen + 8;
    }
    if config.n_ctx < n_prompt + n_gen {
        return Err(format!(
            "a context of {} cannot hold {} tokens",
            config.n_ctx,
            n_prompt + n_gen
        ));
    }
    let mut session = Session::new(model.clone(), config).map_err(|error| error.to_string())?;
    let memory_ready = memory_in_use();

    // Token ids that cycle through the low end of the vocabulary. Their
    // meaning does not matter, only that every run sees the same ones.
    let span = model.config().n_vocab.clamp(2, 1000) - 1;
    let prompt: Vec<u32> = (0..n_prompt)
        .map(|index| (1 + (index * 7919) % span) as u32)
        .collect();

    // One short pass pages the weights in and starts the threads.
    session
        .eval(&prompt[..n_prompt.min(8)])
        .map_err(|error| error.to_string())?;

    let mut best = Run {
        prompt: 0.0,
        generate: 0.0,
    };
    for _ in 0..repeat {
        let run = measure(&mut session, &prompt, n_gen)?;
        best.prompt = best.prompt.max(run.prompt);
        best.generate = best.generate.max(run.generate);
    }

    let row = |label: &str, value: String| println!("{label:<10}{value}");
    row("model", describe(model));
    row(
        "engine",
        format!(
            "smallm {}, {} kernels, {}",
            smallm::VERSION,
            session.kernels(),
            super::thread_counts(&session)
        ),
    );
    row(
        "load",
        format!("{:.1} ms", loaded.load_time.as_secs_f64() * 1e3),
    );
    row(
        "prompt",
        format!("{:>8.1} tokens/s  ({n_prompt} tokens)", best.prompt),
    );
    if n_gen > 0 {
        row(
            "generate",
            format!("{:>8.1} tokens/s  ({n_gen} tokens)", best.generate),
        );
    }
    row(
        "cache",
        format!(
            "{} for {} tokens",
            human_bytes(session.cache_bytes() as u64),
            session.tokens().len()
        ),
    );

    if let (Some(loaded), Some(ready), Some(now)) = (memory_loaded, memory_ready, memory_in_use()) {
        row(
            "memory",
            format!(
                "{} with the model loaded, {} with a session, {} now",
                human_bytes(loaded.own),
                human_bytes(ready.own),
                human_bytes(now.own)
            ),
        );
        row(
            "",
            format!(
                "plus {} of the model file, which the system can drop",
                human_bytes(now.file)
            ),
        );
    }

    if args.has("profile") && n_gen > 0 {
        profile(&mut session, &prompt, n_gen)?;
    }
    Ok(())
}

/// What the process holds in memory.
struct Memory {
    /// Bytes the process allocated itself.
    own: u64,
    /// Bytes of mapped files in memory: the model. The system can drop
    /// these pages at any time and read them again from the file.
    file: u64,
}

/// Reads the memory in use from the kernel. Only Linux reports it in this
/// form.
fn memory_in_use() -> Option<Memory> {
    if !cfg!(any(target_os = "linux", target_os = "android")) {
        return None;
    }
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kilobytes = |name: &str| {
        let line = status.lines().find(|line| line.starts_with(name))?;
        line[name.len()..]
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse::<u64>()
            .ok()
    };
    // A model on a memory file system is counted as shared memory instead
    // of as a file, so the two are added.
    Some(Memory {
        own: kilobytes("RssAnon:")? * 1024,
        file: (kilobytes("RssFile:")? + kilobytes("RssShmem:").unwrap_or(0)) * 1024,
    })
}

/// Times each stage of the forward pass, first while the prompt is read
/// and then while tokens are generated.
fn profile(session: &mut Session, prompt: &[u32], n_gen: usize) -> Result<(), String> {
    session.clear();
    session.set_profiling(true);
    session.eval(prompt).map_err(|error| error.to_string())?;
    if let Some(profile) = session.profile() {
        print_profile("reading the prompt", &profile, prompt.len());
    }

    session.set_profiling(true);
    for _ in 0..n_gen {
        let next = argmax(session.logits());
        session.eval(&[next]).map_err(|error| error.to_string())?;
    }
    if let Some(profile) = session.profile() {
        print_profile("generation", &profile, n_gen);
    }
    session.set_profiling(false);
    Ok(())
}

/// Prints the time each stage took, per token.
fn print_profile(what: &str, profile: &smallm::model::Profile, tokens: usize) {
    let total = profile.total().as_secs_f64();
    let per_token = |time: std::time::Duration| time.as_secs_f64() * 1e6 / tokens.max(1) as f64;
    println!("\n{what}, time per token by stage:");
    for (phase, time) in Phase::ALL.iter().zip(profile.phases) {
        let share = 100.0 * time.as_secs_f64() / total.max(1e-12);
        println!(
            "  {:<18}{:>9.1} us  {share:>5.1}%",
            phase.name(),
            per_token(time)
        );
    }
    println!("  {:<18}{:>9.1} us", "total", per_token(profile.total()));
    println!(
        "  of which quantizing activations: {:.1} us",
        per_token(profile.quantize)
    );
}

/// Reads the prompt into an empty context, then generates greedily.
fn measure(session: &mut Session, prompt: &[u32], n_gen: usize) -> Result<Run, String> {
    session.clear();
    let start = Instant::now();
    session.eval(prompt).map_err(|error| error.to_string())?;
    let prompt_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..n_gen {
        let next = argmax(session.logits());
        session.eval(&[next]).map_err(|error| error.to_string())?;
    }
    let generate_time = start.elapsed().as_secs_f64();
    Ok(Run {
        prompt: prompt.len() as f64 / prompt_time.max(1e-9),
        generate: if n_gen == 0 {
            0.0
        } else {
            n_gen as f64 / generate_time.max(1e-9)
        },
    })
}

//! `smallm run`: answer one prompt, or chat.
//!
//! With a prompt on the command line, in a file or piped in, the model
//! answers once and the program exits. Without one, it opens a chat that
//! keeps the conversation in context from turn to turn.

use std::io::{self, BufRead, IsTerminal, Read, Write};

use smallm::chat::{Message, Role, Template};
use smallm::generate::{self, Options, Stats};
use smallm::{Model, Session};

use super::{
    CONTEXT_SPECS, SAMPLER_SPECS, SESSION_SPECS, TEMPLATE_SPECS, chat_template, describe,
    load_model, sampler_config, stats_line,
};
use crate::args::{self, Args, Spec, switch, value};
use crate::term;

const SPECS: [Spec; 7] = [
    value(
        "max-tokens",
        Some('n'),
        "Stop after this many tokens (default: no limit)",
    ),
    value(
        "system",
        Some('s'),
        "A system message that sets the model's behavior",
    ),
    value("file", Some('f'), "Read the prompt from a file"),
    value("stop", None, "Stop when this text appears. May be repeated"),
    switch("show-special", None, "Show control tokens in the output"),
    switch(
        "verbose",
        Some('v'),
        "Print how the model loaded, and the speed after every answer",
    ),
    switch(
        "quiet",
        Some('q'),
        "Print only the answer, without the speed and context line",
    ),
];

/// Tokens kept free for the answer when a long chat is trimmed to fit.
const ANSWER_RESERVE: usize = 256;

fn specs() -> Vec<Spec> {
    SPECS
        .into_iter()
        .chain(SESSION_SPECS)
        .chain(CONTEXT_SPECS)
        .chain(SAMPLER_SPECS)
        .chain(TEMPLATE_SPECS)
        .collect()
}

pub fn help() -> String {
    format!(
        "Usage: smallm run <model> [prompt] [options]\n\n\
         Answers the prompt and exits. Without a prompt, opens a chat.\n\
         The model is a .gguf file or the name of a downloaded model.\n\n\
         Options:\n{}",
        args::describe(&specs())
    )
}

pub fn main(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &specs())?;
    let name = args
        .positional(0)
        .ok_or("which model? Usage: smallm run <model> [prompt]")?;
    let verbose = args.has("verbose");
    // The line after each answer goes to stderr, so it is shown where a
    // person reads it and stays out of output that is piped on.
    let report = Report {
        stats: verbose || (term::stderr_is_terminal() && !args.has("quiet")),
    };

    let loaded = load_model(name)?;
    let model = &loaded.model;
    let template = chat_template(&args, model)?;
    let mut session =
        Session::new(model.clone(), super::session_config(&args)?).map_err(|e| e.to_string())?;
    let (overflow, keep) = super::overflow_options(&args)?;
    let options = Options {
        max_tokens: args.number("max-tokens")?.unwrap_or(0),
        sampler: sampler_config(&args)?,
        stop: args.all("stop"),
        show_special: args.has("show-special"),
        overflow,
        keep,
    };
    if verbose {
        eprintln!(
            "{}\nloaded in {:.1} ms, {}, {} kernels, context {}",
            describe(model),
            loaded.load_time.as_secs_f64() * 1e3,
            super::thread_counts(&session),
            session.kernels(),
            session.n_ctx()
        );
    }
    term::install_interrupt_handler();

    let system = args.get("system").map(str::to_string);
    match one_shot_prompt(&args)? {
        Some(prompt) => {
            let stats = match template {
                Some(template) => {
                    let mut messages: Vec<Message> = system
                        .iter()
                        .map(|text| Message::new(Role::System, text))
                        .collect();
                    messages.push(Message::new(Role::User, prompt));
                    answer(&mut session, template, &messages, &options)?.1
                }
                None => complete(&mut session, &prompt, &options)?,
            };
            println!();
            report.answer(&stats);
            Ok(())
        }
        None => chat(&mut session, model, template, system, &options, report),
    }
}

/// What to print about each answer.
#[derive(Clone, Copy)]
struct Report {
    /// The speed and context line.
    stats: bool,
}

impl Report {
    fn answer(self, stats: &Stats) {
        if self.stats {
            eprintln!("{}", term::paint("2", &stats_line(stats)));
        }
    }
}

/// The prompt of a one-shot run: from the command line, a file, or a pipe.
/// `None` when the user should be asked interactively.
fn one_shot_prompt(args: &Args) -> Result<Option<String>, String> {
    if let Some(path) = args.get("file") {
        return std::fs::read_to_string(path)
            .map(Some)
            .map_err(|error| format!("{path}: {error}"));
    }
    let words = args.rest(1);
    if !words.is_empty() {
        return Ok(Some(words.join(" ")));
    }
    if !io::stdin().is_terminal() {
        let mut prompt = String::new();
        io::stdin()
            .read_to_string(&mut prompt)
            .map_err(|error| format!("reading the prompt: {error}"))?;
        return Ok(Some(prompt));
    }
    Ok(None)
}

/// Prints a piece of the answer as it arrives. Returns false to stop the
/// generation, after Ctrl+C or when nobody is reading the output any more.
fn print_piece(piece: &str) -> bool {
    let mut out = io::stdout().lock();
    out.write_all(piece.as_bytes())
        .and_then(|()| out.flush())
        .is_ok()
        && !term::interrupted()
}

/// Completes a plain text prompt.
fn complete(session: &mut Session, prompt: &str, options: &Options) -> Result<Stats, String> {
    term::begin_generation();
    let result = generate::run(session, prompt, options, print_piece);
    term::end_generation();
    result.map_err(|error| error.to_string())
}

/// Answers a conversation. Returns the answer and how it went.
fn answer(
    session: &mut Session,
    template: Template,
    messages: &[Message],
    options: &Options,
) -> Result<(String, Stats), String> {
    let tokenizer = session.model().tokenizer();
    let (tokens, dropped) = generate::chat_tokens_fitting(
        tokenizer,
        template,
        messages,
        session.n_ctx(),
        ANSWER_RESERVE,
        options.overflow,
    );
    // In a chat, the start that `middle` keeps is the opening: the system
    // prompt and the first question.
    let mut options = options.clone();
    options.keep = options.keep.or_else(|| {
        Some(generate::opening_tokens(
            tokenizer, template, messages, &tokens,
        ))
    });
    let options = &options;
    if dropped > 0 {
        eprintln!("smallm: the context is full, so the {dropped} oldest messages were left out");
    }
    let mut reply = String::new();
    term::begin_generation();
    let result = generate::run_tokens(session, &tokens, options, |piece| {
        reply.push_str(piece);
        print_piece(piece)
    });
    term::end_generation();
    let stats = result.map_err(|error| error.to_string())?;
    Ok((reply, stats))
}

const CHAT_HELP: &str = "\
Commands:
  /clear            forget the conversation
  /system <text>    set the system message and start over
  /help             show this list
  /exit             leave (Ctrl+D works too)
Start a message with \"\"\" to write several lines, and end it the same way.
Ctrl+C stops an answer.";

/// The interactive chat.
fn chat(
    session: &mut Session,
    model: &Model,
    template: Option<Template>,
    mut system: Option<String>,
    options: &Options,
    report: Report,
) -> Result<(), String> {
    eprintln!("{}", describe(model));
    match template {
        Some(_) => eprintln!("Type a message, or /help for commands."),
        None => eprintln!(
            "This model has no chat format, so each line you type is completed as plain text."
        ),
    }
    let mut messages: Vec<Message> = Vec::new();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!(">>> ");
        let _ = io::stderr().flush();
        let Some(input) = read_message(&mut lines) else {
            break;
        };
        let input = input.trim();
        match input {
            "" => continue,
            "/exit" | "/quit" | "/bye" => break,
            "/help" | "/?" => {
                eprintln!("{CHAT_HELP}");
                continue;
            }
            "/clear" => {
                messages.clear();
                session.clear();
                eprintln!("Conversation cleared.");
                continue;
            }
            _ => {}
        }
        if let Some(text) = input.strip_prefix("/system") {
            system = Some(text.trim().to_string()).filter(|text| !text.is_empty());
            messages.clear();
            session.clear();
            eprintln!(
                "System message {}.",
                if system.is_some() { "set" } else { "removed" }
            );
            continue;
        }
        if input.starts_with('/') {
            eprintln!("Unknown command. Type /help for the list.");
            continue;
        }

        let stats = match template {
            Some(template) => {
                messages.push(Message::new(Role::User, input));
                let mut prompt: Vec<Message> = system
                    .iter()
                    .map(|text| Message::new(Role::System, text))
                    .collect();
                prompt.extend(messages.iter().cloned());
                let (reply, stats) = answer(session, template, &prompt, options)?;
                messages.push(Message::new(Role::Assistant, reply));
                stats
            }
            None => complete(session, input, options)?,
        };
        println!();
        report.answer(&stats);
    }
    Ok(())
}

/// Reads one message: a line, or several lines between `"""` marks.
/// `None` at the end of input.
fn read_message(lines: &mut impl Iterator<Item = io::Result<String>>) -> Option<String> {
    let first = lines.next()?.ok()?;
    let Some(rest) = first.trim_start().strip_prefix("\"\"\"") else {
        return Some(first);
    };
    if let Some(single) = rest.strip_suffix("\"\"\"") {
        return Some(single.to_string());
    }
    let mut message = rest.to_string();
    for line in lines {
        let line = line.ok()?;
        message.push('\n');
        match line.strip_suffix("\"\"\"") {
            Some(last) => {
                message.push_str(last);
                break;
            }
            None => message.push_str(&line),
        }
    }
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(input: &str) -> Vec<String> {
        let mut lines = input.lines().map(|line| Ok(line.to_string()));
        std::iter::from_fn(|| read_message(&mut lines)).collect()
    }

    #[test]
    fn messages_are_lines_or_quoted_blocks() {
        assert_eq!(read("one\ntwo"), ["one", "two"]);
        assert_eq!(
            read("\"\"\"first\nsecond\nthird\"\"\"\nafter"),
            ["first\nsecond\nthird", "after"]
        );
        assert_eq!(read("\"\"\"inline\"\"\""), ["inline"]);
        assert_eq!(read("\"\"\"\ncode\n\"\"\""), ["\ncode\n"]);
        // An unfinished block ends with the input.
        assert_eq!(read("\"\"\"open\nstill"), ["open\nstill"]);
        assert!(read("").is_empty());
    }

    #[test]
    fn the_prompt_comes_from_words_or_a_file() {
        let parse =
            |words: &[&str]| Args::parse(words.iter().map(|w| w.to_string()), &specs()).unwrap();
        let args = parse(&["model", "What", "is", "this?"]);
        assert_eq!(
            one_shot_prompt(&args).unwrap().as_deref(),
            Some("What is this?")
        );

        let path = std::env::temp_dir().join(format!("smallm-prompt-{}", std::process::id()));
        std::fs::write(&path, "from a file").unwrap();
        let args = parse(&["model", "-f", path.to_str().unwrap()]);
        assert_eq!(
            one_shot_prompt(&args).unwrap().as_deref(),
            Some("from a file")
        );
        std::fs::remove_file(&path).unwrap();
        assert!(one_shot_prompt(&parse(&["model", "-f", "/nonexistent/prompt"])).is_err());
    }

    #[test]
    fn every_option_has_a_unique_name() {
        let specs = specs();
        for (index, spec) in specs.iter().enumerate() {
            for other in &specs[index + 1..] {
                assert_ne!(spec.long, other.long);
                assert!(
                    spec.short.is_none() || spec.short != other.short,
                    "-{:?}",
                    spec.short
                );
            }
        }
        assert!(help().contains("--max-tokens"));
    }
}

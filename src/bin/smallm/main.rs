//! The `smallm` command line program.
//!
//! | File          | Role                                               |
//! |---------------|----------------------------------------------------|
//! | `args.rs`     | Parses options.                                    |
//! | `commands/`   | One file per command.                              |
//! | `server/`     | The HTTP API behind `smallm serve`.                |
//! | `models.rs`   | Finds downloaded models.                           |
//! | `json.rs`     | JSON for the API and the downloader.               |
//! | `term.rs`     | Ctrl+C handling, and what the terminal can show.   |
//! | `screen.rs`   | A scrolling log with one progress bar under it.    |
//! | `sha256.rs`   | Checks that a download arrived intact.             |
//! | `config.rs`   | The settings file, `smallm.conf`.                  |

mod args;
mod commands;
mod config;
mod json;
mod models;
mod screen;
mod server;
mod sha256;
mod term;

use std::process::ExitCode;

const HELP: &str = "\
smallm: a small, fast LLM engine

Usage: smallm <command> [arguments]

Commands:
  run <model> [prompt]   Answer a prompt, or chat when none is given
  serve <model>          Serve an OpenAI-compatible API
  pull <user/repo>       Download a model from Hugging Face
  list                   List downloaded models
  rm <model>             Delete a downloaded model
  info <model>           Show what a model is
  bench <model>          Measure speed on this machine
  tokenize <model> <text>  Show how a model reads a text
  config                 Show the settings file, or reset it with --reset
  uninstall              Remove SmaLLM and the models it downloaded
  version                Print the version

Run 'smallm <command> --help' for the options of a command.
";

fn main() -> ExitCode {
    config::init();
    let mut raw = std::env::args().skip(1);
    let Some(command) = raw.next() else {
        print!("{HELP}");
        return ExitCode::SUCCESS;
    };
    let rest: Vec<String> = raw.collect();
    let wants_help = rest
        .iter()
        .take_while(|arg| *arg != "--")
        .any(|arg| arg == "--help" || arg == "-h");
    let rest = rest.into_iter();

    let result = match (command.as_str(), wants_help) {
        ("help" | "--help" | "-h", _) => show(HELP.to_string()),
        ("version" | "--version" | "-V", _) => show(format!("smallm {}\n", smallm::VERSION)),
        ("run", true) => show(commands::run::help()),
        ("serve", true) => show(commands::serve::help()),
        ("pull", true) => show(commands::pull::pull_help()),
        ("info", true) => show(commands::info::help()),
        ("bench", true) => show(commands::bench::help()),
        ("tokenize", true) => show(commands::info::tokenize_help()),
        ("uninstall", true) => show(commands::uninstall::help()),
        ("config", true) => show(commands::config::help()),
        ("run", false) => commands::run::main(rest),
        ("serve", false) => commands::serve::main(rest),
        ("pull", false) => commands::pull::pull(rest),
        ("list" | "ls", _) => commands::pull::list(rest),
        ("rm" | "remove", _) => commands::pull::remove(rest),
        ("info", false) => commands::info::main(rest),
        ("bench", false) => commands::bench::main(rest),
        ("tokenize", false) => commands::info::tokenize(rest),
        ("uninstall", false) => commands::uninstall::main(rest),
        ("config", false) => commands::config::main(rest),
        (other, _) => Err(format!(
            "unknown command '{other}'. Run 'smallm help' for the list."
        )),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("smallm: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Prints help or version text.
fn show(text: String) -> Result<(), String> {
    print!("{text}");
    Ok(())
}

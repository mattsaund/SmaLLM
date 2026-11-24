//! `smallm config`: shows the settings file, and resets it.

use crate::args::{self, Args, Spec, switch};
use crate::config::{self, SETTINGS};

const SPECS: [Spec; 2] = [
    switch(
        "reset",
        None,
        "Write the default settings, replacing what the file holds",
    ),
    switch(
        "path",
        None,
        "Print where the settings file is, and nothing else",
    ),
];

pub fn help() -> String {
    format!(
        "Usage: smallm config [options]\n\n\
         Shows the settings in effect and where they are kept. The settings\n\
         file holds every setting of the program, one `name = value` per line,\n\
         named like the command-line options. An option on the command line\n\
         wins over the file. The first run writes the file with the defaults.\n\n\
         Options:\n{}",
        args::describe(&SPECS)
    )
}

pub fn main(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &SPECS)?;
    let path = config::path();
    if args.has("path") {
        println!("{}", path.display());
        return Ok(());
    }
    if args.has("reset") {
        config::write_defaults()?;
        println!("Reset {} to the default settings.", path.display());
        return Ok(());
    }
    if !path.exists() {
        config::write_defaults()?;
        println!("Wrote the default settings to {}.", path.display());
    }
    // What a fresh read of the file gives, which is what the next command
    // will use.
    let settings = config::get();
    println!("Settings in {}:", path.display());
    for (name, default, _) in SETTINGS {
        let value = settings.and_then(|settings| settings.value(name));
        let shown = match value {
            Some(value) if value != default => format!("{value}  (default: {})", show(default)),
            _ => show(default),
        };
        println!("  {name:<18} {shown}");
    }
    println!("Edit the file to change them, or run 'smallm config --reset' to start over.");
    Ok(())
}

/// A default as it reads best in a listing.
fn show(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
        value.to_string()
    }
}

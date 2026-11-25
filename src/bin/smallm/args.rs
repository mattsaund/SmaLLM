//! Command line argument parsing.
//!
//! Each command declares the options it accepts as a list of [`Spec`]s.
//! Anything else on the command line is an error, so a mistyped option is
//! reported instead of ignored.

use std::str::FromStr;

/// One option a command accepts.
pub struct Spec {
    /// The long name, used as `--name`.
    pub long: &'static str,
    /// An optional one-letter name, used as `-n`.
    pub short: Option<char>,
    /// True when the option is followed by a value.
    pub takes_value: bool,
    /// One line for the help text.
    pub help: &'static str,
}

/// Shorthand for an option that takes a value.
pub const fn value(long: &'static str, short: Option<char>, help: &'static str) -> Spec {
    Spec {
        long,
        short,
        takes_value: true,
        help,
    }
}

/// Shorthand for an on/off option.
pub const fn switch(long: &'static str, short: Option<char>, help: &'static str) -> Spec {
    Spec {
        long,
        short,
        takes_value: false,
        help,
    }
}

/// The parsed arguments of one command.
#[derive(Debug, Default)]
pub struct Args {
    positional: Vec<String>,
    options: Vec<(&'static str, Option<String>)>,
    /// The options the settings file filled in.
    from_settings: Vec<&'static str>,
}

impl Args {
    /// Parses `raw` against the options in `specs`.
    pub fn parse(raw: impl Iterator<Item = String>, specs: &[Spec]) -> Result<Args, String> {
        let mut args = Args::default();
        let mut raw = raw.peekable();
        let mut options_ended = false;
        while let Some(arg) = raw.next() {
            if options_ended || arg == "-" || !arg.starts_with('-') {
                args.positional.push(arg);
                continue;
            }
            if arg == "--" {
                options_ended = true;
                continue;
            }
            // `--name=value` carries its value inline.
            let (name, inline) = match arg.split_once('=') {
                Some((name, value)) if arg.starts_with("--") => (name, Some(value.to_string())),
                _ => (arg.as_str(), None),
            };
            let spec = specs
                .iter()
                .find(|spec| match name.strip_prefix("--") {
                    Some(long) => spec.long == long,
                    None => {
                        let mut letters = name[1..].chars();
                        letters.next() == spec.short && letters.next().is_none()
                    }
                })
                .ok_or_else(|| format!("unknown option '{name}'"))?;
            let value = match (spec.takes_value, inline) {
                (true, Some(value)) => Some(value),
                (true, None) => Some(
                    raw.next()
                        .ok_or_else(|| format!("option '{name}' needs a value"))?,
                ),
                (false, None) => None,
                (false, Some(_)) => return Err(format!("option '{name}' does not take a value")),
            };
            args.options.push((spec.long, value));
        }
        // Settings from the settings file fill in options not given here.
        if let Some(settings) = crate::config::get() {
            for spec in specs.iter().filter(|spec| spec.takes_value) {
                if let (false, Some(value)) = (args.has(spec.long), settings.value(spec.long)) {
                    args.options.push((spec.long, Some(value.to_string())));
                    args.from_settings.push(spec.long);
                }
            }
        }
        Ok(args)
    }

    /// True when `long` was given on the command line, rather than filled
    /// in from the settings file.
    pub fn given(&self, long: &str) -> bool {
        self.has(long) && !self.from_settings.contains(&long)
    }

    /// The positional argument at `index`.
    pub fn positional(&self, index: usize) -> Option<&str> {
        self.positional.get(index).map(String::as_str)
    }

    /// Every positional argument from `index` on.
    pub fn rest(&self, index: usize) -> &[String] {
        self.positional.get(index..).unwrap_or_default()
    }

    /// True when the on/off option `long` was given.
    pub fn has(&self, long: &str) -> bool {
        self.options.iter().any(|(name, _)| *name == long)
    }

    /// The value of option `long`. The last one wins when it is repeated.
    pub fn get(&self, long: &str) -> Option<&str> {
        self.options
            .iter()
            .rev()
            .find(|(name, _)| *name == long)
            .and_then(|(_, value)| value.as_deref())
    }

    /// Every value of a repeatable option, in order.
    pub fn all(&self, long: &str) -> Vec<String> {
        self.options
            .iter()
            .filter(|(name, _)| *name == long)
            .filter_map(|(_, value)| value.clone())
            .collect()
    }

    /// The value of option `long`, parsed as a number.
    pub fn number<T: FromStr>(&self, long: &str) -> Result<Option<T>, String> {
        match self.get(long) {
            None => Ok(None),
            Some(text) => text
                .parse()
                .map(Some)
                .map_err(|_| format!("option '--{long}': '{text}' is not a valid number")),
        }
    }
}

/// Formats the options of a command for its help text.
pub fn describe(specs: &[Spec]) -> String {
    let mut out = String::new();
    for spec in specs {
        let short = spec
            .short
            .map_or("    ".to_string(), |letter| format!("-{letter}, "));
        let name = if spec.takes_value {
            format!("--{} <value>", spec.long)
        } else {
            format!("--{}", spec.long)
        };
        out.push_str(&format!("  {short}{name:<26} {}\n", spec.help));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: [Spec; 3] = [
        value("threads", Some('t'), "threads"),
        value("stop", None, "stop string"),
        switch("verbose", Some('v'), "more output"),
    ];

    fn parse(words: &[&str]) -> Result<Args, String> {
        Args::parse(words.iter().map(|word| word.to_string()), &SPECS)
    }

    #[test]
    fn options_and_positionals_mix_freely() {
        let args = parse(&[
            "model.gguf",
            "-t",
            "4",
            "hello world",
            "--verbose",
            "--stop=END",
        ])
        .unwrap();
        assert_eq!(args.positional(0), Some("model.gguf"));
        assert_eq!(args.positional(1), Some("hello world"));
        assert_eq!(args.positional(2), None);
        assert_eq!(args.number::<usize>("threads"), Ok(Some(4)));
        assert!(args.has("verbose"));
        assert_eq!(args.get("stop"), Some("END"));
        assert_eq!(args.rest(1), ["hello world"]);
        assert!(args.rest(5).is_empty());
    }

    #[test]
    fn repeated_options_keep_every_value() {
        let args = parse(&["--stop", "a", "--stop", "b", "-t", "1", "-t", "2"]).unwrap();
        assert_eq!(args.all("stop"), ["a", "b"]);
        assert_eq!(args.get("threads"), Some("2"));
    }

    #[test]
    fn a_double_dash_ends_the_options() {
        let args = parse(&["--", "-t", "--verbose"]).unwrap();
        assert_eq!(args.rest(0), ["-t", "--verbose"]);
        assert!(!args.has("verbose"));
        // A lone dash is a positional, by convention standard input.
        assert_eq!(parse(&["-"]).unwrap().positional(0), Some("-"));
    }

    #[test]
    fn mistakes_are_reported() {
        assert!(parse(&["--nope"]).unwrap_err().contains("--nope"));
        assert!(parse(&["-x"]).is_err());
        assert!(parse(&["-tv"]).is_err());
        assert!(parse(&["--threads"]).unwrap_err().contains("needs a value"));
        assert!(parse(&["--verbose=yes"]).is_err());
        assert!(
            parse(&["-t", "many"])
                .unwrap()
                .number::<usize>("threads")
                .is_err()
        );
        assert_eq!(parse(&[]).unwrap().number::<usize>("threads"), Ok(None));
    }

    #[test]
    fn help_lists_every_option() {
        let text = describe(&SPECS);
        assert!(text.contains("-t, --threads <value>"));
        assert!(text.contains("    --stop <value>"));
        assert!(text.contains("-v, --verbose"));
    }
}

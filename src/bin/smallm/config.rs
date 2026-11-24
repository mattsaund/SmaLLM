//! The settings file, `smallm.conf`: every setting of the program in one
//! place.
//!
//! Each line is `name = value`, where the name is the long form of a
//! command-line option. An option given on the command line wins over the
//! file, and the file wins over the built-in default. `auto`, or no value,
//! means the built-in default. `smallm config --reset` writes the file with
//! the defaults and a comment on each setting.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Every setting the file can hold: its name, its default and what it
/// does. The names are those of the command-line options they stand for.
pub const SETTINGS: [(&str, &str, &str); 23] = [
    (
        "ctx",
        "auto",
        "Context window in tokens. auto: the model's training context, up to 4096.",
    ),
    (
        "threads",
        "auto",
        "Threads to generate with. auto: one per performance core.",
    ),
    (
        "threads-batch",
        "auto",
        "Threads to read prompts with. auto: every physical core, or threads when set.",
    ),
    (
        "batch",
        "32",
        "Prompt tokens read together. More reads prompts faster and takes more memory.",
    ),
    (
        "cache-type",
        "f16",
        "What the key/value cache keeps. f16: half the memory. f32: exact.",
    ),
    (
        "overflow",
        "middle",
        "When the context is full. middle: keep the start and drop the oldest tokens\n\
         # after it. slide: drop the oldest tokens. stop: end the answer.",
    ),
    (
        "keep",
        "auto",
        "Tokens at the start that middle keeps. auto: the system prompt and the first question.",
    ),
    ("temp", "0.8", "Randomness of the answers, 0 for none."),
    (
        "top-k",
        "40",
        "Choose among the k most likely tokens, 0 for all.",
    ),
    (
        "top-p",
        "0.95",
        "Choose among the tokens that make up this much of the probability.",
    ),
    (
        "min-p",
        "0.05",
        "Leave out tokens less likely than this share of the best one.",
    ),
    (
        "repeat-penalty",
        "1.0",
        "Penalty on recent tokens, 1 for none.",
    ),
    (
        "repeat-last-n",
        "64",
        "How many tokens back the penalties look.",
    ),
    (
        "frequency-penalty",
        "0",
        "Lower a token's score for each recent use of it.",
    ),
    (
        "presence-penalty",
        "0",
        "Lower the score of every token used recently.",
    ),
    (
        "seed",
        "auto",
        "Seed of the random choices. auto: a new one every time.",
    ),
    (
        "max-tokens",
        "auto",
        "Stop an answer after this many tokens. auto: no limit.",
    ),
    (
        "system",
        "",
        "A system message for every chat. Empty for none.",
    ),
    (
        "template",
        "auto",
        "Chat format. auto: the one the model file describes.",
    ),
    (
        "host",
        "127.0.0.1",
        "Address smallm serve listens on. 127.0.0.1 is this machine only.",
    ),
    ("port", "8080", "Port smallm serve listens on."),
    (
        "api-key",
        "",
        "Key smallm serve requires as a bearer token. Empty for none.",
    ),
    (
        "models",
        "auto",
        "Folder downloaded models are kept in. auto: the system's place for app data.",
    ),
];

/// The settings in effect for this run.
pub struct Config {
    /// Each setting in the file, in file order.
    values: Vec<(String, String)>,
}

static LOADED: OnceLock<Config> = OnceLock::new();

/// Reads the settings file, once, and reports lines it cannot use. Until
/// this runs, as in tests, no settings apply.
pub fn init() {
    LOADED.get_or_init(|| {
        let text = std::fs::read_to_string(path()).unwrap_or_default();
        let (values, problems) = parse(&text);
        for problem in problems {
            eprintln!("smallm: {}: {problem}", path().display());
        }
        Config { values }
    });
}

/// The settings of this run, once [`init`] has read them.
pub fn get() -> Option<&'static Config> {
    LOADED.get()
}

impl Config {
    /// The value of setting `name`, unless it is unset or `auto`.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .rev()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
            .filter(|value| !value.is_empty() && *value != "auto")
    }
}

/// Where the settings file is. `SMALLM_CONFIG` overrides the default,
/// which follows each system's convention for settings.
pub fn path() -> PathBuf {
    if let Some(path) = std::env::var_os("SMALLM_CONFIG").filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        var("APPDATA")
            .or_else(|| var("USERPROFILE").map(|home| home.join("AppData").join("Roaming")))
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| home.join("Library").join("Application Support"))
    } else {
        var("XDG_CONFIG_HOME").or_else(|| var("HOME").map(|home| home.join(".config")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("smallm")
        .join("smallm.conf")
}

/// Reads `name = value` lines. Returns the settings and a description of
/// each line that could not be used.
fn parse(text: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut values = Vec::new();
    let mut problems = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            problems.push(format!("line {}: expected 'name = value'", index + 1));
            continue;
        };
        let name = name.trim();
        if !SETTINGS.iter().any(|(known, _, _)| *known == name) {
            problems.push(format!("line {}: unknown setting '{name}'", index + 1));
            continue;
        }
        // Quotes keep spaces at the ends of a value.
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|inner| inner.strip_suffix('"'))
            .unwrap_or(value);
        values.push((name.to_string(), value.to_string()));
    }
    (values, problems)
}

/// The settings file with every setting at its default.
pub fn defaults() -> String {
    let mut text = String::from(
        "# SmaLLM settings.\n\
         #\n\
         # Each line is `name = value`, and a line that starts with # is a comment.\n\
         # An option on the command line wins over the value here. auto, or no\n\
         # value, means the built-in default. `smallm config --reset` writes this\n\
         # file again with the defaults.\n",
    );
    for (name, value, about) in SETTINGS {
        text.push_str(&format!("\n# {about}\n{name} = {value}\n"));
    }
    text
}

/// Writes the default settings to the settings file, making its folder.
pub fn write_defaults() -> Result<PathBuf, String> {
    let path = path();
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)
            .map_err(|error| format!("could not make {}: {error}", folder.display()))?;
    }
    std::fs::write(&path, defaults())
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_become_settings() {
        let (values, problems) = parse(
            "# a comment\n\nctx = 8192\n  temp=0.2  \nsystem = \" Be brief. \"\nport = auto\n\
             bogus = 1\nno equals sign\n",
        );
        let config = Config { values };
        assert_eq!(config.value("ctx"), Some("8192"));
        assert_eq!(config.value("temp"), Some("0.2"));
        assert_eq!(config.value("system"), Some(" Be brief. "));
        assert_eq!(config.value("port"), None, "auto is the default");
        assert_eq!(config.value("threads"), None);
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(problems[0].contains("bogus") && problems[1].contains("line 8"));
    }

    #[test]
    fn the_defaults_read_back_as_defaults() {
        let (values, problems) = parse(&defaults());
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(values.len(), SETTINGS.len());
        let config = Config { values };
        // Every automatic or empty setting reads as unset.
        assert_eq!(config.value("ctx"), None);
        assert_eq!(config.value("system"), None);
        assert_eq!(config.value("overflow"), Some("middle"));
        assert_eq!(config.value("batch"), Some("32"));
    }
}

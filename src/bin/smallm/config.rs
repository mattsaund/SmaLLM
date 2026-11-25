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
pub const SETTINGS: [(&str, &str, &str); 34] = [
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
        "auto",
        "Prompt tokens read together. More reads prompts faster and takes more memory.\n\
         # auto: 64 on x86 and 32 elsewhere, or 128 for models with experts.",
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
    (
        "temp",
        "auto",
        "Randomness of the answers, 0 for none. auto: the model file's, else 0.8.",
    ),
    (
        "top-k",
        "auto",
        "Choose among the k most likely tokens, 0 for all. auto: the model file's, else 40.",
    ),
    (
        "top-p",
        "auto",
        "Choose among the tokens that make up this much of the probability.\n\
         # auto: the model file's, else 0.95.",
    ),
    (
        "min-p",
        "auto",
        "Leave out tokens less likely than this share of the best one.\n\
         # auto: the model file's, else 0.05.",
    ),
    (
        "repeat-penalty",
        "auto",
        "Penalty on recent tokens, 1 for none. auto: the model file's, else 1.",
    ),
    (
        "repeat-last-n",
        "auto",
        "How many tokens back the penalties look. auto: the model file's, else 64.",
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
        "typical",
        "1",
        "Keep the tokens of typical surprise, up to this much probability. 1 for all.",
    ),
    (
        "top-nsigma",
        "0",
        "Keep tokens within this many standard deviations of the best one. 0 for all.",
    ),
    (
        "xtc-probability",
        "auto",
        "How often XTC drops the most likely tokens, 0 for never.\n\
         # auto: the model file's, else 0.",
    ),
    (
        "xtc-threshold",
        "auto",
        "Probability a token needs for XTC to drop it. auto: the model file's, else 0.1.",
    ),
    (
        "dynatemp-range",
        "0",
        "Let the temperature move this far with the uncertainty, 0 for a fixed one.",
    ),
    (
        "dynatemp-exp",
        "1",
        "How the dynamic temperature follows the uncertainty, 1 for linear.",
    ),
    (
        "mirostat",
        "auto",
        "Mirostat version, 0 for off. auto: the model file's, else 0.",
    ),
    (
        "mirostat-ent",
        "auto",
        "Mirostat's target surprise. auto: the model file's, else 5.",
    ),
    (
        "mirostat-lr",
        "auto",
        "Mirostat's learning rate. auto: the model file's, else 0.1.",
    ),
    (
        "seed",
        "auto",
        "Seed of the random choices. auto: a new one every time.",
    ),
    (
        "max-tokens",
        "auto",
        "Stop an answer after this many tokens, reasoning included. auto: no limit.",
    ),
    (
        "system",
        "",
        "A system message for every chat. Empty for none.",
    ),
    (
        "cpus",
        "auto",
        "The CPUs to run on, such as 6-7 or 0,2,4-5. Linux and Android. auto: all of them.",
    ),
    (
        "template",
        "auto",
        "Chat format. auto: the one the model file describes.",
    ),
    (
        "think",
        "auto",
        "Whether a reasoning model thinks first: on or off, or low, medium or high\n\
         # for gpt-oss and Inkling. auto: as the model does.",
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

/// Values that SmaLLM 0.3.0 and earlier wrote into every settings file, for
/// settings whose default is `auto` now. In a file those versions wrote,
/// they are the old defaults, not choices. Read as written, they would
/// override the model file's sampling settings and the larger prompt batch.
const OLD_DEFAULTS: [(&str, &str); 7] = [
    ("batch", "32"),
    ("temp", "0.8"),
    ("top-k", "40"),
    ("top-p", "0.95"),
    ("min-p", "0.05"),
    ("repeat-penalty", "1.0"),
    ("repeat-last-n", "64"),
];

/// Reads the settings file, once, and reports lines it cannot use. Until
/// this runs, as in tests, no settings apply.
pub fn init() {
    LOADED.get_or_init(|| {
        let text = std::fs::read_to_string(path()).unwrap_or_default();
        let (mut values, problems) = parse(&text);
        for problem in problems {
            eprintln!("smallm: {}: {problem}", path().display());
        }
        forget_old_defaults(&text, &mut values);
        Config { values }
    });
}

/// Reads the [`OLD_DEFAULTS`] as `auto` in a file an older version wrote.
/// Such a file starts with the header this program writes and has no
/// `mirostat` line, which every file written since has. A value the owner
/// changed is kept, and so is every value in a file written by hand.
fn forget_old_defaults(text: &str, values: &mut [(String, String)]) {
    let written_by_us = text.starts_with("# SmaLLM settings.");
    let older = written_by_us && !values.iter().any(|(name, _)| name == "mirostat");
    if !older {
        return;
    }
    for (name, value) in values.iter_mut() {
        if OLD_DEFAULTS.contains(&(name.as_str(), value.as_str())) {
            *value = "auto".to_string();
        }
    }
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
        assert_eq!(config.value("batch"), None);
    }

    #[test]
    fn old_defaults_in_an_older_file_read_as_auto() {
        let older = "# SmaLLM settings.\n\nbatch = 32\ntemp = 0.8\ntop-k = 50\nmin-p = 0.05\n";
        let read = |text: &str| {
            let (mut values, _) = parse(text);
            forget_old_defaults(text, &mut values);
            Config { values }
        };
        let config = read(older);
        assert_eq!(config.value("batch"), None);
        assert_eq!(config.value("temp"), None);
        assert_eq!(config.value("min-p"), None);
        assert_eq!(config.value("top-k"), Some("50"), "a changed value stays");
        // A file written since, or by hand, means what it says.
        let newer = format!("{older}mirostat = auto\n");
        assert_eq!(read(&newer).value("temp"), Some("0.8"));
        assert_eq!(read("temp = 0.8\n").value("temp"), Some("0.8"));
        // The defaults written today are never taken for old ones.
        let today = read(&defaults());
        assert!(SETTINGS.iter().all(|(name, value, _)| {
            today.value(name).is_none() || today.value(name) == Some(value)
        }));
    }
}

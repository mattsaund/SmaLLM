//! The writing rules of this repository, enforced.
//!
//! Every text file in the repository is checked for two things:
//!
//! - Em dashes, middle dots and bullet characters. Sentences here are joined
//!   with commas, periods, colons and parentheses instead.
//! - British spellings. The project writes American English.
//!
//! The files under `tests/golden` are exempt. They are model output and
//! tokenizer test input, not prose.

use std::path::{Path, PathBuf};

/// Characters that may not appear, with their names.
const BANNED: [(char, &str); 3] = [
    ('\u{2014}', "an em dash"),
    ('\u{00b7}', "a middle dot"),
    ('\u{2022}', "a bullet character"),
];

/// Directories that are not searched.
const SKIPPED_DIRS: [&str; 6] = ["target", ".git", ".claude", "__pycache__", "dist", "models"];

/// File extensions that hold text someone wrote.
const TEXT_EXTENSIONS: [&str; 12] = [
    "rs", "md", "toml", "sh", "ps1", "py", "c", "h", "yml", "yaml", "txt", "json",
];

/// Files without an extension that are checked as well.
const TEXT_NAMES: [&str; 2] = ["LICENSE", ".gitignore"];

/// British words, matched from their first letter, and the American
/// spelling of each.
const BRITISH_STARTS: [(&str, &str); 46] = [
    ("colour", "color"),
    ("behaviour", "behavior"),
    ("favour", "favor"),
    ("honour", "honor"),
    ("neighbour", "neighbor"),
    ("flavour", "flavor"),
    ("labour", "labor"),
    ("humour", "humor"),
    ("rumour", "rumor"),
    ("harbour", "harbor"),
    ("cancelled", "canceled"),
    ("cancelling", "canceling"),
    ("labelled", "labeled"),
    ("labelling", "labeling"),
    ("modelled", "modeled"),
    ("modelling", "modeling"),
    ("travelled", "traveled"),
    ("travelling", "traveling"),
    ("signalled", "signaled"),
    ("signalling", "signaling"),
    ("levelled", "leveled"),
    ("fuelled", "fueled"),
    ("channelled", "channeled"),
    ("totalled", "totaled"),
    ("centre", "center"),
    ("metre", "meter"),
    ("kilometre", "kilometer"),
    ("millimetre", "millimeter"),
    ("litre", "liter"),
    ("fibre", "fiber"),
    ("theatre", "theater"),
    ("licence", "license"),
    ("defence", "defense"),
    ("offence", "offense"),
    ("greyscale", "grayscale"),
    ("artefact", "artifact"),
    ("catalogue", "catalog"),
    ("analogue", "analog"),
    ("whilst", "while"),
    ("amongst", "among"),
    ("ageing", "aging"),
    ("sceptic", "skeptic"),
    ("speciality", "specialty"),
    ("manoeuvre", "maneuver"),
    ("skilful", "skillful"),
    ("instalment", "installment"),
];

/// British words that only count as a whole word.
const BRITISH_WORDS: [(&str, &str); 11] = [
    ("grey", "gray"),
    ("greys", "grays"),
    ("greyed", "grayed"),
    ("programme", "program"),
    ("programmes", "programs"),
    ("analyse", "analyze"),
    ("analysed", "analyzed"),
    ("analysing", "analyzing"),
    ("analyser", "analyzer"),
    ("fulfil", "fulfill"),
    ("enrol", "enroll"),
];

/// Stems of the verbs that end in "ize" in American English. A word that
/// continues one of them with "ise", "isa" or "isi" is the British form.
const IZE_STEMS: [&str; 55] = [
    "initial",
    "normal",
    "optim",
    "serial",
    "token",
    "quant",
    "organ",
    "recogn",
    "real",
    "summar",
    "general",
    "special",
    "synchron",
    "priorit",
    "custom",
    "util",
    "emphas",
    "apolog",
    "categor",
    "character",
    "visual",
    "equal",
    "final",
    "standard",
    "vector",
    "parallel",
    "memor",
    "random",
    "linear",
    "author",
    "penal",
    "stabil",
    "local",
    "capital",
    "central",
    "minim",
    "maxim",
    "critic",
    "digit",
    "anonym",
    "modular",
    "virtual",
    "material",
    "neutral",
    "regular",
    "discret",
    "factor",
    "diagonal",
    "canonical",
    "modern",
    "familiar",
    "popular",
    "symbol",
    "item",
    "harmon",
];

/// The American spelling to suggest for `word`, when `word` is British.
fn american(word: &str) -> Option<String> {
    for (british, american) in BRITISH_WORDS {
        if word == british {
            return Some(american.to_string());
        }
    }
    for (british, american) in BRITISH_STARTS {
        if let Some(rest) = word.strip_prefix(british) {
            return Some(format!("{american}{rest}"));
        }
    }
    for stem in IZE_STEMS {
        let Some(start) = word.find(stem) else {
            continue;
        };
        let Some(rest) = word[start + stem.len()..].strip_prefix("is") else {
            continue;
        };
        if rest.starts_with('e') || rest.starts_with('a') || rest.starts_with("ing") {
            return Some(format!("{}{stem}iz{rest}", &word[..start]));
        }
    }
    None
}

/// Collects every text file under `dir`.
fn text_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        if path.is_dir() {
            let is_golden = name == "golden" && dir.ends_with("tests");
            if !SKIPPED_DIRS.contains(&name.as_str()) && !is_golden {
                text_files(&path, found);
            }
            continue;
        }
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default();
        if TEXT_EXTENSIONS.contains(&extension) || TEXT_NAMES.contains(&name.as_str()) {
            found.push(path);
        }
    }
}

/// The text files of the repository, with their paths relative to its root.
fn repository_text() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    text_files(root, &mut paths);
    paths.sort();
    assert!(
        paths.len() > 20,
        "found only {} text files under {}",
        paths.len(),
        root.display()
    );
    paths
        .into_iter()
        .filter_map(|path| {
            let name = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            // A file that is not UTF-8 is not prose.
            let text = std::fs::read_to_string(&path).ok()?;
            Some((name, text))
        })
        .collect()
}

#[test]
fn no_em_dashes_or_dot_separators() {
    let mut problems = Vec::new();
    for (name, text) in repository_text() {
        for (number, line) in text.lines().enumerate() {
            for (character, what) in BANNED {
                if line.contains(character) {
                    problems.push(format!("{name}:{}: {what}", number + 1));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "characters this project does not use:\n{}",
        problems.join("\n")
    );
}

#[test]
fn spelling_is_american() {
    let mut problems = Vec::new();
    for (name, text) in repository_text() {
        // This file has to name the words it looks for.
        if name == "tests/writing.rs" {
            continue;
        }
        for (number, line) in text.lines().enumerate() {
            let lower = line.to_lowercase();
            for word in lower
                .split(|c: char| !c.is_ascii_alphabetic())
                .filter(|word| word.len() > 3)
            {
                if let Some(american) = american(word) {
                    problems.push(format!(
                        "{name}:{}: '{word}' should be '{american}'",
                        number + 1
                    ));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "British spellings:\n{}",
        problems.join("\n")
    );
}

#[test]
fn the_spelling_check_knows_its_words() {
    assert_eq!(american("colour").as_deref(), Some("color"));
    assert_eq!(american("behaviours").as_deref(), Some("behaviors"));
    assert_eq!(american("cancelled").as_deref(), Some("canceled"));
    assert_eq!(american("normalise").as_deref(), Some("normalize"));
    assert_eq!(american("tokenisation").as_deref(), Some("tokenization"));
    assert_eq!(american("deserialising").as_deref(), Some("deserializing"));
    assert_eq!(american("unauthorised").as_deref(), Some("unauthorized"));
    assert_eq!(american("grey").as_deref(), Some("gray"));
    // American words that look close to a British one.
    for word in [
        "color",
        "canceled",
        "normalize",
        "organism",
        "optimism",
        "realistic",
        "emphasis",
        "specialist",
        "parallelism",
        "characteristic",
        "criticism",
        "finalist",
        "analyses",
        "programmer",
        "greyhound",
        "center",
        "otherwise",
        "precise",
        "promise",
        "raise",
        "bitwise",
        "pairwise",
    ] {
        assert_eq!(american(word), None, "{word}");
    }
}

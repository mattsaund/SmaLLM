//! Where downloaded models live, and how a model name becomes a file.

use std::path::{Path, PathBuf};

/// The folder downloaded models are kept in.
///
/// `SMALLM_MODELS`, then the `models` setting, override the default, which
/// follows each system's convention for application data.
pub fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SMALLM_MODELS").filter(|dir| !dir.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(dir) = crate::config::get().and_then(|settings| settings.value("models")) {
        return PathBuf::from(dir);
    }
    let var = |name: &str| {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        var("LOCALAPPDATA")
            .or_else(|| var("USERPROFILE").map(|home| home.join("AppData").join("Local")))
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|home| home.join("Library").join("Application Support"))
    } else {
        var("XDG_DATA_HOME").or_else(|| var("HOME").map(|home| home.join(".local").join("share")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("smallm")
        .join("models")
}

/// One downloaded model.
pub struct Entry {
    /// The file name without its `.gguf` extension.
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
}

/// The models in `dir`, sorted by name.
pub fn list(dir: &Path) -> Vec<Entry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut models: Vec<Entry> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path
                .file_name()?
                .to_str()?
                .strip_suffix(".gguf")?
                .to_string();
            let bytes = path.metadata().ok()?.len();
            Some(Entry { name, path, bytes })
        })
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Turns what the user typed into a model file.
///
/// A path to an existing file is used as it is. Anything else is looked up
/// among the downloaded models: first by exact name, then as a part of a
/// name, ignoring case, when that singles out one model.
pub fn resolve(name: &str) -> Result<PathBuf, String> {
    let path = Path::new(name);
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    resolve_in(&models_dir(), name)
}

/// The lookup half of [`resolve`], in a given folder.
pub fn resolve_in(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let models = list(dir);
    let wanted = name.strip_suffix(".gguf").unwrap_or(name);
    if let Some(exact) = models.iter().find(|model| model.name == wanted) {
        return Ok(exact.path.clone());
    }
    let lower = wanted.to_lowercase();
    let matches: Vec<&Entry> = models
        .iter()
        .filter(|model| model.name.to_lowercase().contains(&lower))
        .collect();
    match matches.as_slice() {
        [only] => Ok(only.path.clone()),
        [] => Err(format!(
            "no model named '{name}'. Give a path to a .gguf file, or download one with 'smallm pull'."
        )),
        several => {
            let names: Vec<&str> = several.iter().map(|model| model.name.as_str()).collect();
            Err(format!(
                "'{name}' matches several models: {}",
                names.join(", ")
            ))
        }
    }
}

/// Formats a byte count for people: `105.5 MB`.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch folder that is removed when the test ends.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("smallm-models-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        fn add(&self, name: &str, bytes: usize) {
            std::fs::write(self.0.join(name), vec![0u8; bytes]).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn lists_only_model_files() {
        let dir = TempDir::new("list");
        dir.add("b-model.gguf", 20);
        dir.add("a-model.gguf", 10);
        dir.add("notes.txt", 5);
        dir.add("partial.gguf.part", 5);
        let models = list(&dir.0);
        let names: Vec<&str> = models.iter().map(|model| model.name.as_str()).collect();
        assert_eq!(names, ["a-model", "b-model"]);
        assert_eq!(models[0].bytes, 10);
        assert!(list(&dir.0.join("missing")).is_empty());
    }

    #[test]
    fn names_resolve_exactly_then_by_part() {
        let dir = TempDir::new("resolve");
        dir.add("SmolLM2-135M-Instruct-Q4_K_M.gguf", 1);
        dir.add("SmolLM2-360M-Instruct-Q4_K_M.gguf", 1);
        dir.add("qwen.gguf", 1);
        let file =
            |name: &str| resolve_in(&dir.0, name).map(|path| path.file_name().unwrap().to_owned());
        assert_eq!(file("qwen").unwrap(), "qwen.gguf");
        assert_eq!(file("qwen.gguf").unwrap(), "qwen.gguf");
        assert_eq!(file("135m").unwrap(), "SmolLM2-135M-Instruct-Q4_K_M.gguf");
        assert!(file("smollm2").unwrap_err().contains("several"));
        assert!(file("llama").unwrap_err().contains("smallm pull"));
    }

    #[test]
    fn an_existing_path_wins() {
        let dir = TempDir::new("path");
        dir.add("direct.gguf", 1);
        let path = dir.0.join("direct.gguf");
        assert_eq!(resolve(path.to_str().unwrap()).unwrap(), path);
    }

    #[test]
    fn byte_counts_are_readable() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1500), "1.5 KB");
        assert_eq!(human_bytes(105_454_432), "105.5 MB");
        assert_eq!(human_bytes(4_700_000_000), "4.7 GB");
    }
}

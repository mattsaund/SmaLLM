//! `smallm pull`, `list` and `rm`: manage downloaded models.
//!
//! Downloads go through `curl`, which every supported system ships. That
//! keeps a TLS stack and an HTTP client out of this program.

use std::path::Path;
use std::process::{Command, Stdio};

use crate::args::Args;
use crate::json::Json;
use crate::models::{self, human_bytes};

/// Quantizations to prefer when the user names none, best first. Q4_K_M is
/// the usual balance of size and quality.
const PREFERRED: [&str; 5] = ["q4_k_m", "q4_0", "q5_k_m", "q8_0", "q6_k"];

pub fn pull_help() -> String {
    "Usage: smallm pull <user/repo>[:quantization]\n       smallm pull <url>\n\n\
     Downloads a GGUF model from Hugging Face, or from any URL.\n\n\
     Examples:\n\
     \x20 smallm pull bartowski/SmolLM2-135M-Instruct-GGUF\n\
     \x20 smallm pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0\n"
        .to_string()
}

/// What `smallm pull` was asked to fetch.
#[derive(Debug, PartialEq, Eq)]
enum Source {
    /// A direct link to a model file.
    Url(String),
    /// A Hugging Face repository and an optional quantization tag.
    Repo { repo: String, tag: Option<String> },
}

/// Parses the argument of `smallm pull`.
fn parse_source(arg: &str) -> Result<Source, String> {
    if arg.starts_with("https://") || arg.starts_with("http://") {
        return Ok(Source::Url(arg.to_string()));
    }
    let arg = arg
        .strip_prefix("hf.co/")
        .or_else(|| arg.strip_prefix("huggingface.co/"))
        .unwrap_or(arg);
    let (repo, tag) = match arg.split_once(':') {
        Some((repo, tag)) => (repo, Some(tag.to_string()).filter(|tag| !tag.is_empty())),
        None => (arg, None),
    };
    // Repository names are a user and a name, in a small alphabet.
    let is_name = |part: &str| {
        !part.is_empty()
            && !part.starts_with('.')
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    match repo.split_once('/') {
        Some((user, name)) if is_name(user) && is_name(name) => Ok(Source::Repo {
            repo: repo.to_string(),
            tag,
        }),
        _ => Err(format!(
            "'{arg}' is not a repository like user/name, nor a URL"
        )),
    }
}

/// Picks the model file to download from a repository listing.
///
/// `files` are `(path, size)` pairs. Returns the path, or an error that
/// lists what the repository holds.
fn choose_file(files: &[(String, u64)], tag: Option<&str>) -> Result<String, String> {
    // Vision projectors and pieces of split models are not whole models.
    let models: Vec<&(String, u64)> = files
        .iter()
        .filter(|(path, _)| {
            let lower = path.to_lowercase();
            lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.contains("-of-")
        })
        .collect();
    if models.is_empty() {
        return Err("the repository has no GGUF model files".to_string());
    }
    let matching = |tag: &str| -> Vec<&String> {
        let tag = tag.to_lowercase();
        models
            .iter()
            .map(|(path, _)| path)
            .filter(|path| path.to_lowercase().contains(&tag))
            .collect()
    };
    let available = || {
        let names: Vec<String> = models
            .iter()
            .map(|(path, size)| format!("  {path} ({})", human_bytes(*size)))
            .collect();
        names.join("\n")
    };
    match tag {
        Some(tag) => match matching(tag).as_slice() {
            [] => Err(format!(
                "no file matches '{tag}'. The repository has:\n{}",
                available()
            )),
            // Several matches: the shortest name is the plain variant.
            found => Ok(found
                .iter()
                .min_by_key(|path| path.len())
                .map(|path| path.to_string())
                .unwrap_or_default()),
        },
        None => {
            for preferred in PREFERRED {
                if let Some(path) = matching(preferred)
                    .into_iter()
                    .min_by_key(|path| path.len())
                {
                    return Ok(path.clone());
                }
            }
            match models.as_slice() {
                [(only, _)] => Ok(only.clone()),
                _ => Err(format!(
                    "add a quantization, like ':q8_0'. The repository has:\n{}",
                    available()
                )),
            }
        }
    }
}

/// Runs curl and returns what it printed.
fn curl_text(url: &str) -> Result<String, String> {
    let output = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "60",
            url,
        ])
        .stdin(Stdio::null())
        .output()
        .map_err(curl_missing)?;
    if !output.status.success() {
        return Err(format!(
            "could not fetch {url}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| format!("{url} did not return text"))
}

fn curl_missing(error: std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::NotFound {
        "curl is needed to download models, and it was not found".to_string()
    } else {
        format!("could not run curl: {error}")
    }
}

/// Lists the files of a Hugging Face repository as `(path, size)` pairs.
fn repo_files(repo: &str) -> Result<Vec<(String, u64)>, String> {
    let listing = curl_text(&format!(
        "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    ))?;
    let json = Json::parse(&listing)
        .map_err(|error| format!("unexpected answer from Hugging Face: {error}"))?;
    let entries = json
        .as_array()
        .ok_or("unexpected answer from Hugging Face")?;
    Ok(entries
        .iter()
        .filter(|entry| entry.get("type").and_then(Json::as_str) == Some("file"))
        .filter_map(|entry| {
            let path = entry.get("path")?.as_str()?.to_string();
            let size = entry.get("size").and_then(Json::as_f64).unwrap_or(0.0) as u64;
            Some((path, size))
        })
        .collect())
}

/// Downloads `url` to `dest`, resuming a partial download when there is one.
fn download(url: &str, dest: &Path) -> Result<(), String> {
    let partial = dest.with_extension("gguf.part");
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--progress-bar",
            "--continue-at",
            "-",
            "--output",
        ])
        .arg(&partial)
        .arg(url)
        .stdin(Stdio::null())
        .status()
        .map_err(curl_missing)?;
    if !status.success() {
        return Err(format!(
            "the download failed. Run the command again to resume it. ({status})"
        ));
    }
    // A model file starts with the letters GGUF. Anything else is an error
    // page or a wrong link.
    let mut magic = [0u8; 4];
    let is_model = std::fs::File::open(&partial)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
        .is_ok_and(|()| &magic == b"GGUF");
    if !is_model {
        let _ = std::fs::remove_file(&partial);
        return Err("the downloaded file is not a GGUF model".to_string());
    }
    std::fs::rename(&partial, dest)
        .map_err(|error| format!("could not save {}: {error}", dest.display()))
}

pub fn pull(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &[])?;
    let source = parse_source(args.positional(0).ok_or_else(pull_help)?)?;
    let (url, file_name) = match source {
        Source::Url(url) => {
            let path = url.split(['?', '#']).next().unwrap_or(&url);
            let name = path.rsplit('/').next().unwrap_or_default().to_string();
            if !name.to_lowercase().ends_with(".gguf") {
                return Err("the URL must point at a .gguf file".to_string());
            }
            (url, name)
        }
        Source::Repo { repo, tag } => {
            eprintln!("looking up {repo} on Hugging Face");
            let path = choose_file(&repo_files(&repo)?, tag.as_deref())?;
            let name = path.rsplit('/').next().unwrap_or(&path).to_string();
            (
                format!("https://huggingface.co/{repo}/resolve/main/{path}"),
                name,
            )
        }
    };

    let dir = models::models_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("could not create {}: {error}", dir.display()))?;
    let dest = dir.join(&file_name);
    let name = file_name.strip_suffix(".gguf").unwrap_or(&file_name);
    if dest.is_file() {
        println!("{name} is already downloaded. Run it with: smallm run {name}");
        return Ok(());
    }
    eprintln!("downloading {file_name} to {}", dir.display());
    download(&url, &dest)?;
    println!("downloaded {name}. Run it with: smallm run {name}");
    Ok(())
}

pub fn list(raw: impl Iterator<Item = String>) -> Result<(), String> {
    Args::parse(raw, &[])?;
    let dir = models::models_dir();
    let models = models::list(&dir);
    if models.is_empty() {
        println!(
            "No models in {}. Download one with: smallm pull <user/repo>",
            dir.display()
        );
        return Ok(());
    }
    let width = models
        .iter()
        .map(|model| model.name.len())
        .max()
        .unwrap_or(0);
    for model in &models {
        println!("{:<width$}  {:>9}", model.name, human_bytes(model.bytes));
    }
    eprintln!("in {}", dir.display());
    Ok(())
}

pub fn remove(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &[])?;
    let name = args
        .positional(0)
        .ok_or("which model? Usage: smallm rm <model>")?;
    // Only downloaded models are removed, never an arbitrary path.
    let path = models::resolve_in(&models::models_dir(), name)?;
    std::fs::remove_file(&path)
        .map_err(|error| format!("could not remove {}: {error}", path.display()))?;
    println!(
        "removed {}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(name)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(repo: &str, tag: Option<&str>) -> Source {
        Source::Repo {
            repo: repo.to_string(),
            tag: tag.map(str::to_string),
        }
    }

    #[test]
    fn sources_are_recognized() {
        assert_eq!(
            parse_source("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            Ok(repo("Qwen/Qwen2.5-0.5B-Instruct-GGUF", None))
        );
        assert_eq!(
            parse_source("hf.co/a/b:Q8_0"),
            Ok(repo("a/b", Some("Q8_0")))
        );
        assert_eq!(parse_source("huggingface.co/a/b:"), Ok(repo("a/b", None)));
        assert_eq!(
            parse_source("https://example.com/m.gguf?download=true"),
            Ok(Source::Url(
                "https://example.com/m.gguf?download=true".to_string()
            ))
        );
        for bad in [
            "model", "a/b/c", "/b", "a/", "a b/c", "../etc", "a/..", "a/b;rm",
        ] {
            assert!(parse_source(bad).is_err(), "{bad}");
        }
    }

    fn files(names: &[&str]) -> Vec<(String, u64)> {
        names
            .iter()
            .map(|name| (name.to_string(), 1_000_000))
            .collect()
    }

    #[test]
    fn the_preferred_quantization_is_chosen_by_default() {
        let listing = files(&[
            "README.md",
            "m-Q2_K.gguf",
            "m-Q4_K_M.gguf",
            "m-Q8_0.gguf",
            "mmproj-f16.gguf",
        ]);
        assert_eq!(choose_file(&listing, None).unwrap(), "m-Q4_K_M.gguf");
        assert_eq!(choose_file(&listing, Some("q8_0")).unwrap(), "m-Q8_0.gguf");
        assert!(
            choose_file(&listing, Some("q6_k"))
                .unwrap_err()
                .contains("m-Q2_K.gguf")
        );
        // With no preferred quantization, a lone model is taken.
        assert_eq!(
            choose_file(&files(&["only-f16.gguf"]), None).unwrap(),
            "only-f16.gguf"
        );
        assert!(choose_file(&files(&["a-f16.gguf", "b-bf16.gguf"]), None).is_err());
        assert!(choose_file(&files(&["README.md"]), None).is_err());
    }

    #[test]
    fn the_plain_variant_wins_among_several_matches() {
        let listing = files(&[
            "m-Q4_K_M-imat.gguf",
            "m-Q4_K_M.gguf",
            "m-Q4_K_M-00001-of-00002.gguf",
        ]);
        assert_eq!(
            choose_file(&listing, Some("Q4_K_M")).unwrap(),
            "m-Q4_K_M.gguf"
        );
    }
}

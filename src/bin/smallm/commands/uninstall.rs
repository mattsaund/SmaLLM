//! `smallm uninstall`: removes SmaLLM from this computer.
//!
//! The installers put one program file on a machine. Downloads go to one
//! folder of SmaLLM's own, settings to one file, and on Windows the
//! installer also adds the program's folder to the user's PATH. This
//! command removes all of that.
//! It lists what it will remove and asks before it removes anything.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use crate::args::{self, Args, Spec, switch};
use crate::models::{self, human_bytes};

const SPECS: [Spec; 2] = [
    switch("yes", Some('y'), "Remove without asking first"),
    switch(
        "keep-models",
        None,
        "Leave downloaded models where they are",
    ),
];

pub fn help() -> String {
    format!(
        "Usage: smallm uninstall [options]\n\n\
         Removes SmaLLM from this computer: this program, the models it\n\
         downloaded and the folder they are kept in, and on Windows the PATH\n\
         entry the installer added. It lists what it will remove and asks\n\
         first.\n\n\
         Options:\n{}",
        args::describe(&SPECS)
    )
}

pub fn main(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &SPECS)?;
    let program = std::env::current_exe()
        .map_err(|error| format!("cannot find this program's file: {error}"))?;
    let program = program.canonicalize().unwrap_or(program);
    // A program that cargo built in a source checkout belongs to the
    // checkout, and `cargo clean` removes it.
    let in_checkout = program
        .components()
        .any(|part| part.as_os_str() == "target");
    let custom = std::env::var_os("SMALLM_MODELS").is_some_and(|dir| !dir.is_empty());
    let data = Data::find(models::models_dir(), custom, args.has("keep-models"));

    println!("This removes:");
    if in_checkout {
        println!("  nothing of the program, which is a build in a source checkout");
    } else {
        println!("  the program, {}", program.display());
    }
    data.describe();
    let settings = crate::config::path();
    if settings.exists() {
        println!("  the settings, {}", settings.display());
    }
    if cfg!(windows) && !in_checkout {
        println!("  its folder from your PATH, where the installer added it");
    }

    if !args.has("yes") {
        if !std::io::stdin().is_terminal() {
            return Err("nothing was removed. Pass --yes to uninstall without being asked.".into());
        }
        print!("Remove all of this? [y/N] ");
        std::io::stdout().flush().ok();
        let mut answer = String::new();
        std::io::stdin().lock().read_line(&mut answer).ok();
        if !matches!(answer.trim(), "y" | "Y" | "yes" | "Yes" | "YES") {
            println!("Nothing was removed.");
            return Ok(());
        }
    }

    data.remove()?;
    if settings.exists() {
        std::fs::remove_file(&settings)
            .map_err(|error| format!("could not remove {}: {error}", settings.display()))?;
        println!("Removed {}", settings.display());
        // The folder goes too, unless something else is kept in it.
        if let Some(folder) = settings.parent() {
            let _ = std::fs::remove_dir(folder);
        }
    }
    if !in_checkout {
        if let Some(folder) = program.parent() {
            remove_from_path(folder);
        }
        remove_program(&program)?;
        println!("Removed {}", program.display());
    }
    println!("SmaLLM is uninstalled.");
    if !cfg!(windows) && !in_checkout {
        println!(
            "If you added {} to your PATH yourself, that line is still in your shell's profile.",
            program
                .parent()
                .map_or(String::new(), |folder| folder.display().to_string())
        );
    }
    Ok(())
}

/// The downloads, and the folders they are kept in.
struct Data {
    /// The folder models are downloaded to.
    models: PathBuf,
    /// SmaLLM's own data folder, removed whole. `None` when the models
    /// folder was chosen with `SMALLM_MODELS`, which may hold other files.
    owned: Option<PathBuf>,
    /// The model files and unfinished downloads in the models folder.
    files: Vec<PathBuf>,
    bytes: u64,
    /// Leave all of it alone.
    keep: bool,
}

impl Data {
    /// Looks at `models`, the models folder. `custom` says the user chose
    /// it, in which case only SmaLLM's files in it are removed.
    fn find(models: PathBuf, custom: bool, keep: bool) -> Data {
        let mut files = Vec::new();
        let mut bytes = 0;
        if let Ok(entries) = std::fs::read_dir(&models) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with(".gguf") || name.ends_with(".gguf.part") {
                    bytes += entry.metadata().map_or(0, |meta| meta.len());
                    files.push(path);
                }
            }
        }
        files.sort();
        // The default folder is `<data>/smallm/models`. Its parent is ours.
        let owned = if custom {
            None
        } else {
            models
                .parent()
                .filter(|parent| parent.file_name().is_some_and(|name| name == "smallm"))
                .map(Path::to_path_buf)
        };
        Data {
            models,
            owned,
            files,
            bytes,
            keep,
        }
    }

    /// Says what [`remove`](Self::remove) will remove.
    fn describe(&self) {
        if self.keep {
            println!(
                "  no models: the {} in {} stay",
                plural(self.files.len(), "model file"),
                self.models.display()
            );
            return;
        }
        if !self.files.is_empty() {
            println!(
                "  {} in {}, {}",
                plural(self.files.len(), "model file"),
                self.models.display(),
                human_bytes(self.bytes)
            );
        }
        match &self.owned {
            Some(folder) if folder.exists() => println!("  the folder {}", folder.display()),
            _ => {}
        }
    }

    fn remove(&self) -> Result<(), String> {
        if self.keep {
            return Ok(());
        }
        let failed = |path: &Path, error: std::io::Error| {
            format!("could not remove {}: {error}", path.display())
        };
        match &self.owned {
            Some(folder) if folder.exists() => {
                std::fs::remove_dir_all(folder).map_err(|error| failed(folder, error))?;
                println!("Removed {}", folder.display());
            }
            _ => {
                for file in &self.files {
                    std::fs::remove_file(file).map_err(|error| failed(file, error))?;
                }
                if !self.files.is_empty() {
                    println!(
                        "Removed {} from {}",
                        plural(self.files.len(), "model file"),
                        self.models.display()
                    );
                }
                // Gone only if nothing else was kept in it.
                let _ = std::fs::remove_dir(&self.models);
            }
        }
        Ok(())
    }
}

/// "1 model file", "3 model files".
fn plural(count: usize, thing: &str) -> String {
    if count == 1 {
        format!("1 {thing}")
    } else {
        format!("{count} {thing}s")
    }
}

/// Deletes the program's file. A running program on Unix can delete its
/// own file: the system keeps it until the program exits.
#[cfg(not(windows))]
fn remove_program(program: &Path) -> Result<(), String> {
    std::fs::remove_file(program)
        .map_err(|error| format!("could not remove {}: {error}", program.display()))
}

/// Deletes the program's file. Windows does not let a running program
/// delete itself, so a separate, hidden `cmd` waits for this one to exit
/// and deletes it then.
#[cfg(windows)]
fn remove_program(program: &Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let line = format!(
        "/d /c ping -n 3 127.0.0.1 >nul & del /f /q \"{}\"",
        program.display()
    );
    std::process::Command::new("cmd.exe")
        .raw_arg(line)
        .creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW)
        .spawn()
        .map(drop)
        .map_err(|error| format!("could not remove {}: {error}", program.display()))
}

/// Takes `folder` out of the user's PATH, as the Windows installer put it
/// there. PowerShell edits the setting, as the installer did.
#[cfg(windows)]
fn remove_from_path(folder: &Path) {
    let folder = folder.display().to_string().replace('\'', "''");
    let script = format!(
        "$path = [Environment]::GetEnvironmentVariable('PATH', 'User'); \
         if ($path) {{ \
           $kept = @($path -split ';' | Where-Object {{ $_ -and $_ -ne '{folder}' }}); \
           [Environment]::SetEnvironmentVariable('PATH', ($kept -join ';'), 'User') \
         }}"
    );
    let _ = std::process::Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .status();
}

/// The Unix installers do not edit the PATH, so there is nothing to undo.
#[cfg(not(windows))]
fn remove_from_path(_folder: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh folder under the system's temporary folder.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("smallm-uninstall-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_default_folder_goes_whole() {
        let root = scratch("default");
        let models = root.join("smallm").join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("a.gguf"), [0u8; 10]).unwrap();
        std::fs::write(models.join("b.gguf.part"), [0u8; 5]).unwrap();

        let data = Data::find(models.clone(), false, false);
        assert_eq!((data.files.len(), data.bytes), (2, 15));
        assert_eq!(data.owned.as_deref(), Some(root.join("smallm").as_path()));
        data.remove().unwrap();
        assert!(!root.join("smallm").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_chosen_folder_keeps_other_files() {
        let models = scratch("custom");
        std::fs::write(models.join("a.gguf"), [0u8; 10]).unwrap();
        std::fs::write(models.join("notes.txt"), "mine").unwrap();

        let data = Data::find(models.clone(), true, false);
        assert!(data.owned.is_none());
        data.remove().unwrap();
        assert!(!models.join("a.gguf").exists());
        assert!(models.join("notes.txt").exists());

        // Keeping the models removes nothing.
        std::fs::write(models.join("b.gguf"), [0u8; 3]).unwrap();
        Data::find(models.clone(), true, true).remove().unwrap();
        assert!(models.join("b.gguf").exists());
        std::fs::remove_dir_all(&models).unwrap();
    }
}

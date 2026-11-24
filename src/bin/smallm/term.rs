//! The terminal: Ctrl+C handling, and what the terminal can show.
//!
//! While the model is writing an answer, the first Ctrl+C stops the answer
//! and returns to the prompt. At any other time, and on a second press,
//! Ctrl+C ends the program as usual.
//!
//! Status lines and progress bars go to stderr, so that what a command
//! prints on stdout can be piped without them. They use ANSI escape codes
//! only when stderr is a terminal that understands them.

use std::io::IsTerminal;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// True while an answer is being generated.
static GENERATING: AtomicBool = AtomicBool::new(false);

/// True once Ctrl+C was pressed during the current generation.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Starts handling Ctrl+C as described above. Does nothing on systems
/// without a handler below.
pub fn install_interrupt_handler() {
    sys::install();
}

/// Marks the start of a generation.
pub fn begin_generation() {
    INTERRUPTED.store(false, Ordering::Relaxed);
    GENERATING.store(true, Ordering::Relaxed);
}

/// Marks the end of a generation. Returns whether it was interrupted.
pub fn end_generation() -> bool {
    GENERATING.store(false, Ordering::Relaxed);
    INTERRUPTED.swap(false, Ordering::Relaxed)
}

/// True when Ctrl+C was pressed during the current generation.
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// Decides what a Ctrl+C means right now. Returns true when it was taken as
/// a request to stop the answer, false when the program should end.
fn take_interrupt() -> bool {
    GENERATING.load(Ordering::Relaxed) && !INTERRUPTED.swap(true, Ordering::Relaxed)
}

/// True when stderr is a terminal a person is looking at.
pub fn stderr_is_terminal() -> bool {
    std::io::stderr().is_terminal()
}

/// True when stderr is a terminal that understands ANSI escape codes: colors,
/// and clearing a line to redraw a progress bar.
///
/// `NO_COLOR` turns them off, as does `TERM=dumb`. On Windows the console
/// is asked to interpret them first, which Windows 10 and later can do.
pub fn ansi() -> bool {
    static ANSI: OnceLock<bool> = OnceLock::new();
    *ANSI.get_or_init(|| {
        stderr_is_terminal()
            && std::env::var_os("NO_COLOR").is_none()
            && std::env::var("TERM").map_or(true, |term| term != "dumb")
            && sys::enable_ansi()
    })
}

/// True when the terminal can be expected to draw Unicode block characters.
/// Windows consoles can. Elsewhere the locale says whether text is UTF-8.
pub fn unicode() -> bool {
    if cfg!(windows) {
        return true;
    }
    ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
        .is_some_and(|locale| {
            let locale = locale.to_lowercase();
            locale.contains("utf-8") || locale.contains("utf8")
        })
}

/// Width of the terminal in columns. 80 when it cannot be told.
pub fn width() -> usize {
    sys::columns()
        .or_else(|| std::env::var("COLUMNS").ok()?.parse().ok())
        .filter(|&columns| columns >= 20)
        .unwrap_or(80)
}

/// Wraps `text` in an ANSI style when the terminal shows styles: `"1;32"`
/// is bold green, `"2"` is faint.
pub fn paint(style: &str, text: &str) -> String {
    if ansi() {
        format!("\x1b[{style}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::c_int;

    const SIGINT: c_int = 2;

    unsafe extern "C" {
        fn signal(signal: c_int, handler: extern "C" fn(c_int)) -> usize;
        fn _exit(code: c_int) -> !;
    }

    /// Runs inside the signal, so it only touches atomics and `_exit`.
    extern "C" fn on_interrupt(_signal: c_int) {
        if !super::take_interrupt() {
            // SAFETY: `_exit` is safe to call from a signal handler. 130 is
            // the conventional status of a program ended by Ctrl+C.
            unsafe { _exit(130) }
        }
    }

    pub fn install() {
        // SAFETY: the handler is async-signal-safe.
        unsafe {
            signal(SIGINT, on_interrupt);
        }
    }

    /// Unix terminals take ANSI codes as they are.
    pub fn enable_ansi() -> bool {
        true
    }

    /// The size of a terminal, as the kernel reports it.
    #[repr(C)]
    #[derive(Default)]
    struct WindowSize {
        rows: u16,
        columns: u16,
        width_pixels: u16,
        height_pixels: u16,
    }

    /// The request that reads a `WindowSize`. Linux numbers it differently
    /// from the BSDs and macOS.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const GET_WINDOW_SIZE: std::ffi::c_ulong = 0x5413;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    const GET_WINDOW_SIZE: std::ffi::c_ulong = 0x4008_7468;

    unsafe extern "C" {
        fn ioctl(fd: c_int, request: std::ffi::c_ulong, ...) -> c_int;
    }

    /// Columns of the terminal on stderr.
    pub fn columns() -> Option<usize> {
        let mut size = WindowSize::default();
        // SAFETY: the request writes one `WindowSize`, which `size` is.
        let status = unsafe { ioctl(2, GET_WINDOW_SIZE, &raw mut size) };
        (status == 0 && size.columns > 0).then_some(size.columns as usize)
    }
}

#[cfg(windows)]
mod sys {
    const CTRL_C_EVENT: u32 = 0;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetConsoleCtrlHandler(
            handler: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
    }

    /// Returning 1 tells Windows the event was handled. Returning 0 passes
    /// it on to the default handler, which ends the program.
    unsafe extern "system" fn on_control(kind: u32) -> i32 {
        i32::from(kind == CTRL_C_EVENT && super::take_interrupt())
    }

    pub fn install() {
        // SAFETY: the handler only touches atomics.
        unsafe {
            SetConsoleCtrlHandler(Some(on_control), 1);
        }
    }

    type Handle = *mut std::ffi::c_void;

    /// `GetStdHandle` names stderr with this number.
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    /// The console mode bit that makes ANSI codes work.
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    #[repr(C)]
    #[derive(Default)]
    struct Coord {
        x: i16,
        y: i16,
    }

    #[repr(C)]
    #[derive(Default)]
    struct SmallRect {
        left: i16,
        top: i16,
        right: i16,
        bottom: i16,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ScreenBufferInfo {
        size: Coord,
        cursor: Coord,
        attributes: u16,
        window: SmallRect,
        maximum_window_size: Coord,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> Handle;
        fn GetConsoleMode(console: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(console: Handle, mode: u32) -> i32;
        fn GetConsoleScreenBufferInfo(console: Handle, info: *mut ScreenBufferInfo) -> i32;
    }

    /// Asks the console on stderr to interpret ANSI codes. False on a
    /// console too old to do it.
    pub fn enable_ansi() -> bool {
        // SAFETY: the calls only read and write the local `mode`.
        unsafe {
            let console = GetStdHandle(STD_ERROR_HANDLE);
            let mut mode = 0u32;
            if GetConsoleMode(console, &raw mut mode) == 0 {
                return false;
            }
            mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0
                || SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }
    }

    /// Columns of the console window on stderr.
    pub fn columns() -> Option<usize> {
        let mut info = ScreenBufferInfo::default();
        // SAFETY: the call writes one `ScreenBufferInfo`, which `info` is.
        let status =
            unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &raw mut info) };
        let columns = info.window.right as i32 - info.window.left as i32 + 1;
        (status != 0 && columns > 0).then_some(columns as usize)
    }
}

#[cfg(not(any(unix, windows)))]
mod sys {
    pub fn install() {}

    pub fn enable_ansi() -> bool {
        false
    }

    pub fn columns() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_first_press_during_a_generation_is_taken() {
        // Outside a generation, Ctrl+C is not ours to take.
        assert!(!take_interrupt());
        begin_generation();
        assert!(!interrupted());
        assert!(take_interrupt());
        assert!(interrupted());
        // A second press is left to end the program.
        assert!(!take_interrupt());
        assert!(end_generation());
        // The flag does not leak into the next generation.
        begin_generation();
        assert!(!interrupted());
        assert!(!end_generation());
    }
}

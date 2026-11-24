//! Ctrl+C handling for the interactive commands.
//!
//! While the model is writing an answer, the first Ctrl+C stops the answer
//! and returns to the prompt. At any other time, and on a second press,
//! Ctrl+C ends the program as usual.

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
}

#[cfg(not(any(unix, windows)))]
mod sys {
    pub fn install() {}
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

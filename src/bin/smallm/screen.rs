//! A scrolling log with one progress bar under it.
//!
//! Each step of a long task prints a line, and the lines scroll up like any
//! terminal output. One progress bar for the whole task stays on the last
//! line. Printing a line clears the bar, writes the line, and draws the bar
//! again below it, so the bar never scrolls away and never multiplies.
//!
//! Everything goes to stderr. When stderr is not a terminal, the log lines
//! are printed as plain text and the bar is left out.

use std::io::Write;
use std::time::{Duration, Instant};

use crate::models::human_bytes;
use crate::term;

/// The least time between two redraws of the bar. Drawing more often only
/// costs time, since nobody reads faster.
const REDRAW_EVERY: Duration = Duration::from_millis(80);

/// Width of the tag column of the log, like `Downloading`.
const TAG_WIDTH: usize = 12;

/// A log with a progress bar, measured in bytes.
pub struct Screen {
    /// True when the bar is drawn: stderr is a terminal that takes ANSI.
    live: bool,
    /// Bytes the whole task will move. Zero while unknown.
    total: u64,
    /// Bytes moved so far.
    done: u64,
    /// Bytes that were already there when the task began, as when a
    /// download resumes. They count for the bar but not for the speed.
    skipped: u64,
    started: Instant,
    last_draw: Option<Instant>,
    /// True while the bar occupies the last line.
    drawn: bool,
}

impl Screen {
    pub fn new() -> Screen {
        Screen {
            live: term::ansi(),
            total: 0,
            done: 0,
            skipped: 0,
            started: Instant::now(),
            last_draw: None,
            drawn: false,
        }
    }

    /// Prints one log line: a tag such as `Downloading` and the details.
    pub fn log(&mut self, tag: &str, text: &str) {
        self.line("1;32", tag, text);
    }

    /// Prints one log line about something that went wrong but did not stop
    /// the task. The tag is yellow instead of green.
    pub fn warn(&mut self, tag: &str, text: &str) {
        self.line("1;33", tag, text);
    }

    fn line(&mut self, style: &str, tag: &str, text: &str) {
        let mut err = std::io::stderr().lock();
        if self.drawn {
            let _ = write!(err, "\r\x1b[2K");
        }
        let tag = format!("{tag:>TAG_WIDTH$}");
        let _ = writeln!(err, "{} {text}", term::paint(style, &tag));
        drop(err);
        self.drawn = false;
        if self.live && self.total > 0 {
            self.draw();
        }
    }

    /// Sets how many bytes the task moves in all, and how many of them were
    /// done before it started.
    pub fn begin(&mut self, total: u64, already: u64) {
        self.total = total;
        self.done = already;
        self.skipped = already;
        self.started = Instant::now();
        self.last_draw = None;
    }

    /// Counts bytes moved, and redraws the bar when it is time to.
    pub fn advance(&mut self, bytes: u64) {
        self.done += bytes;
        if !self.live {
            return;
        }
        let due = self
            .last_draw
            .is_none_or(|last| last.elapsed() >= REDRAW_EVERY);
        if due {
            self.draw();
        }
    }

    /// Bytes per second since `begin`, not counting what was there before.
    pub fn speed(&self) -> f64 {
        let seconds = self.started.elapsed().as_secs_f64();
        if seconds <= 0.0 {
            0.0
        } else {
            (self.done - self.skipped) as f64 / seconds
        }
    }

    /// Time since `begin`.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Takes the bar off the screen, for good.
    pub fn finish(&mut self) {
        if self.drawn {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            self.drawn = false;
        }
        self.live = false;
    }

    /// Draws the bar on the last line, over whatever bar was there.
    fn draw(&mut self) {
        // The first moments of a download say little about its speed.
        let speed = if self.elapsed() < Duration::from_secs(1) {
            0.0
        } else {
            self.speed()
        };
        let line = bar_line(self.done, self.total, speed, term::width(), term::unicode());
        let mut err = std::io::stderr().lock();
        let _ = write!(err, "\r\x1b[2K{line}");
        let _ = err.flush();
        self.drawn = true;
        self.last_draw = Some(Instant::now());
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        self.finish();
    }
}

/// The text of the progress bar for `done` of `total` bytes at `speed`
/// bytes per second, fitted to `width` columns.
///
/// It reads like `[██████████▌        ]  56%  59.2 MB / 105.5 MB  12.1 MB/s  0:04`.
fn bar_line(done: u64, total: u64, speed: f64, width: usize, unicode: bool) -> String {
    let share = if total == 0 {
        0.0
    } else {
        (done as f64 / total as f64).clamp(0.0, 1.0)
    };
    let left = if speed > 0.0 && done < total {
        clock(Duration::from_secs_f64((total - done) as f64 / speed))
    } else {
        "     ".to_string()
    };
    let numbers = format!(
        " {:>3}%  {} / {}  {}/s  {left}",
        (share * 100.0).floor() as u32,
        human_bytes(done),
        human_bytes(total),
        human_bytes(speed as u64),
    );
    // The bar takes what the numbers leave, within reason.
    let room = width
        .saturating_sub(numbers.chars().count() + 3)
        .clamp(10, 50);
    let mut bar = String::with_capacity(room * 3);
    let filled = share * room as f64;
    for cell in 0..room {
        let fill = (filled - cell as f64).clamp(0.0, 1.0);
        bar.push(if unicode {
            // Eighths of a cell, so the bar moves smoothly.
            [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'][(fill * 8.0).floor() as usize]
        } else if fill >= 1.0 {
            '#'
        } else {
            '-'
        });
    }
    format!("[{bar}]{numbers}")
}

/// A duration as minutes and seconds, or hours and minutes past an hour.
pub fn clock(time: Duration) -> String {
    let seconds = time.as_secs();
    if seconds >= 3600 {
        format!("{}:{:02}h", seconds / 3600, seconds % 3600 / 60)
    } else {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bar_fills_with_the_share_done() {
        let empty = bar_line(0, 1000, 0.0, 80, false);
        let half = bar_line(500, 1000, 100.0, 80, false);
        let full = bar_line(1000, 1000, 100.0, 80, false);
        assert!(empty.starts_with("[-") && empty.contains("  0%"), "{empty}");
        assert!(half.contains(" 50%"), "{half}");
        let (filled, open) = (half.matches('#').count(), half.matches('-').count());
        assert!(filled.abs_diff(open) <= 1, "{half}");
        assert!(full.contains("100%") && !full.contains('-'), "{full}");
        // Five seconds left at 100 bytes a second.
        assert!(half.contains("0:05"), "{half}");
    }

    #[test]
    fn the_bar_fits_the_terminal() {
        for width in [40, 80, 120, 300] {
            let line = bar_line(123_456, 9_876_543, 4321.0, width, true);
            assert!(line.chars().count() <= width.max(60), "{width}: {line}");
        }
        // Partial cells use the finer block characters.
        let line = bar_line(1, 3, 0.0, 80, true);
        assert!(line.contains('█') && line.contains(' '), "{line}");
    }

    #[test]
    fn durations_read_as_clocks() {
        assert_eq!(clock(Duration::from_secs(0)), "0:00");
        assert_eq!(clock(Duration::from_secs(75)), "1:15");
        assert_eq!(clock(Duration::from_secs(3725)), "1:02h");
    }
}

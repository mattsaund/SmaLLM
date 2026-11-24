//! The generation loop: prompt in, text out.
//!
//! [`run`] is the short way to use the engine. It brings a session's context
//! in line with a prompt, then samples tokens and hands their text to a
//! callback until the model finishes, a limit is reached or the callback
//! asks to stop.
//!
//! The context is reused between calls. When a new prompt starts with the
//! tokens already in the session, as every turn of a chat does, only the new
//! part is evaluated.

use std::time::{Duration, Instant};

use crate::chat::{Message, Template};
use crate::error::{Error, Result};
use crate::model::Session;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

/// What generation does when the context window is full.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Overflow {
    /// Keep the start of the context, such as the system prompt, and drop
    /// the oldest tokens after it.
    #[default]
    Middle,
    /// Drop the oldest tokens and slide forward. Only the first token
    /// stays, when it marks the beginning of the text.
    Slide,
    /// End the answer.
    Stop,
}

impl Overflow {
    /// Every choice, for listings.
    pub const ALL: [Overflow; 3] = [Overflow::Middle, Overflow::Slide, Overflow::Stop];

    /// The name of the choice, as given on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Overflow::Middle => "middle",
            Overflow::Slide => "slide",
            Overflow::Stop => "stop",
        }
    }

    /// Looks up a choice by its name.
    pub fn from_name(name: &str) -> Option<Overflow> {
        Overflow::ALL
            .into_iter()
            .find(|choice| choice.name() == name)
    }
}

/// Options for one generation.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Stop after this many tokens. Zero continues until the model ends its
    /// answer or the context is full.
    pub max_tokens: usize,
    /// How tokens are chosen.
    pub sampler: SamplerConfig,
    /// Stop when the output contains any of these strings. The matching
    /// string is not part of the output.
    pub stop: Vec<String>,
    /// Show control tokens in the output instead of hiding them.
    pub show_special: bool,
    /// What to do when the context window is full.
    pub overflow: Overflow,
    /// Tokens at the start of the context that [`Overflow::Middle`] keeps.
    /// `None` keeps the prompt. At most half the window is kept either way.
    pub keep: Option<usize>,
}

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model produced an end token.
    End,
    /// The token limit was reached.
    Length,
    /// A stop string appeared.
    Stop,
    /// The context window is full.
    Context,
    /// The callback asked to stop.
    Canceled,
}

/// What a generation did.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Tokens in the prompt.
    pub prompt_tokens: usize,
    /// Prompt tokens that were already in the context and not re-evaluated.
    pub reused_tokens: usize,
    /// Time spent evaluating the prompt.
    pub prompt_time: Duration,
    /// Tokens generated.
    pub generated_tokens: usize,
    /// Time spent generating.
    pub generate_time: Duration,
    /// Why the generation ended.
    pub finish: Finish,
    /// Tokens in the context when the generation ended: the prompt and the
    /// answer, as far as the session holds them.
    pub context_tokens: usize,
    /// Size of the context window in tokens.
    pub context_size: usize,
    /// Tokens dropped from the context to make room, from the prompt or
    /// while generating.
    pub dropped_tokens: usize,
    /// Bytes of memory the context holds, for the key/value cache and any
    /// other per-token state.
    pub context_bytes: usize,
}

impl Stats {
    /// Prompt tokens evaluated per second.
    pub fn prompt_speed(&self) -> f64 {
        rate(self.prompt_tokens - self.reused_tokens, self.prompt_time)
    }

    /// Tokens generated per second.
    pub fn generate_speed(&self) -> f64 {
        rate(self.generated_tokens, self.generate_time)
    }

    /// The share of the context window in use, from 0 to 1.
    pub fn context_used(&self) -> f64 {
        if self.context_size == 0 {
            0.0
        } else {
            (self.context_tokens as f64 / self.context_size as f64).min(1.0)
        }
    }
}

fn rate(count: usize, time: Duration) -> f64 {
    if time.is_zero() {
        0.0
    } else {
        count as f64 / time.as_secs_f64()
    }
}

/// Completes `prompt` as plain text.
///
/// `on_text` receives each piece of output as it is produced, and returns
/// `false` to stop early.
pub fn run(
    session: &mut Session,
    prompt: &str,
    options: &Options,
    on_text: impl FnMut(&str) -> bool,
) -> Result<Stats> {
    let tokens = session.model().tokenizer().encode(prompt, true, true);
    run_tokens(session, &tokens, options, on_text)
}

/// Answers a conversation, formatted with `template`.
pub fn chat(
    session: &mut Session,
    template: Template,
    messages: &[Message],
    options: &Options,
    on_text: impl FnMut(&str) -> bool,
) -> Result<Stats> {
    let tokens = chat_tokens(session.model().tokenizer(), template, messages);
    run_tokens(session, &tokens, options, on_text)
}

/// The tokens of a conversation that ends with an open assistant turn.
pub fn chat_tokens(tokenizer: &Tokenizer, template: Template, messages: &[Message]) -> Vec<u32> {
    let eos = tokenizer
        .eos()
        .map(|token| tokenizer.token_text(token))
        .unwrap_or_default();
    let parts = template.render(messages, true, &String::from_utf8_lossy(eos));
    let parts: Vec<(&str, bool)> = parts
        .iter()
        .map(|part| (part.text.as_str(), part.markup))
        .collect();
    tokenizer.encode_parts(&parts, true)
}

/// Like [`chat_tokens`], for a conversation that may have outgrown the
/// context window.
///
/// Drops the oldest turns until the prompt leaves `reserve` tokens of room
/// in a window of `n_ctx`. System messages and the newest message are always
/// kept. Returns the tokens and the number of messages dropped.
pub fn chat_tokens_fitting(
    tokenizer: &Tokenizer,
    template: Template,
    messages: &[Message],
    n_ctx: usize,
    reserve: usize,
    overflow: Overflow,
) -> (Vec<u32>, usize) {
    use crate::chat::Role;
    let mut kept: Vec<Message> = messages.to_vec();
    loop {
        let tokens = chat_tokens(tokenizer, template, &kept);
        if tokens.len() + reserve <= n_ctx {
            return (tokens, messages.len() - kept.len());
        }
        // The oldest turn, unless it is the only one left. Sliding drops
        // system messages too. The other choices keep them.
        let droppable = kept
            .iter()
            .position(|message| overflow == Overflow::Slide || message.role != Role::System);
        match droppable {
            Some(index) if index + 1 < kept.len() => {
                kept.remove(index);
                // A conversation should not open with half a turn.
                while kept.get(index).is_some_and(|m| m.role == Role::Assistant)
                    && index + 1 < kept.len()
                {
                    kept.remove(index);
                }
            }
            _ => return (tokens, messages.len() - kept.len()),
        }
    }
}

/// The number of tokens a chat prompt spends on its opening: the system
/// messages and the first message of the user. That is the start
/// [`Overflow::Middle`] keeps in a chat, so the model never loses its
/// instructions or the question that began the conversation.
pub fn opening_tokens(
    tokenizer: &Tokenizer,
    template: Template,
    messages: &[Message],
    prompt: &[u32],
) -> usize {
    use crate::chat::Role;
    let system = messages
        .iter()
        .take_while(|message| message.role == Role::System)
        .count();
    let first_user = usize::from(messages.get(system).is_some_and(|m| m.role == Role::User));
    if system + first_user == 0 {
        return 0;
    }
    // The opening renders the same in front of any conversation, up to the
    // turn after it.
    let alone = chat_tokens(tokenizer, template, &messages[..system + first_user]);
    alone.iter().zip(prompt).take_while(|(a, b)| a == b).count()
}

/// Where the protected start of the context ends, for `overflow`. The rest
/// of the context is what may be dropped.
fn kept_start(session: &Session, overflow: Overflow, keep: usize) -> usize {
    let tokens = session.tokens();
    let half = session.n_ctx() / 2;
    let bos = session.model().tokenizer().bos();
    let first_is_bos = usize::from(bos.is_some() && tokens.first().copied() == bos);
    match overflow {
        Overflow::Middle => keep.max(first_is_bos).min(half),
        Overflow::Slide | Overflow::Stop => first_is_bos,
    }
}

/// Generates from a prompt that is already tokenized.
pub fn run_tokens(
    session: &mut Session,
    prompt: &[u32],
    options: &Options,
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<Stats> {
    if prompt.is_empty() {
        return Err(Error::invalid("the prompt is empty"));
    }
    let model = std::sync::Arc::clone(session.model());
    let tokenizer = model.tokenizer();
    let keep = options.keep.unwrap_or(prompt.len());
    let mut dropped = 0;

    // A prompt too long for the window loses its oldest tokens after the
    // kept start, leaving a quarter of the window for the answer.
    let n_ctx = session.n_ctx();
    let cut: Vec<u32>;
    let prompt = if prompt.len() + n_ctx / 4 > n_ctx && options.overflow != Overflow::Stop {
        let head = match options.overflow {
            Overflow::Middle => keep.min(n_ctx / 2),
            _ => {
                usize::from(tokenizer.bos().is_some() && prompt.first().copied() == tokenizer.bos())
            }
        };
        let tail = (n_ctx - n_ctx / 4).saturating_sub(head).max(1);
        dropped = prompt.len() - head - tail;
        cut = [&prompt[..head], &prompt[prompt.len() - tail..]].concat();
        &cut[..]
    } else {
        prompt
    };

    // Keep the part of the context the prompt shares with it. The last
    // prompt token is always evaluated, to get fresh logits.
    let start = Instant::now();
    let shared = session
        .tokens()
        .iter()
        .zip(prompt)
        .take_while(|(have, want)| have == want)
        .count();
    let reused = shared.min(prompt.len() - 1);
    session.truncate(reused);
    session.eval(&prompt[reused..])?;
    let prompt_time = start.elapsed();

    let start = Instant::now();
    let mut sampler = Sampler::new(options.sampler.clone());
    let mut text = TextStream::new(&options.stop);
    let mut generated = 0;
    // A sampled token is evaluated only when another one is wanted, so the
    // last token of an answer costs nothing.
    let mut pending: Option<u32> = None;
    let finish = loop {
        if let Some(token) = pending.take() {
            if session.tokens().len() >= session.n_ctx() {
                // Make room by dropping the older half of what may go.
                let len = session.tokens().len();
                let start = kept_start(session, options.overflow, keep);
                if options.overflow == Overflow::Stop || len <= start + 1 {
                    break Finish::Context;
                }
                session.shift(start, (len - start) / 2);
                dropped += len - session.tokens().len();
                if session.tokens().len() == len {
                    break Finish::Context;
                }
            }
            session.eval(&[token])?;
        }
        if options.max_tokens > 0 && generated >= options.max_tokens {
            break Finish::Length;
        }
        let token = sampler.sample(session.logits(), session.tokens());
        if tokenizer.is_end(token) {
            break Finish::End;
        }
        generated += 1;
        pending = Some(token);

        let mut piece = Vec::new();
        tokenizer.append_piece(token, options.show_special, &mut piece);
        match text.push(&piece, &mut on_text) {
            Flow::Continue => {}
            Flow::Stop => break Finish::Stop,
            Flow::Cancel => break Finish::Canceled,
        }
    };
    if !matches!(finish, Finish::Stop | Finish::Canceled) {
        text.finish(&mut on_text);
    }

    Ok(Stats {
        prompt_tokens: prompt.len(),
        reused_tokens: reused,
        prompt_time,
        generated_tokens: generated,
        generate_time: start.elapsed(),
        finish,
        // A sampled token that was not evaluated yet still belongs to the
        // conversation, and the next turn evaluates it.
        context_tokens: session.tokens().len() + usize::from(pending.is_some()),
        context_size: session.n_ctx(),
        dropped_tokens: dropped,
        context_bytes: session.cache_bytes(),
    })
}

/// What the text stream wants the generation loop to do next.
enum Flow {
    Continue,
    Stop,
    Cancel,
}

/// Turns token bytes into text for the callback.
///
/// Two things are held back until more bytes arrive: the start of a UTF-8
/// character whose other bytes are in a later token, and text that could be
/// the beginning of a stop string.
struct TextStream<'a> {
    stop: &'a [String],
    /// Bytes not yet known to be whole characters.
    bytes: Vec<u8>,
    /// Text not yet known to be clear of a stop string.
    held: String,
}

impl<'a> TextStream<'a> {
    fn new(stop: &'a [String]) -> TextStream<'a> {
        TextStream {
            stop,
            bytes: Vec::new(),
            held: String::new(),
        }
    }

    /// Adds the bytes of one token and delivers what can be delivered.
    fn push(&mut self, piece: &[u8], on_text: &mut impl FnMut(&str) -> bool) -> Flow {
        self.bytes.extend_from_slice(piece);
        // Move every complete character to `held`.
        loop {
            match std::str::from_utf8(&self.bytes) {
                Ok(valid) => {
                    self.held.push_str(valid);
                    self.bytes.clear();
                    break;
                }
                Err(error) => {
                    let (valid, rest) = self.bytes.split_at(error.valid_up_to());
                    self.held
                        .push_str(std::str::from_utf8(valid).unwrap_or_default());
                    match error.error_len() {
                        // Bytes that can never be valid become U+FFFD.
                        Some(bad) => {
                            self.held.push('\u{fffd}');
                            self.bytes = rest[bad..].to_vec();
                        }
                        // The start of a character: wait for the rest.
                        None => {
                            self.bytes = rest.to_vec();
                            break;
                        }
                    }
                }
            }
        }

        // A stop string ends the output just before it.
        if let Some(at) = self
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            let before = self.held[..at].to_string();
            self.held.clear();
            if !before.is_empty() {
                on_text(&before);
            }
            return Flow::Stop;
        }

        // Hold back the longest tail that could still become a stop string.
        let mut keep = 0;
        for stop in self.stop {
            for len in (1..stop.len().min(self.held.len() + 1)).rev() {
                if stop.is_char_boundary(len) && self.held.ends_with(&stop[..len]) {
                    keep = keep.max(len);
                    break;
                }
            }
        }
        let deliver = self.held.len() - keep;
        if deliver > 0 {
            let ready: String = self.held.drain(..deliver).collect();
            if !on_text(&ready) {
                return Flow::Cancel;
            }
        }
        Flow::Continue
    }

    /// Delivers everything still held, at the end of a generation.
    fn finish(&mut self, on_text: &mut impl FnMut(&str) -> bool) {
        if !self.bytes.is_empty() {
            self.held.push_str(&String::from_utf8_lossy(&self.bytes));
            self.bytes.clear();
        }
        if !self.held.is_empty() {
            on_text(&self.held);
            self.held.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Model, SessionConfig};
    use crate::tensor::TensorType;
    use crate::testing::TinyModel;
    use std::sync::Arc;

    /// Generates 80 tokens greedily in a window of 32.
    fn overflowing(overflow: Overflow, keep: Option<usize>) -> (Stats, Vec<u32>) {
        let model =
            Arc::new(Model::from_bytes(TinyModel::new("llama", TensorType::F32).build()).unwrap());
        let config = SessionConfig {
            n_ctx: 32,
            threads: 1,
            ..Default::default()
        };
        let mut session = Session::new(model, config).unwrap();
        let options = Options {
            max_tokens: 80,
            sampler: SamplerConfig {
                temperature: 0.0,
                ..Default::default()
            },
            overflow,
            keep,
            ..Default::default()
        };
        let prompt = [1, 5, 9, 12, 7, 30, 8, 8, 21, 3];
        let stats = run_tokens(&mut session, &prompt, &options, |_| true).unwrap();
        (stats, session.tokens().to_vec())
    }

    #[test]
    fn a_full_context_slides_or_keeps_its_start() {
        // Stopping ends the answer where the window ends.
        let (stats, _) = overflowing(Overflow::Stop, None);
        assert_eq!((stats.finish, stats.dropped_tokens), (Finish::Context, 0));

        // The others make room and finish the answer.
        for (overflow, keep, kept) in [
            (Overflow::Slide, None, 1),
            (Overflow::Middle, Some(4), 4),
            (Overflow::Middle, None, 10),
        ] {
            let (stats, tokens) = overflowing(overflow, keep);
            assert_eq!(stats.finish, Finish::Length, "{overflow:?}");
            assert_eq!(stats.generated_tokens, 80);
            assert!(stats.dropped_tokens > 0 && tokens.len() <= 32);
            // The kept start is still the prompt's.
            let prompt = [1u32, 5, 9, 12, 7, 30, 8, 8, 21, 3];
            assert_eq!(tokens[..kept], prompt[..kept], "{overflow:?}");
        }
    }

    /// Feeds byte pieces through a stream and collects what it delivers.
    fn stream(stop: &[&str], pieces: &[&[u8]]) -> (Vec<String>, bool) {
        let stop: Vec<String> = stop.iter().map(|s| s.to_string()).collect();
        let mut text = TextStream::new(&stop);
        let mut out = Vec::new();
        let mut sink = |piece: &str| {
            out.push(piece.to_string());
            true
        };
        let mut stopped = false;
        for piece in pieces {
            if matches!(text.push(piece, &mut sink), Flow::Stop) {
                stopped = true;
                break;
            }
        }
        if !stopped {
            text.finish(&mut sink);
        }
        (out, stopped)
    }

    #[test]
    fn whole_characters_pass_straight_through() {
        let (out, stopped) = stream(&[], &[b"Hello", b", ", b"world"]);
        assert_eq!(out, ["Hello", ", ", "world"]);
        assert!(!stopped);
    }

    #[test]
    fn characters_split_across_tokens_are_joined() {
        // U+1F600 is four bytes, delivered here one or two at a time.
        let (out, _) = stream(&[], &[b"a\xf0", b"\x9f\x98", b"\x80b"]);
        assert_eq!(out, ["a", "\u{1f600}b"]);
        // An unfinished character at the very end becomes U+FFFD.
        let (out, _) = stream(&[], &[b"x\xe4\xb8"]);
        assert_eq!(out, ["x", "\u{fffd}"]);
        // Bytes that cannot start a character become U+FFFD at once.
        let (out, _) = stream(&[], &[b"a\xffb"]);
        assert_eq!(out, ["a\u{fffd}b"]);
    }

    #[test]
    fn stop_strings_end_the_output_before_them() {
        let (out, stopped) = stream(&["END"], &[b"one ", b"two E", b"N", b"D three"]);
        assert_eq!(out.concat(), "one two ");
        assert!(stopped);
        // The earliest of several stop strings wins.
        let (out, stopped) = stream(&["zz", "\n\n"], &[b"a\n\nbzz"]);
        assert_eq!(out.concat(), "a");
        assert!(stopped);
    }

    #[test]
    fn text_that_only_resembles_a_stop_string_is_released() {
        let (out, stopped) = stream(&["END"], &[b"an E", b"N", b"Tity"]);
        assert_eq!(out.concat(), "an ENTity");
        assert!(!stopped);
        // Held text is delivered when the generation ends.
        let (out, _) = stream(&["END"], &[b"the EN"]);
        assert_eq!(out.concat(), "the EN");
        // Empty stop strings are ignored.
        let (out, stopped) = stream(&[""], &[b"abc"]);
        assert_eq!((out.concat().as_str(), stopped), ("abc", false));
    }

    #[test]
    fn rates_handle_zero_time() {
        assert_eq!(rate(10, Duration::ZERO), 0.0);
        assert_eq!(rate(10, Duration::from_secs(2)), 5.0);
    }
}

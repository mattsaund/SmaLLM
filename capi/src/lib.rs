//! The C API of SmaLLM.
//!
//! This crate builds the engine as a shared and a static library that any
//! language with a C foreign function interface can call. The functions are
//! declared in `include/smallm.h`, which documents how to use them.
//!
//! Conventions:
//!
//! - Objects are opaque pointers made by a `_new` or `_load` function and
//!   released by the matching `_free` function.
//! - A function that can fail returns null or a negative [`Status`].
//!   `smallm_last_error` then describes the failure for the calling thread.
//! - Strings are UTF-8. Input strings end with a NUL byte unless a length is
//!   passed alongside.
//! - Passing a null pointer where an object is required is an error, not a
//!   crash.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Arc;

use smallm::chat::{Message, Role, Template};
use smallm::generate::{self, Finish, Options, Overflow};
use smallm::{Error, Model, Sampler, SamplerConfig, Session, SessionConfig};

/// Result codes. Zero is success, negative values are failures.
#[repr(i32)]
#[derive(Clone, Copy)]
enum Status {
    Ok = 0,
    /// An argument was null, malformed or out of range.
    Invalid = -1,
    /// The context window has no room for the request.
    ContextFull = -2,
    /// The model uses something the engine does not implement.
    Unsupported = -3,
    /// The model file could not be read or is not valid.
    BadFile = -4,
    /// The model has no chat format the engine recognizes.
    NoChatFormat = -5,
}

thread_local! {
    /// The message of the most recent failure on this thread.
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Records a failure message and returns its status.
fn fail(status: Status, message: impl Into<String>) -> Status {
    let message = message.into().replace('\0', " ");
    LAST_ERROR.with_borrow_mut(|slot| *slot = CString::new(message).unwrap_or_default());
    status
}

/// Records an engine error and returns the status that fits it.
fn fail_with(error: &Error) -> Status {
    let status = match error {
        Error::Io(_) | Error::Format(_) => Status::BadFile,
        Error::Unsupported(_) => Status::Unsupported,
        Error::Invalid(_) => Status::Invalid,
        Error::ContextFull { .. } => Status::ContextFull,
    };
    fail(status, error.to_string())
}

/// Reads a NUL-terminated string argument. Invalid UTF-8 is replaced.
///
/// # Safety
///
/// `text` must be null or point at a NUL-terminated string.
unsafe fn text_arg(text: *const c_char) -> Option<String> {
    if text.is_null() {
        return None;
    }
    // SAFETY: guaranteed by the caller.
    Some(
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Copies `bytes` into a caller's buffer of `len` bytes and returns the
/// number of bytes the full value needs. A NUL byte is added when there is
/// room for it.
///
/// # Safety
///
/// `buf` must be null or point at `len` writable bytes.
unsafe fn copy_out(bytes: &[u8], buf: *mut c_char, len: usize) -> usize {
    if !buf.is_null() && len > 0 {
        let n = bytes.len().min(len);
        // SAFETY: `n <= len`, and the caller guarantees `len` writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast::<u8>(), n);
            if n < len {
                *buf.add(n) = 0;
            }
        }
    }
    bytes.len()
}

/// A loaded model. Sessions share it, so it may be freed while they live.
pub struct SmallmModel(Arc<Model>);

/// One conversation with a model.
pub struct SmallmSession(Session);

/// A token sampler.
pub struct SmallmSampler(Sampler);

/// Options for a new session. Zero means "choose for me" in every field.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SmallmSessionParams {
    /// Context window in tokens.
    pub n_ctx: u32,
    /// Threads to compute with.
    pub n_threads: u32,
    /// Prompt tokens evaluated together.
    pub n_batch: u32,
    /// Where to compute. Zero is the CPU, the only device so far.
    pub device: u32,
    /// Threads to read prompts with.
    pub n_threads_batch: u32,
}

/// Options for sampling. See `smallm_sampler_default_params`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SmallmSamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u64,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl SmallmSamplerParams {
    fn to_config(self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k.max(0) as usize,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n.max(0) as usize,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
        }
    }
}

/// Options for `smallm_generate` and `smallm_chat`.
#[repr(C)]
pub struct SmallmGenerateParams {
    /// Stop after this many tokens. Zero for no limit.
    pub max_tokens: i32,
    pub sampler: SmallmSamplerParams,
    /// Strings that end the output when they appear, or null.
    pub stop: *const *const c_char,
    /// Number of entries in `stop`.
    pub n_stop: usize,
}

/// What a generation did.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SmallmStats {
    pub prompt_tokens: i32,
    /// Prompt tokens that were already in the context.
    pub reused_tokens: i32,
    pub generated_tokens: i32,
    pub prompt_seconds: f64,
    pub generate_seconds: f64,
    /// Why it ended: 0 the model finished, 1 token limit, 2 stop string,
    /// 3 context full, 4 the callback asked to stop.
    pub finish: i32,
}

/// Receives each piece of generated text. Return false to stop.
pub type SmallmTextCallback =
    Option<unsafe extern "C" fn(text: *const c_char, len: usize, user: *mut c_void) -> bool>;

/// The version of the library, such as "0.1.0".
#[unsafe(no_mangle)]
pub extern "C" fn smallm_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// The message of the last failure on this thread. Valid until the next
/// call into the library from the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn smallm_last_error() -> *const c_char {
    LAST_ERROR.with_borrow(|message| message.as_ptr())
}

/// Opens a model file. Returns null on failure.
///
/// # Safety
///
/// `path` must be null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_load(path: *const c_char) -> *mut SmallmModel {
    // SAFETY: guaranteed by the caller.
    let Some(path) = (unsafe { text_arg(path) }) else {
        fail(Status::Invalid, "the model path is null");
        return std::ptr::null_mut();
    };
    match Model::load(&path) {
        Ok(model) => Box::into_raw(Box::new(SmallmModel(Arc::new(model)))),
        Err(error) => {
            fail(fail_with(&error), format!("{path}: {error}"));
            std::ptr::null_mut()
        }
    }
}

/// Releases a model. Sessions made from it stay valid.
///
/// # Safety
///
/// `model` must be null or a pointer from `smallm_model_load` that has not
/// been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_free(model: *mut SmallmModel) {
    if !model.is_null() {
        // SAFETY: guaranteed by the caller.
        drop(unsafe { Box::from_raw(model) });
    }
}

/// Number of tokens in the model's vocabulary, or a negative status.
///
/// # Safety
///
/// `model` must be null or a live model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_n_vocab(model: *const SmallmModel) -> i32 {
    // SAFETY: guaranteed by the caller.
    match unsafe { model.as_ref() } {
        Some(model) => model.0.config().n_vocab as i32,
        None => fail(Status::Invalid, "the model is null") as i32,
    }
}

/// The context length the model was trained with, or a negative status.
///
/// # Safety
///
/// `model` must be null or a live model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_n_ctx_train(model: *const SmallmModel) -> i32 {
    // SAFETY: guaranteed by the caller.
    match unsafe { model.as_ref() } {
        Some(model) => model.0.config().n_ctx_train.min(i32::MAX as usize) as i32,
        None => fail(Status::Invalid, "the model is null") as i32,
    }
}

/// Total number of weights in the model. Zero for a null model.
///
/// # Safety
///
/// `model` must be null or a live model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_n_params(model: *const SmallmModel) -> u64 {
    // SAFETY: guaranteed by the caller.
    unsafe { model.as_ref() }.map_or(0, |model| model.0.n_params())
}

/// Copies a metadata value of the model file into `buf` as text, for
/// example the value of "general.name". Returns the length of the value,
/// which may exceed `len`, or zero when the key is missing.
///
/// # Safety
///
/// `model` must be null or a live model, `key` null or NUL-terminated, and
/// `buf` null or `len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_meta(
    model: *const SmallmModel,
    key: *const c_char,
    buf: *mut c_char,
    len: usize,
) -> usize {
    // SAFETY: guaranteed by the caller.
    let (Some(model), Some(key)) = (unsafe { model.as_ref() }, unsafe { text_arg(key) }) else {
        return 0;
    };
    let Some(value) = model.0.gguf().get(&key) else {
        return 0;
    };
    let text = match value.as_str() {
        Some(text) => text.to_string(),
        None => match (value.as_bool(), value.as_f64()) {
            (Some(flag), _) => flag.to_string(),
            (_, Some(number)) => number.to_string(),
            _ => return 0,
        },
    };
    // SAFETY: guaranteed by the caller.
    unsafe { copy_out(text.as_bytes(), buf, len) }
}

/// True when the model has a chat format `smallm_chat` can use.
///
/// # Safety
///
/// `model` must be null or a live model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_model_has_chat_format(model: *const SmallmModel) -> bool {
    // SAFETY: guaranteed by the caller.
    unsafe { model.as_ref() }
        .is_some_and(|model| model.0.chat_template().and_then(Template::detect).is_some())
}

/// Converts text to tokens.
///
/// Writes up to `max_tokens` ids to `tokens` and returns how many the text
/// has. When that is more than `max_tokens`, nothing useful was written:
/// call again with a larger buffer. Returns a negative status on failure.
///
/// # Safety
///
/// `model` must be null or a live model, `text` null or `text_len` readable
/// bytes, and `tokens` null or `max_tokens` writable ids.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_tokenize(
    model: *const SmallmModel,
    text: *const c_char,
    text_len: usize,
    tokens: *mut i32,
    max_tokens: usize,
    add_special: bool,
    parse_special: bool,
) -> i32 {
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        return fail(Status::Invalid, "the model is null") as i32;
    };
    if text.is_null() && text_len > 0 {
        return fail(Status::Invalid, "the text is null") as i32;
    }
    // SAFETY: guaranteed by the caller.
    let bytes = if text_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(text.cast::<u8>(), text_len) }
    };
    let ids =
        model
            .0
            .tokenizer()
            .encode(&String::from_utf8_lossy(bytes), add_special, parse_special);
    if !tokens.is_null() && ids.len() <= max_tokens {
        for (index, id) in ids.iter().enumerate() {
            // SAFETY: `index < ids.len() <= max_tokens`.
            unsafe { *tokens.add(index) = *id as i32 };
        }
    }
    ids.len().min(i32::MAX as usize) as i32
}

/// Copies the bytes a token stands for into `buf` and returns their count,
/// which may exceed `len`. Control tokens are empty unless `special` is
/// set. The bytes of one token are not always whole UTF-8 characters.
///
/// # Safety
///
/// `model` must be null or a live model, and `buf` null or `len` writable
/// bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_token_to_piece(
    model: *const SmallmModel,
    token: i32,
    buf: *mut c_char,
    len: usize,
    special: bool,
) -> i32 {
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        return fail(Status::Invalid, "the model is null") as i32;
    };
    let Ok(token) = u32::try_from(token) else {
        return 0;
    };
    let mut piece = Vec::new();
    model.0.tokenizer().append_piece(token, special, &mut piece);
    // SAFETY: guaranteed by the caller.
    (unsafe { copy_out(&piece, buf, len) }) as i32
}

/// True when `token` ends a generation.
///
/// # Safety
///
/// `model` must be null or a live model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_token_is_end(model: *const SmallmModel, token: i32) -> bool {
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        return false;
    };
    u32::try_from(token).is_ok_and(|token| model.0.tokenizer().is_end(token))
}

/// Session options with every field set to "choose for me".
#[unsafe(no_mangle)]
pub extern "C" fn smallm_session_default_params() -> SmallmSessionParams {
    SmallmSessionParams {
        n_ctx: 0,
        n_threads: 0,
        n_batch: 0,
        device: 0,
        n_threads_batch: 0,
    }
}

/// Starts an empty conversation with a model. `params` may be null for the
/// defaults. Returns null on failure.
///
/// # Safety
///
/// `model` must be null or a live model, and `params` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_session_new(
    model: *const SmallmModel,
    params: *const SmallmSessionParams,
) -> *mut SmallmSession {
    // SAFETY: guaranteed by the caller.
    let Some(model) = (unsafe { model.as_ref() }) else {
        fail(Status::Invalid, "the model is null");
        return std::ptr::null_mut();
    };
    // SAFETY: guaranteed by the caller.
    let params = unsafe { params.as_ref() }
        .copied()
        .unwrap_or_else(|| smallm_session_default_params());
    if params.device != 0 {
        fail(Status::Unsupported, "only device 0, the CPU, is available");
        return std::ptr::null_mut();
    }
    let config = SessionConfig {
        n_ctx: params.n_ctx as usize,
        threads: params.n_threads as usize,
        batch_threads: params.n_threads_batch as usize,
        batch: params.n_batch as usize,
        kernels: None,
    };
    match Session::new(Arc::clone(&model.0), config) {
        Ok(session) => Box::into_raw(Box::new(SmallmSession(session))),
        Err(error) => {
            fail_with(&error);
            std::ptr::null_mut()
        }
    }
}

/// Releases a session.
///
/// # Safety
///
/// `session` must be null or a pointer from `smallm_session_new` that has
/// not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_session_free(session: *mut SmallmSession) {
    if !session.is_null() {
        // SAFETY: guaranteed by the caller.
        drop(unsafe { Box::from_raw(session) });
    }
}

/// The context window of a session in tokens, or a negative status.
///
/// # Safety
///
/// `session` must be null or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_session_n_ctx(session: *const SmallmSession) -> i32 {
    // SAFETY: guaranteed by the caller.
    match unsafe { session.as_ref() } {
        Some(session) => session.0.n_ctx().min(i32::MAX as usize) as i32,
        None => fail(Status::Invalid, "the session is null") as i32,
    }
}

/// Number of tokens in the context, or a negative status.
///
/// # Safety
///
/// `session` must be null or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_session_n_tokens(session: *const SmallmSession) -> i32 {
    // SAFETY: guaranteed by the caller.
    match unsafe { session.as_ref() } {
        Some(session) => session.0.tokens().len().min(i32::MAX as usize) as i32,
        None => fail(Status::Invalid, "the session is null") as i32,
    }
}

/// Removes every token from position `n_tokens` on.
///
/// # Safety
///
/// `session` must be null or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_session_truncate(session: *mut SmallmSession, n_tokens: usize) {
    // SAFETY: guaranteed by the caller.
    if let Some(session) = unsafe { session.as_mut() } {
        session.0.truncate(n_tokens);
    }
}

/// Appends tokens to the context and computes the logits for the token
/// after them. Returns zero or a negative status.
///
/// # Safety
///
/// `session` must be null or a live session, and `tokens` null or `n`
/// readable ids.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_eval(
    session: *mut SmallmSession,
    tokens: *const i32,
    n: usize,
) -> i32 {
    // SAFETY: guaranteed by the caller.
    let Some(session) = (unsafe { session.as_mut() }) else {
        return fail(Status::Invalid, "the session is null") as i32;
    };
    if tokens.is_null() && n > 0 {
        return fail(Status::Invalid, "the tokens are null") as i32;
    }
    // SAFETY: guaranteed by the caller.
    let raw = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(tokens, n) }
    };
    let Ok(ids) = raw
        .iter()
        .map(|&id| u32::try_from(id))
        .collect::<Result<Vec<u32>, _>>()
    else {
        return fail(Status::Invalid, "token ids cannot be negative") as i32;
    };
    match session.0.eval(&ids) {
        Ok(()) => Status::Ok as i32,
        Err(error) => fail_with(&error) as i32,
    }
}

/// The scores for the next token, one float per vocabulary entry. Valid
/// until the session is next changed. Null for a null session.
///
/// # Safety
///
/// `session` must be null or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_logits(session: *const SmallmSession) -> *const f32 {
    // SAFETY: guaranteed by the caller.
    unsafe { session.as_ref() }.map_or(std::ptr::null(), |session| session.0.logits().as_ptr())
}

/// The sampling options most chats use.
#[unsafe(no_mangle)]
pub extern "C" fn smallm_sampler_default_params() -> SmallmSamplerParams {
    let defaults = SamplerConfig::default();
    SmallmSamplerParams {
        temperature: defaults.temperature,
        top_k: defaults.top_k as i32,
        top_p: defaults.top_p,
        min_p: defaults.min_p,
        repeat_penalty: defaults.repeat_penalty,
        repeat_last_n: defaults.repeat_last_n as i32,
        seed: defaults.seed,
        frequency_penalty: defaults.frequency_penalty,
        presence_penalty: defaults.presence_penalty,
    }
}

/// Creates a sampler. `params` may be null for the defaults.
///
/// # Safety
///
/// `params` must be null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_sampler_new(
    params: *const SmallmSamplerParams,
) -> *mut SmallmSampler {
    // SAFETY: guaranteed by the caller.
    let params = unsafe { params.as_ref() }
        .copied()
        .unwrap_or_else(|| smallm_sampler_default_params());
    Box::into_raw(Box::new(SmallmSampler(Sampler::new(params.to_config()))))
}

/// Releases a sampler.
///
/// # Safety
///
/// `sampler` must be null or a pointer from `smallm_sampler_new` that has
/// not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_sampler_free(sampler: *mut SmallmSampler) {
    if !sampler.is_null() {
        // SAFETY: guaranteed by the caller.
        drop(unsafe { Box::from_raw(sampler) });
    }
}

/// Picks the next token from a session's logits. Returns the token id, or
/// a negative status.
///
/// # Safety
///
/// `sampler` and `session` must each be null or live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_sample(
    sampler: *mut SmallmSampler,
    session: *const SmallmSession,
) -> i32 {
    // SAFETY: guaranteed by the caller.
    let (Some(sampler), Some(session)) = (unsafe { sampler.as_mut() }, unsafe { session.as_ref() })
    else {
        return fail(Status::Invalid, "the sampler or the session is null") as i32;
    };
    sampler.0.sample(session.0.logits(), session.0.tokens()) as i32
}

/// Generation options with no token limit and the default sampler.
#[unsafe(no_mangle)]
pub extern "C" fn smallm_generate_default_params() -> SmallmGenerateParams {
    SmallmGenerateParams {
        max_tokens: 0,
        sampler: smallm_sampler_default_params(),
        stop: std::ptr::null(),
        n_stop: 0,
    }
}

/// Turns C generation options into the engine's.
///
/// # Safety
///
/// `params` must be null or valid, with `stop` holding `n_stop` strings.
unsafe fn options(params: *const SmallmGenerateParams) -> Options {
    // SAFETY: guaranteed by the caller.
    let Some(params) = (unsafe { params.as_ref() }) else {
        return Options::default();
    };
    let mut stop = Vec::new();
    if !params.stop.is_null() {
        for index in 0..params.n_stop {
            // SAFETY: the caller guarantees `n_stop` entries.
            stop.extend(unsafe { text_arg(*params.stop.add(index)) });
        }
    }
    Options {
        max_tokens: params.max_tokens.max(0) as usize,
        sampler: params.sampler.to_config(),
        stop,
        show_special: false,
        ..Options::default()
    }
}

/// Runs a generation and reports its outcome through `stats`.
///
/// # Safety
///
/// `callback` must be null or callable with `user`, and `stats` null or
/// writable.
unsafe fn run(
    session: &mut Session,
    tokens: &[u32],
    options: &Options,
    callback: SmallmTextCallback,
    user: *mut c_void,
    stats: *mut SmallmStats,
) -> i32 {
    let result = generate::run_tokens(session, tokens, options, |piece| match callback {
        // SAFETY: the caller guarantees the callback accepts `user`. The
        // text is not NUL-terminated, which is why its length is passed.
        Some(callback) => unsafe { callback(piece.as_ptr().cast(), piece.len(), user) },
        None => true,
    });
    match result {
        Ok(done) => {
            // SAFETY: guaranteed by the caller.
            if let Some(stats) = unsafe { stats.as_mut() } {
                *stats = SmallmStats {
                    prompt_tokens: done.prompt_tokens as i32,
                    reused_tokens: done.reused_tokens as i32,
                    generated_tokens: done.generated_tokens as i32,
                    prompt_seconds: done.prompt_time.as_secs_f64(),
                    generate_seconds: done.generate_time.as_secs_f64(),
                    finish: match done.finish {
                        Finish::End => 0,
                        Finish::Length => 1,
                        Finish::Stop => 2,
                        Finish::Context => 3,
                        Finish::Canceled => 4,
                    },
                };
            }
            Status::Ok as i32
        }
        Err(error) => fail_with(&error) as i32,
    }
}

/// Completes `prompt` as plain text, with no chat format.
///
/// `callback` receives each piece of output as it is produced. `params`,
/// `callback` and `stats` may be null. Returns zero or a negative status.
///
/// # Safety
///
/// `session` must be null or a live session, `prompt` null or
/// NUL-terminated, and the other pointers null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_generate(
    session: *mut SmallmSession,
    prompt: *const c_char,
    params: *const SmallmGenerateParams,
    callback: SmallmTextCallback,
    user: *mut c_void,
    stats: *mut SmallmStats,
) -> i32 {
    // SAFETY: guaranteed by the caller.
    let (Some(session), Some(prompt)) = (unsafe { session.as_mut() }, unsafe { text_arg(prompt) })
    else {
        return fail(Status::Invalid, "the session or the prompt is null") as i32;
    };
    let tokens = session.0.model().tokenizer().encode(&prompt, true, true);
    // SAFETY: guaranteed by the caller.
    unsafe {
        run(
            &mut session.0,
            &tokens,
            &options(params),
            callback,
            user,
            stats,
        )
    }
}

/// Answers a conversation in the model's own chat format.
///
/// `roles` and `contents` each hold `n_messages` strings. A role is
/// "system", "user" or "assistant". The session keeps its context between
/// calls, so a conversation that grew by one message only pays for that
/// message. When the conversation outgrows the context window, the oldest
/// turns are left out.
///
/// # Safety
///
/// `session` must be null or a live session, `roles` and `contents` null
/// or `n_messages` NUL-terminated strings each, and the other pointers
/// null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn smallm_chat(
    session: *mut SmallmSession,
    roles: *const *const c_char,
    contents: *const *const c_char,
    n_messages: usize,
    params: *const SmallmGenerateParams,
    callback: SmallmTextCallback,
    user: *mut c_void,
    stats: *mut SmallmStats,
) -> i32 {
    /// Tokens kept free for the answer when a long chat is trimmed.
    const ANSWER_RESERVE: usize = 256;

    // SAFETY: guaranteed by the caller.
    let Some(session) = (unsafe { session.as_mut() }) else {
        return fail(Status::Invalid, "the session is null") as i32;
    };
    if roles.is_null() || contents.is_null() || n_messages == 0 {
        return fail(Status::Invalid, "the conversation is empty") as i32;
    }
    let model = Arc::clone(session.0.model());
    let Some(template) = model.chat_template().and_then(Template::detect) else {
        return fail(
            Status::NoChatFormat,
            "the model has no chat format. Use smallm_generate.",
        ) as i32;
    };
    let mut messages = Vec::with_capacity(n_messages);
    for index in 0..n_messages {
        // SAFETY: the caller guarantees `n_messages` entries in both arrays.
        let (role, content) =
            unsafe { (text_arg(*roles.add(index)), text_arg(*contents.add(index))) };
        let (Some(role), Some(content)) = (role, content) else {
            return fail(Status::Invalid, format!("message {index} is null")) as i32;
        };
        let Some(role) = Role::from_name(&role) else {
            return fail(
                Status::Invalid,
                format!("message {index} has the unknown role '{role}'"),
            ) as i32;
        };
        messages.push(Message::new(role, content));
    }
    let (tokens, _) = generate::chat_tokens_fitting(
        model.tokenizer(),
        template,
        &messages,
        session.0.n_ctx(),
        ANSWER_RESERVE,
        Overflow::default(),
    );
    // SAFETY: guaranteed by the caller.
    unsafe {
        run(
            &mut session.0,
            &tokens,
            &options(params),
            callback,
            user,
            stats,
        )
    }
}

#[cfg(test)]
mod tests {
    //! The C functions, called the way a C program would call them.

    use super::*;
    use std::ffi::CString;
    use std::path::PathBuf;
    use std::ptr::{null, null_mut};

    /// The text of the last error on this thread.
    fn last_error() -> String {
        // SAFETY: the pointer is a NUL-terminated string owned by this thread.
        unsafe { CStr::from_ptr(smallm_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    /// A model file from the folder in `SMALLM_TEST_MODELS`. `None` means
    /// the test may be skipped. A missing file is a failure when its name is
    /// listed in `SMALLM_TEST_REQUIRE`.
    fn test_model(name: &str) -> Option<CString> {
        let required = std::env::var("SMALLM_TEST_REQUIRE").unwrap_or_default();
        let is_required = required.split(',').any(|required| required.trim() == name);
        let path = std::env::var_os("SMALLM_TEST_MODELS")
            .map(|dir| PathBuf::from(dir).join(format!("{name}.gguf")));
        match path {
            Some(path) if path.is_file() => {
                Some(CString::new(path.to_string_lossy().into_owned()).unwrap())
            }
            _ if is_required => panic!("{name}.gguf is required but was not found"),
            _ => {
                eprintln!("skipping: {name}.gguf is not in SMALLM_TEST_MODELS");
                None
            }
        }
    }

    /// Appends each piece of generated text to the `String` behind `user`.
    unsafe extern "C" fn collect(text: *const c_char, len: usize, user: *mut c_void) -> bool {
        // SAFETY: the test passes a `String` as `user`, and the engine
        // passes `len` valid bytes.
        unsafe {
            let bytes = std::slice::from_raw_parts(text.cast::<u8>(), len);
            (*user.cast::<String>()).push_str(&String::from_utf8_lossy(bytes));
        }
        true
    }

    /// Stops the generation at the first piece.
    unsafe extern "C" fn refuse(_text: *const c_char, _len: usize, _user: *mut c_void) -> bool {
        false
    }

    #[test]
    fn the_version_is_the_crate_version() {
        // SAFETY: the pointer is a static NUL-terminated string.
        let version = unsafe { CStr::from_ptr(smallm_version()) };
        assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        // SAFETY: every pointer passed here is null, which the functions
        // must accept.
        unsafe {
            assert!(smallm_model_load(null()).is_null());
            assert!(!last_error().is_empty());
            smallm_model_free(null_mut());
            assert_eq!(smallm_model_n_vocab(null()), Status::Invalid as i32);
            assert_eq!(smallm_model_n_ctx_train(null()), Status::Invalid as i32);
            assert_eq!(smallm_model_n_params(null()), 0);
            assert_eq!(smallm_model_meta(null(), null(), null_mut(), 0), 0);
            assert!(!smallm_model_has_chat_format(null()));
            assert_eq!(
                smallm_tokenize(null(), null(), 0, null_mut(), 0, true, false),
                Status::Invalid as i32
            );
            assert_eq!(
                smallm_token_to_piece(null(), 0, null_mut(), 0, false),
                Status::Invalid as i32
            );
            assert!(!smallm_token_is_end(null(), 0));
            assert!(smallm_session_new(null(), null()).is_null());
            smallm_session_free(null_mut());
            assert_eq!(smallm_session_n_ctx(null()), Status::Invalid as i32);
            assert_eq!(smallm_session_n_tokens(null()), Status::Invalid as i32);
            smallm_session_truncate(null_mut(), 0);
            assert_eq!(smallm_eval(null_mut(), null(), 0), Status::Invalid as i32);
            assert!(smallm_logits(null()).is_null());
            smallm_sampler_free(null_mut());
            assert_eq!(smallm_sample(null_mut(), null()), Status::Invalid as i32);
            assert_eq!(
                smallm_generate(null_mut(), null(), null(), None, null_mut(), null_mut()),
                Status::Invalid as i32
            );
            let status = smallm_chat(
                null_mut(),
                null(),
                null(),
                0,
                null(),
                None,
                null_mut(),
                null_mut(),
            );
            assert_eq!(status, Status::Invalid as i32);
        }
    }

    #[test]
    fn a_missing_file_says_why() {
        let path = CString::new("/no/such/folder/model.gguf").unwrap();
        // SAFETY: the path is a NUL-terminated string.
        let model = unsafe { smallm_model_load(path.as_ptr()) };
        assert!(model.is_null());
        assert!(last_error().contains("model.gguf"), "{}", last_error());
    }

    #[test]
    fn defaults_are_usable() {
        let session = smallm_session_default_params();
        assert_eq!(
            (
                session.n_ctx,
                session.n_threads,
                session.n_batch,
                session.device
            ),
            (0, 0, 0, 0)
        );
        let sampler = smallm_sampler_default_params();
        assert!(sampler.temperature > 0.0 && sampler.top_k > 0);
        let generate = smallm_generate_default_params();
        assert_eq!(generate.max_tokens, 0);
        assert!(generate.stop.is_null());
        // SAFETY: null parameters select the defaults, and the sampler is
        // freed once.
        unsafe {
            let sampler = smallm_sampler_new(null());
            assert!(!sampler.is_null());
            smallm_sampler_free(sampler);
        }
    }

    #[test]
    fn a_model_runs_through_the_api() {
        let Some(path) = test_model("stories260K") else {
            return;
        };
        // SAFETY: every pointer below is either null where null is allowed,
        // or points at live data of the stated length.
        unsafe {
            let model = smallm_model_load(path.as_ptr());
            assert!(!model.is_null(), "{}", last_error());
            let n_vocab = smallm_model_n_vocab(model);
            assert!(n_vocab > 0);
            assert!(smallm_model_n_ctx_train(model) > 0);
            assert!(smallm_model_n_params(model) > 100_000);

            // Metadata: a size query, then the value.
            let key = CString::new("general.architecture").unwrap();
            let needed = smallm_model_meta(model, key.as_ptr(), null_mut(), 0);
            let mut value = vec![0u8; needed + 1];
            smallm_model_meta(model, key.as_ptr(), value.as_mut_ptr().cast(), value.len());
            assert_eq!(
                CStr::from_bytes_until_nul(&value)
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "llama"
            );
            let missing = CString::new("no.such.key").unwrap();
            assert_eq!(smallm_model_meta(model, missing.as_ptr(), null_mut(), 0), 0);

            // Tokens: a buffer that is too small reports the size it needs.
            let text = "Once upon a time";
            let count = smallm_tokenize(
                model,
                text.as_ptr().cast(),
                text.len(),
                null_mut(),
                0,
                true,
                false,
            );
            assert!(count > 1);
            let mut tokens = vec![0i32; count as usize];
            let written = smallm_tokenize(
                model,
                text.as_ptr().cast(),
                text.len(),
                tokens.as_mut_ptr(),
                tokens.len(),
                true,
                false,
            );
            assert_eq!(written, count);
            let mut round_trip = Vec::new();
            for &token in &tokens {
                let mut piece = [0u8; 64];
                let len = smallm_token_to_piece(
                    model,
                    token,
                    piece.as_mut_ptr().cast(),
                    piece.len(),
                    false,
                );
                round_trip.extend_from_slice(&piece[..len as usize]);
            }
            assert_eq!(String::from_utf8(round_trip).unwrap().trim_start(), text);

            // A session outlives the model handle it was made from.
            let params = SmallmSessionParams {
                n_ctx: 256,
                n_threads: 2,
                n_batch: 0,
                device: 0,
                n_threads_batch: 3,
            };
            let session = smallm_session_new(model, &params);
            assert!(!session.is_null(), "{}", last_error());
            let gpu = SmallmSessionParams {
                device: 1,
                ..params
            };
            assert!(smallm_session_new(model, &gpu).is_null());
            assert!(last_error().contains("CPU"), "{}", last_error());
            smallm_model_free(model);
            assert_eq!(smallm_session_n_ctx(session), 256);

            // Low level: evaluate, read the logits, sample.
            assert_eq!(smallm_eval(session, tokens.as_ptr(), tokens.len()), 0);
            assert_eq!(smallm_session_n_tokens(session), count);
            let logits = std::slice::from_raw_parts(smallm_logits(session), n_vocab as usize);
            assert!(logits.iter().all(|logit| logit.is_finite()));
            let greedy = SmallmSamplerParams {
                temperature: 0.0,
                ..smallm_sampler_default_params()
            };
            let sampler = smallm_sampler_new(&greedy);
            let token = smallm_sample(sampler, session);
            assert!((0..n_vocab).contains(&token));
            smallm_sampler_free(sampler);
            assert_eq!(
                smallm_eval(session, [-1].as_ptr(), 1),
                Status::Invalid as i32
            );
            let too_many = vec![1i32; 300];
            assert_eq!(
                smallm_eval(session, too_many.as_ptr(), too_many.len()),
                Status::ContextFull as i32
            );
            smallm_session_truncate(session, 0);
            assert_eq!(smallm_session_n_tokens(session), 0);

            // High level: generate text into a callback.
            let prompt = CString::new(text).unwrap();
            let mut options = smallm_generate_default_params();
            options.max_tokens = 24;
            options.sampler.temperature = 0.0;
            let mut story = String::new();
            let mut stats = SmallmStats::default();
            let user = (&raw mut story).cast::<c_void>();
            let status = smallm_generate(
                session,
                prompt.as_ptr(),
                &options,
                Some(collect),
                user,
                &mut stats,
            );
            assert_eq!(status, 0, "{}", last_error());
            assert!(story.len() > 20, "{story:?}");
            assert_eq!(stats.generated_tokens, 24);
            assert_eq!(stats.prompt_tokens, count);
            assert_eq!(stats.finish, 1);

            // The same prompt again reuses the context and repeats the text.
            let mut again = String::new();
            let user = (&raw mut again).cast::<c_void>();
            assert_eq!(
                smallm_generate(
                    session,
                    prompt.as_ptr(),
                    &options,
                    Some(collect),
                    user,
                    &mut stats
                ),
                0
            );
            assert_eq!(again, story);
            // All but the last token, which is evaluated again for its logits.
            assert_eq!(stats.reused_tokens, count - 1);

            // A stop string ends the output before it, and a callback that
            // returns false ends it at once.
            let first_word = story.split_whitespace().nth(1).unwrap();
            let stop_text = CString::new(first_word).unwrap();
            let stops = [stop_text.as_ptr()];
            options.stop = stops.as_ptr();
            options.n_stop = 1;
            let mut stopped = String::new();
            let user = (&raw mut stopped).cast::<c_void>();
            assert_eq!(
                smallm_generate(
                    session,
                    prompt.as_ptr(),
                    &options,
                    Some(collect),
                    user,
                    &mut stats
                ),
                0
            );
            assert_eq!(stats.finish, 2);
            assert!(!stopped.contains(first_word), "{stopped:?}");
            options.stop = null();
            options.n_stop = 0;
            assert_eq!(
                smallm_generate(
                    session,
                    prompt.as_ptr(),
                    &options,
                    Some(refuse),
                    null_mut(),
                    &mut stats
                ),
                0
            );
            assert_eq!(stats.finish, 4);

            // This model has no chat format, and says so.
            let roles = [c"user".as_ptr()];
            let contents = [c"Hello".as_ptr()];
            let status = smallm_chat(
                session,
                roles.as_ptr(),
                contents.as_ptr(),
                1,
                null(),
                None,
                null_mut(),
                null_mut(),
            );
            assert_eq!(status, Status::NoChatFormat as i32);
            smallm_session_free(session);
        }
    }

    #[test]
    fn a_chat_runs_through_the_api() {
        let Some(path) = test_model("SmolLM2-135M-Instruct-Q4_K_M") else {
            return;
        };
        // SAFETY: as in the test above.
        unsafe {
            let model = smallm_model_load(path.as_ptr());
            assert!(!model.is_null(), "{}", last_error());
            assert!(smallm_model_has_chat_format(model));
            let session = smallm_session_new(model, null());
            assert!(!session.is_null(), "{}", last_error());
            let mut options = smallm_generate_default_params();
            options.max_tokens = 16;
            options.sampler.temperature = 0.0;

            let roles = [c"system".as_ptr(), c"user".as_ptr()];
            let contents = [
                c"You are a helpful assistant.".as_ptr(),
                c"Name one color.".as_ptr(),
            ];
            let mut answer = String::new();
            let mut stats = SmallmStats::default();
            let user = (&raw mut answer).cast::<c_void>();
            let status = smallm_chat(
                session,
                roles.as_ptr(),
                contents.as_ptr(),
                2,
                &options,
                Some(collect),
                user,
                &mut stats,
            );
            assert_eq!(status, 0, "{}", last_error());
            assert!(!answer.trim().is_empty());
            assert!(stats.generated_tokens > 0 && stats.prompt_tokens > 10);

            // An unknown role is refused with its position.
            let bad_roles = [c"narrator".as_ptr()];
            let status = smallm_chat(
                session,
                bad_roles.as_ptr(),
                contents.as_ptr(),
                1,
                &options,
                None,
                null_mut(),
                null_mut(),
            );
            assert_eq!(status, Status::Invalid as i32);
            assert!(last_error().contains("narrator"), "{}", last_error());
            smallm_session_free(session);
            smallm_model_free(model);
        }
    }
}

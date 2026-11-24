//! The OpenAI-compatible API.
//!
//! | Method | Path                   | Purpose                           |
//! |--------|------------------------|-----------------------------------|
//! | GET    | `/health`              | Liveness check.                   |
//! | GET    | `/v1/models`           | Names the loaded model.           |
//! | POST   | `/v1/chat/completions` | Answers a conversation.           |
//! | POST   | `/v1/completions`      | Completes a plain text prompt.    |
//! | POST   | `/tokenize`            | Text to token ids.                |
//! | POST   | `/detokenize`          | Token ids to text.                |
//!
//! One model is loaded and requests take turns with it. The session keeps
//! its context between requests, so a chat that grows by one message only
//! pays for the new message.

use std::io::{BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use smallm::chat::{Message, Role, Template};
use smallm::generate::{self, Finish, Options, Stats};
use smallm::{SamplerConfig, Session};

use super::http::{self, Request};
use crate::json::Json;

/// Connections served at once. More than this are turned away, which keeps
/// a flood of requests from exhausting threads.
const MAX_CONNECTIONS: usize = 32;

/// How long a client may take to send its request.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Tokens kept free for the answer when a long chat is trimmed to fit.
const ANSWER_RESERVE: usize = 256;

/// Everything the request handlers share.
pub struct Server {
    pub session: Mutex<Session>,
    pub template: Option<Template>,
    /// The name clients see for the loaded model.
    pub model_id: String,
    /// When set, requests must carry it as a bearer token.
    pub api_key: Option<String>,
    connections: AtomicUsize,
    next_id: AtomicU64,
}

impl Server {
    pub fn new(
        session: Session,
        template: Option<Template>,
        model_id: String,
        api_key: Option<String>,
    ) -> Server {
        Server {
            session: Mutex::new(session),
            template,
            model_id,
            api_key,
            connections: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
        }
    }

    /// Accepts connections until the process ends.
    pub fn run(self: Arc<Server>, listener: TcpListener) {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let server = Arc::clone(&self);
            if server.connections.fetch_add(1, Ordering::Relaxed) >= MAX_CONNECTIONS {
                server.connections.fetch_sub(1, Ordering::Relaxed);
                let mut stream = stream;
                let _ = send_error(&mut stream, 503, "the server is busy");
                continue;
            }
            let spawned = std::thread::Builder::new()
                .name("smallm-http".to_string())
                .spawn(move || {
                    server.connection(stream);
                    server.connections.fetch_sub(1, Ordering::Relaxed);
                });
            if spawned.is_err() {
                self.connections.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Serves the one request of a connection.
    fn connection(&self, stream: TcpStream) {
        let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
        let _ = stream.set_nodelay(true);
        let Ok(mut writer) = stream.try_clone() else {
            return;
        };
        let mut reader = BufReader::new(stream);
        let outcome = match Request::read(&mut reader) {
            Ok(Some(request)) => self.route(&request, &mut writer),
            Ok(None) => Ok(()),
            Err(rejection) => send_error(&mut writer, rejection.status, rejection.message),
        };
        // A failed write means the client left. There is nobody to tell.
        let _ = outcome;
    }

    /// Sends a request to its handler.
    fn route(&self, request: &Request, out: &mut impl Write) -> std::io::Result<()> {
        if let Some(key) = &self.api_key {
            let sent = request
                .header("authorization")
                .and_then(|value| value.strip_prefix("Bearer "));
            if sent != Some(key.as_str()) {
                return send_error(out, 401, "missing or wrong API key");
            }
        }
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/health" | "/v1/health") => {
                send_json(out, 200, &Json::object([("status", Json::string("ok"))]))
            }
            ("GET", "/v1/models") => {
                let model = Json::object([
                    ("id", Json::string(&self.model_id)),
                    ("object", Json::string("model")),
                    ("owned_by", Json::string("smallm")),
                ]);
                let list = Json::object([
                    ("object", Json::string("list")),
                    ("data", Json::Array(vec![model])),
                ]);
                send_json(out, 200, &list)
            }
            ("POST", "/v1/chat/completions") => self.generate(request, out, Kind::Chat),
            ("POST", "/v1/completions") => self.generate(request, out, Kind::Text),
            ("POST", "/tokenize") => self.tokens(request, out, true),
            ("POST", "/detokenize") => self.tokens(request, out, false),
            (
                _,
                "/health"
                | "/v1/health"
                | "/v1/models"
                | "/v1/chat/completions"
                | "/v1/completions"
                | "/tokenize"
                | "/detokenize",
            ) => send_error(out, 405, "method not allowed"),
            _ => send_error(out, 404, "no such endpoint"),
        }
    }

    /// Handles `/tokenize` and `/detokenize`, in the shape llama.cpp's
    /// server gave them.
    ///
    /// - Tokenize: `{"content": text}` to `{"tokens": [ids]}`. `add_special`
    ///   adds the tokens the model starts with, and `parse_special` reads
    ///   control tokens in the text. Both are off unless asked for.
    /// - Detokenize: `{"tokens": [ids]}` to `{"content": text}`.
    fn tokens(&self, request: &Request, out: &mut impl Write, encode: bool) -> std::io::Result<()> {
        let body = match std::str::from_utf8(&request.body)
            .map_err(|_| "the body is not UTF-8".to_string())
            .and_then(Json::parse)
        {
            Ok(body) => body,
            Err(message) => return send_error(out, 400, &message),
        };
        // The model is shared, so this does not wait for a generation.
        let model = match self.session.lock() {
            Ok(session) => session.model().clone(),
            Err(poisoned) => poisoned.into_inner().model().clone(),
        };
        let tokenizer = model.tokenizer();
        if encode {
            let Some(text) = body.get("content").and_then(Json::as_str) else {
                return send_error(out, 400, "'content' must be a string");
            };
            let flag = |name: &str| body.get(name).and_then(Json::as_bool).unwrap_or(false);
            let ids = tokenizer.encode(text, flag("add_special"), flag("parse_special"));
            let ids = ids.into_iter().map(|id| Json::Number(id as f64)).collect();
            send_json(out, 200, &Json::object([("tokens", Json::Array(ids))]))
        } else {
            let Some(ids) = body.get("tokens").and_then(Json::as_array) else {
                return send_error(out, 400, "'tokens' must be an array of token ids");
            };
            let mut bytes = Vec::new();
            for id in ids {
                let known = id.as_usize().filter(|id| *id < tokenizer.n_tokens());
                let Some(id) = known else {
                    return send_error(out, 400, "'tokens' holds something that is not a token id");
                };
                tokenizer.append_piece(id as u32, true, &mut bytes);
            }
            let text = String::from_utf8_lossy(&bytes).into_owned();
            send_json(out, 200, &Json::object([("content", Json::string(text))]))
        }
    }

    /// Handles both completion endpoints.
    fn generate(&self, request: &Request, out: &mut impl Write, kind: Kind) -> std::io::Result<()> {
        let parsed = std::str::from_utf8(&request.body)
            .map_err(|_| "the body is not UTF-8".to_string())
            .and_then(Json::parse)
            .and_then(|body| Job::from_json(&body, kind));
        let job = match parsed {
            Ok(job) => job,
            Err(message) => return send_error(out, 400, &message),
        };

        // Requests take turns with the one session.
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let tokens = match &job.input {
            Input::Text(prompt) => session.model().tokenizer().encode(prompt, true, true),
            Input::Chat(messages) => {
                let Some(template) = self.template else {
                    return send_error(
                        out,
                        400,
                        "this model has no chat format. Use /v1/completions.",
                    );
                };
                let tokenizer = session.model().tokenizer();
                generate::chat_tokens_fitting(
                    tokenizer,
                    template,
                    messages,
                    session.n_ctx(),
                    ANSWER_RESERVE,
                )
                .0
            }
        };
        if tokens.is_empty() {
            return send_error(out, 400, "the prompt is empty");
        }
        if tokens.len() >= session.n_ctx() {
            return send_error(out, 400, "the prompt does not fit in the context window");
        }

        let id = format!(
            "{}-{:x}",
            kind.id_prefix(),
            self.next_id.fetch_add(1, Ordering::Relaxed)
        );
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |time| time.as_secs());
        let envelope = Envelope {
            id: &id,
            created,
            model: &self.model_id,
            kind,
        };

        if job.stream {
            http::start_events(out)?;
            if kind == Kind::Chat {
                http::send_event(
                    out,
                    &envelope.chunk(Some("assistant"), Some(""), None).to_text(),
                )?;
            }
            let result = generate::run_tokens(&mut session, &tokens, &job.options, |piece| {
                http::send_event(out, &envelope.chunk(None, Some(piece), None).to_text()).is_ok()
            });
            match result {
                Ok(stats) => {
                    log_request(&id, &stats);
                    let mut last = envelope.chunk(None, None, Some(finish_reason(stats.finish)));
                    if let Json::Object(members) = &mut last {
                        members.push(("timings".to_string(), timings(&stats)));
                    }
                    http::send_event(out, &last.to_text())?;
                    if job.include_usage {
                        http::send_event(out, &envelope.usage_chunk(&stats).to_text())?;
                    }
                    http::send_event(out, "[DONE]")
                }
                Err(error) => http::send_event(out, &error_json(&error.to_string()).to_text()),
            }
        } else {
            let mut text = String::new();
            let result = generate::run_tokens(&mut session, &tokens, &job.options, |piece| {
                text.push_str(piece);
                true
            });
            match result {
                Ok(stats) => {
                    log_request(&id, &stats);
                    send_json(out, 200, &envelope.complete(&text, &stats))
                }
                Err(error) => send_error(out, 500, &error.to_string()),
            }
        }
    }
}

/// Which completion endpoint a request came in on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chat,
    Text,
}

impl Kind {
    fn id_prefix(self) -> &'static str {
        match self {
            Kind::Chat => "chatcmpl",
            Kind::Text => "cmpl",
        }
    }
}

/// What to generate from.
#[derive(Debug)]
enum Input {
    Chat(Vec<Message>),
    Text(String),
}

/// A parsed completion request.
#[derive(Debug)]
struct Job {
    input: Input,
    options: Options,
    stream: bool,
    /// Send token counts as a last event of a stream.
    include_usage: bool,
}

impl Job {
    /// Reads a request body. Unknown fields are ignored, as API clients
    /// send many that only matter to other servers.
    fn from_json(body: &Json, kind: Kind) -> Result<Job, String> {
        let input = match kind {
            Kind::Chat => {
                let list = body
                    .get("messages")
                    .and_then(Json::as_array)
                    .ok_or("'messages' must be an array")?;
                let messages = list.iter().map(message).collect::<Result<Vec<_>, _>>()?;
                if messages.is_empty() {
                    return Err("'messages' is empty".to_string());
                }
                Input::Chat(messages)
            }
            Kind::Text => match body.get("prompt") {
                Some(Json::String(prompt)) => Input::Text(prompt.clone()),
                Some(Json::Array(prompts))
                    if prompts.len() == 1 && prompts[0].as_str().is_some() =>
                {
                    Input::Text(prompts[0].as_str().unwrap_or_default().to_string())
                }
                _ => return Err("'prompt' must be a string".to_string()),
            },
        };

        let number = |name: &str| body.get(name).and_then(Json::as_f64);
        let whole = |name: &str| body.get(name).and_then(Json::as_usize);
        let defaults = SamplerConfig::default();
        let clock_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |time| time.as_nanos() as u64);
        let sampler = SamplerConfig {
            temperature: number("temperature").map_or(defaults.temperature, |value| value as f32),
            top_k: whole("top_k").unwrap_or(defaults.top_k),
            top_p: number("top_p").map_or(defaults.top_p, |value| value as f32),
            min_p: number("min_p").map_or(defaults.min_p, |value| value as f32),
            repeat_penalty: number("repeat_penalty")
                .map_or(defaults.repeat_penalty, |value| value as f32),
            repeat_last_n: whole("repeat_last_n").unwrap_or(defaults.repeat_last_n),
            frequency_penalty: number("frequency_penalty")
                .map_or(defaults.frequency_penalty, |value| value as f32),
            presence_penalty: number("presence_penalty")
                .map_or(defaults.presence_penalty, |value| value as f32),
            seed: whole("seed").map_or(clock_seed, |seed| seed as u64),
        };
        let stop = match body.get("stop") {
            Some(Json::String(text)) => vec![text.clone()],
            Some(Json::Array(items)) => items
                .iter()
                .filter_map(Json::as_str)
                .map(str::to_string)
                .collect(),
            _ => Vec::new(),
        };
        let options = Options {
            max_tokens: whole("max_completion_tokens")
                .or_else(|| whole("max_tokens"))
                .unwrap_or(0),
            sampler,
            stop,
            show_special: false,
        };
        let include_usage = body
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Json::as_bool)
            .unwrap_or(false);
        Ok(Job {
            input,
            options,
            stream: body.get("stream").and_then(Json::as_bool).unwrap_or(false),
            include_usage,
        })
    }
}

/// Reads one chat message. Content is a string, or a list of text parts.
fn message(json: &Json) -> Result<Message, String> {
    let role = json
        .get("role")
        .and_then(Json::as_str)
        .ok_or("a message has no 'role'")?;
    let role = Role::from_name(role).ok_or_else(|| format!("unsupported message role '{role}'"))?;
    let content = match json.get("content") {
        Some(Json::String(text)) => text.clone(),
        Some(Json::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Json::as_str) {
                    Some("text") => {
                        text.push_str(part.get("text").and_then(Json::as_str).unwrap_or_default())
                    }
                    other => {
                        return Err(format!(
                            "unsupported content part '{}'",
                            other.unwrap_or("?")
                        ));
                    }
                }
            }
            text
        }
        Some(Json::Null) | None => String::new(),
        _ => return Err("message 'content' must be a string".to_string()),
    };
    Ok(Message::new(role, content))
}

/// The fields every response of one request shares.
struct Envelope<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
    kind: Kind,
}

impl Envelope<'_> {
    fn base(&self, object: &str) -> Vec<(String, Json)> {
        vec![
            ("id".to_string(), Json::string(self.id)),
            ("object".to_string(), Json::string(object)),
            ("created".to_string(), Json::Number(self.created as f64)),
            ("model".to_string(), Json::string(self.model)),
        ]
    }

    /// A whole, unstreamed response.
    fn complete(&self, text: &str, stats: &Stats) -> Json {
        let finish = ("finish_reason", Json::string(finish_reason(stats.finish)));
        let (object, choice) = match self.kind {
            Kind::Chat => {
                let message = Json::object([
                    ("role", Json::string("assistant")),
                    ("content", Json::string(text)),
                ]);
                (
                    "chat.completion",
                    Json::object([("index", Json::Number(0.0)), ("message", message), finish]),
                )
            }
            Kind::Text => (
                "text_completion",
                Json::object([
                    ("index", Json::Number(0.0)),
                    ("text", Json::string(text)),
                    finish,
                ]),
            ),
        };
        let mut members = self.base(object);
        members.push(("choices".to_string(), Json::Array(vec![choice])));
        members.push(("usage".to_string(), usage(stats)));
        members.push(("timings".to_string(), timings(stats)));
        Json::Object(members)
    }

    /// One event of a streamed response.
    fn chunk(&self, role: Option<&str>, text: Option<&str>, finish: Option<&str>) -> Json {
        let finish = ("finish_reason", finish.map_or(Json::Null, Json::string));
        let (object, choice) = match self.kind {
            Kind::Chat => {
                let mut delta = Vec::new();
                if let Some(role) = role {
                    delta.push(("role".to_string(), Json::string(role)));
                }
                if let Some(text) = text {
                    delta.push(("content".to_string(), Json::string(text)));
                }
                let delta = Json::Object(delta);
                (
                    "chat.completion.chunk",
                    Json::object([("index", Json::Number(0.0)), ("delta", delta), finish]),
                )
            }
            Kind::Text => {
                let text = Json::string(text.unwrap_or_default());
                (
                    "text_completion",
                    Json::object([("index", Json::Number(0.0)), ("text", text), finish]),
                )
            }
        };
        let mut members = self.base(object);
        members.push(("choices".to_string(), Json::Array(vec![choice])));
        Json::Object(members)
    }

    /// The closing event that carries token counts, when a client asks.
    fn usage_chunk(&self, stats: &Stats) -> Json {
        let object = if self.kind == Kind::Chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        };
        let mut members = self.base(object);
        members.push(("choices".to_string(), Json::Array(Vec::new())));
        members.push(("usage".to_string(), usage(stats)));
        Json::Object(members)
    }
}

fn usage(stats: &Stats) -> Json {
    Json::object([
        ("prompt_tokens", Json::Number(stats.prompt_tokens as f64)),
        (
            "completion_tokens",
            Json::Number(stats.generated_tokens as f64),
        ),
        (
            "total_tokens",
            Json::Number((stats.prompt_tokens + stats.generated_tokens) as f64),
        ),
    ])
}

/// How long the prompt and the answer took, in the shape llama.cpp's server
/// reports it, so that clients written for it read the same numbers. The
/// context fields are SmaLLM's own.
fn timings(stats: &Stats) -> Json {
    let prompt_n = stats.prompt_tokens - stats.reused_tokens;
    let prompt_ms = stats.prompt_time.as_secs_f64() * 1e3;
    let predicted_ms = stats.generate_time.as_secs_f64() * 1e3;
    let per_token = |ms: f64, n: usize| if n == 0 { 0.0 } else { ms / n as f64 };
    Json::object([
        ("cache_n", Json::Number(stats.reused_tokens as f64)),
        ("prompt_n", Json::Number(prompt_n as f64)),
        ("prompt_ms", Json::Number(prompt_ms)),
        (
            "prompt_per_token_ms",
            Json::Number(per_token(prompt_ms, prompt_n)),
        ),
        ("prompt_per_second", Json::Number(stats.prompt_speed())),
        ("predicted_n", Json::Number(stats.generated_tokens as f64)),
        ("predicted_ms", Json::Number(predicted_ms)),
        (
            "predicted_per_token_ms",
            Json::Number(per_token(predicted_ms, stats.generated_tokens)),
        ),
        ("predicted_per_second", Json::Number(stats.generate_speed())),
        ("context_n", Json::Number(stats.context_tokens as f64)),
        ("context_size", Json::Number(stats.context_size as f64)),
        ("context_bytes", Json::Number(stats.context_bytes as f64)),
    ])
}

/// Prints one line per answered request on the server's console: its speed
/// and how full the context is.
fn log_request(id: &str, stats: &Stats) {
    eprintln!("{id}: {}", crate::commands::stats_line(stats));
}

/// The API's name for why a generation ended.
fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::End | Finish::Stop | Finish::Canceled => "stop",
        Finish::Length | Finish::Context => "length",
    }
}

fn error_json(message: &str) -> Json {
    let kind = Json::string("invalid_request_error");
    Json::object([(
        "error",
        Json::object([("message", Json::string(message)), ("type", kind)]),
    )])
}

fn send_json(out: &mut impl Write, status: u16, body: &Json) -> std::io::Result<()> {
    http::respond(out, status, "application/json", body.to_text().as_bytes())
}

fn send_error(out: &mut impl Write, status: u16, message: &str) -> std::io::Result<()> {
    send_json(out, status, &error_json(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(body: &str, kind: Kind) -> Result<Job, String> {
        Job::from_json(&Json::parse(body).unwrap(), kind)
    }

    #[test]
    fn a_chat_request_is_read() {
        let job = job(
            r#"{"model":"x","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}],
                "max_tokens":32,"temperature":0,"top_p":0.5,"seed":7,"stop":["END","X"],"stream":true,
                "frequency_penalty":0.5,"presence_penalty":0.25,
                "stream_options":{"include_usage":true},"unknown":1}"#,
            Kind::Chat,
        )
        .unwrap();
        let Input::Chat(messages) = &job.input else {
            panic!("expected a chat")
        };
        assert_eq!(messages[0], Message::new(Role::System, "s"));
        assert_eq!(messages[1], Message::new(Role::User, "ab"));
        assert_eq!(job.options.max_tokens, 32);
        assert_eq!(job.options.sampler.temperature, 0.0);
        assert_eq!(job.options.sampler.top_p, 0.5);
        assert_eq!(job.options.sampler.seed, 7);
        assert_eq!(job.options.sampler.frequency_penalty, 0.5);
        assert_eq!(job.options.sampler.presence_penalty, 0.25);
        assert_eq!(job.options.stop, ["END", "X"]);
        assert!(job.stream && job.include_usage);
    }

    #[test]
    fn defaults_apply_to_missing_fields() {
        let job = job(
            r#"{"prompt":"Once","stop":"\n","max_completion_tokens":5}"#,
            Kind::Text,
        )
        .unwrap();
        let Input::Text(prompt) = &job.input else {
            panic!("expected text")
        };
        assert_eq!(prompt, "Once");
        assert_eq!(job.options.stop, ["\n"]);
        assert_eq!(job.options.max_tokens, 5);
        assert_eq!(job.options.sampler.top_k, SamplerConfig::default().top_k);
        assert!(!job.stream && !job.include_usage);
    }

    #[test]
    fn bad_requests_explain_themselves() {
        assert!(job(r#"{}"#, Kind::Chat).unwrap_err().contains("messages"));
        assert!(job(r#"{"messages":[]}"#, Kind::Chat).is_err());
        assert!(
            job(r#"{"messages":[{"content":"x"}]}"#, Kind::Chat)
                .unwrap_err()
                .contains("role")
        );
        assert!(
            job(
                r#"{"messages":[{"role":"tool","content":"x"}]}"#,
                Kind::Chat
            )
            .unwrap_err()
            .contains("tool")
        );
        assert!(
            job(
                r#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#,
                Kind::Chat
            )
            .is_err()
        );
        assert!(job(r#"{"prompt":7}"#, Kind::Text).is_err());
        assert!(job(r#"{"prompt":["a","b"]}"#, Kind::Text).is_err());
    }

    fn stats() -> Stats {
        Stats {
            prompt_tokens: 5,
            reused_tokens: 0,
            prompt_time: Duration::ZERO,
            generated_tokens: 3,
            generate_time: Duration::ZERO,
            finish: Finish::Length,
            context_tokens: 8,
            context_size: 64,
            context_bytes: 1024,
        }
    }

    #[test]
    fn responses_follow_the_api_shape() {
        let chat = Envelope {
            id: "chatcmpl-1",
            created: 10,
            model: "m",
            kind: Kind::Chat,
        };
        let full = chat.complete("hi", &stats());
        assert_eq!(
            full.get("object").unwrap().as_str(),
            Some("chat.completion")
        );
        let choice = &full.get("choices").unwrap().as_array().unwrap()[0];
        assert_eq!(
            choice
                .get("message")
                .unwrap()
                .get("content")
                .unwrap()
                .as_str(),
            Some("hi")
        );
        assert_eq!(
            choice.get("finish_reason").unwrap().as_str(),
            Some("length")
        );
        assert_eq!(
            full.get("usage")
                .unwrap()
                .get("total_tokens")
                .unwrap()
                .as_usize(),
            Some(8)
        );
        let timings = full.get("timings").unwrap();
        assert_eq!(timings.get("prompt_n").unwrap().as_usize(), Some(5));
        assert_eq!(timings.get("predicted_n").unwrap().as_usize(), Some(3));
        assert_eq!(timings.get("context_size").unwrap().as_usize(), Some(64));

        let chunk = chat.chunk(None, Some("x"), None);
        let choice = &chunk.get("choices").unwrap().as_array().unwrap()[0];
        assert_eq!(
            choice
                .get("delta")
                .unwrap()
                .get("content")
                .unwrap()
                .as_str(),
            Some("x")
        );
        assert_eq!(choice.get("finish_reason"), Some(&Json::Null));
        let last = chat.chunk(None, None, Some("stop"));
        assert_eq!(last.to_text().matches("\"delta\":{}").count(), 1);

        let text = Envelope {
            id: "cmpl-1",
            created: 10,
            model: "m",
            kind: Kind::Text,
        };
        let full = text.complete("out", &stats());
        assert_eq!(
            full.get("choices").unwrap().as_array().unwrap()[0]
                .get("text")
                .unwrap()
                .as_str(),
            Some("out")
        );
        assert!(
            text.usage_chunk(&stats())
                .get("choices")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn finish_reasons_use_the_api_names() {
        assert_eq!(finish_reason(Finish::End), "stop");
        assert_eq!(finish_reason(Finish::Stop), "stop");
        assert_eq!(finish_reason(Finish::Length), "length");
        assert_eq!(finish_reason(Finish::Context), "length");
    }
}

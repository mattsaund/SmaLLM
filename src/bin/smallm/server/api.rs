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

use smallm::chat::{Format, Message, Role, Thinking};
use smallm::generate::{self, Finish, Options, Overflow, Stats};
use smallm::model::{Embedder, Pooling, normalize};
use smallm::{Model, SamplerConfig, Session};

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
    /// The session of a model that generates text. `None` for an encoder.
    pub session: Option<Mutex<Session>>,
    /// The embedder of an embedding or reranking model.
    pub embedder: Option<Mutex<Embedder>>,
    /// The model either of them runs.
    model: Arc<Model>,
    /// The chat format, with the server's choice about thinking, which a
    /// request can change.
    pub template: Option<Format>,
    /// The name clients see for the loaded model.
    pub model_id: String,
    /// When set, requests must carry it as a bearer token.
    pub api_key: Option<String>,
    /// What to do when a request fills the context, and the tokens at the
    /// start to keep. A chat keeps its opening, the system prompt and the first
    /// question, when `keep` is unset.
    pub overflow: Overflow,
    pub keep: Option<usize>,
    /// Sampling settings for requests that do not give their own: the
    /// server's options, over the model file's, over the built-in ones.
    pub sampler: SamplerConfig,
    /// A seed for every request that gives none. Without it, each request
    /// draws its own.
    pub seed: Option<u64>,
    connections: AtomicUsize,
    next_id: AtomicU64,
}

impl Server {
    pub fn new(
        session: Session,
        template: Option<Format>,
        model_id: String,
        api_key: Option<String>,
    ) -> Server {
        Server {
            model: Arc::clone(session.model()),
            sampler: SamplerConfig::from_model_file(session.model().gguf()),
            seed: None,
            session: Some(Mutex::new(session)),
            embedder: None,
            template,
            model_id,
            api_key,
            overflow: Overflow::default(),
            keep: None,
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
            ("POST", "/v1/embeddings" | "/embeddings") => self.embeddings(request, out),
            ("POST", "/v1/rerank" | "/rerank" | "/v1/reranking") => self.rerank(request, out),
            (
                _,
                "/health"
                | "/v1/health"
                | "/v1/models"
                | "/v1/chat/completions"
                | "/v1/completions"
                | "/tokenize"
                | "/detokenize"
                | "/v1/embeddings"
                | "/embeddings"
                | "/v1/rerank"
                | "/rerank"
                | "/v1/reranking",
            ) => send_error(out, 405, "method not allowed"),
            _ => send_error(out, 404, "no such endpoint"),
        }
    }

    /// A server for an embedding or reranking model.
    pub fn for_encoder(embedder: Embedder, model_id: String, api_key: Option<String>) -> Server {
        Server {
            model: Arc::clone(embedder.model()),
            session: None,
            embedder: Some(Mutex::new(embedder)),
            template: None,
            model_id,
            api_key,
            overflow: Overflow::default(),
            keep: None,
            sampler: SamplerConfig::default(),
            seed: None,
            connections: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
        }
    }

    /// The embedder, or the error a request for one gets.
    fn embedder(
        &self,
        out: &mut impl Write,
    ) -> std::io::Result<Option<std::sync::MutexGuard<'_, Embedder>>> {
        match &self.embedder {
            Some(embedder) => Ok(Some(
                embedder.lock().unwrap_or_else(|poison| poison.into_inner()),
            )),
            None => {
                send_error(
                    out,
                    400,
                    "this model generates text: serve an embedding or reranking model",
                )?;
                Ok(None)
            }
        }
    }

    /// Handles `/v1/embeddings` in OpenAI's shape: `{"input": text}` or a
    /// list of texts, to one embedding of length one per text.
    fn embeddings(&self, request: &Request, out: &mut impl Write) -> std::io::Result<()> {
        let body = match std::str::from_utf8(&request.body)
            .map_err(|_| "the body is not UTF-8".to_string())
            .and_then(Json::parse)
        {
            Ok(body) => body,
            Err(message) => return send_error(out, 400, &message),
        };
        let texts: Vec<String> = match body.get("input") {
            Some(Json::String(text)) => vec![text.clone()],
            Some(Json::Array(items)) if items.iter().all(|item| item.as_str().is_some()) => items
                .iter()
                .filter_map(Json::as_str)
                .map(str::to_string)
                .collect(),
            _ => return send_error(out, 400, "'input' must be a string or a list of strings"),
        };
        let Some(mut embedder) = self.embedder(out)? else {
            return Ok(());
        };
        if embedder.pooling() == Pooling::Rank {
            return send_error(out, 400, "this is a reranking model: use /v1/rerank");
        }
        let mut data = Vec::with_capacity(texts.len());
        let mut used = 0;
        for (index, text) in texts.iter().enumerate() {
            let tokens = self.model.tokenizer().encode(text, true, true);
            used += tokens.len();
            let mut vector = match embedder.embed(&tokens) {
                Ok(vector) => vector,
                Err(error) => return send_error(out, 400, &error.to_string()),
            };
            normalize(&mut vector);
            let numbers = vector.iter().map(|&v| Json::Number(v as f64)).collect();
            data.push(Json::object([
                ("object", Json::string("embedding")),
                ("index", Json::Number(index as f64)),
                ("embedding", Json::Array(numbers)),
            ]));
        }
        let usage = Json::object([
            ("prompt_tokens", Json::Number(used as f64)),
            ("total_tokens", Json::Number(used as f64)),
        ]);
        send_json(
            out,
            200,
            &Json::object([
                ("object", Json::string("list")),
                ("data", Json::Array(data)),
                ("model", Json::string(&self.model_id)),
                ("usage", usage),
            ]),
        )
    }

    /// Handles `/v1/rerank` in the shape llama.cpp's server and Jina use:
    /// `{"query": text, "documents": [texts], "top_n": n}` to the
    /// documents' indexes and scores, best first.
    fn rerank(&self, request: &Request, out: &mut impl Write) -> std::io::Result<()> {
        let body = match std::str::from_utf8(&request.body)
            .map_err(|_| "the body is not UTF-8".to_string())
            .and_then(Json::parse)
        {
            Ok(body) => body,
            Err(message) => return send_error(out, 400, &message),
        };
        let Some(query) = body.get("query").and_then(Json::as_str) else {
            return send_error(out, 400, "'query' must be a string");
        };
        let documents: Vec<&str> = match body.get("documents").and_then(Json::as_array) {
            Some(items) if items.iter().all(|item| item.as_str().is_some()) => {
                items.iter().filter_map(Json::as_str).collect()
            }
            _ => return send_error(out, 400, "'documents' must be a list of strings"),
        };
        let Some(mut embedder) = self.embedder(out)? else {
            return Ok(());
        };
        if embedder.pooling() != Pooling::Rank {
            return send_error(out, 400, "this model makes embeddings: use /v1/embeddings");
        }
        let mut scored = Vec::with_capacity(documents.len());
        let mut used = 0;
        for (index, document) in documents.iter().enumerate() {
            let tokens = self.model.tokenizer().encode_pair(query, document);
            used += tokens.len();
            match embedder.embed(&tokens) {
                Ok(scores) => scored.push((index, scores.first().copied().unwrap_or(0.0))),
                Err(error) => return send_error(out, 400, &error.to_string()),
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top = body
            .get("top_n")
            .and_then(Json::as_usize)
            .unwrap_or(scored.len());
        let results = scored
            .into_iter()
            .take(top)
            .map(|(index, score)| {
                Json::object([
                    ("index", Json::Number(index as f64)),
                    ("relevance_score", Json::Number(score as f64)),
                ])
            })
            .collect();
        let usage = Json::object([
            ("prompt_tokens", Json::Number(used as f64)),
            ("total_tokens", Json::Number(used as f64)),
        ]);
        send_json(
            out,
            200,
            &Json::object([
                ("model", Json::string(&self.model_id)),
                ("object", Json::string("list")),
                ("usage", usage),
                ("results", Json::Array(results)),
            ]),
        )
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
        let tokenizer = self.model.tokenizer();
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
            .and_then(|body| Job::from_json(&body, kind, &self.sampler, self.seed));
        let mut job = match parsed {
            Ok(job) => job,
            Err(message) => return send_error(out, 400, &message),
        };

        let Some(session) = &self.session else {
            return send_error(
                out,
                400,
                "this model makes embeddings or scores: use /v1/embeddings or /v1/rerank",
            );
        };
        // Requests take turns with the one session.
        let mut session = session.lock().unwrap_or_else(|poison| poison.into_inner());
        let (tokens, system) = match &job.input {
            Input::Text(prompt) => (session.model().tokenizer().encode(prompt, true, true), None),
            Input::Chat(messages) => {
                let Some(mut template) = self.template else {
                    return send_error(
                        out,
                        400,
                        "this model has no chat format. Use /v1/completions.",
                    );
                };
                if let Some(thinking) = job.thinking {
                    template.thinking = thinking;
                }
                let tokenizer = session.model().tokenizer();
                let tokens = generate::chat_tokens_fitting(
                    tokenizer,
                    template,
                    messages,
                    session.n_ctx(),
                    ANSWER_RESERVE,
                    self.overflow,
                )
                .0;
                let system = generate::opening_tokens(tokenizer, template, messages, &tokens);
                (tokens, Some(system))
            }
        };
        job.options.overflow = self.overflow;
        job.options.keep = self.keep.or(system);
        if tokens.is_empty() {
            return send_error(out, 400, "the prompt is empty");
        }
        if tokens.len() >= session.n_ctx() && self.overflow == Overflow::Stop {
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
            let result = generate::run_tokens_with(&mut session, &tokens, &job.options, |piece| {
                let chunk = if piece.reasoning {
                    envelope.reasoning_chunk(piece.text)
                } else {
                    envelope.chunk(None, Some(piece.text), None)
                };
                http::send_event(out, &chunk.to_text()).is_ok()
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
            let (mut text, mut reasoning) = (String::new(), String::new());
            let result = generate::run_tokens_with(&mut session, &tokens, &job.options, |piece| {
                let into = if piece.reasoning {
                    &mut reasoning
                } else {
                    &mut text
                };
                into.push_str(piece.text);
                true
            });
            match result {
                Ok(stats) => {
                    log_request(&id, &stats);
                    send_json(out, 200, &envelope.complete(&text, &reasoning, &stats))
                }
                Err(error) => send_error(out, 500, &error.to_string()),
            }
        }
    }
}

/// Reads a request's `logit_bias`: OpenAI's object of token ids and
/// biases, or llama.cpp's list of `[token, bias]` pairs, where `false`
/// bans the token.
fn logit_bias(value: &Json) -> Result<Vec<(u32, f32)>, String> {
    let bias = |value: &Json| match value {
        Json::Bool(false) => Some(f32::NEG_INFINITY),
        other => other.as_f64().map(|value| value as f32),
    };
    let bad = || "'logit_bias' must map token ids to numbers".to_string();
    match value {
        Json::Object(members) => members
            .iter()
            .map(|(token, value)| {
                Ok((
                    token.parse().map_err(|_| bad())?,
                    bias(value).ok_or_else(bad)?,
                ))
            })
            .collect(),
        Json::Array(pairs) => pairs
            .iter()
            .map(|pair| match pair.as_array() {
                Some([token, value]) => Ok((
                    token
                        .as_usize()
                        .and_then(|t| u32::try_from(t).ok())
                        .ok_or_else(bad)?,
                    bias(value).ok_or_else(bad)?,
                )),
                _ => Err(bad()),
            })
            .collect(),
        _ => Err(bad()),
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
    /// The request's choice about thinking, over the server's.
    thinking: Option<Thinking>,
    stream: bool,
    /// Send token counts as a last event of a stream.
    include_usage: bool,
}

impl Job {
    /// Reads a request body. Unknown fields are ignored, as API clients
    /// send many that only matter to other servers. Sampling settings the
    /// request does not give come from `defaults`, and the seed from
    /// `seed` or the clock.
    fn from_json(
        body: &Json,
        kind: Kind,
        defaults: &SamplerConfig,
        seed: Option<u64>,
    ) -> Result<Job, String> {
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
        let float = |name: &str, default: f32| number(name).map_or(default, |value| value as f32);
        let clock_seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |time| time.as_nanos() as u64);
        let mirostat = match whole("mirostat") {
            Some(version) if version > 2 => return Err("'mirostat' is 0, 1 or 2".to_string()),
            Some(version) => version as u8,
            None => defaults.mirostat,
        };
        let logit_bias = match body.get("logit_bias") {
            None | Some(Json::Null) => defaults.logit_bias.clone(),
            Some(bias) => logit_bias(bias)?,
        };
        let sampler = SamplerConfig {
            temperature: float("temperature", defaults.temperature),
            top_k: whole("top_k").unwrap_or(defaults.top_k),
            top_p: float("top_p", defaults.top_p),
            min_p: float("min_p", defaults.min_p),
            repeat_penalty: float("repeat_penalty", defaults.repeat_penalty),
            repeat_last_n: whole("repeat_last_n").unwrap_or(defaults.repeat_last_n),
            frequency_penalty: float("frequency_penalty", defaults.frequency_penalty),
            presence_penalty: float("presence_penalty", defaults.presence_penalty),
            typical_p: float("typical_p", defaults.typical_p),
            top_n_sigma: float("top_n_sigma", defaults.top_n_sigma),
            xtc_probability: float("xtc_probability", defaults.xtc_probability),
            xtc_threshold: float("xtc_threshold", defaults.xtc_threshold),
            dynatemp_range: float("dynatemp_range", defaults.dynatemp_range),
            dynatemp_exponent: float("dynatemp_exponent", defaults.dynatemp_exponent),
            mirostat,
            mirostat_tau: float("mirostat_tau", defaults.mirostat_tau),
            mirostat_eta: float("mirostat_eta", defaults.mirostat_eta),
            logit_bias,
            seed: whole("seed").map_or(seed.unwrap_or(clock_seed), |seed| seed as u64),
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
            // A chat answer keeps the model's reasoning apart, as
            // `reasoning_content`. A plain completion is raw text.
            reasoning: kind == Kind::Chat,
            overflow: Overflow::default(),
            keep: None,
        };
        let include_usage = body
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Json::as_bool)
            .unwrap_or(false);
        // Thinking is switched as llama.cpp's server takes it, through the
        // template's `enable_thinking`, or set with OpenAI's
        // `reasoning_effort`, at the top level or among the template's
        // arguments.
        let arguments = body.get("chat_template_kwargs");
        let argument = |name: &str| arguments.and_then(|arguments| arguments.get(name));
        let effort = body
            .get("reasoning_effort")
            .or_else(|| argument("reasoning_effort"))
            .and_then(Json::as_str);
        let thinking =
            thinking_choice(argument("enable_thinking").and_then(Json::as_bool), effort)?;
        Ok(Job {
            input,
            options,
            thinking,
            stream: body.get("stream").and_then(Json::as_bool).unwrap_or(false),
            include_usage,
        })
    }
}

/// A request's choice about thinking. `enable_thinking: false` turns it off
/// and `true` on. `reasoning_effort` sets a level: `none` is off, and
/// `minimal` is gpt-oss's lowest, `low`.
fn thinking_choice(enable: Option<bool>, effort: Option<&str>) -> Result<Option<Thinking>, String> {
    if enable == Some(false) {
        return Ok(Some(Thinking::Off));
    }
    let level = match effort {
        None => None,
        Some("none") => Some(Thinking::Off),
        Some("minimal" | "low") => Some(Thinking::Low),
        Some("medium") => Some(Thinking::Medium),
        Some("high") => Some(Thinking::High),
        Some(other) => {
            return Err(format!(
                "'reasoning_effort' is none, minimal, low, medium or high, not '{other}'"
            ));
        }
    };
    Ok(level.or(enable.map(|_| Thinking::On)))
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
    fn complete(&self, text: &str, reasoning: &str, stats: &Stats) -> Json {
        let finish = ("finish_reason", Json::string(finish_reason(stats.finish)));
        let (object, choice) = match self.kind {
            Kind::Chat => {
                let mut message = vec![
                    ("role".to_string(), Json::string("assistant")),
                    ("content".to_string(), Json::string(text)),
                ];
                // The field llama.cpp and DeepSeek's API use.
                if !reasoning.is_empty() {
                    message.push(("reasoning_content".to_string(), Json::string(reasoning)));
                }
                let message = Json::Object(message);
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

    /// One event of a streamed chat answer that carries reasoning.
    fn reasoning_chunk(&self, text: &str) -> Json {
        let delta = Json::object([("reasoning_content", Json::string(text))]);
        let choice = Json::object([
            ("index", Json::Number(0.0)),
            ("delta", delta),
            ("finish_reason", Json::Null),
        ]);
        let mut members = self.base("chat.completion.chunk");
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
        Job::from_json(
            &Json::parse(body).unwrap(),
            kind,
            &SamplerConfig::default(),
            None,
        )
    }

    #[test]
    fn thinking_is_switched_per_request() {
        let thinking = |body: &str| job(body, Kind::Chat).map(|job| job.thinking);
        let chat = r#""messages": [{"role": "user", "content": "Hi"}]"#;
        assert_eq!(thinking(&format!("{{{chat}}}")), Ok(None));
        let off = format!(r#"{{{chat}, "chat_template_kwargs": {{"enable_thinking": false}}}}"#);
        assert_eq!(thinking(&off), Ok(Some(Thinking::Off)));
        let on = format!(r#"{{{chat}, "chat_template_kwargs": {{"enable_thinking": true}}}}"#);
        assert_eq!(thinking(&on), Ok(Some(Thinking::On)));
        let high = format!(r#"{{{chat}, "reasoning_effort": "high"}}"#);
        assert_eq!(thinking(&high), Ok(Some(Thinking::High)));
        let inside =
            format!(r#"{{{chat}, "chat_template_kwargs": {{"reasoning_effort": "low"}}}}"#);
        assert_eq!(thinking(&inside), Ok(Some(Thinking::Low)));
        assert_eq!(
            thinking_choice(Some(true), Some("none")),
            Ok(Some(Thinking::Off))
        );
        assert_eq!(
            thinking_choice(Some(false), Some("high")),
            Ok(Some(Thinking::Off))
        );
        assert!(thinking(&format!(r#"{{{chat}, "reasoning_effort": "max"}}"#)).is_err());
    }

    #[test]
    fn new_sampling_fields_and_server_defaults_are_read() {
        let body = Json::parse(
            r#"{"prompt": "x", "typical_p": 0.9, "top_n_sigma": 1.5, "mirostat": 2,
                "logit_bias": {"5": -100, "7": 2.5}}"#,
        )
        .unwrap();
        let defaults = SamplerConfig {
            temperature: 1.0,
            top_k: 64,
            ..SamplerConfig::default()
        };
        let job = Job::from_json(&body, Kind::Text, &defaults, Some(42)).unwrap();
        let sampler = &job.options.sampler;
        assert_eq!((sampler.temperature, sampler.top_k), (1.0, 64));
        assert_eq!(
            (sampler.typical_p, sampler.top_n_sigma, sampler.mirostat),
            (0.9, 1.5, 2)
        );
        assert_eq!(sampler.logit_bias, [(5, -100.0), (7, 2.5)]);
        assert_eq!(sampler.seed, 42);
        // llama.cpp's form, where false bans a token.
        let pairs = Json::parse(r#"[[3, false], [4, 1]]"#).unwrap();
        assert_eq!(
            logit_bias(&pairs).unwrap(),
            [(3, f32::NEG_INFINITY), (4, 1.0)]
        );
        assert!(logit_bias(&Json::parse(r#"{"x": 1}"#).unwrap()).is_err());
        let bad = Json::parse(r#"{"prompt": "x", "mirostat": 3}"#).unwrap();
        assert!(Job::from_json(&bad, Kind::Text, &defaults, None).is_err());
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
            dropped_tokens: 0,
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
        let full = chat.complete("hi", "", &stats());
        assert_eq!(
            full.get("object").unwrap().as_str(),
            Some("chat.completion")
        );
        let choice = &full.get("choices").unwrap().as_array().unwrap()[0];
        // Reasoning appears only when there was some.
        assert!(
            choice
                .get("message")
                .unwrap()
                .get("reasoning_content")
                .is_none()
        );
        let thought = chat.complete("hi", "plan", &stats());
        let message = thought.get("choices").unwrap().as_array().unwrap()[0]
            .get("message")
            .unwrap();
        assert_eq!(
            message.get("reasoning_content").unwrap().as_str(),
            Some("plan")
        );
        let delta = chat.reasoning_chunk("step");
        let delta = delta.get("choices").unwrap().as_array().unwrap()[0]
            .get("delta")
            .unwrap();
        assert_eq!(
            delta.get("reasoning_content").unwrap().as_str(),
            Some("step")
        );
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
        let full = text.complete("out", "", &stats());
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

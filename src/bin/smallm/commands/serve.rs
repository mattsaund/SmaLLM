//! `smallm serve`: an OpenAI-compatible API for one model.

use std::net::TcpListener;
use std::sync::Arc;

use smallm::Session;
use smallm::model::{Embedder, Pooling};

use super::{
    CONTEXT_SPECS, SAMPLER_SPECS, SESSION_SPECS, TEMPLATE_SPECS, chat_template, describe,
    load_model, overflow_options, sampler_config, session_config,
};
use crate::args::{self, Args, Spec, value};
use crate::server::api::Server;

const SPECS: [Spec; 3] = [
    value(
        "host",
        None,
        "Address to listen on (default: 127.0.0.1, this machine only)",
    ),
    value("port", Some('p'), "Port to listen on (default: 8080)"),
    value("api-key", None, "Require this key as a bearer token"),
];

fn specs() -> Vec<Spec> {
    SPECS
        .into_iter()
        .chain(SESSION_SPECS)
        .chain(CONTEXT_SPECS)
        .chain(TEMPLATE_SPECS)
        .chain(SAMPLER_SPECS)
        .collect()
}

pub fn help() -> String {
    format!(
        "Usage: smallm serve <model> [options]\n\n\
         Serves the model over HTTP, in the API that OpenAI clients speak:\n\
         /v1/chat/completions, /v1/completions and /v1/models.\n\
         Also /tokenize and /detokenize. An embedding model is served at\n\
         /v1/embeddings, and a reranking model at /v1/rerank. The sampling\n\
         options set the defaults for requests that do not give their own.\n\n\
         Options:\n{}",
        args::describe(&specs())
    )
}

pub fn main(raw: impl Iterator<Item = String>) -> Result<(), String> {
    let args = Args::parse(raw, &specs())?;
    let name = args
        .positional(0)
        .ok_or("which model? Usage: smallm serve <model>")?;
    let host = args.get("host").unwrap_or("127.0.0.1");
    let port: u16 = args.number("port")?.unwrap_or(8080);
    let api_key = args.get("api-key").map(str::to_string);

    let loaded = load_model(name, &args)?;
    let model = &loaded.model;

    // Clients name the model in requests. The file name is what people
    // know it by.
    let model_id = crate::models::resolve(name)
        .ok()
        .and_then(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| name.to_string());

    let listener = TcpListener::bind((host, port))
        .map_err(|error| format!("cannot listen on {host}:{port}: {error}"))?;
    let address = listener
        .local_addr()
        .map_or_else(|_| format!("{host}:{port}"), |address| address.to_string());
    eprintln!("{}", describe(model));

    // An encoder model serves embeddings or scores. Others generate.
    let server = if model.is_encoder() {
        let threads = args
            .number("threads-batch")?
            .or(args.number("threads")?)
            .unwrap_or(0);
        let embedder = Embedder::new(model.clone(), threads).map_err(|error| error.to_string())?;
        let endpoint = match embedder.pooling() {
            Pooling::Rank => "/v1/rerank",
            _ => "/v1/embeddings",
        };
        eprintln!("an encoder model, served at {endpoint}");
        Server::for_encoder(embedder, model_id, api_key.clone())
    } else {
        let template = chat_template(&args, model)?;
        let session = Session::new(model.clone(), session_config(&args)?)
            .map_err(|error| error.to_string())?;
        eprintln!(
            "{}, {} kernels, context {}, chat format {}",
            super::thread_counts(&session),
            session.kernels(),
            session.n_ctx(),
            template.map_or("none", |format| format.template.name())
        );
        let mut server = Server::new(session, template, model_id, api_key.clone());
        (server.overflow, server.keep) = overflow_options(&args)?;
        server.sampler = sampler_config(&args, model)?;
        server.seed = args.number("seed")?;
        server
    };
    let reachable_by_others = !matches!(host, "127.0.0.1" | "localhost" | "::1");
    if reachable_by_others && api_key.is_none() {
        eprintln!("smallm: warning: other machines can reach this server and no --api-key is set");
    }
    eprintln!("listening on http://{address}");

    Arc::new(server).run(listener);
    Ok(())
}

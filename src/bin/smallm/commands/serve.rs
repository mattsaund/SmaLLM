//! `smallm serve`: an OpenAI-compatible API for one model.

use std::net::TcpListener;
use std::sync::Arc;

use smallm::Session;

use super::{
    CONTEXT_SPECS, SESSION_SPECS, TEMPLATE_SPECS, chat_template, describe, load_model,
    overflow_options, session_config,
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
        .collect()
}

pub fn help() -> String {
    format!(
        "Usage: smallm serve <model> [options]\n\n\
         Serves the model over HTTP, in the API that OpenAI clients speak:\n\
         /v1/chat/completions, /v1/completions and /v1/models.\n\
         Also /tokenize and /detokenize.\n\n\
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

    let loaded = load_model(name)?;
    let model = &loaded.model;
    let template = chat_template(&args, model)?;
    let session =
        Session::new(model.clone(), session_config(&args)?).map_err(|error| error.to_string())?;

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
    eprintln!(
        "{} threads, {} kernels, context {}, chat format {}",
        session.threads(),
        session.kernels(),
        session.n_ctx(),
        template.map_or("none", |template| template.name())
    );
    let reachable_by_others = !matches!(host, "127.0.0.1" | "localhost" | "::1");
    if reachable_by_others && api_key.is_none() {
        eprintln!("smallm: warning: other machines can reach this server and no --api-key is set");
    }
    eprintln!("listening on http://{address}");

    let mut server = Server::new(session, template, model_id, api_key);
    (server.overflow, server.keep) = overflow_options(&args)?;
    Arc::new(server).run(listener);
    Ok(())
}

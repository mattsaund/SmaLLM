//! The HTTP server behind `smallm serve`.
//!
//! | File      | Role                                              |
//! |-----------|---------------------------------------------------|
//! | `http.rs` | Reads requests and writes responses.              |
//! | `api.rs`  | The OpenAI-compatible endpoints.                  |

pub mod api;
pub mod http;

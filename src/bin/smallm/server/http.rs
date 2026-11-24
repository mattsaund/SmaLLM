//! Just enough HTTP/1.1 for a local API: one request per connection.
//!
//! Every response says `Connection: close`, which keeps the server free of
//! keep-alive and chunked encoding. A streamed response simply ends when the
//! connection does.

use std::io::{self, BufRead, Read, Write};

/// The largest request head, in bytes.
const MAX_HEAD: usize = 64 * 1024;

/// The largest request body, in bytes.
pub const MAX_BODY: usize = 32 * 1024 * 1024;

/// A parsed request.
#[derive(Debug)]
pub struct Request {
    pub method: String,
    /// The path without its query string.
    pub path: String,
    headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Why a request could not be read. Carries the status to answer with.
#[derive(Debug, PartialEq, Eq)]
pub struct Rejection {
    pub status: u16,
    pub message: &'static str,
}

fn reject(status: u16, message: &'static str) -> Rejection {
    Rejection { status, message }
}

impl Request {
    /// Reads one request. `Ok(None)` means the peer closed the connection
    /// without sending anything.
    pub fn read(stream: &mut impl BufRead) -> Result<Option<Request>, Rejection> {
        let mut head = Vec::new();
        loop {
            let mut line = Vec::new();
            let read = Read::take(&mut *stream, (MAX_HEAD - head.len()) as u64 + 1)
                .read_until(b'\n', &mut line)
                .map_err(|_| reject(400, "could not read the request"))?;
            if read == 0 {
                return if head.is_empty() {
                    Ok(None)
                } else {
                    Err(reject(400, "incomplete request"))
                };
            }
            head.extend_from_slice(&line);
            if head.len() > MAX_HEAD {
                return Err(reject(431, "request headers are too large"));
            }
            if line == b"\r\n" || line == b"\n" {
                break;
            }
        }
        let head =
            std::str::from_utf8(&head).map_err(|_| reject(400, "request headers are not text"))?;
        let mut lines = head.lines();
        let mut request_line = lines.next().unwrap_or_default().split(' ');
        let (Some(method), Some(target), Some(version)) = (
            request_line.next(),
            request_line.next(),
            request_line.next(),
        ) else {
            return Err(reject(400, "malformed request line"));
        };
        if !version.starts_with("HTTP/1.") {
            return Err(reject(505, "only HTTP/1.x is supported"));
        }
        let headers: Vec<(String, String)> = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
            .collect();
        let mut request = Request {
            method: method.to_string(),
            path: target
                .split(['?', '#'])
                .next()
                .unwrap_or_default()
                .to_string(),
            headers,
            body: Vec::new(),
        };

        if request.header("transfer-encoding").is_some() {
            return Err(reject(411, "send the body with a Content-Length"));
        }
        if let Some(length) = request.header("content-length") {
            let length: usize = length
                .parse()
                .map_err(|_| reject(400, "invalid Content-Length"))?;
            if length > MAX_BODY {
                return Err(reject(413, "request body is too large"));
            }
            request.body = vec![0; length];
            stream
                .read_exact(&mut request.body)
                .map_err(|_| reject(400, "incomplete request body"))?;
        }
        Ok(Some(request))
    }

    /// The value of a header, by lowercase name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    }
}

/// The standard phrase for the status codes this server sends.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Internal Server Error",
    }
}

/// Sends a complete response.
pub fn respond(
    stream: &mut impl Write,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

/// Starts a server-sent event stream. Events follow with [`send_event`],
/// and the stream ends when the connection is closed.
pub fn start_events(stream: &mut impl Write) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()
}

/// Sends one event of a stream started with [`start_events`].
pub fn send_event(stream: &mut impl Write, data: &str) -> io::Result<()> {
    write!(stream, "data: {data}\n\n")?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Option<Request>, Rejection> {
        Request::read(&mut text.as_bytes())
    }

    #[test]
    fn a_request_with_a_body_is_parsed() {
        let request =
            read("POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 4\r\nX-Key:  v \r\n\r\nbodyEXTRA")
                .unwrap()
                .unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.body, b"body");
        assert_eq!(request.header("x-key"), Some("v"));
        assert_eq!(request.header("missing"), None);
    }

    #[test]
    fn a_request_without_a_body_is_parsed() {
        let request = read("GET /health HTTP/1.1\r\n\r\n").unwrap().unwrap();
        assert_eq!(
            (request.method.as_str(), request.path.as_str()),
            ("GET", "/health")
        );
        assert!(request.body.is_empty());
        // Bare line feeds are tolerated.
        assert!(read("GET / HTTP/1.0\n\n").unwrap().is_some());
    }

    #[test]
    fn a_silent_peer_is_not_an_error() {
        assert!(read("").unwrap().is_none());
    }

    #[test]
    fn bad_requests_are_rejected_with_a_status() {
        let status = |text: &str| read(text).unwrap_err().status;
        assert_eq!(status("GET /\r\n\r\n"), 400);
        assert_eq!(status("GET / HTTP/2\r\n\r\n"), 505);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a"), 400);
        assert_eq!(
            status("POST / HTTP/1.1\r\nContent-Length: ten\r\n\r\n"),
            400
        );
        assert_eq!(
            status("POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort"),
            400
        );
        assert_eq!(
            status("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"),
            411
        );
        assert_eq!(
            status(&format!(
                "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
                MAX_BODY + 1
            )),
            413
        );
        let huge = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert_eq!(status(&huge), 431);
    }

    #[test]
    fn responses_are_well_formed() {
        let mut out = Vec::new();
        respond(&mut out, 404, "application/json", b"{}").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.ends_with("\r\n\r\n{}"));

        let mut out = Vec::new();
        start_events(&mut out).unwrap();
        send_event(&mut out, "{\"a\":1}").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("text/event-stream"));
        assert!(text.ends_with("data: {\"a\":1}\n\n"));
    }
}

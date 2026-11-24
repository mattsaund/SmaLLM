//! A small JSON reader and writer.
//!
//! The API server and the model downloader exchange a few flat objects, so
//! this covers the JSON data model and nothing more: no streaming, no
//! borrowed strings, no number types beyond f64.

use std::fmt::Write;

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    /// Members in the order they were written.
    Object(Vec<(String, Json)>),
}

/// Values nested deeper than this are rejected, which bounds the parser's
/// stack on hostile input.
const MAX_DEPTH: usize = 64;

impl Json {
    /// Parses a complete JSON document.
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = parser.value(0)?;
        parser.skip_whitespace();
        if parser.pos != parser.bytes.len() {
            return Err(parser.error("unexpected data after the document"));
        }
        Ok(value)
    }

    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(number) => Some(*number),
            _ => None,
        }
    }

    /// The value as a whole non-negative number.
    pub fn as_usize(&self) -> Option<usize> {
        let number = self.as_f64()?;
        (number >= 0.0 && number.fract() == 0.0 && number < 1e15).then_some(number as usize)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Builds an object from `(key, value)` pairs.
    pub fn object<const N: usize>(members: [(&str, Json); N]) -> Json {
        Json::Object(
            members
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    /// Builds a string value.
    pub fn string(text: impl Into<String>) -> Json {
        Json::String(text.into())
    }

    /// Serializes the value without extra whitespace.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Json::Number(number) if number.is_finite() => {
                let _ = write!(out, "{number}");
            }
            // JSON has no spelling for infinity or NaN.
            Json::Number(_) => out.push_str("null"),
            Json::String(text) => write_string(text, out),
            Json::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Object(members) => {
                out.push('{');
                for (index, (key, value)) in members.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Writes `text` as a JSON string literal.
fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> String {
        format!("invalid JSON at byte {}: {message}", self.pos)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Consumes `literal` if it is next.
    fn eat(&mut self, literal: &str) -> bool {
        let matches = self.bytes[self.pos..].starts_with(literal.as_bytes());
        if matches {
            self.pos += literal.len();
        }
        matches
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(self.error("nested too deeply"));
        }
        self.skip_whitespace();
        match self.bytes.get(self.pos) {
            Some(b'{') => {
                self.pos += 1;
                let mut members = Vec::new();
                self.skip_whitespace();
                if self.eat("}") {
                    return Ok(Json::Object(members));
                }
                loop {
                    self.skip_whitespace();
                    let key = self.string()?;
                    self.skip_whitespace();
                    if !self.eat(":") {
                        return Err(self.error("expected ':'"));
                    }
                    members.push((key, self.value(depth + 1)?));
                    self.skip_whitespace();
                    if self.eat("}") {
                        return Ok(Json::Object(members));
                    }
                    if !self.eat(",") {
                        return Err(self.error("expected ',' or '}'"));
                    }
                }
            }
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_whitespace();
                if self.eat("]") {
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.skip_whitespace();
                    if self.eat("]") {
                        return Ok(Json::Array(items));
                    }
                    if !self.eat(",") {
                        return Err(self.error("expected ',' or ']'"));
                    }
                }
            }
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("expected a value")),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while matches!(
            self.bytes.get(self.pos),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        // The slice holds ASCII only, so it is valid UTF-8.
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or_default();
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| self.error("malformed number"))
    }

    fn string(&mut self) -> Result<String, String> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return Err(self.error("expected a string"));
        }
        self.pos += 1;
        let mut out = String::new();
        loop {
            // Copy the run of ordinary bytes up to the next quote or escape.
            let start = self.pos;
            while !matches!(self.bytes.get(self.pos), None | Some(b'"' | b'\\')) {
                self.pos += 1;
            }
            let run = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| self.error("invalid UTF-8"))?;
            out.push_str(run);
            match self.bytes.get(self.pos) {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.escape(&mut out)?;
                }
                _ => return Err(self.error("unterminated string")),
            }
        }
    }

    /// Reads the part of an escape sequence after its backslash.
    fn escape(&mut self, out: &mut String) -> Result<(), String> {
        let code = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| self.error("unterminated escape"))?;
        self.pos += 1;
        match code {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{8}'),
            b'f' => out.push('\u{c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let first = self.hex4()?;
                let code_point = if (0xd800..0xdc00).contains(&first) {
                    // A high surrogate must be followed by a low one.
                    if !self.eat("\\u") {
                        return Err(self.error("unpaired surrogate"));
                    }
                    let second = self.hex4()?;
                    if !(0xdc00..0xe000).contains(&second) {
                        return Err(self.error("unpaired surrogate"));
                    }
                    0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                } else {
                    first
                };
                out.push(
                    char::from_u32(code_point).ok_or_else(|| self.error("invalid code point"))?,
                );
            }
            _ => return Err(self.error("unknown escape")),
        }
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.error("short \\u escape"))?;
        let text = std::str::from_utf8(digits).map_err(|_| self.error("invalid \\u escape"))?;
        let value = u32::from_str_radix(text, 16).map_err(|_| self.error("invalid \\u escape"))?;
        self.pos += 4;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_kind_of_value() {
        let json =
            Json::parse(r#" {"a": [1, -2.5e1, true, false, null], "b": {"c": "d"}, "e": ""} "#)
                .unwrap();
        let a = json.get("a").unwrap().as_array().unwrap();
        assert_eq!(a[0].as_usize(), Some(1));
        assert_eq!(a[1].as_f64(), Some(-25.0));
        assert_eq!(a[2].as_bool(), Some(true));
        assert_eq!(a[4], Json::Null);
        assert_eq!(json.get("b").unwrap().get("c").unwrap().as_str(), Some("d"));
        assert_eq!(json.get("e").unwrap().as_str(), Some(""));
        assert!(json.get("missing").is_none());
        assert_eq!(Json::parse("[]").unwrap(), Json::Array(vec![]));
        assert_eq!(Json::parse("{}").unwrap(), Json::Object(vec![]));
    }

    #[test]
    fn strings_decode_escapes() {
        let json = Json::parse(r#""line\nbreak \"quoted\" \u00e9 \ud83d\ude00 \/ \\""#).unwrap();
        assert_eq!(
            json.as_str(),
            Some("line\nbreak \"quoted\" \u{e9} \u{1f600} / \\")
        );
        assert_eq!(
            Json::parse("\"caf\u{e9}\"").unwrap().as_str(),
            Some("caf\u{e9}")
        );
    }

    #[test]
    fn text_round_trips() {
        let value = Json::object([
            ("text", Json::string("tab\there \"q\" \\ \u{1} \u{1f600}")),
            (
                "list",
                Json::Array(vec![
                    Json::Number(1.0),
                    Json::Number(0.5),
                    Json::Null,
                    Json::Bool(false),
                ]),
            ),
            ("nan", Json::Number(f64::NAN)),
        ]);
        let text = value.to_text();
        assert_eq!(
            text,
            "{\"text\":\"tab\\there \\\"q\\\" \\\\ \\u0001 \u{1f600}\",\"list\":[1,0.5,null,false],\"nan\":null}"
        );
        let back = Json::parse(&text).unwrap();
        assert_eq!(back.get("text"), value.get("text"));
        assert_eq!(back.get("list"), value.get("list"));
    }

    #[test]
    fn malformed_documents_are_errors() {
        for text in [
            "",
            "{",
            "[1,",
            "[1 2]",
            "{\"a\" 1}",
            "{\"a\":}",
            "\"open",
            "tru",
            "01x",
            "1 2",
            "\"\\q\"",
            "\"\\ud83d\"",
            "\"\\u12\"",
            "nul",
            "[1,]",
            "{,}",
            "-",
        ] {
            assert!(Json::parse(text).is_err(), "{text:?} should not parse");
        }
    }

    #[test]
    fn deep_nesting_is_rejected_not_overflowed() {
        let deep = "[".repeat(100_000);
        assert!(Json::parse(&deep).is_err());
        let ok = format!("{}1{}", "[".repeat(40), "]".repeat(40));
        assert!(Json::parse(&ok).is_ok());
    }

    #[test]
    fn whole_numbers_convert_to_usize() {
        assert_eq!(Json::Number(42.0).as_usize(), Some(42));
        assert_eq!(Json::Number(-1.0).as_usize(), None);
        assert_eq!(Json::Number(1.5).as_usize(), None);
        assert_eq!(Json::string("42").as_usize(), None);
    }
}

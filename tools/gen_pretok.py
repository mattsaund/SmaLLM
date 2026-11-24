#!/usr/bin/env python3
"""Generates src/tokenizer/pretok_table.rs from llama.cpp's vocabulary code.

A BPE model names its pre-tokenizer in `tokenizer.ggml.pre`, and the name
stands for a list of regular expressions that split text into words, plus a
few switches. llama.cpp keeps that table in src/llama-vocab.cpp, written as
C++ code. This reads the code and writes the same table as Rust data, so
the engine knows every family llama.cpp knows.

Usage:
    python3 tools/gen_pretok.py path/to/llama.cpp > src/tokenizer/pretok_table.rs

Three patterns are names for code inside llama.cpp rather than regular
expressions. They are replaced with expressions that split the same way:
see SUBSTITUTES below.
"""

import re
import sys
from pathlib import Path

# llama.cpp patterns that stand for hand-written code, and expressions that
# do what that code does.
SUBSTITUTES = {
    # Kimi K2: Han characters in runs, and letter words without them. The
    # original uses class intersection, which the engine supports.
    "\\p{Han}+": (
        "[\\p{Han}]+"
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
        "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    ),
    # Digits in groups of three from the right: 1234567 is 1, 234, 567.
    "\\p{AFMoE_digits}": "\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))",
    "\\d{1,3}(?=(?:\\d{3})*\\b)": "\\p{N}{1,3}(?=(?:\\p{N}{3})*(?!\\p{N}))",
}

# The switches a name can turn on, as llama.cpp's code spells them.
SWITCHES = {
    "ignore_merges = true;": "ignore_merges",
    "add_bos = true;": "add_bos",
    "escape_whitespaces = true;": "escape_spaces",
    "special_bos_id = LLAMA_TOKEN_NULL;": "no_bos",
}


def cpp_strings(text):
    """The C++ string literals in text, decoded, ignoring // comments."""
    strings = []
    for line in text.splitlines():
        code = line.split("//")[0] if not line.strip().startswith('"') else line
        # A comment after the last literal of the line.
        for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', code):
            strings.append(decode_cpp(match.group(1)))
    return strings


def decode_cpp(literal):
    """Turns the inside of a C++ string literal into the string it means."""
    out = []
    i = 0
    while i < len(literal):
        c = literal[i]
        if c == "\\" and i + 1 < len(literal):
            nxt = literal[i + 1]
            out.append({"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'"}.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def ascii_pattern(pattern):
    """The pattern with every character outside printable ASCII escaped."""
    out = []
    for c in pattern:
        if c == "\r":
            out.append("\\r")
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif 32 <= ord(c) < 127:
            out.append(c)
        elif ord(c) <= 0xFFFF:
            out.append("\\u%04X" % ord(c))
        else:
            out.append("\\u{%X}" % ord(c))
    return "".join(out)


def patterns_by_type(source):
    """Maps each pre-tokenizer type to its expressions and raw-byte switch."""
    start = source.index("llm_tokenizer_bpe(const llama_vocab & vocab) {")
    end = source.index("std::vector<std::string> regex_exprs;", start)
    body = source[start:end]
    found = {}
    pending = []
    pos = 0
    case = re.compile(r"case LLAMA_VOCAB_PRE_TYPE_(\w+):|default:")
    while True:
        match = case.search(body, pos)
        if not match:
            break
        pending.append(match.group(1) or "DEFAULT")
        pos = match.end()
        nxt = case.search(body, pos)
        block_end = nxt.start() if nxt else len(body)
        block = body[pos:block_end]
        if "regex_exprs = {" not in block:
            continue
        inside = block[block.index("regex_exprs = {") + len("regex_exprs = {"):]
        inside = inside[: inside.index("};")]
        patterns = [SUBSTITUTES.get(p, p) for p in cpp_strings(inside)]
        raw = "byte_encode = false;" in block
        for name in pending:
            found[name] = (patterns, raw)
        pending = []
    return found


def families(source, by_type):
    """The name groups of the loader, with their type and switches."""
    start = source.index('} else if (tokenizer_pre == "flint")')
    start = source.rindex("if (tokenizer_pre.empty())", 0, start)
    end = source.index("unknown pre-tokenizer type", start)
    body = source[start:end]
    groups = []
    for block in re.split(r"\} else if \(", body)[1:]:
        names = re.findall(r'tokenizer_pre == "([^"]+)"', block.split("{")[0])
        kind = re.search(r"pre_type = LLAMA_VOCAB_PRE_TYPE_(\w+);", block)
        if not names or not kind:
            continue
        switches = sorted(field for text, field in SWITCHES.items() if text in block)
        groups.append((names, kind.group(1), switches))
    return groups


def rust_string(text):
    """A Rust raw string literal for text."""
    hashes = "#" * (max((len(m) for m in re.findall(r'"#*', text)), default=0))
    return f'r{hashes}"{text}"{hashes}'


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tools/gen_pretok.py path/to/llama.cpp")
    source = (Path(sys.argv[1]) / "src" / "llama-vocab.cpp").read_text(encoding="utf-8")
    by_type = patterns_by_type(source)
    groups = families(source, by_type)

    print("//! The pre-tokenizer of every BPE family llama.cpp knows.")
    print("//!")
    print("//! Generated by tools/gen_pretok.py from llama.cpp's src/llama-vocab.cpp.")
    print("//! Do not edit by hand.")
    print()
    print("use super::pretok::Family;")
    print()
    print("/// The expressions used for a name that is missing or not in the table.")
    default_patterns, _ = by_type["DEFAULT"]
    print("pub(super) const DEFAULT_PATTERNS: &[&str] = &[")
    for pattern in default_patterns:
        print(f"    {rust_string(ascii_pattern(pattern))},")
    print("];")
    print()
    print("/// Each family: its names, the expressions that split text into words,")
    print("/// and its switches.")
    print("#[rustfmt::skip]")
    print(f"pub(super) static FAMILIES: [Family; {len(groups)}] = [")
    for names, kind, switches in groups:
        patterns, raw = by_type.get(kind, by_type["DEFAULT"])
        print("    Family {")
        print("        names: &[" + ", ".join(f'"{name}"' for name in names) + "],")
        print("        patterns: &[")
        for pattern in patterns:
            print(f"            {rust_string(ascii_pattern(pattern))},")
        print("        ],")
        for field in ["ignore_merges", "add_bos", "escape_spaces", "no_bos"]:
            print(f"        {field}: {'true' if field in switches else 'false'},")
        print(f"        raw_bytes: {'true' if raw else 'false'},")
        print("    },")
    print("];")


if __name__ == "__main__":
    main()

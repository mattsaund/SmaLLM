#!/bin/sh
# Install SmaLLM on Linux or macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.sh | sh
#
# Downloads the ready-made binary from the latest GitHub release and checks
# it against the release's SHA-256 list. Linux binaries are static, so they
# run on any distribution. When there is no binary for this machine, or this
# is run from inside a checkout, it builds from source with cargo.
#
# To remove SmaLLM again:
#
#   curl -fsSL https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.sh | sh -s -- --uninstall
#
# Override anything with the environment:
#   SMALLM_REPO=...         git URL to clone
#   SMALLM_REF=...          branch or tag (default: main). A version tag like
#                           v0.1.0 installs that release
#   SMALLM_PREFIX=...       where the binary lands (default: ~/.local/bin)
#   SMALLM_FROM_SOURCE=1    build even where a download is available

set -eu

REPO="${SMALLM_REPO:-https://github.com/mattsaund/SmaLLM.git}"
REF="${SMALLM_REF:-main}"
PREFIX="${SMALLM_PREFIX:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'install: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# A yes-or-no question, where Enter means yes. It reads from the terminal
# rather than stdin, because piped from the web stdin is this script. With no
# terminal to ask, the answer is no.
ask() {
    printf '  [Y/n] '
    if [ -t 0 ]; then reply_src=/dev/stdin; else reply_src=/dev/tty; fi
    { read -r reply < "$reply_src"; } 2>/dev/null || { reply=n; printf '\n'; }
    case "${reply:-y}" in [Nn]*) return 1 ;; esac
}

# `mktemp -d` with no template is a GNU extension. macOS wants a template.
temp_dir() { mktemp -d 2>/dev/null || mktemp -d -t smallm; }

# Everything to clean up on the way out, however we leave.
WORK=""
cleanup() {
    [ -n "$WORK" ] && rm -rf "$WORK"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- uninstall ----------------------------------------------------------------

if [ "${1:-}" = "--uninstall" ]; then
    if [ -e "$PREFIX/smallm" ]; then
        rm -f "$PREFIX/smallm"
        say "removed $PREFIX/smallm"
    else
        say "no smallm in $PREFIX"
    fi
    say "Downloaded models are kept. They are in ~/.local/share/smallm on Linux and"
    say "~/Library/Application Support/smallm on macOS. Delete that folder to remove them."
    exit 0
fi

# Put the binary at $1 in $PREFIX, over whatever smallm is already there.
# It is copied in beside the old one and renamed over it, which replaces a
# program even while it runs.
put_in_place() {
    mkdir -p "$PREFIX" \
        && cp "$1" "$PREFIX/.smallm.new" \
        && chmod +x "$PREFIX/.smallm.new" \
        && mv -f "$PREFIX/.smallm.new" "$PREFIX/smallm" \
        || { rm -f "$PREFIX/.smallm.new"; return 1; }
}

# What to say once there is a binary, from either route to one.
finish() {
    say ""
    say "installed $PREFIX/smallm ($("$PREFIX/smallm" version 2>/dev/null || echo unknown version))"
    case ":$PATH:" in
        *":$PREFIX:"*) ;;
        *)
            say "$PREFIX is not on your PATH. Add it:"
            say ""
            case "${SHELL##*/}" in
                fish) say "  fish_add_path $PREFIX" ;;
                zsh)  say "  echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.zshrc" ;;
                *)    say "  echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.bashrc" ;;
            esac
            say ""
            ;;
    esac
    say "get a model:   smallm pull bartowski/SmolLM2-135M-Instruct-GGUF"
    say "chat with it:  smallm run SmolLM2-135M"
}

# --- a ready-made binary ------------------------------------------------------
#
# Not when asked to build, and not for a branch: a branch today is not what
# any release was. A version tag names its release exactly, and `main` means
# the newest one.

# The SHA-256 of file $1, with whichever tool this system has.
sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | cut -d' ' -f1
    elif have shasum; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        return 1
    fi
}

prebuilt() {
    [ -z "${SMALLM_FROM_SOURCE:-}" ] || return 1
    case "$REF" in
        main) release=latest/download ;;
        v[0-9]*) release="download/$REF" ;;
        *) return 1 ;;
    esac
    have curl || return 1
    case "$(uname -m)" in
        arm64|aarch64) arch=aarch64 ;;
        x86_64|amd64) arch=x86_64 ;;
        armv7l|armv7) arch=armv7 ;;
        riscv64) arch=riscv64gc ;;
        *) return 1 ;;
    esac
    case "$(uname -s)" in
        Darwin) target="$arch-apple-darwin" ;;
        Linux)
            case "$arch" in
                armv7) target="armv7-unknown-linux-musleabihf" ;;
                *) target="$arch-unknown-linux-musl" ;;
            esac
            ;;
        *) return 1 ;;
    esac
    slug=${REPO#https://github.com/}
    slug=${slug%.git}
    case "$slug" in *://*|*/*/*) return 1 ;; */*) ;; *) return 1 ;; esac
    base="https://github.com/$slug/releases/$release"
    archive="smallm-$target.tar.gz"

    WORK="$(temp_dir)" || return 1
    printf 'downloading smallm for %s ... ' "$target"
    if ! curl -fsSL "$base/$archive" -o "$WORK/$archive" 2>/dev/null; then
        say "none published for it, so building from source instead"
        return 1
    fi

    # Check the download against the release's list of checksums.
    if curl -fsSL "$base/SHA256SUMS" -o "$WORK/SHA256SUMS" 2>/dev/null && actual=$(sha256_of "$WORK/$archive"); then
        expected=$(awk -v name="$archive" '$2 == name || $2 == "*" name { print $1 }' "$WORK/SHA256SUMS")
        if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
            say ""
            die "the download does not match its checksum. Nothing was installed."
        fi
    else
        printf '(checksum not verified) '
    fi

    if ! tar -xzf "$WORK/$archive" -C "$WORK" smallm 2>/dev/null; then
        say "could not unpack it, so building from source instead"
        return 1
    fi
    if ! "$WORK/smallm" version >/dev/null 2>&1; then
        say "it does not run on this system, so building from source instead"
        return 1
    fi
    put_in_place "$WORK/smallm" || die "could not put smallm in $PREFIX"
    say "done"
}

# True when the current folder is a checkout of this project.
in_checkout() {
    [ -f "./Cargo.toml" ] && grep -q '^name = "smallm"' ./Cargo.toml 2>/dev/null
}

# Running from inside a checkout means "build this", so a checkout never
# downloads.
if ! in_checkout && prebuilt; then
    finish
    exit 0
fi

# --- build from source --------------------------------------------------------
#
# SmaLLM has no dependencies, so a build needs only the Rust toolchain and
# the linker that comes with a C compiler.

if ! have cargo; then
    if [ -x "$HOME/.cargo/bin/cargo" ]; then
        PATH="$HOME/.cargo/bin:$PATH"
    else
        say "SmaLLM is written in Rust, and cargo is not installed."
        say "Install the Rust toolchain now? It goes in ~/.rustup and ~/.cargo,"
        say "and 'rustup self uninstall' removes it again."
        ask || die "cargo is required: see https://rustup.rs"
        have curl || die "curl is required to fetch rustup"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --profile minimal --default-toolchain stable
        PATH="$HOME/.cargo/bin:$PATH"
    fi
fi
have cargo || die "cargo still not on PATH"
have cc || have gcc || have clang || die "a C compiler is needed for its linker: install gcc or clang, then run this again"

if in_checkout; then
    SRC="$(pwd)"
    say "building from $SRC"
else
    have git || die "git is required to fetch the source"
    [ -n "$WORK" ] || WORK="$(temp_dir)" || die "could not make a temporary directory"
    SRC="$WORK/src"
    printf 'fetching %s (%s) ... ' "$REPO" "$REF"
    git clone --depth 1 --branch "$REF" "$REPO" "$SRC" >/dev/null 2>&1 \
        || { printf '\n'; die "could not clone $REPO"; }
    say "done"
fi

say "building (about half a minute)"
cargo build --release --locked --manifest-path "$SRC/Cargo.toml" --bin smallm \
    || die "the build failed: the output above says why"
[ -x "$SRC/target/release/smallm" ] || die "the build finished without leaving a binary"
put_in_place "$SRC/target/release/smallm" || die "could not put smallm in $PREFIX"
finish

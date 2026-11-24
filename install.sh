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
# Every step prints a line, and one progress bar for the whole installation
# stays at the bottom of the screen.
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

have() { command -v "$1" >/dev/null 2>&1; }

# --- the screen ----------------------------------------------------------------
#
# A log that scrolls, and one progress bar under it. log() clears the bar,
# prints its line, and draws the bar again, so the bar stays on the last
# line however much is printed above it. Without a terminal on stderr there
# is no bar and no color, only the log.

LIVE=
if [ -t 2 ] && [ "${TERM:-}" != dumb ]; then
    LIVE=1
fi
if [ -n "$LIVE" ] && [ -z "${NO_COLOR:-}" ]; then
    GREEN=$(printf '\033[1;32m')
    YELLOW=$(printf '\033[1;33m')
    RESET=$(printf '\033[0m')
else
    GREEN=
    YELLOW=
    RESET=
fi
# How far along the whole installation is, and what it is doing.
PERCENT=0
ACTIVITY="starting"
# Width of the bar itself, in characters.
BAR_WIDTH=30

# $2 repeated $1 times.
repeat() {
    count=$1
    out=""
    while [ "$count" -gt 0 ]; do
        out="$out$2"
        count=$((count - 1))
    done
    printf '%s' "$out"
}

# Draws the bar on the last line, over the one that was there.
draw_bar() {
    [ -n "$LIVE" ] || return 0
    filled=$((PERCENT * BAR_WIDTH / 100))
    printf '\r\033[2K[%s%s] %3d%%  %s' \
        "$(repeat "$filled" '#')" "$(repeat $((BAR_WIDTH - filled)) '-')" "$PERCENT" "$ACTIVITY" >&2
}

# Takes the bar off the screen.
clear_bar() {
    [ -n "$LIVE" ] && printf '\r\033[2K' >&2
    return 0
}

# log TAG TEXT: one line of the log. The tag is a word like Downloading.
log() {
    clear_bar
    printf '%s%12s%s %s\n' "$GREEN" "$1" "$RESET" "$2" >&2
    draw_bar
}

# warn TAG TEXT: a log line about something that did not stop the install.
warn() {
    clear_bar
    printf '%s%12s%s %s\n' "$YELLOW" "$1" "$RESET" "$2" >&2
    draw_bar
}

# progress PERCENT ACTIVITY: moves the bar.
progress() {
    PERCENT=$1
    ACTIVITY=$2
    draw_bar
}

die() {
    clear_bar
    printf '%s%12s%s %s\n' "$YELLOW" "Failed" "$RESET" "$*" >&2
    exit 1
}

# A byte count for people: 1.2 MB.
human() {
    awk -v bytes="$1" 'BEGIN {
        if (bytes >= 1e9) printf "%.1f GB", bytes / 1e9
        else if (bytes >= 1e6) printf "%.1f MB", bytes / 1e6
        else if (bytes >= 1e3) printf "%.1f KB", bytes / 1e3
        else printf "%d B", bytes
    }'
}

# Seconds as minutes and seconds: 1:05.
clock() {
    printf '%d:%02d' $(($1 / 60)) $(($1 % 60))
}

# A short wait between redraws. Fractions of a second work with the sleep
# of GNU, macOS and BusyBox, and the whole second is the fallback.
nap() {
    sleep 0.2 2>/dev/null || sleep 1
}

# A yes-or-no question, where Enter means yes. It reads from the terminal
# rather than stdin, because piped from the web stdin is this script. With no
# terminal to ask, the answer is no.
ask() {
    clear_bar
    printf '%12s %s [Y/n] ' "" "$1" >&2
    if [ -t 0 ]; then reply_src=/dev/stdin; else reply_src=/dev/tty; fi
    { read -r reply < "$reply_src"; } 2>/dev/null || { reply=n; printf '\n' >&2; }
    case "${reply:-y}" in [Nn]*) return 1 ;; esac
}

# `mktemp -d` with no template is a GNU extension. macOS wants a template.
temp_dir() { mktemp -d 2>/dev/null || mktemp -d -t smallm; }

# Everything to clean up on the way out, however we leave.
WORK=""
cleanup() {
    clear_bar
    [ -n "$WORK" ] && rm -rf "$WORK"
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- uninstall -----------------------------------------------------------------

if [ "${1:-}" = "--uninstall" ]; then
    if [ -e "$PREFIX/smallm" ]; then
        rm -f "$PREFIX/smallm"
        log "Removed" "$PREFIX/smallm"
    else
        log "Absent" "no smallm in $PREFIX"
    fi
    log "Kept" "downloaded models, in ~/.local/share/smallm on Linux and"
    log "" "~/Library/Application Support/smallm on macOS. Delete that folder to remove them."
    exit 0
fi

# --- steps that both routes share ------------------------------------------------

# fetch URL FILE FROM TO: downloads URL to FILE while the bar moves from FROM
# to TO percent. Logs what it fetches and how big it is.
fetch() {
    url=$1
    file=$2
    from=$3
    to=$4
    name=$(basename "$file")
    # The size comes first, from the headers alone, so the bar has a measure.
    total=$(curl -fsSIL "$url" 2>/dev/null | tr -d '\r' |
        awk 'tolower($1) == "content-length:" { size = $2 } END { print size + 0 }')
    if [ "$total" -gt 0 ]; then
        log "Downloading" "$name ($(human "$total"))"
    else
        log "Downloading" "$name"
    fi
    curl -fsSL "$url" -o "$file" 2>/dev/null &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        got=0
        [ -f "$file" ] && got=$(wc -c < "$file" | tr -d ' ')
        if [ "$total" -gt 0 ]; then
            progress $((from + (to - from) * got / total)) \
                "downloading $name  $(human "$got") / $(human "$total")"
        else
            progress "$from" "downloading $name  $(human "$got")"
        fi
        nap
    done
    wait "$pid" || return 1
    got=$(wc -c < "$file" | tr -d ' ')
    log "Received" "$name, $(human "$got")"
    progress "$to" "downloaded $name"
}

# show_line LINE: one line that a tool printed, as a line of the log. cargo
# already writes a tag and its text. rustup writes `info:` and the text.
show_line() {
    line=$(printf '%s' "$1" | sed 's/^[[:space:]]*//')
    case "$line" in
        "") return 0 ;;
        info:*) log "Rust" "${line#info: }" ;;
        warning:*) warn "Warning" "${line#warning: }" ;;
        error:*) warn "Error" "${line#error: }" ;;
        *)
            word=${line%% *}
            case "$word" in
                Compiling | Finished | Downloading | Downloaded | Updating | Locking | Building | Checking | Fresh)
                    rest=${line#"$word"}
                    log "$word" "${rest# }"
                    ;;
                *) log "" "$line" ;;
            esac
            ;;
    esac
}

# run_logged FROM TO ACTIVITY COMMAND...: runs a command, shows each line it
# prints in the log, and moves the bar from FROM toward TO while it runs.
# Nothing measures how far along a build is, so the bar closes in on TO and
# slows down as it does: it is halfway there after a minute.
run_logged() {
    from=$1
    to=$2
    activity=$3
    shift 3
    output="$WORK/step.log"
    : > "$output"
    "$@" > "$output" 2>&1 &
    pid=$!
    shown=0
    started=$(date +%s)
    while :; do
        running=
        kill -0 "$pid" 2>/dev/null && running=1
        # Only whole lines: a line still being written waits for its end.
        lines=$(tr '\r' '\n' < "$output" | wc -l | tr -d ' ')
        if [ "$lines" -gt "$shown" ]; then
            tr '\r' '\n' < "$output" | sed -n "$((shown + 1)),${lines}p" > "$WORK/step.new"
            while IFS= read -r new_line; do
                show_line "$new_line"
            done < "$WORK/step.new"
            shown=$lines
        fi
        [ -n "$running" ] || break
        elapsed=$(($(date +%s) - started))
        progress $((from + (to - from) * elapsed / (elapsed + 60))) "$activity ($(clock "$elapsed"))"
        nap
    done
    # A last line without a line break.
    tail_line=$(tr '\r' '\n' < "$output" | sed -n "$((shown + 1)),\$p")
    [ -n "$tail_line" ] && show_line "$tail_line"
    wait "$pid"
}

# Put the binary at $1 in $PREFIX, over whatever smallm is already there.
# It is copied in beside the old one and renamed over it, which replaces a
# program even while it runs.
put_in_place() {
    mkdir -p "$PREFIX" &&
        cp "$1" "$PREFIX/.smallm.new" &&
        chmod +x "$PREFIX/.smallm.new" &&
        mv -f "$PREFIX/.smallm.new" "$PREFIX/smallm" ||
        { rm -f "$PREFIX/.smallm.new"; return 1; }
}

# What to say once there is a binary, from either route to one.
finish() {
    progress 100 "installed"
    version=$("$PREFIX/smallm" version 2>/dev/null || echo "smallm, unknown version")
    log "Installed" "$version in $PREFIX/smallm"
    clear_bar
    LIVE=
    case ":$PATH:" in
        *":$PREFIX:"*) ;;
        *)
            warn "Path" "$PREFIX is not on your PATH. Add it with:"
            case "${SHELL##*/}" in
                fish) log "" "fish_add_path $PREFIX" ;;
                zsh) log "" "echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.zshrc" ;;
                *) log "" "echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.bashrc" ;;
            esac
            ;;
    esac
    log "Next" "get a model:   smallm pull bartowski/SmolLM2-135M-Instruct-GGUF"
    log "" "chat with it:  smallm run SmolLM2-135M"
}

# --- a ready-made binary ---------------------------------------------------------
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
        arm64 | aarch64) arch=aarch64 ;;
        x86_64 | amd64) arch=x86_64 ;;
        armv7l | armv7) arch=armv7 ;;
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
    case "$slug" in *://* | */*/*) return 1 ;; */*) ;; *) return 1 ;; esac
    base="https://github.com/$slug/releases/$release"
    archive="smallm-$target.tar.gz"

    log "Looking" "for a ready-made smallm for $target"
    progress 5 "looking for a release"
    WORK="$(temp_dir)" || return 1
    if ! curl -fsIL "$base/$archive" >/dev/null 2>&1; then
        warn "Missing" "no release has one for this machine, so it will be built from source"
        return 1
    fi
    fetch "$base/$archive" "$WORK/$archive" 10 85 || {
        warn "Failed" "the download broke off, so it will be built from source"
        return 1
    }

    # Check the download against the release's list of checksums.
    if fetch "$base/SHA256SUMS" "$WORK/SHA256SUMS" 85 90 && actual=$(sha256_of "$WORK/$archive"); then
        expected=$(awk -v name="$archive" '$2 == name || $2 == "*" name { print $1 }' "$WORK/SHA256SUMS")
        if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
            die "the download does not match its checksum. Nothing was installed."
        fi
        log "Verified" "SHA-256 $(printf '%s' "$actual" | cut -c1-16)... matches the release"
    else
        warn "Unverified" "no checksum to compare with"
    fi

    progress 93 "unpacking"
    if ! tar -xzf "$WORK/$archive" -C "$WORK" smallm 2>/dev/null; then
        warn "Failed" "could not unpack it, so it will be built from source"
        return 1
    fi
    log "Unpacked" "smallm, $(human "$(wc -c < "$WORK/smallm" | tr -d ' ')")"
    if ! "$WORK/smallm" version >/dev/null 2>&1; then
        warn "Failed" "it does not run on this system, so it will be built from source"
        return 1
    fi
    progress 97 "installing"
    put_in_place "$WORK/smallm" || die "could not put smallm in $PREFIX"
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

# --- build from source -----------------------------------------------------------
#
# SmaLLM has no dependencies, so a build needs only the Rust toolchain and
# the linker that comes with a C compiler.

[ -n "$WORK" ] || WORK="$(temp_dir)" || die "could not make a temporary directory"

if ! have cargo; then
    if [ -x "$HOME/.cargo/bin/cargo" ]; then
        PATH="$HOME/.cargo/bin:$PATH"
    else
        warn "Missing" "SmaLLM is written in Rust, and cargo is not installed."
        log "" "The Rust toolchain goes in ~/.rustup and ~/.cargo, and"
        log "" "'rustup self uninstall' removes it again."
        ask "Install it now?" || die "cargo is required: see https://rustup.rs"
        have curl || die "curl is required to fetch rustup"
        fetch "https://sh.rustup.rs" "$WORK/rustup-init.sh" 5 8 ||
            die "could not download the Rust installer"
        run_logged 8 40 "installing Rust" sh "$WORK/rustup-init.sh" -y \
            --profile minimal --default-toolchain stable ||
            die "the Rust installer failed: the lines above say why"
        PATH="$HOME/.cargo/bin:$PATH"
    fi
fi
have cargo || die "cargo still not on PATH"
have cc || have gcc || have clang || die "a C compiler is needed for its linker: install gcc or clang, then run this again"
log "Found" "$(cargo --version 2>/dev/null || echo cargo)"

if in_checkout; then
    SRC="$(pwd)"
    log "Building" "the checkout in $SRC"
else
    have git || die "git is required to fetch the source"
    SRC="$WORK/src"
    progress 42 "fetching the source"
    run_logged 42 48 "fetching the source" git clone --depth 1 --branch "$REF" "$REPO" "$SRC" ||
        die "could not clone $REPO"
    log "Fetched" "$REPO ($REF), $(find "$SRC" -type f -not -path '*/.git/*' | wc -l | tr -d ' ') files"
fi

run_logged 50 97 "compiling" cargo build --release --locked --manifest-path "$SRC/Cargo.toml" --bin smallm ||
    die "the build failed: the lines above say why"
[ -x "$SRC/target/release/smallm" ] || die "the build finished without leaving a binary"
log "Built" "smallm, $(human "$(wc -c < "$SRC/target/release/smallm" | tr -d ' ')")"
progress 98 "installing"
put_in_place "$SRC/target/release/smallm" || die "could not put smallm in $PREFIX"
finish

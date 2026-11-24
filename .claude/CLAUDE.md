# SmaLLM, for Claude

This folder is in the repository for a while so that a Claude instance on
another machine can pick the work up. `HANDOFF.md`, next to this file, says
where things stand and what to do next. Everything for that handoff is in
this one folder, `.claude/`, and one command removes it when the work is
done:

```sh
git rm -r .claude
```

The folder stays listed in `.gitignore`, so it was added with `git add -f`
and will not come back by accident.

## What this is

A small, fast LLM inference engine in Rust for GGUF models, aimed at
single-board computers. The goal is to beat llama.cpp on speed, memory and
size. The owner's words: "leave llama in the dust with speed, efficiency,
and size", and "use as little ram as possible".

## Rules that are not up for debate

1. **No dependencies.** The engine, the program and the C API use the Rust
   standard library only. `Cargo.lock` lists two packages, both ours.
2. **No Claude attribution in git.** No `Co-Authored-By` trailer, no
   "generated with" line, in commits or pull requests.
3. **No em dashes and no middle dots as separators**, anywhere: code
   comments, docs, commit messages, replies. Use commas, periods, colons or
   parentheses.
4. **American English.** Color, behavior, canceled, normalize.
5. **No run-on sentences.** Short, plain, technical wording in comments and
   docs. Do not lean on colons and semicolons.
6. **Comment the code.** Say what a function is for and why it is written
   the way it is.
7. **Keep the file layout easy to navigate.** A new file gets a line in the
   project map in `CONTRIBUTING.md`.
8. **Rust 1.85 must build it without warnings.** That is the Rust Debian 13
   and Raspberry Pi OS ship. Newer intrinsics go through inline assembly.
   CI checks this.

`tests/writing.rs` enforces rules 3 and 4 on every text file outside this
folder. Keep to them here as well.

## Commands

```sh
cargo build --release                  # target/release/smallm
cargo build --release -p smallm-capi   # the C library
cargo test --workspace                 # unit and integration tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Tests against llama.cpp output need model files. Missing ones are skipped.
SMALLM_TEST_MODELS=~/models cargo test --release --test golden

# Speed.
target/release/smallm bench model.gguf              # prompt and generation
target/release/smallm bench model.gguf --profile    # time per stage
cargo test --release --lib kernel_speed -- --ignored --nocapture
```

## Where things are

- `CONTRIBUTING.md` has the map of every file and the table of common
  changes.
- `docs/ARCHITECTURE.md` follows a prompt through the engine and explains
  the design.
- `.claude/HANDOFF.md` has the current state, the measurements so far and
  the open work.
- `tools/reference/README.md` says how to build the llama.cpp harness that
  the comparisons use.

## How to work here

- Measure before and after any change to a hot path, on a quiet machine,
  best of several runs. Say what was measured.
- Every kernel is tested against the portable one in `src/kernels/tests.rs`.
  A new kernel is covered by adding it to the backend's table.
- A change to the forward pass must keep `tests/golden.rs` passing on real
  models.
- CI runs on Linux, macOS and Windows, on x86-64 and ARM, plus 32-bit ARM
  and RISC-V under emulation. `gh workflow run ci --ref <branch>` runs it on
  a branch.
- Commit messages are one short line that says what changed, like "Add the
  Q4_1 kernel for NEON", with a body when the reason is not obvious.

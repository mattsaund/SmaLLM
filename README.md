# SmaLLM

SmaLLM is a CPU inference engine for language models in GGUF format, the
format llama.cpp and Ollama use. It targets single-board computers, mini PCs
and old laptops. One binary of about 1 MB, no dependencies, no required
configuration. Anonymous memory beyond the mapped model file is a few MB.

**Binary size, release builds:**\
1.22 MB Linux (x86_64), 1.01 MB Linux (arm64)\
0.94 MB Linux (armv7), 0.82 MB Linux (riscv64)\
1.03 MB macOS (Intel), 0.88 MB macOS (Apple silicon)\
1.26 MB Windows (x86_64), 1.04 MB Windows (arm64)

**Anonymous memory, Llama 3.2 1B, 4,096-token context:**\
7.7 MB. llama.cpp b11846: 627 MB. [Memory](#memory) has the method and the
trade-offs.

## Install

**Linux, macOS**

```sh
curl -fsSL https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.sh | sh
```

**Windows**

```powershell
irm https://raw.githubusercontent.com/mattsaund/SmaLLM/main/install.ps1 | iex
```

Downloads the newest [release](https://github.com/mattsaund/SmaLLM/releases)
for the machine and verifies it against the release's SHA-256 checksums.
Installs one binary to `~/.local/bin`. Linux binaries are static and run on
any distribution. Missing dependencies are installed after a prompt: curl
for model downloads, plus Rust and a C compiler when no release exists for
the machine and SmaLLM is built from source.

`smallm uninstall` removes the binary, downloaded models and settings.
What changed: [CHANGELOG](CHANGELOG.md).

## Use

```sh
smallm pull bartowski/SmolLM2-135M-Instruct-GGUF   # download a model, 105 MB
smallm run SmolLM2-135M                            # chat with it
smallm run SmolLM2-135M "Why is the sky blue?"     # one answer, then exit
smallm serve SmolLM2-135M                          # an OpenAI-compatible API
```

A model is a `.gguf` file, or the name of a downloaded model. Part of the
name is enough when it singles one out.

| Command | Does |
|---|---|
| `run <model> [prompt]` | Answers a prompt, or opens a chat when none is given |
| `serve <model>` | Serves `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize` and `/detokenize`, or for an embedding model `/v1/embeddings` and `/v1/rerank` |
| `embed <model> [text...]` | Prints the embedding of each text, one JSON array per line |
| `rerank <model> <query> <document>...` | Scores how well each document answers the query, best first |
| `pull <user/repo>` | Downloads a model from Hugging Face. `user/repo:q8_0` picks a quantization |
| `list`, `rm <model>` | Lists downloaded models, deletes one |
| `info <model>` | Shows what a model is: size, layers, weight format, chat format |
| `bench <model>` | Measures speed on this machine. `--profile` shows where the time goes |
| `tokenize <model> <text>` | Shows how a model reads a text |
| `config` | Shows the settings and where they are kept. `--reset` restores the defaults |
| `uninstall` | Removes the program, its downloaded models and its settings |

`smallm <command> --help` lists the options of a command: threads and the
CPUs to run on, context size, what to do when the context is full,
temperature and the other sampling settings, stop strings, system message.
Sampling settings that the model file recommends are the defaults. In a
chat, `/clear` starts over, `/system` sets a system message and `/exit`
leaves.

**Reasoning output.** For models that emit reasoning before the answer
(Qwen3, LFM3, Gemma 4, Inkling 2, DeepSeek R1, gpt-oss), the chat prints it
dimmed and keeps it out of the conversation history. Piped output omits it.
The server returns it as `reasoning_content`, the field llama.cpp uses.
`--think off` makes Qwen3, LFM3 and RWKV-7 G1 answer without reasoning, and
`--think on` turns it on for Gemma 4. gpt-oss always reasons and takes
`--think low`, `medium` or `high`. Inkling 2 sets its effort from the same
choices: off is 0, which shortens its reasoning but does not end it. In the
server, `chat_template_kwargs.enable_thinking` and `reasoning_effort` choose
per request.

**Settings.** Every setting has a line in one file, `smallm.conf`, named like
its command-line option. `smallm config` shows the settings in effect and
where the file is, and writes it with the defaults the first time.
`smallm config --reset` puts the defaults back. An option on the command
line wins over the file.

## What it runs

**Models.** Twenty-seven architectures for chat and text:

- Transformers: Llama, Qwen2, Qwen3, Gemma 2, Gemma 3, Gemma 4, Phi-2,
  Phi-3, Granite, SmolLM3, and gpt-oss.
- Mixtures of experts: Qwen2-MoE, Qwen3-MoE, OLMoE, Granite MoE, LFM2-MoE,
  DeepSeek-MoE, DeepSeek V2 and V3 with their latent attention, and Inkling
  2 and Inkling 2 Mini, whose blocks attend over the outputs of earlier
  blocks and whose middle layers run three times.
- Hybrids and models that are not transformers: LFM2 and LFM3 (attention
  and short convolutions), Mamba, Mamba 2, Granite 4.0 H (Mamba 2 and
  attention), RWKV-6 and RWKV-7.

That covers Llama 2 and 3, Mistral, Mixtral, TinyLlama, SmolLM2 and 3,
Qwen2.5, Qwen3, the DeepSeek R1 distillations, Gemma 2, 3 and 4, Phi-2 to
Phi-4-mini, gpt-oss, DeepSeek V2 and V3, Kimi and Moonlight, Granite 3 and
4, OLMoE, Falcon Mamba, Liquid AI's LFM2, LFM2.5 and LFM3, and Thinking
Machines' Inkling 2 Mini. BERT and nomic-bert models make embeddings and
rerank. The engine is tested against llama.cpp on 28 models. Tokens match
exactly, and greedy output matches up to near ties between two candidates.
LFM3 is newer than the llama.cpp it is tested against, so it is checked
against llama.cpp's source instead. Inkling 2 Mini is tested against
llama.cpp b12711.

**Tokenizers.** SentencePiece, byte-level BPE with the word splitting rules
of all 71 families llama.cpp knows, Gemma 4's, RWKV's World tokenizer, and
the encoder tokenizers WordPiece (BERT) and Unigram (T5, XLM-RoBERTa).
llama.cpp's own tokenizer test texts pass on every vocabulary it ships that
uses one of these, and on RWKV's.

**Weights.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, the i-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
IQ3_S, IQ4_NL and IQ4_XS, the ternary TQ1_0 and TQ2_0, MXFP4, NVFP4, and the
1- and 2-bit Q1_0 and Q2_0.
Dequantization is bit-exact with ggml for every format, checked by
`tests/quant.rs`. Weights stay in the file's format and are memory mapped.
Loading takes milliseconds and allocates no memory for weights.

**Platforms.** Linux, macOS and Windows. x86-64 with AVX2 and ARM64 with
NEON get hand-written kernels, selected at startup. Everything else,
including 32-bit ARM and RISC-V, runs portable code.

**Chat formats.** ChatML, Llama 3, Llama 2, Mistral, Gemma, Gemma 4, Granite,
Phi-3, Phi-4-mini, Zephyr, DeepSeek, DeepSeek V2, Kimi, OpenAI's Harmony
(gpt-oss), RWKV World and Inkling, recognized from the model file.

## Speed

Llama 3.2 1B, Q4_K_M, in tokens per second:

| Threads | Prompt, SmaLLM | Prompt, llama.cpp | Generation, SmaLLM | Generation, llama.cpp |
|---|---|---|---|---|
| 1 | **126** | 101 | 24.0 | **25.2** |
| 4 | **470** | 373 | **49.6** | 44.4 |
| 8 | **829** | 707 | **51.7** | 45.6 |

Measured on an Intel Core i7-14700K under Linux, against llama.cpp b11846
with its `alderlake` CPU backend: a 256-token prompt and 96 generated
tokens, the best of three runs, with the engines taking turns.

## Memory

Weights are never copied out of the model file, and nothing is allocated
before it is used. Anonymous memory (RssAnon), Llama 3.2 1B Q4_K_M:

| | Model loaded | Session, 4,096-token context | After 256 tokens |
|---|---|---|---|
| SmaLLM | **7.6 MB** | **7.7 MB** | **24 MB** |
| llama.cpp | 498 MB | 627 MB | 636 MB |

The difference grows with the weights llama.cpp repacks. On gpt-oss-20b
(MXFP4) it is 59 MB against 9,966 MB.

### How it works

- **Weights are read in place.** The GGUF file is memory mapped, and the
  kernels read its quantized blocks in the file's layout. llama.cpp repacks
  most weight tensors into an interleaved layout in anonymous memory at
  load, which is most of its 498 MB. SmaLLM packs a few rows into a
  cache-sized tile per prompt batch instead, the packing step of BLAS-style
  matrix multiplication (GotoBLAS, BLIS). Generation reads the file layout
  directly.
- **The key/value cache is allocated on demand**, in 256-position chunks of
  16-bit floats. llama.cpp allocates the whole context window when the
  session starts. Memory scales with the tokens in the context, not with
  the configured window.
- **Small working set.** Scratch buffers are sized for a 64-token batch,
  or 128 for models with experts. The vocabulary is read from the mapped
  file.
  Logits are computed for the last token only.
- **Per-architecture state.** DeepSeek's latent attention caches one
  compressed row per position and layer. Mamba and RWKV keep a fixed-size
  recurrent state and one saved copy of it. Gemma 4's shared layers have no
  cache of their own.

### Use cases

- **Low-RAM boards.** RAM goes to the model's pages and to other processes.
- **Models close to RAM size, especially mixtures of experts.** A token
  reads only its routed experts. The kernel evicts the pages of cold experts
  and faults them back in when they are needed.
- **Several processes on one model.** Mapped file pages are shared through
  the page cache, so every process uses the same copy of the weights.
- **Millisecond loads.** Nothing is copied or repacked at load.

### Trade-offs

- **Total memory still includes the model.** The figures above leave out
  file-backed pages, and full speed needs the file resident. The total for
  Llama 3.2 1B after 256 tokens is 0.83 GB for SmaLLM and 1.4 GB for
  llama.cpp.
- **Under memory pressure, speed depends on storage.** The kernel evicts
  file pages and faults them back in on access. That is workable on NVMe or
  eMMC and slow on an SD card. llama.cpp's repacked copy is anonymous memory
  and can only go to swap. `--mlock` keeps the whole file resident instead,
  which costs reading all of it at load and needs a locked-memory limit
  (`ulimit -l`) as large as the file.
- **No up-front check.** A context window too large for the machine fails
  when the conversation reaches it, not when the session starts.
- **Packing on every prompt batch.** Tiles are packed per batch, not once at
  load. On x86 the kernels absorb the cost (see [Speed](#speed)). On an
  Allwinner A733's Cortex-A76 cores packing takes about 6 percent of a
  prompt's time, and Q4_0 prompts run 9 percent behind llama.cpp on the two
  fast cores and ahead of it on all eight ([docs/PARITY.md](docs/PARITY.md)).
  Q4_K_M prompts run 1.8 times as fast there.

Measured with `smallm bench`, which prints these figures on Linux, against
llama.cpp b11846 on an Intel Core i7-14700K with a 4,096-token context.

## Size

The program (chat, server, downloader) is 0.82 to 1.26 MB depending on the
platform. The engine as a shared library is 0.58 to 0.86 MB. llama.cpp
b11846's CPU-only libraries total 7.7 MB on the same machine. llama.cpp has
features SmaLLM lacks, listed in [docs/TODO.md](docs/TODO.md).

Release builds compile the standard library for size (`-Z build-std` with
`optimize_for_size`), which saves about 20 percent. A source build with
stable Rust is about 1.5 MB.

## Context

The context window has no built-in limit. The key/value cache is allocated
in 256-token chunks as the context fills, so a large window allocates
nothing up front.

```sh
smallm run model.gguf --ctx 262144
```

Tested end to end at 262,144 tokens. The practical limit is memory, 32 KB
per token for Llama 3.2 1B. Output quality degrades past the model's
training length. The default window is the model's training length, capped
at 4,096 tokens.

When the window is full, `--overflow` (or the `overflow` setting) decides:

| Choice | Does |
|---|---|
| `middle` (default) | Keeps the start, the system prompt and the first question, and drops the oldest tokens after it |
| `slide` | Drops the oldest tokens and slides forward |
| `stop` | Ends the answer |

Kept tokens are not evaluated again. Their cached keys are rotated to their
new positions, which takes milliseconds.

## Use it from your code

SmaLLM is a library first. The program is one user of it.

**Rust**

```rust
use std::sync::Arc;
use smallm::{generate, Model, Session, SessionConfig};

let model = Arc::new(Model::load("model.gguf")?);
let mut session = Session::new(model, SessionConfig::default())?;
let options = generate::Options { max_tokens: 64, ..Default::default() };
generate::run(&mut session, "Once upon a time", &options, |piece| {
    print!("{piece}");
    true
})?;
```

**C, and every language that can call C**

```c
#include "smallm.h"

smallm_model *model = smallm_model_load("model.gguf");
smallm_session *session = smallm_session_new(model, NULL);
smallm_generate(session, "Once upon a time", NULL, print, NULL, NULL);
```

**Python**

```python
import smallm

model = smallm.Model("model.gguf")
session = model.session()
print(session.chat([{"role": "user", "content": "Hello!"}]))
```

**Anything that speaks HTTP**

```sh
smallm serve model.gguf
curl http://127.0.0.1:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

Each release has the library for Linux, macOS and Windows, shared and
static, with the header and the Python binding.
[capi/README.md](capi/README.md) covers the C API, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) how the engine works.

## Status

Version 0.4.0. [docs/TODO.md](docs/TODO.md) lists what is done and what is
left, and [docs/PARITY.md](docs/PARITY.md) compares SmaLLM with llama.cpp
and Ollama area by area, with measurements.

## Build from source

Rust 1.85 or newer, and nothing else.

```sh
git clone https://github.com/mattsaund/SmaLLM
cd SmaLLM
cargo build --release                  # target/release/smallm
cargo build --release -p smallm-capi   # the C library, beside it
cargo test --workspace
```

The smaller build that the releases use needs a nightly compiler.
`.github/workflows/release.yml` has the exact flags.

## Contributing

**AI policy.** I am open to AI and agentic coding, but the code written needs
to follow specific guidelines:

1. MUST be human readable, acceptable variable/function names.
2. Easily traceable, following a good program flow.
3. Contributors MUST look at and document code and code changes. You need to
   understand the code that is being written.

**Tests.** `cargo test`, `cargo clippy` and `cargo fmt`. CI runs all three on
Linux, macOS and Windows, on x86-64 and ARM, and compares the engine's output
with llama.cpp's on real models. [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
has the project map and the rules, and
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) what to measure on a
single-board computer. A version is released by tagging it. See
`.github/workflows/release.yml`.

## License and credits

Created by Matthew Saunders, https://msaunders.dev

MIT. See [LICENSE](LICENSE). SmaLLM has no dependencies, so nothing else is
compiled in. The GGUF format and the layouts of the quantized weights come
from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), whose output
is the reference this engine is tested against. The character tables in
`src/tokenizer/unicode_tables.rs` are derived from the Unicode Character
Database.

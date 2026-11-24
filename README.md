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

`smallm <command> --help` lists the options of a command: threads, context
size, what to do when the context is full, temperature and the other
sampling settings, stop strings, system message. In a chat, `/clear` starts
over, `/system` sets a system message and `/exit` leaves.

**Reasoning output.** For models that emit reasoning before the answer
(Qwen3, DeepSeek R1, gpt-oss), the chat prints it dimmed and keeps it out of
the conversation history. Piped output omits it. The server returns it as
`reasoning_content`, the field llama.cpp uses.

**Settings.** Every setting has a line in one file, `smallm.conf`, named like
its command-line option. `smallm config` shows the settings in effect and
where the file is, and writes it with the defaults the first time.
`smallm config --reset` puts the defaults back. An option on the command
line wins over the file.

## What it runs

**Models.** Twenty-five architectures for chat and text:

- Transformers: Llama, Qwen2, Qwen3, Gemma 2, Gemma 3, Gemma 4, Phi-2,
  Phi-3, Granite, SmolLM3, and gpt-oss.
- Mixtures of experts: Qwen2-MoE, Qwen3-MoE, OLMoE, Granite MoE, LFM2-MoE,
  DeepSeek-MoE, and DeepSeek V2 and V3 with their latent attention.
- Hybrids and models that are not transformers: LFM2 and LFM3 (attention
  and short convolutions), Mamba, Mamba 2, Granite 4.0 H (Mamba 2 and
  attention), RWKV-6 and RWKV-7.

That covers Llama 2 and 3, Mistral, Mixtral, TinyLlama, SmolLM2 and 3,
Qwen2.5, Qwen3, the DeepSeek R1 distillations, Gemma 2, 3 and 4, Phi-2 to
Phi-4-mini, gpt-oss, DeepSeek V2 and V3, Kimi and Moonlight, Granite 3 and
4, OLMoE, Falcon Mamba, and Liquid AI's LFM2, LFM2.5 and LFM3. BERT and
nomic-bert models make embeddings and rerank. The engine is tested against
llama.cpp on 27 models. Tokens match exactly, and greedy output matches up
to near ties between two candidates. LFM3 is newer than the llama.cpp it is
tested against, so it is checked against llama.cpp's source instead.

**Tokenizers.** SentencePiece, byte-level BPE with the word splitting rules
of all 67 families llama.cpp knows, Gemma 4's, RWKV's World tokenizer, and
the encoder tokenizers WordPiece (BERT) and Unigram (T5, XLM-RoBERTa).
llama.cpp's own tokenizer test texts pass on every vocabulary it ships that
uses one of these, and on RWKV's.

**Weights.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, the i-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
IQ3_S, IQ4_NL and IQ4_XS, the ternary TQ1_0 and TQ2_0, and MXFP4.
Dequantization is bit-exact with ggml for every format, checked by
`tests/quant.rs`. Weights stay in the file's format and are memory mapped.
Loading takes milliseconds and allocates no memory for weights.

**Platforms.** Linux, macOS and Windows. x86-64 with AVX2 and ARM64 with
NEON get hand-written kernels, selected at startup. Everything else,
including 32-bit ARM and RISC-V, runs portable code.

**Chat formats.** ChatML, Llama 3, Llama 2, Mistral, Gemma, Gemma 4, Granite,
Phi-3, Phi-4-mini, Zephyr, DeepSeek, DeepSeek V2, Kimi, OpenAI's Harmony
(gpt-oss) and RWKV World, recognized from the model file.

## Speed

Llama 3.2 1B, Q4_K_M, in tokens per second:

| Threads | Prompt, SmaLLM | Prompt, llama.cpp | Generation, SmaLLM | Generation, llama.cpp |
|---|---|---|---|---|
| 1 | **121** | 102 | 23.6 | **25.0** |
| 4 | **448** | 372 | **47.6** | 42.7 |
| 8 | **765** | 657 | **50.4** | 44.6 |

Measured on an Intel Core i7-14700K under Linux, against llama.cpp b11846
with its `alderlake` CPU backend: a 256-token prompt and 96 generated
tokens, the best of three runs, with the engines taking turns.

## Memory

Weights are never copied out of the model file, and nothing is allocated
before it is used. Anonymous memory (RssAnon), Llama 3.2 1B Q4_K_M:

| | Model loaded | Session, 4,096-token context | After 256 tokens |
|---|---|---|---|
| SmaLLM | **7.5 MB** | **7.7 MB** | **21 MB** |
| llama.cpp | 498 MB | 627 MB | 636 MB |

The difference grows with the weights llama.cpp repacks. On gpt-oss-20b
(MXFP4) it is 33 MB against 9,966 MB.

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
- **Small working set.** Scratch buffers are sized for a 32-token batch. The
  vocabulary is read from the mapped file. Logits are computed for the last
  token only.
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
  and can only go to swap.
- **No up-front check.** A context window too large for the machine fails
  when the conversation reaches it, not when the session starts.
- **Packing on every prompt batch.** Tiles are packed per batch, not once at
  load. On x86 the kernels absorb the cost (see [Speed](#speed)). On ARM it
  has not been measured.

Measured with `smallm bench`, which prints these figures on Linux, against
llama.cpp b11846 on an Intel Core i7-14700K with a 4,096-token context.

## Size

The program (chat, server, downloader) is 0.82 to 1.26 MB depending on the
platform. The engine as a shared library is 0.58 to 0.86 MB. llama.cpp
b11846's CPU-only libraries total 7.7 MB on the same machine. llama.cpp has
features SmaLLM lacks, listed under [To do](#to-do).

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

## Project status

- [x] GGUF loading by memory map, with every read bounds-checked
- [x] Twenty-five architectures, including mixtures of experts, hybrids, and the state-space and RWKV families
- [x] Twenty-five weight formats, float, quantized, i-quants, ternary and MXFP4, each expanding exactly as in ggml
- [x] Kernels for AVX2, AVX-VNNI, NEON and the ARM dot product extension
- [x] Prompt reading in packed tiles
- [x] Anonymous memory in megabytes, with weights read in place from the mapped file
- [x] SentencePiece and byte-level BPE tokenizers for every word splitting family llama.cpp knows, RWKV's, and
      the encoder tokenizers WordPiece and Unigram, matching it token for token on its test texts
- [x] Fifteen chat formats, recognized from the model file
- [x] Embedding and reranking models, from the command line and the server
- [x] Reasoning output separated from the answer in the chat, in pipes and in the server
- [x] A context window that slides or keeps its start when it fills, chosen with `--overflow`
- [x] One settings file for everything, with a reset to the defaults
- [x] A context window with no built-in limit, tested at 262,144 tokens
- [x] Chat, one-shot answers, model downloads and benchmarks from one program
- [x] An OpenAI-compatible server with streaming
- [x] A C API as a shared and a static library, and a Python binding
- [x] One-line installers that bring what SmaLLM needs, a full uninstall, and static Linux binaries for x86-64,
      ARM64, ARMv7 and RISC-V
- [x] CI on Linux, macOS and Windows, on x86-64 and ARM, with ARMv7 and RISC-V under emulation
- [ ] Speed and memory measured on ARM boards
- [x] Tagged releases, the latest 0.3.0

## To do

Features llama.cpp, Ollama or other engines have and SmaLLM lacks. Heavy
items are planned as optional parts of the build, so the default binary
stays small. [docs/PARITY.md](docs/PARITY.md) compares the three engines
area by area, with measurements.

**Hardware**

- [ ] GPUs, through Vulkan compute. It is the one API that reaches AMD, Intel and NVIDIA cards and the ARM GPUs in
      phones and boards, so one backend covers them. The engine runs on the CPU only today.
- [ ] Apple GPUs through Metal, and NVIDIA through CUDA, where they outperform Vulkan.
- [ ] NPUs. Every vendor has its own toolkit (Rockchip RKNN, Qualcomm QNN, Intel OpenVINO, Apple Core ML), so each
      one is a backend of its own. None exists yet.
- [ ] TPUs. Google's Coral Edge TPU cannot hold a language model: it runs small, fully 8-bit TensorFlow Lite graphs
      and keeps about 8 MB of weights. Support waits for hardware that can.
- [ ] AVX-512 kernels, NEON kernels for 32-bit ARM, and vector kernels for RISC-V.
- [x] A separate thread count for reading prompts, every core by default. Efficiency cores help there and not
      in generation.

**Models**

- [x] Mixture-of-experts models: Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, Granite MoE and LFM2-MoE.
- [x] Hybrids of attention and short convolutions: LFM2, LFM2.5 and LFM3.
- [x] More architectures: Phi, Gemma 2, the MoE models of DeepSeek and gpt-oss, and the families that are not
      transformers (RWKV, Mamba).
- [ ] The other hybrids and distillations: Falcon-H1, Jamba, Nemotron-H, Qwen3-Next, and RWKV's distillations of
      Qwen (rwkv6qwen2, arwkv7).
- [x] Encoder tokenizers (WordPiece and Unigram), which only the encoder models below use.
- [x] Encoder models, for embeddings and reranking.
- [ ] Images and audio as input.
- [x] The remaining weight formats: IQ1, IQ2, IQ3, IQ4_XS, the ternary ones, and MXFP4.
- [ ] LoRA adapters and control vectors.

**Generation**

- [ ] Grammar-constrained output, and JSON that follows a schema.
- [ ] Jinja chat templates, for models whose format is not one of the fifteen built in.
- [ ] More samplers: Mirostat, typical, DRY, XTC and dynamic temperature, and a bias on chosen tokens.
- [ ] Token probabilities in the output.
- [ ] Speculative decoding with a small draft model.
- [ ] Saving a session's context to a file and loading it again.
- [x] Sliding the context window when it fills, or keeping its start and dropping the middle.
- [x] A half-precision key/value cache, to halve the memory of long contexts.

**Server**

- [ ] Several requests at once.
- [x] Endpoints for embeddings and reranking.
- [ ] An endpoint for code infill.
- [ ] Tool calls.
- [ ] The Ollama API next to the OpenAI one, and several models loaded and unloaded on demand.
- [ ] A built-in web page, and metrics.

**Tools**

- [ ] Converting and quantizing models. Today the GGUF file comes from elsewhere.
- [ ] Perplexity and the other evaluation tools.
- [ ] Bindings beyond Python, starting with Node.js and Go.
- [ ] A `no_std` core, for boards without an operating system.

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

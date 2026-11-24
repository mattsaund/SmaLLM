# SmaLLM

SmaLLM is a small, fast engine that runs language models on your own
hardware. It reads the same GGUF model files as llama.cpp and Ollama, and it
is built for the low end: single-board computers, mini PCs and old laptops.
One program of under a megabyte, which holds a few megabytes of memory
besides the model. No dependencies, and the defaults need no changes.

**Full installation size:**\
0.87 MB Linux (x86_64), 0.74 MB Linux (arm64)\
0.70 MB Linux (armv7), 0.60 MB Linux (riscv64)\
0.70 MB macOS (Intel), 0.62 MB macOS (Apple silicon)\
0.90 MB Windows (x86_64), 0.78 MB Windows (arm64)

**Memory of its own, for Llama 3.2 1B with a 4,096-token context:**\
7.7 MB, where llama.cpp holds 627 MB. [Memory](#memory) says how it was
measured.

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
for your machine and checks it against the release's checksums. One file
lands in `~/.local/bin`. Linux binaries are static, so they run on any
distribution. Anything else SmaLLM needs is installed too, after asking:
curl for downloading models, and when there is no release for the machine,
Rust and a C compiler to build it from source.

`smallm uninstall` removes SmaLLM again, with its models and settings.
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
| `serve <model>` | Serves `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize` and `/detokenize` |
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

**Settings.** Every setting has a line in one file, `smallm.conf`, named like
its command-line option. `smallm config` shows the settings in effect and
where the file is, and writes it with the defaults the first time.
`smallm config --reset` puts the defaults back. An option on the command
line wins over the file.

## What it runs

**Models.** Fourteen architectures: Llama, Qwen2, Qwen3, Gemma 3, Gemma 4,
LFM2, LFM3, Granite, OLMoE, SmolLM3, and the mixture-of-experts Qwen2-MoE,
Qwen3-MoE, Granite MoE and LFM2-MoE. That covers Llama 2 and 3, Mistral,
Mixtral, TinyLlama, SmolLM2 and 3, Qwen2.5, Qwen3, the DeepSeek R1
distillations, Gemma 3 and 4, Granite 3, OLMoE, and Liquid AI's LFM2,
LFM2.5 and LFM3, including the hybrids of attention and convolution. The
engine is tested against llama.cpp on thirteen models. Tokens match
exactly, and greedy output matches up to near ties between two candidates.
LFM3 is newer than the llama.cpp it is tested against, so it is checked
against llama.cpp's source instead.

**Tokenizers.** SentencePiece, byte-level BPE with the word splitting rules
of all 67 families llama.cpp knows, and Gemma 4's. llama.cpp's own tokenizer
test texts pass on every vocabulary it ships that uses one of these.

**Weights.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K and IQ4_NL. Weights stay in the file's format and are memory
mapped, so a model loads in milliseconds and costs no memory beyond the
file.

**Machines.** Linux, macOS and Windows. x86-64 with AVX2 and ARM64 with NEON
get hand-written kernels, picked when the program starts. Everything else,
including 32-bit ARM and RISC-V, runs portable code.

**Chat formats.** ChatML, Llama 3, Llama 2, Mistral, Gemma, Gemma 4, Granite,
Phi-3, Zephyr and DeepSeek, recognized from the model file.

## Speed

Llama 3.2 1B, Q4_K_M, in tokens per second:

| Threads | Prompt, SmaLLM | Prompt, llama.cpp | Generation, SmaLLM | Generation, llama.cpp |
|---|---|---|---|---|
| 1 | **120** | 103 | **24.4** | 23.6 |
| 4 | **427** | 383 | **46.7** | 45.2 |
| 8 | **757** | 681 | **50.8** | 49.2 |

Measured on an Intel Core i7-14700K under Linux, against llama.cpp b11846
with its `alderlake` CPU backend: a 256-token prompt and 96 generated
tokens, the best of three runs, with the engines taking turns.

## Memory

SmaLLM never copies weights out of the model file. The file is mapped into
memory, and the system keeps as much of it there as it has room for. What
the engine holds of its own is small. For Llama 3.2 1B:

| | Model loaded | With a 4,096-token context | After reading 256 tokens |
|---|---|---|---|
| SmaLLM | **7.5 MB** | **7.7 MB** | **29 MB** |
| llama.cpp | 498 MB | 627 MB | 658 MB |

This is memory the process allocated, which the system cannot take back.
Both engines also have the 808 MB model file in memory, and the system can
drop those pages whenever it needs the room.

Two things make the difference. llama.cpp reserves the cache for the whole
context window when a session starts, and SmaLLM grows it as the
conversation grows. llama.cpp also keeps a rearranged copy of most weights
for speed, and SmaLLM rearranges a few rows at a time in a few kilobytes of
scratch memory. `smallm bench` prints these numbers for any model on Linux.

## Size

The whole program, with chat, server and downloader, is 0.60 to 0.90 MB
depending on the platform. The engine as a shared library is 0.37 to
0.55 MB. For scale, the CPU-only libraries of llama.cpp b11846 total 7.7 MB
on the same machine. llama.cpp also does far more, as the to do list below
shows.

Released binaries are built with a standard library that is compiled for
size, which takes a third off. Building from source with stable Rust gives
the same program at about 1.2 MB.

## Context

The context window has no built-in limit. The memory for it grows in steps
of 256 tokens as the conversation grows, so a large window costs nothing
until it is used.

```sh
smallm run model.gguf --ctx 262144
```

A 262,144-token context has been run end to end. What limits the window in
practice is memory, 66 KB per token for Llama 3.2 1B. A model also only
stays coherent as far as it was trained to. The default window is the
model's own training length, up to 4,096 tokens.

When the window is full, `--overflow` (or the `overflow` setting) decides:

| Choice | Does |
|---|---|
| `middle` (default) | Keeps the start, the system prompt and the first question, and drops the oldest tokens after it |
| `slide` | Drops the oldest tokens and slides forward |
| `stop` | Ends the answer |

Either way the kept tokens are not read again: their cached keys are moved
to their new positions, which takes milliseconds.

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
- [x] Fourteen architectures, including mixtures of experts and hybrids of attention and convolution layers
- [x] Fourteen weight formats, float and quantized
- [x] Kernels for AVX2, AVX-VNNI, NEON and the ARM dot product extension
- [x] Prompt reading in tiles, up to twice as fast as llama.cpp
- [x] A few megabytes of memory besides the model file
- [x] SentencePiece and byte-level BPE tokenizers for every word splitting family llama.cpp knows, matching it
      token for token on its test texts
- [x] Ten chat formats, recognized from the model file
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
- [x] A first tagged release, 0.2.0

## To do

Everything here is something llama.cpp, Ollama or another engine does today
and SmaLLM does not. The aim is to close the list without giving up the
size, so the heavy items are meant to be parts of the build that can be
left out.

**Hardware**

- [ ] GPUs, through Vulkan compute. It is the one API that reaches AMD, Intel and NVIDIA cards and the ARM GPUs in
      phones and boards, so one backend covers them. The engine runs on the CPU only today.
- [ ] Apple GPUs through Metal, and NVIDIA through CUDA, where they beat what Vulkan gives.
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
- [ ] More architectures: Phi, Gemma 2, the MoE models of DeepSeek and gpt-oss, and the families that are not
      transformers (RWKV, Mamba).
- [ ] Encoder tokenizers (WordPiece and Unigram), which only the encoder models below use.
- [ ] Encoder models, for embeddings and reranking.
- [ ] Images and audio as input.
- [ ] The remaining weight formats: IQ1, IQ2, IQ3, IQ4_XS and the ternary ones.
- [ ] LoRA adapters and control vectors.

**Generation**

- [ ] Grammar-constrained output, and JSON that follows a schema.
- [ ] Jinja chat templates, for models whose format is not one of the ten built in.
- [ ] More samplers: Mirostat, typical, DRY, XTC and dynamic temperature, and a bias on chosen tokens.
- [ ] Token probabilities in the output.
- [ ] Speculative decoding with a small draft model.
- [ ] Saving a session's context to a file and loading it again.
- [x] Sliding the context window when it fills, or keeping its start and dropping the middle.
- [ ] A half-precision key/value cache, to halve the memory of long contexts.

**Server**

- [ ] Several requests at once.
- [ ] Endpoints for embeddings, reranking and code infill.
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
with llama.cpp's on real models. [CONTRIBUTING.md](CONTRIBUTING.md) has the
project map and the rules. A version is released by tagging it. See
`.github/workflows/release.yml`.

## License and credits

Created by Matthew Saunders, https://msaunders.dev

MIT. See [LICENSE](LICENSE). SmaLLM has no dependencies, so nothing else is
compiled in. The GGUF format and the layouts of the quantized weights come
from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), whose output
is the reference this engine is tested against. The character tables in
`src/tokenizer/unicode_tables.rs` are derived from the Unicode Character
Database.

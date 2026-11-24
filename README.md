# SmaLLM

SmaLLM is a small, fast engine that runs language models on your own
hardware. It reads the same GGUF model files as llama.cpp and Ollama, and it
is built for the low end: single-board computers, mini PCs and old laptops.
One program of about half a megabyte, which holds a few megabytes of memory
besides the model. No dependencies, nothing to configure.

**Full installation size:**\
0.67 MB Linux (x86_64), 0.57 MB Linux (arm64)\
0.54 MB Linux (armv7), 0.47 MB Linux (riscv64)\
0.51 MB macOS (Intel), 0.47 MB macOS (Apple silicon)\
0.70 MB Windows (x86_64), 0.61 MB Windows (arm64)

**Memory of its own, with a model loaded and a 4,096-token context:**\
2.2 MB for SmolLM2 135M, where llama.cpp holds 116 MB\
6.5 MB for Qwen2.5 0.5B, where llama.cpp holds 124 MB\
7.7 MB for Llama 3.2 1B, where llama.cpp holds 627 MB

That is 19 to 81 times less. [Memory](#memory) has the full table and how it
was measured.

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
for your machine and checks it against the release's checksums. Where there
is no release for the machine, it builds from source with cargo. One file
lands in `~/.local/bin`. Linux binaries are static, so they run on any
distribution. The top of each script says how to uninstall.
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

`smallm <command> --help` lists the options of a command: threads, context
size, temperature and the other sampling settings, stop strings, system
message. In a chat, `/clear` starts over, `/system` sets a system message and
`/exit` leaves.

## What it runs

**Models.** The Llama, Qwen2, Qwen3 and Gemma 3 architectures. That covers
Llama 2 and 3, TinyLlama, SmolLM2, Mistral, Qwen2.5, Qwen3, the DeepSeek R1
distillations and Gemma 3. The engine is tested against llama.cpp on eight
models. Tokens match exactly, and greedy output matches up to near ties
between two candidates.

**Weights.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K and IQ4_NL. Weights stay in the file's format and are memory
mapped, so a model loads in milliseconds and costs no memory beyond the
file.

**Machines.** Linux, macOS and Windows. x86-64 with AVX2 and ARM64 with NEON
get hand-written kernels, picked when the program starts. Everything else,
including 32-bit ARM and RISC-V, runs portable code.

**Chat formats.** ChatML, Llama 3, Llama 2, Mistral, Gemma, Phi-3, Zephyr and
DeepSeek, recognized from the model file.

## Speed

Tokens per second on an Intel Core i7-14700K, against llama.cpp b11846 with
its `alderlake` CPU backend. Both ran on the CPU only, on the same Q4_K_M
files.

**Reading a prompt**

| Model | Threads | SmaLLM | llama.cpp | |
|---|---|---|---|---|
| SmolLM2 135M | 1 | **843** | 431 | 2.0x |
| | 4 | **2780** | 1550 | 1.8x |
| | 8 | **4263** | 2724 | 1.6x |
| Gemma 3 270M | 1 | **896** | 484 | 1.9x |
| | 4 | **3002** | 1731 | 1.7x |
| | 8 | **4241** | 2548 | 1.7x |
| Qwen2.5 0.5B | 1 | **267** | 147 | 1.8x |
| | 4 | **964** | 539 | 1.8x |
| | 8 | **1638** | 931 | 1.8x |
| Llama 3.2 1B | 1 | **116** | 102 | 1.1x |
| | 4 | **434** | 375 | 1.2x |
| | 8 | **740** | 688 | 1.1x |
| TinyLlama 1.1B | 1 | **115** | 99 | 1.2x |
| | 4 | **420** | 367 | 1.1x |
| | 8 | 696 | 706 | 1.0x |

**Generating**

| Model | Threads | SmaLLM | llama.cpp | |
|---|---|---|---|---|
| SmolLM2 135M | 1 | **146** | 139 | 1.0x |
| | 4 | **338** | 240 | 1.4x |
| | 8 | **363** | 196 | 1.9x |
| Gemma 3 270M | 1 | 68 | 67 | 1.0x |
| | 4 | **146** | 131 | 1.1x |
| | 8 | **157** | 129 | 1.2x |
| Qwen2.5 0.5B | 1 | 41 | 41 | 1.0x |
| | 4 | **96** | 87 | 1.1x |
| | 8 | **104** | 87 | 1.2x |
| Llama 3.2 1B | 1 | 23 | **24** | 1.0x |
| | 4 | **49** | 46 | 1.1x |
| | 8 | **52** | 47 | 1.1x |
| TinyLlama 1.1B | 1 | 27 | **29** | 0.9x |
| | 4 | **59** | 54 | 1.1x |
| | 8 | **65** | 57 | 1.1x |

Reading a prompt is up to twice as fast. A batch of tokens multiplies
several rows of weights with one instruction, and the rows are unpacked once
per batch. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the details.

Generating is level on one thread and ahead on several. One token at a time,
both engines wait for memory to deliver the weights more than they compute,
so neither can pull far ahead. On the 1B models llama.cpp is about 5 percent
faster on a single thread.

These are numbers from one x86 desktop. The ARM kernels pass the same tests
but have not been timed on a board yet.

How it was measured: a 256-token prompt, then 96 generated tokens, with the
two engines taking turns. Each number is the engine's best run over three
sessions, because the desktop was in use and a busy machine only makes a
run slower. `smallm bench` measures SmaLLM.
[tools/reference](tools/reference/README.md) has the program that measures
llama.cpp.

## Memory

SmaLLM never copies weights out of the model file. The file is mapped into
memory, and the system keeps as much of it there as it has room for. What
the engine holds of its own is small:

| Model | | Model loaded | With a 4,096-token context | After reading 256 tokens |
|---|---|---|---|---|
| SmolLM2 135M | SmaLLM | **1.9 MB** | **2.2 MB** | **15 MB** |
| | llama.cpp | 25 MB | 116 MB | 132 MB |
| Qwen2.5 0.5B | SmaLLM | **6.1 MB** | **6.5 MB** | **15 MB** |
| | llama.cpp | 76 MB | 124 MB | 142 MB |
| Llama 3.2 1B | SmaLLM | **7.5 MB** | **7.7 MB** | **29 MB** |
| | llama.cpp | 498 MB | 627 MB | 658 MB |

This is memory the process allocated, which the system cannot take back.
Both engines also have the model file in memory, 105, 491 and 808 MB for
these three, and the system can drop those pages whenever it needs the room.

With a session ready, SmaLLM holds 19 to 81 times less than llama.cpp. After
reading 256 tokens it is still about 9 to 23 times less.

These numbers are from an x86-64 desktop running Linux, against llama.cpp
b11846. They come from how the engine is built and not from the processor,
so ARM boards should show the same. Those measurements are next.

Two things make the difference. llama.cpp reserves the cache for the whole
context window when a session starts, and SmaLLM grows it as the
conversation grows. llama.cpp also keeps a rearranged copy of most weights
for speed, and SmaLLM rearranges a few rows at a time in a few kilobytes of
scratch memory.

`smallm bench` prints these numbers for any model on Linux, and
`ref mem` in [tools/reference](tools/reference/README.md) does the same for
llama.cpp.

## Size

The whole program, with chat, server and downloader, is 0.47 to 0.70 MB
depending on the platform. The engine as a shared library is 0.29 to
0.43 MB. For scale, the CPU-only libraries of llama.cpp b11846 total 7.7 MB
on the same machine. llama.cpp also does far more, as the to do list below
shows.

Released binaries are built with a standard library that is compiled for
size, which takes a third off. Building from source with stable Rust gives
the same program at about 0.9 MB.

## Context

The context window has no built-in limit. The memory for it grows in steps
of 256 tokens as the conversation grows, so a large window costs nothing
until it is used.

```sh
smallm run model.gguf --ctx 262144
```

A 262,144-token context has been run end to end: 336 MB of history on the
smallest test model, read at 213 tokens per second and continued at 24. What
limits the window in practice is memory, at
`layers x key/value heads x head size x 8` bytes per token. That is 46 KB
for SmolLM2 135M and 66 KB for Llama 3.2 1B. A model also only stays
coherent as far as it was trained to. The default window is the model's own
training length, up to 4,096 tokens.

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
- [x] The Llama, Qwen2, Qwen3 and Gemma 3 architectures
- [x] Fourteen weight formats, float and quantized
- [x] Kernels for AVX2, AVX-VNNI, NEON and the ARM dot product extension
- [x] Prompt reading in tiles, up to twice as fast as llama.cpp
- [x] A few megabytes of memory besides the model file
- [x] SentencePiece and byte-level BPE tokenizers that match llama.cpp token for token on the test models
- [x] Eight chat formats, recognized from the model file
- [x] A context window with no built-in limit, tested at 262,144 tokens
- [x] Chat, one-shot answers, model downloads and benchmarks from one program
- [x] An OpenAI-compatible server with streaming
- [x] A C API as a shared and a static library, and a Python binding
- [x] One-line installers, and static Linux binaries for x86-64, ARM64, ARMv7 and RISC-V
- [x] CI on Linux, macOS and Windows, on x86-64 and ARM, with ARMv7 and RISC-V under emulation
- [ ] Speed and memory measured on ARM boards
- [ ] A first tagged release

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
- [ ] A separate thread count for reading prompts. Efficiency cores help there and not in generation.

**Models**

- [ ] More architectures: Phi, Gemma 2, mixture-of-experts models such as Mixtral, and the families that are not
      transformers (RWKV, Mamba).
- [ ] Encoder models, for embeddings and reranking.
- [ ] Images and audio as input.
- [ ] The remaining weight formats: IQ1, IQ2, IQ3, IQ4_XS and the ternary ones.
- [ ] LoRA adapters and control vectors.

**Generation**

- [ ] Grammar-constrained output, and JSON that follows a schema.
- [ ] Jinja chat templates, for models whose format is not one of the eight built in.
- [ ] More samplers: Mirostat, typical, DRY, XTC and dynamic temperature, and a bias on chosen tokens.
- [ ] Token probabilities in the output.
- [ ] Speculative decoding with a small draft model.
- [ ] Saving a session's context to a file and loading it again.
- [ ] Sliding the context window when it fills, instead of dropping whole turns.
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

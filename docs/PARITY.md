# Parity with llama.cpp and Ollama

Where SmaLLM stands against the engines it is measured against, area by
area. This page tracks the work toward parity, and [TODO.md](TODO.md) turns
its gaps into a list. Update it when a gap closes, and when the competitors
move.

State as of SmaLLM 0.3.0 (2026-11-24).

- **llama.cpp:** counted from the source of b11846 (see "Keeping this page
  current" below).
- **Ollama:** from what is known of its features, not checked against its
  source. Treat that column as approximate.

## Measured

x86 CPU, Intel Core i7-14700K, 8 threads, llama.cpp b11846 with its
`alderlake` backend, engines taking turns, best of two runs.

| | SmaLLM | llama.cpp | Ollama |
|---|---|---|---|
| Binary size | 0.82 to 1.26 MB, no dependencies | 7.7 MB of CPU libraries | Much larger, bundles GPU runtimes |
| Anonymous memory, Llama 3.2 1B, 4K context | 21 MB | 636 MB | Not measured (built on llama.cpp) |
| Anonymous memory, gpt-oss-20b | 33 MB | 9,966 MB | Not measured |
| Generation speed | Faster on 9 of 9 models, by 5 to 33% | 5% faster on one thread (Llama 3.2 1B) | Not measured |
| Prompt speed | Faster on 7 of 9, tie on RWKV-7, DeepSeek 0.83x | | Not measured |
| ARM speed | Unmeasured | Mature NEON, i8mm and SVE kernels | llama.cpp's |

Speed by model, tokens per second, 256-token prompt, 64 generated:

| Model | Prompt: SmaLLM | Prompt: llama.cpp | Generation: SmaLLM | Generation: llama.cpp |
|---|---|---|---|---|
| Llama 3.2 1B | 798 | 683 | 51.7 | 46.6 |
| Qwen3 0.6B | 1405 | 1296 | 97.8 | 78.5 |
| Gemma 2 2B | 359 | 303 | 23.9 | 21.8 |
| Phi-4-mini 3.8B | 228 | 199 | 16.5 | 15.7 |
| gpt-oss-20b (MXFP4) | 130 | 104 | 16.2 | 15.4 |
| DeepSeek-Coder-V2-Lite | 132 | 158 | 24.6 | 21.6 |
| Granite 4.0 H 1B | 453 | 418 | 41.2 | 30.9 |
| Mamba 2.8B | 199 | 135 | 24.3 | 19.8 |
| RWKV-7 1.5B | 388 | 393 | 43.5 | 33.1 |

Anonymous memory (RssAnon) with a 4,096-token context and 256 tokens read:

| Model | SmaLLM | llama.cpp |
|---|---|---|
| Llama 3.2 1B | 21 MB | 658 MB |
| Qwen3 0.6B | 39 MB | 723 MB |
| Gemma 2 2B | 39 MB | 1,403 MB |
| Phi-4-mini 3.8B | 47 MB | 1,831 MB |
| gpt-oss-20b | 33 MB | 9,966 MB |
| DeepSeek-Coder-V2-Lite | 28 MB | 6,699 MB |
| Granite 4.0 H 1B | 130 MB | 787 MB |
| Mamba 2.8B | 88 MB | 1,450 MB |
| RWKV-7 1.5B | 38 MB | 723 MB |

These figures leave out the mapped model file, which both engines need in
memory at full speed. The README's Memory section has the trade-offs.

## Hardware

| | SmaLLM | llama.cpp | Ollama |
|---|---|---|---|
| x86 CPU | AVX2, AVX-VNNI | AVX2, AVX-512, AMX and more | Yes (llama.cpp) |
| ARM64 CPU | NEON, dot product | NEON, dot product, i8mm, SVE, KleidiAI | Yes (llama.cpp) |
| ARMv7, RISC-V | Portable code | Vector kernels for RISC-V | No builds |
| GPU | None | CUDA, HIP, Metal, Vulkan, SYCL, OpenCL, MUSA, WebGPU | CUDA, ROCm, Metal, Vulkan |
| NPU | None | Hexagon, OpenVINO, CANN | None |
| Split across machines | No | Yes (RPC) | No |

## Models and formats

| | SmaLLM | llama.cpp | Ollama |
|---|---|---|---|
| Architectures | 25 decoders and 2 encoders | 164, including vision, audio and TTS | Most popular families |
| Missing from SmaLLM | Llama 4, Qwen3.5, Qwen3-Next, GLM-4.x, Gemma 3n, Mistral 3, OLMo 2, Command-R, ERNIE 4.5, MiniMax, Jamba, Falcon-H1, BitNet, T5, StarCoder2 and more | Has them | Has most |
| Image and audio input | No | Yes | Images |
| Weight formats | 25: every common one, bit-exact with ggml | Same, plus NVFP4, Q1_0, Q2_0 | Same as llama.cpp |
| Tokenizers | SentencePiece, BPE (67 families), WordPiece, Unigram, RWKV | Same, plus PLaMo 2 | Same |
| Embeddings | BERT, nomic-bert | Many, including decoder-based embedding models | Yes |
| Reranking | Yes | Yes | No |

## Generation

| | SmaLLM | llama.cpp | Ollama |
|---|---|---|---|
| Chat formats | 15 built in, no Jinja | 54 built in, plus a Jinja engine that runs any template | Go templates per model |
| Samplers | Temperature, top-k, top-p, min-p, repetition penalty | Also typical, XTC, DRY, Mirostat, dynamic temperature, top-n-sigma, logit bias | Most of llama.cpp's |
| Grammar and JSON schema | No | GBNF and JSON schema | JSON schema |
| Tool calls | No | Yes | Yes |
| Reasoning kept separate | Yes | Yes | Yes |
| Speculative decoding | No | Draft model, EAGLE3, lookup | No |
| LoRA, control vectors | No | Yes | LoRA |
| Key/value cache | 16- or 32-bit floats, allocated on demand | Down to 4-bit, reserved up front | 16, 8 or 4 bits |
| Save and restore a session | No | Yes | No |
| Full context | Keeps the start or slides, keys rotated | Context shift | Truncation or shift |
| Token probabilities | No | Yes | Unknown |

## Server and tools

| | SmaLLM | llama.cpp | Ollama |
|---|---|---|---|
| OpenAI chat and completions, streaming | Yes | Yes | Yes |
| Other APIs | Embeddings, rerank | Plus Anthropic Messages, OpenAI Responses, infill, transcription, TTS | Native Ollama API |
| Parallel requests | No, requests queue | Yes (slots, continuous batching) | Yes |
| Several models, loaded on demand | No | Yes | Yes |
| Web UI, metrics | No | Both | Desktop app |
| Model download | Hugging Face, checksummed | Hugging Face | Own registry and Hugging Face |
| Convert and quantize | No | Yes, with importance matrices | Quantize on create |
| Perplexity and evaluation | No | Yes | No |
| Benchmark | `bench`, with a per-stage profile | `llama-bench` | Timings only |
| C API | Yes | Yes | No, HTTP only |
| Bindings | Python | Many, mostly community | HTTP clients |

## Summary

**Ahead:** memory (6 to 300 times less anonymous memory), size (about a
sixth of llama.cpp's libraries), and CPU speed on x86.

**At parity:** CPU inference for the most-used model families, every
common weight format, tokenizers, the basic OpenAI API, embeddings and
reranking, reasoning output, and full-context handling.

**Behind, ordered by how much each blocks users:**

1. No GPU or NPU backend. It decides speed on any machine with a GPU.
2. No structured output or tool calls. Most agent and app integrations need
   them.
3. About a sixth of llama.cpp's architecture coverage, and no image input.
   Llama 4, Qwen3.5, GLM-4.x, Gemma 3n and Mistral 3 matter most.
4. No Jinja templates. A model whose chat format is not one of the 15 has
   no chat.
5. The server handles one request at a time, and one model.
6. Samplers, token probabilities, speculative decoding, LoRA, session
   files, and converting or quantizing models.

**Unknown:** speed on ARM boards, the actual target. See
[DEVELOPMENT.md](DEVELOPMENT.md) for what to measure first.

## Keeping this page current

- **SmaLLM:** tick [TODO.md](TODO.md) and change this page in the same
  commit when a gap closes.
- **Speed and memory:** `smallm bench <model> -p 256 -n 64 -r 1 -t 8`
  against `ref bench <model> 256 64 8`, and
  `smallm bench <model> -c 4096 -p 240 -n 16 -r 1 -t 4` against
  `ref mem <model> 4096 256 4`, the engines taking turns.
  [tools/reference/README.md](../tools/reference/README.md) builds `ref`.
- **llama.cpp:** count from its source when updating the reference version:

  ```sh
  sed -n '/LLM_ARCH_NAMES = {/,/^};/p' src/llama-arch.cpp | grep -c '{ LLM_ARCH_'
  sed -n '/LLM_CHAT_TEMPLATES = {/,/^};/p' src/llama-chat.cpp | grep -c '{ "'
  ls -d ggml/src/ggml-*/                     # backends
  grep -o 'llama_sampler_init_[a-z0-9_]*' include/llama.h | sort -u
  ```

- **Ollama:** check its release notes. Mark anything unconfirmed as
  approximate.

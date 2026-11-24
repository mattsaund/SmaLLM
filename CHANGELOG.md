# Changelog

Each version is a `v` tag, with ready-made binaries and libraries on its
[release page](https://github.com/mattsaund/SmaLLM/releases).

## Unreleased

- **Architectures.** `gemma4`, `lfm2` and `lfm3`. Layers can now differ in
  how they mix positions, in their head count and width, in rotation and
  window, and in feed-forward width. Layers that reuse an earlier layer's
  keys and values (Gemma 4) cost no cache memory, and after the last layer
  that keeps state, only the token whose logits are wanted is computed.
- **Hybrid models.** Short convolution blocks, as in LFM2 and LFM3, keep a
  few kilobytes of state. A rollback of up to eight tokens costs nothing,
  and a longer one goes back to a state saved every 256 tokens.
- **Tokenizers.** The word splitting rules of all 67 BPE families llama.cpp
  knows, run by a small regular expression engine, and Gemma 4's tokenizer.
  End tokens are found as llama.cpp finds them.
- **Chat formats.** Gemma 4's.
- **Readouts.** Speed and context use after each answer in a terminal, and
  in the server's responses as llama.cpp's `timings`.
- **Downloads.** One progress bar under a log of every step, with resumed
  downloads checked against Hugging Face's SHA-256. The installers show the
  same.

## 0.1.0 - 2026-11-23

The first version.

- **The engine.** Loads GGUF models by memory mapping and runs them on the
  CPU. No dependencies outside the Rust standard library.
- **Architectures.** `llama`, `qwen2`, `qwen3` and `gemma3`, which covers
  Llama 2 and 3, TinyLlama, SmolLM2, Mistral, Qwen2.5, Qwen3 and Gemma 3.
- **Weight formats.** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K,
  Q3_K, Q4_K, Q5_K, Q6_K and IQ4_NL.
- **Kernels.** AVX2 on x86-64, with AVX-VNNI where the CPU has it, and
  NEON on AArch64, with the dot product extension where the CPU has it. A
  portable fallback runs everywhere else.
- **Prompt reading in tiles.** A batch of tokens multiplies several weight
  rows at once, unpacked once per batch and interleaved so that one SIMD
  multiply serves all of them.
- **Memory.** Weights are never copied out of the model file. With a model
  loaded and a 4,096-token context, the engine holds 2 to 8 MB of its own.
- **Tokenizers.** SentencePiece and byte-level BPE, with the Llama 3, Qwen,
  SmolLM and GPT-2 word splitting rules.
- **Context.** The key/value cache grows in chunks as the context fills, so
  the window has no built-in limit. Tested with 262,144 tokens.
- **The program.** `smallm run`, `serve`, `pull`, `list`, `rm`, `info`,
  `bench` and `tokenize`.
- **The server.** An OpenAI-compatible API with streaming, and the
  tokenize and detokenize endpoints of llama.cpp's server.
- **Sampling.** Temperature, top-k, top-p, min-p, and repetition, frequency
  and presence penalties.
- **Libraries.** A C API as a shared and a static library, with a header,
  and a Python binding built on it.

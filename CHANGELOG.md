# Changelog

Each version is a `v` tag, with ready-made binaries and libraries on its
[release page](https://github.com/mattsaund/SmaLLM/releases).

## 0.3.0 - 2026-11-24

- **Prompt threads.** Prompts are read on every physical core, efficiency
  cores included, while generation stays on the performance cores.
  `--threads-batch` (and the `threads-batch` setting) chooses the count.
  Prompts read about 28 percent faster on an Intel Core i7-14700K.
- **Embeddings and reranking.** BERT and nomic-bert models run with
  `smallm embed` and `smallm rerank`, and the server answers
  `/v1/embeddings` and `/v1/rerank`. Their vectors and scores match
  llama.cpp's (cosine similarity 0.9999996 or better).
- **Encoder tokenizers.** WordPiece (BERT) and Unigram (T5, XLM-RoBERTa),
  both matching llama.cpp token for token on its test texts.
- **Weight formats.** The i-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S and IQ4_XS, the ternary TQ1_0 and TQ2_0, and MXFP4. Each
  expands to exactly ggml's numbers, which a test checks for every format.
- **Gemma 2 and Phi.** `gemma2` caps attention scores and windows every
  other layer. `phi3` covers Phi-3, Phi-3.5 and Phi-4-mini, with LongRoPE's
  two sets of rotary divisors chosen by the context length. `phi2` covers
  Phi-2 and Phi-1.5: LayerNorm, and attention and the feed-forward network
  side by side.
- **gpt-oss.** Attention sinks, biased experts, OpenAI's clamped SwiGLU and
  the Harmony chat format. MXFP4, IQ4_NL and IQ4_XS got AVX2 and NEON
  kernels, so gpt-oss-20b generates about as fast as llama.cpp and reads
  prompts faster.
- **DeepSeek.** `deepseek2` (DeepSeek V2, V2-Lite, V3, R1, Moonlight, Kimi)
  with latent attention, whose cache holds one compressed row per position:
  4.4 times smaller than llama.cpp's for older files. V3's routing by
  groups of experts. `deepseek` for DeepSeek-MoE-16B. The DeepSeek V2 and
  Kimi chat formats.
- **Mamba.** `mamba` (with Falcon Mamba), `mamba2`, and `granitehybrid`
  (IBM Granite 4.0 H, Mamba 2 blocks with attention and experts). Mamba
  2.8B holds 88 MB of its own memory, where llama.cpp holds 1,436 MB.
- **RWKV.** `rwkv6` and `rwkv7`, with the RWKV World tokenizer and chat
  format. RWKV-7 1.5B holds 38 MB of its own memory against llama.cpp's
  723 MB.
- **Recurrent states** of Mamba and RWKV blocks are saved just before the
  last token of each prompt, so the next turn of a chat, or an answer
  asked for again, does not read the conversation again.
- **Half the cache memory.** The key/value cache keeps 16-bit floats, as
  llama.cpp does. A full 4,096-token context of Llama 3.2 1B now holds
  176 MB of SmaLLM's own memory instead of 310 MB (llama.cpp: 636 MB), and
  generation there is about 14 percent faster. `--cache-type f32` (and the
  `cache-type` setting, and `cache_type` in the C API) keeps 32-bit floats.
- **YaRN** position scaling, which gpt-oss and DeepSeek use and some Qwen
  files ask for.
- **Docs.** The contributor guide moved to `docs/CONTRIBUTING.md`, next to
  a new `docs/DEVELOPMENT.md` on running and measuring SmaLLM on
  single-board computers. `.gitignore` covers key and environment files.
- **Reasoning.** What thinking models write before their answer (between
  `<think>` tags, or on Harmony's analysis channel) is shown faint in a
  chat and kept out of the conversation. The server returns it as
  `reasoning_content`, as llama.cpp does.

## 0.2.0 - 2026-11-24

- **Architectures.** Fourteen in all. New: `gemma4`, `lfm2`, `lfm3`,
  `granite`, `olmoe`, `smollm3`, and the mixture-of-experts `qwen2moe`,
  `qwen3moe`, `granitemoe` and `lfm2moe`, as well as Mixtral, which is a
  `llama` with experts. Layers can differ in how they mix positions, in
  their head count and width, rotation, window and feed-forward width.
  Layers that reuse an earlier layer's keys and values (Gemma 4) cost no
  cache memory, and after the last layer that keeps state, only the token
  whose logits are wanted is computed.
- **Mixtures of experts.** Every expert in use runs in one multiply over
  the tokens routed to it, so a generated token wakes the threads twice
  per layer, however many experts it uses.
- **Hybrid models.** Short convolution blocks, as in LFM2 and LFM3, keep a
  few kilobytes of state. A rollback of up to eight tokens costs nothing,
  and a longer one goes back to a state saved every 256 tokens.
- **A full context.** `--overflow middle` keeps the start (the system prompt
  and first question) and drops the oldest tokens after it, `slide` drops
  the oldest tokens, and `stop` ends the answer. The kept tokens are not
  read again: their cached keys are rotated to their new positions.
- **Settings.** One file, `smallm.conf`, holds every setting. `smallm config`
  shows it, and `smallm config --reset` restores the defaults.
- **Uninstall.** `smallm uninstall` removes the program, its models and its
  settings, and the installers' uninstall options use it.
- **Installers.** They install what SmaLLM needs, after asking: curl for
  downloads, and Rust and a C compiler for a build from source. Without
  git, the source comes as an archive.
- **Tokenizers.** The word splitting rules of all 67 BPE families llama.cpp
  knows, run by a small regular expression engine, and Gemma 4's tokenizer.
  End tokens are found as llama.cpp finds them.
- **Chat formats.** Gemma 4's and Granite's.
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

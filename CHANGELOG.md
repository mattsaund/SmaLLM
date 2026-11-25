# Changelog

Each version is a `v` tag, with ready-made binaries and libraries on its
[release page](https://github.com/mattsaund/SmaLLM/releases).

## Unreleased

- **Faster on ARM chips with the dot product extension** (Cortex-A55, A76
  and later, such as the Raspberry Pi 5 and the RK3588, Apple's M chips and
  AWS Graviton). Prompts: the tile kernel multiplies four vectors at once
  with `sdot` by lane, tiles keep their weights signed so zero points cost
  nothing, formats with a scale per 32 weights sum 32 columns before they
  scale, and tile fills store with `st4`. Generation: the row kernels of the 32-weight formats (Q4_0, Q8_0,
  Q5_0, IQ4_NL, MXFP4 and others) take four blocks at a time, and Q6_K's
  merges its bits with shifts that insert. On an Allwinner A733's two
  Cortex-A76 cores, LFM3 1.7B (Q4_0) reads a 128-token prompt at 39.1
  tokens per second (12.8 before, llama.cpp 44.5) and generates at 8.6 (7.5
  before, llama.cpp 8.5, both at the board's memory limit). On all 8 cores
  it reads that prompt at 47.4. Nemotron 3.5 Bolt E4B (Q4_K_M) reads
  prompts at 14.3 (8.9 before, llama.cpp 8.5).
- **A prompt batch of 64 on ARM**, as on x86. It was 32. With the faster
  kernels, 64 reads prompts 7 percent faster on the A733, for about 5 MB
  more scratch memory.
- **Prompts read ahead of the multiply.** While a tile of rows meets the
  batch, the next tile's rows are already being fetched from memory. On x86
  Llama 3.2 1B (Q4_K_M) reads a 256-token prompt about 6 percent faster (876
  against 828 tokens per second on 8 cores), and Gemma 4 E2B (Q4_0) is
  unchanged.
- **Thread counts on ARM boards.** Physical cores are now told apart by the
  CPUs that share them. Boards whose clusters number their cores from zero
  (the Allwinner A733) were counted 6 cores short of 8, and read prompts
  about 15 percent slower. Thread counts also follow the CPU affinity mask
  now, so `taskset` limits them.
- **`smallm rm` asks first**, as `uninstall` does. `--yes` skips the
  question. `smallm list --help` and `smallm rm --help` work.
- **Sampling defaults from the model file.** The settings a model file
  recommends (`general.sampling.*`: temperature, top-k, top-p, min-p, the
  repetition penalty and its window, XTC and Mirostat) are the defaults,
  as llama.cpp reads them. `smallm.conf` and the command line still win,
  and the file's sampling settings default to `auto`. A settings file
  written by 0.3.0 or earlier spells out the old defaults (`temp = 0.8`,
  `batch = 32` and the rest). Those values now read as `auto`, so they do
  not hide the model's settings. Values that were changed are kept.
- **Six more samplers**, in llama.cpp's order and with its option names:
  typical (`--typical`), top-n-sigma (`--top-nsigma`), XTC
  (`--xtc-probability`, `--xtc-threshold`), dynamic temperature
  (`--dynatemp-range`, `--dynatemp-exp`), Mirostat 1 and 2 (`--mirostat`,
  `--mirostat-ent`, `--mirostat-lr`) and logit bias
  (`--logit-bias TOKEN+BIAS`, repeatable). The server reads them from
  requests: `typical_p`, `top_n_sigma`, `xtc_probability`,
  `xtc_threshold`, `dynatemp_range`, `dynatemp_exponent`, `mirostat`,
  `mirostat_tau`, `mirostat_eta`, and `logit_bias` as OpenAI's object or
  llama.cpp's list of pairs.
- **`smallm serve` takes the sampling options** as defaults for requests
  that do not give their own.
- **A switch for thinking.** `--think off` makes Qwen3, LFM3, SmolLM3 and
  RWKV-7 G1 answer without reasoning, written the way each model's own
  template writes it. `--think on` turns it on for Gemma 4, whose reasoning
  is now kept apart from its answer. gpt-oss always reasons and takes
  `--think low`, `medium` or `high`. The server takes
  `chat_template_kwargs.enable_thinking` and `reasoning_effort` per
  request, and `serve --think` sets its default.
- **Larger prompt batches on x86 and for models with experts.** Prompts
  are read 64 tokens at a time on x86 (32 before), and 128 for models with
  experts on every machine, whose batch is split among the experts. On an
  Intel Core i7-14700K, dense models read prompts up to 11 percent faster
  for 2 to 6 MB more scratch memory. The expert models read them about 44
  percent faster: DeepSeek-Coder-V2-Lite at 189 tokens per second against
  llama.cpp's 162 (0.83 times llama.cpp before), for 44 MB. ARM keeps 32,
  which was as fast as 64 on the Allwinner A733 and took 5 to 6 MB less.
- **`--cpus`** runs on chosen CPUs, such as one cluster of a board, and the
  thread counts follow them. Linux and Android. The `cpus` setting does the
  same.
- **`--mlock`** reads the whole model into memory at load and keeps it
  there, for boards whose storage is too slow to read pages back. It needs
  a locked-memory limit (`ulimit -l`) as large as the file.
- `--max-tokens` says in its help that it counts reasoning tokens.
- **Inkling 2 and Inkling 2 Mini** (Thinking Machines), the `inkling2` and
  `inkling2_loop` architectures. Their blocks attend over the outputs of
  earlier blocks instead of reading one residual stream. Attention has a
  learned bias on distances instead of rotation, and short convolutions run
  over the keys, the values and both block outputs. Inkling 2 Mini runs
  layers 6 to 17 three times, each pass with its own cache. Its tokenizer
  rule, its chat format (`inkling`, with the thinking effort set by
  `--think`), and its reasoning markers come with it. Greedy output
  matches llama.cpp b12711, and logits agree at a cosine similarity of
  0.9997 (0.99994 on a K-quant copy). With the 9.1 GB UD-Q2_K_XL file,
  SmaLLM reads prompts 15 percent faster and generates 13 percent faster
  than llama.cpp, and holds 154 MB of anonymous memory after 256 tokens
  where llama.cpp holds 1.45 GB.
- **NVFP4, Q1_0 and Q2_0 weights**, which Inkling 2 Mini's experts use,
  each expanding to exactly ggml's numbers. NVFP4 and Q2_0 have AVX2 and
  NEON kernels and run at Q4_0's speed on x86. Q1_0 runs portable code.
  Expert tensors may also carry one scale per expert (`.scale`), which
  conversions from NVFP4 keep apart from the weights.
- **Saved convolution states stay under 16 MB together**, the oldest
  dropped first. They grew with the context before, by 3.8 MB every 256
  positions for Inkling 2 Mini. A cut before the oldest saved state reads
  the context again from its start.
- **Word splitting** for the four families llama.cpp added since 0.3.0's
  reference: `inkling2`, `tekken-num3`, `soprano` and `zonos3` (71 in all).

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

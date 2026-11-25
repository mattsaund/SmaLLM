# To do

What SmaLLM has and what it lacks. The open items are features llama.cpp,
Ollama or other engines have today, and the gaps the measurements show.
[PARITY.md](PARITY.md) has the comparison they come from.

Finished items are ticked, not deleted. When one lands, tick it here,
update PARITY.md, and add it to the [CHANGELOG](../CHANGELOG.md), in the
same commit. Heavy items are planned as optional parts of the build, so the
default binary stays small.

## Done

- [x] GGUF loading by memory map, with every read bounds-checked
- [x] Twenty-five architectures, including mixtures of experts, hybrids, and the state-space and RWKV families
- [x] Twenty-five weight formats, float, quantized, i-quants, ternary and MXFP4, each expanding exactly as in ggml
- [x] Kernels for AVX2, AVX-VNNI, NEON and the ARM dot product extension
- [x] Prompt reading in packed tiles
- [x] Anonymous memory in megabytes, with weights read in place from the mapped file
- [x] A half-precision key/value cache, allocated as the context fills
- [x] SentencePiece and byte-level BPE tokenizers for every word splitting family llama.cpp knows, RWKV's, and
      the encoder tokenizers WordPiece and Unigram, matching it token for token on its test texts
- [x] Fifteen chat formats, recognized from the model file
- [x] Embedding and reranking models, from the command line and the server
- [x] Reasoning output separated from the answer in the chat, in pipes and in the server
- [x] A context window that slides or keeps its start when it fills, chosen with `--overflow`
- [x] A context window with no built-in limit, tested at 262,144 tokens
- [x] A separate thread count for reading prompts, every core by default
- [x] One settings file for everything, with a reset to the defaults
- [x] Chat, one-shot answers, model downloads and benchmarks from one program
- [x] An OpenAI-compatible server with streaming
- [x] A C API as a shared and a static library, and a Python binding
- [x] One-line installers that bring what SmaLLM needs, a full uninstall, and static Linux binaries for x86-64,
      ARM64, ARMv7 and RISC-V
- [x] CI on Linux, macOS and Windows, on x86-64 and ARM, with ARMv7 and RISC-V under emulation
- [x] Tagged releases, the latest 0.3.0

## Fixes from board testing

Bugs the first board found (the Allwinner A733, 2026-11-24).

- [x] Physical cores counted wrong on boards whose clusters number their cores from zero, which cost 15 percent on
      prompts. Cores are now named by the CPUs that share them.
- [x] Thread counts ignored the CPU affinity mask, so `taskset` did not limit them.
- [x] `smallm list --help` and `smallm rm --help` answered "unknown option".
- [x] `smallm rm` deleted without asking. It asks now, and `--yes` skips the question.
- [x] `docs/ARCHITECTURE.md` listed fourteen architectures, ten chat formats and no embeddings.

## Measurement

- [x] Speed and memory measured on a first ARM board, the Allwinner A733. Results in
      [DEVELOPMENT.md](DEVELOPMENT.md).
- [x] Thread counts right on boards whose clusters number their cores from zero, and within the CPU affinity mask.
- [ ] Llama 3.2 1B measured on a board, for the README's tables.
- [ ] More boards: the RK3588 and the Raspberry Pi 5.
- [ ] The model-by-model comparison in PARITY.md repeated on ARM, against llama.cpp built on the board.

## Hardware

- [ ] GPUs, through Vulkan compute. It is the one API that reaches AMD, Intel and NVIDIA cards and the ARM GPUs in
      phones and boards, so one backend covers them. The engine runs on the CPU only today. Board GPUs need 32-bit
      storage fallbacks and small shared-memory tiles, and some (the A733's PowerVR) have less bandwidth than the
      CPU, so they would not speed up generation.
- [ ] Apple GPUs through Metal, and NVIDIA through CUDA, where they outperform Vulkan.
- [ ] OpenCL, for the mobile GPUs llama.cpp reaches that way (Qualcomm Adreno).
- [ ] NPUs. Every vendor has its own toolkit (Rockchip RKNN, Qualcomm QNN and Hexagon, Intel OpenVINO, Huawei CANN,
      Apple Core ML), so each one is a backend of its own. None exists yet.
- [ ] TPUs. Google's Coral Edge TPU cannot hold a language model: it runs small, fully 8-bit TensorFlow Lite graphs
      and keeps about 8 MB of weights. Support waits for hardware that can.
- [ ] AVX-512 kernels, and AMX on Intel Xeon.
- [ ] ARM i8mm (8-bit matrix multiply) and SVE kernels, which llama.cpp uses on newer ARM cores.
- [ ] NEON kernels for 32-bit ARM, and vector kernels for RISC-V.
- [ ] NEON kernels for the formats that have none: IQ1, IQ2, IQ3, TQ1_0, TQ2_0, Q2_K and Q3_K.
- [ ] A model split across machines over the network, as llama.cpp's RPC backend does.
- [ ] `--cpus`, to run on a chosen set of CPUs (one cluster) without `taskset`, as llama.cpp's `--cpu-range` does.
- [ ] `--mlock`, opt-in, to keep the weights resident on boards with slow storage. On an SD card, pages dropped
      under memory pressure are slow to read again (6.2 GB takes about 95 seconds at 65 MB/s).
- [ ] Reading ahead at load (`MADV_WILLNEED`). Low value: at most 15 percent off a cold start from an SD card.

## Models

- [x] Mixture-of-experts models: Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, Granite MoE and LFM2-MoE.
- [x] Hybrids of attention and short convolutions: LFM2, LFM2.5 and LFM3.
- [x] More architectures: Phi, Gemma 2, the MoE models of DeepSeek and gpt-oss, and the families that are not
      transformers (RWKV, Mamba).
- [x] Encoder tokenizers (WordPiece and Unigram), which only the encoder models use.
- [x] Encoder models, for embeddings and reranking.
- [x] The remaining weight formats: IQ1, IQ2, IQ3, IQ4_XS, the ternary ones, and MXFP4.
- [ ] Current architectures in wide use: Llama 4, Qwen3.5 and Qwen3.5-MoE, Qwen3-Next, GLM-4 and its MoE
      (GLM-4.5, 4.6), Gemma 3n, Mistral 3, OLMo 2, Command-R and Cohere 2, ERNIE 4.5, MiniMax M2, EXAONE 4,
      Hunyuan, InternLM2, MiniCPM, Phi-MoE, Kimi Linear and DeepSeek V3.2.
- [ ] The other hybrids and distillations: Falcon-H1, Jamba, Nemotron-H, and RWKV's distillations of Qwen
      (rwkv6qwen2, arwkv7).
- [ ] Older families still in circulation: GPT-2, GPT-J, GPT-NeoX (Pythia), Falcon, StableLM, StarCoder2, Bloom
      and MPT. Phi-2's LayerNorm and parallel blocks cover part of what they need.
- [ ] BitNet. The ternary formats are in, the architecture is not.
- [ ] T5 and other encoder-decoder models.
- [ ] Decoder-based embedding models (Qwen3-Embedding, Gemma embedding), and more encoders (ModernBERT, Jina BERT
      v2 and v3, NeoBERT, EuroBERT).
- [ ] Images and audio as input.
- [ ] Text to speech models (OuteTTS, Qwen3-TTS).
- [ ] LoRA adapters and control vectors.
- [ ] The weight formats NVFP4, Q1_0 and Q2_0.
- [ ] PLaMo 2's tokenizer.
- [ ] Models split across several GGUF files (`-00001-of-00004.gguf`), the way large models ship. The downloader
      skips them today, and the loader reads one file.

## Generation

- [x] Sliding the context window when it fills, or keeping its start and dropping the middle.
- [x] A half-precision key/value cache, to halve the memory of long contexts.
- [ ] Grammar-constrained output (GBNF), and JSON that follows a schema.
- [ ] Jinja chat templates, for models whose format is not one of the fifteen built in.
- [ ] More samplers: Mirostat, typical, DRY, XTC, top-n-sigma and dynamic temperature, presence and frequency
      penalties, and a bias on chosen tokens.
- [ ] Token probabilities in the output.
- [ ] Speculative decoding: a small draft model, EAGLE3, and prompt lookup.
- [ ] Saving a session's context to a file and loading it again.
- [ ] An 8-bit and a 4-bit key/value cache, for long contexts. High value on boards.
- [ ] Freeing the cache of sliding-window layers past their window (Gemma 2, Gemma 3, Gemma 4, gpt-oss), as
      llama.cpp does. High value on boards: Gemma 4 and Nemotron 3.5 Bolt keep 512-position windows in 35 of
      their 42 layers.
- [ ] Sampling defaults from the model file (`general.sampling.*`: temperature, top-k, top-p, min-p, repetition
      penalty and its window, and the rest llama.cpp reads), below `smallm.conf` and the command line. Nemotron 3.5
      Bolt E4B's file asks for temperature 1.0 and top-k 64, where SmaLLM uses 0.8 and 40.
- [ ] A switch for thinking: `run --think on|off`, and `chat_template_kwargs.enable_thinking` and
      `reasoning_effort` in the server, for the formats that support it. Qwen3 turns it off with an empty block
      after the assistant's opening (`<think>\n\n</think>\n\n`), LFM3 with `<think></think>`, SmolLM3 with
      `/no_think` in the system message, and gpt-oss sets its level on the Harmony `Reasoning:` line, fixed at
      medium today. On a board, a thinking model can spend tens of seconds before its answer.
- [ ] `--max-tokens` counting reasoning and answer tokens apart, so that a small limit does not end before the
      answer begins.

## Speed

Gaps the measurements in PARITY.md and DEVELOPMENT.md show.

- [ ] Q4_0 prompt reading on NEON with the dot product extension: 3.6 times slower than llama.cpp on the A733's
      A76 cores (12.5 against 44.5 tokens per second). The matrix multiplies take 96 percent of the time, and the
      batch size is not the cause. Rework the NEON tile kernel's register blocking (4 rows by 4 tokens of
      accumulators, `sdot` by lane, as llama.cpp's Q4_0 GEMM does), or add an opt-in repack at load for boards
      with memory to spare.
- [ ] Q4_0 generation on NEON, 7 percent behind llama.cpp on the A733 (7.4 against 8.0 tokens per second). Both
      are memory-bound, at 7.7 against 8.3 GB/s, so the matrix-vector kernel has some headroom. Q4_K_M is 8 percent
      ahead.
- [ ] The default prompt batch on ARM: 64 read 6 percent faster than 32 on the A733. Measure the scratch memory
      first.

- [ ] Prompt reading for mixtures of experts. DeepSeek reads prompts at 0.83x llama.cpp with the default batch of
      32 tokens, which leaves each of 64 experts about 3 tokens. A batch of 256 reads them 1.6 times as fast as
      llama.cpp. A larger default for expert models costs scratch memory.
- [ ] Generation on one thread, about 5 percent behind llama.cpp on Llama 3.2 1B.
- [ ] RWKV-7 prompt reading, level with llama.cpp. Its low-rank maps take about 13 percent of single-threaded
      prompt time, because the float route handles their short rows one dot product at a time.
- [ ] Long contexts with latent attention (DeepSeek). Each unit of work reads a layer's whole latent cache, and
      attention takes 29 percent of generation time at 3,000 tokens. Splitting by position, with partial softmaxes
      merged at the end, would fix it.
- [ ] Mamba 1's scan on x86: an AVX2 exponential would halve its share of the prompt time.

## Server

- [x] Endpoints for embeddings and reranking.
- [ ] Several requests at once.
- [ ] Tool calls.
- [ ] An endpoint for code infill.
- [ ] The Anthropic Messages API and the OpenAI Responses API.
- [ ] Speech endpoints (transcription and text to speech), with the models above.
- [ ] The Ollama API next to the OpenAI one, and several models loaded and unloaded on demand.
- [ ] Sampling options on `serve`, as defaults for requests. Today two servers can only differ through each
      request or the shared `smallm.conf`.
- [ ] A built-in web page, and metrics.

## Tools

- [ ] Converting and quantizing models, with importance matrices. Today the GGUF file comes from elsewhere.
- [ ] Perplexity and the other evaluation tools.
- [ ] Bindings beyond Python, starting with Node.js and Go.
- [ ] A `no_std` core, for boards without an operating system.

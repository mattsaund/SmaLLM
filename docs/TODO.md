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
- [x] Twenty-seven architectures, including mixtures of experts, hybrids, the state-space and RWKV families, and
      Inkling 2 with its loop
- [x] Twenty-eight weight formats, float, quantized, i-quants, ternary, MXFP4, NVFP4, Q1_0 and Q2_0, each expanding
      exactly as in ggml
- [x] Kernels for AVX2, AVX-VNNI, NEON and the ARM dot product extension
- [x] Prompt reading in packed tiles
- [x] Anonymous memory in megabytes, with weights read in place from the mapped file
- [x] A half-precision key/value cache, allocated as the context fills
- [x] SentencePiece and byte-level BPE tokenizers for every word splitting family llama.cpp knows, RWKV's, and
      the encoder tokenizers WordPiece and Unigram, matching it token for token on its test texts
- [x] Sixteen chat formats, recognized from the model file
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
- [x] Tagged releases, the latest 0.4.0

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
- [ ] Round the scales of 32-element activation groups to 16 bits, as ggml does. SmaLLM keeps them in 32 bits, which
      leaves its logits a little apart from llama.cpp's on the formats that use these groups (Q4_0, Q8_0, MXFP4,
      NVFP4, Q2_0). Rounding them moved Inkling 2 Mini's UD-Q2_K_XL logits from a cosine similarity of 0.9997 to
      0.99994 against llama.cpp b12711. It changes every such model's numbers a little, so it waits for a decision.

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
- [ ] NEON kernels for the formats that have none: IQ1, IQ2, IQ3, TQ1_0, TQ2_0, Q1_0, Q2_K and Q3_K. AVX2 kernels for
      Q1_0.
- [ ] A model split across machines over the network, as llama.cpp's RPC backend does.
- [x] `--cpus`, to run on a chosen set of CPUs (one cluster) without `taskset`, as llama.cpp's `--cpu-range` does.
      Linux and Android.
- [x] `--mlock`, opt-in, to keep the weights resident on boards with slow storage. On an SD card, pages dropped
      under memory pressure are slow to read again (6.2 GB takes about 95 seconds at 65 MB/s). Unix systems only.
- [ ] `--mlock` on Windows, through `VirtualLock`.
- [ ] Reading ahead at load (`MADV_WILLNEED`). Low value: at most 15 percent off a cold start from an SD card.

## Models

- [x] Mixture-of-experts models: Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE, Granite MoE and LFM2-MoE.
- [x] Hybrids of attention and short convolutions: LFM2, LFM2.5 and LFM3.
- [x] More architectures: Phi, Gemma 2, the MoE models of DeepSeek and gpt-oss, and the families that are not
      transformers (RWKV, Mamba).
- [x] Encoder tokenizers (WordPiece and Unigram), which only the encoder models use.
- [x] Encoder models, for embeddings and reranking.
- [x] Inkling 2 and Inkling 2 Mini (Thinking Machines): attention over the outputs of earlier blocks, a learned bias
      on distances instead of rotation, short convolutions on keys, values and block outputs, and Mini's loop of
      layers 6 to 17, run three times with 48 cache slots. Its tokenizer, chat format and reasoning markers too.
- [ ] Inkling's image input (its vision tower) and its multi-token prediction head (DSpark, for speculative
      decoding).
- [ ] Inkling's tool messages and tool calls, and the effort levels it has past `Thinking` (minimal at 0.1, max at
      0.99).
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
- [x] The weight formats NVFP4, Q1_0 and Q2_0, with AVX2 and NEON kernels for NVFP4 and Q2_0.
- [ ] The weight format F8_E4M3, which llama.cpp added after b11846.
- [ ] PLaMo 2's tokenizer.
- [ ] Models split across several GGUF files (`-00001-of-00004.gguf`), the way large models ship. The downloader
      skips them today, and the loader reads one file.

## Generation

- [x] Sliding the context window when it fills, or keeping its start and dropping the middle.
- [x] A half-precision key/value cache, to halve the memory of long contexts.
- [ ] Grammar-constrained output (GBNF), and JSON that follows a schema.
- [ ] Jinja chat templates, for models whose format is not one of the sixteen built in.
- [x] More samplers: Mirostat 1 and 2, typical, XTC, top-n-sigma, dynamic temperature, presence and frequency
      penalties, and a bias on chosen tokens, in llama.cpp's order and with its option names.
- [ ] DRY, llama.cpp's sampler against repeated sequences.
- [ ] Token probabilities in the output.
- [ ] Speculative decoding: a small draft model, EAGLE3, and prompt lookup.
- [ ] Saving a session's context to a file and loading it again.
- [ ] An 8-bit and a 4-bit key/value cache, for long contexts. High value on boards.
- [ ] Freeing the cache of sliding-window layers past their window (Gemma 2, Gemma 3, Gemma 4, gpt-oss, Inkling),
      as llama.cpp does. High value on boards: Gemma 4 and Nemotron 3.5 Bolt keep 512-position windows in 35 of
      their 42 layers, and 40 of Inkling 2 Mini's 48 cache slots look back 1,024 positions. Inkling keeps 4 KB per
      position and slot today, 805 MB at 4,096 tokens.
- [x] Saved convolution states capped at 16 MB together, the oldest dropped first. Inkling 2 Mini's are 3.8 MB each
      and grew to 61 MB at 4,096 tokens. It keeps four now, LFM2 dozens.
- [x] Sampling defaults from the model file (`general.sampling.*`: temperature, top-k, top-p, min-p, repetition
      penalty and its window, XTC and Mirostat), below `smallm.conf` and the command line. Nemotron 3.5 Bolt E4B's
      file asks for temperature 1.0 and top-k 64, where SmaLLM used 0.8 and 40.
- [x] A switch for thinking: `--think`, and `chat_template_kwargs.enable_thinking` and `reasoning_effort` in the
      server. Off is written the way the model's own template writes it, found in that template: Qwen3's
      `<think>\n\n</think>\n\n`, LFM3's `<think></think>`, SmolLM3's and RWKV-7 G1's. Gemma 4 is turned on with
      `<|think|>`, and gpt-oss takes low, medium or high on its Harmony `Reasoning:` line. On a board, a thinking
      model can spend tens of seconds before its answer.
- [ ] SmolLM3's system block (`## Metadata`, with its reasoning mode), which the ChatML format does not write.
- [ ] `--max-tokens` counting reasoning and answer tokens apart, so that a small limit does not end before the
      answer begins. The help says today that the limit includes reasoning.
- [ ] The new samplers, the thinking switch and the model file's sampling defaults in the C API and the Python
      binding.

## Speed

Gaps the measurements in PARITY.md and DEVELOPMENT.md show.

- [x] Q4_0 prompt reading on NEON with the dot product extension, 3.5 times slower than llama.cpp b12711 on the
      A733's A76 cores. A four-vector tile kernel (`sdot` by lane), signed tiles, 32 columns per scale and faster
      fills took it from 12.8 to 39.1 tokens per second. Nemotron's Q4_K_M prompts went from 8.9 to 14.3.
- [x] NEON attention, softmax, gates and activation quantizing. The portable loops took 9 percent of a prompt
      token on the A733. With 16-bit keys and values widened by `fcvtl`, attention went from 1.52 to 0.38 ms per
      token, the SiLU gate from 0.39 to 0.26 and quantizing from 0.40 to 0.32. `quantize` gives exactly the
      portable bytes.
- [ ] Q4_0 prompt reading on NEON, still 9 percent behind llama.cpp b12711 on the A733's two A76 cores (40.6
      against 44.6 tokens per second), and 7 percent on one (21.1 against 22.8), though ahead on all 8 cores
      (50.3). What was tried and what is left, in DEVELOPMENT.md:
      - [x] Zero the paired tile loops' sums with `fmov` from `xzr` instead of `movi`: 27.0 to 26.6 picoseconds
            per weight. In the other loops it cost more than it saved.
      - [x] Share each group of vectors between up to three tiles: 29.2 to 28.0 against a batch of 64, about 1
            percent end to end.
      - [ ] Interleave four vectors' activations, 16 bytes of each in turn, so that a pass reads one stream
            instead of four: 2 to 3 percent of the multiplies in a test.
      - [ ] Fill the next tiles inside the passes over the current ones. The fill is limited by its stores and
            the passes by the vector pipes, so they could overlap: about 3 percent.
      - [ ] Skip the tile offsets' stores when they are all zero, and run the norms on every thread: about 1
            percent together.
      - [ ] Fuse the gate with quantizing for the down projection: both stream the batch's 7680-wide activations
            through memory.
      - Splitting the columns of wide matrices was tried and made LFM3's feed-forward down 36 percent slower:
        rows read in halves stream badly.
      - An opt-in repack at load stays possible for boards with memory to spare.
- [x] Q4_0 generation on NEON, 14 percent behind llama.cpp b12711 on the A733. Row kernels that take four blocks at
      a time, Q6_K's for the tied output layer, and the NEON attention kernels made it 8.8 against 8.7 tokens per
      second. Both engines are now at the board's memory limit (9.9 GB/s, about 9.2 tokens per second for LFM3).
- [x] Q6_K's NEON row kernel, which the tied output layers of LFM3 and of Q4_K_M files use: 152 to 110 picoseconds
      per weight on an A76.
- [x] The default prompt batch on ARM. With the faster NEON kernels, 64 reads LFM3's prompt 7 percent faster than
      32 on the A733 for 4.7 MB more, and 128 is slower again, so every machine now defaults to 64. x86 moved to 64
      first: prompts up to 11 percent faster for 2 to 6 MB.
- [x] Prompt reading for mixtures of experts. A default batch of 128 tokens for models with experts: on x86
      DeepSeek reads prompts at 189 tokens per second against llama.cpp's 162 (0.83x before, at 32), and gpt-oss at
      187 against 103, for 44 and 26 MB of scratch memory.
- [ ] The prompt batch for expert models on ARM boards, not measured yet (no expert model on the A733).
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
- [x] Sampling options on `serve`, as defaults for requests.
- [ ] A built-in web page, and metrics.

## Tools

- [ ] Converting and quantizing models, with importance matrices. Today the GGUF file comes from elsewhere.
- [ ] Perplexity and the other evaluation tools.
- [ ] Bindings beyond Python, starting with Node.js and Go.
- [ ] A `no_std` core, for boards without an operating system.

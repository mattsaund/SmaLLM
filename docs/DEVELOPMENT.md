# Developing on single-board computers

SmaLLM is built for small machines, and most of what is left to learn about
its speed can only be learned on one. This page says what is known, what to
measure first on a board, and the open questions a board can answer. The
rules for changes and the map of the files are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Where things stand

SmaLLM is developed on an x86-64 desktop, an Intel Core i7-14700K. The
first ARM board was measured on 2026-11-24: an Allwinner A733 (results
below). CI runs every NEON kernel against the portable one on ARM runners,
and the engine matches llama.cpp on real models there.

- **Kernels.** AVX2, AVX2 with VNNI, NEON, and NEON with the dot product
  extension. The RK3588's cores have that extension, so expect the engine
  to report `neon+dotprod`.
- **Two routes through a quantized matrix** (see `src/model/matmul.rs`).
  The direct route takes one vector at a time, which is generation. Tiles
  take a batch of 8 or more vectors, which is prompt reading, with 4 rows
  per tile on NEON.
- **Threads.** Generation runs on the performance cores, prompts on every
  physical core. `default_threads` and `default_batch_threads` in
  `src/threads.rs` count them, by `cpu_capacity` on Linux, among the CPUs
  in the process's affinity mask. Physical cores are told apart by their
  sibling lists, since some boards number each cluster's cores from zero.
- **Memory.** The key/value cache keeps 16-bit floats and grows in chunks
  of 256 positions as the context fills. Recurrent models (Mamba, RWKV)
  keep a fixed state plus one saved copy of it.

The desktop's numbers against llama.cpp are in the README.

## Measure first

Nothing below is worth changing until it is measured on the board.

```sh
cargo build --release
lscpu                                    # which cores, which features
smallm pull bartowski/SmolLM2-135M-Instruct-GGUF
smallm pull bartowski/Llama-3.2-1B-Instruct-GGUF
smallm bench SmolLM2-135M                # check the 'engine' line
smallm bench SmolLM2-135M --profile
smallm bench Llama-3.2-1B --profile
cargo test --release --lib kernel_speed -- --ignored --nocapture
```

`kernel_speed` prints, for every weight format, picoseconds per weight of
the direct kernel, of filling a tile, and of the tile kernel. On the desktop
the tile kernel takes 6 to 8, the fill 20 to 30, and the direct kernels 21
to 47. The same three numbers on the board decide most of what follows.

Then build the llama.cpp harness on the board and compare, the way
[tools/reference/README.md](../tools/reference/README.md) describes:
`ref bench` and `ref mem` against `smallm bench`, the two taking turns,
best of several runs.

Run the tests on the board too. The golden tests want model files:

```sh
SMALLM_TEST_MODELS=~/.local/share/smallm cargo test --release --workspace
```

Keep large models on disk, not in `/tmp`, which is often memory.

## First board: Allwinner A733

Orange Pi Zero 3W with an Allwinner A733: 2 Cortex-A76 cores at 2.0 GHz
(cpu6 and cpu7) and 6 Cortex-A55 cores at 1.8 GHz (cpu0 to cpu5), the dot
product extension but no i8mm, 12 GB of LPDDR5, an SD card reading about 65
MB/s, and Armbian's 6.6 vendor kernel. Measured 2026-11-24 with SmaLLM
0.3.0, which picked the `neon+dotprod` kernels, against llama.cpp v0.9.0
built on the board.

Matched runs, same lengths, prompt and generation on the two A76 cores, 2
threads, tokens per second:

| Model (format) | Prompt: SmaLLM | Prompt: llama.cpp | Generation: SmaLLM | Generation: llama.cpp | SmaLLM, prompts on 8 cores |
|---|---|---|---|---|---|
| LFM3-1.7B (Q4_0), 128 prompt, 64 generated | 12.5 | 44.5 | 7.4 | 8.0 | 18.3 |
| Nemotron 3.5 Bolt E4B (Q4_K_M), 64 prompt, 32 generated | 8.4 | 8.3 | 2.7 | 2.5 | 12.6 |

Anonymous memory: SmaLLM 12 to 21 MB, llama.cpp 1.3 GB (LFM3) and 6.8 GB
(Nemotron). Pinning the generation threads to the A76s changed nothing:
the scheduler already keeps them there.

Findings:

- **Q4_0 prompt reading is 3.6 times slower than llama.cpp.** About 96
  percent of the prompt time is in the matrix multiplies (feed-forward up
  48, down 26, query, key and value 17). The batch size is not the cause:
  32, 64, 128 and 256 give 17.7, 18.8, 18.0 and 16.3 tokens per second on 8
  cores, so packing tiles per batch is not the main cost. llama.cpp's Q4_0
  GEMM for this CPU keeps a block of 4 rows by 4 tokens of 32-bit
  accumulators and uses `sdot` by lane, so each loaded weight register
  serves four tokens. The NEON tile kernel's register blocking is the first
  place to look.
- **Generation is memory-bound and close to llama.cpp.** Q4_0 is 7 percent
  behind (7.4 against 8.0), Q4_K_M 8 percent ahead (2.7 against 2.5). On
  Q4_0, SmaLLM reads weights at about 7.7 GB/s and llama.cpp at 8.3, so the
  Q4_0 matrix-vector kernel has some headroom. In LFM3's generation profile the
  feed-forward up projection takes 40 percent, down 21, the logits 18 (a
  tied 128,000 by 2,048 Q6_K embedding, about 215 MB per token) and query,
  key and value 13.
- **Prompts on every core pay off.** On mixed cores the work-claiming pool
  read Nemotron's prompt 46 percent faster on 8 threads than on 2 (12.6
  against 8.4). llama.cpp got slower when the A55s joined. This is a
  strength to keep.
- **Thread counts were wrong** in 0.3.0. The A76s report `core_id` 0 and 1
  in package 0, like the first two A55s (their `cluster_id` is 1), and were
  counted as their hardware threads. SmaLLM counted 6 physical cores
  instead of 8 and read prompts on 6 threads: 15.6 tokens per second on
  LFM3's 128-token prompt, against 18.3 with `--threads-batch 8`. The fast
  core count came out right only by chance. Thread counts also ignored the
  affinity mask: under `taskset -c 0-5` (the six A55s) SmaLLM still found
  two fast cores and generated on 2 threads, where 6 was right. Both are
  fixed after 0.3.0.
- **A prompt batch of 64** read 6 percent faster than 32 here (18.8
  against 17.7 tokens per second). Its scratch memory cost is not measured
  yet.
- **Storage.** Weights are mapped, so under memory pressure their pages are
  dropped and read again from the card: Nemotron's 6.2 GB takes about 95
  seconds at 65 MB/s. Locking the weights in memory (opt-in, for boards with
  RAM to spare) would prevent that. A cold start is fine: LFM3's first token
  (1.05 GB of weights) took 18.7 seconds straight off the card through page
  faults, against 16.1 to read the file in order and 1.2 warm. Asking the
  kernel to read ahead at load (`MADV_WILLNEED`) would save at most 15
  percent.
- **Thinking costs a lot here.** At about 7 tokens per second on two A76
  cores, LFM3 reasons for tens of seconds before every answer, and SmaLLM
  has no switch to turn that off. `--max-tokens` counts reasoning tokens
  too, so a small limit can end before the answer starts.
- **Sampling settings in the file are ignored.** Nemotron 3.5 Bolt E4B's
  file asks for temperature 1.0, top-k 64 and top-p 0.95. SmaLLM uses its
  own defaults, 0.8, 40 and 0.95.
- **The GPU would not help generation.** The PowerVR BXM-4-64 (vendor DDK
  24.2) measures 5.27 GB/s and 37 GFLOPS with clpeak at its 600 MHz, below
  the CPU's 8 GB/s. Its Vulkan shader compiler fails on 16-bit storage
  (llama.cpp's kernels stop at `createComputePipeline: ErrorUnknown`), and
  it has 16 KB of shared memory, a subgroup size of 1 and no integer dot
  product. A Vulkan backend meant for board GPUs needs 32-bit storage
  fallbacks and small shared-memory tiles.
- **The NPU does not fit language models.** The VeriSilicon VIP9000's
  runtime (VIPLite) runs only NBG graphs, compiled ahead of time on an x86
  PC with VeriSilicon's ACUITY tools, with static shapes and no key/value
  cache. Language models stop at SmolLM2 in 16-bit integers with a 32-token
  window, so it suits vision models, not SmaLLM's.

## Questions the board can answer

1. **Is the tile route a win on these cores, and from which batch size?**
   `TILE_BATCH` in `src/model/matmul.rs` is 8, chosen on x86. A tile costs
   one fill and then one kernel call per vector. If the NEON fill is slow,
   the break-even is higher. `kernel_speed` gives both costs. To switch
   tiles off for a test, set `tile_rows` to 0 in `src/kernels/arm.rs`.
2. **Are the thread counts right?** An RK3588 should generate on its 4 fast
   cores and read prompts on all 8, and the A733 on its 2 A76s and all 8.
   Check what `smallm bench` reports, then time other counts with
   `--threads` and `--threads-batch`.
3. **Is generation limited by memory or by the kernels?** Compare
   `kernel_speed` with the time per weight that `bench --profile` implies.
   On the desktop, generation waits for memory, and a faster kernel would
   not help. On slower cores with slower memory it may be the other way
   around, and then the direct NEON kernels in `src/kernels/arm.rs` are
   where to look.
4. **What does the prompt profile say?** Anything that is not a matrix
   multiply and takes more than a few percent is a target. The NEON backend
   still uses the portable versions of `quantize`, `silu_mul`, the softmax
   and the attention kernels (`dot_rows`, `mix_rows` and their 16-bit
   forms). Writing them with NEON is the likely next step if they show up.
5. **How large should a prompt batch be for mixtures of experts?** The
   default batch of 32 tokens leaves each of 64 experts about 3 tokens,
   too few for the tile route. On the desktop, `-b 256` read the prompts
   of DeepSeek-Coder-V2-Lite 1.6 times as fast as llama.cpp, where the
   default batch was slower than it. A larger default for expert models
   costs scratch memory.
6. **Memory.** `smallm bench` prints what the process holds. Compare with
   `ref mem`. Models of 10 to 12 GB, such as gpt-oss-20b and
   DeepSeek-V2-Lite at Q4_K_M, leave a 12 GB board little room. The
   weights are memory mapped, so the system can page them. See how fast
   they run when it does.

## Known unfinished work

- **Q4_0 prompt reading on NEON** is 3.6 times slower than llama.cpp on the
  A733 (above). It is the largest gap on boards.

- **NEON speed is unmeasured.** The tile fills, and the MXFP4, IQ4_NL and
  IQ4_XS kernels, were written without an ARM machine. CI shows they are
  correct, not how fast they are.
- **Formats without SIMD.** IQ1, IQ2 and IQ3, TQ1_0, Q2_K and Q3_K have no
  NEON kernels, and use the portable tile fill on both backends.
- **32-bit ARM and RISC-V** run portable code only.
- **Long contexts with latent attention** (DeepSeek). Each unit of work
  reads a layer's whole latent cache, eight times over during generation.
  Splitting long contexts by position, with partial softmaxes merged at the
  end, would fix it.
- **RWKV's low-rank maps** take about 13 percent of single-threaded prompt
  time, because their rows are short floats that the engine's float route
  handles one dot product at a time.
- **Mamba 1** keeps a 32-bit copy of its decay rates, 21 MB for the 2.8B
  model.
- **GPU, NPU**: nothing yet. The RK3588 has a Mali GPU and an NPU. The seam
  is the `Kernels` table and the `device` field of the C API. Do not start
  on this before the CPU numbers are in.

## How the desktop numbers were measured

- Speed: `smallm bench model.gguf -p 256 -n 96 -r 1 -t N`, and
  `ref bench model.gguf 256 96 N` for llama.cpp, three rounds, the two
  taking turns, best run of each.
- Memory: `smallm bench model.gguf -c 4096 -p 240 -n 16 -r 1 -t 4` (256
  tokens in all, one cache chunk) against `ref mem model.gguf 4096 256 4`.
- A busy machine only makes runs slower, so the best run is the honest
  one. Check `uptime` before trusting a number.

## When you have numbers

Put the board's results in the README next to the desktop's, with the
board's name and how they were measured. Update
[ARCHITECTURE.md](ARCHITECTURE.md) if a finding changes the design, and the
[CHANGELOG](../CHANGELOG.md) for anything a user would notice.

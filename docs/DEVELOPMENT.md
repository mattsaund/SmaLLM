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
  per tile on NEON, met by four vectors at a time with the dot product
  extension.
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
the direct kernel, of filling a tile, and of the tile kernel, and on NEON
with the dot product extension of the tile kernel for four vectors at once
(`tile4`, per weight and vector). Then the Q4_0 tile kernel against a whole
batch of 64 vectors, and the direct kernels over a matrix too large for
the caches, as generation reads it. On the desktop the tile kernel takes 6
to 8, the fill 20 to 30, and the direct kernels 21 to 47. The same numbers
on the board decide most of what follows.

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

Measured again after 0.3.0, against llama.cpp b12711 (its prebuilt
`ubuntu-arm64` build, which picks its `armv8.2_2` CPU variant here). Both
models with a 128-token prompt and 64 generated tokens on the two A76
cores, the engines alternating, each run started under 62°C, best of three.
The rounds agreed to within 1 percent.

| Model (format) | Prompt: SmaLLM | Prompt: llama.cpp | Generation: SmaLLM | Generation: llama.cpp |
|---|---|---|---|---|
| LFM3-1.7B (Q4_0) | 12.8 | 44.6 | 7.5 | 8.7 |
| Nemotron 3.5 Bolt E4B (Q4_K_M) | 8.9 | 8.5 | 2.8 | 2.7 |

This llama.cpp generates Q4_0 faster than the build measured first (8.7
against 8.0), which doubles the Q4_0 generation gap.

Findings:

- **Q4_0 prompt reading is 3.6 times slower than llama.cpp.** About 96
  percent of the prompt time is in the matrix multiplies (feed-forward up
  48, down 26, query, key and value 17). The batch size is not the cause:
  32, 64, 128 and 256 give 17.7, 18.8, 18.0 and 16.3 tokens per second on 8
  cores, so packing tiles per batch is not the main cost. llama.cpp's Q4_0
  GEMM for this CPU keeps a block of 4 rows by 4 tokens of 32-bit
  accumulators and uses `sdot` by lane, so each loaded weight register
  serves four tokens. The NEON tile kernel's register blocking is the first
  place to look. (Done: see "After the NEON work" below.)
- **Generation is memory-bound, and Q4_0 falls behind.** Against the build
  measured first, Q4_0 was 7 percent behind (7.4 against 8.0) and Q4_K_M 8
  percent ahead (2.7 against 2.5). Against b12711, Q4_0 is 14 percent
  behind (7.5 against 8.7) and Q4_K_M 4 percent ahead (2.8 against 2.7). On
  Q4_0, SmaLLM reads weights at about 7.7 GB/s, so the Q4_0 matrix-vector
  kernel leaves bandwidth unused. In LFM3's generation profile the
  feed-forward up projection takes 40 percent, down 21, the logits 18 (a
  tied 128,000 by 2,048 Q6_K embedding, about 215 MB per token) and query,
  key and value 13. (Closed below: both engines now sit near the board's
  memory limit.)
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
- **A prompt batch of 64 does not pay here.** The first runs showed 64 6
  percent faster than 32 (18.8 against 17.7 tokens per second), but those
  runs followed each other and heated the board. Alternated, each started
  under 62°C, best of three on all 8 cores, a 128-token prompt: LFM3 read
  19.3 tokens per second at both sizes, and Nemotron 11.9 at 32 against
  11.5 at 64. 64 took 4.7 MB (LFM3) and 6.4 MB (Nemotron) more. ARM kept
  32, and x86, where 64 is faster, moved to 64 (after 0.3.0). The faster
  NEON kernels below changed that, and ARM moved to 64 as well.
- **Heat decides back-to-back measurements.** The A76 cluster starts idle
  injection at 70°C (`cpub_idle_zone`) and throttles at 90°C, and the board
  idles at 60°C with its fan at full speed. One run of a 128-token prompt
  takes it to 81 to 84°C. Over a series of runs without pauses, LFM3's
  generation fell from 7.4 to 6.2 tokens per second. Alternate what is
  compared, and start each run at the same temperature: wait until
  `/sys/class/thermal/thermal_zone0/temp` is under 62000.
- **Storage.** Weights are mapped, so under memory pressure their pages are
  dropped and read again from the card: Nemotron's 6.2 GB takes about 95
  seconds at 65 MB/s. `--mlock` (opt-in, added after 0.3.0) keeps the
  weights in memory instead. It needs a locked-memory limit as large as the
  file, and the board's is 8 MB (`ulimit -l`), so a service needs
  `LimitMEMLOCK=infinity`. A cold start is fine: LFM3's first token
  (1.05 GB of weights) took 18.7 seconds straight off the card through page
  faults, against 16.1 to read the file in order and 1.2 warm. Asking the
  kernel to read ahead at load (`MADV_WILLNEED`) would save at most 15
  percent.
- **Thinking costs a lot here.** At about 7 tokens per second on two A76
  cores, LFM3 reasons for tens of seconds before every answer. 0.3.0 had no
  switch to turn that off. `--think off` does now (added after 0.3.0).
  `--max-tokens` counts reasoning tokens too, so a small limit can end
  before the answer starts.
- **Sampling settings in the file were ignored** in 0.3.0. Nemotron 3.5
  Bolt E4B's file asks for temperature 1.0, top-k 64 and top-p 0.95, where
  SmaLLM used its own 0.8, 40 and 0.95. They are the defaults now.
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

### After the NEON work

Measured 2026-11-25 as above: a 128-token prompt and 64 generated tokens on
the two A76 cores, 2 threads, the engines alternating, each run started
under 62°C, best of three, against llama.cpp b12711. Tokens per second:

| Model (format) | Prompt: SmaLLM | Prompt: llama.cpp | Generation: SmaLLM | Generation: llama.cpp |
|---|---|---|---|---|
| LFM3-1.7B (Q4_0) | 40.6 | 44.6 | 8.8 | 8.7 |
| Nemotron 3.5 Bolt E4B (Q4_K_M) | 15.0 | 8.5 | 2.9 | 2.7 |

On all 8 cores, its default for prompts, SmaLLM reads LFM3's prompt at
50.3 tokens per second and Nemotron's at 19.1 (19.3 and 11.9 before).
Nemotron's rounds varied, 11.7 to 15.0 for SmaLLM and 8.0 to 8.5 for
llama.cpp: with llama.cpp's 6.8 GB and a 6.2 GB file on a 12 GB board, the
mapped weights were partly read again from the card.

What changed, with `kernel_speed` on one A76 in picoseconds per weight:

- **Four vectors per pass over a tile.** `sdot` by lane repeats four bytes
  of one vector across the register, so each weight register loaded serves
  four vectors. With signed tiles (below), Q4_0's tile kernel went from
  99.7 to 38.1 per weight and vector.
- **Signed tiles.** `sdot` multiplies signed bytes, so zero points come off
  the weights at fill time, and most formats lose the offset pass that
  x86's unsigned byte multiply needs.
- **32 columns per scale.** Formats with one scale per 32 weights or more
  sum 32 columns as integers before they scale: 38.8 to 28.5.
- **One `sdot` chain per vector.** The A76 forwards `sdot`'s accumulator: a
  dependent `sdot` issues the next cycle, and four chains reach two per
  cycle (by lane and mixed with other work, about 1.74). Two chains per
  vector, added at the end, only cost the additions: 28.5 to 27.0.
- **Row kernels four blocks at a time** for the 32-weight formats. Four
  block sums go into one register, four scales are gathered through general
  registers and converted together, and a format's bias comes out in
  integers through the activations' sums. Q4_0 went from 119 to 76 in the
  cache and from 121 to 85 from memory.
- **Q6_K's row kernel** merges its 6-bit values with shifts that insert
  (`sli`, `sri`) and scales four sub-blocks with one multiply-add: 152 to
  110 in the cache, 160 to 125 from memory. The tied output layers of
  Q4_K_M files and LFM3 use it.
- **Tile fills** store four rows' columns with one `st4` and the scales and
  offsets as whole registers: Q4_0 162 to 123, Q8_0 144 to 99.
- **The next tile's rows are prefetched** while the current tile meets the
  batch, since four rows read at once are four streams.
- **NEON attention, softmax, gates and quantizing.** These were still the
  portable loops, and took 9 percent of a prompt token. Widening the 16-bit
  keys and values with `fcvtl` (four per instruction, where the portable
  conversion takes several instructions per value) took attention from
  1.52 to 0.38 ms per prompt token. The SiLU gate went from 0.39 to 0.26
  ms, and quantizing from 0.40 to 0.32, both now held back by streaming the
  batch's 7680-wide activations through memory. Generation gained too:
  8.6 to 8.8 tokens per second.
- **Tiles in sets.** The Q4_0 tile kernel runs at 26.5 picoseconds per
  weight and vector with four vectors in the nearest cache, but 29.2 to
  29.5 against a batch of 64 vectors of 2048 columns, whose 128 KB of
  activations come from the second-level cache. A thread now fills up to
  three tiles, as many as fit in 32 KB, and passes each group of four
  vectors over all of them: 28.0 with two tiles, 27.6 with three. About 1
  percent on LFM3's prompt.
- **Zeroing sums with `fmov` from `xzr`.** `movi` takes a slot on the vector
  pipes, and a bare copy of the tile kernel's step ran 4 percent faster
  with `fmov`. In the real kernel it gave 27.0 to 26.6. Elsewhere (the row
  kernels, Q6_K's tile kernel) the assembly kept the compiler from moving
  it and cost more than it saved, so only the paired tile loops use it.
- **A batch of 64 on ARM too.** With the faster kernels, LFM3 reads a
  128-token prompt at 37.0 tokens per second at 64, against 34.7 at 32 and
  35.9 at 128 (best of two), for 4.7 MB more.

Findings:

- **Generation is at the memory limit.** The board reads 9.9 GB/s with one
  core or two, and 8.2 with all eight. LFM3 reads 1.07 GB per generated
  token, so no engine passes about 9.2 tokens per second here. Both are
  close. A faster kernel shows on one core, and on chips with more
  bandwidth.
- **One stream beats several from memory.** Four rows of Q4_0 read at once
  were 20 percent faster than one row from the cache, but 20 percent slower
  from memory (Q8_0: 55 percent slower). Prefetching ahead within the rows
  changed nothing. Prefetching the next group of four rows recovered Q4_0
  but not Q8_0. So the row kernels read one row at a time.
- **The batch's activations cost 5 to 15 percent.** The Q4_0 tile kernel
  runs at 27.0 with four vectors in the nearest cache, 28.3 against 64
  vectors of 2048 columns, and 32.1 against 64 vectors of 7680 columns
  (LFM3's feed-forward down). Sharing each group of vectors between 2 or 4
  tiles gained nothing at 2048 columns. Splitting the columns of wide
  matrices is what is left there.
- **Tried and dropped.** Splitting the columns of wide matrices, so that a
  tile of LFM3's feed-forward down fits the nearest cache, made that
  multiply 36 percent slower (6.7 to 9.2 ms per token): its rows were then
  read in halves 4 KB apart, which stream badly. Spreading the next tile's
  prefetches over the passes, instead of asking for them at once, changed
  nothing. Prefetching activations a few steps ahead inside the tile
  kernel made it slower everywhere, 26.6 to 28.6 even in the cache.
- **Per core, the gap is the same.** On one A76, SmaLLM reads LFM3's prompt
  at 21.1 tokens per second and llama.cpp at 22.8. Both scale by 1.9 to
  two cores. What is left: interleaving four vectors' activations, 16
  bytes of each in turn, so that a pass reads one stream instead of four
  (27.2 against 27.8 in a test, 2 to 3 percent of the multiplies), and
  doing the tile fill inside the passes over the previous tiles. The fill
  is limited by its stores, the passes by the vector pipes, so they could
  overlap: about 3 percent.
- **The rest of the prompt gap is in the multiplies.** LFM3 spends about
  32 picoseconds per weight and vector on each core in its multiplies. The
  tile kernel accounts for 28 to 32 of that, the fill for 1.9 at a batch
  of 64, and the rest is threads and memory. The fill is limited by its
  stores, 192 bytes per block of four rows: the A76 writes 16 bytes per
  cycle into its nearest cache with `stp` or `st1`, and 10.7 with `st4`.
  In a copy of the loop, its loads, stores and arithmetic barely
  overlapped, and storing through shuffles and `st1` instead of `st4`
  gained nothing.

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
   multiply and takes more than a few percent is a target. On the A733 the
   portable attention kernels took 6 percent of a prompt token, so NEON
   has its own now, with the softmax, the gates and `quantize` (see "After
   the NEON work"). `sum_squares` and `to_half` are still portable.
5. **Are the prompt batch sizes right here?** The defaults are 64 tokens,
   and 128 for models with experts, whose batch is split among the experts
   its tokens are routed to. The dense default was measured on the A733,
   the expert one only on x86. Time `-b 32`, `64`,
   `128` and `256` on an expert model on a board, alternating and starting
   each run at the same temperature, and watch the memory `bench` reports.
6. **Memory.** `smallm bench` prints what the process holds. Compare with
   `ref mem`. Models of 10 to 12 GB, such as gpt-oss-20b and
   DeepSeek-V2-Lite at Q4_K_M, leave a 12 GB board little room. The
   weights are memory mapped, so the system can page them. See how fast
   they run when it does.

## Known unfinished work

- **Q4_0 prompt reading on NEON** is 9 percent behind llama.cpp on the
  A733's two A76 cores (40.6 against 44.6 tokens per second; it was 3.6
  times slower). It is still the largest gap on boards. See "After the NEON
  work" above for what is left.
- **NEON speed is measured only in part.** `kernel_speed` has timed every
  format's kernels and fills on an A76, but only Q4_0 and Q4_K_M models
  were timed end to end. The MXFP4, NVFP4, Q2_0, IQ4_NL and IQ4_XS kernels
  were written without an ARM machine. CI shows they are correct.
- **Formats without SIMD.** IQ1, IQ2 and IQ3, TQ1_0, Q2_K and Q3_K have no
  NEON kernels, and use the portable tile fill on both backends. Q1_0 has
  no SIMD kernels at all.
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

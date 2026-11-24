# Developing on single-board computers

SmaLLM is built for small machines, and most of what is left to learn about
its speed can only be learned on one. This page says what is known, what to
measure first on a board, and the open questions a board can answer. The
rules for changes and the map of the files are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Where things stand

Everything has been built and measured on one x86-64 desktop, an Intel
Core i7-14700K. The ARM code has never been timed. It is correct: CI runs
every NEON kernel against the portable one on ARM runners, and the engine
matches llama.cpp on real models there. Nobody knows yet how fast it is.

- **Kernels.** AVX2, AVX2 with VNNI, NEON, and NEON with the dot product
  extension. The RK3588's cores have that extension, so expect the engine
  to report `neon+dotprod`.
- **Two routes through a quantized matrix** (see `src/model/matmul.rs`).
  The direct route takes one vector at a time, which is generation. Tiles
  take a batch of 8 or more vectors, which is prompt reading, with 4 rows
  per tile on NEON.
- **Threads.** Generation runs on the performance cores, prompts on every
  physical core. `default_threads` and `default_batch_threads` in
  `src/threads.rs` count them, by `cpu_capacity` on Linux.
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

## Questions the board can answer

1. **Is the tile route a win on these cores, and from which batch size?**
   `TILE_BATCH` in `src/model/matmul.rs` is 8, chosen on x86. A tile costs
   one fill and then one kernel call per vector. If the NEON fill is slow,
   the break-even is higher. `kernel_speed` gives both costs. To switch
   tiles off for a test, set `tile_rows` to 0 in `src/kernels/arm.rs`.
2. **Are the thread counts right?** An RK3588 should generate on its 4 fast
   cores and read prompts on all 8. Check what `smallm bench` reports, then
   time other counts with `--threads` and `--threads-batch`.
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

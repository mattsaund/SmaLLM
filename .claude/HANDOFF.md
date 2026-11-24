# Handoff: SmaLLM on a real single-board computer

Written on 2026-11-24 for the Claude instance on the Orange Pi (12 GB RAM).
Read `CLAUDE.md` in this folder first: it has the rules and the commands.
This folder is in the repository only for this handoff. Remove it when the
work is done:

```sh
git rm -r .claude
```

## Where things stand

Everything was built and measured on one x86-64 desktop, an Intel Core
i7-14700K. The ARM code has never been timed. It is correct: CI runs every
NEON kernel against the portable one on ARM runners, and the engine matches
llama.cpp on real models there. Nobody knows yet how fast it is.

What exists:

- The engine, the `smallm` program, the C API and a Python binding.
- Kernels for AVX2, AVX2 with VNNI, NEON, and NEON with the dot product
  extension. The RK3588's cores have that extension, so expect the engine
  to report `neon+dotprod`.
- Two routes through a quantized matrix (see `src/model/matmul.rs`):
  - **Direct**, for one vector at a time. This is generation.
  - **Tiles**, for a batch of 8 or more vectors. This is prompt reading.
    4 rows per tile on NEON.

Numbers from the desktop are in the README, measured against llama.cpp
b11846 on Llama 3.2 1B: prompt reading is about 1.1 to 1.2 times as fast,
generation level or slightly ahead, and the engine holds 7.7 MB of its own
memory with a 4,096-token context where llama.cpp holds 627 MB. The
program is 0.60 to 0.90 MB.

Version 0.2.0 (2026-11-24) added, all tested on the desktop:

- **Fourteen architectures.** Each layer has its own `LayerSpec` in
  `src/model/config.rs`: attention or a short convolution, heads, widths,
  rotation, window, the layer whose cache it reads. Gemma 4, LFM2, LFM3,
  Granite, OLMoE, SmolLM3 and mixtures of experts (Mixtral, Qwen2/3-MoE,
  Granite MoE, LFM2-MoE) all run through one forward pass.
- **Grouped expert multiplies.** `Target::inputs` in `src/model/matmul.rs`
  lets every expert in use run in one multiply over its own rows.
- **Convolution state with rollback** in `src/model/conv.rs`.
- **Context overflow.** `--overflow middle|slide|stop`. `Session::shift`
  drops a span of tokens and rotates the kept keys back. No re-reading.
- **Settings file** `smallm.conf` (`src/bin/smallm/config.rs`), shown by
  `smallm config`, reset by `smallm config --reset`. It fills in options
  that are not on the command line.
- **`smallm uninstall`**, which the installers' uninstall options call.
- **Installers** install curl, and for source builds Rust and a C
  compiler, through the system's package manager, after asking.
- **Tokenizers** for all 67 BPE families llama.cpp knows, via a small regex
  engine, and Gemma 4's.

Test models live in `~/models` on the desktop. The golden tests cover 13
models; `tests/compare.rs` compares logits with llama.cpp's for a new
architecture (see its header). LFM3 has no llama.cpp reference in b11846,
so it was checked against upstream llama.cpp's source and behaviorally.

The owner also wants feature parity with llama.cpp and the other engines.
The README's to do list is the full gap list, grouped by area, with what is
done ticked.

## What to do first: measure

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

Then build llama.cpp on the board and compare, the way
`tools/reference/README.md` describes: `ref bench` and `ref mem` against
`smallm bench`, the two taking turns, best of several runs.

Run the tests on the board too. The golden tests want model files:

```sh
SMALLM_TEST_MODELS=~/.local/share/smallm cargo test --release --workspace
```

## Questions the board can answer

1. **Is the tile route a win on these cores, and from which batch size?**
   `TILE_BATCH` in `src/model/matmul.rs` is 8, chosen on x86. A tile costs
   one fill and then one kernel call per vector. If the NEON fill is slow,
   the break-even is higher. `kernel_speed` gives both costs. To switch
   tiles off for a test, set `tile_rows` to 0 in `src/kernels/arm.rs`.
2. **How many threads?** `default_threads` in `src/threads.rs` counts the
   fast cores by `cpu_capacity`, so an RK3588 should get 4. Check that it
   does, then time 4, 6 and 8 threads for prompt reading and for
   generation. On the desktop the slow cores helped prompt reading by a
   third and did nothing for generation. If the board agrees, the to-do
   item "a separate thread count for reading prompts" is worth doing.
3. **Is generation limited by memory or by the kernels?** Compare
   `kernel_speed` with the time per weight that `bench --profile` implies.
   On the desktop, generation is waiting for memory, and a faster kernel
   would not help. On slower cores with slower memory it may be the other
   way around, and then the direct NEON kernels in `src/kernels/arm.rs` are
   where to look.
4. **What does the prompt profile say?** `bench --profile` now covers
   prompt reading. Anything that is not a matrix multiply and takes more
   than a few percent is a target. On x86 that was rounding the
   activations and the feed-forward gate, both now parallel. The NEON
   backend still uses the portable versions of `quantize`, `silu_mul`,
   `dot_rows` and `mix_rows`. Writing them with NEON is the likely next
   step if they show up.
5. **Memory.** `smallm bench` prints what the process holds. Compare with
   `ref mem`. The key/value cache is 32-bit floats. A 16-bit cache would
   halve it and is on the to-do list.

## Things that are known to be unfinished

- **NEON tile fill**, `tile_fill` in `src/kernels/arm.rs`: written without
  an ARM machine. Correct, by CI. Its speed is unknown.
- **Q4_1, Q5_1, Q2_K, Q3_K and IQ4_NL** use the portable tile fill, and
  the last three have no direct SIMD kernel either. They are rare.
- **32-bit ARM and RISC-V** run portable code only.
- **Generation on one thread** trails llama.cpp by about 5 percent on 1B
  models on x86. Both are waiting for memory there.
- **GPU, NPU**: nothing yet. The RK3588 has a Mali GPU and an NPU. The seam
  is the `Kernels` table and the `device` field of the C API. Do not start
  on this before the CPU numbers are in.

## How the measuring was done on the desktop

- `smallm bench model.gguf -p 256 -n 96 -r 1 -t N`, and
  `ref bench model.gguf 256 96 N` for llama.cpp, three rounds, the two
  taking turns, best run of each.
- Memory: `smallm bench model.gguf -c 4096 -p 240 -n 16 -r 1 -t 4` (256
  tokens in all, one cache chunk) against `ref mem model.gguf 4096 256 4`.
- A busy machine only makes runs slower, so the best run is the honest
  one. Check `uptime` before trusting a number.

## When you have numbers

Put the board's results in the README next to the desktop's, with the
board's name and how they were measured. Update `docs/ARCHITECTURE.md` if a
finding changes the design, and `CHANGELOG.md` for anything a user would
notice.

# Reference results from llama.cpp

SmaLLM is tested against llama.cpp. The files here produce the reference
side of that comparison.

| File | Role |
|---|---|
| `ref.c` | A small program on the llama.cpp library: tokens, greedy generation, raw logits and speed. |
| `golden.sh` | Runs `ref` over a table of models and writes `tests/golden/*.golden`. |
| `quant.c` | Quantizes test rows with ggml and expands them again, for `tests/golden/quant`. |

The golden files are checked in, so running the tests needs none of this.
Use it to add a model to the tests, to refresh the results after llama.cpp
changes, or to measure llama.cpp on your own machine.

## Build the harness

`ref.c` needs a llama.cpp build with loadable backends. One CPU backend is
then loaded by name, so the comparison is CPU against CPU even on a machine
with a GPU.

```sh
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
cmake --build llama.cpp/build --config Release -j

cc -O2 tools/reference/ref.c -o ref \
    -Illama.cpp/include -Illama.cpp/ggml/include \
    -Lllama.cpp/build/bin -lllama -lggml -lggml-base

export LD_LIBRARY_PATH=llama.cpp/build/bin
export REF_CPU_BACKEND=llama.cpp/build/bin/libggml-cpu.so
```

## Write the golden files

```sh
tools/reference/golden.sh ./ref ~/models
```

The table at the end of `golden.sh` names each model file, its prompt and
the number of tokens to generate. Models that are not in the folder are
skipped, and their golden files stay as they are.

Each golden file records, for one model:

- The tokens of `tests/golden/texts/mixed.txt`, with control tokens read as
  text and again read as control tokens.
- The tokens of a prompt from `tests/golden/prompts`.
- The tokens llama.cpp generates after it with greedy sampling.

`tests/golden.rs` replays the same inputs through SmaLLM.

## Write the weight format vectors

`tests/quant.rs` checks that SmaLLM expands the blocks of the IQ, TQ and
MXFP4 formats to the same floats as ggml, bit for bit. Its vectors in
`tests/golden/quant` come from `quant.c`, which needs only ggml's base
library:

```sh
cc -O2 tools/reference/quant.c -o quant -Illama.cpp/ggml/include \
    -Lllama.cpp/build/bin -lggml-base -lm
LD_LIBRARY_PATH=llama.cpp/build/bin ./quant tests/golden/quant
```

For each format it quantizes the same pseudo-random rows with
`ggml_quantize_chunk` and expands the blocks with the format's `to_float`.
A file holds the element count, the blocks and ggml's floats. The rows are
the same on every run, so the files only change when ggml does. To check
another format, add it to `TYPES` in `quant.c` and to `FORMATS` in
`tests/quant.rs`.

The grids these formats look weights up in come from ggml's source as
well. `tools/gen_iq_tables.py` writes them to `src/tensor/iq_tables.rs`:

```sh
python3 tools/gen_iq_tables.py llama.cpp > src/tensor/iq_tables.rs
```

## Compare speed

```sh
./ref bench model.gguf 256 96 4      # prompt tokens, generated tokens, threads
smallm bench model.gguf -p 256 -n 96 -t 4
```

Both read the same synthetic prompt and generate greedily. Run each several
times, taking turns, and keep the best run of each. A busy machine only ever
makes a run slower.

## Compare accuracy

`ref ids` and `ref logits` write the raw logits llama.cpp computes. Set
`SMALLM_KERNELS=exact` to make SmaLLM compute without rounding activations
to 8 bits. The distance from each engine to that exact result is the error
its integer arithmetic adds. [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
has the numbers.

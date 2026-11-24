# How SmaLLM works

This page follows a prompt through the engine, from the model file to the
next token, and says why each part is built the way it is.
[CONTRIBUTING.md](../CONTRIBUTING.md) has the map of the files.

## The layers

```
text ──> tokenizer ──> forward pass ──> sampler ──> tokenizer ──> text
                           │
            ┌──────────────┼──────────────┐
          matmul        attention      key/value
            │              │             cache
         kernels  <────────┘
            │
      model file (memory mapped)
```

Each module depends only on the ones below it in this list:

| Module | Role |
|---|---|
| `error` | The error type. |
| `gguf` | Reads the model file: metadata and the tensor table. |
| `tensor` | Tensor types and the bit layout of quantized blocks. |
| `kernels` | Dot products, scalar and SIMD, picked at startup. |
| `threads` | The thread pool that runs the kernels in parallel. |
| `tokenizer` | Text to token ids and back. |
| `model` | Weights, the forward pass and the key/value cache. |
| `sampler` | Picks the next token from the logits. |
| `chat` | Formats a conversation the way a model was trained on. |
| `generate` | The generation loop: prompt in, text out. |

The program in `src/bin/smallm` and the C API in `capi` are two users of
this library. Neither reaches past its public functions.

## Loading a model

A GGUF file is a header, a list of metadata, a table of tensors and then the
tensor data. SmaLLM maps the file into memory and parses the first three
parts. Nothing large is copied:

- **Weights** stay in the mapping, in the format they were stored in. A
  weight matrix is an offset, a row count and a row size.
- **The vocabulary** stays in the mapping too. The tokenizer records where
  each token's text sits and builds a hash table of 4-byte slots over it.
- **Small vectors**, the norm weights and biases, are expanded to floats.

This is why a model loads in a few milliseconds. The pages of the file are
shared with the system's page cache, so the weights cost no memory of the
process's own, and a second program on the same model shares them.

A model file is untrusted input. Every read goes through a cursor that
checks its bounds, sizes are checked before they are multiplied, and each
tensor's shape is checked against the configuration before the forward pass
may index it. The tests truncate a file at every byte and flip every byte of
its header to make sure a broken file is an error and never a crash.

## Text to tokens

Two tokenizer kinds cover the supported models. SentencePiece models rank a
merge by the score of the token it forms. Byte-level BPE models rank it by
its place in the model's merge list. One loop serves both: it starts from
characters and joins the best adjacent pair until none is left. A heap keeps
the cost of a long word near `n log n` steps instead of `n * n`.

BPE models split text into words first, and each model family defines a
word with its own regular expression. A regex engine would be the largest
part of the tokenizer, so the few expressions in use are written out as
small scanners in `tokenizer/pretok.rs`.

Control tokens such as `<|im_end|>` are only recognized in text the engine
trusts. A chat is rendered as a list of parts, markup and message text
apart, and only markup is searched for control tokens. Somebody typing
`<|im_end|>` into a chat gets those characters, not the end of the turn.

## The forward pass

Every supported architecture is the same decoder-only transformer with a
few switches, so there is one forward pass and no per-architecture code:

```
h      = norm(x)
q k v  = h * Wq, h * Wk, h * Wv          (optional per-head norm)
q k    = rotate by position (RoPE)
a      = softmax(q . K) . V               over every earlier position
x     += [norm] (a * Wo)
h      = norm(x)
x     += [norm] ((act(h * Wgate) * (h * Wup)) * Wdown)
```

`model/config.rs` turns the architecture name into the switches: how
positions are rotated, which activation the feed-forward network uses,
whether embeddings are scaled, whether queries and keys are normalized, and
whether some layers only look at a sliding window.

Three choices keep it fast:

- **Matrices that read the same input run together.** The query, key and
  value projections share one input, and so do the two halves of the
  feed-forward gate. The input is prepared once and the threads are woken
  once for the group.
- **The last layer does less.** Only the final token's logits are read, so
  in the last layer only that token goes through the output projection and
  the feed-forward network.
- **Positions are exact.** Rotation angles are computed in 64-bit floats,
  which keeps them accurate hundreds of thousands of tokens in.

## Multiplying by quantized weights

Nearly all the time goes to one operation: a weight matrix times an
activation vector. The weights are quantized, typically to 4 or 5 bits each.
SmaLLM never expands them to floats. It does the multiply in integers:

1. The activation vector is quantized to 8-bit integers, once per multiply.
   It is cut into groups that match the weight format, 32 or 256 values, and
   each group gets its own scale so that its largest value maps to 127.
2. Each weight row is reduced against it with integer dot products, 32
   weights at a time in one register.
3. One floating-point multiply per block applies the weight scale and the
   activation scale.

Every weight format reduces to the same shape, which is what makes one plan
fit all of them:

```
weight[i] = scale[i / 16] * q[i] + offset[i / 16]
```

`q` is a small integer, and the scales and offsets are floats shared by 16
weights. The `q` part becomes the integer dot product. The offset part needs
only the sum of each 16 activations, which is computed once when the vector
is quantized.

`tensor/quant.rs` is the single description of each format's bit layout, as
plain scalar code. The kernels for the common formats are written again
with SIMD in `kernels/x86.rs` and `kernels/arm.rs`, and the tests hold every
one of them to the scalar result on random blocks, on extreme values and on
every row length.

| Backend | Runs on | Byte multiply |
|---|---|---|
| `scalar` | Everything. The reference for the others. | Plain loops the compiler vectorizes. |
| `avx2` | x86-64 with AVX2, FMA and F16C | `vpmaddubsw`, `vpmaddwd` |
| `neon` | Every ARM64 CPU | `smull` and pairwise adds |
| `neon+dotprod` | ARM64 with the dot product extension, such as the Raspberry Pi 5 | `sdot` |
| `exact` | Everything. For measuring accuracy. | None: weights are expanded to floats. |

The backend is chosen once, when the process starts, from what the CPU
reports. `SMALLM_KERNELS=scalar` or `SMALLM_KERNELS=exact` overrides the
choice.

A backend is a table of function pointers, `kernels::Kernels`. That table is
the seam for other devices. A session takes its kernels as a parameter, and
the C API already carries a `device` field for choosing one.

## Attention and the key/value cache

Each layer stores one key and one value vector for every position it has
seen, per key/value head. A new token's attention is two passes over that
history: a dot product of its query with every key, then a weighted sum of
every value. Models share one key/value head between several query heads,
and the kernels handle all the query heads of a group in one pass over the
keys.

The cache is allocated in chunks of 256 positions, as positions are filled:

- Memory follows the tokens actually in the context, not the configured
  window.
- The window has no built-in maximum. It ends where memory does.
- Nothing is reserved up front, which matters on systems that do not
  overcommit memory and in 32-bit address spaces.

A position costs `layers x key/value heads x head size x 8` bytes. A
conversation can be rolled back to any earlier position, which is how a
chat that shares its beginning with the last one skips the shared part.

## Threads

A forward pass runs a few hundred short parallel loops per token, so the
cost of starting a loop matters more than anything else in the pool:

- Workers spin for 200 microseconds after a loop ends before they sleep. The
  next loop usually arrives within that time and starts in nanoseconds.
- Work is cut into chunks that the threads claim from a shared counter. A
  fast core takes more chunks than a slow one, so mixed CPUs are used
  evenly without knowing which core is which.
- Small jobs run on the calling thread. Waking the pool for them would cost
  more than the work.

The default thread count is one per performance core. Two hardware threads
on one core do not help these kernels, and on CPUs with fast and slow cores
the slow ones add nothing to generation.

## Sampling and the generation loop

The sampler narrows the candidates in a fixed order: repetition penalty,
top-k, top-p, min-p, temperature, then a random draw. A temperature of zero
takes the best token.

The loop in `generate.rs` does the bookkeeping around it:

- **It reuses the context.** When a new prompt starts with the tokens
  already in the session, only the rest is evaluated. Every turn of a chat
  after the first works this way.
- **It streams whole characters.** A token can end in the middle of a UTF-8
  character. The text is held back until the character is complete.
- **It holds back possible stop strings.** Text that could be the start of a
  stop string is not shown until it is clear that it is not one.
- **It trims long chats.** When a conversation outgrows the window, the
  oldest turns are left out and the system message is kept.

Chat formats are recognized from the template text in the model file and
written out by hand. Running the template itself would take an interpreter
several times the size of everything else in the module.

## Accuracy

Float models are computed exactly: the engine's logits equal a slow 64-bit
reference to rounding error. Quantized models add one approximation, the
8-bit activations of the integer multiply. llama.cpp makes the same
approximation in a slightly different way.

The table measures it: the root mean square difference between each
engine's logits and the logits computed with exact activations, at the end
of a test prompt. A logit is a natural-log score, so 0.1 moves a token's
probability by about ten percent of itself.

| Model | SmaLLM | llama.cpp |
|---|---|---|
| stories260K (F32) | 0.000 | 0.002 |
| stories15M (Q4_0) | 0.057 | 0.030 |
| SmolLM2 135M (Q4_K_M) | 0.197 | 0.278 |
| Qwen2.5 0.5B (Q4_K_M) | 0.171 | 0.186 |
| Qwen3 0.6B (Q4_K_M) | 0.245 | 0.202 |
| Llama 3.2 1B (Q4_K_M) | 0.130 | 0.123 |
| TinyLlama 1.1B (Q4_K_M) | 0.124 | 0.163 |
| Gemma 3 270M (Q4_K_M) | 4.17 | 1.68 |

The two engines are equally close to exact, each ahead on some models.
Gemma 3 270M is the outlier for both. At the measured position that model
amplifies rounding: moving SmaLLM's rounding point by half a percent swings
its number between 0.8 and 4.2. Its greedy output still matches llama.cpp's
token for token.

The golden tests in `tests/golden.rs` hold the engine to llama.cpp on real
models. Tokens must match exactly. Generation is checked one step at a time
along llama.cpp's own output: each of its tokens must be the engine's
choice, or within a small margin of it at the few steps where two tokens
are nearly tied.

## What the engine does not do

- It computes on the CPU. GPUs and NPUs are planned as further backends.
- It runs text models of four architectures. No images, audio or
  embeddings.
- It knows eight chat formats and does not interpret templates.
- It is little-endian only, like the GGUF files themselves.

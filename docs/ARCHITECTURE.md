# How SmaLLM works

This page follows a prompt through the engine, from the model file to the
next token, and says why each part is built the way it is.
[CONTRIBUTING.md](CONTRIBUTING.md) has the map of the files.

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
word with its own regular expressions. `tokenizer/pretok_table.rs` holds the
expressions of every family llama.cpp knows, generated from its source by
`tools/gen_pretok.py`. A small backtracking regex engine in
`tokenizer/regex.rs` runs them. The common ones are also written out as
scanners in `tokenizer/pretok.rs`, which are several times faster, and a
test checks every scanner against its expression on thousands of texts.
Gemma 4 has a family of its own: it splits only at newlines and merges raw
UTF-8 bytes with spaces written as `▁`.

Embedding models bring two more kinds. WordPiece (BERT) cleans the text,
lowercasing it and stripping accents with tables taken from llama.cpp, cuts
it into words, and covers each word from the left with the longest tokens
that fit. Unigram (T5, XLM-RoBERTa) first rewrites the text with the map of
replacements the model carries, then picks, of all the ways to cover the
text with tokens, the one whose token scores add up highest.

Control tokens such as `<|im_end|>` are only recognized in text the engine
trusts. A chat is rendered as a list of parts, markup and message text
apart, and only markup is searched for control tokens. Somebody typing
`<|im_end|>` into a chat gets those characters, not the end of the turn.

## The forward pass

Every supported architecture is a stack of the same block, so there is one
forward pass and no per-architecture code:

```
h      = norm(x)
x     += [norm] mix(h)                   attention or a short convolution
h      = norm(x)
x     += [norm] ((act(h * Wgate) * (h * Wup)) * Wdown)
```

Most blocks mix positions with attention:

```
q k v  = h * Wq, h * Wk, h * Wv          (optional per-head norms)
q k    = rotate by position (RoPE)
a      = softmax(q . K) . V               over the visible positions
mix    = [gate] a * Wo
```

The LFM models replace some of them with a short convolution, which looks
at the last few positions only:

```
b c s  = h * Wb, h * Wc, h * Ws
y      = sum over taps i of kernel[i] * (b * s)[position - taps + 1 + i]
mix    = (c * y) * Wout
```

Mamba blocks (Mamba, Falcon Mamba, Mamba 2, and most blocks of Granite 4.0
H) carry a state of fixed size from one position to the next, and most
Mamba models have no feed-forward network:

```
x z    = h * Wx, h * Wz
x      = silu(conv(x) + bias)            over the last few positions
dt     = softplus(time step)             per head and token
s      = s * exp(dt * A) + dt * x * B    a row of the state per channel
mix    = [norm] ((s . C + D * x) * silu(z)) * Wout
```

The scan runs the tokens in order, but its heads are independent, so the
threads split it by heads, and each takes its heads through the whole
batch while their state stays in its cache. The projections around it read
the batch at once, like every other multiply. `model/forward/mamba.rs` has
the details.

A few architectures change the block a little. Switches cover them, not
code of their own:

- Gemma 2 caps attention scores, `cap * tanh(score / cap)`. gpt-oss gives
  each head a learned score, a sink, that joins the softmax with no value
  behind it, so a head can attend to nothing.
- Phi-2 normalizes with LayerNorm and biases, and runs attention and an
  ungated network, `down(act(up(h)))`, side by side on the same input.
- gpt-oss gives its experts biases and OpenAI's SwiGLU, which clamps the
  gate and the up projection.

Rotation is plain, linearly scaled, or stretched for long contexts in one
of two ways. LongRoPE (Phi-3) divides each frequency by a factor from the
file, from a short set or a long one by the length of the context. YaRN
(gpt-oss, DeepSeek) scales the low frequencies, keeps the high ones, blends
the two along a ramp in between, and scales the rotated values up a little.

RWKV models have no attention. Each head keeps a square matrix that every
token decays and writes its key and value into, and the token's output is
that matrix read with its receptance. RWKV-7 also removes what the matrix
held for a key before it writes the new value. The blocks normalize with
layer norms, and every input moves part of the way toward the previous
token's. They run a path of their own, in `model/rwkv.rs`.

Inkling 2 (Thinking Machines) runs a path of its own too, in
`model/forward/inkling.rs`, with its weights bound by `model/inkling.rs`. Its
blocks read more than one residual stream. The first block of every group
of six keeps its input aside and starts a new stream, and the inputs of each
block's attention and experts attend over depth: the kept streams and the
current one are normalized, scored against a learned query, and summed with
the softmax of the scores. Attention has no rotation. Each head also makes a
short relative query that dots with a learned row per distance, and that
bias, minus a sink score per head, joins the scores after their scale.
Queries and keys are capped in length, values are normalized, and short
convolutions with a residual run over the keys, the values and both block
outputs. Inkling 2 Mini runs layers 6 to 17 three times. Each later pass
starts from the same input, takes gated messages mixed from the previous
pass's outputs, and keeps its own key/value cache and convolution state, so
the 24 layers have 48 cache slots.

`model/config.rs` turns the architecture name and its metadata into a few
model-wide switches (how positions are rotated, the activation, embedding
and logit scaling) and one `LayerSpec` per block. A layer's spec says how it
mixes positions, how many heads it has and how wide they are, how much of
each head rotates and with which base, how far back it looks, and whether
it reads another layer's keys and values. Layers can differ in all of
these, which is what Gemma 4 and the LFM models need. Gemma 4 also gives
each block a small input of its own, looked up per token.

In a mixture of experts, the feed-forward network is many small networks.
A router scores them for each token, the best few run, and their outputs
are summed with the router's weights. The tokens of a batch are sorted by
expert, so each expert multiplies all of its tokens at once, and every
expert in use runs in the same multiply. A generated token wakes the
threads twice per layer, whatever the number of experts it uses.

Embedding and reranking models (BERT, nomic-bert) run on the same matrix
engine through `model/encoder.rs`. They read a whole text at once, with
attention that looks both ways, and put out one vector for the text instead
of a next token: the mean of all positions, or the first or last one. A
reranker reads a query and a document together and scores the pair with a
small classifier on that vector.

Three choices keep it fast:

- **Matrices that read the same input run together.** The query, key and
  value projections share one input, and so do the two halves of the
  feed-forward gate. The input is prepared once and the threads are woken
  once for the group.
- **The last layers do less.** Only the final token's logits are read.
  Once no later layer keeps anything for later tokens, only that token goes
  on through the rest of the model. That is the last layer in most models,
  and more than half of Gemma 4, whose last twenty layers reuse the caches
  of earlier ones.
- **Positions are exact.** Rotation angles are computed in 64-bit floats,
  which keeps them accurate hundreds of thousands of tokens in.

## Multiplying by quantized weights

Nearly all the time goes to one operation: a weight matrix times an
activation vector. The weights are quantized, typically to 4 or 5 bits each.
SmaLLM never expands them to floats. It does the multiply in integers:

1. The activation vector is quantized to 8-bit integers, once per multiply.
   It is cut into groups that match the weight format, 32 or 256 values, and
   each group gets its own scale so that its largest value maps to 127. A
   few formats have blocks of 64 or 128 that span several groups of 32, as
   they do in ggml.
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

The i-quants (IQ1 to IQ3) store indices into grids of eight or four values
each, which `tensor/iq_tables.rs` holds, generated from ggml's source. Their
grid values become `q`. The ternary formats store -1, 0 and 1, and MXFP4
stores 4-bit floats with a power-of-two scale for every 32 weights. NVFP4
stores the same 4-bit floats with an 8-bit float scale for every 16, Q2_0
stores 2-bit values with a scale for every 64, and Q1_0 stores only signs,
with a scale for every 128. They all fit the same shape. `tests/quant.rs`
checks every format against ggml's own expansion, number for number.

`tensor/quant.rs` is the single description of each format's bit layout, as
plain scalar code. The kernels for the common formats are written again
with SIMD in `kernels/x86.rs` and `kernels/arm.rs`, and the tests hold every
one of them to the scalar result on random blocks, on extreme values and on
every row length. On x86, the rarer formats are decoded by the scalar code
a block at a time and multiplied with AVX2, which is most of the gain for
little code.

| Backend | Runs on | Byte multiply |
|---|---|---|
| `scalar` | Everything. The reference for the others. | Plain loops the compiler vectorizes. |
| `avx2` | x86-64 with AVX2, FMA and F16C | `vpmaddubsw`, `vpmaddwd` |
| `avx2+vnni` | x86-64 with AVX-VNNI as well: Intel since the 12th generation and the N100, AMD since Zen 5 | `vpdpbusd` for tiles |
| `neon` | Every ARM64 CPU | `smull` and pairwise adds |
| `neon+dotprod` | ARM64 with the dot product extension, such as the Raspberry Pi 5 and the RK3588 | `sdot` |
| `exact` | Everything. For measuring accuracy. | None: weights are expanded to floats. |

The backend is chosen once, when the process starts, from what the CPU
reports. `SMALLM_KERNELS=scalar` or `SMALLM_KERNELS=exact` overrides the
choice.

A backend is a table of function pointers, `kernels::Kernels`. That table is
the seam for other devices. A session takes its kernels as a parameter, and
the C API already carries a `device` field for choosing one.

## Reading a prompt: tiles

Generating a token multiplies each weight row by one vector. Reading a
prompt multiplies it by a batch of them, 64 by default, or 128 for a model
with experts, and that changes what is worth doing.

The kernels above unpack a row's blocks while they multiply. For a batch
that would unpack every row once per vector. So a batch of 8 or more
vectors takes another route:

1. A few rows are unpacked once into a **tile**: 8 rows with AVX2, 4 with
   NEON. Unpacking 6-bit weights costs the same as multiplying them, so
   paying it once per row instead of once per vector is most of the gain.
2. The rows of a tile are interleaved. Columns are taken four at a time,
   and for each run of four the tile holds those weights of its first row,
   then of its second, and so on. A register loaded from that spot has one
   32-bit lane per row.
3. One multiply then serves every row of the tile: the four activation
   bytes are repeated in every lane, and each lane's sum is one row's
   result. Nothing is added across lanes at the end, which is what a
   row-at-a-time kernel spends a third of its time on.

x86's byte multiply takes unsigned weights, so on x86 signed weights are
shifted to be zero or more when a tile is filled, and the shift moves into
the tile's offsets. NEON's `sdot` multiplies signed bytes, so NEON tiles go
the other way: a format's zero point (8 for Q4_0, 32 for Q6_K) comes off the
weights at fill time, and the tiles of most formats need no offset term.

With the dot product extension, NEON also multiplies a tile by four vectors
at once. `sdot` by lane repeats four bytes of one vector across the
register, so each weight register loaded serves four vectors, and the
tile's four rows times four vectors keep four sums going. Formats with one
scale per 32 weights or more (Q4_0, Q8_0, Q4_K and others) sum 32 columns
as integers before scaling, where the tile's scale layout steps every 16.

A thread fills up to three tiles at a time, as many as fit in about 32 KB,
and each group of vectors meets all of them before the next group. A
batch's activations outgrow the nearest cache, so this way they come from
further out once for several tiles instead of once for each.

A tile lives in a few kilobytes of scratch memory per thread, so this route
costs no memory to speak of. The weights stay in the model file. llama.cpp
gets its batch speed from a rearranged copy of the weights that it keeps in
memory, which is why it holds hundreds of megabytes of its own for a model
that SmaLLM runs in a few.

`tensor/quant.rs` has the portable way to fill a tile, and each backend
fills the common formats with its own instructions. A test holds those to
the portable fill bit for bit.

Generation does not use tiles. One vector per row leaves nothing to share
the unpacking with, and at that point speed is set by how fast memory
delivers the weights, not by the multiply. The row kernels read each row
from start to end, one stream that the hardware prefetcher follows well.
On NEON they take four blocks of 32 at a time and scale them together:
four sums in the lanes of one register, four scales gathered and converted
at once, and the format's zero point taken off through the activations'
sums. Reading four rows at once shares more work, but four streams 1 KB
apart came from memory 20 to 50 percent slower on a Cortex-A76.

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

Keys and values are stored as 16-bit floats, as llama.cpp stores them, so
a position costs `key/value heads x head size x 4` bytes for every layer
that has a cache. That halves both the memory and what attention reads, and
at a 4,096-token context generation is about 14 percent faster than with
32-bit floats. `--cache-type f32` keeps them exact, which the tests that
compare against exact math use. DeepSeek's latent attention caches one
compressed row per position instead of a key and a value per head. Layers that reuse another layer's cache, as the last
twenty of Gemma 4 do, and convolution layers cost nothing per position. A
conversation can be rolled back to any earlier position, which is how a
chat that shares its beginning with the last one skips the shared part.

DeepSeek V2 and V3 attend through a latent. Each position keeps one small
latent vector and one rotary key that every head shares: 576 floats per
layer in DeepSeek V2 Lite, where its heads' keys and values would take
5,120. A head's key and value are projections of the latent, but the engine
never computes them. It moves each query into the latent space instead,
through the key half of the head's projection, and scores the latents
directly. The weighted sum of latents then goes through the value half.
Older files store both halves in one matrix, whose key rows are read
transposed, one row at a time. Newer files store the key half transposed
already. `model/forward/mla.rs` has the details.

A convolution layer keeps the inputs of its last few positions instead of
a cache, a few kilobytes. Rolling it back needs inputs that are gone, so
`model/conv.rs` keeps the last eight positions' inputs as well, and saves
every layer's state each 256 positions. A short rollback costs nothing. A
longer one restores the last saved state before the cut and reads the
tokens after it again.

A recurrent block (Mamba, RWKV) folds each token into a state of fixed
size, a matrix per head, so it needs no cache that grows. A state cannot be
taken apart again, though. `model/recurrent.rs` saves every block's state at
the end of each prompt and keeps the last three. Going back into a reply
restores the state after the prompt before it, and the reply is read again
as one batch, which is quick.

When the window is full, the session drops the oldest tokens after a kept
start: the first token, or the conversation's opening. The tokens after the
gap move down, and their cached keys are rotated back by the size of the
gap, which is all a rotary position embedding needs to place a key at a new
position. Nothing is read again, and the kept tokens still carry what they
took from the dropped ones, as in llama.cpp's context shift.

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
- The steps around the matrices run in parallel too once a batch is large:
  rounding the activations to 8 bits, and the gate of the feed-forward
  network. Left serial, they were a fifth of the time on eight threads.

Generation and prompt reading get separate pools of threads, because
they are limited by different things. Generating one token reads every
weight once, so it waits on memory: one thread per performance core, and
the slow cores would only add threads for the fast ones to wait on. A
prompt is read in batches, where each weight serves many tokens, so it is
arithmetic and every core helps: one thread per physical core, slow cores
included. On an Intel Core i7-14700K that reads a prompt 28 percent faster
than the eight performance cores alone. Two hardware threads on one core
help neither.

## Sampling and the generation loop

The sampler narrows the candidates in llama.cpp's default order: logit
bias, the repetition, frequency and presence penalties, top-n-sigma, top-k,
typical, top-p, min-p and XTC. Then the temperature, fixed or moved with the
entropy of what is left, and a random draw. A step whose setting is off is
skipped, and a temperature of zero takes the best token. Mirostat replaces
the truncating steps: it keeps the tokens whose surprise is under a running
estimate, which it moves after each choice toward a target.

The defaults come from the model file where it recommends them
(`general.sampling.*`), below the settings file and the command line.

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
- **It keeps reasoning apart.** Thinking models write their reasoning
  between `<think>` tags, in Gemma 4's `<|channel>thought` block, in
  Inkling 2's `<|content_thinking|>` messages or, in OpenAI's Harmony
  format, on a channel other than the final one. A small state machine
  follows those markers, so the reasoning can be shown faint, sent as its
  own field, or left out, and the markers never reach the output.

Chat formats are recognized from the template text in the model file and
written out by hand. Running the template itself would take an interpreter
several times the size of everything else in the module.

The template text also says how a model's thinking is switched. A template
with an `enable_thinking` argument writes a block of empty reasoning after
the assistant's opening to turn it off, and that block is found in the text:
`<think>\n\n</think>\n\n` for Qwen3, `<think></think>` for LFM3. Gemma 4
turns thinking on with `<|think|>` at the top of the system turn, and gpt-oss
sets a level on the `Reasoning:` line of its system message. Inkling 2 is
given an effort from 0 to 0.99 in a system message of its own. `Format` in
`chat.rs` pairs a template with that choice.

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
- It runs text models of twenty-five architectures, and BERT and nomic-bert
  models for embeddings and reranking. No image or audio input.
- It knows sixteen chat formats and does not interpret Jinja templates.
- It reads one GGUF file per model. Models split across several files are
  not loaded.
- It is little-endian only, like the GGUF files themselves.

[TODO.md](TODO.md) has the full list of what is left.

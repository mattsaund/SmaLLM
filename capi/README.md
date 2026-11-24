# The SmaLLM library

The engine as a library that any language can call. It comes in two forms
with the same functions:

| File | Use it when |
|---|---|
| `libsmallm.so`, `libsmallm.dylib`, `smallm.dll` | The program loads the engine at run time. This is what bindings in other languages use. |
| `libsmallm.a`, `smallm.lib` | The engine is linked into the program, which then ships as one file. |

[include/smallm.h](include/smallm.h) declares every function and documents
it. This page shows how the pieces fit.

## Get it

Each [release](https://github.com/mattsaund/SmaLLM/releases) has a
`libsmallm-<target>` package for Linux, macOS and Windows, on x86-64 and
ARM64. Or build it:

```sh
cargo build --release -p smallm-capi
```

The libraries land in `target/release`. The shared library is about 0.6 MB
and depends on nothing but the system's C library.

## From C

```c
#include <stdio.h>
#include "smallm.h"

static bool print(const char *text, size_t len, void *user) {
    (void)user;
    fwrite(text, 1, len, stdout);
    return true;
}

int main(void) {
    smallm_model *model = smallm_model_load("model.gguf");
    if (!model) {
        fprintf(stderr, "%s\n", smallm_last_error());
        return 1;
    }
    smallm_session *session = smallm_session_new(model, NULL);

    const char *roles[] = {"user"};
    const char *contents[] = {"Why is the sky blue?"};
    smallm_chat(session, roles, contents, 1, NULL, print, NULL, NULL);

    smallm_session_free(session);
    smallm_model_free(model);
    return 0;
}
```

```sh
cc chat.c -Iinclude -Llib -lsmallm -o chat
```

[examples/chat.c](examples/chat.c) is a complete chat program that keeps a
conversation going.

## From Python

[smallm.py](../bindings/python/smallm.py) is one file with no requirements
beyond Python itself. Put it next to your program with the shared library,
or set `SMALLM_LIB` to the library's path.

```python
import smallm

model = smallm.Model("model.gguf")
session = model.session()

messages = [{"role": "user", "content": "Why is the sky blue?"}]
answer = session.chat(messages, on_text=lambda piece: print(piece, end="", flush=True))

messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": "And at sunset?"}]
session.chat(messages, on_text=lambda piece: print(piece, end="", flush=True))
```

The second call only reads the new messages. The session still holds the
first exchange.

## From any other language

Everything a binding needs is in the header: opaque pointers, plain structs
of numbers, NUL-terminated UTF-8 strings and one callback type. The Python
binding is about 400 lines and is a fair model for one in another language.

## The parts

| Object | Made by | Holds |
|---|---|---|
| `smallm_model` | `smallm_model_load` | The model file, mapped into memory. Shared by its sessions. |
| `smallm_session` | `smallm_session_new` | One conversation: the context, its cache and the threads. |
| `smallm_sampler` | `smallm_sampler_new` | The state of a random token choice. Only for the low-level loop. |

There are two ways to run a model.

**The short way.** `smallm_generate` completes a text, and `smallm_chat`
answers a conversation in the model's own chat format. Both stream the
output to a callback and fill in a `smallm_stats`.

**The low-level loop.** `smallm_tokenize`, `smallm_eval`, `smallm_logits`,
`smallm_sample` and `smallm_token_to_piece` are the steps the short way is
made of. Use them to drive generation yourself.

## Rules

- **Errors.** A function that can fail returns `NULL` or a negative
  `smallm_status`. `smallm_last_error()` then has the reason, for the
  calling thread.
- **Threads.** A model can be used from several threads at once. A session
  or a sampler belongs to one thread at a time. A session runs its own
  worker threads, so one session already uses every core it is given.
- **Memory.** Whatever a `_load` or `_new` function returns is released by
  its `_free` function. A model may be freed while its sessions live.
  Strings and buffers you pass in are only read during the call.
- **Text.** Strings are UTF-8. The callback's text is not NUL-terminated.
  Use its length.
- **Devices.** `smallm_session_params.device` selects where to compute.
  Device 0 is the CPU and the only one so far.
- **Stability.** Until version 1.0 the functions may change between
  versions. `smallm_version()` says which one is loaded.

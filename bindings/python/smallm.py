"""Python binding for SmaLLM, on the C API.

One file, standard library only. Copy it next to your program, or put its
folder on PYTHONPATH.

    import smallm

    model = smallm.Model("model.gguf")
    session = model.session()
    answer = session.chat([{"role": "user", "content": "Hello!"}])
    print(answer)

Pass on_text=callback to chat or generate to receive the answer piece by
piece while it is written.

The shared library is looked up in this order: the SMALLM_LIB environment
variable, the folder of this file, then the system's library path.
"""

import ctypes
import ctypes.util
import os
import sys

__all__ = ["Model", "Session", "SmallmError", "version"]


class SmallmError(RuntimeError):
    """A failure reported by the engine."""


class _SessionParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_threads", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("n_threads_batch", ctypes.c_uint32),
    ]


class _SamplerParams(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("repeat_last_n", ctypes.c_int32),
        ("seed", ctypes.c_uint64),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
    ]


class _GenerateParams(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int32),
        ("sampler", _SamplerParams),
        ("stop", ctypes.POINTER(ctypes.c_char_p)),
        ("n_stop", ctypes.c_size_t),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("prompt_tokens", ctypes.c_int32),
        ("reused_tokens", ctypes.c_int32),
        ("generated_tokens", ctypes.c_int32),
        ("prompt_seconds", ctypes.c_double),
        ("generate_seconds", ctypes.c_double),
        ("finish", ctypes.c_int32),
    ]


_TextCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_void_p)

# The names of smallm_finish values, by number.
_FINISH = ["end", "length", "stop", "context", "canceled"]


def _library_names():
    """File names the shared library may have on this system."""
    if sys.platform == "win32":
        return ["smallm.dll"]
    if sys.platform == "darwin":
        return ["libsmallm.dylib"]
    return ["libsmallm.so"]


def _load_library():
    """Finds and loads the shared library, and declares its functions."""
    candidates = []
    if os.environ.get("SMALLM_LIB"):
        candidates.append(os.environ["SMALLM_LIB"])
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.join(here, name) for name in _library_names()]
    found = ctypes.util.find_library("smallm")
    if found:
        candidates.append(found)
    candidates += _library_names()

    lib = None
    errors = []
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            break
        except OSError as error:
            errors.append(f"{path}: {error}")
    if lib is None:
        raise SmallmError("the SmaLLM library was not found. Set SMALLM_LIB to its path.\n" + "\n".join(errors))

    c_void, c_char, c_int, c_size = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_size_t
    signatures = {
        "smallm_version": ([], c_char),
        "smallm_last_error": ([], c_char),
        "smallm_model_load": ([c_char], c_void),
        "smallm_model_free": ([c_void], None),
        "smallm_model_n_vocab": ([c_void], c_int),
        "smallm_model_n_ctx_train": ([c_void], c_int),
        "smallm_model_n_params": ([c_void], ctypes.c_uint64),
        "smallm_model_meta": ([c_void, c_char, ctypes.c_char_p, c_size], c_size),
        "smallm_model_has_chat_format": ([c_void], ctypes.c_bool),
        "smallm_tokenize": (
            [c_void, c_char, c_size, ctypes.POINTER(c_int), c_size, ctypes.c_bool, ctypes.c_bool],
            c_int,
        ),
        "smallm_token_to_piece": ([c_void, c_int, ctypes.c_char_p, c_size, ctypes.c_bool], c_int),
        "smallm_token_is_end": ([c_void, c_int], ctypes.c_bool),
        "smallm_session_new": ([c_void, ctypes.POINTER(_SessionParams)], c_void),
        "smallm_session_free": ([c_void], None),
        "smallm_session_n_ctx": ([c_void], c_int),
        "smallm_session_n_tokens": ([c_void], c_int),
        "smallm_session_truncate": ([c_void, c_size], None),
        "smallm_eval": ([c_void, ctypes.POINTER(c_int), c_size], c_int),
        "smallm_logits": ([c_void], ctypes.POINTER(ctypes.c_float)),
        "smallm_sampler_default_params": ([], _SamplerParams),
        "smallm_generate": (
            [c_void, c_char, ctypes.POINTER(_GenerateParams), _TextCallback, c_void, ctypes.POINTER(_Stats)],
            c_int,
        ),
        "smallm_chat": (
            [
                c_void,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_char_p),
                c_size,
                ctypes.POINTER(_GenerateParams),
                _TextCallback,
                c_void,
                ctypes.POINTER(_Stats),
            ],
            c_int,
        ),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = restype
    return lib


_lib = _load_library()


def _last_error():
    return (_lib.smallm_last_error() or b"unknown error").decode("utf-8", "replace")


def version():
    """The version of the engine, such as "0.1.0"."""
    return _lib.smallm_version().decode()


class Model:
    """A model file, opened and ready to run."""

    def __init__(self, path):
        self._handle = _lib.smallm_model_load(os.fsencode(path))
        if not self._handle:
            raise SmallmError(_last_error())

    def close(self):
        """Releases the model. Sessions made from it keep working."""
        if self._handle:
            _lib.smallm_model_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def n_vocab(self):
        """Number of tokens in the vocabulary."""
        return _lib.smallm_model_n_vocab(self._handle)

    @property
    def n_ctx_train(self):
        """The context length the model was trained with."""
        return _lib.smallm_model_n_ctx_train(self._handle)

    @property
    def n_params(self):
        """Total number of weights."""
        return _lib.smallm_model_n_params(self._handle)

    @property
    def has_chat_format(self):
        """True when Session.chat works with this model."""
        return _lib.smallm_model_has_chat_format(self._handle)

    def meta(self, key):
        """A metadata value of the file as text, or None when it is missing."""
        needed = _lib.smallm_model_meta(self._handle, key.encode(), None, 0)
        if needed == 0:
            return None
        buffer = ctypes.create_string_buffer(needed + 1)
        _lib.smallm_model_meta(self._handle, key.encode(), buffer, needed + 1)
        return buffer.value.decode("utf-8", "replace")

    def tokenize(self, text, add_special=True, parse_special=False):
        """Converts text to a list of token ids."""
        data = text.encode("utf-8")
        capacity = len(data) + 16
        while True:
            tokens = (ctypes.c_int32 * capacity)()
            count = _lib.smallm_tokenize(self._handle, data, len(data), tokens, capacity, add_special, parse_special)
            if count < 0:
                raise SmallmError(_last_error())
            if count <= capacity:
                return list(tokens[:count])
            capacity = count

    def piece(self, token, special=False):
        """The bytes a token stands for."""
        buffer = ctypes.create_string_buffer(64)
        needed = _lib.smallm_token_to_piece(self._handle, token, buffer, len(buffer), special)
        if needed > len(buffer):
            buffer = ctypes.create_string_buffer(needed)
            needed = _lib.smallm_token_to_piece(self._handle, token, buffer, len(buffer), special)
        return buffer.raw[: max(needed, 0)]

    def is_end(self, token):
        """True when the token ends a generation."""
        return _lib.smallm_token_is_end(self._handle, token)

    def session(self, n_ctx=0, threads=0, batch=0):
        """Starts an empty conversation. Zero lets the engine choose."""
        return Session(self, n_ctx, threads, batch)


class Session:
    """One conversation with a model."""

    def __init__(self, model, n_ctx=0, threads=0, batch=0):
        self.model = model
        params = _SessionParams(n_ctx, threads, batch, 0)
        self._handle = _lib.smallm_session_new(model._handle, ctypes.byref(params))
        if not self._handle:
            raise SmallmError(_last_error())
        #: What the last generation did, as a dict.
        self.stats = None

    def close(self):
        """Releases the session."""
        if self._handle:
            _lib.smallm_session_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def n_ctx(self):
        """The context window in tokens."""
        return _lib.smallm_session_n_ctx(self._handle)

    @property
    def n_tokens(self):
        """Number of tokens in the context."""
        return _lib.smallm_session_n_tokens(self._handle)

    def clear(self):
        """Empties the context."""
        _lib.smallm_session_truncate(self._handle, 0)

    def eval(self, tokens):
        """Appends tokens to the context."""
        array = (ctypes.c_int32 * len(tokens))(*tokens)
        if _lib.smallm_eval(self._handle, array, len(tokens)) != 0:
            raise SmallmError(_last_error())

    def logits(self):
        """The scores for the next token, one per vocabulary entry."""
        pointer = _lib.smallm_logits(self._handle)
        return pointer[: self.model.n_vocab]

    def _params(self, max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, seed, stop):
        """Builds the generation options. Keeps the stop strings alive."""
        sampler = _lib.smallm_sampler_default_params()
        if temperature is not None:
            sampler.temperature = temperature
        if top_k is not None:
            sampler.top_k = top_k
        if top_p is not None:
            sampler.top_p = top_p
        if min_p is not None:
            sampler.min_p = min_p
        if repeat_penalty is not None:
            sampler.repeat_penalty = repeat_penalty
        # A fresh seed per call unless one is given, so answers vary.
        sampler.seed = int.from_bytes(os.urandom(8), "little") if seed is None else seed
        stop = [stop] if isinstance(stop, str) else list(stop or [])
        stop_array = (ctypes.c_char_p * len(stop))(*[text.encode("utf-8") for text in stop])
        return _GenerateParams(max_tokens, sampler, stop_array, len(stop)), stop_array

    def _run(self, call, on_text):
        """Runs a generation, passing each piece of text to on_text."""
        pending = bytearray()

        def receive(text, length, _user):
            # A piece always ends on a whole character, so this decodes.
            pending.extend(ctypes.string_at(text, length))
            piece = pending.decode("utf-8", "replace")
            pending.clear()
            return on_text(piece) is not False

        stats = _Stats()
        status = call(_TextCallback(receive), ctypes.byref(stats))
        if status != 0:
            raise SmallmError(_last_error())
        self.stats = {
            "prompt_tokens": stats.prompt_tokens,
            "reused_tokens": stats.reused_tokens,
            "generated_tokens": stats.generated_tokens,
            "prompt_seconds": stats.prompt_seconds,
            "generate_seconds": stats.generate_seconds,
            "finish": _FINISH[stats.finish] if 0 <= stats.finish < len(_FINISH) else "unknown",
        }

    def generate(self, prompt, on_text=None, max_tokens=0, temperature=None, top_k=None, top_p=None, min_p=None,
                 repeat_penalty=None, seed=None, stop=None):
        """Completes a plain text prompt and returns the completion.

        on_text, when given, receives each piece as it is produced. Return
        False from it to stop early.
        """
        params, _keep = self._params(max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, seed, stop)
        pieces = []

        def collect(piece):
            pieces.append(piece)
            return on_text(piece) if on_text else True

        self._run(
            lambda callback, stats: _lib.smallm_generate(
                self._handle, prompt.encode("utf-8"), ctypes.byref(params), callback, None, stats
            ),
            collect,
        )
        return "".join(pieces)

    def chat(self, messages, on_text=None, max_tokens=0, temperature=None, top_k=None, top_p=None, min_p=None,
             repeat_penalty=None, seed=None, stop=None):
        """Answers a conversation and returns the answer.

        messages is a list of {"role": ..., "content": ...} dicts, with the
        roles "system", "user" and "assistant". The context is kept between
        calls, so only new messages cost time.
        """
        params, _keep = self._params(max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, seed, stop)
        count = len(messages)
        roles = (ctypes.c_char_p * count)(*[message["role"].encode("utf-8") for message in messages])
        contents = (ctypes.c_char_p * count)(*[message["content"].encode("utf-8") for message in messages])
        pieces = []

        def collect(piece):
            pieces.append(piece)
            return on_text(piece) if on_text else True

        self._run(
            lambda callback, stats: _lib.smallm_chat(
                self._handle, roles, contents, count, ctypes.byref(params), callback, None, stats
            ),
            collect,
        )
        return "".join(pieces)


if __name__ == "__main__":
    # A small demonstration: python3 smallm.py model.gguf "your prompt"
    if len(sys.argv) < 3:
        sys.exit("usage: python3 smallm.py model.gguf prompt")
    def show(piece):
        print(piece, end="", flush=True)

    with Model(sys.argv[1]) as demo_model, demo_model.session() as demo_session:
        if demo_model.has_chat_format:
            demo_session.chat([{"role": "user", "content": sys.argv[2]}], on_text=show, max_tokens=256)
        else:
            demo_session.generate(sys.argv[2], on_text=show, max_tokens=256)
        print()
        print(demo_session.stats, file=sys.stderr)

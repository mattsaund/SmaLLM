"""Checks the Python binding against a real model.

    SMALLM_LIB=path/to/libsmallm.so python3 test_smallm.py model.gguf

Exits with an error when a check fails. CI runs this on every platform.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smallm


def check(model_path):
    """Runs every part of the binding once."""
    print("engine", smallm.version())
    model = smallm.Model(model_path)
    assert model.n_vocab > 0 and model.n_ctx_train > 0 and model.n_params > 0
    assert model.meta("general.architecture")
    assert model.meta("no.such.key") is None

    # Text survives the trip to tokens and back.
    text = "Once upon a time"
    tokens = model.tokenize(text, add_special=False)
    assert len(tokens) > 1
    assert b"".join(model.piece(token) for token in tokens).decode().strip() == text

    session = model.session(n_ctx=512, threads=2)
    assert session.n_ctx == 512

    # Low level: evaluate tokens and read the scores for the next one.
    session.eval(tokens)
    assert session.n_tokens == len(tokens)
    logits = session.logits()
    assert len(logits) == model.n_vocab
    session.clear()
    assert session.n_tokens == 0

    # High level: the same seedless greedy run gives the same text twice,
    # and the pieces handed to the callback add up to the result.
    pieces = []
    first = session.generate(text, on_text=pieces.append, max_tokens=24, temperature=0)
    assert first and "".join(pieces) == first
    assert session.stats["generated_tokens"] > 0
    assert session.generate(text, max_tokens=24, temperature=0) == first
    print("generate:", repr(first))

    # A callback that returns False stops the run.
    session.generate(text, on_text=lambda piece: False, max_tokens=24, temperature=0)
    assert session.stats["finish"] == "canceled"

    if model.has_chat_format:
        answer = session.chat([{"role": "user", "content": "Name one color."}], max_tokens=16, temperature=0)
        assert answer.strip()
        print("chat:", repr(answer))
    else:
        try:
            session.chat([{"role": "user", "content": "Hello"}])
        except smallm.SmallmError as error:
            print("chat refused as expected:", error)
        else:
            raise AssertionError("a model without a chat format answered a chat")

    # Failures arrive as exceptions with the engine's message.
    try:
        smallm.Model("/no/such/model.gguf")
    except smallm.SmallmError as error:
        assert "model.gguf" in str(error)
    else:
        raise AssertionError("a missing file loaded")

    session.close()
    model.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 test_smallm.py model.gguf [more.gguf ...]")
    for path in sys.argv[1:]:
        check(path)
    print("ok")

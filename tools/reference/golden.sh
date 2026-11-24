#!/bin/sh
# Writes the golden files in tests/golden from a reference llama.cpp build.
#
#   tools/reference/golden.sh REF MODELS_DIR
#
# REF is the harness built from ref.c (see README.md in this folder).
# MODELS_DIR holds the model files named in the table below. Models that are
# missing are skipped, and their golden files are left as they are.
set -eu

REF=$1
MODELS=$2
HERE=$(cd "$(dirname "$0")/../.." && pwd)
GOLDEN="$HERE/tests/golden"

# model file, prompt file, tokens to generate
while read -r model prompt count; do
    case "$model" in ''|'#'*) continue ;; esac
    path="$MODELS/$model"
    if [ ! -f "$path" ]; then
        echo "skipping $model (not in $MODELS)"
        continue
    fi
    out="$GOLDEN/${model%.gguf}.golden"
    {
        echo "model $model"
        echo "tokens plain $("$REF" tok "$path" "$GOLDEN/texts/mixed.txt" 2>/dev/null)"
        echo "tokens special $("$REF" tok "$path" "$GOLDEN/texts/mixed.txt" special 2>/dev/null)"
        echo "prompt $prompt"
        "$REF" gen "$path" "$GOLDEN/prompts/$prompt" "$count" 4 2>/dev/null \
            | sed -n -e 's/^prompt: /context /p' -e 's/^generated: /greedy /p'
    } > "$out"
    echo "wrote $out"
done <<'TABLE'
stories260K.gguf                           plain.txt   48
stories15M-q4_0.gguf                       plain.txt   48
SmolLM2-135M-Instruct-Q4_K_M.gguf          chatml.txt  32
qwen2.5-0.5b-instruct-q4_k_m.gguf          chatml.txt  32
Qwen3-0.6B-Q4_K_M.gguf                     chatml.txt  32
gemma-3-270m-it-Q4_K_M.gguf                gemma.txt   32
Llama-3.2-1B-Instruct-Q4_K_M.gguf          llama3.txt  32
tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf       zephyr.txt  32
TABLE

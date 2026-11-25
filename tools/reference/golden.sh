#!/bin/sh
# Writes the golden files in tests/golden from a reference llama.cpp build.
#
#   tools/reference/golden.sh REF MODELS_DIR [MODEL]
#
# REF is the harness built from ref.c (see README.md in this folder).
# MODELS_DIR holds the model files named in the table below. Models that are
# missing are skipped, and their golden files are left as they are. Naming
# one MODEL file writes only its golden file.
set -eu

REF=$1
MODELS=$2
ONLY=${3:-}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
GOLDEN="$HERE/tests/golden"

# model file, prompt file, tokens to generate
while read -r model prompt count; do
    case "$model" in ''|'#'*) continue ;; esac
    if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then
        continue
    fi
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
LFM2.5-350M-Q4_K_M.gguf                    chatml.txt  32
LFM2.5-1.2B-Instruct-Q4_K_M.gguf           chatml.txt  32
gemma-4-E2B-it-Q4_0.gguf                   gemma4.txt  32
LFM2-8B-A1B-Q4_K_M.gguf                    chatml.txt  32
granite-3.1-1b-a400m-instruct-Q4_K_M.gguf  granite.txt 32
gemma-2-2b-it-Q4_K_M.gguf                  gemma.txt   32
microsoft_Phi-4-mini-instruct-Q4_K_M.gguf  phi4-mini.txt 32
phi-2.Q4_K_M.gguf                          plain.txt   32
gpt-oss-20b-MXFP4.gguf                     harmony.txt 32
mamba-370m-hf.Q8_0.gguf                    plain.txt   48
mamba2-130m-hf-q8_0.gguf                   plain.txt   48
granite-4.0-h-350m-Q4_K_M.gguf             granite.txt 32
granite-4.0-h-tiny-Q4_K_M.gguf             granite.txt 32
rwkv7-g1d-0.1b-Q8_0.gguf                   rwkv-world.txt 32
rwkv-6-world-1.6b-Q4_K.gguf                rwkv-world.txt 32
DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf deepseek2.txt 32
Moonlight-16B-A3B-Instruct-Q4_K_M.gguf     kimi.txt    32
deepseek-moe-16b-chat.Q4_K_M.gguf          deepseek2.txt 32
Inkling-2-Mini-UD-Q2_K_XL.gguf             inkling.txt 32
TABLE

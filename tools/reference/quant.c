// Test vectors for the weight formats: blocks quantized by ggml, and the
// floats ggml's own code expands them to.
//
//   quant OUTDIR
//
// For each format in TYPES, this quantizes pseudo-random rows with
// ggml_quantize_chunk and expands the blocks again with the format's
// to_float. OUTDIR/<name>.bin then holds, all little-endian:
//
//   the number of elements, as a 32-bit integer
//   the quantized blocks
//   ggml's float for each element, 32 bits each
//
// tests/quant.rs checks that SmaLLM expands the same blocks to the same
// floats. Only ggml's base library is needed. README.md in this folder
// says how to build and run it.
#include "ggml.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Rows per format, and elements per row. A row is a whole number of
// blocks for every format. Two rows keep the files small and still use
// every field of a block. The IQ grids are checked in full by a test of
// their own, in src/tensor/quant.rs.
#define ROWS 2
#define COLS 256

// Every quantized format SmaLLM reads. A new format goes at the end: the
// rows come from one stream of random numbers, so the formats before it
// keep the vectors they have.
static const enum ggml_type TYPES[] = {
    GGML_TYPE_Q4_0,    GGML_TYPE_Q4_1,   GGML_TYPE_Q5_0,  GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q2_K,    GGML_TYPE_Q3_K,   GGML_TYPE_Q4_K,  GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ1_S,   GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_TQ1_0,   GGML_TYPE_TQ2_0,
    GGML_TYPE_MXFP4,
    GGML_TYPE_NVFP4,   GGML_TYPE_Q1_0,   GGML_TYPE_Q2_0,
};

// The formats that round each weight to -1, 0 or 1 times the largest
// magnitude of its block: the TQ formats, and Q2_0, whose fourth level
// ggml's quantizer never uses. They get the ternary weights of `fill`, or
// most of their weights would be zero.
static int is_ternary(enum ggml_type type) {
    return type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0;
}

// A small xorshift generator, so that every run writes the same vectors.
static uint64_t state = 0x9e3779b97f4a7c15ull;

static uint64_t next_random(void) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// A float in [-1, 1).
static float uniform(void) {
    return (float) (next_random() >> 40) / (float) (1 << 23) - 1.0f;
}

// Fills `x` with `n` weights. Each run of 256 gets a magnitude from about
// 1/20 to 20, so the blocks have different scales. Each run of 32 in it
// varies that by up to 1.6 times either way, so the scales inside a block
// differ too. One run of 32 in every eight is zero, which gives some
// groups a scale of zero.
//
// With `ternary`, the weights are near -1, 0 or 1 times the magnitude of
// their run of 256 instead, like those of the models the TQ formats are
// made for. Noise would round almost all of them to zero.
static void fill(float * x, int n, int ternary) {
    float block = 1.0f;
    for (int i = 0; i < n; i += 32) {
        if (i % 256 == 0) block = expf(3.0f * uniform());
        if (ternary) {
            for (int j = 0; j < 32; j++) {
                float u = uniform();
                float level = u < -1.0f / 3 ? -1.0f : u < 1.0f / 3 ? 0.0f : 1.0f;
                x[i + j] = block * (level + 0.1f * uniform());
            }
            continue;
        }
        float magnitude = (i / 32) % 8 == 5 ? 0.0f : block * expf(0.5f * uniform());
        for (int j = 0; j < 32; j++) {
            x[i + j] = magnitude * uniform();
        }
    }
}

// Writes a 32-bit integer, least significant byte first.
static void put_u32(FILE * f, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char) value, (unsigned char) (value >> 8),
        (unsigned char) (value >> 16), (unsigned char) (value >> 24),
    };
    fwrite(bytes, 1, 4, f);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: quant OUTDIR\n");
        return 2;
    }
    const int n = ROWS * COLS;
    float * source = malloc(n * sizeof(float));
    float * expanded = malloc(n * sizeof(float));
    // The formats that need an importance matrix get a uniform one.
    float * importance = malloc(COLS * sizeof(float));
    for (int i = 0; i < COLS; i++) importance[i] = 1.0f;

    for (size_t t = 0; t < sizeof TYPES / sizeof TYPES[0]; t++) {
        const enum ggml_type type = TYPES[t];
        fill(source, n, is_ternary(type));
        const size_t row_size = ggml_row_size(type, COLS);
        uint8_t * blocks = malloc(ROWS * row_size);
        ggml_quantize_init(type);
        const float * imatrix = ggml_quantize_requires_imatrix(type) ? importance : NULL;
        const size_t size = ggml_quantize_chunk(type, source, blocks, 0, ROWS, COLS, imatrix);
        if (size != ROWS * row_size) {
            fprintf(stderr, "%s: quantized to %zu bytes, expected %zu\n",
                    ggml_type_name(type), size, ROWS * row_size);
            return 1;
        }
        ggml_get_type_traits(type)->to_float(blocks, expanded, n);

        // Files are named in lowercase: ggml writes q4_K, but iq4_xs.
        char name[64];
        snprintf(name, sizeof name, "%s", ggml_type_name(type));
        for (char * c = name; *c; c++) *c = (char) tolower((unsigned char) *c);
        char path[4096];
        snprintf(path, sizeof path, "%s/%s.bin", argv[1], name);
        FILE * f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", path);
            return 1;
        }
        put_u32(f, (uint32_t) n);
        fwrite(blocks, 1, size, f);
        for (int i = 0; i < n; i++) {
            uint32_t bits;
            memcpy(&bits, &expanded[i], sizeof bits);
            put_u32(f, bits);
        }
        fclose(f);
        printf("wrote %s\n", path);
        free(blocks);
    }
    ggml_quantize_free();
    free(source);
    free(expanded);
    free(importance);
    return 0;
}

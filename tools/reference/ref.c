// Reference harness: drives the llama.cpp library that is already installed
// on this machine, to produce ground truth for SmaLLM tests and benchmarks.
//
//   ref tok    MODEL TEXTFILE [special]        token ids of the text
//   ref tokraw MODEL TEXTFILE                  the same, without start and end tokens
//   ref gen    MODEL TEXTFILE N [threads]      greedy generation: ids and text
//   ref logits MODEL TEXTFILE OUTFILE          raw f32 logits after the prompt
//   ref ids    MODEL IDSFILE OUTFILE           raw f32 logits after the token
//                                              ids listed in IDSFILE
//   ref embed  MODEL TEXTFILE OUTFILE [rank]   an encoder's pooled output as
//                                              raw f32: an embedding, or with
//                                              rank, a reranker's scores
//   ref bench  MODEL N_PROMPT N_GEN THREADS    prompt and generation speed
//   ref mem    MODEL N_CTX N_TOKENS THREADS    memory in use after loading,
//                                              with a context, and after
//                                              reading tokens (Linux only)
#include "llama.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void quiet_log(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static char * read_file(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t) n) { fprintf(stderr, "short read\n"); exit(1); }
    buf[n] = 0;
    fclose(f);
    *len = (size_t) n;
    return buf;
}

// Prints what this process holds in memory, from /proc/self/status.
// "own" is memory the process allocated. "file" is the part of the mapped
// model file that is in memory, which the system can drop and read again.
static void print_memory(const char * when) {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) { printf("%s: no /proc/self/status on this system\n", when); return; }
    char line[256];
    long own = 0, file = 0, shared = 0;
    while (fgets(line, sizeof line, f)) {
        sscanf(line, "RssAnon: %ld", &own);
        sscanf(line, "RssFile: %ld", &file);
        // A model on a memory file system is counted here instead.
        sscanf(line, "RssShmem: %ld", &shared);
    }
    fclose(f);
    printf("%-18s own %8.1f MB   file %8.1f MB\n", when, own / 1024.0, (file + shared) / 1024.0);
}

static int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see the top of ref.c\n"); return 2; }
    const char * mode = argv[1];
    const char * model_path = argv[2];

    llama_log_set(quiet_log, NULL);
    const char * cpu = getenv("REF_CPU_BACKEND");
    if (!cpu) { fprintf(stderr, "set REF_CPU_BACKEND to a libggml-cpu-*.so\n"); return 2; }
    if (!ggml_backend_load(cpu)) { fprintf(stderr, "cannot load backend %s\n", cpu); return 1; }
    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    /* Tokenizing needs only the vocabulary, so a vocabulary file will do. */
    const int tokenize_only = strcmp(mode, "tok") == 0 || strcmp(mode, "tokraw") == 0;
    mp.vocab_only = tokenize_only;
    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "cannot load model\n"); return 1; }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (strcmp(mode, "mem") == 0) {
        if (argc < 6) { fprintf(stderr, "mem MODEL N_CTX N_TOKENS THREADS\n"); return 2; }
        int n_ctx = atoi(argv[3]), n_tokens = atoi(argv[4]), threads = atoi(argv[5]);
        print_memory("model loaded");
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.no_perf = true;
        struct llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "cannot make context\n"); return 1; }
        print_memory("context ready");
        llama_token * toks = malloc(sizeof(llama_token) * (n_tokens + 1));
        for (int i = 0; i < n_tokens; i++) toks[i] = 1 + (i * 7919) % (n_vocab > 1000 ? 1000 : n_vocab - 1);
        for (int done = 0; done < n_tokens; done += 32) {
            int n = n_tokens - done < 32 ? n_tokens - done : 32;
            if (llama_decode(ctx, llama_batch_get_one(toks + done, n))) { fprintf(stderr, "decode failed\n"); return 1; }
        }
        print_memory("tokens read");
        return 0;
    }

    if (strcmp(mode, "bench") == 0) {
        if (argc < 6) { fprintf(stderr, "bench MODEL N_PROMPT N_GEN THREADS\n"); return 2; }
        int n_prompt = atoi(argv[3]), n_gen = atoi(argv[4]), threads = atoi(argv[5]);
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_prompt + n_gen + 8;
        cp.n_batch = n_prompt > 512 ? n_prompt : 512;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.no_perf = true;
        struct llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "cannot make context\n"); return 1; }
        llama_token * toks = malloc(sizeof(llama_token) * (n_prompt + 1));
        // The same synthetic prompt SmaLLM's bench uses: token ids cycling
        // through the low part of the vocabulary.
        for (int i = 0; i < n_prompt; i++) toks[i] = 1 + (i * 7919) % (n_vocab > 1000 ? 1000 : n_vocab - 1);
        // Warm up: page the weights in and start the threads.
        llama_decode(ctx, llama_batch_get_one(toks, n_prompt < 8 ? n_prompt : 8));
        llama_memory_clear(llama_get_memory(ctx), true);

        double t0 = now_s();
        if (llama_decode(ctx, llama_batch_get_one(toks, n_prompt))) { fprintf(stderr, "decode failed\n"); return 1; }
        double t1 = now_s();
        llama_token next = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        for (int i = 0; i < n_gen; i++) {
            if (llama_decode(ctx, llama_batch_get_one(&next, 1))) { fprintf(stderr, "decode failed\n"); return 1; }
            next = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        double t2 = now_s();
        printf("threads %d  prompt %d tok %.2f tok/s  gen %d tok %.2f tok/s\n",
               threads, n_prompt, n_prompt / (t1 - t0), n_gen, n_gen / (t2 - t1));
        return 0;
    }

    if (strcmp(mode, "embed") == 0) {
        if (argc < 5) { fprintf(stderr, "embed MODEL TEXTFILE OUTFILE\n"); return 2; }
        size_t len;
        char * text = read_file(argv[3], &len);
        int cap = (int) len + 16;
        llama_token * toks = malloc(sizeof(llama_token) * cap);
        int n = llama_tokenize(vocab, text, (int) len, toks, cap, true, true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n + 8;
        cp.n_batch = n + 8;
        cp.n_ubatch = n + 8;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.embeddings = true;
        cp.no_perf = true;
        /* A reranker is only ranked when asked, as llama-server's --reranking does. */
        if (argc > 5 && strcmp(argv[5], "rank") == 0) cp.pooling_type = LLAMA_POOLING_TYPE_RANK;
        struct llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "cannot make context\n"); return 1; }
        if (llama_encode(ctx, llama_batch_get_one(toks, n))) { fprintf(stderr, "encode failed\n"); return 1; }
        const float * embedding = llama_get_embeddings_seq(ctx, 0);
        if (!embedding) { fprintf(stderr, "no pooled output\n"); return 1; }
        int width = llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_RANK
            ? (int) llama_model_n_cls_out(model) : llama_model_n_embd(model);
        FILE * out = fopen(argv[4], "wb");
        fwrite(embedding, sizeof(float), width, out);
        fclose(out);
        printf("tokens:");
        for (int i = 0; i < n; i++) printf(" %d", toks[i]);
        printf("\n%d values\n", width);
        return 0;
    }

    if (strcmp(mode, "ids") == 0) {
        if (argc < 5) { fprintf(stderr, "ids MODEL IDSFILE OUTFILE\n"); return 2; }
        size_t len;
        char * list = read_file(argv[3], &len);
        llama_token * toks = malloc(sizeof(llama_token) * (len + 1));
        int n = 0;
        for (char * word = strtok(list, " \n"); word; word = strtok(NULL, " \n")) toks[n++] = atoi(word);
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n + 8;
        cp.n_batch = n > 512 ? n : 512;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.no_perf = true;
        struct llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "cannot make context\n"); return 1; }
        if (llama_decode(ctx, llama_batch_get_one(toks, n))) { fprintf(stderr, "decode failed\n"); return 1; }
        FILE * out = fopen(argv[4], "wb");
        fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), n_vocab, out);
        fclose(out);
        printf("%d tokens, %d logits\n", n, n_vocab);
        return 0;
    }

    size_t text_len;
    char * text = read_file(argv[3], &text_len);
    const int special = (argc > 4 && strcmp(argv[4], "special") == 0) || !tokenize_only;
    /* tokraw adds no start and end tokens, like the tokenizer test files. */
    const int add_special = strcmp(mode, "tokraw") != 0;

    int cap = (int) text_len + 16;
    llama_token * toks = malloc(sizeof(llama_token) * cap);
    int n = llama_tokenize(vocab, text, (int) text_len, toks, cap, add_special, special);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    if (tokenize_only) {
        for (int i = 0; i < n; i++) printf("%d%c", toks[i], i + 1 < n ? ' ' : '\n');
        if (n == 0) printf("\n");
        return 0;
    }

    int n_gen = (strcmp(mode, "gen") == 0 && argc > 4) ? atoi(argv[4]) : 0;
    int threads = (strcmp(mode, "gen") == 0 && argc > 5) ? atoi(argv[5]) : 4;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + n_gen + 8;
    cp.n_batch = n > 512 ? n : 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = true;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot make context\n"); return 1; }

    if (llama_decode(ctx, llama_batch_get_one(toks, n))) { fprintf(stderr, "decode failed\n"); return 1; }
    const float * logits = llama_get_logits_ith(ctx, -1);

    if (strcmp(mode, "logits") == 0) {
        FILE * out = fopen(argv[4], "wb");
        fwrite(logits, sizeof(float), n_vocab, out);
        fclose(out);
        printf("prompt:");
        for (int i = 0; i < n; i++) printf(" %d", toks[i]);
        printf("\nargmax %d\n", argmax(logits, n_vocab));
        return 0;
    }

    printf("prompt:");
    for (int i = 0; i < n; i++) printf(" %d", toks[i]);
    printf("\ngenerated:");
    char piece[256];
    char * out_text = malloc(n_gen * 64 + 1);
    size_t out_len = 0;
    for (int i = 0; i < n_gen; i++) {
        llama_token next = argmax(logits, n_vocab);
        printf(" %d", next);
        if (llama_vocab_is_eog(vocab, next)) break;
        int k = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);
        if (k > 0) { memcpy(out_text + out_len, piece, k); out_len += k; }
        if (llama_decode(ctx, llama_batch_get_one(&next, 1))) { fprintf(stderr, "decode failed\n"); return 1; }
        logits = llama_get_logits_ith(ctx, -1);
    }
    out_text[out_len] = 0;
    printf("\ntext: %s\n", out_text);
    return 0;
}

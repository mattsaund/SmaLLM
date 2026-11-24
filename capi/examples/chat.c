/*
 * A chat program on the SmaLLM C API.
 *
 *     cc chat.c -I../include -L../../target/release -lsmallm -o chat
 *     LD_LIBRARY_PATH=../../target/release ./chat model.gguf
 *
 * Type a message and press Enter. An empty line or the end of input quits.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smallm.h"

#define MAX_TURNS 64
#define MAX_TEXT 4096

/* The answer being written, kept so it can join the conversation. */
struct answer {
    char text[MAX_TEXT];
    size_t len;
};

/* Prints each piece as it arrives and appends it to the answer. */
static bool on_text(const char *text, size_t len, void *user) {
    struct answer *answer = user;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    if (answer->len + len < MAX_TEXT) {
        memcpy(answer->text + answer->len, text, len);
        answer->len += len;
        answer->text[answer->len] = '\0';
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    smallm_model *model = smallm_model_load(argv[1]);
    if (!model) {
        fprintf(stderr, "%s\n", smallm_last_error());
        return 1;
    }
    smallm_session *session = smallm_session_new(model, NULL);
    if (!session) {
        fprintf(stderr, "%s\n", smallm_last_error());
        return 1;
    }
    if (!smallm_model_has_chat_format(model)) {
        fprintf(stderr, "this model has no chat format\n");
        return 1;
    }

    const char *roles[MAX_TURNS];
    char *contents[MAX_TURNS];
    size_t turns = 0;
    char line[MAX_TEXT];

    smallm_generate_params params = smallm_generate_default_params();
    params.max_tokens = 512;

    while (turns + 2 <= MAX_TURNS) {
        printf("> ");
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') break;

        roles[turns] = "user";
        contents[turns] = strdup(line);
        turns++;

        struct answer answer = {.len = 0};
        smallm_stats stats;
        int32_t status = smallm_chat(session, roles, (const char *const *)contents, turns, &params, on_text, &answer, &stats);
        if (status != SMALLM_OK) {
            fprintf(stderr, "\n%s\n", smallm_last_error());
            break;
        }
        printf("\n[%d tokens, %.1f tokens/s]\n", stats.generated_tokens,
               stats.generated_tokens / (stats.generate_seconds > 0 ? stats.generate_seconds : 1));

        roles[turns] = "assistant";
        contents[turns] = strdup(answer.text);
        turns++;
    }

    for (size_t i = 0; i < turns; i++) free(contents[i]);
    smallm_session_free(session);
    smallm_model_free(model);
    return 0;
}

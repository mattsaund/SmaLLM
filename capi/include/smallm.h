/*
 * SmaLLM C API.
 *
 * Link against libsmallm (shared or static) and include this header.
 *
 * Rules that hold for every function:
 *
 * - Objects are opaque pointers. Each is made by a _load or _new function
 *   and released by the matching _free function.
 * - A function that can fail returns NULL or a negative smallm_status.
 *   smallm_last_error() then describes the failure for the calling thread.
 * - Strings are UTF-8. Input strings end with a NUL byte unless a length is
 *   passed alongside.
 * - Passing NULL where an object is required is reported as an error.
 * - A model may be used from several threads at once. A session or a
 *   sampler may be used from one thread at a time.
 *
 * The shortest complete program:
 *
 *     smallm_model *model = smallm_model_load("model.gguf");
 *     smallm_session *session = smallm_session_new(model, NULL);
 *     smallm_generate(session, "Once upon a time", NULL, print, NULL, NULL);
 *     smallm_session_free(session);
 *     smallm_model_free(model);
 *
 * where print is a smallm_text_callback that writes its text to stdout.
 */

#ifndef SMALLM_H
#define SMALLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A loaded model. Sessions share it, so it may be freed while they live. */
typedef struct smallm_model smallm_model;

/* One conversation with a model: its context and its scratch memory. */
typedef struct smallm_session smallm_session;

/* A token sampler. */
typedef struct smallm_sampler smallm_sampler;

/* Result codes. Zero is success, negative values are failures. */
typedef enum smallm_status {
    SMALLM_OK = 0,
    SMALLM_INVALID = -1,        /* An argument was null or out of range. */
    SMALLM_CONTEXT_FULL = -2,   /* The context window has no room left. */
    SMALLM_UNSUPPORTED = -3,    /* The model needs something not implemented. */
    SMALLM_BAD_FILE = -4,       /* The model file is missing or not valid. */
    SMALLM_NO_CHAT_FORMAT = -5  /* The model has no recognized chat format. */
} smallm_status;

/* Options for a new session. Zero means "choose for me" in every field. */
typedef struct smallm_session_params {
    uint32_t n_ctx;     /* Context window in tokens. Any size is accepted. */
    uint32_t n_threads; /* Threads to generate with. */
    uint32_t n_batch;   /* Prompt tokens evaluated together. */
    uint32_t device;    /* Where to compute. 0 is the CPU, the only device. */
    uint32_t n_threads_batch; /* Threads to read prompts with. 0 picks every
                                 physical core, or n_threads when that is set. */
    uint32_t cache_type; /* What the key/value cache keeps: 0 for 16-bit
                            floats, half the memory, 1 for 32-bit floats. */
} smallm_session_params;

/* Options for choosing tokens. */
typedef struct smallm_sampler_params {
    float temperature;     /* Randomness. 0 always picks the likeliest. */
    int32_t top_k;         /* Keep the k likeliest tokens. 0 keeps all. */
    float top_p;           /* Keep tokens within this probability mass. */
    float min_p;           /* Drop tokens below this share of the best. */
    float repeat_penalty;  /* Penalty on recent tokens. 1 for none. */
    int32_t repeat_last_n; /* Tokens the penalties look back. */
    uint64_t seed;         /* The same seed repeats the same choices. */
    float frequency_penalty; /* Taken off a score per recent use. 0 for none. */
    float presence_penalty;  /* Taken off the score of any recent token. */
} smallm_sampler_params;

/* Options for smallm_generate and smallm_chat. */
typedef struct smallm_generate_params {
    int32_t max_tokens;            /* Stop after this many. 0 for no limit. */
    smallm_sampler_params sampler;
    const char *const *stop;       /* Strings that end the output, or NULL. */
    size_t n_stop;                 /* Number of entries in stop. */
} smallm_generate_params;

/* Why a generation ended. */
typedef enum smallm_finish {
    SMALLM_FINISH_END = 0,      /* The model finished its answer. */
    SMALLM_FINISH_LENGTH = 1,   /* The token limit was reached. */
    SMALLM_FINISH_STOP = 2,     /* A stop string appeared. */
    SMALLM_FINISH_CONTEXT = 3,  /* The context window is full. */
    SMALLM_FINISH_CANCELED = 4 /* The callback returned false. */
} smallm_finish;

/* What a generation did. */
typedef struct smallm_stats {
    int32_t prompt_tokens;
    int32_t reused_tokens;    /* Prompt tokens already in the context. */
    int32_t generated_tokens;
    double prompt_seconds;
    double generate_seconds;
    int32_t finish;           /* A smallm_finish value. */
} smallm_stats;

/*
 * Receives each piece of generated text. The text is len bytes long and
 * is not NUL-terminated. Return false to stop the generation.
 */
typedef bool (*smallm_text_callback)(const char *text, size_t len, void *user);

/* The version of the library, such as "0.1.0". */
const char *smallm_version(void);

/*
 * The message of the last failure on this thread. Valid until the next
 * call into the library from the same thread.
 */
const char *smallm_last_error(void);

/* ---- Models ----------------------------------------------------------- */

/* Opens a GGUF model file. Returns NULL on failure. */
smallm_model *smallm_model_load(const char *path);

/* Releases a model. Sessions made from it stay valid. */
void smallm_model_free(smallm_model *model);

/* Number of tokens in the vocabulary, or a negative status. */
int32_t smallm_model_n_vocab(const smallm_model *model);

/* The context length the model was trained with, or a negative status. */
int32_t smallm_model_n_ctx_train(const smallm_model *model);

/* Total number of weights in the model. */
uint64_t smallm_model_n_params(const smallm_model *model);

/*
 * Copies a metadata value of the model file into buf as text, for example
 * the value of "general.name". Returns the length of the value, which may
 * exceed len, or 0 when the key is missing.
 */
size_t smallm_model_meta(const smallm_model *model, const char *key, char *buf, size_t len);

/* True when the model has a chat format that smallm_chat can use. */
bool smallm_model_has_chat_format(const smallm_model *model);

/* ---- Tokens ----------------------------------------------------------- */

/*
 * Converts text to tokens. Writes up to max_tokens ids and returns how many
 * the text has. When that is more than max_tokens, call again with a
 * larger buffer. Returns a negative status on failure.
 *
 * add_special adds the beginning and end tokens the model expects.
 * parse_special turns control token text such as "<|im_end|>" into the
 * control token. Leave it off for text that comes from a user.
 */
int32_t smallm_tokenize(const smallm_model *model, const char *text, size_t text_len, int32_t *tokens,
                        size_t max_tokens, bool add_special, bool parse_special);

/*
 * Copies the bytes a token stands for into buf and returns their count,
 * which may exceed len. Control tokens are empty unless special is set.
 * The bytes of one token are not always whole UTF-8 characters.
 */
int32_t smallm_token_to_piece(const smallm_model *model, int32_t token, char *buf, size_t len, bool special);

/* True when the token ends a generation. */
bool smallm_token_is_end(const smallm_model *model, int32_t token);

/* ---- Sessions --------------------------------------------------------- */

/* Session options with every field set to "choose for me". */
smallm_session_params smallm_session_default_params(void);

/* Starts an empty conversation. params may be NULL. NULL on failure. */
smallm_session *smallm_session_new(const smallm_model *model, const smallm_session_params *params);

/* Releases a session. */
void smallm_session_free(smallm_session *session);

/* The context window in tokens, or a negative status. */
int32_t smallm_session_n_ctx(const smallm_session *session);

/* Number of tokens in the context, or a negative status. */
int32_t smallm_session_n_tokens(const smallm_session *session);

/* Removes every token from position n_tokens on. 0 empties the context. */
void smallm_session_truncate(smallm_session *session, size_t n_tokens);

/*
 * Appends tokens to the context and computes the logits for the token
 * after them. Returns SMALLM_OK or a negative status.
 */
int32_t smallm_eval(smallm_session *session, const int32_t *tokens, size_t n);

/*
 * The scores for the next token, one float per vocabulary entry. Valid
 * until the session is next changed.
 */
const float *smallm_logits(const smallm_session *session);

/* ---- Sampling --------------------------------------------------------- */

/* The sampling options most chats use. */
smallm_sampler_params smallm_sampler_default_params(void);

/* Creates a sampler. params may be NULL. */
smallm_sampler *smallm_sampler_new(const smallm_sampler_params *params);

/* Releases a sampler. */
void smallm_sampler_free(smallm_sampler *sampler);

/* Picks the next token from a session's logits, or a negative status. */
int32_t smallm_sample(smallm_sampler *sampler, const smallm_session *session);

/* ---- Generation ------------------------------------------------------- */

/* Generation options with no token limit and the default sampler. */
smallm_generate_params smallm_generate_default_params(void);

/*
 * Completes prompt as plain text, with no chat format. callback receives
 * each piece of output. params, callback and stats may be NULL.
 * Returns SMALLM_OK or a negative status.
 */
int32_t smallm_generate(smallm_session *session, const char *prompt, const smallm_generate_params *params,
                        smallm_text_callback callback, void *user, smallm_stats *stats);

/*
 * Answers a conversation in the model's own chat format. roles and
 * contents each hold n_messages strings. A role is "system", "user" or
 * "assistant".
 *
 * The session keeps its context between calls, so a conversation that grew
 * by one message only pays for that message. When the conversation
 * outgrows the context window, the oldest turns are left out.
 */
int32_t smallm_chat(smallm_session *session, const char *const *roles, const char *const *contents,
                    size_t n_messages, const smallm_generate_params *params, smallm_text_callback callback,
                    void *user, smallm_stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* SMALLM_H */

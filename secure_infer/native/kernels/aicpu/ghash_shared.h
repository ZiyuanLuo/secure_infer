#pragma once

#include <stdint.h>

// ABI shared by the ASC launcher and the AICPU kernel. Device addresses are
// represented as integers so the structure has the same layout in both
// compilation modes.
struct PipeGhashArgs {
    uint64_t input;
    uint64_t hash_subkey;
    uint64_t output;
    uint64_t aad;
    uint64_t tag_mask;
    uint64_t expected_tag;
    uint64_t keystream;
    uint64_t plaintext_output;
    uint64_t status_output;
    uint64_t input_bytes;
    uint64_t aad_bytes;
    uint32_t iterations;
    uint32_t flags;
};

enum : uint32_t {
    PIPE_GHASH_FINALIZE_TAG = 1U << 0,
    PIPE_GHASH_VERIFY_TAG = 1U << 1,
    PIPE_GHASH_XOR_ON_SUCCESS = 1U << 2,
};

struct PipeAuthGateArgs {
    uint64_t data;
    uint64_t status;
    uint64_t bytes;
};

struct PipeAesCtrArgs {
    uint64_t input;
    uint64_t key;
    uint64_t output;
    uint64_t bytes;
    uint32_t iv_words[4];
};

// Background key-stream generation does not consume a payload.  Keep this
// separate from PipeAesCtrArgs so the AICPU kernel never has to read a dummy
// all-zero input buffer or execute a payload XOR.
struct PipeAesKeystreamArgs {
    uint64_t key;
    uint64_t output;
    uint64_t bytes;
    uint32_t iv_words[4];
};

// One speculative GCM ticket has one unique nonce/IV, so a contiguous CTR
// stream cannot be shared across tickets.  This bounded descriptor instead
// batches independent (IV, output) pairs into one AICPU launch.  The AES key
// is common, which avoids repeating key expansion and, more importantly,
// removes repeated host-to-AICPU launch submissions.  Thirty-two 256-KiB
// tickets keep one AICPU worker busy long enough for a real multi-request
// serving burst to expose sustained device contention; every ticket retains
// an independent IV and output buffer.
constexpr uint32_t PIPE_AES_KEYSTREAM_BATCH_MAX = 32;
struct PipeAesKeystreamBatchArgs {
    uint64_t key;
    uint64_t outputs[PIPE_AES_KEYSTREAM_BATCH_MAX];
    uint64_t bytes;
    uint32_t count;
    uint32_t iv_words[PIPE_AES_KEYSTREAM_BATCH_MAX][4];
};

// H2D AES-GCM verification and decryption are one indivisible security
// operation: the plaintext output may be consumed only if the ciphertext tag
// is valid.  The fused ABI avoids four independently scheduled AICPU tasks
// (tag-mask, GHASH, CTR, gate) for every 256-KiB physical KV slice.
struct PipeGcmDecryptVerifyArgs {
    uint64_t ciphertext;
    uint64_t key;
    uint64_t hash_subkey;
    uint64_t expected_tag;
    uint64_t plaintext_output;
    uint64_t status_output;
    uint64_t bytes;
    uint32_t iv_words[4];
    // Optional GCM additional authenticated data.  Appended so the legacy
    // wrapper's aggregate initializer remains ABI-compatible with zero AAD.
    uint64_t aad;
    uint64_t aad_bytes;
};

// D2H fused protection: encrypt the device plaintext with AES-CTR and
// authenticate the resulting ciphertext in the same AICPU task.  The host
// still performs the final GCM verification before publishing the protected
// cache entry; this kernel only removes the intermediate tag-mask/AES/GHASH
// launch chain on the device side.
struct PipeGcmEncryptTagArgs {
    uint64_t plaintext;
    uint64_t key;
    uint64_t hash_subkey;
    uint64_t ciphertext_output;
    uint64_t tag_output;
    uint64_t bytes;
    uint32_t iv_words[4];
    uint64_t aad;
    uint64_t aad_bytes;
};

// A ready speculative AES-CTR stream can remove payload AES from the H2D
// critical path.  Authentication remains inseparable from release: this ABI
// verifies GHASH, XORs only with the ticket's exact keystream, and masks the
// output on authentication failure in one AICPU task.
struct PipeGcmVerifyXorArgs {
    uint64_t ciphertext;
    uint64_t key;
    uint64_t hash_subkey;
    uint64_t expected_tag;
    uint64_t keystream;
    uint64_t plaintext_output;
    uint64_t status_output;
    uint64_t bytes;
    uint32_t iv_words[4];
    uint64_t aad;
    uint64_t aad_bytes;
};

// Authentication-only half of a strict verified-release pipeline.  A
// caller may decrypt/XOR into request-private device storage concurrently,
// but it must join this status before publishing or consuming that storage.
// Keeping the verifier independent lets GHASH use AICPU PMULL while the
// payload-independent XOR runs on the AI Vector engine.
struct PipeGcmVerifyArgs {
    uint64_t ciphertext;
    uint64_t key;
    uint64_t hash_subkey;
    uint64_t expected_tag;
    uint64_t status_output;
    uint64_t bytes;
    uint32_t iv_words[4];
    uint64_t aad;
    uint64_t aad_bytes;
};

// H2D reloads naturally contain several independent physical KV slices with
// the same AES key and GHASH subkey.  Grouping a small number of such slices
// amortizes AICPU launch/key-setup overhead without ever combining their
// nonce, tag, or release decision.  Every entry still has its own ciphertext,
// IV, expected tag, plaintext destination, and validity flag.
constexpr uint32_t PIPE_GCM_VERIFY_XOR_BATCH_MAX = 4;
struct PipeGcmVerifyXorBatchArgs {
    uint64_t key;
    uint64_t hash_subkey;
    uint64_t ciphertexts[PIPE_GCM_VERIFY_XOR_BATCH_MAX];
    uint64_t expected_tags[PIPE_GCM_VERIFY_XOR_BATCH_MAX];
    uint64_t keystreams[PIPE_GCM_VERIFY_XOR_BATCH_MAX];
    uint64_t plaintext_outputs[PIPE_GCM_VERIFY_XOR_BATCH_MAX];
    uint64_t status_outputs[PIPE_GCM_VERIFY_XOR_BATCH_MAX];
    uint64_t bytes;
    uint32_t count;
    uint32_t iv_words[PIPE_GCM_VERIFY_XOR_BATCH_MAX][4];
    uint64_t aad[PIPE_GCM_VERIFY_XOR_BATCH_MAX];
    uint64_t aad_bytes;
};

#include "secure_infer.h"
#include "hack.h"
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <thread>
#include <cstring>
#include <strings.h>
#include <chrono>
#include <cassert>
#include <dlfcn.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <array>
#include <unordered_set>
#include <condition_variable>
#include <future>
#include <deque>
#include <new>
#include <fcntl.h>
#include <unistd.h>
#include <openssl/rand.h>
std::mutex g_ctx_map_mutex;
static bool enable_encrypt = false;
static bool enable_h2d_encrypt = false;
static bool enable_d2h_encrypt = false;
static bool enable_decrypt = false;
extern size_t encrypt_magic_sz = 0xabcde;
extern size_t decrypt_magic_sz = 0xabcde;
std::map<EVP_CIPHER_CTX *, std::shared_ptr<encrypt_metadata>> m_ctx_metadata;
std::map<size_t, EVP_CIPHER_CTX *> m_magic_encctx;
std::set<size_t> s_magic;
std::map<EVP_CIPHER_CTX *, std::shared_ptr<decrypt_metadata>> m_ctx_dmetadata;
std::map<size_t, EVP_CIPHER_CTX *> m_magic_decctx;
std::set<size_t> s_magic_dec;
constexpr static int encrypt_workers_num_max = 16;
static int encrypt_workers_num = 1;
static int encrypt_workers_per_commit = 1;
constexpr static int decrypt_workers_num_max = 2;
static int decrypt_workers_num = 1;
int memcpy_thread_num = 1;
int commit_workers_num = encrypt_workers_num / encrypt_workers_per_commit;
encrypt_worker_entry *encrypt_worker_entrys[encrypt_workers_num_max];
std::thread *encrypt_worker_threads[encrypt_workers_num_max];
std::thread *commit_worker_threads[encrypt_workers_num_max];
decrypt_worker_entry *decrypt_worker_entrys[decrypt_workers_num_max];
std::thread *decrypt_worker_threads[decrypt_workers_num_max];
std::thread *decrypt_manager_threads[decrypt_workers_num_max];

Predictor predictor;
void sync_predictor();
void assign_to_workers(void *dst, const void *src, size_t count, bool decrypt,
                       bool predict, bool commit, bool update_predict_iv,
                       bool batch_commit, aclrtStream user_stream,
                       uint64_t transfer_ticket = 0,
                       uint64_t transfer_block_offset = 0,
                       uint64_t prefetch_group_id = 0);
static bool sampled_predict_log(std::atomic<uint64_t>& counter,
                                uint64_t first,
                                uint64_t every);
static void load_aes_library();
static bool gcm_h2d_coalesced_aad_enabled();
static void secure_infer_record_h2d_fault_event(const char* phase,
                                            uint32_t layer,
                                            uint32_t tensor,
                                            size_t bytes);
// 适配 add_custom_do: (uint32_t blockDim, void* stream, uint8_t* input, uint8_t* key, uint8_t* iv, uint8_t* output, uint32_t totalLength, uint32_t mode)
typedef void (*LaunchAESKernel_t)(void*, void*, void*, void*, void*, uint32_t);
LaunchAESKernel_t g_launch_kernel = nullptr;
using LaunchInlineAESKernel_t = void (*)(uint32_t, void*, uint8_t*, uint8_t*,
                                         uint8_t*, uint32_t, uint32_t,
                                         uint32_t, uint32_t, uint32_t,
                                         uint32_t);
static LaunchInlineAESKernel_t g_launch_inline_iv_kernel = nullptr;
static void* g_aes_lib_handle = nullptr;
// This is the independently verified AIV mode-1 entry.  It takes a device
// IV pointer rather than packed inline scalars; do not substitute the older
// inline-IV implementation for GCM keystream generation.
using LaunchVerifiedAivKeystream_t = void (*)(uint32_t, void*, uint8_t*,
                                               uint8_t*, uint8_t*, uint8_t*,
                                               uint32_t, uint32_t);
static LaunchVerifiedAivKeystream_t g_launch_verified_aiv_keystream = nullptr;
static void* g_verified_aiv_lib_handle = nullptr;
LaunchAicpuGhash_t g_launch_aicpu_ghash = nullptr;
LaunchAicpuAuthGate_t g_launch_aicpu_auth_gate = nullptr;
LaunchAicpuAesCtr_t g_launch_aicpu_aes_ctr = nullptr;
LaunchAicpuKeystream_t g_launch_aicpu_keystream = nullptr;
LaunchAicpuGcmDecryptVerify_t g_launch_aicpu_gcm_decrypt_verify = nullptr;
LaunchAicpuGcmDecryptVerifyAad_t g_launch_aicpu_gcm_decrypt_verify_aad = nullptr;
LaunchAicpuGcmEncryptTag_t g_launch_aicpu_gcm_encrypt_tag = nullptr;
LaunchAicpuGcmVerifyXor_t g_launch_aicpu_gcm_verify_xor = nullptr;
LaunchAicpuGcmVerifyXorAad_t g_launch_aicpu_gcm_verify_xor_aad = nullptr;
LaunchAicpuGcmVerifyXorBatch_t g_launch_aicpu_gcm_verify_xor_batch = nullptr;
LaunchAicpuGcmVerifyOnly_t g_launch_aicpu_gcm_verify_only = nullptr;
LaunchAicpuGcmVerifyOnlyAad_t g_launch_aicpu_gcm_verify_only_aad = nullptr;
LaunchAicpuGcmVerifyXor_t g_launch_aicpu_gcm_xor_only = nullptr;
static void* g_ghash_lib_handle = nullptr;
static std::mutex g_gcm_backend_init_mutex;
void* g_dev_key = nullptr;
void* g_dev_iv = nullptr;
// H2D key/IV allocation is process-lifetime state for the serving context.
// Publishing it once removes an otherwise unnecessary global mutex round-trip
// from every background AIV ticket submission.
static std::atomic<bool> g_h2d_device_crypto_ready{false};
// The key/IV copies above are enqueued on one framework/init stream, while
// ticketed AICPU and AIV keystreams may run on independent producer streams.
// Publish a single event only after both copies and retain it for the entire
// process lifetime; every producer waits on this event before reading the
// device key.  The event is intentionally never returned to a pool or
// destroyed because producers can still be submitting tickets until process
// teardown.
static std::atomic<void*> g_h2d_device_crypto_ready_event{nullptr};
static std::atomic<bool> g_h2d_gcm_batch_reported{false};
static std::atomic<uint64_t> g_h2d_gcm_batch_launches{0};
static std::atomic<uint64_t> g_h2d_gcm_batched_records{0};
static std::mutex g_gcm_ticket_iv_lock;
static std::array<unsigned char, iv_length> g_gcm_ticket_base_iv{};
static std::atomic<bool> g_gcm_ticket_base_iv_ready{false};

std::atomic<bool> g_running(true);
static std::atomic<bool> g_encrypt_workers_ready(false);

static std::once_flag g_transfer_iv_once;
static std::atomic<uint64_t> g_next_transfer_ticket{1};
static uint64_t g_transfer_session_nonce = 0;

bool secure_infer_gcm_enabled()
{
    static const bool enabled = [] {
        const char* mode = std::getenv("PIPE_CRYPTO_MODE");
        const bool use_gcm =
            mode != nullptr && strcasecmp(mode, "GCM") == 0;
        fprintf(stderr, "[SecureInfer] transfer crypto mode: %s\n",
                use_gcm ? "AES-GCM" : "AES-CTR");
        return use_gcm;
    }();
    return enabled;
}

// Evaluation-only directional controls used to measure how much H2D and D2H
// authentication is exposed in end-to-end latency.  A control is honored
// only together with precise profiling; production serving therefore cannot
// silently disable authentication through either variable.
static bool profile_auth_control_allowed()
{
    const char* experiment =
        std::getenv("PIPE_PROFILE_DIRECTIONAL_AUTH_EXPERIMENT");
    return experiment != nullptr && std::atoi(experiment) != 0;
}

static bool secure_infer_h2d_gcm_enabled()
{
    if (!secure_infer_gcm_enabled()) return false;
    static const bool skip = [] {
        const char* env = std::getenv("PIPE_PROFILE_SKIP_H2D_AUTH");
        const bool requested = env != nullptr && std::atoi(env) != 0;
        const bool enabled = requested && profile_auth_control_allowed();
        if (requested) {
            fprintf(stderr,
                    "[SecureInfer TEST ONLY] H2D authentication control: %s\n",
                    enabled ? "disabled" : "ignored outside directional experiment");
        }
        return enabled;
    }();
    return !skip;
}

static bool secure_infer_d2h_gcm_enabled()
{
    if (!secure_infer_gcm_enabled()) return false;
    static const bool skip = [] {
        const char* env = std::getenv("PIPE_PROFILE_SKIP_D2H_AUTH");
        const bool requested = env != nullptr && std::atoi(env) != 0;
        const bool enabled = requested && profile_auth_control_allowed();
        if (requested) {
            fprintf(stderr,
                    "[SecureInfer TEST ONLY] D2H authentication control: %s\n",
                    enabled ? "disabled" : "ignored outside directional experiment");
        }
        return enabled;
    }();
    return !skip;
}

// Evaluation-only directional plaintext controls.  The authentication
// ablation above intentionally keeps AES-CTR and DMA in the path; these two
// controls are narrower: they select the native plaintext copy for exactly
// one KV-transfer direction so a D2H-only or H2D-only experiment does not
// accidentally exercise the opposite direction's crypto implementation.
// They are accepted only inside the explicit directional experiment and are
// never enabled by normal serving configurations.
static bool secure_infer_plain_h2d_enabled()
{
    const char* env = std::getenv("PIPE_PROFILE_PLAIN_H2D");
    const bool requested = env != nullptr && std::atoi(env) != 0;
    const bool enabled = requested && profile_auth_control_allowed();
    if (requested) {
        static std::once_flag log_once;
        std::call_once(log_once, [enabled] {
            fprintf(stderr,
                    "[SecureInfer TEST ONLY] plaintext H2D control: %s\n",
                    enabled ? "enabled" : "ignored outside directional experiment");
        });
    }
    return enabled;
}

static bool secure_infer_plain_d2h_enabled()
{
    const char* env = std::getenv("PIPE_PROFILE_PLAIN_D2H");
    const bool requested = env != nullptr && std::atoi(env) != 0;
    const bool enabled = requested && profile_auth_control_allowed();
    if (requested) {
        static std::once_flag log_once;
        std::call_once(log_once, [enabled] {
            fprintf(stderr,
                    "[SecureInfer TEST ONLY] plaintext D2H control: %s\n",
                    enabled ? "enabled" : "ignored outside directional experiment");
        });
    }
    return enabled;
}

bool secure_infer_gcm_aad_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_AUTH_METADATA");
        // Metadata authentication is enabled by default for GCM.  Set the
        // variable to 0 only for compatibility with old ciphertext captures.
        enabled = env == nullptr ? 1 : (std::atoi(env) != 0 ? 1 : 0);
        fprintf(stderr, "[SecureInfer] GCM transfer metadata AAD: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0 && secure_infer_gcm_enabled();
}

static void init_transfer_iv_session()
{
    uint64_t nonce = 0;
    if (RAND_bytes(reinterpret_cast<unsigned char*>(&nonce), sizeof(nonce)) != 1) {
        // RAND_bytes should be available through libcrypto.  Keep a robust
        // non-zero fallback so a diagnostics-only environment still runs.
        nonce = static_cast<uint64_t>(std::chrono::steady_clock::now()
                                          .time_since_epoch().count()) ^
                (static_cast<uint64_t>(getpid()) << 32) ^
                reinterpret_cast<uintptr_t>(&g_transfer_session_nonce);
    }
    if (nonce == 0) nonce = 0x504950454c4c4dULL;
    g_transfer_session_nonce = nonce;
    fprintf(stderr,
            "[SecureInfer] IV manager initialized: random session nonce + monotonic transfer tickets\n");
}

uint64_t secure_infer_new_transfer_ticket()
{
    std::call_once(g_transfer_iv_once, init_transfer_iv_session);
    uint64_t ticket = g_next_transfer_ticket.fetch_add(1, std::memory_order_relaxed);
    // Ticket zero is reserved for "not assigned".  Reaching wraparound is
    // practically impossible, but never silently reuse the reserved value.
    if (ticket == 0) {
        fprintf(stderr, "[SecureInfer FATAL] transfer ticket space exhausted\n");
        std::abort();
    }
    return ticket;
}

void secure_infer_derive_transfer_iv(const unsigned char base_iv[iv_length],
                                uint64_t transfer_ticket,
                                uint64_t block_offset,
                                unsigned char out_iv[iv_length])
{
    std::call_once(g_transfer_iv_once, init_transfer_iv_session);
    const uint64_t counter_base = secure_infer_gcm_enabled() ? 2U : 0U;
    if (transfer_ticket == 0 || block_offset > UINT32_MAX - counter_base) {
        fprintf(stderr,
                "[SecureInfer FATAL] invalid CTR assignment: ticket=%lu block_offset=%lu\n",
                static_cast<unsigned long>(transfer_ticket),
                static_cast<unsigned long>(block_offset));
        std::abort();
    }
    memcpy(out_iv, base_iv, iv_length);
    const uint64_t transfer_nonce = g_transfer_session_nonce + transfer_ticket;
    for (int i = 0; i < 8; ++i) {
        out_iv[4 + i] = static_cast<unsigned char>(
            transfer_nonce >> (56 - i * 8));
    }
    const uint32_t counter = static_cast<uint32_t>(
        block_offset + counter_base);
    out_iv[12] = static_cast<unsigned char>(counter >> 24);
    out_iv[13] = static_cast<unsigned char>(counter >> 16);
    out_iv[14] = static_cast<unsigned char>(counter >> 8);
    out_iv[15] = static_cast<unsigned char>(counter);
}

static bool secure_infer_aes128_block(const unsigned char key[key_length],
                                 const unsigned char input[16],
                                 unsigned char output[16])
{
    unsigned char zero[16] = {0};
    int updated = 0;
    int final_len = 0;
    EVP_CIPHER_CTX* ctx = real_EVP_CIPHER_CTX_new();
    if (ctx == nullptr) return false;
    const bool ok =
        real_EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), nullptr,
                                key, input) == 1 &&
        real_EVP_EncryptUpdate(ctx, output, &updated, zero, 16) == 1 &&
        real_EVP_EncryptFinal_ex(ctx, output + updated, &final_len) == 1;
    real_EVP_CIPHER_CTX_free(ctx);
    return ok && updated + final_len == 16;
}

bool secure_infer_compute_gcm_tag_mask(const unsigned char key[key_length],
                                  const unsigned char payload_iv[iv_length],
                                  unsigned char tag_mask[16])
{
    // payload_iv contains nonce || 2. GCM's tag mask is E_K(J0), where
    // J0=nonce||1 for a 96-bit nonce.
    unsigned char j0[16];
    memcpy(j0, payload_iv, 12);
    j0[12] = 0;
    j0[13] = 0;
    j0[14] = 0;
    j0[15] = 1;
    return secure_infer_aes128_block(key, j0, tag_mask);
}

static bool load_ghash_library()
{
    if (g_launch_aicpu_ghash != nullptr &&
        g_launch_aicpu_auth_gate != nullptr &&
        g_launch_aicpu_aes_ctr != nullptr &&
        g_launch_aicpu_keystream != nullptr) {
        return true;
    }
    const char* configured = std::getenv("PIPE_GHASH_AICPU_LIB");
    const char* path = configured;
    if (path == nullptr || path[0] == '\0') {
        fprintf(stderr, "[runtime] PIPE_GHASH_AICPU_LIB is required\n");
        return false;
    }
    g_ghash_lib_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (g_ghash_lib_handle == nullptr) {
        fprintf(stderr, "[SecureInfer ERROR] cannot load AICPU GHASH library %s: %s\n",
                path, dlerror());
        return false;
    }
    g_launch_aicpu_ghash = reinterpret_cast<LaunchAicpuGhash_t>(
        dlsym(g_ghash_lib_handle, "secure_infer_aicpu_ghash_do"));
    g_launch_aicpu_auth_gate = reinterpret_cast<LaunchAicpuAuthGate_t>(
        dlsym(g_ghash_lib_handle, "secure_infer_aicpu_auth_gate_do"));
    g_launch_aicpu_aes_ctr = reinterpret_cast<LaunchAicpuAesCtr_t>(
        dlsym(g_ghash_lib_handle, "secure_infer_aicpu_aes_ctr_do"));
    g_launch_aicpu_keystream = reinterpret_cast<LaunchAicpuKeystream_t>(
        dlsym(g_ghash_lib_handle, "secure_infer_aicpu_keystream_do"));
    // Optional fused H2D path.  Keep old AICPU images deployable: when the
    // symbol is absent the call site uses the established GHASH -> CTR ->
    // auth-gate sequence.
    g_launch_aicpu_gcm_decrypt_verify =
        reinterpret_cast<LaunchAicpuGcmDecryptVerify_t>(
            dlsym(g_ghash_lib_handle,
                  "secure_infer_aicpu_gcm_decrypt_verify_do"));
    g_launch_aicpu_gcm_decrypt_verify_aad =
        reinterpret_cast<LaunchAicpuGcmDecryptVerifyAad_t>(
            dlsym(g_ghash_lib_handle,
                  "secure_infer_aicpu_gcm_decrypt_verify_aad_do"));
    g_launch_aicpu_gcm_encrypt_tag =
        reinterpret_cast<LaunchAicpuGcmEncryptTag_t>(
            dlsym(g_ghash_lib_handle, "secure_infer_aicpu_gcm_encrypt_tag_do"));
    g_launch_aicpu_gcm_verify_xor =
        reinterpret_cast<LaunchAicpuGcmVerifyXor_t>(
            dlsym(g_ghash_lib_handle, "secure_infer_aicpu_gcm_verify_xor_do"));
    g_launch_aicpu_gcm_verify_xor_aad =
        reinterpret_cast<LaunchAicpuGcmVerifyXorAad_t>(
            dlsym(g_ghash_lib_handle,
                  "secure_infer_aicpu_gcm_verify_xor_aad_do"));
    g_launch_aicpu_gcm_verify_xor_batch =
        reinterpret_cast<LaunchAicpuGcmVerifyXorBatch_t>(
            dlsym(g_ghash_lib_handle,
                  "secure_infer_aicpu_gcm_verify_xor_batch_do"));
    g_launch_aicpu_gcm_verify_only =
        reinterpret_cast<LaunchAicpuGcmVerifyOnly_t>(
            dlsym(g_ghash_lib_handle, "secure_infer_aicpu_gcm_verify_only_do"));
    g_launch_aicpu_gcm_verify_only_aad =
        reinterpret_cast<LaunchAicpuGcmVerifyOnlyAad_t>(
            dlsym(g_ghash_lib_handle,
                  "secure_infer_aicpu_gcm_verify_only_aad_do"));
    g_launch_aicpu_gcm_xor_only =
        reinterpret_cast<LaunchAicpuGcmVerifyXor_t>(
            dlsym(g_ghash_lib_handle, "secure_infer_aicpu_gcm_xor_only_do"));
    if (g_launch_aicpu_ghash == nullptr ||
        g_launch_aicpu_auth_gate == nullptr ||
        g_launch_aicpu_aes_ctr == nullptr ||
        g_launch_aicpu_keystream == nullptr) {
        fprintf(stderr, "[SecureInfer ERROR] incomplete AICPU GHASH ABI: %s\n",
                dlerror());
        return false;
    }
    fprintf(stderr,
            "[SecureInfer] AICPU PMULL GHASH backend loaded: %s "
            "(fused H2D GCM: %s, ticket-XOR: %s, verify-only: %s, profile-XOR: %s, "
            "ticketed keystream: per-ticket ABI)\n",
            path, g_launch_aicpu_gcm_decrypt_verify != nullptr ? "enabled" : "legacy",
            g_launch_aicpu_gcm_verify_xor != nullptr ? "enabled" : "legacy",
            g_launch_aicpu_gcm_verify_only != nullptr ? "enabled" : "absent",
            g_launch_aicpu_gcm_xor_only != nullptr ? "enabled" : "absent");
    fprintf(stderr,
            "[SecureInfer] AAD/fused GCM ABI: D2H_encrypt_tag=%s decrypt=%s verify_xor=%s verify_only=%s\n",
            g_launch_aicpu_gcm_encrypt_tag != nullptr ? "enabled" : "legacy",
            g_launch_aicpu_gcm_decrypt_verify_aad != nullptr ? "enabled" : "legacy",
            g_launch_aicpu_gcm_verify_xor_aad != nullptr ? "enabled" : "legacy",
            g_launch_aicpu_gcm_verify_only_aad != nullptr ? "enabled" : "legacy");
    return true;
}

// CANN discovers/registers the AICPU image while the runtime is initialized.
// Loading an ASC/AICPU shared object for the first time after aclInit can
// successfully resolve its host symbol yet later fail every device task with
// `aicpu exception (0x2a)`.  SecureInfer itself is LD_PRELOADed, so register the
// optional AICPU crypto image in an early constructor. CTR keystream
// pre-generation uses the same image, so it must obey the same registration
// rule as GCM.
__attribute__((constructor(200)))
static void secure_infer_preload_gcm_aicpu_image()
{
    const char* mode = std::getenv("PIPE_CRYPTO_MODE");
    const char* prefetch = std::getenv("PIPE_NPU_KEYSTREAM_PREFETCH");
    const char* aicpu_ctr = std::getenv("PIPE_AICPU_AES_CTR");
    const bool requested =
        (mode != nullptr && strcasecmp(mode, "GCM") == 0) ||
        (prefetch != nullptr && std::atoi(prefetch) != 0) ||
        (aicpu_ctr != nullptr && std::atoi(aicpu_ctr) != 0);
    if (requested && !load_ghash_library()) {
        fprintf(stderr,
                "[SecureInfer FATAL] AICPU crypto requested but image could not be preloaded\n");
        std::abort();
    }
}

bool secure_infer_prepare_gcm_backend(encrypt_metadata& metadata,
                                 aclrtStream stream)
{
    if (!secure_infer_gcm_enabled()) return true;
    if (!load_ghash_library()) return false;
    std::lock_guard<std::mutex> init_lock(g_gcm_backend_init_mutex);
    if (metadata.gcm_backend_ready) return true;
    fprintf(stderr, "[SecureInfer] GCM backend cold start: begin\n");

    // Never synchronize the framework's in-progress transfer stream here.
    // A D2H group may already contain waits whose matching records are added
    // only by group_end; synchronizing it from the first intercepted slice
    // would create a circular dependency.  Use a private one-shot stream for
    // H upload and custom-AICPU cold start, then publish a fully ready backend.
    (void)stream;
    aclrtStream init_stream = nullptr;
    if (real_aclrtCreateStream(&init_stream) != ACL_SUCCESS ||
        init_stream == nullptr) {
        return false;
    }
    fprintf(stderr, "[SecureInfer] GCM backend cold start: init stream ready\n");

    unsigned char zero[16] = {0};
    unsigned char h[16];
    if (!secure_infer_aes128_block(metadata.key, zero, h)) {
        aclrtDestroyStream(init_stream);
        return false;
    }
    void* dev_ghash_h = nullptr;
    if (real_aclrtMalloc(&dev_ghash_h, 32) != ACL_SUCCESS ||
        dev_ghash_h == nullptr) {
        aclrtDestroyStream(init_stream);
        return false;
    }
    if (real_aclrtMemcpyAsync(dev_ghash_h, 32, h, 16,
                              ACL_MEMCPY_HOST_TO_DEVICE, init_stream) != ACL_SUCCESS ||
        real_aclrtSynchronizeStream(init_stream) != ACL_SUCCESS) {
        real_aclrtFree(dev_ghash_h);
        aclrtDestroyStream(init_stream);
        return false;
    }
    fprintf(stderr, "[SecureInfer] GCM backend cold start: H uploaded\n");

    // Start the custom AICPU executor while the model worker is still in its
    // initialization path.  Deferring the first AICPU task until a 5,040-slice
    // store is already queued can make the cold executor initialization race
    // with active AI-Core/copy streams on some CANN releases.  Besides making
    // startup deterministic, this turns an unusable custom image into a clean
    // initialization failure instead of poisoning the serving context later.
    void* warmup_input = nullptr;
    void* warmup_tag_mask = nullptr;
    void* warmup_output = nullptr;
    unsigned char zero_block[32] = {0};
    fprintf(stderr, "[SecureInfer] GCM backend cold start: launch AICPU warmup\n");
    bool warmup_ok =
        real_aclrtMalloc(&warmup_input, 32) == ACL_SUCCESS &&
        warmup_input != nullptr &&
        real_aclrtMalloc(&warmup_tag_mask, 32) == ACL_SUCCESS &&
        warmup_tag_mask != nullptr &&
        real_aclrtMalloc(&warmup_output, 32) == ACL_SUCCESS &&
        warmup_output != nullptr &&
        real_aclrtMemcpyAsync(warmup_input, 32, zero_block, 32,
                              ACL_MEMCPY_HOST_TO_DEVICE, init_stream) == ACL_SUCCESS &&
        real_aclrtMemcpyAsync(warmup_tag_mask, 32, zero_block, 32,
                              ACL_MEMCPY_HOST_TO_DEVICE, init_stream) == ACL_SUCCESS &&
        g_launch_aicpu_ghash(
            init_stream, warmup_input, dev_ghash_h, warmup_output,
            nullptr, warmup_tag_mask, nullptr, nullptr, nullptr, nullptr,
            16, 0, PIPE_GHASH_FINALIZE_TAG) == 0 &&
        real_aclrtSynchronizeStream(init_stream) == ACL_SUCCESS;
    fprintf(stderr, "[SecureInfer] GCM backend cold start: AICPU warmup returned=%d\n",
            warmup_ok ? 1 : 0);
    if (warmup_output != nullptr) real_aclrtFree(warmup_output);
    if (warmup_tag_mask != nullptr) real_aclrtFree(warmup_tag_mask);
    if (warmup_input != nullptr) real_aclrtFree(warmup_input);
    if (!warmup_ok) {
        real_aclrtFree(dev_ghash_h);
        aclrtDestroyStream(init_stream);
        fprintf(stderr,
                "[SecureInfer ERROR] AICPU GHASH warmup failed during worker initialization\n");
        return false;
    }

    aclrtDestroyStream(init_stream);
    metadata.dev_ghash_h = dev_ghash_h;
    metadata.gcm_backend_ready = true;
    fprintf(stderr, "[SecureInfer] GCM hash subkey H initialized on NPU\n");
    return true;
}

// A request that reloads 63 logical KV blocks emits 5,040 physical H2D
// transfers.  Creating and destroying an ACL event for every transfer puts
// allocator/driver work directly on the Python submission thread.  Events are
// safe to reuse after aclrtQueryEventWaitStatus reports that the cross-stream
// wait has completed, so retain a small process-local pool.
static std::mutex g_commit_event_pool_lock;
static std::vector<aclrtEvent> g_commit_event_pool;
static std::atomic<uint64_t> g_commit_event_created{0};
static std::atomic<uint64_t> g_commit_event_reused{0};

static bool commit_event_pool_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_COMMIT_EVENT_POOL");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] commit dependency event pool: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

aclrtEvent acquire_commit_sync_event()
{
    if (commit_event_pool_enabled()) {
        std::lock_guard<std::mutex> lock(g_commit_event_pool_lock);
        if (!g_commit_event_pool.empty()) {
            aclrtEvent event = g_commit_event_pool.back();
            g_commit_event_pool.pop_back();
            g_commit_event_reused.fetch_add(1, std::memory_order_relaxed);
            return event;
        }
    }

    aclrtEvent event = nullptr;
    if (real_aclrtCreateEvent(&event) != ACL_SUCCESS) return nullptr;
    uint64_t created = g_commit_event_created.fetch_add(
        1, std::memory_order_relaxed) + 1;
    if (created == 1) {
        fprintf(stderr, "[SecureInfer] commit event pool initialized\n");
    }
    return event;
}

void recycle_commit_sync_event(aclrtEvent event)
{
    if (event == nullptr) return;
    if (!commit_event_pool_enabled()) {
        real_aclrtDestroyEvent(event);
        return;
    }

    // The number of simultaneous dependencies is bounded by the commit
    // in-flight window; the cap only prevents unbounded retention after a
    // pathological burst.
    constexpr size_t kMaxPooledCommitEvents = 256;
    std::lock_guard<std::mutex> lock(g_commit_event_pool_lock);
    if (g_commit_event_pool.size() < kMaxPooledCommitEvents) {
        g_commit_event_pool.push_back(event);
    } else {
        real_aclrtDestroyEvent(event);
    }
}

struct host_range {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    uint32_t layer_id = UINT32_MAX;
    uint32_t tensor_kind = UINT32_MAX;
    size_t block_bytes = 0;
};
static std::mutex g_host_ranges_lock;
static std::vector<host_range> g_host_ranges;

static std::mutex g_gcm_replay_lock;
static std::unordered_set<uint64_t> g_gcm_replay_reserved;
static void init_transfer_iv_session();

static void gcm_store_be32(unsigned char* dst, uint32_t value)
{
    dst[0] = static_cast<unsigned char>(value >> 24);
    dst[1] = static_cast<unsigned char>(value >> 16);
    dst[2] = static_cast<unsigned char>(value >> 8);
    dst[3] = static_cast<unsigned char>(value);
}

static void gcm_store_be64(unsigned char* dst, uint64_t value)
{
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<unsigned char>(value >> (56 - 8 * i));
    }
}

static bool gcm_lookup_host_range(const void* host_ptr, size_t bytes,
                                  host_range* result,
                                  uint64_t* block_id,
                                  uint64_t* byte_offset)
{
    if (host_ptr == nullptr || result == nullptr || block_id == nullptr ||
        byte_offset == nullptr) {
        return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(host_ptr);
    if (bytes > UINTPTR_MAX - begin) return false;
    const uintptr_t end = begin + bytes;
    std::lock_guard<std::mutex> lock(g_host_ranges_lock);
    for (const auto& range : g_host_ranges) {
        if (begin < range.begin || end > range.end) continue;
        *result = range;
        const uint64_t relative = static_cast<uint64_t>(begin - range.begin);
        const uint64_t stride = range.block_bytes != 0
            ? static_cast<uint64_t>(range.block_bytes) : 1U;
        *block_id = relative / stride;
        *byte_offset = relative % stride;
        return true;
    }
    return false;
}

bool secure_infer_build_gcm_aad(const void* host_ptr, size_t bytes,
                           uint8_t direction, uint64_t transfer_ticket,
                           uint64_t transfer_offset,
                           unsigned char out[gcm_aad_length])
{
    if (out == nullptr || host_ptr == nullptr || bytes == 0 ||
        transfer_ticket == 0 ||
        (direction != PIPE_GCM_DIRECTION_H2D &&
         direction != PIPE_GCM_DIRECTION_D2H)) {
        return false;
    }
    std::call_once(g_transfer_iv_once, init_transfer_iv_session);

    host_range range;
    uint64_t block_id = 0;
    uint64_t byte_offset = 0;
    if (!gcm_lookup_host_range(host_ptr, bytes, &range, &block_id,
                               &byte_offset)) {
        // Compatibility callers may not register their host tensor.  Keep a
        // deterministic descriptor rather than silently authenticating an
        // empty AAD; the registered vLLM path supplies real layer/block IDs.
        const uintptr_t address = reinterpret_cast<uintptr_t>(host_ptr);
        range.layer_id = UINT32_MAX;
        range.tensor_kind = UINT32_MAX;
        range.block_bytes = 1;
        block_id = static_cast<uint64_t>(address);
        byte_offset = 0;
    }

    memset(out, 0, gcm_aad_length);
    // Layout (big endian): magic/version/direction/tensor-kind, layer,
    // logical block, in-block offset, payload length, ticket, transfer
    // offset, and session epoch.  The trailing bytes are reserved and zero.
    out[0] = 'S'; out[1] = 'F'; out[2] = 'A'; out[3] = 'D';
    out[4] = 1;  // descriptor version
    out[5] = direction;
    out[6] = static_cast<unsigned char>(range.tensor_kind & 0xffU);
    gcm_store_be32(out + 8, range.layer_id);
    gcm_store_be64(out + 12, block_id);
    gcm_store_be64(out + 20, byte_offset);
    gcm_store_be64(out + 28, static_cast<uint64_t>(bytes));
    gcm_store_be64(out + 36, transfer_ticket);
    gcm_store_be64(out + 44, transfer_offset);
    gcm_store_be64(out + 52, g_transfer_session_nonce);
    return true;
}

bool secure_infer_gcm_replay_reserve(uint64_t transfer_ticket)
{
    if (!secure_infer_gcm_aad_enabled() || transfer_ticket == 0) return true;
    std::lock_guard<std::mutex> lock(g_gcm_replay_lock);
    const auto inserted = g_gcm_replay_reserved.insert(transfer_ticket);
    if (!inserted.second) {
        fprintf(stderr,
                "[SecureInfer SECURITY] replayed transfer ticket rejected: %lu\n",
                static_cast<unsigned long>(transfer_ticket));
        return false;
    }
    return true;
}
struct transfer_prefetch_record {
    size_t bytes = 0;
    uint64_t generation = 0;
    uint64_t ticket = 0;
    uint64_t prefetch_group_id = 0;
    uint64_t transfer_block_offset = 0;
    uint32_t ticket_batch_members = 1;
    uint64_t descriptor_ns = 0;
    // Zero is the legacy deduplicated descriptor scope.  A non-zero H2D job
    // scope permits two queued real loads to legitimately reference the same
    // CPU cache block while retaining independent IV/ticket ownership.
    uint64_t admission_scope = 0;
};

// One physical CPU KV block may be used by more than one queued H2D job (for
// example, a shared-prefix external-cache hit).  Keep a FIFO ticket queue per
// address: the native memcpy hook has no request ID.  The default submission
// path is FIFO; the opt-in multi-stream path dispatches only source-disjoint
// jobs, so shared-address reservations are still consumed in FIFO order.
static std::unordered_map<uintptr_t, std::deque<transfer_prefetch_record>>
    g_transfer_prefetch_records;
static constexpr size_t TRANSFER_PREFETCH_KEY_LIMIT = 65536;
std::atomic<uint64_t> g_retired_transfer_generation{0};
std::atomic<uint64_t> g_transfer_prefetch_ready{0};
std::atomic<uint64_t> g_transfer_prefetch_busy_fallback{0};
std::atomic<uint64_t> g_transfer_prefetch_missing_fallback{0};

// Python's connector polls this symbol after each transfer operation.  These
// counters describe CPU-ciphertext readiness.  The AICPU-keystream/AIV-XOR
// path has separate counters exported by worker.cpp, so the two kinds of
// speculation are never conflated in experiment logs.
HOOK_C_API HOOK_DECL_EXPORT void secure_infer_get_transfer_prefetch_stats(
    uint64_t* ready, uint64_t* busy_fallback, uint64_t* missing_fallback)
{
    if (ready != nullptr) {
        *ready = g_transfer_prefetch_ready.load(std::memory_order_relaxed);
    }
    if (busy_fallback != nullptr) {
        *busy_fallback = g_transfer_prefetch_busy_fallback.load(
            std::memory_order_relaxed);
    }
    if (missing_fallback != nullptr) {
        *missing_fallback = g_transfer_prefetch_missing_fallback.load(
            std::memory_order_relaxed);
    }
}

// Optional descriptor lead-time trace.  It is disabled unless a destination
// path is supplied, so normal interception does not pay for timestamping or
// file I/O.
static std::mutex g_descriptor_leadtime_lock;
static FILE* g_descriptor_leadtime_file = nullptr;
static bool g_descriptor_leadtime_initialized = false;
struct descriptor_leadtime_sample {
    const void* src = nullptr;
    size_t bytes = 0;
    uint64_t ticket = 0;
    uint64_t lead_us = 0;
};
static std::vector<descriptor_leadtime_sample> g_descriptor_leadtime_samples;

static uint64_t secure_infer_now_ns()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

static const char* descriptor_leadtime_path()
{
    const char* path = std::getenv("PIPE_DESCRIPTOR_LEADTIME_PATH");
    return (path != nullptr && path[0] != '\0') ? path : nullptr;
}

static void record_descriptor_lead_time(const void* src,
                                        size_t bytes,
                                        uint64_t ticket,
                                        uint64_t descriptor_ns)
{
    if (descriptor_ns == 0) return;
    if (descriptor_leadtime_path() == nullptr) return;
    // Capture demand before acquiring a profiling lock.  Samples are buffered
    // and written once per transfer scope so tracing does not serialize each
    // physical H2D submission on stdio/fflush.
    const uint64_t demand_ns = secure_infer_now_ns();
    const uint64_t lead_us = demand_ns >= descriptor_ns
        ? (demand_ns - descriptor_ns) / 1000ULL : 0ULL;
    std::lock_guard<std::mutex> lock(g_descriptor_leadtime_lock);
    g_descriptor_leadtime_samples.push_back({src, bytes, ticket, lead_us});
}

static void flush_descriptor_lead_times()
{
    const char* path = descriptor_leadtime_path();
    if (path == nullptr) return;
    std::lock_guard<std::mutex> lock(g_descriptor_leadtime_lock);
    if (!g_descriptor_leadtime_initialized) {
        g_descriptor_leadtime_file = std::fopen(path, "w");
        g_descriptor_leadtime_initialized = true;
        if (g_descriptor_leadtime_file != nullptr) {
            std::fprintf(g_descriptor_leadtime_file,
                         "src,bytes,ticket,descriptor_to_demand_us\n");
        }
    }
    if (g_descriptor_leadtime_file == nullptr) return;
    for (const auto& sample : g_descriptor_leadtime_samples) {
        std::fprintf(g_descriptor_leadtime_file, "%p,%zu,%lu,%lu\n",
                     sample.src, sample.bytes,
                     static_cast<unsigned long>(sample.ticket),
                     static_cast<unsigned long>(sample.lead_us));
    }
    g_descriptor_leadtime_samples.clear();
    std::fflush(g_descriptor_leadtime_file);
}

// Experiment/control knob for a genuinely serial H2D secure path.  Disabling
// transfer-layer lookahead alone is insufficient because the transparent
// physical-address predictor can still pre-encrypt future slices.  When this
// flag is set, no H2D prediction workers are created and every demanded slice
// executes the inline CPU-encrypt -> H2D -> NPU-auth/decrypt path.
static bool h2d_force_serial_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_H2D_FORCE_SERIAL");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] H2D force-serial mode: enabled "
                    "(descriptor/stride prediction disabled)\n");
        }
    }
    return enabled != 0;
}

bool secure_infer_linear_fast_fallback_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_LINEAR_FAST_FALLBACK");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] linear fast-fallback: enabled "
                    "(demand priority, stale-work cancellation, reusable EVP)\n");
        }
    }
    return enabled != 0;
}

static std::mutex g_cancelled_speculative_ticket_lock;
static std::unordered_set<uint64_t> g_cancelled_speculative_tickets;

void secure_infer_cancel_speculative_ticket(uint64_t transfer_ticket)
{
    if (transfer_ticket == 0 || !secure_infer_linear_fast_fallback_enabled()) return;
    std::lock_guard<std::mutex> lock(g_cancelled_speculative_ticket_lock);
    g_cancelled_speculative_tickets.insert(transfer_ticket);
}

bool secure_infer_speculative_ticket_cancelled(uint64_t transfer_ticket)
{
    if (transfer_ticket == 0 || !secure_infer_linear_fast_fallback_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_cancelled_speculative_ticket_lock);
    return g_cancelled_speculative_tickets.count(transfer_ticket) != 0;
}

void secure_infer_retire_prediction_generation(uint64_t generation)
{
    uint64_t observed =
        g_retired_transfer_generation.load(std::memory_order_relaxed);
    while (observed < generation &&
           !g_retired_transfer_generation.compare_exchange_weak(
               observed, generation,
           std::memory_order_release,
           std::memory_order_relaxed)) {
    }
    // A superseded descriptor must not retain a large host ciphertext slab.
    // Running/queued work observes cancellation and is reclaimed once it
    // reaches a terminal state, so this never frees a worker-owned buffer.
    secure_infer_gcm_cpu_cipher_cleanup();
}

static bool mark_transfer_prefetch_locked(const void* src,
                                          size_t bytes,
                                          uint64_t* ticket_out,
                                          uint64_t* block_offset_out = nullptr,
                                          uint64_t reserved_ticket = 0,
                                          uint64_t reserved_block_offset = 0,
                                          uint32_t ticket_batch_members = 1,
                                          uint64_t reserved_prefetch_group_id = 0,
                                          uint64_t admission_scope = 0)
{
    assert(predictor.locking());
    uintptr_t addr = reinterpret_cast<uintptr_t>(src);
    auto existing = g_transfer_prefetch_records.find(addr);
    if (existing != g_transfer_prefetch_records.end()) {
        for (const auto& prior : existing->second) {
            // Legacy calls retain the previous idempotent behaviour. Scoped
            // calls deduplicate only within the same logical H2D job, not
            // across queued jobs that happen to share an address.
            if (prior.bytes == bytes &&
                prior.generation == predictor.generation &&
                prior.admission_scope == admission_scope) {
                if (ticket_out != nullptr) *ticket_out = prior.ticket;
                if (block_offset_out != nullptr) {
                    *block_offset_out = prior.transfer_block_offset;
                }
                return false;
            }
        }
    }

    if (g_transfer_prefetch_records.size() >= TRANSFER_PREFETCH_KEY_LIMIT) {
        g_transfer_prefetch_records.clear();
    }
    transfer_prefetch_record record;
    record.bytes = bytes;
    record.generation = predictor.generation;
    record.ticket = reserved_ticket != 0
        ? reserved_ticket
        : secure_infer_new_transfer_ticket();
    record.prefetch_group_id = reserved_prefetch_group_id != 0
        ? reserved_prefetch_group_id
        : record.ticket;
    record.transfer_block_offset = reserved_block_offset;
    record.ticket_batch_members = std::max<uint32_t>(ticket_batch_members, 1);
    record.descriptor_ns = descriptor_leadtime_path() != nullptr
        ? secure_infer_now_ns() : 0;
    record.admission_scope = admission_scope;
    g_transfer_prefetch_records[addr].push_back(record);
    if (ticket_out != nullptr) *ticket_out = record.ticket;
    if (block_offset_out != nullptr) {
        *block_offset_out = record.transfer_block_offset;
    }
    return true;
}

static bool consume_transfer_prefetch_locked(const void* src,
                                             size_t bytes,
                                             uint64_t* ticket_out,
                                             uint64_t* block_offset_out,
                                             uint64_t* prefetch_group_id_out,
                                             uint64_t* descriptor_ns_out = nullptr,
                                             uint64_t* admission_scope_out = nullptr)
{
    assert(predictor.locking());
    uintptr_t addr = reinterpret_cast<uintptr_t>(src);
    auto it = g_transfer_prefetch_records.find(addr);
    if (it == g_transfer_prefetch_records.end() || it->second.empty()) {
        return false;
    }
    const auto& record = it->second.front();
    if (record.bytes != bytes || record.generation != predictor.generation) {
        return false;
    }
    if (ticket_out != nullptr) *ticket_out = record.ticket;
    if (block_offset_out != nullptr) {
        *block_offset_out = record.transfer_block_offset;
    }
    if (prefetch_group_id_out != nullptr) {
        *prefetch_group_id_out = record.prefetch_group_id;
    }
    if (descriptor_ns_out != nullptr) {
        *descriptor_ns_out = record.descriptor_ns;
    }
    if (admission_scope_out != nullptr) {
        *admission_scope_out = record.admission_scope;
    }
    it->second.pop_front();
    if (it->second.empty()) g_transfer_prefetch_records.erase(it);
    return true;
}

extern "C" void secure_infer_register_host_range_ex(const void* base, size_t bytes,
                                                 uint32_t layer_id,
                                                 uint32_t tensor_kind,
                                                 size_t block_bytes)
{
    if (base == nullptr || bytes == 0) return;
    uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    if (bytes > UINTPTR_MAX - begin) return;
    uintptr_t end = begin + bytes;

    std::lock_guard<std::mutex> lock(g_host_ranges_lock);
    for (auto& range : g_host_ranges) {
        if (range.begin == begin && range.end == end) {
            range.layer_id = layer_id;
            range.tensor_kind = tensor_kind;
            range.block_bytes = block_bytes;
            return;
        }
    }
    g_host_ranges.push_back({begin, end, layer_id, tensor_kind, block_bytes});
    fprintf(stderr,
            "[SecureInfer] registered host KV range: [%p, %p), bytes=%zu, "
            "layer=%u tensor=%u block_bytes=%zu\n",
            base, reinterpret_cast<void*>(end), bytes, layer_id,
            tensor_kind, block_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT void secure_infer_register_host_range(const void* base,
                                                              size_t bytes)
{
    secure_infer_register_host_range_ex(base, bytes, UINT32_MAX, UINT32_MAX, 0);
}

// A Hook-only policy cannot remove entries from the commit worker directly.
// Bumping the generation makes every outstanding speculative result stale,
// while real H2D transfers keep the normal serial fallback for correctness.
HOOK_C_API HOOK_DECL_EXPORT void secure_infer_invalidate_transfer_prefetches()
{
    predictor.lock();
    const uint64_t retired_generation = predictor.generation;
    predictor.generation++;
    g_transfer_prefetch_records.clear();
    predictor.linear_mode = false;
    predictor.linear_stride = 0;
    predictor.linear_expected_src = nullptr;
    predictor.linear_prefetch_until = nullptr;
    predictor.linear_count = 0;
    predictor.read_only_swap_record.clear();
    predictor.periodic_last_src = nullptr;
    predictor.periodic_copy_size = 0;
    predictor.periodic_deltas.clear();
    predictor.periodic_cycle.clear();
    predictor.periodic_next_phase = 0;
    predictor.periodic_run_patterns.clear();
    predictor.periodic_current_run_blocks = 0;
    predictor.periodic_active_run_blocks = 0;
    predictor.periodic_pattern_clock = 0;
    predictor.secure_infer_repeat_record.clear();
    predictor.secure_infer_repeat_profiled = false;
    predictor.secure_infer_repeat_next_index = 0;
    predictor.unlock();
    secure_infer_retire_prediction_generation(retired_generation);
    flush_descriptor_lead_times();
}

// A logical transfer job may finish with speculative tickets that were not
// consumed by any real H2D demand.  Retire those tickets and their crypto
// buffers, but keep the observation state used by history-based predictors.
// This is deliberately separate from secure_infer_invalidate_transfer_prefetches:
// the latter is the explicit full predictor reset used after a mismatch or a
// new predictor session.
HOOK_C_API HOOK_DECL_EXPORT void secure_infer_retire_transfer_prefetches()
{
    predictor.lock();
    const uint64_t retired_generation = predictor.generation;
    predictor.generation++;
    g_transfer_prefetch_records.clear();
    predictor.unlock();
    secure_infer_retire_prediction_generation(retired_generation);
    flush_descriptor_lead_times();
}

static bool host_prediction_in_same_range(const void* current,
                                          const void* predicted,
                                          size_t bytes)
{
    uintptr_t cur = reinterpret_cast<uintptr_t>(current);
    uintptr_t pred = reinterpret_cast<uintptr_t>(predicted);
    if (bytes > UINTPTR_MAX - pred) return false;
    uintptr_t pred_end = pred + bytes;

    std::lock_guard<std::mutex> lock(g_host_ranges_lock);
    for (const auto& range : g_host_ranges) {
        if (cur >= range.begin && cur < range.end &&
            pred >= range.begin && pred_end <= range.end) {
            return true;
        }
    }
    return false;
}

static bool host_address_in_registered_range(const void* address, size_t bytes)
{
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address);
    if (bytes > UINTPTR_MAX - begin) return false;
    const uintptr_t end = begin + bytes;
    std::lock_guard<std::mutex> lock(g_host_ranges_lock);
    for (const auto& range : g_host_ranges) {
        if (begin >= range.begin && end <= range.end) return true;
    }
    return false;
}

static int linear_prefetch_window()
{
    static int window = -1;
    if (window < 0) {
        const char* env = std::getenv("PIPE_LINEAR_PREFETCH_WINDOW");
        window = env ? std::atoi(env) : 8;
        if (window < 1) window = 1;
        // Eight blocks balanced readiness and unused work in the matched
        // multi-prefix sweep.  Longer values remain available for ablations.
        if (window > 128) window = 128;
        fprintf(stderr, "[SecureInfer] 线性预测窗口: %d blocks\n", window);
    }
    return window;
}

static bool periodic_address_predictor_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_PERIODIC_ADDRESS_PREDICTOR");
        const char* prefetch = std::getenv("PIPE_TRANSFER_PREFETCH");
        const bool predictor = env != nullptr && std::atoi(env) != 0;
        const bool transfer_prefetch =
            prefetch == nullptr || std::atoi(prefetch) != 0;
        enabled = (predictor && transfer_prefetch) ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] experimental history-only periodic predictor enabled\n");
        }
    }
    return enabled != 0;
}

using periodic_cpu_submission =
    std::tuple<const void*, size_t, uint64_t, uint64_t>;

static bool secure_infer_repetitive_predictor_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_SECURE_INFER_REPETITIVE_PREDICTOR");
        const char* prefetch = std::getenv("PIPE_TRANSFER_PREFETCH");
        const bool requested = env != nullptr && std::atoi(env) != 0;
        const bool transfer_prefetch =
            prefetch == nullptr || std::atoi(prefetch) != 0;
        enabled = requested && transfer_prefetch ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] public repetitive-sequence predictor enabled\n");
        }
    }
    return enabled != 0;
}

// The Figure-1 control keeps H2D consumption strictly serialized.  For the
// matched SecureInfer experiment we allow only the repetitive predictor's
// preparation workers to run ahead; demand consumption still takes the same
// inline serial path.  The flag is opt-in and has no effect on normal runs.
static bool secure_infer_repetitive_predictor_allow_force_serial()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env =
            std::getenv("PIPE_SECURE_INFER_ALLOW_FORCE_SERIAL_PREDICTOR");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] repetitive predictor allowed on the "
                    "force-serial demand path (experiment only)\n");
        }
    }
    return enabled != 0;
}

static int secure_infer_repetitive_window()
{
    static int window = -1;
    if (window < 0) {
        const char* env = std::getenv("PIPE_SECURE_INFER_REPETITIVE_WINDOW");
        window = env ? std::atoi(env) : 1;
        if (window < 1) window = 1;
        if (window > 128) window = 128;
        fprintf(stderr,
                "[SecureInfer] public repetitive predictor window: %d transfers\n",
                window);
    }
    return window;
}

static void reset_secure_infer_repetitive_predictor_locked(bool retire)
{
    if (retire) {
        const uint64_t retired_generation = predictor.generation;
        predictor.generation++;
        g_transfer_prefetch_records.clear();
        secure_infer_retire_prediction_generation(retired_generation);
    }
    predictor.secure_infer_repeat_record.clear();
    predictor.secure_infer_repeat_profiled = false;
    predictor.secure_infer_repeat_next_index = 0;
}

static int observe_and_schedule_secure_infer_repetitive_locked(
    const void* current, size_t count, aclrtStream stream,
    std::vector<periodic_cpu_submission>* cpu_submissions)
{
    using Entry = std::pair<const void*, size_t>;
    const Entry observed(current, count);
    auto& record = predictor.secure_infer_repeat_record;

    if (predictor.secure_infer_repeat_profiled) {
        const Entry& expected = record[predictor.secure_infer_repeat_next_index];
        if (observed != expected) {
            static std::atomic<uint64_t> mismatches{0};
            const uint64_t n = mismatches.fetch_add(1) + 1;
            if (n <= 16 || n % 1024 == 0) {
                fprintf(stderr,
                        "[SecureInfer] repetitive predictor mismatch: expected=%p/%zu actual=%p/%zu\n",
                        expected.first, expected.second, current, count);
            }
            reset_secure_infer_repetitive_predictor_locked(true);
            record.push_back(observed);
            return 0;
        }
        predictor.secure_infer_repeat_next_index =
            (predictor.secure_infer_repeat_next_index + 1) % record.size();
    } else {
        record.push_back(observed);
        const size_t size = record.size();
        bool found = size >= 64 && size % 2 == 0;
        if (found) {
            const size_t half = size / 2;
            for (size_t index = 0; index < half; ++index) {
                if (record[index] != record[half + index]) {
                    found = false;
                    break;
                }
            }
        }
        if (!found) return 0;

        record.resize(size / 2);
        predictor.secure_infer_repeat_profiled = true;
        predictor.secure_infer_repeat_next_index = 0;
        static std::atomic<uint64_t> learned{0};
        const uint64_t n = learned.fetch_add(1) + 1;
        if (n <= 16 || n % 1024 == 0) {
            fprintf(stderr,
                    "[SecureInfer] repetitive sequence learned: period=%zu transfers\n",
                    record.size());
        }
    }

    int scheduled = 0;
    const int window = secure_infer_repetitive_window();
    for (int offset = 0; offset < window; ++offset) {
        const size_t index =
            (predictor.secure_infer_repeat_next_index + offset) % record.size();
        const Entry& predicted = record[index];
        if (!host_address_in_registered_range(
                predicted.first, predicted.second)) {
            break;
        }
        uint64_t ticket = 0;
        if (!mark_transfer_prefetch_locked(
                predicted.first, predicted.second, &ticket)) {
            continue;
        }
        assign_to_workers(nullptr, predicted.first, predicted.second,
                          false, true, false, false, false, stream, ticket);
        if (cpu_submissions != nullptr && secure_infer_gcm_enabled() &&
            secure_infer_gcm_cpu_cipher_prefetch_enabled()) {
            cpu_submissions->emplace_back(
                predicted.first, predicted.second, ticket,
                predictor.generation);
        }
        scheduled++;
    }
    return scheduled;
}

static bool periodic_stride_bootstrap_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_PERIODIC_STRIDE_BOOTSTRAP");
        // A periodic pattern needs two complete cycles before it can be
        // trusted.  Do not leave the intervening fixed-stride runs entirely
        // uncovered: after two block-sized deltas, start the conservative
        // local-stride lookahead and let the periodic predictor take over once
        // a recurring cross-boundary pattern has been learned.  Keep an
        // explicit opt-out for the periodic-only ablation.
        enabled = env == nullptr ? 1 : (std::atoi(env) != 0 ? 1 : 0);
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] periodic predictor stride bootstrap enabled\n");
        }
    }
    return enabled != 0;
}

static bool periodic_run_transition_cache_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_PERIODIC_RUN_TRANSITION_CACHE");
        enabled = env == nullptr ? 1 : (std::atoi(env) != 0 ? 1 : 0);
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] periodic run-transition cache enabled\n");
        }
    }
    return enabled != 0;
}

static void update_periodic_run_pattern_locked(size_t copy_size,
                                               size_t run_blocks,
                                               intptr_t transition)
{
    constexpr size_t kMaxRunPatterns = 4;
    if (!periodic_run_transition_cache_enabled() || run_blocks < 2 ||
        transition == static_cast<intptr_t>(copy_size)) {
        return;
    }

    auto* pattern = static_cast<Predictor::PeriodicRunPattern*>(nullptr);
    for (auto& candidate : predictor.periodic_run_patterns) {
        if (candidate.copy_size == copy_size &&
            candidate.run_blocks == run_blocks) {
            pattern = &candidate;
            break;
        }
    }
    if (pattern == nullptr) {
        if (predictor.periodic_run_patterns.size() >= kMaxRunPatterns) {
            auto victim = std::min_element(
                predictor.periodic_run_patterns.begin(),
                predictor.periodic_run_patterns.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.last_used < rhs.last_used;
                });
            *victim = Predictor::PeriodicRunPattern{};
            pattern = &*victim;
        } else {
            predictor.periodic_run_patterns.emplace_back();
            pattern = &predictor.periodic_run_patterns.back();
        }
        pattern->copy_size = copy_size;
        pattern->run_blocks = run_blocks;
    }

    bool found = false;
    for (auto& observed : pattern->transitions) {
        if (observed.first == transition) {
            observed.second++;
            found = true;
            break;
        }
    }
    if (!found) pattern->transitions.emplace_back(transition, 1);
    pattern->last_transition = transition;
    pattern->last_used = ++predictor.periodic_pattern_clock;
}

static bool lookup_periodic_run_transition_locked(size_t copy_size,
                                                  size_t run_blocks,
                                                  intptr_t* transition)
{
    if (!periodic_run_transition_cache_enabled() || transition == nullptr ||
        run_blocks < 2) {
        return false;
    }
    for (auto& pattern : predictor.periodic_run_patterns) {
        if (pattern.copy_size != copy_size ||
            pattern.run_blocks != run_blocks ||
            pattern.transitions.empty()) {
            continue;
        }
        uint32_t best_count = 0;
        intptr_t best_transition = pattern.last_transition;
        for (const auto& observed : pattern.transitions) {
            if (observed.second > best_count ||
                (observed.second == best_count &&
                 observed.first == pattern.last_transition)) {
                best_count = observed.second;
                best_transition = observed.first;
            }
        }
        pattern.last_used = ++predictor.periodic_pattern_clock;
        *transition = best_transition;
        return true;
    }
    return false;
}

static void reset_periodic_address_predictor_locked()
{
    predictor.periodic_last_src = nullptr;
    predictor.periodic_copy_size = 0;
    predictor.periodic_deltas.clear();
    predictor.periodic_cycle.clear();
    predictor.periodic_next_phase = 0;
    // Preserve learned run patterns across local discontinuities.  Explicit
    // invalidation still clears them together with all speculative state.
    predictor.periodic_current_run_blocks = 0;
    predictor.periodic_active_run_blocks = 0;
}

static int observe_and_schedule_periodic_prefetches_locked(
    const void* current, size_t count, aclrtStream stream,
    std::vector<periodic_cpu_submission>* cpu_submissions = nullptr)
{
    constexpr size_t kMaxPeriod = 128;
    const uintptr_t current_addr = reinterpret_cast<uintptr_t>(current);

    if (predictor.periodic_last_src == nullptr ||
        predictor.periodic_copy_size != count ||
        !host_address_in_registered_range(
            predictor.periodic_last_src, count) ||
        !host_address_in_registered_range(current, count)) {
        reset_periodic_address_predictor_locked();
        predictor.periodic_last_src = current;
        predictor.periodic_copy_size = count;
        predictor.periodic_current_run_blocks = 1;
        return 0;
    }

    const uintptr_t previous_addr =
        reinterpret_cast<uintptr_t>(predictor.periodic_last_src);
    const intptr_t observed_delta =
        current_addr >= previous_addr
            ? static_cast<intptr_t>(current_addr - previous_addr)
            : -static_cast<intptr_t>(previous_addr - current_addr);

    if (observed_delta == static_cast<intptr_t>(count)) {
        predictor.periodic_current_run_blocks = std::max<size_t>(
            1, predictor.periodic_current_run_blocks) + 1;
    } else {
        const size_t completed_run_blocks = std::max<size_t>(
            1, predictor.periodic_current_run_blocks);
        update_periodic_run_pattern_locked(
            count, completed_run_blocks, observed_delta);
        predictor.periodic_active_run_blocks = completed_run_blocks;
        predictor.periodic_current_run_blocks = 1;
    }

    if (!predictor.periodic_cycle.empty()) {
        if (observed_delta == predictor.periodic_cycle[
                                  predictor.periodic_next_phase]) {
            predictor.periodic_next_phase =
                (predictor.periodic_next_phase + 1) %
                predictor.periodic_cycle.size();
        } else {
            predictor.periodic_cycle.clear();
            predictor.periodic_next_phase = 0;
        }
    }
    predictor.periodic_deltas.push_back(observed_delta);
    predictor.periodic_last_src = current;

    if (predictor.periodic_cycle.empty()) {
        const size_t n = predictor.periodic_deltas.size();
        const size_t upper = std::min(kMaxPeriod, n / 2);
        for (size_t period = 2; period <= upper; ++period) {
            bool equal = true;
            bool has_boundary = false;
            for (size_t offset = 0; offset < period; ++offset) {
                const intptr_t lhs =
                    predictor.periodic_deltas[n - 2 * period + offset];
                const intptr_t rhs =
                    predictor.periodic_deltas[n - period + offset];
                if (lhs != rhs) {
                    equal = false;
                    break;
                }
                if (rhs != static_cast<intptr_t>(count)) has_boundary = true;
            }
            if (equal && has_boundary) {
                predictor.periodic_cycle.assign(
                    predictor.periodic_deltas.end() - period,
                    predictor.periodic_deltas.end());
                predictor.periodic_next_phase = 0;
                static std::atomic<uint64_t> detected{0};
                const uint64_t detected_count =
                    detected.fetch_add(1, std::memory_order_relaxed) + 1;
                if (detected_count <= 32 || detected_count % 1024 == 0) {
                    fprintf(stderr,
                            "[SecureInfer] history-only periodic pattern learned: period=%zu\n",
                            period);
                }
                break;
            }
        }
    }

    // Bound retained history without affecting an active cycle.  The most
    // recent two maximum-size periods are sufficient for relearning.
    if (predictor.periodic_deltas.size() > 2 * kMaxPeriod) {
        predictor.periodic_deltas.erase(
            predictor.periodic_deltas.begin(),
            predictor.periodic_deltas.end() - 2 * kMaxPeriod);
    }

    const bool stride_bootstrap =
        predictor.periodic_cycle.empty() &&
        periodic_stride_bootstrap_enabled() &&
        predictor.periodic_deltas.size() >= 2 &&
        predictor.periodic_deltas[predictor.periodic_deltas.size() - 1] ==
            static_cast<intptr_t>(count) &&
        predictor.periodic_deltas[predictor.periodic_deltas.size() - 2] ==
            static_cast<intptr_t>(count);
    intptr_t run_transition = 0;
    const bool run_transition_bootstrap =
        predictor.periodic_cycle.empty() &&
        lookup_periodic_run_transition_locked(
            count, predictor.periodic_active_run_blocks, &run_transition);
    if (predictor.periodic_cycle.empty() && !stride_bootstrap &&
        !run_transition_bootstrap) {
        return 0;
    }

    uintptr_t predicted_addr = current_addr;
    size_t predicted_run_blocks = predictor.periodic_current_run_blocks;
    int scheduled = 0;
    const int window = linear_prefetch_window();
    for (int offset = 0; offset < window; ++offset) {
        intptr_t delta = 0;
        if (!predictor.periodic_cycle.empty()) {
            delta = predictor.periodic_cycle[
                (predictor.periodic_next_phase + offset) %
                predictor.periodic_cycle.size()];
        } else if (run_transition_bootstrap &&
                   predicted_run_blocks ==
                       predictor.periodic_active_run_blocks) {
            delta = run_transition;
            predicted_run_blocks = 1;
        } else {
            delta = static_cast<intptr_t>(count);
            predicted_run_blocks++;
        }
        if ((delta > 0 && static_cast<uintptr_t>(delta) >
                              UINTPTR_MAX - predicted_addr) ||
            (delta < 0 && static_cast<uintptr_t>(-delta) > predicted_addr)) {
            break;
        }
        predicted_addr = delta >= 0
            ? predicted_addr + static_cast<uintptr_t>(delta)
            : predicted_addr - static_cast<uintptr_t>(-delta);
        const void* predicted = reinterpret_cast<const void*>(predicted_addr);
        if (!host_address_in_registered_range(predicted, count)) break;
        uint64_t ticket = 0;
        if (!mark_transfer_prefetch_locked(predicted, count, &ticket)) continue;
        assign_to_workers(nullptr, predicted, count, false, true, false,
                          false, false, stream, ticket);
        if (cpu_submissions != nullptr && secure_infer_gcm_enabled() &&
            secure_infer_gcm_cpu_cipher_prefetch_enabled()) {
            cpu_submissions->emplace_back(
                predicted, count, ticket, predictor.generation);
        }
        scheduled++;
    }
    if (scheduled > 0) {
        static std::atomic<uint64_t> schedule_logs{0};
        const uint64_t n =
            schedule_logs.fetch_add(1, std::memory_order_relaxed) + 1;
        const char* verbose_env = std::getenv("PIPE_VERBOSE_PREDICT_LOGS");
        const bool log_all = verbose_env && std::atoi(verbose_env) != 0;
        if (log_all || n <= 16 || n % 2048 == 0) {
            fprintf(stderr,
                    "[SecureInfer] 周期窗口预取: 当前=%p, 新增=%d, period=%zu, stride_bootstrap=%d, run_transition=%d\n",
                    current, scheduled, predictor.periodic_cycle.size(),
                    stride_bootstrap ? 1 : 0,
                    run_transition_bootstrap ? 1 : 0);
        }
    }
    return scheduled;
}

static bool verbose_predict_logs()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_VERBOSE_PREDICT_LOGS");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static bool sampled_predict_log(std::atomic<uint64_t>& counter,
                                uint64_t first = 16,
                                uint64_t every = 1024)
{
    if (verbose_predict_logs()) return true;
    uint64_t n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    return n <= first || (every > 0 && n % every == 0);
}

static int schedule_linear_prefetches(const void* current, size_t count, aclrtStream stream)
{
    const int window = linear_prefetch_window();
    uintptr_t cur = reinterpret_cast<uintptr_t>(current);
    uintptr_t stride = predictor.linear_stride ? predictor.linear_stride : count;
    uintptr_t scheduled_until = reinterpret_cast<uintptr_t>(predictor.linear_prefetch_until);
    int scheduled = 0;
    const void* last_scheduled = nullptr;

    uintptr_t next = cur;
    for (int i = 1; i <= window; ++i) {
        if (stride > UINTPTR_MAX - next) break;
        next += stride;

        // linear_prefetch_until is exclusive. Any address below it has already
        // been queued by an earlier refill; avoid duplicate encryption work.
        if (predictor.linear_prefetch_until != nullptr && next < scheduled_until) {
            continue;
        }

        const void* predicted = reinterpret_cast<const void*>(next);
        if (!host_prediction_in_same_range(current, predicted, count)) {
            break;
        }

        uint64_t ticket = 0;
        if (!mark_transfer_prefetch_locked(predicted, count, &ticket)) {
            continue;
        }
        assign_to_workers(nullptr, predicted, count, false, true, false,
                          false, false, stream, ticket);
        if (stride > UINTPTR_MAX - next) break;
        predictor.linear_prefetch_until = reinterpret_cast<const void*>(next + stride);
        scheduled_until = reinterpret_cast<uintptr_t>(predictor.linear_prefetch_until);
        last_scheduled = predicted;
        scheduled++;
    }

    if (scheduled > 0) {
        static std::atomic<uint64_t> log_count{0};
        if (sampled_predict_log(log_count, 16, 2048)) {
            fprintf(stderr,
                    "[SecureInfer] 线性窗口预取: 当前=%p, 新增=%d, 最远=%p\n",
                    current, scheduled, last_scheduled);
        }
    }
    return scheduled;
}

static int predict_encrypt_workers_default()
{
    static int workers = -1;
    if (workers < 0) {
        const char* env = std::getenv("PIPE_PREDICT_ENCRYPT_WORKERS");
        workers = env ? std::atoi(env) : 4;
        if (workers < 1) workers = 1;
        if (workers > 8) workers = 8;
        fprintf(stderr, "[SecureInfer] 预测加密 worker 数: %d\n", workers);
    }
    return workers;
}

static std::mutex g_profile_lock;
static FILE* g_profile_file = nullptr;
static bool g_profile_init = false;
static char g_profile_run_id[128] = "";
static uint64_t g_profile_rows_since_flush = 0;
static uint64_t g_d2h_count_since_persist = 0;
static const char* PROFILE_MARKER_PATH = "runtime_profile_on";
static uint64_t g_d2h_kv_count = 0;
static uint64_t g_d2h_kv_bytes = 0;

// This is only an I/O batching threshold; it must not encode model geometry.
// Runtime descriptor and memcpy records carry the actual logical boundaries.
static uint64_t profile_flush_rows()
{
    static uint64_t rows = 0;
    if (rows == 0) {
        const char* env = std::getenv("PIPE_PROFILE_FLUSH_ROWS");
        long parsed = env ? std::strtol(env, nullptr, 10) : 256;
        rows = parsed > 0 ? static_cast<uint64_t>(parsed) : 256;
    }
    return rows;
}

uint64_t secure_infer_now_us()
{
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

static bool read_profile_run_id(char* buf, size_t buf_size)
{
    FILE* f = fopen(PROFILE_MARKER_PATH, "r");
    if (!f) return false;
    if (!fgets(buf, buf_size, f)) {
        fclose(f);
        return false;
    }
    fclose(f);
    buf[buf_size - 1] = '\0';
    size_t len = strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' ')) {
        buf[len - 1] = '\0';
        len--;
    }
    return len > 0;
}

static const char* profile_path_env()
{
    const char* path = getenv("PIPE_ACL_PROFILE_PATH");
    if (!path || path[0] == '\0') path = getenv("PIPE_SERIAL_PROFILE_PATH");
    if (!path || path[0] == '\0') path = "runtime_profile.csv";
    return path;
}

static bool profile_env_enabled()
{
    const char* enable = getenv("PIPE_ACL_PROFILE");
    if (!enable || enable[0] == '\0') enable = getenv("PIPE_SERIAL_PROFILE");
    const char* path = getenv("PIPE_ACL_PROFILE_PATH");
    if (!path || path[0] == '\0') path = getenv("PIPE_SERIAL_PROFILE_PATH");
    // An explicit enable/disable flag takes precedence over the output path.
    // Previously PIPE_ACL_PROFILE=0 still enabled profiling whenever a path
    // was configured, which made clean profile-on/off experiments impossible.
    if (enable && enable[0] != '\0') return atoi(enable) != 0;
    return path && path[0] != '\0';
}

static void init_profile_file_for_run(const char* run_id)
{
    if (!profile_env_enabled()) return;
    const char* path = profile_path_env();
    if (g_profile_file) {
        fclose(g_profile_file);
        g_profile_file = nullptr;
    }
    g_d2h_kv_count = 0;
    g_d2h_kv_bytes = 0;
    g_profile_rows_since_flush = 0;
    g_d2h_count_since_persist = 0;
    g_profile_file = fopen(path, "w");
    if (!g_profile_file) {
        fprintf(stderr, "[SecureInfer ACL] Failed to open profile file '%s': %s\n", path, strerror(errno));
        return;
    }
    snprintf(g_profile_run_id, sizeof(g_profile_run_id), "%s", run_id);
    g_profile_init = true;
    fprintf(g_profile_file,
            "direction,bytes,aligned_bytes,total_us,cpu_crypto_us,device_pipeline_us,host_wait_us,copy_enqueue_us,kernel_enqueue_us,h2d_device_us,kernel_execute_us,d2d_device_us,device_total_us,device_timing_valid,crypto_mode,cpu_prediction_hit\n");
    fflush(g_profile_file);
    fprintf(stderr, "[SecureInfer ACL] Profiling run %s to %s\n", run_id, path);
}

static bool profile_prepare_if_enabled()
{
    char run_id[128] = "";
    if (!read_profile_run_id(run_id, sizeof(run_id))) return false;
    if (!g_profile_init || strcmp(g_profile_run_id, run_id) != 0) {
        init_profile_file_for_run(run_id);
    }
    return g_profile_file != nullptr;
}

void profile_record(const char* direction,
                    size_t bytes,
                    size_t aligned_bytes,
                    uint64_t total_us,
                    uint64_t cpu_crypto_us,
                    uint64_t device_pipeline_us,
                    uint64_t host_wait_us,
                    uint64_t copy_enqueue_us,
                    uint64_t kernel_enqueue_us)
{
    profile_record_precise(nullptr,
                           direction,
                           bytes,
                           aligned_bytes,
                           total_us,
                           cpu_crypto_us,
                           device_pipeline_us,
                           host_wait_us,
                           copy_enqueue_us,
                           kernel_enqueue_us,
                           0,
                           0,
                           0,
                           0,
                           false);
}

bool profile_begin_precise_capture(char* run_id, size_t run_id_size)
{
    if (run_id == nullptr || run_id_size == 0) return false;
    run_id[0] = '\0';
    const char* enabled = getenv("PIPE_ACL_PRECISE_PROFILE");
    if (!enabled || atoi(enabled) == 0) return false;

    std::lock_guard<std::mutex> lock(g_profile_lock);
    if (!profile_prepare_if_enabled() || g_profile_run_id[0] == '\0') return false;
    snprintf(run_id, run_id_size, "%s", g_profile_run_id);
    return true;
}

void profile_record_precise(const char* run_id,
                            const char* direction,
                            size_t bytes,
                            size_t aligned_bytes,
                            uint64_t total_us,
                            uint64_t cpu_crypto_us,
                            uint64_t device_pipeline_us,
                            uint64_t host_wait_us,
                            uint64_t copy_enqueue_us,
                            uint64_t kernel_enqueue_us,
                            uint64_t h2d_device_us,
                            uint64_t kernel_execute_us,
                            uint64_t d2d_device_us,
                            uint64_t device_total_us,
                            bool device_timing_valid,
                            int crypto_mode,
                            int cpu_prediction_hit)
{
    std::lock_guard<std::mutex> lock(g_profile_lock);
    if (run_id != nullptr) {
        // Asynchronous samples may finish just after the client removes the
        // marker.  Accept them only while their original output file/run is
        // still current; this prevents a late sample contaminating the next
        // request's CSV.
        if (!g_profile_file || strcmp(g_profile_run_id, run_id) != 0) return;
    } else if (!profile_prepare_if_enabled()) {
        return;
    }
    fprintf(g_profile_file, "%s,%zu,%zu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%d,%d,%d\n",
            direction,
            bytes,
            aligned_bytes,
            (unsigned long)total_us,
            (unsigned long)cpu_crypto_us,
            (unsigned long)device_pipeline_us,
            (unsigned long)host_wait_us,
            (unsigned long)copy_enqueue_us,
            (unsigned long)kernel_enqueue_us,
            (unsigned long)h2d_device_us,
            (unsigned long)kernel_execute_us,
            (unsigned long)d2d_device_us,
            (unsigned long)device_total_us,
            device_timing_valid ? 1 : 0,
            crypto_mode,
            cpu_prediction_hit);
    g_profile_rows_since_flush++;
    if (g_profile_rows_since_flush >= profile_flush_rows()) {
        fflush(g_profile_file);
        g_profile_rows_since_flush = 0;
    }
}

static void persist_d2h_count_locked()
{
    char count_path[1024];
    snprintf(count_path, sizeof(count_path), "%s.d2h_count", profile_path_env());
    FILE* f = fopen(count_path, "w");
    if (f) {
        fprintf(f, "%lu,%lu\n", (unsigned long)g_d2h_kv_count, (unsigned long)g_d2h_kv_bytes);
        fclose(f);
    }
}

static void profile_count_d2h_kv_totals(uint64_t count,
                                        uint64_t total_bytes,
                                        const char* run_id = nullptr)
{
    if (count == 0 || total_bytes == 0) return;
    std::lock_guard<std::mutex> lock(g_profile_lock);
    if (run_id != nullptr) {
        if (!g_profile_file || strcmp(g_profile_run_id, run_id) != 0) return;
    } else if (!profile_prepare_if_enabled()) {
        return;
    }
    g_d2h_kv_count += count;
    g_d2h_kv_bytes += total_bytes;
    g_d2h_count_since_persist += count;
    if (g_d2h_count_since_persist >= profile_flush_rows()) {
        persist_d2h_count_locked();
        g_d2h_count_since_persist %= profile_flush_rows();
    }
}

void profile_count_d2h_kv_batch(size_t bytes,
                                uint64_t count,
                                const char* run_id = nullptr)
{
    if (bytes == 0 || count == 0 || count > UINT64_MAX / bytes) return;
    profile_count_d2h_kv_totals(count, bytes * count, run_id);
}

void profile_count_d2h_kv(size_t bytes)
{
    profile_count_d2h_kv_batch(bytes, 1, nullptr);
}

// void __attribute__((destructor)) secure_infer_fini() {
//     fprintf(stderr, "[SecureInfer] 🛑 正在停止 Worker 线程...\n");
//     g_running = false;
//     // 稍微等一下让线程有机会退出
//     usleep(200000);
//     fprintf(stderr, "[SecureInfer] 👋 SecureInfer 退出完成\n");
// }

// [修改加载函数，加载新NPU算子]
static void load_aes_library() {
    if (g_launch_kernel && g_launch_inline_iv_kernel) return;

    fprintf(stderr, "[SecureInfer] 🔄 正在加载 NPU 算子库...\n");

    // 1. 先加载 ACL 库 (RTLD_GLOBAL)
    void* acl_handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!acl_handle) {
        acl_handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!acl_handle) {
        fprintf(stderr, "[SecureInfer] ⚠️ ACL库加载失败（非致命）: %s\n", dlerror());
    }

    // 2. 加载 kernel 库并获取符号
    const char* kernel_lib_path = std::getenv("PIPE_AES_KERNEL_LIB");
    if (kernel_lib_path == nullptr || kernel_lib_path[0] == '\0') {
        fprintf(stderr, "[runtime] PIPE_AES_KERNEL_LIB is required\n");
        return;
    }
    void* kernel_handle = dlopen(kernel_lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!kernel_handle) {
        fprintf(stderr, "[SecureInfer] ❌ 加载 Kernel 库失败: %s\n", dlerror());
        return;
    }

    // 尝试找 C++ 修饰的符号名
    void* sym = dlsym(kernel_handle, "_Z13add_custom_dojPvPhS0_S0_S0_jj");
    if (sym) {
        g_launch_kernel = (LaunchAESKernel_t)sym;
        fprintf(stderr, "[SecureInfer] ✅ 成功加载 NPU 加解密算子库 (C++ 符号)\n");
        fprintf(stderr, "[SecureInfer]    Addr: %p\n", (void*)g_launch_kernel);
    } else {
        // 尝试 C 符号
        sym = dlsym(kernel_handle, "add_custom_do");
        if (sym) {
            g_launch_kernel = (LaunchAESKernel_t)sym;
            fprintf(stderr, "[SecureInfer] ✅ 成功加载 NPU 加解密算子库 (C 符号)\n");
            fprintf(stderr, "[SecureInfer]    Addr: %p\n", (void*)g_launch_kernel);
        } else {
            fprintf(stderr, "[SecureInfer] ❌ 无法找到 add_custom_do 符号: %s\n", dlerror());
        }
    }

    g_launch_inline_iv_kernel = reinterpret_cast<LaunchInlineAESKernel_t>(
        dlsym(kernel_handle, "add_custom_iv_do"));
    if (g_launch_inline_iv_kernel != nullptr) {
        fprintf(stderr,
                "[SecureInfer] ✅ 加载统一 inline-IV 算子入口: %p\n",
                reinterpret_cast<void*>(g_launch_inline_iv_kernel));
    } else {
        fprintf(stderr,
                "[SecureInfer] ❌ 无法找到 add_custom_iv_do: %s\n",
                dlerror());
    }
}

static bool load_verified_aiv_keystream_library()
{
    if (g_launch_verified_aiv_keystream != nullptr) return true;
    const char* configured = std::getenv("PIPE_GCM_AIV_KEYSTREAM_LIB");
    const char* path = configured;
    if (path == nullptr || path[0] == '\0') {
        fprintf(stderr, "[runtime] PIPE_GCM_AIV_KEYSTREAM_LIB is required\n");
        return false;
    }
    g_verified_aiv_lib_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (g_verified_aiv_lib_handle == nullptr) {
        fprintf(stderr, "[SecureInfer ERROR] cannot load verified AIV AES: %s: %s\n",
                path, dlerror());
        return false;
    }
    g_launch_verified_aiv_keystream =
        reinterpret_cast<LaunchVerifiedAivKeystream_t>(
            dlsym(g_verified_aiv_lib_handle, "add_custom_do"));
    if (g_launch_verified_aiv_keystream == nullptr) {
        fprintf(stderr, "[SecureInfer ERROR] verified AIV AES entry missing: %s\n",
                dlerror());
        return false;
    }
    fprintf(stderr, "[SecureInfer] verified AIV AES-CTR keystream backend loaded: %s\n",
            path);
    return true;
}

static void pack_inline_iv(const unsigned char iv[iv_length],
                           uint32_t words[4])
{
    for (int word = 0; word < 4; ++word) {
        int base = word * 4;
        words[word] = (static_cast<uint32_t>(iv[base]) << 24) |
                      (static_cast<uint32_t>(iv[base + 1]) << 16) |
                      (static_cast<uint32_t>(iv[base + 2]) << 8) |
                      static_cast<uint32_t>(iv[base + 3]);
    }
}

static bool enqueue_gcm_tag_mask(aclrtStream stream, void* tag_mask,
                                 void* device_key,
                                 const unsigned char payload_iv[iv_length])
{
    if (g_launch_aicpu_aes_ctr == nullptr || tag_mask == nullptr ||
        device_key == nullptr) {
        return false;
    }
    uint32_t words[4] = {0, 0, 0, 1};
    pack_inline_iv(payload_iv, words);
    // J0 for a 96-bit GCM nonce is nonce||1; the payload starts at nonce||2.
    words[3] = 1;
    return aclrtMemsetAsync(tag_mask, 16, 0, 16, stream) == ACL_SUCCESS &&
        g_launch_aicpu_aes_ctr(
            stream, tag_mask, device_key, tag_mask, 16,
            words[0], words[1], words[2], words[3]) == 0;
}

static std::mutex g_inline_h2d_lock;
static std::mutex g_d2h_crypto_lock;

static bool wait_h2d_device_crypto_ready(aclrtStream stream)
{
    if (stream == nullptr) return false;
    void* event_ptr = g_h2d_device_crypto_ready_event.load(
        std::memory_order_acquire);
    if (event_ptr == nullptr) {
        // A producer must never read a key whose initialization ordering is
        // unknown.  The normal path creates this event in
        // ensure_h2d_device_crypto(); a missing event is therefore a safe
        // admission failure rather than an unchecked launch.
        return false;
    }
    const aclError ret = real_aclrtStreamWaitEvent(
        stream, static_cast<aclrtEvent>(event_ptr));
    if (ret != ACL_SUCCESS) {
        fprintf(stderr,
                "[SecureInfer ERROR] device-crypto-ready event wait failed "
                "stream=%p event=%p ret=%d\n",
                stream, event_ptr, static_cast<int>(ret));
        return false;
    }
    return true;
}

struct d2h_stream_staging {
    void* temp_in = nullptr;
    void* temp_out = nullptr;
    size_t bytes = 0;
};
static std::unordered_map<void*, d2h_stream_staging> g_d2h_staging_by_stream;

// The secure D2H data plane uses the framework D2H stream for NPU-side
// encrypt/pack work and a dedicated copy stream for PCIe.  Several staging
// slots let the producer encrypt tensor batch i+1 while batch i is copied.
// A slot is recycled only after its previous D2H event completes.
struct d2h_pipeline_slot {
    void* temp_in = nullptr;
    void* temp_out = nullptr;
    size_t bytes = 0;
    aclrtEvent crypto_done = nullptr;
    aclrtEvent copy_done = nullptr;
    aclrtEvent ghash_done = nullptr;
    bool used = false;
};

struct d2h_stream_pipeline {
    aclrtStream user_stream = nullptr;
    aclrtStream copy_stream = nullptr;
    aclrtStream ghash_stream = nullptr;
    std::vector<d2h_pipeline_slot> slots;
    size_t next_slot = 0;
};

static std::unordered_map<void*, std::unique_ptr<d2h_stream_pipeline>>
    g_d2h_pipelines;
// Old allocations are intentionally retained when a stream grows.  They may
// still be referenced by already queued work; freeing them on the submission
// thread would reintroduce a synchronization.  KV transfers are fixed-size in
// the current backend, so this list normally remains empty.
static std::vector<std::pair<void*, void*>> g_retired_d2h_staging;
static void* g_inline_h2d_temp_in = nullptr;
static void* g_inline_h2d_temp_out = nullptr;
static size_t g_inline_h2d_temp_size = 0;
static void* g_inline_cipher_host = nullptr;
static size_t g_inline_cipher_host_size = 0;
static bool g_inline_cipher_host_pinned = false;

static bool d2h_async_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_D2H_ASYNC");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] ordered async D2H host decrypt: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static bool d2h_stage_pipeline_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_D2H_STAGE_PIPELINE");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] three-stage secure D2H pipeline: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static bool d2h_ghash_stream_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_D2H_GHASH_STREAM");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] independent D2H GHASH stream: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

// Keep the legacy AES-CTR + separate GHASH path available for controlled
// A/B measurements.  The fused encrypt-and-tag kernel remains the default;
// setting PIPE_D2H_FUSED_GCM=0 only disables that D2H optimization and does
// not change the crypto mode, AAD, or host-side authentication checks.
static bool d2h_fused_gcm_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_D2H_FUSED_GCM");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] fused D2H AES-GCM encrypt/tag: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static size_t d2h_pipeline_slot_count()
{
    static size_t slots = 0;
    if (slots == 0) {
        const char* env = std::getenv("PIPE_D2H_PIPELINE_SLOTS");
        long parsed = env ? std::atol(env) : 3;
        slots = static_cast<size_t>(std::max(1L, std::min(8L, parsed)));
    }
    return slots;
}

static d2h_stream_pipeline* get_d2h_pipeline(aclrtStream user_stream)
{
    if (!d2h_stage_pipeline_enabled()) return nullptr;
    auto& entry = g_d2h_pipelines[user_stream];
    if (entry) return entry.get();

    auto pipeline = std::make_unique<d2h_stream_pipeline>();
    pipeline->user_stream = user_stream;
    if (real_aclrtCreateStream(&pipeline->copy_stream) != ACL_SUCCESS ||
        pipeline->copy_stream == nullptr) {
        fprintf(stderr,
                "[SecureInfer WARN] cannot create secure D2H copy stream; "
                "using ordered compatibility path\n");
        return nullptr;
    }

    pipeline->slots.resize(d2h_pipeline_slot_count());
    for (auto& slot : pipeline->slots) {
        if (real_aclrtCreateEvent(&slot.crypto_done) != ACL_SUCCESS ||
            real_aclrtCreateEvent(&slot.copy_done) != ACL_SUCCESS ||
            slot.crypto_done == nullptr || slot.copy_done == nullptr) {
            fprintf(stderr,
                    "[SecureInfer WARN] cannot create secure D2H pipeline events; "
                    "using ordered compatibility path\n");
            return nullptr;
        }
    }

    if (d2h_ghash_stream_enabled()) {
        if (real_aclrtCreateStream(&pipeline->ghash_stream) != ACL_SUCCESS ||
            pipeline->ghash_stream == nullptr) {
            fprintf(stderr,
                    "[SecureInfer WARN] cannot create secure D2H GHASH stream; "
                    "using inline GHASH path\n");
            pipeline->ghash_stream = nullptr;
        } else {
            for (auto& slot : pipeline->slots) {
                if (real_aclrtCreateEvent(&slot.ghash_done) != ACL_SUCCESS ||
                    slot.ghash_done == nullptr) {
                    fprintf(stderr,
                            "[SecureInfer WARN] cannot create GHASH done event; "
                            "using inline GHASH path\n");
                    pipeline->ghash_stream = nullptr;
                    break;
                }
            }
        }
    }
    fprintf(stderr,
            "[SecureInfer] secure D2H pipeline initialized: slots=%zu\n",
            pipeline->slots.size());
    entry = std::move(pipeline);
    return entry.get();
}

static bool ensure_d2h_pipeline_slot(d2h_pipeline_slot& slot, size_t bytes,
                                     aclrtStream copy_stream)
{
    if (slot.used) {
        // The submission thread is intentionally paced by a small number of
        // slots.  This wait never blocks EngineCore (submission runs in the
        // write-behind worker) and prevents event/buffer reuse before DMA.
        const uint64_t wait_start_us = secure_infer_now_us();
        aclError wait_ret = real_aclrtSynchronizeEvent(slot.copy_done);
        if (wait_ret != ACL_SUCCESS) {
            // Some CANN builds reject host synchronization on a reusable
            // event after a cross-stream wait.  Draining the dedicated copy
            // stream is conservative but still happens only when the bounded
            // slot ring wraps; it must not turn a recoverable pressure point
            // into a failed KV store.
            wait_ret = copy_stream != nullptr
                ? real_aclrtSynchronizeStream(copy_stream)
                : wait_ret;
        }
        if (wait_ret != ACL_SUCCESS) {
            fprintf(stderr,
                    "[SecureInfer ERROR] cannot retire D2H staging slot ret=%d\n",
                    static_cast<int>(wait_ret));
            return false;
        }
        const uint64_t wait_us = secure_infer_now_us() - wait_start_us;
        if (wait_us >= 10000 &&
            std::getenv("PIPE_DIAG_SLOT_WAIT") != nullptr) {
            fprintf(stderr,
                    "[SecureInfer DIAG] D2H slot recycle wait_us=%llu "
                    "copy_stream=%p bytes=%zu\n",
                    static_cast<unsigned long long>(wait_us),
                    copy_stream, bytes);
        }
    }
    if (slot.bytes >= bytes && slot.temp_out) return true;

    constexpr size_t kMaxIntegrityBytes = gcm_metadata_slab_bytes;
    void* new_temp_out = nullptr;
    aclError alloc_out = real_aclrtMalloc(
        &new_temp_out, bytes + kMaxIntegrityBytes);
    if (alloc_out != ACL_SUCCESS || new_temp_out == nullptr) {
        if (new_temp_out) real_aclrtFree(new_temp_out);
        return false;
    }
    if (slot.temp_in || slot.temp_out) {
        g_retired_d2h_staging.emplace_back(slot.temp_in, slot.temp_out);
    }
    // The AES kernel can read the original KV slice directly.  Keeping a
    // second tensor-sized plaintext staging buffer doubled the per-slot NPU
    // footprint and added an unnecessary D2D copy.  temp_out is therefore
    // both the encrypt-and-pack destination and the GCM metadata slab.
    slot.temp_in = nullptr;
    slot.temp_out = new_temp_out;
    slot.bytes = bytes;
    return true;
}

struct d2h_host_decrypt_task {
    void* dst = nullptr;
    void* device_ciphertext = nullptr;
    size_t bytes = 0;
    unsigned char key[key_length] = {};
    unsigned char iv[iv_length] = {};
    unsigned char inline_tag[16] = {};
    unsigned char aad[gcm_aad_length] = {};
    bool aad_valid = false;
    const unsigned char* auth_tag = nullptr;
    void* ghash_tag_out = nullptr;
    void* ghash_tag_mask = nullptr;
    uint64_t submit_start_us = 0;
    uint64_t copy_enqueue_us = 0;
    uint64_t kernel_enqueue_us = 0;
    bool profile_active = false;
    char profile_run_id[128] = {};
};

struct d2h_host_decrypt_batch;

struct d2h_host_decrypt_group {
    std::vector<d2h_host_decrypt_task> tasks;
    d2h_host_decrypt_batch* batch = nullptr;
    d2h_stream_pipeline* pipeline = nullptr;
    d2h_pipeline_slot* slot = nullptr;
    aclrtStream ready_stream = nullptr;
    size_t expected_tasks = 0;
    size_t staging_bytes = 0;
    size_t payload_caruntimety = 0;
    unsigned char* host_tags = nullptr;
    unsigned char* host_tag_masks = nullptr;
    void* dev_ghash_h = nullptr;
    std::vector<void*> ghash_tag_out_stack;
    std::vector<void*> ghash_tag_mask_stack;
    std::vector<void*> ghash_aad_stack;
    // Wall-clock interval from the point at which the deferred GHASH batch is
    // fully enqueued to the copy-stream callback that is ordered after its
    // completion.  This includes AICPU queueing, so it is the relevant
    // authentication gate latency under speculative-keystream pressure.
    uint64_t ghash_gate_submit_us = 0;
    size_t ghash_gate_tasks = 0;

    ~d2h_host_decrypt_group()
    {
        if (host_tags != nullptr) real_aclrtFreeHost(host_tags);
        if (host_tag_masks != nullptr) real_aclrtFreeHost(host_tag_masks);
    }
};

struct d2h_host_decrypt_batch {
    std::vector<std::unique_ptr<d2h_host_decrypt_group>> groups;
    std::vector<aclrtStream> streams;
    std::vector<aclrtStream> copy_streams;
    std::vector<aclrtEvent> join_events;
    std::atomic<size_t> completed_groups{0};
    size_t physical_slices = 0;
    size_t pcie_submissions = 0;
    std::mutex completion_lock;
    std::condition_variable completion_cv;
    bool profile_checked = false;
    bool profile_active = false;
    char profile_run_id[128] = {};
};

// transfer_async is invoked synchronously on one EngineCore submission
// thread.  The external SecureInfer module brackets a D2H transfer job, allowing
// all physical KV slices to share one stream-ordered host callback while
// preserving the final event's plaintext-ready semantics.
static thread_local std::unique_ptr<d2h_host_decrypt_batch> g_active_d2h_batch;
static thread_local d2h_host_decrypt_group* g_active_d2h_group = nullptr;

static std::mutex g_d2h_cpu_queue_lock;
static std::condition_variable g_d2h_cpu_queue_cv;
static std::deque<d2h_host_decrypt_group*> g_d2h_cpu_queue;
static std::once_flag g_d2h_cpu_workers_once;

static bool capture_d2h_profile_run(char* run_id, size_t run_id_size)
{
    if (run_id == nullptr || run_id_size == 0) return false;
    run_id[0] = '\0';
    std::lock_guard<std::mutex> lock(g_profile_lock);
    char current_run[128] = {};
    if (!read_profile_run_id(current_run, sizeof(current_run))) return false;
    if (!g_profile_init || strcmp(g_profile_run_id, current_run) != 0) {
        init_profile_file_for_run(current_run);
    }
    if (!g_profile_file) return false;
    snprintf(run_id, run_id_size, "%s", g_profile_run_id);
    return true;
}

// Test-only fault injection at the D2H trust boundary.  The flip is applied
// after ciphertext reaches host memory and before host AES-GCM
// authentication.  Production never sets this environment variable.
static bool secure_infer_maybe_tamper_gcm_d2h(void* ciphertext, size_t bytes)
{
    static int enabled = -1;
    static std::atomic<bool> injected{false};
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_TAMPER_D2H");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
    }
    if (enabled == 0 || ciphertext == nullptr || bytes == 0 ||
        injected.exchange(true, std::memory_order_acq_rel)) {
        return false;
    }
    static_cast<unsigned char*>(ciphertext)[0] ^= 0x01;
    return true;
}

static void d2h_host_decrypt_one(d2h_host_decrypt_task& task,
                                 EVP_CIPHER_CTX* host_ctx)
{
    const uint64_t callback_start_us = secure_infer_now_us();
    const bool tamper_injected = secure_infer_d2h_gcm_enabled() &&
        secure_infer_maybe_tamper_gcm_d2h(task.dst, task.bytes);
    int updated_len = 0;
    int final_len = 0;
    bool ok = false;
    if (secure_infer_d2h_gcm_enabled()) {
        ok = task.auth_tag != nullptr &&
            real_EVP_DecryptInit_ex(host_ctx, EVP_aes_128_gcm(), nullptr,
                                    nullptr, nullptr) == 1 &&
            real_EVP_CIPHER_CTX_ctrl(host_ctx, EVP_CTRL_GCM_SET_IVLEN,
                                     12, nullptr) == 1 &&
            real_EVP_DecryptInit_ex(host_ctx, nullptr, nullptr,
                                    task.key, task.iv) == 1;
        if (ok && task.aad_valid) {
            int aad_len = 0;
            ok = real_EVP_DecryptUpdate(
                host_ctx, nullptr, &aad_len, task.aad,
                static_cast<int>(gcm_aad_length)) == 1;
        }
        ok = ok && real_EVP_DecryptUpdate(
            host_ctx, static_cast<unsigned char*>(task.dst), &updated_len,
            static_cast<const unsigned char*>(task.dst),
            static_cast<int>(task.bytes)) == 1 &&
            real_EVP_CIPHER_CTX_ctrl(host_ctx, EVP_CTRL_GCM_SET_TAG, 16,
                                     const_cast<unsigned char*>(task.auth_tag)) == 1 &&
            real_EVP_DecryptFinal_ex(
                host_ctx, static_cast<unsigned char*>(task.dst) + updated_len,
                &final_len) == 1;
    } else {
        ok = real_EVP_EncryptInit_ex(host_ctx, EVP_aes_128_ctr(), 0,
                                     task.key, task.iv) == 1 &&
            real_EVP_EncryptUpdate(host_ctx,
                                   static_cast<unsigned char*>(task.dst),
                                   &updated_len,
                                   static_cast<const unsigned char*>(task.dst),
                                   static_cast<int>(task.bytes)) == 1 &&
            real_EVP_EncryptFinal_ex(
                host_ctx, static_cast<unsigned char*>(task.dst) + updated_len,
                &final_len) == 1;
    }
    if (!ok) {
        memset(task.dst, 0, task.bytes);
        if (tamper_injected) {
            fprintf(stderr,
                    "[SecureInfer TEST PASS] GCM D2H tamper rejected; cache entry withheld\n");
        }
        fprintf(stderr,
                "[SecureInfer FATAL] async D2H authentication/decrypt failed; "
                "cache entry withheld\n");
        std::abort();
    }

    const uint64_t callback_end_us = secure_infer_now_us();
    if (task.profile_active) {
        // The framework event is recorded after this callback.  total_us is
        // therefore the true submit-to-plaintext-ready latency.  The device
        // field includes time queued behind earlier operations on this stream;
        // it must not be summed as isolated kernel execution time.
        profile_record_precise(
            task.profile_run_id,
            "D2H",
            task.bytes,
            task.bytes,
            callback_end_us - task.submit_start_us,
            callback_end_us - callback_start_us,
            callback_start_us - task.submit_start_us,
            0,
            task.copy_enqueue_us,
            task.kernel_enqueue_us,
            0, 0, 0, 0, false,
            0,
            0);
    }
}

static EVP_CIPHER_CTX* new_d2h_host_ctx_or_abort()
{
    EVP_CIPHER_CTX* host_ctx = real_EVP_CIPHER_CTX_new();
    if (host_ctx == nullptr) {
        fprintf(stderr, "[SecureInfer FATAL] async D2H decrypt context allocation failed\n");
        std::abort();
    }
    return host_ctx;
}

static void d2h_host_decrypt_callback(void* opaque)
{
    std::unique_ptr<d2h_host_decrypt_task> task(
        static_cast<d2h_host_decrypt_task*>(opaque));
    EVP_CIPHER_CTX* host_ctx = new_d2h_host_ctx_or_abort();
    d2h_host_decrypt_one(*task, host_ctx);
    real_EVP_CIPHER_CTX_free(host_ctx);
}

static size_t d2h_cpu_worker_count()
{
    static size_t workers = 0;
    if (workers == 0) {
        const char* env = std::getenv("PIPE_D2H_CPU_WORKERS");
        long parsed = env ? std::atol(env) : 4;
        workers = static_cast<size_t>(std::max(1L, std::min(16L, parsed)));
    }
    return workers;
}

static void d2h_cpu_worker_loop()
{
    EVP_CIPHER_CTX* host_ctx = new_d2h_host_ctx_or_abort();
    while (true) {
        d2h_host_decrypt_group* group = nullptr;
        {
            std::unique_lock<std::mutex> lock(g_d2h_cpu_queue_lock);
            g_d2h_cpu_queue_cv.wait(lock, [] { return !g_d2h_cpu_queue.empty(); });
            group = g_d2h_cpu_queue.front();
            g_d2h_cpu_queue.pop_front();
        }

        for (auto& task : group->tasks) {
            d2h_host_decrypt_one(task, host_ctx);
        }

        d2h_host_decrypt_batch* batch = group->batch;
        batch->completed_groups.fetch_add(1, std::memory_order_release);
        batch->completion_cv.notify_all();
    }
}

static void ensure_d2h_cpu_workers_started()
{
    std::call_once(g_d2h_cpu_workers_once, [] {
        const size_t workers = d2h_cpu_worker_count();
        for (size_t i = 0; i < workers; ++i) {
            std::thread(d2h_cpu_worker_loop).detach();
        }
        fprintf(stderr,
                "[SecureInfer] secure D2H CPU decrypt workers: %zu\n",
                workers);
    });
}

static std::mutex g_ghash_gate_trace_lock;
static FILE* g_ghash_gate_trace_file = nullptr;
static bool g_ghash_gate_trace_initialized = false;

static void record_d2h_ghash_gate(const d2h_host_decrypt_group& group)
{
    if (group.ghash_gate_submit_us == 0 || group.ghash_gate_tasks == 0) return;
    const char* path = std::getenv("PIPE_GHASH_GATE_TRACE_PATH");
    if (path == nullptr || path[0] == '\0') return;

    const uint64_t callback_us = secure_infer_now_us();
    std::lock_guard<std::mutex> lock(g_ghash_gate_trace_lock);
    if (!g_ghash_gate_trace_initialized) {
        g_ghash_gate_trace_file = fopen(path, "w");
        g_ghash_gate_trace_initialized = true;
        if (g_ghash_gate_trace_file != nullptr) {
            fprintf(g_ghash_gate_trace_file,
                    "callback_us,ghash_tasks,payload_bytes,ghash_gate_us\n");
        }
    }
    if (g_ghash_gate_trace_file == nullptr) return;
    fprintf(g_ghash_gate_trace_file, "%llu,%zu,%zu,%llu\n",
            static_cast<unsigned long long>(callback_us),
            group.ghash_gate_tasks, group.staging_bytes,
            static_cast<unsigned long long>(
                callback_us - group.ghash_gate_submit_us));
    // The trace is a diagnostic experiment artifact. Flush it so an aborted
    // server still retains the completed authentication gates.
    fflush(g_ghash_gate_trace_file);
}

// This callback deliberately does no cryptography.  It only moves a completed
// tensor batch to an independent CPU queue and immediately releases the ACL
// callback thread, so the copy stream can proceed to the next batch.
static void d2h_group_ready_callback(void* opaque)
{
    auto* group = static_cast<d2h_host_decrypt_group*>(opaque);
    record_d2h_ghash_gate(*group);
    {
        std::lock_guard<std::mutex> lock(g_d2h_cpu_queue_lock);
        g_d2h_cpu_queue.push_back(group);
    }
    g_d2h_cpu_queue_cv.notify_one();
}

// This callback is ordered after the copy-stream join event.  Consequently
// all lightweight group callbacks have already run before it waits for CPU
// plaintext restoration, avoiding callback-thread dependency cycles.
static void d2h_host_decrypt_batch_callback(void* opaque)
{
    std::unique_ptr<d2h_host_decrypt_batch> batch(
        static_cast<d2h_host_decrypt_batch*>(opaque));
    {
        std::unique_lock<std::mutex> lock(batch->completion_lock);
        batch->completion_cv.wait(lock, [&batch] {
            return batch->completed_groups.load(std::memory_order_acquire) ==
                   batch->groups.size();
        });
    }
    for (aclrtEvent event : batch->join_events) {
        real_aclrtDestroyEvent(event);
    }
}

static bool d2h_job_batch_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_D2H_JOB_BATCH");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] logical-job D2H callback batching: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

HOOK_C_API HOOK_DECL_EXPORT int secure_infer_d2h_batch_begin(size_t expected_tasks)
{
    if (!d2h_job_batch_enabled() || !d2h_async_enabled()) return 0;
    if (g_active_d2h_batch) {
        fprintf(stderr, "[SecureInfer WARN] nested D2H batch begin; using compatibility path\n");
        return -1;
    }
    auto batch = std::make_unique<d2h_host_decrypt_batch>();
    // A Qwen-style KV cache normally contributes two groups (K/V) per layer.
    // expected_tasks is physical slices, so reserve a conservative group count
    // without allocating one object per slice here.
    if (expected_tasks > 0) batch->groups.reserve(std::min<size_t>(expected_tasks, 256));
    ensure_d2h_cpu_workers_started();
    g_active_d2h_batch = std::move(batch);
    return 1;
}

HOOK_C_API HOOK_DECL_EXPORT int secure_infer_d2h_group_begin(size_t expected_tasks)
{
    if (!g_active_d2h_batch || g_active_d2h_group) return -1;
    auto group = std::make_unique<d2h_host_decrypt_group>();
    group->batch = g_active_d2h_batch.get();
    group->expected_tasks = expected_tasks;
    if (expected_tasks > 0) group->tasks.reserve(expected_tasks);
    if (secure_infer_gcm_enabled() && expected_tasks > 0) {
        if (expected_tasks > SIZE_MAX / 16 ||
            real_aclrtMallocHost(
                reinterpret_cast<void**>(&group->host_tags),
                expected_tasks * 16) != ACL_SUCCESS ||
            group->host_tags == nullptr ||
            real_aclrtMallocHost(
                reinterpret_cast<void**>(&group->host_tag_masks),
                expected_tasks * 16) != ACL_SUCCESS ||
            group->host_tag_masks == nullptr) {
            return -1;
        }
        memset(group->host_tags, 0, expected_tasks * 16);
        memset(group->host_tag_masks, 0, expected_tasks * 16);
    }
    g_active_d2h_group = group.get();
    g_active_d2h_batch->groups.emplace_back(std::move(group));
    return 1;
}

HOOK_C_API HOOK_DECL_EXPORT int secure_infer_d2h_group_end()
{
    d2h_host_decrypt_group* group = g_active_d2h_group;
    if (group == nullptr) return 0;
    g_active_d2h_group = nullptr;
    if (group->tasks.empty()) {
        group->batch->completed_groups.fetch_add(1, std::memory_order_release);
        group->batch->completion_cv.notify_all();
        return 0;
    }
    if (group->ready_stream == nullptr) return -1;

    if (group->pipeline != nullptr) {
        auto* pipeline = group->pipeline;
        auto* slot = group->slot;
        aclError ret = real_aclrtRecordEvent(
            slot->crypto_done, pipeline->user_stream);
        if (ret != ACL_SUCCESS) return static_cast<int>(ret);
        ret = real_aclrtStreamWaitEvent(pipeline->copy_stream, slot->crypto_done);
        if (ret != ACL_SUCCESS) return static_cast<int>(ret);

        // Launch deferred GHASH ops on the independent ghash_stream so
        // they overlap with the D2H ciphertext copy below.  Inline-GHASH
        // (compatibility) is already queued on the framework stream and
        // ordered by crypto_done.  ghash_stream must wait for crypto_done
        // before reading ciphertext from the staging slot.
        bool ghash_deferred = false;
        if (pipeline->ghash_stream != nullptr &&
            secure_infer_d2h_gcm_enabled() &&
            !group->ghash_tag_out_stack.empty()) {
            ret = real_aclrtStreamWaitEvent(
                pipeline->ghash_stream, slot->crypto_done);
            if (ret != ACL_SUCCESS) return static_cast<int>(ret);
            for (size_t i = 0; i < group->tasks.size(); ++i) {
                if (i >= group->ghash_tag_out_stack.size()) break;
                void* tag_out = group->ghash_tag_out_stack[i];
                if (tag_out == nullptr) continue;
                void* aad = i < group->ghash_aad_stack.size()
                    ? group->ghash_aad_stack[i] : nullptr;
                int ghash_ret = g_launch_aicpu_ghash(
                    pipeline->ghash_stream,
                    group->tasks[i].device_ciphertext,
                    group->dev_ghash_h,
                    tag_out,
                    aad,
                    group->ghash_tag_mask_stack[i],
                    nullptr, nullptr, nullptr, nullptr,
                    group->tasks[i].bytes,
                    secure_infer_gcm_aad_enabled() ? gcm_aad_length : 0,
                    PIPE_GHASH_FINALIZE_TAG);
                if (ghash_ret != 0) {
                    fprintf(stderr,
                            "[SecureInfer FATAL] deferred GHASH launch "
                            "rejected ret=%d\n", ghash_ret);
                    std::abort();
                }
            }
            ret = real_aclrtRecordEvent(
                slot->ghash_done, pipeline->ghash_stream);
            if (ret != ACL_SUCCESS) return static_cast<int>(ret);
            group->ghash_gate_submit_us = secure_infer_now_us();
            group->ghash_gate_tasks = group->tasks.size();
            ghash_deferred = true;
        }

        // Ciphertext is packed contiguously in the slot.  Coalesce adjacent
        // CPU destinations as well, reducing a typical 63-block tensor group
        // from 63 PCIe submissions to one without changing per-slice IVs.
        // These D2H memcpys overlap with deferred GHASH above.
        for (size_t i = 0; i < group->tasks.size();) {
            auto& first = group->tasks[i];
            size_t merged_bytes = first.bytes;
            size_t next = i + 1;
            while (next < group->tasks.size()) {
                const auto& prev = group->tasks[next - 1];
                const auto& current = group->tasks[next];
                const auto prev_device_end =
                    reinterpret_cast<uintptr_t>(prev.device_ciphertext) + prev.bytes;
                const auto prev_host_end =
                    reinterpret_cast<uintptr_t>(prev.dst) + prev.bytes;
                if (reinterpret_cast<uintptr_t>(current.device_ciphertext) !=
                        prev_device_end ||
                    reinterpret_cast<uintptr_t>(current.dst) != prev_host_end) {
                    break;
                }
                merged_bytes += current.bytes;
                ++next;
            }
            const uint64_t copy_start_us = secure_infer_now_us();
            ret = real_aclrtMemcpyAsync(
                first.dst, merged_bytes,
                first.device_ciphertext, merged_bytes,
                ACL_MEMCPY_DEVICE_TO_HOST, pipeline->copy_stream);
            first.copy_enqueue_us += secure_infer_now_us() - copy_start_us;
            if (ret != ACL_SUCCESS) return static_cast<int>(ret);
            group->batch->pcie_submissions++;
            i = next;
        }
        // GCM tag copy must wait for GHASH (deferred or inline).
        if (secure_infer_d2h_gcm_enabled()) {
            if (ghash_deferred) {
                ret = real_aclrtStreamWaitEvent(
                    pipeline->copy_stream, slot->ghash_done);
                if (ret != ACL_SUCCESS) return static_cast<int>(ret);
            }
            const size_t tag_bytes = group->tasks.size() * 16;
            const uint64_t tag_copy_start_us = secure_infer_now_us();
            ret = real_aclrtMemcpyAsync(
                group->host_tags, tag_bytes,
                static_cast<uint8_t*>(slot->temp_out) +
                    group->payload_caruntimety + group->expected_tasks * 16,
                tag_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                pipeline->copy_stream);
            if (ret != ACL_SUCCESS) return static_cast<int>(ret);
            if (!group->tasks.empty()) {
                group->tasks.front().copy_enqueue_us +=
                    secure_infer_now_us() - tag_copy_start_us;
            }
            group->batch->pcie_submissions++;
        }
        group->batch->physical_slices += group->tasks.size();
        ret = real_aclrtRecordEvent(slot->copy_done, pipeline->copy_stream);
        if (ret != ACL_SUCCESS) return static_cast<int>(ret);
        slot->used = true;
    } else {
        group->batch->physical_slices += group->tasks.size();
        group->batch->pcie_submissions += group->tasks.size();
    }

    aclError callback_ret = real_aclrtLaunchHostFunc(
        group->ready_stream, d2h_group_ready_callback, group);
    if (callback_ret == ACL_SUCCESS) return 0;

    // Host functions are required for a non-blocking data plane.  This
    // compatibility fallback is outside the normal production configuration
    // and preserves correctness if the runtime rejects the callback.
    fprintf(stderr,
            "[SecureInfer WARN] tensor-batch D2H callback failed (%d); "
            "synchronizing compatibility path\n",
            static_cast<int>(callback_ret));
    real_aclrtSynchronizeStream(group->ready_stream);
    d2h_group_ready_callback(group);
    return 0;
}

// Join the dedicated PCIe stream back into the framework D2H stream.  Python
// records DEVICE_COPY_DONE immediately after this function; CPU decryption is
// intentionally not part of this barrier.
HOOK_C_API HOOK_DECL_EXPORT int secure_infer_d2h_batch_device_barrier()
{
    if (!g_active_d2h_batch || g_active_d2h_group) return -1;
    auto& batch = *g_active_d2h_batch;
    for (size_t i = 0; i < batch.streams.size(); ++i) {
        if (batch.streams[i] == batch.copy_streams[i]) {
            // Compatibility path is already ordered on one stream.
            continue;
        }
        aclrtEvent join = nullptr;
        aclError ret = real_aclrtCreateEvent(&join);
        if (ret != ACL_SUCCESS || join == nullptr) return static_cast<int>(ret);
        ret = real_aclrtRecordEvent(join, batch.copy_streams[i]);
        if (ret == ACL_SUCCESS) {
            ret = real_aclrtStreamWaitEvent(batch.streams[i], join);
        }
        if (ret != ACL_SUCCESS) {
            real_aclrtDestroyEvent(join);
            return static_cast<int>(ret);
        }
        batch.join_events.push_back(join);
    }
    return 0;
}

HOOK_C_API HOOK_DECL_EXPORT int secure_infer_d2h_batch_end()
{
    if (!g_active_d2h_batch) return 0;
    if (g_active_d2h_group) return -1;
    std::unique_ptr<d2h_host_decrypt_batch> batch =
        std::move(g_active_d2h_batch);
    if (batch->groups.empty()) return 0;

    uint64_t profiled_rows = 0;
    uint64_t profiled_bytes = 0;
    for (const auto& group : batch->groups) {
        for (const auto& task : group->tasks) {
            if (task.profile_active && task.bytes > 0 &&
                profiled_bytes <= UINT64_MAX - task.bytes) {
                profiled_rows++;
                profiled_bytes += task.bytes;
            }
        }
    }
    if (profiled_rows > 0) {
        profile_count_d2h_kv_totals(
            profiled_rows, profiled_bytes, batch->profile_run_id);
    }
    fprintf(stderr,
            "[SecureInfer] secure D2H job pipeline: groups=%zu slices=%zu "
            "pcie_submissions=%zu cpu_workers=%zu\n",
            batch->groups.size(), batch->physical_slices,
            batch->pcie_submissions, d2h_cpu_worker_count());

    // The final callback is queued on the framework stream after its wait on
    // the dedicated copy stream.  It therefore cannot run before all tensor
    // groups have reached the CPU worker queue.
    if (batch->streams.size() == 1) {
        d2h_host_decrypt_batch* raw = batch.release();
        aclError ret = real_aclrtLaunchHostFunc(
            raw->streams.front(), d2h_host_decrypt_batch_callback, raw);
        if (ret == ACL_SUCCESS) return 0;
        batch.reset(raw);
        fprintf(stderr,
                "[SecureInfer WARN] batched D2H HostFunc failed (%d), synchronizing fallback\n",
                static_cast<int>(ret));
    }

    for (aclrtStream stream : batch->streams) real_aclrtSynchronizeStream(stream);
    d2h_host_decrypt_batch_callback(batch.release());
    return 0;
}

static void* alloc_inline_cipher_buffer(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (g_inline_cipher_host != nullptr && g_inline_cipher_host_size >= bytes) {
        return g_inline_cipher_host;
    }

    if (g_inline_cipher_host != nullptr) {
        if (g_inline_cipher_host_pinned) {
            real_aclrtFreeHost(g_inline_cipher_host);
        } else {
            std::free(g_inline_cipher_host);
        }
        g_inline_cipher_host = nullptr;
        g_inline_cipher_host_size = 0;
        g_inline_cipher_host_pinned = false;
    }

    void* ptr = nullptr;
    bool pinned = false;
    const char* env = std::getenv("PIPE_USE_PINNED_CIPHER");
    bool use_pinned = !env || std::atoi(env) != 0;
    if (use_pinned) {
        aclError ret = real_aclrtMallocHost(&ptr, bytes);
        pinned = (ret == ACL_SUCCESS && ptr != nullptr);
    }
    if (!pinned) {
        long page_size = sysconf(_SC_PAGESIZE);
        size_t alignment = page_size > 0 ? static_cast<size_t>(page_size) : 4096;
        int ret = posix_memalign(&ptr, alignment, bytes);
        if (ret != 0 || ptr == nullptr) {
            ptr = std::malloc(bytes);
        }
    }
    assert(ptr != nullptr);
    g_inline_cipher_host = ptr;
    g_inline_cipher_host_size = bytes;
    g_inline_cipher_host_pinned = pinned;
    return g_inline_cipher_host;
}

static void free_inline_cipher_buffer(void* ptr) {
    (void)ptr;
}

static uint32_t calc_aes_block_dim(size_t bytes) {
    constexpr uint32_t core_num = 20;
    uint64_t avg = bytes / core_num;
    if (avg < 4096) return 1;
    return core_num;
}

static bool ensure_h2d_device_crypto(encrypt_metadata& meta, aclrtStream stream)
{
    if (!g_launch_inline_iv_kernel) load_aes_library();
    if (meta.dev_key == nullptr || meta.dev_iv == nullptr) {
        if (real_aclrtMalloc(&meta.dev_key, 32) != ACL_SUCCESS) return false;
        // The AscendC kernel performs one aligned 32-byte GM load for IV.
        if (real_aclrtMalloc(&meta.dev_iv, 32) != ACL_SUCCESS) return false;
        if (real_aclrtMemcpyAsync(meta.dev_key, 32, meta.key, 32,
                                  ACL_MEMCPY_HOST_TO_DEVICE, stream) !=
                ACL_SUCCESS ||
            real_aclrtMemcpyAsync(meta.dev_iv, 16, meta.init_iv, 16,
                                  ACL_MEMCPY_HOST_TO_DEVICE, stream) !=
                ACL_SUCCESS) {
            return false;
        }
        fprintf(stderr, "[SecureInfer] ✅ H2D serial fallback 初始化 NPU Key/IV\n");
    }
    g_dev_key = meta.dev_key;
    g_dev_iv = meta.dev_iv;
    if (g_launch_inline_iv_kernel == nullptr || g_dev_key == nullptr ||
        stream == nullptr) {
        return false;
    }

    // The copies and this record are ordered on the initialization stream.
    // Publish the handle only after aclrtRecordEvent succeeds so a producer
    // that observes the pointer can safely install its cross-stream wait.
    if (g_h2d_device_crypto_ready_event.load(std::memory_order_acquire) ==
            nullptr) {
        aclrtEvent ready_event = nullptr;
        if (real_aclrtCreateEvent(&ready_event) != ACL_SUCCESS ||
            ready_event == nullptr) {
            fprintf(stderr,
                    "[SecureInfer ERROR] cannot create device-crypto-ready event\n");
            return false;
        }
        if (real_aclrtRecordEvent(ready_event, stream) != ACL_SUCCESS) {
            fprintf(stderr,
                    "[SecureInfer ERROR] cannot record device-crypto-ready event "
                    "on stream=%p\n",
                    stream);
            real_aclrtDestroyEvent(ready_event);
            return false;
        }
        g_h2d_device_crypto_ready_event.store(ready_event,
                                               std::memory_order_release);
        fprintf(stderr,
                "[SecureInfer] device-crypto-ready event recorded stream=%p event=%p\n",
                stream, static_cast<void*>(ready_event));
    }
    return true;
}

bool secure_infer_prepare_h2d_device_crypto(aclrtStream stream)
{
    if (g_h2d_device_crypto_ready.load(std::memory_order_acquire) &&
        g_dev_key != nullptr) {
        return true;
    }
    if (stream == nullptr || m_ctx_metadata.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> inline_lock(g_inline_h2d_lock);
    auto& meta = *m_ctx_metadata.begin()->second;
    if (secure_infer_gcm_enabled() && !secure_infer_prepare_gcm_backend(meta, stream)) {
        return false;
    }
    const bool ready = ensure_h2d_device_crypto(meta, stream);
    if (ready) {
        g_h2d_device_crypto_ready.store(true, std::memory_order_release);
    }
    return ready;
}

bool secure_infer_enqueue_ticketed_keystream(aclrtStream stream,
                                        void* output,
                                        size_t bytes,
                                        uint64_t transfer_ticket,
                                        uint64_t block_offset)
{
    if (stream == nullptr || output == nullptr || bytes == 0 ||
        transfer_ticket == 0 || m_ctx_metadata.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> inline_lock(g_inline_h2d_lock);
    auto& meta = *m_ctx_metadata.begin()->second;
    if (!ensure_h2d_device_crypto(meta, stream) ||
        !load_ghash_library() || g_launch_aicpu_keystream == nullptr) {
        return false;
    }
    if (!wait_h2d_device_crypto_ready(stream)) return false;

    unsigned char iv[iv_length];
    secure_infer_derive_transfer_iv(meta.init_iv, transfer_ticket,
                               block_offset, iv);
    uint32_t iv_words[4] = {0, 0, 0, 0};
    pack_inline_iv(iv, iv_words);

    const size_t aligned_bytes = (bytes + 31) & ~static_cast<size_t>(31);
    if (aligned_bytes == 0 || aligned_bytes > UINT32_MAX) return false;
    return g_launch_aicpu_keystream(
        stream, g_dev_key, output, aligned_bytes,
        iv_words[0], iv_words[1], iv_words[2], iv_words[3]) == 0;
}

bool secure_infer_enqueue_precomputed_xor(aclrtStream stream,
                                     void* input,
                                     void* keystream,
                                     void* output,
                                     size_t bytes)
{
    if (stream == nullptr || input == nullptr || keystream == nullptr ||
        output == nullptr || bytes == 0) {
        return false;
    }

    if (!g_launch_inline_iv_kernel) load_aes_library();
    const size_t aligned_bytes = (bytes + 31) & ~static_cast<size_t>(31);
    if (g_launch_inline_iv_kernel == nullptr || aligned_bytes == 0 ||
        aligned_bytes > UINT32_MAX) {
        return false;
    }

    // Mode 2 interprets the second pointer as a precomputed keystream.  IV
    // scalars are intentionally zero because this path performs no AES.
    g_launch_inline_iv_kernel(
        calc_aes_block_dim(aligned_bytes),
        stream,
        static_cast<uint8_t*>(input),
        static_cast<uint8_t*>(keystream),
        static_cast<uint8_t*>(output),
        static_cast<uint32_t>(aligned_bytes),
        2, 0, 0, 0, 0);
    return true;
}

// --------------------------------------------------------------------------
// GCM ticketed keystream pool
// --------------------------------------------------------------------------
// The existing CTR pool is private to commit_worker because its ciphertext
// preparation and consume path live there.  GCM is different: authenticated
// H2D uses inline_h2d_encrypt_commit, so the worker may only produce a
// ticketed keystream while the inline path must own consumption after GHASH
// succeeds.  Keep the bridge process-local and opt-in; no unauthenticated
// plaintext can leave this pool because the inline path still executes the
// existing GHASH -> auth gate before the XOR result is copied to its target.

enum class gcm_keystream_backend : uint8_t { aicpu, aicore, drop };

struct gcm_keystream_record {
    void* allocation = nullptr;
    aclrtEvent ready_event = nullptr;
    size_t bytes = 0;
    uint64_t generation = 0;
    gcm_keystream_backend backend = gcm_keystream_backend::aicpu;
    // AIV mode-1 performs an async H2D copy of this IV on the producer
    // stream.  The map node remains alive until its ready event completes,
    // which gives that copy a stable host lifetime without per-ticket pinned
    // allocations.
    std::array<unsigned char, 32> aiv_iv{};
    bool launching = false;
    bool consumed = false;
    bool in_use = false;
};

static std::mutex g_gcm_keystream_lock;
static std::unordered_map<uint64_t, gcm_keystream_record>
    g_gcm_keystream_records;
static std::atomic<size_t> g_gcm_keystream_unconsumed{0};
static std::atomic<uint64_t> g_gcm_keystream_critical{0};
// Highest ticket that has reached the authenticated H2D consume path.  Ticket
// IDs are allocated monotonically, so this exposes a conservative distance
// from the currently active restore window to the background scheduler.
static std::atomic<uint64_t> g_gcm_h2d_ticket_high_water{0};
// Scope of the logical H2D job currently consuming authenticated slices.
// This is supplied by the transfer-layer descriptor, not inferred from a
// ticket number, so a queued request that shares CPU cache blocks cannot be
// mistaken for the foreground request.
static std::atomic<uint64_t> g_gcm_active_h2d_scope{0};
static std::mutex g_gcm_keystream_control_lock;
static uint64_t g_gcm_last_demand_us = 0;
static double g_gcm_demand_bytes_per_s = 0.0;
static gcm_keystream_backend g_gcm_last_backend =
    gcm_keystream_backend::drop;
// aclrtGetDeviceUtilizationRate costs about 100ms on this platform.  Query it
// only on the non-critical commit worker at a coarse interval and make every
// placement decision use the cached result; never call it from inline H2D.
static std::mutex g_gcm_aicpu_util_lock;
static uint64_t g_gcm_aicpu_util_sample_us = 0;
static std::atomic<int> g_gcm_aicpu_utilization{-1};

static std::atomic<uint64_t> g_gcm_keystream_ready{0};
static std::atomic<uint64_t> g_gcm_keystream_busy{0};
static std::atomic<uint64_t> g_gcm_keystream_missing{0};
static std::atomic<uint64_t> g_gcm_keystream_aicpu_submitted{0};
static std::atomic<uint64_t> g_gcm_keystream_aicore_submitted{0};
// Placement counters are deliberately captured at submission time. The
// critical counter may be zero by the time a later H2D consumes its ticket,
// so consume-time sampling cannot prove that the scheduler actually migrated
// a real background ticket away from AICPU.
static std::atomic<uint64_t> g_gcm_keystream_critical_aicpu_submitted{0};
static std::atomic<uint64_t> g_gcm_keystream_critical_aicore_submitted{0};
static std::atomic<uint64_t> g_gcm_keystream_dropped{0};
static std::atomic<uint64_t> g_gcm_keystream_switches{0};
static std::atomic<uint64_t> g_gcm_keystream_finished{0};
static std::atomic<uint64_t> g_gcm_keystream_device_waits{0};

// AIV mode-1 has no inter-ticket data dependency, but a single producer
// stream serializes every IV copy and kernel launch.  Keep a small pool of
// producer streams with distinct 32-byte IV buffers.  The ticket event is
// recorded on its selected stream, so the existing no-wait consume rule and
// IV lifetime invariants remain unchanged.
struct gcm_aiv_producer {
    aclrtStream stream = nullptr;
    void* device_iv = nullptr;
    bool owns_stream = false;
    bool owns_iv = false;
};
static std::mutex g_gcm_aiv_producer_lock;
static std::vector<gcm_aiv_producer> g_gcm_aiv_producers;
static std::atomic<uint64_t> g_gcm_aiv_next_producer{0};

// Keep a bounded cache of device buffers.  A ticket may be consumed before
// its producer finishes (the intentional no-wait fallback); without this
// cache, every such ticket also pays an aclrtMalloc/Free pair and allocation
// latency becomes a confounder in the scheduling experiment.
static std::mutex g_gcm_keystream_buffer_pool_lock;
static std::unordered_map<size_t, std::vector<void*>>
    g_gcm_keystream_buffer_pool;
static size_t g_gcm_keystream_cached_bytes = 0;

// Creating one ACL event for every exact GCM ticket is surprisingly costly
// on the host.  A 4x8K reload exposes 20,160 independent 256-KiB tickets; if
// events are created only when the descriptor arrives, the single ordered
// commit worker spends more time in aclrtCreateEvent than feeding AICPU.  Keep
// a cold, unrecorded event pool beside the device-buffer pool.  Each event is
// still owned by exactly one ticket after acquisition and is destroyed by the
// established record cleanup path, so completion and verify-before-use
// semantics are unchanged.
static std::mutex g_gcm_keystream_event_pool_lock;
static std::vector<aclrtEvent> g_gcm_keystream_event_pool;

// --------------------------------------------------------------------------
// GCM CPU ciphertext lookahead
// --------------------------------------------------------------------------
// The NPU ticket pool removes payload AES from the demand stream, but the
// original GCM path still invoked EVP_Encrypt* synchronously for every 256KiB
// H2D slice.  Descriptors already expose the exact source address and ticket
// before that demand.  This bounded host pool prepares ciphertext+tag in the
// background and is intentionally opportunistic: a busy/missing entry falls
// straight back to the existing inline OpenSSL path.
enum class gcm_cpu_cipher_state : uint8_t { queued, running, ready, failed };

struct gcm_cpu_cipher_record {
    const void* src = nullptr;
    size_t bytes = 0;
    uint64_t ticket = 0;
    uint64_t generation = 0;
    unsigned char key[key_length] = {};
    unsigned char iv[iv_length] = {};
    unsigned char aad[gcm_aad_length] = {};
    bool aad_valid = false;
    unsigned char* ciphertext = nullptr;
    unsigned char tag[16] = {};
    // A direct asynchronous H2D DMA must read pinned memory until the slot's
    // completion event.  The old path allocated pageable memory here and
    // copied 1.23 GiB back into a per-slot pinned staging buffer on demand.
    bool ciphertext_pinned = false;
    bool handed_off = false;
    std::atomic<gcm_cpu_cipher_state> state{gcm_cpu_cipher_state::queued};
    std::atomic<bool> cancelled{false};
    uint64_t crypto_us = 0;

    ~gcm_cpu_cipher_record()
    {
        if (ciphertext == nullptr) return;
        if (ciphertext_pinned) {
            real_aclrtFreeHost(ciphertext);
        } else {
            std::free(ciphertext);
        }
    }
};

static std::mutex g_gcm_cpu_cipher_lock;
static std::condition_variable g_gcm_cpu_cipher_cv;
static std::unordered_map<uint64_t, std::shared_ptr<gcm_cpu_cipher_record>>
    g_gcm_cpu_cipher_records;
static std::deque<std::shared_ptr<gcm_cpu_cipher_record>>
    g_gcm_cpu_cipher_queue;
static size_t g_gcm_cpu_cipher_reserved_bytes = 0;
static std::once_flag g_gcm_cpu_cipher_workers_once;
static std::atomic<uint64_t> g_gcm_cpu_cipher_zero_copy_handoffs{0};

static unsigned char* alloc_gcm_cpu_cipher_buffer(size_t bytes, bool* pinned)
{
    if (pinned != nullptr) *pinned = false;
    void* allocation = nullptr;
    if (real_aclrtMallocHost(&allocation, bytes) == ACL_SUCCESS &&
        allocation != nullptr) {
        if (pinned != nullptr) *pinned = true;
        return static_cast<unsigned char*>(allocation);
    }
    return static_cast<unsigned char*>(std::malloc(bytes));
}

bool secure_infer_gcm_cpu_cipher_prefetch_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_CPU_PREENCRYPT");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] GCM CPU ciphertext pre-encryption: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static size_t gcm_cpu_cipher_record_limit()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env = std::getenv("PIPE_GCM_CPU_CIPHER_MAX_TICKETS");
        long parsed = env != nullptr ? std::atol(env) : 1024;
        if (parsed < 1) parsed = 1;
        // Keep the small default, but allow a bounded real multi-request
        // experiment to prepare every independently authenticated slice.
        if (parsed > 65536) parsed = 65536;
        limit = static_cast<size_t>(parsed);
    }
    return limit;
}

static size_t gcm_cpu_cipher_byte_limit()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env = std::getenv("PIPE_GCM_CPU_CIPHER_POOL_BYTES");
        unsigned long long parsed = env != nullptr
            ? std::strtoull(env, nullptr, 10)
            : 256ULL * 1024ULL * 1024ULL;
        if (parsed < 262144ULL) parsed = 262144ULL;
        if (parsed > 16ULL * 1024ULL * 1024ULL * 1024ULL) {
            parsed = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        }
        limit = static_cast<size_t>(parsed);
    }
    return limit;
}

static int gcm_cpu_cipher_worker_count()
{
    static int workers = 0;
    if (workers == 0) {
        const char* env = std::getenv("PIPE_GCM_CPU_PREENCRYPT_WORKERS");
        int parsed = env != nullptr ? std::atoi(env) : 4;
        workers = std::max(1, std::min(8, parsed));
    }
    return workers;
}

static void gcm_cpu_cipher_worker_loop()
{
    EVP_CIPHER_CTX* ctx = real_EVP_CIPHER_CTX_new();
    if (ctx == nullptr) return;
    while (g_running.load(std::memory_order_acquire)) {
        std::shared_ptr<gcm_cpu_cipher_record> record;
        {
            std::unique_lock<std::mutex> lock(g_gcm_cpu_cipher_lock);
            g_gcm_cpu_cipher_cv.wait_for(lock, std::chrono::milliseconds(1), [] {
                return !g_gcm_cpu_cipher_queue.empty() ||
                    !g_running.load(std::memory_order_acquire);
            });
            if (!g_running.load(std::memory_order_acquire)) break;
            if (g_gcm_cpu_cipher_queue.empty()) continue;
            record = std::move(g_gcm_cpu_cipher_queue.front());
            g_gcm_cpu_cipher_queue.pop_front();
            if (record->cancelled.load(std::memory_order_acquire) ||
                record->generation <=
                    g_retired_transfer_generation.load(std::memory_order_acquire)) {
                record->state.store(gcm_cpu_cipher_state::failed,
                                    std::memory_order_release);
                continue;
            }
            record->state.store(gcm_cpu_cipher_state::running,
                                std::memory_order_release);
        }

        const size_t aligned_bytes =
            (record->bytes + 31) & ~static_cast<size_t>(31);
        // Keep ciphertext, the 16-byte GCM tag, the trusted tag-mask gap, and
        // AAD in one pinned allocation.  A ready record can then reach the
        // device with one DMA submission instead of a second tiny AAD copy.
        bool ciphertext_pinned = false;
        const size_t prepared_bytes = aligned_bytes + 32 +
            (secure_infer_gcm_aad_enabled() && gcm_h2d_coalesced_aad_enabled()
                 ? gcm_aad_length : 0);
        unsigned char* ciphertext = alloc_gcm_cpu_cipher_buffer(
            prepared_bytes, &ciphertext_pinned);
        int updated_len = 0;
        int final_len = 0;
        bool ok = ciphertext != nullptr &&
            record->bytes <= static_cast<size_t>(INT_MAX) &&
            !record->cancelled.load(std::memory_order_acquire);
        const uint64_t begin_us = secure_infer_now_us();
        if (ok) {
            ok = real_EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr,
                                         nullptr, nullptr) == 1 &&
                real_EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12,
                                         nullptr) == 1 &&
                real_EVP_EncryptInit_ex(ctx, nullptr, nullptr, record->key,
                                        record->iv) == 1;
            if (ok && record->aad_valid) {
                int aad_len = 0;
                ok = real_EVP_EncryptUpdate(
                    ctx, nullptr, &aad_len, record->aad,
                    static_cast<int>(gcm_aad_length)) == 1;
            }
            ok = ok && real_EVP_EncryptUpdate(
                ctx, ciphertext, &updated_len,
                static_cast<const unsigned char*>(record->src),
                static_cast<int>(record->bytes)) == 1 &&
                real_EVP_EncryptFinal_ex(ctx, ciphertext + updated_len,
                                         &final_len) == 1 &&
                real_EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16,
                                         record->tag) == 1 &&
                !record->cancelled.load(std::memory_order_acquire);
        }
        record->crypto_us = secure_infer_now_us() - begin_us;
        if (ok) {
            if (aligned_bytes > record->bytes) {
                memset(ciphertext + record->bytes, 0,
                       aligned_bytes - record->bytes);
            }
            memcpy(ciphertext + aligned_bytes, record->tag, 16);
            memset(ciphertext + aligned_bytes + 16, 0, 16);
            if (secure_infer_gcm_aad_enabled() &&
                gcm_h2d_coalesced_aad_enabled()) {
                memcpy(ciphertext + aligned_bytes + 32, record->aad,
                       gcm_aad_length);
            }
            record->ciphertext = ciphertext;
            record->ciphertext_pinned = ciphertext_pinned;
            record->state.store(gcm_cpu_cipher_state::ready,
                                std::memory_order_release);
        } else {
            if (ciphertext_pinned) {
                real_aclrtFreeHost(ciphertext);
            } else {
                std::free(ciphertext);
            }
            record->state.store(gcm_cpu_cipher_state::failed,
                                std::memory_order_release);
        }
    }
    real_EVP_CIPHER_CTX_free(ctx);
}

static void ensure_gcm_cpu_cipher_workers()
{
    std::call_once(g_gcm_cpu_cipher_workers_once, [] {
        const int workers = gcm_cpu_cipher_worker_count();
        for (int index = 0; index < workers; ++index) {
            auto* worker = new std::thread(gcm_cpu_cipher_worker_loop);
            worker->detach();
        }
        fprintf(stderr,
                "[SecureInfer] GCM CPU ciphertext pool: workers=%d, "
                "records=%zu, bytes=%.1f MiB\n",
                workers, gcm_cpu_cipher_record_limit(),
                static_cast<double>(gcm_cpu_cipher_byte_limit()) /
                    (1024.0 * 1024.0));
    });
}

void secure_infer_gcm_cpu_cipher_cleanup()
{
    const uint64_t retired =
        g_retired_transfer_generation.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lock(g_gcm_cpu_cipher_lock);
    for (auto it = g_gcm_cpu_cipher_records.begin();
         it != g_gcm_cpu_cipher_records.end();) {
        const auto& record = it->second;
        const auto state = record->state.load(std::memory_order_acquire);
        if (record->generation <= retired && !record->handed_off) {
            record->cancelled.store(true, std::memory_order_release);
        }
        const bool terminal = state == gcm_cpu_cipher_state::ready ||
            state == gcm_cpu_cipher_state::failed;
        if (!record->handed_off && terminal &&
            (record->cancelled.load(std::memory_order_acquire) ||
                         record->generation <= retired)) {
            g_gcm_cpu_cipher_reserved_bytes -= record->bytes;
            it = g_gcm_cpu_cipher_records.erase(it);
        } else {
            ++it;
        }
    }
}

bool secure_infer_gcm_schedule_cpu_ciphertext(const void* src, size_t bytes,
                                         uint64_t transfer_ticket,
                                         uint64_t generation)
{
    if (!secure_infer_gcm_cpu_cipher_prefetch_enabled() || !secure_infer_gcm_enabled() ||
        src == nullptr || bytes == 0 || bytes > static_cast<size_t>(INT_MAX) ||
        transfer_ticket == 0 ||
        generation <= g_retired_transfer_generation.load(std::memory_order_acquire)) {
        return false;
    }

    unsigned char key[key_length] = {};
    unsigned char iv[iv_length] = {};
    {
        std::lock_guard<std::mutex> lock(g_inline_h2d_lock);
        if (m_ctx_metadata.empty()) return false;
        const auto& metadata = *m_ctx_metadata.begin()->second;
        memcpy(key, metadata.key, sizeof(key));
        secure_infer_derive_transfer_iv(metadata.init_iv, transfer_ticket, 0, iv);
    }

    secure_infer_gcm_cpu_cipher_cleanup();
    auto record = std::make_shared<gcm_cpu_cipher_record>();
    record->src = src;
    record->bytes = bytes;
    record->ticket = transfer_ticket;
    record->generation = generation;
    memcpy(record->key, key, sizeof(key));
    memcpy(record->iv, iv, sizeof(iv));
    record->aad_valid = !secure_infer_gcm_aad_enabled() ||
        secure_infer_build_gcm_aad(src, bytes, PIPE_GCM_DIRECTION_H2D,
                              transfer_ticket, 0, record->aad);
    if (!record->aad_valid) return false;
    {
        std::lock_guard<std::mutex> lock(g_gcm_cpu_cipher_lock);
        if (g_gcm_cpu_cipher_records.find(transfer_ticket) !=
            g_gcm_cpu_cipher_records.end()) {
            return true;
        }
        if (g_gcm_cpu_cipher_records.size() >= gcm_cpu_cipher_record_limit() ||
            bytes > gcm_cpu_cipher_byte_limit() ||
            g_gcm_cpu_cipher_reserved_bytes >
                gcm_cpu_cipher_byte_limit() - bytes) {
            return false;
        }
        g_gcm_cpu_cipher_reserved_bytes += bytes;
        g_gcm_cpu_cipher_records.emplace(transfer_ticket, record);
        g_gcm_cpu_cipher_queue.push_back(record);
    }
    ensure_gcm_cpu_cipher_workers();
    g_gcm_cpu_cipher_cv.notify_one();
    return true;
}

bool secure_infer_gcm_take_ready_cpu_ciphertext(uint64_t transfer_ticket,
                                           const void* src, size_t bytes,
                                           unsigned char* destination,
                                           size_t destination_bytes,
                                           uint64_t* crypto_us)
{
    if (crypto_us != nullptr) *crypto_us = 0;
    if (!secure_infer_gcm_cpu_cipher_prefetch_enabled() || transfer_ticket == 0 ||
        src == nullptr || bytes == 0 || destination == nullptr) {
        return false;
    }
    const size_t aligned = (bytes + 31) & ~static_cast<size_t>(31);
    const size_t prepared_bytes = aligned + 32 +
        (secure_infer_gcm_aad_enabled() && gcm_h2d_coalesced_aad_enabled()
             ? gcm_aad_length : 0);
    if (aligned == 0 || destination_bytes < prepared_bytes) return false;

    std::shared_ptr<gcm_cpu_cipher_record> record;
    {
        std::lock_guard<std::mutex> lock(g_gcm_cpu_cipher_lock);
        auto it = g_gcm_cpu_cipher_records.find(transfer_ticket);
        if (it == g_gcm_cpu_cipher_records.end() ||
            it->second->src != src || it->second->bytes != bytes ||
            it->second->generation <=
                g_retired_transfer_generation.load(std::memory_order_acquire)) {
            g_transfer_prefetch_missing_fallback.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        record = it->second;
        if (record->state.load(std::memory_order_acquire) !=
                gcm_cpu_cipher_state::ready ||
            record->ciphertext == nullptr) {
            record->cancelled.store(true, std::memory_order_release);
            g_transfer_prefetch_busy_fallback.fetch_add(
                1, std::memory_order_relaxed);
            return false;
        }
        g_gcm_cpu_cipher_reserved_bytes -= record->bytes;
        g_gcm_cpu_cipher_records.erase(it);
    }

    memcpy(destination, record->ciphertext, prepared_bytes);
    if (crypto_us != nullptr) *crypto_us = record->crypto_us;
    g_transfer_prefetch_ready.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool secure_infer_gcm_handoff_ready_cpu_ciphertext(uint64_t transfer_ticket,
                                               const void* src, size_t bytes,
                                               unsigned char** ciphertext,
                                               size_t* ciphertext_bytes,
                                               uint64_t* crypto_us)
{
    if (ciphertext != nullptr) *ciphertext = nullptr;
    if (ciphertext_bytes != nullptr) *ciphertext_bytes = 0;
    if (crypto_us != nullptr) *crypto_us = 0;
    if (!secure_infer_gcm_cpu_cipher_prefetch_enabled() || transfer_ticket == 0 ||
        src == nullptr || bytes == 0 || ciphertext == nullptr) {
        return false;
    }
    const size_t aligned = (bytes + 31) & ~static_cast<size_t>(31);
    if (aligned == 0) return false;

    std::lock_guard<std::mutex> lock(g_gcm_cpu_cipher_lock);
    auto it = g_gcm_cpu_cipher_records.find(transfer_ticket);
    if (it == g_gcm_cpu_cipher_records.end() || it->second->src != src ||
        it->second->bytes != bytes ||
        it->second->generation <=
            g_retired_transfer_generation.load(std::memory_order_acquire)) {
        return false;
    }
    const auto& record = it->second;
    // Only an ACL-pinned allocation is safe to retain after the hook returns.
    // A pageable fallback remains correct through the established copy path.
    if (record->state.load(std::memory_order_acquire) !=
            gcm_cpu_cipher_state::ready ||
        record->ciphertext == nullptr || !record->ciphertext_pinned ||
        record->handed_off) {
        return false;
    }
    record->handed_off = true;
    *ciphertext = record->ciphertext;
    if (ciphertext_bytes != nullptr) {
        *ciphertext_bytes = aligned + 32 +
            (secure_infer_gcm_aad_enabled() && gcm_h2d_coalesced_aad_enabled()
                 ? gcm_aad_length : 0);
    }
    if (crypto_us != nullptr) *crypto_us = record->crypto_us;
    g_transfer_prefetch_ready.fetch_add(1, std::memory_order_relaxed);
    const uint64_t handoffs = g_gcm_cpu_cipher_zero_copy_handoffs.fetch_add(
        1, std::memory_order_relaxed) + 1;
    if (handoffs == 1 || handoffs % 1024 == 0) {
        fprintf(stderr,
                "[SecureInfer] CPU pre-encryption zero-copy H2D handoffs=%lu\n",
                static_cast<unsigned long>(handoffs));
    }
    return true;
}

void secure_infer_gcm_release_handed_off_cpu_ciphertext(uint64_t transfer_ticket)
{
    if (transfer_ticket == 0) return;
    std::lock_guard<std::mutex> lock(g_gcm_cpu_cipher_lock);
    auto it = g_gcm_cpu_cipher_records.find(transfer_ticket);
    if (it == g_gcm_cpu_cipher_records.end() || !it->second->handed_off) {
        return;
    }
    g_gcm_cpu_cipher_reserved_bytes -= it->second->bytes;
    g_gcm_cpu_cipher_records.erase(it);
}

bool secure_infer_gcm_keystream_scheduler_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_SCHEDULER");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] GCM ticketed keystream scheduler: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static gcm_keystream_backend gcm_keystream_policy()
{
    static const gcm_keystream_backend policy = [] {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_POLICY");
        if (env != nullptr && strcasecmp(env, "aicore") == 0) {
            return gcm_keystream_backend::aicore;
        }
        if (env != nullptr && strcasecmp(env, "dynamic") == 0) {
            return gcm_keystream_backend::drop; // dynamic sentinel
        }
        return gcm_keystream_backend::aicpu;
    }();
    return policy;
}

static bool gcm_keystream_policy_is_dynamic()
{
    const char* env = std::getenv("PIPE_GCM_KEYSTREAM_POLICY");
    return env != nullptr && strcasecmp(env, "dynamic") == 0;
}

bool secure_infer_gcm_dynamic_keystream_policy_enabled()
{
    return secure_infer_gcm_keystream_scheduler_enabled() &&
        gcm_keystream_policy_is_dynamic();
}

static size_t gcm_keystream_record_limit()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_MAX_TICKETS");
        // A four-layer descriptor horizon contains 256 256-KiB GQA tickets
        // for the 4K external-KV reload, leaving no room to admit the next
        // known layer before the consumer reaches it.  Keep a bounded
        // eight-layer reservoir by default; this is still far below a full
        // request epoch and preserves the producer's existing stream count.
        long parsed = env != nullptr ? std::atol(env) : 512;
        if (parsed < 1) parsed = 1;
        // An 8K external-KV reload can expose 80 tensor slices × 63 blocks
        // (5040 exact GCM IVs) before the first H2D is issued.  A real
        // multi-request burst needs one independent ticket per request and
        // slice, so permit a larger explicitly requested experiment window.
        // The ordinary default remains 256 and therefore does not increase
        // production memory use.
        if (parsed > 32768) parsed = 32768;
        limit = static_cast<size_t>(parsed);
        fprintf(stderr, "[SecureInfer] GCM keystream ticket pool: %zu tickets\n", limit);
    }
    return limit;
}

// Keep the speculative producer far enough ahead to absorb launch jitter,
// but do not make every ticket irrevocable before the first authenticated
// H2D exposes its deadline and resource pressure.  The storage limit and the
// in-flight scheduling window are separate: completed/consumed records may
// remain cached until periodic cleanup without blocking new useful work.
static size_t gcm_keystream_inflight_limit()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env =
            std::getenv("PIPE_GCM_KEYSTREAM_INFLIGHT_TICKETS");
        unsigned long long parsed = env != nullptr
            ? std::strtoull(env, nullptr, 10)
            : static_cast<unsigned long long>(gcm_keystream_record_limit());
        parsed = std::max<unsigned long long>(1, parsed);
        parsed = std::min<unsigned long long>(
            parsed, gcm_keystream_record_limit());
        limit = static_cast<size_t>(parsed);
        fprintf(stderr,
                "[SecureInfer] GCM unconsumed ticket window: %zu tickets\n",
                limit);
    }
    return limit;
}

// A demand that missed the producer has already used inline AES. Do not
// admit its delayed speculative job later and pin an unused ticket slot.
// The history lives only for the active prediction generation; retired jobs
// are terminal at admission. Existing in-use records keep their event owner.
static std::unordered_set<uint64_t> g_gcm_demanded_tickets;
static uint64_t g_gcm_demand_history_retired = 0;
static void gcm_refresh_demand_history_locked()
{
    const uint64_t retired = g_retired_transfer_generation.load(std::memory_order_acquire);
    if (retired != g_gcm_demand_history_retired) {
        g_gcm_demanded_tickets.clear();
        g_gcm_demand_history_retired = retired;
    }
}

static size_t gcm_unconsumed_ticket_count_locked()
{
    return g_gcm_keystream_unconsumed.load(std::memory_order_relaxed);
}

// g_gcm_keystream_lock must be held by the caller.  Keeping an exact counter
// avoids an O(number-of-records) map scan on every ticket admission.
static void gcm_mark_ticket_consumed_locked(gcm_keystream_record& record)
{
    if (record.consumed) return;
    record.consumed = true;
    g_gcm_keystream_unconsumed.fetch_sub(1, std::memory_order_relaxed);
}

static size_t gcm_keystream_cache_byte_limit()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_POOL_BYTES");
        unsigned long long parsed = env != nullptr
            ? std::strtoull(env, nullptr, 10)
            : 64ULL * 1024ULL * 1024ULL;
        if (parsed < 262144ULL) parsed = 262144ULL;
        if (parsed > 8ULL * 1024ULL * 1024ULL * 1024ULL) {
            parsed = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        }
        limit = static_cast<size_t>(parsed);
        fprintf(stderr, "[SecureInfer] GCM keystream buffer cache: %.1f MiB\n",
                static_cast<double>(limit) / (1024.0 * 1024.0));
    }
    return limit;
}

static aclrtEvent alloc_gcm_keystream_event()
{
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_event_pool_lock);
        if (!g_gcm_keystream_event_pool.empty()) {
            aclrtEvent event = g_gcm_keystream_event_pool.back();
            g_gcm_keystream_event_pool.pop_back();
            return event;
        }
    }
    aclrtEvent event = nullptr;
    return real_aclrtCreateEvent(&event) == ACL_SUCCESS ? event : nullptr;
}

static void* alloc_gcm_keystream_buffer(size_t bytes)
{
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_buffer_pool_lock);
        auto it = g_gcm_keystream_buffer_pool.find(bytes);
        if (it != g_gcm_keystream_buffer_pool.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            g_gcm_keystream_cached_bytes -= bytes;
            return ptr;
        }
    }
    void* ptr = nullptr;
    return real_aclrtMalloc(&ptr, bytes) == ACL_SUCCESS ? ptr : nullptr;
}

static void release_gcm_keystream_buffer(void* ptr, size_t bytes)
{
    if (ptr == nullptr) return;
    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_buffer_pool_lock);
        const size_t limit = gcm_keystream_cache_byte_limit();
        if (bytes <= limit && g_gcm_keystream_cached_bytes <= limit - bytes) {
            g_gcm_keystream_buffer_pool[bytes].push_back(ptr);
            g_gcm_keystream_cached_bytes += bytes;
            cached = true;
        }
    }
    if (!cached) real_aclrtFree(ptr);
}

void secure_infer_gcm_keystream_prewarm(aclrtStream stream)
{
    if (!secure_infer_gcm_keystream_scheduler_enabled()) return;
    const char* count_env =
        std::getenv("PIPE_GCM_KEYSTREAM_PREWARM_TICKETS");
    long requested = count_env != nullptr ? std::atol(count_env) : 0;
    if (requested <= 0) return;
    requested = std::min<long>(requested,
                               static_cast<long>(gcm_keystream_record_limit()));

    const char* bytes_env =
        std::getenv("PIPE_GCM_KEYSTREAM_PREWARM_BYTES");
    unsigned long long parsed_bytes = bytes_env != nullptr
        ? std::strtoull(bytes_env, nullptr, 10)
        : 262144ULL;
    if (parsed_bytes == 0 || parsed_bytes > UINT32_MAX) return;
    const size_t bytes =
        (static_cast<size_t>(parsed_bytes) + 31) & ~static_cast<size_t>(31);
    if (bytes == 0) return;

    // Prewarming must obey the same byte budget as the reusable buffer
    // cache.  Count-only clamping allowed 4096 x 1.25-MiB allocations during
    // model startup even with a 1-GiB cache budget, stealing HBM from vLLM's
    // KV cache before its caruntimety probe.  The cache cannot retain more than
    // this number anyway, so allocating the excess has no warmup value.
    const size_t byte_budget = gcm_keystream_cache_byte_limit();
    const size_t budget_count = std::max<size_t>(1, byte_budget / bytes);
    requested = std::min<long>(
        requested, static_cast<long>(std::min<size_t>(
                       budget_count, static_cast<size_t>(LONG_MAX))));

    std::vector<void*> buffers;
    buffers.reserve(static_cast<size_t>(requested));
    const uint64_t begin = secure_infer_now_us();
    for (long i = 0; i < requested; ++i) {
        void* ptr = alloc_gcm_keystream_buffer(bytes);
        if (ptr == nullptr) break;
        buffers.push_back(ptr);
    }
    // aclrtMalloc may leave the physical backing/page mappings cold.  The
    // pressure control reuses a tiny hot output set, whereas a real 8K reload
    // writes thousands of distinct ticket buffers exactly once.  Opt-in
    // device-side touching makes the comparison about AICPU AES execution,
    // not first-write mapping latency.  It does not generate a keystream and
    // therefore cannot create a false ticket hit.
    const char* touch_env = std::getenv("PIPE_GCM_KEYSTREAM_PREWARM_TOUCH");
    const bool touch = touch_env != nullptr && std::atoi(touch_env) != 0;
    bool touch_ok = true;
    if (touch && stream != nullptr) {
        for (void* ptr : buffers) {
            if (aclrtMemsetAsync(ptr, bytes, 0, bytes, stream) != ACL_SUCCESS) {
                touch_ok = false;
                break;
            }
        }
        if (real_aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            touch_ok = false;
        }
    }
    for (void* ptr : buffers) release_gcm_keystream_buffer(ptr, bytes);
    std::vector<aclrtEvent> events;
    events.reserve(static_cast<size_t>(requested));
    for (long i = 0; i < requested; ++i) {
        aclrtEvent event = nullptr;
        if (real_aclrtCreateEvent(&event) != ACL_SUCCESS || event == nullptr) {
            break;
        }
        events.push_back(event);
    }
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_event_pool_lock);
        g_gcm_keystream_event_pool.insert(g_gcm_keystream_event_pool.end(),
                                          events.begin(), events.end());
    }
    size_t cached_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_buffer_pool_lock);
        cached_bytes = g_gcm_keystream_cached_bytes;
    }
    fprintf(stderr,
            "[SecureInfer] GCM keystream prewarm: buffers=%zu/%ld, events=%zu/%ld, "
            "bytes_each=%zu, cached=%.1f MiB, touch=%s, time=%.3f ms\n",
            buffers.size(), requested, events.size(), requested, bytes,
            static_cast<double>(cached_bytes) / (1024.0 * 1024.0),
            touch ? (touch_ok ? "ok" : "failed") : "off",
            (secure_infer_now_us() - begin) / 1000.0);
}

static const char* gcm_keystream_stats_path()
{
    const char* path = std::getenv("PIPE_GCM_KEYSTREAM_STATS_PATH");
    return path != nullptr && path[0] != '\0' ? path : nullptr;
}

static uint64_t gcm_keystream_stats_every()
{
    static uint64_t every = 0;
    if (every == 0) {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_STATS_EVERY");
        unsigned long long parsed = env != nullptr ? std::strtoull(env, nullptr, 10)
                                                    : 64ULL;
        every = std::max<uint64_t>(parsed, 1ULL);
    }
    return every;
}

static void gcm_emit_keystream_stats(const char* reason)
{
    const char* path = gcm_keystream_stats_path();
    if (path == nullptr) return;
    size_t records = 0;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        records = g_gcm_keystream_records.size();
    }
    double demand_bytes_per_s = 0.0;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_control_lock);
        demand_bytes_per_s = g_gcm_demand_bytes_per_s;
    }
    size_t cached_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_buffer_pool_lock);
        cached_bytes = g_gcm_keystream_cached_bytes;
    }
    FILE* file = std::fopen(path, "a+");
    if (file == nullptr) {
        fprintf(stderr, "[SecureInfer] GCM stats open failed: %s: %s\n",
                path, std::strerror(errno));
        return;
    }
    std::fseek(file, 0, SEEK_END);
    if (std::ftell(file) == 0) {
        fprintf(file, "time_us,reason,ready,busy,missing,aicpu_submitted,"
                "aicore_submitted,critical_aicpu_submitted,"
                "critical_aicore_submitted,dropped,switches,finished,records,"
                "critical,demand_bps,cached_bytes,aicpu_util_pct\n");
    }
    fprintf(file,
            "%lu,%s,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%zu,%lu,%.3f,%zu,%d\n",
            static_cast<unsigned long>(secure_infer_now_us()), reason,
            static_cast<unsigned long>(g_gcm_keystream_ready.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_busy.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_missing.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_aicpu_submitted.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_aicore_submitted.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_critical_aicpu_submitted.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_critical_aicore_submitted.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_dropped.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_switches.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(g_gcm_keystream_finished.load(std::memory_order_relaxed)),
            records,
            static_cast<unsigned long>(g_gcm_keystream_critical.load(std::memory_order_relaxed)),
            demand_bytes_per_s, cached_bytes,
            g_gcm_aicpu_utilization.load(std::memory_order_relaxed));
    std::fclose(file);
}

static size_t gcm_dynamic_watermark(const char* env_name,
                                    size_t default_tickets)
{
    const char* env = std::getenv(env_name);
    long parsed = env != nullptr ? std::atol(env)
                                 : static_cast<long>(default_tickets);
    if (parsed < 1) parsed = 1;
    parsed = std::min<long>(parsed,
                            static_cast<long>(gcm_keystream_record_limit()));
    return static_cast<size_t>(parsed);
}

static uint32_t gcm_aiv_block_dim(size_t bytes)
{
    const char* env = std::getenv("PIPE_GCM_AIV_BLOCK_DIM");
    long parsed = env != nullptr ? std::atol(env) : 20;
    if (parsed < 1) parsed = 1;
    if (parsed > 20) parsed = 20;
    // A single core has lower launch/partition overhead for tiny transfers.
    if (bytes < 4096) parsed = 1;
    return static_cast<uint32_t>(parsed);
}

static uint64_t gcm_aicpu_util_sample_interval_us()
{
    static uint64_t interval = 0;
    if (interval == 0) {
        const char* env = std::getenv("PIPE_GCM_AICPU_UTIL_SAMPLE_MS");
        long parsed = env != nullptr ? std::atol(env) : 100;
        parsed = std::max(20L, std::min(1000L, parsed));
        interval = static_cast<uint64_t>(parsed) * 1000;
    }
    return interval;
}

static int gcm_aicpu_util_high_threshold()
{
    static int threshold = -1;
    if (threshold < 0) {
        const char* env = std::getenv("PIPE_GCM_AICPU_UTIL_HIGH");
        threshold = env != nullptr ? std::atoi(env) : 75;
        threshold = std::max(1, std::min(100, threshold));
        fprintf(stderr,
                "[SecureInfer] dynamic GCM AICPU high-watermark: %d%% "
                "(sample=%lu ms)\n",
                threshold, static_cast<unsigned long>(
                    gcm_aicpu_util_sample_interval_us() / 1000));
    }
    return threshold;
}

// The Ascend utilization runtime can reject a query from a detached worker
// context on some CANN 8.5 builds, even though the exact same device accepts
// it from a short-lived monitor process.  Reuse that monitor's last sample
// when supplied, preserving a resource-API based decision without placing a
// slow or context-sensitive query on the serving path.
static bool gcm_read_aicpu_utilization_trace(const char* path, int* value)
{
    if (path == nullptr || path[0] == '\0' || value == nullptr) return false;
    FILE* file = std::fopen(path, "r");
    if (file == nullptr) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        return false;
    }
    const long end = std::ftell(file);
    if (end <= 0) {
        std::fclose(file);
        return false;
    }
    const long begin = std::max(0L, end - 2048L);
    if (std::fseek(file, begin, SEEK_SET) != 0) {
        std::fclose(file);
        return false;
    }
    std::array<char, 2049> buffer{};
    const size_t bytes = std::fread(buffer.data(), 1, buffer.size() - 1, file);
    std::fclose(file);
    if (bytes == 0) return false;
    buffer[bytes] = '\0';

    char* line = buffer.data() + bytes;
    while (line > buffer.data() && *(line - 1) != '\n') --line;
    // The file often ends in a newline.  In that case skip it and locate the
    // preceding complete row; otherwise `line` already begins the last row.
    if (*line == '\0' && line > buffer.data()) {
        --line;
        while (line > buffer.data() && *(line - 1) != '\n') --line;
    }
    unsigned long long timestamp = 0;
    int cube = -1, vector = -1, aicpu = -1, memory = -1;
    if (std::sscanf(line, "%llu,%d,%d,%d,%d", &timestamp, &cube, &vector,
                    &aicpu, &memory) != 5 || aicpu < 0 || aicpu > 100) {
        return false;
    }
    *value = aicpu;
    return true;
}

static int gcm_cached_aicpu_utilization()
{
    const uint64_t now = secure_infer_now_us();
    std::lock_guard<std::mutex> lock(g_gcm_aicpu_util_lock);
    if (g_gcm_aicpu_util_sample_us != 0 &&
        now - g_gcm_aicpu_util_sample_us <
            gcm_aicpu_util_sample_interval_us()) {
        return g_gcm_aicpu_utilization.load(std::memory_order_acquire);
    }

    // Mark before the slow runtime call so simultaneous admission attempts do
    // not pile up utilization queries.  A failed query is represented by -1,
    // which retains the fast AICPU policy rather than fabricating congestion.
    g_gcm_aicpu_util_sample_us = now;
    int utilization = -1;
    const char* trace_path =
        std::getenv("PIPE_GCM_AICPU_UTILIZATION_TRACE_PATH");
    if (!gcm_read_aicpu_utilization_trace(trace_path, &utilization)) {
        aclrtUtilizationInfo info{};
        info.utilizationExtend = nullptr;
        info.cubeUtilization = info.vectorUtilization = -1;
        info.aicpuUtilization = info.memoryUtilization = -1;
        if (real_aclrtGetDeviceUtilizationRate(0, &info) == ACL_SUCCESS) {
            utilization = info.aicpuUtilization;
        }
    }
    g_gcm_aicpu_utilization.store(utilization, std::memory_order_release);
    return utilization;
}

static void gcm_note_demand(size_t bytes)
{
    const uint64_t now = secure_infer_now_us();
    std::lock_guard<std::mutex> lock(g_gcm_keystream_control_lock);
    if (g_gcm_last_demand_us != 0 && now > g_gcm_last_demand_us) {
        const double interval_us = static_cast<double>(now - g_gcm_last_demand_us);
        // Ignore long idle gaps; they are not a sustainable transfer rate.
        if (interval_us <= 1000000.0) {
            const double instant = static_cast<double>(bytes) * 1.0e6 / interval_us;
            g_gcm_demand_bytes_per_s = g_gcm_demand_bytes_per_s == 0.0
                ? instant
                : 0.2 * instant + 0.8 * g_gcm_demand_bytes_per_s;
        }
    }
    g_gcm_last_demand_us = now;
}

// Count only AIV work that is still owned by the ticket pool. The old
// dynamic policy compared its watermark to *all* records, including AICPU
// tickets, which could suppress AIV exactly when GHASH needed AICPU relief.
static size_t gcm_active_aiv_ticket_count()
{
    std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
    size_t active = 0;
    for (const auto& item : g_gcm_keystream_records) {
        const auto& record = item.second;
        if (record.backend == gcm_keystream_backend::aicore &&
            !record.consumed) {
            ++active;
        }
    }
    return active;
}

static bool gcm_dynamic_switch_on_ghash()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_DYNAMIC_SWITCH_ON_GHASH");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr,
                "[SecureInfer] dynamic real-ticket GHASH protection: %s\n",
                enabled ? "AIV preferred" : "utilization-only");
    }
    return enabled != 0;
}

// Utilization is useful for observability, but it is not by itself a safe
// reason to move speculative tickets to AIV: before an authenticated H2D
// begins, the faster AICPU path can still use its lead time to fill the pool.
// Keep this opt-in for stress sweeps; the production dynamic policy reacts to
// the actual GHASH critical section.
static bool gcm_dynamic_switch_on_utilization()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_DYNAMIC_SWITCH_ON_UTIL");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr,
                "[SecureInfer] dynamic real-ticket utilization preemption: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

// AIV is a relief valve, not an always-on alternate AES path.  Require an
// observed AICPU high-water condition before moving real tickets during a
// GHASH interval; otherwise its lower throughput and host launch overhead can
// only reduce the hit rate.  Disable explicitly only for controlled sweeps.
static bool gcm_dynamic_require_aicpu_high()
{
    static int required = -1;
    if (required < 0) {
        const char* env =
            std::getenv("PIPE_GCM_DYNAMIC_REQUIRE_AICPU_HIGH");
        required = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr,
                "[SecureInfer] dynamic GCM AIV requires high AICPU: %s\n",
                required ? "enabled" : "disabled");
    }
    return required != 0;
}

// During an authenticated H2D, the next few tickets are latency-critical:
// moving them to AIV would replace a ready AICPU keystream with an avoidable
// demand-time fallback.  Retain this near window on AICPU; only tickets that
// have enough lead time are eligible for AIV.  The value is in independently
// reserved transfer tickets (not bytes), which keeps the policy independent
// of the current tensor packing.
static uint64_t gcm_dynamic_aicpu_reserve_tickets()
{
    static uint64_t reserve = 0;
    if (reserve == 0) {
        const char* env =
            std::getenv("PIPE_GCM_DYNAMIC_AICPU_RESERVE_TICKETS");
        unsigned long long parsed = env != nullptr
            ? std::strtoull(env, nullptr, 10)
            : 512ULL;
        reserve = std::max<uint64_t>(parsed, 1ULL);
        fprintf(stderr,
                "[SecureInfer] dynamic GCM AICPU near-ticket reserve: %lu\n",
                static_cast<unsigned long>(reserve));
    }
    return reserve;
}

static bool gcm_ticket_is_near_active_h2d(uint64_t ticket)
{
    if (ticket == 0) return true;
    const uint64_t current =
        g_gcm_h2d_ticket_high_water.load(std::memory_order_acquire);
    // With no observed demand (or a conservatively out-of-order ticket), keep
    // the faster path.  AIV is only selected when we can establish slack.
    if (current == 0 || ticket <= current) return true;
    return ticket - current <= gcm_dynamic_aicpu_reserve_tickets();
}

// The AIV AES kernel is useful caruntimety, but it is not a replacement for the
// whole AICPU producer under a fast KV restore.  Admit only the share it can
// sustainably finish before the consumer catches up.  The conservative
// default (60 MB/s * 0.8) is intentionally far below the standalone AIV rate:
// integrated inference has launch and vector-core interference that a
// microbenchmark does not capture. Deployment can calibrate it with
// PIPE_GCM_AIV_SUSTAIN_BPS.
static int gcm_dynamic_aiv_share_pct()
{
    static double aiv_bps = 0.0;
    static int max_share = -1;
    static double safety = 0.0;
    if (aiv_bps == 0.0) {
        const char* env = std::getenv("PIPE_GCM_AIV_SUSTAIN_BPS");
        aiv_bps = env != nullptr ? std::strtod(env, nullptr) : 6.0e7;
        if (aiv_bps < 1.0e6) aiv_bps = 6.0e7;
    }
    if (max_share < 0) {
        const char* env = std::getenv("PIPE_GCM_DYNAMIC_AIV_MAX_SHARE_PCT");
        max_share = env != nullptr ? std::atoi(env) : 10;
        max_share = std::max(0, std::min(100, max_share));
    }
    if (safety == 0.0) {
        const char* env = std::getenv("PIPE_GCM_DYNAMIC_AIV_SAFETY");
        safety = env != nullptr ? std::strtod(env, nullptr) : 0.8;
        safety = std::max(0.1, std::min(1.0, safety));
        fprintf(stderr,
                "[SecureInfer] dynamic GCM AIV budget: sustain=%.1f MB/s, "
                "safety=%.2f, cap=%d%%\n",
                aiv_bps / 1.0e6, safety, max_share);
    }
    double demand_bps = 0.0;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_control_lock);
        demand_bps = g_gcm_demand_bytes_per_s;
    }
    // No observed restore rate yet: preserve the fast/known-safe AICPU path
    // rather than guessing that a newly started AIV producer has slack.
    if (demand_bps < 1.0e6) return 0;
    const int sustainable = static_cast<int>(
        std::floor(100.0 * aiv_bps * safety / demand_bps));
    return std::max(0, std::min(max_share, sustainable));
}

static bool gcm_dynamic_ticket_uses_aiv_share(uint64_t ticket, int share_pct)
{
    if (share_pct <= 0 || ticket == 0) return false;
    // Ticket IDs are monotonic but can interleave independent requests; a
    // multiplicative mix avoids routing long contiguous ticket runs entirely
    // to one backend while remaining deterministic and stateless.
    const uint64_t mixed = ticket * 11400714819323198485ULL;
    return static_cast<int>(mixed % 100ULL) < share_pct;
}

static gcm_keystream_backend gcm_choose_keystream_backend(
    uint64_t transfer_ticket, uint64_t admission_scope,
    bool* ghash_critical_out = nullptr)
{
    const gcm_keystream_backend configured = gcm_keystream_policy();
    gcm_keystream_backend selected = configured;
    bool ghash_critical = false;
    if (gcm_keystream_policy_is_dynamic()) {
        ghash_critical =
            g_gcm_keystream_critical.load(std::memory_order_acquire) != 0;
        const bool protect_ghash = ghash_critical &&
            gcm_dynamic_switch_on_ghash();
        const bool utilization_preemption =
            gcm_dynamic_switch_on_utilization();
        bool protect_from_utilization = false;
        bool aicpu_high = false;
        const bool need_utilization_sample = utilization_preemption ||
            (protect_ghash && gcm_dynamic_require_aicpu_high());
        if (need_utilization_sample) {
            const int aicpu_utilization = gcm_cached_aicpu_utilization();
            aicpu_high = aicpu_utilization >= gcm_aicpu_util_high_threshold();
            protect_from_utilization = utilization_preemption && aicpu_high;
        }
        const bool ghash_aiv_allowed = protect_ghash &&
            (!gcm_dynamic_require_aicpu_high() || aicpu_high);
        (void)admission_scope;
        const bool keep_near_ticket_on_aicpu = protect_ghash &&
            (!ghash_aiv_allowed ||
             gcm_ticket_is_near_active_h2d(transfer_ticket));
        if ((!ghash_aiv_allowed && !protect_from_utilization) ||
            keep_near_ticket_on_aicpu) {
            // AICPU is the measured fast path.  In particular, do not switch
            // merely because background pressure is high *before* GHASH
            // starts: this lead time is what makes the AICPU ticket pool hit.
            // Likewise, when GHASH is active the near window remains on
            // AICPU so a ticket is ready before its authenticated consumer.
            selected = gcm_keystream_backend::aicpu;
        } else {
            // A real GHASH is now in flight (or an explicitly enabled
            // utilization-only stress policy fired).  Far tickets use only a
            // rate-limited AIV share; routing all of them to AIV is slower
            // than the H2D consumer and turns a scheduler into a miss source.
            const size_t water = gcm_dynamic_watermark(
                "PIPE_GCM_AIV_WATERMARK_TICKETS", 64);
            const size_t active_aiv = gcm_active_aiv_ticket_count();
            const int share_pct = gcm_dynamic_aiv_share_pct();
            if (active_aiv < water &&
                gcm_dynamic_ticket_uses_aiv_share(transfer_ticket,
                                                   share_pct)) {
                selected = gcm_keystream_backend::aicore;
            } else {
                // GHASH is the deadline-critical operation.  Continuing to
                // submit arbitrary far-future keystream work to AICPU after
                // the AIV relief budget is full defeats the purpose of the
                // scheduler: it steals exactly the resource GHASH needs.
                // Return a retryable low-priority deferral instead.  worker
                // rotates this ticket behind runnable work; after the
                // critical interval it naturally resumes on the fast AICPU
                // path.  Near tickets were already retained above.
                selected = gcm_keystream_backend::drop;
            }
        }
        if (selected == gcm_keystream_backend::drop) {
            g_gcm_keystream_dropped.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (ghash_critical_out != nullptr) *ghash_critical_out = ghash_critical;
    if (selected != gcm_keystream_backend::drop) {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_control_lock);
        if (g_gcm_last_backend != gcm_keystream_backend::drop &&
            g_gcm_last_backend != selected) {
            g_gcm_keystream_switches.fetch_add(1, std::memory_order_relaxed);
        }
        g_gcm_last_backend = selected;
    }
    return selected;
}

// The worker uses this same gate before it spends admission budget on a later
// queued request.  Keeping the queue policy aligned with backend selection is
// important: without it, fair queueing could slow the urgent request even
// when no AIV relief is actually permitted.
bool secure_infer_gcm_dynamic_relief_active()
{
    if (!secure_infer_gcm_dynamic_keystream_policy_enabled() ||
        !gcm_dynamic_switch_on_ghash() ||
        !secure_infer_gcm_ghash_critical_active()) {
        return false;
    }
    return !gcm_dynamic_require_aicpu_high() ||
        gcm_cached_aicpu_utilization() >= gcm_aicpu_util_high_threshold();
}

static void release_gcm_keystream_record(gcm_keystream_record& record)
{
    if (record.ready_event != nullptr) {
        real_aclrtDestroyEvent(record.ready_event);
    }
    if (record.allocation != nullptr) {
        release_gcm_keystream_buffer(record.allocation, record.bytes);
    }
    record = gcm_keystream_record{};
}

void secure_infer_gcm_keystream_cleanup()
{
    std::vector<gcm_keystream_record> releasable;
    const uint64_t retired =
        g_retired_transfer_generation.load(std::memory_order_acquire);
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        for (auto it = g_gcm_keystream_records.begin();
             it != g_gcm_keystream_records.end();) {
            auto& record = it->second;
            const bool discard = record.consumed || record.generation <= retired;
            if (!discard || record.in_use || record.launching ||
                record.ready_event == nullptr) {
                ++it;
                continue;
            }
            aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            const aclError query = aclrtQueryEventStatus(record.ready_event, &status);
            if (query != ACL_SUCCESS || status != ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                ++it;
                continue;
            }
            gcm_mark_ticket_consumed_locked(record);
            releasable.push_back(record);
            it = g_gcm_keystream_records.erase(it);
        }
    }
    for (auto& record : releasable) release_gcm_keystream_record(record);
    secure_infer_gcm_cpu_cipher_cleanup();
}

// Ticket admission and ticket retirement are both hot paths: one 8K reload
// contains 5,040 independent authenticated slices.  Calling the full cleanup
// routine for *every* slice scans the complete ticket table and queries every
// ACL event, which turns a bounded cache into O(n^2) host work exactly when
// the producer needs to get ahead of H2D.  A small deferred window is safe:
// records remain owned until their completion event, and at most this many
// already-consumed 256-KiB buffers stay temporarily reserved.
static size_t gcm_keystream_cleanup_interval()
{
    static size_t interval = 0;
    if (interval == 0) {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_CLEANUP_INTERVAL");
        long parsed = env != nullptr ? std::atol(env) : 64;
        parsed = std::max(1L, std::min(4096L, parsed));
        interval = static_cast<size_t>(parsed);
        fprintf(stderr, "[SecureInfer] GCM ticket cleanup interval: %zu\n",
                interval);
    }
    return interval;
}

static void secure_infer_gcm_keystream_maybe_cleanup()
{
    static std::atomic<uint64_t> operations{0};
    const uint64_t operation = operations.fetch_add(
        1, std::memory_order_relaxed) + 1;
    if (operation % gcm_keystream_cleanup_interval() == 0) {
        secure_infer_gcm_keystream_cleanup();
    }
}

bool secure_infer_gcm_keystream_known(uint64_t transfer_ticket)
{
    if (!secure_infer_gcm_keystream_scheduler_enabled() || transfer_ticket == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
    return g_gcm_keystream_records.find(transfer_ticket) !=
        g_gcm_keystream_records.end();
}

static size_t gcm_aiv_producer_count()
{
    static size_t count = 0;
    if (count == 0) {
        const char* env = std::getenv("PIPE_GCM_AIV_STREAMS");
        long parsed = env != nullptr ? std::atol(env) : 1;
        count = static_cast<size_t>(std::max(1L, std::min(4L, parsed)));
    }
    return count;
}

static bool acquire_gcm_aiv_producer(aclrtStream base_stream,
                                     aclrtStream* selected_stream,
                                     void** selected_iv)
{
    if (base_stream == nullptr || selected_stream == nullptr ||
        selected_iv == nullptr || g_dev_iv == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_gcm_aiv_producer_lock);
    if (g_gcm_aiv_producers.empty()) {
        g_gcm_aiv_producers.push_back(
            {base_stream, g_dev_iv, false, false});
    }
    const size_t target = gcm_aiv_producer_count();
    while (g_gcm_aiv_producers.size() < target) {
        gcm_aiv_producer producer;
        if (real_aclrtCreateStream(&producer.stream) != ACL_SUCCESS ||
            producer.stream == nullptr ||
            real_aclrtMalloc(&producer.device_iv, 32) != ACL_SUCCESS ||
            producer.device_iv == nullptr) {
            if (producer.device_iv != nullptr) real_aclrtFree(producer.device_iv);
            if (producer.stream != nullptr) aclrtDestroyStream(producer.stream);
            break;
        }
        producer.owns_stream = true;
        producer.owns_iv = true;
        g_gcm_aiv_producers.push_back(producer);
    }
    const size_t index = static_cast<size_t>(
        g_gcm_aiv_next_producer.fetch_add(1, std::memory_order_relaxed) %
        g_gcm_aiv_producers.size());
    const auto& producer = g_gcm_aiv_producers[index];
    *selected_stream = producer.stream;
    *selected_iv = producer.device_iv;
    return *selected_stream != nullptr && *selected_iv != nullptr;
}

static bool enqueue_gcm_aiv_keystream(aclrtStream stream, void* device_iv,
                                      void* output,
                                      size_t bytes,
                                      const unsigned char aiv_iv[32])
{
    if (stream == nullptr || output == nullptr || bytes == 0 ||
        aiv_iv == nullptr || g_dev_key == nullptr || device_iv == nullptr) {
        return false;
    }
    if (!load_verified_aiv_keystream_library()) {
        return false;
    }
    const size_t aligned_bytes = (bytes + 31) & ~static_cast<size_t>(31);
    if (aligned_bytes == 0 || aligned_bytes > UINT32_MAX) return false;
    if (!wait_h2d_device_crypto_ready(stream)) return false;
    // The verified AIV image takes an IV pointer.  Each producer owns its
    // 32-byte device IV buffer, so independent streams may enqueue safely in
    // parallel.  The host array belongs to the ticket record and remains
    // alive through its ready event.
    if (real_aclrtMemcpyAsync(device_iv, 32, aiv_iv, 32,
                              ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
        return false;
    }
    // Mode 1 generates AES-CTR keystream and does not read x.  Pass output
    // as x only to satisfy the ABI's non-null device-pointer contract.
    g_launch_verified_aiv_keystream(
        gcm_aiv_block_dim(aligned_bytes), stream,
        static_cast<uint8_t*>(output), static_cast<uint8_t*>(g_dev_key),
        static_cast<uint8_t*>(device_iv), static_cast<uint8_t*>(output),
        static_cast<uint32_t>(aligned_bytes), 1);
    return true;
}

static bool gcm_copy_ticket_base_iv(unsigned char out[iv_length])
{
    if (out == nullptr) return false;
    if (!g_gcm_ticket_base_iv_ready.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_gcm_ticket_iv_lock);
        if (!g_gcm_ticket_base_iv_ready.load(std::memory_order_relaxed)) {
            // Metadata is initialized before the first ticket scheduler runs;
            // copy it exactly once while taking the same lock used by the
            // H2D path, then never put individual AIV submissions behind
            // foreground GHASH merely to read immutable IV bytes.
            std::lock_guard<std::mutex> inline_lock(g_inline_h2d_lock);
            if (m_ctx_metadata.empty()) return false;
            std::memcpy(g_gcm_ticket_base_iv.data(),
                        m_ctx_metadata.begin()->second->init_iv, iv_length);
            g_gcm_ticket_base_iv_ready.store(true, std::memory_order_release);
        }
    }
    std::memcpy(out, g_gcm_ticket_base_iv.data(), iv_length);
    return true;
}

// The batch is deliberately small.  AICPU executes its payload loop on one
// core, so a very large descriptor would turn a background optimization into
// a long non-preemptible interval for GHASH.  Two tickets are the serving
// default: this amortizes launch/key expansion while keeping the first
// ticket's added wait below one 256-KiB AES invocation.  Wider batches remain
// available as an explicit pressure-sweep setting.
static size_t gcm_aicpu_batch_width()
{
    static size_t width = 0;
    if (width == 0) {
        const char* env = std::getenv("PIPE_GCM_AICPU_BATCH_TICKETS");
        long parsed = env != nullptr ? std::atol(env) : 2;
        parsed = std::max(1L, std::min(32L, parsed));
        width = static_cast<size_t>(parsed);
        fprintf(stderr, "[SecureInfer] GCM AICPU multi-IV batch width: %zu\n", width);
    }
    return width;
}

bool secure_infer_gcm_aicpu_batching_enabled()
{
    if (!secure_infer_gcm_keystream_scheduler_enabled() ||
        !secure_infer_gcm_enabled() || gcm_aicpu_batch_width() <= 1) {
        return false;
    }
    if (!gcm_keystream_policy_is_dynamic()) {
        return gcm_keystream_policy() == gcm_keystream_backend::aicpu;
    }
    // Dynamic placement is identical to fixed AICPU outside a relief
    // interval, so denying it the same launch/event batching creates an
    // artificial baseline gap before the scheduler has made any decision.
    // Utilization-only switching can fire without a GHASH boundary and thus
    // remains per-ticket.  The common GHASH-driven policy is rechecked by
    // every batch below; once critical, worker falls back to exact per-ticket
    // placement for AICPU/AIV/defer decisions.
    return !gcm_dynamic_switch_on_utilization();
}

size_t secure_infer_gcm_aicpu_batch_ticket_limit()
{
    return gcm_aicpu_batch_width();
}

int secure_infer_gcm_schedule_ticketed_keystream(aclrtStream stream,
                                            uint64_t transfer_ticket,
                                            size_t bytes,
                                            uint64_t generation,
                                            uint64_t admission_scope)
{
    if (!secure_infer_gcm_keystream_scheduler_enabled() || !secure_infer_gcm_enabled() ||
        stream == nullptr || transfer_ticket == 0 || bytes == 0) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        gcm_refresh_demand_history_locked();
        if (generation <= g_gcm_demand_history_retired ||
            g_gcm_demanded_tickets.count(transfer_ticket) != 0) return 1;
    }
    secure_infer_gcm_keystream_maybe_cleanup();
    const size_t aligned_bytes = (bytes + 31) & ~static_cast<size_t>(31);
    if (aligned_bytes == 0 || aligned_bytes > UINT32_MAX) {
        return 0;
    }

    // Select before taking g_inline_h2d_lock in the launch helper.  If this
    // ran after prepare_h2d_device_crypto(), a worker blocked behind the
    // foreground H2D would observe GHASH only after it had finished and
    // incorrectly select AICPU again.
    // A recorded AICPU operation cannot be cancelled safely, so this is a
    // dispatch decision for not-yet-launched real tickets; existing ticket
    // records deliberately retain their original backend and IV/output.
    bool ghash_critical_at_submit = false;
    const gcm_keystream_backend backend = gcm_choose_keystream_backend(
        transfer_ticket, admission_scope, &ghash_critical_at_submit);
    if (backend == gcm_keystream_backend::drop) {
        // This is not a failed launch.  The caller can defer this far-future
        // ticket behind later near-deadline work and retry it after an AIV
        // completion; treating it as an ordinary false result previously
        // pinned it at the head of worker::enc_tasks.
        return -1;
    }
    if (!secure_infer_prepare_h2d_device_crypto(stream)) return 0;

    std::array<unsigned char, 32> aiv_iv{};
    if (backend == gcm_keystream_backend::aicore) {
        unsigned char base_iv[iv_length] = {};
        if (!gcm_copy_ticket_base_iv(base_iv)) return 0;
        secure_infer_derive_transfer_iv(base_iv, transfer_ticket, 0, aiv_iv.data());
    }

    const unsigned char* aiv_iv_ptr = nullptr;
    gcm_keystream_record evicted;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        gcm_refresh_demand_history_locked();
        if (generation <= g_gcm_demand_history_retired ||
            g_gcm_demanded_tickets.count(transfer_ticket) != 0 ||
            g_gcm_keystream_records.find(transfer_ticket) !=
                g_gcm_keystream_records.end()) {
            return 1;
        }
        if (g_gcm_keystream_records.size() >= gcm_keystream_record_limit() ||
            gcm_unconsumed_ticket_count_locked() >=
                gcm_keystream_inflight_limit()) {
            // Completed unused predictions are disposable cache entries, not
            // a reason to block every later useful prediction. Evict the
            // oldest ticket only after its producer event has completed.
            // Never reclaim a launching or consumer-owned allocation.
            auto victim = g_gcm_keystream_records.end();
            for (auto it = g_gcm_keystream_records.begin();
                 it != g_gcm_keystream_records.end(); ++it) {
                const auto& record = it->second;
                if (record.in_use || record.launching || record.ready_event == nullptr) continue;
                if (victim == g_gcm_keystream_records.end() || it->first < victim->first) victim = it;
            }
            if (victim == g_gcm_keystream_records.end()) return 0;
            aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            if (aclrtQueryEventStatus(victim->second.ready_event, &status) != ACL_SUCCESS ||
                status != ACL_EVENT_RECORDED_STATUS_COMPLETE) return 0;
            gcm_mark_ticket_consumed_locked(victim->second);
            evicted = std::move(victim->second);
            g_gcm_keystream_records.erase(victim);
            static std::atomic<uint64_t> evictions{0};
            const uint64_t count = evictions.fetch_add(1, std::memory_order_relaxed) + 1;
            if (count <= 4 || count % 1024 == 0)
                fprintf(stderr, "[SecureInfer] completed unused keystream evictions: %lu\n", count);
        }
        gcm_keystream_record pending;
        pending.bytes = aligned_bytes;
        pending.generation = generation;
        pending.backend = backend;
        pending.aiv_iv = aiv_iv;
        pending.launching = true;
        auto inserted = g_gcm_keystream_records.emplace(transfer_ticket,
                                                         std::move(pending));
        if (!inserted.second) return 1;
        g_gcm_keystream_unconsumed.fetch_add(1, std::memory_order_relaxed);
        if (backend == gcm_keystream_backend::aicore) {
            aiv_iv_ptr = inserted.first->second.aiv_iv.data();
        }
    }

    if (evicted.allocation != nullptr) release_gcm_keystream_record(evicted);

    // AICPU tickets stay on the caller's stream.  AIV tickets use one of the
    // independent producer streams, with its own IV buffer, so a slow ticket
    // does not serialize all subsequent AIV pre-generation.
    aclrtStream launch_stream = stream;
    void* aiv_device_iv = nullptr;
    const bool producer_ready = backend != gcm_keystream_backend::aicore ||
        acquire_gcm_aiv_producer(stream, &launch_stream, &aiv_device_iv);

    void* allocation = nullptr;
    aclrtEvent event = nullptr;
    allocation = alloc_gcm_keystream_buffer(aligned_bytes);
    if (allocation != nullptr) event = alloc_gcm_keystream_event();
    bool launched = producer_ready && allocation != nullptr && event != nullptr;
    if (launched) {
        launched = backend == gcm_keystream_backend::aicpu
            ? secure_infer_enqueue_ticketed_keystream(stream, allocation,
                                                 aligned_bytes, transfer_ticket, 0)
            : enqueue_gcm_aiv_keystream(launch_stream, aiv_device_iv, allocation,
                                        aligned_bytes, aiv_iv_ptr);
    }
    if (launched) {
        launched = real_aclrtRecordEvent(event, launch_stream) == ACL_SUCCESS;
    }
    if (!launched) {
        // A failed record may follow a successfully queued kernel.  Drain the
        // producer stream before reclaiming its output so correctness is never
        // traded for an experimental admission failure.
        real_aclrtSynchronizeStream(launch_stream);
        if (event != nullptr) real_aclrtDestroyEvent(event);
        if (allocation != nullptr) {
            release_gcm_keystream_buffer(allocation, aligned_bytes);
        }
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        auto it = g_gcm_keystream_records.find(transfer_ticket);
        if (it != g_gcm_keystream_records.end()) {
            gcm_mark_ticket_consumed_locked(it->second);
            g_gcm_keystream_records.erase(it);
        }
        return 0;
    }

    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        auto it = g_gcm_keystream_records.find(transfer_ticket);
        if (it == g_gcm_keystream_records.end()) {
            // Defensive: another cleanup must never remove a launching entry,
            // but retain a safe path should that invariant change.
            real_aclrtSynchronizeStream(launch_stream);
            real_aclrtDestroyEvent(event);
            release_gcm_keystream_buffer(allocation, aligned_bytes);
            return 0;
        }
        it->second.allocation = allocation;
        it->second.ready_event = event;
        it->second.launching = false;
    }
    if (backend == gcm_keystream_backend::aicpu) {
        g_gcm_keystream_aicpu_submitted.fetch_add(1, std::memory_order_relaxed);
        if (ghash_critical_at_submit) {
            g_gcm_keystream_critical_aicpu_submitted.fetch_add(
                1, std::memory_order_relaxed);
        }
    } else {
        g_gcm_keystream_aicore_submitted.fetch_add(1, std::memory_order_relaxed);
        if (ghash_critical_at_submit) {
            g_gcm_keystream_critical_aicore_submitted.fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    return 1;
}

bool secure_infer_gcm_schedule_ticketed_keystream_batch(
    aclrtStream stream,
    const std::vector<gcm_keystream_submission>& submissions,
    std::vector<int>* admissions)
{
    if (admissions != nullptr) admissions->assign(submissions.size(), 0);
    if (!secure_infer_gcm_aicpu_batching_enabled() || stream == nullptr ||
        submissions.size() < 2 ||
        submissions.size() > gcm_aicpu_batch_width()) {
        return false;
    }

    // Do not call the historical multi-IV AICPU ABI here.  Its descriptor
    // owns an output/IV array and a single completion point, which is not a
    // valid ownership model for independently consumed GCM tickets.  Keep the
    // admission batch as a scheduling hint, but dispatch every member through
    // the audited single-ticket path.  That path allocates one output and one
    // ACL event per ticket and records the exact event in its own map entry.
    for (size_t index = 0; index < submissions.size(); ++index) {
        const auto& submission = submissions[index];
        const int admission = secure_infer_gcm_schedule_ticketed_keystream(
            stream, submission.transfer_ticket, submission.bytes,
            submission.generation, submission.admission_scope);
        if (admissions != nullptr) admissions->at(index) = admission;
    }
    return true;
}

int secure_infer_gcm_ticket_backend(uint64_t transfer_ticket)
{
    if (transfer_ticket == 0) return -1;
    std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
    const auto it = g_gcm_keystream_records.find(transfer_ticket);
    if (it == g_gcm_keystream_records.end()) return -1;
    return it->second.backend == gcm_keystream_backend::aicore ? 1 : 0;
}

static bool gcm_ticket_device_wait_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_TICKET_DEVICE_WAIT");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] GCM ticket device-event wait: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

bool secure_infer_gcm_take_ready_keystream(aclrtStream consumer_stream,
                                      uint64_t transfer_ticket,
                                      size_t bytes,
                                      void** keystream_out)
{
    if (keystream_out != nullptr) *keystream_out = nullptr;
    if (!secure_infer_gcm_keystream_scheduler_enabled() || transfer_ticket == 0 ||
        bytes == 0 || keystream_out == nullptr) {
        return false;
    }
    gcm_note_demand(bytes);
    uint64_t observed =
        g_gcm_h2d_ticket_high_water.load(std::memory_order_acquire);
    while (observed < transfer_ticket &&
           !g_gcm_h2d_ticket_high_water.compare_exchange_weak(
               observed, transfer_ticket, std::memory_order_acq_rel,
               std::memory_order_acquire)) {
        // compare_exchange refreshes observed.
    }
    std::unique_lock<std::mutex> lock(g_gcm_keystream_lock);
    gcm_refresh_demand_history_locked();
    g_gcm_demanded_tickets.insert(transfer_ticket);
    auto it = g_gcm_keystream_records.find(transfer_ticket);
    if (it == g_gcm_keystream_records.end()) {
        g_gcm_keystream_missing.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto& record = it->second;
    gcm_mark_ticket_consumed_locked(record);
    if (record.launching || record.ready_event == nullptr ||
        record.bytes < bytes) {
        g_gcm_keystream_busy.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    const aclError query = aclrtQueryEventStatus(record.ready_event, &status);
    if (query != ACL_SUCCESS) {
        g_gcm_keystream_busy.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    record.in_use = true;
    *keystream_out = record.allocation;
    aclrtEvent producer_done = record.ready_event;
    const bool needs_device_wait =
        status != ACL_EVENT_RECORDED_STATUS_COMPLETE;
    if (needs_device_wait &&
        (!gcm_ticket_device_wait_enabled() || consumer_stream == nullptr)) {
        record.in_use = false;
        *keystream_out = nullptr;
        g_gcm_keystream_busy.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Ownership is pinned by in_use before releasing the map lock, so the
    // producer event and output buffer cannot be reclaimed while the runtime
    // installs this cross-stream dependency.  This converts a producer race
    // into device-side pipelining instead of redundantly executing online AES.
    lock.unlock();
    if (needs_device_wait &&
        real_aclrtStreamWaitEvent(consumer_stream, producer_done) != ACL_SUCCESS) {
        std::lock_guard<std::mutex> retry_lock(g_gcm_keystream_lock);
        auto retry = g_gcm_keystream_records.find(transfer_ticket);
        if (retry != g_gcm_keystream_records.end()) {
            retry->second.in_use = false;
        }
        *keystream_out = nullptr;
        g_gcm_keystream_busy.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (needs_device_wait) {
        const uint64_t waits = g_gcm_keystream_device_waits.fetch_add(
            1, std::memory_order_relaxed) + 1;
        if (waits <= 8 || waits % 1024 == 0) {
            fprintf(stderr,
                    "[SecureInfer] GCM ticket device-event dependency: count=%lu\n",
                    static_cast<unsigned long>(waits));
        }
    }
    g_gcm_keystream_ready.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void secure_infer_gcm_finish_keystream_ticket(uint64_t transfer_ticket)
{
    if (transfer_ticket == 0) return;
    {
        std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
        auto it = g_gcm_keystream_records.find(transfer_ticket);
        if (it != g_gcm_keystream_records.end()) {
            gcm_mark_ticket_consumed_locked(it->second);
            it->second.in_use = false;
        }
    }
    secure_infer_gcm_keystream_maybe_cleanup();
    const uint64_t finished =
        g_gcm_keystream_finished.fetch_add(1, std::memory_order_relaxed) + 1;
    if (finished % gcm_keystream_stats_every() == 0) {
        gcm_emit_keystream_stats("consume");
    }
}

void secure_infer_gcm_ghash_critical_enter()
{
    const uint64_t previous =
        g_gcm_keystream_critical.fetch_add(1, std::memory_order_acq_rel);
    // Optional cross-process scheduling signal for the controlled resource
    // experiment.  It changes no serving-path decision by default.  A
    // cooperating background keystream producer observes this flag and moves
    // from AICPU to AIV while an authenticated H2D is outstanding.  Only the
    // 0 -> 1 transition touches the filesystem, never each H2D slice.
    if (previous == 0) {
        const char* path = std::getenv("PIPE_GHASH_CRITICAL_FLAG_PATH");
        if (path != nullptr && path[0] != '\0') {
            const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (fd >= 0) close(fd);
        }
    }
}

bool secure_infer_gcm_ghash_critical_active()
{
    return g_gcm_keystream_critical.load(std::memory_order_acquire) != 0;
}

void secure_infer_gcm_ghash_critical_exit()
{
    uint64_t current =
        g_gcm_keystream_critical.load(std::memory_order_acquire);
    while (current != 0 &&
           !g_gcm_keystream_critical.compare_exchange_weak(
               current, current - 1, std::memory_order_acq_rel,
               std::memory_order_acquire)) {
        // compare_exchange reloads current; retry until the paired enter is
        // observed or a defensive zero is reached.
    }
    if (current == 1) {
        const char* path = std::getenv("PIPE_GHASH_CRITICAL_FLAG_PATH");
        if (path != nullptr && path[0] != '\0') {
            unlink(path);
        }
    }
}

extern "C" __attribute__((visibility("default"))) void
secure_infer_get_gcm_keystream_stats(uint64_t* ready, uint64_t* busy,
                                uint64_t* missing, uint64_t* aicpu_submitted,
                                uint64_t* aicore_submitted, uint64_t* dropped,
                                uint64_t* switches)
{
    if (ready != nullptr) *ready = g_gcm_keystream_ready.load(std::memory_order_relaxed);
    if (busy != nullptr) *busy = g_gcm_keystream_busy.load(std::memory_order_relaxed);
    if (missing != nullptr) *missing = g_gcm_keystream_missing.load(std::memory_order_relaxed);
    if (aicpu_submitted != nullptr) *aicpu_submitted =
        g_gcm_keystream_aicpu_submitted.load(std::memory_order_relaxed);
    if (aicore_submitted != nullptr) *aicore_submitted =
        g_gcm_keystream_aicore_submitted.load(std::memory_order_relaxed);
    if (dropped != nullptr) *dropped = g_gcm_keystream_dropped.load(std::memory_order_relaxed);
    if (switches != nullptr) *switches = g_gcm_keystream_switches.load(std::memory_order_relaxed);
}

extern "C" __attribute__((visibility("default"))) uint64_t
secure_infer_count_ready_unconsumed_gcm_tickets()
{
    uint64_t ready = 0;
    std::lock_guard<std::mutex> lock(g_gcm_keystream_lock);
    for (const auto& item : g_gcm_keystream_records) {
        const auto& record = item.second;
        if (record.consumed || record.in_use || record.launching ||
            record.ready_event == nullptr) {
            continue;
        }
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        if (aclrtQueryEventStatus(record.ready_event, &status) == ACL_SUCCESS &&
            status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
            ++ready;
        }
    }
    return ready;
}

bool secure_infer_enqueue_ticketed_ctr_xor(aclrtStream stream,
                                      void* input,
                                      void* output,
                                      size_t bytes,
                                      uint64_t transfer_ticket,
                                      uint64_t block_offset)
{
    if (stream == nullptr || input == nullptr || output == nullptr ||
        bytes == 0 || transfer_ticket == 0 || secure_infer_gcm_enabled() ||
        m_ctx_metadata.empty()) {
        return false;
    }

    // The lock only protects lazy key/kernel initialization.  Input/output
    // buffers belong to the commit worker, so the queued kernel itself does
    // not share the serial fallback's staging buffers.
    std::lock_guard<std::mutex> inline_lock(g_inline_h2d_lock);
    auto& meta = *m_ctx_metadata.begin()->second;
    if (!ensure_h2d_device_crypto(meta, stream)) return false;

    unsigned char iv[iv_length];
    secure_infer_derive_transfer_iv(meta.init_iv, transfer_ticket,
                               block_offset, iv);
    uint32_t iv_words[4] = {0, 0, 0, 0};
    pack_inline_iv(iv, iv_words);

    const size_t aligned_bytes = (bytes + 31) & ~static_cast<size_t>(31);
    if (aligned_bytes == 0 || aligned_bytes > UINT32_MAX) return false;
    if (!load_ghash_library() || g_launch_aicpu_aes_ctr == nullptr) {
        return false;
    }
    return g_launch_aicpu_aes_ctr(
        stream, input, g_dev_key, output, aligned_bytes,
        iv_words[0], iv_words[1], iv_words[2], iv_words[3]) == 0;
}

struct inline_device_timing {
    aclrtEvent pipeline_start = nullptr;
    aclrtEvent kernel_start = nullptr;
    aclrtEvent kernel_end = nullptr;
    aclrtEvent pipeline_end = nullptr;
    char run_id[128] = "";
    bool active = false;
    bool records_ok = false;
};

static void destroy_inline_device_timing(inline_device_timing& timing)
{
    if (timing.pipeline_start) real_aclrtDestroyEvent(timing.pipeline_start);
    if (timing.kernel_start) real_aclrtDestroyEvent(timing.kernel_start);
    if (timing.kernel_end) real_aclrtDestroyEvent(timing.kernel_end);
    if (timing.pipeline_end) real_aclrtDestroyEvent(timing.pipeline_end);
    timing = inline_device_timing{};
}

static bool begin_inline_device_timing(inline_device_timing& timing,
                                       aclrtStream stream)
{
    if (!profile_begin_precise_capture(timing.run_id, sizeof(timing.run_id))) return false;
    // A GCM H2D pipeline slot reuses its events after the slot's completion
    // event has retired.  The legacy serial path still creates/destroys this
    // object per request.  Supporting both forms keeps precise profiling
    // correct while avoiding an event allocation for every 256-KiB slice.
    if (!timing.active) {
        if (real_aclrtCreateEventWithFlag(&timing.pipeline_start, ACL_EVENT_TIME_LINE) != ACL_SUCCESS ||
            real_aclrtCreateEventWithFlag(&timing.kernel_start, ACL_EVENT_TIME_LINE) != ACL_SUCCESS ||
            real_aclrtCreateEventWithFlag(&timing.kernel_end, ACL_EVENT_TIME_LINE) != ACL_SUCCESS ||
            real_aclrtCreateEventWithFlag(&timing.pipeline_end, ACL_EVENT_TIME_LINE) != ACL_SUCCESS) {
            destroy_inline_device_timing(timing);
            return false;
        }
        timing.active = true;
    }
    timing.records_ok = real_aclrtRecordEvent(timing.pipeline_start, stream) == ACL_SUCCESS;
    return true;
}

static uint64_t inline_event_elapsed_us(aclrtEvent start,
                                        aclrtEvent end,
                                        bool& valid)
{
    float ms = 0.0f;
    if (real_aclrtEventElapsedTime(&ms, start, end) != ACL_SUCCESS || ms < 0.0f) {
        valid = false;
        return 0;
    }
    return static_cast<uint64_t>(ms * 1000.0f + 0.5f);
}

// H2D GCM used one global staging pair and synchronized the whole framework
// stream after every physical KV slice.  An 8K reload has 5,040 slices, so
// that choice turned authentication into a host-side stop-and-wait protocol.
//
// These slots preserve the required order for each slice:
//   ciphertext H2D -> tag mask -> GHASH verify -> CTR/XOR -> auth gate -> KV
// but allow several independent slices to be queued before the host waits to
// recycle their staging buffers.  The plaintext D2D copy remains after the
// auth gate on the same user stream, so an unauthenticated plaintext is never
// published to the KV cache.
struct h2d_gcm_pipeline_slot {
    void* temp_in = nullptr;
    void* temp_out = nullptr;
    void* cipher_host = nullptr;
    size_t bytes = 0;
    aclrtEvent done = nullptr;
    // Optional heterogeneous fork/join state.  The two child streams never
    // publish a layer event themselves; the parent records completion only
    // after both tasks and the fail-closed gate have finished.
    aclrtEvent fork_ready = nullptr;
    aclrtEvent verify_done = nullptr;
    aclrtEvent xor_done = nullptr;
    aclrtStream verify_stream = nullptr;
    aclrtStream xor_stream = nullptr;
    bool used = false;
    // A ready ticket remains owned by its producer record until this slot's
    // D2D publication completes.  Releasing it any earlier could recycle the
    // keystream buffer while the verified XOR kernel still reads it.
    uint64_t keystream_ticket = 0;
    // Non-zero only when the CPU pre-encryption record's pinned buffer was
    // directly submitted to this slot's H2D DMA.  Its ownership ends at the
    // same authenticated-publication event as the other slot resources.
    uint64_t cpu_cipher_ticket = 0;
    bool ghash_critical = false;

    inline_device_timing timing;
    bool profile_pending = false;
    bool profile_precise = false;
    uint64_t profile_total_start_us = 0;
    uint64_t profile_cpu_crypto_us = 0;
    uint64_t profile_host_device_start_us = 0;
    uint64_t profile_copy_enqueue_us = 0;
    uint64_t profile_kernel_enqueue_us = 0;
    size_t profile_count = 0;
    size_t profile_aligned_bytes = 0;
    bool profile_cpu_prediction_hit = false;
    // These four bits form two exhaustive partitions for every profiled GCM
    // H2D: ticket-XOR vs. on-demand crypto, and CPU-preencrypted vs. inline
    // CPU encryption.  Keep them with the slot until its completion rather
    // than sampling global ticket counters, which cannot account for a
    // descriptor reservation that never reached take_ready().
    bool profile_gcm_ticket_xor = false;
    // 0=AICPU, 1=AIV, -1=no ready ticket. This captures the backend of the
    // ticket actually consumed by this H2D, not merely aggregate admission.
    int profile_ticket_backend = -1;
    bool profile_gcm_on_demand_fallback = false;
    bool profile_cpu_inline_fallback = false;
    bool profile_tamper_injected = false;
    bool profile_auth_bypassed = false;
    void* profile_auth_status = nullptr;
    void* profile_plaintext_output = nullptr;
    const void* profile_expected_plaintext = nullptr;
    uint32_t profile_logical_layer = UINT32_MAX;
    uint32_t profile_tensor_kind = UINT32_MAX;
    std::shared_future<int> authentication_observed;
};

// Recovery experiments observe every verifier result, not injector metadata.
// The worker waits on a real completion event; there is no layer-count timer.
static bool recovery_event_monitor_enabled()
{
    const char* path = std::getenv("PIPE_GCM_FAULT_EVENT_PATH");
    return path != nullptr && *path != '\0';
}

struct recovery_auth_record {
    aclrtContext context = nullptr;
    aclrtEvent ready = nullptr;
    uint32_t* host_status = nullptr;
    uint32_t layer = UINT32_MAX;
    uint32_t tensor = UINT32_MAX;
    size_t bytes = 0;
    uint64_t job_id = 0;
    size_t destination_caruntimety = 0;
    void* destination = nullptr;
    const void* source = nullptr;
    aclrtStream stream = nullptr;
    std::promise<int> result;
    std::shared_future<int> observed = result.get_future().share();
};
struct recovery_auth_observer {
    std::mutex lock;
    std::condition_variable changed;
    std::deque<std::shared_ptr<recovery_auth_record>> pending;
};
static std::mutex g_recovery_auth_lock;
static std::unordered_map<void*, recovery_auth_observer*> g_recovery_observers;
static std::map<uint32_t, std::vector<std::shared_ptr<recovery_auth_record>>>
    g_recovery_layer_records;
static thread_local uint64_t g_recovery_job_id = 0;

extern "C" void secure_infer_recovery_set_job_id(uint64_t job_id)
{
    g_recovery_job_id = job_id;
}

static std::shared_ptr<recovery_auth_record> observe_recovery_authentication(
    void* device_status, aclrtStream verifier_stream, uint32_t layer,
    uint32_t tensor, void* destination, size_t destination_caruntimety,
    const void* source, size_t bytes, aclrtStream transfer_stream)
{
    auto record = std::make_shared<recovery_auth_record>();
    record->layer = layer;
    record->tensor = tensor;
    record->bytes = bytes;
    record->job_id = g_recovery_job_id;
    record->destination = destination;
    record->destination_caruntimety = destination_caruntimety;
    record->source = source;
    record->stream = transfer_stream;
    if (aclrtGetCurrentContext(&record->context) != ACL_SUCCESS ||
        real_aclrtMallocHost(reinterpret_cast<void**>(&record->host_status),
                            sizeof(uint32_t)) != ACL_SUCCESS ||
        real_aclrtCreateEvent(&record->ready) != ACL_SUCCESS ||
        real_aclrtMemcpyAsync(record->host_status, sizeof(uint32_t),
                             device_status, sizeof(uint32_t),
                             ACL_MEMCPY_DEVICE_TO_HOST, verifier_stream) != ACL_SUCCESS ||
        real_aclrtRecordEvent(record->ready, verifier_stream) != ACL_SUCCESS) {
        fprintf(stderr, "[SecureInfer FATAL] cannot observe authentication result\n");
        std::abort();
    }
    recovery_auth_observer* observer = nullptr;
    {
        std::lock_guard<std::mutex> guard(g_recovery_auth_lock);
        g_recovery_layer_records[layer].push_back(record);
        auto& entry = g_recovery_observers[reinterpret_cast<void*>(verifier_stream)];
        if (entry == nullptr) {
            entry = new recovery_auth_observer();
            auto* worker = entry;
            std::thread([worker]() {
                for (;;) {
                    std::shared_ptr<recovery_auth_record> item;
                    {
                        std::unique_lock<std::mutex> guard(worker->lock);
                        worker->changed.wait(guard, [worker]() {
                            return !worker->pending.empty();
                        });
                        item = worker->pending.front();
                        worker->pending.pop_front();
                    }
                    aclrtSetCurrentContext(item->context);
                    const aclError ret = real_aclrtSynchronizeEvent(item->ready);
                    const int status = ret == ACL_SUCCESS
                        ? static_cast<int>(*item->host_status) : -1;
                    // Publish the failure signal before diagnostic cleanup.
                    if (status == 0) {
                        secure_infer_record_h2d_fault_event("authentication_complete",
                            item->layer, item->tensor, item->bytes);
                        secure_infer_record_h2d_fault_event("authentication_failed",
                            item->layer, item->tensor, item->bytes);
                    }
                    secure_infer_record_h2d_fault_event("authentication_record_observed",
                        item->layer, item->tensor, item->bytes);
                    real_aclrtDestroyEvent(item->ready);
                    real_aclrtFreeHost(item->host_status);
                    item->result.set_value(status);
                }
            }).detach();
        }
        observer = entry;
    }
    {
        std::lock_guard<std::mutex> guard(observer->lock);
        observer->pending.push_back(record);
    }
    observer->changed.notify_one();
    return record;
}

struct h2d_gcm_stream_pipeline {
    // Slots and their host-side ring cursor are private to one ACL stream.
    // Keeping this lock per stream is important: a full ring on one stripe
    // must not stall submission on the other independent stripes.
    std::mutex host_lock;
    aclrtStream stream = nullptr;
    aclrtStream verify_stream = nullptr;
    aclrtStream xor_stream = nullptr;
    std::vector<h2d_gcm_pipeline_slot> slots;
    size_t next_slot = 0;
};

static std::unordered_map<void*, std::unique_ptr<h2d_gcm_stream_pipeline>>
    g_h2d_gcm_pipelines;
static std::mutex g_h2d_gcm_pipelines_lock;

struct pending_h2d_gcm_batch_task {
    aclrtStream stream = nullptr;
    h2d_gcm_pipeline_slot* slot = nullptr;
    void* ciphertext = nullptr;
    void* expected_tag = nullptr;
    void* keystream = nullptr;
    void* plaintext_output = nullptr;
    void* status_output = nullptr;
    void* aad = nullptr;
    void* key = nullptr;
    void* hash_subkey = nullptr;
    size_t count = 0;
    size_t aligned_bytes = 0;
    uint32_t iv_words[4] = {};
    uint64_t transfer_ticket = 0;
    uint64_t total_start_us = 0;
    uint64_t device_start_us = 0;
    uint64_t cpu_crypto_us = 0;
    uint64_t copy_enqueue_us = 0;
    bool cpu_prediction_hit = false;
    int ticket_backend = -1;
    bool ghash_critical = false;
};

struct h2d_gcm_batch_context {
    unsigned depth = 0;
    std::vector<pending_h2d_gcm_batch_task> pending;
};

static thread_local h2d_gcm_batch_context g_h2d_gcm_batch_context;

static std::mutex g_gcm_h2d_path_stats_lock;
static FILE* g_gcm_h2d_path_stats_file = nullptr;
static char g_gcm_h2d_path_stats_run_id[128] = "";
static uint64_t g_gcm_h2d_path_stats_rows_since_flush = 0;

// A separate per-transfer ledger makes the measured H2D rows auditable.  It
// deliberately records mutually-exclusive paths instead of trying to infer
// them from asynchronous producer counters.
static void record_gcm_h2d_path_stats(const h2d_gcm_pipeline_slot& slot)
{
    if (!slot.profile_precise || slot.timing.run_id[0] == '\0') return;
    const char* path = std::getenv("PIPE_GCM_H2D_PATH_STATS_PATH");
    if (path == nullptr || path[0] == '\0') return;

    std::lock_guard<std::mutex> lock(g_gcm_h2d_path_stats_lock);
    if (g_gcm_h2d_path_stats_file == nullptr ||
        strcmp(g_gcm_h2d_path_stats_run_id, slot.timing.run_id) != 0) {
        if (g_gcm_h2d_path_stats_file != nullptr) {
            fclose(g_gcm_h2d_path_stats_file);
            g_gcm_h2d_path_stats_file = nullptr;
        }
        g_gcm_h2d_path_stats_file = fopen(path, "w");
        if (g_gcm_h2d_path_stats_file == nullptr) {
            fprintf(stderr,
                    "[SecureInfer WARN] cannot open GCM H2D path ledger '%s': %s\n",
                    path, strerror(errno));
            return;
        }
        snprintf(g_gcm_h2d_path_stats_run_id,
                 sizeof(g_gcm_h2d_path_stats_run_id), "%s", slot.timing.run_id);
        g_gcm_h2d_path_stats_rows_since_flush = 0;
        fprintf(g_gcm_h2d_path_stats_file,
                "run_id,bytes,ticket_xor,on_demand_fallback,cpu_precrypto_hit,"
                "cpu_inline_fallback,ticket_backend,tamper_injected,"
                "auth_bypassed\n");
    }
    fprintf(g_gcm_h2d_path_stats_file, "%s,%zu,%d,%d,%d,%d,%d,%d,%d\n",
            slot.timing.run_id, slot.profile_count,
            slot.profile_gcm_ticket_xor ? 1 : 0,
            slot.profile_gcm_on_demand_fallback ? 1 : 0,
            slot.profile_cpu_prediction_hit ? 1 : 0,
            slot.profile_cpu_inline_fallback ? 1 : 0,
            slot.profile_ticket_backend,
            slot.profile_tamper_injected ? 1 : 0,
            slot.profile_auth_bypassed ? 1 : 0);
    if (++g_gcm_h2d_path_stats_rows_since_flush >= profile_flush_rows()) {
        fflush(g_gcm_h2d_path_stats_file);
        g_gcm_h2d_path_stats_rows_since_flush = 0;
    }
}

static bool h2d_gcm_pipeline_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_H2D_PIPELINE");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] asynchronous verified GCM H2D pipeline: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

// Insecure by construction and therefore limited to the precise profiling
// window.  The control keeps the same encrypted bytes, ticket keystream,
// AICPU launch count, output buffer, stream ordering, and D2D publication,
// but omits GHASH/tag verification so its E2E delta against production GCM
// attributes the authentication work exposed on the request path.
static bool gcm_profile_trust_tag_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_PROFILE_TRUST_TAG");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer TEST ONLY] precise-profile GCM authentication "
                    "bypass enabled; results are not security-valid\n");
        }
    }
    return enabled != 0;
}

// The fused AICPU kernels compute and compare the GCM tag before entering
// their plaintext loop.  They can therefore place verified plaintext in the
// final KV address directly; an invalid tag writes only zeros.  The layer or
// request completion event is still recorded after the kernel, preserving
// verify-before-use while removing one full-size temporary-output D2D copy.
// Keep an explicit switch for paired evaluation and compatibility fallback.
static bool gcm_direct_verified_release_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_DIRECT_VERIFIED_RELEASE");
        if (env != nullptr) {
            enabled = std::atoi(env) != 0 ? 1 : 0;
        } else {
            // In the verified-layer pipeline the fused kernel already gates
            // visibility on a successful tag comparison.  Direct placement
            // therefore removes only a redundant temporary-output D2D copy;
            // it neither weakens verify-before-use nor changes the legacy
            // request-atomic path.
            const char* layerwise = std::getenv("PIPE_GCM_H2D_LAYERWISE");
            enabled = (layerwise != nullptr && std::atoi(layerwise) != 0)
                ? 1 : 0;
        }
        fprintf(stderr, "[SecureInfer] direct verified GCM KV placement: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

// Batch a few independent authenticated releases from one logical layer into
// one AICPU launch. This changes only command granularity: every transfer
// keeps its own IV, tag, AAD, status, and final destination.
static bool gcm_h2d_layer_batch_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_H2D_LAYER_BATCH");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] layer-batched GCM H2D release: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static bool gcm_h2d_coalesced_aad_enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("PIPE_GCM_H2D_COALESCE_AAD");
        const bool value = env == nullptr || std::atoi(env) != 0;
        fprintf(stderr, "[SecureInfer] coalesced H2D ciphertext/tag/AAD DMA: %s\n",
                value ? "enabled" : "disabled");
        return value;
    }();
    return enabled;
}

// Split a ticket-hit release across heterogeneous engines.  GHASH and tag
// comparison run on AICPU while the payload-independent XOR runs on AIV into
// a request-private destination.  The parent stream joins both and applies a
// fail-closed gate before the layer event, preserving strict V_i -> C_i.
static bool gcm_parallel_verify_xor_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_PARALLEL_VERIFY_XOR");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr,
                "[SecureInfer] heterogeneous parallel GCM verify/XOR: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

// Experimental Fastrack-style control.  This mode is disabled by default and
// is only intended for a separately measured eager-use comparison.  It lets
// the parent H2D stream publish the XOR result before the independent
// verify-only stream completes.  The resulting model output is tentative:
// this experimental mode does not implement Fastrack's complete result
// notification or recovery protocol.
static bool gcm_fasttrack_eager_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_FASTTRACK_EAGER");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        if (enabled && !gcm_parallel_verify_xor_enabled()) {
            fprintf(stderr,
                    "[SecureInfer WARN] Fastrack-style eager mode requires "
                    "PIPE_GCM_PARALLEL_VERIFY_XOR=1; disabling\n");
            enabled = 0;
        }
        fprintf(stderr, "[SecureInfer] Fastrack-style eager use: %s\n",
                enabled ? "enabled (experimental)" : "disabled");
    }
    return enabled != 0;
}

static size_t h2d_gcm_pipeline_slot_count()
{
    static size_t slots = 0;
    if (slots == 0) {
        const char* env = std::getenv("PIPE_GCM_H2D_PIPELINE_SLOTS");
        // Treat this as a *global* H2D staging budget.  The old implementation
        // allocated the configured count independently on every stripe
        // stream.  Three streams with 1.25-MiB KV slices therefore created 96
        // input/output/host staging triples in the target request, and their
        // allocation cost erased the GHASH parallelism benefit.  Divide the
        // budget across the configured independent streams while retaining at
        // least two slots per stream for copy/verify overlap.
        long parsed = env ? std::atol(env) : 32;
        const long total_budget = std::max(2L, std::min(256L, parsed));
        long stream_count = 1;
        const char* slice_env = std::getenv("PIPE_GCM_H2D_SLICE_STREAMS");
        if (slice_env != nullptr) {
            // Python resolves "auto" to the measured three-stream policy.
            // Mirror that value here so the slot count remains a *global*
            // budget instead of accidentally allocating the full budget on
            // each stream because atol("auto") is zero.
            const long parsed_streams = strcmp(slice_env, "auto") == 0
                ? 3L : std::atol(slice_env);
            stream_count = std::max(stream_count, parsed_streams);
        }
        const char* admission_env =
            std::getenv("PIPE_GCM_H2D_ASYNC_ADMISSION");
        if (admission_env != nullptr && std::atoi(admission_env) != 0) {
            const char* worker_env =
                std::getenv("PIPE_GCM_H2D_SUBMIT_WORKERS");
            if (worker_env != nullptr) {
                stream_count = std::max(stream_count, std::atol(worker_env));
            }
        }
        stream_count = std::max(1L, std::min(16L, stream_count));
        slots = static_cast<size_t>(std::max(
            2L, (total_budget + stream_count - 1) / stream_count));
    }
    return slots;
}

static void record_completed_h2d_gcm_slot(h2d_gcm_pipeline_slot& slot)
{
    if (!slot.profile_pending) return;

    const uint64_t host_device_us = secure_infer_now_us() -
        slot.profile_host_device_start_us;
    if (slot.profile_precise) {
        bool valid = slot.timing.records_ok;
        const uint64_t h2d_device_us = inline_event_elapsed_us(
            slot.timing.pipeline_start, slot.timing.kernel_start, valid);
        const uint64_t kernel_execute_us = inline_event_elapsed_us(
            slot.timing.kernel_start, slot.timing.kernel_end, valid);
        const uint64_t d2d_device_us = inline_event_elapsed_us(
            slot.timing.kernel_end, slot.timing.pipeline_end, valid);
        const uint64_t device_total_us = inline_event_elapsed_us(
            slot.timing.pipeline_start, slot.timing.pipeline_end, valid);
        profile_record_precise(slot.timing.run_id,
                               "H2D",
                               slot.profile_count,
                               slot.profile_aligned_bytes,
                               secure_infer_now_us() - slot.profile_total_start_us,
                               slot.profile_cpu_crypto_us,
                               valid ? device_total_us : host_device_us,
                               0,
                               slot.profile_copy_enqueue_us,
                               slot.profile_kernel_enqueue_us,
                               h2d_device_us,
                               kernel_execute_us,
                               d2d_device_us,
                               device_total_us,
                               valid,
                               0,
                               slot.profile_cpu_prediction_hit ? 1 : 0);
        record_gcm_h2d_path_stats(slot);
    } else {
        profile_record("H2D",
                       slot.profile_count,
                       slot.profile_aligned_bytes,
                       secure_infer_now_us() - slot.profile_total_start_us,
                       slot.profile_cpu_crypto_us,
                       host_device_us,
                       0,
                       slot.profile_copy_enqueue_us,
                       slot.profile_kernel_enqueue_us);
    }
    slot.profile_pending = false;
}

static void retire_h2d_gcm_pipeline_slot(h2d_gcm_pipeline_slot& slot)
{
    if (slot.authentication_observed.valid()) {
        if (slot.authentication_observed.get() < 0) std::abort();
        slot.authentication_observed = {};
    }
    if (!recovery_event_monitor_enabled() &&
        slot.profile_tamper_injected && slot.profile_auth_status != nullptr &&
        gcm_fasttrack_eager_enabled()) {
        uint32_t auth_status_host = UINT32_MAX;
        const aclError status_ret = real_aclrtMemcpy(
            &auth_status_host, sizeof(auth_status_host),
            slot.profile_auth_status, sizeof(auth_status_host),
            ACL_MEMCPY_DEVICE_TO_HOST);
        std::vector<unsigned char> recovered(slot.profile_count);
        const aclError plain_ret = real_aclrtMemcpy(
            recovered.data(), recovered.size(), slot.profile_plaintext_output,
            recovered.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        size_t mismatched_bytes = recovered.size();
        if (plain_ret == ACL_SUCCESS &&
            slot.profile_expected_plaintext != nullptr) {
            mismatched_bytes = 0;
            const auto* expected = static_cast<const unsigned char*>(
                slot.profile_expected_plaintext);
            for (size_t index = 0; index < recovered.size(); ++index) {
                mismatched_bytes += recovered[index] != expected[index];
            }
        }
        if (status_ret == ACL_SUCCESS && auth_status_host == 0) {
            secure_infer_record_h2d_fault_event(
                "authentication_complete", slot.profile_logical_layer,
                slot.profile_tensor_kind, slot.profile_count);
            secure_infer_record_h2d_fault_event(
                "authentication_failed", slot.profile_logical_layer,
                slot.profile_tensor_kind, slot.profile_count);
        }
        fprintf(stderr,
                "[SecureInfer TEST] eager tampered KV authentication completed: "
                "layer=%u tensor=%u auth=%u status_ret=%d plain_ret=%d "
                "mismatched_bytes=%zu/%zu\n",
                slot.profile_logical_layer, slot.profile_tensor_kind,
                auth_status_host, static_cast<int>(status_ret),
                static_cast<int>(plain_ret), mismatched_bytes,
                recovered.size());
    }
    record_completed_h2d_gcm_slot(slot);
    if (slot.keystream_ticket != 0) {
        secure_infer_gcm_finish_keystream_ticket(slot.keystream_ticket);
        slot.keystream_ticket = 0;
    }
    if (slot.cpu_cipher_ticket != 0) {
        secure_infer_gcm_release_handed_off_cpu_ciphertext(
            slot.cpu_cipher_ticket);
        slot.cpu_cipher_ticket = 0;
    }
    if (slot.ghash_critical) {
        secure_infer_gcm_ghash_critical_exit();
        slot.ghash_critical = false;
    }
    slot.profile_auth_status = nullptr;
    slot.profile_plaintext_output = nullptr;
    slot.profile_expected_plaintext = nullptr;
    slot.profile_logical_layer = UINT32_MAX;
    slot.profile_tensor_kind = UINT32_MAX;
    slot.used = false;
}

static bool flush_h2d_gcm_batch_tasks()
{
    auto& pending = g_h2d_gcm_batch_context.pending;
    if (pending.empty()) return true;
    constexpr size_t kBatchMax = 4;
    const auto& first = pending.front();
    if (pending.size() > kBatchMax || first.stream == nullptr ||
        first.slot == nullptr || g_launch_aicpu_gcm_verify_xor_batch == nullptr) {
        return false;
    }

    void* ciphertexts[kBatchMax] = {};
    void* expected_tags[kBatchMax] = {};
    void* keystreams[kBatchMax] = {};
    void* plaintext_outputs[kBatchMax] = {};
    void* status_outputs[kBatchMax] = {};
    void* aads[kBatchMax] = {};
    uint32_t iv_words[kBatchMax * 4] = {};
    for (size_t index = 0; index < pending.size(); ++index) {
        const auto& task = pending[index];
        if (task.stream != first.stream || task.key != first.key ||
            task.hash_subkey != first.hash_subkey ||
            task.aligned_bytes != first.aligned_bytes) {
            return false;
        }
        ciphertexts[index] = task.ciphertext;
        expected_tags[index] = task.expected_tag;
        keystreams[index] = task.keystream;
        plaintext_outputs[index] = task.plaintext_output;
        status_outputs[index] = task.status_output;
        aads[index] = task.aad;
        memcpy(iv_words + index * 4, task.iv_words, sizeof(task.iv_words));
    }

    const uint64_t launch_start_us = secure_infer_now_us();
    const int launch_ret = g_launch_aicpu_gcm_verify_xor_batch(
        first.stream, first.key, first.hash_subkey, ciphertexts,
        expected_tags, keystreams, plaintext_outputs, status_outputs, aads,
        first.aligned_bytes,
        secure_infer_gcm_aad_enabled() ? gcm_aad_length : 0,
        iv_words, static_cast<uint32_t>(pending.size()));
    const uint64_t launch_us = secure_infer_now_us() - launch_start_us;
    if (launch_ret != 0) {
        fprintf(stderr,
                "[SecureInfer FATAL] batched GCM H2D release failed ret=%d count=%zu\n",
                launch_ret, pending.size());
        return false;
    }
    g_h2d_gcm_batch_launches.fetch_add(1, std::memory_order_relaxed);
    g_h2d_gcm_batched_records.fetch_add(
        pending.size(), std::memory_order_relaxed);
    if (!g_h2d_gcm_batch_reported.exchange(true,
                                            std::memory_order_relaxed)) {
        fprintf(stderr,
                "[SecureInfer] layer-batched H2D GCM active: first_count=%zu "
                "record_bytes=%llu\n",
                pending.size(),
                static_cast<unsigned long long>(first.aligned_bytes));
    }

    for (auto& task : pending) {
        auto& slot = *task.slot;
        slot.profile_pending = true;
        slot.profile_precise = false;
        slot.profile_total_start_us = task.total_start_us;
        slot.profile_cpu_crypto_us = task.cpu_crypto_us;
        slot.profile_host_device_start_us = task.device_start_us;
        slot.profile_copy_enqueue_us = task.copy_enqueue_us;
        slot.profile_kernel_enqueue_us = launch_us / pending.size();
        slot.profile_count = task.count;
        slot.profile_aligned_bytes = task.aligned_bytes;
        slot.profile_cpu_prediction_hit = task.cpu_prediction_hit;
        slot.profile_gcm_ticket_xor = true;
        slot.profile_ticket_backend = task.ticket_backend;
        slot.profile_gcm_on_demand_fallback = false;
        slot.profile_cpu_inline_fallback = !task.cpu_prediction_hit;
        slot.profile_tamper_injected = false;
        slot.profile_auth_bypassed = false;
        slot.keystream_ticket = task.transfer_ticket;
        slot.ghash_critical = task.ghash_critical;
        const aclError event_ret = real_aclrtRecordEvent(slot.done, task.stream);
        if (event_ret != ACL_SUCCESS) {
            fprintf(stderr,
                    "[SecureInfer FATAL] cannot record batched GCM slot event ret=%d\n",
                    static_cast<int>(event_ret));
            return false;
        }
        slot.used = true;
    }
    pending.clear();
    return true;
}

static bool enqueue_h2d_gcm_batch_task(pending_h2d_gcm_batch_task task)
{
    auto& context = g_h2d_gcm_batch_context;
    if (context.depth == 0) return false;
    if (!context.pending.empty()) {
        const auto& first = context.pending.front();
        if (first.stream != task.stream || first.key != task.key ||
            first.hash_subkey != task.hash_subkey ||
            first.aligned_bytes != task.aligned_bytes) {
            if (!flush_h2d_gcm_batch_tasks()) return false;
        }
    }
    context.pending.push_back(task);
    if (context.pending.size() == 4 && !flush_h2d_gcm_batch_tasks()) {
        return false;
    }
    return true;
}

extern "C" __attribute__((visibility("default"))) int
secure_infer_h2d_layer_group_begin(size_t expected_tasks)
{
    (void)expected_tasks;
    if (!gcm_h2d_layer_batch_enabled()) return 0;
    auto& context = g_h2d_gcm_batch_context;
    if (context.depth++ == 0) context.pending.clear();
    return 0;
}

extern "C" __attribute__((visibility("default"))) int
secure_infer_h2d_layer_group_end()
{
    if (!gcm_h2d_layer_batch_enabled()) return 0;
    auto& context = g_h2d_gcm_batch_context;
    if (context.depth == 0) return -1;
    if (--context.depth != 0) return 0;
    const bool ok = flush_h2d_gcm_batch_tasks();
    context.pending.clear();
    return ok ? 0 : -1;
}

static bool ensure_h2d_gcm_pipeline_slot(h2d_gcm_pipeline_slot& slot,
                                         size_t bytes,
                                         aclrtStream stream)
{
    if (slot.used) {
        const uint64_t wait_start_us = secure_infer_now_us();
        aclError wait_ret = real_aclrtSynchronizeEvent(slot.done);
        if (wait_ret != ACL_SUCCESS) {
            // A per-slot wait is deliberately the only fallback: synchronizing
            // the full stream here would defeat the bounded pipeline.
            wait_ret = real_aclrtSynchronizeStream(stream);
        }
        if (wait_ret != ACL_SUCCESS) {
            fprintf(stderr,
                    "[SecureInfer ERROR] cannot retire GCM H2D pipeline slot ret=%d\n",
                    static_cast<int>(wait_ret));
            return false;
        }
        const uint64_t wait_us = secure_infer_now_us() - wait_start_us;
        if (wait_us >= 10000 &&
            std::getenv("PIPE_DIAG_SLOT_WAIT") != nullptr) {
            fprintf(stderr,
                    "[SecureInfer DIAG] H2D slot recycle wait_us=%llu "
                    "stream=%p bytes=%zu\n",
                    static_cast<unsigned long long>(wait_us), stream, bytes);
        }
        retire_h2d_gcm_pipeline_slot(slot);
    }

    if (slot.bytes >= bytes && slot.temp_in != nullptr &&
        slot.temp_out != nullptr && slot.cipher_host != nullptr) {
        return true;
    }

    if (slot.temp_in != nullptr) real_aclrtFree(slot.temp_in);
    if (slot.temp_out != nullptr) real_aclrtFree(slot.temp_out);
    if (slot.cipher_host != nullptr) real_aclrtFreeHost(slot.cipher_host);
    slot.temp_in = nullptr;
    slot.temp_out = nullptr;
    slot.cipher_host = nullptr;
    slot.bytes = 0;

    constexpr size_t kMaxIntegrityBytes = gcm_metadata_slab_bytes;
    // The coalesced path appends the authenticated descriptor to the same
    // host DMA as ciphertext and tag.  Keep the slot allocation identical to
    // the size used by inline_h2d_encrypt_commit(); otherwise every pinned
    // CPU-preencrypted handoff overwrites the next host allocation by one AAD
    // slab (64 bytes), and the overwrite becomes visible only when a ticketed
    // multi-stream pipeline is active.
    const size_t cipher_bytes = bytes + 32 +
        (secure_infer_gcm_aad_enabled() && gcm_h2d_coalesced_aad_enabled()
             ? gcm_aad_length : 0);  // GCM tag, mask gap, optional AAD
    if (real_aclrtMalloc(&slot.temp_in, bytes + kMaxIntegrityBytes) != ACL_SUCCESS ||
        real_aclrtMalloc(&slot.temp_out, bytes + kMaxIntegrityBytes) != ACL_SUCCESS ||
        real_aclrtMallocHost(&slot.cipher_host, cipher_bytes) != ACL_SUCCESS ||
        slot.temp_in == nullptr || slot.temp_out == nullptr ||
        slot.cipher_host == nullptr) {
        if (slot.temp_in != nullptr) real_aclrtFree(slot.temp_in);
        if (slot.temp_out != nullptr) real_aclrtFree(slot.temp_out);
        if (slot.cipher_host != nullptr) real_aclrtFreeHost(slot.cipher_host);
        slot.temp_in = nullptr;
        slot.temp_out = nullptr;
        slot.cipher_host = nullptr;
        return false;
    }
    slot.bytes = bytes;
    return true;
}

static h2d_gcm_pipeline_slot* acquire_h2d_gcm_pipeline_slot(
    aclrtStream stream, size_t bytes, std::unique_lock<std::mutex>& slot_lock)
{
    h2d_gcm_stream_pipeline* pipeline_ptr = nullptr;
    {
        std::lock_guard<std::mutex> map_lock(g_h2d_gcm_pipelines_lock);
        auto& entry = g_h2d_gcm_pipelines[reinterpret_cast<void*>(stream)];
    if (!entry) {
        auto pipeline = std::make_unique<h2d_gcm_stream_pipeline>();
        pipeline->stream = stream;
        if (gcm_parallel_verify_xor_enabled()) {
            // Both controls use the same verifier/XOR stream topology.  The
            // only experimental difference is whether the parent stream
            // waits for verify_done before exposing the XOR output.
            const bool verify_ok =
                real_aclrtCreateStream(&pipeline->verify_stream) == ACL_SUCCESS &&
                pipeline->verify_stream != nullptr;
            const bool xor_ok =
                real_aclrtCreateStream(&pipeline->xor_stream) == ACL_SUCCESS &&
                pipeline->xor_stream != nullptr;
            if (!verify_ok || !xor_ok) {
                fprintf(stderr,
                        "[SecureInfer WARN] cannot create parallel GCM streams; "
                        "using fused verified release\n");
                pipeline->verify_stream = nullptr;
                pipeline->xor_stream = nullptr;
            }
        }
        pipeline->slots.resize(h2d_gcm_pipeline_slot_count());
        for (auto& slot : pipeline->slots) {
            if (real_aclrtCreateEvent(&slot.done) != ACL_SUCCESS ||
                slot.done == nullptr) {
                fprintf(stderr,
                        "[SecureInfer WARN] cannot create GCM H2D pipeline event; "
                        "using serial compatibility path\n");
                return nullptr;
            }
            slot.verify_stream = pipeline->verify_stream;
            slot.xor_stream = pipeline->xor_stream;
            if (pipeline->xor_stream != nullptr) {
                const bool events_ok =
                    real_aclrtCreateEvent(&slot.fork_ready) == ACL_SUCCESS &&
                    slot.fork_ready != nullptr &&
                    real_aclrtCreateEvent(&slot.xor_done) == ACL_SUCCESS &&
                    slot.xor_done != nullptr &&
                    real_aclrtCreateEvent(&slot.verify_done) == ACL_SUCCESS &&
                    slot.verify_done != nullptr;
                if (!events_ok) {
                    fprintf(stderr,
                            "[SecureInfer WARN] cannot create parallel GCM events; "
                            "using fused verified release\n");
                    pipeline->verify_stream = nullptr;
                    pipeline->xor_stream = nullptr;
                    for (auto& reset_slot : pipeline->slots) {
                        reset_slot.verify_stream = nullptr;
                        reset_slot.xor_stream = nullptr;
                    }
                }
            }
        }
        fprintf(stderr,
                "[SecureInfer] secure GCM H2D pipeline initialized: slots=%zu "
                "parallel_verify_xor=%s\n",
                pipeline->slots.size(),
                pipeline->verify_stream != nullptr ? "yes" : "no");
        entry = std::move(pipeline);
    }
        pipeline_ptr = entry.get();
    }
    // The heap allocation behind unique_ptr is stable across unordered-map
    // rehashes.  Hold this lock through enqueue and event recording so slot
    // reuse remains ordered on this stream only.
    const uint64_t lock_start_us = secure_infer_now_us();
    slot_lock = std::unique_lock<std::mutex>(pipeline_ptr->host_lock);
    const uint64_t lock_wait_us = secure_infer_now_us() - lock_start_us;
    if (lock_wait_us >= 10000 &&
        std::getenv("PIPE_DIAG_SLOT_WAIT") != nullptr) {
        fprintf(stderr,
                "[SecureInfer DIAG] H2D pipeline host-lock wait_us=%llu "
                "stream=%p bytes=%zu\n",
                static_cast<unsigned long long>(lock_wait_us), stream, bytes);
    }
    auto& slot = pipeline_ptr->slots[
        pipeline_ptr->next_slot++ % pipeline_ptr->slots.size()];
    return ensure_h2d_gcm_pipeline_slot(slot, bytes, stream) ? &slot : nullptr;
}

// A framework transfer event is recorded after all H2D slices.  Its wait may
// happen before the slot ring wraps, in which case the final few profiles
// would otherwise remain pending even though the transfer is complete.  Poll
// and retire completed slots at stream/event completion boundaries; this does
// not add a device wait and keeps profile row counts aligned with real H2D.
static void flush_completed_h2d_gcm_slots(aclrtStream only_stream = nullptr)
{
    std::lock_guard<std::mutex> map_lock(g_h2d_gcm_pipelines_lock);
    for (auto& item : g_h2d_gcm_pipelines) {
        auto& pipeline = *item.second;
        if (only_stream != nullptr && pipeline.stream != only_stream) continue;
        std::lock_guard<std::mutex> pipeline_lock(pipeline.host_lock);
        for (auto& slot : pipeline.slots) {
            if (!slot.used) continue;
            aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            if (aclrtQueryEventStatus(slot.done, &status) == ACL_SUCCESS &&
                status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                retire_h2d_gcm_pipeline_slot(slot);
            }
        }
    }
}

// Framework event polling does not necessarily call the interposed ACL
// synchronize symbols.  Expose a cheap query-only flush so the offload
// handler can retire the final (not-yet-reused) pipeline slots once its
// framework-visible load event is complete.  This changes neither request
// ordering nor measured latency; it only makes the per-copy ledger complete.
extern "C" void secure_infer_flush_completed_h2d_profile()
{
    flush_completed_h2d_gcm_slots();
    // The harness terminates the serving process after each isolated run.
    // Persist a partial final I/O batch here instead of relying on C stdio
    // teardown after SIGTERM; otherwise the last few completed rows disappear
    // even though their copies and authentication have finished.
    {
        std::lock_guard<std::mutex> profile_lock(g_profile_lock);
        if (g_profile_file != nullptr) std::fflush(g_profile_file);
    }
    {
        std::lock_guard<std::mutex> path_lock(g_gcm_h2d_path_stats_lock);
        if (g_gcm_h2d_path_stats_file != nullptr) {
            std::fflush(g_gcm_h2d_path_stats_file);
        }
    }
}

// Stronger ledger drain used only after the offload worker has observed the
// framework's final logical H2D event and has no other H2D job in flight.
// At that point every slot event is causally before the completed framework
// event, so synchronize is non-blocking in the serving critical path; it only
// avoids a driver event-query visibility race for the last ring entries.
extern "C" void secure_infer_drain_h2d_profile_after_job()
{
    {
        std::lock_guard<std::mutex> map_lock(g_h2d_gcm_pipelines_lock);
        for (auto& item : g_h2d_gcm_pipelines) {
            std::lock_guard<std::mutex> pipeline_lock(item.second->host_lock);
            for (auto& slot : item.second->slots) {
                if (!slot.used) continue;
                if (real_aclrtSynchronizeEvent(slot.done) == ACL_SUCCESS) {
                    retire_h2d_gcm_pipeline_slot(slot);
                }
            }
        }
    }
    {
        std::lock_guard<std::mutex> profile_lock(g_profile_lock);
        if (g_profile_file != nullptr) std::fflush(g_profile_file);
    }
    {
        std::lock_guard<std::mutex> path_lock(g_gcm_h2d_path_stats_lock);
        if (g_gcm_h2d_path_stats_file != nullptr) {
            std::fflush(g_gcm_h2d_path_stats_file);
        }
    }
}

// Test-only fault injection for the *measured* external-KV reload.  The bit
// flip happens after CPU AES-GCM has produced a valid ciphertext/tag and
// before the ciphertext DMA, thereby exercising the same untrusted-link
// threat boundary as a corrupted cache entry.  It is intentionally opt-in
// and one-shot; production never sets this environment variable.
static uint32_t secure_infer_gcm_aad_u32(const unsigned char* aad, size_t offset)
{
    return (static_cast<uint32_t>(aad[offset]) << 24) |
           (static_cast<uint32_t>(aad[offset + 1]) << 16) |
           (static_cast<uint32_t>(aad[offset + 2]) << 8) |
           static_cast<uint32_t>(aad[offset + 3]);
}

static void secure_infer_record_h2d_fault_event(const char* phase,
                                            uint32_t layer,
                                            uint32_t tensor,
                                            size_t bytes)
{
    if (std::strcmp(phase, "authentication_failed") == 0) {
        const char* strict_retry = std::getenv("PIPE_GCM_TAMPER_STRICT_RETRY");
        const bool recovered_before_use =
            strict_retry != nullptr && std::atoi(strict_retry) != 0;
        const char* flag_path = std::getenv("PIPE_GCM_FAULT_FLAG_PATH");
        if (!recovered_before_use && flag_path != nullptr && *flag_path != '\0') {
            FILE* flag_file = std::fopen(flag_path, "w");
            if (flag_file != nullptr) {
                std::fprintf(flag_file, "%u,%u,%zu\n", layer, tensor, bytes);
                std::fclose(flag_file);
            }
        }
    }
    const char* path = std::getenv("PIPE_GCM_FAULT_EVENT_PATH");
    if (path == nullptr || *path == '\0') return;
    static std::mutex event_lock;
    const uint64_t realtime_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::lock_guard<std::mutex> lock(event_lock);
    FILE* event_file = std::fopen(path, "a");
    if (event_file == nullptr) return;
    std::fprintf(event_file,
                 "{\"phase\":\"%s\",\"realtime_ns\":%llu,"
                 "\"layer\":%u,\"tensor\":%u,\"bytes\":%zu}\n",
                 phase, static_cast<unsigned long long>(realtime_ns),
                 layer, tensor, bytes);
    std::fclose(event_file);
}

static bool secure_infer_maybe_tamper_profiled_gcm_h2d(
    bool precise_profile_active, const unsigned char* gcm_aad)
{
    static int enabled = -1;
    static uint64_t target_ordinal = 1;
    static std::atomic<uint64_t> profiled_ordinal{0};
    static std::atomic<uint64_t> logical_match_ordinal{0};
    static std::atomic<bool> arm_state{false};
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_TAMPER_H2D");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        const char* nth_env = std::getenv("PIPE_GCM_TAMPER_H2D_NTH");
        if (nth_env != nullptr) {
            const unsigned long long parsed =
                std::strtoull(nth_env, nullptr, 10);
            if (parsed > 0) target_ordinal = static_cast<uint64_t>(parsed);
        }
    }
    if (enabled == 0 || !precise_profile_active) return false;
    // A complete-recovery run first records an unmodified reference request
    // and arms fault injection only for the following request.  Leaving this
    // variable unset preserves the standalone integrity-test behavior.
    if (const char* arm_path = std::getenv("PIPE_GCM_TAMPER_ARM_PATH")) {
        const bool armed = *arm_path != '\0' && ::access(arm_path, F_OK) == 0;
        if (!armed) {
            // A new arm-file epoch starts a fresh one-shot injection.  This
            // lets a single server process run repeated paired samples while
            // preserving one fault per attacked request.
            arm_state.store(false, std::memory_order_release);
            return false;
        }
        bool was_armed = false;
        if (arm_state.compare_exchange_strong(
                was_armed, true, std::memory_order_acq_rel)) {
            profiled_ordinal.store(0, std::memory_order_release);
            logical_match_ordinal.store(0, std::memory_order_release);
        }
    }
    const char* layer_env = std::getenv("PIPE_GCM_TAMPER_H2D_LAYER");
    if (layer_env != nullptr && gcm_aad != nullptr) {
        const uint32_t layer = secure_infer_gcm_aad_u32(gcm_aad, 8);
        const uint32_t target_layer = static_cast<uint32_t>(
            std::strtoul(layer_env, nullptr, 10));
        const char* tensor_env = std::getenv("PIPE_GCM_TAMPER_H2D_TENSOR");
        const uint32_t tensor = static_cast<uint32_t>(gcm_aad[6]);
        const uint32_t target_tensor = tensor_env != nullptr
            ? static_cast<uint32_t>(std::strtoul(tensor_env, nullptr, 10))
            : tensor;
        if (layer != target_layer || tensor != target_tensor) return false;
        uint64_t target_match = 1;
        const char* match_env =
            std::getenv("PIPE_GCM_TAMPER_H2D_MATCH_ORDINAL");
        if (match_env != nullptr) {
            const unsigned long long parsed =
                std::strtoull(match_env, nullptr, 10);
            if (parsed > 0) target_match = static_cast<uint64_t>(parsed);
        }
        const uint64_t match =
            logical_match_ordinal.fetch_add(1, std::memory_order_acq_rel) + 1;
        return match == target_match;
    }
    const uint64_t ordinal =
        profiled_ordinal.fetch_add(1, std::memory_order_acq_rel) + 1;
    return ordinal == target_ordinal;
}

static bool secure_infer_allow_eager_tampered_kv()
{
    const char* env = std::getenv("PIPE_GCM_TAMPER_ALLOW_EAGER_USE");
    return gcm_fasttrack_eager_enabled() && env != nullptr &&
        std::atoi(env) != 0;
}

static aclError inline_h2d_encrypt_commit(void* dst,
                                          size_t destMax,
                                          const void* src,
                                          size_t count,
                                          aclrtStream stream,
                                          const char* reason)
{
    uint64_t total_start = secure_infer_now_us();
    const bool h2d_gcm_enabled = secure_infer_h2d_gcm_enabled();

    // Diagnostic plain baseline: bypass *this* inline H2D crypto path as
    // well.  Previously PIPE_SERIAL_BYPASS_CRYPTO only affected older paths,
    // so a purported plain run still performed CPU AES and NPU AES here.
    const char* bypass_crypto = std::getenv("PIPE_SERIAL_BYPASS_CRYPTO");
    if (bypass_crypto != nullptr && std::atoi(bypass_crypto) != 0) {
        const uint64_t copy_start = secure_infer_now_us();
        const aclError ret = real_aclrtMemcpyAsync(
            dst, destMax, src, count, ACL_MEMCPY_HOST_TO_DEVICE, stream);
        // Evaluation-only common timing boundary.  CTR and the compatibility
        // GCM path already wait before recycling their per-slice staging
        // buffers; the native-copy baseline does not.  Without this explicit
        // fence, a "serial" plain/CTR/GCM comparison mixes an asynchronous
        // copy baseline with synchronous crypto paths.  It is opt-in so
        // normal serving keeps the original native asynchronous H2D path.
        const char* sync_every_slice =
            std::getenv("PIPE_H2D_SYNC_EVERY_SLICE");
        if (ret == ACL_SUCCESS && sync_every_slice != nullptr &&
            std::atoi(sync_every_slice) != 0) {
            const aclError sync_ret = real_aclrtSynchronizeStream(stream);
            if (sync_ret != ACL_SUCCESS) return sync_ret;
        }
        // Keep the diagnostic plain baseline visible to the same reload
        // validator as GCM.  This records enqueue cost only: adding a stream
        // synchronization here would itself corrupt the baseline we want to
        // measure.  It is intentionally restricted to the explicit bypass.
        if (ret == ACL_SUCCESS) {
            const uint64_t enqueue_us = secure_infer_now_us() - copy_start;
            profile_record("H2D", count, count, enqueue_us, 0, 0, 0,
                           enqueue_us, 0);
        }
        return ret;
    }

    if (m_ctx_metadata.empty()) {
        fprintf(stderr,
                "[SecureInfer] ⚠️ %s: 缺少加密元数据，回退 plaintext H2D: src=%p, size=%zu\n",
                reason, src, count);
        return real_aclrtMemcpyAsync(dst, destMax, src, count,
                                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }

    auto& meta = *m_ctx_metadata.begin()->second;
    // Historically this lock covered the complete CPU-encrypt -> H2D ->
    // verify submission sequence.  A full ring on one stripe then blocked
    // every other stripe and the background keystream producer.  Keep the
    // legacy scratch fallback serialized, but use a per-stream lock for the
    // bounded GCM pipeline below.
    std::unique_lock<std::mutex> legacy_inline_lock(g_inline_h2d_lock,
                                                    std::defer_lock);

    size_t aligned_bytes = (count + 31) & ~static_cast<size_t>(31);
    if (aligned_bytes == 0) aligned_bytes = 32;

    {
        // Device key allocation is lazy and shared.  Only that one-time
        // initialization needs the global lock; queued cryptographic work
        // itself is stream-private.
        std::lock_guard<std::mutex> init_lock(g_inline_h2d_lock);
        if (!ensure_h2d_device_crypto(meta, stream)) {
        fprintf(stderr,
                "[SecureInfer] ⚠️ %s: NPU crypto kernel/key 未就绪，回退 plaintext H2D: src=%p, size=%zu\n",
                reason, src, count);
        return real_aclrtMemcpyAsync(dst, destMax, src, count,
                                     ACL_MEMCPY_HOST_TO_DEVICE, stream);
        }
    }

    // The ticket record is retired by the slot completion event below, so a
    // ready speculative keystream can use the same bounded verified pipeline
    // as a demand-generated one.
    const bool use_gcm_async_pipeline =
        h2d_gcm_enabled && h2d_gcm_pipeline_enabled();
    h2d_gcm_pipeline_slot* gcm_pipeline_slot = nullptr;
    std::unique_lock<std::mutex> pipeline_slot_lock;
    if (use_gcm_async_pipeline) {
        gcm_pipeline_slot = acquire_h2d_gcm_pipeline_slot(
            stream, aligned_bytes, pipeline_slot_lock);
        if (gcm_pipeline_slot == nullptr) {
            fprintf(stderr,
                    "[SecureInfer WARN] GCM H2D pipeline staging unavailable; "
                    "using serial compatibility path\n");
        }
    }
    if (gcm_pipeline_slot == nullptr) {
        // The compatibility path owns global scratch buffers and therefore
        // retains the old serialization.  Production verified pipelines do
        // not take this lock past initialization.
        legacy_inline_lock.lock();
    }

    void* h2d_temp_in = nullptr;
    void* h2d_temp_out = nullptr;
    if (gcm_pipeline_slot != nullptr) {
        h2d_temp_in = gcm_pipeline_slot->temp_in;
        h2d_temp_out = gcm_pipeline_slot->temp_out;
    } else {
        if (g_inline_h2d_temp_size < aligned_bytes) {
            if (g_inline_h2d_temp_in != nullptr) real_aclrtFree(g_inline_h2d_temp_in);
            if (g_inline_h2d_temp_out != nullptr) real_aclrtFree(g_inline_h2d_temp_out);
            constexpr size_t kMaxIntegrityBytes = gcm_metadata_slab_bytes;
            assert(real_aclrtMalloc(&g_inline_h2d_temp_in,
                                    aligned_bytes + kMaxIntegrityBytes) == ACL_SUCCESS);
            // mode-0 appends one 32-byte checksum per launched AIV core.
            assert(real_aclrtMalloc(&g_inline_h2d_temp_out,
                                    aligned_bytes + kMaxIntegrityBytes) == ACL_SUCCESS);
            g_inline_h2d_temp_size = aligned_bytes;
        }
        h2d_temp_in = g_inline_h2d_temp_in;
        h2d_temp_out = g_inline_h2d_temp_out;
    }

    const size_t cipher_bytes = aligned_bytes +
        (h2d_gcm_enabled
             ? 32 + (secure_infer_gcm_aad_enabled() &&
                          gcm_h2d_coalesced_aad_enabled()
                      ? gcm_aad_length : 0)
             : 0);
    unsigned char* cipher = gcm_pipeline_slot != nullptr
        ? static_cast<unsigned char*>(gcm_pipeline_slot->cipher_host)
        : static_cast<unsigned char*>(alloc_inline_cipher_buffer(cipher_bytes));
    memset(cipher, 0, cipher_bytes);
    uint64_t transfer_ticket = secure_infer_new_transfer_ticket();
    // A history prediction may reserve a GCM ticket before the real H2D
    // arrives.  Either speculative pool (device keystream or host GCM
    // ciphertext) consumes that same reservation; the baseline keeps the
    // historical allocation behavior unchanged.
    if (h2d_gcm_enabled &&
        (secure_infer_gcm_keystream_scheduler_enabled() ||
         secure_infer_gcm_cpu_cipher_prefetch_enabled())) {
        predictor.lock();
        uint64_t reserved_ticket = 0;
        uint64_t reserved_offset = 0;
        uint64_t ignored_group = 0;
        uint64_t admission_scope = 0;
        uint64_t prediction_ns = 0;
        if (consume_transfer_prefetch_locked(src, count, &reserved_ticket,
                                             &reserved_offset, &ignored_group,
                                             &prediction_ns,
                                             &admission_scope)) {
            // GCM reserves a separate ticket per physical transfer; an
            // offset here would indicate an invariant violation rather than
            // a safe reuse of CTR state.
            if (reserved_offset != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] GCM ticket offset must be zero: ticket=%lu offset=%lu\n",
                        static_cast<unsigned long>(reserved_ticket),
                        static_cast<unsigned long>(reserved_offset));
                std::abort();
            }
            transfer_ticket = reserved_ticket;
            record_descriptor_lead_time(src, count, transfer_ticket,
                                        prediction_ns);
            if (admission_scope != 0) {
                g_gcm_active_h2d_scope.store(
                    admission_scope, std::memory_order_release);
            }
        }
        predictor.unlock();
    }
    unsigned char iv[iv_length];
    secure_infer_derive_transfer_iv(meta.init_iv, transfer_ticket, 0, iv);
    unsigned char gcm_aad[gcm_aad_length] = {};
    const bool gcm_aad_valid = !secure_infer_gcm_aad_enabled() ||
        secure_infer_build_gcm_aad(src, count, PIPE_GCM_DIRECTION_H2D,
                              transfer_ticket, 0, gcm_aad);
    if (!gcm_aad_valid) {
        fprintf(stderr,
                "[SecureInfer FATAL] cannot build H2D GCM metadata for %p/%zu\n",
                src, count);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (h2d_gcm_enabled &&
        !secure_infer_gcm_replay_reserve(transfer_ticket)) {
        fprintf(stderr,
                "[SecureInfer SECURITY] duplicate H2D transfer ticket rejected: %lu\n",
                static_cast<unsigned long>(transfer_ticket));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // A history-predicted ticket may already have CPU-produced GCM ciphertext
    // and tag.  Prefer to hand its pinned buffer directly to this H2D slot;
    // pageable records retain the established copy fallback.  A non-ready
    // record is only an optimization miss, never a host wait.
    bool cpu_prediction_hit = false;
    uint64_t background_cpu_crypto_us = 0;
    if (h2d_gcm_enabled &&
        secure_infer_gcm_cpu_cipher_prefetch_enabled()) {
        // The ordinary CPU pool used a pageable allocation and copied it into
        // this slot's pinned buffer here.  If the worker obtained a pinned
        // buffer, hand it directly to the asynchronous DMA and retain its
        // ticket in the slot until the verified publication event.  The
        // established memcpy route remains the safe fallback on allocation
        // pressure or a non-ready prediction.
        unsigned char* handed_off_cipher = nullptr;
        size_t handed_off_bytes = 0;
        if (gcm_pipeline_slot != nullptr &&
            secure_infer_gcm_handoff_ready_cpu_ciphertext(
                transfer_ticket, src, count, &handed_off_cipher,
                &handed_off_bytes, &background_cpu_crypto_us)) {
            if (handed_off_bytes != cipher_bytes) {
                fprintf(stderr,
                        "[SecureInfer FATAL] CPU ciphertext handoff size mismatch "
                        "ticket=%lu got=%zu expected=%zu\n",
                        static_cast<unsigned long>(transfer_ticket),
                        handed_off_bytes, cipher_bytes);
                std::abort();
            }
            cipher = handed_off_cipher;
            gcm_pipeline_slot->cpu_cipher_ticket = transfer_ticket;
            cpu_prediction_hit = true;
        } else {
            cpu_prediction_hit = secure_infer_gcm_take_ready_cpu_ciphertext(
                transfer_ticket, src, count, cipher, cipher_bytes,
                &background_cpu_crypto_us);
        }
    }

    uint64_t cpu_crypto_us = 0;
    if (!cpu_prediction_hit) {
        const uint64_t crypto_start = secure_infer_now_us();
        // The legacy fallback constructed and destroyed an EVP context for
        // every 256-KiB slice. Fast-fallback mode reuses one protected by the
        // existing inline lock.
        // Calls on independent H2D stripes may now progress concurrently.
        // OpenSSL contexts are not shared between them; a thread-local
        // fallback context retains the old allocation amortization safely.
        thread_local EVP_CIPHER_CTX* reusable_inline_ctx = nullptr;
        const bool reuse_inline_ctx = secure_infer_linear_fast_fallback_enabled();
        EVP_CIPHER_CTX* local_ctx = nullptr;
        if (reuse_inline_ctx) {
            if (reusable_inline_ctx == nullptr) {
                reusable_inline_ctx = real_EVP_CIPHER_CTX_new();
            }
            local_ctx = reusable_inline_ctx;
        } else {
            local_ctx = real_EVP_CIPHER_CTX_new();
        }
        assert(local_ctx != nullptr);
        int updated_len = 0;
        int final_len = 0;
        if (h2d_gcm_enabled) {
            unsigned char tag[16];
            assert(real_EVP_EncryptInit_ex(local_ctx, EVP_aes_128_gcm(), nullptr,
                                           nullptr, nullptr) == 1);
            assert(real_EVP_CIPHER_CTX_ctrl(local_ctx, EVP_CTRL_GCM_SET_IVLEN,
                                            12, nullptr) == 1);
            assert(real_EVP_EncryptInit_ex(local_ctx, nullptr, nullptr,
                                           meta.key, iv) == 1);
            if (secure_infer_gcm_aad_enabled()) {
                int aad_len = 0;
                assert(real_EVP_EncryptUpdate(
                           local_ctx, nullptr, &aad_len, gcm_aad,
                           static_cast<int>(gcm_aad_length)) == 1);
            }
            assert(real_EVP_EncryptUpdate(local_ctx, cipher, &updated_len,
                                          static_cast<const unsigned char*>(src),
                                          static_cast<int>(count)) == 1);
            assert(real_EVP_EncryptFinal_ex(local_ctx, cipher + updated_len,
                                            &final_len) == 1);
            assert(real_EVP_CIPHER_CTX_ctrl(local_ctx, EVP_CTRL_GCM_GET_TAG,
                                            16, tag) == 1);
            memcpy(cipher + aligned_bytes, tag, 16);
            // E_K(J0) is generated inside the trusted NPU domain. Never
            // expose the tag mask on the untrusted PCIe link.
            memset(cipher + aligned_bytes + 16, 0, 16);
            if (secure_infer_gcm_aad_enabled() &&
                gcm_h2d_coalesced_aad_enabled()) {
                memcpy(cipher + aligned_bytes + 32, gcm_aad,
                       gcm_aad_length);
            }
        } else {
            assert(real_EVP_EncryptInit_ex(local_ctx, EVP_aes_128_ctr(), 0,
                                           meta.key, iv) == 1);
            assert(real_EVP_EncryptUpdate(local_ctx, cipher, &updated_len,
                                          static_cast<const unsigned char*>(src),
                                          static_cast<int>(count)) == 1);
            real_EVP_EncryptFinal_ex(local_ctx, cipher + updated_len, &final_len);
        }
        if (!reuse_inline_ctx) {
            real_EVP_CIPHER_CTX_free(local_ctx);
        }
        cpu_crypto_us = secure_infer_now_us() - crypto_start;
    }

    uint64_t copy_enqueue_us = 0;
    uint64_t kernel_enqueue_us = 0;
    uint64_t device_start = secure_infer_now_us();
    inline_device_timing serial_timing;
    inline_device_timing& timing = gcm_pipeline_slot != nullptr
        ? gcm_pipeline_slot->timing : serial_timing;
    bool precise_profile_active = begin_inline_device_timing(timing, stream);

    const bool tamper_injected = h2d_gcm_enabled &&
        secure_infer_maybe_tamper_profiled_gcm_h2d(
            precise_profile_active, gcm_aad);
    if (tamper_injected) {
        // Flip either ciphertext or its transmitted tag after CPU AES-GCM.
        // The in-device tag mask remains trusted.  Both modes must yield
        // auth=0; tag-only corruption leaves the decrypted numerical input
        // unchanged and isolates recovery-management overhead.
        size_t tamper_bytes = 1;
        unsigned tamper_mask = 0x01;
        if (const char* bytes_env = std::getenv("PIPE_GCM_TAMPER_H2D_BYTES")) {
            const unsigned long long parsed =
                std::strtoull(bytes_env, nullptr, 10);
            if (parsed > 0) {
                tamper_bytes = std::min(
                    aligned_bytes, static_cast<size_t>(parsed));
            }
        }
        if (const char* mask_env = std::getenv("PIPE_GCM_TAMPER_H2D_XOR")) {
            const unsigned long parsed = std::strtoul(mask_env, nullptr, 0);
            tamper_mask = static_cast<unsigned>(parsed & 0xffU);
            if (tamper_mask == 0) tamper_mask = 0x01;
        }
        const char* target_env = std::getenv("PIPE_GCM_TAMPER_H2D_TARGET");
        const bool tamper_tag = target_env != nullptr &&
            strcasecmp(target_env, "tag") == 0;
        if (tamper_tag) tamper_bytes = std::min<size_t>(tamper_bytes, 16);
        unsigned char* tamper_target = tamper_tag
            ? cipher + aligned_bytes : cipher;
        for (size_t index = 0; index < tamper_bytes; ++index) {
            tamper_target[index] ^= static_cast<unsigned char>(tamper_mask);
        }
        const uint32_t tamper_layer = secure_infer_gcm_aad_u32(gcm_aad, 8);
        const uint32_t tamper_tensor = static_cast<uint32_t>(gcm_aad[6]);
        secure_infer_record_h2d_fault_event(
            tamper_tag ? "tag_tamper_injected" : "ciphertext_tamper_injected",
            tamper_layer, tamper_tensor, tamper_bytes);
        fprintf(stderr,
                "[SecureInfer TEST] injected GCM %s tamper for profiled H2D "
                "validation (layer=%u tensor=%u bytes=%zu xor=0x%02x)\n",
                tamper_tag ? "tag" : "ciphertext", tamper_layer,
                tamper_tensor, tamper_bytes, tamper_mask);
    }

    uint64_t copy_start = secure_infer_now_us();
    real_aclrtMemcpyAsync(h2d_temp_in,
                          aligned_bytes + gcm_metadata_slab_bytes,
                          cipher, cipher_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE, stream);
    void* h2d_aad_device = static_cast<unsigned char*>(h2d_temp_in) +
                           aligned_bytes + 32;
    if (secure_infer_gcm_aad_enabled() && !gcm_h2d_coalesced_aad_enabled()) {
        real_aclrtMemcpyAsync(h2d_aad_device, gcm_aad_length,
                              gcm_aad, gcm_aad_length,
                              ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }
    if (!h2d_gcm_enabled && aligned_bytes > count) {
        uint8_t zero_pad[32] = {0};
        real_aclrtMemcpyAsync(static_cast<uint8_t*>(h2d_temp_in) + count,
                              aligned_bytes - count,
                              zero_pad,
                              aligned_bytes - count,
                              ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }
    copy_enqueue_us += secure_infer_now_us() - copy_start;
    if (precise_profile_active &&
        real_aclrtRecordEvent(timing.kernel_start, stream) != ACL_SUCCESS) {
        timing.records_ok = false;
    }

    uint32_t iv_words[4] = {0, 0, 0, 0};
    pack_inline_iv(iv, iv_words);
    uint64_t kernel_start = secure_infer_now_us();
    uint32_t* gcm_auth_status = nullptr;
    bool gcm_ticket_pool_active = false;
    bool gcm_ghash_critical = false;
    bool gcm_precomputed_xor = false;
    bool gcm_auth_bypassed = false;
    bool gcm_direct_verified_release = false;
    bool gcm_eager_slot_done_recorded = false;
    int gcm_ticket_backend = -1;
    if (h2d_gcm_enabled) {
        // Authentication is latency-critical even when the optional ticket
        // pool is disabled.  Expose this full interval to a cooperating
        // system scheduler; ticket ownership remains conditional below.
        secure_infer_gcm_ghash_critical_enter();
        gcm_ghash_critical = true;
        if (secure_infer_gcm_keystream_scheduler_enabled()) {
            gcm_ticket_pool_active = true;
        }
        if (!secure_infer_prepare_gcm_backend(meta, stream)) {
            fprintf(stderr, "[SecureInfer FATAL] inline GCM backend unavailable\n");
            std::abort();
        }
        auto* expected_tag = static_cast<uint8_t*>(h2d_temp_in) +
            aligned_bytes;
        auto* auth_status = reinterpret_cast<uint32_t*>(expected_tag + 16);
        gcm_auth_status = auth_status;
        void* ready_keystream = nullptr;
        gcm_precomputed_xor = gcm_ticket_pool_active &&
            secure_infer_gcm_take_ready_keystream(stream, transfer_ticket,
                                              aligned_bytes,
                                              &ready_keystream);
        if (gcm_precomputed_xor) {
            gcm_ticket_backend = secure_infer_gcm_ticket_backend(transfer_ticket);
        }
        const bool aad_enabled = secure_infer_gcm_aad_enabled();
        // The profiling control deliberately ignores both the ciphertext
        // authentication and AAD authentication while retaining the same
        // metadata copy, ticket, DMA, XOR, destination, and completion
        // ordering.  Requiring AAD to be disabled changed one more variable
        // than the intended GHASH ablation and made the paired E2E result
        // harder to interpret.
        const bool request_auth_bypass = precise_profile_active &&
            gcm_profile_trust_tag_enabled();
        const char* validate_h2d_env = std::getenv("PIPE_GCM_VALIDATE_H2D");
        const bool validation_requested = validate_h2d_env != nullptr &&
            std::atoi(validate_h2d_env) != 0;
        const bool batch_release =
            gcm_precomputed_xor && gcm_pipeline_slot != nullptr &&
            gcm_h2d_layer_batch_enabled() &&
            g_h2d_gcm_batch_context.depth != 0 &&
            g_launch_aicpu_gcm_verify_xor_batch != nullptr &&
            gcm_direct_verified_release_enabled() &&
            destMax >= aligned_bytes && count == aligned_bytes &&
            h2d_gcm_pipeline_slot_count() >= 4 &&
            !precise_profile_active && !tamper_injected &&
            !request_auth_bypass && !validation_requested;
        // Keep stream order and pipeline-slot ownership simple when a layer
        // contains a record that cannot use the batched fast path (for
        // example, an on-demand ticket fallback).  Flush previously queued
        // records before submitting this record through the original path.
        if (!batch_release &&
            !g_h2d_gcm_batch_context.pending.empty() &&
            !flush_h2d_gcm_batch_tasks()) {
            fprintf(stderr,
                    "[SecureInfer FATAL] cannot flush layer-batched GCM H2D tasks\n");
            std::abort();
        }
        if (batch_release) {
            pending_h2d_gcm_batch_task task;
            task.stream = stream;
            task.slot = gcm_pipeline_slot;
            task.ciphertext = h2d_temp_in;
            task.expected_tag = expected_tag;
            task.keystream = ready_keystream;
            task.plaintext_output = dst;
            task.status_output = auth_status;
            task.aad = h2d_aad_device;
            task.key = g_dev_key;
            task.hash_subkey = meta.dev_ghash_h;
            task.count = count;
            task.aligned_bytes = aligned_bytes;
            memcpy(task.iv_words, iv_words, sizeof(iv_words));
            task.transfer_ticket = transfer_ticket;
            task.total_start_us = total_start;
            task.device_start_us = device_start;
            task.cpu_crypto_us = cpu_crypto_us;
            task.copy_enqueue_us = copy_enqueue_us;
            task.cpu_prediction_hit = cpu_prediction_hit;
            task.ticket_backend = gcm_ticket_backend;
            task.ghash_critical = gcm_ghash_critical;
            if (!enqueue_h2d_gcm_batch_task(task)) {
                fprintf(stderr,
                        "[SecureInfer FATAL] cannot enqueue layer-batched GCM H2D task\n");
                std::abort();
            }
            // group_end flushes all remaining records before Python records
            // the public layer event. Therefore Attention can never observe
            // a destination before this batch authenticates it.
            return ACL_SUCCESS;
        }
        if (request_auth_bypass) {
            // Profiling-only authentication ablation.  Preserve host
            // AES-GCM (including tag generation), the transmitted
            // ciphertext/AAD, DMA, IV, destination, and stream ordering.
            // Remove only device GHASH/tag checking: consume a prepared
            // keystream when available, otherwise execute the same AES-CTR
            // decryption needed by the production fallback path.
            void* plaintext_output = h2d_temp_out;
            if (gcm_direct_verified_release_enabled() &&
                destMax >= aligned_bytes) {
                plaintext_output = dst;
                gcm_direct_verified_release = true;
            }
            int ret = -1;
            if (gcm_precomputed_xor) {
                if (g_launch_aicpu_gcm_xor_only == nullptr) {
                    fprintf(stderr,
                            "[SecureInfer FATAL] profile-only GCM XOR control "
                            "ABI is unavailable\n");
                    std::abort();
                }
                ret = g_launch_aicpu_gcm_xor_only(
                    stream, h2d_temp_in, g_dev_key, meta.dev_ghash_h,
                    expected_tag, ready_keystream, plaintext_output,
                    auth_status, aligned_bytes, iv_words[0], iv_words[1],
                    iv_words[2], iv_words[3]);
            } else if (g_launch_aicpu_aes_ctr != nullptr) {
                ret = g_launch_aicpu_aes_ctr(
                    stream, h2d_temp_in, g_dev_key, plaintext_output,
                    aligned_bytes, iv_words[0], iv_words[1], iv_words[2],
                    iv_words[3]);
            }
            if (ret != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] profile-only GCM authentication "
                        "control launch failed ret=%d ticket_xor=%d\n",
                        ret, gcm_precomputed_xor ? 1 : 0);
                std::abort();
            }
            gcm_auth_bypassed = true;
            // No authentication status is produced by this test-only path.
            // Keeping the pointer null prevents validation code from reading
            // the intentionally skipped status record.
            gcm_auth_status = nullptr;
        } else if (gcm_precomputed_xor &&
                   gcm_pipeline_slot != nullptr &&
                   gcm_pipeline_slot->verify_stream != nullptr &&
                   gcm_pipeline_slot->xor_stream != nullptr &&
                   ((!aad_enabled && g_launch_aicpu_gcm_verify_only != nullptr) ||
                    (aad_enabled && g_launch_aicpu_gcm_verify_only_aad != nullptr)) &&
                   gcm_parallel_verify_xor_enabled() &&
                   destMax >= aligned_bytes) {
            // The parent stream is already ordered after ciphertext H2D and
            // the exact ticket's producer-event dependency.  Fork from that
            // point, then either join both engines before the verified-release
            // gate (normal mode), or let the parent publish XOR output while
            // the independent verifier continues (experimental eager mode).
            const bool eager_use =
                gcm_fasttrack_eager_enabled() &&
                gcm_pipeline_slot->verify_stream != stream &&
                gcm_pipeline_slot->verify_done != nullptr;
            const bool fork_ok =
                real_aclrtRecordEvent(gcm_pipeline_slot->fork_ready, stream) ==
                    ACL_SUCCESS &&
                real_aclrtStreamWaitEvent(gcm_pipeline_slot->xor_stream,
                                           gcm_pipeline_slot->fork_ready) ==
                    ACL_SUCCESS &&
                real_aclrtStreamWaitEvent(gcm_pipeline_slot->verify_stream,
                                           gcm_pipeline_slot->fork_ready) ==
                    ACL_SUCCESS;
            const aclrtStream verify_stream = gcm_pipeline_slot->verify_stream;
            const int verify_ret = fork_ok
                ? (aad_enabled
                    ? g_launch_aicpu_gcm_verify_only_aad(
                          verify_stream, h2d_temp_in,
                          g_dev_key, meta.dev_ghash_h, expected_tag, auth_status,
                          aligned_bytes, iv_words[0], iv_words[1], iv_words[2],
                          iv_words[3], h2d_aad_device, gcm_aad_length)
                    : g_launch_aicpu_gcm_verify_only(
                          verify_stream, h2d_temp_in,
                          g_dev_key, meta.dev_ghash_h, expected_tag, auth_status,
                          aligned_bytes, iv_words[0], iv_words[1], iv_words[2],
                          iv_words[3]))
                : -1;
            const bool xor_ok = verify_ret == 0 &&
                secure_infer_enqueue_precomputed_xor(
                    gcm_pipeline_slot->xor_stream, h2d_temp_in,
                    ready_keystream, dst, aligned_bytes);
            const bool join_ok = xor_ok &&
                real_aclrtRecordEvent(gcm_pipeline_slot->xor_done,
                                      gcm_pipeline_slot->xor_stream) ==
                    ACL_SUCCESS &&
                real_aclrtStreamWaitEvent(stream,
                                           gcm_pipeline_slot->xor_done) ==
                    ACL_SUCCESS;
            aclError verify_done_ret = verify_ret == 0
                ? real_aclrtRecordEvent(gcm_pipeline_slot->verify_done,
                                        verify_stream)
                : ACL_ERROR_FAILURE;
            bool verify_join_ok = verify_done_ret == ACL_SUCCESS;
            if (!eager_use && verify_join_ok) {
                verify_join_ok =
                    real_aclrtStreamWaitEvent(stream,
                                               gcm_pipeline_slot->verify_done) ==
                    ACL_SUCCESS;
            }
            if (eager_use && verify_done_ret == ACL_SUCCESS) {
                // The slot remains live until verification finishes.  No
                // layer event waits for this event, which is the
                // deliberate Fastrack-style use-before-auth comparison.
                // Slot ownership must still extend through GHASH even though
                // the layer event below is allowed to expose the XOR result.
                // This protects slot reuse, not result commit.
                verify_done_ret = real_aclrtStreamWaitEvent(
                    verify_stream, gcm_pipeline_slot->xor_done);
                if (verify_done_ret == ACL_SUCCESS) {
                    verify_done_ret = real_aclrtRecordEvent(
                        gcm_pipeline_slot->done, verify_stream);
                }
                gcm_eager_slot_done_recorded =
                    verify_done_ret == ACL_SUCCESS;
            }
            const int gate_ret = eager_use
                ? (verify_done_ret == ACL_SUCCESS ? 0 : -1)
                : (join_ok && verify_join_ok
                    ? g_launch_aicpu_auth_gate(stream, dst, auth_status,
                                               aligned_bytes)
                    : -1);
            if (gate_ret != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] parallel GCM verify/XOR launch "
                        "failed eager=%d fork=%d verify=%d xor=%d join=%d "
                        "verify_done=%d verify_join=%d gate=%d\n",
                        eager_use ? 1 : 0,
                        fork_ok ? 1 : 0, verify_ret, xor_ok ? 1 : 0,
                        join_ok ? 1 : 0, static_cast<int>(verify_done_ret),
                        verify_join_ok ? 1 : 0, gate_ret);
                std::abort();
            }
            if (!eager_use && !join_ok) {
                fprintf(stderr,
                        "[SecureInfer FATAL] verified GCM join failed\n");
                std::abort();
            }
            // In normal mode no layer event can observe plaintext before the
            // joined verifier and fail-closed gate.  In eager mode the layer
            // event intentionally represents decrypt-ready output and the
            // resulting model output remains tentative.
            gcm_direct_verified_release = true;
        } else if (gcm_precomputed_xor &&
                   ((!aad_enabled && g_launch_aicpu_gcm_verify_xor != nullptr) ||
                    (aad_enabled && g_launch_aicpu_gcm_verify_xor_aad != nullptr))) {
            void* verified_output = h2d_temp_out;
            if (gcm_direct_verified_release_enabled() &&
                destMax >= aligned_bytes) {
                verified_output = dst;
                gcm_direct_verified_release = true;
            }
            const int ret = aad_enabled
                ? g_launch_aicpu_gcm_verify_xor_aad(
                      stream, h2d_temp_in, g_dev_key, meta.dev_ghash_h,
                      expected_tag, ready_keystream, verified_output, auth_status,
                      aligned_bytes, iv_words[0], iv_words[1], iv_words[2],
                      iv_words[3], h2d_aad_device, gcm_aad_length)
                : g_launch_aicpu_gcm_verify_xor(
                      stream, h2d_temp_in, g_dev_key, meta.dev_ghash_h,
                      expected_tag, ready_keystream, verified_output, auth_status,
                      aligned_bytes, iv_words[0], iv_words[1], iv_words[2],
                      iv_words[3]);
            if (ret != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] fused GCM ticket-XOR launch failed ret=%d\n",
                        ret);
                std::abort();
            }
        } else if ((!aad_enabled && g_launch_aicpu_gcm_decrypt_verify != nullptr) ||
                   (aad_enabled && g_launch_aicpu_gcm_decrypt_verify_aad != nullptr)) {
            void* verified_output = h2d_temp_out;
            if (gcm_direct_verified_release_enabled() &&
                destMax >= aligned_bytes) {
                verified_output = dst;
                gcm_direct_verified_release = true;
            }
            const int ret = aad_enabled
                ? g_launch_aicpu_gcm_decrypt_verify_aad(
                      stream, h2d_temp_in, g_dev_key, meta.dev_ghash_h,
                      expected_tag, verified_output, auth_status, aligned_bytes,
                      iv_words[0], iv_words[1], iv_words[2], iv_words[3],
                      h2d_aad_device, gcm_aad_length)
                : g_launch_aicpu_gcm_decrypt_verify(
                      stream, h2d_temp_in, g_dev_key, meta.dev_ghash_h,
                      expected_tag, verified_output, auth_status, aligned_bytes,
                      iv_words[0], iv_words[1], iv_words[2], iv_words[3]);
            if (ret != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] fused GCM H2D launch failed ret=%d\n",
                        ret);
                std::abort();
            }
        } else {
            // Compatibility route for an older AICPU image without either
            // fused verified-release ABI.
            auto* tag_mask = expected_tag + 16;
            auto* computed_tag = tag_mask + 16;
            auth_status = reinterpret_cast<uint32_t*>(computed_tag + 16);
            gcm_auth_status = auth_status;
            if (!enqueue_gcm_tag_mask(stream, tag_mask, g_dev_key, iv)) {
                fprintf(stderr,
                        "[SecureInfer FATAL] inline GCM tag-mask generation failed\n");
                std::abort();
            }
            const int ret = g_launch_aicpu_ghash(
                stream, h2d_temp_in, meta.dev_ghash_h,
                computed_tag, h2d_aad_device,
                tag_mask, expected_tag,
                nullptr, nullptr, auth_status, count,
                secure_infer_gcm_aad_enabled() ? gcm_aad_length : 0,
                PIPE_GHASH_FINALIZE_TAG | PIPE_GHASH_VERIFY_TAG);
            if (ret != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] inline GHASH launch failed ret=%d\n",
                        ret);
                std::abort();
            }
            if (gcm_precomputed_xor) {
                if (!secure_infer_enqueue_precomputed_xor(
                        stream, h2d_temp_in, ready_keystream,
                        h2d_temp_out, aligned_bytes)) {
                    fprintf(stderr,
                            "[SecureInfer FATAL] inline GCM precomputed-XOR launch failed\n");
                    std::abort();
                }
            } else if (g_launch_aicpu_aes_ctr(
                           stream, h2d_temp_in, g_dev_key,
                           h2d_temp_out, aligned_bytes,
                           iv_words[0], iv_words[1], iv_words[2], iv_words[3]) != 0) {
                fprintf(stderr, "[SecureInfer FATAL] inline AICPU AES-CTR launch failed\n");
                std::abort();
            }
            g_launch_aicpu_auth_gate(stream, h2d_temp_out,
                                     auth_status, count);
        }
    } else {
        if (!load_ghash_library() || g_launch_aicpu_aes_ctr == nullptr ||
            g_launch_aicpu_aes_ctr(
                stream, h2d_temp_in, g_dev_key,
                h2d_temp_out, aligned_bytes,
                iv_words[0], iv_words[1],
                iv_words[2], iv_words[3]) != 0) {
            fprintf(stderr,
                    "[SecureInfer FATAL] inline CTR AICPU AES launch failed\n");
            std::abort();
        }
    }
    kernel_enqueue_us += secure_infer_now_us() - kernel_start;
    if (tamper_injected) {
        const uint32_t logical_layer = secure_infer_gcm_aad_u32(gcm_aad, 8);
        const uint32_t tensor_kind = static_cast<uint32_t>(gcm_aad[6]);
        // This is a host submission timestamp for device authentication.
        // The completion event is emitted only after the device status has
        // been observed, so the pair brackets the authentication path without
        // instrumenting normal transfers.
        secure_infer_record_h2d_fault_event(
            "authentication_submitted", logical_layer, tensor_kind, count);
    }
    if (precise_profile_active &&
        real_aclrtRecordEvent(timing.kernel_end, stream) != ACL_SUCCESS) {
        timing.records_ok = false;
    }

    if (!gcm_direct_verified_release) {
        copy_start = secure_infer_now_us();
        real_aclrtMemcpyAsync(dst, destMax,
                              h2d_temp_out, count,
                              ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        copy_enqueue_us += secure_infer_now_us() - copy_start;
    }
    if (precise_profile_active &&
        real_aclrtRecordEvent(timing.pipeline_end, stream) != ACL_SUCCESS) {
        timing.records_ok = false;
    }

    bool run_gcm_validation = false;
    if (gcm_auth_status != nullptr) {
        const char* validate_env = std::getenv("PIPE_GCM_VALIDATE_H2D");
        if (validate_env != nullptr && std::atoi(validate_env) != 0) {
            const char* samples_env =
                std::getenv("PIPE_GCM_VALIDATE_H2D_SAMPLES");
            long validation_samples = samples_env != nullptr
                ? std::atol(samples_env) : 4;
            if (validation_samples < 1) validation_samples = 1;
            static std::atomic<uint64_t> validation_count{0};
            run_gcm_validation = validation_count.fetch_add(
                1, std::memory_order_relaxed) <
                static_cast<uint64_t>(validation_samples);
        }
    }
    // The tamper test must observe the specific corrupted slice even if
    // ordinary validation samples have already been consumed during warmup.
    run_gcm_validation = run_gcm_validation || tamper_injected;
    if (recovery_event_monitor_enabled()) {
        // Generic event-driven handling replaces injector-selected validation.
        run_gcm_validation = false;
        if (gcm_auth_status == nullptr || gcm_pipeline_slot == nullptr) {
            fprintf(stderr, "[SecureInfer FATAL] recovery requires per-record GCM status\n");
            std::abort();
        }
        const aclrtStream observation_stream =
            gcm_precomputed_xor && gcm_direct_verified_release &&
            gcm_parallel_verify_xor_enabled() &&
            gcm_pipeline_slot->verify_stream != nullptr
                ? gcm_pipeline_slot->verify_stream : stream;
        const auto record = observe_recovery_authentication(
            gcm_auth_status, observation_stream,
            secure_infer_gcm_aad_u32(gcm_aad, 8),
            static_cast<uint32_t>(gcm_aad[6]), dst, destMax, src, count, stream);
        gcm_pipeline_slot->authentication_observed = record->observed;
    }
    if (tamper_injected && secure_infer_allow_eager_tampered_kv()) {
        // The propagation experiment deliberately lets the tentative result
        // reach computation.  Authentication status is read when this slot
        // retires, after the independent verifier stream has completed.
        run_gcm_validation = false;
    }

    if (gcm_pipeline_slot != nullptr) {
        // The completion event is after the auth gate and the D2D publication
        // copy.  Reusing this slot therefore cannot overwrite ciphertext or
        // device scratch that an outstanding verified transfer still needs.
        gcm_pipeline_slot->profile_pending = true;
        gcm_pipeline_slot->profile_precise = precise_profile_active;
        gcm_pipeline_slot->profile_total_start_us = total_start;
        gcm_pipeline_slot->profile_cpu_crypto_us = cpu_crypto_us;
        gcm_pipeline_slot->profile_host_device_start_us = device_start;
        gcm_pipeline_slot->profile_copy_enqueue_us = copy_enqueue_us;
        gcm_pipeline_slot->profile_kernel_enqueue_us = kernel_enqueue_us;
        gcm_pipeline_slot->profile_count = count;
        gcm_pipeline_slot->profile_aligned_bytes = aligned_bytes;
        gcm_pipeline_slot->profile_cpu_prediction_hit = cpu_prediction_hit;
        gcm_pipeline_slot->profile_gcm_ticket_xor = gcm_precomputed_xor;
        gcm_pipeline_slot->profile_ticket_backend = gcm_ticket_backend;
        gcm_pipeline_slot->profile_gcm_on_demand_fallback =
            h2d_gcm_enabled && !gcm_precomputed_xor;
        gcm_pipeline_slot->profile_cpu_inline_fallback =
            h2d_gcm_enabled && !cpu_prediction_hit;
        gcm_pipeline_slot->profile_tamper_injected = tamper_injected;
        gcm_pipeline_slot->profile_auth_bypassed = gcm_auth_bypassed;
        gcm_pipeline_slot->profile_auth_status =
            tamper_injected ? gcm_auth_status : nullptr;
        gcm_pipeline_slot->profile_plaintext_output =
            tamper_injected ? dst : nullptr;
        gcm_pipeline_slot->profile_expected_plaintext =
            tamper_injected ? src : nullptr;
        gcm_pipeline_slot->profile_logical_layer = tamper_injected
            ? ((static_cast<uint32_t>(gcm_aad[8]) << 24) |
               (static_cast<uint32_t>(gcm_aad[9]) << 16) |
               (static_cast<uint32_t>(gcm_aad[10]) << 8) |
               static_cast<uint32_t>(gcm_aad[11]))
            : UINT32_MAX;
        gcm_pipeline_slot->profile_tensor_kind = tamper_injected
            ? static_cast<uint32_t>(gcm_aad[6]) : UINT32_MAX;
        gcm_pipeline_slot->keystream_ticket =
            gcm_precomputed_xor ? transfer_ticket : 0;
        gcm_pipeline_slot->ghash_critical = gcm_ghash_critical;
        aclError event_ret = ACL_SUCCESS;
        if (!gcm_eager_slot_done_recorded) {
            event_ret = real_aclrtRecordEvent(gcm_pipeline_slot->done,
                                              stream);
        }
        if (event_ret != ACL_SUCCESS) {
            fprintf(stderr,
                    "[SecureInfer WARN] cannot record GCM H2D slot event (%d); "
                    "draining this stream once\n", static_cast<int>(event_ret));
            real_aclrtSynchronizeStream(stream);
            // Mark complete before retiring so a ready ticket is released
            // only after the stream has drained.
            gcm_pipeline_slot->used = true;
            retire_h2d_gcm_pipeline_slot(*gcm_pipeline_slot);
        } else {
            gcm_pipeline_slot->used = true;
        }

        // Validation is diagnostic-only and samples the first few blocks.  It
        // waits on this slot's event rather than on all queued transfers.
        if (run_gcm_validation || event_ret != ACL_SUCCESS) {
            if (event_ret == ACL_SUCCESS) {
                aclError wait_ret = real_aclrtSynchronizeEvent(
                    gcm_pipeline_slot->done);
                if (wait_ret != ACL_SUCCESS) real_aclrtSynchronizeStream(stream);
            }
            if (gcm_pipeline_slot->used) {
                retire_h2d_gcm_pipeline_slot(*gcm_pipeline_slot);
            }
        }
    } else {
        // Compatibility path for ticketed pre-generation: its ticket buffer
        // may be reclaimed only after this stream consumes it.
        real_aclrtSynchronizeStream(stream);
        if (gcm_ticket_pool_active) {
            secure_infer_gcm_finish_keystream_ticket(transfer_ticket);
        }
        if (gcm_ghash_critical) {
            secure_infer_gcm_ghash_critical_exit();
        }
    }

    if (run_gcm_validation) {
            uint32_t auth_status_host = 0;
            std::vector<unsigned char> recovered(count);
            aclError status_ret = real_aclrtMemcpy(
                &auth_status_host, sizeof(auth_status_host),
                gcm_auth_status, sizeof(auth_status_host),
                ACL_MEMCPY_DEVICE_TO_HOST);
            const void* validation_plaintext = gcm_direct_verified_release
                ? dst : h2d_temp_out;
            aclError plain_ret = real_aclrtMemcpy(
                recovered.data(), count, validation_plaintext, count,
                ACL_MEMCPY_DEVICE_TO_HOST);
            size_t first_mismatch = plain_ret == ACL_SUCCESS ? 0 : count;
            if (plain_ret == ACL_SUCCESS) {
                const auto* expected =
                    static_cast<const unsigned char*>(src);
                while (first_mismatch < count &&
                       recovered[first_mismatch] == expected[first_mismatch]) {
                    ++first_mismatch;
                }
            }
            fprintf(stderr,
                    "[SecureInfer] GCM H2D validation: auth=%u status_ret=%d "
                    "copy_ret=%d first_mismatch=%zu/%zu "
                    "ticket_xor=%d ticket_backend=%d\n",
                    auth_status_host, static_cast<int>(status_ret),
                    static_cast<int>(plain_ret), first_mismatch, count,
                    gcm_precomputed_xor ? 1 : 0, gcm_ticket_backend);
            if (tamper_injected) {
                secure_infer_record_h2d_fault_event(
                    "authentication_complete",
                    secure_infer_gcm_aad_u32(gcm_aad, 8),
                    static_cast<uint32_t>(gcm_aad[6]), count);
                const bool no_plaintext_published =
                    plain_ret == ACL_SUCCESS && std::all_of(
                        recovered.begin(), recovered.end(),
                        [](unsigned char value) { return value == 0; });
                if (auth_status_host == 0 && status_ret == ACL_SUCCESS &&
                    no_plaintext_published) {
                    fprintf(stderr,
                            "[SecureInfer TEST PASS] GCM tamper rejected; auth gate "
                            "published no plaintext\n");
                } else {
                    fprintf(stderr,
                            "[SecureInfer TEST FAIL] tampered GCM H2D was not "
                            "fail-closed (auth=%u, zero_output=%d)\n",
                            auth_status_host, no_plaintext_published ? 1 : 0);
                }
                const char* recover_env =
                    std::getenv("PIPE_GCM_TAMPER_STRICT_RETRY");
                if (auth_status_host == 0 && status_ret == ACL_SUCCESS &&
                    no_plaintext_published && recover_env != nullptr &&
                    std::atoi(recover_env) != 0) {
                    const uint32_t layer = secure_infer_gcm_aad_u32(gcm_aad, 8);
                    const uint32_t tensor = static_cast<uint32_t>(gcm_aad[6]);
                    secure_infer_record_h2d_fault_event(
                        "authentication_failed", layer, tensor, count);
                    secure_infer_record_h2d_fault_event(
                        "strict_retry_begin", layer, tensor, count);
                    // The failed attempt has completed and its fail-closed
                    // gate exposed only zeros.  Release the per-stream host
                    // lock before recursively submitting the trusted source
                    // with a fresh transfer ticket and IV.  The one-shot
                    // injector cannot corrupt this retry.  Python records the
                    // public layer event only after this call returns, so the
                    // retry remains ordered before any attention consumer.
                    if (pipeline_slot_lock.owns_lock()) {
                        pipeline_slot_lock.unlock();
                    }
                    if (legacy_inline_lock.owns_lock()) {
                        legacy_inline_lock.unlock();
                    }
                    const aclError retry_ret = inline_h2d_encrypt_commit(
                        dst, destMax, src, count, stream,
                        "authenticated H2D retry");
                    secure_infer_record_h2d_fault_event(
                        retry_ret == ACL_SUCCESS ? "strict_retry_submitted"
                                                : "strict_retry_failed",
                        layer, tensor, count);
                    return retry_ret;
                }
                // Do not let a deliberately corrupted KV slice proceed to
                // model execution.  An asynchronous ACL hook cannot return
                // a per-slice error after publication has been queued, so
                // terminating this opt-in test process is the fail-closed
                // end-to-end oracle.
                std::abort();
            }
            const char* strict_env =
                std::getenv("PIPE_GCM_VALIDATE_H2D_STRICT");
            if (strict_env != nullptr && std::atoi(strict_env) != 0 &&
                (auth_status_host != 1 || status_ret != ACL_SUCCESS ||
                 plain_ret != ACL_SUCCESS || first_mismatch != count)) {
                fprintf(stderr,
                        "[SecureInfer FATAL] strict GCM H2D validation failed\n");
                std::abort();
            }
    }

    if (gcm_pipeline_slot != nullptr) {
        static std::atomic<uint64_t> async_inline_log_count{0};
        if (sampled_predict_log(async_inline_log_count, 16, 2048)) {
            fprintf(stderr,
                    "[SecureInfer] %s: GCM verified H2D queued asynchronously "
                    "src=%p, size=%zu, cpu_crypto=%lu us\n",
                    reason, src, count,
                    static_cast<unsigned long>(cpu_crypto_us));
        }
        return ACL_SUCCESS;
    }

    free_inline_cipher_buffer(cipher);
    uint64_t host_device_us = secure_infer_now_us() - device_start;
    if (precise_profile_active) {
        bool valid = timing.records_ok;
        uint64_t h2d_device_us = inline_event_elapsed_us(
            timing.pipeline_start, timing.kernel_start, valid);
        uint64_t kernel_execute_us = inline_event_elapsed_us(
            timing.kernel_start, timing.kernel_end, valid);
        uint64_t d2d_device_us = inline_event_elapsed_us(
            timing.kernel_end, timing.pipeline_end, valid);
        uint64_t device_total_us = inline_event_elapsed_us(
            timing.pipeline_start, timing.pipeline_end, valid);
        profile_record_precise(timing.run_id,
                               "H2D",
                               count,
                               aligned_bytes,
                               secure_infer_now_us() - total_start,
                               cpu_crypto_us,
                               valid ? device_total_us : host_device_us,
                               0,
                               copy_enqueue_us,
                               kernel_enqueue_us,
                               h2d_device_us,
                               kernel_execute_us,
                               d2d_device_us,
                               device_total_us,
                               valid,
                               0,
                               cpu_prediction_hit ? 1 : 0);
        destroy_inline_device_timing(timing);
    } else {
        profile_record("H2D",
                       count,
                       aligned_bytes,
                       secure_infer_now_us() - total_start,
                       cpu_crypto_us,
                       host_device_us,
                       0,
                       copy_enqueue_us,
                       kernel_enqueue_us);
    }
    static std::atomic<uint64_t> inline_log_count{0};
    if (sampled_predict_log(inline_log_count, 16, 2048)) {
        fprintf(stderr,
                "[SecureInfer] %s: 原地串行加密传输完成 src=%p, size=%zu, cpu_crypto=%lu us\n",
                reason, src, count, static_cast<unsigned long>(cpu_crypto_us));
    }
    return ACL_SUCCESS;
}

extern "C" int secure_infer_recovery_wait_layer_authentication(uint32_t layer)
{
    if (!recovery_event_monitor_enabled() || gcm_fasttrack_eager_enabled()) return 0;
    for (;;) {
        std::vector<std::shared_ptr<recovery_auth_record>> records;
        {
            std::lock_guard<std::mutex> guard(g_recovery_auth_lock);
            auto it = g_recovery_layer_records.find(layer);
            if (it == g_recovery_layer_records.end()) return 0;
            records.swap(it->second);
            g_recovery_layer_records.erase(it);
        }
        for (const auto& record : records) {
            const int status = record->observed.get();
            if (status == 1) continue;
            if (status != 0) return -1;
            secure_infer_record_h2d_fault_event("strict_retry_begin",
                record->layer, record->tensor, record->bytes);
            // The failed record is identified exclusively by its verifier.
            // Issue a fresh ticket/IV from the trusted host copy.  The new
            // verifier result is checked on the next loop before layer use.
            if (inline_h2d_encrypt_commit(record->destination,
                    record->destination_caruntimety, record->source, record->bytes,
                    record->stream, "event-triggered authenticated H2D retry") != ACL_SUCCESS)
                return -1;
            if (real_aclrtSynchronizeStream(record->stream) != ACL_SUCCESS) return -1;
            secure_infer_record_h2d_fault_event("strict_retry_submitted",
                record->layer, record->tensor, record->bytes);
        }
    }
}

// The eager-use evaluation waits for every verifier result before the
// request's output or reusable KV may be published. The caller invokes this
// after submitting the complete H2D reload.
extern "C" int secure_infer_recovery_wait_job_authentication(uint64_t job_id)
{
    if (!recovery_event_monitor_enabled() || !gcm_fasttrack_eager_enabled())
        return -1;
    std::vector<std::shared_ptr<recovery_auth_record>> records;
    {
        std::lock_guard<std::mutex> guard(g_recovery_auth_lock);
        for (auto it = g_recovery_layer_records.begin();
             it != g_recovery_layer_records.end();) {
            auto& pending = it->second;
            for (auto record_it = pending.begin(); record_it != pending.end();) {
                if ((*record_it)->job_id == job_id) {
                    records.push_back(*record_it);
                    record_it = pending.erase(record_it);
                } else {
                    ++record_it;
                }
            }
            if (pending.empty()) it = g_recovery_layer_records.erase(it);
            else ++it;
        }
    }
    if (records.empty()) return -1;
    bool valid = true;
    for (const auto& record : records) {
        valid = record->observed.get() == 1 && valid;
    }
    secure_infer_record_h2d_fault_event(
        valid ? "authentication_commit_ready" : "authentication_commit_failed",
        UINT32_MAX, UINT32_MAX, records.size());
    return valid ? 1 : 0;
}
/*
 * [替换] cudaDeviceSynchronize
 */
HOOK_C_API HOOK_DECL_EXPORT aclError aclrtSynchronizeDevice()
{
    predictor.lock();
    sync_predictor();
    predictor.unlock();

    return real_aclrtSynchronizeDevice(); // <-- [修改]
}

/*
 * [替换] cudaStreamSynchronize
 */
HOOK_C_API HOOK_DECL_EXPORT aclError aclrtSynchronizeStream(aclrtStream stream) // <-- [修改]
{
    predictor.lock();
    sync_predictor();
    predictor.unlock();

    const aclError ret = real_aclrtSynchronizeStream(stream); // <-- [修改]
    if (ret == ACL_SUCCESS) {
        flush_completed_h2d_gcm_slots(stream);
    }
    return ret;
}

// torch.npu.Event.synchronize() reaches the ACL event API directly rather
// than aclrtSynchronizeStream.  Retire completed GCM H2D slots here as well,
// so the final pipeline window is profiled without inserting a new fence.
HOOK_C_API HOOK_DECL_EXPORT aclError aclrtSynchronizeEvent(aclrtEvent event)
{
    const aclError ret = real_aclrtSynchronizeEvent(event);
    if (ret == ACL_SUCCESS) {
        flush_completed_h2d_gcm_slots();
    }
    return ret;
}

/*
 * [替换] cudaStreamWaitEvent
 * [注意] ACL API 没有 flags 参数
 */
HOOK_C_API HOOK_DECL_EXPORT aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) // <-- [修改]
{
    predictor.lock();
    sync_predictor();
    predictor.unlock();

    return real_aclrtStreamWaitEvent(stream, event); // <-- [修改]
}

static bool first = 0;

static inline void init_encrypt_ctx(aclrtMemcpyKind kind, aclrtStream stream)
{
    static std::once_flag init_once;
    std::call_once(init_once, [kind, stream] {
        (void)kind;

        // 加载NPU算子库
        load_aes_library();

        fprintf(stderr, "[SecureInfer Debug] ===== init_encrypt_ctx 开始 =====\n");

        // 读取环境变量
        auto enable_encrypt_env = std::getenv("SECURE_INFER_ENABLE_ENCRYPT");
        auto enable_decrypt_env = std::getenv("SECURE_INFER_ENABLE_DECRYPT");

        auto enable_h2d_env = std::getenv("PIPE_ENABLE_H2D");
        auto enable_d2h_env = std::getenv("PIPE_ENABLE_D2H");

        enable_h2d_encrypt = enable_h2d_env ? std::atoi(enable_h2d_env) != 0 : false;
        enable_d2h_encrypt = enable_d2h_env ? std::atoi(enable_d2h_env) != 0 : false;

        // bool enable_encrypt = enable_encrypt_env ? std::atoi(enable_encrypt_env) != 0 : false;
        // bool enable_decrypt = enable_decrypt_env ? std::atoi(enable_decrypt_env) != 0 : false;

        // 内部使用全局变量控制worker初始化
        bool global_enable_encrypt = (enable_h2d_encrypt || enable_d2h_encrypt);

        fprintf(stderr, "[SecureInfer Debug] enable_h2d_encrypt=%d, enable_d2h_encrypt=%d, global_enable=%d\n",
                enable_h2d_encrypt, enable_d2h_encrypt, global_enable_encrypt);

        // H2D lookahead workers are not used by the D2H path.  Do not create
        // their polling threads in a D2H-only diagnostic run; this both avoids
        // unrelated CPU contention and gives us a clean control-overhead A/B.
        // The default bidirectional configuration is unchanged.
        // The serial baseline normally suppresses speculative workers.  The
        // opt-in GCM ticket pool is the one exception: demand consumption is
        // still serial/verified, but its *keystream producer* must run ahead
        // on the private commit-worker stream or no reserved ticket can ever
        // become ready. CPU ciphertext lookahead is the other exception: its
        // producer runs on the host and is driven by observed address history.
        const bool gcm_ticketed_scheduler =
            secure_infer_gcm_enabled() && secure_infer_gcm_keystream_scheduler_enabled();
        const bool gcm_cpu_cipher_lookahead =
            secure_infer_gcm_enabled() && secure_infer_gcm_cpu_cipher_prefetch_enabled();
        if (enable_h2d_encrypt &&
            (!h2d_force_serial_enabled() || gcm_ticketed_scheduler ||
             gcm_cpu_cipher_lookahead)) {
            fprintf(stderr, "[SecureInfer Debug] 阶段2: 创建加密工作线程\n");

            // 一个 commit worker + 多个 CPU 预测加密 worker。
            // KV block 是 256KB，小于 block_unit，不会被 commit worker 再切片；
            // 多 worker 的作用是让未来块的 CPU 加密真正跑在当前块传输之前。
            // GCM only needs the commit worker to fill device keystreams; it
            // never creates CPU ciphertext predictions.  One dormant helper
            // keeps the legacy worker-entry lifecycle intact without adding
            // the normal four CPU AES polling threads to the experiment.
            encrypt_workers_per_commit =
                (gcm_ticketed_scheduler || gcm_cpu_cipher_lookahead)
                ? 1 : predict_encrypt_workers_default();
            encrypt_workers_num = encrypt_workers_per_commit;
            commit_workers_num = 1;

            for (int i = 0; i < commit_workers_num; i++) {
                fprintf(stderr, "[SecureInfer Debug] 创建commit worker %d\n", i);
                auto worker_entry = new encrypt_worker_entry;
                encrypt_worker_entrys[i] = worker_entry;
                worker_entry->id = i;
                worker_entry->magic_sz = encrypt_magic_sz + i;
                worker_entry->commit_init_done.store(false);
                worker_entry->encrypt_workers_per_commit = encrypt_workers_per_commit;
                s_magic.insert(worker_entry->magic_sz);

                // 创建commit worker
                commit_worker_threads[i] = new std::thread(commit_worker, worker_entry);
                while (!worker_entry->commit_init_done.load()) std::this_thread::yield();


                // 在创建 encrypt worker 之前添加检查
                for (int j = 0; j < encrypt_workers_per_commit; j++) {
                    fprintf(stderr, "[SecureInfer Debug] 准备创建encrypt worker %d-%d\n", i, j);

                    // 检查 worker_entry 状态
                    fprintf(stderr, "[SecureInfer Debug] worker_entry->id=%d\n", worker_entry->id);
                    fprintf(stderr, "[SecureInfer Debug] worker_entry->local_id=%d\n", worker_entry->local_id);
                    fprintf(stderr, "[SecureInfer Debug] worker_entry->encrypt_workers_per_commit=%d\n",
                            worker_entry->encrypt_workers_per_commit);
                    fprintf(stderr, "[SecureInfer Debug] worker_entry->magic_sz=%zu\n", worker_entry->magic_sz);

                    worker_entry->enc_init_done.store(false);
                    worker_entry->local_id = j;

                    fprintf(stderr, "[SecureInfer Debug] 创建encrypt worker线程\n");
                    fflush(stderr);

                    encrypt_worker_threads[i * encrypt_workers_per_commit + j] =
                        new std::thread(encrypt_worker, worker_entry);

                    fprintf(stderr, "[SecureInfer Debug] 等待encrypt worker初始化\n");
                    fflush(stderr);

                    while (!worker_entry->enc_init_done.load()) std::this_thread::yield();

                    fprintf(stderr, "[SecureInfer Debug] encrypt worker %d-%d 初始化完成\n", i, j);
                    fflush(stderr);
                }

            }
            fprintf(stderr, "[SecureInfer Debug] 加密工作线程创建完成\n");
            g_encrypt_workers_ready.store(true, std::memory_order_release);
        }

        // This hook runs on the EngineCore ACL submission thread, before the
        // first intercepted copy is forwarded to CANN.  It is the safe point
        // to cold-start the preloaded custom AICPU image.  D2H write-behind
        // itself runs on a detached Python worker, where making the first ASC
        // launch can stall indefinitely on current CANN releases.
        if (secure_infer_gcm_enabled() && !m_ctx_metadata.empty()) {
            auto& gcm_meta = *m_ctx_metadata.begin()->second;
            if (!secure_infer_prepare_gcm_backend(gcm_meta, stream)) {
                fprintf(stderr,
                        "[SecureInfer FATAL] GCM backend cold start failed on ACL submission thread\n");
                std::abort();
            }
        }

        fprintf(stderr, "[SecureInfer Debug] ===== init_encrypt_ctx 完成 =====\n");
        fflush(stderr);
    });
}

/*
 * (此函数 100% 保持不变)
 */

void assign_to_workers(void *dst, const void *src, size_t count,
                       bool decrypt = false, bool predict = true,
                       bool commit = false, bool update_predict_iv = false,
                       bool batch_commit = false,
                       aclrtStream user_stream = nullptr,
                       uint64_t transfer_ticket,
                       uint64_t transfer_block_offset,
                       uint64_t prefetch_group_id)
{
    const char *src_char = (const char *)src;
    char *dst_char = (char *)dst;
    if (transfer_ticket == 0) {
        transfer_ticket = secure_infer_new_transfer_ticket();
    }
    size_t div = (count + commit_workers_num - 1) / commit_workers_num;
    div = (div + block_unit - 1) / block_unit * block_unit;

    for (int i = 0; i < commit_workers_num; i++) {
        auto worker_entry = encrypt_worker_entrys[i];
        auto sz = std::min(count, div);

        if (predict) {
            // Keep transfer-ticket reservation and history tracking active
            // while allowing an ablation to suppress only proactive device
            // keystream generation.  The real H2D path still consumes the
            // reserved ticket and executes the exact ticketed CTR kernel on
            // demand, so disabling this option changes timing, not crypto
            // identity or correctness.  The production default remains on.
            static const bool speculative_device_keystream = [] {
                const char* env = std::getenv(
                    "PIPE_GCM_SPECULATIVE_DEVICE_KEYSTREAM");
                return env == nullptr || std::atoi(env) != 0;
            }();
            if (secure_infer_gcm_enabled() &&
                secure_infer_gcm_keystream_scheduler_enabled() &&
                !speculative_device_keystream) {
                src_char += sz;
                transfer_block_offset += (sz + 15) / 16;
                count -= sz;
                if (count == 0) break;
                continue;
            }
            // 预测路径保持不变...
            encryption_task task;
            task.src = src_char;
            task.size = sz;
            task.iv_increment = 32;
            task.generation = predictor.generation;
            task.transfer_ticket = transfer_ticket;
            task.prefetch_group_id = prefetch_group_id;
            task.transfer_block_offset = transfer_block_offset;
            {
                std::lock_guard<std::mutex> lock(worker_entry->lock);
                worker_entry->enc_tasks.push_back(task);
            }
        } else {
            commit_task task;
            task.enc_task.src = src_char;
            task.enc_task.size = sz;
            task.enc_task.generation = predictor.generation;
            task.enc_task.transfer_ticket = transfer_ticket;
            task.enc_task.prefetch_group_id = prefetch_group_id;
            task.enc_task.transfer_block_offset = transfer_block_offset;
            task.dst = dst_char;
            task.using_predict = commit;
            task.update_predict_iv = update_predict_iv;
            task.profile_start_us = secure_infer_now_us();

            if (commit && user_stream != nullptr) {
                task.sync = std::make_shared<commit_sync_state>();
                task.sync->event = acquire_commit_sync_event();
                if (task.sync->event == nullptr) {
                    // Event exhaustion must not turn into a silent drop or a
                    // permanently true commit flag. Keep the user-stream
                    // ordering and preserve the original H2D result.
                    real_aclrtMemcpyAsync(dst_char, sz, src_char, sz,
                                          ACL_MEMCPY_HOST_TO_DEVICE, user_stream);
                    src_char += sz;
                    if (dst_char != nullptr) dst_char += sz;
                    transfer_block_offset += (sz + 15) / 16;
                    count -= sz;
                    if (count == 0) break;
                    continue;
                }
            }

            // 1. 推送任务
            {
                std::lock_guard<std::mutex> lock(worker_entry->lock);
                worker_entry->commit.store(true, std::memory_order_release);
                worker_entry->commit_tasks.push_back(task);
            }

            // 2. [关键修复] 自旋等待：确保 Worker 已经执行了 RecordEvent
            // 只有当 is_submitted 变为 true 后，主线程才能继续往下走。
            // 因为 Record 只是向队列发指令，这个过程极快 (10~30us)，不会造成性能瓶颈。
            if (commit && user_stream != nullptr) {
                while (!task.sync->submitted.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                if (task.sync->status.load(std::memory_order_acquire) == ACL_SUCCESS) {
                    aclError ret = real_aclrtStreamWaitEvent(user_stream, task.sync->event);
                    if (ret != ACL_SUCCESS) {
                        // Rare error path: retain correctness even when the
                        // cross-stream dependency could not be enqueued.
                        aclrtSynchronizeEvent(task.sync->event);
                    }
                    task.sync->status.store(ret, std::memory_order_release);
                }
                task.sync->wait_enqueued.store(true, std::memory_order_release);
            }
        }

        src_char += sz;
        if (dst_char != nullptr) dst_char += sz;
        transfer_block_offset += (sz + 15) / 16;
        count -= sz;
        if (count == 0) break;
    }
}


void sync_predictor()
{
    // Caller must hold the predictor's lock
    if (!first || !(enable_h2d_encrypt || enable_d2h_encrypt)) return;
    assert(predictor.locking());
    if (!predictor.pending_commit.empty()) {
        assert(commit_workers_num == 1);
        {
            std::lock_guard<std::mutex> lock(encrypt_worker_entrys[0]->lock);
            encrypt_worker_entrys[0]->commit.store(true, std::memory_order_release);
            for (auto &task: predictor.pending_commit) {
                commit_task task0;
                task0.enc_task.src = task.first;
                task0.enc_task.size = task.second.second;
                task0.enc_task.generation = predictor.generation;
                task0.dst = task.second.first;
                task0.using_predict = true;
                task0.update_predict_iv = false;
                task0.profile_start_us = secure_infer_now_us();
                encrypt_worker_entrys[0]->commit_tasks.push_back(task0);
            }
        }
        predictor.pending_commit.clear();
    }

    if (!predictor.pending_decrypt.empty()) {
        assert(decrypt_workers_num == 1);
        {
            std::lock_guard<std::mutex> lock(decrypt_worker_entrys[0]->lock);
            for (auto &task: predictor.pending_decrypt) {
                decrypt_worker_entrys[0]->dec_tasks.push_back(task);
            }
        }
        predictor.pending_decrypt.clear();
    }
    for (int i = 0; i < commit_workers_num; i++) {
        while (1) {
            {
                if (!encrypt_worker_entrys[i]->commit.load(std::memory_order_acquire)) {
                    break;
                }
            }
        }
    }
    predictor.pending_commit.clear();
}

aclrtContext g_main_ctx = nullptr;
HOOK_C_API HOOK_DECL_EXPORT aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                                                       aclrtMemcpyKind kind, aclrtStream stream)
{
    if (g_main_ctx == nullptr) {
        aclrtGetCurrentContext(&g_main_ctx);
        if (g_main_ctx != nullptr) {
            fprintf(stderr, "[SecureInfer] ✅ 捕获主线程 Context: %p\n", g_main_ctx);
        }
    }
    // 1. 初始化加密上下文
    init_encrypt_ctx(kind, stream);
    // 2. 基础参数检查
    if (count == 0) return ACL_SUCCESS;
    if (dst == nullptr || src == nullptr) {
        return real_aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    }

    // 3. 设备内部拷贝 (D2D) 直接放行
    if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
        return real_aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    }

    // Evaluation-only plaintext control for the serial baseline.  The older
    // bypass lived inside inline_h2d_encrypt_commit(), so D2H still executed
    // the secure path and a run labelled "plain" was not actually plaintext
    // when the request wrote KV blocks back.  Keep the same per-slice stream
    // boundary as the serial GCM path and retain transfer accounting in both
    // directions; no production configuration enables this flag.
    const char* serial_bypass = std::getenv("PIPE_SERIAL_BYPASS_CRYPTO");
    const bool serial_bypass_enabled =
        serial_bypass != nullptr && std::atoi(serial_bypass) != 0;
    const bool plain_h2d = secure_infer_plain_h2d_enabled();
    const bool plain_d2h = secure_infer_plain_d2h_enabled();
    if (count >= encrypt_threshold_sz &&
        ((kind == ACL_MEMCPY_HOST_TO_DEVICE &&
          (serial_bypass_enabled || plain_h2d)) ||
         (kind == ACL_MEMCPY_DEVICE_TO_HOST &&
          (serial_bypass_enabled || plain_d2h)))) {
        const bool measured_h2d =
            kind == ACL_MEMCPY_HOST_TO_DEVICE &&
            host_prediction_in_same_range(src, src, count);
        const bool measured_d2h =
            kind == ACL_MEMCPY_DEVICE_TO_HOST &&
            host_prediction_in_same_range(dst, dst, count);
        if (measured_h2d || measured_d2h) {
            const uint64_t copy_start = secure_infer_now_us();
            const aclError copy_ret = real_aclrtMemcpyAsync(
                dst, destMax, src, count, kind, stream);
            if (copy_ret != ACL_SUCCESS) return copy_ret;
            const char* sync_env = measured_h2d
                ? std::getenv("PIPE_H2D_SYNC_EVERY_SLICE")
                : std::getenv("PIPE_D2H_SYNC_EVERY_SLICE");
            if (sync_env != nullptr && std::atoi(sync_env) != 0) {
                const aclError sync_ret = real_aclrtSynchronizeStream(stream);
                if (sync_ret != ACL_SUCCESS) return sync_ret;
            }
            const uint64_t copy_us = secure_infer_now_us() - copy_start;
            profile_record(measured_h2d ? "H2D" : "D2H",
                           count, count, copy_us, 0, 0, 0, copy_us, 0);
            return ACL_SUCCESS;
        }
    }

    // =================================================================
    // 4. 加密路径 (Host -> Device) —— 全量加密
    // =================================================================
    if (enable_h2d_encrypt && kind == ACL_MEMCPY_HOST_TO_DEVICE &&
        count >= encrypt_threshold_sz &&
        host_prediction_in_same_range(src, src, count)) {
        static std::atomic<uint64_t> h2d_log_count{0};
        if (sampled_predict_log(h2d_log_count, 16, 4096)) {
            fprintf(stderr, "[SecureInfer] H2D+Encrypt: src=%p, size=%zu (NPU)\n", src, count);
        }

        // Publish the AICPU backend from the framework's ACL submission
        // thread before speculative workers can consume a descriptor.  The
        // worker path only uses an already-ready backend; it never performs a
        // custom-image cold start itself.
        if (secure_infer_gcm_enabled() && !m_ctx_metadata.empty()) {
            auto& gcm_meta = *m_ctx_metadata.begin()->second;
            if (!secure_infer_prepare_gcm_backend(gcm_meta, stream)) {
                fprintf(stderr,
                        "[SecureInfer FATAL] H2D GCM backend initialization failed\n");
                std::abort();
            }
        }

        // History-only GCM preparation uses exactly the same speculative
        // addresses as the validated CTR experiment.  It still enters the
        // existing verified inline consume path below: the predictor only
        // reserves a ticket and starts CPU ciphertext/NPU keystream work; it
        // never releases KV or bypasses GHASH.  Schedule host AES after
        // dropping predictor.lock() to preserve the established lock order.
        if (secure_infer_gcm_enabled() &&
            (periodic_address_predictor_enabled() ||
             secure_infer_repetitive_predictor_enabled()) &&
            (!h2d_force_serial_enabled() ||
             secure_infer_repetitive_predictor_allow_force_serial())) {
            std::vector<periodic_cpu_submission> cpu_submissions;
            predictor.lock();
            if (periodic_address_predictor_enabled()) {
                observe_and_schedule_periodic_prefetches_locked(
                    src, count, stream, &cpu_submissions);
            } else {
                observe_and_schedule_secure_infer_repetitive_locked(
                    src, count, stream, &cpu_submissions);
            }
            predictor.unlock();
            for (const auto& submission : cpu_submissions) {
                (void)secure_infer_gcm_schedule_cpu_ciphertext(
                    std::get<0>(submission), std::get<1>(submission),
                    std::get<2>(submission), std::get<3>(submission));
            }
        }

        // GCM ticket pre-generation is consumed by the verified inline path.
        // The legacy commit-worker consumer is CTR-only, so selecting the
        // opt-in GCM pool must also select this safe serial consume path even
        // when an experiment did not set PIPE_H2D_FORCE_SERIAL explicitly.
        if (h2d_force_serial_enabled() ||
            (secure_infer_gcm_enabled() &&
             (secure_infer_gcm_keystream_scheduler_enabled() ||
              secure_infer_gcm_cpu_cipher_prefetch_enabled()))) {
            return inline_h2d_encrypt_commit(dst, destMax, src, count, stream,
                                             "GCM ticketed H2D");
        }

        predictor.lock();

        if (periodic_address_predictor_enabled()) {
            observe_and_schedule_periodic_prefetches_locked(
                src, count, stream);
        } else if (secure_infer_repetitive_predictor_enabled()) {
            observe_and_schedule_secure_infer_repetitive_locked(
                src, count, stream, nullptr);
        }

        uint64_t demand_ticket = 0;
        uint64_t demand_block_offset = 0;
        uint64_t demand_prefetch_group_id = 0;
        uint64_t descriptor_ns = 0;
        if (consume_transfer_prefetch_locked(src, count, &demand_ticket,
                                             &demand_block_offset,
                                             &demand_prefetch_group_id,
                                             &descriptor_ns)) {
            record_descriptor_lead_time(src, count, demand_ticket,
                                        descriptor_ns);
            static std::atomic<uint64_t> transfer_hit_log_count{0};
            if (sampled_predict_log(transfer_hit_log_count, 32, 2048)) {
                fprintf(stderr,
                        "[SecureInfer] transfer层预测命中: src=%p, size=%zu\n",
                        src, count);
            }
            assign_to_workers(dst, src, count, false, false, true, false,
                              false, stream, demand_ticket,
                              demand_block_offset,
                              demand_prefetch_group_id);
            if (secure_infer_linear_fast_fallback_enabled() &&
                predictor.linear_mode &&
                count == predictor.linear_stride) {
                // A linear prediction is stored in the same transfer-record
                // table as descriptor lookahead.  The old early return did
                // not advance the stride state or refill the window, forcing
                // a mismatch after every fixed-size window.  Keep the window
                // rolling only in the opt-in hook-only branch.
                const uintptr_t current_addr =
                    reinterpret_cast<uintptr_t>(src);
                if (predictor.linear_stride <= UINTPTR_MAX - current_addr) {
                    const void* predicted_next = reinterpret_cast<const void*>(
                        current_addr + predictor.linear_stride);
                    if (host_prediction_in_same_range(
                            src, predicted_next, count)) {
                        predictor.linear_expected_src =
                            const_cast<void*>(predicted_next);
                        schedule_linear_prefetches(src, count, stream);
                    } else {
                        predictor.linear_mode = false;
                        predictor.linear_stride = 0;
                        predictor.linear_expected_src = nullptr;
                        predictor.linear_prefetch_until = nullptr;
                        predictor.linear_count = 0;
                        predictor.read_only_swap_record.clear();
                        predictor.read_only_swap_record.push_back(
                            std::make_pair(src, count));
                    }
                }
            }
            goto POST_PROCESS;
        }

        // The evaluation-only periodic predictor owns prediction in this
        // mode.  A demand not covered by its history-only forecast takes the
        // normal serial correctness fallback; do not also train the legacy
        // local-stride state machine on the same trace.
        if (periodic_address_predictor_enabled() ||
            secure_infer_repetitive_predictor_enabled()) {
            goto SERIAL_FALLBACK;
        }

        if (predictor.linear_mode && count != predictor.linear_stride) {
            static std::atomic<uint64_t> miss_size_log_count{0};
            if (sampled_predict_log(miss_size_log_count, 32, 1024)) {
                fprintf(stderr,
                        "[SecureInfer] ⚠️ 预测未命中，重新学习: expected_size=%zu, actual_size=%zu\n",
                        predictor.linear_stride, count);
            }
            if (secure_infer_linear_fast_fallback_enabled()) {
                secure_infer_retire_prediction_generation(predictor.generation);
            }
            predictor.generation++;
            predictor.linear_mode = false;
            predictor.linear_stride = 0;
            predictor.linear_expected_src = nullptr;
            predictor.linear_prefetch_until = nullptr;
            predictor.linear_count = 0;
            predictor.read_only_swap_record.clear();
            g_transfer_prefetch_records.clear();
            predictor.read_only_swap_record.push_back(std::make_pair(src, count));
            goto SERIAL_FALLBACK;
        }

        // ==================== 情况 1: 线性预测模式（vLLM 专用） ====================
        if (predictor.linear_mode && count == predictor.linear_stride) {
            if (predictor.linear_expected_src != nullptr &&
                src != predictor.linear_expected_src) {
                static std::atomic<uint64_t> miss_addr_log_count{0};
                if (sampled_predict_log(miss_addr_log_count, 32, 1024)) {
                    fprintf(stderr,
                            "[SecureInfer] ⚠️ 预测未命中，重新学习: expected=%p, actual=%p, size=%zu\n",
                            predictor.linear_expected_src, src, count);
                }
                if (secure_infer_linear_fast_fallback_enabled()) {
                    secure_infer_retire_prediction_generation(predictor.generation);
                }
                predictor.generation++;
                predictor.linear_mode = false;
                predictor.linear_stride = 0;
                predictor.linear_expected_src = nullptr;
                predictor.linear_prefetch_until = nullptr;
                predictor.linear_count = 0;
                predictor.read_only_swap_record.clear();
                g_transfer_prefetch_records.clear();
                predictor.read_only_swap_record.push_back(std::make_pair(src, count));
                goto SERIAL_FALLBACK;
            }

            void* predicted_next = (uint8_t*)src + predictor.linear_stride;

            if (host_prediction_in_same_range(src, predicted_next, count)) {
                predictor.linear_expected_src = predicted_next;
                static std::atomic<uint64_t> hit_log_count{0};
                if (sampled_predict_log(hit_log_count, 16, 4096)) {
                    fprintf(stderr, "[SecureInfer] 线性预测命中: 当前=%p → 下一个=%p\n", src, predicted_next);
                }
                assign_to_workers(dst, src, count, false, false, true, false, false, stream);
                schedule_linear_prefetches(src, count, stream);
            } else {
                static std::atomic<uint64_t> boundary_log_count{0};
                if (sampled_predict_log(boundary_log_count, 32, 1024)) {
                    fprintf(stderr,
                            "[SecureInfer] 到达 CPU KV tensor 边界，停止局部预测: 当前=%p, predicted=%p\n",
                            src, predicted_next);
                }
                if (secure_infer_linear_fast_fallback_enabled()) {
                    secure_infer_retire_prediction_generation(predictor.generation);
                    predictor.generation++;
                }
                predictor.linear_mode = false;
                predictor.linear_stride = 0;
                predictor.linear_expected_src = nullptr;
                predictor.linear_prefetch_until = nullptr;
                predictor.linear_count = 0;
                predictor.read_only_swap_record.clear();
                predictor.read_only_swap_record.push_back(std::make_pair(src, count));
                assign_to_workers(dst, src, count, false, false, true, false, false, stream);
            }
            goto POST_PROCESS;
        }

        // ==================== 情况 2: 重复循环模式（原有） ====================
        // if (predictor.read_only_swap_profiled) {
        //     auto &record = predictor.read_only_swap_record;
        //     bool hit = src == record[predictor.read_only_swap_cur_idx].first &&
        //                count == record[predictor.read_only_swap_cur_idx].second;

        //     if (!hit) goto SERIAL_FALLBACK;

        //     predictor.read_only_swap_cur_idx = (predictor.read_only_swap_cur_idx + 1) % record.size();
        //     predictor.read_only_swap_pred_idx = (predictor.read_only_swap_pred_idx + 1) % record.size();

        //     auto &entry = record[predictor.read_only_swap_pred_idx];
        //     assign_to_workers(nullptr, entry.first, entry.second, false, true, false, false, false, stream);
        //     assign_to_workers(dst, src, count, false, false, true, false, false, stream);
        //     goto POST_PROCESS;
        // }

        // // ==================== 情况 3: 其他已知模式 ====================
        // else if (predictor.other_swap_set.find(std::make_pair(src, count)) != predictor.other_swap_set.end()) {
        //     predictor.other_swap_set.erase(std::make_pair(src, count));
        //     assign_to_workers(nullptr, src, count, false, true, false, false, false, stream);
        //     assign_to_workers(dst, src, count, false, false, true, false, false, stream);
        //     goto POST_PROCESS;
        // }

        // ==================== 情况 4: 模式发现阶段（新增线性检测） ====================
        else if (!predictor.read_only_swap_profiled && !predictor.linear_mode) {
            // fprintf(stderr, "[SecureInfer] 模式发现中 (size=%zu)\n", count);

            auto &record = predictor.read_only_swap_record;
            record.push_back(std::make_pair(src, count));

            // --- 线性模式检测 ---
            if (record.size() >= 2) {
                auto& prev = record[record.size() - 2];
                if (count == prev.second &&
                    (uint8_t*)src > (uint8_t*)prev.first &&
                    ((uint8_t*)src - (uint8_t*)prev.first) == count) {
                    predictor.linear_count++;
                    if (predictor.linear_count >= predictor.LINEAR_THRESHOLD) {
                        const void* predicted_next = (const uint8_t*)src + count;
                        if (host_prediction_in_same_range(src, predicted_next, count)) {
                            predictor.linear_mode = true;
                            predictor.linear_stride = count;
                            predictor.linear_expected_src = predicted_next;
                            predictor.linear_prefetch_until = nullptr;
                            static std::atomic<uint64_t> enter_log_count{0};
                            if (sampled_predict_log(enter_log_count, 32, 1024)) {
                                fprintf(stderr, "[SecureInfer] ✅ 进入线性预测模式: 当前=%p → 下一个=%p (stride=%zu, threshold=%d)\n",
                                        src, predicted_next, count, predictor.LINEAR_THRESHOLD);
                            }
                            schedule_linear_prefetches(src, count, stream);
                        } else {
                            static std::atomic<uint64_t> no_start_log_count{0};
                            if (sampled_predict_log(no_start_log_count, 32, 1024)) {
                                fprintf(stderr,
                                        "[SecureInfer] 连续序列已到 CPU KV tensor 边界，不启动预测: 当前=%p, predicted=%p\n",
                                        src, predicted_next);
                            }
                            predictor.linear_count = 0;
                            predictor.linear_prefetch_until = nullptr;
                            predictor.read_only_swap_record.clear();
                            predictor.read_only_swap_record.push_back(std::make_pair(src, count));
                        }
                    }
                } else {
                    predictor.linear_count = 0;
                    predictor.linear_prefetch_until = nullptr;
                }
            }

            // 先把未来块交给预测 worker，再走当前块的原地串行 fallback。
            // 这样 CPU 预测加密能和当前块 H2D/NPU pipeline 重叠；比盲目
            // 增大 window/worker 更克制，也更不容易重新触发 ACL backpressure。
            inline_h2d_encrypt_commit(dst, destMax, src, count, stream,
                                      "线性学习阶段 miss");

            // --- 原有重复模式检测 ---
            // bool found = true;
            // const int repeat = 32;
            // auto size = record.size();
            // if (size < repeat * 2 || (size % 2 == 1)) found = false;
            // else {
            //     auto mid = size / 2;
            //     for (int i = 0; i < size / 2; i++) {
            //         if (record[i] != record[mid + i]) { found = false; break; }
            //     }
            // }

            // if (found) {
            //     fprintf(stderr, "[SecureInfer] 发现重复模式! 开启预测优化\n");
            //     for (int i = 0; i < size / 2; i++) record.pop_back();
            //     predictor.read_only_swap_profiled = true;
            //     predictor.read_only_swap_cur_idx = (size / 2) % record.size();
            //     predictor.read_only_swap_pred_idx = (predictor.read_only_swap_cur_idx + 1) % record.size();

            //     auto &entry1 = record[predictor.read_only_swap_pred_idx];
            //     assign_to_workers(nullptr, entry1.first, entry1.second, false, true, false, false, false, stream);
            // }
            goto POST_PROCESS;
        }

    SERIAL_FALLBACK:
        inline_h2d_encrypt_commit(dst, destMax, src, count, stream,
                                  "预测未命中 fallback");

    POST_PROCESS:
        predictor.unlock();
        return ACL_SUCCESS;
    }

    // =================================================================
    // 5. 解密路径 (Device -> Host) - stream-ordered asynchronous host decrypt
    // =================================================================
    if (enable_d2h_encrypt && kind == ACL_MEMCPY_DEVICE_TO_HOST &&
        count >= decrypt_threshold_sz &&
        host_prediction_in_same_range(dst, dst, count)) {
        std::lock_guard<std::mutex> d2h_lock(g_d2h_crypto_lock);
        const bool d2h_gcm_enabled = secure_infer_d2h_gcm_enabled();
        const uint64_t d2h_submit_start_us = secure_infer_now_us();
        static std::atomic<uint64_t> d2h_log_count{0};
        bool log_d2h = sampled_predict_log(d2h_log_count, 16, 4096);
        if (log_d2h) {
            fprintf(stderr, "[SecureInfer] D2H+Decrypt START: src=%p, dst=%p, size=%zu\n", src, dst, count);
        }

        if (m_ctx_metadata.empty()) {
            return real_aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
        }

        auto &meta = *m_ctx_metadata.begin()->second;
        if (meta.dev_key == nullptr) {
            real_aclrtMalloc(&meta.dev_key, 32);
            // CTR.cpp reads an aligned 32-byte IV record.
            real_aclrtMalloc(&meta.dev_iv, 32);
            real_aclrtMemcpyAsync(meta.dev_key, 32, meta.key, 32, ACL_MEMCPY_HOST_TO_DEVICE, stream);
            real_aclrtMemcpyAsync(meta.dev_iv, 16, meta.init_iv, 16, ACL_MEMCPY_HOST_TO_DEVICE, stream);
            fprintf(stderr, "[SecureInfer] ✅ NPU Key/IV 内存初始化完成\n");
        }
        g_dev_key = meta.dev_key;
        g_dev_iv = meta.dev_iv;

        const uint64_t d2h_ticket = secure_infer_new_transfer_ticket();
        unsigned char d2h_iv[iv_length];
        secure_infer_derive_transfer_iv(meta.init_iv, d2h_ticket, 0, d2h_iv);
        unsigned char d2h_aad[gcm_aad_length] = {};
        const bool d2h_aad_valid = !secure_infer_gcm_aad_enabled() ||
            secure_infer_build_gcm_aad(dst, count, PIPE_GCM_DIRECTION_D2H,
                                  d2h_ticket, 0, d2h_aad);
        if (!d2h_aad_valid) {
            fprintf(stderr,
                    "[SecureInfer FATAL] cannot build D2H GCM metadata for %p/%zu\n",
                    dst, count);
            return ACL_ERROR_INTERNAL_ERROR;
        }
        if (d2h_gcm_enabled &&
            !secure_infer_gcm_replay_reserve(d2h_ticket)) {
            fprintf(stderr,
                    "[SecureInfer SECURITY] duplicate D2H transfer ticket rejected: %lu\n",
                    static_cast<unsigned long>(d2h_ticket));
            return ACL_ERROR_INTERNAL_ERROR;
        }
        d2h_stream_pipeline* d2h_pipeline = nullptr;
        d2h_pipeline_slot* d2h_slot = nullptr;
        void* d2h_temp_in = nullptr;
        void* d2h_temp_out = nullptr;

        // In the production path one slot holds a complete tensor batch.  All
        // slices are encrypted into distinct offsets first; group_end then
        // hands the batch to a dedicated PCIe stream.  This is the boundary
        // that permits NPU(i+1), PCIe(i), and CPU(i-1) to overlap.
        if (g_active_d2h_group != nullptr && d2h_stage_pipeline_enabled()) {
            auto* group = g_active_d2h_group;
            d2h_pipeline = get_d2h_pipeline(stream);
            if (d2h_pipeline != nullptr) {
                if (group->pipeline == nullptr) {
                    group->pipeline = d2h_pipeline;
                    d2h_slot = &d2h_pipeline->slots[
                        d2h_pipeline->next_slot++ % d2h_pipeline->slots.size()];
                    size_t reserve_tasks = std::max<size_t>(1, group->expected_tasks);
                    // Keep GHASH tag-mask input and tag output disjoint.  The
                    // AICPU executor does not guarantee input/output aliasing,
                    // even though a byte-wise in-place implementation appears
                    // safe at the C++ level.
                    const bool metadata_valid = !d2h_gcm_enabled ||
                        reserve_tasks <= SIZE_MAX / (32 + gcm_aad_length);
                    const size_t metadata_bytes = metadata_valid &&
                        d2h_gcm_enabled
                        ? reserve_tasks * (32 + gcm_aad_length) : 0;
                    const bool size_valid = metadata_valid &&
                        reserve_tasks <= SIZE_MAX / count &&
                        reserve_tasks * count <= SIZE_MAX - metadata_bytes;
                    const size_t required_bytes = size_valid
                        ? reserve_tasks * count + metadata_bytes : 0;
                    bool slot_ready = size_valid && ensure_d2h_pipeline_slot(
                        *d2h_slot, required_bytes, d2h_pipeline->copy_stream);
                    if (!slot_ready && size_valid) {
                        // Large models can leave room for one tensor staging
                        // slot but not the configured 2--3.  Reuse an already
                        // allocated slot after its copy_done event instead of
                        // failing the store (and crashing the framework after
                        // a partial swap).  This is caruntimety-aware scheduling,
                        // not utilization sampling: correctness and bounded
                        // memory take priority over overlap depth.
                        for (auto& fallback_slot : d2h_pipeline->slots) {
                            if (&fallback_slot == d2h_slot) continue;
                            if (ensure_d2h_pipeline_slot(
                                    fallback_slot, required_bytes,
                                    d2h_pipeline->copy_stream)) {
                                d2h_slot = &fallback_slot;
                                slot_ready = true;
                                fprintf(stderr,
                                        "[SecureInfer] D2H staging pressure: reusing an allocated slot; overlap depth reduced\n");
                                break;
                            }
                        }
                    }
                    if (!slot_ready) {
                        fprintf(stderr,
                                "[SecureInfer ERROR] tensor-batch D2H staging allocation failed: tasks=%zu payload=%zu tags=%zu\n",
                                reserve_tasks, size_valid ? reserve_tasks * count : 0,
                                metadata_bytes);
                        return ACL_ERROR_INTERNAL_ERROR;
                    }
                    group->slot = d2h_slot;
                    group->payload_caruntimety = reserve_tasks * count;
                    group->ready_stream = d2h_pipeline->copy_stream;
                    auto& batch = *group->batch;
                    if (std::find(batch.streams.begin(), batch.streams.end(), stream)
                        == batch.streams.end()) {
                        batch.streams.push_back(stream);
                        batch.copy_streams.push_back(d2h_pipeline->copy_stream);
                    }
                } else if (group->pipeline != d2h_pipeline) {
                    fprintf(stderr,
                            "[SecureInfer ERROR] one D2H tensor group used multiple streams\n");
                    return ACL_ERROR_INTERNAL_ERROR;
                }
                d2h_slot = group->slot;
                if (group->staging_bytes > group->payload_caruntimety - count) {
                    fprintf(stderr,
                            "[SecureInfer ERROR] D2H tensor group exceeded reserved staging\n");
                    return ACL_ERROR_INTERNAL_ERROR;
                }
                d2h_temp_in = const_cast<void*>(src);
                d2h_temp_out = static_cast<uint8_t*>(d2h_slot->temp_out) +
                               group->staging_bytes;
                group->staging_bytes += count;
            }
        }

        if (d2h_pipeline == nullptr) {
            // Compatibility path: one staging pair per framework stream and
            // strictly ordered D2D -> AES -> D2H operations.
            auto &d2h_staging = g_d2h_staging_by_stream[stream];
            if (d2h_staging.bytes < count || d2h_staging.temp_out == nullptr) {
                if (d2h_staging.temp_in || d2h_staging.temp_out) {
                    g_retired_d2h_staging.emplace_back(
                        d2h_staging.temp_in, d2h_staging.temp_out);
                }
                constexpr size_t kMaxIntegrityBytes = gcm_metadata_slab_bytes;
                void* new_temp_out = nullptr;
                aclError alloc_out = real_aclrtMalloc(
                    &new_temp_out, count + kMaxIntegrityBytes);
                if (alloc_out != ACL_SUCCESS || new_temp_out == nullptr) {
                    fprintf(stderr, "[SecureInfer ERROR] async D2H staging allocation failed\n");
                    if (new_temp_out) real_aclrtFree(new_temp_out);
                    return real_aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
                }
                d2h_staging.temp_in = nullptr;
                d2h_staging.temp_out = new_temp_out;
                d2h_staging.bytes = count;
            }
            d2h_temp_in = const_cast<void*>(src);
            d2h_temp_out = d2h_staging.temp_out;
            if (g_active_d2h_group != nullptr) {
                auto* group = g_active_d2h_group;
                group->ready_stream = stream;
                auto& batch = *group->batch;
                if (std::find(batch.streams.begin(), batch.streams.end(), stream)
                    == batch.streams.end()) {
                    batch.streams.push_back(stream);
                    batch.copy_streams.push_back(stream);
                }
            }
        }

        uint8_t* d2h_device_tag_mask = nullptr;
        uint8_t* d2h_device_tag = nullptr;
        uint8_t* d2h_device_aad = nullptr;
        bool d2h_fused_gcm = false;
        if (d2h_gcm_enabled) {
            if (!secure_infer_prepare_gcm_backend(meta, stream)) {
                fprintf(stderr,
                        "[SecureInfer FATAL] D2H GCM backend/material preparation failed\n");
                std::abort();
            }
            if (d2h_pipeline != nullptr) {
                const size_t tag_index = g_active_d2h_group->tasks.size();
                d2h_device_tag_mask =
                    static_cast<uint8_t*>(d2h_slot->temp_out) +
                    g_active_d2h_group->payload_caruntimety + tag_index * 16;
                d2h_device_tag =
                    static_cast<uint8_t*>(d2h_slot->temp_out) +
                    g_active_d2h_group->payload_caruntimety +
                    g_active_d2h_group->expected_tasks * 16 + tag_index * 16;
                d2h_device_aad =
                    static_cast<uint8_t*>(d2h_slot->temp_out) +
                    g_active_d2h_group->payload_caruntimety +
                    g_active_d2h_group->expected_tasks * 32 +
                    tag_index * gcm_aad_length;
            } else {
                d2h_device_tag_mask =
                    static_cast<uint8_t*>(d2h_temp_out) + count;
                d2h_device_tag = d2h_device_tag_mask + 16;
                d2h_device_aad = d2h_device_tag_mask + 32;
            }
            d2h_fused_gcm = g_launch_aicpu_gcm_encrypt_tag != nullptr &&
                            d2h_fused_gcm_enabled();
            if (secure_infer_gcm_aad_enabled()) {
                if (real_aclrtMemcpyAsync(
                        d2h_device_aad, gcm_aad_length, d2h_aad,
                        gcm_aad_length, ACL_MEMCPY_HOST_TO_DEVICE,
                        stream) != ACL_SUCCESS) {
                    return ACL_ERROR_INTERNAL_ERROR;
                }
            }
            if (!d2h_fused_gcm) {
                if (!enqueue_gcm_tag_mask(
                        stream, d2h_device_tag_mask, meta.dev_key, d2h_iv)) {
                    fprintf(stderr,
                            "[SecureInfer FATAL] D2H GCM tag-mask generation failed\n");
                    std::abort();
                }
            }
        }

        // Step1: the NPU kernel reads the original KV slice directly and
        // writes packed ciphertext into the slot.  This fuses staging with
        // encryption and avoids one plaintext-sized D2D copy.
        if (log_d2h) {
            fprintf(stderr,
                    "[SecureInfer] D2H Step1 direct encrypt-and-pack: src=%p -> tmp_out=%p, size=%zu\n",
                    src, d2h_temp_out, count);
        }
        uint64_t copy_enqueue_us = 0;
        uint64_t kernel_enqueue_us = 0;
        uint64_t enqueue_start_us = secure_infer_now_us();

        // Step2: Kernel
        if (log_d2h) {
            fprintf(stderr, "[SecureInfer] D2H Step2 Kernel: tmp_in=%p, key=%p, iv=%p, tmp_out=%p, size=%zu\n",
                    d2h_temp_in, meta.dev_key, meta.dev_iv, d2h_temp_out, count);
        }
        if (d2h_fused_gcm) {
            uint32_t iv_words[4] = {0, 0, 0, 0};
            pack_inline_iv(d2h_iv, iv_words);
            enqueue_start_us = secure_infer_now_us();
            if (g_launch_aicpu_gcm_encrypt_tag(
                    stream, d2h_temp_in, meta.dev_key, meta.dev_ghash_h,
                    d2h_temp_out, d2h_device_tag, count,
                    iv_words[0], iv_words[1], iv_words[2], iv_words[3],
                    d2h_device_aad,
                    secure_infer_gcm_aad_enabled() ? gcm_aad_length : 0) != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] fused D2H AES-GCM encrypt/tag launch failed\n");
                std::abort();
            }
            kernel_enqueue_us += secure_infer_now_us() - enqueue_start_us;
        } else if (load_ghash_library() &&
                   g_launch_aicpu_aes_ctr != nullptr && meta.dev_key) {
            uint32_t iv_words[4] = {0, 0, 0, 0};
            pack_inline_iv(d2h_iv, iv_words);
            enqueue_start_us = secure_infer_now_us();
            if (g_launch_aicpu_aes_ctr(
                    stream, d2h_temp_in, meta.dev_key, d2h_temp_out,
                    count, iv_words[0], iv_words[1],
                    iv_words[2], iv_words[3]) != 0) {
                fprintf(stderr,
                        "[SecureInfer FATAL] D2H AICPU AES-CTR launch failed\n");
                std::abort();
            }
            kernel_enqueue_us += secure_infer_now_us() - enqueue_start_us;
        } else {
            fprintf(stderr, "[SecureInfer ERROR] D2H AES kernel is unavailable\n");
            return ACL_ERROR_INTERNAL_ERROR;
        }

        if (d2h_gcm_enabled && !d2h_fused_gcm) {
            // When an independent GHASH stream is available, defer GHASH
            // to group_end so it can overlap with D2H ciphertext copy.
            // Otherwise launch inline on the framework D2H stream.
            if (d2h_pipeline != nullptr &&
                d2h_pipeline->ghash_stream != nullptr &&
                g_active_d2h_group != nullptr) {
                // Defer: store params, launch in group_end on ghash_stream.
                // The task is appended below; stash the tag addresses now.
                g_active_d2h_group->ghash_tag_out_stack.push_back(
                    d2h_device_tag);
                g_active_d2h_group->ghash_tag_mask_stack.push_back(
                    d2h_device_tag_mask);
                g_active_d2h_group->ghash_aad_stack.push_back(
                    d2h_device_aad);
                if (g_active_d2h_group->dev_ghash_h == nullptr) {
                    g_active_d2h_group->dev_ghash_h = meta.dev_ghash_h;
                }
            } else {
                int ghash_ret = g_launch_aicpu_ghash(
                    stream,
                    d2h_temp_out,
                    meta.dev_ghash_h,
                    d2h_device_tag,
                    d2h_device_aad,
                    d2h_device_tag_mask,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    count,
                    secure_infer_gcm_aad_enabled() ? gcm_aad_length : 0,
                    PIPE_GHASH_FINALIZE_TAG);
                if (ghash_ret != 0) {
                    fprintf(stderr,
                            "[SecureInfer FATAL] D2H AICPU GHASH launch "
                            "rejected ret=%d\n",
                            ghash_ret);
                    std::abort();
                }
            }
        }

        // Step3 is deferred to group_end for the independent PCIe stream.  In
        // compatibility mode it remains on the framework stream.
        if (d2h_pipeline == nullptr) {
            if (log_d2h) {
                fprintf(stderr, "[SecureInfer] D2H Step3 D2H: tmp_out=%p -> dst=%p, size=%zu\n", d2h_temp_out, dst, count);
            }
            enqueue_start_us = secure_infer_now_us();
            real_aclrtMemcpyAsync(dst, destMax, d2h_temp_out, count,
                                  ACL_MEMCPY_DEVICE_TO_HOST, stream);
            copy_enqueue_us += secure_infer_now_us() - enqueue_start_us;
        }

        if (d2h_async_enabled()) {
            if (g_active_d2h_group) {
                auto& batch = *g_active_d2h_batch;
                if (!batch.profile_checked) {
                    batch.profile_active = capture_d2h_profile_run(
                        batch.profile_run_id, sizeof(batch.profile_run_id));
                    batch.profile_checked = true;
                }

                d2h_host_decrypt_task task;
                task.dst = dst;
                task.device_ciphertext = d2h_temp_out;
                task.bytes = count;
                memcpy(task.key, meta.key, key_length);
                memcpy(task.iv, d2h_iv, iv_length);
                task.aad_valid = d2h_aad_valid;
                if (task.aad_valid) memcpy(task.aad, d2h_aad, gcm_aad_length);
                if (d2h_gcm_enabled) {
                    const size_t tag_index =
                        g_active_d2h_group->tasks.size();
                    task.auth_tag =
                        g_active_d2h_group->host_tags + tag_index * 16;
                }
                task.submit_start_us = d2h_submit_start_us;
                task.copy_enqueue_us = copy_enqueue_us;
                task.kernel_enqueue_us = kernel_enqueue_us;
                task.profile_active = batch.profile_active;
                if (task.profile_active) {
                    snprintf(task.profile_run_id,
                             sizeof(task.profile_run_id),
                             "%s", batch.profile_run_id);
                }
                g_active_d2h_group->tasks.emplace_back(std::move(task));
                if (log_d2h) {
                    fprintf(stderr,
                            "[SecureInfer] D2H batched decrypt staged: dst=%p, size=%zu\n",
                            dst, count);
                }
                return ACL_SUCCESS;
            }

            auto* task = new (std::nothrow) d2h_host_decrypt_task();
            if (task != nullptr) {
                task->dst = dst;
                task->bytes = count;
                memcpy(task->key, meta.key, key_length);
                memcpy(task->iv, d2h_iv, iv_length);
                task->aad_valid = d2h_aad_valid;
                if (task->aad_valid) memcpy(task->aad, d2h_aad, gcm_aad_length);
                if (d2h_gcm_enabled) {
                    task->auth_tag = task->inline_tag;
                    aclError tag_copy_ret = real_aclrtMemcpyAsync(
                        task->inline_tag, 16, d2h_device_tag, 16,
                        ACL_MEMCPY_DEVICE_TO_HOST, stream);
                    if (tag_copy_ret != ACL_SUCCESS) {
                        delete task;
                        return tag_copy_ret;
                    }
                }
                task->submit_start_us = d2h_submit_start_us;
                task->copy_enqueue_us = copy_enqueue_us;
                task->kernel_enqueue_us = kernel_enqueue_us;
                task->profile_active = capture_d2h_profile_run(
                    task->profile_run_id, sizeof(task->profile_run_id));
                aclError callback_ret = real_aclrtLaunchHostFunc(
                    stream, d2h_host_decrypt_callback, task);
                if (callback_ret == ACL_SUCCESS) {
                    profile_count_d2h_kv(count);
                    if (log_d2h) {
                        fprintf(stderr,
                                "[SecureInfer] D2H async decrypt queued: dst=%p, size=%zu\n",
                                dst, count);
                    }
                    return ACL_SUCCESS;
                }
                if (d2h_gcm_enabled) {
                    // The tag DMA targets task-owned storage.  If HostFunc
                    // submission fails, drain it before releasing the task.
                    real_aclrtSynchronizeStream(stream);
                }
                delete task;
                fprintf(stderr,
                        "[SecureInfer WARN] aclrtLaunchHostFunc failed (%d), using synchronous D2H fallback\n",
                        static_cast<int>(callback_ret));
            }
        }

        unsigned char synchronous_tag[16] = {0};
        if (d2h_gcm_enabled) {
            aclError tag_copy_ret = real_aclrtMemcpyAsync(
                synchronous_tag, 16, d2h_device_tag, 16,
                ACL_MEMCPY_DEVICE_TO_HOST, stream);
            if (tag_copy_ret != ACL_SUCCESS) return tag_copy_ret;
        }

        // Safe fallback when host functions are unavailable.  Keeping this
        // path is essential: returning before CPU plaintext is ready would let
        // vLLM mark ciphertext as a completed external-cache block.
        real_aclrtSynchronizeStream(stream);

        // The NPU transform above protects the D2H link.  The CPU KV cache is
        // inside the trusted host-memory boundary used by this project, so it
        // must receive plaintext.  Without this inverse transform, a later
        // H2D path encrypts already-encrypted KV and inference reads corrupt
        // cache values even though all transfers themselves succeed.
        EVP_CIPHER_CTX* host_ctx = real_EVP_CIPHER_CTX_new();
        if (host_ctx == nullptr) {
            fprintf(stderr, "[SecureInfer ERROR] D2H host decrypt context allocation failed\n");
            return ACL_ERROR_INTERNAL_ERROR;
        }
        int host_updated_len = 0;
        int host_final_len = 0;
        bool host_ok = false;
        if (d2h_gcm_enabled) {
            host_ok =
                real_EVP_DecryptInit_ex(host_ctx, EVP_aes_128_gcm(), nullptr,
                                        nullptr, nullptr) == 1 &&
                real_EVP_CIPHER_CTX_ctrl(host_ctx, EVP_CTRL_GCM_SET_IVLEN,
                                         12, nullptr) == 1 &&
                real_EVP_DecryptInit_ex(host_ctx, nullptr, nullptr,
                                        meta.key, d2h_iv) == 1;
            if (host_ok && d2h_aad_valid) {
                int aad_len = 0;
                host_ok = real_EVP_DecryptUpdate(
                    host_ctx, nullptr, &aad_len, d2h_aad,
                    static_cast<int>(gcm_aad_length)) == 1;
            }
            host_ok = host_ok && real_EVP_DecryptUpdate(
                host_ctx, static_cast<unsigned char*>(dst),
                &host_updated_len,
                static_cast<const unsigned char*>(dst),
                static_cast<int>(count)) == 1 &&
                real_EVP_CIPHER_CTX_ctrl(host_ctx, EVP_CTRL_GCM_SET_TAG,
                                         16, synchronous_tag) == 1 &&
                real_EVP_DecryptFinal_ex(
                    host_ctx,
                    static_cast<unsigned char*>(dst) + host_updated_len,
                    &host_final_len) == 1;
        } else {
            host_ok =
                real_EVP_EncryptInit_ex(host_ctx, EVP_aes_128_ctr(), 0,
                                        meta.key, d2h_iv) == 1 &&
                real_EVP_EncryptUpdate(
                    host_ctx, static_cast<unsigned char*>(dst),
                    &host_updated_len,
                    static_cast<const unsigned char*>(dst),
                    static_cast<int>(count)) == 1 &&
                real_EVP_EncryptFinal_ex(
                    host_ctx,
                    static_cast<unsigned char*>(dst) + host_updated_len,
                    &host_final_len) == 1;
        }
        real_EVP_CIPHER_CTX_free(host_ctx);
        if (!host_ok) {
            memset(dst, 0, count);
            fprintf(stderr,
                    "[SecureInfer FATAL] synchronous D2H authentication failed\n");
            std::abort();
        }
        if (log_d2h) {
            fprintf(stderr, "[SecureInfer] D2H+Decrypt END: dst=%p, size=%zu\n", dst, count);
        }
        profile_count_d2h_kv(count);

        return ACL_SUCCESS;
    }

    // 6. 默认回传
    if (count >= 100000) {
        static std::atomic<uint64_t> d2h_plain_log_count{0};
        if (sampled_predict_log(d2h_plain_log_count, 16, 4096)) {
            fprintf(stderr, "[SecureInfer] D2H: dst=%p, src=%p, size=%zu\n", dst, src, count);
        }
    }
    return real_aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
}





HOOK_C_API HOOK_DECL_EXPORT
void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx)
{
    // 真实的 OpenSSL 会处理 nullptr，我们也一样
    if (ctx == nullptr) {
        real_EVP_CIPHER_CTX_free(ctx);
        return;
    }

    // [!!! 关键修复 !!!]
    // 在调用真实的 free 之前，必须从我们的 map 中移除这个 ctx，
    // 以防止“用后释放” (Use-After-Free) 漏洞。
    // 同时，必须使用互斥锁来保证线程安全。
    {
        std::lock_guard<std::mutex> lock(g_ctx_map_mutex);

        m_ctx_metadata.erase(ctx);
        m_ctx_dmetadata.erase(ctx);
    }
    // 互斥锁在这里自动释放

    // 现在 ctx 已经从我们的 map 中安全移除，
    // 我们可以安全地调用真实的 free 函数了。
    real_EVP_CIPHER_CTX_free(ctx);
}


// 修正后的桥接函数
extern "C" HOOK_DECL_EXPORT aclError aclrtUseStreamResInCurrentThread(aclrtStream stream) {
    // 定义函数指针类型，匹配 aclError(aclrtStream)
    typedef aclError (*func_ptr)(aclrtStream);

    // 1. 优先寻找带 Impl 后缀的真实实现（CANN 8.5.1 内部实际使用的符号）
    static func_ptr real_func = (func_ptr)dlsym(RTLD_NEXT, "aclrtUseStreamResInCurrentThreadImpl");

    if (!real_func) {
        // 2. 如果没找到 Impl，尝试找不带后缀的原始符号
        // 注意：这里必须用 RTLD_NEXT 避免死循环指向自己
        real_func = (func_ptr)dlsym(RTLD_NEXT, "aclrtUseStreamResInCurrentThread");
    }

    if (real_func) {
        return real_func(stream);
    }

    // 3. 兜底逻辑：如果底层库里真的两个符号都没有，
    // 我们返回成功(0)，防止 Python 因为加载 .so 失败而直接崩溃
    return ACL_SUCCESS;
}

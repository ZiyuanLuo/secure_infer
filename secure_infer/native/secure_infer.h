/*
* 文件名: secure_infer.h (已修改为 ACL 和 ARM 架构)
* 职责: 定义项目的核心数据结构和共享头文件
*/
#pragma once

#include <openssl/ssl.h>
// #include <cuda.h> // <-- [已替换]
#include <acl/acl_rt.h> // <-- [新添加] 华为 AscendCL 运行时头文件
#include <map>
#include <sys/mman.h>
#include <cassert>
#include <unistd.h>
#include <thread>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <stdio.h>
#include <atomic>
#include <cstdint>
// #include <immintrin.h> // <-- [已替换]

// [!!! 关键修改 !!!]
// 替换 <immintrin.h> 为 ARM 和 x86 的条件编译
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h> // Intel/AMD x86 架构
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>  // ARM (华为 NPU) 架构
#else
    #warning "Target architecture not recognized. SIMD optimizations may be unavailable."
#endif
// [!!! 修改结束 !!!]

#include <cstdlib>
#include <memory>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h> // 注意：您包含了两次 unistd.h，这里保留

extern size_t encrypt_magic_sz;
extern size_t decrypt_magic_sz;
const static size_t encrypt_threshold_sz = 100 * 1024;
const static size_t decrypt_threshold_sz = 100 * 1024;
const static size_t decrypt_threshold_top = 0x80000;
const static size_t block_unit = 1 << 20;
constexpr static size_t key_length = 32;
constexpr static size_t iv_length = 16;
constexpr static int profile_bytes = 3;
constexpr static uint64_t iv_mod = 1 << (8 * profile_bytes);
// Canonical, fixed-size metadata authenticated by AES-GCM.  The byte layout
// is serialized explicitly in secure_infer.cpp so host/device endianness and C++
// struct padding never become part of the security contract.
constexpr static size_t gcm_aad_length = 64;
constexpr static size_t gcm_metadata_slab_bytes = 4096;
enum : uint8_t {
    PIPE_GCM_DIRECTION_H2D = 1,
    PIPE_GCM_DIRECTION_D2H = 2,
};
const static char iv_profile_path[] = "profile_iv.txt";
extern std::mutex g_ctx_map_mutex;
extern "C" void aes_cipher_kernel_npu(void *input_x, void *key, void *iv, void *output, uint32_t totalLength, uint32_t mode);
using iv_t = uint64_t;

// [新增] 定义函数指针类型
typedef void (*LaunchAESKernel_t)(void*, void*, void*, void*, void*, uint32_t);

// [新增] 声明全局变量
extern LaunchAESKernel_t g_launch_kernel;
extern void* g_dev_key; // 偷懒做法：暂时用全局变量存 NPU Key 地址
extern void* g_dev_iv;  // 偷懒做法：暂时用全局变量存 NPU IV 地址


struct encryption_entry;
struct buffer_allocator {
    std::vector<unsigned char *> buffers;
    const static size_t buffer_size = block_unit;
    unsigned char *alloc() {
        if (buffers.empty()) {
            return new unsigned char[buffer_size];
        } else {
            auto ptr = buffers.back();
            buffers.pop_back();
            return ptr;
        }
    }
    void free(unsigned char *ptr) {
        buffers.push_back(ptr);
    }
};

extern int memcpy_thread_num;
static constexpr int memcpy_thread_num_max = 5;
struct memcpy_entry {
    volatile bool busy;
    const void *src;
    void *dst;
    size_t size;
    int core;
};

// struct encrypt_metadata 定义 (保持原样)
struct encrypt_metadata {
    unsigned char key[key_length];
    unsigned char init_iv[iv_length];
    iv_t cur_iv_offset;
    EVP_CIPHER_CTX *encrypt_ctx;
    bool iv_inited;
    int commit_worker_id;

    // Predictor
    std::map<iv_t, encryption_entry *> m_iv_encentry;

    // For openssl
    int remain;

    // Buffer allocator
    buffer_allocator allocator;

    // For memcpy
    std::thread *memcpy_threads[memcpy_thread_num_max];
    memcpy_entry memcpy_entries[memcpy_thread_num_max];

    // [新增字段] NPU 上的内存地址
    void* dev_key = nullptr;
    void* dev_iv = nullptr;
    void* dev_ghash_h = nullptr;
    bool gcm_backend_ready = false;
};

// struct decrypt_metadata 定义 (保持原样)
struct decrypt_metadata {
    unsigned char key[key_length];
    unsigned char cur_iv[iv_length];
    unsigned char cur_tag[16];
    unsigned char *cur_buffer;

    // For openssl
    int remain;

    // Buffer allocator
    buffer_allocator allocator;

    // For memcpy
    std::thread *memcpy_threads[memcpy_thread_num_max];
    memcpy_entry memcpy_entries[memcpy_thread_num_max];
};

// extern map 声明 (保持原样)
extern std::map<EVP_CIPHER_CTX *, std::shared_ptr<encrypt_metadata>> m_ctx_metadata;
extern std::map<size_t, EVP_CIPHER_CTX *> m_magic_encctx;
extern std::set<size_t> s_magic;
// Register the logical layout of a host KV tensor.  The history predictor
// still observes only addresses; this metadata is used solely to construct a
// stable authenticated descriptor when a transfer is actually encrypted.
extern "C" void secure_infer_register_host_range_ex(const void* base, size_t bytes,
                                                uint32_t layer_id,
                                                uint32_t tensor_kind,
                                                size_t block_bytes);
// Highest predictor generation whose logical H2D transfer scope has finished
// submitting all real demands. Commit workers may reclaim unused predictions
// at or below this watermark after their NPU ready event completes.
extern std::atomic<uint64_t> g_retired_transfer_generation;
// Descriptor-prefetch outcome counters.  These describe CPU ciphertext
// readiness, not NPU keystream readiness (which is tracked separately once a
// real device keystream pool is connected).
extern std::atomic<uint64_t> g_transfer_prefetch_ready;
extern std::atomic<uint64_t> g_transfer_prefetch_busy_fallback;
extern std::atomic<uint64_t> g_transfer_prefetch_missing_fallback;

// Opt-in engineering branch for the hook-only linear-address predictor.
// The default path remains unchanged unless PIPE_LINEAR_FAST_FALLBACK=1.
bool secure_infer_linear_fast_fallback_enabled();
void secure_infer_retire_prediction_generation(uint64_t generation);
// Retire speculative transfer records at a logical job boundary while
// preserving the observation state of history-based predictors.  The full
// invalidation API below remains the explicit session/reset operation.
extern "C" void secure_infer_retire_transfer_prefetches();
// A speculative ticket whose demand has already taken the serial fallback no
// longer has any useful CPU work.  Queue and AES workers consult this marker
// so stale encryption cannot delay later, still-feasible predictions.
void secure_infer_cancel_speculative_ticket(uint64_t transfer_ticket);
bool secure_infer_speculative_ticket_cancelled(uint64_t transfer_ticket);

// Optional AES-GCM CPU ciphertext lookahead.  The transfer-layer descriptor
// reserves an exact ticket before the real H2D arrives; workers may encrypt
// the host block (including its authentication tag) in the meantime.  Demand
// never waits for this pool: a non-ready record uses the verified inline
// encryption path instead.
bool secure_infer_gcm_cpu_cipher_prefetch_enabled();
bool secure_infer_gcm_schedule_cpu_ciphertext(const void* src, size_t bytes,
                                         uint64_t transfer_ticket,
                                         uint64_t generation);
bool secure_infer_gcm_take_ready_cpu_ciphertext(uint64_t transfer_ticket,
                                           const void* src, size_t bytes,
                                           unsigned char* destination,
                                           size_t destination_bytes,
                                           uint64_t* crypto_us);
// Zero-copy variant for the asynchronous GCM H2D ring.  It hands the pinned
// ciphertext buffer to a slot and keeps its record alive until the slot event
// completes.  Call release only after that event; otherwise the runtime could
// DMA from a recycled host buffer.
bool secure_infer_gcm_handoff_ready_cpu_ciphertext(uint64_t transfer_ticket,
                                              const void* src, size_t bytes,
                                              unsigned char** ciphertext,
                                              size_t* ciphertext_bytes,
                                              uint64_t* crypto_us);
void secure_infer_gcm_release_handed_off_cpu_ciphertext(uint64_t transfer_ticket);
void secure_infer_gcm_cpu_cipher_cleanup();

extern std::map<EVP_CIPHER_CTX *, std::shared_ptr<decrypt_metadata>> m_ctx_dmetadata;
extern std::map<size_t, EVP_CIPHER_CTX *> m_magic_decctx;
extern std::set<size_t> s_magic_dec;
extern std::mutex g_ctx_map_mutex;
// extern function 声明 (保持原样)
extern void encrypt_worker(void *entry);
extern void commit_worker(void *entry);
extern void decrypt_worker(void *entry);
void decrypt_manager(void *entry);

// struct ...task/entry... 定义 (全部保持原样)
struct encryption_task {
    const void *src;
    size_t size;
    size_t iv_increment;
    uint64_t generation = 0;
    // A transfer ticket identifies one logical link-encryption operation.
    // CPU pre-encryption, NPU keystream pre-generation, and the demand miss
    // path must all derive the same IV from this value.
    uint64_t transfer_ticket = 0;
    // Scheduling identity for one history-predicted batch.  This is
    // deliberately separate from transfer_ticket: AES-GCM gives every block
    // an independent crypto ticket/IV, while a predictor may group related
    // submissions for bounded queue management.
    uint64_t prefetch_group_id = 0;
    // Optional logical scope for queue-management experiments.  The serving
    // path does not obtain this value from a future framework block list.
    uint64_t admission_scope = 0;
    // AES-block offset when one logical transfer is split across workers.
    uint64_t transfer_block_offset = 0;
    // Number of physical transfers sharing this ticket's reserved counter
    // range.  It is metadata for batching/admission; every member still has a
    // distinct transfer_block_offset.
    uint32_t ticket_batch_members = 1;
    size_t ticket_batch_bytes = 0;
    // CPU ciphertext preparation is lightweight host work; NPU keystream
    // admission allocates device slabs/events and launches AICPU kernels.
    // Keep their progress independent so an NPU admission deadline never
    // prevents CPU preparation from running ahead.
    bool cpu_prefetch_admitted = false;
    bool npu_prefetch_admitted = false;
};
struct commit_sync_state {
    aclrtEvent event = nullptr;
    std::atomic<bool> submitted{false};
    std::atomic<bool> wait_enqueued{false};
    std::atomic<aclError> status{ACL_SUCCESS};
};
aclrtEvent acquire_commit_sync_event();
void recycle_commit_sync_event(aclrtEvent event);
struct commit_task {
    encryption_task enc_task;
    void *dst;
    bool using_predict;
    bool update_predict_iv;
    std::shared_ptr<commit_sync_state> sync;
    uint64_t profile_start_us = 0;
};
struct encrypt_worker_entry {
    int id;
    int local_id;
    int encrypt_workers_per_commit;
    size_t magic_sz;
    volatile int state;
    std::deque<encryption_task> enc_tasks;
    std::deque<commit_task> commit_tasks;
    std::deque<encryption_entry *> enc_entries[10];
    std::mutex lock;
    std::mutex enc_lock[10];
    std::condition_variable enc_cv[10];
    std::atomic<bool> enc_init_done{false};
    std::atomic<bool> commit_init_done{false};
    std::atomic<bool> commit{false};
    uint64_t predict_iv_offset;
};

struct encryption_entry {
    std::atomic<bool> busy{false};
    // A busy speculative entry may already be queued on a CPU AES worker
    // when the real demand arrives.  Fast-fallback mode marks it cancelled so
    // the worker can skip AES if execution has not started yet.
    std::atomic<bool> cancelled{false};
    const void *src;
    unsigned char *buffer;
    size_t size;
    unsigned char tag[16];
    uint64_t iv_offset;
    uint64_t transfer_ticket;
    uint64_t transfer_block_offset;
    unsigned char iv[iv_length];
    uint64_t crypto_us;
    uint64_t generation;
};

// Process-unique transfer-ticket/IV management.  The derived IV keeps a
// per-process random session nonce in bytes 4..11 and the in-transfer AES
// block counter in bytes 12..15.  A ticket is never reused in one process.
uint64_t secure_infer_new_transfer_ticket();
void secure_infer_derive_transfer_iv(const unsigned char base_iv[iv_length],
                                uint64_t transfer_ticket,
                                uint64_t block_offset,
                                unsigned char out_iv[iv_length]);

// Enqueue the device-side CTR XOR for a ciphertext prepared with the exact
// transfer ticket above.  The operation is ordered on `stream`; callers may
// enqueue the following D2D copy immediately after it.  Keeping this helper in
// secure_infer.cpp makes the speculative worker use the same key/IV derivation and
// the same inline-IV kernel as the serial correctness fallback.
bool secure_infer_enqueue_ticketed_ctr_xor(aclrtStream stream,
                                      void* input,
                                      void* output,
                                      size_t bytes,
                                      uint64_t transfer_ticket,
                                      uint64_t block_offset);

// Device-keystream primitives used by the speculative commit worker.  The
// generator writes AES-CTR keystream for the exact ticket/counter assignment;
// the consume primitive performs mode-2 XOR only.  Both calls merely enqueue
// work on the supplied stream.
bool secure_infer_prepare_h2d_device_crypto(aclrtStream stream);
bool secure_infer_enqueue_ticketed_keystream(aclrtStream stream,
                                        void* output,
                                        size_t bytes,
                                        uint64_t transfer_ticket,
                                        uint64_t block_offset);
bool secure_infer_enqueue_precomputed_xor(aclrtStream stream,
                                     void* input,
                                     void* keystream,
                                     void* output,
                                     size_t bytes);

// Opt-in AES-GCM ticket pool.  History prediction reserves a transfer ticket,
// while the commit worker fills this pool on its private
// stream. A demand never waits for the pool: it consumes a ready entry or
// takes the existing authenticated AES-CTR fallback. In dynamic mode, a
// GHASH-critical interval migrates only sufficiently-far *new real tickets*
// to the verified AIV backend; a small near-consumer window remains on AICPU
// to preserve hit rate. Utilization-only preemption is an explicit
// experimental opt-in. A bounded AIV queue protects both AIV and AICPU; a
// full queue declines speculation rather than delaying GHASH.
bool secure_infer_gcm_keystream_scheduler_enabled();
bool secure_infer_gcm_dynamic_keystream_policy_enabled();
bool secure_infer_gcm_ghash_critical_active();
bool secure_infer_gcm_dynamic_relief_active();
// Returns 1 when handled (admitted/known or no longer useful), 0 for a retryable
// resource failure, and -1 when the dynamic AIV queue is intentionally full.
// The latter is a low-priority deferral signal: callers must continue
// admitting later near-deadline tickets instead of letting it cause
// head-of-line blocking.
int secure_infer_gcm_schedule_ticketed_keystream(aclrtStream stream,
                                            uint64_t transfer_ticket,
                                            size_t bytes,
                                            uint64_t generation,
                                            uint64_t admission_scope = 0);
// A bounded batch keeps one IV and output allocation per ticket, while one
// AICPU launch generates several independent streams using their common key.
// `admissions` receives 1 for admitted/known tickets and 0 for retryable
// backpressure.  Dynamic/AIV policy intentionally stays on the single-ticket
// path so this optimization cannot weaken GHASH-priority decisions.
struct gcm_keystream_submission {
    uint64_t transfer_ticket = 0;
    size_t bytes = 0;
    uint64_t generation = 0;
    uint64_t admission_scope = 0;
};
bool secure_infer_gcm_aicpu_batching_enabled();
size_t secure_infer_gcm_aicpu_batch_ticket_limit();
bool secure_infer_gcm_schedule_ticketed_keystream_batch(
    aclrtStream stream,
    const std::vector<gcm_keystream_submission>& submissions,
    std::vector<int>* admissions);
bool secure_infer_gcm_keystream_known(uint64_t transfer_ticket);
bool secure_infer_gcm_take_ready_keystream(uint64_t transfer_ticket,
                                      size_t bytes,
                                      void** keystream_out);
// Return the backend recorded for an admitted real ticket: 0=AICPU,
// 1=AIV, -1=unknown/not admitted. This is diagnostic-only and lets the
// per-H2D ledger prove where a consumed ticket was actually generated.
int secure_infer_gcm_ticket_backend(uint64_t transfer_ticket);
void secure_infer_gcm_finish_keystream_ticket(uint64_t transfer_ticket);
void secure_infer_gcm_keystream_cleanup();
void secure_infer_gcm_keystream_prewarm(aclrtStream stream);
void secure_infer_gcm_ghash_critical_enter();
void secure_infer_gcm_ghash_critical_exit();
extern "C" void secure_infer_get_gcm_keystream_stats(
    uint64_t* ready, uint64_t* busy, uint64_t* missing,
    uint64_t* aicpu_submitted, uint64_t* aicore_submitted,
    uint64_t* dropped, uint64_t* switches);

// Optional AES-GCM authentication backend. The default serving path remains
// CTR unless PIPE_CRYPTO_MODE=GCM is set explicitly.
constexpr uint32_t PIPE_GHASH_FINALIZE_TAG = 1U << 0;
constexpr uint32_t PIPE_GHASH_VERIFY_TAG = 1U << 1;
constexpr uint32_t PIPE_GHASH_XOR_ON_SUCCESS = 1U << 2;
using LaunchAicpuGhash_t = int (*)(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    uint64_t, uint64_t, uint32_t);
using LaunchAicpuAuthGate_t = int (*)(void*, void*, void*, uint64_t);
using LaunchAicpuAesCtr_t = int (*)(
    void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t);
using LaunchAicpuKeystream_t = int (*)(
    void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t);
// Multi-ticket AICPU ABI intentionally has no in-process binding.  Ticketed
// GCM pre-generation always uses the single-ticket entry below so each
// ticket owns its own output buffer and completion event.
// One AICPU task for the H2D GCM demand path: derive E_K(J0), authenticate
// ciphertext, decrypt only after verification, and report status.  Keeping
// this separate from the general GHASH ABI makes the verified-release
// contract explicit at the call site.
using LaunchAicpuGcmDecryptVerify_t = int (*)(
    void*, void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t);
using LaunchAicpuGcmDecryptVerifyAad_t = int (*)(
    void*, void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, void*, uint64_t);
using LaunchAicpuGcmEncryptTag_t = int (*)(
    void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, void*, uint64_t);
// Same authenticated-release contract, but payload AES is supplied by an
// exact ready ticket rather than recomputed on the demand stream.
using LaunchAicpuGcmVerifyXor_t = int (*)(
    void*, void*, void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t);
using LaunchAicpuGcmVerifyXorAad_t = int (*)(
    void*, void*, void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, void*, uint64_t);
using LaunchAicpuGcmVerifyXorBatch_t = int (*)(
    void*, void*, void*, void* const*, void* const*, void* const*,
    void* const*, void* const*, void* const*, uint64_t, uint64_t,
    const uint32_t*, uint32_t);
using LaunchAicpuGcmVerifyOnly_t = int (*)(
    void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t);
using LaunchAicpuGcmVerifyOnlyAad_t = int (*)(
    void*, void*, void*, void*, void*, void*, uint64_t,
    uint32_t, uint32_t, uint32_t, uint32_t, void*, uint64_t);
extern LaunchAicpuGhash_t g_launch_aicpu_ghash;
extern LaunchAicpuAuthGate_t g_launch_aicpu_auth_gate;
extern LaunchAicpuAesCtr_t g_launch_aicpu_aes_ctr;
extern LaunchAicpuKeystream_t g_launch_aicpu_keystream;
extern LaunchAicpuGcmDecryptVerify_t g_launch_aicpu_gcm_decrypt_verify;
extern LaunchAicpuGcmDecryptVerifyAad_t g_launch_aicpu_gcm_decrypt_verify_aad;
extern LaunchAicpuGcmEncryptTag_t g_launch_aicpu_gcm_encrypt_tag;
extern LaunchAicpuGcmVerifyXor_t g_launch_aicpu_gcm_verify_xor;
extern LaunchAicpuGcmVerifyXorAad_t g_launch_aicpu_gcm_verify_xor_aad;
extern LaunchAicpuGcmVerifyXorBatch_t g_launch_aicpu_gcm_verify_xor_batch;
extern LaunchAicpuGcmVerifyOnly_t g_launch_aicpu_gcm_verify_only;
extern LaunchAicpuGcmVerifyOnlyAad_t g_launch_aicpu_gcm_verify_only_aad;
extern LaunchAicpuGcmVerifyXor_t g_launch_aicpu_gcm_xor_only;
bool secure_infer_gcm_enabled();
bool secure_infer_gcm_aad_enabled();
bool secure_infer_build_gcm_aad(const void* host_ptr, size_t bytes,
                           uint8_t direction, uint64_t transfer_ticket,
                           uint64_t transfer_offset,
                           unsigned char out[gcm_aad_length]);
bool secure_infer_gcm_replay_reserve(uint64_t transfer_ticket);
bool secure_infer_prepare_gcm_backend(encrypt_metadata& metadata,
                                 aclrtStream stream);
bool secure_infer_compute_gcm_tag_mask(const unsigned char key[key_length],
                                  const unsigned char payload_iv[iv_length],
                                  unsigned char tag_mask[16]);

struct decryption_task {
    const void *src;
    void *dst;
    size_t size;
};

struct decryption_entry {
    volatile bool busy;
    const unsigned char *buffer;
    void *dst;
    size_t size;
    unsigned char tag[16];
    unsigned char iv[iv_length];
};

struct decrypt_worker_entry {
    int id;
    size_t magic_sz;
    std::deque<decryption_task> dec_tasks;
    std::mutex lock;

    std::deque<decryption_entry *> dec_entries;
    std::mutex dec_lock;

    std::atomic<bool> dec_init_done{false};
    std::atomic<bool> commit_init_done{false};
    std::atomic<bool> commit{false};

    struct encrypt_worker_entry *enc_worker_entry;
};

uint64_t next_iv(unsigned char cur_iv[], unsigned char dest_iv[], uint64_t incr = 1);
uint64_t secure_infer_now_us();
void profile_record(const char* direction,
                    size_t bytes,
                    size_t aligned_bytes,
                    uint64_t total_us,
                    uint64_t cpu_crypto_us,
                    uint64_t device_pipeline_us,
                    uint64_t host_wait_us,
                    uint64_t copy_enqueue_us,
                    uint64_t kernel_enqueue_us);
// Starts an asynchronous precise-profile sample and snapshots the current
// run id.  The run id lets the worker write the row after the marker file has
// been removed, provided that the same profiling run is still active in the
// output file.
bool profile_begin_precise_capture(char* run_id, size_t run_id_size);
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
                            int crypto_mode = -1,
                            int cpu_prediction_hit = -1);
void profile_count_d2h_kv(size_t bytes);

struct Predictor {
    // For read-only swap（原有）
    std::vector<std::pair<const void *, size_t>> read_only_swap_record;
    bool read_only_swap_profiled = false;
    int read_only_swap_cur_idx = 0;
    int read_only_swap_pred_idx = 0;

    // For other swaps（原有）
    std::vector<std::pair<const void *, size_t>> other_swap_record;
    std::set<std::pair<const void *, size_t>> other_swap_set;

    // For commit / decrypt（原有）
    std::deque<std::pair<const void*, std::pair<void *, size_t>>> pending_commit;
    std::deque<decryption_task> pending_decrypt;

    // ==================== 新增：线性预测模式（vLLM 支持） ====================
    bool linear_mode = false;
    void* linear_base = nullptr;
    size_t linear_stride = 0;
    const void* linear_expected_src = nullptr;
    const void* linear_prefetch_until = nullptr;  // exclusive: first address not yet scheduled
    int linear_count = 0;
    uint64_t generation = 1;
    const int LINEAR_THRESHOLD = 2;   // 连续2次相同大小 + 地址递增 → 进入线性模式

    // Evaluation-only hook predictor.  It learns a repeating address-delta
    // sequence from prior memcpy calls; it never consumes framework metadata.
    const void* periodic_last_src = nullptr;
    size_t periodic_copy_size = 0;
    std::vector<intptr_t> periodic_deltas;
    std::vector<intptr_t> periodic_cycle;
    size_t periodic_next_phase = 0;

    // A small history-only cache for recurring fixed-stride runs.  Each
    // entry records which address delta most often followed a run of a given
    // length.  It accelerates the first few repetitions before the stricter
    // full-cycle detector has observed two complete cycles.
    struct PeriodicRunPattern {
        size_t copy_size = 0;
        size_t run_blocks = 0;
        std::vector<std::pair<intptr_t, uint32_t>> transitions;
        intptr_t last_transition = 0;
        uint64_t last_used = 0;
    };
    std::vector<PeriodicRunPattern> periodic_run_patterns;
    size_t periodic_current_run_blocks = 0;
    size_t periodic_active_run_blocks = 0;
    uint64_t periodic_pattern_clock = 0;

    // Evaluation control that follows the repetitive detector in the public
    // SecureInfer source: prediction starts only after the complete observed H2D
    // sequence consists of two identical halves.  It is isolated from the
    // SealFlow delta predictor so the two policies can be compared under the
    // same crypto, buffering, and fallback implementation.
    std::vector<std::pair<const void*, size_t>> secure_infer_repeat_record;
    bool secure_infer_repeat_profiled = false;
    size_t secure_infer_repeat_next_index = 0;

    void lock() {
        bool expected = false;
        while (!_lock.compare_exchange_weak(expected, true, std::memory_order_acquire)) {
            expected = false;
            std::this_thread::yield();
        }
    }

    void unlock() {
        assert(this->_lock.load(std::memory_order_relaxed));
        this->_lock.store(false, std::memory_order_release);
    }

    bool locking() {
        return this->_lock.load(std::memory_order_relaxed);
    }
private:
    std::atomic<bool> _lock{false};
};

// bind_core 函数 (保持原样)
static inline void bind_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    pthread_t currentThread = pthread_self();
    int rc = pthread_setaffinity_np(currentThread, sizeof(cpu_set_t), &cpuset);
    assert(rc == 0);
}

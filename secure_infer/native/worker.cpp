/*
 * 文件名: worker.cpp (已修改为 ACL 版本)
 * 职责: 实现加密/解密/提交的工作线程
 */

 #include "secure_infer.h"
 #include "hack.h"
 #include "resource_monitor.h"
 #include <unistd.h>
 #include <cstring>
 #include <cassert>
 #include <chrono>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern LaunchAESKernel_t g_launch_kernel;
extern void* g_dev_key;
extern void* g_dev_iv;
extern aclrtContext g_main_ctx;
extern std::atomic<bool> g_running;

static std::atomic<uint64_t> g_ready_npu_keystream_commits{0};
static std::atomic<uint64_t> g_busy_npu_keystream_fallbacks{0};
static std::atomic<uint64_t> g_missing_npu_keystream_fallbacks{0};
static std::atomic<uint64_t> g_submitted_npu_keystream_blocks{0};
static std::atomic<uint64_t> g_capacity_npu_keystream_fallbacks{0};

extern "C" __attribute__((visibility("default"))) void
secure_infer_get_npu_keystream_stats(uint64_t* ready,
                                uint64_t* busy_fallback,
                                uint64_t* missing_fallback)
{
    if (ready != nullptr) {
        *ready = g_ready_npu_keystream_commits.load(
            std::memory_order_relaxed);
    }
    if (busy_fallback != nullptr) {
        *busy_fallback = g_busy_npu_keystream_fallbacks.load(
            std::memory_order_relaxed);
    }
    if (missing_fallback != nullptr) {
        *missing_fallback = g_missing_npu_keystream_fallbacks.load(
            std::memory_order_relaxed);
    }
}

extern "C" __attribute__((visibility("default"))) void
secure_infer_get_npu_keystream_pool_stats(uint64_t* submitted_blocks,
                                     uint64_t* capacity_fallbacks)
{
    if (submitted_blocks != nullptr) {
        *submitted_blocks = g_submitted_npu_keystream_blocks.load(
            std::memory_order_relaxed);
    }
    if (capacity_fallbacks != nullptr) {
        *capacity_fallbacks = g_capacity_npu_keystream_fallbacks.load(
            std::memory_order_relaxed);
    }
}

static bool npu_keystream_prefetch_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_NPU_KEYSTREAM_PREFETCH");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] NPU keystream prefetch: %s\n",
                enabled ? "enabled" : "disabled");
    }
    return enabled != 0;
}

static size_t max_npu_keystream_blocks()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env = std::getenv("PIPE_NPU_KEYSTREAM_MAX_BLOCKS");
        long value = env != nullptr ? std::atol(env) : 512;
        if (value < 1) value = 1;
        if (value > 4096) value = 4096;
        limit = static_cast<size_t>(value);
        fprintf(stderr, "[SecureInfer] NPU keystream cache: %zu blocks\n",
                limit);
    }
    return limit;
}

static size_t npu_keystream_pool_byte_limit()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env =
            std::getenv("PIPE_NPU_KEYSTREAM_POOL_BYTES");
        unsigned long long value = env != nullptr
            ? std::strtoull(env, nullptr, 10)
            : static_cast<unsigned long long>(
                  max_npu_keystream_blocks()) * 262144ULL;
        if (value < 262144ULL) value = 262144ULL;
        if (value > 4ULL * 1024ULL * 1024ULL * 1024ULL) {
            value = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        }
        limit = static_cast<size_t>(value);
        fprintf(stderr,
                "[SecureInfer] NPU keystream slab pool: %.1f MiB\n",
                static_cast<double>(limit) / (1024.0 * 1024.0));
    }
    return limit;
}

static std::mutex g_npu_keystream_buffer_pool_lock;
static std::unordered_map<size_t, std::vector<void*>>
    g_npu_keystream_buffer_pool;
static size_t g_npu_keystream_cached_bytes = 0;

static void* alloc_npu_keystream_buffer(size_t bytes)
{
    {
        std::lock_guard<std::mutex> lock(
            g_npu_keystream_buffer_pool_lock);
        auto it = g_npu_keystream_buffer_pool.find(bytes);
        if (it != g_npu_keystream_buffer_pool.end() &&
            !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            g_npu_keystream_cached_bytes -= bytes;
            return ptr;
        }
    }
    void* ptr = nullptr;
    if (real_aclrtMalloc(&ptr, bytes) != ACL_SUCCESS) {
        return nullptr;
    }
    return ptr;
}

static void release_npu_keystream_buffer(void* ptr, size_t bytes)
{
    if (ptr == nullptr) return;
    bool cached = false;
    {
        std::lock_guard<std::mutex> lock(
            g_npu_keystream_buffer_pool_lock);
        const size_t limit = npu_keystream_pool_byte_limit();
        if (bytes <= limit &&
            g_npu_keystream_cached_bytes <= limit - bytes) {
            g_npu_keystream_buffer_pool[bytes].push_back(ptr);
            g_npu_keystream_cached_bytes += bytes;
            cached = true;
        }
    }
    if (!cached) {
        real_aclrtFree(ptr);
    }
}

static void prewarm_npu_keystream_buffer_pool()
{
    const char* enabled_env =
        std::getenv("PIPE_NPU_KEYSTREAM_POOL_PREWARM");
    if (enabled_env != nullptr && std::atoi(enabled_env) == 0) return;
    const char* bytes_env =
        std::getenv("PIPE_NPU_KEYSTREAM_PREWARM_BYTES");
    unsigned long long parsed = bytes_env != nullptr
        ? std::strtoull(bytes_env, nullptr, 10)
        : 63ULL * 262144ULL;
    if (parsed == 0 || parsed > UINT32_MAX) return;
    const size_t bytes =
        (static_cast<size_t>(parsed) + 31) &
        ~static_cast<size_t>(31);
    const size_t count =
        npu_keystream_pool_byte_limit() / bytes;
    if (count == 0) return;

    std::vector<void*> buffers;
    buffers.reserve(count);
    const uint64_t start = secure_infer_now_us();
    for (size_t i = 0; i < count; ++i) {
        void* ptr = alloc_npu_keystream_buffer(bytes);
        if (ptr == nullptr) break;
        buffers.push_back(ptr);
    }
    for (void* ptr : buffers) {
        release_npu_keystream_buffer(ptr, bytes);
    }
    fprintf(stderr,
            "[SecureInfer] NPU keystream slab prewarm: buffers=%zu, "
            "bytes_each=%zu, time=%.3f ms\n",
            buffers.size(), bytes,
            (secure_infer_now_us() - start) / 1000.0);
}

static void drain_npu_keystream_buffer_pool()
{
    std::unordered_map<size_t, std::vector<void*>> buffers;
    {
        std::lock_guard<std::mutex> lock(
            g_npu_keystream_buffer_pool_lock);
        buffers.swap(g_npu_keystream_buffer_pool);
        g_npu_keystream_cached_bytes = 0;
    }
    for (auto& [bytes, entries] : buffers) {
        (void)bytes;
        for (void* ptr : entries) {
            real_aclrtFree(ptr);
        }
    }
}

static size_t npu_keystream_admission_tickets()
{
    static size_t limit = 0;
    if (limit == 0) {
        const char* env =
            std::getenv("PIPE_NPU_KEYSTREAM_ADMISSION_TICKETS");
        long value = env != nullptr ? std::atol(env) : 8;
        if (value < 1) value = 1;
        // A short host admission quantum creates device-side bubbles even when
        // thousands of history-predicted tickets are already available.  In
        // particular, 512 tickets at 32 IVs/launch produce only 16 AICPU
        // kernels, shorter than the utilization API's ~128-ms observation
        // window on this platform.  Preserve the normal default of eight, but
        // let an explicitly configured real-request burst enqueue enough work
        // to keep the bounded ticket pool and AICPU executor continuously
        // occupied.  The record/byte limits remain the memory authorities.
        if (value > 8192) value = 8192;
        limit = static_cast<size_t>(value);
    }
    return limit;
}

// One commit worker is enough to keep the legacy FIFO predictor correct, but
// one ACL stream also serializes every real AICPU ticket launch.  Keep that
// historical behavior by default.  An opt-in bounded stream pool lets a
// pressure experiment (or a future multi-chain implementation) expose the
// device's actual AICPU parallelism without creating additional predictor
// threads or changing the ticket ordering/identity rules.
static size_t keystream_producer_stream_count()
{
    static size_t count = 0;
    if (count == 0) {
        // The original variable is retained for every existing GCM script.
        // CTR uses the generic spelling because its slab pool is consumed by
        // this same commit worker rather than the GCM inline path.
        const char* env = std::getenv("PIPE_NPU_KEYSTREAM_PRODUCER_STREAMS");
        if (env == nullptr || env[0] == '\0') {
            env = std::getenv("PIPE_GCM_KEYSTREAM_PRODUCER_STREAMS");
        }
        long parsed = env != nullptr ? std::atol(env) : 1;
        // A 20B model can expose MiB-scale real KV slices.  Sixteen streams
        // still left most AICPU capacity idle on 910B3, so permit a wider
        // bounded pool for the real-request saturation sweep.  The serving
        // default remains one and experiments must opt in explicitly.
        parsed = std::max(1L, std::min(64L, parsed));
        count = static_cast<size_t>(parsed);
        fprintf(stderr,
                "[SecureInfer] NPU keystream producer streams: %zu\n", count);
    }
    return count;
}

static bool gcm_keystream_fast_streams_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_KEYSTREAM_FAST_STREAMS");
        enabled = env != nullptr && std::atoi(env) != 0 ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "[SecureInfer] GCM producer streams: ACL fast-launch\n");
        }
    }
    return enabled != 0;
}

static aclError create_gcm_keystream_stream(aclrtStream* stream)
{
    if (stream == nullptr) return ACL_ERROR_INVALID_PARAM;
    if (!gcm_keystream_fast_streams_enabled()) {
        return real_aclrtCreateStream(stream);
    }
    // CANN documents FAST_LAUNCH as preallocating the internal submission
    // resources used by a long-lived stream.  That matches thousands of
    // short AICPU ticket kernels.  CPU_SCHEDULE is explicitly a reserved
    // model-queue flag and is deliberately not used here.
    return aclrtCreateStreamWithConfig(stream, 0, ACL_STREAM_FAST_LAUNCH);
}

// Transfer descriptors arrive one by one.  Without a short explicit
// coalescing window the commit worker wakes on the first ticket, so a nominal
// 32-ticket batch often degenerates into one or two launches followed by a
// host-side queue scan.  The default is zero to preserve serving latency.  A
// real-request pressure experiment can opt in and trade a bounded amount of
// lookahead slack for dense AICPU submission.
static uint64_t gcm_admission_coalesce_us()
{
    static uint64_t value = UINT64_MAX;
    if (value == UINT64_MAX) {
        const char* env = std::getenv("PIPE_GCM_ADMISSION_COALESCE_US");
        unsigned long long parsed = env != nullptr
            ? std::strtoull(env, nullptr, 10) : 0ULL;
        value = std::min<unsigned long long>(parsed, 100000ULL);
        if (value != 0) {
            fprintf(stderr,
                    "[SecureInfer] GCM admission coalescing window: %lu us\n",
                    static_cast<unsigned long>(value));
        }
    }
    return value;
}

static size_t gcm_admission_coalesce_target()
{
    static size_t target = 0;
    if (target == 0) {
        const char* env = std::getenv("PIPE_GCM_ADMISSION_COALESCE_TICKETS");
        unsigned long long parsed = env != nullptr
            ? std::strtoull(env, nullptr, 10)
            : npu_keystream_admission_tickets();
        parsed = std::max<unsigned long long>(1ULL,
            std::min<unsigned long long>(parsed,
                                         npu_keystream_admission_tickets()));
        target = static_cast<size_t>(parsed);
    }
    return target;
}

// During a genuinely high-pressure GHASH interval, a later queued H2D job
// is the only work with enough slack to use the AIV relief valve.  Keep this
// opt-out for diagnostics, but enable it by default for dynamic GCM policy.
static bool gcm_scope_aware_admission_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_GCM_SCOPE_AWARE_ADMISSION");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static bool latency_worker_pinning_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_WORKER_PINNING");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// Keep the latency-sensitive commit worker and the four CPU AES workers off
// the same run queue.  Respect the process' inherited affinity mask instead
// of assuming a machine-wide CPU numbering scheme; the serving launcher may
// already constrain EngineCore to one NUMA node.
static void pin_latency_worker(int slot, const char* name)
{
    if (name != nullptr) {
        pthread_setname_np(pthread_self(), name);
    }
    if (!latency_worker_pinning_enabled()) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (pthread_getaffinity_np(
            pthread_self(), sizeof(allowed), &allowed) != 0) {
        return;
    }

    const char* base_env = std::getenv("PIPE_WORKER_CPU_BASE");
    int selected = -1;
    if (base_env != nullptr && base_env[0] != '\0') {
        const long base = std::strtol(base_env, nullptr, 10);
        const long candidate = base + slot;
        if (candidate >= 0 && candidate < CPU_SETSIZE &&
            CPU_ISSET(static_cast<int>(candidate), &allowed)) {
            selected = static_cast<int>(candidate);
        }
    } else {
        // slot 0 is the commit worker and receives the highest allowed CPU;
        // encryption slots follow on the next-highest CPUs.
        int remaining = slot;
        for (int cpu = CPU_SETSIZE - 1; cpu >= 0; --cpu) {
            if (!CPU_ISSET(cpu, &allowed)) continue;
            if (remaining-- == 0) {
                selected = cpu;
                break;
            }
        }
    }
    if (selected < 0) return;

    cpu_set_t target;
    CPU_ZERO(&target);
    CPU_SET(selected, &target);
    const int rc = pthread_setaffinity_np(
        pthread_self(), sizeof(target), &target);
    if (rc == 0) {
        fprintf(stderr, "[SecureInfer] pinned %s to CPU %d\n",
                name != nullptr ? name : "worker", selected);
    } else {
        fprintf(stderr, "[SecureInfer] unable to pin %s to CPU %d: rc=%d\n",
                name != nullptr ? name : "worker", selected, rc);
    }
}

// 声明 add_custom_do 函数指针类型
typedef void (*AddCustomDo_t)(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t);
static AddCustomDo_t g_add_custom_do = nullptr;
static void* g_kernel_lib_handle = nullptr;

static void init_npu_kernel() {
    if (g_add_custom_do) return;

    const char* kernel_lib_path = std::getenv("PIPE_AES_KERNEL_LIB");
    if (kernel_lib_path == nullptr || kernel_lib_path[0] == '\0') {
        fprintf(stderr, "[runtime] PIPE_AES_KERNEL_LIB is required\n");
        return;
    }
    g_kernel_lib_handle = dlopen(kernel_lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!g_kernel_lib_handle) {
        fprintf(stderr, "[Worker] Failed to load kernel lib: %s\n", dlerror());
        return;
    }

    // 尝试C++符号
    g_add_custom_do = (AddCustomDo_t)dlsym(g_kernel_lib_handle, "_Z13add_custom_dojPvPhS0_S0_S0_jj");
    if (!g_add_custom_do) {
        g_add_custom_do = (AddCustomDo_t)dlsym(g_kernel_lib_handle, "add_custom_do");
    }

    if (g_add_custom_do) {
        fprintf(stderr, "[Worker] NPU kernel loaded: %p\n", (void*)g_add_custom_do);
    } else {
        fprintf(stderr, "[Worker] Failed to find add_custom_do: %s\n", dlerror());
    }
}

struct cipher_buffer_info {
    size_t size;
    bool pinned;
};

static std::mutex g_cipher_buffer_pool_lock;
static std::unordered_map<size_t, std::vector<void*>> g_cipher_buffer_pool;
static std::unordered_map<void*, cipher_buffer_info> g_cipher_buffer_info;

static bool use_pinned_cipher_buffers()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("PIPE_USE_PINNED_CIPHER");
        enabled = (!env || std::atoi(env) != 0) ? 1 : 0;
        fprintf(stderr, "[SecureInfer] CPU cipher buffer: %s\n",
                enabled ? "aclrtMallocHost pinned" : "malloc");
    }
    return enabled != 0;
}

static size_t cipher_buffer_pool_limit_per_size()
{
    static size_t limit = 0;
    static bool initialized = false;
    if (!initialized) {
        const char* env = std::getenv("PIPE_CIPHER_POOL_BLOCKS");
        long value = env ? std::atol(env) : 128;
        if (value < 0) value = 0;
        if (value > 4096) value = 4096;
        limit = static_cast<size_t>(value);
        initialized = true;
        fprintf(stderr, "[SecureInfer] CPU cipher buffer pool: %zu blocks/size\n", limit);
    }
    return limit;
}

static void* alloc_cipher_buffer(size_t bytes) {
    if (bytes == 0) bytes = 1;

    {
        std::lock_guard<std::mutex> lock(g_cipher_buffer_pool_lock);
        auto pool_it = g_cipher_buffer_pool.find(bytes);
        if (pool_it != g_cipher_buffer_pool.end() && !pool_it->second.empty()) {
            void* ptr = pool_it->second.back();
            pool_it->second.pop_back();
            return ptr;
        }
    }

    void* ptr = nullptr;
    bool pinned = false;
    if (use_pinned_cipher_buffers()) {
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
    {
        std::lock_guard<std::mutex> lock(g_cipher_buffer_pool_lock);
        g_cipher_buffer_info[ptr] = {bytes, pinned};
    }
    return ptr;
}

static void free_cipher_buffer(void* ptr) {
    if (ptr == nullptr) return;

    cipher_buffer_info info{0, false};
    bool known = false;
    {
        std::lock_guard<std::mutex> lock(g_cipher_buffer_pool_lock);
        auto info_it = g_cipher_buffer_info.find(ptr);
        if (info_it != g_cipher_buffer_info.end()) {
            info = info_it->second;
            known = true;
            auto& pool = g_cipher_buffer_pool[info.size];
            if (pool.size() < cipher_buffer_pool_limit_per_size()) {
                pool.push_back(ptr);
                return;
            }
            g_cipher_buffer_info.erase(info_it);
        }
    }

    if (known && info.pinned) {
        real_aclrtFreeHost(ptr);
    } else {
        std::free(ptr);
    }
}

static size_t cipher_buffer_prewarm_blocks()
{
    static size_t blocks = 0;
    static bool initialized = false;
    if (!initialized) {
        const char* env = std::getenv("PIPE_CIPHER_POOL_PREWARM_BLOCKS");
        long value = env ? std::atol(env) : static_cast<long>(cipher_buffer_pool_limit_per_size());
        if (value < 0) value = 0;
        if (value > 4096) value = 4096;
        blocks = static_cast<size_t>(value);
        initialized = true;
        fprintf(stderr, "[SecureInfer] CPU cipher buffer prewarm: %zu blocks\n", blocks);
    }
    return blocks;
}

static size_t cipher_buffer_prewarm_bytes()
{
    static size_t bytes = 0;
    if (bytes == 0) {
        const char* env = std::getenv("PIPE_CIPHER_POOL_PREWARM_BYTES");
        long value = env ? std::atol(env) : 262144;
        if (value < 4096) value = 4096;
        bytes = static_cast<size_t>(value);
        fprintf(stderr, "[SecureInfer] CPU cipher buffer prewarm size: %zu bytes\n", bytes);
    }
    return bytes;
}

static void prewarm_cipher_buffer_pool()
{
    static std::atomic<bool> done{false};
    bool expected = false;
    if (!done.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;

    size_t blocks = cipher_buffer_prewarm_blocks();
    size_t bytes = cipher_buffer_prewarm_bytes();
    if (blocks == 0) return;

    std::vector<void*> buffers;
    buffers.reserve(blocks);
    uint64_t start = secure_infer_now_us();
    for (size_t i = 0; i < blocks; ++i) {
        void* ptr = alloc_cipher_buffer(bytes);
        // First-touch the host pages during startup.  Otherwise the first
        // measured request pays a burst of pinned-buffer page faults/allocation
        // cost that shows up as fake "CPU crypto" long tail.
        std::memset(ptr, 0, bytes);
        buffers.push_back(ptr);
    }
    for (void* ptr : buffers) {
        free_cipher_buffer(ptr);
    }
    fprintf(stderr,
            "[SecureInfer] CPU cipher buffer prewarm done: blocks=%zu, bytes=%zu, time=%.3f ms\n",
            blocks, bytes, (secure_infer_now_us() - start) / 1000.0);
}

static uint32_t calc_aes_block_dim(size_t bytes) {
    // Keep this aligned with the reference block-dimension calculation.
    // For the common 256KB KV block this uses 20 cores instead of the old 8,
    // reducing stream backpressure from slow under-parallelized NPU kernels.
    constexpr uint32_t core_num = 20;
    uint64_t avg = bytes / core_num;
    if (avg < 4096) return 1;
    return core_num;
}

static size_t max_predicted_blocks()
{
    static size_t max_blocks = 0;
    if (max_blocks == 0) {
        const char* env = std::getenv("PIPE_MAX_PREDICT_BLOCKS");
        long value = env ? std::atol(env) : 64;
        if (value < 16) value = 16;
        if (value > 4096) value = 4096;
        max_blocks = static_cast<size_t>(value);
        fprintf(stderr, "[SecureInfer] 最大保留预测块: %zu\n", max_blocks);
    }
    return max_blocks;
}

static size_t speculative_cpu_admission_quantum()
{
    static size_t quantum = 0;
    if (quantum == 0) {
        const char* env = std::getenv("PIPE_SPEC_CPU_ADMISSION_QUANTUM");
        long parsed = env ? std::atol(env) : 8;
        if (parsed < 1) parsed = 1;
        if (parsed > 128) parsed = 128;
        quantum = static_cast<size_t>(parsed);
        fprintf(stderr,
                "[SecureInfer] speculative CPU admission quantum: %zu\n",
                quantum);
    }
    return quantum;
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

struct GCBatch {
    aclrtEvent event;           // 用于标记这一批次何时完成
    std::vector<void*> buffers; // 待释放的 char* buffer
    std::vector<encryption_entry*> entries; // 待释放的 entry 对象
    // A consumed tensor slab must outlive the mode-2 XOR queued on the commit
    // stream, even if the transfer scope retires it from the lookup table.
    std::vector<std::shared_ptr<void>> device_owners;
    size_t blocks = 0;
};

static size_t max_commit_inflight_blocks()
{
    static size_t limit = 0;
    static bool initialized = false;
    if (!initialized) {
        const char* env = std::getenv("PIPE_COMMIT_MAX_INFLIGHT_BLOCKS");
        long value = env ? std::atol(env) : 8;
        if (value < 1) value = 1;
        if (value > 4096) value = 4096;
        limit = static_cast<size_t>(value);
        initialized = true;
        fprintf(stderr, "[SecureInfer] commit stream max inflight: %zu blocks\n", limit);
    }
    return limit;
}

static void process_sync_events(std::deque<std::shared_ptr<commit_sync_state>>& events) {
    for (auto it = events.begin(); it != events.end();) {
        auto& sync = *it;
        if (!sync->wait_enqueued.load(std::memory_order_acquire)) {
            ++it;
            continue;
        }
        if (sync->status.load(std::memory_order_acquire) != ACL_SUCCESS) {
            real_aclrtDestroyEvent(sync->event);
            sync->event = nullptr;
            it = events.erase(it);
            continue;
        }
        aclrtEventWaitStatus status = ACL_EVENT_WAIT_STATUS_NOT_READY;
        aclError ret = aclrtQueryEventWaitStatus(sync->event, &status);
        if (ret == ACL_SUCCESS && status == ACL_EVENT_WAIT_STATUS_COMPLETE) {
            real_aclrtDestroyEvent(sync->event);
            sync->event = nullptr;
            it = events.erase(it);
        } else {
            ++it;
        }
    }
}

static size_t release_gc_batch(GCBatch& batch) {
    size_t blocks = batch.blocks;
    for (void* buf : batch.buffers) {
        free_cipher_buffer(buf);
    }
    for (auto* entry : batch.entries) {
        delete entry;
    }
    if (batch.event != nullptr) {
        real_aclrtDestroyEvent(batch.event);
        batch.event = nullptr;
    }
    return blocks;
}

static size_t wait_one_gc_batch(std::deque<GCBatch>& gc_queue) {
    if (gc_queue.empty()) return 0;
    GCBatch& front = gc_queue.front();
    if (front.event != nullptr) {
        real_aclrtSynchronizeEvent(front.event);
    }
    size_t blocks = release_gc_batch(front);
    gc_queue.pop_front();
    return blocks;
}

size_t process_garbage_collection(std::deque<GCBatch>& gc_queue) {
    size_t released_blocks = 0;
    while (!gc_queue.empty()) {
        GCBatch& front = gc_queue.front();
        aclrtEventStatus status = ACL_EVENT_STATUS_NOT_READY;

        // 非阻塞查询事件状态
        aclError ret = aclrtQueryEvent(front.event, &status);

        if (ret == ACL_SUCCESS && status == ACL_EVENT_STATUS_COMPLETE) {
            released_blocks += release_gc_batch(front);
            gc_queue.pop_front();

            // fprintf(stderr, "[GC] 释放了一批内存\n");
        } else {
            // 队列头部的事件还没完成，后面的肯定也没完成（因为是顺序流），直接退出
            break;
        }
    }
    return released_blocks;
}
/*
 * ==========================================================================
 * 函数: encrypt_worker (增强版本)
 * ==========================================================================
 */
    void encrypt_worker(void *entry)
    {
        // while(1) sleep(1000);
        fprintf(stderr, "[encrypt_worker] 🚀 函数开始执行\n");
        fflush(stderr);

        auto x = (struct encrypt_worker_entry *)entry;
        auto worker_id = x->id;
        auto local_id = x->local_id;
        char worker_name[16];
        std::snprintf(worker_name, sizeof(worker_name),
                      "si-enc%d", local_id);
        pin_latency_worker(local_id + 1, worker_name);

        fprintf(stderr, "[encrypt_worker] worker_id=%d, local_id=%d, magic_sz=%zu\n",
                worker_id, local_id, x->magic_sz);
        fflush(stderr);

        // 绑定核心
        // bind_core((x->encrypt_workers_per_commit + 1) * worker_id + local_id + 1);
        fprintf(stderr, "[encrypt_worker] ✅ 核心绑定完成\n");
        fflush(stderr);

        // 🎯 [关键修复] 增强的等待机制
        fprintf(stderr, "[encrypt_worker] 🔍 开始等待加密上下文初始化...\n");
        fflush(stderr);

        EVP_CIPHER_CTX *enc_ctx = nullptr;
     int retry_count = 0;
     const int max_retry = 500; // 最多重试500次，约5秒

     while (enc_ctx == nullptr && retry_count < max_retry) {
         // 安全地访问映射表
         { // [!!! 关键修复：添加大括号和锁 !!!]

            // 在访问 m_magic_encctx 之前加锁
            // std::lock_guard<std::mutex> lock(g_ctx_map_mutex);

            auto it = m_magic_encctx.find(x->magic_sz);

            // (您之前的正确逻辑)
            if (it != m_magic_encctx.end()) {
                // 找到了 key
                if (it->second != nullptr) {
                    // 找到了有效的 ctx
                    enc_ctx = it->second;
                    fprintf(stderr, "[encrypt_worker] 循环 %d: ✅ 成功找到 key 且 ctx 有效: %p\n", retry_count, (void*)enc_ctx);
                    fflush(stderr);
                } else {
                    // 找到了 key，但是 value 是 nullptr (占位符)
                    // fprintf(stderr, "[encrypt_worker] 循环 %d: ⚠️  找到 key, 但 ctx 仍为 nullptr (正在等待 OpenSSL 握手...)\n", retry_count);
                    fflush(stderr);
                }
            } else {
                // 连 key 都没找到 (这不应该发生)
                fprintf(stderr, "[encrypt_worker] 循环 %d: ❌ 未找到 key (magic_sz: %zu)! secure_infer.cpp 的 init 是否先运行了?\n", retry_count, x->magic_sz);
                fflush(stderr);
            }

        } // [!!! 锁在这里自动释放 !!!]
        if (enc_ctx != nullptr) {
            break; // 如果找到了，退出循环
        }
         // 等待并重试
         usleep(100); // 等待10ms
         retry_count++;

         if (retry_count % 50 == 0) {
             fprintf(stderr, "[encrypt_worker] ⏳ 等待加密上下文... (%d/%d)\n", retry_count, max_retry);
             fprintf(stderr, "[encrypt_worker] m_magic_encctx 大小: %zu\n", m_magic_encctx.size());
             fflush(stderr);
         }
     }

     // 🎯 [关键修复] 检查等待结果
     if (enc_ctx == nullptr) {
         fprintf(stderr, "[encrypt_worker] ❌ 致命错误: 无法获取加密上下文\n");
         fprintf(stderr, "[encrypt_worker] ❌ 重试次数: %d/%d\n", retry_count, max_retry);
         fprintf(stderr, "[encrypt_worker] ❌ m_magic_encctx 内容:\n");

         for (const auto& pair : m_magic_encctx) {
             fprintf(stderr, "  - magic_sz=%zu -> ctx=%p\n", pair.first, pair.second);
         }

         fprintf(stderr, "[encrypt_worker] ❌ 期望的 magic_sz: %zu\n", x->magic_sz);
         fflush(stderr);

         // 设置初始化完成标志，但退出线程
         x->enc_init_done = true;
         return;
     }

     fprintf(stderr, "[encrypt_worker] ✅ 加密上下文获取成功: %p (重试: %d次)\n", enc_ctx, retry_count);
     fflush(stderr);

     // 🎯 [关键修复] 安全地获取元数据
     if (m_ctx_metadata.find(enc_ctx) == m_ctx_metadata.end()) {
         fprintf(stderr, "[encrypt_worker] ❌ 错误: 加密上下文没有对应的元数据\n");
         fflush(stderr);
         x->enc_init_done = true;
         return;
     }

     auto &enc_metadata = *m_ctx_metadata.at(enc_ctx);
     fprintf(stderr, "[encrypt_worker] ✅ 元数据获取成功\n");
     fflush(stderr);

     // 复制密钥和IV
     unsigned char encrypt_key[key_length];
     unsigned char encrypt_init_iv[iv_length];
     unsigned char iv[iv_length];

     memcpy(encrypt_key, enc_metadata.key, key_length);
     memcpy(encrypt_init_iv, enc_metadata.init_iv, iv_length);
     fprintf(stderr, "[encrypt_worker] ✅ 密钥和IV复制完成\n");
     fflush(stderr);

     // 分配假数据缓冲区
     auto fake_src = new unsigned char [1 << 20];
     fprintf(stderr, "[encrypt_worker] ✅ 假数据缓冲区分配完成\n");
     fflush(stderr);

     // 🎯 [关键修复] 设置初始化完成标志
     x->enc_init_done.store(true, std::memory_order_release);
     fprintf(stderr, "[encrypt_worker] ✅ 初始化标记设置完成，进入主循环\n");
     fflush(stderr);

     // 创建本地OpenSSL上下文
     auto local_ctx = real_EVP_CIPHER_CTX_new();
     if (local_ctx == nullptr) {
         fprintf(stderr, "[encrypt_worker] ❌ 错误: OpenSSL上下文创建失败\n");
         fflush(stderr);
         return;
     }
     fprintf(stderr, "[encrypt_worker] ✅ 本地OpenSSL上下文创建完成: %p\n", local_ctx);
     fflush(stderr);

     std::deque<encryption_entry *> enc_entries;

     // 主工作循环
     fprintf(stderr, "[encrypt_worker] 🔄 进入主工作循环\n");
     fflush(stderr);

     while (g_running.load()) {
         bool got_work = false;
         while (g_running.load() && !got_work) {
             {
                 std::lock_guard<std::mutex> lock(x->enc_lock[local_id]);
                 if (!x->enc_entries[local_id].empty()) {
                     enc_entries.swap(x->enc_entries[local_id]);
                     got_work = true;
                 }
             }
             if (got_work) break;
             constexpr int polling_us = 1;
             auto start = std::chrono::system_clock::now();
             while (1) {
                 auto end = std::chrono::system_clock::now();
                 if (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() >= polling_us) {
                     break;
                 }
             }
         }

         if (!g_running.load()) break;

         // Each worker receives a contiguous counter interval from a tensor
         // ticket.  Keep CTR state across its discontiguous host buffers:
         // one EncryptInit per interval, then one Update per physical block.
         // The previous per-256-KiB reset multiplied EVP setup and scheduler
         // overhead by all 5040 H2D fragments of an 8K reload.
         for (size_t begin = 0; begin < enc_entries.size();) {
             auto* first_entry = enc_entries[begin];
             const bool first_cancelled =
                 secure_infer_linear_fast_fallback_enabled() &&
                 (first_entry->cancelled.load(std::memory_order_acquire) ||
                  secure_infer_speculative_ticket_cancelled(
                      first_entry->transfer_ticket) ||
                  first_entry->generation <=
                      g_retired_transfer_generation.load(
                          std::memory_order_acquire));
             if (first_cancelled) {
                 first_entry->crypto_us = 0;
                 first_entry->busy.store(false, std::memory_order_release);
                 ++begin;
                 continue;
             }

             size_t end = begin + 1;
             uint64_t expected_offset =
                 first_entry->transfer_block_offset +
                 (first_entry->size + 15) / 16;
             while (end < enc_entries.size()) {
                 auto* previous = enc_entries[end - 1];
                 auto* candidate = enc_entries[end];
                 const bool candidate_cancelled =
                     secure_infer_linear_fast_fallback_enabled() &&
                     (candidate->cancelled.load(std::memory_order_acquire) ||
                      secure_infer_speculative_ticket_cancelled(
                          candidate->transfer_ticket) ||
                      candidate->generation <=
                          g_retired_transfer_generation.load(
                              std::memory_order_acquire));
                 if (candidate_cancelled ||
                     candidate->generation != previous->generation ||
                     candidate->transfer_ticket !=
                         previous->transfer_ticket ||
                     candidate->transfer_block_offset != expected_offset) {
                     break;
                 }
                 expected_offset += (candidate->size + 15) / 16;
                 ++end;
             }

             if (secure_infer_linear_fast_fallback_enabled() &&
                 first_entry->transfer_ticket != 0) {
                 secure_infer_derive_transfer_iv(
                     encrypt_init_iv, first_entry->transfer_ticket,
                     first_entry->transfer_block_offset, iv);
                 memcpy(first_entry->iv, iv, iv_length);
             } else {
                 next_iv(encrypt_init_iv, iv, first_entry->iv_offset);
             }
             uint64_t init_start = secure_infer_now_us();
             assert(real_EVP_EncryptInit_ex(
                        local_ctx, EVP_aes_128_ctr(), 0,
                        encrypt_key, iv) == 1);
             const uint64_t init_us = secure_infer_now_us() - init_start;

             for (size_t index = begin; index < end; ++index) {
                 auto* current = enc_entries[index];
                 int updated_len = 0;
                 uint64_t update_start = secure_infer_now_us();
                 assert(real_EVP_EncryptUpdate(
                            local_ctx, current->buffer, &updated_len,
                            static_cast<unsigned char*>(
                                const_cast<void*>(current->src)),
                            static_cast<int>(current->size)) == 1);
                 current->crypto_us =
                     secure_infer_now_us() - update_start +
                     (index == begin ? init_us : 0);
                 current->busy.store(false, std::memory_order_release);
             }
             unsigned char final_scratch[EVP_MAX_BLOCK_LENGTH] = {0};
             int final_len = 0;
             assert(real_EVP_EncryptFinal_ex(
                        local_ctx, final_scratch, &final_len) == 1);
             begin = end;
         }
         enc_entries.clear();
     }

     // 清理资源
     real_EVP_CIPHER_CTX_free(local_ctx);
     delete[] fake_src;
 }



 void commit_worker(void *entry)
 {
     auto x = (struct encrypt_worker_entry *)entry;
     auto worker_id = x->id;
     auto encrypt_workers_per_commit = x->encrypt_workers_per_commit;
     pin_latency_worker(0, "secure_infer-commit");

     // 绑定核心
     // bind_core((encrypt_workers_per_commit + 1) * worker_id);

     aclrtStream stream;
     aclrtStream keystream_stream = nullptr;
     std::vector<aclrtStream> gcm_keystream_streams;
     size_t next_gcm_keystream_stream = 0;

      // 等待 Context
      while (g_main_ctx == nullptr) { usleep(1000); }
      assert(aclrtSetCurrentContext(g_main_ctx) == ACL_SUCCESS);
      {
          int32_t cur_dev = -1;
          real_aclrtGetDevice(&cur_dev);
          int32_t dev_count = 0;
          real_aclrtGetDeviceCount(&dev_count);
          fprintf(stderr, "[CommitWorker] aclrtGetDevice=%d, deviceCount=%d\n", cur_dev, dev_count);
          fflush(stderr);
      }
      assert(real_aclrtCreateStream(&stream) == ACL_SUCCESS);
      if (npu_keystream_prefetch_enabled() ||
          secure_infer_gcm_keystream_scheduler_enabled()) {
          assert(create_gcm_keystream_stream(&keystream_stream) == ACL_SUCCESS);
          if (npu_keystream_prefetch_enabled() ||
              (secure_infer_gcm_enabled() &&
               secure_infer_gcm_keystream_scheduler_enabled())) {
              gcm_keystream_streams.push_back(keystream_stream);
              const size_t requested =
                  keystream_producer_stream_count();
              for (size_t index = 1; index < requested; ++index) {
                  aclrtStream extra_stream = nullptr;
                  if (create_gcm_keystream_stream(&extra_stream) != ACL_SUCCESS ||
                      extra_stream == nullptr) {
                      fprintf(stderr,
                              "[SecureInfer WARN] could not create GCM ticket "
                              "producer stream %zu; using %zu stream(s)\n",
                              index, gcm_keystream_streams.size());
                      break;
                  }
                  gcm_keystream_streams.push_back(extra_stream);
              }
          }
      }

     // 预分配临时显存
     void* reusable_temp_in = nullptr;
     void* reusable_temp_out = nullptr;
     size_t temp_buffer_size = 32 * 1024 * 1024; // 32MB
     assert(real_aclrtMalloc(&reusable_temp_in, temp_buffer_size) == ACL_SUCCESS);
     assert(real_aclrtMalloc(&reusable_temp_out, temp_buffer_size) == ACL_SUCCESS);

     sleep(1);
     x->commit_init_done.store(true, std::memory_order_release);

     // 获取元数据
     while(m_magic_encctx.find(x->magic_sz) == m_magic_encctx.end()) usleep(100);
     auto enc_ctx = m_magic_encctx[x->magic_sz];
     auto &enc_metadata = *m_ctx_metadata.at(enc_ctx);
     enc_metadata.commit_worker_id = worker_id;
     x->predict_iv_offset = enc_metadata.cur_iv_offset + 1;
     unsigned char commit_encrypt_key[key_length];
     unsigned char commit_encrypt_init_iv[iv_length];
     unsigned char commit_iv[iv_length];
     memcpy(commit_encrypt_key, enc_metadata.key, key_length);
     memcpy(commit_encrypt_init_iv, enc_metadata.init_iv, iv_length);
     if (keystream_stream != nullptr) {
         // Publish the immutable device key before either the speculative
         // stream or the demand stream can consume it.
         assert(secure_infer_prepare_h2d_device_crypto(keystream_stream));
         assert(real_aclrtSynchronizeStream(keystream_stream) == ACL_SUCCESS);
         if (secure_infer_gcm_enabled() &&
             secure_infer_gcm_keystream_scheduler_enabled()) {
             // GCM has one independent IV per physical H2D.  Optional
             // prewarming removes device-allocation latency from the short
             // descriptor-to-first-transfer lead time; it is off by default.
             secure_infer_gcm_keystream_prewarm(keystream_stream);
         } else {
             prewarm_npu_keystream_buffer_pool();
         }
     }
     EVP_CIPHER_CTX* inline_encrypt_ctx = real_EVP_CIPHER_CTX_new();
     assert(inline_encrypt_ctx != nullptr);
     prewarm_cipher_buffer_pool();

     // 任务队列
     std::deque<encryption_task> enc_tasks;
     std::deque<commit_task> commit_tasks;
     std::deque<commit_task> remain_commit_tasks;
     // 预测结果存储
     std::vector<std::tuple<const void*, size_t, void*, encryption_entry*>> predict_vec;
     std::deque<std::pair<void*, encryption_entry*>> abandoned_predictions;
     size_t next_encrypt_worker = 0;

     struct npu_keystream_slab {
         uint64_t generation = 0;
         uint64_t transfer_ticket = 0;
         size_t bytes = 0;
         uint32_t members = 1;
         size_t remaining_members = 1;
         void* allocation = nullptr;
         aclrtEvent ready_event = nullptr;
         std::unordered_set<uint64_t> consumed_block_offsets;

         ~npu_keystream_slab() {
             if (ready_event != nullptr) {
                 real_aclrtDestroyEvent(ready_event);
             }
             if (allocation != nullptr) {
                 release_npu_keystream_buffer(allocation, bytes);
             }
         }
     };
     std::unordered_map<uint64_t, std::shared_ptr<npu_keystream_slab>>
         npu_keystream_slabs;
     std::deque<std::shared_ptr<npu_keystream_slab>>
         retired_npu_keystream_slabs;
     std::unordered_map<uint64_t, uint64_t> rejected_npu_keystream_tickets;
     std::unordered_map<uint64_t, uint64_t> finished_npu_keystream_tickets;

     auto reserved_npu_keystream_members = [&]() {
         size_t reserved = 0;
         for (const auto& item : npu_keystream_slabs) {
             reserved += item.second->members;
         }
         for (const auto& slab : retired_npu_keystream_slabs) {
             reserved += slab->members;
         }
         return reserved;
     };

     auto cleanup_stale_npu_keystreams = [&]() {
         const uint64_t retired =
             g_retired_transfer_generation.load(std::memory_order_acquire);
         for (auto it = npu_keystream_slabs.begin();
              it != npu_keystream_slabs.end();) {
             auto slab = it->second;
             if (slab->generation > retired) {
                 ++it;
                 continue;
             }
             retired_npu_keystream_slabs.push_back(std::move(slab));
             it = npu_keystream_slabs.erase(it);
         }
         for (auto it = retired_npu_keystream_slabs.begin();
              it != retired_npu_keystream_slabs.end();) {
             const auto& slab = *it;
             aclrtEventRecordedStatus status =
                 ACL_EVENT_RECORDED_STATUS_NOT_READY;
             aclError ret =
                 aclrtQueryEventStatus(slab->ready_event, &status);
             if (ret == ACL_SUCCESS &&
                 status == ACL_EVENT_RECORDED_STATUS_COMPLETE &&
                 slab.use_count() == 1) {
                 it = retired_npu_keystream_slabs.erase(it);
             } else {
                 ++it;
             }
         }
         for (auto it = rejected_npu_keystream_tickets.begin();
              it != rejected_npu_keystream_tickets.end();) {
             if (it->second <= retired) {
                 it = rejected_npu_keystream_tickets.erase(it);
             } else {
                 ++it;
             }
         }
         for (auto it = finished_npu_keystream_tickets.begin();
              it != finished_npu_keystream_tickets.end();) {
             if (it->second <= retired) {
                 it = finished_npu_keystream_tickets.erase(it);
             } else {
                 ++it;
             }
         }
     };

     auto consume_npu_keystream_member =
         [&](uint64_t transfer_ticket, uint64_t transfer_block_offset) {
             auto it = npu_keystream_slabs.find(transfer_ticket);
             if (it == npu_keystream_slabs.end()) return;
             auto slab = it->second;
             if (!slab->consumed_block_offsets
                      .insert(transfer_block_offset).second) {
                 return;
             }
             if (slab->remaining_members > 0) {
                 slab->remaining_members--;
             }
             if (slab->remaining_members == 0) {
                 finished_npu_keystream_tickets.emplace(
                     transfer_ticket, slab->generation);
                 retired_npu_keystream_slabs.push_back(std::move(slab));
                 npu_keystream_slabs.erase(it);
             }
         };

     // 1=admitted/known, 0=ordinary retry, -1=dynamic low-priority deferral.
     // A deferred AIV ticket must never pin the FIFO head while a later
     // near-deadline ticket could still preserve the H2D hit rate.
     auto schedule_npu_keystream = [&](const encryption_task& task) -> int {
         if (keystream_stream == nullptr || task.transfer_ticket == 0 ||
             task.size == 0) {
             return 0;
         }
         const uint64_t retired =
             g_retired_transfer_generation.load(std::memory_order_acquire);
         if (task.generation <= retired) return 0;
         // GCM uses the global bridge because its verified consume path is
         // inline_h2d_encrypt_commit, not this CTR commit worker.  Do not
         // create a CPU-CTR prediction for a GCM task: the worker's sole
         // responsibility here is to fill the exact NPU ticket pool.
         if (secure_infer_gcm_enabled() &&
             secure_infer_gcm_keystream_scheduler_enabled()) {
            aclrtStream ticket_stream = keystream_stream;
            if (!gcm_keystream_streams.empty()) {
                ticket_stream = gcm_keystream_streams[
                    next_gcm_keystream_stream++ %
                    gcm_keystream_streams.size()];
            }
            return secure_infer_gcm_schedule_ticketed_keystream(
                 ticket_stream, task.transfer_ticket, task.size,
                 task.generation, task.admission_scope);
         }
         if (!npu_keystream_prefetch_enabled()) return 0;
         if (npu_keystream_slabs.find(task.transfer_ticket) !=
             npu_keystream_slabs.end()) {
             return 1;
         }
         if (finished_npu_keystream_tickets.find(task.transfer_ticket) !=
             finished_npu_keystream_tickets.end()) {
             return 1;
         }

         cleanup_stale_npu_keystreams();
         const uint32_t members =
             std::max<uint32_t>(task.ticket_batch_members, 1);
         if (reserved_npu_keystream_members() + members >
             max_npu_keystream_blocks()) {
             g_capacity_npu_keystream_fallbacks.fetch_add(
                 members, std::memory_order_relaxed);
             return 0;
         }

         size_t slab_bytes =
             task.ticket_batch_bytes != 0 ? task.ticket_batch_bytes : task.size;
         slab_bytes = (slab_bytes + 31) & ~static_cast<size_t>(31);
         if (slab_bytes == 0 || slab_bytes > UINT32_MAX) return false;

         auto slab = std::make_shared<npu_keystream_slab>();
         slab->generation = task.generation;
         slab->transfer_ticket = task.transfer_ticket;
         slab->bytes = slab_bytes;
         slab->members = members;
         slab->remaining_members = members;
         slab->allocation =
             alloc_npu_keystream_buffer(slab_bytes);
         if (slab->allocation == nullptr) {
                 return 0;
         }
         if (real_aclrtCreateEvent(&slab->ready_event) != ACL_SUCCESS) {
             return false;
         }
         // Every slab owns its event.  Submitting independent tickets on
         // distinct producer streams therefore preserves the exact demand
         // dependency: the consumer queries the event belonging to this
         // slab before it ever reads its keystream buffer.
         aclrtStream ticket_stream = keystream_stream;
         if (!gcm_keystream_streams.empty()) {
             ticket_stream = gcm_keystream_streams[
                 next_gcm_keystream_stream++ %
                 gcm_keystream_streams.size()];
         }
         if (!secure_infer_enqueue_ticketed_keystream(
                 ticket_stream, slab->allocation, slab_bytes,
                 task.transfer_ticket, 0)) {
             return false;
         }
         if (real_aclrtRecordEvent(slab->ready_event,
                                   ticket_stream) != ACL_SUCCESS) {
             real_aclrtSynchronizeStream(ticket_stream);
             return false;
         }
         npu_keystream_slabs.emplace(task.transfer_ticket, std::move(slab));
         g_submitted_npu_keystream_blocks.fetch_add(
             members, std::memory_order_relaxed);
         return 1;
     };

     auto process_abandoned_predictions = [&]() {
         for (auto it = abandoned_predictions.begin(); it != abandoned_predictions.end();) {
             auto* entry_ptr = it->second;
             if (entry_ptr->busy.load(std::memory_order_acquire)) {
                 ++it;
                 continue;
             }
             free_cipher_buffer(it->first);
             delete entry_ptr;
             it = abandoned_predictions.erase(it);
         }
     };
     auto retire_stale_predictions = [&]() {
         if (!secure_infer_linear_fast_fallback_enabled()) return;
         const uint64_t retired =
             g_retired_transfer_generation.load(std::memory_order_acquire);
         for (auto it = predict_vec.begin(); it != predict_vec.end();) {
             auto* entry_ptr = std::get<3>(*it);
             if (entry_ptr->generation > retired) {
                 ++it;
                 continue;
             }
             entry_ptr->cancelled.store(true, std::memory_order_release);
             if (entry_ptr->busy.load(std::memory_order_acquire)) {
                 abandoned_predictions.push_back(
                     std::make_pair(std::get<2>(*it), entry_ptr));
             } else {
                 free_cipher_buffer(std::get<2>(*it));
                 delete entry_ptr;
             }
             it = predict_vec.erase(it);
         }
     };
     auto drain_abandoned_predictions = [&]() {
         for (auto& abandoned : abandoned_predictions) {
             auto* entry_ptr = abandoned.second;
             while (entry_ptr->busy.load(std::memory_order_acquire)) {
                 std::this_thread::yield();
             }
             free_cipher_buffer(abandoned.first);
             delete entry_ptr;
         }
         abandoned_predictions.clear();
     };

     // [新增] 垃圾回收队列
     std::deque<GCBatch> gc_queue;
     std::deque<std::shared_ptr<commit_sync_state>> sync_event_gc;
     size_t inflight_async_blocks = 0;

    fprintf(stderr, "[Commit Worker] 🚀 启动成功 (Pipeline Mode)\n");


    while (g_running.load()) {

         // 1. 尝试清理垃圾 (非阻塞)
         {
             size_t released = process_garbage_collection(gc_queue);
             inflight_async_blocks = released > inflight_async_blocks
                 ? 0
                 : inflight_async_blocks - released;
         }
         process_sync_events(sync_event_gc);
         retire_stale_predictions();
         process_abandoned_predictions();
         cleanup_stale_npu_keystreams();

         // 2. Polling 任务
         while (g_running.load() && commit_tasks.empty()) {
             bool has_queued_work = false;
             {
                 std::lock_guard<std::mutex> lock(x->lock);
                 has_queued_work = !x->commit_tasks.empty() || !x->enc_tasks.empty();
             }
             if (has_queued_work) break;
             // 在空闲等待时也可以顺便清理垃圾
             {
                 size_t released = process_garbage_collection(gc_queue);
                 inflight_async_blocks = released > inflight_async_blocks
                     ? 0
                     : inflight_async_blocks - released;
             }
             process_sync_events(sync_event_gc);
             retire_stale_predictions();
             process_abandoned_predictions();
             cleanup_stale_npu_keystreams();
             if (secure_infer_gcm_keystream_scheduler_enabled()) {
                 secure_infer_gcm_keystream_cleanup();
             }

             constexpr int polling_us = 1;
             auto start = std::chrono::system_clock::now();
             while (1) {
                 auto end = std::chrono::system_clock::now();
                 if (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() >= polling_us) break;
             }
         }
         if (!g_running.load()) break;

         // Coalesce only speculative GCM tickets.  A real commit task is a
         // fills x->enc_tasks under the same short lock, while this worker
         // sleeps outside it, so descriptor discovery continues normally.
         const uint64_t coalesce_us = gcm_admission_coalesce_us();
         if (coalesce_us != 0 &&
             secure_infer_gcm_keystream_scheduler_enabled() &&
             commit_tasks.empty()) {
             const auto deadline = std::chrono::steady_clock::now() +
                 std::chrono::microseconds(coalesce_us);
             while (g_running.load() &&
                    std::chrono::steady_clock::now() < deadline) {
                 size_t queued = enc_tasks.size();
                 bool demand_waiting = false;
                 {
                     std::lock_guard<std::mutex> lock(x->lock);
                     queued += x->enc_tasks.size();
                     demand_waiting = !x->commit_tasks.empty();
                 }
                 if (demand_waiting ||
                     queued >= gcm_admission_coalesce_target()) {
                     break;
                 }
                 std::this_thread::sleep_for(std::chrono::microseconds(50));
             }
         }

         // 3. 获取任务 (加锁)
         std::vector<encryption_task> npu_admission_tasks;
         {
             std::lock_guard<std::mutex> lock(x->lock);
             // Unadmitted jobs from a finished scope cannot become useful.
             // Remove them before the bounded admission scan; otherwise an
             // old head of queue can prevent every later request's producer.
             if (secure_infer_gcm_keystream_scheduler_enabled()) {
                 const uint64_t retired = g_retired_transfer_generation.load(std::memory_order_acquire);
                 auto discard_retired = [retired](auto& queue) {
                     queue.erase(std::remove_if(queue.begin(), queue.end(),
                         [retired](const auto& task) { return task.generation <= retired; }), queue.end());
                 };
                 discard_retired(enc_tasks);
                 discard_retired(x->enc_tasks);
             }
             for (auto &task : x->commit_tasks) commit_tasks.push_back(task);
             x->commit_tasks.clear();
             if (!secure_infer_linear_fast_fallback_enabled()) {
                 enc_tasks.swap(x->enc_tasks);
             } else if (!x->enc_tasks.empty()) {
                 // Do not starve preparation under a continuous H2D burst.
                 // The former demand-first rule admitted no speculative CPU
                 // work whenever a commit was pending, so a correctly
                 // predicted rolling window remained stuck in x->enc_tasks
                 // and appeared as a "missing" result at consumption.  Pull
                 // a bounded prefix on every round.  The tensor-boundary
                 // preemption below still returns the unused suffix before
                 // processing the demand, limiting added host scheduling
                 // latency while guaranteeing forward progress.
                 const size_t take = std::min(
                     speculative_cpu_admission_quantum(),
                     x->enc_tasks.size());
                 for (size_t index = 0; index < take; ++index) {
                     enc_tasks.push_back(std::move(x->enc_tasks.front()));
                     x->enc_tasks.pop_front();
                 }
             }
             if ((npu_keystream_prefetch_enabled() ||
                  secure_infer_gcm_keystream_scheduler_enabled()) &&
                 (!x->enc_tasks.empty() || !enc_tasks.empty())) {
                 const bool gcm_ticket_pool = secure_infer_gcm_enabled() &&
                     secure_infer_gcm_keystream_scheduler_enabled();
                 // The backend switch is useful only if a future request's
                 // tickets are actually admitted while GHASH is under high
                 // AICPU pressure.  At all other times preserve FIFO/EDF-like
                 // behaviour; when the native relief gate opens, split just
                 // this admission round among queued logical H2D jobs.
                 std::unordered_set<uint64_t> queued_scopes;
                 const bool scope_aware = gcm_ticket_pool &&
                     gcm_scope_aware_admission_enabled() &&
                     secure_infer_gcm_dynamic_relief_active();
                 if (scope_aware) {
                     auto collect_scopes = [&](const auto& queue) {
                         for (const auto& queued : queue) {
                             if (queued.transfer_ticket != 0 &&
                                 queued.admission_scope != 0) {
                                 queued_scopes.insert(
                                     queued.admission_scope);
                             }
                         }
                     };
                     collect_scopes(enc_tasks);
                     collect_scopes(x->enc_tasks);
                 }
                 const size_t per_scope_budget = queued_scopes.empty()
                     ? npu_keystream_admission_tickets()
                     : std::max<size_t>(1, npu_keystream_admission_tickets() /
                                            queued_scopes.size());
                 std::unordered_map<uint64_t, size_t> selected_per_scope;
                 std::unordered_set<uint64_t> selected_tickets;
                 auto collect_npu_admission =
                     [&](const std::deque<encryption_task>& queue) {
                         for (const auto& queued : queue) {
                             if (queued.transfer_ticket == 0 ||
                                 (!gcm_ticket_pool &&
                                  (npu_keystream_slabs.count(
                                      queued.transfer_ticket) != 0 ||
                                   finished_npu_keystream_tickets.count(
                                      queued.transfer_ticket) != 0)) ||
                                 (gcm_ticket_pool &&
                                  secure_infer_gcm_keystream_known(
                                      queued.transfer_ticket)) ||
                                 !selected_tickets.insert(
                                     queued.transfer_ticket).second) {
                                 continue;
                             }
                             if (scope_aware &&
                                 queued.admission_scope != 0 &&
                                 selected_per_scope[queued.admission_scope] >=
                                     per_scope_budget) {
                                 // This was only a duplicate probe.  Leave
                                 // the ticket eligible for the next round.
                                 selected_tickets.erase(
                                     queued.transfer_ticket);
                                 continue;
                             }
                             npu_admission_tasks.push_back(queued);
                             if (scope_aware &&
                                 queued.admission_scope != 0) {
                                 ++selected_per_scope[
                                     queued.admission_scope];
                             }
                             if (npu_admission_tasks.size() >=
                                 npu_keystream_admission_tickets()) {
                                 return;
                             }
                         }
                     };
                 collect_npu_admission(enc_tasks);
                 if (npu_admission_tasks.size() <
                     npu_keystream_admission_tickets()) {
                     collect_npu_admission(x->enc_tasks);
                 }
             }
         }

         // 4. 处理纯加密任务 (Predict Prefetch)
         // Admit distinct future tensor tickets without waiting for all 63
         // CPU-prediction members of the current tensor to drain first.
         const bool gcm_ticket_pool = secure_infer_gcm_enabled() &&
             secure_infer_gcm_keystream_scheduler_enabled();
         std::unordered_set<uint64_t> admitted_gcm_tickets;
         std::unordered_set<uint64_t> deferred_gcm_tickets;
         if (gcm_ticket_pool && secure_infer_gcm_aicpu_batching_enabled()) {
             // The batch producer is restricted to fixed AICPU policy.  It
             // removes per-ticket launch/key-setup overhead without merging
             // IVs, outputs, or completion events.  Dynamic policy keeps the
             // existing individual admission path below.
             const size_t batch_limit =
                 secure_infer_gcm_aicpu_batch_ticket_limit();
             for (size_t begin = 0; begin < npu_admission_tasks.size();) {
                 const size_t bytes = npu_admission_tasks[begin].size;
                 size_t end = begin;
                 while (end < npu_admission_tasks.size() &&
                        end - begin < batch_limit &&
                        npu_admission_tasks[end].size == bytes) {
                     ++end;
                 }
                 std::vector<gcm_keystream_submission> submissions;
                 submissions.reserve(end - begin);
                 for (size_t index = begin; index < end; ++index) {
                     const auto& task = npu_admission_tasks[index];
                     submissions.push_back({task.transfer_ticket, task.size,
                                            task.generation,
                                            task.admission_scope});
                }
                std::vector<int> admissions;
                aclrtStream ticket_stream = keystream_stream;
                if (!gcm_keystream_streams.empty()) {
                    ticket_stream = gcm_keystream_streams[
                        next_gcm_keystream_stream++ %
                        gcm_keystream_streams.size()];
                }
                const bool batched = submissions.size() > 1 &&
                    secure_infer_gcm_schedule_ticketed_keystream_batch(
                        ticket_stream, submissions, &admissions);
                 for (size_t index = begin; index < end; ++index) {
                     const auto& task = npu_admission_tasks[index];
                     const int admission = batched
                         ? admissions[index - begin]
                         : schedule_npu_keystream(task);
                     if (admission > 0) {
                         admitted_gcm_tickets.insert(task.transfer_ticket);
                     } else if (admission < 0) {
                         deferred_gcm_tickets.insert(task.transfer_ticket);
                     }
                 }
                 begin = end;
             }
         } else {
             for (const auto& task : npu_admission_tasks) {
                 const int admission = schedule_npu_keystream(task);
                 if (admission > 0 && gcm_ticket_pool) {
                     admitted_gcm_tickets.insert(task.transfer_ticket);
                 } else if (admission < 0 && gcm_ticket_pool) {
                     deferred_gcm_tickets.insert(task.transfer_ticket);
                 }
             }
         }
         if (gcm_ticket_pool) {
             // GCM consumes the ticket pool in the inline verified path.  Do
             // not let this CTR worker create host ciphertext predictions for
             // the same source addresses.  Crucially, retain tickets that
             // were not admitted in this round: clearing the whole queue here
             // silently capped each descriptor batch at the admission quantum
             // (normally 8) and produced artificial "missing" fallbacks.
             auto reconcile_gcm_admission = [&](auto& queue) {
                 std::deque<encryption_task> deferred;
                 for (auto it = queue.begin(); it != queue.end();) {
                     if (admitted_gcm_tickets.count(it->transfer_ticket) != 0) {
                         it = queue.erase(it);
                     } else if (deferred_gcm_tickets.count(
                                    it->transfer_ticket) != 0) {
                         // Keep the original task intact, but place it after
                         // all currently runnable tasks.  It becomes eligible
                         // again once AIV has room or its deadline enters the
                         // protected AICPU window.
                         deferred.push_back(std::move(*it));
                         it = queue.erase(it);
                     } else {
                         ++it;
                     }
                 }
                 queue.insert(queue.end(),
                              std::make_move_iterator(deferred.begin()),
                              std::make_move_iterator(deferred.end()));
             };
             reconcile_gcm_admission(enc_tasks);
             if (!admitted_gcm_tickets.empty() ||
                 !deferred_gcm_tickets.empty()) {
                 std::lock_guard<std::mutex> lock(x->lock);
                 reconcile_gcm_admission(x->enc_tasks);
             } else if (!enc_tasks.empty()) {
                 // Water-level backpressure is intentionally retryable.  Do
                 // not convert it into a tight CPU polling loop while GHASH
                 // owns AICPU; the next short tick observes either a consumed
                 // ticket or the end of the critical interval.
                 usleep(50);
             }
         } else {
         // CPU admission follows the same logical-tensor boundary as the
         // ticket slab. If a real transfer appears, finish the current tensor
         // and put all later tensors back before serving demand.
         std::array<std::vector<encryption_entry*>, 10>
             staged_cpu_entries;
         auto flush_staged_cpu_entries = [&]() {
             for (int worker = 0;
                  worker < encrypt_workers_per_commit; ++worker) {
                 auto& staged = staged_cpu_entries[worker];
                 if (staged.empty()) continue;
                 std::lock_guard<std::mutex> lock(x->enc_lock[worker]);
                 x->enc_entries[worker].insert(
                     x->enc_entries[worker].end(),
                     staged.begin(), staged.end());
                 staged.clear();
             }
         };
         uint64_t active_prediction_group = 0;
         auto task_it = enc_tasks.begin();
         for (; task_it != enc_tasks.end(); ++task_it) {
              // If the real demand already used the serial correctness path,
              // completing its queued prediction has zero value and only
              // pushes later deadlines back.
              if (secure_infer_speculative_ticket_cancelled(
                      task_it->transfer_ticket)) {
                  continue;
              }
              const uint64_t task_group =
                  task_it->prefetch_group_id != 0
                      ? task_it->prefetch_group_id
                      : task_it->transfer_ticket;
              const bool tensor_boundary =
                  active_prediction_group == 0 ||
                  active_prediction_group != task_group;
              if (tensor_boundary && active_prediction_group != 0) {
                  flush_staged_cpu_entries();
                  std::lock_guard<std::mutex> lock(x->lock);
                  if (!x->commit_tasks.empty()) {
                      x->enc_tasks.insert(
                          x->enc_tasks.begin(), task_it, enc_tasks.end());
                      break;
                  }
              }
              active_prediction_group = task_group;
              auto& task = *task_it;
              // One descriptor task is sufficient to launch the full
              // tensor-ticket slab. Later members only observe the existing
              // ticket and return, so demand can consume any block even when
              // CPU prediction admission remains deadline-bounded.
              schedule_npu_keystream(task);
              auto count = task.size;
              auto base = (const char*)task.src;
              x->predict_iv_offset += task.iv_increment;
              uint64_t ticket_block_offset = task.transfer_block_offset;
              size_t div = (count + encrypt_workers_per_commit - 1) / encrypt_workers_per_commit;
              div = (div + block_unit - 1) / block_unit * block_unit;
              auto blocks = (div + block_unit - 1) / block_unit;
              for (int i = 0; i < encrypt_workers_per_commit; i++) {
                  for (int j = 0; j < blocks; j++) {
                      auto chunk_size = std::min(count, block_unit);
                      void* cipher_buffer = alloc_cipher_buffer(chunk_size);
                      auto enc_entry = new encryption_entry;
                      enc_entry->src = base;
                      enc_entry->buffer = (unsigned char*)cipher_buffer;
                      enc_entry->size = chunk_size;
                      enc_entry->busy.store(true, std::memory_order_relaxed);
                      enc_entry->iv_offset = x->predict_iv_offset;
                      enc_entry->transfer_ticket = task.transfer_ticket;
                      enc_entry->transfer_block_offset = ticket_block_offset;
                      enc_entry->crypto_us = 0;
                      enc_entry->generation = task.generation;
                      predict_vec.push_back(std::make_tuple((const void*)base, chunk_size, cipher_buffer, enc_entry));
                      int target_worker = 0;
                      if (task.ticket_batch_members > 1 && task.size > 0) {
                          const size_t member_index = std::min<size_t>(
                              static_cast<size_t>(
                                  (ticket_block_offset * 16) / task.size),
                              task.ticket_batch_members - 1);
                          target_worker = static_cast<int>(
                              std::min<size_t>(
                                  encrypt_workers_per_commit - 1,
                                  member_index *
                                      encrypt_workers_per_commit /
                                      task.ticket_batch_members));
                      } else {
                          target_worker = static_cast<int>(
                              next_encrypt_worker++ %
                              static_cast<size_t>(
                                  encrypt_workers_per_commit));
                      }
                      staged_cpu_entries[target_worker].push_back(enc_entry);
                      x->predict_iv_offset += (chunk_size + 15) / 16;
                      ticket_block_offset += (chunk_size + 15) / 16;
                      if (predict_vec.size() > max_predicted_blocks()) {
                          auto old = predict_vec.front();
                          auto* old_entry = std::get<3>(old);
                          if (old_entry->busy.load(std::memory_order_acquire)) {
                              if (secure_infer_linear_fast_fallback_enabled()) {
                                  old_entry->cancelled.store(
                                      true, std::memory_order_release);
                              } else {
                                  old_entry->generation = 0;
                              }
                              abandoned_predictions.push_back(
                                  std::make_pair(std::get<2>(old), old_entry));
                          } else {
                              free_cipher_buffer(std::get<2>(old));
                              delete old_entry;
                          }
                          predict_vec.erase(predict_vec.begin());
                      }
                      base += chunk_size;
                      count -= chunk_size;
                      if (count == 0) break;
                  }
                  if (count == 0) break;
              }
         }
         flush_staged_cpu_entries();
         enc_tasks.clear();
         }

         // Demands can arrive while one tensor's CPU entries are admitted.
         // Pull them into this iteration so the hook thread does not wait for
         // another polling cycle.
         {
             std::lock_guard<std::mutex> lock(x->lock);
             for (auto& demand : x->commit_tasks) {
                 commit_tasks.push_back(std::move(demand));
             }
             x->commit_tasks.clear();
         }

         // 5. 处理提交任务 (Pipeline Commit)
         // 准备当前的 GC Batch
         GCBatch current_gc_batch;
         bool has_async_ops = false;

         for (auto &task : commit_tasks) {
             if (task.using_predict) {
                 auto count = task.enc_task.size;
                 auto base_src = (const char*)task.enc_task.src;
                 auto base_dst = (char*)task.dst;
                 uint64_t demand_block_offset =
                     task.enc_task.transfer_block_offset;

                 size_t div = (count + encrypt_workers_per_commit - 1) / encrypt_workers_per_commit;
                 div = (div + block_unit - 1) / block_unit * block_unit;
                 auto blocks = (div + block_unit - 1) / block_unit;

                 for (int i = 0; i < encrypt_workers_per_commit; i++) {
                     for (int j = 0; j < blocks; j++) {
                         auto chunk_size = std::min(count, block_unit);
                         void* cipher_buffer = nullptr;
                         encryption_entry* entry_ptr = nullptr;

                         // 在 predict_vec 中查找对应的加密块
                         auto it = predict_vec.begin();
                         bool found = false;
                         for (; it != predict_vec.end(); ++it) {
                             auto* candidate = std::get<3>(*it);
                             if (std::get<0>(*it) == base_src &&
                                 std::get<1>(*it) == chunk_size &&
                                 candidate->generation == task.enc_task.generation &&
                                 (!secure_infer_linear_fast_fallback_enabled() ||
                                  (candidate->transfer_ticket ==
                                       task.enc_task.transfer_ticket &&
                                   candidate->transfer_block_offset ==
                                       demand_block_offset)) &&
                                 !candidate->cancelled.load(std::memory_order_acquire)) {
                                 cipher_buffer = std::get<2>(*it);
                                 entry_ptr = std::get<3>(*it);
                                 found = true;
                                 break;
                             }
	                         }

	                         bool used_ready_prediction = false;
	                         uint64_t host_wait_us = 0;
	                         uint64_t cpu_crypto_us = 0;

	                         if (found && !entry_ptr->busy.load(std::memory_order_acquire)) {
	                             used_ready_prediction = true;
	                             cpu_crypto_us = entry_ptr->crypto_us;
	                             g_transfer_prefetch_ready.fetch_add(
	                                 1, std::memory_order_relaxed);
	                         } else {
	                             // The demand deadline has passed.  Cancel a
	                             // task that may still be waiting in the
	                             // commit-worker or AES-worker queue, not just
	                             // an entry already visible in predict_vec.
	                             secure_infer_cancel_speculative_ticket(
	                                 task.enc_task.transfer_ticket);
	                             if (found) {
	                                 g_transfer_prefetch_busy_fallback.fetch_add(
	                                     1, std::memory_order_relaxed);
	                                 // The address prediction was right, but CPU encryption has not
	                                 // finished yet. Waiting here was the source of 15-20ms
	                                 // host_wait spikes. Treat it as a miss and fall back to
	                                 // immediate serial encryption for the current block.
	                                 if (secure_infer_linear_fast_fallback_enabled()) {
	                                     entry_ptr->cancelled.store(
	                                         true, std::memory_order_release);
	                                 } else {
	                                     entry_ptr->generation = 0;
	                                 }
	                                 abandoned_predictions.push_back(std::make_pair(cipher_buffer, entry_ptr));
	                                 predict_vec.erase(it);
	                                 static std::atomic<uint64_t> busy_prediction_log_count{0};
	                                 if (sampled_predict_log(busy_prediction_log_count, 32, 2048)) {
	                                     fprintf(stderr,
	                                             "[SecureInfer] 预测命中但尚未完成，当前块改走原地串行加密: src=%p, size=%zu\n",
	                                             base_src, chunk_size);
	                                 }
	                             } else {
	                                 g_transfer_prefetch_missing_fallback.fetch_add(
	                                     1, std::memory_order_relaxed);
	                                 static std::atomic<uint64_t> missing_prediction_log_count{0};
	                                 if (sampled_predict_log(missing_prediction_log_count, 16, 512)) {
	                                     fprintf(stderr,
	                                             "[SecureInfer] 预测结果不存在，当前块改走原地串行加密: src=%p, size=%zu\n",
	                                             base_src,
	                                             chunk_size);
	                                 }
	                             }

	                             cipher_buffer = alloc_cipher_buffer(chunk_size);
	                             if (secure_infer_linear_fast_fallback_enabled()) {
	                                 secure_infer_derive_transfer_iv(
	                                     commit_encrypt_init_iv,
	                                     task.enc_task.transfer_ticket,
	                                     demand_block_offset,
	                                     commit_iv);
	                             } else {
	                                 iv_t iv_offset =
	                                     enc_metadata.cur_iv_offset + 1;
	                                 enc_metadata.cur_iv_offset++;
	                                 next_iv(commit_encrypt_init_iv, commit_iv,
	                                         iv_offset);
	                             }
	                             int updated_len = 0;
	                             int final_len = 0;
	                             uint64_t crypto_start = secure_infer_now_us();
	                             assert(real_EVP_EncryptInit_ex(inline_encrypt_ctx, EVP_aes_128_ctr(), 0,
	                                                            commit_encrypt_key, commit_iv) == 1);
	                             assert(real_EVP_EncryptUpdate(inline_encrypt_ctx,
	                                                           (unsigned char*)cipher_buffer,
	                                                           &updated_len,
	                                                           (unsigned char*)base_src,
	                                                           (int)chunk_size) == 1);
	                             real_EVP_EncryptFinal_ex(inline_encrypt_ctx,
	                                                      (unsigned char*)cipher_buffer + updated_len,
	                                                      &final_len);
	                             cpu_crypto_us = secure_infer_now_us() - crypto_start;
	                         }

	                         uint64_t copy_enqueue_us = 0;
	                         uint64_t kernel_enqueue_us = 0;
	                         uint64_t device_start = secure_infer_now_us();
	                         size_t kernel_size = (chunk_size + 31) & ~31;  // 强制 32 字节对齐
	                         size_t aligned_bytes = kernel_size;

	                         uint64_t copy_start = secure_infer_now_us();
	                         real_aclrtMemcpyAsync(reusable_temp_in, chunk_size, cipher_buffer, chunk_size,
	                                               ACL_MEMCPY_HOST_TO_DEVICE, stream);

	                         if (kernel_size > chunk_size) {
	                             uint8_t zero_pad[32] = {0};
	                             real_aclrtMemcpyAsync((uint8_t*)reusable_temp_in + chunk_size,
	                                                   kernel_size - chunk_size,
	                                                   zero_pad, kernel_size - chunk_size,
	                                                   ACL_MEMCPY_HOST_TO_DEVICE, stream);
	                         }
	                         copy_enqueue_us += secure_infer_now_us() - copy_start;

	                         uint64_t kernel_start = secure_infer_now_us();
	                         bool kernel_enqueued = false;
	                         if (secure_infer_linear_fast_fallback_enabled()) {
	                             uint64_t ticket = task.enc_task.transfer_ticket;
	                             void* ready_keystream = nullptr;
	                             std::shared_ptr<npu_keystream_slab>
	                                 ready_keystream_owner;
	                             if (npu_keystream_prefetch_enabled()) {
	                                 auto slab_it =
	                                     npu_keystream_slabs.find(ticket);
	                                 if (slab_it !=
	                                     npu_keystream_slabs.end() &&
	                                     demand_block_offset <=
	                                         SIZE_MAX / 16) {
	                                     auto slab = slab_it->second;
	                                     const size_t byte_offset =
	                                         static_cast<size_t>(
	                                             demand_block_offset) * 16;
	                                     if (byte_offset <= slab->bytes &&
	                                         kernel_size <=
	                                             slab->bytes - byte_offset) {
	                                         aclrtEventRecordedStatus status =
	                                             ACL_EVENT_RECORDED_STATUS_NOT_READY;
	                                         aclError query_ret =
	                                             aclrtQueryEventStatus(
	                                                 slab->ready_event,
	                                                 &status);
	                                         if (query_ret == ACL_SUCCESS &&
	                                             status ==
	                                                 ACL_EVENT_RECORDED_STATUS_COMPLETE) {
	                                             ready_keystream =
	                                                 static_cast<uint8_t*>(
	                                                     slab->allocation) +
	                                                 byte_offset;
	                                             ready_keystream_owner =
	                                                 std::move(slab);
	                                             g_ready_npu_keystream_commits
	                                                 .fetch_add(
	                                                     1,
	                                                     std::memory_order_relaxed);
	                                         } else {
	                                             g_busy_npu_keystream_fallbacks
	                                                 .fetch_add(
	                                                     1,
	                                                     std::memory_order_relaxed);
	                                         }
	                                         consume_npu_keystream_member(
	                                             ticket,
	                                             demand_block_offset);
	                                     } else {
	                                         g_missing_npu_keystream_fallbacks
	                                             .fetch_add(
	                                                 1,
	                                                 std::memory_order_relaxed);
	                                     }
	                                 } else {
	                                     g_missing_npu_keystream_fallbacks
	                                         .fetch_add(
	                                             1,
	                                             std::memory_order_relaxed);
	                                 }
	                             }

	                             if (ready_keystream != nullptr) {
	                                 kernel_enqueued =
	                                     secure_infer_enqueue_precomputed_xor(
	                                         stream, reusable_temp_in,
	                                         ready_keystream,
	                                         reusable_temp_out,
	                                         kernel_size);
	                                 if (kernel_enqueued) {
	                                     current_gc_batch.device_owners
	                                         .push_back(
	                                             std::static_pointer_cast<void>(
	                                                 std::move(
	                                                     ready_keystream_owner)));
	                                 }
	                             }
	                             if (!kernel_enqueued) {
	                                 // Non-ready speculation never delays a
	                                 // demand: immediately execute full AES
	                                 // with the exact same ticket/counter.
	                                 kernel_enqueued =
	                                     secure_infer_enqueue_ticketed_ctr_xor(
	                                         stream, reusable_temp_in,
	                                         reusable_temp_out, kernel_size,
	                                         ticket,
	                                         demand_block_offset);
	                             }
	                         } else {
	                             if (!g_add_custom_do) init_npu_kernel();
	                             if (g_add_custom_do && g_dev_key && g_dev_iv) {
	                                 uint32_t blockDim = calc_aes_block_dim(kernel_size);
	                                 uint32_t mode = 0;
	                                 g_add_custom_do(
	                                     blockDim, stream,
	                                     (uint8_t*)reusable_temp_in,
	                                     (uint8_t*)g_dev_key,
	                                     (uint8_t*)g_dev_iv,
	                                     (uint8_t*)reusable_temp_out,
	                                     (uint32_t)kernel_size, mode);
	                                 kernel_enqueued = true;
	                             }
	                         }
	                         kernel_enqueue_us += secure_infer_now_us() - kernel_start;
	                         if (!kernel_enqueued) {
	                             fprintf(stderr,
	                                     "[SecureInfer FATAL] ticketed NPU CTR kernel enqueue failed\n");
	                             std::abort();
	                         }

	                         copy_start = secure_infer_now_us();
	                         real_aclrtMemcpyAsync(base_dst, chunk_size, reusable_temp_out, chunk_size,
	                                               ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
	                         copy_enqueue_us += secure_infer_now_us() - copy_start;

	                         uint64_t profile_start = task.profile_start_us ? task.profile_start_us : device_start;
	                         profile_record("H2D",
	                                        chunk_size,
	                                        aligned_bytes,
	                                        secure_infer_now_us() - profile_start,
	                                        cpu_crypto_us,
	                                        secure_infer_now_us() - device_start,
	                                        host_wait_us,
	                                        copy_enqueue_us,
	                                        kernel_enqueue_us);

	                         current_gc_batch.buffers.push_back(cipher_buffer);
	                         if (used_ready_prediction) {
	                             current_gc_batch.entries.push_back(entry_ptr);
	                             predict_vec.erase(it);
	                         }
	                         current_gc_batch.blocks++;
	                         has_async_ops = true;

                         // [关键修改] 移除了 aclrtSynchronizeStream(stream);
                         // 这实现了完全的流水线：CPU 不断塞任务，NPU 后台排队执行。

                         base_src += chunk_size;
                         base_dst += chunk_size;
                         demand_block_offset += (chunk_size + 15) / 16;
                         count -= chunk_size;
                         if (count == 0) break;
                     }
                     if (count == 0) break;
                 }
             } else {
                 // Fallback (非预测路径)
                 auto chunk_size = task.enc_task.size;
                 assert(real_aclrtMemcpyAsync(task.dst, chunk_size, task.enc_task.src, chunk_size, ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS);
                 // Fallback 路径为了安全可以保留同步，或者也可以优化
                //  aclrtSynchronizeStream(stream);
             }


             if (task.sync) {
                // 在 worker 的流 (stream) 上记录 event
                // 意义：当 NPU 执行完上面的 H2D 之后，触发这个 event
                aclError ret = real_aclrtRecordEvent(task.sync->event, stream);
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[SecureInfer ERROR] RecordEvent failed ret=%d (kernel 之前可能已出错)\n", ret);
                    aclError sync_ret = real_aclrtSynchronizeStream(stream);
                    if (sync_ret != ACL_SUCCESS) {
                        fprintf(stderr, "[SecureInfer ERROR] Stream fallback sync failed ret=%d\n", sync_ret);
                    }
                }
                task.sync->status.store(ret, std::memory_order_release);
                task.sync->submitted.store(true, std::memory_order_release);
                sync_event_gc.push_back(task.sync);
            }
         }

         // 6. 提交垃圾回收 Event
         if (has_async_ops) {
             aclrtEvent event = nullptr;
             aclError ret = real_aclrtCreateEvent(&event);
             if (ret == ACL_SUCCESS) {
                 ret = real_aclrtRecordEvent(event, stream);
             }
             if (ret == ACL_SUCCESS) {
                 current_gc_batch.event = event;
                 inflight_async_blocks += current_gc_batch.blocks;
                 gc_queue.push_back(std::move(current_gc_batch));
                 while (inflight_async_blocks > max_commit_inflight_blocks() && !gc_queue.empty()) {
                     size_t released = wait_one_gc_batch(gc_queue);
                     inflight_async_blocks = released > inflight_async_blocks
                         ? 0
                         : inflight_async_blocks - released;
                 }
             } else {
                 // Resource exhaustion is an exceptional slow path. Waiting
                 // here avoids both a leak and freeing buffers still in use.
                 real_aclrtSynchronizeStream(stream);
                 for (void* buf : current_gc_batch.buffers) free_cipher_buffer(buf);
                 for (auto* enc_entry : current_gc_batch.entries) delete enc_entry;
                 if (event != nullptr) real_aclrtDestroyEvent(event);
             }
         }

         commit_tasks = remain_commit_tasks;
         remain_commit_tasks.clear();
         {
             std::lock_guard<std::mutex> lock(x->lock);
             if (x->commit_tasks.empty() && commit_tasks.empty()) {
                 x->commit.store(false, std::memory_order_release);
             }
         }
     }

     // 退出前清理
     {
         size_t released = process_garbage_collection(gc_queue);
         inflight_async_blocks = released > inflight_async_blocks
             ? 0
             : inflight_async_blocks - released;
     }
     // 如果还有剩余的，强制同步并释放
     if (!gc_queue.empty()) {
         aclrtSynchronizeStream(stream);
         for (auto& batch : gc_queue) {
            release_gc_batch(batch);
         }
     }
     drain_abandoned_predictions();
     if (keystream_stream != nullptr) {
         // A GCM ticket may have been submitted on any producer stream.  Do
         // not reclaim its globally tracked record/buffer until each stream
         // is drained.  The vector contains the primary stream as element 0.
         const bool has_gcm_ticket_streams = !gcm_keystream_streams.empty();
         for (aclrtStream ticket_stream : gcm_keystream_streams) {
             if (ticket_stream != nullptr) {
                 real_aclrtSynchronizeStream(ticket_stream);
             }
         }
         npu_keystream_slabs.clear();
         retired_npu_keystream_slabs.clear();
         rejected_npu_keystream_tickets.clear();
         finished_npu_keystream_tickets.clear();
         for (aclrtStream ticket_stream : gcm_keystream_streams) {
             if (ticket_stream != nullptr && ticket_stream != keystream_stream) {
                 aclrtDestroyStream(ticket_stream);
             }
         }
         gcm_keystream_streams.clear();
         // A legacy one-stream producer does not populate the vector.
         if (!has_gcm_ticket_streams) {
             real_aclrtSynchronizeStream(keystream_stream);
         }
         aclrtDestroyStream(keystream_stream);
         keystream_stream = nullptr;
         drain_npu_keystream_buffer_pool();
     }
     real_EVP_CIPHER_CTX_free(inline_encrypt_ctx);

     real_aclrtFree(reusable_temp_in);
     real_aclrtFree(reusable_temp_out);
 }

 /*
  * ==========================================================================
  * 函数: decrypt_worker (无需修改)
  * ==========================================================================
  */
 void decrypt_worker(void *entry)
 {
     // (此函数 100% 保持不变，它与 CUDA/ACL 无关)
     auto x = (struct decrypt_worker_entry *)entry;
     auto worker_id = x->id;
     bind_core(2 * worker_id + 4 + 1);

     std::deque<decryption_entry *> dec_entries;
     unsigned char decrypt_key[key_length];

     auto dec_ctx = m_magic_decctx[x->magic_sz];
     auto &dec_metadata = *m_ctx_dmetadata.at(dec_ctx);

     memcpy(decrypt_key, dec_metadata.key, key_length);

     x->dec_init_done = true;
     auto local_ctx = real_EVP_CIPHER_CTX_new();
     while (1) {
         while (x->dec_entries.empty()) {
             // Polling 1 us
             constexpr int polling_us = 1;
             auto start = std::chrono::system_clock::now();
             while (1) {
                 auto end = std::chrono::system_clock::now();
                 if (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() >= polling_us) {
                     break;
                 }
             }
         }

         if (!x->dec_entries.empty()){
             std::lock_guard<std::mutex> lock(x->dec_lock);
             dec_entries = x->dec_entries;
             x->dec_entries.clear();
         }

         for (auto &entry : dec_entries) {
             int updated_len;
             real_EVP_DecryptInit_ex(local_ctx, EVP_aes_128_ctr(), 0, decrypt_key, entry->iv);
             real_EVP_CIPHER_CTX_ctrl(local_ctx, EVP_CTRL_GCM_SET_TAG, 16, entry->tag);
             assert(real_EVP_DecryptUpdate(local_ctx, (unsigned char *)entry->dst, &updated_len, entry->buffer, entry->size) == 1);
             real_EVP_DecryptFinal_ex(local_ctx, (unsigned char *)entry->dst + updated_len, &updated_len);
             assert(updated_len == 0);
             entry->busy = false;
         }
         dec_entries.clear();
     }
 }

 /*
  * ==========================================================================
  * 函数: decrypt_manager (已迁移到 ACL)
  * ==========================================================================
  */
 void decrypt_manager(void *entry)
 {
     auto x = (struct decrypt_worker_entry *)entry;
     auto worker_id = x->id;
     bind_core(2 * worker_id + 4);

     // [!!! 关键修改: CUDA -> ACL !!!]
     // CUcontext cuda_ctx;  <-- [已移除]
     // CUdevice dev;         <-- [已移除]
     // cudaStream_t stream;  <-- [已移除]

     aclrtContext acl_ctx;    // <-- [新添加]
     aclrtStream stream;      // <-- [新添加]

     void *fake_src, *fake_dst; // fake_src 在您的代码中未被使用

     // [修改] 替换 CUDA Driver API 为 ACL Runtime API (直接调用)
    //  assert(aclInit(nullptr) == ACL_SUCCESS);
     assert(aclrtSetDevice(0) == ACL_SUCCESS);
     assert(aclrtCreateContext(&acl_ctx, 0) == ACL_SUCCESS);
     assert(aclrtSetCurrentContext(acl_ctx) == ACL_SUCCESS);

     // [修改] 替换 CUDA Runtime Hooks 为 ACL Runtime Hooks
     assert(real_aclrtCreateStream(&stream) == ACL_SUCCESS);
     assert(real_aclrtMallocHost(&fake_dst, 1ul << 20) == ACL_SUCCESS);
     sleep(1);
     void *dev_buffer;
     real_aclrtMalloc(&dev_buffer, 1ul << 20);

     // [修改] 调用 real_aclrt... 并添加 destMax 参数
     real_aclrtMemcpyAsync(fake_dst, x->magic_sz, dev_buffer, x->magic_sz, ACL_MEMCPY_DEVICE_TO_HOST, stream);
     real_aclrtMemcpyAsync(fake_dst, x->magic_sz, dev_buffer, x->magic_sz, ACL_MEMCPY_DEVICE_TO_HOST, stream);
     real_aclrtSynchronizeDevice();
     // [!!! 修改结束 !!!]

     x->commit_init_done = true;

     std::deque<decryption_task> dec_tasks;
     std::deque<decryption_entry *> dec_entries;
     std::deque<encryption_task> enc_tasks;
     auto dec_ctx = m_magic_decctx[x->magic_sz];
     auto &dec_metadata = *m_ctx_dmetadata.at(dec_ctx);
     dec_metadata.remain = 0;

     while (1) {
         // (内部轮询逻辑保持不变)
         while (x->dec_tasks.empty() && dec_entries.empty()) {
             // Polling 1 us
             constexpr int polling_us = 1;
             auto start = std::chrono::system_clock::now();
             while (1) {
                 auto end = std::chrono::system_clock::now();
                 if (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() >= polling_us) {
                     break;
                 }
             }
         }

         // (内部任务获取逻辑保持不变)
         if (!x->dec_tasks.empty()){
             std::lock_guard<std::mutex> lock(x->lock);
             dec_tasks = x->dec_tasks;
             x->dec_tasks.clear();
         }

         for (auto &task : dec_tasks) {
             dec_metadata.remain = 0;

             // [!!! 关键修改: CUDA -> ACL !!!]
             // (添加 destMax = task.size)
             assert(real_aclrtMemcpyAsync(dev_buffer, task.size, task.src, task.size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) == ACL_SUCCESS); // [修改]
             assert(real_aclrtMemcpyAsync(fake_dst, task.size, dev_buffer, task.size, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS); // [修改]
             // [!!! 修改结束 !!!]

             auto entry = new decryption_entry;
             entry->buffer = dec_metadata.cur_buffer;
             memcpy(entry->tag, dec_metadata.cur_tag, 16);
             memcpy(entry->iv, dec_metadata.cur_iv, iv_length);
             entry->size = task.size;
             entry->busy = true;
             entry->dst = task.dst;
             entry->busy = true;
             {
                 std::lock_guard<std::mutex> lock(x->dec_lock);
                 x->dec_entries.push_back(entry);
                 dec_entries.push_back(entry);
             }
         }


         // (内部 "Send to encryption" 逻辑保持不变)
         bool batch_first = true;
         for (auto iter = dec_entries.begin(); iter != dec_entries.end();) {
             auto &entry = *iter;
             if (entry->busy) {
                 ++iter;
                 continue;
             }
             encryption_task task;
             task.src = entry->dst;
             task.size = entry->size;
             task.iv_increment = batch_first ? dec_entries.size() : 0;
             batch_first = false;
             enc_tasks.push_back(task);

             // Remove buffer
             dec_metadata.allocator.free((unsigned char *)entry->buffer);
             delete entry;
             iter = dec_entries.erase(iter);
         }

         if (!enc_tasks.empty()) {
             auto worker_entry = x->enc_worker_entry;
             std::lock_guard<std::mutex> lock(worker_entry->lock);
             for (auto &task : enc_tasks) {
                 worker_entry->enc_tasks.push_back(task);
             }
         }

         dec_tasks.clear();
         enc_tasks.clear();
     }
 }

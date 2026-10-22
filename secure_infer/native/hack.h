/*
 * 文件名: hack.h (已修改为 ACL 完整版本)
 * 职责: 拦截 AscendCL (ACL) 和 OpenSSL API
 */
 #pragma once

 #include <dlfcn.h>
 #include <openssl/ssl.h>
 #include <stdio.h> // 添加 stdio.h 用于 fprintf
 #include <cstdlib>

 // 移除了 <cuda.h> 和 <cuda_runtime.h>
 // 增加了 ACL 的头文件
 #include <acl/acl.h>
 #include <acl/acl_rt.h>

 #define HOOK_C_API extern "C"
 #define HOOK_DECL_EXPORT __attribute__((visibility("default")))

 // 定义一个自定义错误码
 static int ACL_ERROR_LD_SO_NOT_LOADED = 500000;
 /*
  * ==========================================================================
  * ACL (AscendCL) 动态链接库钩子
  * (原 get_symbol_cuda 及 real_cuda... 函数已被替换)
  * ==========================================================================
  */

 static inline void *get_symbol_acl(const char *name)
 {
     const char *acl_lib_path = std::getenv("ASCENDCL_LIBRARY");
     if (acl_lib_path == nullptr || acl_lib_path[0] == '\0') {
         acl_lib_path = "libascendcl.so";
     }

     // 使用 static 确保只在第一次调用时 dlopen
     static void *handle = nullptr;
     if (handle == nullptr) {
         handle = dlopen(acl_lib_path, RTLD_NOW | RTLD_LOCAL);
         if (!handle) {
             // 增加一个错误处理，以便调试
             fprintf(stderr, "[SecureInfer Error] 关键错误: 无法 dlopen ACL 库: %s. 错误: %s\n", acl_lib_path, dlerror());
             // 尝试一个备用路径（例如，如果它在系统标准路径中）
             handle = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
             if (!handle) {
                  fprintf(stderr, "[SecureInfer Error] 关键错误: 无法从标准路径 dlopen libascendcl.so. 错误: %s\n", dlerror());
                  return nullptr; // 严重错误，无法继续
             }
             fprintf(stderr, "[SecureInfer Info] 从标准路径成功加载 libascendcl.so\n");
         } else {
             fprintf(stderr, "[SecureInfer Info] 从 %s 成功加载 libascendcl.so\n", acl_lib_path);
         }
     }

     void* symbol = dlsym(handle, name);
     if (!symbol) {
         fprintf(stderr, "[SecureInfer Error] 关键错误: 无法 dlsym 符号 %s. 错误: %s\n", name, dlerror());
     }
     // 注意：我们不再 dlclose(handle)，以保持库加载状态
     return symbol;
 }

 // 替换 real_cudaMemcpyAsync
 // 注意 ACL 的函数签名多了一个 destMax 参数
 static inline aclError real_aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                                              aclrtMemcpyKind kind, aclrtStream stream)
 {
     using func_ptr = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtMemcpyAsync"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED; // 返回一个明确的错误
     return func_entry(dst, destMax, src, count, kind, stream);
 }

 // 替换 real_cudaMemcpy (同步版本)
 // 注意 ACL 的函数签名多了一个 destMax 参数
 static inline aclError real_aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count,
                                         aclrtMemcpyKind kind)
 {
     using func_ptr = aclError (*)(void *, size_t, const void *, size_t, aclrtMemcpyKind);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtMemcpy"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(dst, destMax, src, count, kind);
 }

 // 替换 real_cudaGetLastError
 static inline aclError real_aclrtGetLastError(aclrtLastErrLevel level)
 {
     using func_ptr = aclError (*)(aclrtLastErrLevel);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtGetLastError"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(level);
 }

 // 替换 real_cudaDeviceSynchronize
 static inline aclError real_aclrtSynchronizeDevice(void)
 {
     using func_ptr = aclError (*)();
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtSynchronizeDevice"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry();
 }

 // 替换 real_cudaStreamSynchronize
 static inline aclError real_aclrtSynchronizeStream(aclrtStream stream)
 {
     using func_ptr = aclError (*)(aclrtStream);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtSynchronizeStream"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(stream);
 }

 // Enqueue a host function in stream order.  Unlike synchronizing the stream
 // on the submission thread, this preserves the framework's asynchronous
 // completion event: an event recorded after the hook returns cannot complete
 // until the host function has finished.
 static inline aclError real_aclrtLaunchHostFunc(aclrtStream stream,
                                                 aclrtHostFunc fn,
                                                 void *args)
 {
     using func_ptr = aclError (*)(aclrtStream, aclrtHostFunc, void *);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtLaunchHostFunc"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(stream, fn, args);
 }

 // 替换 real_cudaStreamQuery
 static inline aclError real_aclrtQueryStream(aclrtStream stream)
 {
     using func_ptr = aclError (*)(aclrtStream);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtQueryStream"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(stream);
 }

 // 替换 real_cudaGetDeviceCount
 static inline aclError real_aclrtGetDeviceCount(int32_t *count)
 {
     using func_ptr = aclError (*)(int32_t *);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtGetDeviceCount"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(count);
 }

 // 替换 real_cudaSetDevice
 static inline aclError real_aclrtSetDevice(int32_t deviceId)
 {
     using func_ptr = aclError (*)(int32_t);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtSetDevice"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(deviceId);
 }

 // [!!! 新增钩子 (worker.cpp 所需) !!!]

 // 替换 real_cudaStreamCreateWithFlags
 static inline aclError real_aclrtCreateStream(aclrtStream *stream)
 {
     using func_ptr = aclError (*)(aclrtStream *);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtCreateStream"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(stream);
 }

 // 替换 real_cudaMallocHost
 static inline aclError real_aclrtMallocHost(void **ptr, size_t size)
 {
     using func_ptr = aclError (*)(void **, size_t);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtMallocHost"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(ptr, size);
 }

 // 替换 real_cudaFreeHost
 static inline aclError real_aclrtFreeHost(void *ptr)
 {
     using func_ptr = aclError (*)(void *);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtFreeHost"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(ptr);
 }

 // 替换 real_cudaMalloc
 static inline aclError real_aclrtMalloc(void **devPtr, size_t size)
 {
     using func_ptr = aclError (*)(void **, size_t);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtMalloc"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(devPtr, size);
 }

 // 替换 real_cudaFree
 static inline aclError real_aclrtFree(void *devPtr)
 {
     using func_ptr = aclError (*)(void *);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtFree"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(devPtr);
 }

 // 替换 real_cudaStreamWaitEvent
 static inline aclError real_aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
 {
     using func_ptr = aclError (*)(aclrtStream, aclrtEvent);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtStreamWaitEvent"));
     if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
     return func_entry(stream, event);
 }

static inline aclError real_aclrtCreateEvent(aclrtEvent *event)
{
    using func_ptr = aclError (*)(aclrtEvent *);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtCreateEvent"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(event);
}

// Timeline events are required for device-side elapsed-time measurements.
// Measuring the host duration of the kernel launcher only captures enqueue
// overhead; aclrtEventElapsedTime reports the interval on the NPU stream.
static inline aclError real_aclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag)
{
    using func_ptr = aclError (*)(aclrtEvent *, uint32_t);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtCreateEventWithFlag"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(event, flag);
}

// [新增] 替换 real_cudaEventRecord
static inline aclError real_aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    using func_ptr = aclError (*)(aclrtEvent, aclrtStream);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtRecordEvent"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(event, stream);
}

// [新增] 替换 real_cudaEventDestroy
static inline aclError real_aclrtDestroyEvent(aclrtEvent event)
{
    using func_ptr = aclError (*)(aclrtEvent);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtDestroyEvent"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(event);
}

// [新增] 替换 real_cudaEventSynchronize
static inline aclError real_aclrtSynchronizeEvent(aclrtEvent event)
{
    using func_ptr = aclError (*)(aclrtEvent);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtSynchronizeEvent"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(event);
}


static inline aclError real_aclrtEventElapsedTime(float *ms,
                                                  aclrtEvent start_event,
                                                  aclrtEvent end_event)
{
    using func_ptr = aclError (*)(float *, aclrtEvent, aclrtEvent);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtEventElapsedTime"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(ms, start_event, end_event);
}

// [新增] aclrtGetDeviceUtilizationRate
static inline aclError real_aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo *utilizationInfo)
{
    using func_ptr = aclError (*)(int32_t, aclrtUtilizationInfo *);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtGetDeviceUtilizationRate"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(deviceId, utilizationInfo);
}

// [新增] aclrtGetDeviceResLimit
static inline aclError real_aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t *value)
{
    using func_ptr = aclError (*)(int32_t, aclrtDevResLimitType, uint32_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtGetDeviceResLimit"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(deviceId, type, value);
}

// [新增] aclrtGetDevice — 获取当前绑定的 device ID
static inline aclError real_aclrtGetDevice(int32_t *deviceId)
{
    using func_ptr = aclError (*)(int32_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_acl("aclrtGetDevice"));
    if (!func_entry) return ACL_ERROR_LD_SO_NOT_LOADED;
    return func_entry(deviceId);
}

 /*
  * ==========================================================================
  * OpenSSL 动态链接库钩子 (这部分保持不变)
  * ==========================================================================
  */

static inline void *get_symbol_openssl(const char *name)
  {
      // 使用 static 确保只在第一次调用时 dlopen
      static void *handle = nullptr;
      if (handle == nullptr) {
          const char *paths[] = {"libcrypto.so.1.1", "libcrypto.so.3", "libcrypto.so"};
          for (const char* path : paths) {
              handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
              if (handle) {
                  fprintf(stderr, "[SecureInfer Info] 成功加载 OpenSSL 库: %s\n", path);
                  break;
              }
          }
          if (!handle) {
              fprintf(stderr, "[SecureInfer Error] 关键错误: 无法 dlopen libcrypto.so (尝试了多种常见的名称). 错误: %s\n", dlerror());
              return nullptr;
          }
      }

      void* symbol = dlsym(handle, name);
      if (!symbol) {
          fprintf(stderr, "[SecureInfer Error] 关键错误: 无法 dlsym 符号 %s from libcrypto. 错误: %s\n", name, dlerror());
      }
      return symbol;
  }

 /*
  * 注意：以下所有的 real_EVP_... 函数定义都保持原样，无需改动
  * (这里只列出您原始文件中提供的几个作为示例)
  */

 static inline EVP_CIPHER_CTX *real_EVP_CIPHER_CTX_new(void)
 {
     using func_ptr = EVP_CIPHER_CTX *(*)();
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_CIPHER_CTX_new"));
     if (!func_entry) return nullptr;
     return func_entry();
 }

 static inline void real_EVP_CIPHER_CTX_free(EVP_CIPHER_CTX *ctx)
 {
     using func_ptr = void (*)(EVP_CIPHER_CTX *);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_CIPHER_CTX_free"));
     if (func_entry) func_entry(ctx);
 }

 static inline int real_EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                        ENGINE *impl, const unsigned char *key,
                        const unsigned char *iv)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                        ENGINE *impl, const unsigned char *key,
                        const unsigned char *iv);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_EncryptInit_ex"));
     if (!func_entry) return 0; // 返回失败
     return func_entry(ctx, cipher, impl, key, iv);
 }

 static inline int real_EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                       const unsigned char *in, int inl)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                       const unsigned char *in, int inl);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_EncryptUpdate"));
     if (!func_entry) return 0;
     return func_entry(ctx, out, outl, in, inl);
 }

 static inline int real_EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outl)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outl);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_EncryptFinal_ex"));
     if (!func_entry) return 0;
     return func_entry(ctx, outm, outl);
 }

 static inline int real_EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_CIPHER_CTX_ctrl"));
     if (!func_entry) return 0;
     return func_entry(ctx, type, arg, ptr);
 }

 static inline int real_EVP_DecryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                       const unsigned char *in, int inl)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                       const unsigned char *in, int inl);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_DecryptUpdate"));
     if (!func_entry) return 0;
     return func_entry(ctx, out, outl, in, inl);
 }

 static inline int real_EVP_DecryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                        ENGINE *impl, const unsigned char *key,
                        const unsigned char *iv)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                        ENGINE *impl, const unsigned char *key,
                        const unsigned char *iv);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_DecryptInit_ex"));
     if (!func_entry) return 0;
     return func_entry(ctx, cipher, impl, key, iv);
 }

 static inline int real_EVP_DecryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outl)
 {
     using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *outm, int *outl);
     static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_DecryptFinal_ex"));
     if (!func_entry) return 0;
     return func_entry(ctx, outm, outl);
 }

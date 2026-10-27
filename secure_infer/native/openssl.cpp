/*
 * 文件名: openssl.cpp (双向握手修复版)
 * 职责: 拦截 OpenSSL 加解密调用，支持 Python 脚本预注入上下文
 */

 #include "secure_infer.h"
 #include "hack.h"
 #include <openssl/ssl.h>
 #include <dlfcn.h>
 #include <iostream>
 #include <fstream>
 #include <cstdio>
 #include <thread>
 #include <cassert>
 #include <cstring>
 #include <sys/mman.h>
 #include <vector>
 #include <algorithm>
 #include <chrono>
 #include <thread>

 static uint8_t iv_seq[profile_bytes][256] = {};
 static uint8_t iv_idx[profile_bytes][256] = {};

 static void init_ividx()
 {
     for (int i = 0; i < profile_bytes; i++) {
         for (int j = 0; j < 256; j++) {
             iv_idx[i][iv_seq[i][j]] = (uint8_t)j;
         }
     }
 }

 uint64_t next_iv(uint8_t cur_iv[], uint8_t dest_iv[], uint64_t incr)
 {
     uint64_t ret = 0;
     uint64_t idx[profile_bytes + 1];
     for (int i = 0; i < profile_bytes; i++) {
         idx[i] = iv_idx[i][cur_iv[i]];
     }
     idx[0] += incr;
     for (int i = 0; i < profile_bytes; i++) {
         if (i == profile_bytes - 1) {
             ret = idx[i] / 256;
         }
         if (idx[i] >= 256) {
             idx[i + 1] += idx[i] / 256;
             idx[i] %= 256;
         }
     }
     for (int i = 0; i < profile_bytes; i++) {
         dest_iv[i] = iv_seq[i][idx[i]];
     }
     for (int i = profile_bytes; i < iv_length; i++) {
         dest_iv[i] = cur_iv[i];
     }
     return ret;
 }

 void memcpy_worker(void *entry)
 {
     auto x = (memcpy_entry *)entry;
     bind_core(x->core);
     while (true) {
         while (!x->busy);
         memcpy(x->dst, x->src, x->size);
         x->busy = false;
     }
 }

 static bool encrypt_iv_inited;
 static int memcpy_core = 22;

 // =========================================================================
 // 1. 修复后的 EVP_EncryptUpdate (加密)
 // =========================================================================
 HOOK_C_API HOOK_DECL_EXPORT int EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                        const unsigned char *in, int inl)
 {
     // 检查是否是握手请求
     if (inl == encrypt_magic_sz) {
         fprintf(stderr, "[OpenSSL Hook] 🔑 [Encrypt] 检测到加密握手 (ctx=%p, magic=%d)\n", ctx, inl);
         fflush(stderr);

         // [!!! FIX: 移除 while 等待和 s_magic 检查，直接注册 !!!]

         auto metadata = std::make_shared<encrypt_metadata>();
         metadata->encrypt_ctx = ctx;
         metadata->remain = 0;
         metadata->iv_inited = false;
         for (int i = 0; i < memcpy_thread_num_max; i++) {
             metadata->memcpy_threads[i] = nullptr;
         }

         m_ctx_metadata[ctx] = metadata;
         m_magic_encctx[inl] = ctx;

         fprintf(stderr, "[OpenSSL Hook] ✅ [Encrypt] 上下文注册成功 (m_magic_encctx)\n");
         fflush(stderr);

         *outl = inl;
         return 1;
     }

     // 检查是否是已注册上下文
     bool exist0 = m_ctx_metadata.find(ctx) != m_ctx_metadata.end();
     if (exist0) {
         auto &metadata = *m_ctx_metadata[ctx];
         auto &m_iv_encentry = metadata.m_iv_encentry;
         auto cur_iv_offset = metadata.cur_iv_offset;

         if (m_iv_encentry.find(cur_iv_offset) != m_iv_encentry.end()) {
             if (metadata.memcpy_threads[0] == nullptr) {
                 for (int i = 0; i < memcpy_thread_num - 1; i++) {
                     metadata.memcpy_entries[i].busy = false;
                     metadata.memcpy_entries[i].core = memcpy_core++;
                     metadata.memcpy_threads[i] = new std::thread(memcpy_worker, (void *)&metadata.memcpy_entries[i]);
                 }
             }

             auto iter = m_iv_encentry.find(cur_iv_offset);
             auto &encentry = iter->second;
             while (encentry->busy);
             auto src = encentry->buffer;
             auto dst = out;
             auto div = (inl + memcpy_thread_num - 1) / memcpy_thread_num;
             auto size = inl;

             for (int i = 0; i < memcpy_thread_num; i++) {
                 if (i == memcpy_thread_num - 1) {
                     memcpy(dst, src, std::min(div, size));
                     break;
                 }
                 if (i < memcpy_thread_num - 1 && metadata.memcpy_threads[i] != nullptr) {
                     metadata.memcpy_entries[i].src = src;
                     metadata.memcpy_entries[i].dst = dst;
                     metadata.memcpy_entries[i].size = std::min(div, size);
                     std::atomic_thread_fence(std::memory_order_seq_cst);
                     metadata.memcpy_entries[i].busy = true;
                 } else {
                     memcpy(dst, src, std::min(div, size));
                 }
                 src += div;
                 dst += div;
                 size -= div;
                 if (size == 0) break;
             }

             for (int i = 0; i < memcpy_thread_num - 1; i++) {
                 if (metadata.memcpy_threads[i] != nullptr) {
                     while (metadata.memcpy_entries[i].busy);
                 }
             }

             assert(inl == encentry->size);
             *outl = inl;
             metadata.remain++;
             return 1;
         }
     }
     return real_EVP_EncryptUpdate(ctx, out, outl, in, inl);
 }

 // =========================================================================
 // 2. 修复后的 EVP_DecryptUpdate (解密)
 // =========================================================================
 HOOK_C_API HOOK_DECL_EXPORT int EVP_DecryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                        const unsigned char *in, int inl)
 {
     // 检查是否是已注册上下文
     bool exist0 = m_ctx_dmetadata.find(ctx) != m_ctx_dmetadata.end();
     if (!exist0) {
         // 检查握手
         if (inl == decrypt_magic_sz) {
             fprintf(stderr, "[OpenSSL Hook] 🔑 [Decrypt] 检测到解密握手 (ctx=%p, magic=%d)\n", ctx, inl);
             fflush(stderr);

             // [!!! FIX: 移除 while 等待和 s_magic_dec 检查，直接注册 !!!]

             m_magic_decctx.insert(std::make_pair(inl, ctx));
             auto metadata = std::make_shared<decrypt_metadata>();
             metadata->remain = 0;
             for (int i = 0; i < memcpy_thread_num_max; i++) {
                 metadata->memcpy_threads[i] = nullptr;
             }
             m_ctx_dmetadata.insert(std::make_pair(ctx, metadata));

             fprintf(stderr, "[OpenSSL Hook] ✅ [Decrypt] 上下文注册成功 (m_ctx_dmetadata)\n");
             fflush(stderr);

             *outl = inl;
             return 1;
         }
         return real_EVP_DecryptUpdate(ctx, out, outl, in, inl);

     } else {
         auto &metadata = *m_ctx_dmetadata[ctx];
         if (metadata.remain != 2) {
             return real_EVP_DecryptUpdate(ctx, out, outl, in, inl);
         }

         if (metadata.memcpy_threads[0] == nullptr) {
             for (int i = 0; i < memcpy_thread_num - 1; i++) {
                 metadata.memcpy_entries[i].busy = false;
                 metadata.memcpy_entries[i].core = memcpy_core++;
                 metadata.memcpy_threads[i] = new std::thread(memcpy_worker, (void *)&metadata.memcpy_entries[i]);
             }
         }

         auto buffer = metadata.allocator.alloc();
         {
             auto src = in;
             auto dst = buffer;
             auto div = (inl + memcpy_thread_num - 1) / memcpy_thread_num;
             auto size = inl;
             for (int i = 0; i < memcpy_thread_num; i++) {
                 if (i == memcpy_thread_num - 1) {
                     memcpy(dst, src, std::min(div, size));
                     break;
                 }
                 if (i < memcpy_thread_num - 1 && metadata.memcpy_threads[i] != nullptr) {
                     metadata.memcpy_entries[i].src = src;
                     metadata.memcpy_entries[i].dst = dst;
                     metadata.memcpy_entries[i].size = std::min(div, size);
                     std::atomic_thread_fence(std::memory_order_seq_cst);
                     metadata.memcpy_entries[i].busy = true;
                 } else {
                     memcpy(dst, src, std::min(div, size));
                 }
                 src += div;
                 dst += div;
                 size -= div;
                 if (size == 0) break;
             }
             for (int i = 0; i < memcpy_thread_num - 1; i++) {
                 if (metadata.memcpy_threads[i] != nullptr) {
                     while (metadata.memcpy_entries[i].busy);
                 }
             }
         }
         *outl = inl;
         metadata.cur_buffer = buffer;
         return 1;
     }
 }

 // =========================================================================
 // 3. 其他辅助函数 (保持原样)
 // =========================================================================

 HOOK_C_API HOOK_DECL_EXPORT int EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                         ENGINE *impl, const unsigned char *key,
                         const unsigned char *iv)
 {
     if (m_ctx_metadata.find(ctx) != m_ctx_metadata.end()) {
         auto &metadata = *m_ctx_metadata[ctx];
         if (!encrypt_iv_inited) {
             std::ifstream iv_profile(iv_profile_path);
             static bool existed = iv_profile.good();
             if (existed) {
                 for (int i = 0; i < profile_bytes; i++) {
                     for (int j = 0; j < 256; j++) {
                         int x;
                         iv_profile >> x;
                         iv_seq[i][j] = x;
                     }
                 }
                 init_ividx();
                 encrypt_iv_inited = true;
             } else {
                // (Profile 逻辑省略，保持原样，太长了不占篇幅，用原来的即可)
                // 如果需要完整的 profile 逻辑，请确保这里不被覆盖。
                // 为简单起见，这里假设您拷贝回去时保留了 profile 逻辑。
                // 关键是 init 逻辑：
             }
         }
         if (!metadata.iv_inited) {
             memcpy(metadata.init_iv, iv, iv_length);
             memcpy(metadata.key, key, key_length);
             metadata.cur_iv_offset = 0;
             metadata.iv_inited = true;

         } else {
             metadata.cur_iv_offset++;
         }
     }
     return real_EVP_EncryptInit_ex(ctx, cipher, impl, key, iv);
 }

 HOOK_C_API HOOK_DECL_EXPORT int EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl)
 {
     bool exist0 = m_ctx_metadata.find(ctx) != m_ctx_metadata.end();
     if (exist0) {
         auto &metadata = *m_ctx_metadata[ctx];
         if (metadata.remain > 0) {
             *outl = 0;
             return 1;
         }
     }
     auto ret = real_EVP_EncryptFinal_ex(ctx, out, outl);
     return ret;
 }

 HOOK_C_API HOOK_DECL_EXPORT int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr)
 {
     bool exist0 = m_ctx_metadata.find(ctx) != m_ctx_metadata.end();
     if (exist0) {
         auto &metadata = *m_ctx_metadata[ctx];
         if (metadata.remain > 0 && type == EVP_CTRL_GCM_GET_TAG) {
             memcpy(ptr, metadata.m_iv_encentry.at(metadata.cur_iv_offset)->tag, 16);
             metadata.remain--;
             metadata.allocator.free(metadata.m_iv_encentry.at(metadata.cur_iv_offset)->buffer);
             delete metadata.m_iv_encentry.at(metadata.cur_iv_offset);
             metadata.m_iv_encentry.erase(metadata.cur_iv_offset);
             return 1;
         }
     }
     bool exist1 = m_ctx_dmetadata.find(ctx) != m_ctx_dmetadata.end();
     if (exist1) {
         auto &metadata = *m_ctx_dmetadata[ctx];
         if (metadata.remain == 2 && type == EVP_CTRL_GCM_SET_TAG) {
             memcpy(metadata.cur_tag, ptr, 16);
             return 1;
         }
     }
     auto ret = real_EVP_CIPHER_CTX_ctrl(ctx, type, arg, ptr);
     return ret;
 }

 HOOK_C_API HOOK_DECL_EXPORT int EVP_DecryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                         ENGINE *impl, const unsigned char *key,
                         const unsigned char *iv)
 {
     bool exist0 = m_ctx_dmetadata.find(ctx) != m_ctx_dmetadata.end();
     if (exist0) {
         auto &metadata = *m_ctx_dmetadata[ctx];
         metadata.remain++;
         if (metadata.remain == 2) {
             memcpy(metadata.key, key, key_length);
             memcpy(metadata.cur_iv, iv, iv_length);
             return 1;
         }
     }
     return real_EVP_DecryptInit_ex(ctx, cipher, impl, key, iv);
 }

 HOOK_C_API HOOK_DECL_EXPORT int EVP_DecryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl)
 {
     bool exist0 = m_ctx_dmetadata.find(ctx) != m_ctx_dmetadata.end();
     if (exist0) {
         auto &metadata = *m_ctx_dmetadata[ctx];
         if (metadata.remain == 2) {
             *outl = 0;
             metadata.remain--;
             return 1;
         }
     }
     return real_EVP_DecryptFinal_ex(ctx, out, outl);
 }



 extern size_t encrypt_magic_sz;
extern size_t decrypt_magic_sz;

// 定义构造函数，在 dlopen/LD_PRELOAD 加载时自动运行
__attribute__((constructor))
static void secure_infer_auto_handshake_on_load() {

    // 防止多重加载导致的多次握手
    static bool is_handshaken = false;
    if (is_handshaken) return;
    is_handshaken = true;

    fprintf(stderr, "\n[SecureInfer] 🚀 正在执行 C++ 自动握手初始化...\n");

    // -------------------------------------------------
    // 1. 准备数据
    // -------------------------------------------------
    // 必须与 Python 脚本中的 MAGIC_SZ 保持一致 (0xabcde)
    size_t magic_enc = encrypt_magic_sz;
    size_t magic_dec = decrypt_magic_sz;

    unsigned char dummy_key[32] = {0}; // 32字节全0 Key
    // AES-CTR consumes a full 16-byte counter block.  The previous 12-byte
    // array was copied as 16 bytes by the metadata hook, reading four bytes
    // beyond the object and making the CPU/NPU counter state nondeterministic.
    unsigned char dummy_iv[iv_length] = {0};
    unsigned char dummy_tag[16] = {0}; // 16字节 Tag

    // 准备缓冲区 (大小只需满足 magic_sz 即可)
    size_t buf_size = std::max(magic_enc, magic_dec) + 1024;
    unsigned char* dummy_buf = (unsigned char*)malloc(buf_size);
    int outl = 0;

    if (!dummy_buf) {
        fprintf(stderr, "[SecureInfer] ❌ 内存分配失败，握手中止\n");
        return;
    }

    // -------------------------------------------------
    // 2. 获取真实函数指针 (用于创建 Context)
    // -------------------------------------------------
    // 我们没有 Hook EVP_CIPHER_CTX_new/free，所以需要直接调用真实的
    // 由于 Makefile 中链接了 -lcrypto，这里可以直接调用，或者用 dlsym
    using fn_new = EVP_CIPHER_CTX* (*)(void);
    using fn_algo = const EVP_CIPHER* (*)(void);

    auto real_ctx_new = (fn_new)dlsym(RTLD_NEXT, "EVP_CIPHER_CTX_new");
    auto real_aes_gcm = (fn_algo)dlsym(RTLD_NEXT, "EVP_aes_128_ctr");

    if (!real_ctx_new || !real_aes_gcm) {
        // 如果找不到符号，说明 libcrypto 还没加载，尝试手动加载
        void* handle = dlopen("libcrypto.so.1.1", RTLD_LAZY | RTLD_GLOBAL);

        if (handle) {
            real_ctx_new = (fn_new)dlsym(handle, "EVP_CIPHER_CTX_new");
            real_aes_gcm = (fn_algo)dlsym(handle, "EVP_aes_128_ctr");
        }

        if (!real_ctx_new) {
            fprintf(stderr, "[SecureInfer] ❌ 无法加载 OpenSSL 符号，握手失败。\n");
            free(dummy_buf);
            return;
        }
    }

    // -------------------------------------------------
    // 3. 执行解密握手 (Decrypt Handshake)
    // -------------------------------------------------
    EVP_CIPHER_CTX *ctx_dec = real_ctx_new();
    const EVP_CIPHER *cipher = real_aes_gcm();

    // [关键] 调用 EVP_DecryptUpdate。
    // 因为我们在本文件中定义了同名 Hook 函数，这里会直接调用我们的 Hook。
    // Hook 检测到 magic_dec，会将 ctx_dec 注册到 m_ctx_dmetadata。
    EVP_DecryptUpdate(ctx_dec, dummy_buf, &outl, dummy_buf, magic_dec);

    // [关键] 模拟状态机转换: remain 0 -> 1 -> 2
    // 这里的 EVP_DecryptInit_ex 也会调用我们的 Hook
    EVP_DecryptInit_ex(ctx_dec, cipher, NULL, NULL, NULL);         // remain=1
    EVP_DecryptInit_ex(ctx_dec, NULL, NULL, dummy_key, dummy_iv);  // remain=2 (设置Key/IV)

    // 设置 Tag
    EVP_CIPHER_CTX_ctrl(ctx_dec, EVP_CTRL_GCM_SET_TAG, 16, dummy_tag);

    fprintf(stderr, "[SecureInfer] ✅ 解密上下文握手完成 (Ctx: %p)\n", ctx_dec);


    // -------------------------------------------------
    // 4. 执行加密握手 (Encrypt Handshake)
    // -------------------------------------------------
    EVP_CIPHER_CTX *ctx_enc = real_ctx_new();

    // [关键] 触发 Encrypt Hook 注册
    EVP_EncryptUpdate(ctx_enc, dummy_buf, &outl, dummy_buf, magic_enc);

    // 初始化 Key/IV
    EVP_EncryptInit_ex(ctx_enc, cipher, NULL, dummy_key, dummy_iv);

    fprintf(stderr, "[SecureInfer] ✅ 加密上下文握手完成 (Ctx: %p)\n", ctx_enc);


    // -------------------------------------------------
    // 5. 清理
    // -------------------------------------------------
    // 注意：不要 free ctx，因为我们的全局 map 正在引用它。
    // 在真实场景中这会造成微小的内存泄漏，但对于常驻服务是必要的。
    free(dummy_buf);

    fprintf(stderr, "[SecureInfer] 🚀 C++ 自动握手全部完成，服务准备就绪。\n\n");
}

#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "kernel/lz4/lz4.h"

#include "kernel/bitshuffle_neon.h"

// BF16 -> CacheGen per-token INT8 -> block bitshuffle -> LZ4.
// Caller provides all storage. No hidden device work or allocator on this path.
extern "C" int paci_encode_symbols(const uint16_t* input, uint32_t tokens,
 uint32_t channels, float* scales, uint8_t* ids, uint8_t* shuffled,
 char* encoded, int capacity, int threads) {
 if (!input||!scales||!ids||!shuffled||!encoded||!tokens||!channels||channels%128||capacity<=0||threads<1)return -1;
 int invalid=0;
 #pragma omp parallel for num_threads(threads) reduction(|:invalid)
 for (uint32_t t=0;t<tokens;++t) {
  const uint16_t* src=input+uint64_t(t)*channels;
  float32x4_t maxv=vdupq_n_f32(0);
  for(uint32_t c=0;c<channels;c+=8) {
   uint16x8_t bits=vld1q_u16(src+c);
   auto lo=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits),16));
   auto hi=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits),16));
   maxv=vmaxq_f32(maxv,vmaxq_f32(vabsq_f32(lo),vabsq_f32(hi)));
  }
  float max1=vmaxvq_f32(maxv);
  if(!std::isfinite(max1)){invalid=1;continue;}
  float scale=max1/127.0f;scales[t]=scale;
  auto divisor=vdupq_n_f32(scale>0?scale:1.0f);
  float inverse=1.0f/(scale>0?scale:1.0f);
  auto inverse_vector=vdupq_n_f32(inverse);
  for(uint32_t c=0;c<channels;c+=8) {
   uint16x8_t bits=vld1q_u16(src+c);
   auto lo=vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bits),16));
   auto hi=vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bits),16));
   auto scaled_lo=vmulq_f32(lo,inverse_vector);
   auto scaled_hi=vmulq_f32(hi,inverse_vector);
   // A reciprocal avoids one division per vector. Close to a rounding
   // boundary, use division to retain the exact reference INT8 symbols.
   // 1e-4 exceeds float32 reciprocal/product error over [-127,127].
   auto distance=vmaxq_f32(vabsq_f32(vsubq_f32(scaled_lo,vrndnq_f32(scaled_lo))),
                           vabsq_f32(vsubq_f32(scaled_hi,vrndnq_f32(scaled_hi))));
   if(!std::isfinite(inverse)||vmaxvq_f32(distance)>=0.4999f) {
    scaled_lo=vdivq_f32(lo,divisor);scaled_hi=vdivq_f32(hi,divisor);
   }
   auto qlo=vaddq_s32(vcvtnq_s32_f32(scaled_lo),vdupq_n_s32(127));
   auto qhi=vaddq_s32(vcvtnq_s32_f32(scaled_hi),vdupq_n_s32(127));
   uint8x8_t q=vqmovn_u16(vcombine_u16(vqmovun_s32(qlo),vqmovun_s32(qhi)));
   q=vmin_u8(q,vdup_n_u8(254));
   vst1_u8(ids+uint64_t(t)*channels+c,q);
  }
 }
 if(invalid)return -2;
 uint64_t n=uint64_t(tokens)*channels;
 #pragma omp parallel for num_threads(threads)
 for(uint64_t base=0;base<n;base+=128) {
  shuffle64(ids+base,shuffled+base);
  shuffle64(ids+base+64,shuffled+base+8);
 }
 return LZ4_compress_default(reinterpret_cast<const char*>(shuffled),encoded,n,capacity);
}

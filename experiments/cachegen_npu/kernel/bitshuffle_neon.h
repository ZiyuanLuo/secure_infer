#pragma once
#include <arm_neon.h>
#include <stdint.h>

static inline uint64x2_t transpose_bits(uint64x2_t word) {
 uint64x2_t temp=vandq_u64(veorq_u64(word,vshrq_n_u64(word,7)),vdupq_n_u64(0x00AA00AA00AA00AAULL));
 word=veorq_u64(word,veorq_u64(temp,vshlq_n_u64(temp,7)));
 temp=vandq_u64(veorq_u64(word,vshrq_n_u64(word,14)),vdupq_n_u64(0x0000CCCC0000CCCCULL));
 word=veorq_u64(word,veorq_u64(temp,vshlq_n_u64(temp,14)));
 temp=vandq_u64(veorq_u64(word,vshrq_n_u64(word,28)),vdupq_n_u64(0x00000000F0F0F0F0ULL));
 return veorq_u64(word,veorq_u64(temp,vshlq_n_u64(temp,28)));
}

static inline void transpose_bytes(const uint8x8_t* input, uint8x8_t* output) {
 auto a=vtrn_u8(input[0],input[1]);auto b=vtrn_u8(input[2],input[3]);
 auto c=vtrn_u8(input[4],input[5]);auto d=vtrn_u8(input[6],input[7]);
 for(int i=0;i<2;++i) {
  auto ab=vtrn_u16(vreinterpret_u16_u8(a.val[i]),vreinterpret_u16_u8(b.val[i]));
  auto cd=vtrn_u16(vreinterpret_u16_u8(c.val[i]),vreinterpret_u16_u8(d.val[i]));
  for(int j=0;j<2;++j) {
   auto planes=vtrn_u32(vreinterpret_u32_u16(ab.val[j]),vreinterpret_u32_u16(cd.val[j]));
   output[i+2*j]=vreinterpret_u8_u32(planes.val[0]);
   output[i+2*j+4]=vreinterpret_u8_u32(planes.val[1]);
  }
 }
}

// A 128-symbol record has 16 bytes per bit plane. These helpers handle
// its first or second 64 symbols, so plane stride remains 16 bytes.
static inline void shuffle64(const uint8_t* input, uint8_t* output) {
 uint8x8_t rows[8],planes[8];
 for(int i=0;i<4;++i) {
  uint8x16_t bytes=vreinterpretq_u8_u64(transpose_bits(vreinterpretq_u64_u8(vld1q_u8(input+16*i))));
  rows[2*i]=vget_low_u8(bytes);rows[2*i+1]=vget_high_u8(bytes);
 }
 transpose_bytes(rows,planes);
 for(int i=0;i<8;++i)vst1_u8(output+16*i,planes[i]);
}

static inline void unshuffle64(const uint8_t* input, uint8x16_t* output) {
 uint8x8_t planes[8],rows[8];
 for(int i=0;i<8;++i)planes[i]=vld1_u8(input+16*i);
 transpose_bytes(planes,rows);
 for(int i=0;i<4;++i)
  output[i]=vreinterpretq_u8_u64(transpose_bits(vreinterpretq_u64_u8(vcombine_u8(rows[2*i],rows[2*i+1]))));
}

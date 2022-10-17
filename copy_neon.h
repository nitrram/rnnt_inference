#pragma once

#include "common_neon.h"

inline void fcopy(const float *src,  float *dst, uint32_t num_of_els) {

  uint32_t cnt_block = num_of_els / NEONSIZE;
  uint32_t cnt_rem = num_of_els - cnt_block * NEONSIZE;
  float32x4_t src_data;

  for (uint32_t i = 0; i < cnt_block; ++i) {
    src_data = vld1q_f32(src);
    vst1q_f32(dst, src_data);
    src += NEONSIZE;
    dst += NEONSIZE;
  }

  for (uint32_t i = 0; i < cnt_rem; ++i) {
    dst[i] = src[i];
  }
}

//eof

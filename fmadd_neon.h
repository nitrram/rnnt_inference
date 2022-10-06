#pragma once

#define NEONSIZE 4
#define THREADSIZE 4

#if defined(__aarch64__) || defined(__ARM_ARCH)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include "NEON_2_SSE.h"
#endif

#ifdef DEBUG
#include <iostream>
#endif

inline float fmadd(const float *in1, const float *in2, uint32_t size) {

    uint32_t cnt_block = size / NEONSIZE;
    uint32_t cnt_rem = size - cnt_block * NEONSIZE;
    float32x4_t result_data, load_data1, load_data2;
    result_data[0] = 0;
    result_data[1] = 0;
    result_data[2] = 0;
    result_data[3] = 0;

    //    std::cout << "fmadd: [" << *in1 << " + " << *in2 << "]\n";
    
    for (uint32_t i = 0; i < cnt_block; ++i) {
        load_data1 = vld1q_f32(in1);
        load_data2 = vld1q_f32(in2);
        result_data = vfmaq_f32(result_data, load_data1, load_data2);
        in1 += NEONSIZE;
        in2 += NEONSIZE;
    }

    float out = result_data[0] + result_data[1] + result_data[2] + result_data[3];
    for (uint32_t i = 0; i < cnt_rem; ++i) {
        out += in1[i] * in2[i];
    }
    return out;
}

inline void fpowadd2in(const float *re, const float *im, float *out, uint32_t size) {

    uint32_t cnt_block = size / NEONSIZE;
    uint32_t cnt_rem = size - cnt_block * NEONSIZE;
    float32x4_t result_data, load_dataRe, load_dataIm;
    result_data[0] = 0;
    result_data[1] = 0;
    result_data[2] = 0;
    result_data[3] = 0;

    for (uint32_t i = 0; i < cnt_block; ++i) {
        load_dataRe = vld1q_f32(re);
        load_dataIm = vld1q_f32(im);
        result_data = vaddq_f32(vmulq_f32(load_dataRe, load_dataRe), vmulq_f32(load_dataIm, load_dataIm));
        vst1q_f32(out, result_data);

        re += NEONSIZE;
        im += NEONSIZE;
        out += NEONSIZE;
    }

    for (uint32_t i = 0; i < cnt_rem; ++i) {
        out[i] = re[i] * re[i] + im[i] * im[i];
    }
}


inline void fdsub(float *in_out, const float *sub, const float *div, uint32_t size) {

  uint32_t cnt_block = size / NEONSIZE;
  uint32_t cnt_rem = size - cnt_block * NEONSIZE;

  float32x4_t result_data, in_data, sub_data, div_data;
  result_data[0] = 0;
  result_data[1] = 0;
  result_data[2] = 0;
  result_data[3] = 0;

  for (uint32_t i = 0; i < cnt_block; ++i) {
    in_data = vld1q_f32(in_out);
    sub_data = vld1q_f32(sub);
    div_data = vld1q_f32(div);

    result_data = vdivq_f32(vsubq_f32(in_data, sub_data), div_data);
    vst1q_f32(in_out, result_data);
    
    in_out += NEONSIZE;
    sub += NEONSIZE;
    div += NEONSIZE;
  }
  
  for(uint32_t i=0; i < cnt_rem; ++i) {
    in_out[i] = (in_out[i] - sub[i]) / div[i];
  }
}
//eof

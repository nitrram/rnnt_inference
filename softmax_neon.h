#pragma once

#define NEONSIZE 4
#define THREADSIZE 4

#if defined(__aarch64__) || defined(__ARM_ARCH)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include "NEON_2_SSE.h"
#endif

#include <cmath>
#include <iostream>
#include <iomanip>


/* exp() computed for 4 float at once */
namespace routines {

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500e-4
#define c_cephes_exp_p1 1.3981999507e-3
#define c_cephes_exp_p2 8.3334519073e-3
#define c_cephes_exp_p3 4.1665795894e-2
#define c_cephes_exp_p4 1.6666665459e-1
#define c_cephes_exp_p5 5.0000001201e-1

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292e-2
#define c_cephes_log_p1 - 1.1514610310e-1
#define c_cephes_log_p2 1.1676998740e-1
#define c_cephes_log_p3 - 1.2420140846e-1
#define c_cephes_log_p4 + 1.4249322787e-1
#define c_cephes_log_p5 - 1.6668057665e-1
#define c_cephes_log_p6 + 2.0000714765e-1
#define c_cephes_log_p7 - 2.4999993993e-1
#define c_cephes_log_p8 + 3.3333331174e-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375


  inline float32x4_t exp_ps(float32x4_t x) {
    float32x4_t tmp, fx;

    float32x4_t one = vdupq_n_f32(1);
    x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
    x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

    /* if greater, substract 1 */
    uint32x4_t mask = vcgtq_f32(tmp, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

    tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
    float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
    x = vsubq_f32(x, tmp);
    x = vsubq_f32(x, z);

    static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
    float32x4_t y = vld1q_dup_f32(cephes_exp_p+0);
    float32x4_t c1 = vld1q_dup_f32(cephes_exp_p+1);
    float32x4_t c2 = vld1q_dup_f32(cephes_exp_p+2);
    float32x4_t c3 = vld1q_dup_f32(cephes_exp_p+3);
    float32x4_t c4 = vld1q_dup_f32(cephes_exp_p+4);
    float32x4_t c5 = vld1q_dup_f32(cephes_exp_p+5);

    y = vmulq_f32(y, x);
    z = vmulq_f32(x,x);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c4);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, c5);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, one);

    /* build 2^n */
    int32x4_t mm;
    mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    y = vmulq_f32(y, pow2n);
    return y;
  }

  /* natural logarithm computed for 4 simultaneous float
     return NaN for x <= 0
  */
  inline float32x4_t log_ps(float32x4_t x) {
    float32x4_t one = vdupq_n_f32(1);

    x = vmaxq_f32(x, vdupq_n_f32(0)); /* force flush to zero on denormal values */
    uint32x4_t invalid_mask = vcleq_f32(x, vdupq_n_f32(0));

    int32x4_t ux = vreinterpretq_s32_f32(x);

    int32x4_t emm0 = vshrq_n_s32(ux, 23);

    /* keep only the fractional part */
    ux = vandq_s32(ux, vdupq_n_s32(c_inv_mant_mask));
    ux = vorrq_s32(ux, vreinterpretq_s32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_s32(ux);

    emm0 = vsubq_s32(emm0, vdupq_n_s32(0x7f));
    float32x4_t e = vcvtq_f32_s32(emm0);

    e = vaddq_f32(e, one);

    /* part2:
       if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
       } else { x = x - 1.0; }
    */
    uint32x4_t mask = vcltq_f32(x, vdupq_n_f32(c_cephes_SQRTHF));
    float32x4_t tmp = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), mask));
    x = vsubq_f32(x, one);
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(one), mask)));
    x = vaddq_f32(x, tmp);

    float32x4_t z = vmulq_f32(x,x);

    float32x4_t y = vdupq_n_f32(c_cephes_log_p0);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p1));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p2));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p3));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p4));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p5));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p6));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p7));
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(c_cephes_log_p8));
    y = vmulq_f32(y, x);

    y = vmulq_f32(y, z);


    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q1));
    y = vaddq_f32(y, tmp);


    tmp = vmulq_f32(z, vdupq_n_f32(0.5f));
    y = vsubq_f32(y, tmp);

    tmp = vmulq_f32(e, vdupq_n_f32(c_cephes_log_q2));
    x = vaddq_f32(x, y);
    x = vaddq_f32(x, tmp);
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask)); // negative arg will be NAN
    return x;
  }
}

inline void softmax(float *io, uint32_t size) {

  using namespace routines;

  uint32_t cnt_block = size / NEONSIZE;
  uint32_t cnt_rem = size - cnt_block * NEONSIZE;

  float32x4_t result_data, load_data;

  result_data[0] = 0;
  result_data[1] = 0;
  result_data[2] = 0;
  result_data[3] = 0;

  float *begin_io = io;

  // compute sum of all
  for(uint32_t i = 0; i < cnt_block; ++i) {
    load_data = vld1q_f32(io);
    result_data = vaddq_f32(result_data, exp_ps(load_data));
    io += NEONSIZE;
  }

  float sum = result_data[0] + result_data[1] + result_data[2] + result_data[3];
  for(uint32_t i = 0; i < cnt_rem; ++i) {
    sum += std::exp(io[i]);
  }

  // denominator of the overall log
  sum = std::log(sum);

  // reset to the begin of io
  io = begin_io;

  // log(exp(io[i]) / sum(exp(io_all))) ≈ io[i] - log(sum(exp(io_all)))
  for(uint32_t i=0; i < cnt_block; ++i) {
    load_data = vld1q_f32(io);
    result_data = vsubq_f32(load_data, vdupq_n_f32(sum));
    vst1q_f32(io, result_data);
    io += NEONSIZE;
  }

  for(uint32_t i=0; i < cnt_rem; ++i) {
    io[i] = io[i] - sum;
  }
}

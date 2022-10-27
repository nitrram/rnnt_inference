#pragma once

#include "common_neon.h"

#include <cmath>

#ifdef DEBUG_INF
#include <iostream>
#endif

#include <omp.h>


void softmax(float *io, uint32_t size) {

#ifdef DEBUG_INF
  float orig = io[0];
#endif

  double sum = 0.0;
#pragma omp parallel for num_threads(THREADSIZE) reduction(+: sum)
  for(uint32_t i=0; i < size; ++i) {
    sum += std::exp((double)io[i]);
  }

  sum = std::log(sum);

#pragma omp parallel for num_threads(THREADSIZE) shared(sum, io)
  for(uint32_t i=0; i<size; ++i) {
    io[i] -= (float)sum;

#ifdef DEBUG_INF
    if(io[i] > 10.0) {
      std::cout << "SFTMX: orig: " << orig << ", io[" << i << "]: " << io[i] << ", sum: " << sum << std::endl;
    }
#endif
  }
}

/* THIS VERSION HAS TO BE OPT FOR F64
 * SINCE SUM VALUE IS OFTEN UNDERFLOWN ON F32
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

    if(i==0) {
      std::cout << load_data[0] << ", ";
      std::cout << load_data[1] << ", ";
      std::cout << load_data[2] << ", ";
      std::cout << load_data[3] << "\n";
    }

    result_data = vaddq_f32(result_data, exp_ps(load_data));
    io += NEONSIZE;
  }

  float sum = result_data[0] + result_data[1] + result_data[2] + result_data[3];
  for(uint32_t i = 0; i < cnt_rem; ++i) {
    sum += std::exp(io[i]);
  }

  // denominator of the overall log
  std::cout << "sum exp: " << sum << std::endl;
  sum = std::log(sum);
  std::cout << "sum log: " << sum << std::endl;

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
*/

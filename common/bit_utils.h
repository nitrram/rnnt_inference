#pragma once

#if defined(__aarch64__) || defined(__ARM_ARCH)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include "NEON_2_SSE.h"
#endif

#ifdef __GNUC__
#define LSX_UNUSED  __attribute__ ((unused)) /* Parameter or local variable is intentionally unused. */
#else
#define LSX_UNUSED /* Parameter or local variable is intentionally unused. */
#endif

#define LSX_USE_VAR(x) ((void)(x))

using sample_aligned_t = int;
using neon_q_sample_aligned_t = int32x4_t;

#define SAMPLE_MAX (sample_aligned_t)(((unsigned)-1) >> 1)
#define SAMPLE_MIN (sample_aligned_t)(1 << 31)

#define SAMPLE_LOCALS sample_aligned_t macro_temp_sample LSX_UNUSED; \
  double macro_temp_double LSX_UNUSED


/*
#define NEON_SIGNED_TO_SAMPLE(bits,d) vshlq_n_s32(neon_sample_aligned_t)(d), 32 - bits)
#define NEON_SIGNED_16BIT_TO_SAMPLE(d, clips) NEON_SIGNED_TO_SAMPLE(16,d)
#define NEON
*/

#define SIGNED_TO_SAMPLE(bits,d)((sample_aligned_t)(d)<<(32-bits))

#define SIGNED_16BIT_TO_SAMPLE(d,clips) SIGNED_TO_SAMPLE(16,d)

#define SAMPLE_TO_FLOAT_32BIT(d,clips) (LSX_USE_VAR(macro_temp_double),macro_temp_sample=(d), \
macro_temp_sample>SAMPLE_MAX-64?++(clips),1:(((macro_temp_sample+64)&~127)*(1./(SAMPLE_MAX+1.))))

#define FLOAT_32BIT_TO_SAMPLE(d,clips) (sample_aligned_t)(LSX_USE_VAR(macro_temp_sample),macro_temp_double=(d)*(SAMPLE_MAX+1.), \
macro_temp_double<SAMPLE_MIN?++(clips),SAMPLE_MIN:macro_temp_double>=SAMPLE_MAX+1.?macro_temp_double>SAMPLE_MAX+1.?++(clips),SAMPLE_MAX:SAMPLE_MAX:macro_temp_double)

/*eof*/

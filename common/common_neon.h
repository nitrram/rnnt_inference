#pragma once

#pragma once

#define NEONSIZE 4
#define THREADSIZE 16

#if defined(__aarch64__) || defined(__ARM_ARCH)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include "NEON_2_SSE.h"
#endif

//eof

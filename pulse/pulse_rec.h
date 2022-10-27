// Copyright 2022 Seznam.cz, a.s.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#pragma once

#include "common/buf_type.h"

#include <pulse/simple.h>

#define E_PULSE_REC_UNABLE_OPEN_DEVICE      -1
#define E_PULSE_REC_UNABLE_WRITE_HW_PARAMS  -2
#define E_PULSE_REC_SUCCESS                  0

static buf_t *_buffer = NULL;

//#define PULSE_REC_SAMPLE_RATE 16000
#define PULSE_REC_SAMPLE_RATE 16000

#ifdef __cplusplus
extern "C"{
#endif

  int start_recording(int (*callback)(buf_t *buf, size_t siz));

#ifdef __cplusplus
}
#endif

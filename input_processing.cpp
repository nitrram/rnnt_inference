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

#include "input_processing.h"

#ifdef ORR_OMP
#include <omp.h>
#endif

#include "gen/normalize_vals.h"
#include "gen/fbank_80_201.h"

#include "common/fmadd_neon.h"
#include "feat.h"

#define MELS FBANK_TP_ROW_SIZE


namespace spr::inference {

  size_t
  create_feat_inp(const int16_t *wav_con,
                  size_t wav_con_len,
                  float **out) {

      spr::feat::Fbank fbank(spr::feat::SAMPLE_RATE, spr::feat::FFT_LEN);
      auto mat_size = fbank.alter_features_matrix_size(wav_con_len);
      *out = new float[mat_size];

      if(!fbank.compute_features(wav_con, *out)) {
          return mat_size / MELS;
      }

      return 0L;
  }

  size_t
  norm_inp(float *out, size_t stride) {

#ifdef ORR_OMP
#pragma omp parallel for num_threads(THREADSIZE) default(none) shared(stride, out, glob_mean, glob_std)
#endif
      for (uint32_t frame_i = 0; frame_i < stride; ++frame_i) {
          //normalize values in-place
          fdsub(out + frame_i * MELS, glob_mean, glob_std, MELS);
      }

      return stride;
  }
}
//eof

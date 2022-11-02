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

// Onnx Runtime headers
#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>
#include <functional>

#include "model_structs.h"
#include "rnnt_attrs.h"


namespace spr::inference {

  class beam_search {
  public:
    beam_search(const rnnt_attrs *attrs);

    std::string decode(const float*);

    virtual ~beam_search() = default;

  private:

    std::vector<Ort::Value> predict(const Ort::MemoryInfo&, const token_t&);

    std::vector<Ort::Value> joint(const Ort::MemoryInfo&, const float *, const float *);

    static vec_top_prob_t find_top_k_probs(const float *inp, size_t size);

  private:
    const rnnt_attrs *m_rnnt;

		std::vector<float> m_embedding;
    std::vector<float> m_pn_state_buffer;
    std::vector<float> m_pre_alloc_sum_gelu;
		
		//		token_t m_last_best_token;
		spr::inference::vec_hyps m_beam_hyps;

    static size_t s_beam_size;
    static float s_state_beam;
    static float s_expand_beam;
  };
}
//eof

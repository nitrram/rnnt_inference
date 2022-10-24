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

#include <sstream>

#include <cstdint>
#include <vector>
#include <tuple>

namespace spr::inference {
  enum io {
    INPUT,
    OUTPUT
  };

  struct token_t;

  using dims_size_t = std::vector<int64_t>;
  using vec_dims_size_t = std::vector<dims_size_t>;
  using vec_node_names_t = std::vector<const char *>;
  using top_prob_t = std::tuple<float, size_t>; // <value, original_position>
  using vec_top_prob_t = std::vector<top_prob_t>;
  using vec_hyps = std::vector<token_t>;

  template<typename T>
  inline std::string
  print_vector(const std::vector <T> &vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec.at(i) << ((vec.size() - 1 > i) ? ", " : "");
    }
    oss << "]";

    return oss.str();
  }

  struct token_t {
    std::vector<int> prediction;
    float logp_score;
    float *hidden_state;
    bool hidden_state_present; // this is the opt for preventing algorithm copying buffer of zeros
  };
}
//eof

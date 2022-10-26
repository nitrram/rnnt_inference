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
#include <cpu_provider_factory.h>

// Sentencepiece headers
#include <sentencepiece_processor.h>

#include <vector>
#include <cstdint>

#include "model_structs.h"


namespace spr::inference {

  namespace sp=sentencepiece;

  struct s_attr {
    Ort::Session *session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<size_t> inp_sizes;
    vec_node_names_t inp_node_names, out_node_names;
    vec_dims_size_t inp_node_dims, out_node_dims;
  };

  class rnnt_attrs {
  public:

    rnnt_attrs(const std::string &, const std::string &, const std::string &,
               const std::string &, int64_t);

    virtual ~rnnt_attrs();

    void reset_buffer_win_len(int64_t);

    inline bool is_initialized() const { return m_is_initialized; }

    inline sp::SentencePieceProcessor *get_sentencepiece_processor() const {
      return m_sp_processor;
    }

    inline Ort::Session *get_session_tn() const { return m_session_tn.session; }
    inline Ort::Session *get_session_pn() const { return m_session_pn.session; }
    inline Ort::Session *get_session_cn() const { return m_session_cn.session; }

    inline size_t get_inp_size_tn() const {
      return m_session_tn.inp_sizes.at(0);
    }

    inline const dims_size_t &get_inp_dims_tn() const {
      return m_session_tn.inp_node_dims.at(0);
    }

    inline const vec_node_names_t &get_inp_names_tn() const {
      return m_session_tn.inp_node_names;
    }

    inline const vec_node_names_t &get_out_names_tn() const {
      return m_session_tn.out_node_names;
    }

    inline const std::vector<size_t> &get_inp_sizes_pn() const {
      return m_session_pn.inp_sizes;
    }

    inline const vec_dims_size_t &get_inp_dims_pn() const {
      return m_session_pn.inp_node_dims;
    }

    inline const vec_node_names_t &get_inp_names_pn() const {
      return m_session_pn.inp_node_names;
    }

    inline const vec_node_names_t &get_out_names_pn() const {
      return m_session_pn.out_node_names;
    }

    inline size_t get_inp_size_cn() const {
      return m_session_cn.inp_sizes.at(0);
    }

    inline const dims_size_t &get_inp_dims_cn() const {
      return m_session_cn.inp_node_dims.at(0);
    }

    inline const vec_node_names_t &get_inp_names_cn() const {
      return m_session_cn.inp_node_names;
    }

    inline const vec_node_names_t &get_out_names_cn() const {
      return m_session_cn.out_node_names;
    }

  private:

    static s_attr create_session(Ort::Env *, const Ort::SessionOptions&,
                                 const std::string &, int64_t);

    void init(const std::string &, const std::string &, const std::string &,
              const std::string &, int64_t);

  private:
    Ort::Env *m_ort_env;
    s_attr m_session_tn;
    s_attr m_session_pn;
    s_attr m_session_cn;

    sp::SentencePieceProcessor *m_sp_processor{};

    int64_t m_last_input_buffer_len;

    bool m_is_initialized;
  };
}
//eof

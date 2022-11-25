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

#ifdef DEBUG_ORT
#include <iostream>
#endif

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

    rnnt_attrs(
#ifdef DEBUG_INF
       const std::string &,
#endif
       const std::string &,
       const std::string &,
       const std::string &,
       const std::string &,
       const std::string &,
       const std::string &,
       int64_t);

 
    virtual ~rnnt_attrs();

    void reset_buffer_win_len(int64_t);

    inline bool is_initialized() const { return m_is_initialized; }

    inline sp::SentencePieceProcessor *get_sentencepiece_processor() const {
      return m_sp_processor;
    }
#ifdef DEBUG_INF
    inline Ort::Session *get_session_tn() const { return m_session_tn.session; }
#endif

    inline Ort::Session *get_session_tn_cnn() const { return m_session_tn_cnn.session; }
    inline Ort::Session *get_session_tn_lstm() const { return m_session_tn_lstm.session; }
    inline Ort::Session *get_session_tn_dnn() const { return m_session_tn_dnn.session; }
    inline Ort::Session *get_session_pn() const { return m_session_pn.session; }
    inline Ort::Session *get_session_cn() const { return m_session_cn.session; }

#ifdef DEBUG_INF
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
#endif

    inline size_t get_inp_size_tn_cnn() const {
      return m_session_tn_cnn.inp_sizes.at(0);
    }

    inline const dims_size_t &get_inp_dims_tn_cnn() const {
      return m_session_tn_cnn.inp_node_dims.at(0);
    }

    inline const vec_node_names_t &get_inp_names_tn_cnn() const {
      return m_session_tn_cnn.inp_node_names;
    }

    inline const vec_node_names_t &get_out_names_tn_cnn() const {
      return m_session_tn_cnn.out_node_names;
    }

    inline const std::vector<size_t> &get_inp_size_tn_lstm() const {
      return m_session_tn_lstm.inp_sizes;
    }

    inline const vec_dims_size_t &get_inp_dims_tn_lstm() const {
      return m_session_tn_lstm.inp_node_dims;
    }

    inline const vec_node_names_t &get_inp_names_tn_lstm() const {
      return m_session_tn_lstm.inp_node_names;
    }

    inline const vec_node_names_t &get_out_names_tn_lstm() const {
      return m_session_tn_lstm.out_node_names;
    }

    inline size_t get_inp_size_tn_dnn() const {
      return m_session_tn_dnn.inp_sizes.at(0);
    }
    
    inline const dims_size_t &get_inp_dims_tn_dnn() const {
      return m_session_tn_dnn.inp_node_dims.at(0);
    }

    inline const vec_node_names_t &get_inp_names_tn_dnn() const {
      return m_session_tn_dnn.inp_node_names;
    }

    inline const vec_node_names_t &get_out_names_tn_dnn() const {
      return m_session_tn_dnn.out_node_names;
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

    static void fetch_session_input_params(s_attr &, int64_t);

    void init(
#ifdef DEBUG_INF
       const std::string &,
#endif
       const std::string &,
       const std::string &,
       const std::string &,
       const std::string &,
       const std::string &,
       const std::string &,
       int64_t);


#ifdef DEBUG_ORT
    inline void print(std::string &&tag, const s_attr &session_attr) const {
      std::cout << tag << ": (input: \n\t";
      size_t i;
      size_t j;
      for(i=0;i<session_attr.inp_node_names.size()-1;++i) {
        std::cout << session_attr.inp_node_names.at(i) << ", ";
      }
      std::cout << session_attr.inp_node_names.at(i) << ") ";

      std::cout << "[";
      for(i=0; i < session_attr.inp_node_dims.size() - 1; ++i) {
      
        std::cout << "[";
        for(j=0; j<session_attr.inp_node_dims.at(i).size() - 1; ++j) {
          std::cout << session_attr.inp_node_dims.at(i).at(j) << ", ";
        }
        std::cout << session_attr.inp_node_dims.at(i).at(j) << "], ";
      }

      std::cout << "[";
      for(j=0; j<session_attr.inp_node_dims.at(i).size() - 1; ++j) {
        std::cout << session_attr.inp_node_dims.at(i).at(j) << ", ";
      }
      std::cout << session_attr.inp_node_dims.at(i).at(j) << "]]\n)\n";



      std::cout << tag << ": (output: \n\t";
      for(i=0;i<session_attr.out_node_names.size()-1;++i) {
        std::cout << session_attr.out_node_names.at(i) << ", ";
      }
      std::cout << session_attr.out_node_names.at(i) << ") ";

      std::cout << "[";
      for(i=0; i < session_attr.out_node_dims.size() - 1; ++i) {
      
        std::cout << "[";
        for(j=0; j<session_attr.out_node_dims.at(i).size() - 1; ++j) {
          std::cout << session_attr.out_node_dims.at(i).at(j) << ", ";
        }
        std::cout << session_attr.out_node_dims.at(i).at(j) << "], ";
      }

      std::cout << "[";
      for(j=0; j<session_attr.out_node_dims.at(i).size() - 1; ++j) {
        std::cout << session_attr.out_node_dims.at(i).at(j) << ", ";
      }
      std::cout << session_attr.out_node_dims.at(i).at(j) << "]]\n)\n";


      std::cout << std::endl;


      /*
      std::cout << "(output: \n\t";
      for(i=0;i<session_attr.out_node_names.size()-1;++i) {
        std::cout << session_attr.out_node_names.at(i) << ", ";
      }
      std::cout << session_attr.out_node_names.at(i) << ") ";
      std::cout << "[";
      for(i=0<session_attr.out_node_dims.size() - 1; ++i) {
        std::cout << session_attr.out_node_dims.at(i) << ", ";
      }
      std::cout << session_attr.out_node_dims.at(i) << "])\n";
      */
    }
  public:

    inline void print_all() const {

#ifdef DEBUG_INF
      print("encoder", m_session_tn);
#endif
      
      print("tn_cnn", m_session_tn_cnn);
      print("tn_lstm", m_session_tn_lstm);
      print("tn_dnn", m_session_tn_dnn);
      print("pn", m_session_pn);
      print("cn", m_session_cn);
    }
#endif

  private:
    Ort::Env *m_ort_env;
#ifdef DEBUG_INF
    s_attr m_session_tn;
#endif
    s_attr m_session_tn_cnn;
    s_attr m_session_tn_lstm;
    s_attr m_session_tn_dnn;
    s_attr m_session_pn;
    s_attr m_session_cn;

    sp::SentencePieceProcessor *m_sp_processor{};

    int64_t m_last_input_buffer_len;

    bool m_is_initialized;
  };
}
//eof

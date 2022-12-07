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

#include "rnnt_attrs.h"

#include <iostream>


namespace spr::inference {

  rnnt_attrs::rnnt_attrs(
#ifdef DEBUG_INF
                         const std::string &model_tn_path,
#endif
                         const std::string &model_tn_cnn_path,
                         const std::string &model_tn_lstm_path,
                         const std::string &model_tn_dnn_path,
                         const std::string &model_pn_path,
                         const std::string &model_cn_path,
                         const std::string &model_sp_path,
                         int64_t input_buffer_win_len) :
    m_ort_env(nullptr),
    m_is_initialized(false) {

    init(
#ifdef DEBUG_INF
         model_tn_path,
#endif
         model_tn_cnn_path, model_tn_lstm_path, model_tn_dnn_path, model_pn_path,
         model_cn_path, model_sp_path, input_buffer_win_len);
  }

  void rnnt_attrs::init(
#ifdef DEBUG_INF
                        const std::string &model_tn_path,
#endif
                        const std::string &model_tn_cnn_path,
                        const std::string &model_tn_lstm_path,
                        const std::string &model_tn_dnn_path,
                        const std::string &model_pn_path,
                        const std::string &model_cn_path,
                        const std::string &model_sp_path,
                        int64_t input_buffer_win_len) {

    //    m_ort_env = new Ort::Env{ORT_LOGGING_LEVEL_VERBOSE, "Default"};
    m_ort_env = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Default"};
    Ort::SessionOptions so;
    uint32_t exec_flags = 0;
    //    exec_flags |= NNAPI_FLAG_USE_FP16;

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(so, (int)exec_flags));
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, (int)exec_flags));

#ifdef DEBUG_INF
    m_session_tn = create_session(m_ort_env, so, model_tn_path, input_buffer_win_len);
#endif
    m_session_tn_cnn = create_session(m_ort_env, so, model_tn_cnn_path, input_buffer_win_len);
    m_session_tn_lstm = create_session(m_ort_env, so, model_tn_lstm_path, input_buffer_win_len);
    m_session_tn_dnn = create_session(m_ort_env, so, model_tn_dnn_path, input_buffer_win_len);
    m_session_pn = create_session(m_ort_env, so, model_pn_path, input_buffer_win_len);
    m_session_cn = create_session(m_ort_env, so, model_cn_path, input_buffer_win_len);

    m_last_input_buffer_len = input_buffer_win_len;

    m_sp_processor = new sp::SentencePieceProcessor();
    const auto status = m_sp_processor->Load(model_sp_path);
    if (!status.ok()) {
      // error
      std::cerr << "SentencePiece processor could not be initialized with: " << model_sp_path << std::endl;
      return; //skipping m_is_initialized = true;
    }

    m_is_initialized = true;
  }

  s_attr rnnt_attrs::create_session(Ort::Env *env, const Ort::SessionOptions &so,
                                    const std::string &model_path, int64_t variable_len) {

    Ort::AllocatorWithDefaultOptions alloc;
    auto *session = new Ort::Session(*env, model_path.c_str(), so);
    size_t inp_cnt = session->GetInputCount(), out_cnt = session->GetOutputCount();
    spr::inference::vec_node_names_t inp_names(inp_cnt), out_names(out_cnt);
    spr::inference::vec_dims_size_t inp_dims(inp_cnt), out_dims(out_cnt);
    std::vector<size_t> inp_sizes(inp_cnt);

    // input
    for(size_t i=0; i< inp_cnt; ++i) {
      inp_names[i] = session->GetInputName(i, alloc);

      auto type_info = session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      inp_dims[i] = tensor_info.GetShape();

      inp_sizes[i] = 1L;

      for(size_t j=0; j<inp_dims[i].size(); ++j) {
        if(inp_dims[i][j] < 1) inp_dims[i][j] = variable_len;
        inp_sizes[i] *= inp_dims[i][j];
      }
    }

    // output
    for(size_t i=0; i< out_cnt; ++i) {
      out_names[i] = session->GetOutputName(i, alloc);

      auto type_info = session->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      out_dims[i] = tensor_info.GetShape();

      // out sizes are not needed
    }

    return {
      .session = session,
      .allocator = alloc,
      .inp_sizes = std::move(inp_sizes),
      .inp_node_names = std::move(inp_names),
      .out_node_names = std::move(out_names),
      .inp_node_dims = std::move(inp_dims),
      .out_node_dims = std::move(out_dims)
    };
  }

  rnnt_attrs::~rnnt_attrs() {
    delete m_sp_processor;

    auto del_nodes = [](auto  &session_attr) {

      for(const char *node_name : session_attr.inp_node_names)
        session_attr.allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));
      for(const char *node_name : session_attr.out_node_names)
        session_attr.allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

      delete session_attr.session;
    };

    // release buffers allocated by ORT alloctor
#ifdef DEBUG_INF
    del_nodes(m_session_tn);
#endif

    del_nodes(m_session_tn_cnn);
    del_nodes(m_session_tn_lstm);
    del_nodes(m_session_tn_dnn);
    del_nodes(m_session_pn);
    del_nodes(m_session_cn);

    delete m_ort_env;
  }

  void rnnt_attrs::reset_buffer_win_len(int64_t variable_input_len) {

    if(m_last_input_buffer_len != variable_input_len) {

      // TN_CNN
      fetch_session_input_params(m_session_tn_cnn, variable_input_len);

      // TN_LSTM
      fetch_session_input_params(m_session_tn_lstm, variable_input_len);

      // TN_DNN
      fetch_session_input_params(m_session_tn_dnn, variable_input_len);

      // PN
      fetch_session_input_params(m_session_pn, variable_input_len);

      // CN
      fetch_session_input_params(m_session_cn, variable_input_len);

      m_last_input_buffer_len = variable_input_len;
    } /*else {
      std::cerr << "Reseting buffer length with no effect.\n";
      } */
  }


  void rnnt_attrs::fetch_session_input_params(s_attr &session_attr, int64_t variable_input_len) {
    auto *session = session_attr.session;
    for(size_t i=0; i< session->GetInputCount(); ++i) {

      auto type_info = session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      session_attr.inp_node_dims[i] = tensor_info.GetShape();

      session_attr.inp_sizes[i] = 1L;

      for(size_t j=0; j < session_attr.inp_node_dims[i].size(); ++j) {
        if(session_attr.inp_node_dims[i][j] < 1) session_attr.inp_node_dims[i][j] = variable_input_len;
        session_attr.inp_sizes[i] *= session_attr.inp_node_dims[i][j];
      }
    }
  }
}
//eof

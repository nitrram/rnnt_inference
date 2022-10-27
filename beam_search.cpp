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

#include "beam_search.h"
#include "common/softmax_neon.h"
#include "common/copy_neon.h"

// Onnx headers
#include <cpu_provider_factory.h>

// Sentencepiece headers
#include <sentencepiece_processor.h>

#include <iostream>
#include <cmath>
#include <cassert>

namespace spr::inference {

  size_t beam_search::s_beam_size = 3;
  float beam_search::s_state_beam = 4.6f;
  float beam_search::s_expand_beam = 2.3f;

  beam_search::beam_search(const spr::inference::rnnt_attrs *attrs) :
    m_rnnt(attrs) {
  }

  std::string beam_search::decode(const float *input, std::function<void(std::string)> callback) const {

    std::string result;
    auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto inp_dims_tn = m_rnnt->get_inp_dims_tn();
    auto input_tensor_tn = Ort::Value::CreateTensor<float>(
                                                           memory_info , const_cast<float*>(input), m_rnnt->get_inp_size_tn(),
                                                           inp_dims_tn.data(), inp_dims_tn.size());

    // run transcription network upon the whole utterance
    auto output_tensors_tn =
      m_rnnt->get_session_tn()->Run(Ort::RunOptions{nullptr}, m_rnnt->get_inp_names_tn().data(),
                                    &input_tensor_tn, 1, m_rnnt->get_out_names_tn().data(), 1);

    assert(output_tensors_tn.size() == 1);

    auto tn_shape = output_tensors_tn[0].GetTensorTypeAndShapeInfo().GetShape();

    float embedding[m_rnnt->get_inp_sizes_pn().at(0)];
    float pn_state_buffer[m_rnnt->get_inp_sizes_pn().at(1)];
    float pre_alloc_sum_gelu[m_rnnt->get_inp_size_cn()];

    // init embedding
    for(size_t i=0; i<m_rnnt->get_inp_sizes_pn().at(0); embedding[i++]=0.0f);
    // init pn state
    for(size_t i=0; i<m_rnnt->get_inp_sizes_pn().at(1); pn_state_buffer[i++]=0.0f);

    std::vector<Ort::Value> input_tensors_pn;
    spr::inference::vec_hyps beam_hyps = {{ .prediction = {0}, .logp_score = .0f, .hidden_state = pn_state_buffer, .hidden_state_present = false }};
    spr::inference::vec_hyps process_hyps;

    spr::inference::vec_hyps::const_iterator a_best_it, b_best_it;

    auto best_fun = [](const auto &left, const auto &right) {
      return left.logp_score / left.prediction.size() < right.logp_score / right.prediction.size(); };

    //e.g. 89x [1,1,512]
    for(int step_t = 0; step_t < tn_shape.at(1); ++step_t) {
      process_hyps = beam_hyps;
      beam_hyps.clear();

      while (true) {
        if (beam_hyps.size() >= s_beam_size) break;

        a_best_it = std::max_element(process_hyps.cbegin(), process_hyps.cend(), best_fun);

        if (a_best_it == process_hyps.cend()) {
          std::cerr << "Processed hypothesises's been empty. Breaking out of the loop" << std::endl;
          break;
        }

        auto a_best_tok = *a_best_it;

        if (!beam_hyps.empty()) {
          b_best_it = std::max_element(beam_hyps.cbegin(), beam_hyps.cend(), best_fun);
          auto b_best_tok = *b_best_it;

          /*state_beam = 4.6*/
          if (b_best_tok.logp_score >= s_state_beam + a_best_tok.logp_score) break;
        }

        if (a_best_it != process_hyps.cend())
          process_hyps.erase(a_best_it);

        // forward PN
        auto output_tensors_pn =
          predict(memory_info,
                  embedding,
                  a_best_tok);

        // forward CN
        auto output_tensors_cn =
          joint(memory_info,
                output_tensors_tn[0].GetTensorData<float>() +
                (step_t * m_rnnt->get_inp_size_cn()),
                output_tensors_pn[0].GetTensorData<float>(),
                pre_alloc_sum_gelu);

        // sum the joint probabilities to 1
        softmax(output_tensors_cn.at(0).GetTensorMutableData<float>(),
                output_tensors_cn.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2));

        const auto *log_probs_raw = output_tensors_cn.at(0).GetTensorData<float>();

        auto top_log_probs =
          find_top_k_probs(log_probs_raw,
                           output_tensors_cn.at(
                                                0).GetTensorTypeAndShapeInfo().GetShape()[2]);

        float best_logp = ((std::get<1>(top_log_probs.at(0)) != 0) ?
                           std::get<0>(top_log_probs.at(0)) : std::get<0>(
                                                                          top_log_probs.at(1)));
        for (const auto & top_log_prob : top_log_probs) {

          token_t top_hyp = {
            .prediction = a_best_tok.prediction,
            .logp_score = a_best_tok.logp_score +
            std::get<0>(top_log_prob),
            .hidden_state = pn_state_buffer,
            .hidden_state_present = false
          };

          // if the found top scored label is equal to blank, expand beam hypothesis
          if (std::get<1>(top_log_prob) == 0) {
            if (a_best_tok.hidden_state_present) {
              top_hyp.hidden_state = new float[m_rnnt->get_inp_sizes_pn().at(1)];
              fcopy(a_best_tok.hidden_state, top_hyp.hidden_state,
                    m_rnnt->get_inp_sizes_pn().at(1));
              top_hyp.hidden_state_present = a_best_tok.hidden_state_present; //true
            }

            beam_hyps.emplace_back(std::move(top_hyp));

            continue; //for( step_e )
          }

          if(std::get<0>(top_log_prob) >= best_logp - s_expand_beam) {
            top_hyp.prediction.emplace_back(std::get<1>(top_log_prob));
            if(output_tensors_pn.at(1) != Ort::Value(nullptr)) {
              top_hyp.hidden_state = new float[m_rnnt->get_inp_sizes_pn().at(1)];
              fcopy(output_tensors_pn.at(1).GetTensorData<float>(), top_hyp.hidden_state,
                    m_rnnt->get_inp_sizes_pn().at(1));
              top_hyp.hidden_state_present = true;
            }

            process_hyps.emplace_back(std::move(top_hyp));
          }
        } //for(int step_e = 0; step_e < top_log_probs.size(); ++step_e)
      } // while(true)

      auto tmp_best_hyp = *std::max_element(beam_hyps.cbegin(), beam_hyps.cend(), best_fun);
      // un-tokenize
      auto tmp_particies = std::vector(tmp_best_hyp.prediction.cbegin() + 1, tmp_best_hyp.prediction.cend());
      m_rnnt->get_sentencepiece_processor()->Decode(tmp_particies, &result);
      callback(result);

    } //for(int step_t = 0; step_t < tn_shape[1]; ++step_t)

    auto best_hyp = *std::max_element(beam_hyps.cbegin(), beam_hyps.cend(), best_fun);

    // un-tokenize
    auto particies = std::vector(best_hyp.prediction.cbegin() + 1, best_hyp.prediction.cend());
    m_rnnt->get_sentencepiece_processor()->Decode(particies, &result);

    return result;
  }

  std::vector<Ort::Value> beam_search::predict(const Ort::MemoryInfo &memory_info,
                                               float *pre_alloc_embedding,
                                               const token_t &best_hyp) const {
    // create one hot vector for the prediction
    ssize_t bpe_idx = !best_hyp.prediction.empty() ? best_hyp.prediction.back() : -1;
    if(bpe_idx > 0) // && (size_t)bpe_idx <= target_sizes_pn.at(0))
      pre_alloc_embedding[ bpe_idx -1 ] = 1.0f;


    std::vector<Ort::Value> input_tensors_pn;
    input_tensors_pn.push_back(
                               Ort::Value::CreateTensor<float>(memory_info, pre_alloc_embedding,
                                                               m_rnnt->get_inp_sizes_pn().at(0),
                                                               m_rnnt->get_inp_dims_pn().at(0).data(),
                                                               m_rnnt->get_inp_dims_pn().at(0).size())
                               );

    // previous state's data should be already copied and isolated
    input_tensors_pn.push_back(
                               Ort::Value::CreateTensor<float>(memory_info, best_hyp.hidden_state,
                                                               m_rnnt->get_inp_sizes_pn().at(1),
                                                               m_rnnt->get_inp_dims_pn().at(1).data(),
                                                               m_rnnt->get_inp_dims_pn().at(1).size())
                               );

    auto output_tensors_pn =
      m_rnnt->get_session_pn()->Run(Ort::RunOptions(nullptr),
                                    m_rnnt->get_inp_names_pn().data(), input_tensors_pn.data(),
                                    input_tensors_pn.size() , m_rnnt->get_out_names_pn().data(),
                                    m_rnnt->get_out_names_pn().size());

    //reset embedding
    if(bpe_idx > 0) {// && (size_t)bpe_idx <= target_sizes_pn.at(0)) {
      pre_alloc_embedding[ bpe_idx - 1 ] = 0.0f;
    }

    return output_tensors_pn;
  }

  std::vector<Ort::Value> beam_search::joint(const Ort::MemoryInfo &memory_info,
                                             const float *tn_data,
                                             const float *pn_data,
                                             float *pre_alloc_sum_gelu) const {
    float sum;

    // lin_input = gelu(logp)
#ifdef ORR_OMP
#pragma omp parallel for num_threads(THREADSIZE) default(none) private(sum) shared(tn_data, pn_data, pre_alloc_sum_gelu)
#endif
    for(size_t i=0; i<m_rnnt->get_inp_size_cn(); ++i) {

      // sum log probabilities from TN and PN
      sum = tn_data[i] + pn_data[i];

      // lin_input
      pre_alloc_sum_gelu[i] = 0.5f * sum * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) *
                                                             (sum + 0.044715f * std::pow(sum, 3.0f))));
    }

    Ort::Value joint_in = Ort::Value::CreateTensor<float>(memory_info, pre_alloc_sum_gelu,
                                                          m_rnnt->get_inp_size_cn(),
                                                          m_rnnt->get_inp_dims_cn().data(),
                                                          m_rnnt->get_inp_dims_cn().size());

    auto result =
      m_rnnt->get_session_cn()->Run(Ort::RunOptions(nullptr), m_rnnt->get_inp_names_cn().data(),
                                    &joint_in, 1, m_rnnt->get_out_names_cn().data(),
                                    m_rnnt->get_out_names_cn().size());



    return result;
  }

  vec_top_prob_t beam_search::find_top_k_probs(const float *inp, size_t size) {
    std::vector<top_prob_t> inp2sort;

    // fill tuples of values and theirs accompanied indexes
    for(size_t i=0; i<size; ++i) {
      inp2sort.emplace_back(std::make_tuple(inp[i], i));
    }

    // sort the tuples
    std::sort(inp2sort.begin(), inp2sort.end(), [](const top_prob_t &left, const top_prob_t &right){
      return std::get<0>(left) > std::get<0>(right); });

    return {inp2sort.begin(), inp2sort.begin() + 3};
  }
}
//eof

/*** Created by Kodo in '22 **/

/** Legend:
 * PN/pn ~ prediction network
 * TN/tn ~ transcription network
 * CN/cn ~ classifier network (linearization of the joint PN&TN)
 * BPE ~ byte-pairs
 */

#include <omp.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <queue>
#include <tuple>
#include <cassert>
#include <algorithm> /*std::max_element*/
#include <cstddef> /*std::nullptr_t*/

//#ifdef DEBUG_INF
#include <chrono>
#include <ctime>
//#endif

// sentencepiece
#include "sp/sentencepiece_processor.h"

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "fbank_80_201.h"
#include "normalize_vals.h"
#include "softmax_neon.h"
#include "tokenizer.h"
#include "wavread.h"
#include "feat.h"
#include "fmadd_neon.h"
#include "copy_neon.h"


#define WAV_FILE "common_voice_cs_25695144_16.wav"

#define BPE_MODEL "80_bpe.model"
#define TN_FILE "rnnt_tn.ort"
#define PN_FILE "rnnt_pn.ort"
#define CN_FILE "rnnt_cn.ort"

#define BEAM_SIZE 3
#define STATE_BEAM 4.6f
#define EXPAND_BEAM 2.3f
#define MELS FBANK_TP_ROW_SIZE

// forward declarations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

namespace sp=sentencepiece;

enum io {
  INPUT,
  OUTPUT
};

struct token_t;

using dims_size_t=std::vector<int64_t>;
using vec_dims_size_t=std::vector<dims_size_t>;
using vec_node_names_t=std::vector<const char*>;
using top_prob_t = std::tuple<float, size_t>; // <value, original_position>
using vec_top_prob_t = std::vector<top_prob_t>;
using vec_hyps = std::vector<token_t>;

template<typename T>
inline std::string
print_vector(const std::vector<T> &vec) {
  std::ostringstream oss;
  oss << "[";
  for(size_t i=0; i < vec.size(); ++i) {
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

size_t create_feat_inp(const int16_t*, size_t, float**);
size_t norm_inp(float*, size_t);

std::vector<Ort::Value>
predict(Ort::Session *,
        const Ort::MemoryInfo &,
        float *,
        float *,
        const token_t &,
        const std::vector<size_t>,
        const vec_dims_size_t &,
        const vec_node_names_t &,
        const vec_node_names_t &);


std::vector<Ort::Value>
joint(Ort::Session *,
      const Ort::MemoryInfo &,
      const float *,
      const float *,
      float *,
      size_t,
      const dims_size_t&,
      const vec_node_names_t &,
      const vec_node_names_t &);


void
obtain_io_attrs(const Ort::Session *,
                Ort::AllocatorWithDefaultOptions&,
                vec_node_names_t&,
                vec_dims_size_t&,
                size_t,
                std::vector<size_t>&,
                size_t,
                io);

vec_top_prob_t
find_top_k_probs(const float *,
                 size_t);


std::string
the_model_inference(Ort::Session *,
                    Ort::Session *,
                    Ort::Session *,
                    const float *,
                    size_t,
                    const std::vector<size_t>&,
                    size_t,
                    const dims_size_t&,
                    const vec_dims_size_t&,
                    const dims_size_t&,
                    const vec_node_names_t&,
                    const vec_node_names_t&,
                    const vec_node_names_t&,
                    const vec_node_names_t&,
                    const vec_node_names_t&,
                    const vec_node_names_t&,
                    const sp::SentencePieceProcessor &);


// forward declarations <<<<<<<<<<<<<<<<<<<<<<<<<<<<<



int
main(int argc,
     char *argv[]) {

  //#ifdef DEBUG_INF
  std::chrono::time_point<std::chrono::system_clock> start, end;
  //#endif

  auto *wav = new spr::wavread();
  wav->init(WAV_FILE);

  auto *ortenv = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "onnxinfere"};
  Ort::SessionOptions so;

  uint32_t exec_flags = 0;

  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(so, (int)exec_flags));
  auto *session_tn = new Ort::Session(*ortenv, TN_FILE, so);
  auto *session_pn = new Ort::Session(*ortenv, PN_FILE, so);
  auto *session_cn = new Ort::Session(*ortenv, CN_FILE, so);

  Ort::AllocatorWithDefaultOptions allocator_for_tn;
  Ort::AllocatorWithDefaultOptions allocator_for_pn;
  Ort::AllocatorWithDefaultOptions allocator_for_cn;

  size_t tn_inp_cnt = session_tn->GetInputCount(), tn_out_cnt = session_tn->GetOutputCount();
  size_t pn_inp_cnt = session_pn->GetInputCount(), pn_out_cnt = session_pn->GetOutputCount();
  size_t cn_inp_cnt = session_cn->GetInputCount(), cn_out_cnt = session_cn->GetOutputCount();

  std::vector<size_t> inp_sizes_tn(tn_inp_cnt), out_sizes_tn(tn_out_cnt);
  std::vector<size_t> inp_sizes_pn(pn_inp_cnt), out_sizes_pn(pn_out_cnt);
  std::vector<size_t> inp_sizes_cn(cn_inp_cnt), out_sizes_cn(cn_out_cnt);

  vec_node_names_t inp_node_names_tn(tn_inp_cnt), out_node_names_tn(tn_out_cnt);
  vec_node_names_t inp_node_names_pn(pn_inp_cnt), out_node_names_pn(pn_out_cnt);
  vec_node_names_t inp_node_names_cn(cn_inp_cnt), out_node_names_cn(cn_out_cnt);

  vec_dims_size_t inp_node_dims_tn(tn_inp_cnt), out_node_dims_tn(tn_out_cnt);
  vec_dims_size_t inp_node_dims_pn(pn_inp_cnt), out_node_dims_pn(pn_out_cnt);
  vec_dims_size_t inp_node_dims_cn(cn_inp_cnt), out_node_dims_cn(cn_out_cnt);


  int16_t *wav_content = NULL;
  float *feat_inp = NULL;

  size_t feat_mat_size = 0;
  size_t variable_len = 1;

  std::string result = "[no result]";

#ifdef DEBUG_INF
  std::ostringstream ossw;
#endif

  sp::SentencePieceProcessor sp_processor;
  // init sentencepiece processor
  const auto status = sp_processor.Load(BPE_MODEL);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    // error
    goto bail_out;
  }

  if(!wav->prepare_to_read()) goto bail_out;

  wav_content = new int16_t[wav->get_num_samples()];

  // create int16_t* buffer for features
  wav->read_data_to_int16(wav_content, wav->get_num_samples());

  // fill feat_inp with the wav data and transform it into the model's shape
  feat_mat_size = create_feat_inp(wav_content, wav->get_num_samples(), &feat_inp);

  // this is the number of processed windows altogether
  variable_len = feat_mat_size / MELS;

  // normalize input
  norm_inp(feat_inp, variable_len);

  //TN - transcription network
  obtain_io_attrs(session_tn, allocator_for_tn, inp_node_names_tn, inp_node_dims_tn, session_tn->GetInputCount(), inp_sizes_tn, variable_len, INPUT);
  obtain_io_attrs(session_tn, allocator_for_tn, out_node_names_tn, out_node_dims_tn, session_tn->GetOutputCount(), out_sizes_tn, variable_len, OUTPUT);

  //PN - prediction network
  obtain_io_attrs(session_pn, allocator_for_pn, inp_node_names_pn, inp_node_dims_pn, session_pn->GetInputCount(), inp_sizes_pn, variable_len, INPUT);
  obtain_io_attrs(session_pn, allocator_for_pn, out_node_names_pn, out_node_dims_pn, session_pn->GetOutputCount(), out_sizes_pn, variable_len, OUTPUT);

  //CN - classifier (linear) network
  obtain_io_attrs(session_cn, allocator_for_cn, inp_node_names_cn, inp_node_dims_cn, session_cn->GetInputCount(), inp_sizes_cn, variable_len, INPUT);
  obtain_io_attrs(session_cn, allocator_for_cn, out_node_names_cn, out_node_dims_cn, session_cn->GetOutputCount(), out_sizes_cn, variable_len, OUTPUT);

  assert(inp_sizes_tn.at(0) == feat_mat_size);

#ifdef DEBUG_INF
  for(size_t x=0; x < tn_inp_cnt; ++x) {
    std::cout << "TN ~input size[" << x << "] = " << inp_sizes_tn[x] << " ("<< print_vector(inp_node_dims_tn[x]) << ") <---- " << variable_len << " features in!\n";
  }
  for(size_t x=0; x < tn_out_cnt; ++x) {
    std::cout << "TN ~out size[" << x << "] = " << out_sizes_tn[x] << " ("<< print_vector(out_node_dims_tn[x]) << ") <---- OUT \n";
  }
  for(size_t x=0; x < pn_inp_cnt; ++x) {
    std::cout << "PN ~input size[" << x << "] = " << inp_sizes_pn[x] << " ("<< print_vector(inp_node_dims_pn[x]) << ((x == 0) ? "<---- PN + TN joint (one hot BPE vec))\n" : "<---- neural out )\n");
  }
  for(size_t x=0; x < pn_out_cnt; ++x) {
    std::cout << "PN ~out size[" << x << "] = " << out_sizes_pn[x] << " ("<< print_vector(out_node_dims_pn[x]) << ")" << ((x == 0) ? "<---- neural out\n" :"<---- parameters IN (recurrent))\n");
  }
  for(size_t x=0; x < cn_inp_cnt; ++x) {
    std::cout << "CN ~input size[" << x << "] = " << inp_sizes_cn[x] << " ("<< print_vector(inp_node_dims_cn[x]) << ((x == 0) ? "<---- joint in\n" : "<---- joint in )\n");
  }
  for(size_t x=0; x < cn_out_cnt; ++x) {
    std::cout << "CN ~out size[" << x << "] = " << out_sizes_cn[x] << " ("<< print_vector(out_node_dims_cn[x]) << ")" << ((x == 0) ? "<---- joint out\n" :"<---- joint out\n");
  }
  std::cout << "===============================================\n\n";
#endif //DEBUG_INF

  assert(inp_node_dims_tn.size() == 1);
  start = std::chrono::system_clock::now();
  result = the_model_inference(session_tn,
                               session_pn,
                               session_cn,
                               feat_inp,
                               inp_sizes_tn[0],
                               inp_sizes_pn,
                               inp_sizes_cn[0],
                               inp_node_dims_tn[0],
                               inp_node_dims_pn,
                               inp_node_dims_cn[0],
                               inp_node_names_tn,
                               out_node_names_tn,
                               inp_node_names_pn,
                               out_node_names_pn,
                               inp_node_names_cn,
                               out_node_names_cn,
                               sp_processor);

  end = std::chrono::system_clock::now();

  std::cout << std::endl << result << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms])\n";

  // release buffers allocated by ORT alloctor
  for(const char *node_name : inp_node_names_tn)
    allocator_for_tn.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  for(const char *node_name : out_node_names_tn)
    allocator_for_tn.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  for(const char *node_name : inp_node_names_pn)
    allocator_for_tn.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  for(const char *node_name : out_node_names_pn)
    allocator_for_tn.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));


  delete []feat_inp;
  delete []wav_content;

 bail_out:
  delete session_pn;
  delete session_tn;
  delete ortenv;

  wav->deallocate();
  delete wav;

  return 0;
}

void
obtain_io_attrs(const Ort::Session *session,
                     Ort::AllocatorWithDefaultOptions &allocator,
                     vec_node_names_t &names,
                     vec_dims_size_t &dims,
                     size_t size,
                     std::vector<size_t> &out_size,
                     size_t win_size,
                     io inp_or_out) {

  for(size_t i=0; i<size; ++i) {
    names[i] = (inp_or_out == INPUT ? session->GetInputName(i, allocator) : session->GetOutputName(i, allocator) );

    auto type_info = ( inp_or_out ==  INPUT ? session->GetInputTypeInfo(i) : session->GetOutputTypeInfo(i) );
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    dims[i] = tensor_info.GetShape();

    out_size[i] = 1L;

    if(inp_or_out != OUTPUT) {
      for(size_t j=0; j<dims[i].size(); ++j) {
        if(dims[i][j] < 1) dims[i][j] = win_size;
        out_size[i] *= dims[i][j];
      }
    }
  }
}

std::string
the_model_inference(Ort::Session *session_tn,
                    Ort::Session *session_pn,
                    Ort::Session *session_cn,
                    const float *enc_inp,
                    size_t target_size_tn/* win_inc * out_bpe*/,
                    const std::vector<size_t> &target_sizes_pn, /*size of embedding  ~ BPE -1 and hidden_state */
                    size_t target_size_cn,
                    const dims_size_t &tn_inp_dims,
                    const vec_dims_size_t &pn_inp_dims_vec,
                    const dims_size_t &cn_inp_dims,
                    const vec_node_names_t &input_node_names_tn,
                    const vec_node_names_t &output_node_names_tn,
                    const vec_node_names_t &input_node_names_pn,
                    const vec_node_names_t &output_node_names_pn,
                    const vec_node_names_t &input_node_names_cn,
                    const vec_node_names_t &output_node_names_cn,
                    const sp::SentencePieceProcessor &sp_processor) {

  auto memory_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

#ifdef DEBUG_INF
  std::cout << "enc inp size: " << target_size_tn << std::endl;
  std::cout << "pn inp size: " << target_sizes_pn.at(0) <<  std::endl;
  std::cout << "joint inp size: " << target_size_cn << std::endl;
#endif

  Ort::Value input_tensor_tn = Ort::Value::CreateTensor<float>(
    memory_info , const_cast<float*>(enc_inp), target_size_tn,
    tn_inp_dims.data(), tn_inp_dims.size());

  // run transcription network upon the whole utterance
  auto output_tensors_tn =
    session_tn->Run(Ort::RunOptions{nullptr}, input_node_names_tn.data(),
                    &input_tensor_tn, 1, output_node_names_tn.data(), 1);

  assert(output_tensors_tn.size() == 1);

  auto tn_shape = output_tensors_tn[0].GetTensorTypeAndShapeInfo().GetShape();

  float embedding[target_sizes_pn.at(0)];
  float pn_state_buffer[target_sizes_pn.at(1)];
#ifdef DEBUG_INF
  std::cout << "PN state buffer size: " << target_sizes_pn.at(1) << std::endl;
#endif

  float pre_alloc_sum_gelu[target_size_cn];

  // init embedding
  for(size_t i=0; i<target_sizes_pn.at(0); embedding[i++]=0.0f);
  // init pn state
  for(size_t i=0; i<target_sizes_pn.at(1); pn_state_buffer[i++]=0.0f);

  std::vector<Ort::Value> input_tensors_pn;
  vec_hyps beam_hyps = {{ .prediction = {0}, .logp_score = .0f, .hidden_state = pn_state_buffer, .hidden_state_present = false }};
  vec_hyps process_hyps;

  vec_hyps::const_iterator a_best_it, b_best_it;

  auto best_fun = [](const auto &left, const auto &right) {
    return left.logp_score / left.prediction.size() < right.logp_score / right.prediction.size(); };

#ifdef DEBUG_INF
	//  bool _tmp_rem = false;
  int _tmp_debug_step = 0;
#endif

  //e.g. 89x [1,1,512]
  for(int step_t = 0; step_t < tn_shape[1]; ++step_t) {

    process_hyps = beam_hyps;
    beam_hyps.clear();

#ifdef DEBUG_INF
    std::cout << "[" << step_t << "] = process_hyps: " << process_hyps.size() << " = " <<
      (process_hyps.size() == 1 ? process_hyps.back().logp_score : a_best_it->logp_score) << std::endl;
#endif

    while(1) {
      if(beam_hyps.size() >= BEAM_SIZE) {

#ifdef DEBUG_INF
        std::cout << "beam size reached in step [" << step_t << "]\n";
#endif

        break;
      }

      a_best_it = std::max_element(process_hyps.cbegin(), process_hyps.cend(), best_fun);

      if(a_best_it == process_hyps.cend()) {
        std::cerr << "Processed hypothesises's been empty. Breaking out of the loop\n";
        break;
      }

      auto a_best_tok = *a_best_it;

      if(beam_hyps.size() > 0) {
        b_best_it = std::max_element(beam_hyps.cbegin(), beam_hyps.cend(), best_fun);
        auto b_best_tok = *b_best_it;

        /*state_beam = 4.6*/
        if(b_best_tok.logp_score >= STATE_BEAM + a_best_tok.logp_score) {
          break;
        }
      }

      if(a_best_it != process_hyps.cend())
        process_hyps.erase(a_best_it);

#ifdef DEBUG_INF
      if(!_tmp_rem && step_t == 0) {
        std::cout << "predict: input: [" << a_best_tok.prediction.back() << "]\n";
      }
#endif

      // forward PN
      auto output_tensors_pn =
        predict(session_pn,
                memory_info,
                embedding,
                pn_state_buffer,
                a_best_tok,
                target_sizes_pn,
                pn_inp_dims_vec,
                input_node_names_pn,
                output_node_names_pn);


#ifdef DEBUG_INF
      if(!_tmp_rem && step_t == _tmp_debug_step) {
        auto *enc_t =  output_tensors_tn.at(0).GetTensorData<float>() + (step_t * target_size_cn);
        std::cout << "enc: output: [\n";
        for(size_t i = 0; i < target_size_cn; ++i) {
          std::cout << enc_t[i] << ((target_size_cn - 1 > i) ? ", " : "\n]\n");
        }
      }

      if(step_t == _tmp_debug_step) {
        auto *pn = output_tensors_pn.at(1).GetTensorData<float>();
        size_t pn_size = output_tensors_pn.at(1).GetTensorTypeAndShapeInfo().GetShape().at(2);
        std::cout << "pred: output: [\n";
        for(size_t i = 0; i < pn_size; ++i) {
          std::cout << pn[i] << ((pn_size - 1 > i) ? ", " : "\n]\n");
        }
      }
#endif

      // forward CN
      auto output_tensors_cn =
        joint(session_cn,
              memory_info,
              output_tensors_tn[0].GetTensorData<float>() + (step_t * target_size_cn),
              output_tensors_pn[0].GetTensorData<float>(),
              pre_alloc_sum_gelu,
              target_size_cn,
              cn_inp_dims,
              input_node_names_cn,
              output_node_names_cn);

#ifdef DEBUG_INF
      if(!_tmp_rem && step_t == _tmp_debug_step) {
        auto *jn =  output_tensors_cn.at(0).GetTensorData<float>();
        size_t jn_size = output_tensors_cn.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
        std::cout << "joint: output: [\n";
        for(size_t i = 0; i < jn_size - 1; ++i) {
          std::cout << jn[i] << ((jn_size - 2 > i) ? ", " : "\n]\n");
        }
      }
#endif

      // sum the joint probabilities to 1
      softmax(output_tensors_cn.at(0).GetTensorMutableData<float>(), output_tensors_cn.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2));

#ifdef DEBUG_INF
      if(!_tmp_rem && step_t == _tmp_debug_step) {
        auto *jn =  output_tensors_cn.at(0).GetTensorData<float>();
        size_t jn_size = output_tensors_cn.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
        std::cout << "joint softmax: output: [\n";
        for(size_t i = 0; i < jn_size - 1; ++i) {
          std::cout << jn[i] << ((jn_size - 2 > i) ? ", " : "\n]\n");
        }
        _tmp_rem = true;
      }
#endif

      const float *log_probs_raw = output_tensors_cn.at(0).GetTensorData<float>();

      auto top_log_probs = find_top_k_probs(log_probs_raw, output_tensors_cn.at(0).GetTensorTypeAndShapeInfo().GetShape()[2]);


#ifdef DEBUG_INF
      if(true) {
        std::cout << "top3: [" <<
        std::get<0>(top_log_probs.at(0)) << "|" << std::get<1>(top_log_probs.at(0)) << ", " <<
        std::get<0>(top_log_probs.at(1)) << "|" << std::get<1>(top_log_probs.at(1)) << ", " <<
        std::get<0>(top_log_probs.at(2)) << "|" << std::get<1>(top_log_probs.at(2)) << "]\n";
      }
#endif

      // if top log prob is equal blank position, use the next one's value
      float best_logp = ((std::get<1>(top_log_probs.at(0)) != 0) ? std::get<0>(top_log_probs.at(0)) : std::get<0>(top_log_probs.at(1)));

#ifdef DEBUG_INF
      if(step_t == _tmp_debug_step) {
        std::cout << "best_logp: " << best_logp << std::endl;
      }
#endif

      for(size_t step_e = 0; step_e < top_log_probs.size(); ++step_e) {

        token_t top_hyp = {
              .prediction = a_best_tok.prediction,
              .logp_score = a_best_tok.logp_score + std::get<0>(top_log_probs.at(step_e)),
              .hidden_state = pn_state_buffer,
              .hidden_state_present = false
        };

        // if the found top scored label is equal to blank, expand beam hypothesis
        if(std::get<1>(top_log_probs.at(step_e)) == 0) {

          if(a_best_tok.hidden_state_present) {
            top_hyp.hidden_state = new float[target_sizes_pn.at(1)];
            fcopy(a_best_tok.hidden_state, top_hyp.hidden_state, target_sizes_pn.at(1));
            top_hyp.hidden_state_present = a_best_tok.hidden_state_present; //true
          }

          beam_hyps.emplace_back(std::move(top_hyp));

#ifdef DEBUG_INF
          if(step_t == _tmp_debug_step) {
            std::cout << "\tbeam : [" << beam_hyps.size() << "]\n";
          }
#endif
          continue; //for( step_e )
        }

        if(std::get<0>(top_log_probs.at(step_e)) >= best_logp - EXPAND_BEAM) {

          top_hyp.prediction.emplace_back(std::get<1>(top_log_probs.at(step_e)));
          if(output_tensors_pn.at(1) != Ort::Value(nullptr)) {
            top_hyp.hidden_state = new float[target_sizes_pn.at(1)];
            fcopy(output_tensors_pn.at(1).GetTensorData<float>(), top_hyp.hidden_state, target_sizes_pn.at(1));
            top_hyp.hidden_state_present = true;
          }

          process_hyps.emplace_back(std::move(top_hyp));

#ifdef DEBUG_INF
          if(step_t == _tmp_debug_step) {
            std::cout << "\tproc : [" << process_hyps.size() << "]\n";
          }
#endif
        }
      } //for(int step_e = 0; step_e < top_log_probs.size(); ++step_e)

      assert(output_tensors_cn.size() == 1);
      assert(output_tensors_cn.at(0).GetTensorTypeAndShapeInfo().GetShape().size() == 3);

    } // while(1)


#ifdef DEBUG_INF
  for(const auto &hyp : beam_hyps) {
    std::vector<int> prediction(hyp.prediction.cbegin() + 1, hyp.prediction.cend());
    std::cout << "hyp: " << print_vector(prediction) << std::endl;
    std::string rr;
    sp_processor.Decode(prediction, &rr);
    std::cout << "hyp_str: " << rr << "\n\n";
  }
#endif

  } //for(int step_t = 0; step_t < tn_shape[1]; ++step_t) {


#ifdef DEBUG_INF
  for(const auto &hyp : beam_hyps) {
    std::vector<int> prediction(hyp.prediction.cbegin() + 1, hyp.prediction.cend());
    std::cout << "hyp: " << print_vector(prediction) << std::endl;
  }
#endif


  auto best_hyp = *std::max_element(beam_hyps.cbegin(), beam_hyps.cend(), best_fun);

  std::string result;
  auto particies = std::vector(best_hyp.prediction.cbegin() + 1, best_hyp.prediction.cend());
  sp_processor.Decode(particies, &result);

  return result;
}

std::vector<Ort::Value>
predict(Ort::Session *session_pn,
        const Ort::MemoryInfo &memory_info,
        float *pre_alloc_embedding,
        float *pre_alloc_state,
        const token_t &best_hyp,
        const std::vector<size_t> target_sizes_pn,
        const vec_dims_size_t &pn_inp_dims_vec,
        const vec_node_names_t &input_node_names_pn,
        const vec_node_names_t &output_node_names_pn) {

  // create one hot vector for the prediction
  ssize_t bpe_idx = !best_hyp.prediction.empty() ? best_hyp.prediction.back() : -1;
  if(bpe_idx > 0) // && (size_t)bpe_idx <= target_sizes_pn.at(0))
    pre_alloc_embedding[ bpe_idx -1 ] = 1.0f;


  std::vector<Ort::Value> input_tensors_pn;
  input_tensors_pn.push_back(
    Ort::Value::CreateTensor<float>(memory_info, pre_alloc_embedding,
                                    target_sizes_pn.at(0), pn_inp_dims_vec.at(0).data(),
                                    pn_inp_dims_vec.at(0).size())
  );

  // previous state's data should be already copied and isolated
  input_tensors_pn.push_back(
    Ort::Value::CreateTensor<float>(memory_info, best_hyp.hidden_state,//pre_alloc_state,
                                    target_sizes_pn.at(1), pn_inp_dims_vec.at(1).data(),
                                    pn_inp_dims_vec.at(1).size())
  );

  auto output_tensors_pn =
    session_pn->Run(Ort::RunOptions(nullptr), input_node_names_pn.data(), input_tensors_pn.data(),
                    input_tensors_pn.size(), output_node_names_pn.data(), output_node_names_pn.size());


  //reset embedding
  if(bpe_idx > 0) {// && (size_t)bpe_idx <= target_sizes_pn.at(0)) {
    pre_alloc_embedding[ bpe_idx - 1 ] = 0.0f;
  }

  return output_tensors_pn;
}

std::vector<Ort::Value>
joint(Ort::Session *session_cn,
      const Ort::MemoryInfo &memory_info,
      const float *tn_data,
      const float *pn_data,
      float *pre_alloc_sum_gelu,
      size_t target_size_cn,
      const dims_size_t& cn_inp_dims,
      const vec_node_names_t &input_node_names_cn,
      const vec_node_names_t &output_node_names_cn) {

  float sum;

  // lin_input = gelu(logp)
#pragma omp parallel for num_threads(THREADSIZE) private(sum)
  for(size_t i=0; i<target_size_cn; ++i) {

    // sum log probabilities from TN and PN
    sum = tn_data[i] + pn_data[i];

    // lin_input
    pre_alloc_sum_gelu[i] = 0.5f * sum * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (sum + 0.044715f * std::pow(sum, 3.0f))));
  }


#ifdef DEBUG_INF
  std::cout << "JOINT IN dim: " << print_vector(cn_inp_dims) << std::endl;
  std::cout << "JOINT IN: [\n";
  for(size_t i=0; i< target_size_cn; ++i) {
    std::cout << pre_alloc_sum_gelu[i] << (i < target_size_cn - 1 ? ", " : "\n]\n");
  }
#endif

  Ort::Value joint_in = Ort::Value::CreateTensor<float>(memory_info, pre_alloc_sum_gelu,
                                                        target_size_cn, cn_inp_dims.data(),
                                                        cn_inp_dims.size());

  auto result =
    session_cn->Run(Ort::RunOptions(nullptr), input_node_names_cn.data(),
                    &joint_in, 1, output_node_names_cn.data(), output_node_names_cn.size());

#ifdef DEBUG_INF
  auto *data  = result.at(0).GetTensorMutableData<float>();
  std::cout << "JOINT OUT: [\n";
  for(size_t i=0; i < 512; ++i) {
    std::cout << data[i] << (i < target_size_cn - 1 ? ", " : "\n]\n");
  }
#endif

  return result;
}

size_t
create_feat_inp(const int16_t *wav_con,
                size_t wav_con_len,
                float **out) {

  spr::feat::Fbank fbank(spr::feat::SAMPLE_RATE, MELS, spr::feat::FFT_LEN);
  auto mat_size = fbank.alter_features_matrix_size(wav_con_len);
  *out = new float[mat_size];

  if(!fbank.compute_features(wav_con, *out)) {
    return mat_size;
  }

  return 0L;
}

size_t
norm_inp(float *out, size_t stride) {

#pragma omp parallel for num_threads(THREADSIZE)
  for(uint32_t frame_i=0; frame_i < stride; ++frame_i) {
    //normalize values in-place
    fdsub(out + frame_i * MELS, glob_mean, glob_std, MELS);
  }

  return stride;
}

vec_top_prob_t
find_top_k_probs(const float *inp,
                 size_t size) {

  std::vector<top_prob_t> inp2sort;

  // fill tuples of values and theirs accompanied indexes
  for(size_t i=0; i<size; ++i) {
    inp2sort.emplace_back(std::make_tuple(inp[i], i));
  }

  // sort the tuples
  std::sort(inp2sort.begin(), inp2sort.end(), [](const top_prob_t &left, const top_prob_t &right){
    return std::get<0>(left) > std::get<0>(right); });

  return std::vector(inp2sort.begin(), inp2sort.begin() + 3);
}
//eof

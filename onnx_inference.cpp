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

/** Legend:
 * PN/pn ~ prediction network
 * TN/tn ~ transcription network
 * CN/cn ~ classifier network (linearization of the joint PN&TN)
 * BPE ~ byte-pairs
 */

#include "common/thr_queue.h"
#include "common/buf_type.h"
//#include "pulse/pulse_rec.h"
#include "wav/wav_buf_simulator.h"
#include "wav/wavread.h"

#include "input_processing.h" // transforms wav stream into normalized features
#include "model_structs.h"
#include "beam_search.h"
#include "rnnt_attrs.h"


#ifdef DEBUG_INF
#include <sstream>
#include <chrono>
#include <ctime>
#endif //DEBUG_INF

#include <functional>
#include <iostream>
#include <future>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>


#define WAV_FILE "common_voice_cs_25695144_16.wav"

#define BPE_MODEL "80_bpe.model"

#ifdef DEBUG_INF
#define TN_FILE "rnnt_tn.ort"
#endif

#define TN_CNN_FILE "rnnt_tn_cnn.ort"
#define TN_LSTM_FILE "rnnt_tn_lstm.ort"
#define TN_DNN_FILE "rnnt_tn_dnn.ort"
#define PN_FILE "rnnt_pn.ort"
#define CN_FILE "rnnt_cn.ort"

#define BEAM_SIZE 3
#define STATE_BEAM 4.6f
#define EXPAND_BEAM 2.3f
#define MELS FBANK_TP_ROW_SIZE

enum INFERENCE_STATUS {
  E_OK = 0,
  E_RNNT_ATTRS = -1
};

enum REC_BUF_STATUS {
  E_PREP = 0,
  E_READY = 1,
  E_DONE = 2
};

struct locked_shared_buffer {
  std::promise<void> signal_ready;
  spr::inference::thr_queue data;
  std::atomic_int pass;
};

static void inference(spr::inference::rnnt_attrs *,
                      spr::inference::beam_search &,
                      const spr::inference::buf_size_t &,
                      std::string *);

static int rec_callback(buf_t *, size_t);

static std::promise<void> g_signal_exit;
static std::future<void> g_stop_recording;

static locked_shared_buffer g_buf_ready;

int _exitting = 0; // exitting handler for pulse

int
main(int argc,
     char *argv[]) {

#ifdef DEBUG_INF
  std::chrono::time_point<std::chrono::system_clock> start, end;
#endif //DEBUG_INF

  g_stop_recording = g_signal_exit.get_future();
  g_buf_ready.pass = E_PREP;

  // load models and determine some intermediate values need for signal processing
  auto *rnnt_attrs = new spr::inference::rnnt_attrs(
#ifdef DEBUG_INF
                                                    TN_FILE,
#endif
                                                    TN_CNN_FILE, TN_LSTM_FILE, TN_DNN_FILE, PN_FILE, CN_FILE, BPE_MODEL, -1);
  if(!rnnt_attrs->is_initialized()) {
    std::cerr << "Could not read any of RNN-T models." << std::endl;
    return E_RNNT_ATTRS;
  }

  // producent
  std::thread([&]() {
    //TODO mod the code so the buffer is continuously read from pulse

    /*
      if(start_recording(rec_callback) < 0) {
      std::cerr << "Failure in starting pulse" << std::endl;
      }
    */

    auto *wav = new spr::wavread();
    if(argc == 2) {
      wav->init(argv[1]);
    } else {
      wav->init(WAV_FILE);
    }

    if(!wav->prepare_to_read()) {
      std::cerr << "Could not read the WAV file." << std::endl;

      // making disebark possible + free resources
      g_signal_exit.set_value();
      wav->deallocate();
      delete wav;
    }

    auto * wav_content = new buf_t[wav->get_num_samples()];

    // create int16_t* buffer for features
    wav->read_data_to_int16(wav_content, wav->get_num_samples());

    // fill the global buffer
    auto *wav_sim = new spr::wav_buf_simulator(wav_content, wav->get_num_samples(), rec_callback);
    wav_sim->start_emitting();

    delete wav_sim;

    // making disembark possible + free resources
    g_signal_exit.set_value();
    delete []wav_content;
    wav->deallocate();
    delete wav;

  }).detach();

#ifdef DEBUG_INF
  start = std::chrono::system_clock::now();
#endif

  // consumer
  // run inference on a separate thread
  std::string part_result;
  std::ostringstream result;

  std::thread runner([&]() {

    spr::inference::beam_search searcher(rnnt_attrs);

    auto start_infering = g_buf_ready.signal_ready.get_future();
    start_infering.wait();

    int readyp;
    while((readyp = g_buf_ready.pass.load()) != E_DONE || !g_buf_ready.data.empty()) {
      inference(rnnt_attrs, searcher, g_buf_ready.data.pop(), &part_result);
    }

    result << part_result;
  });
  runner.join();


  // stop detached recording
  _exitting = 1;
  g_stop_recording.wait();

  // print the results
#ifdef DEBUG_INF
  end = std::chrono::system_clock::now();
#endif

  std::ostringstream  oss;
  oss << result.str();

#ifdef DEBUG_INF
  oss << " (" <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count() <<
    "[ms])";
#endif //DEBUG_INF

  std::cout << oss.str() << std::endl;

  return E_OK;
}

void inference(spr::inference::rnnt_attrs *rnnt_attrs,
               spr::inference::beam_search &searcher,
               const spr::inference::buf_size_t &buffer, std::string *result) {

  //  std::cout << "[" << buffer.size() << "]\n";

  float *feat_inp = nullptr;
  size_t feats_num = spr::inference::create_feat_inp(buffer.data(), buffer.size(), &feat_inp);
  spr::inference::norm_inp(feat_inp, feats_num);

  // this is very important: you need to project the matrix size into the attributes
  rnnt_attrs->reset_buffer_win_len((int64_t)feats_num);

  *result = searcher.decode(feat_inp);

  delete []feat_inp;
}

int rec_callback(buf_t *data, size_t size) {

  if(size) {
    //    std::cout << "push: " << size << "(" << g_buf_ready.data.size() << ")\n";
    g_buf_ready.data.push(std::vector<buf_t>(data, data+size));
    if(g_buf_ready.pass.exchange(E_READY) == E_PREP) {
      g_buf_ready.signal_ready.set_value();
    }
    //    std::cout << "push then: " << size << "(" << g_buf_ready.data.size() << ")\n";
  } else {
    //    std::cout << "push: DONE\n";
    g_buf_ready.pass.store(E_DONE);
  }

  return size;
}
//eof

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

#include "common/buf_type.h"
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
#include <thread>


#define WAV_FILE "common_voice_cs_25695144_16.wav"

#define BPE_MODEL "80_bpe.model"
#define TN_FILE "rnnt_tn.ort"
#define PN_FILE "rnnt_pn.ort"
#define CN_FILE "rnnt_cn.ort"

#define BEAM_SIZE 3
#define STATE_BEAM 4.6f
#define EXPAND_BEAM 2.3f
#define MELS FBANK_TP_ROW_SIZE


enum INFERENCE_STATUS {
  E_OK = 0,
  E_RNNT_ATTRS = -1,
  E_WAV = -2
};

using buf_size_t = std::vector<buf_t>;

static void inference(spr::inference::rnnt_attrs *, const buf_size_t &, std::string *);
static void dec_callback(const std::string &);
static int rec_callback(buf_t *, size_t);

static std::promise<buf_size_t> signal_buf_ready;
static std::future<buf_size_t> process_rec_buffer;

static std::promise<void> signal_exit;
static std::future<void> stop_recording;

int _exitting = 0; // exitting handler for pulse

int
main(int argc,
     char *argv[]) {

#ifdef DEBUG_INF
  std::chrono::time_point<std::chrono::system_clock> start, end;
#endif //DEBUG_INF

  stop_recording = signal_exit.get_future();
  process_rec_buffer = signal_buf_ready.get_future();

  // load models and determine some intermediate values need for signal processing
  auto *rnnt_attrs = new spr::inference::rnnt_attrs(TN_FILE, PN_FILE, CN_FILE, BPE_MODEL, -1);
  if(!rnnt_attrs->is_initialized()) {
    std::cerr << "Could not read any of RNN-T models." << std::endl;
    return E_RNNT_ATTRS;
  }

  // producent
  std::thread([&]() {
    //TODO mod the code so the buffer is continuously read from pulse
    
    auto *wav = new spr::wavread();
    if(argc == 2) {
      wav->init(argv[1]);
    } else {
      wav->init(WAV_FILE);
    }
    
    if(!wav->prepare_to_read()) {
      std::cerr << "Could not read the WAV file." << std::endl;

      // making disebark possible + free resources
      signal_exit.set_value();
      wav->deallocate();
      delete wav;
    }
    
    auto * wav_content = new buf_t[wav->get_num_samples()];
    
    // create int16_t* buffer for features
    wav->read_data_to_int16(wav_content, wav->get_num_samples());

    // fill the global buffer
    rec_callback(wav_content, wav->get_num_samples());

    // making disembark possible + free resources
    signal_exit.set_value();
    delete []wav_content;
    wav->deallocate();
    delete wav;

  }).detach();
  
#ifdef DEBUG_INF
  start = std::chrono::system_clock::now();
#endif

  // consumer
  // run inference on a separate thread
  std::string result;

  std::thread runner([&]() {

    auto data = process_rec_buffer.get();

    inference(rnnt_attrs, data, &result);
  });
    //  std::thread runner(inference, rnnt_attrs, wav_content, wav->get_num_samples(), &result);
  runner.join();


  // stop detached recording
  _exitting = 1;
  stop_recording.wait();

  // print the results
#ifdef DEBUG_INF
  end = std::chrono::system_clock::now();
#endif

  std::ostringstream  oss;
  oss << "\r" << result;

#ifdef DEBUG_INF
  oss << " (" <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count() <<
    "[ms])";
#endif //DEBUG_INF

  std::cout << oss.str() << std::endl;

  return 0;
}

void inference(spr::inference::rnnt_attrs *rnnt_attrs, const buf_size_t &buffer, std::string *result) {

  float *feat_inp = nullptr;
  size_t feats_num = spr::inference::create_feat_inp(buffer.data(), buffer.size(), &feat_inp);
  spr::inference::norm_inp(feat_inp, feats_num);

  // this is very important: you need to project the matrix size into the attributes
  rnnt_attrs->reset_buffer_win_len((int64_t)feats_num);


  // result
  spr::inference::beam_search searcher(rnnt_attrs);

  auto dec_callback_fn = std::bind(dec_callback, std::placeholders::_1);

  *result = searcher.decode(feat_inp, dec_callback_fn);

  delete []feat_inp;
}

void dec_callback(const std::string &part_hyp) {
  /// not implemented (yet)
  std::cout << "\r" << part_hyp << std::flush;
}

int rec_callback(buf_t *data, size_t size) {

  signal_buf_ready.set_value(std::vector<buf_t>(data, data+size));
  
  return size;
}
//eof

/*** Created by Kodo in '22 **/

/** Legend:
 * PN/pn ~ prediction network
 * TN/tn ~ transcription network
 * CN/cn ~ classifier network (linearization of the joint PN&TN)
 * BPE ~ byte-pairs
 */

#include "common/buf_type.h"
#include "input_processing.h" // transforms wav stream into normalized features
#include "model_structs.h"
#include "beam_search.h"
#include "rnnt_attrs.h"
#include "wavread.h"


//#ifdef DEBUG_INF
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>
//#endif

#include <thread>
#include <functional>

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


static void inference(spr::inference::rnnt_attrs *, const buf_t *, size_t, std::string *);
static void callback(const std::string &);

int
main(int argc,
     char *argv[]) {

  //#ifdef DEBUG_INF
  std::chrono::time_point<std::chrono::system_clock> start, end;
  //#endif

  auto *wav = new spr::wavread();
  if(argc == 2) {
    wav->init(argv[1]);
  } else {
    wav->init(WAV_FILE);
  }

  auto *rnnt_attrs = new spr::inference::rnnt_attrs(TN_FILE, PN_FILE, CN_FILE, BPE_MODEL, -1);
  if(!rnnt_attrs->is_initialized()) {
    std::cerr << "Could not read any of RNN-T models." << std::endl;
    return E_RNNT_ATTRS;
  }


  if(!wav->prepare_to_read()) {
    std::cerr << "Could not read the WAV file." << std::endl;
    return E_WAV;
  }

  auto * wav_content = new buf_t[wav->get_num_samples()];

  // create int16_t* buffer for features
  wav->read_data_to_int16(wav_content, wav->get_num_samples());

  start = std::chrono::system_clock::now();

  // run inference on a separate thread
  std::string result;
  //  std::thread runner(inference, rnnt_attrs, wav_content, wav->get_num_samples(), result);
  std::thread runner(inference, rnnt_attrs, wav_content, wav->get_num_samples(), &result);
  runner.join();

  end = std::chrono::system_clock::now();

  std::ostringstream  oss;
  oss << "\r" << result << " (" <<
    std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count() <<
    "[ms])";
  std::cout << oss.str() << std::endl;


  delete []wav_content;

  wav->deallocate();
  delete wav;

  return 0;
}

void inference(spr::inference::rnnt_attrs *rnnt_attrs, const buf_t *buffer, size_t buffer_len, std::string *result) {
  
  float *feat_inp = nullptr;
  size_t feats_num = spr::inference::create_feat_inp(buffer, buffer_len, &feat_inp);
  spr::inference::norm_inp(feat_inp, feats_num);
  
  // this is very important: you need to project the matrix size into the attributes
  rnnt_attrs->reset_buffer_win_len((int64_t)feats_num);
  
  
  // result
  spr::inference::beam_search searcher(rnnt_attrs);

  auto callback_fn = std::bind(callback, std::placeholders::_1);
  
  *result = searcher.decode(feat_inp, callback_fn);

  delete []feat_inp;
}

void callback(const std::string &part_hyp) {
  /// not implemented (yet)
  std::cout << "\r" << part_hyp << std::flush;
}

//eof

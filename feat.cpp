#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>
#include "feat.h"
#include "bit_utils.h"

#include "fmadd_neon.h"

#include "stfft.h"
#include "fbank_80_201.h"


namespace spr::feat {

  const float Fbank::MATLAB_pi = 3.141592653589793;

  Fbank::Fbank(int sample_rate, int n_mels, int n_fft) :
    m_n_fft(n_fft),
    m_sr(sample_rate),
    m_process(nullptr) {
    init();
  }

  void Fbank::init() {
    int i;
    float tmp = 0;
    /// stft and mag params for later calculations
    m_params.fft_len = m_n_fft;
    m_params.hamming_en = true;
    m_params.mode = spr::stfft::FFT_MODE_E::STAND;
    ///info: 1 frame ~ 1/16[ms]
    m_params.win_len = m_n_fft;  //25ms ~ 16kHz/1000ms * 25ms [frames]
    m_params.win_shift = m_sr / 100;   //10ms ~ 16kHz/1000ms * 10ms [frames]
  }


  Fbank::~Fbank() {
    if(m_process)
      delete m_process;
  }

  size_t Fbank::alter_features_matrix_size(size_t buf_length) {

    ///info: wav frames - fft_len / win_shift_frames + 1
    ///info: (57600 - 400) / 160 + 1 ~ 359 number of frequency bins example
    m_params.win_inc = (size_t) std::ceil((float)buf_length / (float)m_params.win_shift);
    m_params.pad_size = m_params.win_inc * m_params.win_shift - buf_length;

    m_params.pad_size = m_params.fft_len / 2;
    m_params.in_size = buf_length + m_params.pad_size * 2;

    m_params.in = new float[m_params.in_size];
    m_altering_buf_length = buf_length;

#ifdef DEBUG_FEAT
    std::cout << "padding signal from left: [0 .. " << m_params.pad_size << "]\n";
    std::cout << "padding signal from right: [" << m_params.pad_size + buf_length << " .. " << m_params.in_size << "]\n";
#endif

    for(uint32_t i = 0; i < m_params.pad_size; m_params.in[i++] = 0.0f);
    for(uint32_t i = m_params.pad_size + buf_length; i < m_params.in_size; m_params.in[i++] = 0.0f);

    m_process = new spr::stfft::stfft(&m_params);

    return m_params.win_inc * FBANK_TP_ROW_SIZE;
  }

  int Fbank::compute_features(const buf_t *in, float *out) {

    // internally alters this.params' the way it calculates this.params.mag
    int result;
    if( (result = stft(in)) != spr::stfft::E_OK ) {
      std::cerr << "compute_features error: " << spr::stfft::get_message_from_error_code(result) << std::endl;
      return E_STFT_MAG;
    }

    if( (result = multfq(out)) != E_OK ) {
      std::cerr << "compute_features multfq: error: " << result << std::endl;
      return E_FILTER_BANK_MUL;
    }


    if( (result = amp2db(out)) != E_OK ) {
      std::cerr << "compute_features amp2db: error: " << result << std::endl;
      return E_AMP2DB;
    }


#ifdef DEBUG_FEAT
    std::cout << "\nfbank: samples = [\n";
    //    int row = 0;
    for(int i = 28640; i < 28720; ++i) {
      std::cout << out[i] << (i < 28720 -1? ", " : "]\n\n");
    }
#endif


    return E_OK;
  }

  int Fbank::stft(const buf_t *in) {

    ///info: wav frames - fft_len / win_shift_frames + 1
    ///info: (57600 - 400) / 160 + 1 ~ 359 number of frequency bins example

    SAMPLE_LOCALS;
    int dummy = 0;


#ifdef DEBUG_FEAT
    std::cout << "stft: in_size: " << m_params.in_size << "; pad_size: " << m_params.pad_size << std::endl;
    std::cout << "stft: out_size: " << m_params.out_size << "; mag_size: " << m_params.mag_size << std::endl;
#endif

    for(int n=0; n < m_params.in_size - (m_params.pad_size * 2); ++n) {
      sample_aligned_t sample = SIGNED_16BIT_TO_SAMPLE(in[n], dummy);
      m_params.in[n+m_params.pad_size] = SAMPLE_TO_FLOAT_32BIT(sample, dummy);
    }

#ifdef DEBUG_FEAT
    std::ostringstream ossm;
    ossm << "\nin: 20 samples = [\n";
    for(int i=m_params.pad_size; i < m_params.pad_size + 20; ++i) {
      //      ossm << m_params.mag[i] << ((m_params.mag_size - 1 > i) ? ", " : "]");
      ossm << m_params.in[i] << ((m_params.pad_size + 20 - 1 > i) ? ", " : "\n...]");
    }
    std::cout << ossm.str() << std::endl;

    ossm.str("");

    ossm << "\nin: 20 samples = [...\n";
    for(int i=m_params.in_size - 21 - m_params.pad_size; i < m_params.in_size - m_params.pad_size; ++i) {
      ossm << m_params.in[i] << ((m_params.in_size - m_params.pad_size - 1 > i) ? ", " : "\n]");
    }
    std::cout << ossm.str() << "\n\n";
#endif

    int result = m_process->compute();
    (void)result;

#ifdef DEBUG_FEAT
    std::ostringstream osst;
    osst << "\nstft: 20 samples = [\n";
    for(int i=0; i < 20; ++i) {
      //      osst << m_params.mag[i] << ((m_params.mag_size - 1 > i) ? ", " : "]");
      osst << "[" << m_params.out[i] << ", " << m_params.out[i+201] <<((20 - 1 > i) ? "]\n" : "]\n...]");
    }
    std::cout << osst.str() << std::endl;

    osst.str("");

    osst << "\nstft: 20 samples = [...\n";
    for(int i=201 - 21; i < 201; ++i) {
      osst << "[" << m_params.out[i] << ", " << m_params.out[i+201] << ((201 - 1 > i) ? "]\n" : "]\n]");
    }
    std::cout << osst.str() << "\n\n";
#endif

#ifdef DEBUG_FEAT
    ossm.str("");
    ossm << "\nmag: 20 samples = [\n";
    for(int i=0; i < 20; ++i) {
      //      ossm << m_params.mag[i] << ((m_params.mag_size - 1 > i) ? ", " : "]");
      ossm << m_params.mag[i] << ((20 - 1 > i) ? ", " : "\n...]");
    }
    std::cout << ossm.str() << std::endl;

    ossm.str("");

    ossm << "\nmag: 20 samples = [...\n";
    for(int i=201 - 21; i < 201; ++i) {
      ossm << m_params.mag[i] << ((201 - 1 > i) ? ", " : "\n]");
    }
    std::cout << ossm.str() << "\n\n";
#endif

    return result;
  }

  int Fbank::multfq(float *outmat) const {

    if((m_params.win_inc - 1) *  FBANK_TP_COL_SIZE > m_params.mag_size) {
      std::cerr << "multfq: mag-buffer size mismatch: " << m_params.win_inc * FBANK_TP_COL_SIZE << " > " << m_params.mag_size << std::endl;
      return E_MULTFQ_MAG_SIZE;
    }


    std::cout << "multfq: spec_row_size: " << m_params.win_inc << "; ROW: " << FBANK_TP_ROW_SIZE << "; COL: " << FBANK_TP_COL_SIZE << "\n";
    std::cout << "multfq: mag_size: " << m_params.mag_size << "\n\n";

    // jump along spectrum's rows
#pragma omp parallel for num_threads(THREADSIZE)
    for(uint32_t spec_i=0; spec_i < m_params.win_inc; ++spec_i) {

      // jump along final matrix's rows (columns are processed fmadd())
      for (uint32_t fi = 0; fi < FBANK_TP_ROW_SIZE; ++fi) {


#ifdef DEBUG_FEAT
        if(spec_i == 358 && fi == 0) {
          std::cout <<"multfq: first mag: [";
          for(int i = 0; i < FBANK_TP_COL_SIZE; ++i) {
            std::cout << m_params.mag[i + spec_i*FBANK_TP_COL_SIZE] << (i < FBANK_TP_COL_SIZE -1 ? ", " : "\n]\n");
          }

          std::cout << "multfq: first fbank: [";
          for(int i=0; i < FBANK_TP_COL_SIZE; ++i) {
            std::cout << FBANK_80_MELS_201_FFT_TP[i + fi*FBANK_TP_COL_SIZE] << (i < FBANK_TP_COL_SIZE -1 ? ", " : "\n]\n");
          }
        }
#endif

        outmat[spec_i * FBANK_TP_ROW_SIZE + fi] =
          // sum vectors product
          fmadd(m_params.mag + (spec_i * FBANK_TP_COL_SIZE),
                FBANK_80_MELS_201_FFT_TP + (fi * FBANK_TP_COL_SIZE),
                FBANK_TP_COL_SIZE);
      }
    }

    return E_OK;
  }


  int Fbank::amp2db(float *outmat) const {
    size_t outsize =  m_params.win_inc * FBANK_TP_ROW_SIZE;

    float amax = std::numeric_limits<float>::min();
    float curr;
    for(size_t i=0; i < outsize; ++i) {
      // do the conversion (with multiplier of 10)
      curr = 10.0f * std::log10(std::fmax(outmat[i], 1e-10));
      // memorize the max value
      amax = (curr > amax ? curr : amax);
      // save the non-clamped db value
      outmat[i] = curr;
    }

    for(size_t i=0; i < outsize; ++i) {
      // clamp to max value of `log value - top db value(80)`
      outmat[i] = std::fmax(outmat[i], amax - 80.0f);
    }

    return E_OK;
  }

}

#pragma once

#include <cmath>

#include "stfft.h"
#include "buf_type.h"


namespace spr::feat {

  constexpr int SAMPLE_RATE = 16000;
  constexpr int FFT_LEN = 400;
  

  enum FBANK_ERROR_CAUSE {
    E_OK = 0,
    E_STFT_MAG = -1,
    E_FILTER_BANK_MUL = -2,
    E_MULTFQ_MAG_SIZE = -3,
    E_MULTFQ_OUT_SIZE = -4

  };

  inline const char* get_message_from_error_code(int code) {
    switch (code) {
    case E_STFT_MAG:
      return "Computing features encountered error while stft/spec_mag calculation occurred";
    case E_FILTER_BANK_MUL:
      return "Computing features encountered error while multiplication with fbank occurred";
    default:
      return "Computing features encountered unknown error";
    }
  }

  class Fbank {
  public:

    Fbank(int sample_rate, int n_mels, int n_fft);

    ///info: hz <-> mel translation
    /*
      inline float to_mel(float freq_in_hz) {
      return 2595 * log10f(1 + freq_in_hz / 700);
      }

      inline float to_hz(float freq_in_mel) {
      return 700 * (powf(10, freq_in_mel / 2595) - 1);
      }
    */

    virtual ~Fbank();

    /** Sets win_inc parameter within params self variable
     *  It needs to be called prior to #compute_features()
     *
     *  @returns size of the overall features matrix for a buffer length
     */
    size_t alter_features_matrix_size(size_t length);

    /** Computes input for the model.
     *  The size of the `in' buffer
     *
     * @param in pcm buffer (often sampled within int16_t type) which size is set calling #alter_features_matrix_size() method prior to this call
     * @param out buffer serving as an input for the model. It is a spectrum of the pcm signal multiplied by the feature-matrix
     *
     * @returns
     */
    int compute_features(const buf_t* in, float *out);

  private:

    void init();

    int stft(const buf_t *in);

    /** Multiplies input spectrum and filter bank matrix
     * Fbank matrix is global, imported from within a header
     */
    int multfq(uint16_t spec_row_size, float *outmat) const;

  private:

    static const float MATLAB_pi; //= 3.141592653589793;

    float *m_hann{};
    //float *m_buf{};

    int m_win_len = 25; //number of frames
    int m_hop = 10;
    //int m_mels = 80;
    int m_n_fft = FFT_LEN;
    int m_sr = SAMPLE_RATE;

    spr::stfft::FFT_PARAM_S m_params;
    spr::stfft::stfft *m_process;
    size_t m_altering_buf_length;
  };
}

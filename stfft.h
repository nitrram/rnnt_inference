#pragma once

#include <cstdint>

namespace spr::stfft {

  enum FFT_MODE_E {
    STAND,
    INVERSE
  };

  enum FFT_ERROR_CAUSE {
    E_OK = 0,
    E_COMPUTE_OUT_BUF_SIZE = -2,
    E_COMPUTE_IN_BUF_SIZE = -1,
    E_COMPUTE_MAG_BUF_SIZE = -3
  };

  inline const char* get_message_from_error_code(int code) {

    switch (code) {
    case E_COMPUTE_OUT_BUF_SIZE:
      return "Computing STFT encountered error on output buffer size, which is less than index";
    case E_COMPUTE_IN_BUF_SIZE:
      return "Computing STFT encountered error on input buffer size, which is less than index";
    case E_COMPUTE_MAG_BUF_SIZE:
      return "Computing STFT encountered error on spectral magnitude buffer size, which is less than index";
    default:
      return "Computing STFT encountered unknown error";
    }
  }

  /**
   * @param hamming_en: manual
   * @param mode: manual
   * @param win_inc: manual
   * @param win_shift: manual
   * @param win_len: manual
   * @param fft_len: manual
   * @param pad_size: manual
   * @param other: auto
   */
  struct FFT_PARAM_S {
    bool hamming_en;
    FFT_MODE_E mode;
    uint32_t win_inc;
    uint32_t win_shift;
    uint32_t win_len;
    uint32_t fft_len;
    uint32_t in_size;
    uint32_t out_size;
    uint32_t mag_size;
    uint32_t pad_size;
    float *hamming;
    float *kernel;
    float *in;
    float *out;
    float *mag;
  public:
    FFT_PARAM_S() {
      hamming_en = false;
      mode = STAND;
      win_inc = 1;
      win_shift = 0;
      pad_size = 0;
    }
  };

  class stfft {
  public:
    stfft(FFT_PARAM_S *ffth_param);

    virtual ~stfft();

    int compute();

  private:
    FFT_PARAM_S *ffth_param;
    uint32_t win_inc;
    uint32_t win_shift;
    uint32_t win_len;
    uint32_t fft_len;
    uint32_t pad_len;
  };
} // namespace ffth_aarch64

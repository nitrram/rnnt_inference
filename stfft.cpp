#include "stfft.h"

#include "common/fmadd_neon.h"

#include <omp.h>

#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <cmath>


static float hamming_window_400[] = {
  0.0800, 0.0801, 0.0802, 0.0805, 0.0809, 0.0814, 0.0820, 0.0828, 0.0836,
  0.0846, 0.0857, 0.0868, 0.0881, 0.0896, 0.0911, 0.0927, 0.0945, 0.0963,
  0.0983, 0.1003, 0.1025, 0.1048, 0.1072, 0.1097, 0.1123, 0.1150, 0.1178,
  0.1208, 0.1238, 0.1269, 0.1301, 0.1335, 0.1369, 0.1404, 0.1441, 0.1478,
  0.1516, 0.1555, 0.1595, 0.1637, 0.1679, 0.1721, 0.1765, 0.1810, 0.1856,
  0.1902, 0.1949, 0.1998, 0.2047, 0.2097, 0.2147, 0.2199, 0.2251, 0.2304,
  0.2358, 0.2413, 0.2468, 0.2524, 0.2581, 0.2638, 0.2696, 0.2755, 0.2814,
  0.2874, 0.2935, 0.2997, 0.3058, 0.3121, 0.3184, 0.3248, 0.3312, 0.3376,
  0.3441, 0.3507, 0.3573, 0.3640, 0.3707, 0.3774, 0.3842, 0.3910, 0.3979,
  0.4047, 0.4117, 0.4186, 0.4256, 0.4326, 0.4397, 0.4467, 0.4538, 0.4609,
  0.4680, 0.4752, 0.4823, 0.4895, 0.4967, 0.5039, 0.5111, 0.5183, 0.5256,
  0.5328, 0.5400, 0.5472, 0.5544, 0.5617, 0.5689, 0.5761, 0.5833, 0.5905,
  0.5977, 0.6048, 0.6120, 0.6191, 0.6262, 0.6333, 0.6403, 0.6474, 0.6544,
  0.6614, 0.6683, 0.6753, 0.6821, 0.6890, 0.6958, 0.7026, 0.7093, 0.7160,
  0.7227, 0.7293, 0.7359, 0.7424, 0.7488, 0.7552, 0.7616, 0.7679, 0.7742,
  0.7803, 0.7865, 0.7926, 0.7986, 0.8045, 0.8104, 0.8162, 0.8219, 0.8276,
  0.8332, 0.8387, 0.8442, 0.8496, 0.8549, 0.8601, 0.8653, 0.8703, 0.8753,
  0.8802, 0.8851, 0.8898, 0.8944, 0.8990, 0.9035, 0.9079, 0.9121, 0.9163,
  0.9205, 0.9245, 0.9284, 0.9322, 0.9359, 0.9396, 0.9431, 0.9465, 0.9499,
  0.9531, 0.9562, 0.9592, 0.9622, 0.9650, 0.9677, 0.9703, 0.9728, 0.9752,
  0.9775, 0.9797, 0.9817, 0.9837, 0.9855, 0.9873, 0.9889, 0.9904, 0.9919,
  0.9932, 0.9943, 0.9954, 0.9964, 0.9972, 0.9980, 0.9986, 0.9991, 0.9995,
  0.9998, 0.9999, 1.0000, 0.9999, 0.9998, 0.9995, 0.9991, 0.9986, 0.9980,
  0.9972, 0.9964, 0.9954, 0.9943, 0.9932, 0.9919, 0.9904, 0.9889, 0.9873,
  0.9855, 0.9837, 0.9817, 0.9797, 0.9775, 0.9752, 0.9728, 0.9703, 0.9677,
  0.9650, 0.9622, 0.9592, 0.9562, 0.9531, 0.9499, 0.9465, 0.9431, 0.9396,
  0.9359, 0.9322, 0.9284, 0.9245, 0.9205, 0.9163, 0.9121, 0.9079, 0.9035,
  0.8990, 0.8944, 0.8898, 0.8851, 0.8802, 0.8753, 0.8703, 0.8653, 0.8601,
  0.8549, 0.8496, 0.8442, 0.8387, 0.8332, 0.8276, 0.8219, 0.8162, 0.8104,
  0.8045, 0.7986, 0.7926, 0.7865, 0.7803, 0.7742, 0.7679, 0.7616, 0.7552,
  0.7488, 0.7424, 0.7359, 0.7293, 0.7227, 0.7160, 0.7093, 0.7026, 0.6958,
  0.6890, 0.6821, 0.6753, 0.6683, 0.6614, 0.6544, 0.6474, 0.6403, 0.6333,
  0.6262, 0.6191, 0.6120, 0.6048, 0.5977, 0.5905, 0.5833, 0.5761, 0.5689,
  0.5617, 0.5544, 0.5472, 0.5400, 0.5328, 0.5256, 0.5183, 0.5111, 0.5039,
  0.4967, 0.4895, 0.4823, 0.4752, 0.4680, 0.4609, 0.4538, 0.4467, 0.4397,
  0.4326, 0.4256, 0.4186, 0.4117, 0.4047, 0.3979, 0.3910, 0.3842, 0.3774,
  0.3707, 0.3640, 0.3573, 0.3507, 0.3441, 0.3376, 0.3312, 0.3248, 0.3184,
  0.3121, 0.3058, 0.2997, 0.2935, 0.2874, 0.2814, 0.2755, 0.2696, 0.2638,
  0.2581, 0.2524, 0.2468, 0.2413, 0.2358, 0.2304, 0.2251, 0.2199, 0.2147,
  0.2097, 0.2047, 0.1998, 0.1949, 0.1902, 0.1856, 0.1810, 0.1765, 0.1721,
  0.1679, 0.1637, 0.1595, 0.1555, 0.1516, 0.1478, 0.1441, 0.1404, 0.1369,
  0.1335, 0.1301, 0.1269, 0.1238, 0.1208, 0.1178, 0.1150, 0.1123, 0.1097,
  0.1072, 0.1048, 0.1025, 0.1003, 0.0983, 0.0963, 0.0945, 0.0927, 0.0911,
  0.0896, 0.0881, 0.0868, 0.0857, 0.0846, 0.0836, 0.0828, 0.0820, 0.0814,
  0.0809, 0.0805, 0.0802, 0.0801
};



using hrcpoint_t = std::chrono::high_resolution_clock::time_point;

namespace spr::stfft {
  [[maybe_unused]] void debug_print(float *data, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i){
      std::cout << (i + 1) << ": " << data[i] << std::endl;

    }
  }

  [[maybe_unused]] inline hrcpoint_t now_() {
    return std::chrono::high_resolution_clock::now();
  }

  [[maybe_unused]] void time_ms(hrcpoint_t start_, hrcpoint_t end_) {
    std::chrono::duration<double, std::milli> out1 = end_ - start_;
    std::chrono::duration<double, std::micro> out2 = end_ - start_;
    std::cout << "Time: " << out1.count() << "(ms), " << out2.count() << "(us)\n";
  }

  /*
    stfft::stfft(FFT_PARAM_S *ffth_param) :
    stfft(ffth_param, (ffth_param->win_inc - 1) * ffth_param->win_shift + ffth_param->fft_len) {}
  */

  stfft::stfft(FFT_PARAM_S *ffth_param) {

    win_inc = ffth_param->win_inc;
    win_shift = ffth_param->win_shift;
    win_len = ffth_param->win_len;
    fft_len = ffth_param->fft_len;

    if (ffth_param->mode == STAND) {
      ffth_param->out_size = win_inc * (fft_len + 2);
      ffth_param->mag_size = ffth_param->out_size/2 + 1;
      ffth_param->kernel = new float[(fft_len + 2) * fft_len]();
      ffth_param->out = new float[ffth_param->out_size]();
      ffth_param->mag = new float[ffth_param->mag_size];


      uint32_t imag_start = (fft_len / 2 + 1) * fft_len;
      for (uint32_t i = 0; i < (fft_len / 2 + 1); ++i) {
        for (uint32_t j = 0; j < fft_len; ++j) {
          ffth_param->kernel[i * fft_len + j] = cosf(2.0f * (float)M_PI * (float)j * (float)i / (float)fft_len);
          ffth_param->kernel[i * fft_len + j + imag_start] = -sinf(2.0f * (float)M_PI * (float)j * (float)i / (float)fft_len);
        }
      }

      if (ffth_param->hamming_en) {
        if(win_len == 400) {
          ffth_param->hamming = hamming_window_400;
        } else {
          ffth_param->hamming = new float[fft_len]();
          for (uint32_t i = 0; i < win_len; ++i) {
            ffth_param->hamming[i + (fft_len - win_len) / 2] = 0.54f - 0.46f * cosf(2 * (float)M_PI * (float)i / (float)(win_len));
          }
        }

#ifdef DEBUG_STFT

        std::ostringstream oss;
        oss <<  std::fixed << std::setprecision(4) ;
        for(uint32_t i = 0; i < win_len; ++i) {
          oss << ffth_param->hamming[i] <<  (win_len - 1 > i ? ", " : "]\n\n");
          if(i && ((i+1) % 9) == 0) oss << "\n";
        }
        std::cout << oss.str() << std::endl;
#endif

        for (uint32_t i = 0; i < (fft_len + 2); ++i) {
          for (uint32_t j = 0; j < fft_len; ++j) {
            ffth_param->kernel[i * fft_len + j] *= ffth_param->hamming[j];
          }
        }
      }
    } else {
      ffth_param->in_size = win_inc * (fft_len + 2);
      ffth_param->out_size = (win_inc - 1) * win_shift + fft_len;
      ffth_param->kernel = new float[(fft_len + 2) * fft_len]();
      ffth_param->in = new float[ffth_param->in_size]();
      ffth_param->out = new float[ffth_param->out_size]();

      uint32_t tmp_1 = fft_len + 2;
      uint32_t tmp_2 = tmp_1 / 2;
      for (uint32_t i = 0; i < fft_len; ++i) {
        for (uint32_t j = 0; j < tmp_2; ++j) {
          ffth_param->kernel[i * tmp_1 + j] = cosf(2 * (float)M_PI * (float)j * (float)i / (float)fft_len) / (float)fft_len;
          ffth_param->kernel[i * tmp_1 + j + tmp_2] = -sinf(2 * (float)M_PI * (float)j * (float)i / (float)fft_len) / (float)fft_len;
        }
        for (uint32_t j = 0; j < (tmp_2 - 2); ++j) {
          ffth_param->kernel[i * tmp_1 + j + 1] += cosf(2 * (float)M_PI * (float)(fft_len - 1 - j) * (float)i / (float)fft_len) / (float)fft_len;
          ffth_param->kernel[i * tmp_1 + j + tmp_2 + 1] += sinf(2 * (float)M_PI * (float)(fft_len - 1 - j) * (float)i / (float)fft_len) / (float)fft_len;
        }
      }
    }

    this->ffth_param = ffth_param;
  }

  stfft::~stfft() {
    if(ffth_param->mode != STAND) {
      delete[]ffth_param->in;
      if(ffth_param->hamming_en) {
        delete[]ffth_param->hamming;
      }
    }

    delete[]ffth_param->kernel;
    delete[]ffth_param->out;
    delete[]ffth_param->mag;
  }

  int stfft::compute() {

    float *in = ffth_param->in;
    float *out = ffth_param->out;
    float *mag = ffth_param->mag;
    float *kernel = ffth_param->kernel;


    if (ffth_param->mode == STAND) {
      uint32_t out_w = (fft_len + 2);
      uint32_t size = win_inc * out_w;

      if(size > ffth_param->out_size) {
        std::cerr << "stfft error compute: out-buffer overflow: (size)" << size << " > (params)" << ffth_param->out_size << std::endl;;
        return E_COMPUTE_OUT_BUF_SIZE;
      }

      if( (win_shift * size / out_w) > ffth_param->in_size) {
        std::cout << "stfft err: win_shift: " << win_shift << "; win_inc: " << win_inc << "; size: " << size << "; out_w: " << out_w << std::endl;
        std::cerr << "stfft error compute: in-buffer overflow: (size)" << (win_shift * size/out_w) << " > (params)" << ffth_param->in_size << std::endl;
        return E_COMPUTE_IN_BUF_SIZE;
      }

#ifdef DEBUG_STFT
      std::cout << "\nfirst 20 stft samples:\n";
#endif

#pragma omp parallel for num_threads(THREADSIZE) default(none) shared(out_w, out, in, kernel)
      for (uint32_t idx = 0; idx < ffth_param->out_size; ++idx) {
        uint32_t i, j;
        i = idx / out_w;
        j = idx - i * out_w;

        //e.g. j ~ <201 = Re; j ~>=201 = Im
        //e.g. 1 window has 400 fs ~ 0..200 Re and 201..401 Im
        out[idx] = fmadd(in + i * win_shift, kernel + j * fft_len, fft_len);
      }

      if( (size >> 1) > ffth_param->mag_size) {
        std::cerr << "stfft error compute: mag-buffer overflow: (size)" << (size - 1) <<" * " <<
          (fft_len >> 1) << " > (params)" << ffth_param->mag_size << std::endl;
        return E_COMPUTE_MAG_BUF_SIZE;
      }

      // compute spectral magnitude out of the complex signal
      uint32_t imag_start = (fft_len >> 1) + 1;
#pragma omp parallel for num_threads(THREADSIZE) default(none) shared(size, out, imag_start, mag)
      for(uint32_t idx = 0; idx < size; idx += (fft_len + 2)) {
        //        std::cout << "mag comp idxs: [" << idx << ", " << (idx + imag_start) << ", " << (idx >> 1) << "]\n";
        fpowadd2in(out + idx, out + (idx + imag_start), mag + (idx >> 1), imag_start);
      }

      return E_OK;
    }

    memset(out, 0, ffth_param->out_size * sizeof(float));
    uint32_t in_w = fft_len + 2;
    for (uint32_t i = 0; i < win_inc; ++i) {
#pragma omp parallel for num_threads(THREADSIZE) default(none) shared(out, i, in, in_w, kernel)
      for (uint32_t j = 0; j < fft_len; ++j) {
        out[i * win_shift + j] += fmadd(in + i * in_w, kernel + j * in_w, in_w);
      }
    }
    return E_OK;
  }
} // ffth_aarch64

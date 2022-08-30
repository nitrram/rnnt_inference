#include <omp.h>

#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include "stfft.h"
#include "fmadd_neon.h"


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

    stfft::stfft(FFT_PARAM_S *ffth_param) {

        win_inc = ffth_param->win_inc;
        win_shift = ffth_param->win_shift;
        win_len = ffth_param->win_len;
        fft_len = ffth_param->fft_len;
        if (ffth_param->mode == STAND) {
            ffth_param->in_size = (win_inc - 1) * win_shift + fft_len;
            ffth_param->out_size = win_inc * (fft_len + 2);
            ffth_param->mag_size = ffth_param->out_size/2 + 1;
            ffth_param->kernel = new float[(fft_len + 2) * fft_len]();
            ffth_param->in = new float[ffth_param->in_size]();
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
                ffth_param->hamming = new float[fft_len]();
                for (uint32_t i = 0; i < win_len; ++i) {
                    ffth_param->hamming[i + (fft_len - win_len) / 2] = 0.54f - 0.46f * cosf(2 * (float)M_PI * (float)i / (float)(win_len - 1));
                }
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
        delete[]ffth_param->hamming;
        delete[]ffth_param->kernel;
        delete[]ffth_param->in;
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

            if( ((size - 1)) > ffth_param->out_size) {
              std::cerr << "stfft error compute: out-buffer overflow: (size)" << ((size -1) * out_w) << " > (params)" << ffth_param->out_size << std::endl;;
                return E_COMPUTE_OUT_BUF_SIZE;
            }

            if( (win_shift * size/out_w) > ffth_param->in_size) {
              std::cerr << "stfft error compute: in-buffer overflow: (size)" << (win_shift * size/out_w) << " > (params)" << ffth_param->in_size << std::endl;
                return E_COMPUTE_IN_BUF_SIZE;
            }

            #pragma omp parallel for num_threads(THREADSIZE)
            for (uint32_t idx = 0; idx < size; ++idx) {
                uint32_t i, j;
                i = idx / out_w;
                j = idx - i * out_w;

                //e.g. j ~ <201 = Re; j ~>=201 = Im
                //e.g. 1 window has 400 fs ~ 0..200 Re and 201..401 Im
                out[idx] = fmadd(in + i * win_shift, kernel + j * fft_len, fft_len);
            }


            if( ((size - 1) >> 1) > ffth_param->mag_size) {
              std::cerr << "stfft error compute: mag-buffer overflow: (size)" << (size - 1) <<" * " << (fft_len >> 1) << " > (params)" << ffth_param->mag_size << std::endl;
              return E_COMPUTE_MAG_BUF_SIZE;
            }

            // compute spectral magnitude out of the complex signal
            uint32_t imag_start = fft_len >> 1;
            for (uint32_t idx = 0; idx < (size - fft_len); idx += fft_len) {
                fpowadd2in(out + idx, out + (idx + imag_start), mag + (idx >> 1), imag_start);
            }

            return E_OK;
        }
        
            memset(out, 0, ffth_param->out_size * sizeof(float));
        uint32_t in_w = fft_len + 2;
        for (uint32_t i = 0; i < win_inc; ++i) {
            #pragma omp parallel for num_threads(THREADSIZE)
            for (uint32_t j = 0; j < fft_len; ++j) {
                out[i * win_shift + j] += fmadd(in + i * in_w, kernel + j * in_w, in_w);
            }
        }
        return E_OK;
    }
} // ffth_aarch64

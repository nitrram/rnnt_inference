#include <cmath>
#include <iostream>
#include <sstream>
#include "feat.h"
#include "bit_utils.h"

#include "fmadd_neon.h"

#include "stfft.h"
#include "fbank_80_201.h"


namespace spr::feat {

    const float Fbank::MATLAB_pi = 3.141592653589793;

    Fbank::Fbank() {
        init();
    }

    Fbank::Fbank(int sample_rate, int n_mels, int n_fft) :
            //m_mels(n_mels),
            m_n_fft(n_fft),
            m_sr(sample_rate) {
        init();
    }

    Fbank::~Fbank() {
        delete[] m_hann;
    }

    void Fbank::init() {
        //m_buf = new float[m_n_fft];
        m_hann = new float[m_win_len];

        int i;
        float tmp = 0;

        // win = hann(frame_size,'periodic');
        for (i = 0; i < m_win_len; i++)
            m_hann[i] = 0.5f * (1.0f - cosf(2.0f * MATLAB_pi * ((float) i / (float) m_win_len)));

        // win = win./sqrt(sum(win.^2)/shift_size);
        for (i = 0; i < m_win_len; i++)
            tmp += m_hann[i] * m_hann[i];
        tmp /= (float) m_hop;
        tmp = std::sqrt(tmp);

        for (i = 0; i < m_win_len; i++)
            m_hann[i] /= tmp;

        /// stft and mag params for later calculations

        m_params.fft_len = m_n_fft;
        m_params.hamming_en = true;
        m_params.mode = spr::stfft::FFT_MODE_E::STAND;
        ///info: 1 frame ~ 1/16[ms]
        m_params.win_len = m_n_fft;  //25ms ~ 16kHz/1000ms * 25ms [frames]
        m_params.win_shift = m_sr / 100;   //10ms ~ 16kHz/1000ms * 10ms [frames]
    }

    size_t Fbank::alter_features_matrix_size(size_t buf_length) {

        m_altering_buf_length = buf_length;

        ///info: wav frames - fft_len / win_shift_frames + 1
        ///info: (57600 - 400) / 160 + 1 ~ 359 number of frequency bins example
        m_params.win_inc = (m_altering_buf_length - m_params.fft_len) / m_params.win_shift + 1;

        return m_params.win_inc * FBANK_TP_COL_SIZE + FBANK_TP_COL_SIZE;
    }

    int Fbank::compute_features(const buf_t *in, float *out) {

        // internally alters this.params' the way it calculates this.params.mag
        int result;
        if( (result = stft(in)) != spr::stfft::E_OK ) {
          std::cerr << "compute_features error: " << spr::stfft::get_message_from_error_code(result) << std::endl;
            return E_STFT_MAG;
        }


        if( (result = multfq(m_params.mag_size / (FBANK_TP_COL_SIZE - 1), out)) != E_OK ) {
          std::cerr << "compute_features multfq: error: " << result << std::endl;
            return E_FILTER_BANK_MUL;
        }

        return E_OK;
    }

    int Fbank::stft(const buf_t *in) {

        ///info: wav frames - fft_len / win_shift_frames + 1
        ///info: (57600 - 400) / 160 + 1 ~ 359 number of frequency bins example

        spr::stfft::stfft process(&m_params);

        SAMPLE_LOCALS;
        int dummy = 0;

        for(int n=0; n < m_params.in_size; ++n) {
            sample_aligned_t sample = SIGNED_16BIT_TO_SAMPLE(in[n], dummy);
            m_params.in[n] = SAMPLE_TO_FLOAT_32BIT(sample, dummy);

            auto test = FLOAT_32BIT_TO_SAMPLE(m_params.in[n], dummy);

        }

        int result = process.compute();
        (void)result;

        return result;
    }

    int Fbank::multfq(uint16_t spec_row_size, float *outmat) const {


        if((spec_row_size - 1) *  FBANK_TP_COL_SIZE > m_params.mag_size) {
          std::cerr << "multfq: mag-buffer size mismatch: " << spec_row_size * FBANK_TP_COL_SIZE << " > " << m_params.mag_size << std::endl;
            return E_MULTFQ_MAG_SIZE;
        }

        if( (spec_row_size - 1) > m_params.win_inc + 1 ) {
          std::cerr << "multfq: out-buffer size mismatch: " <<  spec_row_size << " > " <<  m_params.win_inc << std::endl;
            return E_MULTFQ_OUT_SIZE;
        }

        // jump along spectrum's rows
        #pragma omp parallel for num_threads(THREADSIZE)
        for(uint32_t spec_i=0; spec_i < spec_row_size - 1; ++spec_i) {

            // jump along final matrix's rows (columns are processed fmadd())
            for (uint32_t fi = 0; fi < FBANK_TP_ROW_SIZE; ++fi) {

                outmat[spec_i * FBANK_TP_COL_SIZE + fi] =
                        // sum vectors product
                        fmadd(m_params.mag + (spec_i * FBANK_TP_COL_SIZE),
                              FBANK_80_MELS_201_FFT_TP + (fi * FBANK_TP_COL_SIZE),
                              FBANK_TP_COL_SIZE);
            }
        }

        return E_OK;
    }
}

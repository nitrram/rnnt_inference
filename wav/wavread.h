#pragma once

#include <cstdio> //For FILE
#include <cstdint> //For uint8_t, etc.

#include "wavcommon.h"


namespace spr {
    class wavread {

    public:
        wavread();

        virtual ~wavread();

        bool init(const char *file_path);

        bool prepare_to_read();

        bool read_data(uint8_t sample_data[], //WAV format bytes
                       uint32_t num_samples);

        bool read_data_to_int16(
                int16_t sample_data[], //channels interleaved; length = numInt16Samples * numChannels
                uint32_t num_samples);

        bool deallocate();

        //Read int16 sample from an in-memory array of wav-format sample data
        bool read_int16_from_data(const uint8_t sample_data[], //wav-format sample data
                                  uint32_t num_samples,
                                  uint32_t sample_index,
                                  int16_t &channel1,
                                  int16_t &chanel2);

        const char *get_path();

        uint32_t get_sample_rate();

        uint32_t get_num_samples();

        uint32_t get_channels_num();

        bool are_samples_ints();

        uint32_t get_byte_depth();

        uint32_t get_sample_data_size();

    private:
        bool read_metadata();

        bool open_file();

        bool close_file();

        bool close_file(const char *error_msg);

        bool find_subchunk(const char *id, uint32_t *size);

    private:
        char *m_file_path;
        FILE *m_file;

        //Metadata
        uint32_t m_sample_rate;
        uint32_t m_num_samples;
        uint32_t m_num_channels;
        bool m_are_samples_ints; //False if samples are 32 or 64-bit floating point values
        uint32_t m_byte_depth; //Number of significant bytes required to represent a single channel of a sample
        uint32_t m_sample_data_size;
        bool m_is_initialized;
    };
}
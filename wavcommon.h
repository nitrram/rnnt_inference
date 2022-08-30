#pragma once

#include <cstdint>
#include <limits.h> //CHAR_BIT

namespace spr {
//Verify we have float32 and float64
    static char static_assert_float32[
            1 - (2 * ((sizeof(float) * CHAR_BIT) != 32))]; //To ensure float is 32 bits
    static char static_assert_float64[
            1 - (2 * ((sizeof(double) * CHAR_BIT) != 64))]; //To ensure double is 64 bits

    typedef struct {
        char id[4];             //"riff"
        uint32_t file_size_less8;
        char name[4];          //"*.wav"
    } riff_t;
    const uint32_t RIFF_HEADER_SIZE = sizeof(riff_t);

    typedef struct {
        char id[4];
        uint32_t size;           //Number of bytes in subchunk, following this field
    } subchunk_t;
    const uint32_t SUBCHUNK_HEADER_SIZE = sizeof(subchunk_t);  //Total size of subchunk

    typedef struct {
        char id[4];    //"fmt "
        uint32_t size;     //Number of bytes following this field
        uint16_t audio_format;            //1 for PCM, 3 for Float
        uint16_t num_channels;            //1 or 2
        uint32_t sample_rate;             //8000, 44100, etc.
        uint32_t byte_rate;               // == sampleRate * numChannels * (bitsPerSample / 8)
        uint16_t block_align;             //Bytes per sample, including all channels
        uint16_t bits_per_sample;          //Bits per sample, for one channel
    } formsubchunk_t;
    const uint32_t FORMAT_SUBCHUNK_SIZE = sizeof(formsubchunk_t);  //Total size of format subchunk

    const uint8_t AUDIO_FORMAT_FLOAT = 3;
    const uint8_t AUDIO_FORMAT_INT = 1;

//Supposedly required for IEEE floating-point PCM format; see:
// http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    typedef struct {
        char id[4];  //"fact"
        uint32_t size;  //Number of bytes following this field
        uint32_t num_samp_per_channel;
    } factsubchunk_t;
    const uint32_t FACT_SUBCHUNK_SIZE = sizeof(factsubchunk_t); //Total size of fact subchunk

}
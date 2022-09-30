#include <cstring> //memset()
# include <cstdlib>
#include <cstdio>

#include <iostream>

#include "wavread.h"

namespace spr {

  static const uint16_t TWO_POW_7_AS_UINT16 = 128;
  static const uint16_t TWO_POW_8_AS_UINT16 = 256;
  static const uint32_t TWO_POW_8_AS_UINT32 = 256;
  static const float TWO_POW_15_LESS1_AS_FLOAT32 = 32768.0f - 1.0f;
  static const double TWO_POW_15_LESS1_AS_FLOAT64 = 32768.0 - 1.0;
  static const uint32_t TWO_POW_16_AS_UINT32 = 65536;
  static const char *UNINITIALIZED_MSG = "Attempt to call wavread class method before calling initialize().\n";

  wavread::wavread() {
    m_is_initialized = false;
  }

  wavread::~wavread() {
  }

  bool wavread::init(const char *file_path) {
    //Test for file existence...
    FILE *f = fopen(file_path, "r");
    if (!f) {
      std::cerr << "File: " << file_path << " doesn't exist.\n";
      return false;
    }
    fclose(f);

    //Set member vars
    m_file_path = (char *) file_path;
    m_file = nullptr;

    m_is_initialized = true; //Set *before* call to readMetadata()
    bool verifies = read_metadata(); //Sets remaining member variables
    m_is_initialized = verifies; //Update *after* call to readMetadata()

    return verifies;
  }

  bool wavread::open_file() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    if (!m_file) {
      m_file = fopen(m_file_path, "rb");
      if (m_file == NULL) {
        std::cerr << "Error: Unable to open input file for reading.\n";
        return false;
      }
    } else {
      rewind(m_file);
    }

    return true;
  }

  bool wavread::close_file() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    return close_file(nullptr);
  }

  bool wavread::close_file(const char *error_msg) {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
          
      return false;
    }

    if (error_msg) {
      std::cerr << error_msg << std::endl;

    }

    if (m_file) {
      fclose(m_file);
      m_file = nullptr;
    }

    return true;
  }

  bool wavread::find_subchunk(const char *id, uint32_t *size) {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;

      return false;
    }

    if (!open_file()) {
      return false;
    }

    //Skip over RIFF header
    if (fseek(m_file, RIFF_HEADER_SIZE, SEEK_CUR)) {
      close_file("Error: Problem while skipping over RIFF header.\n");
      return false;
    }

    while (true) {

      size_t numToRead = 1;
      size_t numRead = 0;
      uint8_t subchunkHeaderData[SUBCHUNK_HEADER_SIZE];
      numRead = fread(subchunkHeaderData, SUBCHUNK_HEADER_SIZE, 1, m_file);
      if (numRead < numToRead) {
        if (feof(m_file)) {
          std::cerr << "Error: Reached end of file without finding subchunk: " << id << std::endl;
          close_file();
          return false;
        }
        std::cerr << "Error: Problem reading subchunk: " << id << std::endl;
        close_file();
        return false;
      }

      auto *sch = reinterpret_cast<subchunk_t *>(subchunkHeaderData);
      bool scfound = !strncmp(sch->id, id, 4);
      if (scfound) {
        //Set size to pass back
        *size = sch->size;
        //Rewind to the beginning of the subchunk, i.e. including the header
        if (fseek(m_file, -(int) SUBCHUNK_HEADER_SIZE, SEEK_CUR)) {
          std::cerr <<  "Error: Problem advancing to subchunk: " << id << std::endl;
          close_file();
          return false;
        }
        return true;
      }

      //Subchunk not found; advance to next subchunk
      if (fseek(m_file, sch->size, SEEK_CUR)) {
        if (feof(m_file)) {
          std::cerr << "Error: End of file reached without finding subchunk: " << id << std::endl;
          close_file();
          return false;
        } else {
          close_file("Error: Problem while advancing to the next subchunk");
          return false;
        }
      }
    }

    return false;
  }

  bool wavread::read_metadata() {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    if (!open_file()) {
      std::cerr << "Error: Unable to open file to read metadata.\n";
      return false;
    }

    //Read riff header
    uint8_t riffHeaderData[RIFF_HEADER_SIZE];
    size_t numToRead = 1;
    size_t numRead = 0;
    numRead = fread(riffHeaderData, RIFF_HEADER_SIZE, 1, m_file);
    if (numRead < numToRead) {
      close_file("Error: Problem reading RIFF header.");
      return false;
    }
    auto *rh = reinterpret_cast<riff_t *>(riffHeaderData);
    if (strncmp(rh->id, "RIFF", 4) != 0) {
      close_file("Error: RIFF header not included at start.");
      return false;
    }

    //Read format subchunk
    uint32_t subchunkSize = 0;
    if (!find_subchunk("fmt ", &subchunkSize)) {
      close_file("Error: Unable find 'fmt ' subchunk.");
      return false;
    }
    uint8_t formatSubchunkData[FORMAT_SUBCHUNK_SIZE];
    numToRead = 1;
    numRead = 0;
    numRead = fread(formatSubchunkData, FORMAT_SUBCHUNK_SIZE, 1, m_file);
    if (numRead < numToRead) {
      close_file("Error: Problem reading format subchunk.");
      return false;
    }
    auto *fsc = reinterpret_cast<formsubchunk_t *>(formatSubchunkData);

    //Parse format subchunk
    if (strncmp(fsc->id, "fmt ", 4) != 0) {
      close_file("Error: 'fmt ' field not found.");
      return false;
    }

    if (fsc->audio_format == AUDIO_FORMAT_INT) {
      m_are_samples_ints = true;
    } else if (fsc->audio_format == AUDIO_FORMAT_FLOAT) {
      m_are_samples_ints = false;
    } else {
      close_file("Error: Audio format must be WAVE_FORMAT_PCM or WAVE_FORMAT_IEEE_FLOAT.");
      return false;
    }

    m_num_channels = fsc->num_channels;
    if (!(m_num_channels == 1 || m_num_channels == 2)) {
      close_file("Error: Number of channels must be 1 or 2");
      return false;
    }

    m_sample_rate = fsc->sample_rate;
    if (m_sample_rate < 8000) { // Other constraints?
      close_file("Error: Unsupported sample rate.");
      return false;
    }

    m_byte_depth = fsc->bits_per_sample / 8;
    if (!((m_are_samples_ints && (m_byte_depth == 1 || m_byte_depth == 2 || m_byte_depth == 3 ||
                                  m_byte_depth == 4)) ||
          (!m_are_samples_ints && (m_byte_depth == 4 || m_byte_depth == 8)))) {
      close_file(
                 "Error: Invalid bits-per-sample value, or invalid combination of bits-per-sample and number of channels.");
      return false;
    }

    subchunkSize = 0;
    if (!find_subchunk("data", &subchunkSize)) {
      close_file("Error: Data subchunk not found.");
      return false;
    }
    m_sample_data_size = subchunkSize;

    if (fsc->block_align != m_num_channels * m_byte_depth) {
      close_file("Error: block alignment doesn't match number of channels + bit depth.");
      return false;
    }

    m_num_samples = m_sample_data_size / fsc->block_align;

    return true;
  }

  bool wavread::prepare_to_read() {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    //Open file
    if (!open_file()) {
      close_file("Error: Unable to open file, while preparing to read data.");
      return false;
    }

    //Find data subchunk
    uint32_t subchunkSize = 0;
    if (!find_subchunk("data", &subchunkSize)) {
      close_file("Error: Unable to find data subchunk, while preparing to read data.");
      return false;
    }

    //Advance past data subchunk header, to the sample data
    if (fseek(m_file, SUBCHUNK_HEADER_SIZE, SEEK_CUR)) {
      close_file("Error: Unable to advance past data subchunk header.\n");
      return false;
    }

    return true;
  }

  bool wavread::read_data(uint8_t *sample_data, uint32_t num_samples) {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    if (this->m_sample_data_size < num_samples) {
      close_file("Error: Suppled sampleDataSize larger than available data");
      return false;
    }

    if (m_sample_data_size % (m_byte_depth * m_num_channels) > 0) {
      close_file("Error: Suppled sampleDataSize doesn't fall evenly on a sample boundary.");
      return false;
    }

    size_t numToRead = 1;
    size_t numRead = 0;
    numRead = fread((char *) sample_data, 1, num_samples, m_file);
    if (numRead < numToRead) {
      if (feof(m_file)) {
        close_file("Error: Reached end of file while reading data");
        return false;
      }
      close_file("Error: Problem reading data");
      return false;
    }

    return true;
  }

  bool wavread::read_data_to_int16(int16_t *sample_data, uint32_t num_samples) {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    const uint32_t numBytesToRequest = num_samples * m_num_channels * m_byte_depth;
    if (numBytesToRequest > m_sample_data_size) {
      close_file("Error: Suppled numInt16Samples to large for available data");
      return false;
    }

    int16_t sampleCh1 = 0;
    int16_t sampleCh2 = 0;
    const uint32_t numBytes = (m_num_channels * m_byte_depth);
    uint8_t sampleBytes[numBytes];
    uint8_t tmp_byte;
    for (uint32_t i = 0; i < num_samples; i++) {
      read_data(sampleBytes, numBytes);      
      read_int16_from_data(sampleBytes,
                           numBytes,
                           0, //sampleIndex
                           sampleCh1,
                           sampleCh2);
      
      sample_data[i * m_num_channels] = sampleCh1;
      if (m_num_channels == 2) {
        sample_data[i * m_num_channels + 1] = sampleCh2;
      }
    }

    return true;
  }

  bool wavread::deallocate() {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    if (m_file) {
      fclose(m_file);
      m_file = nullptr;
    }

    return true;
  }

  //Read sample from in-memory wav data array
  bool wavread::read_int16_from_data(const uint8_t *sample_data, uint32_t num_samples,
                                     uint32_t sample_index, int16_t &channel1,
                                     int16_t &channel2) {

    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }

    //Verify in bounds
    uint32_t sampleBlockSize = m_num_channels * m_byte_depth;
    if (num_samples < ((sample_index + 1) * sampleBlockSize)) {
      channel1 = 0;
      channel2 = 0;
      return false;
    }

    switch (sampleBlockSize) {
    case 1: { //int8 mono
      //NOTE:
      //"There are some inconsistencies in the WAV format:
      //for example, 8-bit data is unsigned while 16-bit data is signed"
      //https://en.wikipedia.org/wiki/WAV
      channel1 = (int16_t) (
                            ((uint16_t) (sample_data[sample_index]) - TWO_POW_7_AS_UINT16) *
                            TWO_POW_8_AS_UINT16);
      channel2 = 0;

      break;
    }
    case 2: { //int8 stereo or int16 mono
      if (m_num_channels == 2) { //8-bit stereo
        //NOTE:
        //"There are some inconsistencies in the WAV format:
        //for example, 8-bit data is unsigned while 16-bit data is signed"
        //https://en.wikipedia.org/wiki/WAV
        uint64_t sampleIndexX2 = sample_index * 2;
        channel1 = (int16_t) (
                              ((uint16_t) sample_data[sampleIndexX2] - TWO_POW_7_AS_UINT16) *
                              TWO_POW_8_AS_UINT16);
        channel2 = (int16_t) (
                              ((uint16_t) sample_data[sampleIndexX2 + 1] - TWO_POW_7_AS_UINT16) *
                              TWO_POW_8_AS_UINT16);
      } else { //int16 mono
        channel1 = ((int16_t *) sample_data)[sample_index];
        channel2 = 0;
      }

      break;
    }
    case 3: { //int24 mono
      uint64_t sampleIndexX3 = sample_index * 3;
      uint32_t ch1 = sample_data[sampleIndexX3 + 2];
      ch1 <<= 8;
      ch1 |= sample_data[sampleIndexX3 + 1];
      ch1 <<= 8;
      ch1 |= sample_data[sampleIndexX3];
      if (0x800000 & ch1) { //If negative...
        ch1 |= 0xFF000000; //Sign extension
      }
      channel1 = (int16_t) (((uint32_t) ch1) / TWO_POW_8_AS_UINT32);
      channel2 = 0;

      break;
    }
    case 4: { //int16 stereo, int32 mono, float32 mono
      if (!m_are_samples_ints) { //float32 mono
        float ch1float = ((float *) sample_data)[sample_index] *
          TWO_POW_15_LESS1_AS_FLOAT32; //For floats, full scale is 1.0; for int16, 2^15 - 1.
        channel1 = ((int16_t) ch1float);
        channel2 = 0;
      } else if (m_num_channels == 1) { //int32 mono
        channel1 = (int16_t) ((((uint32_t *) sample_data)[sample_index]) /
                              TWO_POW_16_AS_UINT32);
        channel2 = 0;
      } else { //int16 stereo
        uint64_t sampleIndexX2 = sample_index * 2;
        channel1 = ((int16_t *) sample_data)[sampleIndexX2];
        channel2 = ((int16_t *) sample_data)[sampleIndexX2 + 1];
      }

      break;
    }
    case 6: { //int24 stereo
      uint64_t sampleIndexX6 = sample_index * 6;

      uint32_t ch1 = sample_data[sampleIndexX6 + 2];
      ch1 <<= 8;
      ch1 |= sample_data[sampleIndexX6 + 1];
      ch1 <<= 8;
      ch1 |= sample_data[sampleIndexX6];
      if (0x800000 & ch1) { //If negative...
        ch1 |= 0xFF000000; //Sign extension
      }
      channel1 = (int16_t) ((uint32_t) ch1 / TWO_POW_8_AS_UINT32);

      uint32_t ch2 = sample_data[sampleIndexX6 + 5];
      ch2 <<= 8;
      ch2 |= sample_data[sampleIndexX6 + 4];
      ch2 <<= 8;
      ch2 |= sample_data[sampleIndexX6 + 3];
      if (0x800000 & ch2) { //If negative...
        ch2 |= 0xFF000000; //Sign extension
      }
      channel1 = (int16_t) ((uint32_t) ch2 / TWO_POW_8_AS_UINT32);

      break;
    }
    case 8: { //float64 mono, float32 stereo, int32 stereo
      if (!m_are_samples_ints) { //floating point
        if (m_num_channels == 1) { //float64 mono
          double ch1float = ((double *) sample_data)[sample_index] *
            TWO_POW_15_LESS1_AS_FLOAT64; //For floats, full scale is 1.0; for int16, 2^15 - 1.
          channel1 = ((int16_t) ch1float);
          channel2 = 0;
        } else { // (numChannels == 2) - float32 stereo
          uint64_t sampleIndexX2 = sample_index * 2;
          float ch1float = ((float *) sample_data)[sampleIndexX2] *
            TWO_POW_15_LESS1_AS_FLOAT32; //For floats, full scale is 1.0; for int16, 2^15 - 1.
          channel1 = ((int16_t) ch1float);
          float ch2float = ((float *) sample_data)[sampleIndexX2 + 1] *
            TWO_POW_15_LESS1_AS_FLOAT32; //For floats, full scale is 1.0; for int16, 2^15 - 1.
          channel2 = ((int16_t) ch2float);
        }
      } else { //int32 stereo
        uint64_t sampleIndexX2 = sample_index * 2;
        channel1 = (int16_t) ((uint32_t) (((int32_t *) sample_data)[sampleIndexX2]) /
                              TWO_POW_16_AS_UINT32);
        channel2 = (int16_t) (
                              (uint32_t) (((int32_t *) sample_data)[sampleIndexX2 + 1]) /
                              TWO_POW_16_AS_UINT32);
      }

      break;
    }
    case 16: { //float64 stereo
      uint64_t sampleIndexX2 = sample_index * 2;

      double ch1float = ((double *) sample_data)[sampleIndexX2] *
        TWO_POW_15_LESS1_AS_FLOAT64; //For floats, full scale is 1.0; for int16, 2^15 - 1.
      channel1 = ((int16_t) ch1float);

      double ch2float = ((double *) sample_data)[sampleIndexX2 + 1] *
        TWO_POW_15_LESS1_AS_FLOAT64; //For floats, full scale is 1.0; for int16, 2^15 - 1.
      channel2 = ((int16_t) ch2float);

      break;
    }
    default: { //Error case

      channel1 = 0;
      channel2 = 0;

      break;
    }
    }
    return true;
  }

  const char *wavread::get_path() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return nullptr;
    }
    return m_file_path;
  }

  uint32_t wavread::get_sample_rate() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }
    return m_sample_rate;
  }

  uint32_t wavread::get_num_samples() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }
    return m_num_samples;
  }

  uint32_t wavread::get_channels_num() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }
    return m_num_channels;
  }

  bool wavread::are_samples_ints() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }
    return m_are_samples_ints;
  }

  uint32_t wavread::get_byte_depth() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }
    return m_byte_depth;
  }

  uint32_t wavread::get_sample_data_size() {
    if (!m_is_initialized) {
      std::cerr << UNINITIALIZED_MSG << std::endl;
      return false;
    }
    return m_sample_data_size;
  }
}

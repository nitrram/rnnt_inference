#include "wav_buf_simulator.h"
#include <algorithm>

#include <iostream>

#define LEAP 4096 //~1/25s
//#define LEAP 8192
//#define LEAP 16000 //1s
//#define LEAP 64000


// 10ms ~ 160; 25ms ~ 400
#define OFFSET 160

namespace spr {


  wav_buf_simulator::wav_buf_simulator(int16_t *whole_buffer,
                                       size_t whole_buffer_size,
                                       int (*callback)(buf_t *buf, size_t siz)) :
    m_buffer(whole_buffer),
    m_buffer_size(whole_buffer_size),
    m_callback(callback) {
  }

  void wav_buf_simulator::start_emitting() const {

		auto leap = std::min(m_buffer_size, (size_t)LEAP);

    size_t i = 0;
		m_callback(m_buffer, leap);
    for(i=leap-OFFSET; i <  m_buffer_size - leap; i+=leap-OFFSET) {
      m_callback(m_buffer + i, leap);
    }

    // push the rest out	
		if(i+OFFSET < m_buffer_size) {
			auto diff = m_buffer_size - i;
			m_callback(m_buffer + i-OFFSET, diff+OFFSET);
		}

		// finalize
		m_callback(NULL, 0);
  }
}
//eof

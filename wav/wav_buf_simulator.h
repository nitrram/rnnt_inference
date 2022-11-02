#pragma once

#include "common/buf_type.h"
#include <cstddef>

namespace spr {


  class wav_buf_simulator {
  public:
    wav_buf_simulator(buf_t *whole_buffer, size_t whole_buffer_size,
                      int (*callback)(buf_t *buf, size_t siz));

    virtual ~wav_buf_simulator() = default;

    void start_emitting() const;

  private:
    buf_t *m_buffer;
    size_t m_buffer_size;

    int (*m_callback)(buf_t *, size_t);

  };
}
//eof
